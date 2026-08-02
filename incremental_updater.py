#!/usr/bin/env python
"""Incremental fungus-search index updater.

Triggered by .git/hooks/post-commit. Reads git diff since the previous commit,
removes stale chunks from changed/deleted files, re-chunks + re-embeds the
changed files, and saves the updated FAISS + chunks. Designed for sub-30s
runtime per commit, 0% idle CPU between commits.

Architecture (W1 design — see plan-das-mal-soft-wren.md):
    1. git diff -> set of changed indexable files
    2. early exit if no relevant changes (~3s, no model load)
    3. load_persistent_index() to get existing 80k+ documents (no model needed
       for this — embeddings already on disk)
    4. filter out documents whose `# file: <path>` prefix matches a changed file
    5. chunk_python_file() (try-AST-first, line-fallback) on each changed file
    6. add_documents() the new chunks (this loads the model + embeds + rebuilds
       the FAISS index from all existing + new embeddings)
    7. save_persistent_index() — atomic disk write

The `# file:` prefix in each chunk string is our manifest. No separate JSON.

Usage:
    python incremental_updater.py                       # default --since HEAD~1
    python incremental_updater.py --since HEAD~5        # last 5 commits
    python incremental_updater.py --background          # self-detach + return

ASCII-only logging — Windows cp1252 cannot encode arrows etc.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO

# Repo paths (hardcoded by design — see plan, matches build_vibemind_cpu.py:18)
REPO_ROOT = Path("c:/Users/User/Desktop/Vibemind_V1")
CODEBASE = REPO_ROOT / "vibemind-os"
FUNGUS_ROOT = CODEBASE / "la-fungus-search"
LOG_DIR = REPO_ROOT / "backups"
DEFAULT_LOCK_FILE = LOG_DIR / "fungus_incremental_updater.lock"

WINDOWS = [200]
FILE_PREFIX_RE = re.compile(r"^# file: (.+?) \| lines:")

# Match the exclusion set in build_vibemind_cpu.py:20-25
EXCLUDE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", ".pitchdeck_chroma", ".playwright-mcp",
    "uv.lock", ".kilocode", ".vscode",
}


# ── machine-wide singleton ──────────────────────────────────────────────────

def acquire_singleton_lock(lock_path: Path | None = None) -> BinaryIO | None:
    """Acquire the one machine-wide incremental-updater slot.

    The file remains on disk for diagnostics, while the OS lock is tied to the
    open handle and is therefore released automatically if the process dies.
    """
    configured = os.environ.get("FUNGUS_UPDATER_LOCK_FILE", "").strip()
    path = Path(lock_path or configured or DEFAULT_LOCK_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None

    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n".encode("ascii"))
    handle.flush()
    return handle


def release_singleton_lock(handle: BinaryIO) -> None:
    """Release a lock returned by :func:`acquire_singleton_lock`."""
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


# ── logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"fungus_inc_{ts}.log"

    logger = logging.getLogger("fungus-inc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"log file: {log_path}")
    return logger


# ── git diff ─────────────────────────────────────────────────────────────────

def _run_git_diff(cwd: Path, since: str, log: logging.Logger,
                   path_filter: str | None = None) -> set[str]:
    """Run `git diff --name-only since HEAD [-- path_filter]`. Return set of
    repo-relative paths (or empty set on error). Repo-relative to `cwd`."""
    cmd = ["git", "diff", "--name-only", since, "HEAD"]
    if path_filter:
        cmd += ["--", path_filter]
    try:
        out = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except subprocess.CalledProcessError as e:
        log.debug(f"git diff failed in {cwd} (since={since}): {e.stderr[:200]}")
        return set()
    except subprocess.TimeoutExpired:
        log.warning(f"git diff timed out in {cwd}")
        return set()
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def _to_repo_root_rel(repo_cwd: Path, diff_paths: set[str]) -> set[str]:
    """Turn a repo's own diff paths into Vibemind_V1-root-relative paths.

    git diff emits paths relative to the repo it ran in. To merge diffs from
    repos at different submodule depths we re-anchor every path to the
    Vibemind_V1 root, e.g.:
        repo=vibemind-os/openfang, diff='mcp/README.md'
          -> 'vibemind-os/openfang/mcp/README.md'
    """
    try:
        prefix = repo_cwd.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        # repo_cwd is outside Vibemind_V1 — keep paths as-is
        prefix = Path()
    out: set[str] = set()
    for p in diff_paths:
        rel = (prefix / p).as_posix() if str(prefix) != "." else p
        out.add(rel)
    return out


def git_changed_files(since: str, log: logging.Logger,
                       repo_cwd: Path | None = None) -> set[str]:
    """Collect changed files across every relevant repo and return them as a
    set of Vibemind_V1-root-relative paths (forward slashes).

    Three diff sources, deduped:
      1. The repo the commit actually happened in (`repo_cwd` from the hook) —
         this is the authoritative source for commits made in any submodule,
         however deep.
      2. The Vibemind_V1 root repo, filtered to vibemind-os/ — catches
         file-level commits made from the root.
      3. The vibemind-os repo itself — catches commits made directly there.
    Sources 2+3 are belt-and-braces for the manual `--since` path; the hook
    path relies primarily on source 1.
    """
    files: set[str] = set()

    # 1. The repo the hook fired in (most precise — knows exactly what changed)
    if repo_cwd is not None:
        own = _run_git_diff(repo_cwd, since, log)
        if not own and since != "HEAD~1":
            own = _run_git_diff(repo_cwd, "HEAD~1", log)
        files.update(_to_repo_root_rel(repo_cwd, own))

    # 2. Vibemind_V1 root, scoped to vibemind-os/
    parent = _run_git_diff(REPO_ROOT, since, log, path_filter="vibemind-os/")
    files.update(parent)

    # 3. vibemind-os repo's own diff
    sub_paths = _run_git_diff(CODEBASE, since, log)
    if not sub_paths and since != "HEAD~1":
        sub_paths = _run_git_diff(CODEBASE, "HEAD~1", log)
    files.update(f"vibemind-os/{p}" for p in sub_paths if not p.startswith("vibemind-os/"))

    return files


# ── file indexability check (mirrors corpus.py _is_indexable logic) ─────────

# Mirror of _CODE_EXTENSIONS in corpus.py:165-170 — keep in sync if upstream
# changes. Importing the private name would be brittle (underscore = private).
_CODE_EXTENSIONS = {
    '.py', '.ts', '.tsx', '.js', '.jsx', '.rs', '.go', '.java',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift',
    '.kt', '.scala', '.sh', '.bash', '.zsh', '.yaml', '.yml',
    '.toml', '.json', '.sql', '.prisma', '.graphql', '.vue',
    '.svelte', '.css', '.scss', '.html', '.md', '.txt',
}
_DOTFILE_PATTERNS = {
    '.env.example', '.env.template', '.env.sample',
    '.env.hotel', '.env.hotel.example', '.env.electron',
    '.gitignore', '.dockerignore', '.eslintrc', '.prettierrc',
    'Dockerfile', 'Makefile', 'Cargo.toml', 'Cargo.lock',
}


def is_indexable(rel_path: str) -> bool:
    """Return True iff file should be re-embedded. Mirrors corpus._is_indexable."""
    # Reject if any path segment matches an excluded dir
    parts = rel_path.replace("\\", "/").split("/")
    if any(p in EXCLUDE_DIRS for p in parts):
        return False
    fn = parts[-1].lower()
    ext = os.path.splitext(fn)[1]
    if ext in _CODE_EXTENSIONS:
        return True
    if fn in _DOTFILE_PATTERNS:
        return True
    return False


# ── chunk file-path extraction ───────────────────────────────────────────────

def extract_file_from_chunk(chunk_string: str) -> str | None:
    """Parse `# file: <path> | lines: X-Y | window: Z\n...`, return <path>."""
    m = FILE_PREFIX_RE.match(chunk_string)
    return m.group(1) if m else None


def path_match_key(p: str) -> str:
    """Normalize a path for cross-platform comparison.

    git diff emits forward-slashes; corpus.py uses os.path.relpath which on
    Windows emits backslashes. We compare on the slash-normalized form.
    Additionally, corpus.py is called from FUNGUS_ROOT (la-fungus-search/) so
    its relative paths look like '..\\openfang\\foo.py'; git diff produces
    'vibemind-os/openfang/foo.py'. We normalize both to a comparable suffix.
    """
    # Convert to forward slashes
    q = p.replace("\\", "/")
    # Strip leading "../" (corpus.py from la-fungus-search emits ..\\)
    while q.startswith("../"):
        q = q[3:]
    # Strip leading "vibemind-os/" (git diff emits from repo root)
    if q.startswith("vibemind-os/"):
        q = q[len("vibemind-os/"):]
    return q.lower()


# ── background self-fork ─────────────────────────────────────────────────────

def background_self() -> None:
    """Re-exec self detached so git-hook returns instantly.

    On Windows this uses CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS to break
    from the parent's process group entirely. On POSIX, setsid via start_new_session.
    """
    args = [sys.executable, __file__] + [a for a in sys.argv[1:] if a != "--background"]
    LOG_DIR.mkdir(exist_ok=True)
    detach_log = LOG_DIR / f"fungus_inc_detach_{time.strftime('%Y%m%d_%H%M%S')}.log"

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": open(detach_log, "a", encoding="utf-8"),
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen(args, **kwargs)


# ── main ─────────────────────────────────────────────────────────────────────

def _run_with_args(args: argparse.Namespace) -> int:
    log = setup_logging()
    t0 = time.time()
    repo_cwd = Path(args.repo_cwd).resolve() if args.repo_cwd else None
    log.info(
        f"=== fungus-inc start (since={args.since}, dry={args.dry_run}, "
        f"repo_cwd={repo_cwd}) ==="
    )

    # === Step 1: git diff ===
    changed = git_changed_files(args.since, log, repo_cwd=repo_cwd)
    log.info(f"git diff: {len(changed)} candidate files under vibemind-os/")

    changed_indexable = {p for p in changed if is_indexable(p)}
    log.info(f"indexable changed files: {len(changed_indexable)}")
    if changed_indexable and len(changed_indexable) <= 20:
        for p in sorted(changed_indexable):
            log.info(f"  -> {p}")

    if not changed_indexable:
        log.info(f"no relevant changes, exit early ({time.time()-t0:.1f}s)")
        return 0

    if args.dry_run:
        log.info("DRY-RUN: would proceed to load index + filter + re-embed")
        log.info(f"DRY-RUN exit ({time.time()-t0:.1f}s)")
        return 0

    # === Step 2: lazy import + load existing index ===
    sys.path.insert(0, str(FUNGUS_ROOT / "src"))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        os.path.expanduser("~/.cache/huggingface"),
    )
    # Chdir into FUNGUS_ROOT so .fungus_cache/ resolves correctly.
    original_cwd = os.getcwd()
    os.chdir(FUNGUS_ROOT)
    try:
        from embeddinggemma.mcmp_rag import MCPMRetriever
        from embeddinggemma.ui.corpus import chunk_python_file

        # num_agents=1 / max_iterations=1: the ant-colony MCMP simulation is
        # only used for *search*, not for index building. Minimal here to cut
        # constructor overhead.
        r = MCPMRetriever(
            num_agents=1,
            max_iterations=1,
            embed_batch_size=32,
        )
        log.info("embedding role: fungus_search via OpenFang")

        t1 = time.time()
        if not r.load_persistent_index():
            log.error(
                "cannot load persistent index — run "
                "`python build_vibemind_cpu.py` first to seed the cache"
            )
            return 1
        log.info(f"loaded {len(r.documents)} docs from cache ({time.time()-t1:.1f}s)")

        # === Step 3: filter stale chunks ===
        changed_keys = {path_match_key(p) for p in changed_indexable}
        kept = []
        removed_count = 0
        for d in r.documents:
            f = extract_file_from_chunk(d.content)
            if f and path_match_key(f) in changed_keys:
                removed_count += 1
                continue
            kept.append(d)
        log.info(
            f"filter: removed {removed_count} stale chunks, "
            f"kept {len(kept)} ({len(r.documents)} -> {len(kept)})"
        )
        r.documents = kept
        # Force FAISS rebuild on next add (or save, if no new chunks)
        r._faiss_index = None

        # === Step 4: re-chunk changed files ===
        new_chunks: list[str] = []
        chunk_fail = 0
        for rel in sorted(changed_indexable):
            abs_path = REPO_ROOT / rel
            if not abs_path.is_file():
                # File was deleted — already removed from kept above
                continue
            try:
                cs = chunk_python_file(str(abs_path), windows=WINDOWS)
                new_chunks.extend(c for c in cs if len(c) >= 50)
            except Exception as e:  # noqa: BLE001
                chunk_fail += 1
                log.warning(f"chunking failed for {rel}: {e}")
        log.info(
            f"re-chunk: {len(new_chunks)} new chunks "
            f"({chunk_fail} files failed)"
        )

        # === Step 5: embed + add ===
        if new_chunks:
            # add_documents() embeds the new chunks AND rebuilds _faiss_index
            # from the full (already-filtered) document set — so the index
            # stays consistent with the kept + new embeddings.
            t1 = time.time()
            r.add_documents(new_chunks, cache=True)
            log.info(
                f"embedded + indexed: {time.time()-t1:.1f}s "
                f"(total docs now: {len(r.documents)})"
            )
        elif removed_count > 0:
            # Deletion-only commit: chunks were filtered out in Step 3 but no
            # new chunks were added, so add_documents() never ran and never
            # rebuilt FAISS. We MUST rebuild it here from the filtered
            # embeddings — otherwise save_persistent_index() either skips the
            # faiss.index write (leaving a stale file with the OLD count) or
            # the in-memory index stays out of sync with chunks.json.
            # _build_faiss is fast (~0.03s for ~2800 vecs).
            t1 = time.time()
            try:
                import numpy as np
                from embeddinggemma.mcmp.indexing import build_faiss_index
                if r.documents:
                    mat = np.array([d.embedding for d in r.documents],
                                   dtype=np.float32)
                    dim = int(r.documents[0].embedding.shape[0])
                    r._faiss_index = build_faiss_index(mat, dim)
                    r._embed_dim = dim
                else:
                    # Everything was deleted — empty the index cleanly.
                    r._faiss_index = None
                    r._embed_dim = None
                log.info(
                    f"deletion-only: rebuilt FAISS from {len(r.documents)} "
                    f"kept embeddings ({time.time()-t1:.2f}s)"
                )
            except Exception as e:  # noqa: BLE001
                log.error(f"deletion-only FAISS rebuild failed: {e}")
                return 1
        else:
            # No new chunks AND no removals — nothing actually changed for the
            # index (e.g. the changed files produced zero chunks). Skip the
            # save entirely; the on-disk cache is already correct.
            log.info("no index change (0 added, 0 removed); skipping save")
            log.info(
                f"=== fungus-inc done in {time.time()-t0:.1f}s (no-op) ==="
            )
            return 0

        # === Step 6: save ===
        t1 = time.time()
        if not r.documents:
            # Every indexed document was deleted. save_persistent_index()
            # returns False on an empty doc set (by design — nothing to write),
            # which is NOT an error here. Leave the cache as-is; a future
            # commit that adds files will reseed it.
            log.warning(
                "all documents removed — index is now empty; "
                "leaving cache untouched (run build_vibemind_cpu.py to reseed)"
            )
        else:
            ok = r.save_persistent_index()
            log.info(f"save: ok={ok} ({time.time()-t1:.1f}s)")
            if not ok:
                log.error("save_persistent_index returned False")
                return 1

    finally:
        os.chdir(original_cwd)

    log.info(
        f"=== fungus-inc done in {time.time()-t0:.1f}s "
        f"({removed_count} removed, {len(new_chunks)} added) ==="
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--background", action="store_true",
                        help="self-detach and exit immediately (for git-hook)")
    parser.add_argument("--since", default="HEAD~1",
                        help="git ref to diff against (default: HEAD~1)")
    parser.add_argument("--repo-cwd", default=None,
                        help="repo the commit happened in (set by the git hook)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print plan without modifying index")
    args = parser.parse_args()

    if args.background:
        background_self()
        return 0

    lock_handle = acquire_singleton_lock()
    if lock_handle is None:
        print("[fungus-inc] updater already running; exiting", file=sys.stderr)
        return 0
    try:
        return _run_with_args(args)
    finally:
        release_singleton_lock(lock_handle)


if __name__ == "__main__":
    sys.exit(main())
