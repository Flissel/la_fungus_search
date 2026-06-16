from __future__ import annotations

import os
import re
import hashlib
from typing import List, Dict, Any
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = os.path.join(".fungus_cache", "chunks")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Opt-Stage-2 (2026-05-25): path-boost in chunk header. ──
# Embedding score of a chunk was dominated by body tokens; identifier-rich
# filenames + parent dirs (e.g. "face_math/landmark_detector.py") barely
# influenced ranking. We now prepend a small block that repeats the
# filename stem + parent dirs as searchable identifiers, then a snake_case
# split of the same tokens (so "FaceLandmarkDetector" → "face landmark
# detector" matches a natural-language query token-by-token).
_SPLIT_RE = re.compile(r"[_\-.]|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _path_boost_header(rel: str, start: int, end: int, step: int) -> str:
    """Build a chunk header that gives path tokens real semantic weight."""
    parts = [p for p in re.split(r"[\\/]", rel) if p]
    fname = parts[-1] if parts else rel
    stem = os.path.splitext(fname)[0]
    parent_dirs = " ".join(parts[-4:-1])  # up to 3 parent dirs
    # Snake/camel/kebab → individual lowercase tokens
    nat = " ".join(t for t in _SPLIT_RE.split(stem) if t).lower()
    # Header gives the embedder 3 distinct mentions of the path tokens:
    #   1. the verbatim relative path (anchors filename + dir hierarchy)
    #   2. the stem + parent dirs as a phrase
    #   3. natural-language token decomposition (camelCase → "camel case")
    return (
        f"# file: {rel} | lines: {start}-{end} | window: {step}\n"
        f"# path: {stem} in {parent_dirs}\n"
        f"# tokens: {nat}\n"
    )



def _file_sha1(path: str) -> str:
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return ""


def _cache_key(path: str, windows: List[int]) -> str:
    rel = os.path.relpath(path).replace(os.sep, "_")
    win_key = "-".join(str(int(w)) for w in sorted(set(int(w) for w in windows)))
    sha = _file_sha1(path)
    return os.path.join(CACHE_DIR, f"{rel}.{sha}.{win_key}.jsonl")


def _load_cached_chunks(path: str, windows: List[int]) -> List[str]:
    c = _cache_key(path, windows)
    try:
        if os.path.exists(c):
            with open(c, 'r', encoding='utf-8') as f:
                return [line.rstrip("\n").replace('\\n', '\n') for line in f]
    except Exception:
        return []
    return []


def _save_cached_chunks(path: str, windows: List[int], chunks: List[str]) -> None:
    c = _cache_key(path, windows)
    try:
        os.makedirs(os.path.dirname(c), exist_ok=True)
        with open(c, 'w', encoding='utf-8') as f:
            for ch in chunks:
                f.write(ch.replace('\n', '\\n') + '\n')
    except Exception:
        pass


def _chunk_line_windows(path: str, windows: List[int]) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return []
    chunks: List[str] = []
    total = len(lines)
    rel = os.path.relpath(path)
    for w in windows:
        step = max(1, int(w))
        for i in range(0, total, step):
            start = i + 1
            end = min(i + step, total)
            body = ''.join(lines[i:end])
            if body.strip():
                header = _path_boost_header(rel, start, end, step)
                chunks.append(header + body)
    return chunks


def _chunk_python_file_ast(path: str, windows: List[int]) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception:
        return []
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    lines = source.splitlines(keepends=True)
    total = len(lines)
    rel = os.path.relpath(path)
    max_window = max(1, max(int(w) for w in windows) if windows else 1000)

    def slice_block(start_line: int, end_line: int) -> List[str]:
        start_idx = max(1, start_line)
        end_idx = min(end_line, total)
        body = ''.join(lines[start_idx - 1:end_idx])
        if not body.strip():
            return []
        # If block is small enough, keep as one chunk
        if (end_idx - start_idx + 1) <= max_window:
            header = _path_boost_header(rel, start_idx, end_idx, max_window)
            return [header + body]
        # Otherwise split into windowed sub-chunks
        chunks_local: List[str] = []
        step = max_window
        overlap = max(0, int(0.2 * step))
        i = start_idx - 1
        while i < end_idx:
            s = i + 1
            e = min(i + step, end_idx)
            sub = ''.join(lines[s - 1:e])
            if sub.strip():
                header = _path_boost_header(rel, s, e, max_window)
                chunks_local.append(header + sub)
            if e >= end_idx:
                break
            i = e - overlap
        return chunks_local

    chunks: List[str] = []
    # Module-level docstring and imports as a header chunk (optional)
    mod_start = 1
    mod_end = total
    if getattr(tree, 'body', []):
        first_node = tree.body[0]
        last_node = tree.body[-1]
        try:
            mod_start = getattr(first_node, 'lineno', 1)
            mod_end = getattr(last_node, 'end_lineno', total)
        except Exception:
            mod_start, mod_end = 1, total

    # Collect class and function nodes
    nodes: List[ast.AST] = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(n)

    if not nodes:
        return slice_block(1, total)

    for n in nodes:
        try:
            s = getattr(n, 'lineno', None)
            e = getattr(n, 'end_lineno', None)
        except Exception:
            s, e = None, None
        if s is None or e is None:
            # Fallback to full file slice if positions are unavailable
            return slice_block(1, total)
        chunks.extend(slice_block(int(s), int(e)))

    return chunks


def chunk_python_file(path: str, windows: List[int]) -> List[str]:
    # Try AST-based chunking first for semantically coherent chunks
    chunks_ast = _chunk_python_file_ast(path, windows)
    if chunks_ast:
        return chunks_ast
    # Fallback to simple line-window chunking
    return _chunk_line_windows(path, windows)


_CODE_EXTENSIONS = {
    '.py', '.ts', '.tsx', '.js', '.jsx', '.rs', '.go', '.java',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift',
    '.kt', '.scala', '.sh', '.bash', '.zsh', '.yaml', '.yml',
    '.toml', '.json', '.sql', '.prisma', '.graphql', '.vue',
    '.svelte', '.css', '.scss', '.html', '.md', '.txt',
}

# Dotfiles to index (config/env templates — NOT actual secrets)
_DOTFILE_PATTERNS = {
    '.env.example', '.env.template', '.env.sample',
    '.env.hotel', '.env.hotel.example', '.env.electron',
    '.gitignore', '.dockerignore', '.eslintrc', '.prettierrc',
    'Dockerfile', 'Makefile', 'Cargo.toml', 'Cargo.lock',
    'pyproject.toml', 'docker-compose.yml', 'docker-compose.yaml',
}

# Actual .env files contain secrets — index them too but could be excluded
# by setting FUNGUS_SKIP_SECRETS=1
_SECRETS_FILES = {'.env', '.envrc'}
_SKIP_SECRETS = os.environ.get('FUNGUS_SKIP_SECRETS', '') == '1'


def _is_indexable(filename: str) -> bool:
    """Check if a file should be indexed."""
    fn_lower = filename.lower()
    # Standard code extensions
    ext = os.path.splitext(fn_lower)[1]
    if ext in _CODE_EXTENSIONS:
        return True
    # Dotfile patterns (exact name match)
    if fn_lower in _DOTFILE_PATTERNS:
        return True
    # .env files (secrets — optional)
    if fn_lower in _SECRETS_FILES:
        return not _SKIP_SECRETS
    # .env.* variants (templates are safe)
    if fn_lower.startswith('.env.'):
        return True
    return False


def list_code_files(root_dir: str, max_files: int, exclude_dirs: List[str] = None) -> List[str]:
    files: List[str] = []
    count = 0
    for root, dirs, filenames in os.walk(root_dir):
        if exclude_dirs:
            dirs[:] = [d for d in dirs if not any(ex in os.path.join(root, d) for ex in exclude_dirs)]
        for fn in filenames:
            if _is_indexable(fn):
                files.append(os.path.join(root, fn))
                count += 1
                if max_files and count >= max_files:
                    return files
    return files


def collect_codebase_chunks(root_dir: str, windows: List[int], max_files: int, exclude_dirs: List[str] = None, workers: int | None = None) -> List[str]:
    files_to_process = list_code_files(root_dir, max_files, exclude_dirs)
    docs: List[str] = []
    max_workers = workers if (workers and workers > 0) else max(4, min(32, (os.cpu_count() or 8) * 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures: Dict[Any, str] = {}
        for p in files_to_process:
            cached = _load_cached_chunks(p, windows)
            if cached:
                docs.extend(cached)
            else:
                futures[ex.submit(chunk_python_file, p, windows)] = p
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                chunks = fut.result()
            except Exception:
                chunks = []
            if chunks:
                docs.extend(chunks)
                _save_cached_chunks(p, windows, chunks)
    return docs
