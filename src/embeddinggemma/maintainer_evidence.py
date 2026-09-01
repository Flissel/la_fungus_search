"""Evidence CLI for the secondbrain Maintainer-Agent.

The vault's maintainer pipeline (secondbrain `_automation/maintainer/`) needs
code evidence for claims, and its original semantic path was recorded as not
operational: the retriever required an `llm_config.yml` embeddings section, the
shared `.fungus_cache` meant a reindex on one repo poisoned every other, and the
skill paid a ~50 s cold start per checked sentence. This CLI is the replacement
evidence engine, built on the section-27 stack:

- AST manifest per repo, cached in a caller-chosen directory — one index per
  corpus, the shared-cache poisoning cannot happen by construction;
- BM25 + one-hop call-graph expansion (`RetrievalV2`, no snapshot, no embedding
  service, no torch) — loads in about a second;
- **every hit carries the sha256 digest of its document source**, which is what
  the vault-side ledger binds verdicts to: staleness becomes a digest diff.

One process per *run*, all queries batched — the per-sentence cold start dies
here rather than being optimised.

Contract (stdin-free, file-based, deliberately boring)::

    python -m embeddinggemma.maintainer_evidence \
        --repo C:/path/to/repo --queries queries.json \
        --cache-dir C:/path/to/cache [--top-k 5] [--rebuild]

`queries.json` is a JSON list of strings. Output on stdout, one JSON object:

    {"engine": "...", "manifest_digest": "...", "corpus_root": "...",
     "document_count": N,
     "results": [{"query": "...", "hits": [{"file", "start_line", "end_line",
                  "symbol", "score", "digest", "source", "expanded"}]}]}

Every failure is a non-zero exit with a one-line reason on stderr — the vault
adapter turns that into `source: "error"` exactly like the old one did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from embeddinggemma.retrieval_v2 import RetrievalV2, load_index

MANIFEST_NAME = "evidence-manifest.json"

# Vendor and tooling trees. Without this, walking a repo means AST-parsing its
# virtualenv -- measured as a 300 s timeout on the first real target repo.
EXCLUDE_DIRS = frozenset(
    {".git", ".venv", "venv", "env", "node_modules", "site-packages",
     ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
     ".eggs", ".claude"}
)


def _repo_key(repo: Path) -> str:
    return hashlib.sha256(str(repo.resolve()).lower().encode("utf-8")).hexdigest()[:16]


def _manifest_path(cache_dir: Path, repo: Path) -> Path:
    return cache_dir / _repo_key(repo) / MANIFEST_NAME


def ensure_manifest(repo: Path, cache_dir: Path, rebuild: bool) -> Path:
    """Build the AST manifest for `repo` into its own cache slot.

    The build is cheap enough to be the freshness strategy: `--rebuild` (or a
    missing file) re-derives it from the working tree, so evidence is always
    against the code as it is on disk, commit or no commit.
    """
    target = _manifest_path(cache_dir, repo)
    if target.exists() and not rebuild:
        return target
    from benchmarks.gate2.build_local_snapshot import _commit_sha
    from benchmarks.gate2.manifest import build_manifest, save_manifest

    try:
        sha = _commit_sha(repo)
    except Exception:
        sha = "no-git"
    manifest = build_manifest(repo, sha, f"evidence-{_repo_key(repo)}", exclude_dirs=EXCLUDE_DIRS)
    save_manifest(manifest, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="maintainer evidence over retrieval v2")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()

    if not arguments.repo.is_dir():
        print(f"repo does not exist: {arguments.repo}", file=sys.stderr)
        return 2
    try:
        queries = json.loads(arguments.queries.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"queries file unreadable: {error}", file=sys.stderr)
        return 2
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        print("queries must be a JSON list of strings", file=sys.stderr)
        return 2

    try:
        manifest_path = ensure_manifest(arguments.repo, arguments.cache_dir, arguments.rebuild)
        index = load_index(manifest_path)
        engine = RetrievalV2(index)
    except Exception as error:
        print(f"evidence engine failed to load: {error}", file=sys.stderr)
        return 3

    digest_of = {
        document.document_id: hashlib.sha256(document.source.encode("utf-8")).hexdigest()
        for document in index.documents
    }
    results = []
    for query in queries:
        hits = []
        for row in engine.search(query, top_k=arguments.top_k)["results"]:
            meta = row["metadata"]
            hits.append(
                {
                    "file": meta["file"],
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "symbol": meta["symbol"],
                    "score": row["relevance_score"],
                    "digest": digest_of[meta["document_id"]],
                    "source": row["content"],
                    "expanded": meta["expanded"],
                }
            )
        results.append({"query": query, "hits": hits})

    json.dump(
        {
            "engine": engine.engine,
            "manifest_digest": index.manifest_digest,
            "corpus_root": str(arguments.repo),
            "document_count": len(index.documents),
            "results": results,
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
