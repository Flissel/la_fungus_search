"""Context MCP server: the measured retrieval stack, served to agents.

Layer 5 of the context-with-receipts design. Any MCP client — a Claude Code
session, OpenFang, Hermes — gets multi-corpus search over the section-27/28
stack, and every hit carries its receipts: file, line span, symbol, corpus name,
and the sha256 digest of the document source at index time. An agent can quote
context *and* say which version of reality it was quoting.

Why this exists next to `mcp_server.py`: that server fronts the chunk-based
production retriever with a ~50 s cold start and one shared cache. This one
loads manifests in about a second, keeps one index per corpus, and needs neither
an embedding service nor torch (the dense arm arms itself only when
`FUNGUS_V2_EMBEDDER_URL` points at the local embedding service).

Configuration, resolved at first tool call, fail-closed with readable errors:

- ``FUNGUS_V2_CORPORA`` — path to the JSON corpus list understood by
  `retrieval_v2.build_from_env` (name / manifest / snapshot? / rank_rule? /
  embedder_url?). Preferred.
- else ``FUNGUS_V2_MANIFEST`` (+ optional ``FUNGUS_V2_SNAPSHOT``) for a single
  corpus named "default".

Run::

    python -m embeddinggemma.context_mcp            # stdio, for MCP clients
    python -m embeddinggemma.context_mcp --selftest  # load config, print corpora
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from embeddinggemma.retrieval_v2 import (
    HttpQueryEmbedder,
    MultiRetrieval,
    RetrievalV2,
    build_from_env,
    load_index,
)

_STATE: dict[str, Any] = {}


def _build_engine() -> Any:
    environ = dict(os.environ)
    environ.setdefault("FUNGUS_RETRIEVAL_V2", "1")  # being asked via MCP is opting in
    engine = build_from_env(environ)
    if engine is None:
        raise ValueError(
            "no corpora configured: set FUNGUS_V2_CORPORA (JSON list) or "
            "FUNGUS_V2_MANIFEST/FUNGUS_V2_SNAPSHOT"
        )
    return engine


def _engine() -> Any:
    if "engine" not in _STATE:
        _STATE["engine"] = _build_engine()
    return _STATE["engine"]


def _named_engines(engine: Any) -> dict[str, RetrievalV2]:
    if isinstance(engine, MultiRetrieval):
        return dict(engine._engines)  # noqa: SLF001 — same module family, read-only
    return {"default": engine}


def corpora_impl() -> dict[str, Any]:
    """What is searchable, and with which engine configuration."""
    engine = _engine()
    rows = []
    for name, single in _named_engines(engine).items():
        rows.append(
            {
                "corpus": name,
                "engine": single.engine,
                "documents": len(single._index.documents),  # noqa: SLF001
                "rank_rule": single.rank_rule,
                "manifest_digest": single._index.manifest_digest,  # noqa: SLF001
            }
        )
    return {"corpora": rows}


def search_impl(query: str, corpus: str = "", top_k: int = 8) -> dict[str, Any]:
    """Search one corpus or all of them; every hit carries its receipts."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    engine = _engine()
    if corpus:
        engines = _named_engines(engine)
        if corpus not in engines:
            raise ValueError(
                f"unknown corpus {corpus!r}; configured: {sorted(engines)}"
            )
        chosen = engines[corpus]
        answer = chosen.search(query, top_k=top_k)
        for row in answer["results"]:
            row["metadata"].setdefault("corpus", corpus)
        return answer
    return engine.search(query, top_k=top_k)


def main() -> int:
    if "--selftest" in sys.argv:
        try:
            overview = corpora_impl()
        except Exception as error:
            print(f"selftest failed: {error}", file=sys.stderr)
            return 1
        json.dump(overview, sys.stdout, indent=2)
        print()
        return 0

    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "context-v2",
        instructions=(
            "Multi-corpus context search with receipts: every hit names its "
            "corpus, file, line span, symbol and the sha-bound manifest digest. "
            "Use context_corpora first to see what is searchable."
        ),
    )

    @server.tool()
    def context_corpora() -> dict:
        """List the configured corpora, their engines and document counts."""
        return corpora_impl()

    @server.tool()
    def context_search(query: str, corpus: str = "", top_k: int = 8) -> dict:
        """Search the configured corpora (or one of them) for context.

        Returns ranked hits with content and receipt metadata (corpus, file,
        start_line, end_line, symbol, expanded). Empty `corpus` searches all.
        """
        return search_impl(query, corpus=corpus, top_k=top_k)

    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
