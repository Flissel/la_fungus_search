"""
MCP Server for la-fungus-search — semantic code search over vibemind-os.

Exposes the MCMP-RAG search engine as MCP tools so Claude Code (and other
MCP clients) can search the codebase semantically.

Tools:
  - fungus_search:        Semantic search across the indexed codebase
  - fungus_search_multi:  Run multiple queries and merge results
  - fungus_lookup_file:   Find all chunks from a specific file
  - fungus_index_stats:   Show index statistics
  - fungus_reindex:       Rebuild the index from scratch

Usage:
  python mcp_server.py              # stdio transport (for Claude Code)
  python mcp_server.py --http 8412  # HTTP transport (for remote clients)
"""
from __future__ import annotations

import asyncio
import sys
import os
import re
import time
import logging
import warnings

# Ensure src/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("fungus-mcp")

CODEBASE = os.environ.get(
    "FUNGUS_CODEBASE",
    os.path.normpath(os.path.join(_HERE, "..", "..")),
)
EMBED_MODEL = os.environ.get("FUNGUS_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# S.3: device override. Brain uses ~11GB GPU; default to CPU here so reindex
# doesn't OOM. Override with FUNGUS_DEVICE=cuda if GPU has headroom.
DEVICE_MODE = os.environ.get("FUNGUS_DEVICE", "cpu")
EXCLUDE_DIRS = [
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", ".pitchdeck_chroma", ".playwright-mcp",
    "uv.lock", ".kilocode", ".vscode",
]

# ---------------------------------------------------------------------------
# Eagerly load model + index at import time (before mcp.run starts the
# async event loop).  This way the heavy synchronous work is done before
# any tool call arrives.
# ---------------------------------------------------------------------------
_index_meta: dict = {}


def _load_retriever():
    from embeddinggemma.mcmp_rag import MCPMRetriever

    r = MCPMRetriever(
        embedding_model_name=EMBED_MODEL,
        num_agents=50,
        max_iterations=10,
        device_mode=DEVICE_MODE,
        embed_batch_size=256,
    )

    t0 = time.time()
    loaded = r.load_persistent_index()
    load_time = time.time() - t0

    global _index_meta
    if loaded:
        _index_meta = {
            "docs": len(r.documents),
            "dim": r._embed_dim,
            "load_time_s": round(load_time, 2),
            "source": "persistent_cache",
        }
    else:
        _index_meta = {"docs": 0, "dim": None, "load_time_s": 0, "source": "none"}

    return r


# Load synchronously NOW — before the event loop starts
_retriever = _load_retriever()


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "la-fungus-search",
    instructions="Semantic code search over vibemind-os using MCMP-RAG (multi-agent pheromone simulation + vector embeddings)",
)


def _sync_search(query: str, top_k: int) -> dict:
    """Run search_direct synchronously (CPU-bound)."""
    return _retriever.search_direct(query, top_k=top_k)


@mcp.tool()
async def fungus_search(query: str, top_k: int = 10) -> str:
    """Search the vibemind-os codebase semantically.

    Returns the top-k most relevant code chunks for a natural language query.
    Each result includes the file path, line range, relevance score, and code content.

    Args:
        query: Natural language search query (e.g. "WebSocket real-time messaging handler")
        top_k: Number of results to return (default 10, max 30)
    """
    if not _retriever.documents:
        return "ERROR: No index loaded. Run the fungus_reindex tool first."

    top_k = min(max(1, top_k), 30)
    t0 = time.time()
    results = await asyncio.to_thread(_sync_search, query, top_k)
    search_time = time.time() - t0

    items = results.get("results", [])
    if not items:
        return f"No results for: {query}"

    lines = [f"## Search: \"{query}\" ({len(items)} results, {search_time*1000:.0f}ms)\n"]
    for i, item in enumerate(items, 1):
        score = item.get("relevance_score", 0)
        content = item.get("content", "")

        m = re.search(r'# file: (.+?) \| lines: (\d+-\d+) \| window: (\d+)', content)
        if m:
            filepath = m.group(1).replace("\\", "/")
            line_range = m.group(2)
            body = "\n".join(content.split("\n")[1:]).strip()
        else:
            filepath = "unknown"
            line_range = "?"
            body = content.strip()

        if len(body) > 800:
            body = body[:800] + "\n... (truncated)"

        lines.append(f"### {i}. {filepath} (lines {line_range}) — score: {score:.3f}")
        lines.append(f"```\n{body}\n```\n")

    return "\n".join(lines)


@mcp.tool()
async def fungus_search_multi(queries: str, top_k: int = 5) -> str:
    """Run multiple semantic searches and merge results (deduplicated).

    Useful for exploring a topic from multiple angles. Separate queries with newlines or semicolons.

    Args:
        queries: Multiple queries separated by newlines or semicolons
        top_k: Results per query (default 5)
    """
    if not _retriever.documents:
        return "ERROR: No index loaded. Run the fungus_reindex tool first."

    query_list = [q.strip() for q in re.split(r'[;\n]', queries) if q.strip()]
    if not query_list:
        return "No queries provided."

    top_k = min(max(1, top_k), 15)
    seen_files: set[str] = set()
    all_results: list[tuple[str, dict]] = []

    for q in query_list[:5]:
        results = await asyncio.to_thread(_sync_search, q, top_k)
        for item in results.get("results", []):
            content = item.get("content", "")
            m = re.search(r'# file: (.+?) \|', content)
            file_key = m.group(1) if m else content[:100]
            if file_key not in seen_files:
                seen_files.add(file_key)
                all_results.append((q, item))

    if not all_results:
        return f"No results for queries: {query_list}"

    lines = [f"## Multi-search: {len(all_results)} unique results from {len(query_list)} queries\n"]
    for i, (q, item) in enumerate(all_results[:20], 1):
        score = item.get("relevance_score", 0)
        content = item.get("content", "")
        m = re.search(r'# file: (.+?) \| lines: (\d+-\d+)', content)
        if m:
            filepath = m.group(1).replace("\\", "/")
            line_range = m.group(2)
        else:
            filepath = "unknown"
            line_range = "?"

        body = "\n".join(content.split("\n")[1:]).strip()
        if len(body) > 400:
            body = body[:400] + "..."

        lines.append(f"### {i}. {filepath}:{line_range} (score: {score:.3f}, query: \"{q}\")")
        lines.append(f"```\n{body}\n```\n")

    return "\n".join(lines)


@mcp.tool()
async def fungus_lookup_file(filepath: str, top_k: int = 10) -> str:
    """Find all indexed chunks from a specific file path.

    Args:
        filepath: Partial or full file path to search for (e.g. "brain/core/radial_attention.py")
        top_k: Max chunks to return (default 10)
    """
    if not _retriever.documents:
        return "ERROR: No index loaded."

    def _lookup():
        filepath_lower = filepath.lower().replace("\\", "/")
        matches = []
        for doc in _retriever.documents:
            content = doc.content or ""
            m = re.search(r'# file: (.+?) \|', content)
            if m:
                doc_file = m.group(1).replace("\\", "/").lower()
                if filepath_lower in doc_file:
                    matches.append(doc)
        return matches[:top_k]

    matches = await asyncio.to_thread(_lookup)

    if not matches:
        return f"No indexed chunks found for: {filepath}"

    lines = [f"## {len(matches)} chunks from files matching \"{filepath}\"\n"]
    for i, doc in enumerate(matches, 1):
        content = doc.content or ""
        m = re.search(r'# file: (.+?) \| lines: (\d+-\d+)', content)
        if m:
            fp = m.group(1)
            lr = m.group(2)
        else:
            fp = "unknown"
            lr = "?"
        body = "\n".join(content.split("\n")[1:]).strip()
        if len(body) > 600:
            body = body[:600] + "..."
        lines.append(f"### {i}. {fp} (lines {lr})")
        lines.append(f"```\n{body}\n```\n")

    return "\n".join(lines)


@mcp.tool()
async def fungus_index_stats() -> str:
    """Show statistics about the current search index.

    Returns document count, embedding dimensions, file type breakdown, and top directories.
    """
    meta = dict(_index_meta)

    if not _retriever.documents:
        return f"Index is empty. Meta: {meta}"

    from collections import Counter
    file_counter: Counter[str] = Counter()
    dir_counter: Counter[str] = Counter()
    ext_counter: Counter[str] = Counter()

    for doc in _retriever.documents:
        m = re.search(r'# file: (.+?) \|', doc.content or "")
        if m:
            fp = m.group(1).replace("\\", "/")
            file_counter[fp] += 1
            parts = fp.split("/")
            if len(parts) >= 2:
                dir_counter[parts[0] + "/" + parts[1]] += 1
            ext = os.path.splitext(fp)[1]
            ext_counter[ext] += 1

    lines = [
        "## la-fungus-search Index Stats\n",
        f"- **Total chunks**: {len(_retriever.documents)}",
        f"- **Unique files**: {len(file_counter)}",
        f"- **Embedding dim**: {_retriever._embed_dim}",
        f"- **Model**: {_retriever.embedding_model_name}",
        f"- **Load time**: {meta.get('load_time_s', '?')}s",
        "",
        "### File types:",
    ]
    for ext, count in ext_counter.most_common(10):
        lines.append(f"  - `{ext}`: {count} chunks")

    lines.append("\n### Top directories:")
    for d, count in dir_counter.most_common(10):
        lines.append(f"  - `{d}/`: {count} chunks")

    lines.append("\n### Most-chunked files:")
    for fp, count in file_counter.most_common(10):
        lines.append(f"  - `{fp}`: {count} chunks")

    return "\n".join(lines)


@mcp.tool()
async def fungus_reindex(codebase_path: str = "") -> str:
    """Rebuild the search index from scratch.

    This re-scans the codebase, chunks all code files, computes embeddings,
    and saves the persistent index. Takes a few minutes on CPU.

    Args:
        codebase_path: Override codebase root (default: vibemind-os root)
    """
    global _retriever, _index_meta

    target = codebase_path.strip() or CODEBASE
    if not os.path.isdir(target):
        return f"ERROR: Directory not found: {target}"

    def _do_reindex():
        from embeddinggemma.mcmp_rag import MCPMRetriever
        from embeddinggemma.ui.corpus import collect_codebase_chunks

        r = MCPMRetriever(
            embedding_model_name=EMBED_MODEL,
            num_agents=50,
            max_iterations=10,
            device_mode=DEVICE_MODE,
            embed_batch_size=256,
        )

        t0 = time.time()
        raw_chunks = collect_codebase_chunks(
            root_dir=target,
            windows=[200],
            max_files=15000,
            exclude_dirs=EXCLUDE_DIRS,
        )
        chunk_time = time.time() - t0

        filtered = [c for c in raw_chunks if len(c.strip()) >= 50]

        try:
            sys.path.insert(0, _HERE)
            from build_optimized import deduplicate_chunks
            deduped, dupe_count = deduplicate_chunks(filtered, threshold=3)
        except ImportError:
            deduped = filtered
            dupe_count = 0

        t1 = time.time()
        r.add_documents(deduped, cache=True)
        embed_time = time.time() - t1

        return r, len(raw_chunks), len(filtered), len(deduped), dupe_count, chunk_time, embed_time

    r, n_raw, n_filt, n_dedup, n_dupes, ct, et = await asyncio.to_thread(_do_reindex)

    _retriever = r
    _index_meta = {
        "docs": len(r.documents),
        "dim": r._embed_dim,
        "load_time_s": round(ct + et, 2),
        "source": "fresh_build",
    }

    return (
        f"## Reindex complete\n\n"
        f"- **Codebase**: {target}\n"
        f"- **Raw chunks**: {n_raw}\n"
        f"- **After filter**: {n_filt}\n"
        f"- **After dedup**: {n_dedup} (removed {n_dupes})\n"
        f"- **Indexed**: {len(r.documents)} docs, dim={r._embed_dim}\n"
        f"- **Chunk time**: {ct:.1f}s\n"
        f"- **Embed time**: {et:.1f}s\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="la-fungus-search MCP server")
    parser.add_argument("--http", type=int, default=0, help="Run as HTTP server on this port")
    args = parser.parse_args()

    if args.http:
        mcp.settings.port = args.http
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")