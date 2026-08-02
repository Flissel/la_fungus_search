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
import hashlib
import json

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
# The reranker is independent of OpenFang embeddings and remains optional.
RERANKER_DEVICE = os.environ.get("FUNGUS_RERANKER_DEVICE", "cpu")
EXCLUDE_DIRS = [
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", ".pitchdeck_chroma", ".playwright-mcp",
    "uv.lock", ".kilocode", ".vscode",
    # ── Opt-Stage-2 (2026-05-25): dead/duplicate trees that polluted top-K ──
    "Coding_engine",   # old copy under spaces/coding/Coding_engine/
    "_archive",        # coding-engine/_archive/ + similar
    "all_services",    # coding-engine/Data/all_services/ (generated artefacts)
    # ── 2026-07-14: generated/duplicate trees that dominated top-K ──
    "graphify-out",         # 30k chunks from one generated graph.json (35% of index!)
    "temp-merge-parking",   # duplicated Automation_ui tree
]

# ---------------------------------------------------------------------------
# Lazy / background loading — MCP startup and handshake stay lightweight.
# The first tool call that actually needs the retriever starts one daemon
# loader; later queries share the same ready event.
# ---------------------------------------------------------------------------
import threading as _threading

_index_meta: dict = {}
_retriever = None
_ready_event = _threading.Event()  # set when retriever + heavy indexes are ready

import numpy as _np  # for hybrid math
from embeddinggemma.bm25_lite import BM25Lite as _BM25Lite

_bm25: _BM25Lite | None = None
try:
    _bm25_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".fungus_cache", "bm25.npz",
    )
    if os.path.exists(_bm25_path):
        _bm25 = _BM25Lite.load(_bm25_path)
        logger.info("BM25 hybrid ready: vocab=%d docs=%d", len(_bm25.doc_freq), _bm25.N)
    else:
        logger.info("BM25 hybrid disabled (no bm25.npz at %s)", _bm25_path)
except Exception as e:
    logger.warning("BM25 load failed (continuing without hybrid): %s", e)
    _bm25 = None


def _minmax(arr: _np.ndarray) -> _np.ndarray:
    """Min-max normalise an array to [0, 1]. Returns zeros for constant input."""
    if arr.size == 0:
        return arr
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return _np.zeros_like(arr, dtype=_np.float32)
    return ((arr - mn) / (mx - mn)).astype(_np.float32)


_multivec_embs: _np.ndarray | None = None
_multivec_chunk_ids: _np.ndarray | None = None


def _background_load():
    """Load retriever + multivec in a daemon thread; sets _ready_event when done."""
    global _retriever, _index_meta, _multivec_embs, _multivec_chunk_ids

    try:
        from embeddinggemma.mcmp_rag import MCPMRetriever
        r = MCPMRetriever(
            num_agents=50,
            max_iterations=10,
            embed_batch_size=256,
        )
        t0 = time.time()
        loaded = r.load_persistent_index()
        load_time = time.time() - t0
        if loaded:
            _index_meta = {
                "docs": len(r.documents),
                "dim": r._embed_dim,
                "load_time_s": round(load_time, 2),
                "source": "persistent_cache",
            }
        else:
            _index_meta = {"docs": 0, "dim": None, "load_time_s": 0, "source": "none"}
        _retriever = r
        logger.info("Retriever ready: %s docs=%d", _index_meta["source"], _index_meta["docs"])
    except Exception as e:
        logger.warning("Retriever load failed: %s", e)
        _index_meta = {"docs": 0, "dim": None, "load_time_s": 0, "source": "error"}

    try:
        _mv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".fungus_cache", "multivec.npz",
        )
        if os.path.exists(_mv_path):
            from embeddinggemma.multivec import load_multivec as _load_mv
            _multivec_embs, _multivec_chunk_ids = _load_mv(_mv_path)
            logger.info("ColBERT-Lite ready: %d views over %d chunks",
                        _multivec_embs.shape[0],
                        int(_multivec_chunk_ids.max()) + 1)
        else:
            logger.info("ColBERT-Lite disabled (no multivec.npz)")
    except Exception as e:
        logger.warning("ColBERT-Lite load failed: %s", e)
        _multivec_embs = None
        _multivec_chunk_ids = None

    _ready_event.set()
    logger.info("Background load complete.")


_bg_thread: _threading.Thread | None = None
_bg_thread_lock = _threading.Lock()


def _start_background_load() -> None:
    """Start the heavy retriever load once, on first real query."""
    global _bg_thread
    if _ready_event.is_set():
        return
    with _bg_thread_lock:
        if _ready_event.is_set():
            return
        if _bg_thread is None or not _bg_thread.is_alive():
            _bg_thread = _threading.Thread(
                target=_background_load,
                daemon=True,
                name="fungus-bg-load",
            )
            _bg_thread.start()


def _ensure_ready(timeout: float = 300.0) -> bool:
    """Start the lazy loader and wait until it finishes (or timeout).

    The cold start loads model weights + index in ~50s.
    A 30s wait expired mid-load, so the FIRST search after a reconnect returned a
    false "Index empty". Search tools genuinely need the retriever, so they wait
    here; index_stats does NOT (it reports load progress non-blockingly)."""
    _start_background_load()
    return _ready_event.wait(timeout=timeout)


# ── Stage-8 (2026-05-25): Cross-Encoder Reranker ────────────────────────
# BM25 and bi-encoder both score query vs doc independently. A cross-encoder
# *reads them together* and predicts a relevance score — much stronger but
# 200-400ms per batch of 30 candidates. We lazy-load on first call to keep
# MCP startup fast, and cap candidates at 30 to stay within ~250ms p95.
_RERANKER_MODEL = os.environ.get("FUNGUS_RERANKER",
                                 "BAAI/bge-reranker-base")  # 280MB, fast
_reranker = None
_reranker_load_failed = False


def _get_reranker():
    """Lazy-load the cross-encoder. Returns None on failure (graceful degrade)."""
    global _reranker, _reranker_load_failed
    if _reranker is not None:
        return _reranker
    if _reranker_load_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder
        device = RERANKER_DEVICE if RERANKER_DEVICE in ("cuda", "cpu") else "auto"
        # max_length=512 matches our chunk window (~512 tokens for code).
        _reranker = CrossEncoder(_RERANKER_MODEL, max_length=512, device=device)
        logger.info("Reranker loaded: %s on %s", _RERANKER_MODEL, device)
        return _reranker
    except Exception as e:
        logger.warning("Reranker load failed (continuing without rerank): %s", e)
        _reranker_load_failed = True
        return None


def _rerank_cross_encoder(query: str, items: list[dict], top_n: int = 30) -> list[dict]:
    """Score top_n items with cross-encoder, blend with prior score 60/40.

    Mutates items in place: sets `_ce_score` and updates `relevance_score`.
    Returns the input list re-sorted; never raises (graceful degradation).
    """
    ce = _get_reranker()
    if ce is None or not items:
        return items
    top = items[:top_n]
    # Strip our path-boost header for cleaner cross-encoder input — keep
    # only the natural-language token line + actual code body.
    pairs = []
    for it in top:
        content = it.get("content", "")
        # keep the "# tokens:" line (helps the model) + body
        kept_lines = []
        for ln in content.split("\n"):
            if ln.startswith("# file:"):
                continue
            if ln.startswith("# path:"):
                continue
            kept_lines.append(ln)
        ce_text = "\n".join(kept_lines).strip()
        if len(ce_text) > 2000:  # keep it tight for speed
            ce_text = ce_text[:2000]
        pairs.append((query, ce_text))
    try:
        scores = ce.predict(pairs, batch_size=16, show_progress_bar=False)
    except Exception as e:
        logger.warning("Reranker predict failed: %s", e)
        return items
    # Min-max normalise CE scores into [0,1] for blending.
    arr = _np.array(scores, dtype=_np.float32)
    ce_n = _minmax(arr)
    prior_n = _minmax(_np.array([float(it.get("relevance_score", 0)) for it in top],
                                dtype=_np.float32))
    for i, it in enumerate(top):
        it["_ce_score"] = float(arr[i])
        # Blend: 60% cross-encoder, 40% prior (hybrid+rerank score).
        # CE is much stronger but we keep prior as a safety net against weird CE outputs.
        it["relevance_score"] = float(0.6 * ce_n[i] + 0.4 * prior_n[i])
    top.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    # Append any items beyond top_n at the end, unchanged.
    return top + items[top_n:]


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "la-fungus-search",
    instructions="Semantic code search over vibemind-os using MCMP-RAG (multi-agent pheromone simulation + vector embeddings)",
)


def _sync_search(query: str, top_k: int, alpha: float = 0.65) -> dict:
    """Hybrid search: alpha*cosine + (1-alpha)*BM25 over a top-N candidate pool.

    alpha=1.0 → pure semantic (old behaviour). alpha=0.0 → pure BM25.
    Default 0.65 prefers semantic but lets BM25 break ties on rare-token queries.
    Falls back to pure semantic if BM25 isn't loaded.
    """
    _ensure_ready()
    if _retriever is None or not _retriever.documents:
        return {"results": []}
    if _bm25 is None or alpha >= 0.999:
        return _retriever.search_direct(query, top_k=top_k)

    # Step 1: fetch a wide semantic pool (top 200 by cosine).
    POOL = 200
    pool = _retriever.search_direct(query, top_k=POOL).get("results", [])
    if not pool:
        return {"results": []}

    # Step 2: map content → doc id (chunks.json index). Document ordering in
    # _retriever.documents matches BM25 fit order, so we use position.
    # Build a content→id map once and cache on the retriever.
    if not hasattr(_retriever, "_content_to_id"):
        _retriever._content_to_id = {
            d.content: d.id for d in _retriever.documents
        }
    cmap: dict = _retriever._content_to_id
    cand_ids = _np.array(
        [cmap.get(it.get("content", ""), -1) for it in pool],
        dtype=_np.int32,
    )

    # Step 3: BM25 over just the candidate set (fast: ~200 lookups).
    bm_raw = _bm25.score(query, candidate_ids=cand_ids)

    # Step 4: normalise both score streams to [0,1] then mix.
    sem_raw = _np.array([float(it.get("relevance_score", 0)) for it in pool],
                        dtype=_np.float32)
    sem_n = _minmax(sem_raw)
    bm_n = _minmax(bm_raw)
    fused = alpha * sem_n + (1.0 - alpha) * bm_n

    # Step 5: replace relevance_score with fused score, sort, slice.
    for i, it in enumerate(pool):
        it["relevance_score"] = float(fused[i])
        it["_sem_score"] = float(sem_raw[i])
        it["_bm25_score"] = float(bm_raw[i])
    pool.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return {"results": pool[:top_k]}


# ── Opt-Stage-3 (2026-05-25): file-type boost ─────────────────────────────
# Code queries (what users mostly want) were losing to data/doc files
# (capabilities.yaml, mcp_tools_cache.json, design markdown) because those
# embeddings happen to be longer/richer. We bias the score by file type
# *after* the FAISS search, then re-sort + re-cut top_k. Cheap, no reindex.
_CODE_EXT = (".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".swift", ".go", ".java", ".cpp", ".c", ".h")
_DOC_EXT  = (".md", ".txt", ".rst")
_DATA_EXT = (".yaml", ".yml", ".json", ".toml", ".xml", ".csv")
_TEST_HINTS = ("/tests/", "\\tests\\", "/test_", "\\test_", "_test.py", "test_")
_BOOST_CODE  = 1.0    # neutral
_BOOST_DOC   = 0.85   # markdown is fine for context, not as primary answer
_BOOST_DATA  = 0.65   # YAML/JSON usually = data registries, not code answers
_BOOST_TEST  = 0.80   # tests touch the code but aren't the code itself
_HEADER_RE = re.compile(r"# file: (.+?) \| lines:")


def _filepath_from_chunk(content: str) -> str:
    m = _HEADER_RE.search(content)
    return m.group(1) if m else ""


# Python-only indicators in the query → push .py results, push others down
_PY_ONLY_HINTS = ("importlib", "import_module", "__import__", "sys.modules",
                  "asyncio.", "self.", "def ", "class ", "pytest", "fastapi",
                  "pydantic", "django", "flask", "celery", "sqlalchemy")
_JS_ONLY_HINTS = ("typescript", "tsx", "jsx", "react", "vue", "npm",
                  "package.json", "useeffect", "usestate")


def _language_bias(query_lower: str) -> dict[str, float]:
    """Detect language hint in query; return per-extension multiplier overrides."""
    py = any(h in query_lower for h in _PY_ONLY_HINTS)
    js = any(h in query_lower for h in _JS_ONLY_HINTS)
    overrides: dict[str, float] = {}
    if py and not js:
        overrides[".py"] = 1.15
        overrides[".ts"] = 0.55
        overrides[".tsx"] = 0.55
        overrides[".js"] = 0.55
        overrides[".jsx"] = 0.55
    elif js and not py:
        overrides[".py"] = 0.55
        overrides[".ts"] = 1.10
        overrides[".tsx"] = 1.10
    return overrides


def _file_type_boost(path: str, lang_overrides: dict[str, float] | None = None) -> float:
    p = path.lower()
    boost = _BOOST_CODE
    if p.endswith(_DATA_EXT):
        boost = _BOOST_DATA
    elif p.endswith(_DOC_EXT):
        boost = _BOOST_DOC
    elif p.endswith(_CODE_EXT):
        boost = _BOOST_CODE
    # Language bias from query (e.g. "importlib" → favor .py, penalize .ts)
    if lang_overrides:
        for ext, mul in lang_overrides.items():
            if p.endswith(ext):
                boost *= mul
                break
    # Tests are deprioritised vs production code
    if any(h in p for h in _TEST_HINTS):
        boost *= _BOOST_TEST
    return boost


def _query_tokens(query: str) -> set[str]:
    """Lowercase content-word tokens of length >= 3, no stopwords."""
    stop = {"the", "and", "for", "with", "from", "this", "that", "which",
            "what", "where", "how", "who", "are", "is", "was", "were", "has",
            "have", "had", "can", "use", "all", "any", "not", "yes", "but",
            "into", "out", "by", "to", "in", "on", "at", "of"}
    toks = re.findall(r"[a-zA-Z_]{3,}", query.lower())
    return {t for t in toks if t not in stop}


def _path_token_bonus(path: str, qtoks: set[str]) -> float:
    """Multiplier 1.0–1.5 based on how many query tokens match the file path stem + parent dirs.

    The path tokens are split by /, \\, _, -, ., and camelCase. Each query
    token that matches a path token adds 0.10 (capped at +0.50). This makes
    a query for "capability router dispatch" prefer brain/core/capability_router.py
    over brain/core/routing_matrix_autotrain.py even when bodies are similar.
    """
    if not qtoks or not path:
        return 1.0
    parts = re.split(r"[\\/_\-. ]|(?<=[a-z0-9])(?=[A-Z])", path.lower())
    ptoks = {p for p in parts if len(p) >= 3}
    overlap = len(qtoks & ptoks)
    return min(1.8, 1.0 + 0.15 * overlap)


def _chunk_body(content: str) -> str:
    """Strip path-boost header lines (# file:, # path:, # tokens:) — return body."""
    lines = content.split("\n")
    return "\n".join(l for l in lines if not l.startswith("# file:")
                     and not l.startswith("# path:")
                     and not l.startswith("# tokens:")).strip()


def _dedup_path_family(items: list[dict], max_per_family: int = 2,
                       max_per_body: int = 1,
                       max_per_pattern: int = 2) -> list[dict]:
    """Three-tier dedup:
       1) keep at most N chunks per (first-3-path-segments) family
       2) keep only 1 chunk per identical body (boilerplate copy-paste defence)
       3) keep at most M chunks with the same AST pattern signature (so
          5 different files using identical `spec_from_file_location(X, X)`
          collapse to 2 representatives)
    """
    keep: list[dict] = []
    family_counts: dict[str, int] = {}
    seen_bodies: dict[str, int] = {}
    pattern_counts: dict[str, int] = {}
    for it in items:
        content = it.get("content", "")
        path = _filepath_from_chunk(content)
        body_key = hashlib.md5(_chunk_body(content).encode("utf-8", "replace")).hexdigest()
        if seen_bodies.get(body_key, 0) >= max_per_body:
            continue
        parts = [p for p in re.split(r"[\\/]", path) if p and ".." != p]
        if "worktrees" in parts:
            wi = parts.index("worktrees")
            parts = parts[:wi-1] + parts[wi+2:]
        family = "/".join(parts[:3])
        if family_counts.get(family, 0) >= max_per_family:
            continue
        # AST-pattern dedup (only applies when the chunk has an _ast_pattern_sig)
        ast_sig = it.get("_ast_pattern_sig")
        if ast_sig and pattern_counts.get(ast_sig, 0) >= max_per_pattern:
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        seen_bodies[body_key] = seen_bodies.get(body_key, 0) + 1
        if ast_sig:
            pattern_counts[ast_sig] = pattern_counts.get(ast_sig, 0) + 1
        keep.append(it)
    return keep


def _rerank_results(items: list[dict], prefer_code: bool = True,
                    query: str = "") -> list[dict]:
    """Apply file-type boost + path-token bonus, dedupe families, re-sort."""
    if not prefer_code:
        return items
    qtoks = _query_tokens(query)
    lang_overrides = _language_bias(query.lower())
    for it in items:
        path = _filepath_from_chunk(it.get("content", ""))
        raw = float(it.get("relevance_score", 0))
        boosted = raw * _file_type_boost(path, lang_overrides) * _path_token_bonus(path, qtoks)
        it["_raw_score"] = raw
        it["relevance_score"] = boosted
    items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return _dedup_path_family(items)


@mcp.tool()
async def fungus_search(query: str, top_k: int = 10, prefer_code: bool = True,
                        alpha: float = 0.65, rerank: bool = False) -> str:
    """Search the vibemind-os codebase using hybrid semantic + BM25 (+ optional cross-encoder rerank).

    Returns the top-k most relevant code chunks for a natural language query.
    Each result includes the file path, line range, relevance score, and code content.

    Args:
        query: Natural language search query (e.g. "WebSocket real-time messaging handler")
        top_k: Number of results to return (default 10, max 30)
        prefer_code: If True (default), re-rank to prefer source code over
            data files (yaml/json) and docs (markdown). Set False to see
            raw embedding ranking (useful when explicitly searching docs).
        alpha: Hybrid weight in [0,1]. 1.0 = pure semantic (old behaviour),
            0.0 = pure BM25 (keyword), default 0.65 = semantic-dominant
            with BM25 tie-break (best for rare-token queries like
            "importlib" or specific function names).
        rerank: If True, run a cross-encoder (bge-reranker-base) over the
            top-30 candidates as a final re-rank stage. +200-400ms latency
            but much higher precision on naming-gap queries. First call
            triggers a one-time model download (~280MB). Default False.
    """
    _ensure_ready()
    if not _retriever or not _retriever.documents:
        return "ERROR: No index loaded. Run the fungus_reindex tool first."

    top_k = min(max(1, top_k), 30)
    # Fetch a wider pool so re-ranking + family-dedup have room to push code up.
    # When reranking, fetch even wider so the cross-encoder has 30 candidates.
    fetch_k = min(80, top_k * 8) if prefer_code else top_k
    if rerank:
        fetch_k = max(fetch_k, 30)
    t0 = time.time()
    results = await asyncio.to_thread(_sync_search, query, fetch_k, alpha)
    search_time = time.time() - t0

    items = _rerank_results(results.get("results", []), prefer_code, query=query)
    if rerank:
        items = await asyncio.to_thread(_rerank_cross_encoder, query, items, 30)
    items = items[:top_k]
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
    _ensure_ready()
    if not _retriever or not _retriever.documents:
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


# ═════════════════════════════════════════════════════════════════════════
# Stage-6 (Ant-Colony) + Stage-7 (Query-Expansion) + Stage-5 (LLM-Judge)
# ═════════════════════════════════════════════════════════════════════════

# --- fixed OpenFang LLM paths (used by stages 5, 7, and 13) ------------
def _generate_summary(prompt: str) -> str:
    """Generate through the fixed OpenFang fungus_summary role."""
    from embeddinggemma.rag.generation import generate_text

    return generate_text(prompt=prompt)


def _generate_judge(prompt: str) -> str:
    """Generate through the fixed OpenFang fungus_judge role."""
    from embeddinggemma.rag.generation import generate_judge_text

    return generate_judge_text(prompt=prompt)


def _strip_json(text: str) -> str:
    """Extract the first JSON object/array from a (possibly chatty) LLM response."""
    # find first { or [, then last matching brace
    s = text.find("{")
    a = text.find("[")
    if s == -1 and a == -1:
        return ""
    start = a if (s == -1 or (a != -1 and a < s)) else s
    closer = "]" if text[start] == "[" else "}"
    end = text.rfind(closer)
    if end <= start:
        return ""
    return text[start:end + 1]


# ── Stage-6: deep / ant-colony / MCMP search ────────────────────────────
@mcp.tool()
async def fungus_search_deep(query: str, top_k: int = 10, n_steps: int = 8,
                             alpha: float = 0.65) -> str:
    """Multi-hop semantic search via ant-colony pheromone simulation.

    Use when a single-shot search isn't enough and you want the engine to
    explore the *neighbourhood* of the strongest hits — agents walk the
    embedding sphere, deposit pheromones on relevant documents, and reinforce
    multi-hop trails between conceptually-linked chunks. This surfaces
    concept clusters that pure cosine misses.

    Slower (~1-3 s vs 50 ms for fungus_search) but better for "show me everything
    around concept X" or "how are X and Y connected" type questions.

    Args:
        query: Natural-language query.
        top_k: How many top documents to return after the simulation.
        n_steps: How many simulation steps to run (default 8; more = deeper,
            slower; 4-12 is the sweet spot).
        alpha: Hybrid semantic/BM25 weight applied BEFORE the simulation
            (to seed agents near high-quality starting points).
    """
    _ensure_ready()
    if not _retriever or not _retriever.documents:
        return "ERROR: No index loaded."

    top_k = min(max(1, top_k), 30)
    n_steps = min(max(1, n_steps), 20)

    def _deep():
        t0 = time.time()
        # Seed: run hybrid first so doc.relevance_score reflects initial fit.
        # This gives the colony a better starting heat-map than random.
        seed_results = _sync_search(query, top_k=200, alpha=alpha)
        seed_pool = seed_results.get("results", [])
        seed_ids = set()
        if hasattr(_retriever, "_content_to_id"):
            cmap = _retriever._content_to_id
            for it in seed_pool[:50]:
                doc_id = cmap.get(it.get("content", ""), -1)
                if doc_id >= 0:
                    seed_ids.add(doc_id)
                    _retriever.documents[doc_id].relevance_score = float(it.get("relevance_score", 0))

        # Initialise simulation: spawn agents around query embedding.
        if not _retriever.initialize_simulation(query):
            return {"results": [], "error": "simulation init failed"}
        # Run the pheromone walk.
        sim_stats = _retriever.step(n_steps)
        # Rank by accumulated relevance_score (visits × pheromone).
        ranked = sorted(_retriever.documents,
                        key=lambda d: d.relevance_score, reverse=True)[:top_k * 3]
        items = [{
            "content": d.content,
            "metadata": d.metadata,
            "relevance_score": float(d.relevance_score),
            "_visits": int(d.visit_count),
        } for d in ranked]
        return {"results": items, "elapsed": time.time() - t0, "sim": sim_stats}

    t0 = time.time()
    out = await asyncio.to_thread(_deep)
    elapsed = time.time() - t0
    if "error" in out:
        return f"ERROR: {out['error']}"
    items = _rerank_results(out["results"], prefer_code=True, query=query)[:top_k]
    lines = [f"## Deep (ant-colony) Search: \"{query}\" "
             f"({len(items)} results, {elapsed*1000:.0f}ms, "
             f"{out.get('sim',{}).get('pheromone_trails',0)} trails)\n"]
    for i, it in enumerate(items, 1):
        score = it.get("relevance_score", 0)
        content = it.get("content", "")
        m = _HEADER_RE.search(content)
        if m:
            filepath = m.group(1).replace("\\", "/")
            body = "\n".join(content.split("\n")[1:]).strip()
        else:
            filepath, body = "unknown", content.strip()
        if len(body) > 600:
            body = body[:600] + "\n... (truncated)"
        lines.append(f"### {i}. {filepath} — score: {score:.3f}, visits: {it.get('_visits', 0)}")
        lines.append(f"```\n{body}\n```\n")
    return "\n".join(lines)


# ── Stage-7: query expansion via LLM ────────────────────────────────────
@mcp.tool()
async def fungus_search_expanded(query: str, top_k: int = 10,
                                 n_subqueries: int = 4) -> str:
    """Expand the user query into N sub-queries via LLM, then merge results.

    The OpenFang summary role rewrites the query into
    several angles ("synonyms", "related concepts", "specific identifiers",
    "alternative wordings") so that the embedder + BM25 catch matches that
    a single phrasing would miss. Especially helpful when the user query
    uses one term but the code uses another (face_landmark vs FaceMesh,
    capability dispatcher vs route() method).

    Args:
        query: The original user query.
        top_k: How many merged top results to return.
        n_subqueries: How many sub-queries to generate (3-6 is typical).
    """
    _ensure_ready()
    if not _retriever or not _retriever.documents:
        return "ERROR: No index loaded."

    top_k = min(max(1, top_k), 30)
    n_subqueries = min(max(2, n_subqueries), 6)

    # Step 1: ask LLM for sub-queries (very small prompt).
    expand_prompt = (
        f"You are a code-search query rewriter. Generate exactly {n_subqueries} "
        f"alternative search queries for the original query below. Vary the angle: "
        f"include synonyms, exact identifier names, related concepts, alternative "
        f"phrasings. Return ONLY a JSON array of strings, nothing else.\n\n"
        f"Original query: {query}\n\nJSON array:"
    )
    raw = await asyncio.to_thread(_generate_summary, expand_prompt)
    sub_queries: list[str] = []
    js = _strip_json(raw)
    try:
        parsed = json.loads(js) if js else []
        if isinstance(parsed, list):
            sub_queries = [str(q).strip() for q in parsed if str(q).strip()]
    except Exception:
        sub_queries = []
    if not sub_queries:
        sub_queries = [query]
    # Always include the original — protects against bad LLM expansions.
    if query not in sub_queries:
        sub_queries.insert(0, query)

    # Step 2: run each sub-query (hybrid), merge by max-fused-score.
    def _multi():
        merged: dict[str, dict] = {}  # content_key → best item
        for q in sub_queries[:n_subqueries + 1]:
            for it in _sync_search(q, top_k=20).get("results", []):
                key = it.get("content", "")[:200]  # first 200 chars as id
                prev = merged.get(key)
                if prev is None or it.get("relevance_score", 0) > prev.get("relevance_score", 0):
                    merged[key] = it
        return list(merged.values())

    t0 = time.time()
    merged = await asyncio.to_thread(_multi)
    items = _rerank_results(merged, prefer_code=True, query=query)[:top_k]
    elapsed = time.time() - t0

    lines = [f"## Expanded Search: \"{query}\" ({len(items)} results, {elapsed*1000:.0f}ms)"]
    lines.append(f"**Sub-queries ({len(sub_queries)}):** " + " · ".join(
        f"`{q}`" for q in sub_queries[:6]))
    lines.append("")
    for i, it in enumerate(items, 1):
        score = it.get("relevance_score", 0)
        content = it.get("content", "")
        m = _HEADER_RE.search(content)
        filepath = m.group(1).replace("\\", "/") if m else "unknown"
        body = "\n".join(content.split("\n")[1:]).strip()
        if len(body) > 500:
            body = body[:500] + "\n... (truncated)"
        lines.append(f"### {i}. {filepath} — score: {score:.3f}")
        lines.append(f"```\n{body}\n```\n")
    return "\n".join(lines)


# ── Stage-5: LLM judge / steering ───────────────────────────────────────
@mcp.tool()
async def fungus_search_judged(query: str, top_k: int = 10,
                               candidates: int = 25) -> str:
    """Hybrid search + LLM judge: model scores each candidate for relevance
    and entry-point status, then we re-rank by combined score.

    This emulates the Explore-agent pattern: fungus produces a wide candidate
    pool fast and cheap, then a small LLM acts as the "judge" — for each
    candidate it answers is_relevant, is_entry_point, why, and suggests
    follow-up queries you should run next. Slower than a plain search
    (~5-15s for 25 candidates) but precision-recall-wise the best mode.

    Args:
        query: The user query.
        top_k: How many results to return after judging.
        candidates: How many hybrid candidates to send to the LLM (10-30 typical).
    """
    _ensure_ready()
    if not _retriever or not _retriever.documents:
        return "ERROR: No index loaded."

    top_k = min(max(1, top_k), 20)
    candidates = min(max(5, candidates), 40)

    # Step 1: hybrid search to get a candidate pool.
    pool_raw = _sync_search(query, top_k=candidates * 2)
    pool = _rerank_results(pool_raw.get("results", []), prefer_code=True,
                           query=query)[:candidates]
    if not pool:
        return f"No candidates found for: {query}"

    # Step 2: build a compact list for the LLM judge.
    items_for_llm = []
    for i, it in enumerate(pool):
        content = it.get("content", "")
        m = _HEADER_RE.search(content)
        filepath = m.group(1) if m else "unknown"
        body = "\n".join(content.split("\n")[1:]).strip()[:800]
        items_for_llm.append({
            "id": i,
            "file": filepath,
            "score": round(float(it.get("relevance_score", 0)), 3),
            "code": body,
        })

    judge_prompt = (
        f"You are evaluating code chunks for relevance to a developer query.\n"
        f"For each chunk, decide:\n"
        f"  - is_relevant (bool): does this chunk actually answer or contain logic that answers the query?\n"
        f"  - is_entry_point (bool): is this the primary/canonical place a developer should land?\n"
        f"  - why (string, max 1 short sentence)\n"
        f"Return ONLY JSON in this exact shape: "
        f'{{"items":[{{"id":<int>,"is_relevant":<bool>,"is_entry_point":<bool>,"why":"<str>"}}]}}.\n\n'
        f"Query: {query}\n\nChunks:\n"
        f"{json.dumps(items_for_llm, ensure_ascii=False)[:24000]}\n\nJSON:"
    )

    t0 = time.time()
    raw = await asyncio.to_thread(_generate_judge, judge_prompt)
    js = _strip_json(raw)
    judgements: dict[int, dict] = {}
    try:
        parsed = json.loads(js) if js else {}
        for it in parsed.get("items", []):
            judgements[int(it.get("id", -1))] = it
    except Exception as e:
        logger.warning("judge JSON parse failed: %s", e)

    # Step 3: re-rank by judge — entry_point > relevant > anything else.
    def _final_score(i: int, base: float) -> float:
        j = judgements.get(i, {})
        bump = 0.0
        if j.get("is_entry_point"):
            bump += 0.5
        if j.get("is_relevant"):
            bump += 0.3
        if j and not j.get("is_relevant"):
            bump -= 0.4
        return base + bump

    for i, it in enumerate(pool):
        it["_judge"] = judgements.get(i, {})
        it["relevance_score"] = _final_score(i, float(it.get("relevance_score", 0)))
    pool.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    final = pool[:top_k]
    elapsed = time.time() - t0

    lines = [f"## Judged Search: \"{query}\" ({len(final)} results, "
             f"{elapsed:.1f}s, {len(judgements)} judged)\n"]
    for i, it in enumerate(final, 1):
        score = it.get("relevance_score", 0)
        j = it.get("_judge", {})
        content = it.get("content", "")
        m = _HEADER_RE.search(content)
        filepath = m.group(1).replace("\\", "/") if m else "unknown"
        body = "\n".join(content.split("\n")[1:]).strip()
        if len(body) > 500:
            body = body[:500] + "\n... (truncated)"
        tags = []
        if j.get("is_entry_point"):
            tags.append("ENTRY")
        if j.get("is_relevant"):
            tags.append("RELEVANT")
        tag_str = " ".join(f"[{t}]" for t in tags)
        why = j.get("why", "")
        lines.append(f"### {i}. {filepath} — score: {score:.3f} {tag_str}")
        if why:
            lines.append(f"_judge: {why}_")
        lines.append(f"```\n{body}\n```\n")
    return "\n".join(lines)


# ── Stage-11+12: AST-Scan + LLM Re-Search Loop = "Validated Search" ──
@mcp.tool()
async def fungus_search_validated(query: str, top_k: int = 10,
                                  iterations: int = 2,
                                  use_ast: bool = True) -> str:
    """Strongest mode: AST-Scan (recall floor) + LLM re-search loop (precision).

    Combines four passes for maximum coverage on overview-style queries
    ("find all X in the codebase"):

      1. Hybrid+rerank semantic search (Stage 4+8) → initial top-K
      2. AST-Scan (Stage 12) — if query has syntactic hints (importlib,
         getattr, etc.), deterministically scan all .py files for those
         patterns. Guarantees recall: no file with the construct gets missed.
      3. LLM Validator (Stage 11) — given the current top-K, asks the LLM
         "what's missing? what other angles would surface more matches?"
         and gets back N follow-up queries.
      4. Re-search with those queries via hybrid+rerank, merge into pool.

    Two iterations by default. Slower than fungus_search (~5-15s) but the
    most thorough mode — recommended for "list all"/"find every" queries
    that need high recall.

    Args:
        query: User query.
        top_k: How many results to return.
        iterations: How many re-search rounds the LLM can request (1-3).
        use_ast: If True (default), enable AST-Scan for syntactic queries.
    """
    _ensure_ready()
    if not _retriever or not _retriever.documents:
        return "ERROR: No index loaded."

    top_k = min(max(1, top_k), 30)
    iterations = min(max(1, iterations), 3)

    def _initial_pool() -> list[dict]:
        # Hybrid + cross-encoder rerank for the starting pool.
        results = _sync_search(query, top_k=80, alpha=0.65)
        items = _rerank_results(results.get("results", []),
                                prefer_code=True, query=query)
        items = _rerank_cross_encoder(query, items, 30)
        return items

    def _ast_pool() -> list[dict]:
        if not use_ast:
            return []
        try:
            from embeddinggemma.ast_scan import (
                pick_detector_for_query, scan_with_detector,
            )
        except Exception as e:
            logger.warning("ast_scan import failed: %s", e)
            return []
        det = pick_detector_for_query(query)
        if det is None:
            return []
        detector, label = det
        logger.info("[validated] AST scan triggered (detector=%s)", label)
        # Use a focused exclude list (we want voice/python included unlike
        # the index, since the index has it as a junction-deduplicated copy).
        ast_excludes = [d for d in EXCLUDE_DIRS]
        hits = scan_with_detector(CODEBASE, detector, ast_excludes,
                                  max_files=30000, context_lines=4)
        # Each AST hit gets a stable high score so it survives merge cuts,
        # but rerank can still re-order them by relevance to query terms.
        for h in hits:
            h["relevance_score"] = 1.0
        # Apply our standard rerank chain so file-type/path-token boosts
        # still kick in (this also dedupes families/bodies).
        hits = _rerank_results(hits, prefer_code=True, query=query)
        return hits[:80]  # cap to keep merge tractable

    t0 = time.time()
    pool = await asyncio.to_thread(_initial_pool)
    ast_hits = await asyncio.to_thread(_ast_pool)

    def _merge(existing: list[dict], new_hits: list[dict]) -> list[dict]:
        """Union by content-prefix, keeping the max relevance_score."""
        seen: dict[str, dict] = {}
        for it in existing + new_hits:
            key = it.get("content", "")[:200]
            prev = seen.get(key)
            if prev is None or it.get("relevance_score", 0) > prev.get("relevance_score", 0):
                seen[key] = it
        return list(seen.values())

    pool = _merge(pool, ast_hits)

    # ── LLM re-search loop ─────────────────────────────────────────────
    rounds_done = 0
    followup_used: list[str] = []
    for round_i in range(iterations):
        # Snapshot of current top-K (for the LLM)
        cur_top = sorted(pool, key=lambda x: x.get("relevance_score", 0),
                         reverse=True)[:min(15, len(pool))]
        items_for_llm = []
        for i, it in enumerate(cur_top):
            content = it.get("content", "")
            m = _HEADER_RE.search(content)
            filepath = m.group(1) if m else "unknown"
            body = "\n".join(content.split("\n")[1:]).strip()[:400]
            items_for_llm.append({"id": i, "file": filepath, "snippet": body})

        # Collect file paths currently in top-K so the LLM can see what
        # directories are over-represented (and conversely, which ones aren't).
        seen_dirs = set()
        for it in items_for_llm:
            f = it.get("file", "")
            parts = f.replace("\\", "/").split("/")
            if len(parts) > 2:
                seen_dirs.add("/".join(parts[:3]))

        validator_prompt = (
            f"You are a code-search recall auditor for the Vibemind codebase "
            f"(a Python+TypeScript repo with subsystems: brain/, voice/, "
            f"spaces/, coding-engine/, openclaw/, openfang/, security/, "
            f"bridge/, skills/, ops/). All hits MUST be inside this repo — "
            f"never propose generic queries like 'Django', 'Flask', "
            f"'tutorial', 'how to'.\n\n"
            f"User query: {query}\n\n"
            f"Current top-{len(cur_top)} hits (their file paths):\n"
            f"{json.dumps(items_for_llm, ensure_ascii=False)[:12000]}\n\n"
            f"Directories already represented: {sorted(seen_dirs)[:10]}\n\n"
            f"Task: identify GAPS in coverage. Look at the current results "
            f"and ask: which SUBSYSTEMS or NAMING CONVENTIONS are missing? "
            f"Examples of good follow-up queries for an 'importlib' query:\n"
            f"  - 'tool_registry agent loader backend_agents'\n"
            f"  - 'bindings_registry __import__ swarm'\n"
            f"  - 'capability_executor DirectExecutor import_module'\n"
            f"  - 'plugin discovery spec_from_file_location'\n"
            f"Each follow-up MUST use vocabulary that would appear inside "
            f"the Vibemind code — file names, class names, function names. "
            f"Do NOT propose framework names not present in the repo.\n\n"
            f"Return ONLY JSON: "
            f'{{"gaps_identified": <bool>, '
            f'"why": "<one short sentence naming the missing subsystem>", '
            f'"followup_queries": ["<query1>", "<query2>", ...]}}\n'
            f"Generate 0-3 follow-up queries.\n\nJSON:"
        )

        raw = await asyncio.to_thread(_generate_judge, validator_prompt)
        js = _strip_json(raw)
        followups: list[str] = []
        gap_reason = ""
        try:
            parsed = json.loads(js) if js else {}
            if parsed.get("gaps_identified"):
                gap_reason = str(parsed.get("why", ""))[:200]
                followups = [str(q).strip()
                             for q in parsed.get("followup_queries", [])
                             if str(q).strip() and str(q).strip() != query
                             and str(q).strip() not in followup_used][:3]
        except Exception as e:
            logger.warning("[validated] validator JSON parse failed: %s", e)

        if not followups:
            logger.info("[validated] round %d: no gaps identified, stopping", round_i + 1)
            break
        logger.info("[validated] round %d: %d follow-ups (%s)",
                    round_i + 1, len(followups), gap_reason)
        for fq in followups:
            followup_used.append(fq)
            fq_results = await asyncio.to_thread(_sync_search, fq, 30, 0.65)
            fq_items = _rerank_results(fq_results.get("results", []),
                                       prefer_code=True, query=fq)[:20]
            pool = _merge(pool, fq_items)
        rounds_done += 1

    # ── Final cross-encoder pass on the merged pool ────────────────────
    pool.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    pool = _rerank_cross_encoder(query, pool, min(40, len(pool)))
    final = pool[:top_k]
    elapsed = time.time() - t0

    lines = [f"## Validated Search: \"{query}\" "
             f"({len(final)} results, {elapsed:.1f}s, "
             f"AST={len(ast_hits)} hits, "
             f"LLM rounds={rounds_done}, follow-ups={len(followup_used)})"]
    if followup_used:
        lines.append("**LLM follow-ups used:** "
                     + " · ".join(f"`{q}`" for q in followup_used[:5]))
    lines.append("")
    for i, it in enumerate(final, 1):
        score = it.get("relevance_score", 0)
        ast_tag = " [AST]" if it.get("_ast_match") else ""
        ce_tag = f" ce={it.get('_ce_score', 0):.2f}" if "_ce_score" in it else ""
        content = it.get("content", "")
        m = _HEADER_RE.search(content)
        filepath = m.group(1).replace("\\", "/") if m else "unknown"
        body = "\n".join(content.split("\n")[1:]).strip()
        if len(body) > 500:
            body = body[:500] + "\n... (truncated)"
        lines.append(f"### {i}. {filepath} — score: {score:.3f}{ast_tag}{ce_tag}")
        lines.append(f"```\n{body}\n```\n")
    return "\n".join(lines)


# ── Stage-13: LLM synthesizer on top of fungus_search_validated ────────
@mcp.tool()
async def fungus_search_synthesized(query: str, top_k: int = 10,
                                    iterations: int = 2,
                                    use_ast: bool = True) -> str:
    """Strongest mode + LLM synthesis: pipe fungus_search_validated through
    an LLM that produces a *curated* overview — exactly like an Explore agent.

    The validated pipeline (AST recall floor + LLM re-search + cross-encoder
    rerank) gets a high-recall pool. This tool then asks an LLM to:
      - dedupe near-duplicate patterns (e.g. 4× identical load_prompt_from_module
        boilerplate → 1 representative)
      - synthesize one short sentence per remaining file explaining WHAT it does
      - present the result as a numbered list, like a researcher's report

    Use this for overview-style queries:
        "find all importlib usages"
        "show me the plugin loaders in this repo"
        "where is dynamic dispatch happening"

    Args:
        query: User query.
        top_k: How many final synthesized entries to return.
        iterations: LLM re-search rounds (passed to validated pipeline).
        use_ast: Enable AST-Scan recall floor.
    """
    _ensure_ready()
    if not _retriever or not _retriever.documents:
        return "ERROR: No index loaded."
    top_k = min(max(1, top_k), 20)

    # Step 1: run the validated pipeline to get a high-recall pool.
    def _validated_pool() -> list[dict]:
        # Inline a leaner version: skip the formatted output, return raw items.
        # Same flow as fungus_search_validated but no markdown formatting.
        results = _sync_search(query, top_k=80, alpha=0.65)
        items = _rerank_results(results.get("results", []),
                                prefer_code=True, query=query)
        items = _rerank_cross_encoder(query, items, 30)
        # AST recall floor
        if use_ast:
            try:
                from embeddinggemma.ast_scan import (
                    pick_detector_for_query, scan_with_detector,
                )
                det = pick_detector_for_query(query)
                if det is not None:
                    detector, label = det
                    logger.info("[synth] AST scan: %s", label)
                    ast_hits = scan_with_detector(
                        CODEBASE, detector, EXCLUDE_DIRS,
                        max_files=30000, context_lines=4,
                    )
                    for h in ast_hits:
                        h["relevance_score"] = 1.0
                    ast_hits = _rerank_results(ast_hits, prefer_code=True, query=query)[:60]
                    # merge
                    seen = {it["content"][:200]: it for it in items}
                    for h in ast_hits:
                        k = h["content"][:200]
                        prev = seen.get(k)
                        if prev is None or h["relevance_score"] > prev["relevance_score"]:
                            seen[k] = h
                    items = list(seen.values())
            except Exception as e:
                logger.warning("[synth] AST scan failed: %s", e)
        # LLM re-search loop (same as validated)
        for round_i in range(iterations):
            cur_top = sorted(items, key=lambda x: x.get("relevance_score", 0),
                             reverse=True)[:15]
            items_for_llm = []
            for i, it in enumerate(cur_top):
                content = it.get("content", "")
                m = _HEADER_RE.search(content)
                filepath = m.group(1) if m else "unknown"
                body = "\n".join(content.split("\n")[1:]).strip()[:300]
                items_for_llm.append({"id": i, "file": filepath, "snippet": body})
            seen_dirs = set()
            for it in items_for_llm:
                parts = it.get("file", "").replace("\\", "/").split("/")
                if len(parts) > 2:
                    seen_dirs.add("/".join(parts[:3]))
            prompt = (
                f"You are a code-search recall auditor for the Vibemind repo "
                f"(subsystems: brain/, voice/, spaces/, coding-engine/, openclaw/, "
                f"openfang/, security/, bridge/, skills/, ops/). All hits MUST be "
                f"inside this repo — never propose generic queries like 'Django'.\n\n"
                f"User query: {query}\n\n"
                f"Current top-{len(cur_top)} hits:\n"
                f"{json.dumps(items_for_llm, ensure_ascii=False)[:10000]}\n\n"
                f"Directories represented: {sorted(seen_dirs)[:10]}\n\n"
                f"Identify gaps and propose 0-3 follow-up queries using "
                f"vocabulary that would appear in Vibemind code "
                f"(file/class/function names).\n\n"
                f'Return ONLY JSON: {{"gaps_identified": <bool>, '
                f'"why": "<short>", "followup_queries": [...]}}\n\nJSON:'
            )
            raw = _generate_judge(prompt)
            js = _strip_json(raw)
            followups = []
            try:
                p = json.loads(js) if js else {}
                if p.get("gaps_identified"):
                    followups = [str(q).strip() for q in p.get("followup_queries", [])
                                 if str(q).strip() and str(q).strip() != query][:3]
            except Exception:
                pass
            if not followups:
                break
            for fq in followups:
                fq_res = _sync_search(fq, 30, 0.65)
                fq_items = _rerank_results(fq_res.get("results", []),
                                           prefer_code=True, query=fq)[:20]
                seen = {it["content"][:200]: it for it in items}
                for fi in fq_items:
                    k = fi["content"][:200]
                    prev = seen.get(k)
                    if prev is None or fi["relevance_score"] > prev["relevance_score"]:
                        seen[k] = fi
                items = list(seen.values())
        items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        # Final cross-encoder pass on the merged pool
        items = _rerank_cross_encoder(query, items, min(40, len(items)))
        return items[:min(25, len(items))]  # take 25 candidates for synthesis

    t0 = time.time()
    pool = await asyncio.to_thread(_validated_pool)
    pipeline_time = time.time() - t0

    if not pool:
        return f"No results for: {query}"

    # Step 2: build the synthesis prompt — give the LLM all 25 candidates
    # with file paths + code snippet + AST signature, ask for curated top-N.
    synth_items = []
    for i, it in enumerate(pool):
        content = it.get("content", "")
        m = _HEADER_RE.search(content)
        filepath = m.group(1) if m else "unknown"
        body_lines = []
        for ln in content.split("\n"):
            if ln.startswith(("# file:", "# path:", "# tokens:")):
                continue
            body_lines.append(ln)
        body = "\n".join(body_lines).strip()[:400]
        ast_sig = it.get("metadata", {}).get("ast_pattern", "")
        synth_items.append({
            "id": i,
            "file": filepath,
            "ast_pattern": ast_sig[:80],
            "code": body,
        })

    # Pre-filter test files BEFORE we send to the LLM — keeps the prompt small
    # and prevents the LLM from picking tests as "architectural" hits.
    # User can still find tests via fungus_lookup_file("tests/test_X").
    # We match BOTH "tests/" directories AND any file whose BASENAME starts
    # with "test_" or "_test_" (catches "_test_all_pocs.py" etc.).
    def _is_test_file(path: str) -> bool:
        p = path.lower().replace("\\", "/")
        if "/tests/" in p or "/test/" in p:
            return True
        base = p.rsplit("/", 1)[-1]
        return (base.startswith("test_") or base.startswith("_test_")
                or base.endswith("_test.py"))

    is_test_query = any(t in query.lower() for t in ("test_", "pytest", "unittest", "test suite"))
    if not is_test_query:
        before = len(synth_items)
        synth_items = [it for it in synth_items
                       if not _is_test_file(it.get("file", ""))]
        if before > len(synth_items):
            logger.info("[synth] filtered %d test files (query not test-related)",
                        before - len(synth_items))

    # Pre-dedupe: keep AT MOST 1 candidate per (subsystem, ast_pattern) tuple
    # AT THE INPUT LEVEL. This makes the LLM physically incapable of returning
    # duplicates from the same file or near-identical pattern — the bad cases
    # (tool_registry.py twice, poc_os_shield twice) just don't exist as inputs.
    seen_keys: set[tuple[str, str]] = set()
    seen_files: set[str] = set()
    deduped: list[dict] = []
    for it in synth_items:
        f = it.get("file", "").replace("\\", "/")
        subsystem = ""
        for p in f.split("/"):
            if p and not p.startswith(".."):
                subsystem = p
                break
        pattern_key = (it.get("ast_pattern") or "")[:40]
        key = (subsystem, pattern_key)
        # Hard rule: each file may appear AT MOST ONCE (regardless of pattern).
        if f in seen_files:
            continue
        # And each (subsystem, pattern) at most once too.
        if pattern_key and key in seen_keys:
            continue
        seen_files.add(f)
        seen_keys.add(key)
        deduped.append(it)
    if len(deduped) < len(synth_items):
        logger.info("[synth] input dedupe: %d → %d candidates",
                    len(synth_items), len(deduped))
    synth_items = deduped

    # Collect subsystems + ast-patterns already in pool so the LLM can see
    # what diversity is achievable.
    subsystems = set()
    patterns = set()
    for it in synth_items:
        parts = it.get("file", "").replace("\\", "/").split("/")
        # First non-".." segment is the subsystem (brain, voice, etc.)
        for p in parts:
            if p and not p.startswith(".."):
                subsystems.add(p)
                break
        if it.get("ast_pattern"):
            patterns.add(it["ast_pattern"][:40])

    target_n = min(top_k, len(synth_items))

    synth_prompt = (
        f"You are a senior engineer producing a curated overview for a developer.\n\n"
        f"DEVELOPER'S QUERY: {query}\n\n"
        f"You have {len(synth_items)} candidate code chunks (already pre-filtered "
        f"for relevance and high recall, test files removed).\n\n"
        f"AVAILABLE SUBSYSTEMS in the candidates: {sorted(subsystems)[:15]}\n"
        f"AVAILABLE AST PATTERNS (sample): {sorted(patterns)[:6]}\n\n"
        f"**HARD CONSTRAINTS — VIOLATE AT YOUR PERIL:**\n"
        f"1. You MUST return EXACTLY {target_n} items in the 'items' array — "
        f"NOT fewer. If you cannot find {target_n} distinct architecturally "
        f"interesting hits, you MUST fill remaining slots with the highest-"
        f"scoring candidates you have NOT yet picked, even if similar.\n"
        f"2. **DIVERSITY RULE**: Each picked item MUST satisfy at least ONE:\n"
        f"   - Different subsystem than every previously-picked item, OR\n"
        f"   - Different AST pattern than every previously-picked item, OR\n"
        f"   - Different use-case (e.g. 'plugin loader' vs 'lazy import' vs "
        f"     'capability dispatch' vs 'config hot-reload')\n"
        f"   No two items may share BOTH subsystem AND ast_pattern.\n"
        f"3. **NO TEST FILES** — files in /tests/ or matching test_*.py are already "
        f"filtered out. Do not invent any.\n"
        f"4. **DEDUPE BOILERPLATE**: 4× identical `load_prompt_from_module` "
        f"copies → pick 1 representative, count the rest as `skipped_duplicates`.\n"
        f"5. For each picked file, write ONE specific sentence (max 25 words) "
        f"naming WHAT module/class it loads and WHY (not generic 'loads modules').\n\n"
        f"CANDIDATES:\n"
        f"{json.dumps(synth_items, ensure_ascii=False)[:18000]}\n\n"
        f"Return ONLY JSON in this exact shape:\n"
        f'{{"items": [\n'
        f'  {{"id": <int from candidates>, '
        f'"file": "<path>", '
        f'"subsystem": "<top-level dir, e.g. brain>", '
        f'"role": "<2-5 word category>", '
        f'"summary": "<one specific sentence with concrete names>"}}\n'
        f"  // EXACTLY {target_n} items required\n"
        f'  ], "skipped_duplicates": <int>}}\n\n'
        f"Order items by architectural importance (load-bearing first). JSON:"
    )

    t1 = time.time()
    raw = await asyncio.to_thread(_generate_summary, synth_prompt)
    synth_time = time.time() - t1

    js = _strip_json(raw)
    final_items = []
    skipped = 0
    try:
        parsed = json.loads(js) if js else {}
        raw_items = parsed.get("items", [])
        skipped = int(parsed.get("skipped_duplicates", 0))
        # Post-LLM dedup: even with all the constraints, models sometimes
        # double-pick. Keep first occurrence of each file path.
        seen_files: set[str] = set()
        for it in raw_items:
            f = str(it.get("file", "")).replace("\\", "/")
            if not f or f in seen_files:
                continue
            seen_files.add(f)
            final_items.append(it)
            if len(final_items) >= top_k:
                break
    except Exception as e:
        logger.warning("[synth] JSON parse failed: %s", e)
        # Fallback: return the raw pool top-N as a plain list
        final_items = []
        for it in pool[:top_k]:
            content = it.get("content", "")
            m = _HEADER_RE.search(content)
            final_items.append({
                "id": -1,
                "file": m.group(1) if m else "unknown",
                "role": "(synth fallback)",
                "summary": (it.get("content", "")[:100]),
            })

    elapsed = time.time() - t0
    lines = [
        f"## Synthesized Overview: \"{query}\"",
        f"_{len(final_items)} curated results from {len(pool)} candidates, "
        f"{elapsed:.1f}s total (pipeline {pipeline_time:.1f}s + synth {synth_time:.1f}s, "
        f"{skipped} duplicates collapsed)_\n",
    ]
    for i, fi in enumerate(final_items, 1):
        file_path = str(fi.get("file", "unknown")).replace("\\", "/")
        role = fi.get("role", "")
        summary = fi.get("summary", "")
        lines.append(f"### {i}. `{file_path}` — _{role}_")
        lines.append(f"{summary}\n")
    return "\n".join(lines)


# ── Stage-9: ColBERT-Lite multi-vector search ──────────────────────────
@mcp.tool()
async def fungus_search_multivec(query: str, top_k: int = 10,
                                 multi_query: bool = True,
                                 fuse_with_hybrid: bool = True) -> str:
    """ColBERT-Lite multi-vector search using Sum-of-MaxSim retrieval.

    Each chunk was indexed as ~3 views (header, body-first, body-second). The
    query is split into phrases; for each (query-phrase × chunk-view) pair we
    take cosine and sum the maxes per chunk. This recovers naming-gap matches
    that single-vector cosine averages away.

    Args:
        query: User query.
        top_k: How many top chunks to return.
        multi_query: If True (default), split the query into phrases and use
            Sum-of-MaxSim (true ColBERT). If False, just embed the whole query
            once and use single-vector MaxSim.
        fuse_with_hybrid: If True (default), blend 60% multivec + 40% hybrid
            (Stage-4) so the file-type bias and BM25 still apply. False = pure
            multivec ranking (useful for debugging the multivec quality).
    """
    if _multivec_embs is None or _multivec_chunk_ids is None:
        return ("ERROR: multivec index not built. Run:\n"
                "    cd vibemind-os/la-fungus-search\n"
                "    FUNGUS_DEVICE=cuda python build_multivec_index.py")
    _ensure_ready()
    if not _retriever or not _retriever.documents:
        return "ERROR: No index loaded."

    top_k = min(max(1, top_k), 30)

    def _run():
        from embeddinggemma.multivec import (
            maxsim_search, maxsim_search_multi_query, split_query_into_phrases,
        )
        t0 = time.time()
        if multi_query:
            phrases = split_query_into_phrases(query, max_phrases=6)
            q_vecs = _retriever.embedding_model.encode(
                phrases, convert_to_numpy=True, normalize_embeddings=True,
            )
            q_vecs = _np.asarray(q_vecs, dtype=_np.float32)
            ranked = maxsim_search_multi_query(
                q_vecs, _multivec_embs, _multivec_chunk_ids,
                top_k=max(top_k * 4, 40),
            )
        else:
            phrases = [query]
            q = _retriever.embedding_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True,
            )[0]
            q = _np.asarray(q, dtype=_np.float32)
            ranked = maxsim_search(q, _multivec_embs, _multivec_chunk_ids,
                                   top_k=max(top_k * 4, 40))
        elapsed = time.time() - t0
        return ranked, phrases, elapsed

    ranked, phrases, mv_time = await asyncio.to_thread(_run)

    # Build items list with proper content + scores
    items: list[dict] = []
    for chunk_id, score in ranked:
        if 0 <= chunk_id < len(_retriever.documents):
            d = _retriever.documents[chunk_id]
            items.append({
                "content": d.content,
                "metadata": d.metadata,
                "relevance_score": float(score),
                "_mv_score": float(score),
            })

    # Optional fusion with Stage-4 hybrid (gives us BM25 + file-type-boost)
    if fuse_with_hybrid and _bm25 is not None:
        hybrid_res = _sync_search(query, top_k=80, alpha=0.65)
        hybrid_pool = hybrid_res.get("results", [])
        # Build {content_key → hybrid_score}
        cmap = {it.get("content", "")[:200]: float(it.get("relevance_score", 0))
                for it in hybrid_pool}
        # Min-max normalise both score streams
        mv_arr = _np.array([it["_mv_score"] for it in items], dtype=_np.float32)
        hb_arr = _np.array([cmap.get(it["content"][:200], 0.0) for it in items],
                           dtype=_np.float32)
        mv_n = _minmax(mv_arr)
        hb_n = _minmax(hb_arr)
        for i, it in enumerate(items):
            it["_hybrid_score"] = float(hb_arr[i])
            it["relevance_score"] = float(0.6 * mv_n[i] + 0.4 * hb_n[i])
        items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Always apply our rerank tier (file-type, path-token, family-dedup)
    items = _rerank_results(items, prefer_code=True, query=query)[:top_k]

    lines = [f"## ColBERT-Lite Search: \"{query}\" "
             f"({len(items)} results, {mv_time*1000:.0f}ms"
             + (", fused" if fuse_with_hybrid else "") + ")"]
    if multi_query:
        lines.append(f"**Query phrases ({len(phrases)}):** "
                     + " · ".join(f"`{p}`" for p in phrases[:6]))
    lines.append("")
    for i, it in enumerate(items, 1):
        score = it.get("relevance_score", 0)
        mv = it.get("_mv_score", 0)
        content = it.get("content", "")
        m = _HEADER_RE.search(content)
        filepath = m.group(1).replace("\\", "/") if m else "unknown"
        body = "\n".join(content.split("\n")[1:]).strip()
        if len(body) > 500:
            body = body[:500] + "\n... (truncated)"
        lines.append(f"### {i}. {filepath} — score: {score:.3f} (mv={mv:.2f})")
        lines.append(f"```\n{body}\n```\n")
    return "\n".join(lines)


@mcp.tool()
async def fungus_lookup_file(filepath: str, top_k: int = 10) -> str:
    """Find all indexed chunks from a specific file path.

    Args:
        filepath: Partial or full file path to search for (e.g. "brain/core/radial_attention.py")
        top_k: Max chunks to return (default 10)
    """
    _ensure_ready()
    if not _retriever or not _retriever.documents:
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

    # NON-BLOCKING: do not turn a status request into the first heavy query.
    # Report whether the lazy loader has not started or is currently running.
    if not _ready_event.is_set():
        if _bg_thread is None:
            return ("Index not loaded yet (lazy start). Run a search query to "
                    "start the retriever. Meta: " + str(meta))
        return ("Index still loading (cold start ~50s: model weights + index). "
                "Retry in a moment. Meta: " + str(meta))
    if not _retriever or not _retriever.documents:
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
        "- **Embedding role**: fungus_search (OpenFang)",
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
            num_agents=50,
            max_iterations=10,
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


# ═════════════════════════════════════════════════════════════════════════
# Stage-10 (2026-05-25): Debounced file watcher for incremental reindex
# ═════════════════════════════════════════════════════════════════════════
#
# Design:
#   - Background daemon thread walks CODEBASE every 5s, tracks file mtimes.
#   - When a file changes (or a new one appears), it's queued.
#   - After 30s of quiet (no further changes), the queue is drained: each
#     queued file is re-chunked, embedded, and merged into the live index.
#   - Survives "git pull storms" (1000 files at once) because debounce gates
#     the actual embed work into one big batch instead of 1000 single embeds.
#   - State (mtimes + last_scan) is in-memory only — restart re-baselines.
#
import threading

_watcher_state = {
    "thread": None,
    "stop_flag": threading.Event(),
    "mtimes": {},         # path -> last mtime seen (initial baseline)
    "pending": set(),     # paths queued for next reindex batch
    "last_change_ts": 0.0,
    "last_reindex_ts": 0.0,
    "reindexed_files": 0,
    "debounce_s": 30,
    "scan_interval_s": 5,
    "status": "stopped",  # stopped | running | reindexing
}
_watcher_lock = threading.Lock()


def _is_indexable_file(path: str) -> bool:
    """Mirror the file-type filter used by collect_codebase_chunks."""
    p = path.lower()
    if not p.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".swift",
                       ".go", ".java", ".cpp", ".c", ".h", ".md", ".yaml",
                       ".yml", ".json", ".toml")):
        return False
    for ex in EXCLUDE_DIRS:
        if f"{os.sep}{ex}{os.sep}" in path or path.endswith(f"{os.sep}{ex}"):
            return False
    return True


def _scan_mtimes() -> dict[str, float]:
    """Walk CODEBASE and return {path: mtime} for indexable files. Cheap (stat only)."""
    out: dict[str, float] = {}
    for root, dirs, files in os.walk(CODEBASE):
        # Prune excluded dirs at walk time (much faster than checking each file)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            full = os.path.join(root, f)
            if not _is_indexable_file(full):
                continue
            try:
                out[full] = os.path.getmtime(full)
            except OSError:
                continue
    return out


def _reindex_files_incremental(paths: list[str]) -> int:
    """Re-chunk + embed the given files, replace their chunks in the index.

    Strategy: we cannot remove a single doc from a FAISS flat index in-place;
    instead we mark stale chunks (by file path in the chunk header) and rebuild
    if there are too many duplicates accumulating. For a "live edit" workflow
    where each file changes a handful of times per session, the duplicates are
    a minor noise — the deduplicator keeps the highest score by family/body in
    rerank, so stale chunks just get pushed down.

    Returns number of new chunks added.
    """
    from embeddinggemma.ui.corpus import _chunk_line_windows, _chunk_python_file_ast
    n_chunks = 0
    new_docs = []
    seen_paths = set()
    for p in paths:
        if not os.path.exists(p):
            continue  # deleted between detect and reindex — skip silently
        try:
            if p.lower().endswith(".py"):
                chunks = _chunk_python_file_ast(p, [200])
                if not chunks:
                    chunks = _chunk_line_windows(p, [200])
            else:
                chunks = _chunk_line_windows(p, [200])
            # Filter the same way the build does: short + oversized out.
            chunks = [c for c in chunks if 50 <= len(c.strip()) <= 20000]
            if chunks:
                seen_paths.add(p)
                new_docs.extend(chunks)
                n_chunks += len(chunks)
        except Exception as e:
            logger.warning("incremental reindex: failed for %s: %s", p, e)
    if not new_docs:
        return 0
    try:
        # add_documents_incremental: embeds new chunks, appends to FAISS, saves
        # the cache. Built-in dedup by exact content match — so byte-identical
        # re-runs are no-ops, but any real change creates a new chunk version.
        stats = _retriever.add_documents_incremental(new_docs)
        logger.info("[watcher] add_documents_incremental: %s", stats)
        # Invalidate the content→id cache used by hybrid search
        if hasattr(_retriever, "_content_to_id"):
            try: delattr(_retriever, "_content_to_id")
            except Exception: pass
    except Exception as e:
        logger.warning("incremental reindex: add_documents failed: %s", e)
        return 0
    return n_chunks


def _watcher_loop():
    """Background thread: scan mtimes, debounce, batch-reindex."""
    logger.info("[watcher] starting; debounce=%ds scan_every=%ds",
                _watcher_state["debounce_s"], _watcher_state["scan_interval_s"])
    # Baseline: don't fire on already-existing files.
    with _watcher_lock:
        _watcher_state["mtimes"] = _scan_mtimes()
        logger.info("[watcher] baseline: %d files tracked",
                    len(_watcher_state["mtimes"]))
        _watcher_state["status"] = "running"

    stop_flag: threading.Event = _watcher_state["stop_flag"]
    while not stop_flag.is_set():
        try:
            current = _scan_mtimes()
            with _watcher_lock:
                old = _watcher_state["mtimes"]
                changed_or_new = [p for p, m in current.items()
                                  if old.get(p, 0.0) < m]
                if changed_or_new:
                    _watcher_state["pending"].update(changed_or_new)
                    _watcher_state["last_change_ts"] = time.time()
                    _watcher_state["mtimes"] = current
                # Debounce check: any pending + enough quiet → drain.
                quiet = time.time() - _watcher_state["last_change_ts"]
                pending_count = len(_watcher_state["pending"])
                should_drain = (pending_count > 0 and
                                quiet >= _watcher_state["debounce_s"])
            if should_drain:
                with _watcher_lock:
                    batch = list(_watcher_state["pending"])
                    _watcher_state["pending"].clear()
                    _watcher_state["status"] = "reindexing"
                logger.info("[watcher] draining batch of %d files (%.0fs quiet)",
                            len(batch), quiet)
                t0 = time.time()
                n = _reindex_files_incremental(batch)
                logger.info("[watcher] reindexed %d files → %d chunks in %.1fs",
                            len(batch), n, time.time() - t0)
                with _watcher_lock:
                    _watcher_state["reindexed_files"] += len(batch)
                    _watcher_state["last_reindex_ts"] = time.time()
                    _watcher_state["status"] = "running"
        except Exception as e:
            logger.warning("[watcher] loop iteration failed: %s", e)
        # sleep but wake fast on stop
        stop_flag.wait(_watcher_state["scan_interval_s"])
    with _watcher_lock:
        _watcher_state["status"] = "stopped"
    logger.info("[watcher] stopped")


@mcp.tool()
async def fungus_watch_start(debounce_s: int = 30, scan_interval_s: int = 5) -> str:
    """Start a background file-watcher that incrementally reindexes changed files.

    The watcher walks the codebase every `scan_interval_s` seconds (5s default),
    tracks file mtimes, and reindexes any changed/new files after `debounce_s`
    seconds of quiet (30s default). This survives `git pull` storms — instead
    of 1000 single-file embeds, you get one batch reindex.

    Notes:
        - In-memory state only; restart re-baselines (no false hits on existing files).
        - BM25 inverted index is NOT updated live (would need ~12s rebuild);
          full BM25 refresh on next server restart.
        - Cross-encoder reranker reads chunks dynamically, no rebuild needed.

    Args:
        debounce_s: Seconds of quiet before draining the pending batch.
        scan_interval_s: How often to scan for changes (between batches).
    """
    if _watcher_state["thread"] is not None and _watcher_state["thread"].is_alive():
        return f"Watcher already running (status: {_watcher_state['status']})."
    _watcher_state["debounce_s"] = max(5, int(debounce_s))
    _watcher_state["scan_interval_s"] = max(1, int(scan_interval_s))
    _watcher_state["stop_flag"].clear()
    t = threading.Thread(target=_watcher_loop, name="fungus-watcher", daemon=True)
    _watcher_state["thread"] = t
    t.start()
    return (f"Watcher started: scanning every {_watcher_state['scan_interval_s']}s, "
            f"draining after {_watcher_state['debounce_s']}s of quiet.")


@mcp.tool()
async def fungus_watch_stop() -> str:
    """Stop the background file-watcher (any pending changes are discarded)."""
    t = _watcher_state.get("thread")
    if t is None or not t.is_alive():
        return "Watcher not running."
    _watcher_state["stop_flag"].set()
    t.join(timeout=10)
    return "Watcher stopped."


@mcp.tool()
async def fungus_watch_status() -> str:
    """Show the watcher's state: running/stopped, pending count, last reindex stats."""
    with _watcher_lock:
        s = dict(_watcher_state)
    t = s.get("thread")
    alive = t is not None and t.is_alive()
    lines = [
        f"## Watcher status: **{s['status']}** ({'alive' if alive else 'dead'})",
        f"- tracked files: {len(s.get('mtimes', {}))}",
        f"- pending in queue: {len(s.get('pending', set()))}",
        f"- total reindexed since start: {s.get('reindexed_files', 0)}",
        f"- debounce: {s.get('debounce_s', 0)}s, scan interval: {s.get('scan_interval_s', 0)}s",
    ]
    lct = s.get("last_change_ts", 0)
    lrt = s.get("last_reindex_ts", 0)
    if lct > 0:
        lines.append(f"- last change detected: {time.time()-lct:.0f}s ago")
    if lrt > 0:
        lines.append(f"- last reindex: {time.time()-lrt:.0f}s ago")
    return "\n".join(lines)


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
