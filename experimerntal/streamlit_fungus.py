import os
import sys
import re
import json
import uuid
import hashlib
import streamlit as st
import time
from datetime import datetime
import io

# Ensure 'src' is on sys.path so `embeddinggemma.*` imports work when running from repo root
_SRC_PATH = os.path.abspath(os.path.join(os.getcwd(), "src"))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except Exception:
    HAS_GRAPHVIZ = False
import plotly.graph_objects as go
from typing import List, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure project root on path to import MCPMRetriever
sys.path.append(os.path.abspath("."))
try:
    from embeddinggemma.mcmp_rag import MCPMRetriever  # type: ignore
except Exception:
    try:
        from mcmp_rag import MCPMRetriever  # type: ignore
    except Exception as e:
        MCPMRetriever = None  # type: ignore
        st.error(f"MCPMRetriever Import failed: {e}")


CACHE_DIR = os.path.join(".fungus_cache", "chunks")
os.makedirs(CACHE_DIR, exist_ok=True)

# Report artifacts (background runs)
REPORTS_DIR = os.path.join(".fungus_cache", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Lightweight background executor for long-running report jobs
BG_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Optional LangChain/LangGraph agent support (tool-calling). Falls back gracefully if unavailable.
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    # Force modern ChatOllama first
    try:
        import langchain_ollama as _lco  # type: ignore
        LCChatOllama = _lco.ChatOllama  # type: ignore
    except Exception:  # pragma: no cover
        # Deprecated fallback only if modern import fails
        from langchain_community.chat_models import ChatOllama as LCChatOllama  # type: ignore
    try:
        from langgraph.prebuilt import create_react_agent  # preferred modern agent
    except Exception:
        create_react_agent = None  # type: ignore
    HAS_LANGCHAIN = True and (LCChatOllama is not None)
except Exception:
    HAS_LANGCHAIN = False

# Optional Enterprise RAG (persistent store) for the Rag section
try:
    from embeddinggemma.rag_v1 import RagV1  # type: ignore
    HAS_ENTERPRISE_RAG = True
except Exception:
    try:
        from embeddinggemma.enterprise_rag import EnterpriseCodeRAG as RagV1  # type: ignore
        HAS_ENTERPRISE_RAG = True
    except Exception:
        HAS_ENTERPRISE_RAG = False

# Import refactored helpers
from embeddinggemma.ui.state import Settings, init_session  # type: ignore
from embeddinggemma.ui.corpus import collect_codebase_chunks, list_code_files, chunk_python_file  # type: ignore
from embeddinggemma.ui.queries import generate_multi_queries_from_llm, dedup_multi_queries  # type: ignore
from embeddinggemma.ui.mcmp_runner import select_diverse_results, quick_search_with_mcmp  # type: ignore
from embeddinggemma.ui.agent import build_langchain_agent_if_available  # type: ignore


def _make_snippet(item: Dict[str, Any], q: str, radius: int = 5) -> Dict[str, Any]:
    text = item.get('content') or ''
    lines = text.splitlines()
    try:
        pattern = re.compile(re.escape(q), re.IGNORECASE)
    except Exception:
        pattern = None
    idx = None
    if pattern:
        for i, ln in enumerate(lines):
            if pattern.search(ln):
                idx = i
                break
    if idx is None:
        return {**item, 'snippet': '\n'.join(lines[:min(10, len(lines))])}
    s = max(0, idx - radius)
    e = min(len(lines), idx + radius + 1)
    return {**item, 'snippet': '\n'.join(lines[s:e])}

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
    win_key = "-".join(str(w) for w in sorted(set(int(w) for w in windows)))
    sha = _file_sha1(path)
    return os.path.join(CACHE_DIR, f"{rel}.{sha}.{win_key}.jsonl")


def _load_cached_chunks(path: str, windows: List[int], ui: bool = True) -> List[str]:
    cache_path = _cache_key(path, windows)
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                lines = [line.rstrip("\n") for line in f]
                if ui:
                    st.write(f"[cache] hit {os.path.relpath(path)} -> {len(lines)} chunks")
                return lines
    except Exception:
        return []
    return []


def _save_cached_chunks(path: str, windows: List[int], chunks: List[str], ui: bool = True) -> None:
    cache_path = _cache_key(path, windows)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            for c in chunks:
                f.write(c.replace('\n', '\n') + '\n')
        if ui:
            st.write(f"[cache] save {os.path.relpath(path)} -> {len(chunks)} chunks")
    except Exception:
        pass


def identifiers_from_text(text: str) -> List[str]:
    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)


def find_similar_names(docs: List[str]) -> List[Dict[str, Any]]:
    names: List[str] = []
    for d in docs:
        names.extend(identifiers_from_text(d))
    names = list({n for n in names if len(n) >= 3})
    names.sort()
    similar: List[Dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            if a[0].lower() != b[0].lower():
                break
            common = os.path.commonprefix([a.lower(), b.lower()])
            ratio = len(common) / max(len(a), len(b))
            if ratio >= 0.6 and a != b:
                similar.append({"a": a, "b": b, "score": ratio})
    return similar[:200]


def find_redundancy(docs: List[str]) -> List[Dict[str, Any]]:
    seen: Dict[str, List[int]] = {}
    for idx, d in enumerate(docs):
        norm = re.sub(r"\s+", " ", d.strip()).lower()
        h = hashlib.sha256(norm.encode('utf-8')).hexdigest()[:16]
        seen.setdefault(h, []).append(idx)
    duplicates = [{"hash": h, "indices": idxs, "count": len(idxs)} for h, idxs in seen.items() if len(idxs) > 1]
    duplicates.sort(key=lambda x: x["count"], reverse=True)
    return duplicates


def diff_trails(prev: Dict[Any, float], curr: Dict[Any, float]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    keys = set(prev.keys()) | set(curr.keys())
    for k in keys:
        a = prev.get(k, 0.0)
        b = curr.get(k, 0.0)
        if b != a:
            events.append({"edge": k, "delta": float(b - a), "to": float(b), "from": float(a)})
    # sort by absolute change descending
    events.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return events


def make_dot_tree(root_label: str, children: List[str]) -> str:
    # Build a simple DOT graph string without requiring python-graphviz
    def esc(s: str) -> str:
        return s.replace('"', '\"')
    lines = ["digraph G {", "  rankdir=LR;", f"  Q [label=\"{esc(root_label)}\"];" ]
    for i, child in enumerate(children):
        nid = f"n{i+1}"
        lines.append(f"  {nid} [label=\"{esc(child)}\"];")
        lines.append(f"  Q -> {nid};")
    lines.append("}")
    return "\n".join(lines)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def _cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _normalize(a) @ _normalize(b).T


def _normalize_query_text(text: str) -> str:
    """Lowercase, collapse spaces, strip punctuation and common stopwords for dedup."""
    try:
        t = (text or "").strip().lower()
        # remove punctuation
        t = re.sub(r"[\p{P}\p{S}]", " ", t)
    except re.error:
        # Fallback: basic ASCII punctuation
        t = re.sub(r"[^a-z0-9]+", " ", (text or "").strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    stop = {
        "the","a","an","is","are","be","to","of","in","on","for","and","or","with","how","what","where","when","which","that",
        "does","do","can","i","we","you","it","this","these","those","about","use","used","using","run","start","start","guide"
    }
    tokens = [w for w in t.split() if w not in stop]
    return " ".join(tokens)


def _token_set(text: str) -> set:
    return set(_normalize_query_text(text).split())


def _extract_source(meta: Dict[str, Any], content: str) -> str:
    src = (meta or {}).get('file_path') if isinstance(meta, dict) else None
    if not src:
        m = re.search(r"^# file: (.+?) \| lines:", content or "", re.MULTILINE)
        if m:
            src = m.group(1)
    return src or "unknown"


def select_diverse_results(results_items: List[Dict[str, Any]],
                           retr: Any,
                           top_k: int,
                           alpha: float,
                           dedup_tau: float,
                           per_folder_cap: int) -> List[Dict[str, Any]]:
    if not results_items:
        return []
    # Embed contents using retr's embedding model if available
    texts = [(it.get('content') or '')[:2048] for it in results_items]
    try:
        embs = retr.embedding_model.encode(texts)
        embs = np.array(embs, dtype=np.float32)
    except Exception:
        # Fallback: random small vectors to at least run MMR logic (not ideal)
        rng = np.random.default_rng(42)
        embs = rng.normal(0, 1, size=(len(texts), 64)).astype(np.float32)
    sims = _cos_sim(embs, embs)

    selected: List[int] = []
    folder_counts: Dict[str, int] = {}
    # Precompute base scores
    base_scores = np.array([float(it.get('relevance_score', 0.0)) for it in results_items], dtype=np.float32)
    order = np.argsort(-base_scores)  # descending
    for idx in order:
        if len(selected) >= int(top_k):
            break
        it = results_items[int(idx)]
        src = _extract_source(it.get('metadata', {}), it.get('content', ''))
        folder = os.path.dirname(src)
        # Per-folder cap
        if per_folder_cap > 0 and folder_counts.get(folder, 0) >= per_folder_cap:
            continue
        # Dedup threshold vs selected set
        if selected:
            max_sim = float(np.max(sims[idx, selected]))
            if max_sim >= dedup_tau:
                continue
        # MMR score
        if selected:
            diversity_penalty = float(np.max(sims[idx, selected]))
        else:
            diversity_penalty = 0.0
        mmr = float(alpha * base_scores[idx] - (1.0 - alpha) * diversity_penalty)
        # Greedy select by order; mmr is advisory since we already sorted by base score
        selected.append(int(idx))
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    return [results_items[i] for i in selected]


# ---------- Shared helpers for agent/chat + background report ----------

def _prepare_corpus(use_repo: bool,
                    root_folder: str,
                    docs_file: str,
                    windows: List[int],
                    max_files: int,
                    exclude_dirs: List[str]) -> Dict[str, Any]:
    """Build chunk corpus without UI side-effects. Returns payload with docs and file lists."""
    docs: List[str] = []
    discovered_files: List[str] = []
    if use_repo:
        discovered_files = list_code_files('src', int(max_files), exclude_dirs)
        docs.extend(collect_codebase_chunks('src', windows, int(max_files), exclude_dirs, ui=False))
    else:
        rf = (root_folder or '').strip()
        if rf and os.path.isdir(rf):
            discovered_files = list_code_files(rf, int(max_files), exclude_dirs)
            docs.extend(collect_codebase_chunks(rf, windows, int(max_files), exclude_dirs, ui=False))
    if docs_file.strip():
        if docs_file.endswith('.py'):
            docs.extend(chunk_python_file(docs_file, windows))
        else:
            try:
                with open(docs_file, 'r', encoding='utf-8', errors='ignore') as f:
                    docs.append(f.read())
            except Exception:
                pass
    # Parse files represented in embeddings from chunk headers
    loaded_files: List[str] = []
    try:
        pat = re.compile(r"^# file: (.+?) \| lines:", re.MULTILINE)
        seen = set()
        for d in docs:
            m = pat.search(d)
            if m:
                fp = m.group(1)
                if fp not in seen:
                    seen.add(fp)
                    loaded_files.append(fp)
    except Exception:
        pass
    return {
        "docs": docs,
        "discovered_files": discovered_files,
        "loaded_files": loaded_files,
    }


def _ollama_generate(prompt: str, model: str = None, timeout: int = 180) -> str:
    try:
        import requests  # local import to avoid hard dep if unused
        host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
        model_name = model or os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
        r = requests.post(
            f"{host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 384, "top_p": 0.9, "gpu_layers": 999}
            },
            timeout=timeout
        )
        if r.ok:
            return r.json().get('response', '')
        return f"[LLM error] status={r.status_code}"
    except Exception as e:
        return f"[LLM error] {e}"


def _render_snapshot_png_bytes(snapshot: Dict[str, Any], width: int = 800, height: int = 600) -> bytes:
    """Render a minimal snapshot image using matplotlib and return PNG bytes.

    Avoids heavy static image backends by drawing simple scatter/lines.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
    except Exception:
        return b""
    docs_xy = snapshot.get("documents", {}).get("xy", []) or []
    docs_rel = snapshot.get("documents", {}).get("relevance", []) or []
    agents_xy = snapshot.get("agents", {}).get("xy", []) or []
    edges = snapshot.get("edges", []) or []
    fig = plt.figure(figsize=(width/100.0, height/100.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    # Draw edges first
    for e in edges:
        x0, y0, x1, y1 = e.get('x0', 0.0), e.get('y0', 0.0), e.get('x1', 0.0), e.get('y1', 0.0)
        s = float(e.get('s', 0.1))
        ax.plot([x0, x1], [y0, y1], color=(0, 0.6, 0, 0.25), linewidth=max(0.5, s*2))
    # Draw documents
    if docs_xy:
        xs = [p[0] for p in docs_xy]
        ys = [p[1] for p in docs_xy]
        sizes = [10 + 30*float(r) for r in (docs_rel if docs_rel else [0.0]*len(xs))]
        ax.scatter(xs, ys, s=sizes, c='tab:blue', alpha=0.7, edgecolors='none')
    # Draw agents
    if agents_xy:
        xa = [p[0] for p in agents_xy]
        ya = [p[1] for p in agents_xy]
        ax.scatter(xa, ya, s=6, c='crimson', alpha=0.7, edgecolors='none')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    plt.close(fig)
    return buf.getvalue()


def _build_gif_from_snapshots(snaps: List[Dict[str, Any]], out_path: str, fps: int = 6) -> str:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        # If imageio missing, abort gracefully
        return ""
    frames = []
    for sn in snaps:
        png = _render_snapshot_png_bytes(sn)
        if png:
            frames.append(imageio.imread(io.BytesIO(png)))
        if len(frames) >= 1 and len(frames) % 50 == 0:
            pass
    if not frames:
        return ""
    duration = max(0.05, 1.0/float(max(1, int(fps))))
    imageio.mimsave(out_path, frames, duration=duration)
    return out_path

def generate_multi_queries_from_llm(base_query: str, num_queries: int = 5, context_files: List[str] = None, keyword_hints: List[str] = None) -> List[str]:
    """Use local Ollama model to propose specific multi-queries derived from a single intent.

    Returns a list of plain lines; numbering/bullets are stripped when possible.
    """
    try:
        n = max(1, min(int(num_queries), 10))
    except Exception:
        n = 5
    files_hint = "\n".join(sorted(set((context_files or [])[:40])))
    kw_hint = ", ".join(sorted(set([(k or '').strip() for k in (keyword_hints or []) if (k or '').strip()])))
    lines_prompt = [
        "You reformulate a single repository question into multiple concrete search queries.",
        "Focus on files that are embedded (so results are answerable).",
        "Candidate files (hints):",
        files_hint,
    ]
    if kw_hint:
        lines_prompt.append(f"Query keywords (hints): {kw_hint}")
    lines_prompt += [
        "",
        "Rules:",
        "- Output EXACTLY {n} lines, no numbering or bullets.",
        "- Each line should be a direct, concrete code-search question.",
        "- Prefer including exact file paths from the hints when relevant.",
        "- Keep lines under 90 chars.",
    ]
    sys_prompt = "\n".join(lines_prompt).replace("{n}", str(n))
    user = f"Base query: {base_query}\nWrite {n} concrete repository search questions (one per line), grounded in the hinted files."
    text = _ollama_generate(f"System:\n{sys_prompt}\n\nUser:\n{user}")
    lines = [re.sub(r"^[\-\d\.\)\s]+", "", ln).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return lines[:n]

def quick_search_with_mcmp(settings: Dict[str, Any], query_text: str, top_k: int) -> Dict[str, Any]:
    """Run a small, fast MCMP search to power agent replies."""
    if MCPMRetriever is None:
        return {"error": "MCPMRetriever unavailable"}
    corp = _prepare_corpus(
        settings.get('use_repo', True),
        settings.get('root_folder', os.getcwd()),
        settings.get('docs_file', ''),
        settings.get('windows', [100, 200, 300]),
        settings.get('max_files', 200),
        settings.get('exclude_dirs', [])
    )
    docs = corp.get('docs', [])
    if not docs:
        return {"error": "No documents found for quick search"}
    retr = MCPMRetriever(
        num_agents=int(settings.get('num_agents', 100)),
        max_iterations=int(settings.get('max_iterations', 20)),
        exploration_bonus=float(settings.get('exploration_bonus', 0.1)),
        pheromone_decay=float(settings.get('pheromone_decay', 0.95)),
        embed_batch_size=int(settings.get('embed_bs', 64))
    )
    try:
        retr.add_documents(docs)
        res = retr.search(str(query_text), top_k=int(top_k))
    except Exception as e:
        return {"error": f"search failed: {e}"}
    return {"results": res.get('results', []), "docs_count": len(docs)}


def start_background_report(settings: Dict[str, Any], query_text: str) -> None:
    """Kick off a background report job and store a Future in session_state."""
    def _progress_path(job_id: str) -> str:
        return os.path.join(REPORTS_DIR, f"progress_{job_id}.json")

    def _write_progress(job_id: str, payload: Dict[str, Any]) -> None:
        try:
            payload = dict(payload)
            payload.setdefault("updated_at", datetime.utcnow().isoformat() + "Z")
            with open(_progress_path(job_id), 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _job(job_id: str, s: Dict[str, Any], q: str) -> str:
        if MCPMRetriever is None:
            raise RuntimeError("MCPMRetriever unavailable")
        _write_progress(job_id, {"status": "running", "percent": 0, "message": "Preparing corpus…"})
        corp = _prepare_corpus(
            s.get('use_repo', True),
            s.get('root_folder', os.getcwd()),
            s.get('docs_file', ''),
            s.get('windows', [50, 100, 200, 300, 400]),
            s.get('max_files', 1000),
            s.get('exclude_dirs', [])
        )
        docs = corp.get('docs', [])
        _write_progress(job_id, {"status": "running", "percent": 10, "message": f"Corpus ready: {len(docs)} chunks"})
        retr = MCPMRetriever(
            num_agents=int(s.get('num_agents', 200)),
            max_iterations=int(s.get('max_iterations', 60)),
            exploration_bonus=float(s.get('exploration_bonus', 0.1)),
            pheromone_decay=float(s.get('pheromone_decay', 0.95)),
            embed_batch_size=int(s.get('embed_bs', 64))
        )
        if not docs:
            _write_progress(job_id, {"status": "error", "percent": 100, "message": "No documents to analyze"})
            raise RuntimeError("No documents to analyze")
        retr.add_documents(docs)
        # Sharded single-query path similar to UI flow
        total_chunks = len(docs)
        shard_size = int(s.get('max_chunks_per_shard', 2000))
        if shard_size <= 0 or shard_size >= total_chunks:
            shard_ranges = [(0, total_chunks)]
        else:
            shard_ranges = [(i, min(i + shard_size, total_chunks)) for i in range(0, total_chunks, shard_size)]
        aggregated_items: List[Dict[str, Any]] = []
        num_shards = max(1, len(shard_ranges))
        for shard_idx, (s_start, s_end) in enumerate(shard_ranges):
            shard = docs[s_start:s_end]
            try:
                retr.clear_documents()
            except Exception:
                pass
            retr.add_documents(shard)
            retr.initialize_simulation(q)
            for _ in range(int(s.get('max_iterations', 60))):
                retr.step(1)
            res_shard = retr.search(q, top_k=int(s.get('top_k', 5)))
            aggregated_items.extend(res_shard.get('results', []))
            pct = 10 + int(80 * float(shard_idx + 1) / float(num_shards))
            _write_progress(job_id, {"status": "running", "percent": pct, "message": f"Processed shard {shard_idx+1}/{num_shards}"})
        # Optionally diversify results
        if s.get('pure_topk', False):
            items = sorted(aggregated_items, key=lambda it: float(it.get('relevance_score', 0.0)), reverse=True)[:int(s.get('top_k', 5))]
        else:
            items = select_diverse_results(
                aggregated_items, retr,
                int(s.get('top_k', 5)), float(s.get('div_alpha', 0.7)),
                float(s.get('dedup_tau', 0.92)), int(s.get('per_folder_cap', 2))
            )
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": q,
            "settings": {k: v for k, v in s.items() if k not in {"docs"}},
            "results": items,
            "discovered_files": corp.get('discovered_files', []),
            "loaded_files": corp.get('loaded_files', []),
        }
        out_path = os.path.join(REPORTS_DIR, f"report_{job_id}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _write_progress(job_id, {"status": "done", "percent": 100, "message": "Report ready", "report_path": out_path})
        return out_path

    job_id = uuid.uuid4().hex[:12]
    st.session_state['report_job_id'] = job_id
    _write_progress(job_id, {"status": "running", "percent": 0, "message": "Queued", "started_at": datetime.utcnow().isoformat() + "Z"})
    fut = BG_EXECUTOR.submit(_job, job_id, dict(settings), str(query_text))
    st.session_state['report_future'] = fut
    st.session_state['report_started_at'] = time.time()


def build_langchain_agent_if_available(settings: Dict[str, Any]):
    """Create a LangChain agent with tools (if LC is installed)."""
    if not HAS_LANGCHAIN:
        return None

    def tool_search(q: str) -> str:
        res = quick_search_with_mcmp(settings, q, settings.get('top_k', 5))
        if res.get('error'):
            return res['error']
        items = res.get('results', [])
        lines = []
        for i, it in enumerate(items[: settings.get('top_k', 5)]):
            src = (it.get('metadata', {}) or {}).get('file_path', 'chunk')
            lines.append(f"#{i+1} {float(it.get('relevance_score', 0.0)):.3f} {src}")
        return "\n".join(lines) or "No results"

    def tool_get_settings(_: str = "") -> str:
        safe = {k: v for k, v in settings.items() if k not in {"docs"}}
        return json.dumps(safe)

    def tool_set_root_dir(new_dir: str) -> str:
        if os.path.isdir(new_dir):
            settings['use_repo'] = False
            settings['root_folder'] = new_dir
            st.session_state['root_folder_agent_override'] = new_dir
            return f"Root folder set to: {new_dir}"
        return f"Directory does not exist: {new_dir}"

    tools = [
        Tool(name="search_code", func=tool_search, description="Search codebase for a query and return top sources."),
        Tool(name="get_settings", func=tool_get_settings, description="Return current search settings as JSON."),
        Tool(name="set_root_dir", func=tool_set_root_dir, description="Set the root directory for searching. Provide an absolute path."),
    ]

    llm = LCChatOllama(model=os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')) if LCChatOllama else None
    if llm is None:
        return None
    # Prefer LangGraph prebuilt ReAct agent when available; avoid legacy when possible
    if create_react_agent is not None:
        try:
            # Some models (e.g., ChatOllama) don't implement bind_tools; guard and fallback
            if hasattr(llm, "bind_tools"):
                graph = create_react_agent(llm, tools)
                class _GraphWrapper:
                    def __init__(self, g):
                        self._g = g
                    def invoke(self, payload: Dict[str, Any]):
                        msg = [{"role": "user", "content": str(payload.get("input", ""))}]
                        out = self._g.invoke({"messages": msg})
                        try:
                            messages = out.get("messages", [])
                            content = messages[-1].get("content", "") if messages else str(out)
                            return {"output": content}
                        except Exception:
                            return {"output": str(out)}
                return _GraphWrapper(graph)
        except NotImplementedError:
            pass
        except Exception:
            pass
    # Only if LangGraph is missing, fall back to legacy AgentExecutor
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )


st.set_page_config(page_title="Fungus Frontend", layout="wide")
st.title("Fungus (MCPM) Frontend for Code Space")
dbg_state = st.empty()

DOCS_MD = """
### How to use

1. Launch the app:
   - `streamlit run streamlit_fungus.py`

2. Choose a mode in the sidebar:
   - deep, structure, exploratory, summary, similar, redundancy, repair

3. Select corpus:
   - Enable "Use code space (src) as corpus" to analyze `src/`, or
   - Disable it and set a valid path in "Root folder" to analyze any folder on your PC.

4. Configure search:
   - Agents, Max iterations (simulation length)
   - Windows (multi-granular chunk sizes)
   - Diversity & Breadth (MMR alpha, dedup threshold, per-folder cap)
   - Optional: Edit the Mode prompt for LLM-guided summaries

5. Click Run:
   - You will see discovered files, files loaded into embeddings, results JSON,
     an optional results tree, and a pheromone network snapshot.
   - In standard search, the simulation steps stream live (avg relevance, edges, agents).

Tips:
- Increase Agents/Iterations for larger repos.
- For broader coverage, raise diversity (lower dedup, set folder caps, tune MMR alpha).
"""

with st.sidebar:
    st.header("Settings")
    if 'show_docs' not in st.session_state:
        st.session_state.show_docs = False
    if st.button("Docs"):
        st.session_state.show_docs = not st.session_state.show_docs
    mode = st.selectbox("Mode", [
        "deep", "structure", "exploratory", "summary", "similar", "redundancy", "repair", "Rag"
    ], index=0)
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
    windows_str = st.text_input("Windows (lines)", value="50,100,200,300,400")
    windows = [int(x.strip()) for x in windows_str.split(',') if x.strip().isdigit()]
    use_repo = st.checkbox("Use code space (src) as corpus", value=True)
    root_folder = st.text_input("Root folder (used when src is off)", value=os.getcwd())
    max_files = st.number_input("Max files to index (0 = no limit)", min_value=0, max_value=10000, value=1000, step=50)
    exclude_str = st.text_input("Exclude folders (comma-separated substrings)", value=".venv,node_modules,.git,external")
    exclude_dirs = [e.strip() for e in exclude_str.split(',') if e.strip()]
    chunk_workers = st.number_input("Chunk workers (threads)", min_value=1, max_value=64, value=32, step=1, help="Parallel threads for chunking files")
    st.session_state['chunk_workers_override'] = int(chunk_workers)
    docs_file = st.text_input("Optional single docs file", value="")
    num_agents = st.number_input("Agents", min_value=50, max_value=5000, value=200, step=50)
    max_iterations = st.number_input("Max iterations", min_value=10, max_value=1000, value=60, step=10)
    show_tree = st.checkbox("Show results tree", value=True)
    show_network = st.checkbox("Show pheromone network snapshot", value=True)
    gen_answer = st.checkbox("Generate answer with LLM using mode prompt", value=(mode in ["summary","exploratory","structure"]))
    # Diversity controls
    st.markdown("### Diversity & Breadth")
    div_alpha = st.slider("MMR alpha (relevance vs. diversity)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    dedup_tau = st.slider("Dedup cosine threshold", min_value=0.80, max_value=0.99, value=0.92, step=0.01)
    cols_div = st.columns([1, 1])
    with cols_div[0]:
        per_folder_cap = st.number_input("Per-folder cap", min_value=0, max_value=10, value=2, step=1, help="0 = no cap")
    with cols_div[1]:
        pure_topk = st.checkbox("Pure Top-K (disable diversity)", value=False, help="If enabled, results are sorted strictly by relevance_score and truncated to Top K.")
    # Logging & params
    st.markdown("### Simulation & Logging")
    cols_sim = st.columns([1, 1, 1, 1])
    with cols_sim[0]:
        log_every = st.number_input("Log every N iterations", min_value=1, max_value=100, value=10, step=1)
    with cols_sim[1]:
        exploration_bonus = st.slider("Exploration bonus", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    with cols_sim[2]:
        pheromone_decay = st.slider("Pheromone decay", min_value=0.80, max_value=0.999, value=0.95, step=0.001)
    with cols_sim[3]:
        embed_bs = st.number_input("Embedding batch size", min_value=16, max_value=1024, value=64, step=16, help="Batch size for GPU embedding encode()")
    mcp_debug = st.checkbox("MCPM debug logs (stdout)", value=False, help="Print detailed MCPM debug logs to the Streamlit server console.")
    st.markdown("### Keyword Boost (optional)")
    kw_lambda = st.slider("Keyword boost λ", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    kw_terms_str = st.text_input("Query keywords (comma‑separated)", value="")
    # Sharding
    st.markdown("### Sharding")
    max_chunks_per_shard = st.number_input("Max chunks per shard (0 = no sharding)", min_value=0, max_value=20000, value=2000, step=100)
    # Agent settings
    st.markdown("### Agent & Reports")
    enable_agent_chat = st.checkbox("Enable agent chat", value=True)
    prefer_langchain = st.checkbox("Use LangChain agent if available", value=HAS_LANGCHAIN)
    st.caption(f"LangChain available: {'✅' if HAS_LANGCHAIN else '❌'}")
    default_prompts = {
        "deep": (
            "You are a senior code analyst. Given the CONTEXT snippets, produce a thorough, actionable analysis.\n"
            "Output strictly in this structure:\n"
            "1. Key Findings (bullets with short titles + 1-2 lines)\n"
            "2. Risks (bullets; each with severity: low/med/high)\n"
            "3. Gaps & Unknowns (bullets; what is missing)\n"
            "4. Recommended Next Steps (numbered; concrete actions)\n"
            "5. Evidence (file_path:line_range short-quote)\n"
            "Rules: Be concise and precise. No speculation. Cite file paths exactly as seen."
        ),
        "structure": (
            "You are a code structure auditor. From CONTEXT, extract a structured inventory.\n"
            "Output strictly as sections:\n"
            "- Modules: name, purpose\n"
            "- Classes: name, methods (name: purpose), relationships (inherits/uses)\n"
            "- Functions: name, signature(if visible), purpose\n"
            "- Entry Points / CLI / API: description\n"
            "- Config & Env: keys/paths\n"
            "- Evidence: file_path:line_range bullets\n"
            "Rules: Prefer lists, avoid prose. Use exact identifiers and paths."
        ),
        "exploratory": (
            "You are a technical explorer. Explain the repository at a high level and map components.\n"
            "Produce sections:\n"
            "- Overview: one paragraph\n"
            "- Components & Responsibilities (bulleted)\n"
            "- Data Flow / Control Flow (bulleted; mention files)\n"
            "- How to Run / Build (bulleted commands if visible)\n"
            "- Key Dependencies (bulleted)\n"
            "- Evidence (file_path:line_range)\n"
            "Rules: Keep it factual; cite source lines."
        ),
        "summary": (
            "You are a technical summarizer. Summarize the most important information for the QUERY.\n"
            "Output strictly:\n"
            "- TL;DR (2-3 bullets)\n"
            "- Details (bullets grouped by file or feature)\n"
            "- Open Questions\n"
            "- Evidence (file_path:line_range)\n"
            "Rules: Avoid speculation; only use given CONTEXT; cite file paths."
        ),
        "similar": (
            "You are a naming auditor. Identify identifiers that are likely to be confused.\n"
            "Output strictly:\n"
            "- Similar Pairs (a,b, similarity 0-1, reason)\n"
            "- Risk Assessment (bullets)\n"
            "- Evidence (file_path:line_range)\n"
            "Rules: Only list near-collisions that can confuse a coder."
        ),
        "redundancy": (
            "You are a duplication detector. From CONTEXT, list duplicates or near-duplicates.\n"
            "Output strictly:\n"
            "- Duplicates (hash or signature; file paths involved)\n"
            "- Impact (why it matters)\n"
            "- Suggested Merge/Refactor Plan\n"
            "- Evidence (file_path:line_range)\n"
            "Rules: Be specific and actionable."
        ),
        "repair": (
            "You are a refactoring planner. Propose versioned renames for risky/confusing identifiers.\n"
            "Output strictly:\n"
            "- Rename Plan (from -> to_v2, rationale)\n"
            "- Affected Files (bulleted with file_path:line_range)\n"
            "- Risks & Mitigations\n"
            "- Migration Steps (ordered)\n"
            "Rules: Use conservative, reversible changes. Cite evidence."
        )
    }
    mode_prompt = st.text_area("Mode prompt", value=default_prompts.get(mode, ""), height=120)
    st.markdown("### Multi-Query (one per line, optional)")
    auto_multi = st.checkbox("Auto-generate from Query if empty (LLM)", value=True)
    auto_multi_n = st.number_input("How many to generate", min_value=1, max_value=10, value=5, step=1)
    dedup_enable = st.checkbox("Deduplicate multi-queries", value=True)
    dedup_tau_q = st.slider("Dedup threshold (Jaccard)", min_value=0.5, max_value=0.95, value=0.8, step=0.01)
    if 'multi_queries_text' not in st.session_state:
        st.session_state.multi_queries_text = ""
    multi_queries_text = st.text_area(
        "Queries",
        value=st.session_state.get('multi_queries_text', ""),
        height=120,
        help="Provide multiple queries, one per line. Leave empty to auto-generate when enabled."
    )
    if auto_multi and not multi_queries_text.strip() and st.session_state.get('generated_multi_queries'):
        st.caption("Last generated multi-queries:")
        try:
            st.code("\n".join(st.session_state['generated_multi_queries']))
        except Exception:
            pass

    cols_q = st.columns([4, 1])
    with cols_q[0]:
        query = st.text_input(
            "Query",
            value=(
                "Explain how Enterprise RAG and Fungus MCPM are implemented here, and how to run them."
            )
        )
    with cols_q[1]:
        run = st.button("Run")

# Consolidated settings object used by agent and background jobs
settings_obj: Dict[str, Any] = {
    "mode": mode,
    "top_k": int(top_k),
    "windows": windows,
    "use_repo": bool(use_repo),
    "root_folder": root_folder,
    "max_files": int(max_files),
    "exclude_dirs": exclude_dirs,
    "docs_file": docs_file,
    "num_agents": int(num_agents),
    "max_iterations": int(max_iterations),
    "show_tree": bool(show_tree),
    "show_network": bool(show_network),
    "gen_answer": bool(gen_answer),
    "div_alpha": float(div_alpha),
    "dedup_tau": float(dedup_tau),
    "per_folder_cap": int(per_folder_cap),
    "pure_topk": bool(pure_topk),
    "log_every": int(log_every),
    "exploration_bonus": float(exploration_bonus),
    "pheromone_decay": float(pheromone_decay),
    "embed_bs": int(embed_bs),
    "max_chunks_per_shard": int(max_chunks_per_shard),
}

# Keep sidebar and agent in sync if agent set root
if st.session_state.get('root_folder_agent_override'):
    settings_obj['root_folder'] = st.session_state['root_folder_agent_override']
    settings_obj['use_repo'] = False

if run:
    if MCPMRetriever is None:
        st.error("MCPMRetriever unavailable.")
    else:
        with st.spinner("Preparing corpus and running Fungus search..."):
            dbg_state.info(f"[mode={mode}] preparing corpus…")
            # Live log panel
            log_exp = st.expander("Live log", expanded=True)
            log_box = log_exp.empty()
            _logs: List[str] = []
            def _log(msg: str) -> None:
                try:
                    ts = datetime.now().strftime("%H:%M:%S")
                    _logs.append(f"[{ts}] {msg}")
                    log_box.code("\n".join(_logs[-300:]))
                except Exception:
                    pass
            _log(f"mode={mode} | top_k={top_k} | windows={windows} | use_repo={use_repo} | max_files={max_files}")
            # Build corpus
            docs: List[str] = []
            discovered_files: List[str] = []
            # Initialize events accumulator
            events_accum: List[Dict[str, Any]] = []
            if mode == "Rag":
                # Enterprise RAG over persistent store (Qdrant + LlamaIndex)
                st.subheader("Rag (Enterprise store)")
                if not HAS_ENTERPRISE_RAG:
                    st.error("Enterprise RAG (rag_v1) not available. Ensure `embeddinggemma.rag_v1` (or `embeddinggemma.enterprise_rag`) imports and dependencies are installed (Qdrant, LlamaIndex).")
                    st.stop()
                if 'enterprise_rag' not in st.session_state:
                    st.session_state.enterprise_rag = RagV1(use_ollama=True, ollama_model=os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b'))
                er = st.session_state.enterprise_rag
                persist_dir = st.text_input("Index directory", value="./enterprise_index")
                cols_r = st.columns([1,1,2])
                with cols_r[0]:
                    if st.button("Load index"):
                        with st.spinner("Loading index…"):
                            er.load_index(persist_dir=persist_dir)
                            st.success("Index loaded")
                with cols_r[1]:
                    if st.button("Show stats"):
                        stats = er.get_stats()
                        st.json(stats)
                q = st.text_input("Query store:", value="Explain how MCPMRetriever is used")
                k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
                alpha = st.slider("Hybrid alpha (semantic vs keyword)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                if st.button("Search") and q:
                    with st.spinner("Searching enterprise index…"):
                        items = er.hybrid_search(q, top_k=int(k), alpha=float(alpha))
                        if not items:
                            st.warning("No results or index not loaded. Use 'Load index' first.")
                        else:
                            st.subheader("Results")
                            for i, it in enumerate(items):
                                src = it.get('source') or (it.get('metadata') or {}).get('file_path', 'unknown')
                                score = float(it.get('hybrid_score', 0.0))
                                with st.expander(f"#{i+1} {score:.3f} — {src}"):
                                    st.write(it.get('content','')[:1200])
                    st.stop()
                # stop here to avoid running MCPM path when in Rag mode
                st.stop()
            if use_repo:
                discovered_files = list_code_files('src', int(max_files), exclude_dirs)
                docs.extend(collect_codebase_chunks('src', windows, int(max_files), exclude_dirs))
            else:
                rf = root_folder.strip()
                if rf and os.path.isdir(rf):
                    dbg_state.info(f"[mode={mode}] crawling root folder: {rf}")
                    _log(f"Crawl root: {rf}")
                    discovered_files = list_code_files(rf, int(max_files), exclude_dirs)
                    docs.extend(collect_codebase_chunks(rf, windows, int(max_files), exclude_dirs))
                else:
                    st.warning("Root folder is not set or does not exist. Please provide a valid path.")
            if docs_file.strip():
                if docs_file.endswith('.py'):
                    docs.extend(chunk_python_file(docs_file, windows))
                else:
                    try:
                        with open(docs_file, 'r', encoding='utf-8', errors='ignore') as f:
                            docs.append(f.read())
                    except Exception as e:
                        st.warning(f"Failed to read docs_file: {e}")

            retr = MCPMRetriever(num_agents=int(num_agents), max_iterations=int(max_iterations), exploration_bonus=float(exploration_bonus), pheromone_decay=float(pheromone_decay), embed_batch_size=int(embed_bs))
            retr.log_every = int(log_every)
            try:
                retr.set_debug(bool(mcp_debug))
            except Exception:
                pass
            # Apply keyword boost settings
            try:
                retr.kw_lambda = float(kw_lambda)
                retr.kw_terms = {t.strip().lower() for t in (kw_terms_str or "").split(',') if t.strip()}
            except Exception:
                retr.kw_lambda = 0.0
                retr.kw_terms = set()
            # Enable fast GPU stepping to reduce CPU load
            try:
                import torch  # noqa
                retr.fast_gpu_step = True
            except Exception:
                retr.fast_gpu_step = False
            dbg_state.info(f"[mode={mode}] corpus size={len(docs)} chunks")
            _log(f"Discovered files: {len(discovered_files)} | Chunks: {len(docs)}")
            if docs:
                retr.add_documents(docs)
                _log("Embeddings added to retriever")
            else:
                st.error("No documents found to analyze. Enable 'Use code space (src)' or set a valid root folder or file.")
                st.stop()

            # Show discovered files and files represented in embeddings
            with st.expander("Discovered code files", expanded=False):
                st.write(f"Count: {len(discovered_files)}")
                if discovered_files:
                    preview = [os.path.relpath(p) for p in discovered_files[:200]]
                    st.code("\n".join(preview))
            # Parse file names from chunk headers
            loaded_files: List[str] = []
            try:
                pat = re.compile(r"^# file: (.+?) \| lines:", re.MULTILINE)
                seen = set()
                for d in docs:
                    m = pat.search(d)
                    if m:
                        fp = m.group(1)
                        if fp not in seen:
                            seen.add(fp)
                            loaded_files.append(fp)
            except Exception:
                pass
            with st.expander("Files loaded into embeddings", expanded=True):
                st.write(f"Count: {len(loaded_files)}")
                if loaded_files:
                    st.code("\n".join(loaded_files[:200]))

            # Parse multi-queries
            mq_raw = [q.strip() for q in (multi_queries_text or "").splitlines() if q.strip()]
            # If empty and auto-gen enabled, generation happens later
            mq_list = mq_raw

            if mode in ("similar", "redundancy", "repair") and not mq_list:
                if mode == "similar":
                    dbg_state.info(f"[mode=similar] analyzing similar names…")
                    sim = find_similar_names(docs)
                    st.subheader("Similar identifiers")
                    st.json({"count": len(sim), "items": sim})
                elif mode == "redundancy":
                    dbg_state.info(f"[mode=redundancy] scanning duplicates…")
                    dup = find_redundancy(docs)
                    st.subheader("Redundancy (duplicate chunks)")
                    st.json({"count": len(dup), "items": dup})
                else:  # repair
                    dbg_state.info(f"[mode=repair] proposing versioned renames…")
                    sim = find_similar_names(docs)
                    plan = [{"from": s["a"], "to": f"{s['a']}_v2", "reason": f"Similar to {s['b']} ({s['score']:.2f})"} for s in sim[:50]]
                    st.subheader("Rename plan (versioned)")
                    st.json({"count": len(plan), "items": plan})
            else:
                if mode == "summary":
                    # iterative search with interim summarization every 10 iterations (single-query path)
                    retr.max_iterations = int(max_iterations)
                    summaries = []
                    step = 10
                    for i in range(0, retr.max_iterations, step):
                        retr.max_iterations = min(retr.max_iterations, i + step)
                        dbg_state.info(f"[mode=summary] running iterations {i+1}-{i+10}…")
                        results = retr.search(query, top_k=int(top_k))
                        # capture minimal event
                        events_accum.append({"mode": "summary", "iter_to": i+10, "count": len(results.get('results', []))})
                        chunks = [r.get('content','') for r in results.get('results', [])]
                        ctx = "\n\n".join(c[:800] for c in chunks)
                        prompt = f"Context:\n{ctx}\n\nTask:\n{mode_prompt}\n\nAnswer:"
                        try:
                            import requests
                            host = os.environ.get('OLLAMA_HOST','http://127.0.0.1:11434').rstrip('/')
                            model = os.environ.get('OLLAMA_MODEL','qwen2.5-coder:7b')
                            #TODO: Track prompt to debug
                            _log(f"Prompt:Summary {prompt}")
                            r = requests.post(
                                f"{host}/api/generate",
                                json={
                                    "model": model,
                                    "prompt": prompt,
                                    "stream": False,
                                    "options": {"temperature": 0.1, "num_predict": 384, "top_p": 0.9, "gpu_layers": 999}
                                },
                                timeout=180
                            )
                            text = r.json().get('response','') if r.ok else ''
                        except Exception as e:
                            text = f"[LLM error] {e}"
                        summaries.append({"iter": i+10, "summary": text})
                    st.subheader("Summaries (every 10 iterations)")
                    summary_payload = {"mode": "summary", "query": query, "summaries": summaries}
                    st.json(summary_payload)
                    if show_tree:
                        if HAS_GRAPHVIZ:
                            dot = Digraph(comment="Summary Tree")
                            dot.node("Q", f"Query: {query[:50]}…")
                            for s in summaries:
                                preview = (s.get('summary') or '')[:80]
                                label = f"iter {s['iter']}\n{preview}…"
                                node_id = f"iter{s['iter']}"
                                dot.node(node_id, label)
                                dot.edge("Q", node_id)
                            st.graphviz_chart(dot)
                        else:
                            dot_src = make_dot_tree(f"Query: {query[:50]}…", [f"iter {s['iter']}" for s in summaries])
                            st.code(dot_src, language="dot")
                else:
                    dbg_state.info(f"[mode={mode}] running search…")
                    # If multi-queries provided, run multi-query path; else single-query
                    if mq_list or (auto_multi and (not mq_list) and query.strip()):
                        # Deduplicate if requested (either user-provided or generated list)
                        before = len(mq_list)
                        if dedup_enable and mq_list:
                            mq_list = dedup_multi_queries(mq_list, similarity_threshold=float(dedup_tau_q))
                        after = len(mq_list)
                        if before != after:
                            _log(f"Multi-queries dedup: kept {after}/{before}")
                            with st.expander("Final Multi-Queries", expanded=True):
                                st.code("\n".join(mq_list))
                        elif mq_list:
                            _log(f"Multi-queries: {after}")
                        results: Dict[str, Any] = {"results": []}
                        per_query_results: Dict[str, List[Dict[str, Any]]] = {q: [] for q in mq_list}
                        total_chunks = len(docs)
                        shard_size = int(max_chunks_per_shard)
                        if shard_size <= 0 or shard_size >= total_chunks:
                            shard_ranges = [(0, total_chunks)]
                            _log("Sharding: OFF (single pass)")
                        else:
                            shard_ranges = [(i, min(i + shard_size, total_chunks)) for i in range(0, total_chunks, shard_size)]
                            _log(f"Sharding: {len(shard_ranges)} shards of size≈{shard_size}")
                        for s_start, s_end in shard_ranges:
                            shard = docs[s_start:s_end]
                            try:
                                retr.clear_documents()
                            except Exception:
                                pass
                            retr.add_documents(shard)
                            _log(f"Shard {s_start}-{s_end}: docs={len(shard)} | init simulation")
                            for qline in mq_list:
                                retr.initialize_simulation(qline)
                                prev_trails = dict(retr.pheromone_trails)
                                for step_i in range(int(max_iterations)):
                                    stats = retr.step(1)
                                    # capture light event each log window
                                    if (step_i % max(20, int(log_every)) == 0) or (step_i == int(max_iterations) - 1):
                                        # Compute trail diffs vs previous snapshot and order by (start,end)
                                        try:
                                            diffs = diff_trails(prev_trails, retr.pheromone_trails)
                                            diffs_sorted = sorted(diffs, key=lambda d: (d.get('edge',(0,0))[0], d.get('edge',(0,0))[1]))
                                        except Exception:
                                            diffs_sorted = []
                                        events_accum.append({
                                            "mode": "multi",
                                            "q": qline,
                                            "step": step_i,
                                            "avg_relevance": float(stats.get('avg_relevance', 0.0)),
                                            "trail_deltas": diffs_sorted[:50],
                                        })
                                        dbg_state.info(f"[mq shard {s_start}-{s_end}] {qline} step {step_i}/{int(max_iterations)}")
                                        _log(f"[multi] {qline} step {step_i}/{int(max_iterations)} avg_rel={float(stats.get('avg_relevance',0.0)):.3f}")
                                        # advance prev snapshot
                                        prev_trails = dict(retr.pheromone_trails)
                                    if (step_i % int(log_every) == 0 or step_i == int(max_iterations) - 1):
                                        # Compute trail diffs vs previous snapshot and order by (start,end)
                                        try:
                                            diffs = diff_trails(prev_trails, retr.pheromone_trails)
                                            diffs_sorted = sorted(diffs, key=lambda d: (d.get('edge',(0,0))[0], d.get('edge',(0,0))[1]))
                                        except Exception:
                                            diffs_sorted = []
                                        events_accum.append({
                                            "mode": "single",
                                            "step": step_i,
                                            "avg_relevance": float(stats.get('avg_relevance', 0.0)),
                                            "trail_deltas": diffs_sorted[:50],
                                        })
                                        dbg_state.info(f"[shard {s_start}-{s_end}] step {step_i}/{int(max_iterations)} avg_rel={stats.get('avg_relevance',0):.3f}")
                                        _log(f"[single] step {step_i}/{int(max_iterations)} avg_rel={float(stats.get('avg_relevance',0.0)):.3f}")
                                        prev_trails = dict(retr.pheromone_trails)
                                res_shard = retr.search(qline, top_k=int(top_k))
                                per_query_results[qline].extend(res_shard.get('results', []))
                            _log(f"Shard {s_start}-{s_end}: accumulated results updated")
                        # Diversify per query and build merged view
                        tabs = st.tabs(["Merged"] + mq_list)
                        merged_pool: List[Dict[str, Any]] = []
                        for qline in mq_list:
                            items_q = per_query_results[qline]
                            items_q_div = select_diverse_results(items_q, retr, int(top_k), float(div_alpha), float(dedup_tau), int(per_folder_cap)) if items_q else []
                            merged_pool.extend(items_q_div)
                        _log(f"Merged pool size: {len(merged_pool)}")
                        if pure_topk:
                            merged_div = sorted(merged_pool, key=lambda it: float(it.get('relevance_score', 0.0)), reverse=True)[:int(top_k)]
                        else:
                            merged_div = select_diverse_results(merged_pool, retr, int(top_k), float(div_alpha), float(dedup_tau), int(per_folder_cap)) if merged_pool else []
                        _log(f"Merged diversified count: {len(merged_div)}")
                        def _make_snippet(item: Dict[str, Any], q: str, radius: int = 5) -> Dict[str, Any]:
                            text = item.get('content') or ''
                            lines = text.splitlines()
                            # find first match position
                            try:
                                pattern = re.compile(re.escape(q), re.IGNORECASE)
                            except Exception:
                                pattern = None
                            idx = None
                            if pattern:
                                for i, ln in enumerate(lines):
                                    if pattern.search(ln):
                                        idx = i
                                        break
                            if idx is None:
                                return {**item, 'snippet': '\n'.join(lines[:min(10, len(lines))])}
                            s = max(0, idx - radius)
                            e = min(len(lines), idx + radius + 1)
                            return {**item, 'snippet': '\n'.join(lines[s:e])}

                        with tabs[0]:
                            st.subheader("Raw Results (Merged)")
                            # Build header+snippet view
                            view = []
                            for it in merged_div:
                                src = _extract_source(it.get('metadata', {}), it.get('content', ''))
                                sn = _make_snippet(it, query)
                                view.append({'file': src, 'score': it.get('relevance_score', 0.0), 'snippet': sn.get('snippet', '')})
                            st.json({'results': view})
                        # Expose merged results for tree/network rendering below
                        results = {"results": merged_div}
                    else:
                        # Single-query sharded search
                        aggregated_items: List[Dict[str, Any]] = []
                        total_chunks = len(docs)
                        shard_size = int(max_chunks_per_shard)
                        if shard_size <= 0 or shard_size >= total_chunks:
                            shard_ranges = [(0, total_chunks)]
                            _log("Sharding: OFF (single pass)")
                        else:
                            shard_ranges = [(i, min(i + shard_size, total_chunks)) for i in range(0, total_chunks, shard_size)]
                            _log(f"Sharding: {len(shard_ranges)} shards of size≈{shard_size}")
                        for s_start, s_end in shard_ranges:
                            shard = docs[s_start:s_end]
                            try:
                                retr.clear_documents()
                            except Exception:
                                pass
                            retr.add_documents(shard)
                            _log(f"Shard {s_start}-{s_end}: docs={len(shard)} | init simulation")
                            retr.initialize_simulation(query)
                            prev_trails = dict(retr.pheromone_trails)
                            for step_i in range(int(max_iterations)):
                                stats = retr.step(1)
                                if (step_i % max(5, int(log_every)) == 0) or (step_i == int(max_iterations) - 1):
                                    try:
                                        diffs = diff_trails(prev_trails, retr.pheromone_trails)
                                        diffs_sorted = sorted(diffs, key=lambda d: (d.get('edge',(0,0))[0], d.get('edge',(0,0))[1]))
                                    except Exception:
                                        diffs_sorted = []
                                    events_accum.append({
                                        "mode": "single",
                                        "step": step_i,
                                        "avg_relevance": float(stats.get('avg_relevance', 0.0)),
                                        "trail_deltas": diffs_sorted[:50],
                                    })
                                    dbg_state.info(f"[shard {s_start}-{s_end}] step {step_i}/{int(max_iterations)} avg_rel={stats.get('avg_relevance',0):.3f}")
                                    _log(f"[single] step {step_i}/{int(max_iterations)} avg_rel={float(stats.get('avg_relevance',0.0)):.3f}")
                                    prev_trails = dict(retr.pheromone_trails)
                                if (step_i % int(log_every) == 0 or step_i == int(max_iterations) - 1):
                                    try:
                                        diffs = diff_trails(prev_trails, retr.pheromone_trails)
                                        diffs_sorted = sorted(diffs, key=lambda d: (d.get('edge',(0,0))[0], d.get('edge',(0,0))[1]))
                                    except Exception:
                                        diffs_sorted = []
                                    events_accum.append({
                                        "mode": "single",
                                        "step": step_i,
                                        "avg_relevance": float(stats.get('avg_relevance', 0.0)),
                                        "trail_deltas": diffs_sorted[:50],
                                    })
                                    dbg_state.info(f"[shard {s_start}-{s_end}] step {step_i}/{int(max_iterations)} avg_rel={stats.get('avg_relevance',0):.3f}")
                                    _log(f"[single] step {step_i}/{int(max_iterations)} avg_rel={float(stats.get('avg_relevance',0.0)):.3f}")
                                    prev_trails = dict(retr.pheromone_trails)
                            res_shard = retr.search(query, top_k=int(top_k))
                            aggregated_items.extend(res_shard.get('results', []))
                            _log(f"Shard {s_start}-{s_end}: results+={len(res_shard.get('results', []))}")
                        if pure_topk:
                            sorted_items = sorted(aggregated_items, key=lambda it: float(it.get('relevance_score', 0.0)), reverse=True)[:int(top_k)]
                            results = {"query": query, "results": sorted_items}
                        else:
                            items_diverse = select_diverse_results(aggregated_items, retr, int(top_k), float(div_alpha), float(dedup_tau), int(per_folder_cap)) if aggregated_items else []
                            results = {"query": query, "results": items_diverse}
                        _log(f"Aggregated items: {len(aggregated_items)} | Final results: {len(results.get('results', []))}")
                        st.subheader("Raw Results")
                        # Build header+snippet view
                        items = results.get('results', [])
                        view = []
                        for it in items:
                            src = _extract_source(it.get('metadata', {}), it.get('content', ''))
                            sn = _make_snippet(it, query)
                            view.append({'file': src, 'score': it.get('relevance_score', 0.0), 'snippet': sn.get('snippet', '')})
                        st.json({'results': view})
                    # Optional agent summary separated from raw results
                    if gen_answer and results and results.get('results'):
                        try:
                            st.subheader("Agent Summary")
                            top_items = results.get('results', [])[:int(top_k)]
                            context_text = "\n\n".join([(it.get('content') or '')[:800] for it in top_items])
                            prompt_text = (
                                f"Context:\n{context_text}\n\n"
                                f"Task:\n{mode_prompt or 'Summarize key findings and answer the query.'}\n\nAnswer:"
                            )
                            summary = _ollama_generate(prompt_text)
                            st.write(summary)
                        except Exception as _se:
                            st.info(f"Summary unavailable: {_se}")
                    # After any path, if we recorded snapshots, build and display GIF
                    # (Removed GIF functionality by request)
            if show_network:
                # Debounce visual updates and send only deltas vs. previous snapshot
                if 'last_snapshot' not in st.session_state:
                    st.session_state.last_snapshot = None
                snap = retr.get_visualization_snapshot()
                last = st.session_state.last_snapshot or {"documents": {"xy": [], "relevance": []}, "agents": {"xy": []}, "edges": []}
                def snapshot_delta(curr, prev):
                    try:
                        # Only diff edges count and doc relevance, keep positions as-is for now
                        delta = {
                            "documents": {"xy": curr.get("documents", {}).get("xy", []),
                                           "relevance": curr.get("documents", {}).get("relevance", [])},
                            "agents": {"xy": curr.get("agents", {}).get("xy", [])},
                            "edges": curr.get("edges", [])
                        }
                        return delta
                    except Exception:
                        return curr
                snap = snapshot_delta(snap, last)
                st.session_state.last_snapshot = snap
                docs_xy = snap.get("documents", {}).get("xy", [])
                docs_rel = snap.get("documents", {}).get("relevance", [])
                agents_xy = snap.get("agents", {}).get("xy", [])
                edges = snap.get("edges", [])
                fig = go.Figure()
                for e in edges:
                    fig.add_trace(go.Scatter(x=[e['x0'], e['x1']], y=[e['y0'], e['y1']], mode='lines', line=dict(width=max(1, e['s'] * 3), color='rgba(0,150,0,0.3)'), hoverinfo='none', showlegend=False))
                if docs_xy:
                    xs = [p[0] for p in docs_xy]
                    ys = [p[1] for p in docs_xy]
                    sizes = [8 + 12*float(r) for r in (docs_rel or [0]*len(xs))]
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=sizes, color='rgba(0,0,200,0.6)'), name='docs'))
                if agents_xy:
                    xa = [p[0] for p in agents_xy]
                    ya = [p[1] for p in agents_xy]
                    fig.add_trace(go.Scatter(x=xa, y=ya, mode='markers', marker=dict(size=3, color='rgba(200,0,0,0.5)'), name='agents'))
                fig.update_layout(title="Pheromone Network Snapshot", xaxis_title="x", yaxis_title="y", height=500)
                st.plotly_chart(fig, use_container_width=True)

# ---------------- Agent Chat & Background Report ----------------

if enable_agent_chat:

    st.divider()
    st.subheader("💬 Agent Chat (search via tools)")

    if 'agent_messages' not in st.session_state:
        st.session_state.agent_messages = []

    # Show history
    for m in st.session_state.agent_messages:
        st.chat_message(m.get('role', 'assistant')).write(m.get('content', ''))

    # Build agent if requested
    lc_agent = build_langchain_agent_if_available(settings_obj) if prefer_langchain else None

    prompt_text = st.chat_input("Ask about this codebase…")
    if prompt_text:
        st.session_state.agent_messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                if lc_agent is not None:
                    try:
                        res = lc_agent.invoke({"input": prompt_text})
                        if isinstance(res, dict) and "output" in res:
                            reply = res["output"]
                        else:
                            reply = str(res)
                    except Exception as e:
                        reply = f"Agent error: {e}"
                else:
                    # Fallback: run a quick search and summarize with local Ollama
                    quick = quick_search_with_mcmp(settings_obj, prompt_text, int(top_k))
                    if quick.get('error'):
                        reply = quick['error']
                    else:
                        items = quick.get('results', [])
                        summary_ctx = "\n\n".join([(it.get('content') or '')[:800] for it in items])
                        sys_prompt = (
                            "You are a helpful code assistant. Summarize the key findings from the CONTEXT to answer the USER question.\n"
                            "Cite file paths when visible. Keep it concise."
                        )
                        prompt = f"{sys_prompt}\n\nQUESTION: {prompt_text}\n\nCONTEXT:\n{summary_ctx}\n\nANSWER:"
                        reply = _ollama_generate(prompt)
                st.write(reply)
                st.session_state.agent_messages.append({"role": "assistant", "content": reply})

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Start background report"):
            start_background_report(settings_obj, query or prompt_text or "Repository analysis")
            st.info("Report started. You can continue chatting while it runs.")
    with cols[1]:
        # Live progress view instead of manual check
        job_id = st.session_state.get('report_job_id')
        if job_id:
            progress_file = os.path.join(REPORTS_DIR, f"progress_{job_id}.json")
            try:
                if os.path.exists(progress_file):
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        prog = json.load(f)
                else:
                    prog = {"status": "unknown", "percent": 0}
            except Exception:
                prog = {"status": "unknown", "percent": 0}
            pct = int(prog.get('percent', 0))
            status = str(prog.get('status', 'unknown'))
            msg = str(prog.get('message', ''))
            st.progress(min(max(pct, 0), 100) / 100.0, text=f"{status}: {msg}")
            if status == 'done' and prog.get('report_path'):
                st.success("Report ready")
                try:
                    with open(prog['report_path'], 'r', encoding='utf-8') as f:
                        report_payload = json.load(f)
                    st.json(report_payload)
                except Exception as e:
                    st.error(f"Failed to load report: {e}")
        else:
            st.info("No background report started yet.")

if st.session_state.get('show_docs'):
    with st.expander("Docs", expanded=True):
        st.markdown(DOCS_MD)
else:
    st.info("Run: streamlit run streamlit_fungus.py")


