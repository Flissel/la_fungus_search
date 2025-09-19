from __future__ import annotations
from typing import Dict, Any, List
import os
import re
import streamlit as st

try:
    from graphviz import Digraph  # type: ignore
    HAS_GRAPHVIZ = True
except Exception:
    HAS_GRAPHVIZ = False

try:
    import plotly.graph_objects as go  # noqa: F401
except Exception:
    pass


def sidebar(settings: Dict[str, Any]) -> Dict[str, Any]:
    sb = st.sidebar
    sb.header("Settings")
    mode = sb.selectbox("Mode", ["deep", "structure", "exploratory", "summary", "similar", "redundancy", "repair", "Rag"], index=0)
    top_k = sb.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
    windows_str = sb.text_input("Windows (lines)", value=",".join(str(w) for w in settings.get('windows', [50,100,200,300,400])))
    windows = [int(x.strip()) for x in windows_str.split(',') if x.strip().isdigit()]
    use_repo = sb.checkbox("Use code space (src) as corpus", value=True)
    root_folder = sb.text_input("Root folder (used when src is off)", value=os.getcwd())
    max_files = sb.number_input("Max files to index (0 = no limit)", min_value=0, max_value=10000, value=1000, step=50)
    exclude_str = sb.text_input("Exclude folders (comma-separated substrings)", value=".venv,node_modules,.git,external")
    exclude_dirs = [e.strip() for e in exclude_str.split(',') if e.strip()]
    chunk_workers = sb.number_input("Chunk workers (threads)", min_value=1, max_value=64, value=32, step=1)
    num_agents = sb.number_input("Agents", min_value=50, max_value=5000, value=200, step=50)
    max_iterations = sb.number_input("Max iterations", min_value=10, max_value=1000, value=60, step=10)
    show_tree = sb.checkbox("Show results tree", value=True)
    show_network = sb.checkbox("Show pheromone network snapshot", value=True)
    gen_answer = sb.checkbox("Generate answer with LLM using mode prompt", value=(mode in ["summary","exploratory","structure"]))
    div_alpha = sb.slider("MMR alpha (relevance vs. diversity)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    dedup_tau = sb.slider("Dedup cosine threshold", min_value=0.80, max_value=0.99, value=0.92, step=0.01)
    per_folder_cap = sb.number_input("Per-folder cap", min_value=0, max_value=10, value=2, step=1)
    pure_topk = sb.checkbox("Pure Top-K (disable diversity)", value=False)
    log_every = sb.number_input("Log every N iterations", min_value=1, max_value=100, value=10, step=1)
    exploration_bonus = sb.slider("Exploration bonus", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    pheromone_decay = sb.slider("Pheromone decay", min_value=0.80, max_value=0.999, value=0.95, step=0.001)
    embed_bs = sb.number_input("Embedding batch size", min_value=16, max_value=1024, value=64, step=16)
    max_chunks_per_shard = sb.number_input("Max chunks per shard (0 = no sharding)", min_value=0, max_value=20000, value=2000, step=100)
    enable_agent_chat = sb.checkbox("Enable agent chat", value=True)
    return {
        "mode": mode,
        "top_k": int(top_k),
        "windows": windows,
        "use_repo": bool(use_repo),
        "root_folder": root_folder,
        "max_files": int(max_files),
        "exclude_dirs": exclude_dirs,
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
        "chunk_workers_override": int(chunk_workers),
        "enable_agent_chat": bool(enable_agent_chat),
    }


def live_log():
    log_exp = st.expander("Live log", expanded=True)
    return log_exp.empty()


def render_results_tree(query: str, items: List[Dict[str, Any]]):
    if HAS_GRAPHVIZ:
        try:
            dot = Digraph(comment="Results Tree")
            dot.node("Q", f"Query: {query[:50]}…")
            for idx, it in enumerate(items):
                score = it.get('relevance_score', 0.0)
                src = (it.get('metadata', {}) or {}).get('file_path', 'chunk')
                label = f"#{idx+1} {score:.3f}\n{src}"
                nid = f"n{idx+1}"
                dot.node(nid, label)
                dot.edge("Q", nid)
            st.graphviz_chart(dot)
            return
        except Exception:
            pass
    labels = []
    for idx, it in enumerate(items):
        score = it.get('relevance_score', 0.0)
        src = ( it.get('metadata', {}) or {}).get('file_path', 'chunk')
        labels.append(f"#{idx+1} {score:.3f} {src}")
    st.code("\n".join(labels))


def snippet_for(item: Dict[str, Any], q: str, radius: int = 5) -> str:
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
        return "\n".join(lines[:min(10, len(lines))])
    s = max(0, idx - radius)
    e = min(len(lines), idx + radius + 1)
    return "\n".join(lines[s:e])
