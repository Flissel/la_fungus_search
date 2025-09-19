#!/usr/bin/env python3
import os
import sys
import re
import json
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None  # type: ignore

# Ensure 'src' is on sys.path so `embeddinggemma.*` imports work when running from repo root
_SRC_PATH = os.path.abspath(os.path.join(os.getcwd(), "src"))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Imports from refactored UI modules
try:
    from embeddinggemma.ui.state import Settings, init_session  # type: ignore
    from embeddinggemma.ui.corpus import collect_codebase_chunks, list_code_files, chunk_python_file  # type: ignore
    from embeddinggemma.ui.queries import generate_multi_queries_from_llm, dedup_multi_queries  # type: ignore
    from embeddinggemma.ui.mcmp_runner import select_diverse_results, quick_search_with_mcmp  # type: ignore
    from embeddinggemma.ui.agent import build_langchain_agent_if_available  # type: ignore
    from embeddinggemma.ui.components import sidebar, live_log, render_results_tree, snippet_for  # type: ignore
except Exception as e:
    st.error(f"Failed to import refactored UI modules: {e}")

# Try to import retriever
try:
    from embeddinggemma.mcmp_rag import MCPMRetriever  # type: ignore
except Exception:
    try:
        from mcmp_rag import MCPMRetriever  # type: ignore
    except Exception as e:
        MCPMRetriever = None  # type: ignore
        st.error(f"MCPMRetriever Import failed: {e}")

st.set_page_config(page_title="Fungus Backup", layout="wide")
st.title("Fungus (MCMP) Backup Frontend")

init_session(st.session_state)

# Sidebar
defaults = Settings().to_dict()
ui_settings = sidebar(defaults)

# Query input
query = st.text_input(
    "Query",
    value=(
        "Explain how RagV1 and Fungus MCPM are implemented, and how to run them."
    ),
)
run = st.button("Run")

# Live log
log_box = live_log()
_logs: List[str] = []


def _log(msg: str) -> None:
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        _logs.append(f"[{ts}] {msg}")
        log_box.code("\n".join(_logs[-300:]))
    except Exception:
        pass

if run:
    if MCPMRetriever is None:
        st.error("MCPMRetriever unavailable.")
        st.stop()

    _log(f"mode={ui_settings['mode']} | top_k={ui_settings['top_k']} | windows={ui_settings['windows']} | use_repo={ui_settings['use_repo']} | max_files={ui_settings['max_files']}")

    # Build corpus
    docs: List[str] = []
    discovered_files: List[str] = []

    if ui_settings.get('use_repo', True):
        discovered_files = list_code_files('src', int(ui_settings['max_files']), ui_settings['exclude_dirs'])
        docs.extend(collect_codebase_chunks('src', ui_settings['windows'], int(ui_settings['max_files']), ui_settings['exclude_dirs']))
    else:
        rf = ui_settings['root_folder'].strip()
        if rf and os.path.isdir(rf):
            _log(f"Crawl root: {rf}")
            discovered_files = list_code_files(rf, int(ui_settings['max_files']), ui_settings['exclude_dirs'])
            docs.extend(collect_codebase_chunks(rf, ui_settings['windows'], int(ui_settings['max_files']), ui_settings['exclude_dirs']))
        else:
            st.warning("Root folder is not set or does not exist. Please provide a valid path.")

    if not docs:
        st.error("No documents found to analyze.")
        st.stop()

    st.expander("Discovered code files", expanded=False).code("\n".join([os.path.relpath(p) for p in discovered_files[:200]]))

    # Instantiate retriever
    retr = MCPMRetriever(
        num_agents=int(ui_settings['num_agents']),
        max_iterations=int(ui_settings['max_iterations']),
        exploration_bonus=float(ui_settings['exploration_bonus']),
        pheromone_decay=float(ui_settings['pheromone_decay']),
        embed_batch_size=int(ui_settings['embed_bs'])
    )
    try:
        retr.log_every = int(ui_settings['log_every'])
    except Exception:
        pass

    retr.add_documents(docs)
    _log(f"Embeddings added to retriever | Chunks: {len(docs)}")

    # Single-query run (no Rag section, no background reports)
    aggregated_items: List[Dict[str, Any]] = []
    total_chunks = len(docs)
    shard_size = int(ui_settings['max_chunks_per_shard'])
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
        for _ in range(int(ui_settings['max_iterations'])):
            retr.step(1)
        res_shard = retr.search(query, top_k=int(ui_settings['top_k']))
        aggregated_items.extend(res_shard.get('results', []))
        _log(f"Shard {s_start}-{s_end}: results+={len(res_shard.get('results', []))}")

    # Select final
    if ui_settings['pure_topk']:
        items = sorted(aggregated_items, key=lambda it: float(it.get('relevance_score', 0.0)), reverse=True)[:int(ui_settings['top_k'])]
    else:
        items = select_diverse_results(aggregated_items, retr, int(ui_settings['top_k']), float(ui_settings['div_alpha']), float(ui_settings['dedup_tau']), int(ui_settings['per_folder_cap']))

    # Render
    st.subheader("Raw Results")
    view = []
    for it in items:
        src = (it.get('metadata', {}) or {}).get('file_path') or re.search(r"^# file: (.+?) \| lines:", (it.get('content') or ''), re.MULTILINE).group(1) if re.search(r"^# file:", (it.get('content') or ''), re.MULTILINE) else 'unknown'
        sn = snippet_for(it, query)
        view.append({'file': src, 'score': it.get('relevance_score', 0.0), 'snippet': sn})
    st.json({'results': view})

    if ui_settings.get('show_network', True):
        if go is None:
            st.info("Plotly is not installed. Install with: pip install plotly")
        else:
            try:
                snap = retr.get_visualization_snapshot(min_trail_strength=0.0, max_edges=800, method="pca", whiten=False, spread=1.0, jitter=0.0)
                docs_xy = snap.get("documents", {}).get("xy", [])
                docs_rel = snap.get("documents", {}).get("relevance", [])
                agents_xy = snap.get("agents", {}).get("xy", [])
                edges = snap.get("edges", [])
                if not docs_xy and not edges and not agents_xy:
                    st.info("Network snapshot is empty. This can happen if no trails were formed. Try lowering 'Pheromone decay' or increasing 'Max iterations'.")
                else:
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
                    fig.update_layout(title=f"Pheromone Network Snapshot (trails={len(edges)}, docs={len(docs_xy)}, agents={len(agents_xy)})", xaxis_title="x", yaxis_title="y", height=500)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as _:
                st.info("Could not render network snapshot (unexpected error).")

    if ui_settings['show_tree']:
        render_results_tree(query, items)

# Agent Chat (simple)
st.divider()
if ui_settings.get('enable_agent_chat', True):
    st.subheader("Agent Chat (backup)")
    lc_agent = build_langchain_agent_if_available(ui_settings)
    prompt_text = st.chat_input("Ask about this codebase…")
    if prompt_text:
        st.chat_message("user").write(prompt_text)
        with st.chat_message("assistant"):
            if lc_agent is not None:
                try:
                    res = lc_agent.invoke({"input": prompt_text})
                    reply = res["output"] if isinstance(res, dict) and "output" in res else str(res)
                except Exception as e:
                    reply = f"Agent error: {e}"
            else:
                quick = quick_search_with_mcmp(ui_settings, prompt_text, int(ui_settings.get('top_k', 5)))
                if quick.get('error'):
                    reply = quick['error']
                else:
                    items = quick.get('results', [])
                    summary_ctx = "\n\n".join([(it.get('content') or '')[:800] for it in items])
                    reply = f"Top results:\n\n{summary_ctx[:1200]}"
            st.write(reply)
