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
    try:
        from embeddinggemma.rag.generation import generate_with_ollama  # type: ignore
    except Exception:
        generate_with_ollama = None  # type: ignore
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

# Optional simulation telemetry helper
try:
    from embeddinggemma.mcmp.simulation import log_simulation_step as _sim_log  # type: ignore
except Exception:
    _sim_log = None  # type: ignore

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

    def _sim_metrics(_retr) -> Dict[str, Any]:
        try:
            docs = getattr(_retr, 'documents', [])
            agents = getattr(_retr, 'agents', [])
            avg_rel = float(sum(getattr(d, 'relevance_score', 0.0) for d in docs) / max(1, len(docs))) if docs else 0.0
            max_rel = float(max((getattr(d, 'relevance_score', 0.0) for d in docs), default=0.0)) if docs else 0.0
            trails = len(getattr(_retr, 'pheromone_trails', {}))
            avg_speed = 0.0
            if agents:
                import numpy as _np
                avg_speed = float(_np.mean([float(_np.linalg.norm(getattr(a, 'velocity', 0.0))) for a in agents]))
            return {"avg_rel": avg_rel, "max_rel": max_rel, "trails": trails, "avg_speed": avg_speed,
                    "docs": len(docs), "agents": len(agents)}
        except Exception:
            return {"avg_rel": 0.0, "max_rel": 0.0, "trails": 0, "avg_speed": 0.0, "docs": 0, "agents": 0}

    for s_start, s_end in shard_ranges:
        shard = docs[s_start:s_end]
        try:
            retr.clear_documents()
        except Exception:
            pass
        retr.add_documents(shard)
        _log(f"Shard {s_start}-{s_end}: docs={len(shard)} | init simulation")
        retr.initialize_simulation(query)

        # Live network snapshot placeholder + sidebar controls
        net_placeholder = None
        # Wrap controls in a form to avoid reruns during adjustment
        with st.sidebar.form("viz_controls"):
            redraw_every_in = st.number_input('Redraw every N steps', min_value=1, max_value=50, value=int(ui_settings.get('redraw_every', max(1, int(ui_settings.get('log_every', 1)) ))))
            min_trail_in = st.slider('Min trail strength', min_value=0.0, max_value=1.0, value=float(ui_settings.get('min_trail_strength', 0.05)), step=0.01)
            max_edges_in = st.number_input('Max edges', min_value=50, max_value=2000, value=int(ui_settings.get('max_edges', 500)), step=50)
            viz_dims_in = st.selectbox('Viz dimensions', options=[2,3], index=0 if int(ui_settings.get('viz_dims', 2)) == 2 else 1)
            apply_controls = st.form_submit_button("Apply")
        if apply_controls:
            ui_settings['redraw_every'] = int(redraw_every_in)
            ui_settings['min_trail_strength'] = float(min_trail_in)
            ui_settings['max_edges'] = int(max_edges_in)
            ui_settings['viz_dims'] = int(viz_dims_in)

        min_trail = float(ui_settings.get('min_trail_strength', 0.05))
        max_edges = int(ui_settings.get('max_edges', 500))
        redraw_every = max(1, int(ui_settings.get('redraw_every', max(1, int(ui_settings.get('log_every', 1))))))
        viz_dims = int(ui_settings.get('viz_dims', 2))
        if ui_settings.get('show_network', True) and go is not None:
            net_placeholder = st.empty()

        for _i in range(int(ui_settings['max_iterations'])):
            retr.step(1)
            if _i % max(1, int(ui_settings.get('log_every', 1))) == 0:
                try:
                    if _sim_log is not None:
                        _sim_log(retr, _i)
                except Exception:
                    pass
                m = _sim_metrics(retr)
                _log(f"step={_i} avg_rel={m['avg_rel']:.4f} max_rel={m['max_rel']:.4f} trails={m['trails']} avg_speed={m['avg_speed']:.4f}")

                # Live redraw of the network snapshot
                if net_placeholder is not None and (_i % redraw_every == 0):
                    try:
                        snap = retr.get_snapshot(min_trail_strength=min_trail, max_edges=max_edges, method="pca", whiten=False,)
                        docs_xy = snap.get("documents", {}).get("xy", [])
                        docs_rel = snap.get("documents", {}).get("relevance", [])
                        agents_xy = snap.get("agents", {}).get("xy", [])
                        edges = snap.get("edges", [])
                        if viz_dims == 3:
                            fig = go.Figure()
                            # Edges: draw with true z if provided
                            for e in edges:
                                if 'z0' in e and 'z1' in e:
                                    fig.add_trace(go.Scatter3d(x=[e['x0'], e['x1']], y=[e['y0'], e['y1']], z=[e['z0'], e['z1']], mode='lines', line=dict(width=max(1, e['s'] * 3), color='rgba(0,150,0,0.25)'), showlegend=False))
                                else:
                                    fig.add_trace(go.Scatter3d(x=[e['x0'], e['x1']], y=[e['y0'], e['y1']], z=[0,0], mode='lines', line=dict(width=max(1, e['s'] * 3), color='rgba(0,150,0,0.25)'), showlegend=False))
                            if docs_xy:
                                xs = [p[0] for p in docs_xy]
                                ys = [p[1] for p in docs_xy]
                                zs = [p[2] if len(p) > 2 else 0 for p in docs_xy]
                                sizes = [4 + 10*float(r) for r in (docs_rel or [0]*len(xs))]
                                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(size=sizes, color=docs_rel or [0]*len(xs), colorscale='Viridis', opacity=0.8), name='docs'))
                            if agents_xy:
                                xa = [p[0] for p in agents_xy]
                                ya = [p[1] for p in agents_xy]
                                za = [p[2] if len(p) > 2 else 0 for p in agents_xy]
                                fig.add_trace(go.Scatter3d(x=xa, y=ya, z=za, mode='markers', marker=dict(size=3, color='rgba(200,0,0,0.6)'), name='agents'))
                            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=600, title=f"Live Pheromone Network 3D (step={_i}, trails={len(edges)})")
                            net_placeholder.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = go.Figure()
                            for e in edges:
                                fig.add_trace(go.Scatter(x=[e['x0'], e['x1']], y=[e['y0'], e['y1']], mode='lines', line=dict(width=max(1, e['s'] * 3), color='rgba(0,150,0,0.25)'), hoverinfo='none', showlegend=False))
                            if docs_xy:
                                xs = [p[0] for p in docs_xy]
                                ys = [p[1] for p in docs_xy]
                                sizes = [8 + 12*float(r) for r in (docs_rel or [0]*len(xs))]
                                fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=sizes, color='rgba(0,0,200,0.6)'), name='docs'))
                            if agents_xy:
                                xa = [p[0] for p in agents_xy]
                                ya = [p[1] for p in agents_xy]
                                fig.add_trace(go.Scatter(x=xa, y=ya, mode='markers', marker=dict(size=3, color='rgba(200,0,0,0.5)'), name='agents'))
                            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=500, title=f"Live Pheromone Network (step={_i}, trails={len(edges)})", xaxis_title="x", yaxis_title="y")
                            net_placeholder.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
        # Final metrics for shard
        m_final = _sim_metrics(retr)
        _log(f"final shard metrics: docs={m_final['docs']} agents={m_final['agents']} avg_rel={m_final['avg_rel']:.4f} max_rel={m_final['max_rel']:.4f} trails={m_final['trails']}")
        res_shard = retr.search(query, top_k=int(ui_settings['top_k']))
        aggregated_items.extend(res_shard.get('results', []))
        _log(f"Shard {s_start}-{s_end}: results+={len(res_shard.get('results', []))}")

    # Select final
    if ui_settings['pure_topk']:
        items = sorted(aggregated_items, key=lambda it: float(it.get('relevance_score', 0.0)), reverse=True)[:int(ui_settings['top_k'])]
    else:
        items = select_diverse_results(
            aggregated_items,
            retr,
            int(ui_settings['top_k']),
            float(ui_settings['div_alpha']),
            float(ui_settings['dedup_tau']),
            int(ui_settings['per_folder_cap'])
        )

    # Render
    st.subheader("Raw Results")
    view = []
    for it in items:
        src = (it.get('metadata', {}) or {}).get('file_path') or re.search(r"^# file: (.+?) \| lines:", (it.get('content') or ''), re.MULTILINE).group(1) if re.search(r"^# file:", (it.get('content') or ''), re.MULTILINE) else 'unknown'
        sn = snippet_for(it, query)
        view.append({'file': src, 'score': it.get('relevance_score', 0.0), 'snippet': sn})
    st.json({'results': view})

    # Optional answer generation using mode prompt
    if ui_settings.get('gen_answer', False):
        def _task_for_mode(mode: str, q: str) -> str:
            m = (mode or '').lower()
            if m == 'deep':
                return f"Führe eine tiefe Analyse durch und beantworte präzise: {q}\\nNenne Belege aus den Snippets."
            if m == 'structure':
                return f"Analysiere Funktionen/Klassen, extrahiere relevante Definitionen und beantworte: {q}"
            if m == 'exploratory':
                return f"Beantworte explorativ: {q}. Stelle ggf. Anschlussfragen."
            if m == 'summary':
                return f"Fasse die wichtigsten Informationen zu '{q}' zusammen und liste Quellen."
            if m == 'repair':
                return f"Schlage konkrete Verbesserungen/Refactorings vor basierend auf dem Kontext zur Frage: {q}"
            # 'similar' / 'redundancy' or default
            return q

        ctx_items = items if not ui_settings.get('show_all_scored') else aggregated_items
        context = "\n\n".join([(it.get('content') or '')[:800] for it in ctx_items])
        task = _task_for_mode(ui_settings.get('mode', ''), query)
        llm_prompt = f"Kontext:\n{context}\n\nAufgabe:\n{task}\n\nAntwort:".strip()
        st.subheader("Answer (mode prompt)")
        answer_text = ""
        if generate_with_ollama is not None:
            host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
            model = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
            answer_text = generate_with_ollama(llm_prompt, model=model, host=host)
        else:
            answer_text = "[LLM unavailable] Install Ollama and ensure embeddinggemma.rag.generation is importable."
        st.write(answer_text)

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
