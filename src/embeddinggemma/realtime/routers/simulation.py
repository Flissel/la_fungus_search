"""Simulation control router - start, stop, reset, pause, resume, config, status endpoints."""

from __future__ import annotations
import asyncio
import logging
import os
import json
import glob
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embeddinggemma.realtime.server import SnapshotStreamer

from embeddinggemma.realtime.services.settings_manager import SettingsModel, SETTINGS_DIR
from embeddinggemma.ui.reports import create_run_summary

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["simulation"])

# This will be set by server.py after import
_get_streamer_dependency: Any = None

def get_streamer() -> Any:
    """Get streamer from dependency."""
    if _get_streamer_dependency is None:
        raise RuntimeError("Streamer dependency not configured")
    return _get_streamer_dependency()


# Import save_settings_to_disk function reference - will be set by server.py
_save_settings_to_disk: Any = None

def save_settings_to_disk() -> None:
    """Save settings to disk - delegates to server module function."""
    if _save_settings_to_disk is None:
        raise RuntimeError("save_settings_to_disk not configured")
    _save_settings_to_disk()


@router.post("/start")
async def http_start(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    raw = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    body = SettingsModel(**raw)
    try:
        await streamer._broadcast({"type": "log", "message": f"api:/start payload_keys={list(raw.keys()) if isinstance(raw, dict) else 'non-json'}"})
    except Exception:
        pass
    # log values which are provided
    try:
        applied_dict = {}
        try:
            applied_dict = body.model_dump(exclude_none=True)  # pydantic v2
        except Exception:
            applied_dict = {k: v for k, v in getattr(body, '__dict__', {}).items() if v is not None}
        await streamer._broadcast({"type": "log", "message": "api:/start applied: " + " ".join([f"{k}={applied_dict[k]}" for k in applied_dict])})
    except Exception:
        pass
    if body.query is not None:
        streamer.query = body.query
    if body.redraw_every is not None:
        streamer.redraw_every = int(body.redraw_every)
    if body.min_trail_strength is not None:
        streamer.min_trail_strength = float(body.min_trail_strength)
    if body.max_edges is not None:
        streamer.max_edges = int(body.max_edges)
    if body.viz_dims is not None:
        streamer.viz_dims = int(body.viz_dims)
    if body.use_repo is not None:
        streamer.use_repo = bool(body.use_repo)
    if body.root_folder is not None:
        streamer.root_folder = str(body.root_folder)
    if body.max_files is not None:
        streamer.max_files = int(body.max_files)
    if body.exclude_dirs is not None:
        streamer.exclude_dirs = [str(x) for x in body.exclude_dirs]
    if body.windows is not None:
        streamer.windows = [int(x) for x in body.windows]
    if body.chunk_workers is not None:
        streamer.chunk_workers = int(body.chunk_workers)
    if body.max_iterations is not None:
        streamer.max_iterations = int(body.max_iterations)
    if body.num_agents is not None:
        streamer.num_agents = int(body.num_agents)
    if body.exploration_bonus is not None:
        streamer.exploration_bonus = float(body.exploration_bonus)
    if body.pheromone_decay is not None:
        streamer.pheromone_decay = float(body.pheromone_decay)
    if body.embed_batch_size is not None:
        streamer.embed_batch_size = int(body.embed_batch_size)
    if body.max_chunks_per_shard is not None:
        streamer.max_chunks_per_shard = int(body.max_chunks_per_shard)
    if body.top_k is not None:
        streamer.top_k = int(body.top_k)
    if getattr(body, 'report_enabled', None) is not None:
        streamer.report_enabled = bool(getattr(body, 'report_enabled'))
    if getattr(body, 'report_every', None) is not None:
        streamer.report_every = int(getattr(body, 'report_every'))
    if getattr(body, 'report_mode', None) is not None:
        streamer.report_mode = str(getattr(body, 'report_mode'))
    if getattr(body, 'judge_mode', None) is not None:
        streamer.judge_mode = str(getattr(body, 'judge_mode'))
    if getattr(body, 'mq_enabled', None) is not None:
        streamer.mq_enabled = bool(getattr(body, 'mq_enabled'))
    if getattr(body, 'mq_count', None) is not None:
        streamer.mq_count = max(1, min(int(getattr(body, 'mq_count')), 3))
    # follow-up and budget controls
    if getattr(body, 'token_cap', None) is not None:
        try:
            streamer.token_cap = int(getattr(body, 'token_cap'))
        except Exception:
            streamer.token_cap = None
    if getattr(body, 'cost_cap_usd', None) is not None:
        try:
            streamer.cost_cap_usd = float(getattr(body, 'cost_cap_usd'))
        except Exception:
            streamer.cost_cap_usd = None
    if getattr(body, 'stagnation_threshold', None) is not None:
        streamer.stagnation_threshold = max(3, int(getattr(body, 'stagnation_threshold')))
    if getattr(body, 'query_pool_cap', None) is not None:
        streamer._query_pool_cap = max(10, int(getattr(body, 'query_pool_cap')))
    # contextual steering settings
    for k in ["alpha","beta","gamma","delta","epsilon","min_content_chars","import_only_penalty","max_reports","max_report_tokens","judge_enabled"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    # LLM configuration overrides
    for k in ["ollama_model","ollama_host","ollama_system","ollama_num_gpu","ollama_num_thread","ollama_num_batch"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    for k in ["llm_provider","openai_model","openai_api_key","openai_base_url","openai_temperature"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    for k in ["google_model","google_api_key","google_base_url","google_temperature","grok_model","grok_api_key","grok_base_url","grok_temperature"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    # OpenAI overrides
    for k in ["llm_provider","openai_model","openai_api_key","openai_base_url","openai_temperature"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    await streamer.start()
    save_settings_to_disk()
    return JSONResponse({"status": "ok"})


@router.post("/config")
async def http_config(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    raw = await req.json()
    body = SettingsModel(**raw)
    try:
        await streamer._broadcast({"type": "log", "message": f"api:/config payload_keys={list(raw.keys())}"})
    except Exception:
        pass
    # log values which will be applied
    try:
        applied_dict = {}
        try:
            applied_dict = body.model_dump(exclude_none=True)
        except Exception:
            applied_dict = {k: v for k, v in getattr(body, '__dict__', {}).items() if v is not None}
        await streamer._broadcast({"type": "log", "message": "api:/config applied: " + " ".join([f"{k}={applied_dict[k]}" for k in applied_dict])})
    except Exception:
        pass
    if body.redraw_every is not None:
        streamer.redraw_every = int(body.redraw_every)
    if body.min_trail_strength is not None:
        streamer.min_trail_strength = float(body.min_trail_strength)
    if body.max_edges is not None:
        streamer.max_edges = int(body.max_edges)
    if body.viz_dims is not None:
        streamer.viz_dims = int(body.viz_dims)
    if body.use_repo is not None:
        streamer.use_repo = bool(body.use_repo)
    if body.root_folder is not None:
        streamer.root_folder = str(body.root_folder)
    if body.max_files is not None:
        streamer.max_files = int(body.max_files)
    if body.exclude_dirs is not None:
        streamer.exclude_dirs = [str(x) for x in body.exclude_dirs]
    if body.windows is not None:
        streamer.windows = [int(x) for x in body.windows]
    if body.chunk_workers is not None:
        streamer.chunk_workers = int(body.chunk_workers)
    if body.max_iterations is not None:
        streamer.max_iterations = int(body.max_iterations)
    if body.num_agents is not None:
        streamer.num_agents = int(body.num_agents)
    if body.exploration_bonus is not None:
        streamer.exploration_bonus = float(body.exploration_bonus)
    if body.pheromone_decay is not None:
        streamer.pheromone_decay = float(body.pheromone_decay)
    if body.embed_batch_size is not None:
        streamer.embed_batch_size = int(body.embed_batch_size)
    if body.max_chunks_per_shard is not None:
        streamer.max_chunks_per_shard = int(body.max_chunks_per_shard)
    if body.top_k is not None:
        streamer.top_k = int(body.top_k)
    if getattr(body, 'report_enabled', None) is not None:
        streamer.report_enabled = bool(getattr(body, 'report_enabled'))
    if getattr(body, 'report_every', None) is not None:
        streamer.report_every = int(getattr(body, 'report_every'))
    if getattr(body, 'report_mode', None) is not None:
        streamer.report_mode = str(getattr(body, 'report_mode'))
    if getattr(body, 'judge_mode', None) is not None:
        streamer.judge_mode = str(getattr(body, 'judge_mode'))
    if getattr(body, 'mq_enabled', None) is not None:
        streamer.mq_enabled = bool(getattr(body, 'mq_enabled'))
    if getattr(body, 'mq_count', None) is not None:
        streamer.mq_count = int(getattr(body, 'mq_count'))
    # contextual steering settings
    for k in ["alpha","beta","gamma","delta","epsilon","min_content_chars","import_only_penalty","max_reports","max_report_tokens","judge_enabled"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    # LLM configuration overrides
    for k in ["ollama_model","ollama_host","ollama_system","ollama_num_gpu","ollama_num_thread","ollama_num_batch"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    # LLM provider configuration overrides (OpenAI/Google/Grok)
    for k in ["llm_provider","openai_model","openai_api_key","openai_base_url","openai_temperature"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    for k in ["google_model","google_api_key","google_base_url","google_temperature","grok_model","grok_api_key","grok_base_url","grok_temperature"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    save_settings_to_disk()
    return JSONResponse({"status": "ok"})


@router.post("/stop")
async def http_stop(
    streamer: Any = Depends(get_streamer),
    force: bool = False
) -> JSONResponse:
    # Force stop - skip all cleanup and immediately terminate
    if force:
        _logger.info("[STOP] Force stop requested - skipping cleanup")
        try:
            await streamer._broadcast({
                "type": "log",
                "message": "Force stop - cleanup skipped"
            })
        except Exception:
            pass
        await streamer.stop()
        return JSONResponse({"status": "stopped", "forced": True})

    # Get run_id for both cost aggregation and summary creation
    run_id = str(getattr(streamer, 'run_id', 'run'))

    # Aggregate per-step usage files into a run summary on stop
    try:
        base_dir = os.path.join(SETTINGS_DIR, "runs", run_id)
        total = {"by_provider": {}, "by_model": {}, "totals": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        import glob as _glob, json as _json
        for step_dir in sorted(_glob.glob(os.path.join(base_dir, "step_*"))):
            for name in ("usage.json", "judge_usage.json"):
                p = os.path.join(step_dir, name)
                try:
                    if os.path.exists(p):
                        with open(p, 'r', encoding='utf-8') as f:
                            u = _json.load(f)
                        prov = str(u.get('provider', 'unknown'))
                        model = str(u.get('model', 'unknown'))
                        # favor exact tokens; fall back to *_est
                        pt = int(u.get('prompt_tokens', u.get('prompt_tokens_est', 0)) or 0)
                        ct = int(u.get('completion_tokens', u.get('completion_tokens_est', 0)) or 0)
                        tt = int(u.get('total_tokens', u.get('total_tokens_est', pt + ct)) or (pt + ct))
                        total['totals']['prompt_tokens'] += pt
                        total['totals']['completion_tokens'] += ct
                        total['totals']['total_tokens'] += tt
                        total['by_provider'].setdefault(prov, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                        total['by_provider'][prov]['prompt_tokens'] += pt
                        total['by_provider'][prov]['completion_tokens'] += ct
                        total['by_provider'][prov]['total_tokens'] += tt
                        total['by_model'].setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                        total['by_model'][model]['prompt_tokens'] += pt
                        total['by_model'][model]['completion_tokens'] += ct
                        total['by_model'][model]['total_tokens'] += tt
                except Exception:
                    continue
        # Rough cost estimation (USD per 1K tokens); configurable via env
        PRICES = {
            'openai:gpt-4o': {"prompt": float(os.environ.get('PRICE_OPENAI_GPT4O_PROMPT', '0.005')), "completion": float(os.environ.get('PRICE_OPENAI_GPT4O_COMPLETION', '0.015'))},
            'openai:gpt-4o-mini': {"prompt": float(os.environ.get('PRICE_OPENAI_GPT4OM_PROMPT', '0.0005')), "completion": float(os.environ.get('PRICE_OPENAI_GPT4OM_COMPLETION', '0.0015'))},
        }
        costs = {"by_model": {}, "total_usd": 0.0}
        for model, v in total['by_model'].items():
            key = 'openai:' + model if not model.startswith('openai:') else model
            price = PRICES.get(key)
            if price:
                usd = (v['prompt_tokens'] / 1000.0) * price['prompt'] + (v['completion_tokens'] / 1000.0) * price['completion']
                costs['by_model'][model] = round(usd, 6)
                costs['total_usd'] += usd
        costs['total_usd'] = round(costs['total_usd'], 6)
        out = {"usage": total, "costs": costs}
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "run_costs.json"), 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        try:
            await streamer._broadcast({"type": "log", "message": f"run_costs: total_tokens={total['totals']['total_tokens']} total_usd={costs['total_usd']}"})
        except Exception:
            pass

        # Update manifest with final costs
        try:
            # Set streamer totals from aggregated costs
            streamer._total_tokens = total['totals']['total_tokens']
            streamer._total_cost = costs['total_usd']
            streamer._update_manifest()
            _logger.info(f"[STOP] Updated manifest with tokens={total['totals']['total_tokens']} cost={costs['total_usd']}")
        except Exception as e:
            _logger.warning(f"[STOP] Failed to update manifest with costs: {e}")
    except Exception as e:
        _logger.warning(f"[STOP] Cost aggregation failed: {e}")

    # Create run summary with final results
    try:
        _logger.info(f"[SUMMARY] Starting summary creation for run_id={run_id}")
        if streamer.retr is not None and hasattr(streamer, 'query'):
            _logger.info(f"[SUMMARY] Retriever exists, calling search with query='{streamer.query}', top_k={streamer.top_k}")

            # Get final top-k results from retriever with timeout to prevent hanging
            try:
                final_results = await asyncio.wait_for(
                    asyncio.to_thread(
                        streamer.retr.search,
                        streamer.query,
                        int(streamer.top_k)
                    ),
                    timeout=5.0
                )
                _logger.info(f"[SUMMARY] Search completed, got {len(final_results.get('results', []))} results")
                result_items = final_results.get('results', [])

                # Enrich with doc IDs if needed (using existing helper method)
                result_items = streamer._enrich_results_with_ids(result_items)
                _logger.info(f"[SUMMARY] Enriched {len(result_items)} items with IDs")

                # Create summary in run directory
                _logger.info(f"[SUMMARY] Calling create_run_summary")
                summary_path = create_run_summary(
                    run_id=run_id,
                    query=streamer.query,
                    result_items=result_items
                )
                _logger.info(f"[SUMMARY] Summary created at: {summary_path}")

                try:
                    await streamer._broadcast({
                        "type": "log",
                        "message": f"Summary saved to {summary_path}"
                    })
                except Exception:
                    pass
            except asyncio.TimeoutError:
                _logger.warning("[SUMMARY] Search timed out after 5 seconds - skipping summary creation")
                try:
                    await streamer._broadcast({
                        "type": "log",
                        "message": "Summary creation skipped: search timeout"
                    })
                except Exception:
                    pass
        else:
            _logger.warning(f"[SUMMARY] Cannot create summary: retr is None or query missing")
    except Exception as e:
        _logger.warning(f"[SUMMARY] Failed to create run summary: {e}", exc_info=True)
        try:
            await streamer._broadcast({
                "type": "log",
                "message": f"Summary creation failed: {e}"
            })
        except Exception:
            pass

    await streamer.stop()
    return JSONResponse({"status": "stopped"})


@router.post("/reset")
async def http_reset(streamer: Any = Depends(get_streamer)) -> JSONResponse:
    # Fully stop and clear simulation state so a fresh /start rebuilds corpus and retriever
    try:
        await streamer.stop()
    except Exception:
        pass
    try:
        streamer.retr = None
        streamer.step_i = 0
        streamer.last_metrics = None
        streamer._paused = False
        streamer._saved_state = None
        streamer._avg_rel_history = []
        # Clear LLM/reporting related state and caches
        streamer._reports_sent = 0
        streamer._tokens_used = 0
        streamer._judge_cache = {}
        streamer._llm_vote = {}
        streamer._doc_boost = {}
        streamer._query_pool = []
        streamer._seeds_queue = []
        # Clear background jobs and force corpus rebuild detection
        streamer.jobs = {}
        streamer._corpus_fingerprint = None
        # Clear LangChain agent to force recreation with latest code
        streamer.langchain_agent = None
        # Keep configuration (query, windows, etc.) so the next /start can reuse or override
        await streamer._broadcast({"type": "log", "message": "simulation reset"})
    except Exception:
        pass
    return JSONResponse({"status": "reset"})


@router.post("/pause")
async def http_pause(streamer: Any = Depends(get_streamer)) -> JSONResponse:
    streamer._paused = True
    # save minimal state
    try:
        retr = streamer.retr
        if retr is not None:
            streamer._saved_state = {
                "agents": [
                    {
                        "position": getattr(a, 'position', None).tolist() if getattr(a, 'position', None) is not None else None,
                        "velocity": getattr(a, 'velocity', None).tolist() if getattr(a, 'velocity', None) is not None else None,
                        "age": int(getattr(a, 'age', 0)),
                    }
                    for a in getattr(retr, 'agents', [])
                ]
            }
    except Exception:
        streamer._saved_state = None
    return JSONResponse({"status": "paused"})


@router.post("/resume")
async def http_resume(streamer: Any = Depends(get_streamer)) -> JSONResponse:
    streamer._paused = False
    # restore minimal state
    try:
        retr = streamer.retr
        if retr is not None and streamer._saved_state:
            agents = streamer._saved_state.get("agents", [])
            for i, a in enumerate(getattr(retr, 'agents', [])):
                if i < len(agents):
                    st = agents[i]
                    import numpy as _np
                    if st.get('position') is not None:
                        a.position = _np.array(st['position'], dtype=_np.float32)
                    if st.get('velocity') is not None:
                        a.velocity = _np.array(st['velocity'], dtype=_np.float32)
                    a.age = int(st.get('age', getattr(a, 'age', 0)))
    except Exception:
        pass
    return JSONResponse({"status": "resumed"})


@router.get("/status")
async def http_status(streamer: Any = Depends(get_streamer)) -> JSONResponse:
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"running": False})
    return JSONResponse({
        "running": streamer.running,
        "docs": len(getattr(retr, 'documents', [])),
        "agents": len(getattr(retr, 'agents', [])),
        "metrics": streamer.last_metrics or {},
    })
