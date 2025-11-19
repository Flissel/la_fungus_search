"""Misc router - root, introspect, run, jobs, and reports endpoints."""

from __future__ import annotations
import os
import json
import asyncio
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embeddinggemma.realtime.server import SnapshotStreamer

from embeddinggemma.ui.corpus import collect_codebase_chunks  # type: ignore
from embeddinggemma.ui.reports import merge_reports_to_summary  # type: ignore
from embeddinggemma.mcmp_rag import MCPMRetriever

router = APIRouter(tags=["misc"])

# Module-level dependency
_get_streamer_dependency: Any = None

def get_streamer() -> Any:
    """Get streamer from dependency."""
    if _get_streamer_dependency is None:
        raise RuntimeError("Streamer dependency not configured")
    return _get_streamer_dependency()


# Module-level references set by server.py
_static_dir: str | None = None
_app: Any = None


@router.get("/")
async def index() -> HTMLResponse:
    if _static_dir is None:
        raise RuntimeError("static_dir not configured")
    html_path = os.path.join(_static_dir, "index.html")
    if not os.path.exists(html_path):
        # Minimal landing page
        return HTMLResponse("""
<!doctype html>
<html><head><meta charset='utf-8'><title>MCMP Realtime</title></head>
<body>
<h3>MCMP Realtime</h3>
<p>Open <a href='/static/index.html'>client</a> to view the network.</p>
</body></html>
""")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@router.get("/introspect/api")
async def http_introspect_api() -> JSONResponse:
    """Introspect server to list HTTP routes and WebSocket event types."""
    try:
        import inspect as _inspect
        import re as _re
        routes: list[dict] = []
        ws_events: set[str] = set()

        if _app is None:
            raise RuntimeError("app not configured")

        # Extract routes from FastAPI app
        for r in getattr(_app, 'routes', []):
            try:
                path = getattr(r, 'path', None)
                methods = sorted(list(getattr(r, 'methods', []) or []))
                name = getattr(r, 'name', None)
                if not path or path.startswith('/openapi'):
                    continue
                if path in ['/', '/docs', '/redoc']:
                    continue
                routes.append({"path": path, "methods": methods, "name": name})
            except Exception:
                continue
        # Parse this module's source to find _broadcast({"type": "..."}) usages
        try:
            mod = _inspect.getmodule(http_introspect_api)  # type: ignore
            src = _inspect.getsource(mod) if mod else ""
        except Exception:
            src = ""
        if src:
            for m in _re.finditer(r"_broadcast\(\{[^\}]*\"type\"\s*:\s*\"([^\"]+)\"", src):
                try:
                    ev = m.group(1)
                    if ev:
                        ws_events.add(ev)
                except Exception:
                    continue
        return JSONResponse({"routes": routes, "ws_events": sorted(list(ws_events))})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/run/new")
async def http_run_new(streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """DEPRECATED: run_id is now set automatically in start() method based on collection name.
    This endpoint is kept for backward compatibility but should not be used."""
    try:
        # Don't overwrite run_id if it's already set (it's set in start() to collection name)
        if not getattr(streamer, 'run_id', None):
            import time as _t
            streamer.run_id = f"run_{int(_t.time())}"
        return JSONResponse({"status": "ok", "run_id": streamer.run_id})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/jobs/start")
async def http_jobs_start(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    body = await req.json()
    q = str(body.get("query", streamer.query))
    job_id = str(len(streamer.jobs) + 1)
    streamer.jobs[job_id] = {"status": "running", "progress": 0}

    async def _job():
        try:
            # Build corpus
            if streamer.use_repo:
                docs, streamer.corpus_file_count = collect_codebase_chunks('src', streamer.windows, int(streamer.max_files), streamer.exclude_dirs, streamer.chunk_workers)
            else:
                docs, streamer.corpus_file_count = collect_codebase_chunks(streamer.root_folder or os.getcwd(), streamer.windows, int(streamer.max_files), streamer.exclude_dirs, streamer.chunk_workers)
            total_chunks = len(docs)
            shard_size = int(streamer.max_chunks_per_shard)
            if shard_size <= 0 or shard_size >= total_chunks:
                shard_ranges = [(0, total_chunks)]
            else:
                shard_ranges = [(i, min(i + shard_size, total_chunks)) for i in range(0, total_chunks, shard_size)]
            num_shards = max(1, len(shard_ranges))
            agg = []
            for idx, (s, e) in enumerate(shard_ranges):
                retr = MCPMRetriever(num_agents=streamer.num_agents, max_iterations=streamer.max_iterations, exploration_bonus=streamer.exploration_bonus, pheromone_decay=streamer.pheromone_decay, embed_batch_size=streamer.embed_batch_size)
                retr.add_documents(docs[s:e])
                retr.initialize_simulation(q)
                for _ in range(streamer.max_iterations):
                    retr.step(1)
                res = retr.search(q, top_k=5)
                agg.extend(res.get('results', []))
                pct = int(100 * (idx + 1) / num_shards)
                await streamer._broadcast({"type": "job_progress", "job_id": job_id, "percent": pct, "message": f"Processed shard {idx+1}/{num_shards}"})
            streamer.jobs[job_id] = {"status": "done", "progress": 100, "results": agg}
        except Exception as e:
            streamer.jobs[job_id] = {"status": "error", "message": str(e)}

    asyncio.create_task(_job())
    return JSONResponse({"status": "ok", "job_id": job_id})


@router.get("/jobs/status")
async def http_jobs_status(job_id: str, streamer: Any = Depends(get_streamer)):
    j = streamer.jobs.get(job_id)
    if not j:
        return JSONResponse({"status": "error", "message": "unknown job"}, status_code=404)
    return JSONResponse({"status": "ok", "job": j})


@router.post("/reports/merge")
async def http_reports_merge(req: Request) -> JSONResponse:
    """Merge multiple per-step or background reports into a single summary.json.

    Body (optional): {
      paths?: string[]  // explicit report.json paths; if omitted, auto-discover recent ones
      out_path?: string // output path for the summary (default: .fungus_cache/reports/summary.json)
      max_reports?: number // cap auto-discovery
    }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    paths = body.get('paths') if isinstance(body.get('paths'), list) else None
    out_path = body.get('out_path') if isinstance(body.get('out_path'), str) else None
    res = merge_reports_to_summary(paths=paths, out_path=out_path)
    # Best-effort: include the merged summary content for UI download
    summary_obj = None
    try:
        with open(res.get('summary_path') or '', 'r', encoding='utf-8') as f:
            summary_obj = json.load(f)
    except Exception:
        summary_obj = None
    return JSONResponse({"status": "ok", "data": res, "summary": summary_obj})
