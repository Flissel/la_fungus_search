"""Search & query router - search, answer, document detail endpoints."""

from __future__ import annotations
import logging
import os
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embeddinggemma.realtime.server import SnapshotStreamer

from embeddinggemma.llm import generate_text
from embeddinggemma.realtime.services.settings_manager import SETTINGS_DIR

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

# This will be set by server.py after import
_get_streamer_dependency: Any = None

def get_streamer() -> Any:
    """Get streamer from dependency."""
    if _get_streamer_dependency is None:
        raise RuntimeError("Streamer dependency not configured")
    return _get_streamer_dependency()


@router.post("/search")
async def http_search(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    body = await req.json()
    query = str(body.get("query", ""))
    top_k = int(body.get("top_k", 5))
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    try:
        res = retr.search(query, top_k=top_k)
        return JSONResponse({"status": "ok", "results": res.get('results', [])})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/answer")
async def http_answer(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    body = await req.json()
    query = str(body.get("query", ""))
    top_k = int(body.get("top_k", 5))
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    res = retr.search(query, top_k=top_k)
    ctx = "\n\n".join([(it.get('content') or '')[:800] for it in res.get('results', [])])
    prompt = f"Kontext:\n{ctx}\n\nAufgabe:\n{query}\n\nAntwort:"
    llm_opts = {}
    try:
        if os.environ.get('OLLAMA_NUM_GPU'):
            llm_opts['num_gpu'] = int(os.environ.get('OLLAMA_NUM_GPU'))
        if os.environ.get('OLLAMA_NUM_THREAD'):
            llm_opts['num_thread'] = int(os.environ.get('OLLAMA_NUM_THREAD'))
        if os.environ.get('OLLAMA_NUM_BATCH'):
            llm_opts['num_batch'] = int(os.environ.get('OLLAMA_NUM_BATCH'))
    except Exception:
        llm_opts = {}
    answer_prompt_path = os.path.join(SETTINGS_DIR, "reports/answer_prompt.txt")
    try:
        # also write under run folder
        run_dir = os.path.join(SETTINGS_DIR, "runs", str(getattr(streamer, 'run_id', 'run')), f"step_{int(getattr(streamer, 'step_i', 0))}")
        os.makedirs(run_dir, exist_ok=True)
        answer_prompt_path = os.path.join(run_dir, "answer_prompt.txt")
    except Exception:
        pass
    text = generate_text(
        provider=(streamer.llm_provider or 'ollama'),
        prompt=prompt,
        system=streamer.ollama_system,
        ollama_model=streamer.ollama_model,
        ollama_host=streamer.ollama_host,
        ollama_options=(llm_opts or None),
        openai_model=streamer.openai_model,
        openai_api_key=(streamer.openai_api_key or ''),
        openai_base_url=(streamer.openai_base_url or 'https://api.openai.com'),
        openai_temperature=float(getattr(streamer, 'openai_temperature', 0.0)),
        google_model=streamer.google_model,
        google_api_key=(streamer.google_api_key or ''),
        google_base_url=(streamer.google_base_url or 'https://generativelanguage.googleapis.com'),
        google_temperature=float(getattr(streamer, 'google_temperature', 0.0)),
        grok_model=streamer.grok_model,
        grok_api_key=(streamer.grok_api_key or ''),
        grok_base_url=(streamer.grok_base_url or 'https://api.x.ai'),
        grok_temperature=float(getattr(streamer, 'grok_temperature', 0.0)),
        save_prompt_path=answer_prompt_path,
    )
    return JSONResponse({"status": "ok", "answer": text, "results": res.get('results', [])})


@router.get("/doc/{doc_id}")
async def http_doc(doc_id: int, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    try:
        d = next((x for x in getattr(retr, 'documents', []) if int(getattr(x, 'id', -1)) == int(doc_id)), None)
        if d is None:
            return JSONResponse({"status": "error", "message": "doc not found"}, status_code=404)
        emb = getattr(d, 'embedding', None)
        emb_list = emb.tolist() if emb is not None else []
        meta = getattr(d, 'metadata', {}) or {}
        return JSONResponse({
            "status": "ok",
            "doc": {
                "id": int(getattr(d, 'id', -1)),
                "content": getattr(d, 'content', ''),
                "embedding": emb_list,
                "relevance_score": float(getattr(d, 'relevance_score', 0.0)),
                "visit_count": int(getattr(d, 'visit_count', 0)),
                "metadata": meta,
            }
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
