"""Prompts router - mode prompt management endpoints."""

from __future__ import annotations
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embeddinggemma.realtime.server import SnapshotStreamer

from embeddinggemma.realtime.services import prompts_manager

router = APIRouter(prefix="/prompts", tags=["prompts"])

# Module-level dependency
_get_streamer_dependency: Any = None

def get_streamer() -> Any:
    """Get streamer from dependency."""
    if _get_streamer_dependency is None:
        raise RuntimeError("Streamer dependency not configured")
    return _get_streamer_dependency()


@router.post("/save")
async def http_prompts_save(req: Request) -> JSONResponse:
    """Persist mode prompt overrides to .fungus_cache/prompts_overrides.json.

    Body: { overrides: { mode: instructions_text } }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    overrides = body.get("overrides", {}) or {}
    if not isinstance(overrides, dict):
        return JSONResponse({"status": "error", "message": "overrides must be an object"}, status_code=400)
    try:
        prompts_manager.save_prompt_overrides(overrides)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("")
async def http_prompts_get() -> JSONResponse:
    try:
        overrides = prompts_manager.get_prompt_overrides()
        defaults = prompts_manager.get_all_prompt_defaults()
        modes = prompts_manager.AVAILABLE_MODES
        return JSONResponse({"status":"ok", "overrides": overrides, "defaults": defaults, "modes": modes})
    except Exception as e:
        return JSONResponse({"status":"error", "message": str(e)}, status_code=500)
