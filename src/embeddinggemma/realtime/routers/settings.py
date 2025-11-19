"""Settings router - settings persistence and retrieval endpoints."""

from __future__ import annotations
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embeddinggemma.realtime.server import SnapshotStreamer

from embeddinggemma.realtime.services import settings_manager

router = APIRouter(tags=["settings"])

# Module-level dependency
_get_streamer_dependency: Any = None

def get_streamer() -> Any:
    """Get streamer from dependency."""
    if _get_streamer_dependency is None:
        raise RuntimeError("Streamer dependency not configured")
    return _get_streamer_dependency()


# Module-level reference to save function
_save_settings_to_disk: Any = None

def save_settings_to_disk() -> None:
    """Save settings to disk - delegates to server module function."""
    if _save_settings_to_disk is None:
        raise RuntimeError("save_settings_to_disk not configured")
    _save_settings_to_disk()


@router.get("/settings")
async def http_settings_get(streamer: Any = Depends(get_streamer)) -> JSONResponse:
    sd = settings_manager.get_settings_dict(streamer)
    return JSONResponse({"settings": sd, "usage": settings_manager.get_settings_usage_lines(sd)})


@router.post("/settings")
async def http_settings_post(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    try:
        body = await req.json()
        settings_manager.apply_settings_to_streamer(streamer, body)
        save_settings_to_disk()
        sd = settings_manager.get_settings_dict(streamer)
        return JSONResponse({"status": "ok", "settings": sd, "usage": settings_manager.get_settings_usage_lines(sd)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)
