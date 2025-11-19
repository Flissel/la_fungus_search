"""Agents router - agent management endpoints."""

from __future__ import annotations
import logging
import json
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embeddinggemma.realtime.server import SnapshotStreamer

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# This will be set by server.py after import
_get_streamer_dependency: Any = None

def get_streamer() -> Any:
    """Get streamer from dependency."""
    if _get_streamer_dependency is None:
        raise RuntimeError("Streamer dependency not configured")
    return _get_streamer_dependency()


@router.post("/add")
async def http_agents_add(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Append N new agents with random positions/velocities in embedding space.
    Recommended to call while paused to avoid visual glitches.
    Body: { n: int }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    n = int(body.get("n", 0))
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    if not getattr(retr, "documents", []):
        return JSONResponse({"status": "error", "message": "no documents indexed"}, status_code=400)
    if n <= 0:
        return JSONResponse({"status": "error", "message": "n must be > 0"}, status_code=400)
    try:
        dim = int(retr.documents[0].embedding.shape[0])  # type: ignore
        import numpy as _np
        current_len = len(getattr(retr, 'agents', []))
        max_id = max([getattr(a, 'id', -1) for a in getattr(retr, 'agents', [])], default=-1)
        for i in range(n):
            pos = _np.random.normal(0, 1.0, size=(dim,)).astype(_np.float32)
            norm = float(_np.linalg.norm(pos)) or 1.0
            pos = pos / norm
            vel = _np.random.normal(0, 0.05, size=(dim,)).astype(_np.float32)
            agent = retr.Agent(  # type: ignore[attr-defined]
                id=int(max_id + 1 + i),
                position=pos,
                velocity=vel,
                exploration_factor=float(_np.random.uniform(0.05, max(0.05, float(getattr(retr, 'exploration_bonus', 0.1))))),
            )
            retr.agents.append(agent)
        retr.num_agents = int(len(retr.agents))
        return JSONResponse({"status": "ok", "added": int(n), "agents": int(len(retr.agents))})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/resize")
async def http_agents_resize(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Resize agent population to target count. Adds random agents or trims list.
    Body: { count: int }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    target = int(body.get("count", -1))
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    if target < 0:
        return JSONResponse({"status": "error", "message": "count must be >= 0"}, status_code=400)
    cur = len(getattr(retr, 'agents', []))
    if target == cur:
        return JSONResponse({"status": "ok", "agents": cur})
    if target < cur:
        try:
            retr.agents = list(retr.agents)[:target]
            retr.num_agents = int(target)
            return JSONResponse({"status": "ok", "agents": int(len(retr.agents))})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    # Need to add agents
    to_add = int(target - cur)
    # Reuse add logic
    fake_req = Request({'type': 'http'})  # type: ignore
    fake_req._body = json.dumps({"n": to_add}).encode("utf-8")  # type: ignore
    return await http_agents_add(fake_req, streamer=streamer)
