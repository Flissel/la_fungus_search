"""Collections router - Qdrant collection management endpoints."""

from __future__ import annotations
import logging
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embeddinggemma.realtime.server import SnapshotStreamer

from embeddinggemma.realtime.services.qdrant_service import QdrantService

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])

# This will be set by server.py after import
_get_streamer_dependency: Any = None

def get_streamer() -> Any:
    """Get streamer from dependency."""
    if _get_streamer_dependency is None:
        raise RuntimeError("Streamer dependency not configured")
    return _get_streamer_dependency()


def create_qdrant_service_from_streamer(streamer: SnapshotStreamer) -> QdrantService:
    """Helper to create Qdrant service from streamer configuration."""
    return QdrantService(
        url=streamer.qdrant_url,
        api_key=streamer.qdrant_api_key,
        collection=streamer.qdrant_collection
    )


@router.get("/list")
async def list_collections(streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """List all available Qdrant collections with metadata."""
    try:
        backend = (streamer.vector_backend or 'memory').lower().strip()
        _logger.info(f"Collections endpoint: vector_backend='{backend}'")

        # Handle memory backend - return mock collection
        if backend == 'memory':
            try:
                point_count = len(streamer.retr.documents) if streamer.retr and hasattr(streamer.retr, 'documents') else 0
            except Exception as e:
                _logger.warning(f"Could not get chunk count from retriever: {e}")
                point_count = 0

            return JSONResponse({
                "status": "ok",
                "collections": [{
                    "name": "codebase",
                    "point_count": point_count,
                    "dimension": None,
                    "is_active": True
                }],
                "active_collection": "codebase"
            })

        # Qdrant backend
        if backend != 'qdrant':
            return JSONResponse({
                "status": "error",
                "message": f"Unsupported vector_backend: '{backend}'. Must be 'memory' or 'qdrant'."
            }, status_code=400)

        try:
            qdrant = create_qdrant_service_from_streamer(streamer)
            collections_list = []

            for coll_info in qdrant.list_collections():
                try:
                    # Get detailed info for each collection
                    detailed_info = qdrant.get_collection_info(coll_info["name"])
                    if detailed_info:
                        collections_list.append({
                            "name": detailed_info["name"],
                            "point_count": detailed_info.get("points_count", 0),
                            "dimension": detailed_info.get("config", {}).get("vector_size"),
                            "is_active": detailed_info["name"] == streamer.qdrant_collection
                        })
                    else:
                        # Fallback if detailed info fails
                        collections_list.append({
                            "name": coll_info["name"],
                            "point_count": coll_info.get("points_count", 0),
                            "dimension": None,
                            "is_active": coll_info["name"] == streamer.qdrant_collection
                        })
                except Exception as e:
                    _logger.warning(f"Could not get info for collection {coll_info['name']}: {e}")
                    collections_list.append({
                        "name": coll_info["name"],
                        "point_count": 0,
                        "dimension": None,
                        "is_active": coll_info["name"] == streamer.qdrant_collection,
                        "error": str(e)
                    })

            return JSONResponse({
                "status": "ok",
                "collections": collections_list,
                "active_collection": streamer.qdrant_collection
            })
        except Exception as e:
            _logger.error(f"Qdrant connection error: {e}")
            return JSONResponse({
                "status": "error",
                "error_type": "qdrant_connection_error",
                "message": f"Failed to connect to Qdrant at {streamer.qdrant_url}",
                "details": str(e)
            }, status_code=503)

    except Exception as e:
        _logger.error(f"Collections list error: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "error_type": "internal_error",
            "message": str(e)
        }, status_code=500)


@router.post("/switch")
async def switch_collection(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Switch the active collection and reload the simulation."""
    try:
        body = await req.json()
    except Exception:
        body = {}

    collection_name = str(body.get('collection', '')).strip()
    if not collection_name:
        return JSONResponse({"status": "error", "message": "collection name is required"}, status_code=400)

    # Handle memory backend - accept switch but only "codebase" is valid
    if (streamer.vector_backend or 'memory').lower() == 'memory':
        if collection_name != 'codebase':
            return JSONResponse({
                "status": "error",
                "message": f"Memory backend only supports 'codebase' collection"
            }, status_code=400)
        return JSONResponse({
            "status": "ok",
            "message": f"Already using collection 'codebase'",
            "active_collection": "codebase"
        })

    if (streamer.vector_backend or 'memory').lower() != 'qdrant':
        return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)

    try:
        qdrant = create_qdrant_service_from_streamer(streamer)

        # Check if collection exists
        if not qdrant.collection_exists(collection_name):
            return JSONResponse({
                "status": "error",
                "message": f"Collection '{collection_name}' does not exist"
            }, status_code=404)

        # Stop current simulation if running
        if streamer.running:
            await streamer.stop()

        # Switch collection
        streamer.qdrant_collection = collection_name

        # Reload simulation with new collection
        await streamer.start()

        return JSONResponse({
            "status": "ok",
            "message": f"Switched to collection '{collection_name}'",
            "active_collection": collection_name
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/{collection_name}/info")
async def get_collection_info(collection_name: str, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Get detailed information about a specific collection."""
    try:
        if (streamer.vector_backend or 'memory').lower() != 'qdrant':
            return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)

        qdrant = create_qdrant_service_from_streamer(streamer)

        # Check if collection exists
        if not qdrant.collection_exists(collection_name):
            return JSONResponse({
                "status": "error",
                "message": f"Collection '{collection_name}' does not exist"
            }, status_code=404)

        # Get collection details
        info = qdrant.get_collection_info(collection_name)
        if not info:
            return JSONResponse({
                "status": "error",
                "message": f"Could not retrieve info for collection '{collection_name}'"
            }, status_code=500)

        # Try to get a sample point to extract metadata
        sample_metadata = {}
        try:
            points = qdrant.scroll_all_points(collection_name, batch_size=1)
            if points and len(points) > 0:
                sample_payload = points[0].get('payload', {})
                if sample_payload:
                    sample_metadata = {
                        "has_path": "path" in sample_payload,
                        "has_text": "text" in sample_payload,
                        "has_start": "start" in sample_payload,
                        "has_end": "end" in sample_payload,
                        "sample_keys": list(sample_payload.keys())
                    }
        except Exception:
            pass

        return JSONResponse({
            "status": "ok",
            "collection": collection_name,
            "point_count": info.get("points_count", 0),
            "dimension": info.get("config", {}).get("vector_size"),
            "distance": info.get("config", {}).get("distance"),
            "is_active": collection_name == streamer.qdrant_collection,
            "metadata": sample_metadata
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.delete("/{collection_name}")
async def delete_collection(collection_name: str, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Delete a collection (with safety check)."""
    try:
        if (streamer.vector_backend or 'memory').lower() != 'qdrant':
            return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)

        # Safety: don't delete active collection
        if collection_name == streamer.qdrant_collection:
            return JSONResponse({
                "status": "error",
                "message": "Cannot delete the active collection. Switch to another collection first."
            }, status_code=400)

        qdrant = create_qdrant_service_from_streamer(streamer)

        # Check if collection exists
        if not qdrant.collection_exists(collection_name):
            return JSONResponse({
                "status": "error",
                "message": f"Collection '{collection_name}' does not exist"
            }, status_code=404)

        # Delete the collection
        success = qdrant.delete_collection(collection_name)
        if not success:
            return JSONResponse({
                "status": "error",
                "message": f"Failed to delete collection '{collection_name}'"
            }, status_code=500)

        return JSONResponse({
            "status": "ok",
            "message": f"Collection '{collection_name}' deleted successfully"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
