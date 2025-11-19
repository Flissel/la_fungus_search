"""Corpus router - corpus management and indexing endpoints."""

from __future__ import annotations
import os
import json
import hashlib
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embeddinggemma.realtime.server import SnapshotStreamer

from embeddinggemma.ui.corpus import collect_codebase_chunks, list_code_files, chunk_python_file  # type: ignore
from embeddinggemma.mcmp_rag import MCPMRetriever

router = APIRouter(prefix="/corpus", tags=["corpus"])

# Module-level dependency
_get_streamer_dependency: Any = None

def get_streamer() -> Any:
    """Get streamer from dependency."""
    if _get_streamer_dependency is None:
        raise RuntimeError("Streamer dependency not configured")
    return _get_streamer_dependency()


def _parse_chunk_header_line(first_line: str) -> tuple[str | None, int | None, int | None, int | None]:
    try:
        if not first_line.startswith('# file:'):
            return None, None, None, None
        parts = [p.strip() for p in first_line[1:].split('|')]
        file_part = parts[0].split(':', 1)[1].strip() if len(parts) > 0 and ':' in parts[0] else None
        lines_part = parts[1].split(':', 1)[1].strip() if len(parts) > 1 and ':' in parts[1] else None
        win_part = parts[2].split(':', 1)[1].strip() if len(parts) > 2 and ':' in parts[2] else None
        a, b = None, None
        if lines_part and '-' in lines_part:
            try:
                a = int(lines_part.split('-')[0].strip())
                b = int(lines_part.split('-')[1].strip())
            except Exception:
                a, b = None, None
        w = None
        try:
            if win_part:
                w = int(win_part)
        except Exception:
            w = None
        return file_part, a, b, w
    except Exception:
        return None, None, None, None


def _load_embed_client():
    from embeddinggemma.mcmp.embeddings import load_sentence_model  # lazy import
    model_name = os.environ.get('EMBEDDING_MODEL', 'google/embeddinggemma-300m')
    try:
        if not model_name or model_name.strip() == 'google/embeddinggemma-300m':
            if os.environ.get('OPENAI_API_KEY'):
                model_name = 'openai:text-embedding-3-large'
    except Exception:
        pass
    device_mode = os.environ.get('DEVICE_MODE', 'auto')
    return load_sentence_model(model_name, device_mode)


def _encode_texts(embedder, texts: list[str]) -> list[list[float]]:
    try:
        # OpenAI adapter returns list[list[float]] directly
        vecs = embedder.encode(texts)
        if isinstance(vecs, list):
            return [list(map(float, v)) for v in vecs]
        # SentenceTransformers -> numpy array
        import numpy as _np
        if hasattr(vecs, 'tolist'):
            return [list(map(float, v)) for v in _np.asarray(vecs).tolist()]
        return [list(map(float, v)) for v in vecs]
    except Exception as e:
        raise RuntimeError(f"embedding failed: {e}")


@router.get("/list")
async def http_corpus_list(root: str | None = None, page: int = 1, page_size: int = 200, exclude: str | None = None, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    try:
        page = max(1, int(page))
        page_size = max(1, min(2000, int(page_size)))
        ex = [e.strip() for e in (exclude or "").split(',') if e and e.strip()]
        use_root = root or ('src' if streamer.use_repo else streamer.root_folder)
        files = list_code_files(use_root, 0, ex or streamer.exclude_dirs)
        total = len(files)
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        return JSONResponse({"root": use_root, "total": total, "page": page, "page_size": page_size, "files": files[start:end]})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/summary")
async def http_corpus_summary(streamer: Any = Depends(get_streamer)) -> JSONResponse:
    try:
        q_count = None
        try:
            if (streamer.vector_backend or 'memory').lower() == 'qdrant':
                from qdrant_client import QdrantClient  # type: ignore
                client = QdrantClient(url=streamer.qdrant_url, api_key=streamer.qdrant_api_key)
                cnt = client.count(collection_name=streamer.qdrant_collection, exact=True)
                q_count = int(getattr(cnt, 'count', None) or 0)
        except Exception:
            q_count = None
        sim_docs = len(getattr(streamer.retr, 'documents', [])) if streamer.retr else 0
        return JSONResponse({
            "vector_backend": streamer.vector_backend,
            "qdrant_url": streamer.qdrant_url,
            "qdrant_collection": streamer.qdrant_collection,
            "qdrant_points": q_count,
            "simulation_docs": sim_docs,
            "run_id": getattr(streamer, 'run_id', ''),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/add_file")
async def http_corpus_add_file(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Chunk a file, embed chunks, upsert to Qdrant, and reload simulation.

    Body: { path: string }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    path = str(body.get('path', '')).strip()
    if not path:
        return JSONResponse({"status": "error", "message": "path is required"}, status_code=400)
    if (streamer.vector_backend or 'memory').lower() != 'qdrant':
        return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)
    try:
        # Chunk
        chunks = chunk_python_file(path, streamer.windows or [max(1, 1000)])
        if not chunks:
            return JSONResponse({"status": "error", "message": "no chunks produced"}, status_code=400)
        # Prepare payloads and texts
        payloads: list[dict] = []
        texts: list[str] = []
        for ch in chunks:
            first = (ch.splitlines() or [""])[0]
            p, a, b, _w = _parse_chunk_header_line(first)
            body_txt = "\n".join(ch.splitlines()[1:])
            payloads.append({"path": p or os.path.relpath(path), "start": a or 1, "end": b or 1 + len(body_txt.splitlines()), "text": body_txt})
            texts.append(body_txt)
        # Embed
        embedder = _load_embed_client()
        vectors = _encode_texts(embedder, texts)
        # Upsert
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        client = QdrantClient(url=streamer.qdrant_url, api_key=streamer.qdrant_api_key)
        pts = []
        import uuid as _uuid
        for vec, pl in zip(vectors, payloads):
            pts.append(PointStruct(id=str(_uuid.uuid4()), vector=vec, payload=pl))
        client.upsert(collection_name=streamer.qdrant_collection, points=pts)
        try:
            await streamer._broadcast({"type": "log", "message": f"qdrant: upserted chunks={len(pts)} for {path}"})
        except Exception:
            pass
        # Reload simulation
        await streamer.stop()
        await streamer.start()
        return JSONResponse({"status": "ok", "chunks": len(pts)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/update_file")
async def http_corpus_update_file(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Delete existing chunks for a path in Qdrant, then add_file flow, and reload."""
    try:
        body = await req.json()
    except Exception:
        body = {}
    path = str(body.get('path', '')).strip()
    if not path:
        return JSONResponse({"status": "error", "message": "path is required"}, status_code=400)
    if (streamer.vector_backend or 'memory').lower() != 'qdrant':
        return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = QdrantClient(url=streamer.qdrant_url, api_key=streamer.qdrant_api_key)
        flt = Filter(must=[FieldCondition(key="path", match=MatchValue(value=os.path.relpath(path)))])
        client.delete(collection_name=streamer.qdrant_collection, points_selector=flt)
        # reuse add_file logic by faking request
        fake = Request({'type': 'http'})  # type: ignore
        fake._body = json.dumps({"path": path}).encode('utf-8')  # type: ignore
        return await http_corpus_add_file(fake, streamer)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/reindex")
async def http_corpus_reindex(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Rebuild corpus and reinitialize retriever if files changed (or force).

    Body: { force: bool }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    force = bool(body.get("force", False))
    try:
        # Generate new timestamped collection name based on root directory
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        root_dir_name = streamer._get_root_dir_name()
        new_collection_name = f"{root_dir_name}_{timestamp}"
        streamer.qdrant_collection = new_collection_name

        if streamer.use_repo:
            rf = 'src'
        else:
            rf = (streamer.root_folder or os.getcwd())
        files = list_code_files(rf, int(streamer.max_files), streamer.exclude_dirs)
        try:
            await streamer._broadcast({"type": "log", "message": f"reindex: found {len(files)} files"})
            await streamer._broadcast({"type": "collection", "name": new_collection_name})
        except Exception:
            pass
        h = hashlib.sha1()
        for p in sorted(files):
            try:
                h.update(p.encode('utf-8', errors='ignore'))
                h.update(str(os.path.getsize(p)).encode('utf-8'))
            except Exception:
                continue
        new_fp = h.hexdigest()
        changed = (new_fp != streamer._corpus_fingerprint)
        if not changed and not force:
            return JSONResponse({"status": "ok", "changed": False, "message": "No file changes detected"})
        # stop if running
        try:
            await streamer._broadcast({"type": "log", "message": "reindex: starting"})
        except Exception:
            pass
        await streamer.stop()
        # rebuild docs and retriever
        try:
            await streamer._broadcast({"type": "log", "message": "reindex: chunking files..."})
        except Exception:
            pass
        docs, streamer.corpus_file_count = collect_codebase_chunks(rf, streamer.windows, int(streamer.max_files), streamer.exclude_dirs, streamer.chunk_workers)
        try:
            await streamer._broadcast({"type": "log", "message": f"reindex: chunked {len(docs)} docs from {streamer.corpus_file_count} files"})
        except Exception:
            pass
        try:
            await streamer._broadcast({"type": "log", "message": "reindex: embedding chunks..."})
        except Exception:
            pass
        retr = MCPMRetriever(
            embedding_model_name=os.environ.get('EMBEDDING_MODEL', 'google/embeddinggemma-300m'),
            num_agents=streamer.num_agents,
            max_iterations=streamer.max_iterations,
            exploration_bonus=streamer.exploration_bonus,
            pheromone_decay=streamer.pheromone_decay,
            embed_batch_size=streamer.embed_batch_size,
            device_mode=os.environ.get('DEVICE_MODE', 'auto'),
        )
        retr.add_documents(docs)
        try:
            await streamer._broadcast({"type": "log", "message": "reindex: initializing simulation..."})
        except Exception:
            pass
        retr.initialize_simulation(streamer.query)
        streamer.retr = retr
        streamer._corpus_fingerprint = new_fp
        try:
            await streamer._broadcast({"type": "log", "message": f"reindex: complete docs={len(docs)}"})
        except Exception:
            pass
        return JSONResponse({"status": "ok", "changed": True, "docs": len(docs)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/index_repo")
async def http_corpus_index_repo(req: Request, streamer: Any = Depends(get_streamer)) -> JSONResponse:
    """Chunk and embed all repo files to Qdrant, then restart simulation.

    Body: { root?: string, exclude_dirs?: string[] }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    if (streamer.vector_backend or 'memory').lower() != 'qdrant':
        return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)
    root = str(body.get('root') or ('src' if streamer.use_repo else (streamer.root_folder or os.getcwd())))
    exclude = body.get('exclude_dirs') or streamer.exclude_dirs
    try:
        files = list_code_files(root, int(streamer.max_files), exclude)
        if not files:
            return JSONResponse({"status": "error", "message": "no files found to index"}, status_code=400)
        # Embedder and Qdrant client
        embedder = _load_embed_client()
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        client = QdrantClient(url=streamer.qdrant_url, api_key=streamer.qdrant_api_key)
        total_pts = 0
        for i, path in enumerate(files):
            try:
                chunks = chunk_python_file(path, streamer.windows or [max(1, 1000)])
                if not chunks:
                    continue
                payloads: list[dict] = []
                texts: list[str] = []
                for ch in chunks:
                    first = (ch.splitlines() or [""])[0]
                    p, a, b, _w = _parse_chunk_header_line(first)
                    body_txt = "\n".join(ch.splitlines()[1:])
                    payloads.append({"path": p or os.path.relpath(path), "start": a or 1, "end": b or 1 + len(body_txt.splitlines()), "text": body_txt})
                    texts.append(body_txt)
                vectors = _encode_texts(embedder, texts)
                import uuid as _uuid
                pts = [PointStruct(id=str(_uuid.uuid4()), vector=v, payload=pl) for v, pl in zip(vectors, payloads)]
                if pts:
                    client.upsert(collection_name=streamer.qdrant_collection, points=pts)
                    total_pts += len(pts)
                if i % 10 == 0:
                    try:
                        await streamer._broadcast({"type": "log", "message": f"qdrant: indexed files={i+1}/{len(files)} points={total_pts}"})
                    except Exception:
                        pass
            except Exception as e:
                try:
                    await streamer._broadcast({"type": "log", "message": f"index_repo: skipped {path}: {e}"})
                except Exception:
                    pass
                continue
        # Reload simulation
        await streamer.stop()
        await streamer.start()
        return JSONResponse({"status": "ok", "files": len(files), "points": total_pts})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
