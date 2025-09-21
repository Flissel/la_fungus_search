from __future__ import annotations
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import logging

_logger = logging.getLogger("Rag.VectorStore")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    desired_dim: int,
) -> None:
    """Create the collection if missing, or recreate if vector size mismatches."""
    try:
        _logger.info("ensure_collection: name=%s dim=%d", collection_name, desired_dim)
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        if collection_name not in names:
            _create_collection(client, collection_name, desired_dim)
            return
        info = client.get_collection(collection_name=collection_name)
        current_size = getattr(getattr(getattr(info, 'config', None), 'params', None), 'vectors', None)
        current_size = getattr(current_size, 'size', None)
        if isinstance(current_size, int) and current_size != desired_dim:
            _logger.warning("recreate collection due to dim mismatch: current=%d desired=%d", int(current_size), desired_dim)
            client.delete_collection(collection_name)
            _create_collection(client, collection_name, desired_dim)
    except Exception:
        # Bubble up to caller if strict handling is desired
        raise


def _create_collection(client: QdrantClient, name: str, dim: int) -> None:
    from qdrant_client.http.models import HnswConfigDiff, OptimizersConfigDiff, ScalarQuantizationConfig
    _logger.info("create_collection: name=%s dim=%d", name, dim)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=32, ef_construct=128),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000, memmap_threshold=200000),
        quantization_config=ScalarQuantizationConfig(scalar=ScalarQuantizationConfig.Scalar(bits=8), always_ram=False),
    )


