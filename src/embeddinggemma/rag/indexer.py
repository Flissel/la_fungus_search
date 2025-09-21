from __future__ import annotations
from typing import List

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
import logging

_logger = logging.getLogger("Rag.Indexer")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


def build_index(
    documents,
    vector_store: QdrantVectorStore,
    embed_model,
    transformations,
):
    _logger.info("build_index: docs=%d", len(documents or []))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=transformations,
        show_progress=True,
    )


def load_index(persist_dir: str, vector_store: QdrantVectorStore, embed_model):
    from llama_index.core import load_index_from_storage
    _logger.info("load_index: dir=%s", persist_dir)
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
    try:
        return load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
    except Exception:
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context, embed_model=embed_model)


