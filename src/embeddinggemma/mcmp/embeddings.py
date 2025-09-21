from __future__ import annotations
from typing import List, Optional
import sys

from embeddinggemma.rag.embeddings import resolve_device  # reuse
from embeddinggemma.rag.embeddings import log_torch_environment as _log_torch_env
from sentence_transformers import SentenceTransformer
import logging


_logger = logging.getLogger("MCMP.Embeddings")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


def load_sentence_model(model_name: str, device_preference: str = "auto") -> SentenceTransformer:
    _logger.info("load_sentence_model: model=%s device_pref=%s", model_name, device_preference)
    _log_torch_env()
    device = resolve_device(device_preference)
    try:
        model = SentenceTransformer(model_name, device=device)
        _logger.info("load_sentence_model: loaded device=%s", device)
        return model
    except Exception:
        if device != "cpu":
            _logger.warning("load_sentence_model: falling back to CPU")
            return SentenceTransformer(model_name, device="cpu")
        raise


