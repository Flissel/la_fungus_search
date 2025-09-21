from __future__ import annotations
from typing import Optional, List
import os
import sys
import logging

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from sentence_transformers import SentenceTransformer

_logger = logging.getLogger("Rag.Embeddings")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


def resolve_device(preferred: str = "auto") -> str:
    mode = (preferred or "auto").lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        # Only honor explicit CUDA if it is actually available
        try:
            if torch and torch.cuda.is_available():
                return "cuda"
            _logger.warning("CUDA requested but not available; falling back to CPU")
            _log_torch_environment(level=logging.WARNING)
            return "cpu"
        except Exception:
            return "cpu"
    try:
        dev = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        _logger.debug("resolve_device: preferred=%s resolved=%s", preferred, dev)
        return dev
    except Exception:
        return "cpu"


def _cuda_build_tag() -> Optional[str]:
    try:
        if torch is None:
            return None
        cuda_ver = getattr(torch.version, "cuda", None)
        if not cuda_ver:
            return None
        # Map versions like "11.8" -> "cu118"
        cleaned = cuda_ver.replace(".", "")
        return f"cu{cleaned}"
    except Exception:
        return None


def _log_torch_environment(level: int = logging.INFO) -> None:
    try:
        torch_version = getattr(torch, "__version__", "not-installed") if torch else "not-installed"
        cuda_build = _cuda_build_tag() or "cpu-build"
        cuda_available = bool(torch and torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if (torch and cuda_available) else 0
        device_name = torch.cuda.get_device_name(0) if (torch and cuda_available and device_count > 0) else "-"
        cudnn_ver = getattr(torch.backends.cudnn, "version", lambda: None)()
        msg = (
            f"torch={torch_version} build={cuda_build} cuda_available={cuda_available} "
            f"devices={device_count} name={device_name} cudnn={cudnn_ver} venv={sys.prefix}"
        )
        _logger.log(level, msg)
        # Guidance if CUDA not available but a CUDA build is installed
        if (torch and not cuda_available and getattr(torch.version, "cuda", None)):
            _logger.warning(
                "Torch is a CUDA build (%s) but CUDA is not available. Check NVIDIA drivers/CUDA runtime and CUDA_VISIBLE_DEVICES.",
                getattr(torch.version, "cuda", None),
            )
    except Exception as e:
        _logger.debug("torch environment logging failed: %s", e)


def log_torch_environment() -> None:
    """Public helper to log torch/CUDA environment once at INFO level."""
    _log_torch_environment(level=logging.INFO)


class EmbeddingBackend:
    def __init__(self, model_name: str, device_preference: str = "auto"):
        self.model_name = model_name
        self.device_preference = device_preference
        self.model: Optional[SentenceTransformer] = None

    def load(self) -> SentenceTransformer:
        if self.model is not None:
            return self.model
        # Probe environment for helpful diagnostics
        _log_torch_environment(level=logging.INFO)
        device = resolve_device(self.device_preference)
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            _logger.info("Embedding model loaded: %s on %s", self.model_name, device)
        except Exception:
            # fallback to CPU if CUDA build mismatch
            if device != "cpu":
                _logger.warning("Embedding model fallback to CPU: %s", self.model_name)
                self.model = SentenceTransformer(self.model_name, device="cpu")
            else:
                raise
        return self.model

    def encode(self, texts: List[str]) -> List[float]:
        m = self.load()
        _logger.debug("encode: texts=%d", len(texts))
        return m.encode(texts)


