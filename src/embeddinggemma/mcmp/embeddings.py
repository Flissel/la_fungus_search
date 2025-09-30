from __future__ import annotations
from typing import List, Optional, Any
import sys

from embeddinggemma.rag.embeddings import resolve_device  # reuse
from embeddinggemma.rag.embeddings import log_torch_environment as _log_torch_env
from sentence_transformers import SentenceTransformer
import logging
import os
import requests


_logger = logging.getLogger("MCMP.Embeddings")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


class _OpenAIEmbeddingClient:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key or not str(self.api_key).strip():
            _logger.error("OpenAI embeddings: missing API key")
            raise RuntimeError("Missing OPENAI_API_KEY for OpenAI embeddings")
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        # Truncate overly long inputs to avoid 400 errors due to max token limits
        try:
            max_chars = int(os.environ.get('OPENAI_EMBED_MAX_CHARS', '6000'))
        except Exception:
            max_chars = 6000
        try:
            batch_size = int(os.environ.get('OPENAI_EMBED_BATCH', '64'))
        except Exception:
            batch_size = 64
        # Coerce inputs to non-empty strings and trim
        proc = []
        for t in texts:
            s = "" if t is None else str(t)
            s = s[:max_chars]
            s = s if s.strip() else " "
            proc.append(s)
        _logger.info("openai.embed: model=%s base=%s n=%d max_chars=%d batch=%d", self.model, self.base_url, len(proc or []), max_chars, batch_size)
        out: List[List[float]] = []

        def _post_embed(inputs: List[str]) -> List[List[float]]:
            # If single item, send as string per API examples
            payload_input: Any = inputs[0] if len(inputs) == 1 else inputs
            body = {"model": self.model, "input": payload_input, "encoding_format": "float"}
            r = requests.post(url, json=body, headers=headers, timeout=90)
            if r.status_code >= 400:
                try:
                    _logger.warning("openai.embed: status=%s body=%s", r.status_code, r.text[:500])
                except Exception:
                    pass
            r.raise_for_status()
            data = r.json()
            return [it.get('embedding', []) for it in (data.get('data') or [])]

        i = 0
        N = len(proc)
        while i < N:
            chunk = proc[i:i+batch_size]
            try:
                out.extend(_post_embed(chunk))
            except requests.HTTPError as he:
                # If still 400, try stronger truncation then halve batch size and retry once
                if getattr(he.response, 'status_code', None) == 400:
                    try:
                        _logger.warning("openai.embed: 400 on batch, applying stronger truncation and smaller batch")
                        chunk2 = [ s[:4000] for s in chunk ]
                        out.extend(_post_embed(chunk2))
                    except Exception:
                        if batch_size > 8:
                            batch_size = max(8, batch_size // 2)
                            continue  # reattempt with smaller batch
                        raise
                else:
                    raise
            i += batch_size
        return out


def load_sentence_model(model_name: str, device_preference: str = "auto") -> Any:
    _logger.info("load_sentence_model: model=%s device_pref=%s", model_name, device_preference)
    _log_torch_env()
    # OpenAI adapter: use prefix openai:<model>
    if isinstance(model_name, str) and model_name.lower().startswith("openai:"):
        raw = model_name.split(":", 1)[1] or "text-embedding-3-large"
        alias = raw.lower().strip()
        # Aliases for convenience
        if alias in ("large",):
            model = "text-embedding-3-large"
        elif alias in ("small",):
            model = "text-embedding-3-small"
        elif alias in ("ada", "text-embedding-ada-002"):
            # Legacy ADA alias maps to a 1536-dim model; prefer current small
            model = "text-embedding-3-small"
        else:
            model = raw
        api_key = os.environ.get('OPENAI_API_KEY', '')
        base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com')
        _logger.info("load_sentence_model: using OpenAI embeddings model=%s base=%s", model, base_url)
        return _OpenAIEmbeddingClient(model=model, api_key=api_key, base_url=base_url)
    # Default: SentenceTransformers (local/HF)
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


