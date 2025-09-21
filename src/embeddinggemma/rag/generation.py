from __future__ import annotations
from typing import List, Dict, Any, Optional
import requests
import logging

_logger = logging.getLogger("Rag.Generation")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)
def generate_with_ollama(prompt: str, model: str, host: str, timeout: int = 500) -> str:
    try:
        url = f"{host.rstrip('/')}" + "/api/generate"
        _logger.info("generate_with_ollama: model=%s host=%s", model, host)
        r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout)
        r.raise_for_status()
        return r.json().get('response', '')
    except Exception as e:
        _logger.error("generate_with_ollama error: %s", e)
        return f"[LLM error] {e}"


