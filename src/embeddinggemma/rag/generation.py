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
def generate_with_ollama(
    prompt: str,
    model: str,
    host: str,
    timeout: int = 500,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    save_prompt_path: Optional[str] = None,
) -> str:
    try:
        url = f"{host.rstrip('/')}" + "/api/generate"
        if save_prompt_path:
            try:
                with open(save_prompt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            except Exception:
                pass
        _logger.info("generate_with_ollama: model=%s host=%s", model, host)
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        _logger.info("generate_with_ollama: status=%s", r.status_code)
        return r.json().get('response', '')
        
    except Exception as e:
        _logger.error("generate_with_ollama error: %s", e)
        return f"[LLM error] {e}"


