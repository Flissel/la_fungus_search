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


def generate_with_openai(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str = "https://api.openai.com",
    system: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 500,
    save_prompt_path: Optional[str] = None,
) -> str:
    try:
        if save_prompt_path:
            try:
                with open(save_prompt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            except Exception:
                pass
        url = base_url.rstrip('/') + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
        }
        _logger.info("generate_with_openai: model=%s base=%s", model, base_url)
        try:
            _logger.info("generate_with_openai: prompt_len=%d", len(prompt))
        except Exception:
            pass
        r = requests.post(url, json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get('choices') or [{}])[0].get('message', {}).get('content', '')
        try:
            _logger.info("generate_with_openai: status=%s response_len=%d", r.status_code, len(text or ""))
        except Exception:
            pass
        return text or ''
    except Exception as e:
        _logger.error("generate_with_openai error: %s", e)
        return f"[LLM error] {e}"
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
        try:
            _logger.info("generate_with_ollama: prompt_len=%d", len(prompt))
        except Exception:
            pass
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        _logger.info("generate_with_ollama: status=%s", r.status_code)
        resp = r.json().get('response', '')
        try:
            _logger.info("generate_with_ollama: response_len=%d", len(resp or ""))
        except Exception:
            pass
        return resp
        
    except Exception as e:
        _logger.error("generate_with_ollama error: %s", e)
        return f"[LLM error] {e}"


def generate_text(
    provider: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    # Ollama
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    ollama_options: Optional[Dict[str, Any]] = None,
    # OpenAI
    openai_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_temperature: Optional[float] = None,
    # Common
    timeout: int = 500,
    save_prompt_path: Optional[str] = None,
) -> str:
    p = (provider or 'ollama').lower()
    if p == 'openai':
        return generate_with_openai(
            prompt,
            model=openai_model or 'gpt-4o-mini',
            api_key=openai_api_key or '',
            base_url=(openai_base_url or 'https://api.openai.com'),
            system=system,
            temperature=float(openai_temperature if openai_temperature is not None else 0.0),
            timeout=timeout,
            save_prompt_path=save_prompt_path,
        )
    # default: ollama
    return generate_with_ollama(
        prompt,
        model=ollama_model or 'qwen2.5-coder:7b',
        host=(ollama_host or 'http://127.0.0.1:11434'),
        timeout=timeout,
        system=system,
        options=ollama_options,
        save_prompt_path=save_prompt_path,
    )


