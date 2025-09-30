from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
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
    """Backward-compatible wrapper for OpenAI-compatible chat completions.

    This function delegates to the normalized OpenAI-compatible adapter below.
    """
    return generate_with_openai_compatible(
        prompt=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        system=system,
        temperature=temperature,
        timeout=timeout,
        save_prompt_path=save_prompt_path,
    )


def generate_with_openai_compatible(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 500,
    save_prompt_path: Optional[str] = None,
    save_usage_path: Optional[str] = None,
    provider_label: str = "openai",
) -> str:
    try:
        # Fast-fail if API key is missing to avoid 401 spam
        if not api_key or not str(api_key).strip():
            _logger.error("generate_with_openai_compatible missing api_key for base=%s model=%s", base_url, model)
            return (
                "[LLM error] Missing API key for provider. Set OPENAI_API_KEY/GOOGLE_API_KEY/GROK_API_KEY or switch provider to 'ollama'."
            )
        if save_prompt_path:
            try:
                with open(save_prompt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            except Exception:
                pass
        # Single path for any OpenAI-compatible endpoint
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
        _logger.info("generate_with_openai_compatible: model=%s base=%s", model, base_url)
        try:
            _logger.info("generate_with_openai_compatible: prompt_len=%d", len(prompt))
        except Exception:
            pass
        r = requests.post(url, json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get('choices') or [{}])[0].get('message', {}).get('content', '')
        try:
            _logger.info("generate_with_openai_compatible: status=%s response_len=%d", r.status_code, len(text or ""))
        except Exception:
            pass
        # Persist usage details when available
        try:
            if save_usage_path:
                usage = data.get('usage') or {}
                out = {
                    "provider": str(provider_label or "openai"),
                    "model": str(model),
                    "base_url": str(base_url),
                    "prompt_tokens": int(usage.get('prompt_tokens') or 0),
                    "completion_tokens": int(usage.get('completion_tokens') or 0),
                    "total_tokens": int(usage.get('total_tokens') or (int(usage.get('prompt_tokens') or 0) + int(usage.get('completion_tokens') or 0))),
                }
                import json as _json, os as _os
                _os.makedirs(_os.path.dirname(save_usage_path), exist_ok=True)
                with open(save_usage_path, 'w', encoding='utf-8') as _f_u:
                    _json.dump(out, _f_u, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return text or ''
    except Exception as e:
        _logger.error("generate_with_openai_compatible error: %s", e)
        return f"[LLM error] {e}"
def generate_with_ollama(
    prompt: str,
    model: str,
    host: str,
    timeout: int = 500,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    save_prompt_path: Optional[str] = None,
    save_usage_path: Optional[str] = None,
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
        # Approximate token usage for Ollama
        try:
            if save_usage_path:
                approx_prompt_toks = int(max(1, len(prompt) // 4))
                approx_completion_toks = int(max(1, len(resp or "") // 4))
                out = {
                    "provider": "ollama",
                    "model": str(model),
                    "host": str(host),
                    "prompt_tokens_est": approx_prompt_toks,
                    "completion_tokens_est": approx_completion_toks,
                    "total_tokens_est": approx_prompt_toks + approx_completion_toks,
                    "estimated": True,
                }
                import json as _json, os as _os
                _os.makedirs(_os.path.dirname(save_usage_path), exist_ok=True)
                with open(save_usage_path, 'w', encoding='utf-8') as _f_u:
                    _json.dump(out, _f_u, ensure_ascii=False, indent=2)
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
    # Google (OpenAI-compatible endpoint if used)
    google_model: Optional[str] = None,
    google_api_key: Optional[str] = None,
    google_base_url: Optional[str] = None,
    google_temperature: Optional[float] = None,
    # Grok (OpenAI-compatible)
    grok_model: Optional[str] = None,
    grok_api_key: Optional[str] = None,
    grok_base_url: Optional[str] = None,
    grok_temperature: Optional[float] = None,
    # Common
    timeout: int = 500,
    save_prompt_path: Optional[str] = None,
    save_usage_path: Optional[str] = None,
) -> str:
    p = (provider or 'ollama').lower()
    if p in ('openai', 'google', 'grok'):
        # Normalize all OpenAI-compatible providers through one adapter
        if p == 'openai':
            model = openai_model or 'gpt-4o-mini'
            api_key = openai_api_key or os.environ.get('OPENAI_API_KEY', '')
            base = (openai_base_url or 'https://api.openai.com')
            temp = float(openai_temperature if openai_temperature is not None else 0.0)
        elif p == 'google':
            model = google_model or 'gemini-1.5-pro'
            api_key = google_api_key or os.environ.get('GOOGLE_API_KEY', '')
            base = (google_base_url or 'https://generativelanguage.googleapis.com')
            temp = float(google_temperature if google_temperature is not None else 0.0)
        else:  # grok
            model = grok_model or 'grok-2-latest'
            api_key = grok_api_key or os.environ.get('GROK_API_KEY', '')
            base = (grok_base_url or 'https://api.x.ai')
            temp = float(grok_temperature if grok_temperature is not None else 0.0)
        return generate_with_openai_compatible(
            prompt=prompt,
            model=model,
            api_key=api_key,
            base_url=base,
            system=system,
            temperature=temp,
            timeout=timeout,
            save_prompt_path=save_prompt_path,
            save_usage_path=save_usage_path,
            provider_label=p,
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
        save_usage_path=save_usage_path,
    )


