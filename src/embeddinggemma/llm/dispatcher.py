from __future__ import annotations
from typing import Optional, Dict, Any

from embeddinggemma.rag.generation import (
    generate_text as _generate_text,
    generate_with_ollama as _gen_ollama,
    generate_with_openai as _gen_openai,
)


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
    # Google
    google_model: Optional[str] = None,
    google_api_key: Optional[str] = None,
    google_base_url: Optional[str] = None,
    google_temperature: Optional[float] = None,
    # Grok
    grok_model: Optional[str] = None,
    grok_api_key: Optional[str] = None,
    grok_base_url: Optional[str] = None,
    grok_temperature: Optional[float] = None,
    timeout: int = 500,
    save_prompt_path: Optional[str] = None,
) -> str:
    # Delegate to existing implementation to avoid code duplication
    return _generate_text(
        provider=provider,
        prompt=prompt,
        system=system,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        ollama_options=ollama_options,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_temperature=openai_temperature,
        google_model=google_model,
        google_api_key=google_api_key,
        google_base_url=google_base_url,
        google_temperature=google_temperature,
        grok_model=grok_model,
        grok_api_key=grok_api_key,
        grok_base_url=grok_base_url,
        grok_temperature=grok_temperature,
        timeout=timeout,
        save_prompt_path=save_prompt_path,
    )


# Direct provider helpers if needed by callers
def generate_with_ollama(prompt: str, model: str, host: str, *, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, timeout: int = 500) -> str:
    return _gen_ollama(prompt=prompt, model=model, host=host, timeout=timeout, system=system, options=options)


def generate_with_openai(prompt: str, model: str, api_key: str, *, base_url: str = "https://api.openai.com", system: Optional[str] = None, temperature: float = 0.0, timeout: int = 500) -> str:
    return _gen_openai(prompt=prompt, model=model, api_key=api_key, base_url=base_url, system=system, temperature=temperature, timeout=timeout)



