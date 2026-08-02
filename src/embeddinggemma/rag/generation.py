from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from vibemind_shared import get_client_sync, get_model


_ROLE = "fungus_summary"
def _write_prompt(path: Optional[str], prompt: str) -> None:
    if not path:
        return
    prompt_path = Path(path)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("OpenFang returned no completion choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content:
        raise RuntimeError("OpenFang returned an empty completion")
    return content


def generate_text(
    provider: Optional[str] = None,
    prompt: str = "",
    *,
    system: Optional[str] = None,
    save_prompt_path: Optional[str] = None,
    save_usage_path: Optional[str] = None,
    **_legacy: Any,
) -> str:
    """Generate a summary through the configured OpenFang role.

    ``provider`` and the remaining legacy keyword arguments are accepted only
    for source compatibility. Provider, model, credentials, retries, and cost
    authority are owned by ``vibemind_shared`` and OpenFang.
    """
    del provider, save_usage_path, _legacy
    _write_prompt(save_prompt_path, prompt)

    messages: list[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = get_client_sync(_ROLE)
    response = client.chat.completions.create(
        model=get_model(_ROLE),
        messages=messages,
    )
    return _response_text(response)


def generate_with_openai(
    prompt: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 500,
    save_prompt_path: Optional[str] = None,
) -> str:
    """Compatibility wrapper; supplied provider settings are intentionally ignored."""
    del model, api_key, base_url, temperature, timeout
    return generate_text(prompt=prompt, system=system, save_prompt_path=save_prompt_path)


def generate_with_openai_compatible(
    prompt: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    system: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 500,
    save_prompt_path: Optional[str] = None,
    save_usage_path: Optional[str] = None,
    provider_label: Optional[str] = None,
) -> str:
    """Compatibility wrapper; supplied provider settings are intentionally ignored."""
    del model, api_key, base_url, temperature, timeout, save_usage_path, provider_label
    return generate_text(prompt=prompt, system=system, save_prompt_path=save_prompt_path)


def generate_with_ollama(
    prompt: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
    timeout: int = 500,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    save_prompt_path: Optional[str] = None,
    save_usage_path: Optional[str] = None,
) -> str:
    """Compatibility wrapper; supplied provider settings are intentionally ignored."""
    del model, host, timeout, options, save_usage_path
    return generate_text(prompt=prompt, system=system, save_prompt_path=save_prompt_path)
