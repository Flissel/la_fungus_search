from __future__ import annotations

import json
import os
from typing import Dict

import embeddinggemma.prompts as _base_prompts

# Capture original to avoid recursion after monkey-patching
try:
    _ORIG_GET_REPORT_INSTRUCTIONS = _base_prompts.get_report_instructions  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _ORIG_GET_REPORT_INSTRUCTIONS = None  # type: ignore[assignment]

_OVERRIDE_PATH = os.path.join(os.getcwd(), ".fungus_cache", "prompts_overrides.json")
_OVERRIDES: Dict[str, str] = {}


def _load_overrides() -> None:
    global _OVERRIDES
    try:
        if os.path.exists(_OVERRIDE_PATH):
            with open(_OVERRIDE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _OVERRIDES = {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
                else:
                    _OVERRIDES = {}
        else:
            _OVERRIDES = {}
    except Exception:
        _OVERRIDES = {}


def get_report_instructions(mode: str) -> str:
    m = (mode or "deep").lower()
    try:
        if not _OVERRIDES:
            _load_overrides()
        if m in _OVERRIDES and _OVERRIDES[m].strip():
            return _OVERRIDES[m]
    except Exception:
        pass
    # Fallback to original base implementation (captured before patching)
    try:
        if callable(_ORIG_GET_REPORT_INSTRUCTIONS):  # type: ignore[truthy-function]
            return _ORIG_GET_REPORT_INSTRUCTIONS(m)  # type: ignore[misc]
    except Exception:
        pass
    # Last resort: use base module's default instructions
    try:
        return _base_prompts._default_instructions(m)  # type: ignore[attr-defined]
    except Exception:
        return ""


# Monkey-patch base module so its builders use our override-aware instructions
try:
    _base_prompts.get_report_instructions = get_report_instructions  # type: ignore[assignment]
except Exception:
    pass


def build_report_prompt(mode: str, query: str, top_k: int, docs: list[dict]) -> str:
    # Delegate to base; it will call our patched get_report_instructions
    return _base_prompts.build_report_prompt(mode, query, top_k, docs)


def build_judge_prompt(mode: str, query: str, results: list[dict]) -> str:
    # Delegate to base; it will call our patched get_report_instructions
    return _base_prompts.build_judge_prompt(mode, query, results)


__all__ = [
    "build_report_prompt",
    "build_judge_prompt",
    "get_report_instructions",
]


