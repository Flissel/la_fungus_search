from __future__ import annotations

# Public re-exports to present a clean LLM interface
from .dispatcher import generate_text, generate_with_ollama, generate_with_openai  # noqa: F401
from .prompts import build_report_prompt, build_judge_prompt, get_report_instructions  # noqa: F401


