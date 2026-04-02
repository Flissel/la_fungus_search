from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os

# --- Zentrale LLM-Config via vibemind_shared ---
try:
    from vibemind_shared import get_model as _shared_get_model
    _HAS_SHARED_CONFIG = True
except ImportError:
    _HAS_SHARED_CONFIG = False


def _resolve_model(role: str, env_var: str, fallback: str) -> str:
    """Modell aus zentraler Config, env-var oder Fallback."""
    env_val = os.environ.get(env_var)
    if env_val:
        return env_val
    if _HAS_SHARED_CONFIG:
        try:
            return _shared_get_model(role)
        except Exception:
            pass
    return fallback


@dataclass
class OllamaConfig:
    model: str = field(default_factory=lambda: _resolve_model('default', 'OLLAMA_MODEL', 'qwen2.5-coder:7b'))
    host: str = field(default_factory=lambda: os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/'))
    system: Optional[str] = field(default_factory=lambda: os.environ.get('OLLAMA_SYSTEM'))
    num_gpu: Optional[int] = field(default_factory=lambda: int(os.environ.get('OLLAMA_NUM_GPU')) if os.environ.get('OLLAMA_NUM_GPU') else None)
    num_thread: Optional[int] = field(default_factory=lambda: int(os.environ.get('OLLAMA_NUM_THREAD')) if os.environ.get('OLLAMA_NUM_THREAD') else None)
    num_batch: Optional[int] = field(default_factory=lambda: int(os.environ.get('OLLAMA_NUM_BATCH')) if os.environ.get('OLLAMA_NUM_BATCH') else None)


@dataclass
class OpenAIConfig:
    model: str = field(default_factory=lambda: _resolve_model('openai', 'OPENAI_MODEL', 'gpt-4o-mini'))
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get('OPENAI_API_KEY'))
    base_url: str = field(default_factory=lambda: os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com'))
    temperature: float = field(default_factory=lambda: float(os.environ.get('OPENAI_TEMPERATURE', '0.0')))


@dataclass
class GoogleConfig:
    model: str = field(default_factory=lambda: _resolve_model('google', 'GOOGLE_MODEL', 'gemini-1.5-pro'))
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get('GOOGLE_API_KEY'))
    base_url: str = field(default_factory=lambda: os.environ.get('GOOGLE_BASE_URL', 'https://generativelanguage.googleapis.com'))
    temperature: float = field(default_factory=lambda: float(os.environ.get('GOOGLE_TEMPERATURE', '0.0')))


@dataclass
class GrokConfig:
    model: str = field(default_factory=lambda: _resolve_model('grok', 'GROK_MODEL', 'grok-2-latest'))
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get('GROK_API_KEY'))
    base_url: str = field(default_factory=lambda: os.environ.get('GROK_BASE_URL', 'https://api.x.ai'))
    temperature: float = field(default_factory=lambda: float(os.environ.get('GROK_TEMPERATURE', '0.0')))


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.environ.get('LLM_PROVIDER', 'ollama'))
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    grok: GrokConfig = field(default_factory=GrokConfig)


def load_config() -> LLMConfig:
    return LLMConfig()


