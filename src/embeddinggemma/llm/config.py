from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class OllamaConfig:
    model: str = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
    host: str = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
    system: Optional[str] = os.environ.get('OLLAMA_SYSTEM')
    num_gpu: Optional[int] = int(os.environ.get('OLLAMA_NUM_GPU')) if os.environ.get('OLLAMA_NUM_GPU') else None
    num_thread: Optional[int] = int(os.environ.get('OLLAMA_NUM_THREAD')) if os.environ.get('OLLAMA_NUM_THREAD') else None
    num_batch: Optional[int] = int(os.environ.get('OLLAMA_NUM_BATCH')) if os.environ.get('OLLAMA_NUM_BATCH') else None


@dataclass
class OpenAIConfig:
    model: str = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    api_key: Optional[str] = os.environ.get('OPENAI_API_KEY')
    base_url: str = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com')
    temperature: float = float(os.environ.get('OPENAI_TEMPERATURE', '0.0'))


@dataclass
class GoogleConfig:
    model: str = os.environ.get('GOOGLE_MODEL', 'gemini-1.5-pro')
    api_key: Optional[str] = os.environ.get('GOOGLE_API_KEY')
    base_url: str = os.environ.get('GOOGLE_BASE_URL', 'https://generativelanguage.googleapis.com')
    temperature: float = float(os.environ.get('GOOGLE_TEMPERATURE', '0.0'))


@dataclass
class GrokConfig:
    model: str = os.environ.get('GROK_MODEL', 'grok-2-latest')
    api_key: Optional[str] = os.environ.get('GROK_API_KEY')
    base_url: str = os.environ.get('GROK_BASE_URL', 'https://api.x.ai')
    temperature: float = float(os.environ.get('GROK_TEMPERATURE', '0.0'))


@dataclass
class LLMConfig:
    provider: str = os.environ.get('LLM_PROVIDER', 'ollama')
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    grok: GrokConfig = field(default_factory=GrokConfig)


def load_config() -> LLMConfig:
    # Simple loader for now; reads env on import/init
    return LLMConfig()


