from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class RagSettings:
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6337")
    qdrant_api_key: Optional[str] = os.environ.get("QDRANT_API_KEY")
    collection_name: str = os.environ.get("RAG_COLLECTION", "codebase")
    embedding_model: str = os.environ.get("EMBED_MODEL", "google/embeddinggemma-300m")
    llm_model: str = os.environ.get("RAG_LLM_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    llm_device: str = os.environ.get("RAG_LLM_DEVICE", "auto")  # auto|cuda|cpu
    use_ollama: bool = os.environ.get("RAG_USE_OLLAMA", "0") == "1"
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip('/')
    persist_dir: str = os.environ.get("RAG_PERSIST_DIR", "./enterprise_index")


