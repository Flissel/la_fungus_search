Environment configuration (development)

Backend (FastAPI):
- EMBEDDINGGEMMA_BACKEND_PORT: port for backend (default 8011)
- EMBEDDING_MODEL: embeddings model id. If not set and OPENAI_API_KEY is present, defaults to openai:text-embedding-3-large; otherwise google/embeddinggemma-300m
- DEVICE_MODE: embeddings device selection (auto|cpu|cuda)
- OLLAMA_MODEL: LLM model id (e.g., qwen2.5-coder:7b)
- OLLAMA_HOST: Ollama HTTP endpoint (default http://127.0.0.1:11434)
- OLLAMA_SYSTEM: optional system prompt string
- OLLAMA_NUM_GPU: number of GPUs to use (int)
- OLLAMA_NUM_THREAD: CPU threads for generation (int)
- OLLAMA_NUM_BATCH: batch size for generation (int)

- LLM_PROVIDER: which provider to use for judge/report/answer (ollama|openai|google|grok)

- OPENAI_MODEL: OpenAI model id (e.g., gpt-4o-mini)
- OPENAI_API_KEY: OpenAI API key
- OPENAI_BASE_URL: OpenAI-compatible base URL (default https://api.openai.com)
- OPENAI_TEMPERATURE: sampling temperature (0.0-2.0)

- GOOGLE_MODEL: Google model id (e.g., gemini-1.5-pro)
- GOOGLE_API_KEY: Google API key
- GOOGLE_BASE_URL: Google Generative Language API base URL (default https://generativelanguage.googleapis.com)
- GOOGLE_TEMPERATURE: sampling temperature (0.0-2.0)

- GROK_MODEL: Grok model id (e.g., grok-2-latest)
- GROK_API_KEY: Grok API key
- GROK_BASE_URL: Grok API base URL (default https://api.x.ai)
- GROK_TEMPERATURE: sampling temperature (0.0-2.0)

Frontend (Vite):
- VITE_BACKEND_PORT: backend port (for proxy) if not default

Prompts:
- Place mode-specific prompt modules under embeddinggemma/modeprompts/<mode>.py
  with function instructions() -> str to override report instructions.
  Example file: embeddinggemma/modeprompts/deep.py

Artifacts:
- Per-run artifacts are saved under .fungus_cache/runs/<run_id>/step_<i>/
  - report.json
  - answer_prompt.txt

.env support:
- Create a `.env` in repo root; it will be auto-loaded by the backend.
- Example (.env):
  - OLLAMA_MODEL=qwen2.5-coder:7b
  - OLLAMA_HOST=http://127.0.0.1:11434
  - OLLAMA_SYSTEM=You are a precise code analysis assistant. Output strict JSON as requested.
  - OLLAMA_NUM_GPU=1
  - OLLAMA_NUM_THREAD=12
  - OLLAMA_NUM_BATCH=128
  - EMBEDDING_MODEL=openai:text-embedding-3-large
  - DEVICE_MODE=cuda
  # Qdrant (vector backend)
  - VECTOR_BACKEND=qdrant
  - QDRANT_URL=http://localhost:6339
  - QDRANT_COLLECTION=codebase

  - EMBEDDINGGEMMA_BACKEND_PORT=8011
  - LLM_PROVIDER=openai
  - OPENAI_MODEL=gpt-4o-mini
  - OPENAI_API_KEY=sk-...
  - OPENAI_BASE_URL=https://api.openai.com
  - OPENAI_TEMPERATURE=0.0
  - GOOGLE_MODEL=gemini-1.5-pro
  - GOOGLE_API_KEY=...
  - GOOGLE_BASE_URL=https://generativelanguage.googleapis.com
  - GOOGLE_TEMPERATURE=0.0
  - GROK_MODEL=grok-2-latest
  - GROK_API_KEY=...
  - GROK_BASE_URL=https://api.x.ai
  - GROK_TEMPERATURE=0.0


