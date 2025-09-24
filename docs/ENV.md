Environment configuration (development)

Backend (FastAPI):
- EMBEDDINGGEMMA_BACKEND_PORT: port for backend (default 8011)
- EMBEDDING_MODEL: embeddings model id (default google/embeddinggemma-300m)
- DEVICE_MODE: embeddings device selection (auto|cpu|cuda)
- OLLAMA_MODEL: LLM model id (e.g., qwen2.5-coder:7b)
- OLLAMA_HOST: Ollama HTTP endpoint (default http://127.0.0.1:11434)
- OLLAMA_SYSTEM: optional system prompt string
- OLLAMA_NUM_GPU: number of GPUs to use (int)
- OLLAMA_NUM_THREAD: CPU threads for generation (int)
- OLLAMA_NUM_BATCH: batch size for generation (int)

Frontend (Vite):
- VITE_BACKEND_PORT: backend port (for proxy) if not default

Prompts:
- Place mode-specific prompt modules under embeddinggemma/modeprompts/<mode>.py
  with function instructions() -> str to override report instructions.
  Example file: embeddinggemma/modeprompts/deep.py

Artifacts:
- All generated prompts and reports are saved under .fungus_cache/reports/
  - prompt_step_<i>.txt
  - judge_prompt_step_<i>.txt
  - step_<i>.json
  - answer_prompt.txt


