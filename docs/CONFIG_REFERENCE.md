# Configuration Reference

This reference documents all settings available in the frontend and backend. Each setting includes the technical key (backend name), type, default value (if known), and purpose. Values can be set via API endpoints (`/start`, `/config`), environment variables, or through the UI. Backend source: `src/embeddinggemma/realtime/server.py` (SettingsModel, settings_dict, apply_settings).

## General Settings
- **query** (string, default: "Explain the architecture.")
  - Purpose: Search/steering query for simulation, ranking, and reports
- **top_k** (int, default: 5)
  - Purpose: Number of top results per step/response
- **mode/report_mode** (string, default: "deep")
  - Values: deep, structure, exploratory, summary, repair, steering. Prompt style for reports (LLM)
- **judge_mode** (string, default: "steering")
  - Purpose: Prompt style for LLM judge (context steering)
- **mq_enabled** (bool, default: false)
  - Purpose: Enable multi-query mode (LLM-assisted additional queries)
- **mq_count** (int, default: 5)
  - Purpose: Number of additional queries generated in multi-query mode

## Visualization Settings
- **viz_dims** (int, default: 3)
  - Values: 2 or 3; 2D/3D projection (PCA) in snapshot
- **min_trail_strength** (float, default: 0.05)
  - Purpose: Threshold for displaying pheromone edges
- **max_edges** (int, default: 600)
  - Purpose: Edge limit in snapshot visualization
- **redraw_every** (int, default: 2)
  - Purpose: Step interval for snapshot/metrics broadcast via WebSocket

## Corpus & Chunking Settings
- **use_repo** (bool, default: true)
  - Purpose: Use `src` as root directory; otherwise use `root_folder`
- **root_folder** (string, default: current working directory)
  - Purpose: Project root when `use_repo=false`
- **max_files** (int, default: 1000)
  - Purpose: Maximum number of files to index (0 = no limit)
- **exclude_dirs** (string[], default: [".venv","node_modules",".git","external"])
  - Purpose: Directories to exclude from indexing
- **windows** (int[], default: [1000,2000,4000])
  - Purpose: Line window sizes for chunking (e.g., 1000, 2000, 4000)
- **chunk_workers** (int, default: CPU count based)
  - Purpose: Number of threads for chunking operations
- **embed_batch_size** (int, default: 64)
  - Purpose: Batch size for embedding chunks (SentenceTransformers)
- **max_chunks_per_shard** (int, default: 2000)
  - Purpose: Shard size for batch operations (jobs)

## Simulation Settings
- **num_agents** (int, default: 200)
  - Purpose: Number of agents in the simulation
- **max_iterations** (int, default: 60)
  - Purpose: Maximum number of simulation steps
- **exploration_bonus** (float, default: 0.1)
  - Purpose: Weight for exploratory agent movement
- **pheromone_decay** (float, default: 0.95)
  - Purpose: Decay factor for pheromone trails per step

## Reporting & Judge Settings (LLM-powered Context Steering)
- **report_enabled** (bool, default: false)
  - Purpose: Enable periodic step reports
- **report_every** (int, default: 5)
  - Purpose: Interval (in steps) for reports
- **report_mode** (string, default: "deep")
  - Purpose: Report prompt mode (see General Settings)
- **judge_enabled** (bool, default: true)
  - Purpose: Enable LLM judge (context steering)
- **max_reports** (int, default: 20)
  - Purpose: Budget limit for report/judge steps
- **max_report_tokens** (int, default: 20000)
  - Purpose: Token/character budget for reports/judge operations

### Blended Scoring / Pruning Settings
- **alpha** (float, default: 0.7)
  - Purpose: Weight for cosine similarity
- **beta** (float, default: 0.1)
  - Purpose: Weight for visit normalization (visit_norm)
- **gamma** (float, default: 0.1)
  - Purpose: Weight for trail degree
- **delta** (float, default: 0.1)
  - Purpose: Weight for LLM vote (-1/0/1)
- **epsilon** (float, default: 0.0)
  - Purpose: Length/priority weight (BM25-like)
- **min_content_chars** (int, default: 80)
  - Purpose: Minimum character length for chunk evaluation/pruning
- **import_only_penalty** (float, default: 0.4)
  - Purpose: Penalty weight for import-only chunks

## LLM Provider Settings (Central defaults in src/embeddinggemma/llm/config.py)
- **llm_provider** (string, default: "ollama")
  - Values: ollama, openai, google, grok

### Ollama Provider
- **ollama_model** (string, default: "qwen2.5-coder:7b")
- **ollama_host** (string, default: "http://127.0.0.1:11434")
- **ollama_system** (string|null, default: null)
- **ollama_num_gpu** (int|null, default: from env/None)
- **ollama_num_thread** (int|null, default: from env/None)
- **ollama_num_batch** (int|null, default: from env/None)

### OpenAI Provider
- **openai_model** (string, default: "gpt-4o-mini")
- **openai_api_key** (string|null, from OPENAI_API_KEY env var)
- **openai_base_url** (string, default: "https://api.openai.com")
- **openai_temperature** (float, default: 0.0)

### Google Provider
- **google_model** (string, default: "gemini-1.5-pro")
- **google_api_key** (string|null, from GOOGLE_API_KEY env var)
- **google_base_url** (string, default: "https://generativelanguage.googleapis.com")
- **google_temperature** (float, default: 0.0)

### Grok Provider
- **grok_model** (string, default: "grok-2-latest")
- **grok_api_key** (string|null, from GROK_API_KEY env var)
- **grok_base_url** (string, default: "https://api.x.ai")
- **grok_temperature** (float, default: 0.0)

## API Actions (UI)
These actions don't change persistent state but are workflow-relevant:
- **Apply** (POST /config) - Update simulation parameters without restarting
- **Start** (POST /start) - Initialize corpus and begin simulation
- **Stop** (POST /stop) - Halt current simulation
- **Reset** (POST /reset) - Reset simulation completely (configuration persists)
- **Pause/Resume** (POST /pause, /resume) - Pause/resume simulation execution
- **Add Agents** (POST /agents/add) - Add agents during paused simulation
- **Resize Agents** (POST /agents/resize) - Change total number of agents
- **Corpus Listing** (GET /corpus/list) - Browse indexed files and directories
- **Shard Run** (POST /jobs/start) - Run background sharded analysis job
- **Search** (POST /search) - Search for relevant code snippets
- **Answer** (POST /answer) - Generate LLM answer from search results
- **Document Details** (GET /doc/{id}) - Fetch full document content and embeddings

## Setting Usage Mapping
Key usage locations for each setting are documented in `settings_usage_lines` in the backend:

- **viz_dims** → Visualization projection and UI (PCA 2D/3D rendering)
- **num_agents**, **pheromone_decay**, **exploration_bonus** → Agent simulation (`mcmp/simulation.py`)
- **report_***, **judge_***, **alpha**..**epsilon**, **min_content_chars**, **import_only_penalty** → LLM steering and blended scoring (`realtime/server.py`)
- **Provider-specific fields** → LLM dispatcher (`llm/dispatcher.py`) and calls in `realtime/server.py`
- **windows**, **chunk_workers** → Corpus chunking (`ui/corpus.py`)
- **embed_batch_size** → Embedding model processing (`mcmp/embeddings.py`)

## Configuration Management
- **Central defaults**: Modify `src/embeddinggemma/llm/config.py` to change default values
- **Environment override**: `.env` file values override Python defaults
- **Runtime updates**: Frontend sends only changed/non-default values via API
- **Persistence**: Settings automatically saved to `.fungus_cache/settings.json`



