# MCMP Realtime – Configuration Reference

This reference documents all settings available in the frontend and backend. Each setting lists the technical key (backend name), type, default (if known), and purpose. Values can be set via API (/start, /config), through .env, or via the UI. Backend source: src/embeddinggemma/realtime/server.py (SettingsModel, settings_dict, apply_settings).

## General
- query (string, default: "Classify the code into modules.")
  - Purpose: Search/control query for simulation, ranking, and reports.
- top_k (int, default: 10)
  - Purpose: Number of top results per step/response.
- mode (string, default: "deep")
  - Also known as: Task Mode
  - Purpose: Overall objective for the analysis - defines what kind of insights you want from the code
  - Task-Specific Values (New - January 2025):
    - **architecture**: Map system components, layers, dependencies, data flow, design patterns
    - **bugs**: Detect security vulnerabilities, null checks, race conditions, resource leaks with severity ratings (CRITICAL/HIGH/MEDIUM/LOW)
    - **quality**: Assess code complexity, SOLID principles, code smells, maintainability, test coverage
    - **documentation**: Extract comprehensive API documentation with parameters, return values, usage examples
    - **features**: Trace feature implementation end-to-end through all system layers
  - Generic Values (Existing):
    - **deep**: General deep analysis with comprehensive context
    - **structure**: Focus on structural organization and module relationships
    - **exploratory**: Broad exploration for discovering patterns and connections
    - **summary**: Concise summaries of code functionality
    - **repair**: Identify issues and suggest fixes
  - Note: Previously called "report_mode" in backend, but UI now uses "Task Mode" for clarity
- judge_mode (string, default: "steering")
  - Purpose: Steering strategy for the MCMP simulation algorithm - controls how the judge guides agent exploration
  - Values:
    - **steering**: Balanced exploration with adaptive context control (default)
    - **focused**: Deep-first exploration - follows call chains and helper functions to build complete mental model of one area before moving on
    - **exploratory**: Breadth-first exploration - discovers new areas and connections
  - Note: Judge analyzes current findings, interprets them in context of the Task Mode, and produces follow-up queries to fulfill the main task
- mq_enabled (bool, default: false)
  - Purpose: Enable multi-query mode (LLM-assisted additional queries).
- mq_count (int, default: 5)
  - Purpose: Number of generated additional queries in multi-query mode.

## Visualization
- viz_dims (int, default: 3)
  - Values: 2 or 3; 2D/3D projection (PCA) in snapshot.
- min_trail_strength (float, default: 0.05)
  - Purpose: Threshold for displaying pheromone edges.
- max_edges (int, default: 1500)
  - Purpose: Edge limit in snapshot.
- redraw_every (int, default: 2)
  - Purpose: Step interval for snapshot/metrics broadcast via WebSocket.

## Corpus & Chunking
- use_repo (bool, default: true)
  - Purpose: Use src as root; otherwise root_folder.
- root_folder (string, default: working directory)
  - Purpose: Project root when use_repo=false.
- max_files (int, default: 500)
  - Purpose: Upper limit of loaded files.
- exclude_dirs (string[], default: [".venv","node_modules",".git","external"])
  - Purpose: Directories to exclude.
- windows (int[], default: [2000, 4000, 8000] - auto-set if not provided)
  - Purpose: Line windows for chunking - defines the size of code chunks analyzed by LLM
  - Note: Increased from [1000, 2000, 4000] in January 2025 to provide more context
  - Larger windows = more meaningful context for LLM analysis = better insights
  - No truncation applied - LLM receives full chunk content
- chunk_workers (int, default: CPU-based)
  - Purpose: Number of threads for chunking.
- embed_batch_size (int, default: 128)
  - Purpose: Batch size when embedding chunks (SentenceTransformers).
- max_chunks_per_shard (int, default: 2000)
  - Purpose: Shard size for batch runs (jobs).

## Simulation
- num_agents (int, default: 200)
  - Purpose: Number of agents in the simulation.
- max_iterations (int, default: 200)
  - Purpose: Maximum number of steps (runs, jobs).
- exploration_bonus (float, default: 0.1)
  - Purpose: Weight for exploratory agent movement.
- pheromone_decay (float, default: 0.95)
  - Purpose: Decay factor of pheromone trails per step.

## Reporting & Judge (LLM-driven Context Control)
- report_enabled (bool, default: false)
  - Purpose: Enable periodic step reports.
- report_every (int, default: 5)
  - Purpose: Interval (in steps) for reports.
- report_mode (string, see above)
  - Purpose: Report prompt mode.
- judge_enabled (bool, default: true)
  - Purpose: Enable LLM judge (context control).
- max_reports (int, default: 20)
  - Purpose: Budget limit for report/judge steps.
- max_report_tokens (int, default: 20000; characters approximated)
  - Purpose: Rough token/character budget for reports/judge.

### Blended Scoring / Pruning
- alpha (float, default: 0.7)
  - Purpose: Weight for cosine similarity.
- beta (float, default: 0.1)
  - Purpose: Weight for visit norm (visit_norm).
- gamma (float, default: 0.1)
  - Purpose: Weight for trail degree.
- delta (float, default: 0.1)
  - Purpose: Weight for LLM vote (−1/0/1).
- epsilon (float, default: 0.0)
  - Purpose: Length/prior weight (bm25-like).
- min_content_chars (int, default: 80)
  - Purpose: Minimum character length for chunk evaluation/pruning.
- import_only_penalty (float, default: 0.4)
  - Purpose: Penalty weight for import-only chunks.

## LLM Provider (central defaults in src/embeddinggemma/llm/config.py)
- llm_provider (string, default: ollama)
  - Values: ollama, openai, google, grok.

### Ollama
- ollama_model (string, default: qwen2.5-coder:7b)
- ollama_host (string, default: http://127.0.0.1:11434)
- ollama_system (string|null, default: null)
- ollama_num_gpu (int|null, default: env/None)
- ollama_num_thread (int|null, default: env/None)
- ollama_num_batch (int|null, default: env/None)

### OpenAI
- openai_model (string, default: gpt-4o-mini)
- openai_api_key (string|null)
- openai_base_url (string, default: https://api.openai.com)
- openai_temperature (float, default: 0.0)

### Google
- google_model (string, default: gemini-1.5-pro)
- google_api_key (string|null)
- google_base_url (string, default: https://generativelanguage.googleapis.com)
- google_temperature (float, default: 0.0)

### Grok
- grok_model (string, default: grok-2-latest)
- grok_api_key (string|null)
- grok_base_url (string, default: https://api.x.ai)
- grok_temperature (float, default: 0.0)

## Actions (UI)
These don't change persistent state but are workflow-relevant:
- Apply (POST /config)
- Start (POST /start – initializes corpus/simulation)
- Stop (POST /stop)
- Reset (POST /reset – fully resets simulation, configuration remains)
- Pause/Resume (POST /pause, /resume)
- Add Agents/Resize Agents (POST /agents/add, /agents/resize)
- Corpus Listing (GET /corpus/list)
- Shard Run (POST /jobs/start)
- Search/Answer (POST /search, /answer)

## Mapping & Usage
The most important uses per setting are stored in settings_usage_lines in the backend. Examples:
- viz_dims → Projection & UI (PCA 2D/3D)
- num_agents, pheromone_decay, exploration_bonus → mcmp/simulation.py
- report_*, judge_*, alpha..epsilon, min_content_chars, import_only_penalty → LLM control/blended score in realtime/server.py
- Provider-specific fields → LLM dispatcher src/embeddinggemma/llm/dispatcher.py and calls in realtime/server.py

Note: To maintain defaults centrally, src/embeddinggemma/llm/config.py can be modified. .env values override these defaults. The frontend only sends set/modified values.



