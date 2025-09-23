## Maintenance Guide

This guide helps maintain the FastAPI + React MCMP system, tests, and docs.

### Overview
- Backend: `FastAPI` app at `src/embeddinggemma/realtime/server.py` (port 8011)
- Frontend: React + Vite dev server in `frontend/` (port 5173), proxies to backend
- Simulation: MCMP loop via `MCPMRetriever` and `mcmp.simulation.*`
- Docs: see `docs/` (e.g., `mcmp_simulation.md`, `rag_query_prompt.md`)

### Environment
- Python venv recommended: `.venv`
- Install backend deps (includes uvicorn standard):
  - Windows PowerShell: `./run-realtime.ps1` (installs if needed, runs server)
- Frontend deps:
  - `cd frontend && npm i`
  - Install Playwright browsers: `npx playwright install --with-deps`

### Start/Stop
- Backend (dev, auto-reload):
  - `./run-realtime.ps1`
  - Equivalent: `./.venv/Scripts/python.exe -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011`
- Frontend (dev):
  - `cd frontend && npm run dev`
- Streamlit legacy (ensures correct venv/CUDA):
  - `./run-streamlit.ps1` (or `run-streamlit.cmd`)

### Health & Ports
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8011` (landing page) and WebSocket `/ws`
- Snapshot stream: WebSocket to `ws://localhost:8011/ws`
- Verify: open `http://localhost:8011/static/index.html`

### Settings & Persistence
The backend persists settings to `.fungus_cache/settings.json`.

- GET `/settings` → returns current settings and usage lines
- POST `/settings` (JSON) → applies, persists to disk

Primary fields (validated by `SettingsModel`):
- Visualization: `redraw_every`, `min_trail_strength`, `max_edges`, `viz_dims`
- Corpus: `use_repo`, `root_folder`, `max_files`, `exclude_dirs`, `windows`, `chunk_workers`
- Simulation: `max_iterations`, `num_agents`, `exploration_bonus`, `pheromone_decay`, `embed_batch_size`, `max_chunks_per_shard`

Note: `windows` (line chunk sizes) must be provided by the frontend. No hard-coded defaults.

### Simulation Control
- Start: `POST /start` with JSON settings (at least `query`, `windows`, and corpus knobs)
- Live update: `POST /config` for non-destructive changes during a run
- Pause/Resume: `POST /pause`, `POST /resume`
- Stop: `POST /stop`
- Agents: `POST /agents/add?n=NN`, `POST /agents/resize?count=NN`
- Document detail: `GET /doc/{id}` (full content + embedding)

Example minimal `/start` body:
```json
{
  "query": "Explain the architecture.",
  "viz_dims": 3,
  "redraw_every": 2,
  "min_trail_strength": 0.05,
  "max_edges": 600,
  "use_repo": true,
  "root_folder": "src",
  "max_files": 1000,
  "exclude_dirs": [".venv","node_modules",".git","external"],
  "windows": [1000,2000,4000],
  "chunk_workers": 8,
  "max_iterations": 200,
  "num_agents": 200,
  "exploration_bonus": 0.1,
  "pheromone_decay": 0.95,
  "embed_batch_size": 128,
  "max_chunks_per_shard": 2000
}
```

### MCMP Equations & Convergence
See `docs/mcmp_simulation.md` for:
- Agent forces, motion, pheromone deposit/decay
- Relevance formula and bonuses
- Metrics (`avg_rel`, `max_rel`, `trails`, `avg_speed`)
- Convergence heuristics (trail stagnation, avg relevance band)

### Testing
- Pytest (unit): `pytest -q` (mcmp and rag suites)
- Playwright (e2e):
  - Ensure backend is running, then: `cd frontend && npm run test:e2e`

### Troubleshooting
- WebSocket fails (404 or closed): ensure backend is running, installed `uvicorn[standard]`, and Vite proxy is configured (see `frontend/vite.config.ts`).
- CORS errors: backend includes permissive `CORSMiddleware`; verify origins and port alignment.
- FAISS IVF training error on small corpora: `build_faiss_index` auto-falls back to `Flat` for N < 4096.
- CUDA not used in Streamlit: launch via `run-streamlit.ps1` to use venv Python with the correct Torch build.
- Simulation not starting: ensure `/start` body includes `windows` and that the backend logs "started" with docs/agents counts.

### Conventions
- Do not hard-code chunk sizes; always pass via frontend `/start` or `/settings`.
- Keep `SettingsModel` constraints aligned with UI controls.
- Use PCA basis consistently for 2D/3D projections in snapshots.

### Release Checklist
- Backend starts, `/ws` stable
- Frontend UI parity (light/dark/system) and Plotly theming
- Playwright smoke tests green (streaming, search, theme toggle)
- Pytest suites green
- Update docs (`mcmp_simulation.md`, `rag_query_prompt.md`, this file)


