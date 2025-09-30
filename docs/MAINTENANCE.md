## Maintenance Guide

This guide helps maintain the FastAPI + React MCMP system, tests, and docs.

### Overview
- **Backend**: FastAPI application at `src/embeddinggemma/realtime/server.py` (port 8011)
- **Frontend**: React + TypeScript application in `frontend/` (port 5173) with Vite build system
- **Simulation**: MCMP (Multi-agent Codebase Pattern Matching) via `MCPMRetriever` and `mcmp.simulation.*`
- **Documentation**: Available in `docs/` directory

### Environment Setup
- **Python venv recommended**: Create with `python -m venv .venv` and activate
- **Install backend dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
- **Frontend dependencies**:
  ```bash
  cd frontend && npm install
  ```
- **Install Playwright browsers** (for e2e tests):
  ```bash
  cd frontend && npx playwright install --with-deps
  ```

### Development Servers
- **Backend (dev, auto-reload)**:
  ```bash
  # Using the provided script (Windows PowerShell)
  ./run-realtime.ps1

  # Or manually with uvicorn
  python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011
  ```
- **Frontend (dev)**:
  ```bash
  cd frontend && npm run dev
  ```
- **Production build**:
  ```bash
  cd frontend && npm run build
  ```

### Access URLs
- **Frontend development**: `http://localhost:5173`
- **Backend API**: `http://localhost:8011`
- **WebSocket endpoint**: `ws://localhost:8011/ws`
- **API documentation**: `http://localhost:8011/docs` (FastAPI auto-generated docs)
- **Static files**: `http://localhost:8011/static/index.html`

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
- **Unit tests (pytest)**:
  ```bash
  pytest -q  # Run all tests in tests/ directory
  pytest tests/mcmp/ -v  # MCMP-specific tests
  pytest tests/rag/ -v   # RAG-specific tests
  ```
- **End-to-end tests (Playwright)**:
  ```bash
  cd frontend && npm run test:e2e
  ```
  Ensure backend is running before running e2e tests.

### Troubleshooting
- **WebSocket connection issues**: Ensure backend is running and Vite proxy is correctly configured in `frontend/vite.config.ts`
- **CORS errors**: Backend includes permissive CORS middleware; verify origins and port alignment in browser dev tools
- **FAISS index issues**: For small corpora (< 4096 docs), the system automatically falls back to Flat index
- **CUDA not detected**: Ensure PyTorch with CUDA support is installed and CUDA_VISIBLE_DEVICES is set appropriately
- **Simulation not starting**: Verify `/start` request includes required `windows` parameter and check backend logs for "started" message with docs/agents counts
- **Frontend build issues**: Clear `node_modules` and reinstall if dependency issues occur: `cd frontend && rm -rf node_modules && npm install`
- **Memory issues**: For large codebases, increase `max_files` setting gradually and monitor system resources

### Conventions
- Do not hard-code chunk sizes; always pass via frontend `/start` or `/settings`.
- Keep `SettingsModel` constraints aligned with UI controls.
- Use PCA basis consistently for 2D/3D projections in snapshots.

### Development Conventions
- **Configuration**: Never hard-code chunk sizes; always pass via frontend `/start` or `/settings`
- **Validation**: Keep `SettingsModel` constraints aligned with UI controls
- **Visualization**: Use PCA basis consistently for 2D/3D projections in snapshots
- **Error handling**: Implement proper error handling for WebSocket disconnections and API failures

### Release Checklist
- [ ] Backend starts successfully and WebSocket `/ws` remains stable
- [ ] Frontend UI renders correctly with light/dark/system theme support
- [ ] Plotly visualizations display correctly and update in real-time
- [ ] Playwright e2e tests pass (streaming, search, theme toggle, basic interactions)
- [ ] Pytest unit test suites pass for both MCMP and RAG modules
- [ ] Documentation updated (`ARCHITECTURE.md`, `MAINTENANCE.md`, this file)
- [ ] Configuration reference updated to reflect any new settings
- [ ] README.md exists and accurately describes project setup and usage


