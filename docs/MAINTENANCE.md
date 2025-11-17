# Maintenance Guide

This guide helps maintain the FastAPI + React MCMP system after the modular refactoring.

## Overview

- **Backend**: FastAPI app with 8 routers and 3 services (port 8011)
- **Frontend**: React + Vite dev server (port 5173), proxies to backend
- **Simulation**: MCMP loop via `MCPMRetriever` and `mcmp.simulation.*`
- **Architecture**: Modular design with routers, services, hooks, and components
- **Docs**: See `docs/` for detailed documentation

---

## Environment Setup

### Python Backend

**Virtual Environment:**
```bash
# Create venv
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\.venv\Scripts\activate.bat

# Activate (Linux/Mac)
source .venv/bin/activate
```

**Install Dependencies:**
```bash
pip install -e .
pip install uvicorn[standard]
```

**Environment Variables:**
Create a `.env` file in the project root by copying `.env.example`:

```bash
cp .env.example .env
```

Then edit `.env` with your configuration. See `.env.example` for comprehensive documentation of all available environment variables including:
- LLM provider selection (Ollama, OpenAI, Google, Grok)
- Embedding model configuration
- Vector database settings (Qdrant or FAISS)
- Server ports and logging configuration
- Provider-specific options (GPU settings, temperature, etc.)

For full variable reference, see [ENV.md](ENV.md) or [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

### Frontend

**Install Dependencies:**
```bash
cd frontend
npm install
```

**Install Playwright (for E2E tests):**
```bash
npx playwright install --with-deps
```

---

## Start/Stop

### Backend (Development)

**Using PowerShell Script (Recommended for Windows):**
```bash
./run-realtime.ps1
```

**Manual Start:**
```bash
# From project root
.\.venv\Scripts\python.exe -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011
```

**What happens:**
- Loads `.env` variables
- Initializes FastAPI app
- Mounts static files
- Registers 8 routers
- Starts WebSocket endpoint
- Listens on `http://localhost:8011`

### Frontend (Development)

**Start Dev Server:**
```bash
cd frontend
npm run dev
```

**What happens:**
- Starts Vite dev server
- Proxies API calls to backend (port 8011)
- Enables hot module replacement
- Listens on `http://localhost:5173`

### Stop Services

- **Backend**: `Ctrl+C` in terminal
- **Frontend**: `Ctrl+C` in terminal

---

## Health Checks

### Backend Health

**Landing Page:**
```bash
curl http://localhost:8011/
```

**API Introspection:**
```bash
curl http://localhost:8011/introspect/api
```

**Settings Endpoint:**
```bash
curl http://localhost:8011/settings
```

**WebSocket:**
- Connect to `ws://localhost:8011/ws`
- Should receive initial status message

### Frontend Health

**Browser:**
- Open `http://localhost:5173`
- UI should load without errors
- Check browser console for errors

**Build:**
```bash
cd frontend
npm run build
```

---

## Settings & Persistence

### Settings Management

**Location:** `.fungus_cache/settings.json`

**Load Settings:**
```bash
GET http://localhost:8011/settings
```

**Update Settings:**
```bash
POST http://localhost:8011/settings
Content-Type: application/json

{
  "num_agents": 300,
  "exploration_bonus": 0.2
}
```

**Settings Validation:**
All settings are validated via `SettingsModel` in `services/settings_manager.py`:
- Type checking
- Range validation
- Required field validation
- Default values

**Key Setting Categories:**

1. **Visualization**
   - `viz_dims` (2 or 3)
   - `redraw_every` (1-100)
   - `min_trail_strength` (0.0-1.0)
   - `max_edges` (10-5000)

2. **Corpus**
   - `use_repo` (boolean)
   - `root_folder` (string)
   - `max_files` (0-20000)
   - `exclude_dirs` (string[])
   - `windows` (int[] - required for chunking)

3. **Simulation**
   - `max_iterations` (1-5000)
   - `num_agents` (1-10000)
   - `exploration_bonus` (0.01-1.0)
   - `pheromone_decay` (0.5-0.999)

4. **LLM & Reporting**
   - `report_enabled` (boolean)
   - `report_every` (1-100)
   - `report_mode` (deep/structure/exploratory/summary/repair/steering)
   - `judge_enabled` (boolean)
   - `max_reports` (0-1000)
   - `max_report_tokens` (0-1000000)

---

## Simulation Control

### Start Simulation

**Minimal Request:**
```json
POST /start
{
  "query": "Explain the architecture",
  "windows": [1000, 2000, 4000]
}
```

**Full Request:**
See `docs/API_REFERENCE.md` for complete example.

**What Happens:**
1. Validates settings
2. Builds corpus from `root_folder` or `src`
3. Chunks files using multi-window strategy
4. Generates embeddings for all chunks
5. Initializes agents in embedding space
6. Starts simulation loop
7. Broadcasts snapshots via WebSocket

### Control Endpoints

```bash
# Start
POST /start

# Update config during runtime
POST /config
{"redraw_every": 3}

# Stop
POST /stop

# Reset (keeps config)
POST /reset

# Pause
POST /pause

# Resume
POST /resume

# Get status
GET /status
```

### Agent Management

```bash
# Add 50 agents
POST /agents/add?n=50

# Resize to 300 agents
POST /agents/resize?count=300
```

---

## MCMP Equations & Convergence

See `docs/mcmp_simulation.md` for detailed formulas.

**Key Metrics:**
- `avg_rel` - Average relevance across documents
- `max_rel` - Maximum relevance score
- `trails` - Number of pheromone trails
- `avg_speed` - Average agent movement speed

**Convergence Heuristics:**
- Trail count stagnation (no new trails for N steps)
- Average relevance plateau (stable for N steps)
- Maximum iterations reached

---

## Adding New Features

### Adding a New Router

1. **Create Router File:**
   ```python
   # src/embeddinggemma/realtime/routers/my_router.py
   from fastapi import APIRouter, Depends
   from typing import Any

   router = APIRouter(prefix="/myrouter", tags=["myrouter"])

   _get_streamer_dependency: Any = None

   def get_streamer() -> Any:
       if _get_streamer_dependency is None:
           raise RuntimeError("Streamer dependency not configured")
       return _get_streamer_dependency()

   @router.get("/hello")
   async def hello(streamer: Any = Depends(get_streamer)):
       return {"message": "Hello from my router"}
   ```

2. **Register Router in server.py:**
   ```python
   # In server.py, after other router imports
   from embeddinggemma.realtime.routers import my_router

   # After streamer initialization
   my_router._get_streamer_dependency = get_streamer_instance
   app.include_router(my_router.router)
   ```

3. **Test:**
   ```bash
   curl http://localhost:8011/myrouter/hello
   ```

**Best Practices:**
- One router per logical domain
- Use dependency injection for streamer access
- Add type hints
- Document endpoints with docstrings
- Add error handling
- Return JSONResponse for consistency

---

### Adding a New Service

1. **Create Service File:**
   ```python
   # src/embeddinggemma/realtime/services/my_service.py
   from typing import Any

   class MyService:
       def __init__(self, config: dict):
           self.config = config

       def do_something(self, param: str) -> dict:
           # Service logic here
           return {"result": param}
   ```

2. **Import in Router:**
   ```python
   from embeddinggemma.realtime.services.my_service import MyService

   @router.get("/use-service")
   async def use_service(streamer: Any = Depends(get_streamer)):
       service = MyService(config={"key": "value"})
       result = service.do_something("test")
       return result
   ```

**Best Practices:**
- Services encapsulate business logic
- No HTTP concerns in services
- Testable in isolation
- Use type hints
- Add error handling

---

### Adding a New Frontend Component

1. **Create Component File:**
   ```typescript
   // frontend/src/components/MyComponent.tsx
   import React from 'react'

   interface MyComponentProps {
     title: string
     onAction: () => void
   }

   export function MyComponent({ title, onAction }: MyComponentProps) {
     return (
       <div className="my-component">
         <h3>{title}</h3>
         <button onClick={onAction}>Click Me</button>
       </div>
     )
   }
   ```

2. **Add to App.tsx:**
   ```typescript
   import { MyComponent } from '../components/MyComponent'

   function AppContent() {
     // ... existing code

     const handleMyAction = () => {
       console.log('Action triggered')
     }

     return (
       <>
         {/* ... existing components */}
         <MyComponent title="My Feature" onAction={handleMyAction} />
       </>
     )
   }
   ```

**Best Practices:**
- Small, focused components
- Use TypeScript interfaces for props
- Extract logic to custom hooks
- Use context for global state
- Add CSS classes for styling

---

### Adding a New Hook

1. **Create Hook File:**
   ```typescript
   // frontend/src/hooks/useMyHook.ts
   import { useState, useEffect } from 'react'

   export function useMyHook(initialValue: string) {
     const [value, setValue] = useState(initialValue)

     useEffect(() => {
       // Side effects here
     }, [])

     return { value, setValue }
   }
   ```

2. **Use in Component:**
   ```typescript
   import { useMyHook } from '../hooks/useMyHook'

   function MyComponent() {
     const { value, setValue } = useMyHook('initial')

     return <input value={value} onChange={e => setValue(e.target.value)} />
   }
   ```

**Best Practices:**
- Prefix with `use`
- Return object with named properties
- Document parameters and return values
- Handle cleanup in useEffect
- Make reusable across components

---

## Testing

### Backend Tests (Pytest)

**Run Tests:**
```bash
pytest -v
pytest tests/test_mcmp.py
pytest tests/test_rag.py
```

**Coverage:**
```bash
pytest --cov=embeddinggemma --cov-report=html
```

### Frontend Tests (Vitest)

**Run Tests:**
```bash
cd frontend
npm test
```

### E2E Tests (Playwright)

**Prerequisites:**
- Backend must be running
- Frontend must be running

**Run Tests:**
```bash
cd frontend
npm run test:e2e
```

**Test Structure:**
```typescript
test('should load app', async ({ page }) => {
  await page.goto('http://localhost:5173')
  await expect(page.locator('h1')).toContainText('MCMP')
})
```

---

## Troubleshooting

### WebSocket Issues

**Symptom:** WebSocket connection fails (404 or closed)

**Solutions:**
1. Ensure backend is running on port 8011
2. Check `uvicorn[standard]` is installed
3. Verify Vite proxy config in `frontend/vite.config.ts`:
   ```typescript
   proxy: {
     '/ws': {
       target: 'ws://localhost:8011',
       ws: true
     }
   }
   ```

### CORS Errors

**Symptom:** Browser blocks API requests

**Solutions:**
1. Verify CORS middleware in `server.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```
2. Check frontend proxy configuration
3. Ensure ports match (backend 8011, frontend 5173)

### FAISS Training Error

**Symptom:** `FAISS IVF training error on small corpora`

**Solution:**
- Auto-handled by backend
- Falls back to `Flat` index for N < 4096 chunks
- No action needed

### CUDA Not Used

**Symptom:** Embeddings not using GPU

**Solutions:**
1. Launch via `run-realtime.ps1` to use venv Python
2. Verify PyTorch CUDA installation:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
3. Check CUDA drivers and toolkit

### Simulation Not Starting

**Symptom:** `/start` returns success but simulation doesn't run

**Solutions:**
1. Check backend logs for errors
2. Ensure `windows` is provided in `/start` payload
3. Verify corpus has files (check `.fungus_cache/`)
4. Check WebSocket connection is active

---

## File Organization

### Backend Structure
```
src/embeddinggemma/realtime/
├── server.py                 # Main app (1,288 lines)
├── routers/                  # 8 routers (1,475 lines total)
│   ├── collections.py        # Collection management
│   ├── simulation.py         # Simulation control
│   ├── search.py             # Search & answer
│   ├── agents.py             # Agent control
│   ├── settings.py           # Settings API
│   ├── prompts.py            # Prompt management
│   ├── corpus.py             # Corpus management
│   └── misc.py               # Misc endpoints
└── services/                 # 3 services (721 lines total)
    ├── settings_manager.py   # Settings persistence
    ├── prompts_manager.py    # Prompt templates
    └── qdrant_service.py     # Qdrant client
```

### Frontend Structure
```
frontend/src/
├── ui/App.tsx                # Main component (230 lines)
├── types/index.ts            # Type definitions (143 lines)
├── services/api.ts           # API client (149 lines)
├── hooks/                    # 3 hooks (613 lines total)
│   ├── useWebSocket.ts
│   ├── useSimulationState.ts
│   └── useSettings.ts
├── context/                  # 1 context (124 lines)
│   └── SimulationContext.tsx
└── components/               # 10 components (1,618 lines total)
    ├── ControlPanel.tsx
    ├── SearchPanel.tsx
    ├── SettingsPanel.tsx
    ├── MetricsPanel.tsx
    ├── LogPanel.tsx
    ├── ResultsPanel.tsx
    ├── StatusBar.tsx
    ├── VisualizationPanel.tsx
    ├── CorpusModal.tsx
    └── PromptsModal.tsx
```

---

## Refactoring Benefits

### Backend (40% reduction)
- **Before**: 2,029 lines in server.py
- **After**: 1,288 lines in server.py + 2,199 lines in routers/services
- **Benefit**: Modular, testable, maintainable

### Frontend (77.8% reduction)
- **Before**: 1,037 lines in App.tsx
- **After**: 230 lines in App.tsx + 2,647 lines in modules
- **Benefit**: Clean separation, reusable hooks, typed components

---

## Conventions

### Backend Conventions

1. **Router Naming**: Use descriptive, domain-specific names
2. **Dependency Injection**: Always use for streamer access
3. **Error Handling**: Return JSONResponse with status field
4. **Validation**: Use Pydantic models for all inputs
5. **Logging**: Use module-level logger
6. **Type Hints**: Add type hints to all functions

### Frontend Conventions

1. **Component Naming**: PascalCase for components
2. **Hook Naming**: Prefix with `use`
3. **File Naming**: Match component/hook name
4. **Props**: Define interfaces for all props
5. **State**: Use hooks for local state, context for global
6. **API Calls**: Use `services/api.ts` functions

---

## Release Checklist

- [ ] Backend starts without errors
- [ ] Frontend builds without errors
- [ ] WebSocket connection stable
- [ ] All routers responding correctly
- [ ] Settings persistence working
- [ ] Simulation starts and runs
- [ ] Visualization renders correctly
- [ ] Dark/light themes working
- [ ] Pytest suites pass
- [ ] E2E tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

---

## Useful Commands

### Backend

```bash
# Start with auto-reload
uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011

# Run tests
pytest -v

# Check types
mypy src/embeddinggemma/realtime/

# Lint
flake8 src/embeddinggemma/realtime/
```

### Frontend

```bash
# Dev server
npm run dev

# Build
npm run build

# Preview build
npm run preview

# Tests
npm test

# E2E tests
npm run test:e2e

# Lint
npm run lint

# Type check
npm run type-check
```

---

## See Also

- **Architecture**: `docs/ARCHITECTURE.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Frontend Architecture**: `docs/FRONTEND_ARCHITECTURE.md`
- **Configuration**: `docs/CONFIG_REFERENCE.md`
- **Refactoring History**: `docs/REFACTORING_HISTORY.md`
- **MCMP Simulation**: `docs/mcmp_simulation.md`
- **RAG Query Prompt**: `docs/rag_query_prompt.md`
