# Scripts and Entry Points

This document describes how to start and use the LA Fungus Search system.

## Current System Architecture

The system consists of:
- **Backend**: FastAPI server (Python) on port 8011
- **Frontend**: React application (TypeScript) on port 5173
- **Vector DB**: Qdrant (optional) or in-memory FAISS

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture.

---

## Primary Entry Points

### 1. Backend Server (FastAPI)

**Windows PowerShell:**
```powershell
.\run-realtime.ps1
```

**Manual start (cross-platform):**
```bash
# From project root
python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011
```

- **Status**: Active (primary backend)
- **Purpose**: FastAPI server providing REST API and WebSocket for real-time MCMP simulation
- **Port**: 8011
- **Auto-reload**: Enabled for development
- **When to use**: Always required for the system to function

**Key endpoints:**
- `http://localhost:8011/` - API root
- `http://localhost:8011/docs` - Interactive API documentation (Swagger)
- `http://localhost:8011/introspect/api` - API introspection endpoint
- WebSocket: `ws://localhost:8011/ws` - Real-time simulation updates

See [API_REFERENCE.md](API_REFERENCE.md) for complete endpoint documentation (32 endpoints across 8 routers).

### 2. Frontend Application (React)

```bash
cd frontend
npm install  # First time only
npm run dev
```

- **Status**: Active (primary frontend)
- **Purpose**: React+TypeScript UI with real-time visualization of MCMP simulation
- **Port**: 5173 (Vite dev server)
- **URL**: `http://localhost:5173`
- **When to use**: Day-to-day interactive exploration, running searches, viewing results, configuring settings

**Features:**
- Real-time simulation visualization (2D/3D PCA projections)
- Interactive control panel for simulation parameters
- Search and answer interfaces
- Settings management UI
- Corpus management
- Live metrics and logging

See [FRONTEND_ARCHITECTURE.md](FRONTEND_ARCHITECTURE.md) for component details.

### 3. Production Build

```bash
# Frontend production build
cd frontend
npm run build

# Serve production build
npm run preview
```

- **Purpose**: Optimized production build
- **Output**: `frontend/dist/`
- **When to use**: Production deployment

---

## Utility Scripts

### Code Analysis Tools

**src/embeddinggemma/codespace_analyzer.py**
- **Status**: Active
- **Purpose**: Quick code-space analyzer tool for scanning project structure
- **Run**: `python -m embeddinggemma.codespace_analyzer`
- **When to use**: Analyzing project structure and complexity

**src/embeddinggemma/agents/agent_fungus_rag.py**
- **Status**: Active (advanced CLI agent)
- **Purpose**: Combined tools agent for programmatic access to RAG capabilities
- **Run**: `python -m embeddinggemma.agents.agent_fungus_rag [options]`
- **When to use**: Headless runs, automation, CI/CD integration

---

## Development Workflow

### Standard Development Setup

```bash
# Terminal 1 - Backend
python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011

# Terminal 2 - Frontend
cd frontend
npm run dev

# Open browser
# Navigate to http://localhost:5173
```

### With Qdrant (Persistent Storage)

```bash
# Terminal 1 - Start Qdrant
docker-compose -f docker-compose.qdrant.yml up -d

# Terminal 2 - Backend (with Qdrant configured in .env)
python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011

# Terminal 3 - Frontend
cd frontend
npm run dev
```

See [MAINTENANCE.md](MAINTENANCE.md) for detailed setup instructions.

---

## Configuration

All runtime configuration is managed through:
- **Environment variables**: `.env` file (see `.env.example`)
- **UI settings**: Settings panel in the frontend
- **API calls**: POST `/config` endpoint

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for all available settings.

---

## Legacy Scripts (Removed)

The following scripts have been removed in favor of the current React+FastAPI architecture:

- ❌ **streamlit_fungus_backup.py** - Replaced by React frontend
- ❌ **src/embeddinggemma/rag_v1.py** - Integrated into realtime system
- ❌ **src/embeddinggemma/fungus_api.py** - Replaced by realtime/server.py
- ❌ **src/embeddinggemma/app.py** - Streamlit demo removed
- ❌ **src/embeddinggemma/cli.py** - Simple CLI removed

For historical context, see [REFACTORING_HISTORY.md](REFACTORING_HISTORY.md).

---

## Quick Reference

| Task | Command |
|------|---------|
| Start backend | `.\run-realtime.ps1` (Windows) or `python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011` |
| Start frontend | `cd frontend && npm run dev` |
| View API docs | Navigate to `http://localhost:8011/docs` |
| View app | Navigate to `http://localhost:5173` |
| Production build | `cd frontend && npm run build` |
| Start Qdrant | `docker-compose -f docker-compose.qdrant.yml up -d` |

---

## Troubleshooting

**Port 8011 already in use:**
```bash
# Find and kill process using port 8011
# Windows:
netstat -ano | findstr :8011
taskkill /PID <process_id> /F

# Linux/Mac:
lsof -ti:8011 | xargs kill -9
```

**Frontend build errors:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

For more troubleshooting, see [MAINTENANCE.md](MAINTENANCE.md#troubleshooting).
