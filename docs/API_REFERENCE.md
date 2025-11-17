# API Reference

This document provides a complete reference for all API endpoints in the LA Fungus Search system, organized by router.

## Base URL

**Development:**
- Backend: `http://localhost:8011`
- Frontend: `http://localhost:5173` (proxies to backend)

**WebSocket:**
- `ws://localhost:8011/ws` - Real-time simulation updates

---

## API Organization

The API is organized into 8 routers with 32 endpoints total:

| Router | Prefix | Endpoints | Purpose |
|--------|--------|-----------|---------|
| Collections | `/collections` | 4 | Qdrant collection management |
| Simulation | `/` | 7 | Simulation control |
| Search | `/` | 3 | Search and answer |
| Agents | `/agents` | 2 | Agent control |
| Settings | `/` | 2 | Settings management |
| Prompts | `/prompts` | 2 | Prompt management |
| Corpus | `/corpus` | 6 | Corpus management |
| Misc | `/` | 6 | Miscellaneous utilities |

---

## Collections Router

### `GET /collections/list`

List all available Qdrant collections with metadata.

**Response:**
```json
{
  "status": "ok",
  "collections": [
    {
      "name": "codebase",
      "point_count": 1234,
      "dimension": 384,
      "is_active": true
    }
  ],
  "active_collection": "codebase"
}
```

**Note:** In memory mode, returns a mock collection with current chunk count.

---

### `POST /collections/switch`

Switch the active Qdrant collection.

**Request:**
```json
{
  "collection_name": "my_collection"
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Switched to collection 'my_collection'"
}
```

**Error Cases:**
- 404: Collection not found
- 500: Switch failed

---

### `GET /collections/{collection_name}/info`

Get detailed information about a specific collection.

**Response:**
```json
{
  "status": "ok",
  "collection": {
    "name": "codebase",
    "point_count": 1234,
    "dimension": 384,
    "vectors_config": {...},
    "optimizer_status": "ok"
  }
}
```

---

### `DELETE /collections/{collection_name}`

Delete a Qdrant collection.

**Response:**
```json
{
  "status": "ok",
  "message": "Collection 'my_collection' deleted"
}
```

**Error Cases:**
- 404: Collection not found
- 500: Deletion failed

---

## Simulation Router

### `POST /start`

Start a new simulation with the provided settings.

**Request:**
```json
{
  "query": "Explain the authentication flow",
  "viz_dims": 3,
  "redraw_every": 2,
  "min_trail_strength": 0.05,
  "max_edges": 600,
  "use_repo": true,
  "root_folder": "src",
  "max_files": 1000,
  "exclude_dirs": [".venv", "node_modules", ".git"],
  "windows": [1000, 2000, 4000],
  "chunk_workers": 8,
  "max_iterations": 200,
  "num_agents": 200,
  "exploration_bonus": 0.1,
  "pheromone_decay": 0.95,
  "embed_batch_size": 128,
  "max_chunks_per_shard": 2000,
  "report_enabled": false,
  "report_every": 5,
  "report_mode": "deep",
  "judge_enabled": true,
  "judge_mode": "steering",
  "alpha": 0.7,
  "beta": 0.1,
  "gamma": 0.1,
  "delta": 0.1,
  "epsilon": 0.0
}
```

**Response:**
```json
{
  "status": "ok",
  "docs": 1234,
  "files": 56,
  "agents": 200
}
```

**Notes:**
- Builds corpus from specified directory
- Creates embeddings for all chunks
- Initializes agents in embedding space
- Starts simulation loop
- Settings are persisted to `.fungus_cache/settings.json`

---

### `POST /config`

Update simulation configuration during runtime (non-destructive).

**Request:**
```json
{
  "redraw_every": 3,
  "min_trail_strength": 0.1,
  "exploration_bonus": 0.15
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Config updated"
}
```

**Notes:**
- Does not rebuild corpus
- Does not reset simulation state
- Updates are applied immediately

---

### `POST /stop`

Stop the running simulation.

**Response:**
```json
{
  "status": "ok",
  "message": "Simulation stopped"
}
```

---

### `POST /reset`

Reset the simulation (keeps configuration).

**Response:**
```json
{
  "status": "ok",
  "message": "Simulation reset"
}
```

**Notes:**
- Resets agent positions
- Clears pheromone trails
- Resets iteration counter
- Keeps corpus and settings

---

### `POST /pause`

Pause the simulation.

**Response:**
```json
{
  "status": "ok",
  "message": "Simulation paused"
}
```

---

### `POST /resume`

Resume a paused simulation.

**Response:**
```json
{
  "status": "ok",
  "message": "Simulation resumed"
}
```

---

### `GET /status`

Get current simulation status.

**Response:**
```json
{
  "status": "ok",
  "state": "running",
  "iteration": 42,
  "docs": 1234,
  "agents": 200,
  "metrics": {
    "avg_rel": 0.65,
    "max_rel": 0.92,
    "trails": 847,
    "avg_speed": 0.023
  }
}
```

**State Values:**
- `idle` - Not running
- `running` - Simulation active
- `paused` - Simulation paused
- `stopped` - Simulation stopped

---

## Search Router

### `POST /search`

Perform hybrid search (embedding + BM25) on the corpus.

**Request:**
```json
{
  "query": "authentication middleware",
  "top_k": 10
}
```

**Response:**
```json
{
  "status": "ok",
  "results": [
    {
      "relevance_score": 0.87,
      "metadata": {
        "file_path": "src/auth/middleware.py",
        "line_range": [15, 45]
      },
      "content": "def authenticate_request(request):\n    ..."
    }
  ]
}
```

---

### `POST /answer`

Search the corpus and generate an LLM answer.

**Request:**
```json
{
  "query": "How does authentication work?",
  "top_k": 5
}
```

**Response:**
```json
{
  "status": "ok",
  "answer": "Authentication in this system works by...",
  "results": [...]
}
```

**Notes:**
- Performs search first
- Builds context from top results
- Sends to configured LLM provider
- Returns both answer and supporting results

---

### `GET /doc/{doc_id}`

Get full document details by ID.

**Response:**
```json
{
  "status": "ok",
  "doc": {
    "id": 42,
    "content": "full document content...",
    "embedding": [0.123, 0.456, ...],
    "relevance_score": 0.87,
    "visit_count": 15,
    "metadata": {
      "file_path": "src/auth/middleware.py",
      "line_range": [15, 45],
      "window_size": 1000
    }
  }
}
```

---

## Agents Router

### `POST /agents/add`

Add agents to the running simulation.

**Query Parameters:**
- `n` (int, default=10) - Number of agents to add

**Request:**
```
POST /agents/add?n=50
```

**Response:**
```json
{
  "status": "ok",
  "message": "Added 50 agents, new total: 250"
}
```

---

### `POST /agents/resize`

Resize the agent count (add or remove).

**Query Parameters:**
- `count` (int, required) - Target agent count

**Request:**
```
POST /agents/resize?count=300
```

**Response:**
```json
{
  "status": "ok",
  "message": "Resized to 300 agents"
}
```

---

## Settings Router

### `GET /settings`

Get current settings with usage metadata.

**Response:**
```json
{
  "status": "ok",
  "settings": {
    "query": "Explain architecture",
    "viz_dims": 3,
    "num_agents": 200,
    ...
  },
  "usage": {
    "query": "Search query for simulation and reports",
    "viz_dims": "Visualization dimensions (2D or 3D)",
    ...
  }
}
```

**Notes:**
- Returns all current settings
- Includes usage descriptions for each setting
- Useful for building dynamic UIs

---

### `POST /settings`

Update and persist settings.

**Request:**
```json
{
  "num_agents": 300,
  "exploration_bonus": 0.2,
  "pheromone_decay": 0.93
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Settings updated and saved"
}
```

**Notes:**
- Validates all settings via Pydantic model
- Persists to `.fungus_cache/settings.json`
- Does not restart simulation (use `/config` for runtime updates)

---

## Prompts Router

### `GET /prompts`

Get available prompt modes and templates.

**Response:**
```json
{
  "status": "ok",
  "modes": ["deep", "structure", "exploratory", "summary", "repair", "steering"],
  "defaults": {
    "deep": "Analyze the code deeply...",
    "structure": "Describe the structure...",
    ...
  },
  "overrides": {
    "deep": "Custom prompt override..."
  }
}
```

---

### `POST /prompts/save`

Save prompt template override.

**Request:**
```json
{
  "mode": "deep",
  "template": "Custom analysis prompt template..."
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Prompt override saved for mode 'deep'"
}
```

**Notes:**
- Persists to `.fungus_cache/prompts/{mode}.txt`
- Overrides are loaded on next LLM call

---

## Corpus Router

### `GET /corpus/list`

List corpus files with pagination.

**Query Parameters:**
- `page` (int, default=1) - Page number (1-indexed)
- `per_page` (int, default=100) - Items per page

**Request:**
```
GET /corpus/list?page=2&per_page=50
```

**Response:**
```json
{
  "status": "ok",
  "files": [
    "src/auth/middleware.py",
    "src/auth/utils.py",
    ...
  ],
  "total": 567,
  "page": 2,
  "per_page": 50
}
```

---

### `GET /corpus/summary`

Get corpus statistics.

**Response:**
```json
{
  "status": "ok",
  "total_files": 567,
  "total_chunks": 12345,
  "total_size_bytes": 5432100,
  "avg_chunk_size": 450,
  "windows": [1000, 2000, 4000]
}
```

---

### `POST /corpus/add_file`

Add a single file to the corpus.

**Request:**
```json
{
  "file_path": "src/new_module.py"
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "File added, 3 new chunks"
}
```

---

### `POST /corpus/update_file`

Update an existing file in the corpus.

**Request:**
```json
{
  "file_path": "src/existing_module.py"
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "File updated, 5 chunks modified"
}
```

---

### `POST /corpus/reindex`

Reindex the entire corpus.

**Response:**
```json
{
  "status": "ok",
  "message": "Reindex started",
  "job_id": "reindex_20250109_143022"
}
```

**Notes:**
- Runs asynchronously
- Returns job ID for status tracking
- Use `/jobs/status` to monitor progress

---

### `POST /corpus/index_repo`

Index an entire repository.

**Request:**
```json
{
  "root_folder": "path/to/repo",
  "max_files": 1000,
  "exclude_dirs": [".git", "node_modules"]
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Repository indexing started",
  "job_id": "index_20250109_143022"
}
```

---

## Misc Router

### `GET /`

Landing page endpoint.

**Response:**
```html
<html>
  <body>
    <h1>MCMP Realtime API</h1>
    <p>Backend is running on port 8011</p>
    <p>WebSocket available at /ws</p>
  </body>
</html>
```

---

### `GET /introspect/api`

API introspection endpoint (returns all endpoints).

**Response:**
```json
{
  "status": "ok",
  "endpoints": [
    {
      "path": "/start",
      "method": "POST",
      "tags": ["simulation"],
      "summary": "Start simulation"
    },
    ...
  ]
}
```

**Notes:**
- Auto-generated from FastAPI route table
- Useful for API discovery and testing

---

### `POST /run/new`

Create a new run (for tracking multiple simulation sessions).

**Response:**
```json
{
  "status": "ok",
  "run_id": "run_20250109_143022"
}
```

**Notes:**
- Creates directory in `.fungus_cache/runs/{run_id}`
- Used for organizing reports and logs

---

### `POST /jobs/start`

Start a batch processing job.

**Request:**
```json
{
  "query": "Find all authentication code",
  "max_iterations": 100,
  "shard_size": 2000
}
```

**Response:**
```json
{
  "status": "ok",
  "job_id": "job_20250109_143022"
}
```

---

### `GET /jobs/status`

Get job status.

**Query Parameters:**
- `job_id` (string, required) - Job identifier

**Request:**
```
GET /jobs/status?job_id=job_20250109_143022
```

**Response:**
```json
{
  "status": "ok",
  "job_id": "job_20250109_143022",
  "state": "running",
  "progress": 0.65,
  "current_shard": 3,
  "total_shards": 5,
  "eta_seconds": 120
}
```

---

### `POST /reports/merge`

Merge multiple reports into a summary.

**Request:**
```json
{
  "run_id": "run_20250109_143022"
}
```

**Response:**
```json
{
  "status": "ok",
  "summary": "Overall analysis of the codebase shows...",
  "report_count": 15
}
```

---

## WebSocket Protocol

### Connection

**URL:** `ws://localhost:8011/ws`

**Connection Flow:**
1. Client connects to WebSocket
2. Server sends initial status message
3. Server streams updates during simulation
4. Client can send config updates

### Message Types

**Snapshot (Server → Client)**
```json
{
  "type": "snapshot",
  "step": 42,
  "documents": {
    "xy": [[0.1, 0.2, 0.3], ...],
    "relevance": [0.8, 0.7, ...],
    "meta": [{"file_path": "..."}, ...]
  },
  "agents": {
    "xy": [[0.5, 0.6, 0.7], ...]
  },
  "edges": [
    {"source": 12, "target": 34, "strength": 0.8},
    ...
  ]
}
```

**Metrics (Server → Client)**
```json
{
  "type": "metrics",
  "step": 42,
  "avg_rel": 0.65,
  "max_rel": 0.92,
  "trails": 847,
  "avg_speed": 0.023
}
```

**Report (Server → Client)**
```json
{
  "type": "report",
  "step": 42,
  "data": {
    "summary": "Analysis summary...",
    "items": [
      {
        "file_path": "src/auth.py",
        "relevance_to_query": "High relevance because...",
        "code_purpose": "Handles authentication..."
      }
    ]
  }
}
```

**Results (Server → Client)**
```json
{
  "type": "results",
  "query": "authentication",
  "results": [
    {
      "relevance_score": 0.87,
      "content": "...",
      "metadata": {...}
    }
  ]
}
```

**Seed Queries (Server → Client)**
```json
{
  "type": "seed_queries",
  "added": ["query1", "query2"],
  "pool_size": 15
}
```

**Log (Server → Client)**
```json
{
  "type": "log",
  "message": "Simulation started with 200 agents",
  "level": "info",
  "timestamp": "2025-01-09T14:30:22Z"
}
```

**Job Progress (Server → Client)**
```json
{
  "type": "job_progress",
  "job_id": "job_20250109_143022",
  "percent": 65.5
}
```

**Config Update (Client → Server)**
```json
{
  "type": "config",
  "redraw_every": 3,
  "exploration_bonus": 0.15
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "status": "error",
  "message": "Error description",
  "details": "Optional additional details"
}
```

**Common HTTP Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

---

## Rate Limiting

Currently, there are no rate limits on API endpoints. In production deployments, consider adding:
- Rate limiting middleware
- Request throttling
- WebSocket connection limits

---

## Authentication

Currently, the API has no authentication. For production deployments:
- Add API key authentication
- Use JWT tokens for session management
- Implement role-based access control (RBAC)

---

## CORS

CORS is enabled for all origins in development mode. Configure appropriately for production:

```python
CORSMiddleware(
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## See Also

- **Architecture**: `docs/ARCHITECTURE.md`
- **Frontend**: `docs/FRONTEND_ARCHITECTURE.md`
- **Maintenance**: `docs/MAINTENANCE.md`
- **Configuration**: `docs/CONFIG_REFERENCE.md`
