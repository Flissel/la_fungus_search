# API Reference - Realtime Server

This document describes all REST and WebSocket endpoints provided by the EmbeddingGemma realtime server (`src/embeddinggemma/realtime/server.py`).

## Base URL
- Default: `http://localhost:8011`
- Development: `uvicorn src.embeddinggemma.realtime.server:app --reload --port 8011`

## WebSocket Endpoints

### `/ws` - Main WebSocket Connection
Real-time bidirectional communication for live updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8011/ws');
```

**Client → Server Messages:**
```json
{
  "type": "config",
  "viz_dims": 2
}
```

**Server → Client Message Types:**
- `hello`: Connection acknowledgment
- `log`: Status/debug messages  
- `snapshot`: Visualization data (agents, documents, trails)
- `metrics`: Simulation metrics (step, avg_rel, trails count)
- `results`: Top-K search results for current step
- `results_stable`: Results when simulation reaches stability
- `report`: Background LLM analysis reports
- `job_progress`: Background job status updates

**Example Messages:**
```json
// Visualization snapshot
{
  "type": "snapshot",
  "step": 42,
  "data": {
    "agents": [[x, y], ...],
    "documents": [[x, y], ...],
    "edges": [[i, j, strength], ...],
    "method": "pca"
  }
}

// Simulation metrics
{
  "type": "metrics", 
  "data": {
    "step": 42,
    "docs": 150,
    "agents": 200,
    "avg_rel": 0.75,
    "max_rel": 0.95,
    "trails": 45
  }
}

// Search results
{
  "type": "results",
  "step": 42,
  "data": [
    {
      "content": "code snippet...",
      "metadata": {"file": "src/main.py"},
      "relevance_score": 0.89
    }
  ]
}
```

## REST Endpoints

### Simulation Control

#### `POST /start`
Start or restart the simulation with configuration.

**Request Body:**
```json
{
  "query": "Find authentication code",
  "num_agents": 300,
  "max_iterations": 150,
  "exploration_bonus": 0.1,
  "pheromone_decay": 0.95,
  "windows": [1000, 2000, 4000],
  "use_repo": true,
  "root_folder": "/path/to/code",
  "max_files": 500,
  "exclude_dirs": [".git", "__pycache__"],
  "viz_dims": 2,
  "redraw_every": 2,
  "min_trail_strength": 0.05,
  "max_edges": 600,
  "embed_batch_size": 128,
  "top_k": 10,
  "report_enabled": false,
  "report_every": 5,
  "report_mode": "deep"
}
```

**Response:**
```json
{
  "status": "ok"
}
```

#### `POST /stop`
Stop the current simulation.

**Response:**
```json
{
  "status": "stopped"  
}
```

#### `POST /reset`
Reset simulation state, clear all data.

**Response:**
```json
{
  "status": "reset"
}
```

#### `POST /pause`
Pause simulation, save agent state.

**Response:**
```json
{
  "status": "paused"
}
```

#### `POST /resume`  
Resume paused simulation, restore agent state.

**Response:**
```json
{
  "status": "resumed"
}
```

### Configuration

#### `POST /config`
Update simulation parameters without restart.

**Request Body:** (Same as `/start`, any subset of parameters)
```json
{
  "viz_dims": 3,
  "redraw_every": 1,
  "top_k": 15
}
```

**Response:**
```json
{
  "status": "ok"
}
```

#### `GET /settings`
Get current configuration and parameter usage.

**Response:**
```json
{
  "settings": {
    "query": "Find authentication code", 
    "num_agents": 300,
    "viz_dims": 2,
    // ... all current settings
  },
  "usage": [
    "query: Find authentication code -> Scripts: mcmp_rag.py (initialize_simulation/search), realtime/server.py (/start)",
    "viz_dims: 2 -> Scripts: mcmp_rag.py (get_visualization_snapshot), frontend (Plotly 2D/3D)"
    // ... usage info for all parameters
  ]
}
```

#### `POST /settings`
Bulk update settings.

**Request Body:**
```json
{
  "num_agents": 400,
  "exploration_bonus": 0.2,
  "report_enabled": true
}
```

**Response:**
```json
{
  "status": "ok",
  "settings": { /* updated settings */ },
  "usage": [ /* parameter usage info */ ]
}
```

### Agent Management

#### `POST /agents/add`
Add N new agents to running simulation.

**Request Body:**
```json
{
  "n": 50
}
```

**Response:**
```json
{
  "status": "ok",
  "added": 50,
  "agents": 350
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "retriever not started"
}
```

#### `POST /agents/resize`
Resize agent population to target count.

**Request Body:**
```json
{
  "count": 500
}
```

**Response:**
```json
{
  "status": "ok", 
  "agents": 500
}
```

### Search & Results

#### `POST /search`
Perform search against current corpus.

**Request Body:**
```json
{
  "query": "authentication implementation",
  "top_k": 5
}
```

**Response:**
```json
{
  "status": "ok",
  "results": [
    {
      "content": "def authenticate(username, password):\n    ...",
      "metadata": {
        "file": "src/auth.py",
        "lines": "15-25",
        "window": 1000
      },
      "relevance_score": 0.92
    }
  ]
}
```

#### `POST /answer`
Search and generate LLM answer.

**Request Body:**
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
  "answer": "Based on the code, authentication works by...",
  "results": [ /* search results used for context */ ]
}
```

### Document Access

#### `GET /doc/{doc_id}`
Get detailed information about a specific document.

**Response:**
```json
{
  "status": "ok",
  "doc": {
    "id": 42,
    "content": "def main():\n    ...",
    "embedding": [0.1, 0.2, ...],
    "relevance_score": 0.87,
    "visit_count": 15,
    "metadata": {
      "file": "src/main.py",
      "lines": "1-50"
    }
  }
}
```

### Corpus Management

#### `GET /corpus/list`
List files in the corpus.

**Query Parameters:**
- `root`: Custom root directory (optional)
- `page`: Page number (default: 1)
- `page_size`: Results per page (default: 200, max: 2000)
- `exclude`: Comma-separated directories to exclude

**Example:** `GET /corpus/list?page=1&page_size=50&exclude=.git,__pycache__`

**Response:**
```json
{
  "root": "src",
  "total": 1247,
  "page": 1,
  "page_size": 50,
  "files": [
    "src/main.py",
    "src/auth.py",
    "src/models/user.py"
  ]
}
```

### Background Jobs

#### `POST /jobs/start`
Start background processing job with sharding.

**Request Body:**
```json
{
  "query": "Find database connections"
}
```

**Response:**
```json
{
  "status": "ok",
  "job_id": "1"
}
```

#### `GET /jobs/status`
Check background job status.

**Query Parameters:**
- `job_id`: Job identifier

**Example:** `GET /jobs/status?job_id=1`

**Response:**
```json
{
  "status": "ok",
  "job": {
    "status": "running",
    "progress": 65,
    "message": "Processed shard 3/5"
  }
}
```

**Completed Job:**
```json
{
  "status": "ok", 
  "job": {
    "status": "done",
    "progress": 100,
    "results": [
      { /* aggregated search results */ }
    ]
  }
}
```

### System Status

#### `GET /status`
Get system status and metrics.

**Response:**
```json
{
  "running": true,
  "docs": 1247,
  "agents": 300,
  "metrics": {
    "step": 95,
    "avg_rel": 0.78,
    "max_rel": 0.94,
    "trails": 67
  }
}
```

#### `GET /`
Landing page with links to client interface.

**Response:** HTML page with navigation links.

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (unknown job, document)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "status": "error", 
  "message": "Description of the error"
}
```

## CORS Configuration

The server accepts requests from:
- `http://localhost:5173` (Vite frontend)
- `http://127.0.0.1:5173`
- `http://localhost:8011`
- `http://127.0.0.1:8011`
- `*` (wildcard for development)

## Settings Persistence

Settings are automatically saved to `.fungus_cache/settings.json` on:
- `/start` calls
- `/config` calls  
- `/settings` POST calls

Settings are loaded from disk on server startup.

## Example Client Integration

### JavaScript WebSocket Client
```javascript
class MCPMClient {
  constructor(url = 'ws://localhost:8011/ws') {
    this.ws = new WebSocket(url);
    this.setupHandlers();
  }
  
  setupHandlers() {
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'snapshot':
          this.updateVisualization(data.data);
          break;
        case 'metrics':
          this.updateMetrics(data.data);
          break;
        case 'results':
          this.updateResults(data.data);
          break;
      }
    };
  }
  
  async start(config) {
    const response = await fetch('http://localhost:8011/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return response.json();
  }
  
  updateVizDims(dims) {
    this.ws.send(JSON.stringify({
      type: 'config',
      viz_dims: dims
    }));
  }
}
```

### Python Client
```python
import aiohttp
import json

class MCPMClient:
    def __init__(self, base_url="http://localhost:8011"):
        self.base_url = base_url
    
    async def start_simulation(self, config):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/start", 
                                   json=config) as resp:
                return await resp.json()
    
    async def search(self, query, top_k=5):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/search",
                                   json={"query": query, "top_k": top_k}) as resp:
                return await resp.json()
```