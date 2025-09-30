## MCMP Simulation (Formulas, Parameters, Convergence)

This document summarizes the current MCMP simulation behavior implemented in `src/embeddinggemma/mcmp/simulation.py`, used by `MCPMRetriever`.

### State
- **Documents** `d_i`
  - `embedding` ∈ R^D
  - `relevance_score` r_i ∈ R
  - `visit_count` v_i ∈ N
  - `last_visited` timestamp
- **Agents** `a`
  - `position` p_t ∈ R^D (re-normalized to unit norm each step)
  - `velocity` v_t ∈ R^D
  - `exploration_factor` e ∈ (0, 1) (sampled ∈ [0.05, exploration_bonus])
  - `energy` E (default 1.0)
  - `trail_strength` T_s (default 1.0)
  - `visited_docs` (set of doc ids)
- **Pheromone trails**
  - Undirected keys (i, j) with strength s_{ij} ≥ 0 (pruned at < 0.01)

### Per-step Agent Update
Let k = 3 nearest neighbor documents by cosine similarity around p_t.

- Attraction force F_att
  - For each neighbor doc with embedding x and relevance r:
    - dir = x − p_t, dist = ||dir||
    - sim = cosine(normalize(x), normalize(p_t))
    - f_doc = (dir / dist) × sim × (1 + r)
  - F_att = Σ f_doc

- Pheromone force F_pher
  - Find current nearest doc c to p_t. Among trails (c → j) with strengths s_{cj}, pick the max; direction points to doc j:
    - F_pher = normalize(x_j − p_t) × s_{cj} (0 if none)

- Exploration force F_expl
  - Gaussian noise per dimension: N(0, e^2)

- Total force and motion
  - F = 0.6 F_att + 0.3 F_pher + 0.1 F_expl
  - v_{t+1} = 0.8 v_t + 0.2 F
  - p_{t+1}′ = p_t + v_{t+1}
  - p_{t+1} = p_{t+1}′ / ||p_{t+1}′|| (unit normalization)

### Pheromone Mechanics
- Deposit (on visiting nearest doc c)
  - Increment `visit_count` and update `last_visited`
  - Add `c` to `visited_docs`
  - For the last up to 3 previously visited docs `j ≠ c`:
    - s_{cj} ← s_{cj} + E × T_s × 0.1

- Decay & prune (each step)
  - s_{ij} ← pheromone_decay × s_{ij}
  - If s_{ij} < 0.01: delete trail

### Relevance Computation (per document)
- Base similarity r_sim:
  - If GPU torch available: cosine(normalize(q), normalize(x_i)) via cached half-precision tensors
  - Else: sklearn cosine similarity
- Bonuses:
  - visit_bonus = min(0.1 × visit_count_i, 0.5)
  - time_bonus = 0.1 if (now − last_visited_i) < 1.0 s else 0
  - kw_bonus = kw_lambda × (hits_i / max(1, |kw_terms|)), where `hits_i` is number of query keywords in doc content
- Final score:
  - r_i = r_sim + visit_bonus + time_bonus + kw_bonus

### Metrics (reported)
- avg_rel = (1/N) Σ_i r_i
- max_rel = max_i r_i
- trails = |{(i, j)}|
- avg_speed = mean over agents of ||v_t||

### Convergence Heuristics (suggested)
Evaluate over a sliding window of W steps (e.g., W = 20):

- Trail stagnation (no new exploration)
  - Use threshold `s_min` = 0.05 to define present edges E_t
  - No-new-trails: |E_t \ E_{t−W}| = 0
  - Low motion: avg_speed_t < v_eps (e.g., 0.05)
  - Optional: total-trail-mass stability |S_t − S_{t−W}| / max(S_{t−W}, ε) < m_eps (e.g., 0.02)
  - Converged if all hold

- Average relevance band (bounded oscillation)
  - Band [L, U] = [0.7, 0.9]
  - Bounded: L ≤ A_k ≤ U ∀ k ∈ [t−W+1, t]
  - Small amplitude: max(A) − min(A) ≤ Δ (e.g., 0.05)
  - Low drift: |A_t − A_{t−W}| ≤ δ (e.g., 0.02)
  - Converged if all hold

Stop when convergence holds for one full window, or at `max_iterations`.

### Parameters (current defaults)

#### Core Simulation
- **num_agents**: 200 (range: 1-10000)
  - Number of agents exploring the embedding space
  - More agents = better coverage, higher CPU usage

- **max_iterations**: 200 (range: 1-5000)  
  - Maximum simulation steps before auto-stop
  - Higher values = more thorough exploration

- **exploration_bonus**: 0.1 (range: 0.01-1.0)
  - Upper bound for agent noise standard deviation
  - Higher values = more random exploration vs exploitation

- **pheromone_decay**: 0.95 (range: 0.5-0.999)
  - Trail strength decay rate per step
  - Higher values = longer trail persistence

#### Search & Results
- **top_k**: 10 (range: 1-200)
  - Number of top results to return
  - Used for both intermediate and final results

- **kw_lambda**: 0.0 (keyword weighting)
- **kw_terms**: ∅ (empty set, keyword filtering)

#### Visualization & Updates  
- **redraw_every**: 2 (range: 1-100)
  - Steps between visualization updates (WebSocket/live UI)
  - Higher values = less frequent updates, better performance

- **min_trail_strength**: 0.05 (range: 0.0-1.0)
  - Minimum trail strength to display in visualization
  - Higher values = show only strong connections

- **max_edges**: 600 (range: 10-5000)
  - Maximum edges displayed in network visualization
  - Higher values = more detail, slower rendering

- **viz_dims**: 2 or 3
  - Visualization dimensionality (2D/3D)
  - 2D = faster, 3D = more immersive

#### Corpus Building
- **windows**: Required from frontend (e.g., [1000, 2000, 4000])
  - Chunk window sizes in lines for multi-scale chunking
  - Creates overlapping chunks at different granularities

- **max_files**: 500 (range: 0-20000)
  - Maximum files to process from corpus
  - 0 = no limit

- **exclude_dirs**: [".venv", "node_modules", ".git", "external"]
  - Directories to skip during corpus building

- **chunk_workers**: max(4, cpu_count) (range: 1-128)
  - CPU threads for parallel document chunking

- **embed_batch_size**: 128 (range: 1-4096)
  - Batch size for embedding computation
  - Adjust based on GPU memory

#### Background Processing
- **max_chunks_per_shard**: 2000 (range: 0-100000)
  - Chunks per shard in background jobs
  - 0 = no sharding, single job
  - Used in `/jobs/start` endpoint

#### Reporting (Realtime Server)
- **report_enabled**: false
  - Enable background LLM analysis reports

- **report_every**: 5 (range: 1-100)
  - Steps between background reports
  - Higher values = less frequent reports

- **report_mode**: "deep" 
  - Values: "deep", "structure", "exploratory", "summary", "repair"
  - Controls LLM analysis style

## API Integration

### Realtime Server Endpoints
The simulation can be controlled via REST API (see `docs/API_REFERENCE.md`):

#### Simulation Control
- `POST /start`: Initialize simulation with parameters
- `POST /stop`: Stop current simulation
- `POST /reset`: Reset state, clear all data
- `POST /pause`: Pause simulation, save agent state  
- `POST /resume`: Resume with restored agent state

#### Live Updates (WebSocket `/ws`)
- `snapshot`: Visualization data every `redraw_every` steps
- `metrics`: Step count, avg/max relevance, trail count
- `results`: Top-K results for current step
- `results_stable`: Results when stability detected
- `report`: Background LLM analysis (if enabled)

#### Agent Manipulation
- `POST /agents/add`: Add N new agents during simulation
- `POST /agents/resize`: Resize agent population to target count

#### Configuration  
- `POST /config`: Update parameters without restart
- `GET /settings`: Get current configuration + usage info
- `POST /settings`: Bulk settings update

### Parameter Sources
Settings can come from:
1. **Startup defaults**: Hard-coded in server
2. **Disk persistence**: `.fungus_cache/settings.json`
3. **API calls**: `/start`, `/config`, `/settings` endpoints
4. **WebSocket**: Small config updates (e.g., viz_dims)

### Integration Examples
```python
# Start simulation with custom parameters
config = {
    "query": "Find database connections",
    "num_agents": 300,
    "max_iterations": 150, 
    "exploration_bonus": 0.15,
    "pheromone_decay": 0.97,
    "top_k": 15,
    "report_enabled": True,
    "report_every": 10
}
response = await session.post("/start", json=config)
```

```javascript
// Live visualization updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'snapshot') {
        updateVisualization(data.data.agents, data.data.documents, data.data.edges);
    }
    if (data.type === 'metrics') {
        updateMetrics(data.data.step, data.data.avg_rel, data.data.trails);
    }
};
```

## Implementation Notes
- k for agent attraction is fixed to 3 nearest neighbors
- Agent positions are normalized each step to stay on the embedding hypersphere
- Pheromone deposit scale uses constants: E × T_s × 0.1
- Background reports use async LLM calls to avoid blocking simulation loop
- WebSocket broadcasts are non-blocking; failed clients are automatically removed
- Settings persistence occurs automatically on API configuration changes
