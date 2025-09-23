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
- Simulation
  - num_agents: 200
  - max_iterations: 200 (backend loop)
  - exploration_bonus: 0.1 (upper bound for agent noise std)
  - pheromone_decay: 0.95 (API constraint: 0.5 ≤ value ≤ 0.999)
- Relevance shaping
  - kw_lambda: 0.0
  - kw_terms: ∅ (empty set)
- Visualization & cadence
  - redraw_every: 2
  - min_trail_strength: 0.05
  - max_edges: 600
  - viz_dims: 2 or 3
- Corpus/build
  - windows: required from frontend (e.g., [1000, 2000, 4000])
  - max_files, exclude_dirs, chunk_workers
  - embed_batch_size: 128

### Notes
- k for agent attraction is fixed to 3.
- Agent positions are normalized each step to stay on the embedding hypersphere.
- Deposit scale uses constants: E × T_s × 0.1.
