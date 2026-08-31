## MCMP Simulation (Formulas, Parameters, Convergence)

This document describes the MCMP simulation implemented in
`src/embeddinggemma/mcmp/simulation.py` and driven by `MCPMRetriever`
(`src/embeddinggemma/mcmp_rag.py`). **The source is authoritative**; where this
document and the code disagree, the code is right and this document is a bug.

> **Corrected 2026-08-31 against the source.** The previous revision had drifted
> on five points, all of them load-bearing: it gave `k = 3` neighbours where the
> code uses 5, force weights `0.6 / 0.3 / 0.1` where the code uses
> `0.8 / 0.15 / 0.05` — stating the pheromone weight at **double** its real value
> — velocity smoothing `0.8 / 0.2` where the code uses `0.85 / 0.15`, a plain
> difference vector for the attraction where the code uses a sphere-tangent
> projection, and a constructor default of 200 agents/200 iterations where the
> code has 200/50. Two behaviours that no revision has ever documented are
> recorded below under **Two asymmetries in the pheromone mechanism**.
>
> For what these formulas *do* — which terms carry retrieval and which are inert
> — see `docs/MCMP_MECHANICS.md`. This document is the mechanism; that one is the
> measured behaviour.

### State

- **Documents** `d_i` (`mcmp_rag.Document`)
  - `id` ∈ N — assigned in insertion order, and load-bearing (see below)
  - `embedding` ∈ R^D
  - `relevance_score` r_i ∈ R
  - `visit_count` v_i ∈ N
  - `last_visited` timestamp
- **Agents** `a` (`mcmp_rag.Agent`)
  - `position` p_t ∈ R^D, re-normalized to unit norm each step
  - `velocity` v_t ∈ R^D
  - `energy` E (default 1.0), `trail_strength` T_s (default 1.0)
  - `exploration_factor` e, sampled uniformly from [0.05, max(0.05, exploration_bonus)]
  - `visited_docs` — a **set** of document ids
  - `age` — incremented each step, not otherwise read
- **Pheromone trails**
  - Keys `tuple(sorted((i, j)))`, strength s_{ij} ≥ 0, pruned below 0.01

Agents are spawned at `query + N(0, 0.1)`, normalized, with velocity `N(0, 0.05)`.

### Per-step agent update

`update_agent_position`, one call per agent per step. Let `p` be the agent's
position normalized to the unit sphere, and let the neighbourhood be the
**k = 5** nearest documents by cosine similarity around `p`.

- **Attraction force F_att** — sphere-tangent, not a plain difference vector.
  For each neighbour with embedding `x` and relevance `r`:
  - `nd = x / ||x||`
  - `t = nd − (p · nd) p` — the radial component is removed, so the force lies in
    the tangent plane at `p` and moves the agent *along* the hypersphere
  - `f = (t / ||t||) × sim × (1 + r)`, where `sim` is the similarity returned by
    the index
  - `F_att = Σ f`

- **Pheromone force F_pher** — `calculate_pheromone_force`. Find the single
  nearest document `c` to `p`. Among trails whose **first** key element equals
  `c.id`, take the strongest, and point at the other endpoint:
  - `F_pher = normalize(x_j − p) × s_{cj}`, or 0 if there is no such trail

- **Exploration force F_expl** — `N(0, e²)` per dimension.

- **Total force and motion**
  - `F = 0.8 F_att + 0.15 F_pher + 0.05 F_expl`
  - `v_{t+1} = 0.85 v_t + 0.15 F`
  - `p_{t+1}′ = p + v_{t+1}` — note `p`, the *normalized* position, not the raw one
  - `p_{t+1} = p_{t+1}′ / ||p_{t+1}′||`

### Pheromone mechanics

- **Deposit** — `deposit_pheromones`, once per agent per step, on the nearest
  document `c`:
  - `c.visit_count += 1`, `c.last_visited = now`
  - `c.id` is added to `agent.visited_docs`
  - for each `j` in `list(agent.visited_docs)[-3:]` with `j ≠ c.id`:
    `s_{cj} ← s_{cj} + E × T_s × 0.1` under the key `tuple(sorted((c.id, j)))`

- **Decay and prune** — each step: `s_{ij} ← pheromone_decay × s_{ij}`, delete
  below 0.01.

#### Two asymmetries in the pheromone mechanism

Both are properties of the shipped code, both are verified by experiment in
`benchmarks/probes/visit_term.py --experiment defects`, and neither is
documented anywhere else.

1. **A trail is followable from one endpoint only.** Deposit stores the key
   sorted, so a trail between documents 3 and 91 is stored as `(3, 91)`. Following
   matches `doc_a == current_doc.id`, i.e. only the **lower** id. An agent standing
   on document 3 feels the trail; an agent standing on document 91 feels nothing.
   Since ids are assigned in insertion order, which half of the deposited signal is
   usable is decided by corpus load order.

2. **The "last three visited documents" are not the last three.**
   `agent.visited_docs` is a `set`, so `list(...)[-3:]` returns three entries in
   hash order — for small ints, roughly value order — not insertion order. For a
   visit sequence `7, 3, 91, 12, 5, 44, 2` the code deposits against
   `[12, 44, 91]`, while the genuinely most recent three are `[5, 44, 2]`. Trails
   therefore connect documents that are close in *id*, not close in *time*.

Repairing either one, or both, was measured and **reduces** retrieval quality at
every agent count tested. See `docs/MCMP_MECHANICS.md` for why.

### Relevance computation (per document)

`update_document_relevance`, once per step, over **every document in the working
set** — there is no candidate restriction.

- Base similarity `r_sim`: cosine against the query. Half-precision CUDA path when
  torch sees a GPU and `force_cpu` is unset, otherwise sklearn on the CPU.
- Bonuses:
  - `visit_bonus = min(0.1 × v_i, 0.5)` — **saturates at five visits**
  - `time_bonus = 0.1` if `now − last_visited_i < 1.0` s, else 0
  - `kw_bonus = kw_lambda × hits_i / max(1, |kw_terms|)`, where `hits_i` counts
    query keywords occurring in the document text
- `r_i = r_sim + visit_bonus + time_bonus + kw_bonus`

The relevance score feeds back into `F_att` through the `(1 + r)` weight, so the
visit count steers the walk as well as the ranking.

### Step loop

`MCPMRetriever.step(n)` repeats, `n` times:

1. for each agent: `update_agent_position`, then `deposit_pheromones`
2. `update_document_relevance` — the whole working set
3. `decay_pheromones`

There is no candidate-set frontier and no region abstraction: every
`find_nearest_documents` call goes to the one global index. The benchmark
harness's method G adds a bounded frontier on top of this loop without changing
it (`benchmarks/mcmp/adapters.py`).

### Metrics reported

`log_simulation_step`: `avg_rel`, `max_rel`, trail count, and mean agent speed.

### Convergence heuristics (suggested, not implemented in `simulation.py`)

Evaluated over a sliding window of W steps (e.g. W = 20):

- **Trail stagnation** — present edges `E_t` at strength ≥ `s_min` = 0.05;
  converged when `|E_t \ E_{t−W}| = 0`, `avg_speed_t < v_eps` (e.g. 0.05), and
  optionally total trail mass stable within `m_eps` (e.g. 0.02).
- **Average-relevance band** — converged when `A_k ∈ [0.7, 0.9]` for the whole
  window, `max(A) − min(A) ≤ 0.05`, and `|A_t − A_{t−W}| ≤ 0.02`.

Stop when convergence holds for one full window, or at `max_iterations`.

### Parameters

`MCPMRetriever.__init__` defaults:

| parameter | default |
|---|---|
| `num_agents` | 200 |
| `max_iterations` | 50 |
| `pheromone_decay` | 0.95 (API range 0.5–0.999) |
| `exploration_bonus` | 0.1 (upper bound of the agent noise std) |

Other defaults come from the caller, not the retriever: the realtime backend
(`realtime/server.py`) sets `max_iterations = 200`, the Streamlit UI defaults to
60, and `ui/mcmp_runner.py` to 20. `kw_lambda` is 0.0 and `kw_terms` empty unless
a caller sets them, so the keyword bonus is normally inactive.

Visualization and corpus knobs (`redraw_every` 2, `min_trail_strength` 0.05,
`max_edges` 600, `viz_dims` 2 or 3, `embed_batch_size` 128, `windows`,
`max_files`, `exclude_dirs`, `chunk_workers`) are UI/build concerns and do not
enter the simulation above.

### Notes

- `find_nearest_documents` defaults to `k = 3`, but `update_agent_position` passes
  `k = 5` explicitly and `calculate_pheromone_force`/`deposit_pheromones` pass
  `k = 1`. The default is never the value the walk uses.
- Agent positions are normalized every step, so the walk stays on the embedding
  hypersphere.
- Deposit scale is the constant `E × T_s × 0.1`, and with default energy and trail
  strength that is a flat 0.1 per deposit.
