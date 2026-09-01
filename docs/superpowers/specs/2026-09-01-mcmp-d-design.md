# MCMP-D: the colony on the graph the gate measures

## Why this exists

Report sections 17-25 left one live contradiction. Stage 1 proves the structure
exists: far relevant documents are reachable in the mutual k-NN graph at 0.44-0.62
against a permutation null of 0.03-0.14, over paths of 3-4 hops (§22). The
continuous walk cannot traverse it: the documents FAISS misses sit at median rank
200 and MCMP ranks 0 of 8 of them better (§25). The force law explains why — 80%
attraction anchors agents to the query's basin, 5% exploration cannot carry an
agent four hops, and the pheromone term only follows trails the colony has already
laid (§11, §14). The simulation searches continuous space; the structure lives in
the discrete graph. **MCMP-D moves the colony onto the graph.**

## Design

- **Space.** The mutual k-NN graph over the snapshot vectors, `knn_k = 8` (the
  operating point every real-data selection chose), precomputed once. Edge weight
  = cosine similarity, clipped at 0.
- **Agents.** Each agent occupies a node. Start distribution: the FAISS
  `initial_k` pool, weighted by query similarity. Per step an agent restarts to
  the start distribution with probability `alpha = 0.15`, else hops to a
  neighbour with probability proportional to `weight x (1 + pheromone)^beta`.
- **Pheromone.** On edges, symmetric — which incidentally makes both §14.3
  defects unrepresentable. Deposit 0.1 per traversal, decay 0.95 per step.
- **Ranking.** Visit counts. Section 14 proved the visit distribution is the only
  signal that ever carried MCMP's ranking; here it is the ranking, not a bonus
  term fighting a similarity score.
- **Cost.** One hop is O(k) against a precomputed graph. 96 agents x 50 steps
  = 4 800 hops per query, against method C's 57.2 million comparisons and G's
  147 761.

## Controls, from day one

MCMP-D as described converges toward personalized PageRank with edge
reinforcement. That proximity is a threat, not a footnote: if the pheromone adds
nothing over PPR, the simpler method wins and the answer is PPR, not MCMP.

1. **`ppr`** — exact personalized PageRank by power iteration on the same graph,
   same restart distribution, same alpha. The deterministic baseline.
2. **`walk`** — the agent walk with pheromone off. A Monte Carlo PPR estimate;
   isolates stochasticity from reinforcement (the §11 method-F pattern).
3. **`colony`** — the full mechanism. Must beat both to justify itself.

## Oracles, and the leakage rule

- **Call-graph oracle** (§17): evaluated on the **pure semantic graph only**.
  Call edges in the walk graph would make targets trivially adjacent to the
  query's neighbourhood — leakage.
- **Sibling oracle** (§20): sibling pairs share callees/callers but have **no
  direct call edge**, so a hybrid graph (k-NN ∪ call edges) contains no edge that
  is itself a label. The hybrid variant is evaluated here and only here.

## Pre-registered success criteria

Measured on brain sample 0, the corpus and seeds of §22-§25, against FAISS (A):

1. **Reach.** The walk visits at least half of the far relevant documents that
   BFS proves reachable at (knn 8, hops 6) — the stage-1 ceiling, computed per
   seed on the same data.
2. **Promotion.** Median rank of far relevant documents improves from ~200 (§25)
   to inside the top 32.
3. **Ranking.** recall@8 at or above FAISS's on the same seeds.
4. **Cost.** At most method G's comparison budget.

Failing 1 kills the design (the graph walk cannot reach what BFS reaches). Failing
2-3 while passing 1 means discovery without ranking again — §24's shape — and the
visit distribution needs work before any further claim. `colony` beating `walk`
and `ppr` is required before attributing anything to the pheromone.

## Non-goals

No change to `src/`. No production wiring. No claim about the 3072-dimensional
production space. The continuous MCMP stays as measured; this is a successor
candidate, not a patch.
