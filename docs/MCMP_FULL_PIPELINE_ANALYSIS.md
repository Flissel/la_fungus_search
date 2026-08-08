# MCMP Full Pipeline Analysis

## Scope and evidence standard

This is a source-only analysis of the checkout at `22011ea`. It describes
implemented control flow, not intended architecture or marketing claims. No
server, embedding backend, LLM, MCP transport, or benchmark was run for this
document.

The repository contains several retrieval entry points. They must not be
collapsed into one pipeline:

1. `MCPMRetriever.search()` / `search_direct()` is direct vector retrieval.
2. `fungus_search_deep()` and the realtime loop run the MCMP simulation.
3. `fungus_search_expanded()`, `fungus_search_multi()`, and the realtime query
   pool perform different forms of multi-query aggregation.
4. `fungus_search_multivec()` is a separate ColBERT-Lite-style phrase/view
   scoring path.

## Executive finding

The current source does **not** implement a shared multi-query MCMP colony.
MCMP is initialized with one query embedding per simulation run. Multi-query
paths repeatedly invoke direct retrieval and merge result records afterwards.
The realtime path can retain a query pool and a single retriever's document
state, but its extra queries do not get their own agent trajectories and do
not drive the simulation state.

Therefore the following are unverified hypotheses, not current behavior:

- multiple query trajectories collectively exploring the same MCMP state;
- shared pheromone or relevance updates attributable to individual queries;
- candidate-region reuse that avoids repeated ANN work.

## Implemented data structures

`src/embeddinggemma/mcmp_rag.py` defines the state used by the simulation:

- `Document`: embedding, `relevance_score`, `visit_count`, and
  `last_visited`;
- `Agent`: a point and velocity in embedding space, an energy/trail weight,
  and a set of visited document ids;
- one `MCPMRetriever`: documents, agents, `pheromone_trails`, the current
  query embedding, and one FAISS index.

`pheromone_trails` is a dictionary keyed by an undirected pair of document
ids. It is global to that retriever instance, rather than keyed by query or
agent colony. Document relevance is likewise one mutable score per document.

## Path A: direct vector retrieval

```text
query text
  -> embedding backend
  -> L2-normalize query
  -> FAISS inner-product search (or brute-force cosine fallback)
  -> top-k documents
```

`build_faiss_index()` in `src/embeddinggemma/mcmp/indexing.py` normalizes
document vectors and builds `Flat` for fewer than 4096 documents, otherwise
`IVF4096,Flat`, both with `METRIC_INNER_PRODUCT`. `faiss_search()` normalizes
the query and returns FAISS's raw scores.

`MCPMRetriever.search()` delegates directly to `search_direct()`. It does not
call `initialize_simulation()` or `step()`. Consequently, normal calls from
the quick UI runner and the ordinary MCP search tool are single-query ANN
retrieval, not MCMP.

### Metric-semantic observation

`search_direct()` correctly treats the normalized inner product as cosine-like
similarity. In contrast, `find_nearest_documents()` labels the same returned
value as an FAISS L2 distance and computes `1 - dist / 2`. That conversion is
inconsistent with the index construction and changes the neighbour weights
seen by MCMP agents. This is a code-level finding only; it needs a focused
regression test and a separately reported fix before any benchmark conclusion
uses the MCMP results.

## Path B: one-query MCMP / deep search

```text
one query
  -> embed query
  -> spawn N noisy agents around that query vector
  -> repeat steps:
       nearest documents -> attraction + pheromone + noise
       move each agent
       increment document visits and deposit document-pair trails
       recompute every document relevance against the same query
       decay/prune trails
  -> rank documents by current relevance_score
```

### Initialization

`MCPMRetriever.initialize_simulation(query)` embeds exactly one query, stores
it as `_current_query_embedding`, spawns agents around it, clears trails, and
resets every document's visits, timestamps, and relevance. `spawn_agents()`
uses Gaussian position noise (standard deviation `0.1`) and random initial
velocity; each agent's exploration factor is sampled between `0.05` and the
configured exploration bonus.

### Step behavior

For every agent, `update_agent_position()` retrieves five nearest documents,
builds tangential attraction from their normalized embeddings and current
relevance, adds one best outgoing pheromone direction from the agent's nearest
document, then adds random noise. The current source weights the forces as
`0.8` attraction, `0.15` pheromone, and `0.05` exploration; velocity is
updated with `0.85` retention and `0.15` new force.

`deposit_pheromones()` marks the agent's nearest document as visited and adds
trail mass between it and up to three document ids in that agent's visited-id
set. `decay_pheromones()` multiplies all trail values by `pheromone_decay` and
removes values below `0.01`.

After all agents move, `update_document_relevance()` scores *all* documents
against the same current query via GPU torch cosine if available, otherwise
sklearn cosine. It adds visit, recency, and optional keyword bonuses. There is
no candidate-set frontier or region abstraction in this loop; repeated
nearest-document calls use the one global corpus/index.

### `fungus_search_deep()` specifics

The MCP deep tool first runs a hybrid direct search and writes some seed scores
to documents. It then calls `initialize_simulation()`, which resets document
relevance to zero. The implementation therefore does not retain those seed
scores as simulation state. Agents start around the query embedding, not
around the hybrid hits. After the requested number of steps it re-ranks the
simulated document list with the normal result re-ranker.

This path is sequential inside one background thread. It is not a comparison
of multiple query trajectories and it does not expose the set of documents
discovered beyond a separately captured initial direct-search set.

## Path C: multi-query mechanisms

### LLM-generated UI queries

`generate_multi_queries_from_llm()` asks the configured text generator for up
to ten one-line concrete repository queries. It accepts optional file and
keyword hints. It is a one-shot generation call: no retrieved results are
fed back into this generator, so it is not recursive.

`dedup_multi_queries()` applies text normalization, token Jaccard at the
configured threshold (normally `0.8`), then trigram Jaccard at threshold minus
`0.1`. It does not embed queries and does not measure geometry.

### Explicit MCP multi-search

`fungus_search_multi()` accepts user-delimited queries, limits execution to
five, executes `_sync_search()` once per query in a loop, and retains the first
hit per file key. It is sequential. Its merge identity is file path where a
chunk header is available; it does not fuse scores across duplicate hits.

### LLM-expanded MCP search

`fungus_search_expanded()` asks an LLM for two to six alternatives, parses a
JSON array, and always adds the original query if absent. It has no explicit
semantic or textual deduplication after generation. It calls `_sync_search()`
sequentially for each subquery, keeps the highest score for each content
prefix, then applies the normal reranker using the original query.

The merge is max-score union, not reciprocal-rank fusion, learned fusion, or
candidate-region sharing. `asyncio.to_thread()` moves the whole sequential loop
off the event loop; it does not parallelize individual searches.

### Realtime query pool

The realtime `SnapshotStreamer` initializes MCMP with its single `query` and
runs one MCMP step at a time. Optional LLM reports and judgements can add
keywords and concrete follow-up queries to `_query_pool`. When `mq_enabled` is
set, up to three deduplicated pool queries plus the initial query are sent to
`self.retr.search()`, which is direct vector retrieval. Results are united by
full content and the maximum score retained.

Thus the query pool is result-adaptive and may be recursive over report
windows, but it does not create separate MCMP runs. Its indirect interaction
with MCMP is only that judge actions can mutate the single retriever's document
boosts/relevance while the UI is running; the subsequent extra-query searches
still call direct retrieval.

## Candidate sets, state sharing, and fusion

| Question | Source-supported answer |
| --- | --- |
| Where are queries generated? | LLM one-shot generator in `ui/queries.py`; LLM expansion in `mcp_server.py`; follow-ups from realtime LLM reports/judgements; or supplied directly to `fungus_search_multi()`. |
| How many? | UI generator: 1–10 requested. Expanded MCP: 2–6 alternatives, plus original if absent. Explicit MCP: at most 5 supplied queries. Realtime aggregation: initial plus at most 3 extras. |
| Is an LLM involved? | Yes for generated/expanded/follow-up queries; no for manually supplied multi-search or phrase splitting in multivec. |
| Recursive? | UI/MCP expansion: no. Realtime follow-up pool: potentially across report windows. |
| Query dedup? | UI/realtime helper: lexical token/trigram Jaccard. Explicit MCP: implicit first-file retention. Expanded MCP: content-prefix merge only after retrieval. |
| Independent MCMP state per query? | No multi-query MCMP execution path was found. One simulation has one current query and one state. |
| Are pheromones/relevance/agents shared across queries? | They are shared by all agents inside one simulation. No code assigns them to multiple query trajectories. |
| Sequential or parallel? | Multi-query retrieval loops are sequential. Some whole operations run in a background thread. |
| How are results merged? | Depends on entry point: first file, maximum content-prefix score, or maximum full-content score, followed by optional reranking. |

## Separate multivector path

`fungus_search_multivec()` is not MCMP. It splits a query heuristically into
phrases, embeds the phrases, and uses Sum-of-MaxSim across up to six query
phrase vectors and approximately three chunk views. It may blend those scores
with hybrid retrieval. This is a useful multi-vector baseline, but it neither
uses pheromone state nor shares ANN candidate regions between independent
queries.

## Documentation drift and analysis limits

`docs/mcmp_simulation.md` describes different neighbour counts and motion
weights from `simulation.py` (for example, three vs. five neighbours and
`0.6/0.3/0.1` vs. `0.8/0.15/0.05`). The source is treated as authoritative.

The source contains no built-in captures for the following required Phase 3/4
measurements: pre/post query-dedup counts, query embedding dispersion,
candidate-set overlap, MCMP novel candidates, recall/MRR/NDCG, or comparable
latency/compute counters. They must be added as an isolated benchmark harness;
they cannot be reconstructed reliably from the current logs.

## Consequences for the research hypotheses

1. Current code cannot demonstrate whether collective multi-query MCMP state
   helps, because it does not execute that configuration.
2. It can still supply baselines: single direct FAISS, multi-query direct
   FAISS, and one-query MCMP simulation.
3. Before comparing MCMP to FAISS, correct or isolate the FAISS metric
   interpretation in the agent-neighbour path. Keep that fix separate from
   benchmark conclusions.
4. The next experiment should instrument result/candidate sets and query
   embeddings before designing a shared-region or colony mechanism.
