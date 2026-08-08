# MCMP Ablation Harness Design

## Purpose

Build a deterministic, offline-first experiment that answers one narrow
question before any TIG C004 implementation:

> Does the current MCMP simulation discover relevant candidates beyond the
> initial FAISS result set, and is that effect stronger when related queries
> are used?

The harness separates algorithm behavior from LLM generation, RAG synthesis,
MCP transport, OpenFang availability, and production corpus construction.

## Scope

The first delivery has two sequential gates.

### Gate 1: synthetic vector experiment

Run the four required ablations over deterministic vector fixtures with known
relevance labels:

- A: Single Query + FAISS
- B: Multi Query + FAISS
- C: Single Query + MCMP
- D: Multi Query + independent MCMP runs

The synthetic fixtures contain several vector clusters, bridge candidates,
near distractors, and relevant targets outside the smallest initial FAISS
set. Fixed seeds control fixture generation and MCMP randomness.

Gate 1 tests whether the current simulation can produce novel relevant
candidates under any controlled geometry. It does not claim production search
quality.

### Gate 2: labelled Fungus retrieval experiment

After Gate 1 is reproducible, run the same A-D contract over a small,
versioned set of real code-search queries and relevance judgements. Corpus and
query embeddings are materialized once through an explicitly selected backend
and then cached locally. Repeated ablations consume the identical embedding
snapshot without requiring a live provider.

Gate 2 tests whether an observed synthetic effect transfers to Fungus code
retrieval. It remains separate from TIG C004.

## Non-goals

This design does not:

- start or modify the Fungus MCP server;
- invoke an LLM or generate RAG answers;
- implement query colonies or shared pheromone state;
- implement TIG C004 algorithms;
- claim RAG, multi-hop, production, or competitive-search improvement;
- change the production MCMP algorithm to make a benchmark pass.

## Architecture

The harness is split into four units with explicit data contracts.

### Fixture provider

The provider emits a `BenchmarkDataset` containing:

- stable dataset id and seed;
- normalized document vectors and document ids;
- one initial query and zero or more related query vectors;
- relevant document ids per logical query;
- fixture metadata describing clusters and bridge targets.

Synthetic data is generated from source-controlled parameters rather than
committing opaque binary arrays. The real-data provider reads a local cached
embedding snapshot plus a source-controlled judgement manifest.

### Retrieval adapters

Adapters expose one result contract for FAISS and MCMP:

```text
SearchRun
  method
  query_ids
  ranked_document_ids
  initial_candidate_ids
  discovered_candidate_ids
  per_query_candidate_ids
  elapsed_ms
  candidate_comparisons
  mcmp_steps
  document_visits
  pheromone_trails
```

The FAISS adapter calls the existing inner-product index with normalized
vectors. The MCMP adapter uses the existing simulation implementation through
a local in-memory embedding backend. The adapter records state before and
after simulation; it does not alter MCMP scoring.

For Gate 1, D consists of independent MCMP runs whose results are fused after
retrieval. This represents current product behavior honestly. Shared-colony
state is reserved for a later experiment and must not be smuggled into the
baseline.

### Metrics evaluator

The evaluator consumes only `BenchmarkDataset` and `SearchRun`. It computes:

- Recall@K;
- reciprocal rank and MRR across a suite;
- NDCG@K;
- unique relevant documents;
- candidate count and candidates per query;
- candidate overlap between every query pair;
- query-vector pairwise cosine distance, mean dispersion, and maximum
  dispersion;
- MCMP novel candidates;
- novel relevant candidates;
- elapsed time and MCMP exploration counters.

The central set definitions are:

```text
MCMP_novel = MCMP_discovered_set - FAISS_initial_set

novel_relevant = MCMP_novel intersect relevant_document_ids
```

Metrics are computed from literal relevance labels, never from MCMP or FAISS
scores treated as ground truth.

### Result writer

Each run produces a JSON document under `benchmarks/results/`. It includes the
dataset id, seed, method configuration, environment metadata, raw candidate
ids, and derived metrics. A concise Markdown summary may be generated from
the JSON, but JSON remains authoritative.

Results from different corpus or embedding snapshot digests cannot be merged
into one comparison.

## Data flow

```text
source-controlled fixture parameters or judgement manifest
  -> BenchmarkDataset
  -> A/B/C/D retrieval adapters
  -> SearchRun records
  -> metrics evaluator
  -> machine-readable JSON
  -> explicit Gate 1 or Gate 2 conclusion
```

Multi-query generation is outside the first harness. Related queries are
explicit fixture inputs so geometry and overlap are controlled. A later
production experiment can capture LLM-generated queries and replay them from
a fixed manifest.

## Reproducibility

Every run fixes and records:

- Python random seed;
- NumPy random seed;
- dataset version and digest;
- document/query counts and dimensions;
- FAISS factory and metric;
- top-k and initial candidate-pool size;
- MCMP agent count, step count, exploration bonus, and pheromone decay;
- package versions and CPU/GPU mode.

Synthetic tests use CPU FAISS. GPU acceleration is not required to establish
algorithm behavior and must be benchmarked separately if introduced.

## Failure handling

The harness fails closed when:

- vectors have inconsistent dimensions or are not finite;
- ids or relevance labels refer to missing documents;
- normalized-inner-product assumptions are violated;
- a cached real embedding snapshot has a mismatched dataset digest;
- a method returns duplicate or out-of-range document ids;
- a requested metric cannot be computed from the recorded data.

An unavailable real embedding backend blocks snapshot creation only. It does
not block Gate 1 or replay of an existing valid Gate 2 snapshot.

## Test strategy

Implementation follows RED-GREEN cycles.

1. Contract tests reject malformed datasets and search records.
2. Literal vector fixtures verify FAISS ranking and candidate capture.
3. Metric tests use hand-derived rankings and relevance labels for Recall,
   RR, NDCG, overlap, dispersion, and novel relevant candidates.
4. A deterministic MCMP integration test runs the real simulation with a
   local embedding backend and confirms identical output for the same seed.
5. An A-D orchestration test verifies method isolation and result labelling.
6. A JSON round-trip test verifies that every comparison input is persisted.

The existing MCMP suite remains a regression gate. Tests must not assert
source text or mock behavior in place of observable rankings and metrics.

## Decision gates

### Gate 1 conclusion

Proceed to Gate 2 when the harness is deterministic and all A-D runs produce
complete, comparable evidence. Gate 1 may legitimately conclude that MCMP
adds no novel relevant candidates.

### Gate 2 conclusion

Proceed to shared-region/query-colony design only if the labelled Fungus
results demonstrate at least one reproducible benefit attributable to MCMP or
multi-query interaction, such as additional relevant candidates or improved
Recall/NDCG under explicitly reported extra compute.

If Multi Query + FAISS explains the benefit and MCMP adds none, retain
Multi-Query as a retrieval layer and stop MCMP index integration work. If
neither improves the labelled baseline, stop this branch of the research
rather than forcing a positive result.

## Deliverables

The implementation plan will assign exact paths, but the intended ownership
is:

- benchmark contracts and fixture generation under `benchmarks/mcmp/`;
- source-controlled manifests under `benchmarks/fixtures/`;
- machine-readable outputs under `benchmarks/results/`;
- focused tests under `tests/benchmarks/`;
- later findings in `docs/MCMP_TIG_C004_REPORT.md` only after evidence exists.

The current production retriever changes only when a separately failing
regression test establishes a product defect. Benchmark instrumentation stays
outside `MCPMRetriever` unless an observable, reusable hook is required.
