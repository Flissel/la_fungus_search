# MCMP Gate 1 Fixture Redesign Design

Amends `docs/superpowers/specs/2026-08-08-mcmp-ablation-harness-design.md`.
Motivated by `docs/MCMP_TIG_C004_REPORT.md`.

## Purpose

Gate 1 currently cannot measure the retriever. The Task 7 review established that
in all 24 query cases of the existing fixture the relevant documents occupy FAISS
similarity ranks exactly (2, 3) and the most similar document is never relevant.
Every Gate 1 number follows from that single labelling decision: method A's MRR is
forced to 0.500, MCMP's advantage comes from demoting rank 1, and the novelty
conclusion is a step function of `initial_k`.

Two further defects compound it. `initial_k > top_k` is rejected
(`benchmarks/mcmp/adapters.py:293`, `benchmarks/mcmp/run_gate1.py:280`), so
retrieve-then-rerank cannot be run as a control. And `initial_k` does not bound
MCMP at all — `benchmarks/mcmp/adapters.py:190` uses it only for reporting while
`add_documents` receives the entire corpus, so the novelty metric compares a
full-corpus walk against a truncated FAISS list.

This design makes Gate 1 able to distinguish MCMP's behaviour from the fixture's
labelling convention.

## Scope

In scope: two additional fixtures, a fixture registry and runner flag, an
additional retrieval method E, relaxation of the `initial_k` constraint, and an
evidence-schema field naming the fixture.

Out of scope: any change to the production retriever, Gate 2, query colonies,
TIG C004, and method F (equal-budget pheromone-free control), which stays
available as a later additive step.

## Non-goals

- Do not modify `build_synthetic_dataset`, its tests, or the committed evidence
  files. That report carries the standing decision and must remain checkable.
  Note the precise guarantee: because every payload now also carries `runs.E` and
  `dataset.fixture`, a legacy run regenerated with the amended code is **not**
  byte-identical to the committed file. What is preserved is stronger than the
  file bytes and weaker than regeneration: the committed files still replay and
  validate unchanged, and every A-D number the report cites is reproduced exactly,
  because A-D are untouched. Byte-exact regeneration of the original files
  requires checking out the pre-amendment commit. This trade is accepted
  deliberately rather than by adding a conditional that suppresses `runs.E` on the
  legacy fixture.
- Do not change MCMP production behaviour to make any fixture pass.
- Do not remove the `initial_k <= document_count` bound.
- Do not introduce randomness that is not seed-derived.

## Architecture

### Fixture provider

`benchmarks/mcmp/fixtures.py` gains two builders and a registry:

```python
FIXTURES = {
    "legacy":   build_synthetic_dataset,    # unchanged, dataset_id "synthetic-mcmp-v1"
    "neutral":  build_neutral_dataset,      # dataset_id "neutral-mcmp-v1"
    "manifold": build_manifold_dataset,     # dataset_id "manifold-mcmp-v1"
}
```

Both new builders produce 64 documents and 2 queries in 16 dimensions, unit-norm
float32, and validate through the existing `BenchmarkDataset` contract. The seed
varies structure, not only jitter: which documents are relevant, where the chain
lies, and distractor placement all derive from the seed. Seeds are therefore
replicates, not repetitions.

#### Neutral fixture

Purpose: a control in which no method is advantaged by the labelling.

Construction: 64 unit vectors drawn from a seeded RNG. For each query, rank all
documents by similarity, then draw the 4 relevant documents **uniformly at random
from the top 16**.

Relevance therefore correlates with similarity — which is realistic and keeps the
task solvable — while the *rank positions* of the relevant documents vary by seed
instead of being fixed. Drawing relevance uniformly from all 64 was rejected: with
`top_k` around 4 every method would score near zero and the fixture would measure
noise rather than act as a control.

Expected reading: no method should show a systematic advantage. An MCMP advantage
here would indicate a defect in the harness or the metric, not a discovery
capability.

#### Manifold fixture

Purpose: the structure MCMP's pheromone walk is designed to exploit, so that a
positive result is attributable to the mechanism.

Construction: **each of the two queries gets its own chain** of 8 documents,
`c1..c8`, produced by rotating stepwise from that query's vector toward a
seed-chosen orthogonal direction, so that consecutive links are close
(cosine ≈ 0.95) while the far end is distant from the query (cosine ≈ 0.2). The
relevant set per query is the far end of its own chain (`c6..c8`). The two chains
use different orthogonal directions, so neither query's chain is relevant to the
other. That accounts for 16 of the 64 documents; the remaining 48 are distractors
sitting at moderate similarity to both queries (cosine ≈ 0.5), occupying the FAISS
top-k without lying on either chain.

FAISS top-k therefore returns distractors and the near chain links; the relevant
documents are reachable only by traversing the chain. Exact angular constants are
implementation detail, but the invariant is: at least one relevant document ranks
deeper than the default `top_k`, and consecutive chain similarity exceeds
query-to-far-end similarity.

Expected reading: this is where MCMP can legitimately win. If it does not win
here, the mechanism does not work on the structure it was designed for. A win here
is a claim about this structure only; whether real code retrieval has it is
Gate 2's question.

### Retrieval adapters

Methods A–D are unchanged. One method is added:

- **E — Single Query + MCMP restricted to the FAISS pool.** FAISS retrieves
  `initial_k` documents; MCMP receives only those via `add_documents(pool_ids)`.
  Otherwise identical to C: same agent count, steps, decay, exploration bonus,
  fixed clock, seeded RNG.

E exists to separate two effects that C conflates: discovery by walking outside
the pool, and reranking within it. C minus E is the walk's contribution; E minus A
is the reranking contribution.

E is single-query so it compares directly against C. A multi-query pooled variant
is deliberately omitted (YAGNI).

E is orchestrated by `run_gate1` alongside A–D and written into the evidence
payload under `runs.E`. **The `conclusion` rule is unchanged**: it continues to be
derived from the novel-relevant counts of C and D only. E is reported as evidence
for the walk-versus-rerank comparison and deliberately does not influence the
conclusion, so that new and legacy evidence remain comparable on the same rule.

### `initial_k` relaxation

`initial_k > top_k` is permitted. Both guards are removed:
`benchmarks/mcmp/adapters.py:293` and the replay validator at
`benchmarks/mcmp/run_gate1.py:280`. `initial_k <= document_count` is added where
only `top_k <= document_count` was checked. The evidence check at
`benchmarks/mcmp/run_gate1.py:390` already compares against `initial_k` and needs
no change.

### Result writer

The evidence JSON gains `dataset.fixture`, holding the registry key, and a
`runs.E` block. The replay validator treats a missing `fixture` as `"legacy"` and
a missing `runs.E` as a legacy payload. New payloads must carry both.

**Correction, verified empirically.** An earlier draft of this section claimed the
60 committed evidence files "continue to validate unchanged". That claim is wrong
and was not caused by this amendment: `validate_gate1_evidence` compares
`environment` strictly against `_environment_payload()` of the *running*
interpreter, so any payload generated under different NumPy or FAISS versions
already fails replay validation today. `benchmarks/results/gate1-seed-7.json`
(NumPy 1.26.4, FAISS 1.12.0) fails in a NumPy 2.4.6 / FAISS 1.15.0 environment,
while files generated in that environment pass. Gate 1 evidence is therefore not
portable across machines — a pre-existing property recorded here, not fixed here.

The achievable guarantee for this amendment is narrower and is what the tests must
assert: **the schema change introduces no new failure mode for legacy payloads.** A
legacy payload generated in the current environment must still validate after the
change, and must not fail because `dataset.fixture` or `runs.E` is absent.

## Data flow

Unchanged in shape:

```text
fixture registry -> BenchmarkDataset -> adapters A,B,C,D,E -> metrics -> JSON
```

The runner gains `--fixture {legacy,neutral,manifold}`, defaulting to `legacy`.

## Reproducibility

Everything the amended harness records stays as specified in the original design.
Additionally recorded: the fixture registry key and the method E run. The seed
continues to fix the Python and NumPy RNG; because the seed now determines
structure, two runs at the same seed and fixture must remain byte-identical apart
from environment and timing fields, exactly as verified for the legacy fixture in
the Task 7 report. Reproducibility is defined within a fixture and a code version,
not across the amendment boundary — see the note under Non-goals.

## Failure handling

Fixture builders validate through `BenchmarkDataset.validate()` and fail closed.
An unknown `--fixture` value is a hard error naming the valid keys. Method E fails
closed if the requested pool is larger than the corpus.

## Test strategy

RED-GREEN per unit, matching the existing `tests/benchmarks/` convention. The
existing 168 passing tests must stay green; legacy fixture tests are not touched.

The load-bearing new test is the direct regression against the defect this design
exists to fix:

- **Relevant-rank variability:** for the neutral fixture, the tuple of similarity
  ranks of the relevant documents must differ across seeds. The legacy fixture
  returns `(2, 3)` for all 24 query cases; the neutral fixture must not exhibit a
  single constant tuple.

Further coverage:

- Neutral: shapes, dtypes, unit norms, relevance drawn from the top 16, labels
  valid under `BenchmarkDataset.validate()`, determinism per seed.
- Manifold: chain connectivity (consecutive similarity exceeds query-to-far-end
  similarity), at least one relevant document deeper than the default `top_k`,
  determinism per seed.
- Method E: returns only pool documents; accepts `initial_k > top_k`; rejects a
  pool larger than the corpus; deterministic under a fixed seed.
- Constraint relaxation: `initial_k > top_k` is accepted by both the adapter and
  the replay validator; `initial_k > document_count` is still rejected.
- Runner: `--fixture` selects the builder; the JSON carries the fixture key; a
  legacy evidence file without the field still replays.

## Decision gates

This design changes what Gate 1 can measure; it does not change the Gate 1 or
Gate 2 decision rules in the original spec. The standing decision in
`docs/MCMP_TIG_C004_REPORT.md` — do not proceed to Gate 2, query colonies or TIG
C004 — remains in force until Gate 1 is re-run on the new fixtures and reviewed.

A negative result stays valid. If MCMP shows no advantage on the manifold fixture,
that is the answer, and the honest response is to stop the discovery branch rather
than tune the fixture until it passes.

## Deliverables

- `benchmarks/mcmp/fixtures.py`: `build_neutral_dataset`, `build_manifold_dataset`,
  `FIXTURES` registry
- `benchmarks/mcmp/adapters.py`: method E, relaxed constraint
- `benchmarks/mcmp/run_gate1.py`: `--fixture` flag, `dataset.fixture` in the
  payload, relaxed replay validation
- `tests/benchmarks/`: coverage per the test strategy above
- no new evidence files; re-running Gate 1 on the new fixtures is a separate,
  reviewable step
