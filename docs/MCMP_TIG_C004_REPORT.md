# MCMP Gate 1 Evidence Review

Task 7 of `docs/superpowers/plans/2026-08-08-mcmp-ablation-harness.md`.
Design: `docs/superpowers/specs/2026-08-08-mcmp-ablation-harness-design.md`.

This document reviews the Gate 1 evidence and applies the plan's decision rule.
It records no code changes to the harness or to the production retriever.

Facts and interpretation are kept in separate sections on purpose.

---

## 1. Facts: harness verification

**Test suite.** `python -m pytest tests/benchmarks tests/mcmp -q --import-mode=importlib`
→ **133 passed, 2 skipped, 0 failures** (27.85 s).

**Determinism.** The committed evidence `benchmarks/results/gate1-seed-7.json` was
regenerated with its own recorded parameters
(`--seed 7 --top-k 4 --initial-k 1 --num-agents 24 --steps 10`) in a fresh
Python 3.11.0 environment. The re-run differs from the committed file in exactly
four kinds of field:

| Field | Committed | Re-run |
|---|---|---|
| `environment.numpy_version` | 1.26.4 | 2.4.6 |
| `environment.faiss_version` | 1.12.0 | 1.15.0 |
| `runs.{A,B,C,D}.timing.elapsed_ms` | wall clock | wall clock |

Every document id, candidate set, ranking, metric value and the `conclusion`
field are byte-identical. The harness is deterministic and reproduces across a
NumPy major-version boundary (1.26 → 2.4) and a FAISS minor-version change.

Note: `pyproject.toml` pins `numpy>=1.21.0,<2.0`, but the harness reproduces
exactly under NumPy 2.4.6. The pin is stricter than the observed requirement.
No change is proposed here; recorded as an observation.

---

## 2. Facts: A–D measurements at the published parameters

Declared seed set **1–12** (declared before execution, all results reported, none
discarded). Parameters identical to the committed run: `top-k 4`, `initial-k 1`,
`num-agents 24`, `steps 10`, CPU FAISS, `Flat`, inner product.

Evidence: `benchmarks/results/gate1-seed-{1..12}.json`.

Means over the 12 seeds:

| Method | recall@4 | MRR | nDCG@4 | candidates | comparisons | elapsed ms | novel relevant |
|---|---|---|---|---|---|---|---|
| A — Single Query + FAISS | **1.000** | 0.500 | 0.693 | 1.0 | **8** | 0.32 | 0 |
| B — Multi Query + FAISS | 0.500 | 0.500 | 0.363 | 2.0 | 16 | 0.27 | 0 |
| C — Single Query + MCMP | **1.000** | 1.000 | 1.000 | 3.1 | **5 565** | 652.16 | 2 |
| D — Multi Query + MCMP | **1.000** | 1.000 | 1.000 | 4.7 | **11 127** | 43.73 | 4 |

Do not read the `elapsed ms` column as a cost ranking. C runs before D in the
orchestration and absorbs FAISS/NumPy warm-up, which is why C shows 652 ms while
D — at twice the comparison count — shows 43.73 ms. `candidate_comparisons` is
the implementation-independent cost measure and is the one used below.

`conclusion` was `novel_relevant_observed` in **12 of 12** seeds, with identical
counts each time (C = 2, D = 4). Ranked orderings do vary between seeds; the
candidate *sets* and the metric values do not.

**What "novel" is measured against.** `MCMP_novel = MCMP_discovered − FAISS_initial`,
where `FAISS_initial` is the initial candidate pool of size `initial_k`. At the
published `initial_k = 1` that baseline is a single document. Seed 1:

| Method | initial candidates | discovered | ranked top-4 |
|---|---|---|---|
| A | `main-top` | 1 | `main-top, main-near, main-bridge, related-bridge` |
| B | `main-top, related-top` | 2 | `related-top, main-top, related-near, main-near` |
| C | `main-top` | 3 | `main-near, main-bridge, related-bridge, main-top` |
| D | `main-top, related-top` | 4 | `related-near, main-near, related-bridge, main-bridge` |

Ground truth: `q-main → {main-near, main-bridge}`, `q-related → {related-near, related-bridge}`.

The fixture (`benchmarks/mcmp/fixtures.py`) is 8 documents in 3 dimensions with a
hand-designed topology; the seed applies Gaussian jitter of σ = 0.002 and does not
change which documents are relevant or their angular arrangement.

---

## 3. Facts: `initial_k` sensitivity

Same harness, same fixture, only `initial_k` varied. Seeds 1, 7, 12; `top-k 4`,
`num-agents 24`, `steps 10`. Evidence: `benchmarks/results/sensitivity/`.

| `initial_k` | A recall@4 | C recall@4 | C novel relevant | D novel relevant | `conclusion` |
|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 2 | 4 | `novel_relevant_observed` |
| 2 | 1.000 | 1.000 | 1 | 2 | `novel_relevant_observed` |
| 3 | 1.000 | 1.000 | **0** | **0** | `no_novel_relevant_observed` |
| 4 | 1.000 | 1.000 | **0** | **0** | `no_novel_relevant_observed` |

Identical in all three seeds. Method A's recall@4 is 1.000 in every configuration.

---

## 4. Interpretation

The following is interpretation, not measurement.

1. **The Gate 1 headline conclusion is a function of `initial_k`, not of MCMP
   search dynamics.** There are two relevant documents per query. At
   `initial_k ≥ 3` the FAISS initial pool already contains them, so nothing MCMP
   reaches can be *novel* by the metric's own definition, and the conclusion flips
   to `no_novel_relevant_observed`. The published `novel_relevant_observed` result
   is produced by comparing MCMP against a FAISS baseline truncated to one
   document.

2. **FAISS alone already achieves full recall on this fixture.** Method A returns
   all relevant documents within its top-4 ranking at 8 comparisons — in all 12
   seeds at `initial_k = 1`, and in seeds 1, 7 and 12 at every tested
   `initial_k` (1–4). MCMP does not find anything FAISS misses on this fixture;
   it reorders what FAISS already ranks.

3. **The measured MCMP benefit is ranking quality, not coverage** — MRR 1.000 vs
   0.500 and nDCG@4 1.000 vs 0.693 for C over A. That benefit is real in these
   runs but costs ~5 565 comparisons versus 8, a factor of ~696. Method D costs
   ~11 127 versus B's 16. The design spec requires extra compute to be reported
   explicitly; this is that report.

4. **Method B's recall@4 of 0.500 is a fusion-budget effect, not a FAISS defect.**
   B fuses two queries into one top-4 list over a 4-relevant-document union, so it
   is structurally capped. Comparing D against B therefore flatters D.

5. **The 12-seed sweep demonstrates robustness to vector jitter, not independence
   of fixture design.** All seeds share one hand-built 3-dimensional topology.
   12/12 agreement is 12 repetitions of one experimental design, not 12
   independent observations of a phenomenon.

---

## 5. Decision

The plan's decision rule (Task 7, Step 2) reads: if C or D has novel relevant
candidates, repeat Gate 1 over a declared seed set before claiming
reproducibility; in both cases, do not start Gate 2, query colonies, or TIG
without evidence review.

The declared seed set 1–12 has been run. The result reproduces. **However,
reproducibility of the number does not establish the claim the number was meant
to support.**

**Decision: do not proceed to Gate 2, query colonies, or TIG C004 on this
evidence.**

Grounds: the harness satisfies the design spec's Gate 1 precondition — it is
deterministic and all A–D runs produce complete, comparable evidence — but the
`novel_relevant_observed` conclusion does not demonstrate that MCMP discovers
relevant candidates outside what FAISS retrieves. It disappears at
`initial_k ≥ 3`, and FAISS reaches recall@4 = 1.000 without MCMP throughout.

The design spec explicitly permits this outcome: "Gate 1 may legitimately conclude
that MCMP adds no novel relevant candidates," and the plan states "a negative
result is valid; production MCMP behavior must not be changed to make Gate 1
pass." No production MCMP behavior was changed.

### Options for the owner, in order of cost

1. **Re-baseline Gate 1** so `initial_k` equals `top_k`, making the novelty metric
   measure discovery against what FAISS actually returns rather than against a
   truncated pool. Re-run the declared seed set. Cheap; uses the existing harness
   unchanged except for the parameter.
2. **Harden the fixture** before drawing any transfer conclusion: more documents,
   higher dimensionality, relevance that is not a fixed function of angular
   rank. The current 8×3 fixture cannot separate "MCMP works" from "the fixture
   was built so the second and third neighbours are the answers."
3. **Go to Gate 2 anyway** (labelled Fungus retrieval) and treat Gate 1 as
   inconclusive rather than positive. Defensible, but the spec puts Gate 2 behind
   a Gate 1 review, and this review is not a pass.
4. **Stop the MCMP-discovery branch** and keep MCMP, if at all, as a re-ranking
   layer — which is what the MRR/nDCG numbers actually support — under an
   explicit compute budget.

Recommendation: option 1 first, because it is one parameter and it directly tests
whether anything survives. If nothing survives, option 4 is the honest reading.

---

## 6. Repository evidence

```
branch: claude/mcmp-gate1-review-v1  (tracking origin/main)
base:   84fa4aa docs: add README
tests:  133 passed, 2 skipped, 0 failures  (tests/benchmarks tests/mcmp)
python: 3.11.0    numpy: 2.4.6    faiss-cpu: 1.15.0    CPU only
```

Added by this review:

- `benchmarks/results/gate1-seed-{1..6,8..12}.json` — declared seed sweep
- `benchmarks/results/sensitivity/gate1-seed-{1,7,12}-initialk-{1..4}.json`
- this document

`benchmarks/results/gate1-seed-7.json` is left exactly as committed.

## 7. Non-claims

This review did not start or modify the Fungus MCP server, and did not invoke an
LLM, OpenFang, RAG generation, Docker, or a production embedding service. No
Gate 2 run was performed. No TIG C004 algorithm was implemented or evaluated. No
GPU was used. No claim is made about MCMP behaviour on real Fungus code
retrieval — that is Gate 2's question and it remains unanswered.
