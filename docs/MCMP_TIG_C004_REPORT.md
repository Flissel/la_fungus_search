# MCMP Gate 1 Evidence Review

Task 7 of `docs/superpowers/plans/2026-08-08-mcmp-ablation-harness.md`.
Design: `docs/superpowers/specs/2026-08-08-mcmp-ablation-harness-design.md`.

This document reviews the Gate 1 evidence and applies the plan's decision rule.
It records no code changes to the harness or to the production retriever.

Facts and interpretation are kept in separate sections on purpose.

> **Read section 9 first — it carries the current decision.**
>
> This document was written in three rounds and is kept whole rather than rewritten,
> so the reasoning stays auditable.
>
> - **Sections 1-7** review Codex's original Gate 1 evidence on the legacy fixture.
>   Decision at the time: do not proceed to Gate 2.
> - **Section 8** records the re-baseline at `initial_k = top_k`. It sharpened the
>   diagnosis and did not reverse that decision.
> - **Section 9** reviews Gate 1 re-run on the redesigned fixtures, which added a
>   neutral control and a method that isolates MCMP's walk from its reranking.
>   It revises the decision: Gate 2 is justified. Sections 5 and 8.5 remain correct
>   about the legacy fixture; they are superseded, not retracted.
> - **Section 10 corrects section 9.3 point 3, which was wrong.** Section 9 held the
>   agent budget fixed at 24 and read the resulting limit as a property of the
>   mechanism. It is not: at 192 agents MCMP traverses the whole manifold in 12 of
>   12 seeds.
> - **Section 13 is the current state.** A corpus-size sweep shows full-corpus MCMP
>   does not merely get expensive as the corpus grows — it fails: recall 0.611 at 64
>   documents, 0.056 at 1024, while spending 14.6 million comparisons. Method G is
>   flat across the same range at a constant 143 224 comparisons. The bounded
>   frontier is not an optimisation of a working method; it is the only version that
>   survives scale.
> - **Section 12** introduces method G. Method G — a bounded frontier that starts
>   on the FAISS pool and expands toward where the walk goes — matches method C's
>   recall at 16% of the comparisons and discovers more of the chain, while holding
>   about a fifth of the corpus. That is the scaling answer section 11.3 asked for.
> - **Section 11 corrects section 10 in turn.**
>   Section 10 read `steps` as inert and concluded the agents work as parallel
>   random restarts rather than as a colony. Method F — the same walk with the
>   pheromone switched off — refutes that: F reaches exactly one chain document at
>   any agent count while C reaches all three, and `steps` is decisive at 96 agents
>   though it was inert at the 24 where section 10 measured it. **The colony is the
>   mechanism.** Gate 2 stays justified; the ranking degradation stays unsolved.

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

---

## 8. Addendum: re-baseline with `initial_k = top_k`

Section 5 recommended re-baselining so that the novelty metric measures MCMP
against what FAISS actually returns, rather than against a truncated pool. That
run has now been performed.

**Declared design, fixed before execution:** `top_k = initial_k ∈ {2, 3, 4}`,
seeds 1–12, `num_agents 24`, `steps 10`. 36 runs, all reported, none discarded.
Evidence: `benchmarks/results/rebaseline/`.

### 8.1 Facts: does the effect survive?

| `top_k = initial_k` | `conclusion` (12 seeds) | C novel relevant | D novel relevant |
|---|---|---|---|
| 2 | 12 × `novel_relevant_observed` | 1.00 | 2.00 |
| 3 | 12 × `no_novel_relevant_observed` | 0.00 | 0.00 |
| 4 | 12 × `no_novel_relevant_observed` | 0.00 | 0.00 |

Means over 12 seeds per configuration:

| k | Method | recall@k | MRR | nDCG@k | comparisons |
|---|---|---|---|---|---|
| 2 | A | 0.500 | 0.500 | 0.387 | 8 |
| 2 | B | 0.000 | 0.500 | 0.000 | 16 |
| 2 | C | **1.000** | **1.000** | **1.000** | 5 565 |
| 2 | D | 0.500 | 1.000 | 1.000 | 11 127 |
| 3 | A | 1.000 | 0.500 | 0.693 | 8 |
| 3 | B | 0.250 | 0.500 | 0.235 | 16 |
| 3 | C | 1.000 | 1.000 | 1.000 | 5 565 |
| 3 | D | 0.750 | 1.000 | 1.000 | 11 127 |
| 4 | A | 1.000 | 0.500 | 0.693 | 8 |
| 4 | B | 0.500 | 0.500 | 0.363 | 16 |
| 4 | C | 1.000 | 1.000 | 1.000 | 5 565 |
| 4 | D | 1.000 | 1.000 | 1.000 | 11 127 |

Across all 36 runs: C recall@k is 1.000 in every single run (min = max = 1.000).
A ranges 0.500–1.000 (mean 0.833). B ranges 0.000–0.500 (mean 0.250).

### 8.2 Facts: the fixture's relevance structure

Computed directly from `benchmarks/mcmp/fixtures.py`, all 12 seeds × 2 queries
= 24 query cases:

- The relevant documents occupy FAISS similarity ranks **exactly (2, 3)** in
  **all 24** cases.
- The rank-1 document (`main-top` / `related-top`, the exact query match,
  similarity ≈ 1.0) is labelled **irrelevant** in **all 24** cases.

Seed 7, `q-main`: `main-top` 1.0000 · `main-near` 0.9836 (relevant) ·
`main-bridge` 0.8010 (relevant) · `related-bridge` 0.5986 · …

### 8.3 Facts: the standard control cannot be expressed

The natural cheap competitor to MCMP is retrieve-then-rerank: give FAISS a
candidate pool deeper than the returned list. The harness rejects it —
`benchmarks/mcmp/adapters.py:294` raises `ValueError("initial_k must not exceed
top_k")`. `initial_k > top_k` is therefore unreachable, and
`MCMP_novel = MCMP_discovered − FAISS_initial` is evaluated against a pool that
is by construction never deeper than `top_k`.

### 8.4 Interpretation

1. **The re-baseline does not rescue the positive result; it localises it.** The
   novelty effect exists only at `k = 2` and disappears at `k ≥ 3`. This is a
   direct consequence of 8.2: the deepest relevant document sits at rank 3, so a
   pool of 2 misses it and a pool of 3 or more contains it. MCMP reaches one rank
   position beyond the pool. That is pool expansion, not semantic discovery, and
   the fixture cannot distinguish the two.

2. **The ranking advantage survives at every k — and 8.2 explains it entirely.**
   A's MRR is 0.500 in all configurations because the first relevant document is
   always at rank 2, giving RR = 1/2 exactly. MCMP scores MRR 1.000 because it
   demotes the rank-1 document. The fixture labels the single most similar
   document as irrelevant, which is adversarial to similarity ranking by
   construction. Any method that declines to rank purely by similarity gains here.

3. **The measured benefit is therefore a property of the fixture's labelling, not
   demonstrated retrieval behaviour.** Every Gate 1 number — A's MRR of 0.5, C's
   1.0, the novelty counts, their disappearance at `k ≥ 3` — follows
   deterministically from "relevance is defined as ranks 2 and 3, never rank 1."
   The experiment has effectively one degree of freedom.

4. **The missing control is structural, not accidental** (8.3). Until the harness
   permits `initial_k > top_k`, it cannot answer whether a cheap reranker over a
   slightly deeper FAISS pool matches MCMP at ~700× less compute.

### 8.5 Decision after re-baseline

**The section 5 decision stands: do not proceed to Gate 2, query colonies, or
TIG C004 on this evidence.** The re-baseline was the cheapest test available and
it did not produce a result attributable to MCMP's search dynamics.

The blocking issue is no longer the `initial_k` parameter — it is the fixture.
Before any further MCMP ablation work is worth running:

1. **Relevance must stop being a fixed function of similarity rank.** As long as
   the relevant set is exactly ranks (2, 3) and rank 1 is always irrelevant, the
   harness measures the labelling convention, not the retriever.
2. **Allow `initial_k > top_k`** so retrieve-then-rerank can be run as a control
   at comparable cost.
3. Only then is a positive Gate 1 result informative enough to justify Gate 2.

If the owner does not want to invest in (1) and (2), the honest reading of the
current evidence is section 5's option 4: keep MCMP, if at all, as a re-ranking
layer under an explicit compute budget, and stop the discovery branch.

### 8.6 Non-claims for this addendum

No harness or production code was modified. `initial_k > top_k` was attempted and
refused by the existing contract; it was not worked around. Section 8.2 is a read-only
computation over the existing fixture. No Gate 2 run, no TIG C004 implementation,
no MCP server, LLM, OpenFang, Docker or GPU involvement.

---

## 9. Gate 1 on the redesigned fixtures

Run after the fixture redesign
(`docs/superpowers/specs/2026-08-29-mcmp-fixture-redesign-design.md`). This is the
first Gate 1 round in which a control exists and in which MCMP's walk can be
separated from its reranking.

**Declared design, fixed before execution:** fixtures `neutral` and `manifold`;
seeds 1-12; two budget regimes per fixture, `top_k = initial_k = 8` (matched) and
`top_k = 8, initial_k = 16` (retrieve-then-rerank); `num_agents 24`, `steps 10`
unchanged for comparability. 48 runs, all reported, none discarded. Evidence:
`benchmarks/results/gate1-v2/`.

Methods: A Single+FAISS, B Multi+FAISS, C Single+MCMP over the full corpus,
D Multi+MCMP, E Single+MCMP restricted to the FAISS pool.

### 9.1 Facts: the neutral control

Means over 12 seeds at `top_k = initial_k = 8`:

| Method | recall@8 | MRR | nDCG@8 | comparisons |
|---|---|---|---|---|
| A | **0.438** | **0.433** | **0.341** | 64 |
| B | 0.243 | 0.380 | 0.249 | 128 |
| C | 0.417 | 0.404 | 0.322 | 43 936 |
| D | 0.234 | 0.345 | 0.240 | 88 267 |
| E | **0.438** | 0.418 | 0.339 | 44 213 |

Paired per seed, C against A: recall better in 3 seeds, worse in 2, tied in 7.
MRR better in 2, **worse in 5**, tied in 5.

At `initial_k = 16` the picture is unchanged for A-D; E falls to recall 0.396.
The conclusion field moves from 5/12 `novel_relevant_observed` at `initial_k = 8`
to 12/12 `no_novel_relevant_observed` at `initial_k = 16`.

### 9.2 Facts: the manifold fixture

Means over 12 seeds, identical at `initial_k = 8` and `initial_k = 16`:

| Method | recall@8 | MRR | nDCG@8 | comparisons |
|---|---|---|---|---|
| A | 0.000 | 0.000 | 0.000 | 64 |
| B | 0.000 | 0.000 | 0.000 | 128 |
| C | **0.222** | 0.117 | 0.114 | 44 544 |
| D | 0.000 | 0.143 | 0.000 | 89 088 |
| E | 0.000 | 0.000 | 0.000 | 44 544 |

Per seed, C is the only method with nonzero recall. It ranks a relevant document in
its top-8 in **8 of 12 seeds**, and reaches one into its candidate set in **9 of 12**.

In every seed where C succeeds, the document it reaches is `main-chain-6` — and
only `main-chain-6`. `main-chain-7` and `main-chain-8` are never reached in any of
the 24 manifold runs.

D's recall of 0.000 alongside MRR 0.143 is a fusion-budget effect: its per-query
ranking for `q-main` does contain `main-chain-6`, but the fused top-8 is filled by
chain links 1-5 of both chains.

### 9.3 Interpretation

Interpretation, not measurement.

1. **The control works, and MCMP shows no advantage on it.** On the neutral fixture
   MCMP is not better than plain FAISS on recall (3 better / 2 worse / 7 tied) and
   is slightly worse on ranking (2 better / 5 worse / 5 tied), at 686x the
   comparisons. This is the outcome the control was built to be able to show, and
   it is the first time Gate 1 has been able to show it at all. Note also that at
   `initial_k = 8` five neutral seeds report `novel_relevant_observed` while C's
   recall is *below* A's: **novelty and retrieval quality are not the same thing,**
   and the conclusion field tracks only the former.

2. **On the manifold fixture there is a real effect, and it is attributable.**
   C reaches a relevant document that A, B and E never reach. E is the load-bearing
   comparison: it runs identical MCMP machinery with identical parameters and
   differs only in being confined to the FAISS pool, and it scores 0.000 in all 24
   manifold runs. The difference between C and E is therefore attributable to the
   walk leaving the pool, not to MCMP's reranking. This is the first
   mechanism-attributable positive result Gate 1 has produced.

3. **[CORRECTED IN SECTION 10 — this point is wrong.]** It was measured at a fixed
   agent budget of 24 and generalised beyond it; at 192 agents MCMP does traverse
   the chain to its end. The original text follows unchanged for the record.
   **The effect's magnitude is one rank band, not manifold traversal.** C reaches
   `main-chain-6` and never `main-chain-7` or `main-chain-8`. Chain link 6 sits at
   cosine 0.517, just below the distractor band at 0.55-0.75; links 7 and 8 sit at
   0.364 and 0.199. So the walk penetrates roughly one similarity band past what
   FAISS retrieves, and does not follow the chain to its end. This is the same
   magnitude the Task 7 review measured on the legacy fixture, where MCMP reached
   exactly one rank position beyond the pool. Two different fixtures, one
   consistent mechanism.

4. **The cost is unchanged and remains the dominant fact.** 44 544 comparisons
   against 64, a factor of ~696, to recover on average 0.67 of 3 relevant documents
   in 8 of 12 seeds. E's figure is an upper bound by the accounting caveat recorded
   in the plan and is not cited as exact here.

5. **`initial_k` is inert on the manifold fixture.** The 8 and 16 regimes give
   identical A-E numbers, because C searches the whole corpus regardless and E's
   pool never contains chain link 6 at either depth. Deepening the retrieval pool
   is not a substitute for the walk on this structure — which is the honest form of
   the retrieve-then-rerank control the harness previously could not run.

6. **The manifold fixture was built to have the structure MCMP needs.** That is its
   declared purpose and it is what makes the result attributable, but it bounds the
   claim: this shows the mechanism works on chain-structured data, not that code
   retrieval is chain-structured. Per the plan's recorded caveat, the 12 seeds vary
   vector realization at one fixed difficulty; they are not 12 difficulty levels.

### 9.4 Decision

The design spec's Gate 1 rule is about harness quality: proceed to Gate 2 when the
harness is deterministic and all runs produce complete, comparable evidence.

**That condition is now met, and it was not met before.** Gate 1 has a control that
can falsify, a method that isolates the walk from the rerank, and a retrieval-pool
control that can actually be run. The earlier blocking objection — that Gate 1
measured the labelling convention — no longer applies to these fixtures.

**Revised decision: Gate 2 is now justified, which it was not on the previous
evidence.** This supersedes the section 5 and 8.5 decisions, which were made when
the only positive result was a parameterization artifact. It is not a reversal of
those findings: they remain correct about the legacy fixture.

Two conditions on that recommendation:

1. **Gate 2 must answer the structural question, not repeat the metric.** The open
   question is whether real Fungus code retrieval contains chain structure of the
   kind the manifold fixture supplies. If it does not, the mechanism has nothing to
   act on and the manifold result does not transfer.
2. **[REPLACED BY SECTION 10.5 — this condition points the wrong way.]** It asks
   for the walk to be bounded, which is precisely what destroys the gain. Retained
   for the record. **The cost profile belongs in Gate 2's design from the start.**
   A ~696x comparison factor for a one-band depth gain is not viable in production
   as it stands. Gate 2 should measure whether the walk can be bounded — fewer
   agents, fewer steps, or a restricted frontier — while keeping the depth gain,
   because that, not the raw effect, decides whether this is shippable.

If the owner would rather not spend Gate 2 yet, the defensible alternative is
unchanged from section 5's option 4: keep MCMP as a bounded reranking layer and
stop the discovery branch. The neutral control gives no support for MCMP as a
general retrieval improvement.

### 9.5 Non-claims

No production MCMP behaviour was changed. No Gate 2 run was performed, no TIG C004
algorithm implemented, no MCP server, LLM, OpenFang, Docker or GPU involved. E's
novelty count is structurally zero by construction and is not reported as a finding
about MCMP. No claim is made about MCMP on real code retrieval.

---

## 10. Correction: the one-band limit was an agent-budget artifact

Section 9 held `num_agents = 24` and `steps = 10` fixed across all 48 runs, for
comparability with the earlier rounds. Section 9.3 then generalised beyond that
hold. This section corrects it.

**Section 9.3 point 3 is wrong.** It stated that "the walk penetrates roughly one
similarity band past what FAISS retrieves, and does not follow the chain to its
end", and read that as a property of the mechanism. It is not. It is a property of
running MCMP with 24 agents.

The section 9.2 *facts* stand — at 24 agents and 10 steps, `main-chain-7` and
`main-chain-8` were never reached in any of those 24 runs. What was wrong was
treating that as the mechanism's ceiling.

### 10.1 Facts: steps do not matter, agents do

Manifold fixture, `top_k = initial_k = 8`, seed 1. Raising steps 20-fold changes
nothing:

| steps | agents | C recall@8 | relevant found | comparisons |
|---|---|---|---|---|
| 10 | 24 | 0.333 | `main-chain-6` | 44 544 |
| 50 | 24 | 0.333 | `main-chain-6` | 228 864 |
| 200 | 24 | 0.333 | `main-chain-6` | 920 064 |

Raising the agent count does. Manifold, `steps = 50`, means over seeds 1-12:

| agents | recall@8 | seeds with recall>0 | seeds discovering all 3 | comparisons | vs A |
|---|---|---|---|---|---|
| 24 (§9 setting, steps 10) | 0.222 | 8 / 12 | 0 / 12 | 44 544 | 696x |
| 48 | 0.333 | 12 / 12 | 0 / 12 | 457 728 | 7 152x |
| 96 | **0.722** | 12 / 12 | 10 / 12 | 915 456 | 14 304x |
| 192 | 0.611 | 11 / 12 | **12 / 12** | 1 830 912 | 28 608x |
| 384 | 0.139 | 5 / 12 | **12 / 12** | 3 661 824 | 57 216x |

### 10.2 Facts: discovery and ranking come apart

Splitting "reached as a candidate" from "ranked into the top 8", means over 12 seeds:

| agents | relevant discovered | of those, ranked in top-8 |
|---|---|---|
| 48 | 1.00 / 3 | 1.00 / 3 |
| 96 | 2.67 / 3 | 2.17 / 3 |
| 192 | 3.00 / 3 | 1.83 / 3 |
| 384 | 3.00 / 3 | 0.42 / 3 |

Discovery rises monotonically with agents and saturates at complete traversal.
Ranking peaks near 96 agents and then collapses.

### 10.3 Facts: the control still holds at the manifold optimum

Neutral fixture at the manifold-optimal budget (`agents = 96`, `steps = 50`),
means over 12 seeds:

| Method | recall@8 | MRR | nDCG@8 | comparisons |
|---|---|---|---|---|
| A | **0.438** | **0.433** | **0.341** | 64 |
| C | 0.375 | 0.429 | 0.305 | 914 923 |
| E | **0.438** | 0.429 | 0.339 | 915 088 |

Paired against A, C is better on recall in 2 seeds, worse in 4, tied in 6. MCMP
still shows no advantage on the control, at 14 000x the comparisons. The manifold
gain is therefore structure-specific and not a budget effect — which is what makes
the control worth having.

### 10.4 Revised interpretation

1. **MCMP's walk can traverse the whole manifold.** At 192 agents it discovers all
   three chain-end documents in 12 of 12 seeds. The mechanism is not depth-limited
   in the way section 9 claimed. This is a stronger positive result for MCMP than
   section 9 reported.
2. **The bottleneck is the relevance scoring, not the walk.** Past roughly 96
   agents, MCMP keeps finding the right documents and keeps ranking them worse,
   until at 384 agents it discovers everything and surfaces almost none of it.
   Exploration and scoring are in tension in the current implementation.
3. **`steps` is the wrong knob.** A 20-fold increase changes nothing. Whatever
   limits a single agent's reach is not iteration count.
4. **The cost verdict gets worse, not better.** The section 9 figure of ~696x was
   measured at a setting that finds one third of the target. The best-ranking
   setting costs ~14 300x and the complete-discovery setting ~28 600x.
5. **The methodological failure is the same one this report keeps documenting.**
   Round one generalised past a fixed `initial_k`. Round two generalised past a
   fixed fixture labelling. This round generalised past a fixed agent budget. Each
   time the held-fixed parameter, not the mechanism, produced the headline. Any
   future claim here should sweep the parameter it is about to generalise over.

### 10.5 Revised decision

**Gate 2 remains justified** — section 9.4 is unchanged on that point, and the
evidence for it is now stronger, since full traversal is reproducible in 12 of 12
seeds rather than a partial effect in 8.

**Section 9.4's second condition is replaced.** It asked Gate 2 to measure whether
the walk can be *bounded* — fewer agents, fewer steps, a restricted frontier —
while keeping the depth gain. That prescription was based on the mistaken
one-band reading and points the wrong way: the walk needs *more* agents, and
shrinking it is what destroys the gain.

The correct condition is: **Gate 2 should measure whether MCMP's scoring can hold
onto what its walk finds.** Discovery is solved and expensive; ranking is the
unsolved part. A Gate 2 that only reports retrieval metrics at one agent count will
reproduce exactly the error this section corrects — it must sweep the agent budget
and report discovery and ranking separately.

The cost condition stands and hardens: 14 300x at the best operating point is not
a production profile. Whether that is fixable is a question about the scoring, not
about the walk.

### 10.6 Non-claims

No production MCMP behaviour was changed; the agent and step counts are existing
CLI parameters. No Gate 2 run, no TIG C004 implementation, no MCP server, LLM,
OpenFang, Docker or GPU. The exploration bonus and pheromone decay were not varied
and remain at their harness constants; no claim is made about their effect.

---

## 11. Method F: does the colony contribute, or are the agents just a sampling budget?

Section 10 established that discovery scales with agent count while `steps` appeared
inert, and read that as evidence the agents function as parallel random restarts
rather than as a pheromone-coordinated colony. Method F was built to test that
directly. **The hypothesis was wrong, and section 10's `steps` observation was
measured in a regime where the mechanism is not engaged.**

**Method F** is method C with the colony switched off: pheromone deposition runs
normally and the trail memory is then cleared, so the 0.15 pheromone term of the
movement force contributes nothing while agents, steps, seeds, attraction and
exploration stay exactly as C has them. Zero recorded trails is the
self-verifying evidence that the control is pheromone-free.

**Declared design, fixed before execution:** fixtures `manifold` and `neutral`,
seeds 1-12, `agents ∈ {96, 192}`, `steps 50`, `top_k = initial_k = 8`. 48 runs,
all reported. Evidence: `benchmarks/results/method-f/`.

### 11.1 Facts

Means over 12 seeds:

| fixture | agents | method | recall@8 | MRR | nDCG@8 | comparisons |
|---|---|---|---|---|---|---|
| manifold | 96 | C | **0.722** | 0.165 | 0.345 | 915 456 |
| manifold | 96 | F | 0.333 | 0.167 | 0.167 | 614 464 |
| manifold | 192 | C | **0.611** | 0.145 | 0.290 | 1 830 912 |
| manifold | 192 | F | 0.333 | 0.163 | 0.166 | 1 228 864 |
| neutral | 96 | C | 0.375 | 0.429 | 0.305 | 914 923 |
| neutral | 96 | F | 0.375 | 0.429 | 0.305 | 614 464 |
| neutral | 192 | C | 0.375 | 0.419 | 0.309 | 1 830 379 |
| neutral | 192 | F | 0.375 | 0.419 | 0.309 | 1 228 864 |

Paired per seed on recall@8:

| fixture | agents | C better | F better | tied |
|---|---|---|---|---|
| manifold | 96 | **10** | 0 | 2 |
| manifold | 192 | **9** | 1 | 2 |
| neutral | 96 | 0 | 0 | **12** |
| neutral | 192 | 0 | 0 | **12** |

Relevant chain documents discovered, manifold:

| agents | C | F |
|---|---|---|
| 96 | 2.67 / 3 | **1.00 / 3** |
| 192 | 3.00 / 3 | **1.00 / 3** |

`steps` sensitivity re-measured at 96 agents, manifold, 12 seeds:

| steps | C recall@8 | C discovered | comparisons |
|---|---|---|---|
| 10 | 0.333 | 1.00 / 3 | 178 176 |
| 50 | 0.722 | 2.67 / 3 | 915 456 |
| 200 | 0.667 | 2.67 / 3 | 3 680 256 |

### 11.2 Interpretation

1. **The colony is the mechanism. Section 10's reading was wrong.** F reaches
   exactly 1.00 of 3 relevant chain documents at 96 agents and at 192 — adding
   agents does not move it. C reaches 2.67 and 3.00. Chain traversal is what the
   pheromone buys; without it, the walk stops at the first link no matter how
   large the sampling budget. The agents are not parallel restarts.

2. **`steps` is not inert; it was measured where the mechanism is dormant.**
   Section 10 reported a 20-fold step increase changing nothing, at 24 agents. At
   96 agents, 10 → 50 steps takes discovery from 1.00/3 to 2.67/3. The colony
   needs enough agents to lay trails *and* enough steps to follow them; below
   either threshold MCMP degenerates to exactly what F does. This is the fourth
   time in this report that a conclusion came from a parameter held fixed outside
   the regime it was generalised over, and the second time it was mine.

3. **The control behaves as a control must.** On the neutral fixture C and F are
   identical — same recall, same MRR, same nDCG, tied in all 12 seeds at both
   agent counts. Where there is no chain structure the pheromone contributes
   exactly nothing. That is what makes the manifold difference attributable to
   structure rather than to the machinery.

4. **The colony's price is roughly 49% on top of an already expensive walk.**
   The pheromone force computation makes its own nearest-neighbour calls: 915 456
   comparisons against F's 614 464 at the same agent and step count. On the
   manifold fixture that buys 0.333 → 0.722 recall. On the neutral fixture it
   buys nothing at all.

5. **The ranking degradation from section 10 is unaffected and still unsolved.**
   C's recall falls from 0.722 at 96 agents to 0.611 at 192, and 200 steps is
   worse than 50. More exploration keeps finding more and ranking it worse.

### 11.3 What this changes for the road ahead

The question was whether to make the colony scale or replace it. The answer is
now measured: **the colony is what produces the capability**, so replacing the
walk with bounded sampling would discard the only thing MCMP does that FAISS
cannot. The scaling problem has to be solved with the colony intact.

Three constraints any large-corpus design must respect, all measured:

- MCMP walks the entire corpus (`add_documents` receives every document), and
  comparisons scale with corpus size. This is the binding obstacle, not the
  pheromone.
- The mechanism has thresholds in both agents and steps. A larger corpus needs
  more of both to lay and follow trails across it, and cost is roughly the
  product.
- Ranking degrades as exploration grows, so simply scaling the budget up makes
  results worse even where discovery improves.

Method E's pool restriction is the obvious scaling lever, and section 9 measured
it at 0.000 on the manifold fixture — a pool-confined colony finds nothing,
because the chain leaves the pool by construction. A large-corpus design therefore
needs a *bounded but not pool-confined* frontier: something that lets the walk
leave its starting neighbourhood without touching the whole corpus. That is the
open design problem.

### 11.4 Non-claims

No production MCMP behaviour was changed; F is a benchmark-side subclass. No
Gate 2 run, no TIG C004 implementation, no MCP server, LLM, OpenFang, Docker or
GPU. The renormalised variant — rescaling the remaining force weights so F
matches C's total force magnitude — was not run; F therefore differs from C in
two ways, no trail guidance and a slightly smaller total force, and the manifold
difference cannot be attributed to guidance alone without it.

---

## 12. Method G: a bounded frontier, and the scaling answer

Section 11.3 named the open design problem: MCMP walks the entire corpus, method E
shows a pool-confined colony finds nothing on a chain, and a large-corpus design
therefore needs a working set that is **bounded but not pool-confined**. Method G
is that, and it is measured here.

**Method G** starts on the FAISS top-`initial_k` pool. Every `expand_every` steps
the most-visited not-yet-expanded document contributes its `expand_k` nearest
neighbours from the full corpus, capped at `frontier_cap`. It needs no production
change: `add_documents` appends and rebuilds the index without resetting agents,
trails or visit counts, so the colony survives each expansion.

**Declared design, fixed before execution:** fixtures `manifold` and `neutral`,
seeds 1-12, 96 agents, 50 steps, `top_k = initial_k = 8`, `expand_every 10`,
`expand_k 4`, `frontier_cap 24` of a 64-document corpus. Evidence:
`benchmarks/results/method-g/`.

**A prerequisite had to be fixed first.** `candidate_comparisons` was calls
multiplied by the *final* corpus size, which is correct only while the working set
is constant. It overstated method E — a limitation carried since the Gate 1 review
— and would have made G's entire point unmeasurable. Comparisons now accumulate
against the corpus present at each call.

### 12.1 Facts

Means over 12 seeds:

| fixture | method | recall@8 | nDCG@8 | comparisons | vs C |
|---|---|---|---|---|---|
| manifold | A | 0.000 | 0.000 | 64 | 0% |
| manifold | C | 0.722 | 0.345 | 915 456 | 100% |
| manifold | E | 0.000 | 0.000 | 114 488 | 13% |
| manifold | **G** | **0.722** | 0.337 | **143 224** | **16%** |
| neutral | A | **0.438** | 0.341 | 64 | 0% |
| neutral | C | 0.375 | 0.305 | 914 923 | 100% |
| neutral | E | 0.438 | 0.339 | 114 442 | 13% |
| neutral | G | 0.375 | 0.305 | 178 458 | 20% |

Relevant chain documents discovered, manifold: A 0.00/3, E 0.00/3, C 2.67/3,
**G 3.00/3**. Paired per seed on recall, C against G: 3 / 2 / 7 tied.

G added 4.0 documents on average to its 8-document pool — a working set of about
12 of 64 documents.

`frontier_cap` sweep (12, 16, 24, 40): **identical results and identical
comparison counts at every value.** The cap was never binding, because only 4
documents were ever added. It is inert at this configuration, which is not the
same as irrelevant.

`expand_every` sweep, the parameter that does bind:

| `expand_every` | rounds | recall@8 | discovered | comparisons | vs C |
|---|---|---|---|---|---|
| 5 | 9 | 0.694 | 2.83 / 3 | 156 304 | 17% |
| 10 | 4 | **0.722** | **3.00 / 3** | 143 224 | 16% |
| 25 | 1 | **0.000** | 0.00 / 3 | 121 624 | 13% |

### 12.2 Interpretation

1. **G is the scaling answer for this structure.** It matches C's recall exactly
   at 16% of the comparisons, and discovers *more* of the chain — 3.00 of 3
   against C's 2.67 — while holding about a fifth of the corpus. E, confined to
   the same starting pool, finds nothing at all. The difference between E and G
   is one mechanism: the ability to leave the starting neighbourhood.

2. **The optimum in `expand_every` is explained, not tuned.** A single expansion
   round gives 0.000, exactly matching E: with no walking between expansions the
   colony has laid no trails, so there is nothing to tell it where to expand.
   Nine rounds is slightly worse than four, because each round then expands from
   a less settled trail. Expansion has to be interleaved with enough walking for
   the pheromone to identify the frontier — which is the same mechanism section 11
   established, seen from a different angle.

3. **On the neutral control G behaves exactly like C**, and both are worse than
   plain FAISS. G does not rescue MCMP where there is no structure; it makes the
   structured case affordable. That is the correct division and it is what the
   control is for.

4. **`frontier_cap` was inert here and must not be reported as unimportant.**
   It never bound, because expansion added only 4 documents. On a corpus where
   the walk ranges further it would bind, and it is untested in that regime.

5. **The cost saving is measured at 64 documents; its growth with corpus size is
   an extrapolation.** G's per-call cost is bounded by its working set, C's scales
   with the corpus, so the ratio should widen roughly linearly — at 103 000
   chunks C scans 103 000 per call where G scans a few dozen. That follows from
   the cost model, and it is *not* measured. A corpus-size sweep is the obvious
   next measurement and this report does not anticipate its result.

### 12.3 Non-claims

No production MCMP behaviour was changed; G is a benchmark-side orchestration over
existing public methods. Frontier expansions are reported as their own counters
and never folded into `candidate_comparisons`, because in production they are ANN
index queries whose cost is not comparable to the walk's linear scans; any
end-to-end cost claim needs that index's cost model, which this report does not
have. No Gate 2 run, no TIG C004, no live service of any kind.

---

## 13. Corpus scaling: the bounded frontier is not an optimisation, it is the only version that survives

Section 12.2 point 5 said the cost saving's growth with corpus size followed from
the cost model and was **not measured**, and that a corpus-size sweep was the next
measurement. It has been run, and it found something the cost model did not
predict.

The manifold fixture's document count is now a parameter. Only the distractor
field scales; the two chains keep their length and their relevant tail, so a
larger corpus is a harder haystack rather than a different needle.

**Declared design:** manifold at 64, 256 and 1024 documents, seeds 1-6, 96 agents,
50 steps, `top_k = initial_k = 8`, `expand_every 10`, `expand_k 4`,
`frontier_cap 24`. Evidence: `benchmarks/results/scaling/`.

### 13.1 Facts: corpus size

| documents | method | recall@8 | discovered / 3 | comparisons | vs C |
|---|---|---|---|---|---|
| 64 | A | 0.000 | 0.00 | 64 | 0.0% |
| 64 | C | 0.611 | 2.33 | 915 456 | 100% |
| 64 | **G** | **0.667** | **3.00** | 143 224 | 15.6% |
| 256 | C | 0.389 | 2.50 | 3 661 824 | 100% |
| 256 | **G** | **0.722** | **3.00** | 143 224 | 3.9% |
| 1024 | C | **0.056** | 2.33 | 14 647 125 | 100% |
| 1024 | **G** | **0.667** | **3.00** | 143 224 | **1.0%** |

G's working set is 8 + 4.0 = 12 documents at **every** corpus size — 18.8% of the
corpus at 64, 1.2% at 1024 — so its comparison count is identical (143 224) at all
three sizes while C's grows linearly.

### 13.2 Facts: agent count at 256 documents

| agents | method | recall@8 | nDCG@8 | discovered / 3 |
|---|---|---|---|---|
| 96 | C | 0.389 | 0.185 | 2.50 |
| 96 | **G** | **0.722** | **0.334** | 3.00 |
| 192 | C | 0.278 | 0.131 | 3.00 |
| 192 | **G** | **0.500** | **0.226** | 3.00 |
| 384 | C | **0.000** | **0.000** | 3.00 |
| 384 | **G** | **0.389** | **0.176** | 3.00 |

### 13.3 Interpretation

1. **Full-corpus MCMP does not scale — it fails.** C's recall falls from 0.611 at
   64 documents to 0.056 at 1024. This is not a slowdown; at 1024 documents the
   method returns essentially nothing useful while spending 14.6 million
   comparisons. The bounded frontier is therefore not an optimisation of a working
   method. It is the only version that still works.

2. **The failure is ranking, not discovery.** At 1024 documents C still reaches
   2.33 of 3 relevant chain documents — discovery holds — but ranks only about
   0.17 of them into its top 8. The relevance ordering is computed over the whole
   working set, so as the corpus grows the chain documents are pushed out by
   distractors. G ranks over 12 candidates instead of 1024 and does not have the
   problem. This is the ranking collapse from section 10, and corpus size is a
   second, independent way to trigger it.

3. **G's cost is constant, not merely smaller.** 143 224 comparisons at 64, 256
   and 1024 documents. It is bounded by the working set, which is bounded by
   `expand_every × expand_k`, and the corpus size does not enter. The ratio to C
   is therefore whatever the corpus size makes it: 1.0% at 1024, and it would keep
   falling.

4. **G mitigates the agent-driven ranking collapse but does not solve it.** From
   96 to 384 agents G falls 0.722 → 0.389 while C falls 0.389 → 0.000. G at 384
   agents still beats C at 96. The degradation is real in both and the same
   mechanism drives it; a smaller working set makes it much less severe. Section
   10's finding — that MCMP's scoring is the unsolved part — stands.

5. **`frontier_cap` never bound, at any corpus size.** Expansion added exactly 4.0
   documents at 64, 256 and 1024 alike, because it is limited by the number of
   expansion rounds and not by the cap. That G finds the whole chain by adding
   four documents to a 1024-document corpus is the striking part of this result,
   and it is also its narrowest point: the fixture's chain is 8 links and reachable
   in few hops by construction. A structure needing a longer or branching walk
   would need `expand_k` or the round count to grow, and that is untested.

### 13.4 Non-claims

Synthetic fixtures throughout, with a chain the fixture places there by design.
Six seeds per configuration, not twelve. Nothing here says real code retrieval has
this structure — that is Gate 2's question and it remains unrun. Frontier lookups
are still reported separately from walk comparisons; an end-to-end production cost
needs the ANN index's cost model, which this report does not have.
