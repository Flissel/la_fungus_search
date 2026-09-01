# MCMP Gate 1 Evidence Review

Task 7 of `docs/superpowers/plans/2026-08-08-mcmp-ablation-harness.md`.
Design: `docs/superpowers/specs/2026-08-08-mcmp-ablation-harness-design.md`.

This document reviews the Gate 1 evidence and applies the plan's decision rule.
It records no code changes to the harness or to the production retriever.

Facts and interpretation are kept in separate sections on purpose.

> **Read section 22 first — it reverses the verdict of everything before it and
> says why.** Sections 17, 18 and 21 ran Gate 2 stage 1 on a 238-document corpus
> and the gate closed, at one point on an exact tie. Section 22 runs the same
> protocol on two independent 4 000-document samples of a second repository and
> **the gate opens on both, on every condition, by a factor of two.** The confound
> is corpus size: on a small corpus almost everything is reachable from almost
> everything, so the permutation null scores nearly as well as the real labels and
> the test has no power. Sections 19 and 20 (the crawler and sibling relations)
> carry the same caveat. Sections 1-16 are synthetic throughout.**
>
> This document was written in rounds and is kept whole rather than rewritten,
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
> - **Section 15 is the current state, and it corrects section 14 twice.** Section
>   14 compared four shapes of the visit term at one weight; the weight turns out to
>   be the dominant variable. On the corpus where the shipped ceiling returns
>   nothing, a swept term reaches recall 1.000, and the collapse under agent count is
>   inverted rather than mitigated. Separately: the visit term's damage was in the
>   *steering*, not the ranking — `relevance_score` both ranks the result and feeds
>   the attraction force, and splitting those two jobs is a structural fix that costs
>   no parameter. Section 15 also introduces the neutral control into this line of
>   work, which is what separates candidates that manifold alone cannot.
> - **Section 14.** It opens the relevance function, which
>   sections 10, 11 and 13 all pointed at without examining. Removing the visit
>   term drops recall to 0.000 in every configuration measured — it is the entire
>   ranking signal — and it saturates at five visits, which is a mechanism for both
>   documented collapses. It also records two undocumented defects in the pheromone
>   code whose repair makes retrieval *worse*, and explains why.
> - **Section 13.** A corpus-size sweep shows full-corpus MCMP
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

---

## 14. The relevance function: the visit term is the whole ranking signal, and it saturates

Sections 10, 11 and 13 all arrived at the same place from different directions —
discovery holds, ranking fails — and none of them opened the scoring function.
This section does.

**Declared design, fixed before execution:** fixture `manifold`, seeds 1-6,
`agents ∈ {96, 192, 384}`, `steps 50`, `top_k = initial_k = 8`, corpus sizes 256
and 1024, methods C and G. Every variant is an override applied inside a probe;
nothing under `src/` was modified. Reproduce with
`benchmarks/probes/visit_term.py`.

recall@8 is measured over one query with three relevant documents across six
seeds, so the resolution is 1/18 ≈ 0.056. Differences of that size are one
document in one seed and are not read as differences below.

### 14.1 Facts: ablating the visit term

`update_document_relevance` computes
`r_i = cosine(q, d_i) + min(0.1 × visits_i, 0.5) + time_bonus + kw_bonus`.
Removing the visit term alone, at 256 documents:

| method | agents | with visit term | without |
|---|---|---|---|
| C | 96 | 0.389 | **0.000** |
| C | 192 | 0.278 | **0.000** |
| C | 384 | 0.000 | **0.000** |
| G | 96 | 0.722 | **0.000** |
| G | 192 | 0.500 | **0.000** |
| G | 384 | 0.389 | **0.000** |

Under the benchmark's fixed clock (`DETERMINISTIC_CLOCK_VALUE = 2.0`) the time
bonus reduces to a flat +0.1 for any document visited at least once, and
`kw_lambda` is 0. So the "without" column is cosine similarity plus a visited flag.

### 14.2 Facts: replacing the cap

Three shapes that cannot saturate, against the shipped `min(0.1 v, 0.5)`. `log`
is `0.5 · log1p(v) / log1p(v_max)`; `normalised` is `0.5 · v / v_max`; `uncapped`
is `0.1 v`.

256 documents:

| method | agents | capped | uncapped | log | normalised |
|---|---|---|---|---|---|
| C | 96 | 0.389 | **0.556** | 0.333 | 0.167 |
| C | 192 | 0.278 | **0.667** | 0.389 | 0.389 |
| C | 384 | 0.000 | **0.667** | 0.333 | 0.000 |
| G | 96 | **0.722** | 0.000 | 0.667 | 0.444 |
| G | 192 | 0.500 | 0.000 | 0.500 | 0.500 |
| G | 384 | **0.389** | 0.000 | 0.389 | 0.222 |

1024 documents:

| method | agents | capped | uncapped | log | normalised |
|---|---|---|---|---|---|
| C | 96 | 0.056 | 0.000 | **0.333** | 0.167 |
| C | 192 | 0.000 | 0.000 | **0.333** | 0.333 |
| C | 384 | 0.000 | **0.278** | 0.056 | 0.056 |
| G | 96 | **0.667** | 0.000 | 0.611 | 0.500 |
| G | 192 | 0.333 | 0.000 | 0.389 | **0.556** |
| G | 384 | 0.333 | 0.000 | 0.389 | 0.167 |

### 14.3 Facts: two undocumented defects in the pheromone code

Both verified by experiment, both properties of the shipped source.

1. **A trail is followable from one endpoint only.** `deposit_pheromones` stores
   the key as `tuple(sorted((i, j)))`; `calculate_pheromone_force` matches
   `doc_a == current_doc.id`, i.e. the lower id. Direct probe: with one trail
   `(0, 2)` at strength 0.1, an agent on document 0 feels `|F_pher| = 0.1000` and
   an agent on document 2 feels `0.0000`. Document ids are assigned in corpus
   insertion order, so load order decides which half of the signal is usable.

2. **The "last three visited documents" are not the last three.**
   `agent.visited_docs` is a `set`, so `list(...)[-3:]` returns three entries in
   hash order. For the visit sequence `7, 3, 91, 12, 5, 44, 2` the code deposits
   against `[12, 44, 91]`; the genuinely recent three are `[5, 44, 2]`.

Repairing them, 256 documents:

| method | agents | as built | symmetric | recency | both |
|---|---|---|---|---|---|
| C | 96 | **0.389** | 0.056 | 0.389 | 0.000 |
| C | 192 | **0.278** | 0.000 | 0.222 | 0.000 |
| C | 384 | 0.000 | 0.000 | 0.000 | 0.000 |
| G | 96 | **0.722** | 0.389 | 0.611 | 0.278 |
| G | 192 | 0.500 | 0.222 | 0.333 | **0.556** |
| G | 384 | **0.389** | 0.111 | 0.333 | 0.278 |

### 14.4 Interpretation

1. **The visit count is MCMP's entire ranking advantage.** Twelve cells, two
   methods, three agent counts, and recall is 0.000 in every one without the visit
   term. This is the answer to a question this report has circled since section 9:
   what does the walk actually buy? It buys visits, and visits are the only channel
   through which the walk reaches the ranking. Cosine similarity cannot lift a far
   document into the top 8 — by construction, since the fixture defines "far" as
   low cosine similarity.

2. **The channel saturates at five visits, and that is a mechanism for both
   documented collapses.** `min(0.1 v, 0.5)` is constant above five visits. Denser
   walks push more documents past the ceiling; past it, the only discriminating
   signal MCMP has is a tie. Section 10's collapse under agent count and section
   13's collapse under corpus size are then the same failure reached two ways, and
   the earlier framing — "the relevance ordering runs over the whole working set" —
   was the symptom rather than the cause.

3. **Uncapping repairs full-corpus MCMP at 256 documents and destroys the bounded
   frontier.** For C the agent-driven collapse disappears: 0.556 / 0.667 / 0.667
   against 0.389 / 0.278 / 0.000, i.e. 0.000 becomes 0.667 at 384 agents. For G it
   is 0.000 in all six cells, because inside a 12-document working set an unbounded
   visit term swamps similarity and the ranking becomes "most trafficked", which in
   a frontier seeded from the FAISS pool means the pool. The correct shape of this
   term depends on the size of the set being scored.

4. **Log compression is the only shape that never fails.** It matches the cap
   where the cap works and beats it where the cap saturates: 0.333 against 0.056
   and 0.000 for C at 1024 documents. It is below the cap by one document in one
   seed in the sparse cells. On this evidence it is the variant worth testing on
   real data — not the variant to ship.

5. **Repairing a genuine defect made retrieval worse, and the ceiling explains
   why.** Eleven of twelve cells are at or below the shipped behaviour. More trail
   signal concentrates the walk; a concentrated walk pushes more documents past the
   five-visit ceiling; past the ceiling the ranking signal is gone. The defects
   were an accidental brake on a feedback loop with no brake of its own. The
   repairs are therefore **blocked on the visit term**, not independently
   shippable. That they would compose well with a non-saturating term is a
   prediction, and it has not been tested.

6. **The unsolved problem from sections 10 and 13 now has a named cause.** It was
   recorded as "MCMP's scoring is the unsolved part". The scoring's problem is a
   hard ceiling on the only informative term. That is a smaller and more tractable
   statement than the one it replaces.

### 14.5 Non-claims

Synthetic fixtures throughout, with a chain placed there by design. Six seeds, one
query, three relevant documents — resolution 0.056. Two corpus sizes, three agent
counts, `steps` held at 50, frontier parameters at their defaults. The exploration
term has never been ablated. The alternative visit terms were not tuned; `log` and
`normalised` reuse the shipped 0.5 ceiling for comparability, not because 0.5 is
right. The combination of repaired pheromone code with a non-saturating visit term
was not run. No production code was changed on the strength of any of this, and
nothing here says real code retrieval behaves this way — that remains Gate 2's
question, and Gate 2 remains unrun.

---

## 15. The visit term, swept: weight beats shape, and the damage was in the steering

Section 14 identified the visit term as MCMP's whole ranking signal and its
five-visit ceiling as the mechanism behind both collapses. It then compared four
*shapes* of the term at a single weight. **That weight, 0.5, was held fixed in
every measurement in section 14, and it was the dominant variable.** Two of
section 14's conclusions do not survive being swept, and are corrected in 15.5.

**Declared design, fixed before execution:** fixtures `manifold` (256 and 1024
documents) and `neutral`, seeds 1-6, `agents in {96, 192, 384}`, `steps 50`,
`top_k = initial_k = 8`, methods C and G, shapes
`{capped, uncapped, log, rank, visited_rank, normalised}`,
`alpha in {0.1, 0.25, 0.5, 1.0, 2.0, 4.0}`, coupled and decoupled. All results
reported. Reproduce with `benchmarks/probes/visit_term.py`, experiments `alpha`,
`confirm`, `decouple`, `final`.

`neutral` is run as a control throughout: its relevant documents are drawn from
the FAISS top-16, so similarity is the correct signal there and a visit term that
overrides similarity must *hurt*. Resolution on neutral is 1/24 = 0.042 (four
relevant documents, six seeds); on manifold it is 1/18 = 0.056.

### 15.1 Facts: sweeping the weight

manifold, 1024 documents, 192 agents, coupled:

| shape | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| C capped | 0.000 | | | | | |
| C uncapped | 0.000 | | | | | |
| C log | 0.000 | 0.000 | 0.333 | 0.500 | 0.444 | 0.389 |
| C rank | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C visited_rank | 0.000 | 0.000 | 0.000 | 0.278 | 0.333 | 0.333 |
| C normalised | 0.000 | 0.000 | 0.333 | 0.778 | **1.000** | 0.500 |
| G capped | 0.333 | | | | | |
| G log | 0.000 | 0.222 | 0.389 | 0.667 | **1.000** | **1.000** |
| G rank | 0.000 | 0.167 | 0.389 | **1.000** | 0.667 | **1.000** |
| G normalised | 0.000 | 0.222 | 0.556 | 0.778 | **1.000** | 0.833 |

The neutral control at the same setting separates the shapes, which manifold does
not:

| shape | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| C capped | 0.458 | | | | | |
| C log | 0.542 | 0.458 | 0.458 | 0.417 | 0.375 | 0.375 |
| C rank | 0.542 | 0.542 | 0.458 | 0.417 | 0.417 | 0.375 |
| C visited_rank | 0.500 | 0.458 | 0.417 | 0.375 | 0.333 | 0.333 |
| C normalised | 0.542 | 0.542 | 0.542 | 0.542 | 0.500 | 0.542 |

### 15.2 Facts: across the agent budget

manifold 1024, coupled, against the shipped ceiling:

| method | agents | capped | normalised a=1 | normalised a=2 |
|---|---|---|---|---|
| C | 96 | 0.056 | 0.389 | 0.167 |
| C | 192 | 0.000 | 0.778 | **1.000** |
| C | 384 | 0.000 | 0.667 | **1.000** |
| G | 96 | **0.667** | 0.556 | 0.111 |
| G | 192 | 0.333 | 0.778 | **1.000** |
| G | 384 | 0.333 | 0.667 | **1.000** |

### 15.3 Facts: separating steering from ranking

`relevance_score` is read twice: by the attraction force through its `(1 + r)`
weight, and by the harness as the final ranking. Decoupled runs steer with the
shipped term for every step but the last, and apply the replacement only to the
final scoring. manifold, 192 agents:

| corpus | method | shape | coupled | decoupled |
|---|---|---|---|---|
| 256 | C | uncapped | 0.667 | **1.000** |
| 256 | G | uncapped | 0.000 | **1.000** |
| 1024 | C | uncapped | 0.000 | 0.722 |
| 1024 | G | uncapped | 0.000 | **1.000** |

`uncapped` decoupled beats the shipped ceiling in all twelve manifold cells,
including the sparse regime where every coupled candidate fails (G at 96 agents,
1024 documents: 1.000 against 0.667). On the control it does systematic damage
that grows with the agent count:

| method | agents | capped | uncapped decoupled |
|---|---|---|---|
| C | 96 | 0.417 | 0.375 |
| C | 192 | 0.458 | 0.333 |
| C | 384 | 0.542 | **0.292** |

### 15.4 Facts: the candidate that has both properties

`normalised` at alpha = 2.0, decoupled, against the shipped ceiling:

| fixture | method | agents | capped | normalised a=2 decoupled |
|---|---|---|---|---|
| manifold 256 | C | 96 | **0.389** | 0.167 |
| manifold 256 | C | 192 | 0.278 | **0.778** |
| manifold 256 | C | 384 | 0.000 | **1.000** |
| manifold 256 | G | 96 | 0.722 | 0.722 |
| manifold 256 | G | 192 | 0.500 | **1.000** |
| manifold 256 | G | 384 | 0.389 | **0.944** |
| manifold 1024 | C | 96 | 0.056 | **0.278** |
| manifold 1024 | C | 192 | 0.000 | **0.556** |
| manifold 1024 | C | 384 | 0.000 | **1.000** |
| manifold 1024 | G | 96 | 0.667 | **0.722** |
| manifold 1024 | G | 192 | 0.333 | **1.000** |
| manifold 1024 | G | 384 | 0.333 | **1.000** |
| neutral | C | 96 | 0.417 | **0.458** |
| neutral | C | 192 | 0.458 | **0.542** |
| neutral | C | 384 | 0.542 | 0.542 |
| neutral | G | 96 | 0.458 | **0.500** |
| neutral | G | 192 | 0.500 | **0.542** |
| neutral | G | 384 | **0.583** | 0.500 |

Ten of twelve manifold cells improve, one ties, one is worse. Four of six control
cells improve, one ties, one is worse by two documents.

### 15.5 Corrections to section 14

1. **"Log compression is the only shape that never fails" was an artifact of
   alpha = 0.5.** At alpha >= 1 several shapes reach 1.000 on the corpus where the
   shipped ceiling returns nothing, and `log` is not the best of them. Section 14.4
   point 4 is withdrawn.

2. **"The correct shape depends on the size of the set being scored" was the wrong
   axis.** Section 14.4 point 3 read the C/G split as a working-set effect. The
   sweep shows the split is mostly a weight effect: at alpha = 2 the same shape wins
   in both. What genuinely differs by shape is behaviour on the *control*, which
   section 14 did not measure at all.

3. This is the sixth conclusion in this report drawn from a parameter held fixed
   outside the regime it was generalised over, and the third that was mine.

### 15.6 Interpretation

1. **The damage was in the steering, not the ranking.** `uncapped` was dismissed in
   section 14 as swamping similarity. It does not: applied to the ranking alone it
   is the strongest signal measured, 1.000 in three of four cells. Coupled, it
   destroys the walk, because feeding an aggressive visit term back through
   `(1 + r)` is positive feedback that collapses the colony onto what it has
   already visited. The two jobs `relevance_score` does are in conflict, and
   separating them is a structural fix that costs no parameter.

2. **Two properties are needed, and no single tested candidate has both cleanly.**
   Decoupling makes a strong ranking signal usable. Boundedness protects the
   control — an unbounded term ranks by traffic alone, which is wrong wherever
   similarity is right. `uncapped` decoupled has the first and not the second;
   coupled `normalised` has the second and not the first. `normalised` at
   alpha = 2 decoupled has both, and is the recommendation this evidence supports.

3. **Its limitation is the sparse regime.** The one manifold cell where it loses is
   C at 96 agents on 256 documents. With few agents the visit distribution is noisy,
   and weighting noise at twice the similarity is worse than the ceiling. A weight
   that scales with walk density is the obvious next step and is untested.

4. **The agent-count collapse is inverted, not merely mitigated.** The shipped term
   goes 0.389 to 0.278 to 0.000 as agents grow. `normalised` at alpha = 2 goes 0.333
   to 0.889 to 1.000. More agents now help. Section 10's collapse and section 13's
   collapse were the same ceiling reached two ways, and lifting the ceiling
   addresses both.

5. **`rank` failed for method C in every cell of every run — 18 of 18 — and it was
   predicted.** A spread diagnostic run before the sweep showed the empirical CDF
   collapses when almost nothing is visited: with 990 of 1000 documents unvisited,
   all visited documents share the top percentile and score within 0.005 of each
   other. The prediction held exactly. `visited_rank` was then derived to fix that,
   and it is the worst shape on the control (0.250 at alpha = 4, 384 agents) — the
   derivation optimised for maximum spread, which is not the objective.

### 15.7 Non-claims

Synthetic fixtures. Six seeds, one query; resolution 0.056 on manifold and 0.042
on neutral, so single-cell differences of that size are one or two documents.
`steps` held at 50 throughout, frontier parameters at their defaults, the
exploration term still never ablated. The decoupling is implemented by applying
the replacement on the final relevance call only, which is a clean separation for
this harness but is not the same as a production design that maintains two scores
throughout. The combination of these findings with the section 14.3 pheromone
repairs has not been run. **No production code was changed, and none is proposed
on this evidence**: a fixture with a deliberately planted chain is grounds for
testing this on real data, not for shipping it. Gate 2 remains unrun.

---

## 16. Chain reachability: the sweep the Gate 2 spec makes binding

`docs/superpowers/specs/2026-08-30-mcmp-gate2-design.md` blocks the production
stage 1 run on a `knn_k`/`max_hops` sweep, because on the corpora measurable at
the time reachability given "far" was 1.000 for real and permuted labels alike,
which makes the manifold signature numerically identical to the far rate. This
section runs that sweep on the Gate 1 fixtures. It does **not** choose the
production operating point — that must come from the production snapshot — but it
supplies the criterion, and it shows the criterion works.

**Declared design, fixed before execution:** fixtures `manifold` (256 documents)
and `neutral`, seeds 1-6, `top_k = 8`, `hop_threshold = 0.0`,
`knn_k in {2, 3, 4, 6, 8, 12}`, `max_hops in {1, 2, 3, 4, 6}`, 10 permutations per
seed. All cells reported. Reproduce with `benchmarks/probes/reachability.py`.

**The reported quantity is not raw reachability.** It is `reach_given_far` for the
real labels minus the same statistic under the permutation null. A dense graph
reaches everything, related or not; what makes reachability a measurement is
whether it separates real relevance from redrawn relevance. Normalising over far
pairs rather than all pairs keeps the two conditions the signature multiplies
apart: the signature can fall because fewer documents are far, this cannot.

### 16.1 Facts

manifold, 256 documents — real / null / gap:

| knn_k | hops=1 | hops=2 | hops=3 | hops=4 | hops=6 |
|---|---|---|---|---|---|
| 2 | 0.000/0.000/+0.000 | 0.000/0.000/+0.000 | 0.000/0.000/+0.000 | 0.000/0.003/-0.003 | 0.667/0.017/**+0.649** |
| 3 | 0.000/0.000/+0.000 | 0.000/0.000/+0.000 | 0.083/0.006/+0.077 | 0.528/0.014/+0.513 | 1.000/0.021/**+0.979** |
| 4 | 0.000/0.000/+0.000 | 0.000/0.003/-0.003 | 0.667/0.023/+0.644 | 1.000/0.026/**+0.974** | 1.000/0.040/+0.960 |
| 6 | 0.000/0.003/-0.003 | 0.611/0.046/+0.565 | 1.000/0.084/**+0.916** | 1.000/0.151/+0.849 | 1.000/0.369/+0.631 |
| 8 | 0.000/0.021/-0.021 | 0.889/0.084/+0.804 | 1.000/0.233/+0.767 | 1.000/0.416/+0.584 | 1.000/0.483/+0.517 |
| 12 | 0.000/0.032/-0.032 | 1.000/0.212/+0.788 | 1.000/0.423/+0.577 | 1.000/0.479/+0.521 | 1.000/0.483/+0.517 |

neutral (no planted chain) — the same:

| knn_k | hops=1 | hops=2 | hops=3 | hops=4 | hops=6 |
|---|---|---|---|---|---|
| 2 | 0.000/0.005/-0.005 | 0.028/0.014/+0.013 | 0.028/0.022/+0.006 | 0.028/0.024/+0.003 | 0.028/0.034/-0.006 |
| 3 | 0.083/0.022/+0.061 | 0.111/0.046/+0.065 | 0.111/0.106/+0.005 | 0.167/0.168/-0.001 | 0.372/0.341/+0.031 |
| 4 | 0.083/0.029/+0.054 | 0.200/0.089/+0.111 | 0.364/0.228/+0.136 | 0.514/0.440/+0.074 | 0.900/0.846/+0.054 |
| 6 | 0.083/0.057/+0.026 | 0.378/0.247/+0.130 | 0.706/0.678/+0.027 | 1.000/0.955/+0.045 | 1.000/0.994/+0.006 |
| 8 | 0.150/0.076/+0.074 | 0.711/0.438/**+0.273** | 0.967/0.944/+0.023 | 1.000/1.000/+0.000 | 1.000/1.000/+0.000 |
| 12 | 0.225/0.128/+0.097 | 0.925/0.717/+0.208 | 1.000/1.000/+0.000 | 1.000/1.000/+0.000 | 1.000/1.000/+0.000 |

### 16.2 Interpretation

1. **Saturation is real for the labels and not for the null, on this corpus.** At
   the spec's defaults (`knn_k = 8`, `max_hops = 6`) the real reachability is
   1.000, but the null is 0.483, not 1.000. The spec's observation — reachability
   1.000 "for real and permuted labels alike" — was made on the 11- and
   249-document Gate 2 corpora; it does not hold on the 256-document manifold
   fixture. The comparison there has content even at the defaults. It is simply
   much weaker than it needs to be.

2. **A denser graph inflates the null, not the signal.** Across the manifold table
   the real column reaches 1.000 and stays there while the null climbs from 0.021
   at `knn_k = 3` to 0.483 at `knn_k = 8`. The defaults therefore discard about
   half the available separation: +0.517 where +0.979 is on the table. This is the
   opposite of the intuition that a well-connected graph makes reachability easier
   to establish — it makes it easier for *everything*, which is precisely what
   destroys it as evidence.

3. **The procedure validates against a structureless control.** At the best
   manifold cell (`knn_k = 3`, `max_hops = 6`) the gap is +0.979 with a planted
   chain and +0.031 without one — a thirty-fold separation. `knn_k = 6`,
   `max_hops = 3` is nearly as good on both counts (+0.916 and +0.027) with a less
   fragile graph. A statistic that could not tell those two fixtures apart would
   be unusable on a production corpus regardless of what it reported there.

4. **A gap can appear without structure, so "largest gap" is not a safe
   criterion.** neutral at `knn_k = 8`, `max_hops = 2` shows +0.273 with nothing
   planted to find. Selecting the operating point by maximising the gap *on the
   production corpus* is post-hoc selection over a grid of thirty cells, which is
   the garden of forking paths, not a measurement. The spec already forbids
   choosing the parameter after seeing its output; this quantifies how much room
   that choice would have.

5. **`max_hops` does not transfer from synthetic data.** On manifold the real
   column jumps from 0.000 to 1.000 over one or two hop steps, and where that jump
   sits is set by the fixture's chain length, which is 8 links by construction.
   Real code has no such known length. `knn_k` is the parameter this evidence
   speaks to; `max_hops` must be swept on the production corpus.

### 16.3 What this changes for Gate 2

The criterion can now be stated before the production sweep is seen, which is what
the spec requires:

- The operating point is chosen to maximise the **real-minus-null reachability
  gap**, not raw reachability, and not the signature.
- `knn_k` is drawn from the small end. On this evidence 3 to 6 is where the null
  stays low; the spec's default of 8 is where it starts inflating.
- Because point 4 makes free selection over the grid unsound, the production
  sweep must be **split-sample**: choose `(knn_k, max_hops)` on one half of the
  manifest's queries, and run stage 1 for the record on the other half. The full
  sweep is reported either way.

### 16.4 Non-claims

Synthetic fixtures, one of them built to contain the structure being detected.
Six seeds, 10 permutations per seed — the null estimates carry more noise than the
spec's 100-permutation standard, which is acceptable for choosing a criterion and
would not be for a gate decision. `hop_threshold` was held at 0.0 throughout, so
the whole table is a pure graph property and says nothing about similarity-gated
hops. Nothing here is a Gate 2 result: the production snapshot does not exist, and
this sweep does not create it.

---

## 17. Gate 2 stage 1, run on real code: the gate does not open

This is the first measurement in this report taken on real data. Every section
before it is synthetic.

### 17.1 How it was made possible, and what that costs

The production route is dead at the account level, not the infrastructure level.
On 2026-09-01 the `embedding-service` image was built and started standalone: it
reports `{"status":"ok","model":"text-embedding-3-large"}` and the first real call
returns `429 insufficient_quota — "You have no credits remaining."` The service is
a thin wrapper over OpenAI's embeddings API with no local model, so **Docker was
never the blocker; OpenAI credit is.**

The snapshot was therefore built from a locally cached model. Two were available:
`Qwen/Qwen3-Embedding-0.6B` (1024-dim) and `sentence-transformers/all-MiniLM-L6-v2`
(384-dim). Qwen measured at 37 minutes for 40 documents on this contended host —
55 s/document with torch already using 24 of 32 threads — against 33 s for the
same 40 with MiniLM, including model load. MiniLM was used.

**What that costs, stated plainly.** This is a measurement of *this* embedding
space. A 384-dimensional general-purpose sentence model is a weaker instrument
than the 3072-dimensional production one, and it may under-resolve structure that
is there. A negative result here does not establish a negative result for
production, and `build_service_snapshot` records `backend="local-transformers"`
and the model id so no reader can mistake one for the other.

### 17.2 Facts

Corpus `src/embeddinggemma/`, manifest `embeddinggemma-local-v1`, digest
`114b6a66…`: **238 documents** (functions, methods, classes by AST), 156 query
candidates, 11 ambiguous call names discarded fail-closed.

Split-sample per section 16.3, criterion fixed before the run: 23 selection seeds,
23 evaluation seeds, query-disjoint; 2 seeds dropped as unusable or straddling.

Selection half — `reach_given_far`, real / null / gap:

| knn_k | hops=2 | hops=3 | hops=4 | hops=6 | hops=8 |
|---|---|---|---|---|---|
| 2 | 0.082/0.003/+0.080 | 0.094/0.003/+0.092 | 0.094/0.003/+0.091 | 0.094/0.003/+0.091 | 0.094/0.003/+0.091 |
| 3 | 0.094/0.005/+0.089 | 0.106/0.007/+0.099 | 0.118/0.011/+0.106 | 0.118/0.019/+0.099 | 0.118/0.024/+0.094 |
| 4 | 0.129/0.013/+0.117 | 0.200/0.030/+0.170 | 0.212/0.055/+0.157 | 0.271/0.100/+0.170 | 0.294/0.153/+0.142 |
| 6 | 0.176/0.027/+0.149 | 0.306/0.081/+0.225 | 0.376/0.147/+0.230 | 0.471/0.328/+0.143 | 0.647/0.529/+0.118 |
| 8 | 0.271/0.051/+0.220 | 0.412/0.151/**+0.261** | 0.565/0.312/+0.253 | 0.882/0.683/+0.200 | 0.965/0.910/+0.055 |

Operating point selected by the pre-registered rule: `knn_k = 8`, `max_hops = 3`.

Evaluation half, 100-permutation null:

| quantity | value |
|---|---|
| pair_count | 108 |
| far_rate | **0.602** |
| reach_given_far | 0.292 |
| null reach_given_far (median) | 0.155 |
| manifold_signature | 0.176 |
| null median | 0.148 |
| null p95 | 0.204 |
| excess over null median | 0.028 |
| required excess | 0.085 |
| exceeds null p95 | **False** |
| meets absolute minimum | True |
| meets relative excess | **False** |

**STAGE 2 JUSTIFIED: False.**

### 17.3 Interpretation

1. **MCMP's premise survives; its mechanism does not clear the bar.** 60% of
   call-graph neighbours rank deeper than the FAISS top-8. Embedding similarity
   really does miss most call-graph relations, which is exactly the gap MCMP
   exists to close. What fails is the second half: those far documents are
   chain-reachable at 0.292 where redrawn labels reach 0.155. The signal is real
   and roughly two-fold, and it is still too small for the pre-registered gate.

2. **The saturation worry does not apply to this corpus.** Sections 14-16 were
   shaped by reachability measuring 1.000 and collapsing the signature into the
   far rate. Here the real column runs 0.082 to 0.965 across the grid and sits at
   0.412 at the chosen point. The measurement discriminates; the gate closes on
   the evidence, not on a degenerate statistic.

3. **The criterion picked the edge of its own grid, and that is reported rather
   than repaired.** The gap is maximised at `knn_k = 8`, the largest value the
   pre-registered grid allows — and the grid stopped at 8 precisely because
   section 16 found the null inflating above it *on synthetic data*. Real geometry
   disagrees with that synthetic finding. The honest reading is that the true
   optimum may lie outside the grid; widening the grid now and re-running would
   be exactly the post-hoc selection the split-sample protocol exists to prevent.
   It is a finding for the next pre-registration, not a fix for this run.

4. **Two of three conditions fail together, which is the informative pattern.**
   The signature clears the absolute floor (0.176 ≥ 0.10) but misses both the 95th
   percentile (0.176 < 0.204) and the relative excess (0.028 < 0.085). A gate that
   failed only on the absolute floor would mean "too few far pairs to be worth the
   compute"; failing on both null-relative conditions means the structure is not
   distinguishable enough from chance. The Gate 1 cost argument — roughly 14 300x
   FAISS for full-corpus MCMP, 143 224 comparisons for the bounded frontier — has
   nothing to buy here.

### 17.4 Non-claims

One corpus, 238 documents, one embedding space, one model. A 384-dimensional
general-purpose model may simply not resolve code structure that a 3072-dimensional
one would; **this result does not transfer to the production embedding space and
must not be quoted as if it did.** The call graph is resolved fail-closed, so 11
ambiguous names are absent from the relevance oracle and their relations are
invisible to the measurement. `hop_threshold` was held at 0.0 throughout, making
reachability a pure graph property. Stage 2 was not run, so nothing here says what
MCMP's retrieval would have scored — only that the geometry does not justify
spending the compute to find out.

---

## 18. Replication in a stronger embedding space: the same verdict, and a harder finding

Section 17's obvious objection is that its verdict came from a 384-dimensional
general-purpose model, which may simply fail to resolve structure that a better
one would find. This section answers that objection by running the identical
pre-registered protocol on the same manifest in a different space. It is a new
measurement, not a re-run: nothing about the criterion, the grid, the split or
the null was changed, and the earlier result stands as published.

**Why it was possible at all.** `embed_local.py` had `device="cpu"` hard-coded,
which is the entire reason section 17 measured Qwen at 55 s/document and settled
for MiniLM. This host has an RTX 3060 with 12 GB, 1.8 GB in use, and torch sees
it. On CUDA the same 238 documents embed in **61 seconds** — against 37 minutes
for 40 of them on CPU. The model choice in section 17 was forced by a defect in
the measuring apparatus, not by the hardware.

### 18.1 Facts

Same manifest (`114b6a66…`, 238 documents), same split (23/23, 2 dropped), same
grid and criterion. Model `Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions.

Selection half — `reach_given_far`, real / null / gap:

| knn_k | hops=2 | hops=3 | hops=4 | hops=6 | hops=8 |
|---|---|---|---|---|---|
| 2 | 0.000/0.002/-0.002 | 0.000/0.002/-0.002 | 0.000/0.002/-0.002 | 0.000/0.002/-0.002 | 0.000/0.002/-0.002 |
| 3 | 0.000/0.005/-0.005 | 0.013/0.010/+0.003 | 0.013/0.014/-0.002 | 0.013/0.016/-0.003 | 0.013/0.017/-0.004 |
| 4 | 0.025/0.010/+0.016 | 0.038/0.018/+0.020 | 0.038/0.029/+0.009 | 0.038/0.049/-0.011 | 0.063/0.066/-0.003 |
| 6 | 0.089/0.024/+0.065 | 0.114/0.065/+0.049 | 0.165/0.113/+0.051 | 0.278/0.280/-0.001 | 0.506/0.462/+0.045 |
| 8 | 0.177/0.052/+0.125 | 0.278/0.147/+0.132 | 0.544/0.292/**+0.252** | 0.810/0.642/+0.168 | 0.886/0.847/+0.039 |

Operating point: `knn_k = 8`, `max_hops = 4`.

Evaluation half, side by side with section 17:

| quantity | MiniLM (384) | Qwen (1024) |
|---|---|---|
| far_rate | 0.602 | **0.537** |
| reach_given_far | 0.292 | **0.621** |
| null reach_given_far | 0.155 | 0.284 |
| manifold_signature | 0.176 | **0.333** |
| null median | 0.148 | 0.278 |
| null p95 | 0.204 | 0.343 |
| excess over null median | 0.028 | 0.056 |
| required excess | 0.085 | 0.072 |
| exceeds null p95 | False | False |
| meets relative excess | False | False |

**STAGE 2 JUSTIFIED: False**, in both spaces.

### 18.2 Interpretation

1. **The negative result is not an artifact of the weak model.** The stronger
   embedding nearly doubles the signature, 0.176 to 0.333 — and the null rises
   with it, 0.148 to 0.278. Signal and chance move almost in proportion, so the
   excess barely improves (0.028 to 0.056) and the verdict does not change. Two
   independent spaces, one protocol, one answer.

2. **But it is now marginal, and that must be said as plainly as the verdict.**
   The signature misses the 95th percentile by 0.010 — on 108 pairs that is about
   one pair — and the relative excess by 0.017. This is not a comfortable No. A
   third space, or a larger corpus, could plausibly land on the other side. What
   it is not is evidence *for* the mechanism: a gate that closes by one pair has
   still closed, and the pre-registration exists precisely so a near miss cannot
   be read as a pass.

3. **A better embedding shrinks the territory MCMP exists to occupy.** `far_rate`
   falls from 0.602 to 0.537: the stronger model already places more call-graph
   neighbours inside the FAISS top-8, so there is less for a walk to recover. This
   generalises uncomfortably. MCMP's value is bounded above by what the embedding
   misses, and embeddings keep improving. Any case for the mechanism has to be
   made against the best available embedding, not the one that flatters it most.

4. **Both spaces put the operating point at `knn_k = 8`, the grid edge.** Section
   16 derived from synthetic fixtures that the null inflates above `knn_k = 6` and
   capped the grid accordingly. Real geometry contradicts that in both spaces
   independently. In the Qwen space the small end is not merely worse but dead —
   reachability is exactly 0.000 at `knn_k = 2` and 0.013 at `knn_k = 3`, because
   that space's mutual k-NN graph is too sparse at small k to connect anything.
   The synthetic finding does not transfer, and the grid is now known to be
   mis-centred for real data. That is a correction to section 16.3, and it is
   binding on the next pre-registration rather than on this run.

### 18.3 Non-claims

Still one corpus and one manifest; the two spaces share every relevance label, so
these are not independent samples of code, only of embedding. Neither model is the
3072-dimensional production one, which remains unmeasured and unreachable while
the OpenAI account has no credit. The `knn_k = 8` grid edge means neither run
observed its own optimum. `hop_threshold` was 0.0 throughout. Stage 2 has still
never run, in any space.

---

## 19. The crawler framing, tested on real code: the walk finds nothing the pool did not have

Sections 17 and 18 closed the retrieval gate. The crawler framing is a different
question and the gate does not speak to it: ranking is not involved, so the visit
ceiling cannot bite, and cost is not involved either, because a crawl is batch
work where the ~14 300x that sinks MCMP as an interactive retriever is affordable.
The mechanism would earn its keep on a **non-empty difference** — call-graph
neighbours the walk visits that the FAISS pool did not already contain.

**This is not Gate 2 stage 2 and no retrieval claim is drawn from it.**

**Declared design:** manifest `embeddinggemma-local-v1` (238 documents), Qwen
snapshot (1024-dim), 12 seeds, `top_k = initial_k = 8`, `steps 50`, method G. The
measured quantity is `document_visits`, the set the colony actually stood on —
not its ranking. Reproduce with `benchmarks/probes/crawler.py`.

### 19.1 Facts

Means over 12 seeds, per query:

| agents | frontier | relevant | in FAISS pool | visited | relevant visited | **novel relevant** | walk comparisons |
|---|---|---|---|---|---|---|---|
| 96 | every 10, k 4, cap 64 | 1.75 | 0.67 | 8.7 | 0.58 | **0.00** | 154 253 |
| 192 | every 10, k 4, cap 64 | 1.75 | 0.67 | 9.8 | 0.67 | **0.00** | 317 165 |
| 384 | every 10, k 4, cap 64 | 1.75 | 0.67 | 9.6 | 0.67 | **0.00** | 634 349 |
| 192 | every 2, k 12, cap 200 | 1.75 | 0.67 | 13.8 | 0.67 | **0.00** | 1 331 789 |

### 19.2 Interpretation

1. **The walk reaches exactly what the pool already had, and nothing else.**
   `relevant visited` equals `in FAISS pool` at 0.67 in every configuration. Not
   one call-graph neighbour outside the pool was visited, at any agent count, with
   the frontier at its defaults or opened five-fold.

2. **The targets exist and are missed.** This is what makes the result decisive
   rather than vacuous. Each query has 1.75 call-graph neighbours on average and
   the pool holds 0.67 of them, so roughly **1.08 relevant documents per query lie
   outside the pool** — exactly the territory the mechanism is for. The walk visits
   9 to 14 documents and none of them is one of those.

3. **Opening the frontier changes the cost, not the outcome.** Section 13 found G
   adding exactly 4.0 documents at every corpus size because the round count binds
   rather than the cap; the fair crawler test therefore ran `expand_every 2`,
   `expand_k 12`, `cap 200`. The visited set grew from 9.8 to 13.8 documents and
   the comparison count from 317 165 to 1 331 789 — a 4.2x cost for four more
   documents, none of them relevant. The untested regime section 13.5 flagged as
   its narrowest point is now tested, and it does not rescue the method.

4. **This closes the crawler question on this evidence.** The framing was the
   strongest available for MCMP: it uses the half that works on synthetic data and
   discards the half that does not, and it removes the cost objection. On real code
   the half that "works" does not transfer. The colony traverses a planted chain in
   a synthetic manifold and does not traverse a call graph.

### 19.3 Non-claims

One corpus, one manifest, one embedding space, 238 documents. The relevance oracle
is the call graph, so semantic neighbours without a call edge — the category
sections 4 and the crawler discussion identified as MCMP's remaining niche — are
invisible here by construction, and a walk that found them would score zero on
this measurement. That niche is untested and this section does not touch it; what
it rules out is the call-graph case, which is the one with a free oracle. Method G
was the only walk measured; full-corpus C was not run, on the section 13 finding
that it does not survive scale.

---

## 20. The niche sections 17-19 were blind to: a real, small, above-chance effect

Sections 17-19 used the call graph as the relevance oracle and closed on it. All
three are blind by construction to the category MCMP's remaining case rests on:
documents that belong together while neither calls the other. Section 19.3 says a
walk that found exactly those would have scored zero there. **This section removes
that excuse rather than leaving it as a defence**, and the answer is not the one
the preceding three sections would predict.

**The sibling relation.** Two documents are siblings when they share at least *n*
callees or at least *n* callers **and** neither calls the other. Direct call-graph
neighbours are excluded outright, so this oracle and section 17's are disjoint by
construction and this cannot re-measure the same thing under a new name. It is
mechanical, derived from the manifest already committed, and requires no judgement.

**Both thresholds were run and both are reported.** They disagree on one condition
and that disagreement is part of the result, not something to select away.

### 20.1 Facts

Qwen snapshot (1024-dim), manifest `embeddinggemma-local-v1`, `top_k = 8`,
`knn_k = 8`, `max_hops = 4`, 100-permutation nulls.

| | shared ≥ 2 | shared ≥ 1 |
|---|---|---|
| documents with siblings | 34 | 128 |
| usable seeds | 24 | 48 |
| pair_count | 199 | 688 |
| far_rate | 0.794 | 0.823 |
| reach_given_far | 0.462 | 0.371 |
| null reach_given_far | 0.358 | 0.264 |
| signature | 0.367 | 0.305 |
| null median / p95 | 0.347 / 0.402 | 0.254 / **0.283** |
| **exceeds null p95** | False | **True** |
| meets absolute minimum | True | True |
| meets relative excess | False | False (0.051 vs 0.075) |

Crawl, `agents 192`, `expand_every 2`, `expand_k 12`, `cap 200`:

| | shared ≥ 2 | shared ≥ 1 |
|---|---|---|
| relevant per query | 4.29 | 6.46 |
| of those in FAISS pool | 0.50 | 1.48 |
| documents visited | 11.5 | 11.9 |
| relevant visited | 0.75 | 1.75 |
| **novel relevant** | **0.250** | **0.271** |
| novel under permuted labels | 0.062 | 0.137 |
| **ratio** | **4.03x** | **1.97x** |

### 20.2 Interpretation

1. **The walk finds something here, and it did not on the call graph.** Section 19
   measured novel relevant at exactly 0.00 in four configurations. On siblings it
   is 0.25 to 0.27 against a permuted-label chance level of 0.06 to 0.14 — two to
   four times chance. `relevant visited` exceeds `in FAISS pool` in both, which
   never happened in section 19. The category that oracle could not see is a
   category where the mechanism does something.

2. **The effect is real and small.** At threshold 2 the walk recovers 0.25 of the
   3.79 relevant documents lying outside the pool — about 7% — for roughly 1.3
   million comparisons. Four times chance on a small base is still a small
   absolute number: six novel finds across 24 queries. Nothing here says this pays
   for itself; it says the mechanism is not inert on this relation.

3. **The geometry significance test passes once, at the larger sample.** At
   threshold 1 the signature exceeds the null's 95th percentile (0.305 > 0.283) —
   the first time any pre-registered significance condition has passed on real
   data. It fails at threshold 2 (0.367 < 0.402), where the sample is a third the
   size and the null correspondingly wider. The pattern is consistent with a real
   effect being resolved by the larger sample rather than with noise, but two
   thresholds is not a sample of thresholds and this is stated as suggestive.

4. **The full gate still does not open, and on a different condition than before.**
   Both thresholds fail the relative-excess condition (0.051 against a required
   0.075 at threshold 1). Sections 17 and 18 failed the significance condition
   *and* the excess; here the significance condition passes and the effect size
   does not. That is a materially different failure: the structure is
   distinguishable from chance, and the margin over chance is too small to carry
   the compute.

5. **This corrects the summary of sections 17-19, not their contents.** Those
   measurements stand exactly as published. What changes is what may be concluded
   from them: "MCMP does not transfer to real code" was too broad. The supported
   statement is narrower — **it does not transfer to the call-graph relation, and
   on the sibling relation it produces a small above-chance effect that still
   falls short of the pre-registered bar.**

### 20.3 Non-claims

One corpus, one embedding space, one manifest. Two thresholds were run and both
are reported; the p95 pass appears at one of them, so it is a single observation
and not a replication. The crawl null redraws labels while holding the walk and
the pool fixed, which is the right null for "did the walk find these on purpose",
and it does not control for the walk preferring documents that are large, central
or heavily connected for reasons unrelated to relevance. `knn_k = 8` and
`max_hops = 4` were carried over from section 18's split-sample selection, which
was performed against the *call-graph* oracle — they are not tuned for this
relation, and no sweep was run here. Stage 2 has still never run.

---

## 21. The sibling relation under its own pre-registered selection: the gate closes on a tie

Section 20.3 recorded that its `knn_k` and `max_hops` were carried over from
section 18's split-sample selection, which was performed against the *call-graph*
oracle and is not tuned for this relation. This section gives the sibling relation
its own selection under the same protocol — a new pre-registration for a different
relation, not a re-run of an existing one.

**Declared before the run:** same criterion as section 16.3 (maximise the
real-minus-null `reach_given_far` gap; ties to smaller `knn_k`, then smaller
`max_hops`), same grid, same 100-permutation null, split-sample on
query-disjoint halves. Oracle: siblings sharing ≥1 callee or caller with no direct
call edge. 64 seeds offered, 30 evaluation seeds used, 1 dropped.

### 21.1 Facts

Selection half, `reach_given_far` real / null / gap:

| knn_k | hops=2 | hops=3 | hops=4 | hops=6 | hops=8 |
|---|---|---|---|---|---|
| 2 | 0.011/0.002/+0.009 | 0.014/0.003/+0.011 | 0.014/0.003/+0.010 | 0.014/0.004/+0.010 | 0.014/0.004/+0.010 |
| 3 | 0.035/0.004/+0.031 | 0.041/0.006/+0.035 | 0.041/0.006/+0.034 | 0.041/0.008/+0.033 | 0.041/0.008/+0.033 |
| 4 | 0.035/0.008/+0.027 | 0.046/0.015/+0.031 | 0.049/0.024/+0.025 | 0.079/0.044/+0.035 | 0.092/0.065/+0.028 |
| 6 | 0.095/0.034/+0.061 | 0.182/0.077/+0.105 | 0.277/0.146/+0.131 | 0.467/0.324/**+0.144** | 0.622/0.520/+0.102 |
| 8 | 0.128/0.054/+0.074 | 0.231/0.133/+0.098 | 0.416/0.248/**+0.168** | 0.628/0.547/+0.080 | 0.796/0.753/+0.043 |

Operating point: `knn_k = 8`, `max_hops = 4`.

Evaluation half:

| quantity | value |
|---|---|
| pair_count | 138 |
| far_rate | 0.587 |
| reach_given_far | 0.593 |
| null reach_given_far (median) | 0.289 |
| manifold_signature | 0.34782608695652173 |
| null p95 | 0.34782608695652173 |
| excess over null median | 0.07246376811594202 |
| required excess | 0.07246376811594203 |

**STAGE 2 JUSTIFIED: False** — and the two conditions fail in different ways that
must not be conflated.

- **The significance condition ties exactly.** `signature == null_p95` is `True`
  as a float comparison: both are 48/138. The pre-registration says `>`, and a tie
  decides against the hypothesis. This is the rule working as designed.
- **The effect-size condition fails by one unit in the last place.** Excess and
  required differ by `-1.39e-17`, which is floating-point noise on two routes to
  the same number: with this null median, `0.10 × (1 − median)` happens to equal
  `signature − median` exactly. Mathematically the condition is *met*; the `>=`
  comparison on independently computed floats reports otherwise.

### 21.2 Interpretation

1. **The gate closes on a tie, not on a shortfall.** Under its own pre-registered
   selection the sibling relation lands exactly on the bar, to sixteen decimal
   places on one condition and to within one ULP on the other. That is a materially
   different statement from sections 17 and 18, where the signature missed by 0.010
   and 0.028 respectively.

2. **The verdict does not change if the floating-point defect is repaired.** The
   effect-size comparison should use a tolerance; comparing two independently
   computed floats with `>=` is a real defect and it is recorded here as one. But
   even granting that condition, the significance condition still fails on an exact
   tie, so the gate outcome stands. **Nothing is changed in the gate on the strength
   of a run it has already decided** — that is precisely the post-hoc move the
   pre-registration exists to prevent. It is a defect for the next
   pre-registration to fix, and it is written down so the next reader does not have
   to rediscover it.

3. **The statistic's resolution is coarser than the margin it demands.** With 138
   pairs the signature is quantised at 1/138 ≈ 0.0072, while the required excess is
   0.0725 — ten quantisation steps. A gate deciding at that granularity can be
   flipped by a single pair. The corpus is too small for the bar it is being asked
   to clear, and that is a design finding, not a result: the next run needs more
   pairs, from a larger corpus or more query candidates, before its verdict means
   much either way.

4. **Its own selection did not help the sibling relation.** The chosen point,
   `knn_k = 8` / `max_hops = 4`, is the same one carried over in section 20, and
   the signature moved from 0.305 to 0.348 only because the evaluation half
   differs. Both spaces and both oracles now put the operating point at
   `knn_k = 8`, the grid edge — the fourth independent observation that section
   16.3's synthetic small-`knn_k` finding does not transfer.

### 21.3 Non-claims

One corpus, one embedding space, 138 evaluation pairs. The exact tie is a property
of a small, quantised sample and should not be read as the true effect sitting
precisely at the threshold. Two sibling thresholds have now been run (section 20)
plus this one at threshold 1 with its own selection; all are reported, none is
selected. The float comparison defect is recorded, not repaired. Stage 2 has still
never run, in any space, on any relation.

---

## 22. A second corpus opens the gate, twice — and identifies corpus size as the confound

Section 21.3 named the binding problem: 138 evaluation pairs quantised at 1/138
against a required margin ten steps wide, on a corpus of 238 documents. The fix it
asked for was more pairs from a larger corpus. This section supplies one, and the
answer reverses the verdict of sections 17, 18 and 21.

**Corpus.** `vibemind-os/brain`, 741 Python files, which the AST walk resolves to
**16 497 documents and 10 234 query candidates** — 69 times the previous corpus.
The full corpus is not used: `geometry_cache` holds an N x N similarity matrix, so
16 497 documents cost 1.09 GB per cached dataset and the selection sweep holds one
per seed, which is 33 GB across 30 seeds on a host with a documented history of
RAM exhaustion. Two fixed 4 000-document samples were drawn instead, with the call
graph cut consistently so no relevance set names a document the corpus lacks.

**Both samples were run and both are reported.** Same protocol, same criterion,
same grid, same 100-permutation null, query-disjoint halves.

### 22.1 Facts

| | 238-doc corpus (§18) | brain sample 0 | brain sample 1 |
|---|---|---|---|
| documents | 238 | 4 000 | 4 000 |
| query candidates | 156 | 1 186 | 956 |
| operating point | knn 8 / hops 4 | knn 8 / hops 6 | knn 8 / hops 4 |
| pair_count | 108 | 63 | 85 |
| far_rate | 0.537 | 0.571 | 0.482 |
| reach_given_far | 0.621 | 0.556 | 0.439 |
| **null reach_given_far** | **0.284** | **0.143** | **0.035** |
| manifold_signature | 0.333 | 0.317 | 0.212 |
| null median | 0.278 | 0.143 | 0.035 |
| null p95 | 0.343 | 0.222 | 0.071 |
| excess / required | 0.056 / 0.072 | **0.175 / 0.086** | **0.176 / 0.096** |
| exceeds null p95 | False | **True** | **True** |
| meets absolute minimum | True | **True** | **True** |
| meets relative excess | False | **True** | **True** |
| **STAGE 2 JUSTIFIED** | False | **True** | **True** |

### 22.2 Interpretation

1. **The gate opens on both samples, and not narrowly.** Every pre-registered
   condition passes. The excess clears its requirement by a factor of two on both.
   Section 21's exact tie and sections 17-18's near misses were not the shape of
   the underlying effect; they were the shape of the corpus they were measured on.

2. **The mechanism is the null, not the signal.** `reach_given_far` for the real
   labels is *lower* on the brain samples (0.556, 0.439) than on the 238-document
   corpus (0.621). What changes is chance: the null falls from 0.284 to 0.143 to
   0.035. On a small corpus almost everything is reachable from almost everything,
   so redrawn labels score nearly as well as real ones and the test has no power.
   Enlarging the corpus does not make the structure stronger — it makes coincidence
   rarer, which is what lets the structure show.

3. **This is a confound, and it invalidates the reach of sections 17, 18 and 21 —
   not their contents.** Those measurements are exactly what they say they are, on
   the corpora they name. What cannot be carried forward is the generalisation. The
   summary "MCMP does not transfer to real code" was drawn from a corpus too small
   for the test to detect anything, and the same protocol on a larger one decides
   the other way, twice. **Section 20's "the supported statement is narrower" was
   still not narrow enough.**

4. **A corpus-size sweep is now the obvious missing measurement**, and it is
   missing on purpose rather than by oversight: two sizes is not a sweep, and the
   right response to discovering a size confound is not to pick the size that gives
   the agreeable answer. What the next pre-registration needs is the *smallest*
   corpus at which the null stops saturating, measured before any verdict is read
   off it.

5. **Gate 2 stage 2 is now justified for the first time.** The geometry supports
   spending the compute to find out what MCMP's retrieval actually does on real
   code — the question this entire report has been unable to reach.

### 22.3 Non-claims

63 and 85 evaluation pairs are small samples; the margin is wide relative to the
quantisation, which is why they are reported, but they are not large. One
repository, one embedding space, one model, and neither sample is the production
3072-dimensional space. The 4 000-document samples cut the call graph: edges to
dropped documents are removed, so each sample's graph is sparser than the real
one, and how that interacts with reachability is not measured. The operating point
sits at `knn_k = 8`, the grid edge, for the sixth independent time. Stage 2 has
still not run — this section establishes that it may, not what it will find.
