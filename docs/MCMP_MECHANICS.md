# MCMP: what each part of the algorithm actually does

Two documents already describe how MCMP works.
`docs/mcmp_simulation.md` gives the formulas; `docs/MCMP_FULL_PIPELINE_ANALYSIS.md`
traces the call path. Both predate every measurement in this repository, so
neither says which parts of the mechanism carry retrieval and which are inert or
harmful.

This document is that bridge. It walks the step loop term by term and attaches
the evidence to each one. Everything here is measured, not argued; the source of
each number is named, and the things that were *not* measured are listed at the
end.

Evidence lives in `benchmarks/results/` and
`docs/MCMP_TIG_C004_REPORT.md` sections 9–15. The probes below are reproducible:

```bash
# from the repository root; `embeddinggemma` is not installed in the venv,
# so src/ has to be on the path explicitly
PYTHONPATH=src .venv/Scripts/python.exe -m benchmarks.probes.visit_term \
    --experiment alpha --fixture manifold
PYTHONPATH=src .venv/Scripts/python.exe -m benchmarks.probes.visit_term \
    --experiment alpha --fixture neutral
```

`--experiment` takes `ablate`, `terms`, `defects`, `alpha`, `confirm`, `decouple`
or `final`. The sweeps take roughly ten to twenty minutes each; the ablations are
faster. **Always run the `neutral` fixture alongside `manifold`** — it is the
control, and it is what separates candidates that manifold alone cannot.

Figures are recall@8 over six seeds and one query: three relevant documents on
`manifold` (resolution 1/18 ≈ 0.056) and four on `neutral` (1/24 ≈ 0.042). A gap
that size is one or two documents in one seed. Read the large gaps.

---

## The step loop, term by term

Each step, every agent moves once and deposits once; then the whole working set
is rescored and the trails decay.

```text
F = 0.8 · F_attraction + 0.15 · F_pheromone + 0.05 · F_exploration
v ← 0.85 · v + 0.15 · F
r_i = cosine(q, d_i) + min(0.1 · visits_i, 0.5) + time_bonus + kw_bonus
```

### `0.8 · F_attraction` — the similarity gradient

Four fifths of the movement force pulls the agent toward its five nearest
documents, weighted by similarity and by `(1 + relevance)`. This is a gradient
walk on the embedding, and it is the term that makes MCMP look like FAISS with
extra steps — because for this term, it is.

The relevance weighting matters more than it looks: `relevance` includes the
visit count, so the attraction term is where the walk's history feeds back into
where the walk goes next.

### `0.15 · F_pheromone` — the colony, and it is the mechanism

This is the term that distinguishes MCMP from a random-restart sampler, and
method F was built to test whether it does anything. F is method C with the trail
memory cleared after each deposit — same agents, same steps, same seeds, same
attraction, no colony memory.

| manifold, 12 seeds | C (with pheromone) | F (without) |
|---|---|---|
| recall@8 @ 96 agents | **0.722** | 0.333 |
| chain documents reached @ 96 agents | 2.67 / 3 | **1.00 / 3** |
| chain documents reached @ 192 agents | 3.00 / 3 | **1.00 / 3** |
| comparisons @ 96 agents | 915 456 | 614 464 |

F stops at exactly one chain link and adding agents does not move it. The colony
is what walks a chain; without it the extra agents are just more starts in the
same neighbourhood. The pheromone charges 49% more comparisons for that, because
following a trail costs its own nearest-document lookup.

(Report section 11. This corrected an earlier conclusion in section 10 that the
agents were parallel restarts — that had been measured at 24 agents, below the
density at which the colony engages.)

### `0.05 · F_exploration` — untested

Gaussian noise scaled by the agent's `exploration_factor`. No ablation has been
run on it. It is 5% of the force and nothing in this repository says whether that
matters.

---

## The relevance score, and where the real problem lives

The walk is not the weak part. The scoring is.

### The visit term is the entire ranking signal

`update_document_relevance` adds `min(0.1 × visit_count, 0.5)` to every
document's cosine similarity. Removing that one term — leaving similarity, the
time bonus and the keyword bonus untouched — gives this:

| manifold, 256 docs | with visit term | without |
|---|---|---|
| C, 96 agents | 0.389 | **0.000** |
| C, 192 agents | 0.278 | **0.000** |
| C, 384 agents | 0.000 | **0.000** |
| G, 96 agents | 0.722 | **0.000** |
| G, 192 agents | 0.500 | **0.000** |
| G, 384 agents | 0.389 | **0.000** |

Zero in every cell. Nothing else in the score can lift a far document into the
top 8 — which is unsurprising once stated, since the fixture defines "far" as low
cosine similarity, and cosine similarity is the only other real term. Under the
benchmark's fixed clock the time bonus reduces to a flat +0.1 for "visited at
least once", and even that is not enough.

**So MCMP's entire retrieval advantage over FAISS is carried by the visit
count.** The colony's job is to visit the right documents; the visit term is the
only channel through which that reaches the ranking.

### And that channel saturates

`min(0.1 × visits, 0.5)` reaches its ceiling at five visits. Once a walk is dense
enough that many documents pass five visits, the only discriminating signal MCMP
has becomes a constant and the ranking ties.

That is a mechanism for both collapses this project has documented:

- **collapse under agent count** — more agents, more documents past the ceiling
  (report section 10)
- **collapse under corpus size** — C falls from 0.611 at 64 documents to 0.056 at
  1024 while still *reaching* 2.33 of 3 chain documents (report section 13)

In both cases discovery holds and ranking fails. A saturating signal explains
why.

### Replacing the ceiling

Three shapes that cannot saturate, against the shipped cap:

**256 documents**

| | capped | uncapped | log | normalised |
|---|---|---|---|---|
| C, 96 | 0.389 | **0.556** | 0.333 | 0.167 |
| C, 192 | 0.278 | **0.667** | 0.389 | 0.389 |
| C, 384 | 0.000 | **0.667** | 0.333 | 0.000 |
| G, 96 | **0.722** | 0.000 | 0.667 | 0.444 |
| G, 192 | 0.500 | 0.000 | 0.500 | **0.500** |
| G, 384 | **0.389** | 0.000 | 0.389 | 0.222 |

**1024 documents**

| | capped | uncapped | log | normalised |
|---|---|---|---|---|
| C, 96 | 0.056 | 0.000 | **0.333** | 0.167 |
| C, 192 | 0.000 | 0.000 | **0.333** | 0.333 |
| C, 384 | 0.000 | **0.278** | 0.056 | 0.056 |
| G, 96 | **0.667** | 0.000 | 0.611 | 0.500 |
| G, 192 | 0.333 | 0.000 | 0.389 | **0.556** |
| G, 384 | 0.333 | 0.000 | 0.389 | 0.167 |

> **Both tables above were measured at one weight, and that weight was the
> dominant variable.** Every column here uses α = 0.5. Sweeping it changes the
> ranking of the shapes and lifts the best cells to 1.000. The two conclusions
> originally drawn from these tables — "log compression is the only shape that
> never fails" and "the right shape depends on the working-set size" — are both
> withdrawn. See the next section, and report section 15.5.

What survives from these two tables is narrower: **uncapping helps full-corpus C
and destroys the bounded frontier at α = 0.5**, which turns out to be a fact about
coupling rather than about magnitude — see "The two jobs" below.

---

## The weight, not the shape

Sweeping α on the corpus where the shipped ceiling returns nothing at all
(manifold 1024, 192 agents, coupled):

| shape | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| C capped | 0.000 | | | | | |
| C log | 0.000 | 0.000 | 0.333 | 0.500 | 0.444 | 0.389 |
| C normalised | 0.000 | 0.000 | 0.333 | 0.778 | **1.000** | 0.500 |
| G capped | 0.333 | | | | | |
| G log | 0.000 | 0.222 | 0.389 | 0.667 | **1.000** | **1.000** |
| G normalised | 0.000 | 0.222 | 0.556 | 0.778 | **1.000** | 0.833 |

At α = 0.5 the shapes look different and none of them is good. At α ≥ 1 several
reach 1.000. The apparent shape-dependence was a weight artifact.

**And the collapse under agent count inverts.** The shipped term goes
0.389 → 0.278 → 0.000 as agents grow from 96 to 384; `normalised` at α = 2 goes
0.333 → 0.889 → 1.000. More agents now help. Section 10's collapse and section
13's collapse were the same ceiling reached along two different axes, so lifting
the ceiling addresses both at once.

### The control is what separates the shapes

manifold cannot tell the shapes apart at high α — the neutral fixture can, because
its relevant documents are drawn from the FAISS top-16 and similarity is therefore
the *correct* signal there. A visit term that overrides similarity must hurt.
Method C, 192 agents:

| shape | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| capped | 0.458 | | | | | |
| log | 0.542 | 0.458 | 0.458 | 0.417 | 0.375 | 0.375 |
| rank | 0.542 | 0.542 | 0.458 | 0.417 | 0.417 | 0.375 |
| visited_rank | 0.500 | 0.458 | 0.417 | 0.375 | 0.333 | 0.333 |
| normalised | 0.542 | 0.542 | 0.542 | 0.542 | 0.500 | 0.542 |

`log`, `rank` and `visited_rank` all decay as the weight rises; `normalised` does
not. The shapes that discard the *magnitude* of the visit distribution — a
document visited once scoring nearly as high as one visited two hundred times —
are the ones that damage the control. **Any result on manifold that was not
checked here is tuned to one fixture, not improved.**

---

## The two jobs: steering and ranking are in conflict

`update_document_relevance` writes `relevance_score`, and that field is read
twice: by the attraction force through its `(1 + r)` weight, and by the harness as
the final ranking. Those are different jobs and they want different things.

Applying the replacement term to the **ranking only** — steering with the shipped
term for every step but the last — at 192 agents:

| corpus | method | shape | coupled | decoupled |
|---|---|---|---|---|
| 256 | C | uncapped | 0.667 | **1.000** |
| 256 | G | uncapped | 0.000 | **1.000** |
| 1024 | C | uncapped | 0.000 | 0.722 |
| 1024 | G | uncapped | 0.000 | **1.000** |

`uncapped` was dismissed above as swamping similarity. It does not. Applied to the
ranking alone it is the strongest signal measured. Coupled, it destroys the
*walk*: an aggressive visit term fed back through `(1 + r)` is positive feedback
that collapses the colony onto what it has already visited. The damage was never
in the scoring.

This is a structural fix and it costs no parameter. It is also not free: `uncapped`
decoupled does systematic damage on the control that grows with agent count
(C: 0.375 → 0.333 → 0.292 against a baseline of 0.417 → 0.458 → 0.542), because an
unbounded term ranks by traffic alone.

**Two properties are needed.** Decoupling makes a strong ranking signal usable;
boundedness protects the control. `normalised` at α = 2, decoupled, has both:
ten of twelve manifold cells improve over the shipped ceiling, one ties, one is
worse — and on the control four of six improve, one ties, one is worse by two
documents. Its weak spot is the sparse regime (96 agents), where the visit
distribution is too noisy to weight at twice the similarity.

**No production code was changed on the strength of any of this,** and none is
proposed. These are synthetic fixtures with a chain placed there by design. The
evidence supports testing a bounded, decoupled visit term on real data. It does
not support shipping one.

---

## Two defects in the pheromone mechanism, and why repairing them hurt

Both are in `simulation.py`, both are verified by experiment, and both were
undocumented until 2026-08-31.

**Defect 1 — a trail is followable from one endpoint only.** `deposit_pheromones`
stores trails under `tuple(sorted((i, j)))`; `calculate_pheromone_force` matches
`doc_a == current_doc.id`, i.e. only the lower id. A trail between documents 3
and 91 pulls an agent standing on 3 and is invisible to an agent standing on 91.
Since ids are assigned in corpus insertion order, load order decides which half of
the deposited signal is usable.

**Defect 2 — the "last three visited documents" are not the last three.**
`agent.visited_docs` is a `set`, so `list(...)[-3:]` returns three entries in hash
order. For the visit sequence `7, 3, 91, 12, 5, 44, 2` the code deposits against
`[12, 44, 91]`; the genuinely recent three are `[5, 44, 2]`. Trails connect
documents adjacent in *id*, not adjacent in *time* — which for a walk means the
trail graph is not a record of the path taken.

Repairing them, separately and together (256 documents):

| | as built | symmetric | recency | both |
|---|---|---|---|---|
| C, 96 | **0.389** | 0.056 | 0.389 | 0.000 |
| C, 192 | **0.278** | 0.000 | 0.222 | 0.000 |
| C, 384 | 0.000 | 0.000 | 0.000 | 0.000 |
| G, 96 | **0.722** | 0.389 | 0.611 | 0.278 |
| G, 192 | 0.500 | 0.222 | 0.333 | **0.556** |
| G, 384 | **0.389** | 0.111 | 0.333 | 0.278 |

Eleven of twelve cells are at or below the shipped behaviour; the one exception is
a single document in a single seed. Repairing a defect made retrieval worse, which
is worth stating plainly rather than filing as an anomaly.

The consistent explanation is the section above: more trail signal means a more
concentrated walk, a more concentrated walk drives more documents past the
five-visit ceiling, and past the ceiling the ranking signal is gone. The defects
were acting as an accidental brake on a feedback loop that has no brake of its
own. **Both repairs are therefore blocked on fixing the visit term first** — they
are not independently shippable, and shipping them alone would degrade retrieval.

This has not been confirmed by running the repairs together with a non-saturating
visit term. That is the obvious next experiment and it has not been done.

---

## What this means for large repositories

The standing goal is MCMP that produces better results on large codebases. The
measured position:

1. **Full-corpus MCMP does not scale *as shipped*, and the reason is the scoring,
   not the walk.** At 1024 documents C reaches 2.33 of 3 relevant documents and
   ranks 0.056 of them into the top 8, spending 14.6 million comparisons to do it
   (section 13). With the visit term swept, that same configuration reaches
   **1.000** — so the scaling failure is a property of the relevance function, not
   of full-corpus MCMP as such. The cost is unchanged and still ruinous, so this
   does not make full-corpus search viable; it means the bottleneck moved from
   "cannot rank" to "cannot afford". Search over 103k chunks remains unsupported by
   anything measured here.

2. **The bounded frontier (method G) is the only variant that survives scaling** —
   constant 143 224 comparisons at 64, 256 and 1024 documents, recall flat around
   0.667. It works by keeping the *scored* set small, which is consistent with the
   scoring being the bottleneck.

3. **The relevance function has been pulled, and it moved a long way.** Bounded,
   decoupled, weighted above similarity: ten of twelve manifold cells improve, the
   agent-count collapse inverts, and the control takes no systematic damage. What
   remains untested there is a weight that adapts to walk density, which is the
   named weakness of the current best candidate.

4. **Cost is now the binding constraint, not quality.** G holds a constant
   143 224 comparisons against FAISS's 8. Nothing in this repository models what
   that costs against a production ANN index, and no latency has ever been measured.
   For a retrieval path that gap decides everything; for a batch crawler it may not
   matter at all.

5. **The pheromone repairs come after the relevance work, not before** — section
   14.3 showed they degrade retrieval while the ceiling is in place. Whether they
   compose with a swept, decoupled term is a prediction and has not been run.

---

## Non-claims

- Synthetic fixtures throughout, with a chain placed there by design. Nothing here
  says real code retrieval has this geometry; that is Gate 2's question and Gate 2
  has never run.
- Six seeds, one query, three relevant documents per configuration on manifold
  (resolution 0.056) and four on neutral (0.042). Differences that size are one or
  two documents.
- Two corpus sizes and three agent counts. `steps` was held at 50 throughout;
  `expand_every`, `expand_k` and `frontier_cap` were left at their defaults.
- The exploration term has never been ablated.
- The decoupling is implemented by applying the replacement on the final relevance
  call only. That is a clean separation for this harness; it is not the same as a
  production design maintaining two scores throughout, and no such design has been
  built or measured.
- The weight was swept over six values on a log-ish grid. Nothing says α = 2 is
  optimal, only that it is the best of those six on this evidence, and its weakness
  in the sparse regime is visible in the same tables.
- No production code was changed. Every variant in this document is an override
  applied inside a probe.
