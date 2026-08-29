# MCMP Gate 1 Fixture Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gate 1 able to measure the retriever instead of the fixture's labelling convention, by adding a neutral control fixture, a manifold fixture, a pool-restricted MCMP method E, and by allowing `initial_k > top_k`.

**Architecture:** `benchmarks/mcmp/fixtures.py` gains two seeded builders and a `FIXTURES` registry; the existing `build_synthetic_dataset` is not touched. `benchmarks/mcmp/adapters.py` gains method E (MCMP restricted to the FAISS pool) and drops the `initial_k > top_k` guard. `benchmarks/mcmp/run_gate1.py` gains a `--fixture` flag, writes `dataset.fixture` and `runs.E`, and its replay validator becomes fixture-aware while still accepting legacy four-run payloads.

**Tech Stack:** Python 3.11, NumPy, FAISS CPU, pytest, existing `embeddinggemma.mcmp` implementation, JSON.

**Spec:** `docs/superpowers/specs/2026-08-29-mcmp-fixture-redesign-design.md`

## Global Constraints

- Use `.venv\Scripts\python.exe` explicitly. Python 3.11; `.python-version` selects 3.12, where `pandas 1.5.3` fails to build.
- Do not start or modify the Fungus MCP server. Do not invoke an LLM, OpenFang, RAG generation, Docker, or a production embedding service.
- Do not modify `build_synthetic_dataset`, `tests/benchmarks/test_fixtures.py`'s legacy tests, or any file under `benchmarks/results/`.
- Do not change production MCMP behaviour to make a fixture pass. A negative result is valid.
- All randomness must be seed-derived. Fixtures are CPU-only, unit-normalized float32, `faiss.METRIC_INNER_PRODUCT`.
- New fixtures reuse the query ids `("q-main", "q-related")` so `_RUN_SPECS` stays unchanged.
- The `conclusion` rule stays derived from C and D only. E never influences it.
- The full suite must stay green: `168 passed, 2 skipped` before this plan starts.
- Run the full suite with: `.venv\Scripts\python.exe -m pytest tests -q --disable-warnings --import-mode=importlib`

---

## File map

- `benchmarks/mcmp/adapters.py`: relax `initial_k`; add method E via a `pool_only` flag on `run_mcmp`.
- `benchmarks/mcmp/fixtures.py`: add `build_neutral_dataset`, `build_manifold_dataset`, `FIXTURES`.
- `benchmarks/mcmp/run_gate1.py`: `--fixture` flag, `dataset.fixture`, `runs.E`, fixture-aware and legacy-tolerant validator.
- `tests/benchmarks/test_adapters.py`: `initial_k` relaxation, method E.
- `tests/benchmarks/test_fixtures.py`: neutral and manifold coverage; legacy tests untouched.
- `tests/benchmarks/test_gate1_runner.py`: `--fixture`, payload schema, legacy tolerance, five-run ordering.

---

### Task 1: Allow `initial_k > top_k`

**Files:**
- Modify: `benchmarks/mcmp/adapters.py:291-294`
- Modify: `benchmarks/mcmp/run_gate1.py:280-281`
- Test: `tests/benchmarks/test_adapters.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `run_faiss` and `run_mcmp` accept `initial_k > top_k`; both reject `initial_k > len(dataset.document_ids)` with the message `initial_k must not exceed document count`.

**Correction, found during execution.** This plan originally claimed no existing
test asserts the old rejection, based on `grep -rn "must not exceed top_k" tests/`
returning nothing. That grep was too narrow. `test_adapters.py` parametrizes
`test_mcmp_rejects_invalid_scalar_parameters_before_execution` with
`{"initial_k": 5}` against `top_k=4`, which relied on the removed guard without
naming its message. Under the new bound, `initial_k=5` is legal on an 8-document
fixture, so that parameter must become `{"initial_k": 9}` — still invalid, and
mirroring the `{"top_k": 9}` case already present. This preserves the test's
intent rather than weakening it.

The new document-count bound is also what satisfies the spec's Failure handling
requirement that method E fail closed when the requested pool is larger than the
corpus: E's pool size is `initial_k`, validated here for every method.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_adapters.py`:

```python
def test_run_faiss_accepts_initial_k_greater_than_top_k() -> None:
    dataset = build_synthetic_dataset()

    run, _evidence = run_faiss(dataset, "A", ("q-main",), top_k=2, initial_k=5)

    assert len(run.per_query_ranked_document_ids["q-main"]) == 2
    assert len(run.per_query_initial_candidate_ids["q-main"]) == 5


def test_run_faiss_rejects_initial_k_above_document_count() -> None:
    dataset = build_synthetic_dataset()

    with pytest.raises(ValueError, match="initial_k must not exceed document count"):
        run_faiss(dataset, "A", ("q-main",), top_k=8, initial_k=9)
```

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_adapters.py -q -k "initial_k"

Expected: both FAIL — the first with `ValueError: initial_k must not exceed top_k`, the second with the same wrong message instead of the document-count message.

- [ ] **Step 3: Replace the guard in `adapters.py`**

In `_validate_run_inputs`, replace:

```python
    if top_k > document_count:
        raise ValueError("top_k must not exceed document count")
    if initial_k > top_k:
        raise ValueError("initial_k must not exceed top_k")
```

with:

```python
    if top_k > document_count:
        raise ValueError("top_k must not exceed document count")
    if initial_k > document_count:
        raise ValueError("initial_k must not exceed document count")
```

- [ ] **Step 4: Remove the mirrored guard in the replay validator**

In `benchmarks/mcmp/run_gate1.py`, inside `validate_gate1_evidence`, delete:

```python
    if _integer(config["initial_k"], "initial_k") > _integer(config["top_k"], "top_k"):
        raise ValueError("initial_k must not exceed top_k")
```

- [ ] **Step 5: Verify GREEN**

    .venv\Scripts\python.exe -m pytest tests -q --disable-warnings --import-mode=importlib

Expected: 170 passed, 2 skipped, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add -- benchmarks/mcmp/adapters.py benchmarks/mcmp/run_gate1.py tests/benchmarks/test_adapters.py
git diff --cached --check
git commit -m "feat: allow a retrieval pool deeper than the returned list"
```

---

### Task 2: Fixture registry, `--fixture` flag, and `dataset.fixture`

**Files:**
- Modify: `benchmarks/mcmp/fixtures.py`
- Modify: `benchmarks/mcmp/run_gate1.py`
- Test: `tests/benchmarks/test_gate1_runner.py`

**Interfaces:**
- Consumes: Task 1's relaxed validation.
- Produces:
  - `benchmarks.mcmp.fixtures.FIXTURES: dict[str, Callable[[int], BenchmarkDataset]]`, initially `{"legacy": build_synthetic_dataset}`.
  - `benchmarks.mcmp.fixtures.build_dataset(fixture: str, seed: int) -> BenchmarkDataset`, raising `ValueError` naming valid keys.
  - `run_gate1(seed, top_k, initial_k, num_agents, steps, fixture="legacy")`.
  - Payload key `dataset["fixture"]`.

This task adds only the plumbing, with the registry holding one entry. The new fixtures arrive in Tasks 3 and 4, so a failure here is unambiguously a plumbing failure.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_gate1_runner.py`:

```python
def test_payload_records_the_fixture_key() -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=4, steps=2)

    assert payload["dataset"]["fixture"] == "legacy"


def test_unknown_fixture_names_the_valid_keys() -> None:
    with pytest.raises(ValueError, match="unknown fixture 'nope'"):
        run_gate1(seed=7, top_k=4, initial_k=1, num_agents=4, steps=2, fixture="nope")


def test_validator_accepts_a_legacy_payload_without_a_fixture_key() -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=4, steps=2)
    del payload["dataset"]["fixture"]

    validate_gate1_evidence(payload)
```

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_gate1_runner.py -q -k "fixture"

Expected: FAIL — `KeyError: 'fixture'`, then `TypeError: run_gate1() got an unexpected keyword argument 'fixture'`.

- [ ] **Step 3: Add the registry to `fixtures.py`**

Append to `benchmarks/mcmp/fixtures.py`:

```python
FIXTURES = {
    "legacy": build_synthetic_dataset,
}


def build_dataset(fixture: str, seed: int) -> BenchmarkDataset:
    """Build a benchmark dataset by registry key, failing closed on unknown keys."""
    if fixture not in FIXTURES:
        raise ValueError(
            f"unknown fixture {fixture!r}; valid keys are {sorted(FIXTURES)}"
        )
    return FIXTURES[fixture](seed)
```

- [ ] **Step 4: Thread the fixture through `run_gate1.py`**

Replace the import `from benchmarks.mcmp.fixtures import build_synthetic_dataset` with:

```python
from benchmarks.mcmp.fixtures import build_dataset
```

Change the signature and first line of `run_gate1`:

```python
def run_gate1(
    seed: int,
    top_k: int,
    initial_k: int,
    num_agents: int,
    steps: int,
    fixture: str = "legacy",
) -> dict[str, object]:
    """Run the fixed A-D offline ablation and return its complete evidence."""
    dataset = build_dataset(fixture, seed)
```

Add `"fixture": fixture,` to the `"dataset"` block of `payload`, keeping keys otherwise unchanged.

In `validate_gate1_evidence`, replace `dataset = build_synthetic_dataset(_integer(config["seed"], "seed"))` with:

```python
    dataset_payload = _mapping(payload["dataset"], "dataset")
    fixture = dataset_payload.get("fixture", "legacy")
    if not isinstance(fixture, str):
        raise ValueError("dataset.fixture must be a string")
    dataset = build_dataset(fixture, _integer(config["seed"], "seed"))
```

Delete the later duplicate `dataset_payload = _mapping(payload["dataset"], "dataset")` line.

**Correction, found during execution.** `_require_keys` compares with
`set(mapping) != expected`, so it rejects *extra* keys, not just missing ones.
Keeping the six original keys required therefore fails on every new payload with
`dataset has incomplete or unexpected keys`. The required set must depend on
whether `fixture` is present, which is also what makes both payload shapes valid:

```python
    required_dataset_keys = {"id", "digest", "document_ids", "query_ids", "document_vector_shape", "query_vector_shape"}
    if "fixture" in dataset_payload:
        required_dataset_keys = required_dataset_keys | {"fixture"}
    _require_keys(dataset_payload, required_dataset_keys, "dataset")
    expected_dataset = {
        "id": dataset.dataset_id,
        "digest": dataset.digest(),
        "document_ids": list(dataset.document_ids),
        "query_ids": list(dataset.query_ids),
        "document_vector_shape": list(dataset.document_vectors.shape),
        "query_vector_shape": list(dataset.query_vectors.shape),
    }
    if "fixture" in dataset_payload:
        expected_dataset["fixture"] = fixture
    if not _strict_equal(dict(dataset_payload), expected_dataset):
        raise ValueError("dataset evidence does not match the configured seed")
```

- [ ] **Step 5: Add the CLI flag**

In `main`, after the `--steps` argument:

```python
    parser.add_argument(
        "--fixture", choices=sorted(FIXTURES), default="legacy"
    )
```

Import `FIXTURES` alongside `build_dataset`, and pass it through:

```python
    payload = run_gate1(
        args.seed, args.top_k, args.initial_k, args.num_agents, args.steps, args.fixture
    )
```

- [ ] **Step 6: Verify GREEN**

    .venv\Scripts\python.exe -m pytest tests -q --disable-warnings --import-mode=importlib

Expected: 173 passed, 2 skipped, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add -- benchmarks/mcmp/fixtures.py benchmarks/mcmp/run_gate1.py tests/benchmarks/test_gate1_runner.py
git diff --cached --check
git commit -m "feat: select the Gate 1 fixture by registry key"
```

---

### Task 3: Neutral control fixture

**Files:**
- Modify: `benchmarks/mcmp/fixtures.py`
- Test: `tests/benchmarks/test_fixtures.py`

**Interfaces:**
- Consumes: `FIXTURES` from Task 2.
- Produces: `build_neutral_dataset(seed: int = 7) -> BenchmarkDataset` with `dataset_id="neutral-mcmp-v1"`, 64 documents `doc-00..doc-63`, queries `("q-main", "q-related")`, 16 dimensions, 4 relevant documents per query drawn uniformly from that query's top 16 by similarity. Registered under `"neutral"`.

The load-bearing test is rank variability: the legacy fixture yields the constant rank tuple `(2, 3)` in all 24 query cases, which is the defect this fixture exists to remove.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_fixtures.py`:

```python
from benchmarks.mcmp.fixtures import build_dataset, build_neutral_dataset


def _relevant_ranks(dataset, query_id: str) -> tuple[int, ...]:
    query_index = dataset.query_ids.index(query_id)
    similarities = dataset.document_vectors @ dataset.query_vectors[query_index]
    order = np.argsort(-similarities)
    relevant = dataset.relevant_by_query[query_id]
    return tuple(
        rank
        for rank, index in enumerate(order, start=1)
        if dataset.document_ids[index] in relevant
    )


def test_neutral_dataset_has_expected_shape_and_validates() -> None:
    dataset = build_neutral_dataset(7)

    assert dataset.dataset_id == "neutral-mcmp-v1"
    assert dataset.document_vectors.shape == (64, 16)
    assert dataset.query_vectors.shape == (2, 16)
    assert dataset.query_ids == ("q-main", "q-related")
    assert dataset.document_ids[0] == "doc-00"
    assert dataset.document_ids[-1] == "doc-63"
    assert dataset.document_vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(dataset.document_vectors, axis=1), 1.0, atol=1e-4)
    dataset.validate()


def test_neutral_relevance_is_drawn_from_the_similarity_top_16() -> None:
    dataset = build_neutral_dataset(7)

    for query_id in dataset.query_ids:
        ranks = _relevant_ranks(dataset, query_id)
        assert len(ranks) == 4
        assert max(ranks) <= 16


def test_neutral_relevant_ranks_vary_across_seeds() -> None:
    observed = {
        _relevant_ranks(build_neutral_dataset(seed), query_id)
        for seed in range(1, 13)
        for query_id in ("q-main", "q-related")
    }

    assert len(observed) > 1


def test_neutral_dataset_is_deterministic_per_seed() -> None:
    first = build_neutral_dataset(3)
    second = build_neutral_dataset(3)

    assert first.digest() == second.digest()
    assert first.relevant_by_query == second.relevant_by_query
    assert first.digest() != build_neutral_dataset(4).digest()


def test_registry_selects_the_requested_builder() -> None:
    assert build_dataset("neutral", 7).dataset_id == "neutral-mcmp-v1"
    assert build_dataset("legacy", 7).dataset_id == "synthetic-mcmp-v1"
```

This is the first point at which the registry holds more than one entry, so it is
the first point at which "the flag actually selects a different builder" is
testable. Task 2 could only test that the key is recorded and that unknown keys
fail.

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_fixtures.py -q -k "neutral"

Expected: collection failure — `ImportError: cannot import name 'build_neutral_dataset'`.

- [ ] **Step 3: Implement the builder**

Append to `benchmarks/mcmp/fixtures.py`, above the `FIXTURES` definition:

```python
NEUTRAL_DOCUMENT_COUNT = 64
NEUTRAL_DIMENSIONS = 16
NEUTRAL_RELEVANT_PER_QUERY = 4
NEUTRAL_CANDIDATE_DEPTH = 16


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def build_neutral_dataset(seed: int = 7) -> BenchmarkDataset:
    """Build a control fixture whose relevance is not a fixed function of rank."""
    rng = np.random.default_rng(seed)
    documents = _unit_rows(
        rng.normal(size=(NEUTRAL_DOCUMENT_COUNT, NEUTRAL_DIMENSIONS))
    )
    queries = _unit_rows(rng.normal(size=(2, NEUTRAL_DIMENSIONS)))
    document_ids = tuple(f"doc-{index:02d}" for index in range(NEUTRAL_DOCUMENT_COUNT))
    query_ids = ("q-main", "q-related")

    relevant_by_query: dict[str, frozenset[str]] = {}
    for query_index, query_id in enumerate(query_ids):
        similarities = documents @ queries[query_index]
        candidates = np.argsort(-similarities)[:NEUTRAL_CANDIDATE_DEPTH]
        chosen = rng.choice(
            candidates, size=NEUTRAL_RELEVANT_PER_QUERY, replace=False
        )
        relevant_by_query[query_id] = frozenset(
            document_ids[int(index)] for index in chosen
        )

    dataset = BenchmarkDataset(
        dataset_id="neutral-mcmp-v1",
        seed=seed,
        document_ids=document_ids,
        document_vectors=documents,
        query_ids=query_ids,
        query_vectors=queries,
        relevant_by_query=relevant_by_query,
    )
    dataset.validate()
    return dataset
```

- [ ] **Step 4: Register it**

```python
FIXTURES = {
    "legacy": build_synthetic_dataset,
    "neutral": build_neutral_dataset,
}
```

- [ ] **Step 5: Verify GREEN**

    .venv\Scripts\python.exe -m pytest tests -q --disable-warnings --import-mode=importlib

Expected: 177 passed, 2 skipped, 0 failures.

- [ ] **Step 6: Smoke-run the fixture end to end**

    .venv\Scripts\python.exe -m benchmarks.mcmp.run_gate1 --seed 7 --top-k 4 --initial-k 4 --num-agents 24 --steps 10 --fixture neutral --output tmp-neutral.json

Expected: exit 0. Then delete `tmp-neutral.json` — evidence generation is a separate reviewable step, not part of this plan.

- [ ] **Step 7: Commit**

```bash
git add -- benchmarks/mcmp/fixtures.py tests/benchmarks/test_fixtures.py
git diff --cached --check
git commit -m "feat: add a neutral Gate 1 control fixture"
```

---

### Task 4: Manifold fixture

**Files:**
- Modify: `benchmarks/mcmp/fixtures.py`
- Test: `tests/benchmarks/test_fixtures.py`

**Interfaces:**
- Consumes: `_unit_rows` and the registry from Task 3, and the `_relevant_ranks`
  test helper defined in Task 3's test block. Task 4's tests will not run before
  Task 3 has landed.
- Produces: `build_manifold_dataset(seed: int = 7) -> BenchmarkDataset` with `dataset_id="manifold-mcmp-v1"`, 64 documents in 16 dimensions: two 8-document chains `main-chain-1..8` and `related-chain-1..8`, plus 48 distractors `distractor-00..47`. Relevant per query: that query's chain positions 6, 7 and 8. Registered under `"manifold"`.

Geometry: chain document *i* of 8 sits at angle `i * TOTAL_ANGLE / 8` from its query, rotated toward a seed-chosen direction orthogonal to both queries. With `TOTAL_ANGLE = 1.37` rad the cosines run `0.985, 0.942, 0.871, 0.774, 0.655, 0.518, 0.362, 0.199`. Distractors sit at cosine drawn uniformly from `[0.55, 0.75]` of one query, so they outrank chain links 6-8 and the relevant documents are unreachable by shallow similarity retrieval.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_fixtures.py`:

```python
from benchmarks.mcmp.fixtures import build_manifold_dataset


def test_manifold_dataset_has_expected_shape_and_validates() -> None:
    dataset = build_manifold_dataset(7)

    assert dataset.dataset_id == "manifold-mcmp-v1"
    assert dataset.document_vectors.shape == (64, 16)
    assert dataset.query_ids == ("q-main", "q-related")
    assert dataset.relevant_by_query["q-main"] == frozenset(
        {"main-chain-6", "main-chain-7", "main-chain-8"}
    )
    assert dataset.relevant_by_query["q-related"] == frozenset(
        {"related-chain-6", "related-chain-7", "related-chain-8"}
    )
    dataset.validate()


def test_manifold_chain_links_are_closer_to_each_other_than_the_far_end_is_to_the_query() -> None:
    dataset = build_manifold_dataset(7)
    index = {document_id: position for position, document_id in enumerate(dataset.document_ids)}
    vectors = dataset.document_vectors
    query = dataset.query_vectors[dataset.query_ids.index("q-main")]

    far_end_similarity = float(vectors[index["main-chain-8"]] @ query)
    for position in range(1, 8):
        link = float(
            vectors[index[f"main-chain-{position}"]]
            @ vectors[index[f"main-chain-{position + 1}"]]
        )
        assert link > far_end_similarity


def test_manifold_relevant_documents_rank_below_the_default_top_k() -> None:
    dataset = build_manifold_dataset(7)

    ranks = _relevant_ranks(dataset, "q-main")

    assert min(ranks) > 4


def test_manifold_dataset_is_deterministic_per_seed() -> None:
    assert build_manifold_dataset(3).digest() == build_manifold_dataset(3).digest()
    assert build_manifold_dataset(3).digest() != build_manifold_dataset(4).digest()
```

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_fixtures.py -q -k "manifold"

Expected: collection failure — `ImportError: cannot import name 'build_manifold_dataset'`.

- [ ] **Step 3: Implement the builder**

Append to `benchmarks/mcmp/fixtures.py`, above `FIXTURES`:

```python
MANIFOLD_CHAIN_LENGTH = 8
MANIFOLD_TOTAL_ANGLE = 1.37
MANIFOLD_RELEVANT_TAIL = 3
MANIFOLD_DISTRACTOR_COUNT = 48
MANIFOLD_DISTRACTOR_COSINE_RANGE = (0.55, 0.75)


def _orthonormal_basis(rng: np.random.Generator, dimensions: int) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))
    return basis.T


def _chain(query: np.ndarray, direction: np.ndarray, length: int, total_angle: float) -> np.ndarray:
    angles = np.linspace(total_angle / length, total_angle, length)
    return np.stack(
        [np.cos(angle) * query + np.sin(angle) * direction for angle in angles]
    )


def build_manifold_dataset(seed: int = 7) -> BenchmarkDataset:
    """Build a fixture whose relevant documents are reachable only along a chain."""
    rng = np.random.default_rng(seed)
    dimensions = NEUTRAL_DIMENSIONS
    basis = _orthonormal_basis(rng, dimensions)
    queries = basis[:2]
    query_ids = ("q-main", "q-related")

    document_ids: list[str] = []
    rows: list[np.ndarray] = []
    relevant_by_query: dict[str, frozenset[str]] = {}

    for query_index, (query_id, prefix) in enumerate(
        zip(query_ids, ("main", "related"), strict=True)
    ):
        weights = rng.normal(size=dimensions - 2)
        direction = weights @ basis[2:]
        direction = direction / np.linalg.norm(direction)
        chain = _chain(
            queries[query_index], direction, MANIFOLD_CHAIN_LENGTH, MANIFOLD_TOTAL_ANGLE
        )
        ids = [f"{prefix}-chain-{position}" for position in range(1, MANIFOLD_CHAIN_LENGTH + 1)]
        document_ids.extend(ids)
        rows.extend(chain)
        relevant_by_query[query_id] = frozenset(ids[-MANIFOLD_RELEVANT_TAIL:])

    low, high = MANIFOLD_DISTRACTOR_COSINE_RANGE
    for position in range(MANIFOLD_DISTRACTOR_COUNT):
        anchor = queries[position % 2]
        cosine = float(rng.uniform(low, high))
        weights = rng.normal(size=dimensions - 2)
        offset = weights @ basis[2:]
        offset = offset / np.linalg.norm(offset)
        document_ids.append(f"distractor-{position:02d}")
        rows.append(cosine * anchor + np.sqrt(1.0 - cosine**2) * offset)

    documents = _unit_rows(np.stack(rows))
    dataset = BenchmarkDataset(
        dataset_id="manifold-mcmp-v1",
        seed=seed,
        document_ids=tuple(document_ids),
        document_vectors=documents,
        query_ids=query_ids,
        query_vectors=_unit_rows(queries),
        relevant_by_query=relevant_by_query,
    )
    dataset.validate()
    return dataset
```

- [ ] **Step 4: Register it**

```python
FIXTURES = {
    "legacy": build_synthetic_dataset,
    "neutral": build_neutral_dataset,
    "manifold": build_manifold_dataset,
}
```

- [ ] **Step 5: Verify GREEN**

    .venv\Scripts\python.exe -m pytest tests -q --disable-warnings --import-mode=importlib

Expected: 181 passed, 2 skipped, 0 failures.

If `test_manifold_relevant_documents_rank_below_the_default_top_k` fails, the distractor cosine band overlaps the chain tail. Do not weaken the test. Widen `MANIFOLD_DISTRACTOR_COSINE_RANGE` upward, or raise `MANIFOLD_TOTAL_ANGLE`, and re-run — the invariant is the point of the fixture.

- [ ] **Step 6: Commit**

```bash
git add -- benchmarks/mcmp/fixtures.py tests/benchmarks/test_fixtures.py
git diff --cached --check
git commit -m "feat: add a manifold Gate 1 fixture with chain-reachable relevance"
```

---

### Task 5: Method E adapter

**Files:**
- Modify: `benchmarks/mcmp/adapters.py`
- Test: `tests/benchmarks/test_adapters.py`

**Interfaces:**
- Consumes: Task 1's relaxed validation.
- Produces: `run_mcmp(..., pool_only: bool = False)`. Method `"E"` is accepted as a single-query MCMP method. When `pool_only` is true, `add_documents` receives only the FAISS top-`initial_k` ids.

Method E is single-query, so `_validate_run_inputs` must count `"E"` among the single-query methods.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_adapters.py`:

```python
def test_method_e_restricts_mcmp_to_the_faiss_pool() -> None:
    dataset = build_synthetic_dataset()

    run, _evidence = run_mcmp(
        dataset, "E", ("q-main",), top_k=2, initial_k=3,
        seed=7, num_agents=4, steps=2, pool_only=True,
    )

    pool = run.per_query_initial_candidate_ids["q-main"]
    assert len(pool) == 3
    assert run.per_query_candidate_ids["q-main"] <= pool
    assert set(run.per_query_ranked_document_ids["q-main"]) <= pool


def test_method_e_is_deterministic_for_a_fixed_seed() -> None:
    dataset = build_synthetic_dataset()
    kwargs = dict(
        top_k=2, initial_k=3, seed=7, num_agents=4, steps=2, pool_only=True
    )

    first, _ = run_mcmp(dataset, "E", ("q-main",), **kwargs)
    second, _ = run_mcmp(dataset, "E", ("q-main",), **kwargs)

    assert first.ranked_document_ids == second.ranked_document_ids
```

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_adapters.py -q -k "method_e"

Expected: FAIL — `TypeError: run_mcmp() got an unexpected keyword argument 'pool_only'`.

- [ ] **Step 3: Accept `"E"` in the input validator**

In `_validate_run_inputs`, replace:

```python
    expected_query_count = 1 if method in {"A", "C"} else 2
```

with:

```python
    expected_query_count = 1 if method in {"A", "C", "E"} else 2
```

- [ ] **Step 4: Add the `pool_only` path to `run_mcmp`**

Add the parameter to the signature, after `steps`:

```python
    pool_only: bool = False,
```

Widen the allowed methods on the first line of the body:

```python
    _validate_run_inputs(dataset, method, query_ids, {"C", "D", "E"}, top_k, initial_k)
```

Inside the per-query loop, replace:

```python
            retriever.add_documents(list(dataset.document_ids), cache=False)
            execution_backends.add(_execution_backend(retriever))
            initial = retriever.find_nearest_documents(vectors[query_id], k=len(dataset.document_ids))
            initial_scores = {document.content: float(score) for document, score in initial}
            initial_rankings[query_id] = _rank(initial_scores, initial_k)
            initial_candidates[query_id] = frozenset(initial_rankings[query_id])
```

with:

```python
            retriever.add_documents(list(dataset.document_ids), cache=False)
            execution_backends.add(_execution_backend(retriever))
            initial = retriever.find_nearest_documents(vectors[query_id], k=len(dataset.document_ids))
            initial_scores = {document.content: float(score) for document, score in initial}
            initial_rankings[query_id] = _rank(initial_scores, initial_k)
            initial_candidates[query_id] = frozenset(initial_rankings[query_id])
            if pool_only:
                backend = MappingEmbeddingBackend(vectors)
                retriever = CountingRetriever(
                    num_agents=num_agents,
                    max_iterations=steps,
                    pheromone_decay=PHEROMONE_DECAY,
                    exploration_bonus=EXPLORATION_BONUS,
                    build_faiss_after_add=True,
                    force_cpu=True,
                    embedding_backend=(backend, dataset.document_vectors.shape[1]),
                    time_source=FixedClock(DETERMINISTIC_CLOCK_VALUE),
                )
                retriever.add_documents(list(initial_rankings[query_id]), cache=False)
                execution_backends.add(_execution_backend(retriever))
```

The pool ranking is computed on a full-corpus retriever, then a fresh retriever is built over the pool alone. Both retrievers' `nearest_search_calls` are counted, so E's reported comparisons include the cost of forming its own pool.

- [ ] **Step 5: Verify GREEN**

    .venv\Scripts\python.exe -m pytest tests -q --disable-warnings --import-mode=importlib

Expected: 183 passed, 2 skipped, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add -- benchmarks/mcmp/adapters.py tests/benchmarks/test_adapters.py
git diff --cached --check
git commit -m "feat: add pool-restricted MCMP as benchmark method E"
```

---

### Task 6: Wire method E into the runner

**Files:**
- Modify: `benchmarks/mcmp/run_gate1.py`
- Modify: `tests/benchmarks/test_gate1_runner.py:16`, `:166`, `:168`
- Test: `tests/benchmarks/test_gate1_runner.py`

**Interfaces:**
- Consumes: `run_mcmp(..., pool_only=True)` from Task 5.
- Produces: payload `runs` ordered `A, B, C, D, E`; `comparisons` gains `A_vs_E` and `C_vs_E`; the validator accepts both the five-run and the legacy four-run shape.

Three existing assertions lock the old shape and must be updated: `test_gate1_runner.py:16` and `:166` assert `["A", "B", "C", "D"]`, and `:168` asserts `comparisons.keys() == {"A_vs_C", "B_vs_D"}`.

- [ ] **Step 1: Update the three existing assertions**

At `tests/benchmarks/test_gate1_runner.py:16` and `:166`, change the expected list to `["A", "B", "C", "D", "E"]`. At `:168`, change the expected set to `{"A_vs_C", "B_vs_D", "A_vs_E", "C_vs_E"}`.

- [ ] **Step 2: Write the failing tests**

Append to `tests/benchmarks/test_gate1_runner.py`:

```python
def test_method_e_ranks_only_pool_documents() -> None:
    payload = run_gate1(seed=7, top_k=2, initial_k=3, num_agents=4, steps=2)

    pool = set(payload["runs"]["E"]["raw_ids"]["initial_candidate_ids"])
    assert set(payload["runs"]["E"]["raw_ids"]["ranked_document_ids"]) <= pool


def test_conclusion_ignores_method_e() -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=4, num_agents=4, steps=2)
    novel = sum(
        len(payload["runs"][method]["metrics"]["novel_relevant_candidates"])
        for method in ("C", "D")
    )

    expected = "novel_relevant_observed" if novel else "no_novel_relevant_observed"
    assert payload["conclusion"] == expected


def test_validator_accepts_a_legacy_four_run_payload() -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=4, steps=2)
    del payload["runs"]["E"]
    del payload["comparisons"]["A_vs_E"]
    del payload["comparisons"]["C_vs_E"]
    del payload["dataset"]["fixture"]

    validate_gate1_evidence(payload)
```

- [ ] **Step 3: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_gate1_runner.py -q

Expected: FAIL — `KeyError: 'E'` and `runs must be ordered A-D`.

- [ ] **Step 4: Add E to the run specification and orchestration**

Extend `_RUN_SPECS`:

```python
_RUN_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A", ("q-main",)),
    ("B", ("q-main", "q-related")),
    ("C", ("q-main",)),
    ("D", ("q-main", "q-related")),
    ("E", ("q-main",)),
)
```

In `run_gate1`'s loop, replace the `else` branch with:

```python
        else:
            run, evidence = run_mcmp(
                dataset,
                method,
                query_ids,
                top_k,
                initial_k,
                seed,
                num_agents,
                steps,
                pool_only=method == "E",
            )
```

- [ ] **Step 5: Add the two comparisons**

Replace `_comparison_payload` entirely:

```python
def _comparison_payload(runs: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    comparisons = {
        "A_vs_C": _compare_runs(runs["A"], runs["C"]),
        "B_vs_D": _compare_runs(runs["B"], runs["D"]),
    }
    if "E" in runs:
        comparisons["A_vs_E"] = _compare_runs(runs["A"], runs["E"])
        comparisons["C_vs_E"] = _compare_runs(runs["C"], runs["E"])
    return comparisons
```

`A_vs_E` isolates the reranking contribution and `C_vs_E` the walk contribution, per the spec's Retrieval adapters section.

- [ ] **Step 6: Make the validator tolerate both shapes**

In `validate_gate1_evidence`, replace:

```python
    runs = _mapping(payload["runs"], "runs")
    if list(runs) != ["A", "B", "C", "D"]:
        raise ValueError("runs must be ordered A-D")
    for method, query_ids in _RUN_SPECS:
```

with:

```python
    runs = _mapping(payload["runs"], "runs")
    if list(runs) not in (["A", "B", "C", "D"], ["A", "B", "C", "D", "E"]):
        raise ValueError("runs must be ordered A-D, optionally followed by E")
    present = set(runs)
    for method, query_ids in _RUN_SPECS:
        if method not in present:
            continue
```

and replace the comparisons check:

```python
    comparisons = _mapping(payload["comparisons"], "comparisons")
    expected_pairs = (
        (("A_vs_C", "A", "C"), ("B_vs_D", "B", "D"), ("A_vs_E", "A", "E"), ("C_vs_E", "C", "E"))
        if "E" in present
        else (("A_vs_C", "A", "C"), ("B_vs_D", "B", "D"))
    )
    if list(comparisons) != [name for name, _left, _right in expected_pairs]:
        raise ValueError("comparisons must match the runs present")
    for name, left, right in expected_pairs:
```

- [ ] **Step 7: Verify GREEN**

    .venv\Scripts\python.exe -m pytest tests -q --disable-warnings --import-mode=importlib

Expected: 186 passed, 2 skipped, 0 failures.

- [ ] **Step 8: Confirm no new failure mode for legacy evidence**

    .venv\Scripts\python.exe -c "import json,sys; sys.path.insert(0,'.'); from benchmarks.mcmp.run_gate1 import validate_gate1_evidence; validate_gate1_evidence(json.load(open('benchmarks/results/gate1-seed-1.json'))); print('legacy payload still validates')"

Expected: `legacy payload still validates`. Note that `gate1-seed-7.json` fails on environment provenance for reasons predating this plan — see the Result writer correction in the spec. Do not "fix" that here.

- [ ] **Step 9: Commit**

```bash
git add -- benchmarks/mcmp/run_gate1.py tests/benchmarks/test_gate1_runner.py
git diff --cached --check
git commit -m "feat: report pool-restricted MCMP alongside A-D"
```

---

## Out of scope for this plan

Generating Gate 1 evidence on the neutral and manifold fixtures, and reviewing it, is a separate step under the Gate 1 decision rule. Do not draw or record a conclusion about MCMP from within this plan. Method F (equal-budget, pheromone-free control) is deliberately deferred.
