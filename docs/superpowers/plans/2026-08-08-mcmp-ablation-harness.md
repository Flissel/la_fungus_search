# MCMP Ablation Harness Implementation Plan

> **Status (2026-08-29):** Tasks 1-6 are implemented and merged into `main`; all
> their artifacts exist and `pytest tests/benchmarks tests/mcmp` reports 133
> passed, 2 skipped. The step checkboxes below were never ticked and are left
> unticked on purpose — the outcomes are verified, the individual RED/GREEN steps
> are not independently attested.
>
> **Task 7 is done.** Its result is `docs/MCMP_TIG_C004_REPORT.md`, which applies
> the decision rule and concludes: do not proceed to Gate 2, query colonies or
> TIG C004 on the current evidence. Read that report before acting on this plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a deterministic offline Gate-1 harness that compares Single/Multi Query with FAISS/MCMP and records whether MCMP discovers relevant candidates outside the initial FAISS set.

**Architecture:** A source-controlled synthetic fixture feeds four isolated retrieval adapters (A-D). Validated run records flow into pure metric functions and an authoritative JSON writer; MCMP is exercised through an injected in-memory embedding backend without MCP, LLM, OpenFang, RAG, or Docker.

**Tech Stack:** Python 3.11, NumPy, FAISS CPU, pytest, existing embeddinggemma.mcmp implementation, JSON.

## Global Constraints

- Do not start or modify the Fungus MCP server.
- Do not invoke an LLM, OpenFang, RAG generation, Docker, or a production embedding service.
- Gate 1 uses normalized vectors, faiss.METRIC_INNER_PRODUCT, CPU execution, and fixed seeds.
- Methods are A: Single Query + FAISS, B: Multi Query + FAISS, C: Single Query + MCMP, D: Multi Query + independent MCMP runs.
- D shares no agents, relevance, pheromone, or retriever state between queries.
- Literal relevance labels are ground truth; retrieval scores are never ground truth.
- MCMP_novel = MCMP_discovered_set - FAISS_initial_set.
- novel_relevant = MCMP_novel intersect relevant_document_ids.
- JSON under benchmarks/results/ is authoritative and retains raw candidate ids and configuration.
- A negative result is valid; production MCMP behavior must not be changed to make Gate 1 pass.
- Use .venv\Scripts\python.exe explicitly. The lock is reproducible under Python 3.11; .python-version currently selects Python 3.12, where pandas 1.5.3 fails to build.

---

## File map

- src/embeddinggemma/mcmp_rag.py: optional injection seam for an already-constructed embedding backend.
- benchmarks/mcmp/contracts.py: BenchmarkDataset and SearchRun validation.
- benchmarks/mcmp/metrics.py: ranking, geometry, overlap, and novelty metrics.
- benchmarks/mcmp/fixtures.py: deterministic synthetic vectors and labels.
- benchmarks/mcmp/adapters.py: FAISS and independent-run MCMP adapters.
- benchmarks/mcmp/run_gate1.py: A-D orchestration and JSON output.
- tests/benchmarks/: focused RED-GREEN coverage for every harness unit.
- benchmarks/results/gate1-seed-7.json: first machine-readable evidence.

### Task 1: Close the proven metric correction and pipeline analysis

**Files:**
- Modify: src/embeddinggemma/mcmp_rag.py:427
- Modify: tests/mcmp/test_mcmp_rag_openfang.py:96
- Create: docs/MCMP_FULL_PIPELINE_ANALYSIS.md

**Interfaces:**
- Consumes: current faiss_search inner-product scores.
- Produces: find_nearest_documents returns similarity consistently with the brute-force path; the analysis records base-commit behavior.

- [ ] **Step 1: Re-run focused regression**

    .venv\Scripts\python.exe -m pytest tests/mcmp/test_mcmp_rag_openfang.py::test_faiss_neighbours_report_inner_product_as_cosine_similarity -q

Expected: PASS. Recorded RED was [0.5, 1.0, 1.5], expected [1.0, 0.0, -1.0].

- [ ] **Step 2: Re-run all MCMP tests**

    .venv\Scripts\python.exe -m pytest tests/mcmp -q

Expected: 37 passed, 2 skipped or a higher passing count, with zero failures.

- [ ] **Step 3: Commit only code and regression test**

    git add -- src/embeddinggemma/mcmp_rag.py tests/mcmp/test_mcmp_rag_openfang.py
    git diff --cached --check
    git commit -m "fix: preserve FAISS similarity in MCMP neighbours"

- [ ] **Step 4: Validate and commit only the analysis**

    $report = Get-Content -Raw docs/MCMP_FULL_PIPELINE_ANALYSIS.md
    if ($report -notmatch 'current source does \*\*not\*\* implement a shared multi-query MCMP colony') { throw 'missing central finding' }
    git add -- docs/MCMP_FULL_PIPELINE_ANALYSIS.md
    git diff --cached --check
    git commit -m "docs: analyze the complete MCMP pipeline"

### Task 2: Add fail-closed benchmark contracts

**Files:**
- Create: benchmarks/__init__.py
- Create: benchmarks/mcmp/__init__.py
- Create: benchmarks/mcmp/contracts.py
- Create: tests/benchmarks/test_contracts.py

**Interfaces:**
- Produces: BenchmarkDataset.validate() -> None, BenchmarkDataset.digest() -> str, SearchRun.validate(dataset) -> None.

- [ ] **Step 1: Write the failing tests**

Create a literal two-document dataset and assert three behaviors:

    def test_dataset_rejects_unknown_relevant_document():
        dataset = valid_dataset()
        dataset.relevant_by_query["q0"] = frozenset({"missing"})
        with pytest.raises(ValueError, match="unknown relevant document"):
            dataset.validate()

    def test_dataset_digest_is_stable_for_equal_content():
        assert valid_dataset().digest() == valid_dataset().digest()

    def test_search_run_rejects_unknown_ranked_document():
        run = valid_run(ranked_document_ids=("missing",))
        with pytest.raises(ValueError, match="unknown ranked document"):
            run.validate(valid_dataset())

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_contracts.py -q

Expected: collection failure because benchmarks.mcmp.contracts does not exist.

- [ ] **Step 3: Implement exact records**

    @dataclass
    class BenchmarkDataset:
        dataset_id: str
        seed: int
        document_ids: Sequence[str]
        document_vectors: np.ndarray
        query_ids: Sequence[str]
        query_vectors: np.ndarray
        relevant_by_query: dict[str, frozenset[str]]

        # Public methods: validate(self) -> None and digest(self) -> str

    @dataclass(frozen=True)
    class SearchRun:
        method: str
        query_ids: Sequence[str]
        ranked_document_ids: Sequence[str]
        initial_candidate_ids: frozenset[str]
        discovered_candidate_ids: frozenset[str]
        per_query_candidate_ids: dict[str, frozenset[str]]
        per_query_ranked_document_ids: dict[str, Sequence[str]]
        elapsed_ms: float
        candidate_comparisons: int | None
        mcmp_steps: int
        document_visits: dict[str, int]
        pheromone_trails: int

        # Public method: validate(self, dataset: BenchmarkDataset) -> None

Replace the ellipses with checks for two-dimensional finite matrices, matching dimensions and row/id counts, unique ids, known labels, known run ids, duplicate rankings, nonnegative counters, and finite nonnegative elapsed time. Digest canonical ids plus little-endian float32 bytes with SHA-256.

- [ ] **Step 4: Verify GREEN and commit**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_contracts.py -q
    git add -- benchmarks/__init__.py benchmarks/mcmp/__init__.py benchmarks/mcmp/contracts.py tests/benchmarks/test_contracts.py
    git diff --cached --check
    git commit -m "feat: add MCMP benchmark contracts"

Expected: 3 passed.

### Task 3: Implement pure metrics with hand-derived expectations

**Files:**
- Create: benchmarks/mcmp/metrics.py
- Create: tests/benchmarks/test_metrics.py

**Interfaces:**
- Produces: evaluate_run(dataset, run, k), query_geometry(dataset), candidate_overlap(run).

- [ ] **Step 1: Write failing literal tests**

For ranked (d0, d1, d2), relevant {d1, d2}, initial {d0}, discovered {d1, d2}, assert:

    metrics = evaluate_run(dataset, run, k=3)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["reciprocal_rank"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["ndcg_at_k"] == pytest.approx(1.1309297536 / 1.6309297536)
    assert metrics["unique_relevant_documents"] == 2
    assert metrics["candidate_count"] == 2
    assert metrics["novel_candidates"] == ["d1", "d2"]
    assert metrics["novel_relevant_candidates"] == ["d1", "d2"]

For query vectors [1, 0] and [0, 1], assert mean/max cosine distance 1.0. For candidate sets {a, b} and {b, c}, assert Jaccard overlap 1/3 under q0|q1.

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_metrics.py -q

Expected: collection failure because benchmarks.mcmp.metrics does not exist.

- [ ] **Step 3: Implement formulas**

    def reciprocal_rank(ranked, relevant, k):
        for rank, document_id in enumerate(ranked[:k], start=1):
            if document_id in relevant:
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(ranked, relevant, k):
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, document_id in enumerate(ranked[:k], start=1)
            if document_id in relevant
        )
        ideal_hits = min(k, len(relevant))
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / idcg if idcg else 0.0

For multi-query fused rankings, union relevant labels for run.query_ids. Compute MRR as the arithmetic mean of reciprocal rank over per_query_ranked_document_ids and each query's own relevance labels. Sort emitted id lists for deterministic JSON.

- [ ] **Step 4: Verify GREEN and commit**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_metrics.py -q
    git add -- benchmarks/mcmp/metrics.py tests/benchmarks/test_metrics.py
    git diff --cached --check
    git commit -m "feat: measure MCMP ablation outcomes"

### Task 4: Generate a deterministic synthetic dataset

**Files:**
- Create: benchmarks/mcmp/fixtures.py
- Create: tests/benchmarks/test_fixtures.py

**Interfaces:**
- Produces: build_synthetic_dataset(seed: int = 7) -> BenchmarkDataset.

- [ ] **Step 1: Write failing fixture tests**

Assert document shape (8, 3), query shape (2, 3), unit norms, valid labels, identical digest for equal seeds, and different digest for seed 8. Labels are:

    {
        "q-main": frozenset({"main-near", "main-bridge"}),
        "q-related": frozenset({"related-near", "related-bridge"}),
    }

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_fixtures.py -q

Expected: collection failure because benchmarks.mcmp.fixtures does not exist.

- [ ] **Step 3: Implement vectors**

Use Gaussian jitter sigma 0.002 and normalize rows:

    documents = {
        "main-top": [1.0, 0.0, 0.0],
        "main-near": [0.98, 0.18, 0.0],
        "main-bridge": [0.78, 0.58, 0.0],
        "related-top": [0.0, 1.0, 0.0],
        "related-near": [0.18, 0.98, 0.0],
        "related-bridge": [0.58, 0.78, 0.0],
        "z-distractor": [0.0, 0.0, 1.0],
        "opposite": [-1.0, 0.0, 0.0],
    }
    queries = {
        "q-main": [1.0, 0.0, 0.0],
        "q-related": [0.0, 1.0, 0.0],
    }

Call dataset.validate() before returning.

- [ ] **Step 4: Verify GREEN and commit**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_fixtures.py tests/benchmarks/test_contracts.py -q
    git add -- benchmarks/mcmp/fixtures.py tests/benchmarks/test_fixtures.py
    git diff --cached --check
    git commit -m "test: add deterministic MCMP vector fixture"

### Task 5: Add the embedding seam and A-D adapters

**Files:**
- Modify: src/embeddinggemma/mcmp_rag.py:62
- Modify: tests/mcmp/test_mcmp_rag_openfang.py
- Create: benchmarks/mcmp/adapters.py
- Create: tests/benchmarks/test_adapters.py

**Interfaces:**
- Produces: run_faiss(dataset, method, query_ids, top_k, initial_k) and run_mcmp(dataset, method, query_ids, top_k, initial_k, seed, num_agents, steps), each returning SearchRun plus AdapterEvidence.

- [ ] **Step 1: Write and prove RED for injection seam**

    def test_retriever_accepts_explicit_embedding_backend():
        backend = object()
        retriever = mcmp_rag.MCPMRetriever(embedding_backend=(backend, 3))
        assert retriever.embedding_model is backend
        assert retriever._expected_embedding_dim == 3

Run:

    .venv\Scripts\python.exe -m pytest tests/mcmp/test_mcmp_rag_openfang.py::test_retriever_accepts_explicit_embedding_backend -q

Expected: FAIL with unexpected keyword argument embedding_backend.

- [ ] **Step 2: Implement only the optional seam**

Append constructor parameter:

    embedding_backend: Optional[Tuple[Any, int]] = None

Initialize with:

    if embedding_backend is None:
        self.embedding_model, self._expected_embedding_dim = load_embedding_backend()
    else:
        self.embedding_model, self._expected_embedding_dim = embedding_backend

Do not change default runtime behavior or simulation scoring.

- [ ] **Step 3: Verify MCMP GREEN**

    .venv\Scripts\python.exe -m pytest tests/mcmp -q

Expected: all existing cases plus injection test pass.

- [ ] **Step 4: Write adapter RED tests**

Assert FAISS q-main begins with main-top; B exposes both per-query candidate sets and per-query rankings; equal MCMP seeds produce identical rankings, visits, and trails; C reports independent_run_count 1 and D reports 2; no mutable retriever is returned.

- [ ] **Step 5: Implement adapters**

MappingEmbeddingBackend.encode maps document/query ids to copied vectors. CountingRetriever increments nearest_search_calls and delegates to super. For each MCMP query:

    np.random.seed(seed + query_index)
    retriever = CountingRetriever(
        num_agents=num_agents,
        max_iterations=steps,
        build_faiss_after_add=True,
        embedding_backend=(backend, dataset.document_vectors.shape[1]),
    )
    retriever.add_documents(list(dataset.document_ids), cache=False)
    retriever.initialize_simulation(query_id)
    retriever.step(steps)

Define discovered candidates as documents with visit_count > 0. Initial candidates are direct FAISS top initial_k. For Gate-1 Flat indexes, comparisons equal nearest_search_calls times document count. Fuse multiple rankings by maximum relevance, ties by document id. Create a fresh backend and retriever for every query.

- [ ] **Step 6: Verify RED-GREEN scope and commit in two units**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_adapters.py tests/mcmp -q

Then:

    git add -- src/embeddinggemma/mcmp_rag.py tests/mcmp/test_mcmp_rag_openfang.py
    git commit -m "refactor: allow explicit MCMP embedding backend"
    git add -- benchmarks/mcmp/adapters.py tests/benchmarks/test_adapters.py
    git commit -m "feat: add FAISS and MCMP ablation adapters"

Expected: zero failures and two narrow commits.

### Task 6: Orchestrate Gate 1 and persist evidence

**Files:**
- Modify: benchmarks/mcmp/__init__.py
- Create: benchmarks/mcmp/run_gate1.py
- Create: tests/benchmarks/test_gate1_runner.py
- Create: benchmarks/results/gate1-seed-7.json

**Interfaces:**
- Produces: run_gate1(seed, top_k, initial_k, num_agents, steps), write_gate1_result(payload, path), and CLI exit 0 only for complete A-D evidence.

- [ ] **Step 1: Write failing orchestration test**

    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)
    assert list(payload["runs"]) == ["A", "B", "C", "D"]
    assert payload["config"] == {
        "seed": 7,
        "top_k": 4,
        "initial_k": 1,
        "num_agents": 24,
        "steps": 10,
    }
    assert payload["conclusion"] in {
        "novel_relevant_observed",
        "no_novel_relevant_observed",
    }
    assert payload["runs"]["D"]["independent_run_count"] == 2

Write to tmp_path, reload JSON, and assert exact equality.

- [ ] **Step 2: Verify RED**

    .venv\Scripts\python.exe -m pytest tests/benchmarks/test_gate1_runner.py -q

Expected: collection failure because benchmarks.mcmp.run_gate1 does not exist.

- [ ] **Step 3: Implement fixed A-D orchestration**

Record dataset id/digest, Python/NumPy/FAISS versions, CPU mode, raw ids, query geometry, overlap, ranking metrics, timing, comparisons, visits, and trails. Derive conclusion only from C/D:

    novel_count = sum(
        len(payload["runs"][method]["metrics"]["novel_relevant_candidates"])
        for method in ("C", "D")
    )
    payload["conclusion"] = (
        "novel_relevant_observed"
        if novel_count > 0
        else "no_novel_relevant_observed"
    )

Serialize with indent 2, sorted keys, and a final newline.

- [ ] **Step 4: Verify runner GREEN**

    .venv\Scripts\python.exe -m pytest tests/benchmarks -q

Expected: all benchmark tests pass.

- [ ] **Step 5: Generate evidence without editing its conclusion**

    .venv\Scripts\python.exe -m benchmarks.mcmp.run_gate1 --seed 7 --top-k 4 --initial-k 1 --num-agents 24 --steps 10 --output benchmarks/results/gate1-seed-7.json

Expected: exit 0 and valid A-D JSON with one permitted conclusion.

- [ ] **Step 6: Verify full offline scope**

    .venv\Scripts\python.exe -m pytest tests/benchmarks tests/mcmp -q
    git diff --check
    git status --short

Expected: zero failures and only planned files.

- [ ] **Step 7: Commit runner and evidence**

    git add -- benchmarks/mcmp/__init__.py benchmarks/mcmp/run_gate1.py tests/benchmarks/test_gate1_runner.py benchmarks/results/gate1-seed-7.json
    git diff --cached --check
    git commit -m "feat: run offline MCMP Gate 1 ablation"

### Task 7: Review Gate 1 before Gate 2 or TIG

**Files:**
- Read: benchmarks/results/gate1-seed-7.json
- Read: docs/MCMP_FULL_PIPELINE_ANALYSIS.md
- Read: docs/superpowers/specs/2026-08-08-mcmp-ablation-harness-design.md

**Interfaces:**
- Produces: evidence summary and decision; no code changes.

- [ ] **Step 1: Report A-D measurements**

Report Recall@K, RR, NDCG@K, novel candidates, novel relevant candidates, comparisons, visits, trails, and elapsed time. Separate facts from interpretation.

- [ ] **Step 2: Apply decision rule**

    If C and D have zero novel relevant candidates:
      Gate 1 found no MCMP discovery benefit for this fixture.

    If C or D has novel relevant candidates:
      repeat Gate 1 over a declared seed set before claiming reproducibility.

    In both cases:
      do not start Gate 2, query colonies, or TIG without evidence review.

- [ ] **Step 3: Record repository evidence**

    git status --short --branch
    git log --oneline --decorate -12

Report branch, commits, dirty state, test count, Python 3.11 constraint, and non-claims: no MCP, LLM, OpenFang, Docker, Gate 2, or TIG execution.
