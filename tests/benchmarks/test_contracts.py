from __future__ import annotations

import numpy as np
import pytest

from benchmarks.mcmp.contracts import BenchmarkDataset, SearchRun


def valid_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        dataset_id="literal-two-document",
        seed=7,
        document_ids=("d0", "d1"),
        document_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_ids=("q0",),
        query_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        relevant_by_query={"q0": frozenset({"d0"})},
    )


def valid_run(
    *, ranked_document_ids: tuple[str, ...] = ("d0",), elapsed_ms: float = 1.0
) -> SearchRun:
    return SearchRun(
        method="literal",
        query_ids=("q0",),
        ranked_document_ids=ranked_document_ids,
        initial_candidate_ids=frozenset({"d0"}),
        discovered_candidate_ids=frozenset({"d0"}),
        per_query_candidate_ids={"q0": frozenset({"d0"})},
        per_query_ranked_document_ids={"q0": ranked_document_ids},
        elapsed_ms=elapsed_ms,
        candidate_comparisons=1,
        mcmp_steps=0,
        document_visits={"d0": 1},
        pheromone_trails=0,
    )


def test_dataset_rejects_unknown_relevant_document() -> None:
    dataset = valid_dataset()
    dataset.relevant_by_query["q0"] = frozenset({"missing"})

    with pytest.raises(ValueError, match="unknown relevant document"):
        dataset.validate()


def test_dataset_digest_is_stable_for_equal_content() -> None:
    assert valid_dataset().digest() == valid_dataset().digest()


def test_search_run_rejects_unknown_ranked_document() -> None:
    run = valid_run(ranked_document_ids=("missing",))

    with pytest.raises(ValueError, match="unknown ranked document"):
        run.validate(valid_dataset())


def test_search_run_rejects_complex_elapsed_time() -> None:
    run = valid_run(elapsed_ms=np.complex128(1 + 0j))

    with pytest.raises(ValueError, match="elapsed_ms"):
        run.validate(valid_dataset())


def test_dataset_rejects_vectors_that_overflow_float32_digest() -> None:
    dataset = valid_dataset()
    overflowing_float32_value = 2.0 * float(np.finfo(np.float32).max)
    assert np.isfinite(overflowing_float32_value)
    assert overflowing_float32_value > float(np.finfo(np.float32).max)
    dataset.document_vectors = np.asarray(
        [[overflowing_float32_value, 0.0], [0.0, 1.0]], dtype=np.float64
    )

    with pytest.raises(ValueError, match="float32"):
        dataset.validate()


@pytest.mark.parametrize("vector_field", ["document_vectors", "query_vectors"])
def test_dataset_rejects_nonpositive_vector_dimensions(vector_field: str) -> None:
    dataset = valid_dataset()
    if vector_field == "document_vectors":
        dataset.document_ids = ()
        dataset.document_vectors = np.empty((0, 2), dtype=np.float32)
        dataset.relevant_by_query = {"q0": frozenset()}
    else:
        dataset.query_ids = ()
        dataset.query_vectors = np.empty((0, 2), dtype=np.float32)
        dataset.relevant_by_query = {}

    with pytest.raises(ValueError, match="positive dimensions"):
        dataset.validate()


@pytest.mark.parametrize("vector_field", ["document_vectors", "query_vectors"])
def test_dataset_rejects_vectors_that_are_not_unit_normalized(vector_field: str) -> None:
    dataset = valid_dataset()
    setattr(
        dataset,
        vector_field,
        np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        if vector_field == "document_vectors"
        else np.asarray([[2.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="unit-normalized"):
        dataset.validate()


def test_dataset_requires_relevance_labels_for_exactly_its_query_ids() -> None:
    dataset = valid_dataset()
    dataset.relevant_by_query = {}

    with pytest.raises(ValueError, match="exactly match query ids"):
        dataset.validate()


def test_search_run_requires_global_discoveries_to_match_per_query_candidates() -> None:
    run = valid_run()
    run = SearchRun(
        method=run.method,
        query_ids=run.query_ids,
        ranked_document_ids=run.ranked_document_ids,
        initial_candidate_ids=run.initial_candidate_ids,
        discovered_candidate_ids=frozenset(),
        per_query_candidate_ids=run.per_query_candidate_ids,
        per_query_ranked_document_ids=run.per_query_ranked_document_ids,
        elapsed_ms=run.elapsed_ms,
        candidate_comparisons=run.candidate_comparisons,
        mcmp_steps=run.mcmp_steps,
        document_visits=run.document_visits,
        pheromone_trails=run.pheromone_trails,
    )

    with pytest.raises(ValueError, match="discovered candidates must match"):
        run.validate(valid_dataset())
