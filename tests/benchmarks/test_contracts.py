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
    *, ranked_document_ids: tuple[str, ...] = ("d0",)
) -> SearchRun:
    return SearchRun(
        method="literal",
        query_ids=("q0",),
        ranked_document_ids=ranked_document_ids,
        initial_candidate_ids=frozenset({"d0"}),
        discovered_candidate_ids=frozenset(),
        per_query_candidate_ids={"q0": frozenset({"d0"})},
        per_query_ranked_document_ids={"q0": ranked_document_ids},
        elapsed_ms=1.0,
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
