from __future__ import annotations

import numpy as np
import pytest

from benchmarks.mcmp.contracts import BenchmarkDataset, SearchRun
from benchmarks.mcmp.metrics import candidate_overlap, evaluate_run, query_geometry


def literal_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        dataset_id="literal-metrics",
        seed=7,
        document_ids=("d0", "d1", "d2"),
        document_vectors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
        ),
        query_ids=("q0", "q1"),
        query_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        relevant_by_query={
            "q0": frozenset({"d1", "d2"}),
            "q1": frozenset({"d0"}),
        },
    )


def literal_run() -> SearchRun:
    return SearchRun(
        method="literal",
        query_ids=("q0",),
        ranked_document_ids=("d0", "d1", "d2"),
        initial_candidate_ids=frozenset({"d0"}),
        discovered_candidate_ids=frozenset({"d1", "d2"}),
        per_query_candidate_ids={"q0": frozenset({"d1", "d2"})},
        per_query_ranked_document_ids={"q0": ("d0", "d1", "d2")},
        elapsed_ms=1.0,
        candidate_comparisons=2,
        mcmp_steps=1,
        document_visits={"d0": 1, "d1": 1, "d2": 1},
        pheromone_trails=1,
    )


def test_evaluate_run_reports_hand_derived_ranking_and_candidate_metrics() -> None:
    metrics = evaluate_run(literal_dataset(), literal_run(), k=3)

    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["reciprocal_rank"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["ndcg_at_k"] == pytest.approx(1.1309297536 / 1.6309297536)
    assert metrics["unique_relevant_documents"] == 2
    assert metrics["candidate_count"] == 2
    assert metrics["novel_candidates"] == ["d1", "d2"]
    assert metrics["novel_relevant_candidates"] == ["d1", "d2"]


def test_query_geometry_reports_mean_and_max_cosine_distance() -> None:
    geometry = query_geometry(literal_dataset())

    assert geometry["mean_cosine_distance"] == pytest.approx(1.0)
    assert geometry["max_cosine_distance"] == pytest.approx(1.0)


def test_candidate_overlap_reports_pairwise_jaccard_by_query_pair() -> None:
    run = SearchRun(
        method="literal",
        query_ids=("q0", "q1"),
        ranked_document_ids=("d0",),
        initial_candidate_ids=frozenset(),
        discovered_candidate_ids=frozenset(),
        per_query_candidate_ids={
            "q0": frozenset({"a", "b"}),
            "q1": frozenset({"b", "c"}),
        },
        per_query_ranked_document_ids={"q0": ("d0",), "q1": ("d0",)},
        elapsed_ms=1.0,
        candidate_comparisons=0,
        mcmp_steps=0,
        document_visits={},
        pheromone_trails=0,
    )

    assert candidate_overlap(run) == {"q0|q1": pytest.approx(1.0 / 3.0)}
