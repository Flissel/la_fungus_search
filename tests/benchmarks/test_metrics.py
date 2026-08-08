from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.mcmp.contracts import BenchmarkDataset, SearchRun
from benchmarks.mcmp.metrics import (
    candidate_overlap,
    evaluate_run,
    ndcg_at_k,
    query_geometry,
    reciprocal_rank,
)


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


def test_evaluate_run_counts_all_discovered_candidates_separately_from_novel_ones() -> None:
    run = replace(
        literal_run(),
        initial_candidate_ids=frozenset({"d0", "d1"}),
        discovered_candidate_ids=frozenset({"d1", "d2"}),
    )

    metrics = evaluate_run(literal_dataset(), run, k=3)

    assert metrics["candidate_count"] == 2
    assert metrics["novel_candidates"] == ["d2"]
    assert metrics["novel_relevant_candidates"] == ["d2"]


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_ranking_metrics_reject_nonpositive_or_noninteger_k(k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        reciprocal_rank(("d0",), frozenset({"d0"}), k)
    with pytest.raises(ValueError, match="positive integer"):
        ndcg_at_k(("d0",), frozenset({"d0"}), k)
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_run(literal_dataset(), literal_run(), k)


def test_query_geometry_reports_mean_and_max_cosine_distance() -> None:
    geometry = query_geometry(literal_dataset())

    assert geometry["mean_cosine_distance"] == pytest.approx(1.0)
    assert geometry["max_cosine_distance"] == pytest.approx(1.0)


def test_query_geometry_rejects_zero_norm_query_vectors() -> None:
    dataset = literal_dataset()
    dataset.query_vectors = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="zero-norm query vector"):
        query_geometry(dataset)


def test_evaluate_run_averages_reciprocal_rank_over_each_querys_own_labels() -> None:
    run = SearchRun(
        method="literal",
        query_ids=("q0", "q1"),
        ranked_document_ids=("d0", "d1", "d2"),
        initial_candidate_ids=frozenset(),
        discovered_candidate_ids=frozenset({"d0", "d1", "d2"}),
        per_query_candidate_ids={
            "q0": frozenset({"d0", "d1", "d2"}),
            "q1": frozenset({"d0", "d1", "d2"}),
        },
        per_query_ranked_document_ids={
            "q0": ("d0", "d1", "d2"),
            "q1": ("d2", "d1", "d0"),
        },
        elapsed_ms=1.0,
        candidate_comparisons=3,
        mcmp_steps=1,
        document_visits={"d0": 1, "d1": 1, "d2": 1},
        pheromone_trails=1,
    )

    metrics = evaluate_run(literal_dataset(), run, k=3)

    assert metrics["mrr"] == pytest.approx((0.5 + (1.0 / 3.0)) / 2.0)


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


def test_candidate_overlap_returns_zero_for_two_empty_candidate_sets() -> None:
    run = SearchRun(
        method="literal",
        query_ids=("q0", "q1"),
        ranked_document_ids=("d0",),
        initial_candidate_ids=frozenset(),
        discovered_candidate_ids=frozenset(),
        per_query_candidate_ids={"q0": frozenset(), "q1": frozenset()},
        per_query_ranked_document_ids={"q0": ("d0",), "q1": ("d0",)},
        elapsed_ms=1.0,
        candidate_comparisons=0,
        mcmp_steps=0,
        document_visits={},
        pheromone_trails=0,
    )

    assert candidate_overlap(run) == {"q0|q1": 0.0}
