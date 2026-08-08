from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.mcmp.adapters import run_faiss, run_mcmp
from benchmarks.mcmp.fixtures import build_synthetic_dataset


def test_faiss_ranks_main_query_by_inner_product() -> None:
    dataset = build_synthetic_dataset()

    run, evidence = run_faiss(dataset, "A", ("q-main",), top_k=4, initial_k=1)

    assert run.per_query_ranked_document_ids["q-main"][0] == "main-top"
    assert run.initial_candidate_ids == frozenset({"main-top"})
    assert evidence.independent_run_count == 1


def test_multi_query_faiss_preserves_each_query_candidates_and_rankings() -> None:
    dataset = build_synthetic_dataset()

    run, _evidence = run_faiss(
        dataset, "B", ("q-main", "q-related"), top_k=4, initial_k=1
    )

    assert run.per_query_candidate_ids == {
        "q-main": frozenset({"main-top"}),
        "q-related": frozenset({"related-top"}),
    }
    assert run.per_query_ranked_document_ids["q-main"][0] == "main-top"
    assert run.per_query_ranked_document_ids["q-related"][0] == "related-top"


def test_mcmp_seed_reproduces_rankings_visits_and_trails() -> None:
    dataset = build_synthetic_dataset()
    arguments = dict(
        dataset=dataset,
        method="C",
        query_ids=("q-main",),
        top_k=4,
        initial_k=1,
        seed=7,
        num_agents=24,
        steps=10,
    )

    first, _first_evidence = run_mcmp(**arguments)
    second, _second_evidence = run_mcmp(**arguments)

    assert first.ranked_document_ids == second.ranked_document_ids
    assert first.document_visits == second.document_visits
    assert first.pheromone_trails == second.pheromone_trails


def test_mcmp_evidence_counts_independent_runs_without_exposing_retriever() -> None:
    dataset = build_synthetic_dataset()

    _single_run, single_evidence = run_mcmp(
        dataset, "C", ("q-main",), top_k=4, initial_k=1, seed=7, num_agents=24, steps=10
    )
    _multi_run, multi_evidence = run_mcmp(
        dataset,
        "D",
        ("q-main", "q-related"),
        top_k=4,
        initial_k=1,
        seed=7,
        num_agents=24,
        steps=10,
    )

    assert single_evidence.independent_run_count == 1
    assert multi_evidence.independent_run_count == 2
    assert not hasattr(single_evidence, "retriever")
    assert not hasattr(multi_evidence, "retriever")


@pytest.mark.parametrize(
    ("runner", "method", "query_ids"),
    [
        (run_faiss, "C", ("q-main",)),
        (run_faiss, "A", ("q-main", "q-related")),
        (run_faiss, "B", ("q-main",)),
        (run_faiss, "B", ("q-main", "q-main")),
        (run_mcmp, "A", ("q-main",)),
        (run_mcmp, "C", ("q-main", "q-related")),
        (run_mcmp, "D", ("q-main",)),
        (run_mcmp, "D", ("q-main", "q-main")),
    ],
)
def test_adapters_reject_mislabeled_or_wrong_cardinality_ablation_runs(
    runner, method: str, query_ids: tuple[str, ...]
) -> None:
    dataset = build_synthetic_dataset()
    kwargs = dict(dataset=dataset, method=method, query_ids=query_ids, top_k=4, initial_k=1)
    if runner is run_mcmp:
        kwargs.update(seed=7, num_agents=1, steps=1)

    with pytest.raises(ValueError, match="method|query ids"):
        runner(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 0},
        {"top_k": True},
        {"top_k": 9},
        {"initial_k": 0},
        {"initial_k": True},
        {"initial_k": 5},
        {"steps": 0},
        {"steps": True},
        {"num_agents": 0},
        {"num_agents": True},
        {"seed": True},
        {"seed": -1},
    ],
)
def test_mcmp_rejects_invalid_scalar_parameters_before_execution(overrides: dict[str, int]) -> None:
    dataset = build_synthetic_dataset()
    kwargs = dict(
        dataset=dataset,
        method="C",
        query_ids=("q-main",),
        top_k=4,
        initial_k=1,
        seed=7,
        num_agents=1,
        steps=1,
    )
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        run_mcmp(**kwargs)


def test_adapter_allows_document_count_k_boundaries() -> None:
    dataset = build_synthetic_dataset()

    faiss_run, _evidence = run_faiss(dataset, "A", ("q-main",), top_k=8, initial_k=8)
    mcmp_run, _evidence = run_mcmp(
        dataset, "C", ("q-main",), top_k=8, initial_k=8, seed=0, num_agents=1, steps=1
    )

    assert len(faiss_run.ranked_document_ids) == 8
    assert len(mcmp_run.ranked_document_ids) == 8


def test_adapter_rejects_document_query_identifier_collisions() -> None:
    dataset = build_synthetic_dataset()
    colliding = replace(
        dataset,
        query_ids=("main-top", "q-related"),
        relevant_by_query={
            "main-top": frozenset({"main-near"}),
            "q-related": frozenset({"related-near"}),
        },
    )

    with pytest.raises(ValueError, match="document and query ids must be disjoint"):
        run_faiss(colliding, "A", ("main-top",), top_k=4, initial_k=1)


@pytest.mark.parametrize(
    ("method", "query_ids", "seed"),
    [
        ("C", ("q-main",), np.uint64(2**32 - 1)),
        ("D", ("q-main", "q-related"), np.uint64(2**32 - 2)),
    ],
)
def test_mcmp_accepts_safe_uint64_seed_deterministically(
    method: str, query_ids: tuple[str, ...], seed: np.uint64
) -> None:
    dataset = build_synthetic_dataset()
    arguments = dict(
        dataset=dataset,
        method=method,
        query_ids=query_ids,
        top_k=4,
        initial_k=1,
        seed=seed,
        num_agents=1,
        steps=1,
    )

    first, _first_evidence = run_mcmp(**arguments)
    second, _second_evidence = run_mcmp(**arguments)

    assert first.ranked_document_ids == second.ranked_document_ids
    assert first.document_visits == second.document_visits
    assert first.pheromone_trails == second.pheromone_trails
