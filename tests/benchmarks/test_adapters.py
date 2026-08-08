from __future__ import annotations

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
