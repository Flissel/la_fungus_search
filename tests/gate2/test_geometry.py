from __future__ import annotations

import numpy as np

from benchmarks.gate2.geometry import (
    MANIFOLD_SIGNATURE_GATE,
    chain_reachable,
    characterise,
    knn_graph,
    relevant_ranks,
    stage_two_is_justified,
)
from benchmarks.mcmp.contracts import BenchmarkDataset


def _chain_dataset() -> BenchmarkDataset:
    angles = [0.0, 0.35, 0.70, 1.05, 1.40, 1.55]
    documents = np.asarray(
        [[np.cos(angle), np.sin(angle), 0.0] for angle in angles], dtype=np.float32
    )
    return BenchmarkDataset(
        dataset_id="chain",
        seed=0,
        document_ids=tuple(f"d{index}" for index in range(len(angles))),
        document_vectors=documents,
        query_ids=("q:probe",),
        query_vectors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        relevant_by_query={"q:probe": frozenset({"d5"})},
    )


def test_relevant_ranks_are_one_based_similarity_positions() -> None:
    dataset = _chain_dataset()

    assert relevant_ranks(dataset, "q:probe") == (6,)


def test_knn_graph_is_mutual() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32
    )

    graph = knn_graph(vectors, knn_k=1)

    assert graph[0] == frozenset({1})
    assert graph[1] == frozenset({0})
    assert graph[2] == frozenset()


def test_chain_reachable_finds_a_multi_hop_path() -> None:
    dataset = _chain_dataset()

    assert chain_reachable(
        dataset, "q:probe", "d5", knn_k=2, max_hops=6, hop_threshold=0.0
    )


def test_chain_reachable_respects_the_hop_budget() -> None:
    dataset = _chain_dataset()

    assert not chain_reachable(
        dataset, "q:probe", "d5", knn_k=2, max_hops=1, hop_threshold=0.0
    )


def test_characterise_reports_a_manifold_signature() -> None:
    dataset = _chain_dataset()

    report = characterise(dataset, top_k=2, knn_k=2, max_hops=6, hop_threshold=0.0)

    assert report["pair_count"] == 1
    assert report["far_count"] == 1
    assert report["far_and_reachable_count"] == 1
    assert report["manifold_signature"] == 1.0
    assert report["pairs"][0]["document_id"] == "d5"


def test_stage_two_gate_is_pre_registered_at_ten_percent() -> None:
    assert MANIFOLD_SIGNATURE_GATE == 0.10
    assert stage_two_is_justified(0.10)
    assert not stage_two_is_justified(0.09)
