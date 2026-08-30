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


def test_chain_reachable_prunes_hops_below_the_threshold() -> None:
    """The chain's hops sit at cosine 0.9394 (0.9888 for the last one).

    At the default threshold of 0.0 nothing is pruned and d5 is reached. A
    threshold of 0.95 is above every hop out of the start node, so the path
    must be cut and d5 must become unreachable. Deleting the pruning branch
    makes the second assertion fail.
    """
    dataset = _chain_dataset()

    assert chain_reachable(
        dataset, "q:probe", "d5", knn_k=2, max_hops=6, hop_threshold=0.0
    )
    assert not chain_reachable(
        dataset, "q:probe", "d5", knn_k=2, max_hops=6, hop_threshold=0.95
    )


def test_characterise_reports_a_manifold_signature() -> None:
    dataset = _chain_dataset()

    report = characterise(dataset, top_k=2, knn_k=2, max_hops=6, hop_threshold=0.0)

    assert report["pair_count"] == 1
    assert report["far_count"] == 1
    assert report["far_and_reachable_count"] == 1
    assert report["manifold_signature"] == 1.0
    assert report["pairs"][0]["document_id"] == "d5"
    assert report["pairs"][0]["rank"] == relevant_ranks(dataset, "q:probe")[0]
    assert report["pairs"][0]["chain_reachable"] is True


def test_characterise_records_near_pairs_as_not_measured() -> None:
    """Reachability is computed only for far pairs.

    A near pair therefore has no measurement, and must not be recorded as
    ``False`` -- a reader tallying chain_reachable across the raw records
    would count it as "walk cannot reach this" and undercount.
    """
    dataset = _chain_dataset()
    dataset = BenchmarkDataset(
        dataset_id=dataset.dataset_id,
        seed=dataset.seed,
        document_ids=dataset.document_ids,
        document_vectors=dataset.document_vectors,
        query_ids=dataset.query_ids,
        query_vectors=dataset.query_vectors,
        relevant_by_query={"q:probe": frozenset({"d1", "d5"})},
    )

    report = characterise(dataset, top_k=2, knn_k=2, max_hops=6, hop_threshold=0.0)

    near, far = report["pairs"]
    assert near["document_id"] == "d1"
    assert near["far"] is False
    assert near["chain_reachable"] is None
    assert far["document_id"] == "d5"
    assert far["far"] is True
    assert far["chain_reachable"] is True

    # The aggregates gate on `far`, so "not measured" is invisible to them.
    assert report["pair_count"] == 2
    assert report["far_count"] == 1
    assert report["far_and_reachable_count"] == 1
    assert report["manifold_signature"] == 0.5


def test_stage_two_gate_is_pre_registered_at_ten_percent() -> None:
    assert MANIFOLD_SIGNATURE_GATE == 0.10
    assert stage_two_is_justified(0.10)
    assert not stage_two_is_justified(0.09)
