from __future__ import annotations

import numpy as np

import pytest

from benchmarks.gate2.geometry import (
    MIN_ABSOLUTE_SIGNATURE,
    MIN_RELATIVE_EXCESS,
    NULL_ALPHA,
    NULL_PERMUTATIONS,
    chain_reachable,
    characterise,
    geometry_cache,
    knn_graph,
    permuted_labels,
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


def test_null_parameters_are_pre_registered() -> None:
    assert NULL_PERMUTATIONS == 100
    assert NULL_ALPHA == 0.05
    assert MIN_ABSOLUTE_SIGNATURE == 0.10
    assert MIN_RELATIVE_EXCESS == 0.10


def test_stage_two_gate_requires_all_three_conditions() -> None:
    low_null = [0.10] * 100  # median .10, ceiling .90, required excess .09

    assert stage_two_is_justified(0.30, low_null)
    # Significant, above the absolute floor, but the excess is 5 points against
    # a required 9: too small to carry the measured cost.
    assert not stage_two_is_justified(0.15, low_null)
    # A large signature the null reaches just as easily is not evidence.
    assert not stage_two_is_justified(0.30, [0.50] * 100)


def test_the_absolute_floor_bites_independently_of_the_null() -> None:
    """A tiny signature can be significant against a tiny null and still be
    worthless: a walk can only recover pairs that are far in the first place,
    and that is what the Gate 1 cost measurement speaks to."""
    zero_null = [0.0] * 100

    assert 0.05 > 0.0  # significant against this null
    assert not stage_two_is_justified(0.05, zero_null)
    assert stage_two_is_justified(0.20, zero_null)


def test_the_margin_scales_with_the_achievable_ceiling() -> None:
    """The regression against a rule that could not pass.

    A flat 10-point margin is unsatisfiable wherever the null median exceeds
    0.90 -- measured at 0.96 on a synthetic corpus, where the largest possible
    excess is 0.043. Against the ceiling the required margin there is 0.004,
    so real structure can still clear it.
    """
    saturated_null = [0.96] * 100

    assert stage_two_is_justified(0.98, saturated_null)
    # Under the flat-margin rule this was False: 0.98 - 0.96 = 0.02 < 0.10.
    assert (0.98 - 0.96) < 0.10


def test_stage_two_gate_refuses_an_empty_null() -> None:
    with pytest.raises(ValueError, match="null distribution"):
        stage_two_is_justified(0.9, [])


def test_permuted_labels_keep_the_geometry_and_the_set_sizes() -> None:
    dataset = _chain_dataset()

    permuted = permuted_labels(dataset, np.random.default_rng(0))

    assert permuted.document_ids == dataset.document_ids
    assert permuted.query_ids == dataset.query_ids
    assert np.array_equal(permuted.document_vectors, dataset.document_vectors)
    assert np.array_equal(permuted.query_vectors, dataset.query_vectors)
    for query_id in dataset.query_ids:
        assert len(permuted.relevant_by_query[query_id]) == len(
            dataset.relevant_by_query[query_id]
        )
        assert set(permuted.relevant_by_query[query_id]) <= set(dataset.document_ids)
    permuted.validate()
    # The original must be untouched, or the comparison against it would be
    # self-consistent and meaningless.
    assert dataset.relevant_by_query == {"q:probe": frozenset({"d5"})}


def test_permuted_labels_are_deterministic_and_vary_across_seeds() -> None:
    dataset = _chain_dataset()

    first = permuted_labels(dataset, np.random.default_rng(3))
    second = permuted_labels(dataset, np.random.default_rng(3))

    assert first.relevant_by_query == second.relevant_by_query

    observed = {
        frozenset(
            permuted_labels(dataset, np.random.default_rng(seed)).relevant_by_query["q:probe"]
        )
        for seed in range(20)
    }
    assert len(observed) > 1


def test_permutation_does_not_change_the_pair_count() -> None:
    dataset = _chain_dataset()
    permuted = permuted_labels(dataset, np.random.default_rng(1))

    real = characterise(dataset, top_k=2, knn_k=2)
    null = characterise(permuted, top_k=2, knn_k=2)

    assert null["pair_count"] == real["pair_count"]


def test_a_precomputed_cache_does_not_change_the_measurement() -> None:
    dataset = _chain_dataset()
    cache = geometry_cache(dataset, knn_k=2)

    with_cache = characterise(dataset, top_k=2, knn_k=2, cache=cache)
    without_cache = characterise(dataset, top_k=2, knn_k=2)

    assert with_cache == without_cache


def test_a_cache_built_for_a_different_knn_k_is_rejected() -> None:
    """Silently honouring a mismatched cache would answer with the wrong graph."""
    dataset = _chain_dataset()
    cache = geometry_cache(dataset, knn_k=2)

    with pytest.raises(ValueError, match="cache was built for knn_k"):
        characterise(dataset, top_k=2, knn_k=5, cache=cache)

    with pytest.raises(ValueError, match="cache was built for knn_k"):
        chain_reachable(dataset, "q:probe", "d5", 5, 6, 0.0, cache=cache)


def test_a_cache_built_from_another_dataset_is_rejected() -> None:
    """Every stage-1 seed's corpus has the same shape, so a mis-zipped cache
    would otherwise answer with a different corpus's graph in silence. A label
    permutation shares the vector buffer by construction and must still pass."""
    dataset = _chain_dataset()
    other = _chain_dataset()  # identical values, distinct array object
    cache = geometry_cache(dataset, knn_k=2)

    with pytest.raises(ValueError, match="different dataset's vectors"):
        characterise(other, top_k=2, knn_k=2, cache=cache)

    permuted = permuted_labels(dataset, np.random.default_rng(0))
    assert characterise(permuted, top_k=2, knn_k=2, cache=cache)["pair_count"] == 1
