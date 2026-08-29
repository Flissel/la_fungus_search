from __future__ import annotations

import numpy as np

from benchmarks.mcmp.fixtures import build_synthetic_dataset


def _expected_seed_seven_vectors() -> tuple[np.ndarray, np.ndarray]:
    document_bases = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.18, 0.0],
            [0.78, 0.58, 0.0],
            [0.0, 1.0, 0.0],
            [0.18, 0.98, 0.0],
            [0.58, 0.78, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    query_bases = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(7)

    documents = document_bases + rng.normal(0.0, 0.002, size=document_bases.shape).astype(
        np.float32
    )
    queries = query_bases + rng.normal(0.0, 0.002, size=query_bases.shape).astype(
        np.float32
    )
    return (
        documents / np.linalg.norm(documents, axis=1, keepdims=True),
        queries / np.linalg.norm(queries, axis=1, keepdims=True),
    )


def test_synthetic_dataset_has_expected_shape_ids_vectors_and_labels() -> None:
    dataset = build_synthetic_dataset()

    assert dataset.document_vectors.shape == (8, 3)
    assert dataset.query_vectors.shape == (2, 3)
    assert dataset.document_ids == (
        "main-top",
        "main-near",
        "main-bridge",
        "related-top",
        "related-near",
        "related-bridge",
        "z-distractor",
        "opposite",
    )
    assert dataset.query_ids == ("q-main", "q-related")
    assert dataset.document_vectors.dtype == np.float32
    assert dataset.query_vectors.dtype == np.float32
    assert np.all(np.isfinite(dataset.document_vectors))
    assert np.all(np.isfinite(dataset.query_vectors))
    assert np.allclose(np.linalg.norm(dataset.document_vectors, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(dataset.query_vectors, axis=1), 1.0)
    assert dataset.relevant_by_query == {
        "q-main": frozenset({"main-near", "main-bridge"}),
        "q-related": frozenset({"related-near", "related-bridge"}),
    }


def test_synthetic_dataset_digest_is_seed_deterministic() -> None:
    first = build_synthetic_dataset()
    second = build_synthetic_dataset()

    np.testing.assert_array_equal(first.document_vectors, second.document_vectors)
    np.testing.assert_array_equal(first.query_vectors, second.query_vectors)
    assert first.digest() == second.digest()


def test_synthetic_dataset_digest_changes_with_seed() -> None:
    seed_seven = build_synthetic_dataset()
    seed_eight = build_synthetic_dataset(seed=8)

    assert not np.array_equal(seed_seven.document_vectors, seed_eight.document_vectors)
    assert not np.array_equal(seed_seven.query_vectors, seed_eight.query_vectors)
    assert seed_seven.digest() != seed_eight.digest()


def test_synthetic_dataset_seed_seven_matches_prescribed_jittered_bases() -> None:
    expected_documents, expected_queries = _expected_seed_seven_vectors()
    dataset = build_synthetic_dataset(seed=7)

    np.testing.assert_allclose(dataset.document_vectors, expected_documents, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dataset.query_vectors, expected_queries, rtol=0.0, atol=0.0)


def test_synthetic_dataset_does_not_advance_legacy_global_rng() -> None:
    original_state = np.random.get_state()
    try:
        np.random.seed(314159)
        expected = np.random.random(3)
        np.random.seed(314159)

        build_synthetic_dataset()

        np.testing.assert_array_equal(np.random.random(3), expected)
    finally:
        np.random.set_state(original_state)


from benchmarks.mcmp.fixtures import build_dataset, build_neutral_dataset


def _relevant_ranks(dataset, query_id: str) -> tuple[int, ...]:
    query_index = dataset.query_ids.index(query_id)
    similarities = dataset.document_vectors @ dataset.query_vectors[query_index]
    order = np.argsort(-similarities)
    relevant = dataset.relevant_by_query[query_id]
    return tuple(
        rank
        for rank, index in enumerate(order, start=1)
        if dataset.document_ids[index] in relevant
    )


def test_neutral_dataset_has_expected_shape_and_validates() -> None:
    dataset = build_neutral_dataset(7)

    assert dataset.dataset_id == "neutral-mcmp-v1"
    assert dataset.document_vectors.shape == (64, 16)
    assert dataset.query_vectors.shape == (2, 16)
    assert dataset.query_ids == ("q-main", "q-related")
    assert dataset.document_ids[0] == "doc-00"
    assert dataset.document_ids[-1] == "doc-63"
    assert dataset.document_vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(dataset.document_vectors, axis=1), 1.0, atol=1e-4)
    dataset.validate()


def test_neutral_relevance_is_drawn_from_the_similarity_top_16() -> None:
    dataset = build_neutral_dataset(7)

    for query_id in dataset.query_ids:
        ranks = _relevant_ranks(dataset, query_id)
        assert len(ranks) == 4
        assert max(ranks) <= 16


def test_neutral_relevant_ranks_vary_across_seeds() -> None:
    observed = {
        _relevant_ranks(build_neutral_dataset(seed), query_id)
        for seed in range(1, 13)
        for query_id in ("q-main", "q-related")
    }

    assert len(observed) > 1


def test_neutral_dataset_is_deterministic_per_seed() -> None:
    first = build_neutral_dataset(3)
    second = build_neutral_dataset(3)

    assert first.digest() == second.digest()
    assert first.relevant_by_query == second.relevant_by_query
    assert first.digest() != build_neutral_dataset(4).digest()


def test_registry_selects_the_requested_builder() -> None:
    assert build_dataset("neutral", 7).dataset_id == "neutral-mcmp-v1"
    assert build_dataset("legacy", 7).dataset_id == "synthetic-mcmp-v1"


from benchmarks.mcmp.fixtures import build_manifold_dataset


def test_manifold_dataset_has_expected_shape_and_validates() -> None:
    dataset = build_manifold_dataset(7)

    assert dataset.dataset_id == "manifold-mcmp-v1"
    assert dataset.document_vectors.shape == (64, 16)
    assert dataset.query_ids == ("q-main", "q-related")
    assert dataset.relevant_by_query["q-main"] == frozenset(
        {"main-chain-6", "main-chain-7", "main-chain-8"}
    )
    assert dataset.relevant_by_query["q-related"] == frozenset(
        {"related-chain-6", "related-chain-7", "related-chain-8"}
    )
    dataset.validate()


def test_manifold_chain_links_are_closer_to_each_other_than_the_far_end_is_to_the_query() -> None:
    dataset = build_manifold_dataset(7)
    index = {document_id: position for position, document_id in enumerate(dataset.document_ids)}
    vectors = dataset.document_vectors
    query = dataset.query_vectors[dataset.query_ids.index("q-main")]

    far_end_similarity = float(vectors[index["main-chain-8"]] @ query)
    for position in range(1, 8):
        link = float(
            vectors[index[f"main-chain-{position}"]]
            @ vectors[index[f"main-chain-{position + 1}"]]
        )
        assert link > far_end_similarity


def test_manifold_relevant_documents_rank_below_the_default_top_k() -> None:
    dataset = build_manifold_dataset(7)

    ranks = _relevant_ranks(dataset, "q-main")

    assert min(ranks) > 4


def test_manifold_dataset_is_deterministic_per_seed() -> None:
    assert build_manifold_dataset(3).digest() == build_manifold_dataset(3).digest()
    assert build_manifold_dataset(3).digest() != build_manifold_dataset(4).digest()
