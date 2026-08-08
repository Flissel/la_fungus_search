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
