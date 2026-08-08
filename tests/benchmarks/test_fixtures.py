from __future__ import annotations

import numpy as np

from benchmarks.mcmp.fixtures import build_synthetic_dataset


def test_synthetic_dataset_has_expected_shape_normalized_vectors_and_labels() -> None:
    dataset = build_synthetic_dataset()

    assert dataset.document_vectors.shape == (8, 3)
    assert dataset.query_vectors.shape == (2, 3)
    assert np.allclose(np.linalg.norm(dataset.document_vectors, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(dataset.query_vectors, axis=1), 1.0)
    assert dataset.relevant_by_query == {
        "q-main": frozenset({"main-near", "main-bridge"}),
        "q-related": frozenset({"related-near", "related-bridge"}),
    }


def test_synthetic_dataset_digest_is_seed_deterministic() -> None:
    assert build_synthetic_dataset().digest() == build_synthetic_dataset().digest()


def test_synthetic_dataset_digest_changes_with_seed() -> None:
    assert build_synthetic_dataset().digest() != build_synthetic_dataset(seed=8).digest()
