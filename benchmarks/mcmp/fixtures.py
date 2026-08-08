"""Deterministic synthetic inputs for the offline MCMP benchmark."""

from __future__ import annotations

import numpy as np

from benchmarks.mcmp.contracts import BenchmarkDataset


_DOCUMENTS = {
    "main-top": [1.0, 0.0, 0.0],
    "main-near": [0.98, 0.18, 0.0],
    "main-bridge": [0.78, 0.58, 0.0],
    "related-top": [0.0, 1.0, 0.0],
    "related-near": [0.18, 0.98, 0.0],
    "related-bridge": [0.58, 0.78, 0.0],
    "z-distractor": [0.0, 0.0, 1.0],
    "opposite": [-1.0, 0.0, 0.0],
}

_QUERIES = {
    "q-main": [1.0, 0.0, 0.0],
    "q-related": [0.0, 1.0, 0.0],
}


def _jitter_and_normalize(vectors: dict[str, list[float]], rng: np.random.Generator) -> np.ndarray:
    matrix = np.asarray(tuple(vectors.values()), dtype=np.float32)
    matrix += rng.normal(0.0, 0.002, size=matrix.shape).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def build_synthetic_dataset(seed: int = 7) -> BenchmarkDataset:
    """Build the fixed-shape synthetic dataset with reproducible Gaussian jitter."""
    rng = np.random.default_rng(seed)
    dataset = BenchmarkDataset(
        dataset_id="synthetic-mcmp-v1",
        seed=seed,
        document_ids=tuple(_DOCUMENTS),
        document_vectors=_jitter_and_normalize(_DOCUMENTS, rng),
        query_ids=tuple(_QUERIES),
        query_vectors=_jitter_and_normalize(_QUERIES, rng),
        relevant_by_query={
            "q-main": frozenset({"main-near", "main-bridge"}),
            "q-related": frozenset({"related-near", "related-bridge"}),
        },
    )
    dataset.validate()
    return dataset
