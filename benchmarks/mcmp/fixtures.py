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


NEUTRAL_DOCUMENT_COUNT = 64
NEUTRAL_DIMENSIONS = 16
NEUTRAL_RELEVANT_PER_QUERY = 4
NEUTRAL_CANDIDATE_DEPTH = 16


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def build_neutral_dataset(seed: int = 7) -> BenchmarkDataset:
    """Build a control fixture whose relevance is not a fixed function of rank."""
    rng = np.random.default_rng(seed)
    documents = _unit_rows(
        rng.normal(size=(NEUTRAL_DOCUMENT_COUNT, NEUTRAL_DIMENSIONS))
    )
    queries = _unit_rows(rng.normal(size=(2, NEUTRAL_DIMENSIONS)))
    document_ids = tuple(f"doc-{index:02d}" for index in range(NEUTRAL_DOCUMENT_COUNT))
    query_ids = ("q-main", "q-related")

    relevant_by_query: dict[str, frozenset[str]] = {}
    for query_index, query_id in enumerate(query_ids):
        similarities = documents @ queries[query_index]
        candidates = np.argsort(-similarities)[:NEUTRAL_CANDIDATE_DEPTH]
        chosen = rng.choice(
            candidates, size=NEUTRAL_RELEVANT_PER_QUERY, replace=False
        )
        relevant_by_query[query_id] = frozenset(
            document_ids[int(index)] for index in chosen
        )

    dataset = BenchmarkDataset(
        dataset_id="neutral-mcmp-v1",
        seed=seed,
        document_ids=document_ids,
        document_vectors=documents,
        query_ids=query_ids,
        query_vectors=queries,
        relevant_by_query=relevant_by_query,
    )
    dataset.validate()
    return dataset


MANIFOLD_CHAIN_LENGTH = 8
MANIFOLD_TOTAL_ANGLE = 1.37
MANIFOLD_RELEVANT_TAIL = 3
MANIFOLD_DOCUMENT_COUNT = 64
MANIFOLD_DISTRACTOR_COUNT = 48
MANIFOLD_DISTRACTOR_COSINE_RANGE = (0.55, 0.75)


def _orthonormal_basis(rng: np.random.Generator, dimensions: int) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))
    return basis.T


def _chain(query: np.ndarray, direction: np.ndarray, length: int, total_angle: float) -> np.ndarray:
    angles = np.linspace(total_angle / length, total_angle, length)
    return np.stack(
        [np.cos(angle) * query + np.sin(angle) * direction for angle in angles]
    )


def build_manifold_dataset(
    seed: int = 7, document_count: int = MANIFOLD_DOCUMENT_COUNT
) -> BenchmarkDataset:
    """Build a fixture whose relevant documents are reachable only along a chain.

    ``document_count`` scales the distractor field only. The two chains keep
    their length and their relevant tail, so a larger corpus is a harder
    haystack rather than a different needle -- which is what lets a corpus-size
    sweep vary one thing at a time.
    """
    chain_documents = 2 * MANIFOLD_CHAIN_LENGTH
    distractor_count = document_count - chain_documents
    if distractor_count < 1:
        raise ValueError(
            f"document_count must leave room for the two chains: "
            f"{document_count} given, more than {chain_documents} required"
        )
    rng = np.random.default_rng(seed)
    dimensions = NEUTRAL_DIMENSIONS
    basis = _orthonormal_basis(rng, dimensions)
    queries = basis[:2]
    query_ids = ("q-main", "q-related")

    document_ids: list[str] = []
    rows: list[np.ndarray] = []
    relevant_by_query: dict[str, frozenset[str]] = {}

    for query_index, (query_id, prefix) in enumerate(
        zip(query_ids, ("main", "related"), strict=True)
    ):
        weights = rng.normal(size=dimensions - 2)
        direction = weights @ basis[2:]
        direction = direction / np.linalg.norm(direction)
        chain = _chain(
            queries[query_index], direction, MANIFOLD_CHAIN_LENGTH, MANIFOLD_TOTAL_ANGLE
        )
        ids = [f"{prefix}-chain-{position}" for position in range(1, MANIFOLD_CHAIN_LENGTH + 1)]
        document_ids.extend(ids)
        rows.extend(chain)
        relevant_by_query[query_id] = frozenset(ids[-MANIFOLD_RELEVANT_TAIL:])

    low, high = MANIFOLD_DISTRACTOR_COSINE_RANGE
    for position in range(distractor_count):
        anchor = queries[position % 2]
        cosine = float(rng.uniform(low, high))
        weights = rng.normal(size=dimensions - 2)
        offset = weights @ basis[2:]
        offset = offset / np.linalg.norm(offset)
        document_ids.append(f"distractor-{position:04d}")
        rows.append(cosine * anchor + np.sqrt(1.0 - cosine**2) * offset)

    documents = _unit_rows(np.stack(rows))
    dataset = BenchmarkDataset(
        dataset_id="manifold-mcmp-v1",
        seed=seed,
        document_ids=tuple(document_ids),
        document_vectors=documents,
        query_ids=query_ids,
        query_vectors=_unit_rows(queries),
        relevant_by_query=relevant_by_query,
    )
    dataset.validate()
    return dataset


FIXTURES = {
    "legacy": build_synthetic_dataset,
    "neutral": build_neutral_dataset,
    "manifold": build_manifold_dataset,
}


def build_dataset(
    fixture: str, seed: int, document_count: int | None = None
) -> BenchmarkDataset:
    """Build a benchmark dataset by registry key, failing closed on unknown keys."""
    if fixture not in FIXTURES:
        raise ValueError(
            f"unknown fixture {fixture!r}; valid keys are {sorted(FIXTURES)}"
        )
    if document_count is None:
        return FIXTURES[fixture](seed)
    if fixture != "manifold":
        raise ValueError(
            f"document_count is only supported by the manifold fixture, not {fixture!r}"
        )
    return FIXTURES[fixture](seed, document_count=document_count)
