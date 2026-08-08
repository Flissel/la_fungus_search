"""Deterministic local adapters for the offline MCMP ablation benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from collections.abc import Sequence

import numpy as np

from benchmarks.mcmp.contracts import BenchmarkDataset, SearchRun
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from embeddinggemma.mcmp_rag import MCPMRetriever


@dataclass(frozen=True)
class AdapterEvidence:
    """Adapter-level facts which are not part of a retrieval result."""

    independent_run_count: int
    nearest_search_calls: int


class MappingEmbeddingBackend:
    """In-memory embedding backend keyed by benchmark document and query ids."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = {identifier: np.asarray(vector, dtype=np.float32).copy() for identifier, vector in vectors.items()}

    def encode(self, identifiers: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [self._vectors[identifier].copy() for identifier in identifiers], dtype=np.float32
        )


class CountingRetriever(MCPMRetriever):
    """MCMP retriever which records Flat-index nearest-neighbour operations."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.nearest_search_calls = 0

    def find_nearest_documents(self, position: np.ndarray, k: int = 3):
        self.nearest_search_calls += 1
        return super().find_nearest_documents(position, k)


def run_faiss(
    dataset: BenchmarkDataset,
    method: str,
    query_ids: Sequence[str],
    top_k: int,
    initial_k: int,
) -> tuple[SearchRun, AdapterEvidence]:
    """Run normalized inner-product retrieval independently for each query."""
    dataset.validate()
    _validate_query_ids(dataset, query_ids)
    started = perf_counter()
    vectors = _vector_mapping(dataset)
    candidates: dict[str, frozenset[str]] = {}
    rankings: dict[str, Sequence[str]] = {}
    score_maps: list[dict[str, float]] = []
    nearest_search_calls = 0

    for query_id in query_ids:
        backend = MappingEmbeddingBackend(vectors)
        retriever = CountingRetriever(
            build_faiss_after_add=True,
            embedding_backend=(backend, dataset.document_vectors.shape[1]),
        )
        retriever.add_documents(list(dataset.document_ids), cache=False)
        neighbours = retriever.find_nearest_documents(vectors[query_id], k=len(dataset.document_ids))
        scores = {document.content: float(score) for document, score in neighbours}
        ordered = _rank(scores, top_k)
        rankings[query_id] = ordered
        candidates[query_id] = frozenset(_rank(scores, initial_k))
        score_maps.append(scores)
        nearest_search_calls += retriever.nearest_search_calls

    run = SearchRun(
        method=method,
        query_ids=tuple(query_ids),
        ranked_document_ids=_fuse(score_maps, top_k),
        initial_candidate_ids=frozenset().union(*candidates.values()),
        discovered_candidate_ids=frozenset().union(*candidates.values()),
        per_query_candidate_ids=candidates,
        per_query_ranked_document_ids=rankings,
        elapsed_ms=(perf_counter() - started) * 1000.0,
        candidate_comparisons=nearest_search_calls * len(dataset.document_ids),
        mcmp_steps=0,
        document_visits={},
        pheromone_trails=0,
    )
    run.validate(dataset)
    return run, AdapterEvidence(1, nearest_search_calls)


def run_mcmp(
    dataset: BenchmarkDataset,
    method: str,
    query_ids: Sequence[str],
    top_k: int,
    initial_k: int,
    seed: int,
    num_agents: int,
    steps: int,
) -> tuple[SearchRun, AdapterEvidence]:
    """Run a fresh, seeded MCMP simulation for every benchmark query."""
    dataset.validate()
    _validate_query_ids(dataset, query_ids)
    started = perf_counter()
    vectors = _vector_mapping(dataset)
    initial_candidates: dict[str, frozenset[str]] = {}
    discovered_candidates: dict[str, frozenset[str]] = {}
    rankings: dict[str, Sequence[str]] = {}
    score_maps: list[dict[str, float]] = []
    visits: dict[str, int] = {document_id: 0 for document_id in dataset.document_ids}
    trails = 0
    nearest_search_calls = 0
    original_rng_state = np.random.get_state()
    try:
        for query_index, query_id in enumerate(query_ids):
            np.random.seed(seed + query_index)
            backend = MappingEmbeddingBackend(vectors)
            retriever = CountingRetriever(
                num_agents=num_agents,
                max_iterations=steps,
                build_faiss_after_add=True,
                embedding_backend=(backend, dataset.document_vectors.shape[1]),
            )
            retriever.add_documents(list(dataset.document_ids), cache=False)
            initial = retriever.find_nearest_documents(vectors[query_id], k=len(dataset.document_ids))
            initial_scores = {document.content: float(score) for document, score in initial}
            initial_candidates[query_id] = frozenset(_rank(initial_scores, initial_k))
            retriever.initialize_simulation(query_id)
            retriever.step(steps)

            scores = {document.content: float(document.relevance_score) for document in retriever.documents}
            rankings[query_id] = _rank(scores, top_k)
            score_maps.append(scores)
            discovered_candidates[query_id] = frozenset(
                document.content for document in retriever.documents if document.visit_count > 0
            )
            for document in retriever.documents:
                visits[document.content] += int(document.visit_count)
            trails += len(retriever.pheromone_trails)
            nearest_search_calls += retriever.nearest_search_calls
    finally:
        np.random.set_state(original_rng_state)

    run = SearchRun(
        method=method,
        query_ids=tuple(query_ids),
        ranked_document_ids=_fuse(score_maps, top_k),
        initial_candidate_ids=frozenset().union(*initial_candidates.values()),
        discovered_candidate_ids=frozenset().union(*discovered_candidates.values()),
        per_query_candidate_ids=discovered_candidates,
        per_query_ranked_document_ids=rankings,
        elapsed_ms=(perf_counter() - started) * 1000.0,
        candidate_comparisons=nearest_search_calls * len(dataset.document_ids),
        mcmp_steps=steps,
        document_visits=visits,
        pheromone_trails=trails,
    )
    run.validate(dataset)
    return run, AdapterEvidence(len(query_ids), nearest_search_calls)


def _vector_mapping(dataset: BenchmarkDataset) -> dict[str, np.ndarray]:
    return {
        **dict(zip(dataset.document_ids, dataset.document_vectors, strict=True)),
        **dict(zip(dataset.query_ids, dataset.query_vectors, strict=True)),
    }


def _rank(scores: dict[str, float], top_k: int) -> tuple[str, ...]:
    return tuple(
        document_id
        for document_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    )


def _fuse(score_maps: Sequence[dict[str, float]], top_k: int) -> tuple[str, ...]:
    scores: dict[str, float] = {}
    for score_map in score_maps:
        for document_id, score in score_map.items():
            scores[document_id] = max(scores.get(document_id, float("-inf")), score)
    return _rank(scores, top_k)


def _validate_query_ids(dataset: BenchmarkDataset, query_ids: Sequence[str]) -> None:
    if not query_ids:
        raise ValueError("query ids must not be empty")
    if any(query_id not in dataset.query_ids for query_id in query_ids):
        raise ValueError("unknown query id")
