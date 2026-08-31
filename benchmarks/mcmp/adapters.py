"""Deterministic local adapters for the offline MCMP ablation benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from time import perf_counter
from collections.abc import Sequence

import numpy as np

from benchmarks.mcmp.contracts import BenchmarkDataset, SearchRun
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from embeddinggemma.mcmp_rag import MCPMRetriever


PHEROMONE_DECAY = 0.95
EXPLORATION_BONUS = 0.1
DETERMINISTIC_CLOCK_MODE = "fixed"
DETERMINISTIC_CLOCK_VALUE = 2.0


@dataclass(frozen=True)
class FixedClock:
    value: float

    def __call__(self) -> float:
        return self.value


@dataclass(frozen=True)
class AdapterEvidence:
    """Adapter-level facts which are not part of a retrieval result."""

    independent_run_count: int
    nearest_search_calls: int
    execution_backend: str
    per_query_initial_candidate_ids: dict[str, tuple[str, ...]]
    per_query_random_seeds: dict[str, dict[str, int]] | None
    clock_mode: str
    clock_value: float


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


class PheromoneFreeRetriever(CountingRetriever):
    """MCMP with pheromone deposition disabled, for the equal-budget control.

    The colony is the distinguishing claim of this design: agents deposit trails
    and later agents follow them. Overriding the deposit leaves `pheromone_trails`
    permanently empty, so `calculate_pheromone_force` short-circuits and the
    0.15 pheromone term of the movement force contributes nothing, while the
    attraction and exploration terms, the agent count, the step count and every
    seed stay exactly as method C has them. The same nearest-neighbour calls are
    made, so the comparison budget is equal by construction rather than by
    assertion.

    Known confound, recorded rather than hidden: the force weights
    (0.8 attraction, 0.15 pheromone, 0.05 exploration) are fixed, so removing the
    pheromone also removes its contribution to the total force *magnitude*. This
    control therefore differs from C in two ways -- no trail guidance, and a
    slightly slower walk. Separating those requires renormalising the remaining
    weights, which is a further experiment and only worth running if this one
    shows a difference.
    """

    def deposit_pheromones(self, agent: object) -> None:  # type: ignore[override]
        # Deposit normally, then wipe the trail memory. Suppressing the call
        # outright would also suppress what it does besides depositing -- the
        # nearest-document lookup and the visit counter -- which measured as
        # 328 comparisons against C's 896 and zero recorded visits. That is a
        # different algorithm on a smaller budget, not a control. Running the
        # deposit and clearing after it removes the pheromone memory and nothing
        # else: identical calls, identical visits, empty trails.
        super().deposit_pheromones(agent)
        self.pheromone_trails.clear()


def run_faiss(
    dataset: BenchmarkDataset,
    method: str,
    query_ids: Sequence[str],
    top_k: int,
    initial_k: int,
) -> tuple[SearchRun, AdapterEvidence]:
    """Run normalized inner-product retrieval independently for each query."""
    _validate_run_inputs(dataset, method, query_ids, {"A", "B"}, top_k, initial_k)
    started = perf_counter()
    vectors = _vector_mapping(dataset)
    candidates: dict[str, frozenset[str]] = {}
    initial_rankings: dict[str, tuple[str, ...]] = {}
    rankings: dict[str, Sequence[str]] = {}
    score_maps: list[dict[str, float]] = []
    nearest_search_calls = 0
    execution_backends: set[str] = set()

    for query_id in query_ids:
        backend = MappingEmbeddingBackend(vectors)
        retriever = CountingRetriever(
            build_faiss_after_add=True,
            force_cpu=True,
            embedding_backend=(backend, dataset.document_vectors.shape[1]),
            time_source=FixedClock(DETERMINISTIC_CLOCK_VALUE),
        )
        retriever.add_documents(list(dataset.document_ids), cache=False)
        execution_backends.add(_execution_backend(retriever))
        neighbours = retriever.find_nearest_documents(vectors[query_id], k=len(dataset.document_ids))
        scores = {document.content: float(score) for document, score in neighbours}
        ordered = _rank(scores, top_k)
        rankings[query_id] = ordered
        initial_rankings[query_id] = _rank(scores, initial_k)
        candidates[query_id] = frozenset(initial_rankings[query_id])
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
        per_query_initial_candidate_ids=candidates,
    )
    run.validate(dataset)
    return run, AdapterEvidence(
        len(query_ids),
        nearest_search_calls,
        _single_execution_backend(execution_backends),
        initial_rankings,
        None,
        DETERMINISTIC_CLOCK_MODE,
        DETERMINISTIC_CLOCK_VALUE,
    )


def run_mcmp(
    dataset: BenchmarkDataset,
    method: str,
    query_ids: Sequence[str],
    top_k: int,
    initial_k: int,
    seed: int,
    num_agents: int,
    steps: int,
    pool_only: bool = False,
    pheromone_free: bool = False,
) -> tuple[SearchRun, AdapterEvidence]:
    """Run a fresh, seeded MCMP simulation for every benchmark query."""
    _validate_run_inputs(dataset, method, query_ids, {"C", "D", "E", "F"}, top_k, initial_k)
    # Method F is C with the colony switched off; every other knob is identical.
    retriever_class = PheromoneFreeRetriever if pheromone_free else CountingRetriever
    _validate_positive_integer(num_agents, name="num_agents")
    _validate_positive_integer(steps, name="steps")
    seed = _validate_seed(seed, query_count=len(query_ids))
    started = perf_counter()
    vectors = _vector_mapping(dataset)
    initial_candidates: dict[str, frozenset[str]] = {}
    initial_rankings: dict[str, tuple[str, ...]] = {}
    discovered_candidates: dict[str, frozenset[str]] = {}
    rankings: dict[str, Sequence[str]] = {}
    score_maps: list[dict[str, float]] = []
    visits: dict[str, int] = {document_id: 0 for document_id in dataset.document_ids}
    trails = 0
    nearest_search_calls = 0
    execution_backends: set[str] = set()
    per_query_random_seeds: dict[str, dict[str, int]] = {}
    original_python_rng_state = random.getstate()
    original_rng_state = np.random.get_state()
    try:
        for query_index, query_id in enumerate(query_ids):
            query_seed = seed + query_index
            random.seed(query_seed)
            np.random.seed(query_seed)
            per_query_random_seeds[query_id] = {
                "python_random_seed": query_seed,
                "numpy_random_seed": query_seed,
            }
            backend = MappingEmbeddingBackend(vectors)
            retriever = retriever_class(
                num_agents=num_agents,
                max_iterations=steps,
                pheromone_decay=PHEROMONE_DECAY,
                exploration_bonus=EXPLORATION_BONUS,
                build_faiss_after_add=True,
                force_cpu=True,
                embedding_backend=(backend, dataset.document_vectors.shape[1]),
                time_source=FixedClock(DETERMINISTIC_CLOCK_VALUE),
            )
            retriever.add_documents(list(dataset.document_ids), cache=False)
            execution_backends.add(_execution_backend(retriever))
            initial = retriever.find_nearest_documents(vectors[query_id], k=len(dataset.document_ids))
            initial_scores = {document.content: float(score) for document, score in initial}
            initial_rankings[query_id] = _rank(initial_scores, initial_k)
            initial_candidates[query_id] = frozenset(initial_rankings[query_id])
            if pool_only:
                # Method E reranks only the FAISS pool. The pool is formed on a
                # full-corpus retriever, whose search calls are banked here before
                # a second retriever is built over the pool alone.
                nearest_search_calls += retriever.nearest_search_calls
                backend = MappingEmbeddingBackend(vectors)
                retriever = retriever_class(
                    num_agents=num_agents,
                    max_iterations=steps,
                    pheromone_decay=PHEROMONE_DECAY,
                    exploration_bonus=EXPLORATION_BONUS,
                    build_faiss_after_add=True,
                    force_cpu=True,
                    embedding_backend=(backend, dataset.document_vectors.shape[1]),
                    time_source=FixedClock(DETERMINISTIC_CLOCK_VALUE),
                )
                retriever.add_documents(list(initial_rankings[query_id]), cache=False)
                execution_backends.add(_execution_backend(retriever))
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
        random.setstate(original_python_rng_state)
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
        per_query_initial_candidate_ids=initial_candidates,
    )
    run.validate(dataset)
    return run, AdapterEvidence(
        len(query_ids),
        nearest_search_calls,
        _single_execution_backend(execution_backends),
        initial_rankings,
        per_query_random_seeds,
        DETERMINISTIC_CLOCK_MODE,
        DETERMINISTIC_CLOCK_VALUE,
    )


def _execution_backend(retriever: CountingRetriever) -> str:
    return "faiss-cpu" if retriever._faiss_index is not None else "numpy-cpu-fallback"


def _single_execution_backend(backends: set[str]) -> str:
    if len(backends) != 1:
        raise RuntimeError("benchmark execution backend changed within a run")
    return next(iter(backends))


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


def _validate_run_inputs(
    dataset: BenchmarkDataset,
    method: str,
    query_ids: Sequence[str],
    allowed_methods: set[str],
    top_k: int,
    initial_k: int,
) -> None:
    dataset.validate()
    if method not in allowed_methods:
        raise ValueError(f"method must be one of {sorted(allowed_methods)}")
    expected_query_count = 1 if method in {"A", "C", "E", "F"} else 2
    if len(query_ids) != expected_query_count:
        raise ValueError(f"method {method} requires exactly {expected_query_count} query ids")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query ids must be unique")
    if any(query_id not in dataset.query_ids for query_id in query_ids):
        raise ValueError("unknown query id")
    if set(dataset.document_ids) & set(dataset.query_ids):
        raise ValueError("document and query ids must be disjoint")
    _validate_positive_integer(top_k, name="top_k")
    _validate_positive_integer(initial_k, name="initial_k")
    document_count = len(dataset.document_ids)
    if top_k > document_count:
        raise ValueError("top_k must not exceed document count")
    if initial_k > document_count:
        raise ValueError("initial_k must not exceed document count")


def _validate_positive_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_seed(seed: object, *, query_count: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    normalized_seed = int(seed)
    if normalized_seed < 0 or normalized_seed > (2**32 - query_count):
        raise ValueError("seed is outside the supported NumPy range")
    return normalized_seed
