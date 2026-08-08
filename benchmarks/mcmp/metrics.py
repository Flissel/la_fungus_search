"""Pure, deterministic metrics for offline MCMP benchmark runs."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
import math

import numpy as np

from benchmarks.mcmp.contracts import BenchmarkDataset, SearchRun


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    k = _validate_k(k)
    for rank, document_id in enumerate(ranked[:k], start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    k = _validate_k(k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, document_id in enumerate(ranked[:k], start=1)
        if document_id in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_run(dataset: BenchmarkDataset, run: SearchRun, k: int) -> dict[str, object]:
    """Evaluate one fused ranking and its per-query ranking outcomes."""
    k = _validate_k(k)
    dataset.validate()
    run.validate(dataset)
    relevant = frozenset().union(
        *(dataset.relevant_by_query.get(query_id, frozenset()) for query_id in run.query_ids)
    )
    ranked = run.ranked_document_ids[:k]
    relevant_hits = sum(document_id in relevant for document_id in ranked)
    per_query_reciprocal_ranks = [
        reciprocal_rank(
            run.per_query_ranked_document_ids[query_id],
            dataset.relevant_by_query.get(query_id, frozenset()),
            k,
        )
        for query_id in run.query_ids
    ]
    novel_candidates = sorted(run.discovered_candidate_ids - run.initial_candidate_ids)
    novel_relevant_candidates = sorted(set(novel_candidates) & relevant)

    return {
        "recall_at_k": relevant_hits / len(relevant) if relevant else 0.0,
        "reciprocal_rank": reciprocal_rank(run.ranked_document_ids, relevant, k),
        "mrr": (
            sum(per_query_reciprocal_ranks) / len(per_query_reciprocal_ranks)
            if per_query_reciprocal_ranks
            else 0.0
        ),
        "ndcg_at_k": ndcg_at_k(run.ranked_document_ids, relevant, k),
        "unique_relevant_documents": len(relevant),
        "candidate_count": len(run.discovered_candidate_ids),
        "novel_candidates": novel_candidates,
        "novel_relevant_candidates": novel_relevant_candidates,
    }


def query_geometry(dataset: BenchmarkDataset) -> dict[str, float]:
    """Summarize pairwise cosine distance between benchmark query vectors."""
    dataset.validate()
    if np.any(np.linalg.norm(dataset.query_vectors, axis=1) == 0.0):
        raise ValueError("zero-norm query vector")
    distances = [
        _cosine_distance(left, right)
        for left, right in combinations(dataset.query_vectors, 2)
    ]
    if not distances:
        return {"mean_cosine_distance": 0.0, "max_cosine_distance": 0.0}
    return {
        "mean_cosine_distance": float(np.mean(distances)),
        "max_cosine_distance": float(max(distances)),
    }


def candidate_overlap(run: SearchRun) -> dict[str, float]:
    """Return deterministic pairwise Jaccard candidate overlap by query pair."""
    overlaps: dict[str, float] = {}
    for left_query_id, right_query_id in combinations(sorted(run.query_ids), 2):
        left = run.per_query_candidate_ids[left_query_id]
        right = run.per_query_candidate_ids[right_query_id]
        union = left | right
        overlaps[f"{left_query_id}|{right_query_id}"] = len(left & right) / len(union) if union else 0.0
    return overlaps


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("zero-norm query vector")
    return float(1.0 - np.dot(left, right) / denominator)


def _validate_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")
    return int(k)
