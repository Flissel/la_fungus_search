"""Stage 1: characterise real embedding geometry without any retrieval method."""

from __future__ import annotations

from collections import deque

import numpy as np

from benchmarks.mcmp.contracts import BenchmarkDataset

MANIFOLD_SIGNATURE_GATE = 0.10


def _similarities(dataset: BenchmarkDataset, query_id: str) -> np.ndarray:
    query_index = list(dataset.query_ids).index(query_id)
    return dataset.document_vectors @ dataset.query_vectors[query_index]


def relevant_ranks(dataset: BenchmarkDataset, query_id: str) -> tuple[int, ...]:
    """One-based similarity ranks of the relevant documents for a query."""
    order = np.argsort(-_similarities(dataset, query_id))
    relevant = dataset.relevant_by_query[query_id]
    return tuple(
        rank
        for rank, index in enumerate(order, start=1)
        if dataset.document_ids[index] in relevant
    )


def knn_graph(vectors: np.ndarray, knn_k: int) -> dict[int, frozenset[int]]:
    """Mutual k-NN adjacency: an edge needs each endpoint in the other's top-k."""
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    neighbours = {
        index: set(np.argsort(-similarities[index])[:knn_k].tolist())
        for index in range(vectors.shape[0])
    }
    return {
        index: frozenset(other for other in targets if index in neighbours[other])
        for index, targets in neighbours.items()
    }


def chain_reachable(
    dataset: BenchmarkDataset,
    query_id: str,
    document_id: str,
    knn_k: int,
    max_hops: int,
    hop_threshold: float,
) -> bool:
    """Is the target reachable from the query's nearest document within the budget?"""
    similarities = _similarities(dataset, query_id)
    start = int(np.argmax(similarities))
    target = list(dataset.document_ids).index(document_id)
    if start == target:
        return True
    graph = knn_graph(dataset.document_vectors, knn_k)
    pairwise = dataset.document_vectors @ dataset.document_vectors.T
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbour in sorted(graph[node]):
            if neighbour in seen or float(pairwise[node, neighbour]) < hop_threshold:
                continue
            if neighbour == target:
                return True
            seen.add(neighbour)
            queue.append((neighbour, depth + 1))
    return False


def characterise(
    dataset: BenchmarkDataset,
    top_k: int = 8,
    knn_k: int = 8,
    max_hops: int = 6,
    hop_threshold: float = 0.0,
) -> dict[str, object]:
    """Measure relevant-rank distribution and chain reachability for every pair."""
    pairs: list[dict[str, object]] = []
    for query_id in dataset.query_ids:
        similarities = _similarities(dataset, query_id)
        order = np.argsort(-similarities)
        rank_by_id = {
            dataset.document_ids[index]: rank for rank, index in enumerate(order, start=1)
        }
        for document_id in sorted(dataset.relevant_by_query[query_id]):
            rank = rank_by_id[document_id]
            far = rank > top_k
            reachable = (
                chain_reachable(dataset, query_id, document_id, knn_k, max_hops, hop_threshold)
                if far
                else False
            )
            pairs.append(
                {
                    "query_id": query_id,
                    "document_id": document_id,
                    "rank": rank,
                    "far": far,
                    "chain_reachable": reachable,
                }
            )
    far_count = sum(1 for pair in pairs if pair["far"])
    far_and_reachable = sum(1 for pair in pairs if pair["far"] and pair["chain_reachable"])
    signature = far_and_reachable / len(pairs) if pairs else 0.0
    return {
        "config": {
            "top_k": top_k,
            "knn_k": knn_k,
            "max_hops": max_hops,
            "hop_threshold": hop_threshold,
        },
        "dataset_id": dataset.dataset_id,
        "seed": dataset.seed,
        "pair_count": len(pairs),
        "far_count": far_count,
        "far_and_reachable_count": far_and_reachable,
        "manifold_signature": signature,
        "pairs": pairs,
    }


def stage_two_is_justified(signature: float) -> bool:
    """Pre-registered gate: stage 2 runs at or above a 10% manifold signature."""
    return signature >= MANIFOLD_SIGNATURE_GATE
