"""Stage 1: characterise real embedding geometry without any retrieval method."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import NamedTuple

import numpy as np

from benchmarks.mcmp.contracts import BenchmarkDataset

# The gate is pre-registered against a permutation null, not against a bare
# threshold. A bare threshold has no reference point: the manifold signature
# measured 0.36-0.44 on a corpus of pure text hashes with no structure at all,
# three to four times the 0.10 floor that used to stand here. A number that
# noise clears is not a gate.
NULL_PERMUTATIONS = 100
NULL_ALPHA = 0.05
# Absolute headroom. This is the condition the ~14 300x cost argument from
# Gate 1 actually speaks to: a walk can only recover pairs that are far in the
# first place.
MIN_ABSOLUTE_SIGNATURE = 0.10
# Effect size, expressed against the achievable ceiling rather than as a flat
# margin. A flat 10 points is unsatisfiable a priori wherever the null median
# exceeds 0.90 -- measured at 0.96 on a synthetic corpus, where the largest
# possible excess is 0.043. Replacing a threshold that could not fail with one
# that cannot pass is not a repair.
MIN_RELATIVE_EXCESS = 0.10


class GeometryCache(NamedTuple):
    """Precomputed geometry, reusable across label permutations.

    The mutual k-NN graph and the pairwise similarity matrix depend only on the
    document vectors, which a label permutation leaves untouched. Computing them
    once per dataset and reusing them is what makes a 100-permutation null
    affordable. ``knn_k`` travels with them so a cache can never be honoured for
    a different neighbourhood size than it was built for -- that would answer
    with the wrong graph and say nothing about it.
    """

    graph: dict[int, frozenset[int]]
    pairwise: np.ndarray
    knn_k: int
    vectors: np.ndarray


def _resolve_cache(
    dataset: BenchmarkDataset, knn_k: int, cache: "GeometryCache | None"
) -> "GeometryCache":
    if cache is None:
        return geometry_cache(dataset, knn_k)
    if cache.knn_k != knn_k:
        raise ValueError(
            f"cache was built for knn_k={cache.knn_k}, but knn_k={knn_k} was requested"
        )
    # Identity, not equality: a label permutation shares the vector buffer by
    # construction, so this passes for the null path and fails for a cache built
    # from any other dataset. Every stage-1 seed's corpus has the same shape, so
    # a mis-zipped cache would otherwise answer with the wrong graph in silence.
    if cache.vectors is not dataset.document_vectors:
        raise ValueError("cache was built for a different dataset's vectors")
    return cache


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


def geometry_cache(dataset: BenchmarkDataset, knn_k: int) -> GeometryCache:
    """Precompute the parts of the measurement that depend only on the vectors."""
    return GeometryCache(
        graph=knn_graph(dataset.document_vectors, knn_k),
        pairwise=dataset.document_vectors @ dataset.document_vectors.T,
        knn_k=knn_k,
        vectors=dataset.document_vectors,
    )


def permuted_labels(
    dataset: BenchmarkDataset, rng: np.random.Generator
) -> BenchmarkDataset:
    """Redraw relevance uniformly at the same per-query sizes, geometry untouched.

    This is the null the gate is scored against. Vectors, query vectors and
    therefore the k-NN graph are identical to the real dataset; only *which*
    documents count as relevant changes, and each query keeps its own set size
    so the pair count is unchanged. What survives the comparison is the
    call-graph relation, isolated from the graph density that was previously
    inflating the signature.
    """
    document_ids = list(dataset.document_ids)
    relevant_by_query = {
        query_id: frozenset(
            document_ids[int(index)]
            for index in rng.choice(
                len(document_ids),
                size=len(dataset.relevant_by_query[query_id]),
                replace=False,
            )
        )
        for query_id in dataset.query_ids
    }
    return replace(
        dataset,
        dataset_id=f"{dataset.dataset_id}-null",
        relevant_by_query=relevant_by_query,
    )


def chain_reachable(
    dataset: BenchmarkDataset,
    query_id: str,
    document_id: str,
    knn_k: int,
    max_hops: int,
    hop_threshold: float,
    cache: GeometryCache | None = None,
) -> bool:
    """Is the target reachable from the query's nearest document within the budget?"""
    similarities = _similarities(dataset, query_id)
    start = int(np.argmax(similarities))
    target = list(dataset.document_ids).index(document_id)
    if start == target:
        return True
    resolved = _resolve_cache(dataset, knn_k, cache)
    graph, pairwise = resolved.graph, resolved.pairwise
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
    cache: GeometryCache | None = None,
) -> dict[str, object]:
    """Measure relevant-rank distribution and chain reachability for every pair."""
    cache = _resolve_cache(dataset, knn_k, cache)
    pairs: list[dict[str, object]] = []
    for query_id in dataset.query_ids:
        # relevant_ranks is the tested rank function, so it is the one that runs
        # here. Its ranks are positions in this same descending-similarity
        # ordering, so indexing back through `order` recovers the document each
        # rank belongs to exactly, without a parallel rank computation.
        order = np.argsort(-_similarities(dataset, query_id))
        rank_by_id = {
            dataset.document_ids[order[rank - 1]]: rank
            for rank in relevant_ranks(dataset, query_id)
        }
        for document_id in sorted(dataset.relevant_by_query[query_id]):
            rank = rank_by_id[document_id]
            far = rank > top_k
            # Reachability is measured only for far pairs. Near pairs record
            # None -- "not measured" -- rather than False, so a reader tallying
            # chain_reachable across the raw records cannot undercount it.
            reachable = (
                chain_reachable(
                    dataset,
                    query_id,
                    document_id,
                    knn_k,
                    max_hops,
                    hop_threshold,
                    cache=cache,
                )
                if far
                else None
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
    far_and_reachable = sum(
        1 for pair in pairs if pair["far"] and pair["chain_reachable"] is True
    )
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


def stage_two_is_justified(signature: float, null_signatures: list[float]) -> bool:
    """Pre-registered gate: three conditions, all fixed in advance.

    1. The signature exceeds the 95th percentile of the permutation null
       (one-sided, ``NULL_ALPHA``). The result is not noise.
    2. It is at least ``MIN_ABSOLUTE_SIGNATURE``. This is the condition the
       Gate 1 cost measurement speaks to -- there must be enough far pairs for a
       walk to recover for the compute to be worth spending.
    3. It exceeds the null's median by at least ``MIN_RELATIVE_EXCESS`` of the
       *achievable* ceiling, ``1 - median``, rather than by a flat margin. A flat
       margin is unsatisfiable wherever the null median is high, which converts
       the gate from "cannot fail" into "cannot pass".

    Conditions 1 and 3 are different questions: significance alone certifies a
    one-point difference on a large sample, and a large effect that the null
    reaches just as easily is not evidence.
    """
    if not null_signatures:
        raise ValueError(
            "stage 2 justification requires a null distribution to compare against"
        )
    percentile = float(np.quantile(null_signatures, 1.0 - NULL_ALPHA))
    median = float(np.median(null_signatures))
    required_excess = MIN_RELATIVE_EXCESS * (1.0 - median)
    return (
        signature > percentile
        and signature >= MIN_ABSOLUTE_SIGNATURE
        and (signature - median) >= required_excess
    )
