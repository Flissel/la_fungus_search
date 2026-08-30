"""Assemble a BenchmarkDataset from a Gate 2 manifest and embedding snapshot."""

from __future__ import annotations

import numpy as np

from benchmarks.gate2.manifest import Manifest, query_candidates, relevant_for
from benchmarks.gate2.snapshot import Snapshot
from benchmarks.mcmp.contracts import BenchmarkDataset

QUERY_PREFIX = "q:"


def _select_query_pair(candidates: list[str], manifest: Manifest, rng: np.random.Generator) -> list[str]:
    """Deterministically search for a query pair that leaves both queries with
    a non-empty relevant set once the pair itself is removed from the corpus.

    A pair whose two members are each other's only call-graph neighbour (or
    whose relevant sets otherwise collapse once both are excluded) would hand
    that query slot zero ground-truth relevant documents -- a degenerate
    recall target. The permutation makes the search itself seed-derived so
    the result stays deterministic per seed, but the ordering in which
    candidate pairs are tried is not the pair's own sort order.
    """
    order = [candidates[int(index)] for index in rng.permutation(len(candidates))]
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            pair = [order[i], order[j]]
            pair_set = set(pair)
            if all(relevant_for(manifest, document_id) - pair_set for document_id in pair):
                return pair
    raise ValueError("no query pair leaves both queries with a non-empty relevant set")


def build_gate2_dataset(manifest: Manifest, snapshot: Snapshot, seed: int) -> BenchmarkDataset:
    """Build a two-query dataset whose corpus excludes its own query documents."""
    candidates = sorted(query_candidates(manifest))
    if len(candidates) < 2:
        raise ValueError("manifest must supply at least two query candidates")
    rng = np.random.default_rng(seed)
    chosen = _select_query_pair(candidates, manifest, rng)

    index_by_id = {document_id: position for position, document_id in enumerate(snapshot.document_ids)}
    missing = [document_id for document_id in chosen if document_id not in index_by_id]
    if missing:
        raise ValueError("snapshot is missing a query document")

    corpus_ids = tuple(
        document.document_id
        for document in manifest.documents
        if document.document_id not in set(chosen)
    )
    corpus_rows = np.stack([snapshot.vectors[index_by_id[document_id]] for document_id in corpus_ids])
    query_rows = np.stack([snapshot.vectors[index_by_id[document_id]] for document_id in chosen])

    corpus_set = set(corpus_ids)
    relevant_by_query = {
        f"{QUERY_PREFIX}{document_id}": frozenset(relevant_for(manifest, document_id) & corpus_set)
        for document_id in chosen
    }

    dataset = BenchmarkDataset(
        dataset_id=f"gate2-{manifest.manifest_id}",
        seed=seed,
        document_ids=corpus_ids,
        document_vectors=np.asarray(corpus_rows, dtype=np.float32),
        query_ids=tuple(f"{QUERY_PREFIX}{document_id}" for document_id in chosen),
        query_vectors=np.asarray(query_rows, dtype=np.float32),
        relevant_by_query=relevant_by_query,
    )
    dataset.validate()
    return dataset
