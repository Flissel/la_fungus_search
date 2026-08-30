"""Assemble a BenchmarkDataset from a Gate 2 manifest and embedding snapshot."""

from __future__ import annotations

import numpy as np

from benchmarks.gate2.manifest import Manifest, query_candidates, relevant_for
from benchmarks.gate2.snapshot import Snapshot
from benchmarks.mcmp.contracts import BenchmarkDataset

QUERY_PREFIX = "q:"


def build_gate2_dataset(manifest: Manifest, snapshot: Snapshot, seed: int) -> BenchmarkDataset:
    """Build a two-query dataset whose corpus excludes its own query documents."""
    candidates = sorted(query_candidates(manifest))
    if len(candidates) < 2:
        raise ValueError("manifest must supply at least two query candidates")
    rng = np.random.default_rng(seed)
    chosen = [candidates[int(index)] for index in rng.choice(len(candidates), size=2, replace=False)]

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
