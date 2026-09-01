"""Materialise a Gate 2 snapshot from a locally hosted embedding model.

The production route in `snapshot.py` goes through the VibeMind embedding-service,
which is a thin wrapper over OpenAI `text-embedding-3-large`. On 2026-09-01 that
route was verified dead at the account level, not the infrastructure level: the
service builds, starts and reports healthy, and the first real call returns
``429 insufficient_quota -- "You have no credits remaining"``. Docker was never
the blocker.

This module is the offline substitute. It embeds the same manifest with a model
already cached on disk, and it records that fact rather than hiding it: the
snapshot's ``backend`` and ``model`` fields name the local model, so no reader can
mistake the result for a production-embedding measurement.

**What that costs in generalisation, stated up front.** A Gate 2 conclusion drawn
from this snapshot is a statement about *this* embedding space. Whether real code
geometry contains far-but-chain-reachable structure can genuinely differ between a
1024-dimensional local model and the 3072-dimensional production one. This
snapshot answers the question for the space it was built in, and licenses no claim
about production retrieval until the production snapshot exists.

Two stages, because the environments differ:

1. ``export`` runs in the Fungus venv, builds the AST manifest and writes the
   document sources to JSON.
2. ``assemble`` runs in the same venv after an embedding step has produced a
   vectors file, and writes the snapshot.

The embedding step in between runs in whatever interpreter has the model stack;
see ``benchmarks/gate2/embed_local.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from dataclasses import replace

from benchmarks.gate2.manifest import (
    Manifest,
    build_manifest,
    load_manifest,
    manifest_digest,
    query_candidates,
    save_manifest,
)
from benchmarks.gate2.snapshot import build_service_snapshot, save_snapshot


class PrecomputedClient:
    """Serves vectors that were computed elsewhere, keyed by document source.

    `build_service_snapshot` asks a client to encode batches and reads `model_id`
    off it for provenance. Both work unchanged here; the encoding simply happened
    in a different interpreter.
    """

    def __init__(self, sources: Sequence[str], vectors: np.ndarray, model_id: str) -> None:
        if len(sources) != vectors.shape[0]:
            raise ValueError(
                f"{len(sources)} sources but {vectors.shape[0]} vectors -- "
                "the embedding step did not run over this manifest"
            )
        if not model_id:
            raise ValueError("model_id must be a non-empty string")
        self._by_source: dict[str, np.ndarray] = {}
        for source, vector in zip(sources, vectors):
            self._by_source[source] = vector
        self.model_id = model_id

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._by_source]
        if missing:
            raise KeyError(
                f"{len(missing)} document sources have no precomputed vector; "
                "the manifest and the embedding input are out of step"
            )
        return [self._by_source[text].tolist() for text in texts]


def _commit_sha(corpus_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=corpus_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def subsample(manifest: Manifest, limit: int, seed: int) -> Manifest:
    """A fixed random subset of the corpus, with the call graph cut consistently.

    Memory, not taste, forces this. `geometry_cache` holds a full N x N similarity
    matrix, so a 16 497-document corpus costs 1.09 GB per cached dataset and the
    selection sweep holds one per seed -- 33 GB across 30 seeds, on a host with a
    documented history of RAM exhaustion. At 4 000 documents the same cache is
    64 MB.

    Edges to dropped documents are removed rather than left dangling, so the
    surviving call graph is the true graph *of the sample*: a document's relevance
    set never names something the corpus does not contain.
    """
    if limit >= len(manifest.documents):
        return manifest
    rng = np.random.default_rng(seed)
    chosen_positions = sorted(
        int(position)
        for position in rng.choice(len(manifest.documents), size=limit, replace=False)
    )
    documents = tuple(manifest.documents[position] for position in chosen_positions)
    kept = {document.document_id for document in documents}
    callees = {
        document_id: frozenset(target for target in targets if target in kept)
        for document_id, targets in manifest.callees_by_document.items()
        if document_id in kept
    }
    callers = {
        document_id: frozenset(source for source in sources if source in kept)
        for document_id, sources in manifest.callers_by_document.items()
        if document_id in kept
    }
    return replace(
        manifest,
        manifest_id=f"{manifest.manifest_id}-n{limit}s{seed}",
        documents=documents,
        callees_by_document=callees,
        callers_by_document=callers,
    )


def export(
    corpus_root: Path,
    manifest_path: Path,
    sources_path: Path,
    manifest_id: str,
    max_documents: int = 0,
    sample_seed: int = 0,
) -> None:
    manifest = build_manifest(corpus_root, _commit_sha(corpus_root), manifest_id)
    if max_documents > 0:
        manifest = subsample(manifest, max_documents, sample_seed)
    save_manifest(manifest, manifest_path)
    sources = [document.source for document in manifest.documents]
    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(json.dumps(sources), encoding="utf-8")
    queries = query_candidates(manifest)
    print(f"manifest      : {manifest_path}")
    print(f"digest        : {manifest_digest(manifest)}")
    print(f"documents     : {len(manifest.documents)}")
    print(f"query cands   : {len(queries)}")
    print(f"discarded     : {len(manifest.discarded_names)} ambiguous call names")
    print(f"sources       : {sources_path}")


def assemble(
    manifest_path: Path,
    sources_path: Path,
    vectors_path: Path,
    snapshot_path: Path,
    model_id: str,
    backend: str,
) -> None:
    manifest = load_manifest(manifest_path)
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    with np.load(vectors_path) as payload:
        vectors = payload["vectors"]
    client = PrecomputedClient(sources, vectors, model_id)
    snapshot = build_service_snapshot(manifest, client, backend=backend)
    save_snapshot(snapshot, snapshot_path)
    print(f"snapshot      : {snapshot_path}")
    print(f"backend/model : {snapshot.backend} / {snapshot.model}")
    print(f"documents     : {len(snapshot.document_ids)}")
    print(f"dimension     : {snapshot.dimension}")
    print(f"digest        : {snapshot.manifest_digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    exporter = sub.add_parser("export", help="build the manifest and dump sources")
    exporter.add_argument("--corpus-root", type=Path, required=True)
    exporter.add_argument("--manifest", type=Path, required=True)
    exporter.add_argument("--sources", type=Path, required=True)
    exporter.add_argument("--manifest-id", default="embeddinggemma-local-v1")
    exporter.add_argument("--max-documents", type=int, default=0)
    exporter.add_argument("--sample-seed", type=int, default=0)

    assembler = sub.add_parser("assemble", help="turn vectors into a snapshot")
    assembler.add_argument("--manifest", type=Path, required=True)
    assembler.add_argument("--sources", type=Path, required=True)
    assembler.add_argument("--vectors", type=Path, required=True)
    assembler.add_argument("--snapshot", type=Path, required=True)
    assembler.add_argument("--model-id", required=True)
    assembler.add_argument("--backend", default="local-transformers")

    arguments = parser.parse_args()
    if arguments.command == "export":
        export(
            arguments.corpus_root,
            arguments.manifest,
            arguments.sources,
            arguments.manifest_id,
            arguments.max_documents,
            arguments.sample_seed,
        )
    else:
        assemble(
            arguments.manifest,
            arguments.sources,
            arguments.vectors,
            arguments.snapshot,
            arguments.model_id,
            arguments.backend,
        )


if __name__ == "__main__":
    main()
