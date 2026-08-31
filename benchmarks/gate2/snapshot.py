"""Embedding snapshots for Gate 2, with a deterministic offline stub."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.gate2.manifest import Manifest, manifest_digest


@dataclass(frozen=True)
class Snapshot:
    document_ids: tuple[str, ...]
    vectors: np.ndarray
    backend: str
    model: str
    dimension: int
    manifest_digest: str


def stub_embed(texts: Sequence[str], dimension: int = 16) -> np.ndarray:
    """Deterministic offline embedding derived from the text digest."""
    if len(texts) == 0:
        raise ValueError("cannot embed an empty text sequence")
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        rows.append(np.random.default_rng(seed).normal(size=dimension))
    matrix = np.asarray(rows, dtype=np.float32)
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def build_stub_snapshot(manifest: Manifest, dimension: int = 16) -> Snapshot:
    """Build an offline snapshot; never use it to draw a Gate 2 conclusion."""
    document_ids = tuple(document.document_id for document in manifest.documents)
    sources = [document.source for document in manifest.documents]
    return Snapshot(
        document_ids=document_ids,
        vectors=stub_embed(sources, dimension=dimension),
        backend="stub",
        model="sha256-gaussian",
        dimension=dimension,
        manifest_digest=manifest_digest(manifest),
    )


class ServiceEmbeddingClient:
    """Pairs an embedding client with the model identifier a snapshot must record.

    `EmbeddingServiceClient` in `src/embeddinggemma/mcmp/embeddings.py` exposes no
    `model_id`, so `build_service_snapshot`'s fail-closed provenance read rejects
    it outright. The service does not report which model it is serving either, so
    the identifier cannot be discovered — it is the caller's declaration, and this
    wrapper is where that declaration is made once and explicitly rather than
    defaulted to "unknown" somewhere downstream.

    Production use:

        ServiceEmbeddingClient(EmbeddingServiceClient(), model_id="<model>")
    """

    def __init__(self, client: object, model_id: str) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        self._client = client
        self.model_id = model_id

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.encode(texts)


def build_service_snapshot(manifest: Manifest, client: object, batch_size: int = 32) -> Snapshot:
    """Materialise the production snapshot through an injected embedding client.

    The client is injected rather than constructed so this stays testable offline.
    Production passes ``EmbeddingServiceClient()``, which is fail-closed HTTP.
    """
    if len(manifest.documents) == 0:
        raise ValueError("manifest has no documents to embed")
    document_ids = tuple(document.document_id for document in manifest.documents)
    sources = [document.source for document in manifest.documents]
    rows: list[list[float]] = []
    for start in range(0, len(sources), batch_size):
        rows.extend(client.encode(sources[start : start + batch_size]))
    if len(rows) != len(sources):
        raise ValueError("embedding backend returned the wrong number of vectors")
    matrix = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(norms > 0.0):
        raise ValueError("embedding backend returned a zero vector")
    # Fail closed on provenance: a snapshot whose model identifier is a
    # placeholder is indistinguishable from one built by a different model, and
    # this function runs once, unattended, behind an expensive setup.
    model = getattr(client, "model_id", None)
    if not isinstance(model, str) or not model:
        raise ValueError(
            "embedding client exposes no model_id; snapshot provenance would be empty"
        )
    return Snapshot(
        document_ids=document_ids,
        vectors=(matrix / norms).astype(np.float32),
        backend="embedding-service",
        model=model,
        dimension=int(matrix.shape[1]),
        manifest_digest=manifest_digest(manifest),
    )


def save_snapshot(snapshot: Snapshot, path: Path) -> None:
    # Ids and metadata are stored as NumPy unicode arrays, never object arrays, so
    # that load_snapshot never needs allow_pickle. A snapshot file is data, not a
    # code path.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        document_ids=np.asarray(snapshot.document_ids, dtype=np.str_),
        vectors=snapshot.vectors,
        backend=np.asarray(snapshot.backend, dtype=np.str_),
        model=np.asarray(snapshot.model, dtype=np.str_),
        dimension=np.asarray(snapshot.dimension, dtype=np.int64),
        manifest_digest=np.asarray(snapshot.manifest_digest, dtype=np.str_),
    )


def load_snapshot(path: Path, expected_manifest_digest: str) -> Snapshot:
    # allow_pickle stays at its safe default of False.
    with np.load(path) as payload:
        stored_digest = str(payload["manifest_digest"])
        if stored_digest != expected_manifest_digest:
            raise ValueError("snapshot manifest digest does not match")
        return Snapshot(
            document_ids=tuple(str(value) for value in payload["document_ids"]),
            vectors=np.asarray(payload["vectors"], dtype=np.float32),
            backend=str(payload["backend"]),
            model=str(payload["model"]),
            dimension=int(payload["dimension"]),
            manifest_digest=stored_digest,
        )
