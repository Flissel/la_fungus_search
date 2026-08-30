from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.gate2.manifest import build_manifest, manifest_digest
from benchmarks.gate2.snapshot import (
    build_service_snapshot,
    build_stub_snapshot,
    load_snapshot,
    save_snapshot,
    stub_embed,
)


def _corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "alpha.py").write_text(
        "def helper():\n    return 1\n\ndef caller():\n    return helper()\n",
        encoding="utf-8",
    )


def test_stub_embed_is_deterministic_and_unit_norm() -> None:
    first = stub_embed(["alpha", "beta"], dimension=16)
    second = stub_embed(["alpha", "beta"], dimension=16)

    assert first.shape == (2, 16)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(first[0], first[1])


def test_snapshot_round_trips(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _corpus(root)
    manifest = build_manifest(root, commit_sha="sha", manifest_id="m1")
    snapshot = build_stub_snapshot(manifest, dimension=16)
    path = tmp_path / "snap.npz"

    save_snapshot(snapshot, path)
    reloaded = load_snapshot(path, manifest_digest(manifest))

    assert reloaded.document_ids == snapshot.document_ids
    assert np.array_equal(reloaded.vectors, snapshot.vectors)
    assert reloaded.backend == "stub"


def test_snapshot_rejects_a_mismatched_manifest(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _corpus(root)
    manifest = build_manifest(root, commit_sha="sha", manifest_id="m1")
    path = tmp_path / "snap.npz"
    save_snapshot(build_stub_snapshot(manifest, dimension=16), path)

    with pytest.raises(ValueError, match="snapshot manifest digest does not match"):
        load_snapshot(path, "0" * 64)


class _FakeEmbeddingClient:
    """Stands in for EmbeddingServiceClient; keeps the test fully offline."""

    model_id = "fake-model"

    def __init__(self, dimension: int, truncate: bool = False) -> None:
        self.dimension = dimension
        self.truncate = truncate
        self.batches: list[list[str]] = []

    def encode(self, texts):
        self.batches.append(list(texts))
        rows = [[float(len(text) + offset + 1) for offset in range(self.dimension)] for text in texts]
        return rows[:-1] if self.truncate and rows else rows


def test_service_snapshot_preserves_order_batches_and_normalizes(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _corpus(root)
    manifest = build_manifest(root, commit_sha="sha", manifest_id="m1")
    client = _FakeEmbeddingClient(dimension=4)

    snapshot = build_service_snapshot(manifest, client, batch_size=1)

    assert snapshot.document_ids == tuple(d.document_id for d in manifest.documents)
    assert snapshot.backend == "embedding-service"
    assert snapshot.model == "fake-model"
    assert snapshot.dimension == 4
    assert len(client.batches) == len(manifest.documents)
    assert np.allclose(np.linalg.norm(snapshot.vectors, axis=1), 1.0, atol=1e-5)


def test_service_snapshot_rejects_a_short_response(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _corpus(root)
    manifest = build_manifest(root, commit_sha="sha", manifest_id="m1")

    with pytest.raises(ValueError, match="wrong number of vectors"):
        build_service_snapshot(manifest, _FakeEmbeddingClient(4, truncate=True), batch_size=64)


def test_stub_embed_rejects_an_empty_text_sequence() -> None:
    with pytest.raises(ValueError, match="cannot embed an empty text sequence"):
        stub_embed([], dimension=16)


def test_service_snapshot_rejects_an_empty_manifest_without_calling_the_client(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    (root / "empty.py").write_text("x = 1\n", encoding="utf-8")
    manifest = build_manifest(root, commit_sha="sha", manifest_id="m1")
    client = _FakeEmbeddingClient(dimension=4)

    with pytest.raises(ValueError, match="manifest has no documents to embed"):
        build_service_snapshot(manifest, client, batch_size=64)

    assert client.batches == []
