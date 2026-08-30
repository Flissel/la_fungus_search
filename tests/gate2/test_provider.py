from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.gate2.manifest import build_manifest, relevant_for
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.snapshot import build_stub_snapshot


def _corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "alpha.py").write_text(
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def caller():\n"
        "    return helper()\n"
        "\n"
        "def second():\n"
        "    return caller()\n"
        "\n"
        "def third():\n"
        "    return second()\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path):
    root = tmp_path / "corpus"
    _corpus(root)
    manifest = build_manifest(root, commit_sha="sha", manifest_id="m1")
    return manifest, build_stub_snapshot(manifest, dimension=16)


def test_dataset_validates_and_uses_prefixed_query_ids(tmp_path: Path) -> None:
    manifest, snapshot = _fixture(tmp_path)

    dataset = build_gate2_dataset(manifest, snapshot, seed=1)

    dataset.validate()
    assert len(dataset.query_ids) == 2
    assert all(query_id.startswith("q:") for query_id in dataset.query_ids)
    assert not set(dataset.document_ids) & set(dataset.query_ids)


def test_query_documents_are_removed_from_the_corpus(tmp_path: Path) -> None:
    manifest, snapshot = _fixture(tmp_path)

    dataset = build_gate2_dataset(manifest, snapshot, seed=1)

    queried = {query_id[2:] for query_id in dataset.query_ids}
    assert not queried & set(dataset.document_ids)
    assert len(dataset.document_ids) == len(manifest.documents) - 2


def test_relevance_is_call_graph_neighbours_inside_the_corpus(tmp_path: Path) -> None:
    manifest, snapshot = _fixture(tmp_path)

    dataset = build_gate2_dataset(manifest, snapshot, seed=1)

    corpus = set(dataset.document_ids)
    for query_id, relevant in dataset.relevant_by_query.items():
        expected = relevant_for(manifest, query_id[2:]) & corpus
        assert relevant == frozenset(expected)


def test_dataset_is_deterministic_per_seed(tmp_path: Path) -> None:
    manifest, snapshot = _fixture(tmp_path)

    first = build_gate2_dataset(manifest, snapshot, seed=3)
    second = build_gate2_dataset(manifest, snapshot, seed=3)

    assert first.digest() == second.digest()
    assert np.allclose(np.linalg.norm(first.document_vectors, axis=1), 1.0, atol=1e-4)
