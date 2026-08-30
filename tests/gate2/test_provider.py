from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def test_every_query_has_a_nonempty_relevant_set(tmp_path: Path) -> None:
    # Of the 6 possible query pairs over the helper -> caller -> second -> third
    # chain, 2 leave one of their own queries with zero relevant documents once
    # the pair is excluded from the corpus (each pair member was the other's
    # only call-graph neighbour). This must never be observable from the built
    # dataset, for any seed.
    manifest, snapshot = _fixture(tmp_path)

    for seed in range(8):
        dataset = build_gate2_dataset(manifest, snapshot, seed=seed)
        for query_id, relevant in dataset.relevant_by_query.items():
            assert relevant, f"seed={seed} query_id={query_id} has an empty relevant set"


def test_no_valid_query_pair_raises(tmp_path: Path) -> None:
    # Exactly two mutually-calling functions and nothing else: both are query
    # candidates, but the only possible pair is the two of them together, and
    # drawing both empties the corpus each depended on for relevance. No pair
    # can ever satisfy the non-empty-relevant-set requirement here.
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    (root / "pair.py").write_text(
        "def a():\n"
        "    return b()\n"
        "\n"
        "def b():\n"
        "    return a()\n",
        encoding="utf-8",
    )
    manifest = build_manifest(root, commit_sha="sha", manifest_id="pair")
    snapshot = build_stub_snapshot(manifest, dimension=16)

    with pytest.raises(
        ValueError,
        match="no query pair leaves both queries with a non-empty relevant set",
    ):
        build_gate2_dataset(manifest, snapshot, seed=1)
