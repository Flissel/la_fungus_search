from __future__ import annotations

from pathlib import Path

from benchmarks.gate2.manifest import (
    build_manifest,
    load_manifest,
    manifest_digest,
    query_candidates,
    relevant_for,
    save_manifest,
)


def _write_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "alpha.py").write_text(
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def caller():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    (root / "beta.py").write_text(
        "def bridge():\n"
        "    def inner():\n"
        "        return helper()\n"
        "    return inner()\n"
        "\n"
        "def ambiguous_user():\n"
        "    return dupe()\n"
        "\n"
        "def dupe():\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (root / "gamma.py").write_text(
        "def dupe():\n"
        "    return 1\n"
        "\n"
        "def lonely():\n"
        "    return 42\n",
        encoding="utf-8",
    )
    (root / "delta.py").write_text(
        "def class_level_helper():\n"
        "    return 1\n"
        "\n"
        "def method_level_helper():\n"
        "    return 2\n"
        "\n"
        "class Widget:\n"
        "    class_level_helper()\n"
        "\n"
        "    def method(self):\n"
        "        return method_level_helper()\n",
        encoding="utf-8",
    )


def test_manifest_records_documents_and_resolved_edges(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)

    manifest = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")

    ids = {document.document_id for document in manifest.documents}
    assert "alpha.helper" in ids
    assert "alpha.caller" in ids
    assert "beta.bridge" in ids
    assert "beta.inner" not in ids
    assert manifest.callees_by_document["alpha.caller"] == frozenset({"alpha.helper"})
    assert manifest.callers_by_document["alpha.helper"] == frozenset(
        {"alpha.caller", "beta.bridge"}
    )


def test_nested_function_calls_belong_to_the_enclosing_document(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)

    manifest = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")

    assert "alpha.helper" in manifest.callees_by_document["beta.bridge"]


def test_ambiguous_call_names_are_discarded_not_guessed(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)

    manifest = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")

    assert manifest.callees_by_document["beta.ambiguous_user"] == frozenset()
    assert "dupe" in manifest.discarded_names


def test_relevant_is_callers_union_callees_without_self(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)

    manifest = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")

    assert relevant_for(manifest, "alpha.helper") == frozenset(
        {"alpha.caller", "beta.bridge"}
    )
    assert "alpha.helper" not in relevant_for(manifest, "alpha.helper")


def test_query_candidates_exclude_documents_without_neighbours(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)

    manifest = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")

    assert "gamma.lonely" not in query_candidates(manifest)
    assert "alpha.helper" in query_candidates(manifest)


def test_manifest_round_trips_and_digest_is_stable(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)
    manifest = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")
    path = tmp_path / "manifest.json"

    save_manifest(manifest, path)
    reloaded = load_manifest(path)

    assert manifest_digest(reloaded) == manifest_digest(manifest)
    assert reloaded.commit_sha == "abc123"
    assert reloaded.documents == manifest.documents


def test_digest_changes_when_a_function_body_changes(tmp_path: Path) -> None:
    """A refactored body is the mismatch a snapshot digest check has to catch.

    Same corpus root, same commit sha, same document ids, same call structure:
    everything the digest hashed before the source text was added. The
    snapshot builders embed ``document.source``, so two corpora that differ
    only in a body produce different vectors and must not share a digest.
    """
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    template = "def helper():\n    return {value}\n\ndef caller():\n    return helper()\n"

    (root / "alpha.py").write_text(template.format(value=1), encoding="utf-8")
    first = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")
    (root / "alpha.py").write_text(template.format(value=2), encoding="utf-8")
    second = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")

    assert first.corpus_root == second.corpus_root
    assert [d.document_id for d in first.documents] == [
        d.document_id for d in second.documents
    ]
    assert first.callees_by_document == second.callees_by_document
    assert first.documents[0].source != second.documents[0].source

    assert manifest_digest(first) != manifest_digest(second)


def test_zero_match_call_names_are_recorded_as_unresolved(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)

    manifest = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")

    assert "inner" in manifest.unresolved_names
    assert "inner" not in manifest.discarded_names


def test_class_document_only_counts_class_level_calls(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus(root)

    manifest = build_manifest(root, commit_sha="abc123", manifest_id="test-v1")

    ids = {document.document_id for document in manifest.documents}
    assert "delta.Widget" in ids
    assert "delta.Widget.method" in ids

    assert manifest.callees_by_document["delta.Widget.method"] == frozenset(
        {"delta.method_level_helper"}
    )
    assert manifest.callees_by_document["delta.Widget"] == frozenset(
        {"delta.class_level_helper"}
    )
