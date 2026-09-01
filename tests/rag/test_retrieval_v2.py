"""Contract tests for the flagged retrieval v2 path (report section 27).

The load-bearing guarantees:

- the inlined manifest digest matches `benchmarks.gate2.manifest`'s bit-for-bit,
  so a stale snapshot cannot pass one check and fail the other;
- an enabled flag with broken assets raises instead of degrading to v1;
- a query naming an identifier retrieves its definition (the BM25 arm), and a
  call-neighbour of a hit enters the results (the expansion arm);
- a dense arm whose dimensions do not match is disarmed, recorded, and does not
  take BM25 down with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.gate2.manifest import build_manifest, manifest_digest, save_manifest
from benchmarks.gate2.snapshot import build_stub_snapshot, save_snapshot
from embeddinggemma.retrieval_v2 import RetrievalV2, build_from_env, load_index


def _write_corpus(root: Path) -> None:
    (root / "alpha.py").write_text(
        "def parse_config(path):\n"
        "    return normalise_settings(path)\n"
        "\n"
        "def normalise_settings(raw):\n"
        "    return dict(raw)\n",
        encoding="utf-8",
    )
    (root / "beta.py").write_text(
        "def unrelated_worker(queue):\n"
        "    return queue.pop()\n",
        encoding="utf-8",
    )


@pytest.fixture()
def assets(tmp_path: Path) -> dict[str, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus(corpus)
    manifest = build_manifest(corpus, "test-sha", "v2-test")
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    snapshot_path = tmp_path / "snapshot.npz"
    save_snapshot(build_stub_snapshot(manifest, dimension=16), snapshot_path)
    return {"manifest": manifest_path, "snapshot": snapshot_path, "corpus": corpus}


def test_inlined_digest_matches_gate2_implementation(assets: dict[str, Path]) -> None:
    from embeddinggemma.retrieval_v2 import _digest

    payload = json.loads(assets["manifest"].read_text(encoding="utf-8"))
    corpus = assets["corpus"]
    rebuilt = build_manifest(corpus, "test-sha", "v2-test")
    assert _digest(payload) == manifest_digest(rebuilt)


def test_flag_off_returns_none() -> None:
    assert build_from_env({}) is None
    assert build_from_env({"FUNGUS_RETRIEVAL_V2": "0"}) is None


def test_flag_on_without_manifest_raises() -> None:
    with pytest.raises(ValueError, match="FUNGUS_V2_MANIFEST"):
        build_from_env({"FUNGUS_RETRIEVAL_V2": "1"})


def test_stale_snapshot_is_refused(assets: dict[str, Path], tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "gamma.py").write_text("def lonely():\n    return 1\n", encoding="utf-8")
    foreign = build_manifest(other, "other-sha", "other")
    foreign_snapshot = tmp_path / "foreign.npz"
    save_snapshot(build_stub_snapshot(foreign, dimension=16), foreign_snapshot)
    with pytest.raises(ValueError, match="digest does not match"):
        load_index(assets["manifest"], foreign_snapshot)


def test_bm25_arm_finds_the_named_definition(assets: dict[str, Path]) -> None:
    engine = RetrievalV2(load_index(assets["manifest"]))
    result = engine.search("where is normalise_settings defined?", top_k=3)
    symbols = [row["metadata"]["symbol"] for row in result["results"]]
    assert "normalise_settings" in symbols
    assert result["engine"].startswith("v2:bm25+expand")


def test_expansion_adds_the_call_neighbour(assets: dict[str, Path]) -> None:
    engine = RetrievalV2(load_index(assets["manifest"]))
    # The query names only parse_config; normalise_settings arrives through the
    # call edge, and its row says so.
    result = engine.search("parse_config", top_k=4)
    by_symbol = {row["metadata"]["symbol"]: row for row in result["results"]}
    assert "parse_config" in by_symbol
    assert "normalise_settings" in by_symbol


def test_dimension_mismatch_disarms_dense_and_keeps_serving(assets: dict[str, Path]) -> None:
    index = load_index(assets["manifest"], assets["snapshot"])
    engine = RetrievalV2(index, embed_query=lambda _text: [0.5] * 7)  # wrong dims
    result = engine.search("parse_config", top_k=2)
    assert result["results"], "the BM25 arm must keep serving"
    assert "dense off" in engine.engine
    assert "7 dims" in index.dense_disabled_reason


def test_matching_embedder_arms_the_union(assets: dict[str, Path]) -> None:
    index = load_index(assets["manifest"], assets["snapshot"])
    engine = RetrievalV2(index, embed_query=lambda _text: [0.25] * 16)
    result = engine.search("parse_config", top_k=2)
    assert result["results"]
    assert engine.engine.startswith("v2:union+expand")
