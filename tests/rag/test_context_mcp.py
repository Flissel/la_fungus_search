"""The context MCP server's implementation functions, offline.

The FastMCP wrappers are thin by design; what needs testing is the config
resolution (fail-closed), corpus routing, and that hits carry their receipts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.gate2.manifest import build_manifest, save_manifest
from embeddinggemma import context_mcp


@pytest.fixture()
def corpora(tmp_path: Path, monkeypatch) -> Path:
    for name, body in (
        ("alpha", "def parse_config(path):\n    return normalise_settings(path)\n\n"
                  "def normalise_settings(raw):\n    return dict(raw)\n"),
        ("beta", "def weekly_review(entries):\n    return entries[:3]\n"),
    ):
        corpus = tmp_path / name
        corpus.mkdir()
        (corpus / f"{name}.py").write_text(body, encoding="utf-8")
        manifest = build_manifest(corpus, "sha", name)
        save_manifest(manifest, tmp_path / f"{name}.json")
    config = tmp_path / "corpora.json"
    config.write_text(
        json.dumps(
            [
                {"name": "code", "manifest": str(tmp_path / "alpha.json")},
                {"name": "vault", "manifest": str(tmp_path / "beta.json"), "rank_rule": "rrf"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FUNGUS_V2_CORPORA", str(config))
    monkeypatch.delenv("FUNGUS_V2_MANIFEST", raising=False)
    context_mcp._STATE.clear()
    yield config
    context_mcp._STATE.clear()


def test_missing_config_fails_closed(monkeypatch) -> None:
    monkeypatch.delenv("FUNGUS_V2_CORPORA", raising=False)
    monkeypatch.delenv("FUNGUS_V2_MANIFEST", raising=False)
    context_mcp._STATE.clear()
    with pytest.raises(ValueError, match="no corpora configured|FUNGUS_V2_MANIFEST"):
        context_mcp.corpora_impl()
    context_mcp._STATE.clear()


def test_corpora_lists_engines_and_digests(corpora: Path) -> None:
    overview = context_mcp.corpora_impl()
    rows = {row["corpus"]: row for row in overview["corpora"]}
    assert set(rows) == {"code", "vault"}
    assert rows["vault"]["rank_rule"] == "rrf"
    assert len(rows["code"]["manifest_digest"]) == 64


def test_search_all_corpora_tags_hits(corpora: Path) -> None:
    answer = context_mcp.search_impl("parse_config weekly_review", top_k=6)
    tags = {row["metadata"]["corpus"] for row in answer["results"]}
    assert tags == {"code", "vault"}


def test_search_single_corpus_routes_and_rejects_unknown(corpora: Path) -> None:
    answer = context_mcp.search_impl("parse_config", corpus="code", top_k=3)
    assert all(row["metadata"]["corpus"] == "code" for row in answer["results"])
    symbols = {row["metadata"]["symbol"] for row in answer["results"]}
    assert "normalise_settings" in symbols, "expansion must survive the MCP path"
    with pytest.raises(ValueError, match="unknown corpus"):
        context_mcp.search_impl("x", corpus="nope")


def test_empty_query_is_refused(corpora: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        context_mcp.search_impl("   ")
