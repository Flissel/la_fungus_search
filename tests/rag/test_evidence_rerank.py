"""The opt-in LLM rerank step of the evidence CLI — fail-closed, offline tests."""

from __future__ import annotations

import json
import subprocess

from embeddinggemma import maintainer_evidence


def _hits() -> list[dict]:
    return [
        {"file": "a.py", "start_line": 1, "end_line": 3, "symbol": "alpha",
         "score": 3.0, "digest": "a" * 64, "source": "def alpha():\n    return 1\n",
         "expanded": False},
        {"file": "b.py", "start_line": 1, "end_line": 3, "symbol": "beta",
         "score": 2.0, "digest": "b" * 64, "source": "def beta():\n    return 2\n",
         "expanded": False},
        {"file": "c.py", "start_line": 1, "end_line": 3, "symbol": "gamma",
         "score": 1.0, "digest": "c" * 64, "source": "def gamma():\n    return 3\n",
         "expanded": True},
    ]


def _fake_run(reply: str, returncode: int = 0):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=returncode,
            stdout=json.dumps({"result": reply, "total_cost_usd": 0.01}), stderr="",
        )

    return run


def test_valid_permutation_reorders(monkeypatch) -> None:
    monkeypatch.setattr(maintainer_evidence.subprocess, "run", _fake_run("[2, 0, 1]"))
    ordered, state = maintainer_evidence.rerank_with_llm("claim", _hits(), "claude", "haiku")
    assert state == "llm"
    assert [hit["symbol"] for hit in ordered] == ["gamma", "alpha", "beta"]


def test_non_permutation_keeps_order_and_reports(monkeypatch) -> None:
    monkeypatch.setattr(maintainer_evidence.subprocess, "run", _fake_run("[0, 0, 1]"))
    hits = _hits()
    ordered, state = maintainer_evidence.rerank_with_llm("claim", hits, "claude", "haiku")
    assert ordered == hits, "a bad reply must not reorder anything"
    assert state.startswith("failed: not a permutation")


def test_prose_reply_keeps_order(monkeypatch) -> None:
    monkeypatch.setattr(
        maintainer_evidence.subprocess, "run", _fake_run("I think alpha is best.")
    )
    hits = _hits()
    ordered, state = maintainer_evidence.rerank_with_llm("claim", hits, "claude", "haiku")
    assert ordered == hits
    assert state.startswith("failed: no JSON array")


def test_missing_binary_keeps_order() -> None:
    hits = _hits()
    ordered, state = maintainer_evidence.rerank_with_llm(
        "claim", hits, "Z:/does/not/exist/claude.exe", "haiku"
    )
    assert ordered == hits
    assert state.startswith("failed:")


def test_single_hit_skips() -> None:
    single = _hits()[:1]
    ordered, state = maintainer_evidence.rerank_with_llm("claim", single, "claude", "haiku")
    assert ordered == single
    assert state.startswith("skipped")
