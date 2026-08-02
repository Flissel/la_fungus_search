from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def _load_judge_runner(monkeypatch, *, client, model: str = "openfang-judge-model"):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.syspath_prepend(str(src_dir))

    shared = types.ModuleType("vibemind_shared")
    def get_client_sync(role: str):
        assert role == "fungus_judge"
        return client

    def get_model(role: str):
        assert role == "fungus_judge"
        return model

    shared.get_client_sync = get_client_sync
    shared.get_model = get_model
    monkeypatch.setitem(sys.modules, "vibemind_shared", shared)
    sys.modules.pop("embeddinggemma.tools.judge_runner", None)
    return importlib.import_module("embeddinggemma.tools.judge_runner")


def test_main_uses_openfang_judge_role_and_preserves_json_output(monkeypatch, tmp_path, capsys):
    calls: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"items": []}'))]
            )

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    runner = _load_judge_runner(monkeypatch, client=client)
    results = tmp_path / "results.json"
    results.write_text(json.dumps([{"id": 7, "score": 0.8, "content": "relevant chunk"}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["judge_runner", "--query", "find the handler", "--results", str(results)])

    runner.main()

    assert calls["model"] == "openfang-judge-model"
    messages = calls["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    assert "find the handler" in messages[0]["content"]
    assert json.loads(capsys.readouterr().out) == {"items": []}


def test_main_propagates_openfang_client_failure_without_fallback(monkeypatch, tmp_path):
    class UnavailableShared(types.ModuleType):
        def get_client_sync(self, role):
            raise RuntimeError("OpenFang unavailable")

        def get_model(self, role):
            raise AssertionError("model must not be resolved after client failure")

    src_dir = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.syspath_prepend(str(src_dir))
    monkeypatch.setitem(sys.modules, "vibemind_shared", UnavailableShared("vibemind_shared"))
    sys.modules.pop("embeddinggemma.tools.judge_runner", None)
    runner = importlib.import_module("embeddinggemma.tools.judge_runner")
    results = tmp_path / "results.json"
    results.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["judge_runner", "--query", "q", "--results", str(results)])

    with pytest.raises(RuntimeError, match="OpenFang unavailable"):
        runner.main()


def test_cli_rejects_provider_override(monkeypatch, tmp_path):
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=object()))
    runner = _load_judge_runner(monkeypatch, client=client)
    results = tmp_path / "results.json"
    results.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["judge_runner", "--query", "q", "--results", str(results), "--provider", "ollama"],
    )

    with pytest.raises(SystemExit, match="2"):
        runner.main()
