import importlib
import sys
import asyncio
from types import ModuleType

import pytest


def _module(name, monkeypatch, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_server(monkeypatch, generate_summary, generate_judge):
    class Retriever:
        pass

    _module("embeddinggemma.mcmp_rag", monkeypatch, MCPMRetriever=Retriever)
    _module("embeddinggemma.ui.corpus", monkeypatch, collect_codebase_chunks=lambda *_a, **_k: [], list_code_files=lambda *_a, **_k: [])
    _module("embeddinggemma.ui.queries", monkeypatch, dedup_multi_queries=lambda values: values)
    _module("embeddinggemma.ui.reports", monkeypatch, merge_reports_to_summary=lambda values: values)
    _module("embeddinggemma.llm.prompts", monkeypatch, get_report_instructions=lambda *_a: "", build_report_prompt=lambda *_a: "", build_judge_prompt=lambda *_a: "")
    _module("embeddinggemma.prompts", monkeypatch, _default_instructions=lambda *_a: "", report_schema_hint=lambda: "")
    for name in ("deep", "structure", "exploratory", "summary", "repair", "steering"):
        _module(f"embeddinggemma.modeprompts.{name}", monkeypatch, instructions=lambda: "")
    _module(
        "embeddinggemma.rag.generation",
        monkeypatch,
        generate_text=generate_summary,
        generate_judge_text=generate_judge,
    )
    sys.modules.pop("embeddinggemma.realtime.server", None)
    return importlib.import_module("embeddinggemma.realtime.server")


def test_realtime_summary_generation_does_not_accept_provider_overrides(monkeypatch):
    received = {}

    def generate(**kwargs):
        received.update(kwargs)
        return "summary"

    server = _load_server(monkeypatch, generate, lambda **_kwargs: "unused")

    result = server._generate_summary("Explain this", system="Be concise")

    assert result == "summary"
    assert received == {"prompt": "Explain this", "system": "Be concise", "save_prompt_path": None}


def test_realtime_summary_generation_propagates_openfang_failure(monkeypatch):
    def generate(**_kwargs):
        raise RuntimeError("openfang unavailable")

    server = _load_server(monkeypatch, generate, lambda **_kwargs: "unused")

    with pytest.raises(RuntimeError, match="openfang unavailable"):
        server._generate_summary("Explain this")


def test_realtime_judge_generation_uses_fixed_judge_path(monkeypatch):
    calls = []

    def summary(**_kwargs):
        raise AssertionError("judge path must not use fungus_summary")

    def judge(**kwargs):
        calls.append(kwargs)
        return '{"items": []}'

    server = _load_server(monkeypatch, summary, judge)

    async def run_judge():
        return server.streamer._llm_judge([])

    assert asyncio.run(run_judge()) == {}
    assert len(calls) == 1
    assert calls[0]["prompt"] == ""
    assert calls[0]["system"] is None
    assert calls[0]["save_prompt_path"].endswith("judge_prompt_step_0.txt")


def test_realtime_embedding_loader_uses_fixed_openfang_gateway_role(monkeypatch):
    expected_model = object()
    calls = []

    def load_embedding_model():
        calls.append(True)
        return expected_model

    _module(
        "embeddinggemma.mcmp.embeddings",
        monkeypatch,
        load_embedding_model=load_embedding_model,
    )
    server = _load_server(monkeypatch, lambda **_kwargs: "summary", lambda **_kwargs: "judge")

    assert server._load_embed_client() is expected_model
    assert calls == [True]
