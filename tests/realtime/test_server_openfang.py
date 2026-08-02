import importlib
import sys
from types import ModuleType

import pytest


def _module(name, monkeypatch, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_server(monkeypatch, generate):
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
    _module("embeddinggemma.rag.generation", monkeypatch, generate_text=generate)
    sys.modules.pop("embeddinggemma.realtime.server", None)
    return importlib.import_module("embeddinggemma.realtime.server")


def test_realtime_summary_generation_does_not_accept_provider_overrides(monkeypatch):
    received = {}

    def generate(**kwargs):
        received.update(kwargs)
        return "summary"

    server = _load_server(monkeypatch, generate)

    result = server._generate_summary("Explain this", system="Be concise")

    assert result == "summary"
    assert received == {"prompt": "Explain this", "system": "Be concise", "save_prompt_path": None}


def test_realtime_summary_generation_propagates_openfang_failure(monkeypatch):
    def generate(**_kwargs):
        raise RuntimeError("openfang unavailable")

    server = _load_server(monkeypatch, generate)

    with pytest.raises(RuntimeError, match="openfang unavailable"):
        server._generate_summary("Explain this")
