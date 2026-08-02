import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_generation(monkeypatch, client, model="openfang-summary-model"):
    shared = ModuleType("vibemind_shared")
    calls = []

    def get_client_sync(role):
        calls.append(("client", role))
        return client

    def get_model(role):
        calls.append(("model", role))
        return model

    shared.get_client_sync = get_client_sync
    shared.get_model = get_model
    monkeypatch.setitem(sys.modules, "vibemind_shared", shared)
    sys.modules.pop("embeddinggemma.rag.generation", None)
    generation = importlib.import_module("embeddinggemma.rag.generation")
    return generation, calls


def test_generate_text_uses_openfang_summary_role(monkeypatch):
    completions = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generation, calls = _load_generation(monkeypatch, client)

    result = generation.generate_text(prompt="Explain this", system="Be concise")

    assert result == "summary"
    assert calls == [("client", "fungus_summary"), ("model", "fungus_summary")]


def test_generate_judge_text_uses_openfang_judge_role(monkeypatch):
    completions = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="judgement"))]
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generation, calls = _load_generation(monkeypatch, client, model="openfang-judge-model")

    result = generation.generate_judge_text(prompt="Judge this")

    assert result == "judgement"
    assert calls == [("client", "fungus_judge"), ("model", "fungus_judge")]


def test_legacy_provider_wrapper_cannot_override_openfang_authority(monkeypatch):
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    generation, calls = _load_generation(monkeypatch, client, model="configured-by-openfang")

    result = generation.generate_with_openai(
        "Explain this",
        model="local-override",
        api_key="must-not-be-used",
        base_url="https://provider.invalid",
    )

    assert result == "summary"
    assert calls == [("client", "fungus_summary"), ("model", "fungus_summary")]
    assert received["model"] == "configured-by-openfang"


def test_generate_text_propagates_openfang_failure(monkeypatch):
    def unavailable(_role):
        raise RuntimeError("openfang unavailable")

    shared = ModuleType("vibemind_shared")
    shared.get_client_sync = unavailable
    shared.get_model = lambda _role: "unused"
    monkeypatch.setitem(sys.modules, "vibemind_shared", shared)
    sys.modules.pop("embeddinggemma.rag.generation", None)
    generation = importlib.import_module("embeddinggemma.rag.generation")

    with pytest.raises(RuntimeError, match="openfang unavailable"):
        generation.generate_text(prompt="Explain this")


def test_generate_judge_text_propagates_openfang_failure(monkeypatch):
    def unavailable(_role):
        raise RuntimeError("openfang judge unavailable")

    shared = ModuleType("vibemind_shared")
    shared.get_client_sync = unavailable
    shared.get_model = lambda _role: "unused"
    monkeypatch.setitem(sys.modules, "vibemind_shared", shared)
    sys.modules.pop("embeddinggemma.rag.generation", None)
    generation = importlib.import_module("embeddinggemma.rag.generation")

    with pytest.raises(RuntimeError, match="openfang judge unavailable"):
        generation.generate_judge_text(prompt="Judge this")
