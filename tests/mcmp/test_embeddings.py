import types
import pytest


def test_load_sentence_model_cpu(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    calls = {"ctor": []}

    class DummyModel:
        def __init__(self, name, device=None):
            calls["ctor"].append((name, device))

    # Patch device resolver and env logger
    monkeypatch.setattr(embeddings, "resolve_device", lambda pref: "cpu")
    monkeypatch.setattr(embeddings, "_log_torch_env", lambda: None)
    monkeypatch.setattr(embeddings, "SentenceTransformer", DummyModel)

    m = embeddings.load_sentence_model("test-model", device_preference="auto")
    assert isinstance(m, DummyModel)
    assert calls["ctor"] == [("test-model", "cpu")]


def test_load_sentence_model_fallback_to_cpu(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    class FailingThenCPU:
        def __init__(self, name, device=None):
            # First attempt with non-CPU should fail to trigger fallback
            if device != "cpu":
                raise RuntimeError("No GPU available")
            self.name = name
            self.device = device

    ctor_calls = []

    def ctor_spy(name, device=None):
        ctor_calls.append((name, device))
        return FailingThenCPU(name, device=device)

    monkeypatch.setattr(embeddings, "resolve_device", lambda pref: "cuda")
    monkeypatch.setattr(embeddings, "_log_torch_env", lambda: None)
    monkeypatch.setattr(embeddings, "SentenceTransformer", ctor_spy)

    m = embeddings.load_sentence_model("test-model", device_preference="auto")
    assert getattr(m, "device", None) == "cpu"
    # Verify we tried cuda then fell back to cpu
    assert ctor_calls == [("test-model", "cuda"), ("test-model", "cpu")]


