import pytest


def test_resolve_device_auto_cpu(monkeypatch):
    from embeddinggemma.rag import embeddings
    # Simulate torch missing or cuda unavailable
    monkeypatch.setattr(embeddings, "torch", None)
    assert embeddings.resolve_device("auto") == "cpu"


def test_resolve_device_cuda_fallback(monkeypatch):
    from embeddinggemma.rag import embeddings

    class DummyCUDA:
        @staticmethod
        def is_available():
            return False

    class DummyTorch:
        cuda = DummyCUDA()
        version = type("V", (), {"cuda": "12.1"})

    monkeypatch.setattr(embeddings, "torch", DummyTorch)
    dev = embeddings.resolve_device("cuda")
    assert dev == "cpu"


def test_embedding_backend_load_success(monkeypatch):
    from embeddinggemma.rag import embeddings

    calls = {"ctor": []}

    class DummyModel:
        def __init__(self, name, device=None):
            calls["ctor"].append((name, device))

        def encode(self, texts):
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(embeddings, "SentenceTransformer", DummyModel)
    monkeypatch.setattr(embeddings, "_log_torch_environment", lambda level=None: None)
    monkeypatch.setattr(embeddings, "resolve_device", lambda pref: "cpu")

    backend = embeddings.EmbeddingBackend(model_name="foo/bar", device_preference="auto")
    model = backend.load()
    assert isinstance(model, DummyModel)
    assert calls["ctor"] == [("foo/bar", "cpu")]


def test_embedding_backend_fallback_cpu(monkeypatch):
    from embeddinggemma.rag import embeddings

    class DummyModel:
        def __init__(self, name, device=None):
            if device != "cpu":
                raise RuntimeError("no gpu")
            self.name = name
            self.device = device

        def encode(self, texts):
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(embeddings, "SentenceTransformer", DummyModel)
    monkeypatch.setattr(embeddings, "_log_torch_environment", lambda level=None: None)
    monkeypatch.setattr(embeddings, "resolve_device", lambda pref: "cuda")

    backend = embeddings.EmbeddingBackend(model_name="foo/bar", device_preference="auto")
    model = backend.load()
    assert model.device == "cpu"


