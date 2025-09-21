import json
from types import SimpleNamespace


def test_generate_with_ollama_success(monkeypatch):
    from embeddinggemma.rag import generation

    class DummyResponse:
        def __init__(self):
            self._json = {"response": "ok"}

        def raise_for_status(self):
            return None

        def json(self):
            return self._json

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - shadow json param ok here
        return DummyResponse()

    monkeypatch.setattr(generation, "requests", SimpleNamespace(post=fake_post))
    out = generation.generate_with_ollama("hi", model="m", host="http://h")
    assert out == "ok"


def test_generate_with_ollama_error(monkeypatch):
    from embeddinggemma.rag import generation

    class DummyResponse:
        def raise_for_status(self):
            raise RuntimeError("bad")

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        return DummyResponse()

    monkeypatch.setattr(generation, "requests", SimpleNamespace(post=fake_post))
    out = generation.generate_with_ollama("hi", model="m", host="http://h")
    assert out.startswith("[LLM error]")


