import pytest


def test_load_embedding_model_uses_fixed_fungus_search_role(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    expected_model = object()
    calls = []

    def fake_get_embedding_model(role):
        calls.append(role)
        return expected_model

    monkeypatch.setattr(embeddings, "get_embedding_model", fake_get_embedding_model)

    assert embeddings.load_embedding_model() is expected_model
    assert calls == ["fungus_search"]


def test_load_embedding_model_propagates_shared_gateway_failure(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    def unavailable(_role):
        raise RuntimeError("OpenFang unreachable")

    monkeypatch.setattr(embeddings, "get_embedding_model", unavailable)

    with pytest.raises(RuntimeError, match="OpenFang unreachable"):
        embeddings.load_embedding_model()
