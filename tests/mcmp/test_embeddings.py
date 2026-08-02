import pytest


def _canonical_embedding_config():
    return {
        "driver": "openai",
        "provider": "openfang",
        "model": "text-embedding-3-large",
        "dim": 3072,
    }


def test_load_embedding_backend_uses_exact_fungus_search_contract(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    expected_model = object()
    config = _canonical_embedding_config()
    calls = {"config": [], "model": []}

    def fake_get_config():
        return {"embeddings": {"fungus_search": config}}

    def fake_get_embedding_config(role):
        calls["config"].append(role)
        return config

    def fake_get_embedding_model(role):
        calls["model"].append(role)
        return expected_model

    monkeypatch.setattr(embeddings, "get_config", fake_get_config)
    monkeypatch.setattr(embeddings, "get_embedding_config", fake_get_embedding_config)
    monkeypatch.setattr(embeddings, "get_embedding_model", fake_get_embedding_model)

    model, dimension = embeddings.load_embedding_backend()

    assert model is expected_model
    assert dimension == 3072
    assert calls == {"config": ["fungus_search"], "model": ["fungus_search"]}


def test_load_embedding_backend_rejects_missing_fungus_search_instead_of_default_fallback(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    fallback = _canonical_embedding_config()
    calls = []

    monkeypatch.setattr(embeddings, "get_config", lambda: {"embeddings": {"default": fallback}})
    monkeypatch.setattr(embeddings, "get_embedding_config", lambda _role: fallback)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda role: calls.append(role))

    with pytest.raises(RuntimeError, match=r"requires explicit embeddings\.fungus_search"):
        embeddings.load_embedding_backend()

    assert calls == []


def test_load_embedding_backend_rejects_direct_provider_drift(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    direct_config = {**_canonical_embedding_config(), "provider": "openai"}
    calls = []

    monkeypatch.setattr(embeddings, "get_config", lambda: {"embeddings": {"fungus_search": direct_config}})
    monkeypatch.setattr(embeddings, "get_embedding_config", lambda _role: direct_config)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda role: calls.append(role))

    with pytest.raises(RuntimeError, match="expected provider='openfang'.*got 'openai'"):
        embeddings.load_embedding_backend()

    assert calls == []


def test_load_embedding_backend_rejects_extra_contract_fields(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    config = {**_canonical_embedding_config(), "fallback_provider": "openai"}
    calls = []

    monkeypatch.setattr(embeddings, "get_config", lambda: {"embeddings": {"fungus_search": config}})
    monkeypatch.setattr(embeddings, "get_embedding_config", lambda _role: config)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda role: calls.append(role))

    with pytest.raises(RuntimeError, match="contract fields must be exactly"):
        embeddings.load_embedding_backend()

    assert calls == []


def test_load_embedding_model_propagates_shared_gateway_failure(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    def unavailable(_role):
        raise RuntimeError("OpenFang unreachable")

    config = _canonical_embedding_config()
    monkeypatch.setattr(embeddings, "get_config", lambda: {"embeddings": {"fungus_search": config}})
    monkeypatch.setattr(embeddings, "get_embedding_config", lambda _role: config)
    monkeypatch.setattr(embeddings, "get_embedding_model", unavailable)

    with pytest.raises(RuntimeError, match="OpenFang unreachable"):
        embeddings.load_embedding_backend()
