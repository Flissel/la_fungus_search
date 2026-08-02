def test_embedding_loader_uses_shared_service_default_without_provider_config(monkeypatch):
    from embeddinggemma.mcmp.embeddings import (
        EMBEDDING_DIMENSION,
        EMBEDDING_SERVICE_DEFAULT_URL,
        load_embedding_backend,
    )

    monkeypatch.delenv("EMBEDDING_SERVICE_URL", raising=False)
    client, dimension = load_embedding_backend()

    assert client._base_url == EMBEDDING_SERVICE_DEFAULT_URL
    assert dimension == EMBEDDING_DIMENSION == 3072


def test_embedding_loader_honors_service_url_override_and_strips_trailing_slash(monkeypatch):
    from embeddinggemma.mcmp.embeddings import load_embedding_backend

    monkeypatch.setenv("EMBEDDING_SERVICE_URL", "http://localhost:9000/")
    client, _dimension = load_embedding_backend()

    assert client._base_url == "http://localhost:9000"
