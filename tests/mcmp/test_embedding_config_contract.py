from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_embedding_loader_requires_explicit_reachable_service_url(monkeypatch):
    from embeddinggemma.mcmp.embeddings import EmbeddingServiceError, load_embedding_backend

    monkeypatch.delenv("EMBEDDING_SERVICE_URL", raising=False)

    with pytest.raises(EmbeddingServiceError, match="EMBEDDING_SERVICE_URL is required"):
        load_embedding_backend()


def test_embedding_loader_honors_service_url_override_and_strips_trailing_slash(monkeypatch):
    from embeddinggemma.mcmp.embeddings import load_embedding_backend

    monkeypatch.setenv("EMBEDDING_SERVICE_URL", "http://localhost:9000/")
    client, _dimension = load_embedding_backend()

    assert client._base_url == "http://localhost:9000"


def test_env_example_keeps_llm_config_and_requires_service_url_without_docker_default():
    example = (ROOT / "_.env.example").read_text(encoding="utf-8")

    assert "VIBEMIND_CONFIG_DIR=/absolute/path/to/vibemind-config" in example
    assert "EMBEDDING_SERVICE_URL=" in example
    assert "EMBEDDING_SERVICE_URL=http://embedding-service:8080" not in example.splitlines()
    assert "Swarm-only" in example
