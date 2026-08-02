from pathlib import Path
import tomllib

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


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("", "is required"),
        ("   ", "is required"),
        ("embedding-service:8080", "absolute HTTP\\(S\\) URL"),
        ("ftp://embedding-service:8080", "absolute HTTP\\(S\\) URL"),
        ("http:///embed", "absolute HTTP\\(S\\) URL"),
        ("https://embed ding.test", "absolute HTTP\\(S\\) URL"),
    ],
)
def test_embedding_loader_rejects_non_absolute_http_service_urls(monkeypatch, url, message):
    from embeddinggemma.mcmp.embeddings import EmbeddingServiceError, load_embedding_backend

    monkeypatch.setenv("EMBEDDING_SERVICE_URL", url)

    with pytest.raises(EmbeddingServiceError, match=message):
        load_embedding_backend()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://embedding-service:8080/", "http://embedding-service:8080"),
        ("https://host.example.test/path///", "https://host.example.test/path"),
    ],
)
def test_embedding_loader_accepts_and_normalizes_absolute_http_urls(monkeypatch, url, expected):
    from embeddinggemma.mcmp.embeddings import load_embedding_backend

    monkeypatch.setenv("EMBEDDING_SERVICE_URL", url)

    client, _dimension = load_embedding_backend()

    assert client._base_url == expected


def test_env_example_keeps_llm_config_and_requires_service_url_without_docker_default():
    example = (ROOT / "_.env.example").read_text(encoding="utf-8")

    assert "VIBEMIND_CONFIG_DIR=/absolute/path/to/vibemind-config" in example
    assert "EMBEDDING_SERVICE_URL=" in example
    assert "EMBEDDING_SERVICE_URL=http://embedding-service:8080" not in example.splitlines()
    assert "Swarm-only" in example


def test_reranker_is_a_declared_optional_extra_while_default_stays_heavy_free():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    extra = project["project"]["optional-dependencies"]["reranker"]
    docs = (ROOT / "docs" / "ENV.md").read_text(encoding="utf-8")

    assert all("sentence-transformers" not in item and "torch" not in item for item in dependencies)
    assert any(item.startswith("sentence-transformers") for item in extra)
    assert any(item.startswith("torch") for item in extra)
    assert ".[reranker]" in docs
