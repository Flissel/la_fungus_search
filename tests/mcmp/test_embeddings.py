from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests


ROOT = Path(__file__).resolve().parents[2]


def _response(payload: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_embedding_client_uses_service_batch_contract_and_preserves_order(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    session = MagicMock()
    session.post.return_value = _response({"vectors": [[0.1] * 3072, [0.2] * 3072]})
    client = embeddings.EmbeddingServiceClient(
        base_url="http://embedding-service:8080/", session=session, max_retries=0
    )

    assert client.encode(["first", "second"]) == [[0.1] * 3072, [0.2] * 3072]
    session.post.assert_called_once_with(
        "http://embedding-service:8080/embed/batch",
        json={"texts": ["first", "second"]},
        timeout=30.0,
    )


def test_embedding_client_retries_only_transient_service_unavailability(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    session = MagicMock()
    session.post.side_effect = [
        requests.exceptions.ConnectionError("service unavailable"),
        _response({"vectors": [[0.1] * 3072]}),
    ]
    sleeps: list[float] = []
    monkeypatch.setattr(embeddings.time, "sleep", sleeps.append)
    client = embeddings.EmbeddingServiceClient(
        base_url="http://embedding-service.test", session=session, max_retries=2, retry_backoff=0.25
    )

    assert client.encode(["retry me"]) == [[0.1] * 3072]
    assert session.post.call_count == 2
    assert sleeps == [0.25]


def test_embedding_client_fails_closed_after_bounded_transient_retries(monkeypatch):
    from embeddinggemma.mcmp import embeddings

    session = MagicMock()
    session.post.side_effect = requests.exceptions.Timeout("service unavailable")
    monkeypatch.setattr(embeddings.time, "sleep", lambda _seconds: None)
    client = embeddings.EmbeddingServiceClient(
        base_url="http://embedding-service.test", session=session, max_retries=1
    )

    with pytest.raises(embeddings.EmbeddingServiceUnavailable, match="after 2 attempts"):
        client.encode(["must not fall back"])

    assert session.post.call_count == 2


def test_embedding_client_does_not_retry_non_transient_http_errors():
    from embeddinggemma.mcmp import embeddings

    response = _response({}, status_code=400)
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=response)
    session = MagicMock()
    session.post.return_value = response
    client = embeddings.EmbeddingServiceClient(
        base_url="http://embedding-service.test", session=session, max_retries=2
    )

    with pytest.raises(embeddings.EmbeddingServiceError, match="status 400"):
        client.encode(["bad request"])

    assert session.post.call_count == 1


def test_embedding_client_rejects_malformed_or_dimension_mismatched_service_response():
    from embeddinggemma.mcmp import embeddings

    session = MagicMock()
    session.post.return_value = _response({"vectors": [[0.1] * 384]})
    client = embeddings.EmbeddingServiceClient(
        base_url="http://embedding-service.test", session=session, max_retries=0
    )

    with pytest.raises(embeddings.EmbeddingServiceError, match="3072.*384"):
        client.encode(["wrong dimension"])


def test_active_embedding_paths_have_no_local_model_fallback_or_provider_bypass():
    source = (ROOT / "src" / "embeddinggemma" / "mcmp" / "embeddings.py").read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(project["project"]["dependencies"]).lower()
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()

    assert "embedding-service" in source
    assert "sentence_transformers" not in source
    assert "get_embedding_model" not in source
    assert "openai" not in source.lower()
    for local_embedding_dependency in ("sentence-transformers", "torch", "transformers"):
        assert local_embedding_dependency not in dependencies
    for local_embedding_dependency in ("sentence-transformers", "torch", "transformers", "ollama"):
        assert local_embedding_dependency not in requirements


def test_importing_mcp_before_a_query_does_not_import_local_embedding_stacks(monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    for module_name in tuple(sys.modules):
        if module_name == "mcp_server" or module_name.startswith("embeddinggemma"):
            sys.modules.pop(module_name, None)
    monkeypatch.syspath_prepend(str(ROOT))

    importlib.import_module("mcp_server")

    forbidden_prefixes = ("torch", "transformers", "sentence_transformers", "ollama")
    assert not any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in forbidden_prefixes
        for module_name in sys.modules
    )
