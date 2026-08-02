import pytest


def _canonical_config() -> str:
    return """
keys:
  openfang: ${OPENFANG_API_KEY}
providers:
  openfang:
    type: openai
    base_url: ${OPENFANG_URL}/v1
    key_ref: openfang
    fail_closed: true
default:
  provider: openfang
  model: gpt-4.1
  temperature: 0
roles:
  fungus_summary:
    provider: openfang
    model: gpt-4.1
    temperature: 0
  fungus_judge:
    provider: openfang
    model: gpt-4.1
    temperature: 0
embeddings:
  fungus_search:
    driver: openai
    provider: openfang
    model: text-embedding-3-large
    dim: 3072
"""


@pytest.fixture(autouse=True)
def clear_shared_config_cache():
    from vibemind_shared import llm_client

    llm_client._load_config.cache_clear()
    yield
    llm_client._load_config.cache_clear()


def test_embedding_loader_requires_canonical_config_dir_when_no_repo_config_exists(monkeypatch, tmp_path):
    from embeddinggemma.mcmp.embeddings import load_embedding_backend

    monkeypatch.delenv("VIBEMIND_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="llm_config.yml not found"):
        load_embedding_backend()


def test_canonical_config_resolves_openfang_roles_and_embedding_contract(monkeypatch, tmp_path):
    from embeddinggemma.mcmp.embeddings import load_embedding_backend
    from vibemind_shared import get_embedding_config, get_provider_info

    (tmp_path / "llm_config.yml").write_text(_canonical_config(), encoding="utf-8")
    monkeypatch.setenv("VIBEMIND_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("OPENFANG_URL", "http://127.0.0.1:4200")
    monkeypatch.setenv("OPENFANG_API_KEY", "test-openfang-key")

    _model, dimension = load_embedding_backend()

    assert dimension == 3072
    assert get_embedding_config("fungus_search") == {
        "driver": "openai",
        "provider": "openfang",
        "model": "text-embedding-3-large",
        "dim": 3072,
    }
    assert get_provider_info("fungus_summary")["provider"] == "openfang"
    assert get_provider_info("fungus_judge")["provider"] == "openfang"
