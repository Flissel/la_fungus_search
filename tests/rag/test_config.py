def test_rag_settings_defaults():
    from embeddinggemma.rag.config import RagSettings

    cfg = RagSettings()
    assert isinstance(cfg.qdrant_url, str)
    assert cfg.collection_name
    assert isinstance(cfg.use_ollama, bool)


