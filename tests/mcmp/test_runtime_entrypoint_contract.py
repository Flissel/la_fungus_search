from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_active_retriever_entrypoints_do_not_override_embedding_contract():
    for relative_path in (
        "mcp_server.py",
        "incremental_updater.py",
        "build_index.py",
        "build_optimized.py",
        "build_brain_focused.py",
        "build_vibemind_cpu.py",
        "build_vibemind_index.py",
        "test_persistent.py",
        "test_search.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "embedding_model_name=" not in source
        assert "device_mode=" not in source


def test_direct_index_builders_do_not_bypass_openfang_embeddings():
    for relative_path in ("build_direct_gpu.py", "build_multivec_index.py"):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "SentenceTransformer" not in source
        assert "FUNGUS_EMBED_MODEL" not in source


def test_service_starters_require_canonical_vibemind_config_dir():
    for relative_path in ("run-realtime.ps1", "run_http_server.cmd"):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "VIBEMIND_CONFIG_DIR" in source
        assert "llm_config.yml" in source
