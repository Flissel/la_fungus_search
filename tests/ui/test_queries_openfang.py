import importlib
import sys
from pathlib import Path
from types import ModuleType


SRC = Path(__file__).resolve().parents[2] / "src"


def _load_queries(monkeypatch):
    generation = ModuleType("embeddinggemma.rag.generation")
    generation.generate_text = lambda **_kwargs: ""
    monkeypatch.setitem(sys.modules, "embeddinggemma.rag.generation", generation)
    sys.path.insert(0, str(SRC))
    sys.modules.pop("embeddinggemma.ui.queries", None)
    return importlib.import_module("embeddinggemma.ui.queries")


def test_multi_query_generation_uses_fixed_openfang_summary_path(monkeypatch):
    queries = _load_queries(monkeypatch)
    calls = []

    def generate_text(**kwargs):
        calls.append(kwargs)
        return "first repository query\nsecond repository query"

    monkeypatch.setattr(queries, "generate_text", generate_text, raising=False)

    result = queries.generate_multi_queries_from_llm("find search routing", num_queries=2)

    assert result == ["first repository query", "second repository query"]
    assert calls and calls[0]["prompt"].startswith("Base query: find search routing")
    assert "System:" not in calls[0]["prompt"]
    assert calls[0]["system"].startswith("You reformulate a single repository question")
    assert not hasattr(queries, "_ollama_generate")
