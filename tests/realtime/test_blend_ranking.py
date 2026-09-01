"""The blended ranking must actually use the signals it is written to blend.

`SnapshotStreamer._compute_blended_topk` scores

    alpha*cosine + beta*visit_norm + gamma*trail_degree
    + delta*llm_vote + epsilon*len_prior + 0.05*boost

but it reads `it['score']` for the cosine and `it['id']` for the document. The
items it is called with come straight from `MCPMRetriever.search()`, which
returns only `content`, `metadata` and `relevance_score` -- no `score`, no `id`.
Every lookup therefore misses: the cosine reads 0.0, the document id resolves to
-1, and visits, trail degree, the LLM vote and the boost all resolve against a
document that does not exist. What survives is `epsilon * len_prior`, so the
"blended" ranking is a ranking by document length.

Nothing raises. The surrounding try/except reports no failure because there is no
failure -- the arithmetic is correct and the inputs are all zero.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def _module(name, monkeypatch, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_server(monkeypatch):
    class Retriever:
        pass

    _module("embeddinggemma.mcmp_rag", monkeypatch, MCPMRetriever=Retriever)
    _module(
        "embeddinggemma.ui.corpus",
        monkeypatch,
        collect_codebase_chunks=lambda *_a, **_k: [],
        list_code_files=lambda *_a, **_k: [],
    )
    _module("embeddinggemma.ui.queries", monkeypatch, dedup_multi_queries=lambda values: values)
    _module("embeddinggemma.ui.reports", monkeypatch, merge_reports_to_summary=lambda values: values)
    _module(
        "embeddinggemma.llm.prompts",
        monkeypatch,
        get_report_instructions=lambda *_a: "",
        build_report_prompt=lambda *_a: "",
        build_judge_prompt=lambda *_a: "",
    )
    _module("embeddinggemma.prompts", monkeypatch, _default_instructions=lambda *_a: "", report_schema_hint=lambda: "")
    for name in ("deep", "structure", "exploratory", "summary", "repair", "steering"):
        _module(f"embeddinggemma.modeprompts.{name}", monkeypatch, instructions=lambda: "")
    _module(
        "embeddinggemma.rag.generation",
        monkeypatch,
        generate_text=lambda **_k: "",
        generate_judge_text=lambda **_k: "",
    )
    sys.modules.pop("embeddinggemma.realtime.server", None)
    return importlib.import_module("embeddinggemma.realtime.server")


class _Document:
    def __init__(self, doc_id: int, content: str, visit_count: int) -> None:
        self.id = doc_id
        self.content = content
        self.visit_count = visit_count
        self.relevance_score = 0.0


class _Retriever:
    def __init__(self, documents):
        self.documents = documents


class _Streamer:
    """The attribute surface `_compute_blended_topk` reads, and nothing else."""

    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    delta = 0.0
    epsilon = 1.0
    top_k = 2
    import_only_penalty = 0.0
    min_content_chars = 0

    def __init__(self, documents):
        self.retr = _Retriever(documents)
        self._llm_vote: dict[int, int] = {}
        self._doc_boost: dict[int, float] = {}
        self._documents = documents

    def _trail_degree_map(self):
        return {document.id: 4 for document in self._documents}

    def _doc_by_id(self, doc_id):
        for document in self._documents:
            if document.id == doc_id:
                return document
        return None

    def _is_import_only(self, _content):
        return False


# A short, highly similar, heavily visited document against a long, dissimilar,
# never-visited one. Every term in the formula except `len_prior` favours the
# first; only length favours the second.
RELEVANT = "def parse_config(path):\n    return json.loads(path.read_text())"
PADDING = "# unrelated boilerplate\n" * 60


@pytest.fixture()
def server(monkeypatch):
    return _load_server(monkeypatch)


def _search_results():
    """Exactly the shape `MCPMRetriever.search_direct` returns."""
    return [
        {"content": RELEVANT, "metadata": {}, "relevance_score": 0.93},
        {"content": PADDING, "metadata": {}, "relevance_score": 0.11},
    ]


def test_blend_ranks_by_similarity_not_by_length(server):
    documents = [_Document(0, RELEVANT, visit_count=40), _Document(1, PADDING, visit_count=0)]
    streamer = _Streamer(documents)

    blended = server.SnapshotStreamer._compute_blended_topk(streamer, _search_results())

    assert [item["content"] for item in blended][0] == RELEVANT, (
        "the long, dissimilar, unvisited document outranked the short relevant one, "
        "which is what happens when every term but len_prior reads as zero"
    )


def test_blend_actually_reads_the_signals(server):
    """The scores must move when the signals move, not only when length does."""
    hot = [_Document(0, RELEVANT, visit_count=40), _Document(1, PADDING, visit_count=0)]
    cold = [_Document(0, RELEVANT, visit_count=0), _Document(1, PADDING, visit_count=0)]

    scored_hot = server.SnapshotStreamer._compute_blended_topk(_Streamer(hot), _search_results())
    scored_cold = server.SnapshotStreamer._compute_blended_topk(_Streamer(cold), _search_results())

    hot_score = next(i["blended_score"] for i in scored_hot if i["content"] == RELEVANT)
    cold_score = next(i["blended_score"] for i in scored_cold if i["content"] == RELEVANT)
    assert hot_score > cold_score, (
        "visit_count changed from 0 to 40 and the blended score did not move, so the "
        "visit term is not reaching the document"
    )
