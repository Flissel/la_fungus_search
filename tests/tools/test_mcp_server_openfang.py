import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _install_fastmcp_stub(monkeypatch):
    mcp_package = ModuleType("mcp")
    mcp_package.__path__ = []
    server_package = ModuleType("mcp.server")
    server_package.__path__ = []
    fastmcp_module = ModuleType("mcp.server.fastmcp")

    class FastMCP:
        def __init__(self, *_args, **_kwargs):
            pass

        def tool(self):
            return lambda function: function

    fastmcp_module.FastMCP = FastMCP
    mcp_package.server = server_package
    server_package.fastmcp = fastmcp_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_package)
    monkeypatch.setitem(sys.modules, "mcp.server", server_package)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)


def _forbid_popen(*_args, **_kwargs):
    raise AssertionError("subprocess.Popen requires an explicit hermetic test stub")


def _semantic_retriever(content):
    document = type("Document", (), {"id": 0, "content": content})()

    class FakeRetriever:
        documents = [document]

        def search_direct(self, _query, *, top_k):
            return {
                "results": [
                    {"content": document.content, "relevance_score": 0.5}
                ]
            }

    return FakeRetriever()


def _load_mcp_server(
    monkeypatch, *, popen_stub=None, bm25_load=None, bm25_cache_exists=None
):
    monkeypatch.setattr(subprocess, "Popen", popen_stub or _forbid_popen)
    calls = []
    if bm25_cache_exists is not None:
        original_exists = os.path.exists
        monkeypatch.setattr(
            os.path,
            "exists",
            lambda path: bm25_cache_exists
            if str(path).endswith("bm25.npz")
            else original_exists(path),
        )
    bm25_module = ModuleType("embeddinggemma.bm25_lite")

    class BM25Lite:
        @staticmethod
        def load(path):
            if bm25_load is None:
                raise AssertionError("BM25Lite.load requires an explicit hermetic test stub")
            return bm25_load(path)

    bm25_module.BM25Lite = BM25Lite
    monkeypatch.setitem(sys.modules, "embeddinggemma.bm25_lite", bm25_module)
    generation = ModuleType("embeddinggemma.rag.generation")

    def generate_text(**kwargs):
        calls.append(("summary", kwargs))
        return "summary"

    def generate_judge_text(**kwargs):
        calls.append(("judge", kwargs))
        return "judgement"

    generation.generate_text = generate_text
    generation.generate_judge_text = generate_judge_text
    monkeypatch.setitem(sys.modules, "embeddinggemma.rag.generation", generation)
    _install_fastmcp_stub(monkeypatch)
    sys.modules.pop("mcp_server", None)
    return importlib.import_module("mcp_server"), calls


def test_mcp_llm_helpers_use_fixed_openfang_paths(monkeypatch):
    server, calls = _load_mcp_server(monkeypatch)

    assert server._generate_summary("Expand this") == "summary"
    assert server._generate_judge("Judge this") == "judgement"
    assert calls == [
        ("summary", {"prompt": "Expand this"}),
        ("judge", {"prompt": "Judge this"}),
    ]


def test_mcp_active_llm_call_sites_do_not_use_direct_provider_helper():
    source = (ROOT / "mcp_server.py").read_text(encoding="utf-8")

    assert "def _ollama_generate" not in source
    assert source.count("asyncio.to_thread(_generate_summary,") == 2
    assert source.count("asyncio.to_thread(_generate_judge,") == 2
    assert "raw = _generate_judge(prompt)" in source


def test_retriever_load_is_lazy_and_starts_once_on_first_query(monkeypatch):
    class DeferredThread:
        starts = 0
        run_targets = False

        def __init__(self, *, target, daemon=None, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name
            self.started = False

        def start(self):
            type(self).starts += 1
            self.started = True
            if type(self).run_targets:
                self.target()

        def is_alive(self):
            return self.started

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    server, _calls = _load_mcp_server(monkeypatch)

    assert DeferredThread.starts == 0, "MCP import must not start the heavy retriever"
    assert server._bg_thread is None

    updater_boundaries = []
    monkeypatch.setattr(
        server,
        "_start_incremental_updater_once",
        lambda: updater_boundaries.append("start"),
    )
    monkeypatch.setattr(server, "_background_load", server._ready_event.set)
    DeferredThread.run_targets = True

    assert server._ensure_ready(timeout=0.1) is True
    assert DeferredThread.starts == 1
    assert server._ensure_ready(timeout=0.1) is True
    assert DeferredThread.starts == 1
    assert updater_boundaries == ["start", "start"]


def test_index_stats_reports_lazy_state_without_starting_loader(monkeypatch):
    class NoStartThread:
        starts = 0

        def __init__(self, *, target, daemon=None, name=None):
            self.target = target

        def start(self):
            type(self).starts += 1

        def is_alive(self):
            return False

    monkeypatch.setattr(threading, "Thread", NoStartThread)
    server, _calls = _load_mcp_server(monkeypatch)

    result = asyncio.run(server.fungus_index_stats())

    assert "not loaded yet" in result.lower()
    assert NoStartThread.starts == 0


def test_bm25_load_waits_for_a_hybrid_search_not_import_or_index_stats(monkeypatch):
    loads = []
    server, _calls = _load_mcp_server(
        monkeypatch,
        bm25_load=lambda path: loads.append(path),
        bm25_cache_exists=True,
    )

    assert loads == []
    assert "not loaded yet" in asyncio.run(server.fungus_index_stats()).lower()
    assert loads == []


def test_parallel_hybrid_searches_hydrate_bm25_once(monkeypatch):
    load_started = threading.Event()
    release_load = threading.Event()
    loads = []

    class FakeBM25:
        doc_freq = {}
        N = 1

        def score(self, _query, *, candidate_ids):
            return server._np.asarray(
                [float(candidate_id) for candidate_id in candidate_ids]
            )

    def fake_load(path):
        loads.append(path)
        load_started.set()
        assert release_load.wait(timeout=2)
        return FakeBM25()

    server, _calls = _load_mcp_server(monkeypatch, bm25_load=fake_load)
    monkeypatch.setattr(server.os.path, "exists", lambda path: path == server._bm25_path)
    monkeypatch.setattr(server, "_ensure_ready", lambda: True)
    server._retriever = _semantic_retriever(
        "# file: example.py | lines: 1-1\ndef search(): pass"
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(server._sync_search, "hybrid", 1, 0.65)
        assert load_started.wait(timeout=2)
        second = executor.submit(server._sync_search, "hybrid", 1, 0.65)
        release_load.set()
        first_results = first.result(timeout=2)["results"]
        second_results = second.result(timeout=2)["results"]

    assert loads == [server._bm25_path]
    assert "_bm25_score" in first_results[0]
    assert "_bm25_score" in second_results[0]


@pytest.mark.parametrize(
    ("cache_exists", "load_error", "expected_loads"),
    [(False, False, 0), (True, True, 1)],
    ids=["missing-cache", "load-error"],
)
def test_bm25_hydration_failure_is_bounded_and_falls_back_to_semantic(
    monkeypatch, cache_exists, load_error, expected_loads
):
    loads = []

    def fake_load(path):
        loads.append(path)
        if load_error:
            raise OSError("corrupt cache")
        raise AssertionError("missing cache must not call BM25Lite.load")

    server, _calls = _load_mcp_server(monkeypatch, bm25_load=fake_load)
    monkeypatch.setattr(server.os.path, "exists", lambda _path: cache_exists)
    monkeypatch.setattr(server, "_ensure_ready", lambda: True)
    server._retriever = _semantic_retriever(
        "# file: fallback.py | lines: 1-1\ndef fallback(): pass"
    )

    assert server._sync_search("fallback", 1, 0.65)["results"]
    assert server._sync_search("fallback", 1, 0.65)["results"]
    assert len(loads) == expected_loads


def test_first_retriever_query_spawns_incremental_updater_once(monkeypatch):
    """A process starts the updater only for the first retriever-requiring call."""
    spawned = []

    class SuccessfulWrapper:
        def __init__(self):
            self.wait_timeouts = []

        def wait(self, *, timeout):
            self.wait_timeouts.append(timeout)
            return 0

    wrapper = SuccessfulWrapper()

    class FakeRetriever:
        documents = [object()]

        def search_direct(self, *_args, **_kwargs):
            return {"results": []}

    def fake_popen(args, **kwargs):
        spawned.append((args, kwargs))
        return wrapper

    server, _calls = _load_mcp_server(monkeypatch, popen_stub=fake_popen)

    assert hasattr(server, "_start_incremental_updater_once"), (
        "retriever-requiring queries must own a process-local lazy updater spawn"
    )
    monkeypatch.setattr(sys, "executable", "isolated-python")
    monkeypatch.setattr(server, "_background_load", server._ready_event.set)
    server._retriever = FakeRetriever()
    server._bm25 = None

    assert "not loaded yet" in asyncio.run(server.fungus_index_stats()).lower()
    assert spawned == []

    start = threading.Barrier(3)

    def enter_retriever_boundary():
        start.wait(timeout=2)
        return server._ensure_ready(timeout=0.1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(enter_retriever_boundary) for _value in range(2)]
        start.wait(timeout=2)
        assert [future.result(timeout=2) for future in futures] == [True, True]

    assert asyncio.run(server.fungus_search("lazy updater")) == "No results for: lazy updater"

    assert len(spawned) == 1
    args, kwargs = spawned[0]
    assert args == [
        "isolated-python",
        str(ROOT / "incremental_updater.py"),
        "--background",
    ]
    assert kwargs["cwd"] == str(ROOT)
    assert kwargs["shell"] is False
    assert wrapper.wait_timeouts == [server._UPDATER_WRAPPER_TIMEOUT_S]


@pytest.mark.parametrize("failure_mode", ["nonzero", "timeout", "oserror"])
def test_retriever_query_fails_closed_when_updater_wrapper_fails(
    monkeypatch, failure_mode
):
    class FailedWrapper:
        def __init__(self):
            self.wait_timeouts = []
            self.terminated = False

        def wait(self, *, timeout):
            self.wait_timeouts.append(timeout)
            if failure_mode == "timeout" and len(self.wait_timeouts) == 1:
                raise subprocess.TimeoutExpired("incremental-updater-wrapper", timeout)
            return 9 if failure_mode == "nonzero" else 0

        def terminate(self):
            self.terminated = True

    wrapper = FailedWrapper()
    attempts = []

    def fail_popen(*args, **kwargs):
        attempts.append((args, kwargs))
        if failure_mode == "oserror":
            raise OSError("blocked")
        return wrapper

    server, _calls = _load_mcp_server(monkeypatch, popen_stub=fail_popen)

    assert hasattr(server, "_start_incremental_updater_once"), (
        "updater spawn failures must be surfaced before retriever loading"
    )

    with pytest.raises(RuntimeError, match="incremental updater"):
        asyncio.run(server.fungus_search("must not pretend to search"))
    with pytest.raises(RuntimeError, match="incremental updater"):
        asyncio.run(server.fungus_search("still fail closed"))

    assert len(attempts) == 1
    assert server._bg_thread is None
    if failure_mode == "timeout":
        assert wrapper.terminated is True
        assert wrapper.wait_timeouts == [
            server._UPDATER_WRAPPER_TIMEOUT_S,
            server._UPDATER_WRAPPER_TERMINATE_TIMEOUT_S,
        ]
