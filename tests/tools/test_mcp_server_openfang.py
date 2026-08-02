import asyncio
import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType


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


def _load_mcp_server(monkeypatch):
    calls = []
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

    monkeypatch.setattr(server, "_background_load", server._ready_event.set)
    DeferredThread.run_targets = True

    assert server._ensure_ready(timeout=0.1) is True
    assert DeferredThread.starts == 1
    assert server._ensure_ready(timeout=0.1) is True
    assert DeferredThread.starts == 1


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
