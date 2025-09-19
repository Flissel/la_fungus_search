#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
GPU-powered Combined Retrieval Orchestrator

Combines:
- Line-based retrieval tool (for line-sliced documents)
- Fungus (MCPMRetriever) exploration search (Physarum/MCPM-inspired)
- Enterprise RAG (optional, if deps available)
- LangChain Tooling Agent with Ollama to plan, enhance queries, call tools, score, and retry

Usage examples:
  python -m embeddinggemma.agent_fungus_rag --docs-file ../../Elevenlabs_API_Codesheet_final.md --md-codeblocks --query "cURL Beispiel: Text-zu-Sprache (TTS)" --ollama-model llama3.1 --device auto

Notes:
- Requires a running Ollama server for the agent (default http://localhost:11434)
- Enterprise RAG tools activate only if dependencies are installed
"""

import os
import sys
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import heapq
import numpy as np
from urllib.parse import urlparse, urlunparse

# Optional heavy deps guarded
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from langchain.agents import initialize_agent
    from langchain.agents.agent_types import AgentType
    from langchain_core.tools import Tool
    try:
        # Preferred modern import (removes deprecation warning)
        from langchain_ollama import ChatOllama  # type: ignore
    except Exception:
        # Fallback (deprecated warnings may appear)
        from langchain_community.chat_models import ChatOllama  # type: ignore
    LANGCHAIN_OK = True
except Exception:
    # Fallback stub for Tool to allow script to run without langchain
    class Tool:  # type: ignore
        def __init__(self, name: str, description: str, func):
            self.name = name
            self.description = description
            self.func = func
    LANGCHAIN_OK = False

# Ensure project root is importable (for mcmp_rag)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
try:
    # MCPM retriever
    from mcmp_rag import MCPMRetriever
except Exception as e:
    print(f"[ERROR] Cannot import MCPMRetriever: {e}")
    MCPMRetriever = None  # type: ignore

# Enterprise RAG (optional)
ENTERPRISE_OK = False
try:
    from embeddinggemma.enterprise_rag import EnterpriseCodeRAG  # type: ignore
    ENTERPRISE_OK = True
except Exception:
    ENTERPRISE_OK = False


# ----------------------------
# Utilities
# ----------------------------

def detect_device(preferred: str = "auto") -> str:
    mode = (preferred or "auto").lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return "cuda"
    if torch is not None:
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


def _json_safe(obj: Any) -> Any:
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None  # type: ignore
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.integer,)):
            return int(obj)
    return obj


def _normalize_ollama_host(value: Optional[str]) -> str:
    default = "http://127.0.0.1:11434"
    if not value:
        return default
    val = value.strip()
    # Ensure scheme
    if not val.startswith("http://") and not val.startswith("https://"):
        val = "http://" + val
    parsed = urlparse(val)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    scheme = parsed.scheme or "http"
    if port is None:
        port = 11434
    netloc = f"{host}:{port}"
    return urlunparse((scheme, netloc, "", "", "", ""))

def load_documents_from_file(file_path: str) -> List[str]:
    texts: List[str] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def load_codeblocks_from_markdown(file_path: str, include_text: bool = False) -> List[str]:
    content = open(file_path, "r", encoding="utf-8", errors="ignore").read()
    # Robust: support CRLF, optional spaces after language, and require closing fence on its own line
    code_pattern = re.compile(r"```\s*([^\r\n]*)\r?\n([\s\S]*?)\r?\n```", re.MULTILINE)
    documents: List[str] = []
    last_end = 0
    for match in code_pattern.finditer(content):
        lang = (match.group(1) or "").strip()
        code = match.group(2)
        # Preserve code as-is; trim only trailing CRLF whitespace
        code = re.sub(r"\s+$", "", code)
        documents.append(f"```{lang}\n{code}\n```")
        if include_text:
            prefix = content[last_end:match.start()].strip()
            if prefix:
                for para in re.split(r"\n\n+", prefix):
                    para = para.strip()
                    if para:
                        documents.append(para)
        last_end = match.end()
    if include_text:
        tail = content[last_end:].strip()
        if tail:
            for para in re.split(r"\n\n+", tail):
                para = para.strip()
                if para:
                    documents.append(para)
    return documents or load_documents_from_file(file_path)


# ----------------------------
# Tools Implementations
# ----------------------------

@dataclass
class LineIndex:
    lines: List[str]

    @classmethod
    def from_file(cls, path: str) -> "LineIndex":
        return cls(load_documents_from_file(path))

    def search(self, query: str, window: int = 2, max_hits: int = 10) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        q = query.lower()
        for idx, line in enumerate(self.lines):
            if q in line.lower():
                start = max(0, idx - window)
                end = min(len(self.lines), idx + window + 1)
                snippet = "\n".join(self.lines[start:end])
                results.append({"line": idx + 1, "start": start + 1, "end": end, "snippet": snippet})
                if len(results) >= max_hits:
                    break
        return results


@dataclass
class CollageIndex:
    """Multi-window chunk index over a list of units (lines or blocks).

    Produces non-overlapping chunks per window size with metadata.
    """
    units: List[str]

    def _iter_chunks(self, window_size: int):
        total = len(self.units)
        i = 0
        while i < total:
            chunk_units = self.units[i:i + window_size]
            if not chunk_units:
                break
            start_line = i + 1
            end_line = i + len(chunk_units)
            content = "\n".join(chunk_units)
            yield {
                "window_size": window_size,
                "start_line": start_line,
                "end_line": end_line,
                "content": content,
            }
            i += window_size

    @staticmethod
    def _score_text(text: str, query: str) -> float:
        tl = text.lower()
        ql = query.lower().strip()
        if not ql:
            return 0.0
        # Simple token frequency score + phrase bonus
        tokens = [t for t in re.split(r"\W+", ql) if t]
        if not tokens:
            return 0.0
        score = 0.0
        for t in tokens:
            score += tl.count(t)
        if ql in tl:
            score += 2.0
        return score

    def search(self, query: str, window_sizes: List[int], top_k_per_size: int = 3) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for w in window_sizes:
            heap: List[Tuple[float, Dict[str, Any]]] = []
            for ch in self._iter_chunks(w):
                s = self._score_text(ch["content"], query)
                if s <= 0:
                    continue
                item = {
                    "window_size": w,
                    "start_line": ch["start_line"],
                    "end_line": ch["end_line"],
                    "score": float(s),
                    "excerpt": ch["content"][:600],
                }
                if len(heap) < top_k_per_size:
                    heapq.heappush(heap, (s, item))
                else:
                    if s > heap[0][0]:
                        heapq.heapreplace(heap, (s, item))
            # highest first
            size_results = [it for _, it in sorted(heap, key=lambda x: -x[0])]
            results.extend(size_results)
        # Sort global by score desc then by larger window
        results.sort(key=lambda r: (r.get("score", 0.0), r.get("window_size", 0)), reverse=True)
        return results


class FungusSearch:
    def __init__(self, device: str = "auto", use_model: bool = True):
        if MCPMRetriever is None:
            raise ImportError("MCPMRetriever not available")
        self.mcmp = MCPMRetriever(num_agents=300, max_iterations=80, device_mode=device, exploration_bonus=0.1)
        self.mcmp.use_embedding_model = bool(use_model)

    def add_docs(self, docs: List[str]) -> bool:
        return self.mcmp.add_documents(docs)

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return self.mcmp.search(query, top_k=top_k, verbose=False)


class EnterpriseTool:
    def __init__(self):
        if not ENTERPRISE_OK:
            raise ImportError("Enterprise RAG dependencies not installed")
        self.rag = EnterpriseCodeRAG()

    # Placeholder: user should build or load index separately
    def status(self) -> str:
        return json.dumps(self.rag.get_stats()) if hasattr(self.rag, "get_stats") else "{}"


# ----------------------------
# LangChain Agent Setup
# ----------------------------

AGENT_SYSTEM_PROMPT = (
    "You are a retrieval planning agent. First, ask the user 1-2 questions to clarify intent. "
    "The corpus is line-sliced text. Start with `collage_retriever` using larger windows first (400/300/200), "
    "then smaller (100/50). Next, use `lines_retriever` to extract precise line ranges around matches. "
    "Use `fungus_search` to explore and rank for supplemental evidence. Optionally consult `enterprise_rag` if available. "
    "Return only the SINGLE best evidence set (top-1). Avoid noisy or duplicate contexts. "
    "After tool calls, produce a STRICT JSON object: {matches: boolean, reasoning: string, confidence: number, "
    "contexts: [{source: string, start_line: number, end_line: number, excerpt: string}], final_answer: string}. "
    "If confidence < 0.6, reformulate the query and try again once."
)

def build_agent(ollama_model: str, tools: List['Tool']):
    if not LANGCHAIN_OK:
        raise RuntimeError("LangChain not available. Install langchain and langchain-ollama, and ensure Ollama is running.")

    llm = ChatOllama(model=ollama_model, temperature=0)

    # Prefer LangGraph ReAct agent when available
    graph = None
    try:
        from langgraph.prebuilt import create_react_agent  # type: ignore
        graph = create_react_agent(llm, tools)
    except Exception:
        graph = None

    if graph is not None:
        class _GraphWrapper:
            def __init__(self, g):
                self._g = g

            def invoke(self, payload: Dict[str, Any]):
                messages = []
                # Accept {input, system} for compatibility
                sys_msg = payload.get("system")
                user_input = payload.get("input", "")
                if sys_msg:
                    messages.append({"role": "system", "content": sys_msg})
                messages.append({"role": "user", "content": user_input})
                return self._g.invoke({"messages": messages})

        return _GraphWrapper(graph)

    # Fallback to legacy AgentExecutor
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
    )
    return agent


# ----------------------------
# Main Orchestrator
# ----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Combined Agent: Line + Fungus + Enterprise (optional)")
    parser.add_argument("--docs-file", type=str, help="Path to input file (.txt lines or .md)")
    parser.add_argument("--md-codeblocks", action="store_true", help="Extract fenced code blocks from markdown")
    parser.add_argument("--md-include-text", action="store_true", help="Also include non-code paragraphs")
    parser.add_argument("--query", type=str, help="User query to answer")
    parser.add_argument("--ollama-model", type=str, default="llama3.1", help="Ollama chat model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Target device for embeddings")
    parser.add_argument("--no-model", action="store_true", help="Use mock embeddings (no embedding model)")
    parser.add_argument("--windows", type=str, default="400,300,200,100,50", help="Comma-separated window sizes for collage retriever")
    parser.add_argument("--max-retry", type=int, default=1, help="Agent reformulation retries when confidence low")
    parser.add_argument("--strict-agent", action="store_true", help="Disable all heuristic fallbacks; require valid agent JSON output")
    parser.add_argument("--debug", action="store_true", help="Print detailed diagnostics for agent and environment")
    args = parser.parse_args()

    # Load documents
    if not args.docs_file or not os.path.exists(args.docs_file):
        print("❌ Provide --docs-file with valid path")
        sys.exit(1)

    if args.docs_file.lower().endswith(".md") and args.md_codeblocks:
        docs = load_codeblocks_from_markdown(args.docs_file, include_text=args.md_include_text)
        source_label = os.path.basename(args.docs_file) + " (codeblocks)"
    else:
        docs = load_documents_from_file(args.docs_file)
        source_label = os.path.basename(args.docs_file) + " (lines)"

    device = detect_device(args.device)
    use_model = not bool(args.no_model)
    # Device verification
    try:
        if device == "cuda" and torch is not None and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🟢 Using CUDA GPU: {gpu_name}")
        else:
            print(f"🟡 Using device: {device} (CUDA available: {bool(torch and torch.cuda.is_available())})")
    except Exception as _e:
        print(f"🟠 Device check error: {_e}")

    # Initialize tools
    line_index = LineIndex(docs)
    collage_index = CollageIndex(docs)
    fungus = FungusSearch(device=device, use_model=use_model)
    fungus.add_docs(docs)

    # LangChain Tools
    tools: List[Tool] = []

    def _tool_lines(q: str) -> str:
        """Fetch contextual lines for a query; returns JSON list with line ranges and snippets."""
        hits = line_index.search(q, window=3, max_hits=12)
        return json.dumps({"source": source_label, "hits": hits}, ensure_ascii=False)

    tools.append(Tool(name="lines_retriever", description="Retrieve contextual lines around matches in the line-sliced corpus.", func=_tool_lines))

    # Collage retriever tool
    try:
        window_sizes = [int(x) for x in str(args.windows).split(",") if str(x).strip().isdigit()]
        if not window_sizes:
            window_sizes = [400, 300, 200, 100, 50]
    except Exception:
        window_sizes = [400, 300, 200, 100, 50]

    def _tool_collage(q: str) -> str:
        hits = collage_index.search(q, window_sizes=window_sizes, top_k_per_size=1)  # strict top-1 per size
        # also derive global top-1
        best = hits[0] if hits else None
        return json.dumps({"source": source_label, "top1": best, "hits": hits}, ensure_ascii=False)

    tools.append(Tool(name="collage_retriever", description="Multi-window chunk search over corpus (400/300/200/100/50).", func=_tool_collage))

    def _tool_fungus(q: str) -> str:
        """Run fungus (MCPM) search; returns JSON with ranked results and stats."""
        res = fungus.search(q, top_k=8)
        return json.dumps(_json_safe(res), ensure_ascii=False)

    tools.append(Tool(name="fungus_search", description="Physarum-inspired exploration search over documents.", func=_tool_fungus))

    if ENTERPRISE_OK:
        # Provide a minimal enterprise status tool to avoid heavy ops here
        ent = EnterpriseTool()

        def _tool_enterprise_status(_: str) -> str:
            return ent.status()

        tools.append(Tool(name="enterprise_rag", description="Inspect enterprise RAG status (optional).", func=_tool_enterprise_status))

    # Build agent or fallback
    agent = None
    if LANGCHAIN_OK:
        try:
            # Ensure normalized host is visible to client
            import os as _os
            _os.environ['OLLAMA_HOST'] = _normalize_ollama_host(_os.environ.get('OLLAMA_HOST'))
            agent = build_agent(args.ollama_model, tools)
        except Exception as be:
            if getattr(args, "debug", False):
                print(f"[DEBUG] build_agent error: {be}")

    # Compose input prompt
    if not args.query:
        print("❌ Provide --query")
        sys.exit(1)

    system_preface = AGENT_SYSTEM_PROMPT + f"\nCorpus: {source_label}."
    user_payload = {
        "query": args.query,
        "hints": [
            "Use lines_retriever first to get context windows",
            "Then call fungus_search to rank and explore",
            "Return STRICT JSON with matches, reasoning, confidence, contexts, final_answer",
        ],
    }

    # One-shot with self-evaluation: If confidence < 0.6, try reformulations
    strict_mode = bool(args.strict_agent)
    if getattr(args, "debug", False):
        try:
            import sys as _sys, os as _os
            print(f"[DEBUG] LANGCHAIN_OK={LANGCHAIN_OK}")
            print(f"[DEBUG] Python={_sys.executable}")
            raw_host = _os.environ.get('OLLAMA_HOST')
            norm_host = _normalize_ollama_host(raw_host)
            print(f"[DEBUG] OLLAMA_HOST={raw_host} -> {norm_host}")
            try:
                import langchain as _lc  # type: ignore
                import langchain_ollama as _lco  # type: ignore
                import langgraph as _lg  # type: ignore
                print(f"[DEBUG] langchain={getattr(_lc,'__version__','?')} langchain-ollama={getattr(_lco,'__version__','?')} langgraph={getattr(_lg,'__version__','?')}")
            except Exception as _e:
                print(f"[DEBUG] langchain stack import error: {_e}")
            # ping ollama
            try:
                import httpx as _httpx  # type: ignore
                host = norm_host
                r = _httpx.get(host.rstrip('/') + '/api/tags', timeout=5.0)
                print(f"[DEBUG] ollama /api/tags status={r.status_code}")
            except Exception as _pe:
                print(f"[DEBUG] ollama ping failed: {_pe}")
        except Exception as _de:
            print(f"[DEBUG] diagnostics error: {_de}")

    def run_once(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        if agent is None:
            if strict_mode:
                return None, json.dumps({"error": "agent_unavailable_in_strict_mode"})
            # Fallback: heuristic combination of tools without LLM (non-strict)
            line_hits = line_index.search(payload.get("query", ""), window=3, max_hits=8)
            try:
                collage_hits = json.loads(_tool_collage(payload.get("query", ""))).get("hits", [])
            except Exception:
                collage_hits = []
            fungus_res = json.loads(_tool_fungus(payload.get("query", "")))
            results = fungus_res.get("results", [])
            matches = bool(results or line_hits)
            contexts = []
            for h in line_hits[:1]:
                contexts.append({
                    "source": source_label,
                    "start_line": h["start"],
                    "end_line": h["end"],
                    "excerpt": h["snippet"][:500],
                })
            for c in collage_hits[:1]:
                contexts.append({
                    "source": source_label,
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                    "excerpt": c.get("excerpt", "")[:500],
                })
            final_answer = "\n\n".join([r.get("content", "")[:300] for r in results[:1]])
            confidence = 0.7 if matches else 0.3
            data = {
                "matches": matches,
                "reasoning": "Heuristic fallback without LLM. Combined line windows and fungus ranking.",
                "confidence": confidence,
                "contexts": contexts,
                "final_answer": final_answer,
            }
            return data, json.dumps(data)
        try:
            raw = agent.invoke({"input": json.dumps(payload, ensure_ascii=False), "system": system_preface})
            text = raw["output"] if isinstance(raw, dict) and "output" in raw else str(raw)
        except Exception as e:
            # Guard: agent failure
            if strict_mode:
                return None, json.dumps({"error": f"agent_invoke_failed: {e}"})
            try:
                line_hits = line_index.search(payload.get("query", ""), window=3, max_hits=8)
                collage_hits = json.loads(_tool_collage(payload.get("query", ""))).get("hits", [])
                fungus_res = json.loads(_tool_fungus(payload.get("query", "")))
                results = fungus_res.get("results", [])
                matches = bool(results or line_hits or collage_hits)
                contexts = []
                for h in line_hits[:5]:
                    contexts.append({
                        "source": source_label,
                        "start_line": h["start"],
                        "end_line": h["end"],
                        "excerpt": h["snippet"][:500],
                    })
                for c in collage_hits[:3]:
                    contexts.append({
                        "source": source_label,
                        "start_line": c.get("start_line"),
                        "end_line": c.get("end_line"),
                        "excerpt": c.get("excerpt", "")[:500],
                    })
                final_answer = "\n\n".join([r.get("content", "")[:300] for r in results[:3]])
                confidence = 0.65 if matches else 0.3
                data = {
                    "matches": matches,
                    "reasoning": f"Agent failed ({e}); heuristic fallback used.",
                    "confidence": confidence,
                    "contexts": contexts,
                    "final_answer": final_answer,
                }
                return data, json.dumps(data)
            except Exception as ee:
                return None, json.dumps({"error": f"agent_invoke_failed: {e}", "fallback_error": str(ee)})
        try:
            data = json.loads(text)
            return data, text
        except Exception:
            # Try to extract a JSON object via regex
            try:
                import re as _re
                m = _re.search(r"\{[\s\S]*\}$", text.strip())
                if m:
                    data = json.loads(m.group(0))
                    return data, m.group(0)
            except Exception:
                pass
            if strict_mode:
                return None, text
            # Heuristic fallback assembly (non-strict)
            try:
                line_hits = line_index.search(payload.get("query", ""), window=3, max_hits=8)
                collage_hits = json.loads(_tool_collage(payload.get("query", ""))).get("hits", [])
                fungus_res = json.loads(_tool_fungus(payload.get("query", "")))
                results = fungus_res.get("results", [])
                matches = bool(results or line_hits or collage_hits)
                contexts = []
                if line_hits:
                    h = line_hits[0]
                    contexts.append({
                        "source": source_label,
                        "start_line": h["start"],
                        "end_line": h["end"],
                        "excerpt": h["snippet"][:500],
                    })
                if collage_hits:
                    c = collage_hits[0]
                    contexts.append({
                        "source": source_label,
                        "start_line": c.get("start_line"),
                        "end_line": c.get("end_line"),
                        "excerpt": c.get("excerpt", "")[:500],
                    })
                final_answer = "\n\n".join([r.get("content", "")[:300] for r in results[:1]])
                data = {
                    "matches": matches,
                    "reasoning": "LLM returned non-JSON; heuristic fallback used.",
                    "confidence": 0.6 if matches else 0.3,
                    "contexts": contexts,
                    "final_answer": final_answer,
                }
                return data, json.dumps(data, ensure_ascii=False)
            except Exception:
                return None, text

    data, raw_text = run_once(user_payload)
    retries = max(0, int(args.max_retry))
    attempt = 0
    while (not strict_mode) and agent is not None and attempt < retries and (not data or not isinstance(data, dict) or data.get("confidence", 0) < 0.6):
        user_payload["hints"].append("Confidence was low; reformulate and try again.")
        data2, raw_text2 = run_once(user_payload)
        data = data2 or data
        raw_text = raw_text2 or raw_text
        attempt += 1

    # Output
    print("\n=== Agent Structured Output ===")
    try:
        if strict_mode:
            # In strict mode, print only parsed JSON or raw text fallback above
            if data and isinstance(data, dict):
                print(json.dumps(_json_safe(data), indent=2, ensure_ascii=False))
            else:
                print(raw_text)
        else:
            # Ensure required keys to avoid empty {}
            if not data or not isinstance(data, dict):
                data = {"matches": False, "reasoning": "no_data", "confidence": 0.0, "contexts": [], "final_answer": ""}
            print(json.dumps(_json_safe(data), indent=2, ensure_ascii=False))
    except Exception:
        print(raw_text)


if __name__ == "__main__":
    main()


