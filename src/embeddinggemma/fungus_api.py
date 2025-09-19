#!/usr/bin/env python3
"""
FastAPI service exposing Fungus (MCPMRetriever) search modes and analyses.

Modes:
- deep: broad multi-granular retrieval
- structure: function/class-centric retrieval
- exploratory: answer open questions about the repo
- summary: summarize retrieved content
- similar: detect similarly named identifiers (to reduce hallucinations)
- redundancy: detect duplicate/near-duplicate chunks
- repair: propose versioned renames for problematic names (hallucination-prone)
"""

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import re
import hashlib
import requests
import ast
import pathlib
import json

# Ensure project root on sys.path to import mcmp_rag
import sys
_THIS_DIR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from mcmp_rag import MCPMRetriever  # type: ignore
except Exception as e:
    MCPMRetriever = None  # type: ignore
    print(f"⚠️ MCPMRetriever Import fehlgeschlagen: {e}")


OLLAMA_HOST_DEFAULT = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip('/')
OLLAMA_MODEL_DEFAULT = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL_DEFAULT, host: str = OLLAMA_HOST_DEFAULT, temperature: float = 0.2) -> str:
    try:
        url = f"{host}/api/generate"
        r = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except Exception as e:
        return f"[LLM error] {e}"


def extract_codeblocks_from_markdown(path: str, include_text: bool = False) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    code_pattern = re.compile(r"```([a-zA-Z0-9_+\-]*)\n([\s\S]*?)```", re.MULTILINE)
    documents: List[str] = []
    last_end = 0
    for m in code_pattern.finditer(content):
        lang = m.group(1) or ""
        code = (m.group(2) or "").strip('\n')
        full = f"```{lang}\n{code}\n```"
        if code.strip():
            documents.append(full)
        if include_text:
            prefix = content[last_end:m.start()].strip()
            if prefix:
                for para in re.split(r"\n\n+", prefix):
                    para = para.strip()
                    if para:
                        documents.append(para)
        last_end = m.end()
    if include_text:
        tail = content[last_end:].strip()
        if tail:
            for para in re.split(r"\n\n+", tail):
                para = para.strip()
                if para:
                    documents.append(para)
    if not documents:
        documents = [content]
    return documents


def chunk_python_file(path: str, windows: Optional[List[int]] = None) -> List[str]:
    if windows is None:
        windows = [50, 100, 200, 300, 400]
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return []
    chunks: List[str] = []
    total = len(lines)
    rel = os.path.relpath(path)
    for w in windows:
        for i in range(0, total, w):
            start = i + 1
            end = min(i + w, total)
            body = ''.join(lines[i:end])
            if body.strip():
                header = f"# file: {rel} | lines: {start}-{end} | window: {w}\n"
                chunks.append(header + body)
    return chunks


def collect_docs(docs_file: Optional[str], md_codeblocks: bool, md_include_text: bool) -> List[str]:
    docs: List[str] = []
    if docs_file and os.path.exists(docs_file):
        if docs_file.endswith('.py'):
            docs = chunk_python_file(docs_file)
            if not docs:
                with open(docs_file, 'r', encoding='utf-8', errors='ignore') as f:
                    docs = [f.read()]
        elif docs_file.endswith('.md') and md_codeblocks:
            docs = extract_codeblocks_from_markdown(docs_file, include_text=md_include_text)
        else:
            with open(docs_file, 'r', encoding='utf-8', errors='ignore') as f:
                docs = [f.read()]
    return docs


def identifiers_from_text(text: str) -> List[str]:
    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)


def find_similar_names(docs: List[str]) -> List[Dict[str, Any]]:
    # naive similarity by prefix and length distance
    names: List[str] = []
    for d in docs:
        names.extend(identifiers_from_text(d))
    names = list({n for n in names if len(n) >= 3})
    names.sort()
    similar: List[Dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            if a[0].lower() != b[0].lower():
                break
            # simple ratio
            common = os.path.commonprefix([a.lower(), b.lower()])
            ratio = len(common) / max(len(a), len(b))
            if ratio >= 0.6 and a != b:
                similar.append({"a": a, "b": b, "score": ratio})
    return similar[:200]


def find_redundancy(docs: List[str]) -> List[Dict[str, Any]]:
    # detect duplicates via hash of normalized content
    seen: Dict[str, List[int]] = {}
    for idx, d in enumerate(docs):
        norm = re.sub(r"\s+", " ", d.strip()).lower()
        h = hashlib.sha256(norm.encode('utf-8')).hexdigest()[:16]
        seen.setdefault(h, []).append(idx)
    duplicates = [{"hash": h, "indices": idxs, "count": len(idxs)} for h, idxs in seen.items() if len(idxs) > 1]
    duplicates.sort(key=lambda x: x["count"], reverse=True)
    return duplicates


class SearchRequest(BaseModel):
    query: str
    mode: str = "deep"
    top_k: int = 5
    docs_file: Optional[str] = None
    md_codeblocks: bool = False
    md_include_text: bool = False
    ollama_model: Optional[str] = None
    ollama_host: Optional[str] = None


app = FastAPI(title="Fungus API", version="0.1.0")


@app.post("/api/fungus/search")
def fungus_search(req: SearchRequest):
    if MCPMRetriever is None:
        return {"error": "mcp_retriever_unavailable"}
    docs = collect_docs(req.docs_file, req.md_codeblocks, req.md_include_text)
    retr = MCPMRetriever(num_agents=200, max_iterations=60)
    if docs:
        retr.add_documents(docs)
    out = retr.search(req.query, top_k=req.top_k)
    results = out.get('results', []) if isinstance(out, dict) else []

    # Mode-specific LLM prompts
    prompt = ""
    if req.mode == "deep":
        prompt = f"Führe eine tiefe Analyse durch und beantworte präzise: {req.query}\nNenne Belege aus den Snippets."
    elif req.mode == "structure":
        prompt = f"Analysiere Funktionen/Klassen, extrahiere relevante Definitionen und beantworte: {req.query}"
    elif req.mode == "exploratory":
        prompt = f"Beantworte explorativ: {req.query}. Stelle ggf. Anschlussfragen."
    elif req.mode == "summary":
        prompt = f"Fasse die wichtigsten Informationen zu '{req.query}' zusammen und liste Quellen."
    elif req.mode == "similar":
        sim = find_similar_names(docs)
        return {"mode": req.mode, "similar_names": sim, "count": len(sim)}
    elif req.mode == "redundancy":
        dup = find_redundancy(docs)
        return {"mode": req.mode, "duplicates": dup, "count": len(dup)}
    elif req.mode == "repair":
        sim = find_similar_names(docs)
        plan = [
            {
                "from": s["a"],
                "to": f"{s['a']}_v2",
                "reason": f"Similar to {s['b']} (score {s['score']:.2f})"
            } for s in sim[:50]
        ]
        return {"mode": req.mode, "rename_plan": plan}
    else:
        prompt = req.query

    # Build context from results
    context = "\n\n".join((r.get('content', '') or '')[:800] for r in results)
    llm_prompt = f"Kontext:\n{context}\n\nAufgabe:\n{prompt}\n\nAntwort:".strip()
    answer = ollama_generate(llm_prompt, model=(req.ollama_model or OLLAMA_MODEL_DEFAULT), host=(req.ollama_host or OLLAMA_HOST_DEFAULT))

    return {
        "mode": req.mode,
        "query": req.query,
        "top_k": req.top_k,
        "results": results,
        "answer": answer
    }


# --------------------- Code Edit Event API ---------------------

HDR_RE = re.compile(r"^# file:\s*(.+?)\s*\|\s*lines:\s*(\d+)-(\d+)(?:\s*\|\s*window:\s*(\d+))?", re.MULTILINE)


def _parse_chunk_header(text: str) -> Optional[Dict[str, Any]]:
    m = HDR_RE.search(text or "")
    if not m:
        return None
    return {
        "file_path": m.group(1),
        "start": int(m.group(2)),
        "end": int(m.group(3)),
        "window": int(m.group(4)) if m.group(4) else None,
    }


def _find_method_span(path: str, method_name: str) -> Optional[Dict[str, int]]:
    try:
        src = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            target = node
            break
        if isinstance(node, ast.ClassDef):
            for b in node.body:
                if isinstance(b, ast.FunctionDef) and b.name == method_name:
                    target = b
                    break
            if target:
                break
    if not target:
        return None
    start = getattr(target, "lineno", 1)
    end = getattr(target, "end_lineno", start)
    return {"start": int(start), "end": int(end)}


def _read_slice(path: str, start: int, end: int) -> str:
    try:
        lines = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(1, int(start)); end = max(start, int(end))
        return "\n".join(lines[start-1:end])
    except Exception:
        return ""


REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_QUEUE_KEY = os.environ.get("CODER_EVENTS_KEY", "coder:events")

try:
    import redis  # type: ignore
    _redis_client = redis.from_url(REDIS_URL)  # lazy connection
except Exception:
    _redis_client = None  # type: ignore


class BuildEditEventRequest(BaseModel):
    # Either provide chunk_text with the header, or explicit file_path/start/end
    chunk_text: Optional[str] = Field(default=None, description="Chunk text containing header '# file: ... | lines: a-b | window: w'")
    file_path: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    prefer_method: Optional[str] = Field(default=None, description="If given, narrow the corridor to this function via AST")
    instructions: Optional[str] = Field(default="", description="Natural language edit instructions for coder agent")
    publish: bool = Field(default=False, description="If true, publish the event to Redis queue")
    extra: Optional[Dict[str, Any]] = None


@app.post("/api/edit/build_event")
def build_edit_event(req: BuildEditEventRequest):
    meta = None
    if req.chunk_text:
        meta = _parse_chunk_header(req.chunk_text)
    if not meta and req.file_path and req.start and req.end:
        meta = {"file_path": req.file_path, "start": int(req.start), "end": int(req.end), "window": None}
    if not meta:
        return {"error": "missing_bounds", "message": "Provide chunk_text with header or explicit file_path/start/end"}

    file_path = meta["file_path"]
    a, b = int(meta["start"]), int(meta["end"])
    if req.prefer_method:
        span = _find_method_span(file_path, req.prefer_method)
        if span:
            a, b = int(span["start"]), int(span["end"])

    before = _read_slice(file_path, a, b)
    event = {
        "type": "code_edit_request",
        "file_path": file_path,
        "start_line": a,
        "end_line": b,
        "before": before,
        "instructions": req.instructions or "",
        "meta": {k: v for k, v in (req.extra or {}).items()},
    }

    published = False
    if req.publish:
        if _redis_client is None:
            return {"error": "redis_unavailable", "event": event}
        try:
            _redis_client.rpush(REDIS_QUEUE_KEY, json.dumps(event, ensure_ascii=False))
            published = True
        except Exception as e:
            return {"error": f"redis_error: {e}", "event": event}

    return {"status": "ok", "event": event, "published": published}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("FUNGUS_API_PORT", "8055"))
    uvicorn.run("embeddinggemma.fungus_api:app", host="0.0.0.0", port=port, reload=True)



