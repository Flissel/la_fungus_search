import os
import re
from typing import List


def _normalize_query_text(text: str) -> str:
    try:
        t = (text or "").strip().lower()
        t = re.sub(r"[^a-z0-9]+", " ", t)
    except re.error:
        t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t).strip()
    stop = {
        "the","a","an","is","are","be","to","of","in","on","for","and","or","with","how","what","where","when","which","that",
        "does","do","can","i","we","you","it","this","these","those","about","use","used","using","run","start","guide"
    }
    tokens = [w for w in t.split() if w not in stop]
    return " ".join(tokens)


def _token_set(text: str) -> set:
    return set(_normalize_query_text(text).split())


def dedup_multi_queries(queries: List[str], similarity_threshold: float = 0.8) -> List[str]:
    if not queries:
        return []
    kept: List[str] = []
    kept_sets: List[set] = []
    thr = max(0.0, min(float(similarity_threshold), 1.0))
    for q in queries:
        ts = _token_set(q)
        if not ts:
            continue
        duplicate = False
        for ks in kept_sets:
            inter = len(ts & ks)
            union = len(ts | ks) or 1
            jacc = inter / union
            if jacc >= thr:
                duplicate = True
                break
        if not duplicate:
            kept.append(q)
            kept_sets.append(ts)
    if not kept and queries:
        kept = [queries[0]]
    return kept


def _ollama_generate(prompt: str, model: str = None, timeout: int = 180) -> str:
    try:
        import requests
        host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
        model_name = model or os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
        r = requests.post(
            f"{host}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=timeout,
        )
        if r.ok:
            return r.json().get('response', '')
        return f"[LLM error] status={r.status_code}"
    except Exception as e:
        return f"[LLM error] {e}"


def generate_multi_queries_from_llm(base_query: str, num_queries: int = 5, context_files: List[str] = None, keyword_hints: List[str] = None) -> List[str]:
    try:
        n = max(1, min(int(num_queries), 10))
    except Exception:
        n = 5
    files_hint = "\n".join(sorted(set((context_files or [])[:40])))
    kw_hint = ", ".join(sorted(set([(k or '').strip() for k in (keyword_hints or []) if (k or '').strip()])))
    lines_prompt = [
        "You reformulate a single repository question into multiple concrete search queries.",
        "Focus on files that are embedded (so results are answerable).",
        "Candidate files (hints):",
        files_hint,
    ]
    if kw_hint:
        lines_prompt.append(f"Query keywords (hints): {kw_hint}")
    lines_prompt += [
        "",
        "Rules:",
        "- Output EXACTLY {n} lines, no numbering or bullets.",
        "- Each line should be a direct, concrete code-search question.",
        "- Prefer including exact file paths from the hints when relevant.",
        "- Keep lines under 90 chars.",
    ]
    sys_prompt = "\n".join(lines_prompt).replace("{n}", str(n))
    user = f"Base query: {base_query}\nWrite {n} concrete repository search questions (one per line), grounded in the hinted files."
    text = _ollama_generate(f"System:\n{sys_prompt}\n\nUser:\n{user}")
    lines = [re.sub(r"^[\-\d\.\)\s]+", "", ln).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return lines[:n]
