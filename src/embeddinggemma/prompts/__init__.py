from __future__ import annotations

def _default_instructions(mode: str) -> str:
    m = (mode or "deep").lower()
    if m == "steering":
        return (
            "You are a RAG query steering agent for code retrieval. "
            "Goal: expand beyond a given chunk by inferring surrounding code and requesting larger windows. "
            "At each step: (1) analyze the chunk header and content, (2) guess the missing context (continuations, imports, callees/callers, adjacent lines, sibling functions/classes), "
            "(3) propose follow_up_queries and keywords to refine retrieval, (4) vote relevance (is_relevant) with why, (5) if strong entry_point, mark entry_point=true and suggest files/functions to inspect, "
            "(6) recommend window expansions (e.g., same file +/- N lines) to retrieve bigger spans. "
            "Optimize for quickly converging to the complete implementation that answers the initial query. "
            "Return strictly structured JSON objects per item with fields: is_relevant, why, entry_point, missing_context, follow_up_queries, keywords, functions_to_inspect, files_to_inspect."
        )
    if m == "structure":
        return (
            "Identify modules, classes, functions, and their relationships. "
            "Prefer summarizing public APIs and entrypoints. "
        )
    if m == "exploratory":
        return (
            "Surface diverse areas of the codebase relevant to the query. "
            "Prefer coverage across files over depth. "
        )
    if m == "summary":
        return (
            "Produce concise high-level summaries for each chunk and reduce redundancy. "
        )
    if m == "repair":
        return (
            "Focus on error-prone or suspicious code and dependencies that may break. "
        )
    # deep (default)
    return (
        "Extract purpose, dependencies, and how the code answers the query. "
        "Prefer depth on the most relevant chunks. "
    )


def get_report_instructions(mode: str) -> str:
    """Return mode-specific instruction text.

    If a custom module embeddinggemma.modeprompts.<mode> exists with a
    function instructions() -> str, it will be used; otherwise default text.
    """
    try:
        import importlib
        mod = importlib.import_module(f"embeddinggemma.modeprompts.{(mode or 'deep').lower()}")
        if hasattr(mod, "instructions"):
            txt = mod.instructions()  # type: ignore
            if isinstance(txt, str) and txt.strip():
                return txt
    except Exception:
        pass
    return _default_instructions(mode)



def report_schema_hint() -> str:
    return (
        "Return ONLY JSON with key 'items' (array). Each item must include: "
        "code_chunk (str), content (str), file_path (str), line_range ([int,int]), "
        "code_purpose (str), code_dependencies (str[] or str), file_type (str), "
        "embedding_score (number), relevance_to_query (str), query_initial (str), follow_up_queries (str[])."
    )


def judge_schema_hint() -> str:
    return (
        "Return ONLY JSON with key 'items' (array). Each item must include: "
        "doc_id (int), is_relevant (bool), why (str), entry_point (bool), "
        "missing_context (str[]), follow_up_queries (str[]), keywords (str[]), inspect (str[])."
    )


def build_report_prompt(mode: str, query: str, top_k: int, docs: list[dict]) -> str:
    m = (mode or "deep").lower()
    ctx = "\n\n".join([(it.get("content") or "")[:1200] for it in (docs or [])])
    base = f"Mode: {m}\nQuery: {query}\nTopK: {int(top_k)}\n\nContext begins:\n{ctx}\n\nContext ends.\n\n"
    schema = report_schema_hint()
    instr = get_report_instructions(m)
    return base + instr + "\n" + schema + " Answer with JSON only."


def build_judge_prompt(mode: str, query: str, results: list[dict]) -> str:
    m = (mode or "steering").lower()
    items = []
    for it in (results or []):
        try:
            items.append({
                'doc_id': int(it.get('id', it.get('doc_id', -1))),
                'score': float(it.get('score', 0.0)),
                'content': (it.get('content') or '')[:1200],
            })
        except Exception:
            continue
    instr = get_report_instructions(m)
    schema = judge_schema_hint()
    return (
        f"Mode: {m}\nQuery: {query}\n\n" +
        instr + "\n" +
        "Evaluate the following code chunks for relevance to the query. "
        "Mark entry_point for main functions, API routes, or top-level orchestrators.\n\n" +
        __import__('json').dumps({'chunks': items}, ensure_ascii=False) + "\n\n" +
        schema
    )



