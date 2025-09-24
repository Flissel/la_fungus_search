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



