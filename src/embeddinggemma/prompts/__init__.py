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
            "Explore broadly to map the codebase. "
            "Surface key modules, responsibilities, and cross-cutting concerns. "
            "Prefer breadth first; highlight surprising or high-impact areas to inspect next."
        )
    if m == "summary":
        return (
            "Produce a concise summary of the most relevant code. "
            "Capture purpose, main entrypoints, and how to use it. "
            "Prefer clarity and brevity; avoid minor details."
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
        "missing_context (str[]), follow_up_queries (str[]), keywords (str[]), inspect (str[]).\n\n"
        "OPTIONAL: You may also include at the top level:\n"
        "- 'suggested_top_k' (int, 5-50): Adjust retrieval depth for next cycle\n"
        "  * Use 5-15 for focused analysis (found specific target, narrowing down)\n"
        "  * Use 15-25 for normal exploration (current default)\n"
        "  * Use 25-50 for broad context (exploring new area, need more examples)\n\n"
        "USING ACCUMULATED KNOWLEDGE:\n"
        "If 'ACCUMULATED KNOWLEDGE' is provided above, use it to inform your decisions:\n"
        "- Recognize already-explored areas (avoid redundant follow-up queries)\n"
        "- Build on previous discoveries (reference known entry points, patterns)\n"
        "- Identify gaps in knowledge (what's missing from accumulated memories?)\n"
        "- Make connections between current chunks and accumulated knowledge\n"
        "- Generate follow-up queries that fill knowledge gaps, not repeat past exploration\n\n"
        "Focus on evaluating code relevance and generating effective follow-up queries. "
        "A separate Memory Manager Agent handles knowledge ingestion decisions.\n\n"
        "Example:\n"
        "{\"items\": [\n"
        "  {\"doc_id\": 42, \"is_relevant\": true, \"why\": \"Contains FastAPI server initialization\",\n"
        "   \"entry_point\": true, \"missing_context\": [\"Route handlers\", \"Middleware setup\"],\n"
        "   \"follow_up_queries\": [\"Find FastAPI route definitions\", \"Explore WebSocket handlers\"],\n"
        "   \"keywords\": [\"FastAPI\", \"async\", \"server\"], \"inspect\": [\"src/server.py\"]}\n"
        "], \"suggested_top_k\": 25}"
    )


def build_report_prompt(mode: str, query: str, top_k: int, docs: list[dict]) -> str:
    m = (mode or "deep").lower()
    # No truncation - keep full chunk context for better LLM analysis
    ctx = "\n\n".join([(it.get("content") or "") for it in (docs or [])])
    base = f"Mode: {m}\nQuery: {query}\nTopK: {int(top_k)}\n\nContext begins:\n{ctx}\n\nContext ends.\n\n"
    schema = report_schema_hint()
    instr = get_report_instructions(m)
    return base + instr + "\n" + schema + " Answer with JSON only."


def build_judge_prompt(
    mode: str,
    query: str,
    results: list[dict],
    task_mode: str | None = None,
    query_history: list[str] | None = None,
    memory_context: str | None = None
) -> str:
    """Build judge prompt with optional task_mode awareness and memory context.

    Args:
        mode: Judge mode (steering, focused, exploratory)
        query: Current search query
        results: List of code chunks to evaluate
        task_mode: Optional main task mode (architecture, bugs, quality, etc.) for context
        query_history: Optional list of previous queries to avoid repetition
        memory_context: Optional context from past insights stored in Supermemory
    """
    m = (mode or "steering").lower()
    items = []
    for it in (results or []):
        try:
            items.append({
                'doc_id': int(it.get('id', it.get('doc_id', -1))),
                'score': float(it.get('score', 0.0)),
                # No truncation - judge needs full context to make informed decisions
                'content': (it.get('content') or ''),
            })
        except Exception:
            continue
    instr = get_report_instructions(m)
    schema = judge_schema_hint()

    # Build task context if task_mode is provided and different from judge mode
    task_context = ""
    if task_mode and task_mode.lower() not in ('steering', 'focused', 'exploratory'):
        task_instr = get_report_instructions(task_mode)
        task_context = (
            f"\n\n**MAIN TASK OBJECTIVE** (Task Mode: {task_mode}):\n"
            f"{task_instr}\n\n"
            f"Your role as judge is to evaluate code chunks and generate follow-up queries "
            f"that help fulfill this MAIN TASK OBJECTIVE. Your follow_up_queries should be "
            f"specifically designed to gather information needed to complete the {task_mode} analysis.\n"
        )

    # Add query history to prevent repetition
    history_context = ""
    if query_history and len(query_history) > 0:
        # Show last 20 queries to avoid overwhelming the prompt
        recent_queries = query_history[-20:]
        history_list = "\n".join([f"  - {q}" for q in recent_queries])
        history_context = (
            f"\n\n**PREVIOUS QUERIES EXPLORED** (avoid repeating these):\n"
            f"{history_list}\n\n"
            f"Generate NEW follow-up queries that explore DIFFERENT aspects not covered above. "
            f"Look for gaps in understanding and unexplored areas of the codebase.\n"
        )

    # Add memory context from past insights
    memory_section = ""
    if memory_context:
        memory_section = f"\n\n{memory_context}\n"

    return (
        f"Judge Mode: {m}\nQuery: {query}\n" +
        task_context +
        memory_section +
        history_context +
        "\n" + instr + "\n" +
        "Evaluate the following code chunks for relevance to the query. "
        "Mark entry_point for main functions, API routes, or top-level orchestrators.\n\n" +
        __import__('json').dumps({'chunks': items}, ensure_ascii=False) + "\n\n" +
        schema
    )



