from __future__ import annotations

def _default_instructions(mode: str) -> str:
    m = (mode or "deep").lower()
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


