"""Deterministic AST scanner for fungus.

When a query is "syntactic" (looking for specific Python constructs like
importlib.import_module, getattr-based dispatch, decorator-registered
plugins, etc.) the embedder is the wrong tool. This module gives 100%
recall via a fast AST walk over the indexed Python files.

Approach:
- Walk CODEBASE on every call (cheap: ~45k chunks → ~5k Python files → 2-3s).
- For each file, parse AST and check for the requested pattern.
- Return matches as fungus-shaped result dicts so they can be merged
  into the normal pipeline.

Three pattern detectors are built in:
- importlib_calls:   importlib.import_module(...), importlib.util.spec_from_file_location(...), __import__(...)
- decorator:         @some_decorator (e.g. @mcp.tool, @app.get, @router.post)
- inheritance:       class X(BaseClass): ...

The scanner is intentionally pluggable — you pass a query-aware detector
function and get matches back.
"""
from __future__ import annotations
import os
import re
import ast
import logging
from typing import Callable, List, Dict, Any, Optional

_logger = logging.getLogger("ASTScan")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


# ── Detectors ──────────────────────────────────────────────────────────

def detect_importlib(tree: ast.AST) -> List[ast.AST]:
    """Return all importlib.*/__import__ call nodes."""
    hits: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # importlib.import_module(...) / importlib.util.spec_from_file_location(...)
            if isinstance(func, ast.Attribute):
                # walk down the attribute chain to find "importlib"
                cur = func
                while isinstance(cur, ast.Attribute):
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id == "importlib":
                    hits.append(node)
                    continue
            # __import__(...)
            if isinstance(func, ast.Name) and func.id == "__import__":
                hits.append(node)
    return hits


def detect_getattr_dispatch(tree: ast.AST) -> List[ast.AST]:
    """getattr(module, 'name')-style dynamic dispatch."""
    hits: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            hits.append(node)
    return hits


def detect_decorator(decorator_substr: str):
    """Build a detector that matches functions/classes decorated with
    something containing the given substring (e.g. 'mcp.tool', 'app.get')."""
    def _det(tree: ast.AST) -> List[ast.AST]:
        hits: List[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    try:
                        src = ast.unparse(dec)
                    except Exception:
                        src = ""
                    if decorator_substr in src:
                        hits.append(node)
                        break
        return hits
    return _det


def detect_inherits_from(base_name: str):
    """Classes inheriting from BaseName."""
    def _det(tree: ast.AST) -> List[ast.AST]:
        hits: List[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for b in node.bases:
                    try:
                        src = ast.unparse(b)
                    except Exception:
                        src = ""
                    if base_name in src:
                        hits.append(node)
                        break
        return hits
    return _det


# ── Walker ─────────────────────────────────────────────────────────────

def _excluded(name: str, exclude_dirs: list[str]) -> bool:
    """Match against EXCLUDE_DIRS with prefix-style for venv-likes.

    Exact match for normal entries, but `.venv` should also match `.venv312`,
    `.venv-py311`, `venv*` etc. — anything starting with `.venv` or `venv`.
    """
    if name in exclude_dirs:
        return True
    if name.startswith(".venv") or name.startswith("venv"):
        return True
    return False


def walk_python_files(root: str, exclude_dirs: list[str], max_files: int = 30000) -> List[str]:
    """Return a list of .py file paths under root, applying directory excludes.

    Default cap raised to 30k — real-world Python repos rarely have more,
    and capping at 8k was silently dropping files like
    voice/python/swarm/orchestrator/tool_registry.py.

    Also: filters site-packages aggressively — a `voice/.venv312/Lib/site-packages`
    is NOT what we mean by "fungus index", and including it floods AST hits
    with transformer-internal importlib calls.
    """
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _excluded(d, exclude_dirs)
                       and "site-packages" not in d]
        # Belt and braces: if we somehow landed inside a site-packages,
        # skip the file entirely.
        if "site-packages" in dirpath or "\\.venv" in dirpath or "/.venv" in dirpath:
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(os.path.join(dirpath, fn))
                if len(files) >= max_files:
                    return files
    return files


# ── Mtime-keyed cache ──────────────────────────────────────────────────
# A full AST scan over 5k Python files takes ~30-40s. That's too slow for
# every query. We cache the parsed AST + a tuple of (path, mtime) per file,
# so repeat scans for the same query are <1s. Cache survives the lifetime
# of the process (typical fungus-MCP-server uptime: hours/days).

_AST_CACHE: Dict[str, tuple[float, ast.AST]] = {}


def _get_cached_ast(path: str) -> Optional[ast.AST]:
    """Return parsed AST for `path`, caching by (path, mtime)."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _AST_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        tree = ast.parse(src, filename=path)
    except (SyntaxError, ValueError):
        return None
    _AST_CACHE[path] = (mtime, tree)
    return tree


def clear_ast_cache() -> None:
    _AST_CACHE.clear()


def scan_with_detector(root: str, detector: Callable[[ast.AST], List[ast.AST]],
                       exclude_dirs: list[str],
                       max_files: int = 30000,
                       context_lines: int = 4) -> List[Dict[str, Any]]:
    """Walk python files under root, apply detector, return fungus-shaped hits.

    Each hit is a dict like:
        {"content": "# file: <rel> | lines: <s>-<e> | window: 200\\n...code...",
         "metadata": {"file": rel, "ast_kind": "importlib_call", ...},
         "relevance_score": 1.0,
         "_ast_match": True}

    `relevance_score=1.0` is a fixed anchor — re-ranker can still adjust it.
    Uses the mtime-keyed AST cache so repeat queries are <1s.
    """
    files = walk_python_files(root, exclude_dirs, max_files=max_files)
    out: List[Dict[str, Any]] = []
    for path in files:
        tree = _get_cached_ast(path)
        if tree is None:
            continue
        hits = detector(tree)
        if not hits:
            continue
        # Only read source if we have hits — saves I/O on most files.
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                src = f.read()
        except OSError:
            continue
        lines = src.splitlines()
        rel = os.path.relpath(path, start=root)
        seen_ranges: set[tuple[int, int]] = set()
        for h in hits:
            lineno = getattr(h, "lineno", 0) or 1
            end_lineno = getattr(h, "end_lineno", lineno) or lineno
            start = max(1, lineno - context_lines)
            end = min(len(lines), end_lineno + context_lines)
            key = (start, end)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            snippet = "\n".join(lines[start - 1:end])
            try:
                code_repr = ast.unparse(h)[:200]
            except Exception:
                code_repr = ""
            content = (
                f"# file: ..\\{rel} | lines: {start}-{end} | window: ast\n"
                f"# ast_match: {code_repr}\n"
                f"{snippet}"
            )
            # Pattern signature: a normalised version of the AST call that
            # ignores argument names so `spec_from_file_location(module_name, ...)`
            # and `spec_from_file_location(name, ...)` count as same pattern.
            try:
                pattern_sig = re.sub(r"[a-zA-Z_][a-zA-Z0-9_]*", "X", code_repr)
                pattern_sig = re.sub(r"\s+", " ", pattern_sig)[:80]
            except Exception:
                pattern_sig = code_repr[:80]
            out.append({
                "content": content,
                "metadata": {"file": rel, "ast_kind": detector.__name__,
                             "lineno": lineno,
                             "ast_pattern": pattern_sig},
                "relevance_score": 1.0,
                "_ast_match": True,
                "_ast_pattern_sig": pattern_sig,
            })
    _logger.info("AST scan: %d files → %d hits", len(files), len(out))
    return out


# ── Query → detector dispatch ──────────────────────────────────────────

_QUERY_DETECTOR_HINTS = (
    # (substring trigger, detector function, label)
    (("importlib", "import_module", "__import__", "sys.modules"),
     detect_importlib, "importlib_calls"),
    (("getattr",),
     detect_getattr_dispatch, "getattr_dispatch"),
)


def pick_detector_for_query(query: str) -> Optional[tuple[Callable, str]]:
    """Heuristic: return a detector + label if the query has syntactic hints.
    Returns None when no detector matches — caller should skip AST scan then.
    """
    q = query.lower()
    for triggers, det, label in _QUERY_DETECTOR_HINTS:
        if any(t in q for t in triggers):
            return det, label
    return None
