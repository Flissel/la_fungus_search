"""Sprachtabelle: Dateipfad -> Sprache, Sprache -> tree-sitter-Parser.

Gepinnt: tree-sitter==0.26.0, tree-sitter-language-pack==1.22.1.
PowerShell: Das Language-Pack liefert einen Parser fuer "powershell"; Task 7
kann ihn nutzen (Regex nur als Rueckfall).
"""
from __future__ import annotations

import posixpath
from functools import lru_cache

import tree_sitter
from tree_sitter_language_pack import get_parser

PARSER_VERSION: int = 1

SPRACHEN: dict[str, tuple[str, ...]] = {
    "python": (".py", ".pyi"),
    "typescript": (".ts", ".mts", ".cts"),
    "tsx": (".tsx",),
    "javascript": (".js", ".mjs", ".cjs", ".jsx"),
    "sql": (".sql",),
    "yaml": (".yml", ".yaml"),
    "json": (".json",),
    "powershell": (".ps1", ".psm1", ".psd1"),
    "bash": (".sh", ".bash"),
    "rust": (".rs",),
    "go": (".go",),
    "markdown": (".md", ".markdown"),
}

_ENDUNG_ZU_SPRACHE: dict[str, str] = {
    endung: sprache for sprache, endungen in SPRACHEN.items() for endung in endungen
}


def sprache_fuer(pfad: str) -> str | None:
    """Sprache zu einem Pfad oder None. `.env*` (ausser `.env`) ist `dotenv`."""
    name = posixpath.basename(pfad.replace("\\", "/")).lower()
    if name.startswith(".env"):
        return "dotenv" if name != ".env" else None
    return _ENDUNG_ZU_SPRACHE.get(posixpath.splitext(name)[1])


@lru_cache(maxsize=None)
def parser_fuer(sprache: str) -> tree_sitter.Parser:
    """Gecachter tree-sitter-Parser fuer eine Sprache."""
    return get_parser(sprache)  # type: ignore[arg-type]
