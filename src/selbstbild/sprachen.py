"""Sprachtabelle: Dateipfad -> Sprache, Sprache -> tree-sitter-Parser.

Gepinnt: tree-sitter==0.26.0, tree-sitter-language-pack==1.22.1.
PowerShell: Das Language-Pack liefert einen Parser fuer "powershell"; Task 7
kann ihn nutzen (Regex nur als Rueckfall).

Grammatiken: Das Language-Pack laedt Grammatik-DLLs beim ersten `get_parser`
selbst nach (Netz). Das ist hier abgeschaltet: `parser_fuer` prueft vorher
`downloaded_languages()` und wirft RuntimeError, wenn die Grammatik fehlt.
Einmalig beim Setup `grammatiken_bereitstellen()` aufrufen (einziger Netzzugriff).
"""
from __future__ import annotations

import posixpath
from functools import lru_cache

import tree_sitter
import tree_sitter_language_pack as _pack

PARSER_VERSION: int = 1
GRAMMATIK_VERSION: str = "1.22.1"  # = gepinnte tree-sitter-language-pack-Version

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
    if name == ".env":
        return None
    if name.startswith(".env."):
        return "dotenv"
    return _ENDUNG_ZU_SPRACHE.get(posixpath.splitext(name)[1])


def grammatiken_bereitstellen() -> None:
    """Laedt alle von SPRACHEN benoetigten Grammatiken vor (einmalig beim Setup)."""
    _pack.download(sorted(SPRACHEN))


@lru_cache(maxsize=None)
def parser_fuer(sprache: str) -> tree_sitter.Parser:
    """Gecachter tree-sitter-Parser; laedt nie selbst nach."""
    if sprache not in _pack.downloaded_languages():
        raise RuntimeError(
            f"Grammatik '{sprache}' fehlt im Cache; einmalig "
            "selbstbild.sprachen.grammatiken_bereitstellen() ausfuehren."
        )
    return _pack.get_parser(sprache)  # type: ignore[arg-type]
