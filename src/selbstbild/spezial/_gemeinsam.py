"""Gemeinsame Hilfen der Spezialextraktoren: ast lesen, Definitionen, Belege."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import warnings
from dataclasses import dataclass

from selbstbild.typen import Beleg, id_datei, id_symbol


@dataclass(frozen=True)
class Def:
    qualname: str
    von: int
    bis: int


def baum_lesen(quelle: bytes) -> ast.Module | None:
    """ast-Baum oder None bei Syntaxfehlern und kaputten Bytes."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # SyntaxWarning fremden Codes (z. B. '\\d')
            return ast.parse(quelle)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def definitionen(baum: ast.Module) -> tuple[dict[int, str], list[Def]]:
    """(id(Knoten) -> qualname, alle Definitionen). Qualnamen wie in parser/python.py."""
    je_knoten: dict[int, str] = {}
    liste: list[Def] = []

    def besuche(knoten: ast.AST, scope: str | None) -> None:
        for k in ast.iter_child_nodes(knoten):
            if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = k.name if scope is None else f"{scope}.{k.name}"
                je_knoten[id(k)] = qual
                liste.append(Def(qual, k.lineno, k.end_lineno or k.lineno))
                besuche(k, qual)
            else:
                besuche(k, scope)

    besuche(baum, None)
    return je_knoten, liste


def umschliessend(defs: list[Def], zeile: int) -> str | None:
    """qualname der innersten Definition, die die Zeile enthaelt."""
    treffer = [d for d in defs if d.von <= zeile <= d.bis]
    if not treffer:
        return None
    return min(treffer, key=lambda d: (d.bis - d.von, -d.von)).qualname


def quelle_id(repo: str, pfad: str, defs: list[Def], zeile: int) -> str:
    """Symbol-Id der umschliessenden Definition, sonst die Datei-Id."""
    qual = umschliessend(defs, zeile)
    return id_symbol(repo, pfad, qual) if qual else id_datei(repo, pfad)


def text_digest(text: str) -> str:
    """sha256 eines Quellausschnitts, Zeilenenden normalisiert (CRLF/CR -> LF)."""
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(norm.encode("utf-8", "surrogateescape")).hexdigest()


def blob_sha(quelle: bytes) -> str:
    """Git-Blob-SHA der Datei (wie `git hash-object`)."""
    return hashlib.sha1(b"blob %d\0" % len(quelle) + quelle).hexdigest()


def quelltext(quelle: bytes) -> str:
    """Quelltext wie ihn `ast` liest (BOM, Kodierungszeile, universelle Zeilenenden)."""
    try:
        return importlib.util.decode_source(quelle)
    except (UnicodeDecodeError, SyntaxError, LookupError):
        return quelle.decode("utf-8", "replace")


def ausschnitt_digest(text: str, knoten: ast.AST) -> str:
    """R19: sha256 des Quellausschnitts eines AST-Knotens (CRLF normalisiert)."""
    teil = ast.get_source_segment(text, knoten)
    return text_digest(teil if teil is not None else ast.unparse(knoten))


def beleg(repo: str, pfad: str, zeile: int, anker: str, digest: str) -> Beleg:
    """Beleg mit `digest` = Digest der belegenden Quelle (R19), nie des Ankers."""
    return Beleg(repo, pfad, zeile, anker, digest)


def ist_testdatei(pfad: str) -> bool:
    """Testcode: Ordner `tests`/`test`, `test_*.py`, `*_test.py`, `conftest.py`."""
    teile = pfad.replace("\\", "/").split("/")
    if any(t in {"tests", "test"} for t in teile[:-1]):
        return True
    name = teile[-1]
    return name == "conftest.py" or (
        name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))
    )
