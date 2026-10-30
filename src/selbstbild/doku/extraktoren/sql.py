"""SQL-Extraktor: CREATE-Definitionen aus .sql-Dateien, schema-qualifizierte Bezuege aus Python-Literalen."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .. import korpus
from ..konfig import Konfig, SpaceKonfig
from .routen import _korpus_relativ, _repo_relativ

if TYPE_CHECKING:
    from .._manifest_gate2 import Manifest

STANDARD_SCHEMATA = ("marketing", "sales", "public")

_IDENT = r'(?:"[^"]+"|[A-Za-z_]\w*)'
_CREATE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:(?:MATERIALIZED|UNLOGGED|TEMP|TEMPORARY|RECURSIVE)\s+)*"
    r"(TABLE|FUNCTION|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    rf"({_IDENT})\s*\.\s*({_IDENT})",
    re.IGNORECASE,
)
_DOLLAR = re.compile(r"\$(?:[A-Za-z_]\w*)?\$")
_NUMMER = re.compile(r"^(\d+)")


@dataclass(frozen=True)
class SqlDefinition:
    datei: str  # repo-relativ, POSIX
    zeile: int
    name: str  # schema.name, klein
    art: str  # table | function | view
    digest: str


@dataclass(frozen=True)
class SqlBezug:
    datei: str  # repo-relativ, POSIX
    zeile: int
    name: str
    symbol: str | None


def _maskieren(text: str) -> str:
    """Kopie gleicher Laenge: Kommentare, '...'-Strings und Dollar-Quotes durch Leerzeichen ersetzt."""
    aus = list(text)
    n, i = len(text), 0

    def leeren(von: int, bis: int) -> None:
        for k in range(von, min(bis, n)):
            if aus[k] != "\n":
                aus[k] = " "

    while i < n:
        c = text[i]
        if c == "-" and text.startswith("--", i):
            ende = text.find("\n", i)
            ende = n if ende < 0 else ende
            leeren(i, ende)
            i = ende
        elif c == "/" and text.startswith("/*", i):
            ende = text.find("*/", i + 2)
            ende = n if ende < 0 else ende + 2
            leeren(i, ende)
            i = ende
        elif c == "'":
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if text.startswith("''", j):
                        j += 2
                        continue
                    break
                j += 1
            leeren(i, j + 1)
            i = j + 1
        elif c == "$" and (m := _DOLLAR.match(text, i)):
            ende = text.find(m.group(0), m.end())
            ende = n if ende < 0 else ende + len(m.group(0))
            leeren(i, ende)
            i = ende
        elif c == '"':  # Bezeichner bleiben lesbar, aber ihr Inhalt darf kein ; / -- verwirren
            j = text.find('"', i + 1)
            i = n if j < 0 else j + 1
        else:
            i += 1
    return "".join(aus)


def _anweisung(text: str, start: int, maske: str | None = None) -> str:
    """Text ab `start` bis einschliesslich des ersten ; ausserhalb von Quotes/Kommentaren."""
    maske = maske if maske is not None else _maskieren(text)
    ende = maske.find(";", start)
    return text[start:] if ende < 0 else text[start:ende + 1]


def _entquoten(teil: str) -> str:
    return (teil[1:-1] if teil.startswith('"') else teil).lower()


def _nummer(datei: str) -> int:
    m = _NUMMER.match(datei.rsplit("/", 1)[-1])
    return int(m.group(1)) if m else 0


def _schluessel(d: SqlDefinition) -> tuple[int, str]:
    return _nummer(d.datei), d.datei


def sql_definitionen(konfig: Konfig) -> tuple[dict[str, SqlDefinition], list[str]]:
    """(Definitionen je Name, Namen mit mehreren Definitionen). Hoehere Migrationsnummer gewinnt."""
    alle: dict[str, list[SqlDefinition]] = {}
    for wurzel in konfig.sql_wurzeln:
        for pfad in korpus.dateien(konfig.repo / wurzel, konfig, (".sql",)):
            try:
                text = pfad.read_bytes().decode("utf-8").replace("\r\n", "\n")
            except (UnicodeDecodeError, OSError):
                continue
            datei = _repo_relativ(pfad, konfig.repo)
            maske = _maskieren(text)
            for m in _CREATE.finditer(maske):
                name = f"{_entquoten(m.group(2))}.{_entquoten(m.group(3))}"
                digest = hashlib.sha256(_anweisung(text, m.start(), maske).encode("utf-8")).hexdigest()
                alle.setdefault(name, []).append(SqlDefinition(
                    datei, text.count("\n", 0, m.start()) + 1, name, m.group(1).lower(), digest))
    ergebnis: dict[str, SqlDefinition] = {}
    doppelte: list[str] = []
    for name in sorted(alle):
        kandidaten = alle[name]
        ergebnis[name] = max(kandidaten, key=_schluessel)
        if len(kandidaten) > 1:
            doppelte.append(name)
    return ergebnis, doppelte


def _docstring_knoten(baum: ast.AST) -> set[int]:
    ids: set[int] = set()
    for k in ast.walk(baum):
        if isinstance(k, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            erst = k.body[0] if k.body else None
            if (isinstance(erst, ast.Expr) and isinstance(erst.value, ast.Constant)
                    and isinstance(erst.value.value, str)):
                ids.add(id(erst.value))
    return ids


def sql_bezuege(
    space: SpaceKonfig,
    konfig: Konfig,
    manifest: "Manifest",
    schemata: tuple[str, ...] = STANDARD_SCHEMATA,
) -> list[SqlBezug]:
    wurzel = konfig.repo / space.wurzel
    muster = re.compile(
        r"(?<![\w.@/:-])(" + "|".join(re.escape(s) for s in schemata) + r")\.([A-Za-z_]\w*)(?!\w|\.\w)",
        re.IGNORECASE,
    )
    ergebnis: list[SqlBezug] = []
    for pfad in korpus.dateien(wurzel, konfig, (".py",)):
        baum = korpus.ast_lesen(pfad)
        if baum is None:
            continue
        docs = _docstring_knoten(baum)
        datei = _repo_relativ(pfad, konfig.repo)
        korpus_pfad = _korpus_relativ(pfad, manifest, wurzel, konfig.repo)
        gesehen: set[tuple[int, str]] = set()
        for k in ast.walk(baum):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)) or id(k) in docs:
                continue
            for m in muster.finditer(k.value):
                name = f"{m.group(1).lower()}.{m.group(2).lower()}"
                zeile = k.lineno + k.value.count("\n", 0, m.start())
                if (zeile, name) in gesehen:
                    continue
                gesehen.add((zeile, name))
                doc = korpus.symbol_an(manifest, korpus_pfad, zeile)
                ergebnis.append(SqlBezug(datei, zeile, name, doc.document_id if doc else None))
    ergebnis.sort(key=lambda b: (b.datei, b.zeile, b.name))
    return ergebnis
