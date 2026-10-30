"""MCP-Werkzeug-Extraktor: welche Werkzeuge ein Space als MCP-Server anbietet (per ast).

Muster je `SpaceKonfig.mcp_muster`:

``decorator`` (brain): `@mcp.tool` / `@mcp.tool()` / `@server.tool(...)` an einer Funktion,
    `Tool(name="x", ...)` und `register_tool("x", ...)` als Aufruf. Werkzeugname = `name=`-Argument
    (bzw. erstes Stringargument bei `register_tool`), sonst der Funktionsname.
    Beispiel: vibemind-os/brain/the_brain/core/mcp_stdio_proxy.py (`@mcp.tool()` an `think`).

``werkzeuge`` (marketing, sales-claw): die Werkzeuge stehen in einem Modul-Literal
    `WERKZEUGE = (f1, f2, ...)`, das der Server beim Start per `mcp.tool()(fn)` bzw.
    `server.tool()(im_thread(fn))` registriert (keine Dekoratoren an den Funktionen):
      - vibemind-os/spaces/marketing/claw/server.py: Eintraege `werkzeuge.versand_beauftragen`
        (Attribut eines Moduls; die Funktion steht in `claw/werkzeuge.py`).
      - vibemind-os/spaces/sales-claw/sales-mcp/server.py: nackte Namen
        (`entwurf_freigeben`, `versandauftrag_uebernehmen`, ...), Funktionen im selben Modul.
    Werkzeugname = Funktionsname. `datei`/`zeile` zeigen auf die Definition, wenn auffindbar
    (Name: gleiche Datei; `modul.name`: `modul.py` neben dem Server), sonst auf den Eintrag im Literal.

`datei` ist repo-relativ (POSIX), `symbol` die umschliessende Manifest-`document_id` oder None.
Testdateien liefern nichts; unparsbare Dateien werden uebersprungen.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .. import korpus
from ..konfig import Konfig, SpaceKonfig
from .routen import _korpus_relativ, _repo_relativ

if TYPE_CHECKING:
    from .._manifest_gate2 import Manifest

_SERVER_OBJEKTE = {"mcp", "server"}
_LISTE = "WERKZEUGE"


@dataclass(frozen=True)
class McpWerkzeug:
    datei: str  # repo-relativ, POSIX
    zeile: int
    name: str
    symbol: str | None


def _str(k: ast.AST | None) -> str | None:
    if isinstance(k, ast.Constant) and isinstance(k.value, str):
        return k.value
    return None


def _tool_deko(d: ast.expr) -> tuple[bool, ast.Call | None]:
    """(ist Werkzeug-Dekorator, Call oder None bei nacktem `@mcp.tool`)."""
    ziel = d.func if isinstance(d, ast.Call) else d
    if (
        isinstance(ziel, ast.Attribute) and ziel.attr == "tool"
        and isinstance(ziel.value, ast.Name) and ziel.value.id in _SERVER_OBJEKTE
    ):
        return True, d if isinstance(d, ast.Call) else None
    return False, None


def _kw_name(c: ast.Call) -> str | None:
    return next((_str(k.value) for k in c.keywords if k.arg == "name"), None)


def _decorator(baum: ast.Module) -> list[tuple[int, str]]:
    ergebnis: list[tuple[int, str]] = []
    for k in ast.walk(baum):
        if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in k.decorator_list:
                ist, call = _tool_deko(d)
                if ist:
                    ergebnis.append((k.lineno, (_kw_name(call) if call else None) or k.name))
        elif isinstance(k, ast.Call):
            f = k.func
            fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if fname == "Tool":
                name = _kw_name(k)
                if name:
                    ergebnis.append((k.lineno, name))
            elif fname == "register_tool":
                name = _str(k.args[0]) if k.args else _kw_name(k)
                if name:
                    ergebnis.append((k.lineno, name))
    return ergebnis


def _top_defs(baum: ast.Module) -> dict[str, int]:
    return {
        k.name: k.lineno for k in baum.body
        if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _eintraege(baum: ast.Module) -> list[ast.expr]:
    """Elemente der `WERKZEUGE`-Literale auf Modulebene (Tuple/List, auch `+`-verkettet)."""
    aus: list[ast.expr] = []

    def lies(w: ast.expr | None) -> None:
        if isinstance(w, (ast.Tuple, ast.List)):
            aus.extend(w.elts)
        elif isinstance(w, ast.BinOp) and isinstance(w.op, ast.Add):
            lies(w.left)
            lies(w.right)

    for k in baum.body:
        if isinstance(k, ast.Assign) and any(
            isinstance(z, ast.Name) and z.id == _LISTE for z in k.targets
        ):
            lies(k.value)
        elif (
            isinstance(k, ast.AnnAssign) and isinstance(k.target, ast.Name)
            and k.target.id == _LISTE
        ):
            lies(k.value)
    return aus


def _werkzeuge_liste(p: Path, baum: ast.Module) -> list[tuple[Path, int, str]]:
    ergebnis: list[tuple[Path, int, str]] = []
    eigene = _top_defs(baum)
    fremde: dict[str, tuple[Path, dict[str, int]] | None] = {}
    for e in _eintraege(baum):
        if isinstance(e, ast.Name):
            ergebnis.append((p, eigene.get(e.id, e.lineno), e.id))
        elif isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            modul = e.value.id
            if modul not in fremde:
                q = p.with_name(f"{modul}.py")
                b = korpus.ast_lesen(q) if q.is_file() else None
                fremde[modul] = (q, _top_defs(b)) if b is not None else None
            treffer = fremde[modul]
            if treffer is not None and e.attr in treffer[1]:
                ergebnis.append((treffer[0], treffer[1][e.attr], e.attr))
            else:
                ergebnis.append((p, e.lineno, e.attr))
    return ergebnis


def mcp_werkzeuge(space: SpaceKonfig, konfig: Konfig, manifest: "Manifest") -> list[McpWerkzeug]:
    wurzel = konfig.repo / space.wurzel
    roh: list[tuple[Path, int, str]] = []
    for p in korpus.dateien(wurzel, konfig, (".py",)):
        if korpus.ist_testdatei(_repo_relativ(p, konfig.repo)):
            continue
        baum = korpus.ast_lesen(p)
        if baum is None:
            continue
        try:
            if space.mcp_muster == "decorator":
                roh.extend((p, z, n) for z, n in _decorator(baum))
            else:
                roh.extend(_werkzeuge_liste(p, baum))
        except RecursionError:
            continue
    gesehen: set[tuple[Path, int, str]] = set()
    ergebnis: list[McpWerkzeug] = []
    for p, zeile, name in roh:
        if (p, zeile, name) in gesehen:
            continue
        gesehen.add((p, zeile, name))
        kpfad = _korpus_relativ(p, manifest, wurzel, konfig.repo)
        doc = korpus.symbol_an(manifest, kpfad, zeile)
        ergebnis.append(McpWerkzeug(
            datei=_repo_relativ(p, konfig.repo), zeile=zeile, name=name,
            symbol=doc.document_id if doc else None,
        ))
    ergebnis.sort(key=lambda x: (x.datei, x.zeile, x.name))
    return ergebnis
