"""MCP-Werkzeug-Extraktor: welche Werkzeuge eine Datei als MCP-Server anbietet (per ast).

Übernommen aus Vibemind_V1 scripts/doku_loop/extraktoren/mcp_werkzeuge.py @ 96b05d24; der Doku-Loop behält bis Stufe 4 seine Fassung.

Beide Muster des Doku-Loops gelten jetzt immer, ohne Space-Konfiguration:

``decorator``: `@mcp.tool` / `@mcp.tool()` / `@server.tool(...)` an einer Funktion, `Tool(name="x", ...)`
    und `register_tool("x", ...)` als Aufruf. Werkzeugname = `name=`-Argument (bzw. erstes
    Stringargument bei `register_tool`), sonst der Funktionsname.
``werkzeuge``: Modul-Literal `WERKZEUGE = (f1, f2, ...)`. Nackte Namen gelten der Funktion in derselben
    Datei; `modul.name` zeigt auf eine andere Datei und ist pro Blob nicht aufloesbar, dort haengt die
    Registrierung an der Datei, die das Literal enthaelt.
"""
from __future__ import annotations

import ast

from selbstbild.spezial._gemeinsam import Def, ausschnitt_digest, beleg, quelle_id
from selbstbild.typen import Einstieg, Kante, Knoten, id_datei, id_mcp, id_symbol

_SERVER_OBJEKTE = {"mcp", "server"}
_LISTE = "WERKZEUGE"


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


def mcp_werkzeuge(
    repo: str, pfad: str, baum: ast.Module, je_knoten: dict[int, str], defs: list[Def], text: str,
) -> tuple[tuple[Knoten, ...], tuple[Kante, ...], tuple[Einstieg, ...]]:
    # (Zeile, Name, Quelle-Id, Symbol-qualname fuer den Einstieg, Digest des Ausschnitts)
    roh: list[tuple[int, str, str, str | None, str]] = []
    for k in ast.walk(baum):
        if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in k.decorator_list:
                ist, call = _tool_deko(d)
                if ist:
                    qual = je_knoten.get(id(k))
                    name = (_kw_name(call) if call else None) or k.name
                    von = id_symbol(repo, pfad, qual) if qual else id_datei(repo, pfad)
                    roh.append((k.lineno, name, von, qual, ausschnitt_digest(text, d)))
        elif isinstance(k, ast.Call):
            f = k.func
            fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            name: str | None = None
            if fname == "Tool":
                name = _kw_name(k)
            elif fname == "register_tool":
                name = _str(k.args[0]) if k.args else _kw_name(k)
            if name:
                roh.append((k.lineno, name, quelle_id(repo, pfad, defs, k.lineno), None,
                            ausschnitt_digest(text, k)))

    top = {k.name: k.lineno for k in baum.body if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for e in _eintraege(baum):
        d_e = ausschnitt_digest(text, e)
        if isinstance(e, ast.Name):
            if e.id in top:
                roh.append((top[e.id], e.id, id_symbol(repo, pfad, e.id), e.id, d_e))
            else:
                roh.append((e.lineno, e.id, id_datei(repo, pfad), None, d_e))
        elif isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            roh.append((e.lineno, e.attr, id_datei(repo, pfad), None, d_e))

    knoten: dict[str, Knoten] = {}
    kanten: dict[tuple[str, str], Kante] = {}
    einstiege: list[Einstieg] = []
    for zeile, name, von, qual, digest in sorted(roh, key=lambda x: (x[0], x[1], x[2], x[4])):
        wid = id_mcp(repo, name)
        b = beleg(repo, pfad, zeile, name, digest)
        knoten.setdefault(wid, Knoten(wid, "mcp_werkzeug", name, b))
        kanten.setdefault((von, wid), Kante("registriert", von, wid, True, b))
        if qual is not None:
            einstiege.append(Einstieg("mcp", qual, zeile))
    return tuple(knoten.values()), tuple(kanten.values()), tuple(einstiege)
