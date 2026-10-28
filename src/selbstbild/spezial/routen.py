"""Routen-Extraktor: FastAPI/Starlette/Flask-Dekoratoren per ast, Praefixe zusammengesetzt.

Übernommen aus Vibemind_V1 scripts/doku_loop/extraktoren/routen.py @ 96b05d24; der Doku-Loop behält bis Stufe 4 seine Fassung.

Anpassung: arbeitet auf einer einzelnen Datei (Bytes statt Dateisystem). Praefixe von
`APIRouter(prefix=...)`/`Blueprint(url_prefix=...)` und `include_router(...)` werden innerhalb
der Datei zusammengesetzt; dateiuebergreifende Includes sind pro Blob nicht aufloesbar.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from selbstbild.spezial._gemeinsam import Def, beleg
from selbstbild.typen import Einstieg, Kante, Knoten, id_route, id_symbol

GENERISCHE_PFADE = frozenset({"/", "/health", "/api/health", "/api/stats", "/healthz", "/ready"})
_METHODEN = {"get", "post", "put", "delete", "patch", "route", "api_route"}
_PARAM = re.compile(r"\{[^{}]*\}|<[^<>]*>")


def _objekt_passt(name: str) -> bool:
    return name in {"app", "router", "bp"} or name.endswith(("_router", "_bp"))


def normalisieren(pfad: str) -> str:
    pfad = pfad.split("?", 1)[0]
    pfad = _PARAM.sub("{}", pfad)
    if len(pfad) > 1:
        pfad = pfad.rstrip("/")
    return pfad or "/"


def _literal(knoten: ast.expr | None) -> str | None:
    if isinstance(knoten, ast.Constant) and isinstance(knoten.value, str):
        return knoten.value
    return None


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _letzter_name(knoten: ast.expr) -> str | None:
    if isinstance(knoten, ast.Name):
        return knoten.id
    if isinstance(knoten, ast.Attribute):
        return knoten.attr
    return None


def _methoden(attr: str, call: ast.Call) -> list[str]:
    if attr not in {"route", "api_route"}:
        return [attr.upper()]
    wert = _kw(call, "methods")
    if isinstance(wert, (ast.List, ast.Tuple, ast.Set)):
        namen = [_literal(e) for e in wert.elts]
        gueltig = [n.upper() for n in namen if n]
        if gueltig:
            return gueltig
    return ["GET"]


@dataclass
class _Roh:
    zeile: int
    methode: str
    pfad: str
    objekt: str
    qualname: str | None


def routen(
    repo: str, pfad: str, baum: ast.Module, je_knoten: dict[int, str], defs: list[Def],
) -> tuple[tuple[Knoten, ...], tuple[Kante, ...], tuple[Einstieg, ...]]:
    praefixe: dict[str, str] = {}
    includes: dict[str, list[str]] = {}
    roh: list[_Roh] = []
    for k in ast.walk(baum):
        wert: ast.expr | None = None
        ziele: list[ast.expr] = []
        if isinstance(k, ast.Assign):
            wert, ziele = k.value, list(k.targets)
        elif isinstance(k, ast.AnnAssign) and k.value is not None:
            wert, ziele = k.value, [k.target]
        if isinstance(wert, ast.Call) and _letzter_name(wert.func) in {"APIRouter", "Blueprint"}:
            p = _literal(_kw(wert, "prefix")) or _literal(_kw(wert, "url_prefix")) or ""
            for ziel in ziele:
                if isinstance(ziel, ast.Name):
                    praefixe[ziel.id] = p
        if isinstance(k, ast.Call) and isinstance(k.func, ast.Attribute):
            if k.func.attr == "include_router" and k.args and isinstance(k.args[0], ast.Name):
                liste = includes.setdefault(k.args[0].id, [])
                p = _literal(_kw(k, "prefix")) or ""
                if p not in liste:
                    liste.append(p)
        elif isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dek in k.decorator_list:
                if not (isinstance(dek, ast.Call) and isinstance(dek.func, ast.Attribute)):
                    continue
                if dek.func.attr not in _METHODEN or not isinstance(dek.func.value, ast.Name):
                    continue
                objekt = dek.func.value.id
                if not _objekt_passt(objekt):
                    continue
                arg = dek.args[0] if dek.args else _kw(dek, "path") or _kw(dek, "rule")
                p = _literal(arg)
                if p is None:
                    continue
                for m in _methoden(dek.func.attr, dek):
                    roh.append(_Roh(dek.lineno, m, p, objekt, je_knoten.get(id(k))))

    knoten: dict[str, Knoten] = {}
    kanten: list[Kante] = []
    einstiege: list[Einstieg] = []
    gesehen: set[tuple[int, str, str]] = set()
    for r in sorted(roh, key=lambda x: (x.zeile, x.methode, x.pfad)):
        eigen = praefixe.get(r.objekt, "")
        fremde = [""] if r.objekt == "app" else (includes.get(r.objekt) or [""])
        for fremd in fremde:
            voll = normalisieren(fremd + eigen + r.pfad)
            schluessel = (r.zeile, r.methode, voll)
            if schluessel in gesehen:
                continue
            gesehen.add(schluessel)
            rid = id_route(repo, r.methode, voll)
            anker = f"{r.methode} {voll}"
            b = beleg(repo, pfad, r.zeile, anker)
            attr = (("generisch", "1"),) if voll in GENERISCHE_PFADE else ()
            knoten.setdefault(rid, Knoten(rid, "route", anker, b, attr))
            if r.qualname is not None:
                kanten.append(Kante("definiert_route", id_symbol(repo, pfad, r.qualname), rid, True, b))
            einstiege.append(Einstieg("route", r.qualname, r.zeile))
    return tuple(knoten.values()), tuple(kanten), tuple(einstiege)
