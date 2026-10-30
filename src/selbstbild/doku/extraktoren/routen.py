"""Routen-Extraktor: FastAPI/Starlette/Flask-Dekoratoren per ast, Praefixe zusammengesetzt."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .. import korpus
from ..konfig import Konfig, SpaceKonfig

if TYPE_CHECKING:
    from .._manifest_gate2 import Manifest

GENERISCHE_PFADE = frozenset({"/", "/health", "/api/health", "/api/stats", "/healthz", "/ready"})
_METHODEN = {"get", "post", "put", "delete", "patch", "route", "api_route"}
_PARAM = re.compile(r"\{[^{}]*\}|<[^<>]*>")


@dataclass(frozen=True)
class Route:
    datei: str  # repo-relativ, POSIX
    zeile: int
    methode: str
    pfad: str
    generisch: bool
    symbol: str | None


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
    datei: Path
    zeile: int
    methode: str
    pfad: str
    objekt: str
    def_zeile: int


@dataclass
class _Datei:
    pfad: Path
    modul: tuple[str, ...]  # Pfad relativ zur Space-Wurzel, ohne .py / __init__
    paket: tuple[str, ...]
    roh: list[_Roh]
    praefixe: dict[str, str]  # Router-Variable -> eigener Praefix
    includes: list[tuple[ast.expr, str]]  # (Argument, Praefix)
    importe: dict[str, tuple[tuple[str, ...], str | None]]  # Alias -> (Modul, Name | None)
    uebersprungen: int


def _modul_von(pfad: Path, wurzel: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        teile = list(pfad.relative_to(wurzel).with_suffix("").parts)
    except ValueError:
        teile = [pfad.stem]
    paket = tuple(teile[:-1])
    if teile and teile[-1] == "__init__":
        teile = teile[:-1]
    return tuple(teile), paket


def _datei_lesen(pfad: Path, wurzel: Path) -> _Datei:
    modul, paket = _modul_von(pfad, wurzel)
    d = _Datei(pfad, modul, paket, [], {}, [], {}, 0)
    baum = korpus.ast_lesen(pfad)
    if baum is None:
        return d
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
                    d.praefixe[ziel.id] = p
        if isinstance(k, ast.ImportFrom):
            basis = paket[: max(len(paket) - (k.level - 1), 0)] if k.level else ()
            mod = basis + tuple((k.module or "").split("."))
            mod = tuple(t for t in mod if t)
            for n in k.names:
                d.importe[n.asname or n.name] = (mod, n.name)
        elif isinstance(k, ast.Import):
            for n in k.names:
                if n.asname:
                    d.importe[n.asname] = (tuple(n.name.split(".")), None)
                else:
                    erst = n.name.split(".")[0]
                    d.importe[erst] = ((erst,), None)
        elif isinstance(k, ast.Call) and isinstance(k.func, ast.Attribute):
            if k.func.attr == "include_router" and k.args:
                d.includes.append((k.args[0], _literal(_kw(k, "prefix")) or ""))
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
                    d.uebersprungen += 1
                    continue
                for m in _methoden(dek.func.attr, dek):
                    d.roh.append(_Roh(pfad, dek.lineno, m, p, objekt, k.lineno))
    return d


class _Aufloeser:
    """Include-Argument -> (definierende Datei, Router-Variable), nur innerhalb des Spaces."""

    def __init__(self, dateien: list[_Datei]) -> None:
        self.je_modul = {d.modul: d for d in dateien}

    def modul(self, mod: tuple[str, ...], von: _Datei) -> _Datei | None:
        if not mod:
            return None
        for kandidat in (mod, von.paket + mod):
            if kandidat in self.je_modul:
                return self.je_modul[kandidat]
        treffer = [
            d for m, d in self.je_modul.items()
            if len(m) >= len(mod) and m[len(m) - len(mod):] == mod
        ]
        if len(treffer) == 1:
            return treffer[0]
        if treffer:
            return None
        # Absoluter Import mit Repo-Praefix (spaces.marketing.api.bilder): laengstes Suffix, das ein Modul ist
        laenger = [
            (len(m), d) for m, d in self.je_modul.items()
            if m and len(m) < len(mod) and mod[len(mod) - len(m):] == m
        ]
        if not laenger:
            return None
        beste = max(n for n, _ in laenger)
        gleich = [d for n, d in laenger if n == beste]
        return gleich[0] if len(gleich) == 1 else None

    def variable(self, datei: _Datei, name: str, tiefe: int = 0) -> tuple[Path, str] | None:
        if name in datei.praefixe:
            return datei.pfad, name
        imp = datei.importe.get(name)
        if imp is None or imp[1] is None or tiefe >= 5:
            return None
        ziel = self.modul(imp[0], datei)
        return self.variable(ziel, imp[1], tiefe + 1) if ziel else None

    def aufloesen(self, arg: ast.expr, von: _Datei) -> tuple[Path, str] | None:
        if isinstance(arg, ast.Name):
            return self.variable(von, arg.id)
        if not isinstance(arg, ast.Attribute):
            return None
        kette: list[str] = [arg.attr]
        knoten = arg.value
        while isinstance(knoten, ast.Attribute):
            kette.insert(0, knoten.attr)
            knoten = knoten.value
        if not isinstance(knoten, ast.Name):
            return None
        imp = von.importe.get(knoten.id)
        if imp is None:
            return None
        mod, name = imp
        if name is not None:  # from pkg import b -> b ist Untermodul
            mod = mod + (name,)
        ziel = self.modul(mod + tuple(kette[:-1]), von)
        return self.variable(ziel, kette[-1]) if ziel else None


def _repo_relativ(pfad: Path, repo: Path) -> str:
    try:
        return pfad.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return pfad.as_posix()


def _korpus_relativ(pfad: Path, manifest: "Manifest", wurzel: Path, repo: Path) -> str:
    cr = Path(manifest.corpus_root)
    if not cr.is_absolute():
        cr = repo / cr
    for basis in (cr, wurzel):
        try:
            return pfad.resolve().relative_to(basis.resolve()).as_posix()
        except ValueError:
            continue
    return pfad.as_posix()


def routen_extrahieren_mit_zaehlung(
    space: SpaceKonfig, konfig: Konfig, manifest: "Manifest"
) -> tuple[list[Route], int, int]:
    """(Routen, uebersprungene nicht-literale Pfade, nicht aufloesbare include_router-Aufrufe)."""
    wurzel = konfig.repo / space.wurzel
    gelesen = [_datei_lesen(p, wurzel) for p in korpus.dateien(wurzel, konfig, (".py",))]
    uebersprungen = sum(g.uebersprungen for g in gelesen)
    aufloeser = _Aufloeser(gelesen)
    include_praefixe: dict[tuple[Path, str], list[str]] = defaultdict(list)
    unaufgeloest = 0
    for g in gelesen:
        for arg, p in g.includes:
            ziel = aufloeser.aufloesen(arg, g)
            if ziel is None:
                unaufgeloest += 1
            elif p not in include_praefixe[ziel]:
                include_praefixe[ziel].append(p)

    vorlaeufig: list[tuple[_Roh, str]] = []
    for g in gelesen:
        for r in g.roh:
            eigen = g.praefixe.get(r.objekt, "")
            if r.objekt == "app":
                fremde = [""]
            else:
                fremde = include_praefixe.get((g.pfad, r.objekt)) or [""]
            for fremd in fremde:
                vorlaeufig.append((r, normalisieren(fremd + eigen + r.pfad)))

    dateien_je_pfad: dict[str, set[Path]] = defaultdict(set)
    for r, pfad in vorlaeufig:
        dateien_je_pfad[pfad].add(r.datei)

    ergebnis: list[Route] = []
    gesehen: set[tuple[Path, int, str, str]] = set()
    # Symbol ueber die Zeile der Funktion (Dekorator liegt oft vor start_line der Definition)
    for r, pfad in vorlaeufig:
        schluessel = (r.datei, r.zeile, r.methode, pfad)
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)
        korpus_pfad = _korpus_relativ(r.datei, manifest, wurzel, konfig.repo)
        doc = korpus.symbol_an(manifest, korpus_pfad, r.def_zeile)
        ergebnis.append(Route(
            datei=_repo_relativ(r.datei, konfig.repo),
            zeile=r.zeile,
            methode=r.methode,
            pfad=pfad,
            generisch=pfad in GENERISCHE_PFADE or len(dateien_je_pfad[pfad]) > 1,
            symbol=doc.document_id if doc else None,
        ))
    ergebnis.sort(key=lambda x: (x.datei, x.zeile, x.methode, x.pfad))
    return ergebnis, uebersprungen, unaufgeloest


def routen_extrahieren(space: SpaceKonfig, konfig: Konfig, manifest: "Manifest") -> list[Route]:
    return routen_extrahieren_mit_zaehlung(space, konfig, manifest)[0]
