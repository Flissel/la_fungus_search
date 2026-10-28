"""HTTP-Client-Extraktor: requests/httpx/urllib-Aufrufe per ast, Env-Namen dateiintern verfolgt.

Übernommen aus Vibemind_V1 scripts/doku_loop/extraktoren/http_clients.py @ 96b05d24; der Doku-Loop behält bis Stufe 4 seine Fassung.

Anpassung: arbeitet auf einem ast-Baum statt Dateisystem. Das Ziel eines Aufrufs ist der Zielrepo-unabhaengige
Route-Literal (normalisierter Pfad) plus der Env-Name der Basis-URL. Ausgegeben wird je Env-Name eine Kante
`nutzt_dienst` vom aufrufenden Symbol zu `env:<NAME>`; Methode und Pfad stehen im Beleg-Anker. Aufrufe ohne
Env-Namen (feste Basis-URL) liefern keine Kante: ein Zielrepo ist pro Blob nicht bekannt.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Union

from selbstbild.spezial._gemeinsam import Def, beleg, quelle_id
from selbstbild.spezial.routen import normalisieren
from selbstbild.typen import Kante, Knoten, id_env

_VERBEN = {"get", "post", "put", "delete", "patch", "head", "options"}
_MODULE = {"requests", "httpx"}
_CLIENT_KLASSEN = {"Client", "AsyncClient", "Session"}
_STR_METHODEN = {"rstrip", "lstrip", "strip", "lower"}
_ENV_OBJEKTE = {"os.environ", "environ"}
_MAX_TIEFE = 6
_PH = "\x00"
_PROZENT = re.compile(r"%(?:\([^)]*\))?[sdrif]")
_KLAMMER = re.compile(r"\{[^{}]*\}")
_HOST = re.compile(r"^[A-Za-z][\w+.-]*://[^/\x00]+(/.*)?$", re.S)

Teil = Union[str, frozenset]



@dataclass
class _Scope:
    parent: "_Scope | None"
    klasse: bool = False
    zuweisungen: dict[str, list[tuple[int, ast.expr]]] = field(default_factory=dict)


@dataclass
class _Roh:
    zeile: int
    methode: str | None
    teile: list[Teil]


def _schluessel(ziel: ast.AST) -> str | None:
    if isinstance(ziel, ast.Name):
        return ziel.id
    if isinstance(ziel, ast.Attribute) and isinstance(ziel.value, ast.Name):
        return f"{ziel.value.id}.{ziel.attr}"
    return None


def _str(knoten: ast.AST | None) -> str | None:
    if isinstance(knoten, ast.Constant) and isinstance(knoten.value, str):
        return knoten.value
    return None


def _env_name(k: ast.AST) -> str | None:
    """Name aus os.getenv("X") / os.environ.get("X") / os.environ["X"], sonst None."""
    if isinstance(k, ast.Call) and k.args:
        f = k.func
        ist_getenv = (
            isinstance(f, ast.Attribute) and f.attr == "getenv"
            and isinstance(f.value, ast.Name) and f.value.id == "os"
        ) or (isinstance(f, ast.Name) and f.id == "getenv")
        ist_get = (
            isinstance(f, ast.Attribute) and f.attr == "get"
            and ast.unparse(f.value) in _ENV_OBJEKTE
        )
        if ist_getenv or ist_get:
            return _str(k.args[0])
    if isinstance(k, ast.Subscript) and ast.unparse(k.value) in _ENV_OBJEKTE:
        return _str(k.slice)
    return None


def _kw(c: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in c.keywords if k.arg == name), None)


class _Datei:
    def __init__(self, baum: ast.Module) -> None:
        self.baum = baum
        self.modul = _Scope(None)
        self.scope_von: dict[int, _Scope] = {}
        self.funktionen: dict[str, list[tuple[ast.expr, _Scope]]] = {}
        self.clients: dict[str, ast.expr | None] = {}  # Client-Variable -> base_url-Ausdruck
        self.request_namen: set[str] = set()  # Variablen, die eine urllib-Request halten
        self.bare_request = False
        self.request_module: set[str] = {"urllib.request", "request"}
        self._sammeln(baum, self.modul)

    # --- Aufbau -----------------------------------------------------------------
    def client_ctor(self, wert: ast.AST | None) -> ast.Call | None:
        if (
            isinstance(wert, ast.Call) and isinstance(wert.func, ast.Attribute)
            and wert.func.attr in _CLIENT_KLASSEN and isinstance(wert.func.value, ast.Name)
            and wert.func.value.id in _MODULE
        ):
            return wert
        return None

    def ist_request(self, c: ast.AST) -> bool:
        if not isinstance(c, ast.Call):
            return False
        f = c.func
        if isinstance(f, ast.Attribute) and f.attr == "Request":
            return ast.unparse(f.value) in self.request_module
        return isinstance(f, ast.Name) and f.id == "Request" and self.bare_request

    def _merke(self, schluessel: str | None, zeile: int, wert: ast.expr, scope: _Scope) -> None:
        if schluessel is None:
            return
        ziel = scope
        if "." in schluessel:  # self.x gilt in der umschliessenden Klasse (sonst dateiweit)
            while ziel.parent is not None and not ziel.klasse:
                ziel = ziel.parent
        ziel.zuweisungen.setdefault(schluessel, []).append((zeile, wert))
        ctor = self.client_ctor(wert)
        if ctor is not None:
            self.clients[schluessel] = _kw(ctor, "base_url")
        if self.ist_request(wert):
            self.request_namen.add(schluessel)

    def _sammeln(self, knoten: ast.AST, scope: _Scope) -> None:
        for k in ast.iter_child_nodes(knoten):
            if isinstance(k, ast.ImportFrom) and k.module == "urllib.request":
                if any(n.name == "Request" for n in k.names):
                    self.bare_request = True
            if isinstance(k, ast.ImportFrom) and k.module == "urllib":
                for n in k.names:
                    if n.name == "request":
                        self.request_module.add(n.asname or n.name)
            if isinstance(k, ast.Import):
                for n in k.names:
                    if n.name == "urllib.request" and n.asname:
                        self.request_module.add(n.asname)
            if isinstance(k, ast.ClassDef):
                neu_k = _Scope(scope, klasse=True)
                self.scope_von[id(k)] = neu_k
                self._sammeln(k, neu_k)
                continue
            if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef)):
                neu = _Scope(scope)
                self.scope_von[id(k)] = neu
                for r in ast.walk(k):
                    if isinstance(r, ast.Return) and r.value is not None:
                        self.funktionen.setdefault(k.name, []).append((r.value, neu))
                self._sammeln(k, neu)
                continue
            if isinstance(k, ast.Assign):
                for z in k.targets:
                    self._merke(_schluessel(z), k.lineno, k.value, scope)
            elif isinstance(k, ast.AnnAssign) and k.value is not None:
                self._merke(_schluessel(k.target), k.lineno, k.value, scope)
            elif isinstance(k, (ast.With, ast.AsyncWith)):
                for item in k.items:
                    if item.optional_vars is not None:
                        self._merke(
                            _schluessel(item.optional_vars), k.lineno, item.context_expr, scope
                        )
            self._sammeln(k, scope)

    # --- Aufloesung -------------------------------------------------------------
    def kandidaten(self, schluessel: str, scope: _Scope, zeile: int) -> list[ast.expr]:
        s: _Scope | None = scope
        while s is not None:
            eintraege = s.zuweisungen.get(schluessel)
            if eintraege:
                if "." in schluessel:
                    return [e for _, e in eintraege]
                davor = [e for z, e in eintraege if z <= zeile or s is not scope]
                return [davor[-1]] if davor else []
            s = s.parent
        return []

    def envs(self, ausdruck: ast.AST, scope: _Scope, zeile: int, tiefe: int = 0) -> set[str]:
        if tiefe > _MAX_TIEFE:
            return set()
        ergebnis: set[str] = set()
        for k in ast.walk(ausdruck):
            name = _env_name(k)
            if name:
                ergebnis.add(name)
            schluessel = None
            if isinstance(k, ast.Name):
                schluessel = k.id
            elif isinstance(k, ast.Attribute):
                schluessel = _schluessel(k)
            if schluessel:
                for e in self.kandidaten(schluessel, scope, zeile):
                    if e is not ausdruck:
                        ergebnis |= self.envs(e, scope, zeile, tiefe + 1)
            if isinstance(k, ast.Call):
                fn = k.func.id if isinstance(k.func, ast.Name) else (
                    k.func.attr if isinstance(k.func, ast.Attribute) else None)
                if fn in self.funktionen:
                    for r, s in self.funktionen[fn]:
                        ergebnis |= self.envs(r, s, r.lineno, tiefe + 1)
        return ergebnis

    def teile(self, a: ast.expr, scope: _Scope, zeile: int, tiefe: int = 0) -> list[Teil]:
        if tiefe > _MAX_TIEFE:
            return [frozenset()]
        s = _str(a)
        if s is not None:
            return [s]
        if isinstance(a, ast.JoinedStr):
            aus: list[Teil] = []
            for v in a.values:
                if isinstance(v, ast.Constant):
                    aus.append(str(v.value))
                elif isinstance(v, ast.FormattedValue):
                    aus.extend(self._eingesetzt(v.value, scope, zeile, tiefe))
            return aus
        if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add):
            return (self.teile(a.left, scope, zeile, tiefe + 1)
                    + self.teile(a.right, scope, zeile, tiefe + 1))
        if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Mod) and _str(a.left) is not None:
            rechts = list(a.right.elts) if isinstance(a.right, ast.Tuple) else [a.right]
            return self._vorlage(_PROZENT, _str(a.left) or "", rechts, scope, zeile, tiefe)
        if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute):
            if a.func.attr == "format" and _str(a.func.value) is not None:
                args = list(a.args) + [k.value for k in a.keywords]
                return self._vorlage(_KLAMMER, _str(a.func.value) or "", args, scope, zeile, tiefe)
            if a.func.attr in _STR_METHODEN:
                return self.teile(a.func.value, scope, zeile, tiefe + 1)
        if isinstance(a, (ast.Name, ast.Attribute)):
            schluessel = _schluessel(a)
            kand = self.kandidaten(schluessel, scope, zeile) if schluessel else []
            if len(kand) == 1 and _env_name(kand[0]) is None:
                return self.teile(kand[0], scope, zeile, tiefe + 1)
        return [frozenset(self.envs(a, scope, zeile))]

    def _eingesetzt(self, a: ast.expr, scope: _Scope, zeile: int, tiefe: int) -> list[Teil]:
        """Eingesetzter Ausdruck: Basis-URL/Pfade inline lesen, alles andere ist ein Platzhalter."""
        inner = self.teile(a, scope, zeile, tiefe + 1)
        if any(isinstance(t, str) and "/" in t for t in inner):
            return inner
        envs: set[str] = set(self.envs(a, scope, zeile))
        for t in inner:
            if isinstance(t, frozenset):
                envs |= t
        return [frozenset(envs)]

    def _vorlage(self, muster: re.Pattern[str], text: str, args: list[ast.expr],
                 scope: _Scope, zeile: int, tiefe: int) -> list[Teil]:
        aus: list[Teil] = []
        pos = 0
        for i, m in enumerate(muster.finditer(text)):
            if m.start() > pos:
                aus.append(text[pos:m.start()])
            if i < len(args):
                aus.extend(self._eingesetzt(args[i], scope, zeile, tiefe))
            else:
                aus.append(frozenset())
            pos = m.end()
        if pos < len(text):
            aus.append(text[pos:])
        return aus


def _text(teile: list[Teil]) -> tuple[str, list[tuple[int, frozenset]]]:
    """Teile -> Text mit Platzhalter-Marken (ohne Query) und Env-Mengen je Platzhalter-Position."""
    text = ""
    phs: list[tuple[int, frozenset]] = []
    for t in teile:
        if isinstance(t, str):
            if "?" in t:
                text += t.split("?", 1)[0]
                break
            text += t
        else:
            phs.append((len(text), t))
            text += _PH
    return text, phs


def _pfad_start(text: str) -> int | None:
    erster = text.find(_PH)
    if erster < 0:
        m = _HOST.match(text)
        if m:
            return len(text) - len(m.group(1)) if m.group(1) else None
        return 0 if text.startswith("/") else None
    vor = text[:erster]
    m = _HOST.match(vor)
    if m and m.group(1):
        return len(vor) - len(m.group(1))
    start = text.find("/", erster)
    return start if start >= 0 else None


def _flach(teile: list[Teil]) -> tuple[str | None, tuple[str, ...]]:
    """(normalisierter Pfad | None, Env-Namen der Platzhalter vor dem Pfad = Basis-URL)."""
    text, phs = _text(teile)
    start = _pfad_start(text)
    pfad = normalisieren(text[start:].replace(_PH, "{}")) if start is not None else None
    envs: set[str] = set()
    if start is None:  # kein Pfad erkennbar: nur der erste Platzhalter gilt als Basis-URL
        phs = phs[:1]
    for pos, e in phs:
        if start is None or pos < start:
            envs |= e
    return pfad, tuple(sorted(envs))


def _daten_methode(c: ast.Call, pos: int) -> str:
    daten = c.args[pos] if len(c.args) > pos else _kw(c, "data")
    if daten is None or (isinstance(daten, ast.Constant) and daten.value is None):
        return "GET"
    return "POST"


def _methode_arg(wert: ast.expr | None) -> str | None:
    s = _str(wert)
    return s.upper() if s else None


def _mit_basis(d: _Datei, basis: ast.expr | None, url: ast.expr, scope: _Scope,
               zeile: int) -> list[Teil]:
    teile = d.teile(url, scope, zeile)
    if basis is not None:
        return d.teile(basis, scope, zeile) + teile
    return teile


def _bewerte(d: _Datei, c: ast.Call, scope: _Scope) -> _Roh | None:
    f = c.func
    if isinstance(f, ast.Attribute):
        empfaenger = f.value
        ist_modul = isinstance(empfaenger, ast.Name) and empfaenger.id in _MODULE
        schl = _schluessel(empfaenger)
        basis: ast.expr | None = None
        ist_client = False
        if schl is not None and schl in d.clients:
            ist_client, basis = True, d.clients[schl]
        elif d.client_ctor(empfaenger) is not None:
            ist_client = True
            basis = _kw(empfaenger, "base_url")  # type: ignore[arg-type]
        if ist_modul or ist_client:
            if f.attr in _VERBEN:
                url = c.args[0] if c.args else _kw(c, "url")
                if url is None:
                    return None
                return _Roh(c.lineno, f.attr.upper(), _mit_basis(d, basis, url, scope, c.lineno))
            if f.attr in {"request", "stream"}:
                meth = c.args[0] if c.args else _kw(c, "method")
                url = c.args[1] if len(c.args) >= 2 else _kw(c, "url")
                if url is not None:
                    return _Roh(c.lineno, _methode_arg(meth),
                                _mit_basis(d, basis, url, scope, c.lineno))
    if d.ist_request(c):
        url = c.args[0] if c.args else _kw(c, "url")
        if url is None:
            return None
        meth = _kw(c, "method")
        methode = _methode_arg(meth) if meth is not None else _daten_methode(c, 1)
        return _Roh(c.lineno, methode, d.teile(url, scope, c.lineno))
    ist_urlopen = (isinstance(f, ast.Attribute) and f.attr == "urlopen") or (
        isinstance(f, ast.Name) and f.id == "urlopen")
    if ist_urlopen and c.args:
        a = c.args[0]
        if d.ist_request(a) or _schluessel(a) in d.request_namen:
            return None  # Request-Aufruf wird dort schon gezaehlt
        return _Roh(c.lineno, _daten_methode(c, 1), d.teile(a, scope, c.lineno))
    return None


def _aufrufe_in(d: _Datei) -> list[_Roh]:
    roh: list[_Roh] = []

    def besuche(knoten: ast.AST, scope: _Scope) -> None:
        for k in ast.iter_child_nodes(knoten):
            if isinstance(k, ast.Call):
                r = _bewerte(d, k, scope)
                if r is not None:
                    roh.append(r)
            inner = d.scope_von.get(id(k), scope)
            besuche(k, inner)

    besuche(d.baum, d.modul)
    return roh



def http_aufrufe(
    repo: str, pfad: str, baum: ast.Module, defs: list[Def],
) -> tuple[tuple[Knoten, ...], tuple[Kante, ...]]:
    try:
        rohe = _aufrufe_in(_Datei(baum))
    except RecursionError:
        return (), ()
    knoten: dict[str, Knoten] = {}
    kanten: dict[tuple[str, str, int, str], Kante] = {}
    for r in sorted(rohe, key=lambda x: (x.zeile, x.methode or "")):
        pfad_literal, envs = _flach(r.teile)
        anker = f"{r.methode or '*'} {pfad_literal or ''}".rstrip()
        b = beleg(repo, pfad, r.zeile, anker)
        von = quelle_id(repo, pfad, defs, r.zeile)
        for name in envs:
            eid = id_env(name)
            knoten.setdefault(eid, Knoten(eid, "env", name, None))
            kanten.setdefault((von, eid, r.zeile, anker), Kante("nutzt_dienst", von, eid, True, b))
    return tuple(knoten.values()), tuple(kanten.values())
