"""Toter Code in drei Stufen (Spec 6.2): sicher_tot, wahrscheinlich_tot, verdaechtig.

Ein falsches `sicher_tot` ist der teure Fehler (daraus werden Issues). Im Zweifel wird abgestuft."""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..quelle import Baum, BlobLeser, Eintrag, baum_lesen, blobs_mit_namen
from ..sprachen import SPRACHEN
from . import CODE_SPRACHEN, ist_generiert, ist_testdatei
from .erreichbar import Erreichbarkeit
from .lader import GraphDaten, datei_von

TOT_TAGE: int = 90
_SEKUNDEN = TOT_TAGE * 86400

Stufe = Literal["sicher_tot", "wahrscheinlich_tot", "verdaechtig"]

_TS_ENDUNGEN = tuple(sorted(e for s in ("typescript", "tsx", "javascript") for e in SPRACHEN[s]))

# Ruling S9: Namen, die Frameworks per Konvention aufrufen (http.server, ast/docutils-Visitor,
# unittest/pytest, Ereignis-Handler, Klassen-Hooks). Nie `sicher_tot`, Grund `konvention`.
# Fix-Welle: pytest-Hooks, alembic upgrade/downgrade, AWS lambda_handler, Django `handle`; Go `init`
# nur in .go-Dateien (siehe `_ausnahmen`).
_KONVENTION = re.compile(
    r"^(do_.*|visit_.*|depart_.*|generic_visit|on_.*|handle_.*|handle|setUp|tearDown|setUpClass|tearDownClass"
    r"|setup_method|teardown_method|__init_subclass__|__class_getitem__|pytest_.*|upgrade|downgrade"
    r"|lambda_handler)$")

# Ruling S10: Decorators, die keine Registrierung bedeuten. Ein dekoriertes Symbol ist nie `sicher_tot`
# (Grund `dekorator`), ausser ALLE seine Decorators stehen hier (`functools.*` als Praefix).
INERTE_DEKORATOREN = frozenset({
    "property", "staticmethod", "classmethod", "dataclass", "dataclasses.dataclass", "abstractmethod",
    "abc.abstractmethod", "overload", "typing.overload", "cached_property", "contextmanager",
    "contextlib.contextmanager", "wraps", "total_ordering", "final", "typing.final", "override"})
INERTE_PRAEFIXE: tuple[str, ...] = ("functools.",)


@dataclass(frozen=True)
class TotFund:
    symbol: str
    stufe: Stufe
    gruende: tuple[str, ...]
    zuletzt: int | None


def _attr(d: GraphDaten, i: str) -> dict[str, str]:
    k = d.knoten.get(i)
    return {} if k is None else dict(k.attr)


def _pfad(symbol_id: str) -> str:
    return symbol_id.split(":", 2)[2].rpartition("#")[0]


def _repo(symbol_id: str) -> str:
    return symbol_id.split(":", 2)[1]


def _dokumentiert(symbol_id: str, doku: Mapping[str, frozenset[str]]) -> bool:
    """Ruling S5: Doku des eigenen Repos oder eines Eltern-Repos (Namenspraefix an `/`-Grenze)."""
    repo = _repo(symbol_id)
    name = symbol_id.rpartition("#")[2].rpartition(".")[2]
    teile = repo.split("/")
    return any(name in doku.get("/".join(teile[:n]), frozenset()) for n in range(1, len(teile) + 1))


def _zuletzt(d: GraphDaten, symbol_id: str) -> int | None:
    a = _attr(d, datei_von(symbol_id))
    if a.get("aenderungen") == "0":
        return None  # keine Historie: Alter unbekannt
    try:
        z = None if a.get("zuletzt") is None else int(a["zuletzt"])
    except ValueError:
        return None  # unlesbar: nicht als alt werten (im Zweifel abstufen)
    return z if z is not None and z > 0 else None


def _inert(dekorator: str) -> bool:
    return dekorator in INERTE_DEKORATOREN or dekorator.startswith(INERTE_PRAEFIXE)


def _dekoriert_aktiv(d: GraphDaten, i: str) -> bool:
    """Hat das Symbol mindestens einen nicht inerten Decorator (Attr `dekoriert`, Ruling S10)?"""
    roh = _attr(d, i).get("dekoriert", "")
    return any(not _inert(x) for x in roh.split(",") if x)


def _s4(d: GraphDaten, i: str) -> set[str]:
    """Eigene Ausnahmen nach Ruling S4 und R22 (ohne Vererbung)."""
    erg: set[str] = set()
    pfad = _pfad(i)
    einfach = i.rpartition("#")[2].rpartition(".")[2]
    art = _attr(d, i).get("symbolart")
    if pfad.endswith(".rs"):
        erg.add("rust_r22")
    if pfad.endswith(".go") and art == "methode":
        erg.add("go_methode")
    if pfad.endswith(_TS_ENDUNGEN) and art in ("klasse", "methode"):
        erg.add("ts_klasse")
    if pfad.endswith(_TS_ENDUNGEN) and art == "funktion" and einfach[:1].isupper():
        erg.add("jsx_komponente")
    return erg


def _ausnahmen(d: GraphDaten, e: Erreichbarkeit, i: str) -> set[str]:
    """Gruende, aus denen ein Symbol nie `sicher_tot` sein darf (Global Constraints, E4, Ruling S4).

    Ruling S8: Verschachteltes (Qualname-Praefix in derselben Datei) erbt die S4/R22-Ausnahmen
    seiner Vorfahren mit demselben Grund, z. B. Handler in einer JSX-Komponente."""
    erg: set[str] = _s4(d, i)
    pfad = _pfad(i)
    kopf, _, qual = i.rpartition("#")
    einfach = qual.rpartition(".")[2]
    art = _attr(d, i).get("symbolart")
    datei = _attr(d, datei_von(i))
    teile = qual.split(".")
    for n in range(1, len(teile)):
        vorfahr = f"{kopf}#{'.'.join(teile[:n])}"
        if vorfahr in d.knoten:
            erg |= _s4(d, vorfahr)
            if _dekoriert_aktiv(d, vorfahr):
                erg.add("dekorator")  # registrierte Klasse/Funktion: Verschachteltes lebt mit
    if _dekoriert_aktiv(d, i):
        erg.add("dekorator")  # Ruling S10: Registrierung per Decorator erzeugt keine Kante
    if datei.get("fehlerhaft") == "1" or _attr(d, i).get("fehlerhaft") == "1":
        erg.add("fehlerhaft")
    if datei.get("spezial_fehlt") == "1":
        erg.add("spezial_fehlt")
    if len(einfach) > 4 and einfach.startswith("__") and einfach.endswith("__"):
        erg.add("dunder")
    if ist_testdatei(pfad):
        erg.add("testdatei")
    if _KONVENTION.match(einfach) or (art == "methode" and einfach.startswith("test")) \
            or (pfad.endswith(".go") and einfach == "init"):
        erg.add("konvention")  # Ruling S9: Aufruf per Namenskonvention (Framework-Dispatch)
    if art == "methode" and "." in qual:
        eltern = f"{i.rpartition('#')[0]}#{qual.rpartition('.')[0]}"
        if eltern in e.irgendwie and _attr(d, eltern).get("symbolart") == "klasse":
            erg.add("methode_lebender_klasse")
    return erg


def generiert_zaehlen(d: GraphDaten) -> int:
    """Symbole in Build-/generierten Dateien, die `tot_finden` ueberspringt (Ruling S9)."""
    return sum(1 for i, k in d.knoten.items() if k.art == "symbol" and ist_generiert(_pfad(i)))


def tot_finden(d: GraphDaten, e: Erreichbarkeit, doku: Mapping[str, frozenset[str]]) -> tuple[TotFund, ...]:
    """Nur Symbolknoten. Sortiert nach Symbol-Id, ohne Uhr (Stichtag = `d.commit_zeit`)."""
    grenze = d.commit_zeit - _SEKUNDEN
    funde: list[TotFund] = []
    for i in sorted(k for k, n in d.knoten.items() if n.art == "symbol"):
        pfad = _pfad(i)
        if ist_generiert(pfad):
            continue  # Ruling S9: Build-/generierte Ausgabe ist nie ein Fund (gezaehlt in generiert_zaehlen)
        zuletzt = _zuletzt(d, i)
        if i in e.nur_test:
            if ist_testdatei(pfad):
                continue  # Testcode ist per Definition nur von Tests erreicht
            funde.append(TotFund(i, "verdaechtig", ("nur_test",), zuletzt))
            continue
        alt = zuletzt is not None and zuletzt <= grenze
        if i in e.irgendwie:
            if i in e.sicher or not alt or pfad.endswith(_TS_ENDUNGEN):
                continue  # TS: unsichere Kante genuegt (R15), kein Fund
            gruende = {"nur_unsicher", "alt_90"}
            ausnahmen = _ausnahmen(d, e, i)
            gruende |= ausnahmen
            if _dokumentiert(i, doku):
                gruende.add("doku_verweis")
            funde.append(TotFund(i, "wahrscheinlich_tot", tuple(sorted(gruende)), zuletzt))
            continue
        verwiesen = _dokumentiert(i, doku)
        ausnahmen = _ausnahmen(d, e, i)
        gruende = {"unerreichbar", "doku_verweis" if verwiesen else "kein_doku_verweis"} | ausnahmen
        if alt:
            gruende.add("alt_90")
        sicher = alt and not verwiesen and not ausnahmen
        funde.append(TotFund(i, "sicher_tot" if sicher else "wahrscheinlich_tot", tuple(sorted(gruende)), zuletzt))
    return tuple(funde)


# --- Ruling S8: Namensverweis-Waechter ----------------------------------------------------------
# Ein `sicher_tot`-Kandidat, dessen einfacher Name (letztes Qualname-Segment, >= NAME_MIN Zeichen)
# als Bezeichner-Token in einer ANDEREN committeten Code-Datei desselben Repos oder eines
# Vorfahren-Repos vorkommt, wird `wahrscheinlich_tot` mit Grund `namensverweis`. In der eigenen
# Datei zaehlen nur Vorkommen ausserhalb der Zeilen der eigenen Definition. Gelesen werden Code-Dateien
# plus HTML/Vorlagen (VORLAGEN_ENDUNGEN), nie generierte Dateien (Ruling S9). Deckt ab: Klassen nur
# als Typ/Attribut, `obj.methode()`, String-Dispatch, nicht aufgeloeste Testimporte.

NAME_MIN: int = 3
# Fix-Welle: Bezeichner Unicode-faehig (`größe_berechnen`), wie Python- und JS-Bezeichner.
_BEZEICHNER = re.compile(r"[^\W\d]\w*")
# S8-Erweiterung (Fix-Runde 3): HTML und Vorlagen verweisen per Name (Alpine `x-data="f"`, `onclick=`).
VORLAGEN_ENDUNGEN: tuple[str, ...] = (".html", ".htm", ".vue", ".svelte", ".jinja", ".jinja2", ".j2", ".hbs",
                                      ".ejs", ".erb", ".twig")
# Fix-Welle: Konfiguration und Daten registrieren per Name (entry points, Dockerfile-CMD, YAML-Handler).
# json/yaml haben eine Sprache und kommen aus `Baum.eintraege`, der Rest ueber den Dateinamen.
SCAN_DATEN_SPRACHEN = frozenset({"json", "yaml"})
KONFIG_ENDUNGEN: tuple[str, ...] = (".toml", ".cfg", ".ini")
KONFIG_NAMEN = frozenset({"Makefile", "Procfile"})
KONFIG_PRAEFIXE: tuple[str, ...] = ("Dockerfile",)
_LOCKDATEI = re.compile(r"^(.*[-.]lock\.(json|ya?ml)|npm-shrinkwrap\.json)$", re.IGNORECASE)


def ist_lockdatei(name: str) -> bool:
    """Lock-Dateien (`package-lock.json`, `pnpm-lock.yaml` ...) verweisen nie (nur Paketnamen)."""
    return _LOCKDATEI.match(name) is not None


def _ohne_sprache_lesen(name: str) -> bool:
    low = name.lower()
    return (low.endswith(VORLAGEN_ENDUNGEN) or low.endswith(KONFIG_ENDUNGEN) or name in KONFIG_NAMEN
            or name.startswith(KONFIG_PRAEFIXE))


def _einfach(symbol_id: str) -> str:
    return symbol_id.rpartition("#")[2].rpartition(".")[2]


def _vorfahren(repo: str) -> list[str]:
    teile = repo.split("/")
    return ["/".join(teile[:n]) for n in range(1, len(teile) + 1)]


def _baeume(b: Baum) -> Iterator[Baum]:
    yield b
    for u in b.unterbaeume:
        yield from _baeume(u)


def _in_eigener_datei(e: Eintrag, roh: bytes, text: str, eigene: set[str]) -> set[str]:
    """Kandidaten dieser Datei, deren Name ausserhalb der eigenen Definitionszeilen vorkommt."""
    from ..vermessen import _parsen_sicher  # spaet: vermessen zieht den ganzen Stufe-1-Lauf

    ergebnis, _ = _parsen_sicher(e.sprache or "", e.pfad, roh, e.repo)
    spannen: dict[str, list[tuple[int, int]]] = {}
    for s in ergebnis.symbole:
        spannen.setdefault(s.qualname, []).append((s.zeile_von, s.zeile_bis))
    gesucht: dict[str, list[str]] = {}
    for sid in sorted(eigene):
        gesucht.setdefault(_einfach(sid), []).append(sid)
    zeilen: dict[str, list[int]] = {}
    zeile, pos = 1, 0
    for m in _BEZEICHNER.finditer(text):
        w = m.group()
        if w in gesucht:
            zeile += text.count("\n", pos, m.start())
            pos = m.start()
            zeilen.setdefault(w, []).append(zeile)
    erg: set[str] = set()
    for name, sids in gesucht.items():
        for sid in sids:
            sp = spannen.get(sid.rpartition("#")[2], [])  # ohne Spanne zaehlt jedes Vorkommen (vorsichtig)
            if any(not any(a <= z <= b for a, b in sp) for z in zeilen.get(name, ())):
                erg.add(sid)
    return erg


def namensverweise(wurzel: Path, commit: str, kandidaten: Iterable[str]) -> frozenset[str]:
    """Die Kandidaten (Symbol-Ids), die ein Namensverweis rettet. Ein Durchlauf ueber die Code-Blobs
    am Commit (nur ueber `quelle`, nie der Arbeitsbaum); behalten werden je (Repo, Name) hoechstens
    zwei Dateipfade, das genuegt fuer "eine andere Datei"."""
    je_datei: dict[tuple[str, str], set[str]] = {}
    namen: set[str] = set()
    alle: list[str] = []
    for sid in sorted(set(kandidaten)):
        name = _einfach(sid)
        if len(name) < NAME_MIN:
            continue
        je_datei.setdefault((_repo(sid), _pfad(sid)), set()).add(sid)
        namen.add(name)
        alle.append(sid)
    if not namen:
        return frozenset()
    dateien: dict[tuple[str, str], set[str]] = {}
    gerettet: set[str] = set()
    for b in _baeume(baum_lesen(Path(wurzel), commit)):
        with BlobLeser(b.wurzel) as leser:
            code = [e for e in b.eintraege
                    if e.sprache in CODE_SPRACHEN or e.sprache in SCAN_DATEN_SPRACHEN]
            for e in code + blobs_mit_namen(b, _ohne_sprache_lesen):  # ein Blob nach dem anderen
                if ist_generiert(e.pfad) or ist_lockdatei(e.pfad.rpartition("/")[2]):
                    continue  # Ruling S9: Build-Ausgabe verweist nicht; Lock-Dateien auch nicht
                roh = leser.lesen(e.blob)
                text = roh.decode("utf-8", errors="replace")
                # gestreamt: grosse JSON-Dateien erzeugen keine Liste aller Bezeichner
                for name in {w for m in _BEZEICHNER.finditer(text) if (w := m.group()) in namen}:
                    s = dateien.setdefault((e.repo, name), set())
                    if len(s) < 2:
                        s.add(e.pfad)
                eigene = je_datei.get((e.repo, e.pfad)) if e.sprache in CODE_SPRACHEN else None
                if eigene:
                    gerettet |= _in_eigener_datei(e, roh, text, eigene)
    for sid in alle:
        if sid in gerettet:
            continue
        name, repo, pfad = _einfach(sid), _repo(sid), _pfad(sid)
        if any(a != repo or p != pfad for a in _vorfahren(repo) for p in dateien.get((a, name), ())):
            gerettet.add(sid)
    return frozenset(gerettet)


def namensverweis_abstufen(funde: tuple[TotFund, ...], gerettet: frozenset[str]) -> tuple[TotFund, ...]:
    """`sicher_tot` mit Namensverweis wird `wahrscheinlich_tot` (Grund `namensverweis`); Reihenfolge bleibt."""
    return tuple(
        TotFund(f.symbol, "wahrscheinlich_tot", tuple(sorted({*f.gruende, "namensverweis"})), f.zuletzt)
        if f.stufe == "sicher_tot" and f.symbol in gerettet else f
        for f in funde)
