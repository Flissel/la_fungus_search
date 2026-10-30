"""Beleg-Drift in KETTEN.md messen (Task 12, Abnahme C.9.1); nur lesend.

Jede Code-Stelle `pfad:zeile[-zeile]` in KETTEN.md ist ein Beleg: in Hop-Zeilen (alle Spalten), in anderen
Tabellen und in Prosa. Aufgeloest wird wie im Seed (`ketten_seed._aufloesen`: voll, Kurzform, Verzeichnis,
Anker, Auslassung). Ausserhalb der Spalte Code-Stelle zaehlt eine Kurzform `:n` nur hinter einem Pfad derselben
Zelle bzw. Einheit und nur, wenn die Datei so viele Zeilen hat (sonst ist es ein Port wie `:8131`).

Klassen (`KLASSEN`):
  exakt            Token steht an der zitierten Zeile (Bereich ±3)
  verschoben       Token steht woanders in derselben Datei (`gefunden_zeile` = naechster Treffer)
  token_fehlt      Datei da, Token nirgends (oder die Zeile nennt kein Kenn-Token)
  datei_fehlt      aufgeloester Pfad existiert nicht bzw. ein Dateiname ist nicht auffindbar
  nicht_aufloesbar Auslassung (`043…sql:297`) oder Kurzform ohne vorigen Pfad

Token-Methode (unveraendert aus der Messung vom 27.10., `wf-ketten/beleg_drift.py`: 298 Belege, 202 exakt,
83 verschoben): Kandidaten sind die Backtick-Begriffe der Spalte Mechanismus derselben Zeile, in ihrer
Reihenfolge (der erste, der in der Datei steht, ist das Token), dazu ihre Bestandteile; reicht das nicht,
die Begriffe der eigenen Zelle, dann die von Von → Nach. Prosa und andere Tabellen nehmen die Begriffe
der ganzen Einheit. Bezeichner werden auf Wortgrenzen gesucht, alles andere als Teilstring.

Gerenderte Bloecke (zwischen den Markern von `render_ketten`): Code-Stellen mit Anker-Paar
`` `datei:zeile` `anker` `` (`hat_anker`, Ruling 37) werden ueber den Anker eingestuft -- `exakt`, wenn der
Anker (symbol / sql / yaml_key, auch TOML) heute innerhalb ±3 Zeilen der gezeigten Zeile beginnt,
`verschoben`, wenn er woanders aufloest, `token_fehlt`, wenn er nicht mehr aufloest (auch: Datei nicht
parsbar, Endung ohne Ankerart), `datei_fehlt`, wenn die Datei weg ist. Belege ohne Anker-Paar (Dateibelege,
`⚠`-Stellen) laufen ueber die Token-Methode. `bericht` zaehlt rohe und gerenderte Zeilen getrennt, gerenderte
zusaetzlich nach `hat_anker` (darueber wird C.9.1 gemessen).
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import anker as anker_modul
from . import ketten_seed as ks
from . import konfig as konfig_modul
from .extraktoren import sql as sql_extraktor
from .konfig import Konfig
from .render_ketten import _yaml_zeile, bloecke_finden

KLASSEN = ("exakt", "verschoben", "token_fehlt", "datei_fehlt", "nicht_aufloesbar")
FENSTER = 3
SPALTE_MECH, SPALTE_CODE, SPALTE_VON = 2, 3, 1


@dataclass(frozen=True)
class BelegBefund:
    zeile: int  # Zeile in KETTEN.md (1-basiert)
    kette: str | None
    stelle: str  # aufgeloest `pfad:zeile[-zeile]`, sonst der Rohtext
    klasse: str
    gefunden_zeile: int | None
    token: str | None  # gefundenes Token bzw. bei Anker-Paaren der Anker
    gerendert: bool = False  # Beleg steht in einem markierten (gerenderten) Block
    hat_anker: bool = False  # Ruling 37: der Beleg traegt eine gerenderte Anker-Spanne (Massstab fuer C.9.1)


# ---------- Token-Kandidaten (Methode der Messung vom 27.10.) ----------

_ENDUNG = r"(?:py|yaml|yml|sql|rs|ts|js|json|toml|ps1|md|sh)"
_PFAD_ZEILE = re.compile(r"(?:[\w\-.~]+/)*(?:[\w\-.]+\." + _ENDUNG + r"|Dockerfile):\d+(?:-\d+)?")
_AUSLASSUNG = re.compile(r"[\w\-.]*…[\w\-.]*:\d+(?:-\d+)?")
_KURZ = re.compile(r"^:\d+(?:-\d+)?$")
_PRAEFIX = re.compile(r"^(?:POST|GET|HEAD|PUT|PATCH|DELETE|HTTP|SQL|MCP|Python|Chat|Datei|Regex|IPC|CDP|SSE)\s+")
_STOPP = frozenset(w.lower() for w in """
POST GET HEAD PUT PATCH DELETE HTTP HTTPS SQL MCP Python Datei Chat IPC SSE CDP Regex Import Modulimport
true false none null stdio json yaml file files name type test tests data core spaces space brain voice
api app apps src lib config scripts docs tools claw header bearer default launcher worker capability
master pin kein keine nicht laut nur auch alle jede jeder dann sonst oder und mit ohne ueber via
""".split())


def _spannen(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def kandidaten(spannen: Sequence[str]) -> list[str]:
    """Token-Kandidaten aus Backtick-Begriffen in Reihenfolge: der Begriff (ohne Methoden-Praefix), ein
    Aufrufname `name(`, dann seine Woerter (laengste zuerst) und deren Segmente; ohne Code-Stellen, Ports,
    Stoppwoerter und Woerter unter 4 Zeichen."""
    aus: list[str] = []
    for roh in spannen:
        t = roh.strip()
        if not t or _PFAD_ZEILE.search(t) or _AUSLASSUNG.search(t) or _KURZ.match(t):
            continue
        t2 = _PRAEFIX.sub("", t)
        if len(t2) >= 4 and not t2.startswith("<"):
            aus.append(t2)
        m = re.match(r"^([A-Za-z_][\w.]*)\(", t2)
        if m and len(m.group(1)) >= 4:
            aus.append(m.group(1))
        woerter = [w.strip("./-") for w in re.findall(r"[A-Za-z0-9_./\-]+", t2) if re.search(r"[A-Za-z]", w)]
        woerter = sorted((w for w in woerter if len(w) >= 4 and w.lower() not in _STOPP), key=len, reverse=True)
        for w in woerter:
            aus.append(w)
            if "." in w or "/" in w:
                aus.extend(s for s in re.split(r"[./]", w) if len(s) >= 5 and s.lower() not in _STOPP)
    return list(dict.fromkeys(aus))


def _treffer(kandidat: str, zeilen: Sequence[str]) -> list[int]:
    if re.fullmatch(r"[A-Za-z_]\w*", kandidat):
        muster = re.compile(r"(?<!\w)" + re.escape(kandidat) + r"(?!\w)")
        return [i + 1 for i, z in enumerate(zeilen) if muster.search(z)]
    return [i + 1 for i, z in enumerate(zeilen) if kandidat in z]


def token_einstufen(stufen: Sequence[Sequence[str]], zeilen: Sequence[str], start: int,
                    ende: int | None) -> tuple[str, int | None, str | None]:
    """(Klasse, gefundene Zeile, Token) nach der Token-Methode; `stufen` sind Kandidatenlisten nach Vorrang.

    Je Stufe: ein Kandidat im Fenster -> exakt; sonst der erste Kandidat mit Treffern -> verschoben (naechster
    Treffer); ohne Treffer weiter zur naechsten Stufe; am Ende token_fehlt."""
    unten, oben = start - FENSTER, (ende or start) + FENSTER
    for stufe in stufen:
        bester: tuple[str, int] | None = None
        for kandidat in stufe:
            treffer = _treffer(kandidat, zeilen)
            if not treffer:
                continue
            im_fenster = [t for t in treffer if unten <= t <= oben]
            if im_fenster:
                return "exakt", im_fenster[0], kandidat
            if bester is None:
                bester = (kandidat, min(treffer, key=lambda t: abs(t - start)))
        if bester is not None:
            return "verschoben", bester[1], bester[0]
    return "token_fehlt", None, None


# ---------- Anker gerenderter Paare ----------

def _paare(zelle: str) -> list[tuple[str, str | None]]:
    """Backtick-Spannen einer gerenderten Zelle: (Spanne, Anker) fuer Paare `` `stelle` `anker` `` (Regel wie
    `ketten_seed.render_anker_trennen`), sonst (Spanne, None)."""
    spannen = [(m.start(), m.end(), m.group(1)) for m in re.finditer(r"`([^`]+)`", zelle)]
    aus: list[tuple[str, str | None]] = []
    i = 0
    while i < len(spannen):
        _, ende, s = spannen[i]
        if i + 1 < len(spannen):
            start2, _, s2 = spannen[i + 1]
            zwischen = zelle[ende:start2]
            if (ks._STELLE.fullmatch(s.strip()) and zwischen and not zwischen.strip()
                    and not ks._STELLE.fullmatch(s2.strip())):
                aus.append((s, s2))
                i += 2
                continue
        aus.append((s, None))
        i += 1
    return aus


def _sql_zeilen(datei: Path, name: str) -> list[int]:
    try:
        text = datei.read_bytes().decode("utf-8").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError):
        return []
    maske = sql_extraktor._maskieren(text)
    zeilen = []
    for m in sql_extraktor._CREATE.finditer(maske):
        gefunden = f"{sql_extraktor._entquoten(m.group(2))}.{sql_extraktor._entquoten(m.group(3))}"
        if gefunden == name.lower():
            zeilen.append(text.count("\n", 0, m.start()) + 1)
    return zeilen


def _modul_namen(pfad: str) -> set[str]:
    """Modulsegment der Datei in einer id: der Dateistamm; bei `__init__.py` auch der Paketname
    (AST-ids ausserhalb der Spaces behalten `__init__`, `anker.modulpfad`)."""
    p = Path(pfad)
    namen = {p.stem} | ({p.parent.name} if p.stem == "__init__" else set())
    return {n.replace("-", "_") for n in namen}


def _py_zeile(defs: Sequence[anker_modul.Definition], pfad: str, anker: str, nahe: int) -> int | None:
    """Startzeile des Symbols `anker` (Manifest- bzw. AST-id `<…>.<modul>.<qualname>`) in einer Datei.

    Voll: das Segment vor dem Qualnamen ist der Modulname der Datei. Sonst (Datei umbenannt/verschoben)
    nur, wenn genau eine Definition dem laengsten Qualnamen-Ende mit mindestens zwei Segmenten gleicht
    (`Klasse.methode`); ein nackter Name allein reicht nicht. Mehrere gleiche Qualnamen: der naechste."""
    je_qual: dict[str, list[anker_modul.Definition]] = {}
    for d in defs:
        je_qual.setdefault(d.document_id.lstrip("."), []).append(d)
    teile = anker.split(".")
    module = _modul_namen(pfad)
    for i in range(1, len(teile)):
        qual = ".".join(teile[i:])
        if teile[i - 1].replace("-", "_") in module and qual in je_qual:
            return min(je_qual[qual], key=lambda d: abs(d.start_line - nahe)).start_line
    for k in range(len(teile) - 1, 1, -1):
        treffer = je_qual.get(".".join(teile[-k:]), [])
        if len(treffer) == 1:
            return treffer[0].start_line
        if treffer:
            return None  # mehrdeutig
    return None


def _toml_zeile(datei: Path, anker: str) -> int | None:
    """Zeile der Tabelle bzw. des Schluessels mit Pfad `anker` (Identitaeten wie im Seed)."""
    try:
        text = datei.read_text(encoding="utf-8")
        daten = tomllib.loads(text)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError):
        return None
    for e in ks._toml_eintraege(text):
        pfad = ks._toml_identitaet(daten, e.knoten)
        if pfad and anker_modul.pfad_text([t for t, _ in pfad]) == anker:
            return e.start
    return None


class _Anker:
    """Aktuelle Startzeile eines gerenderten Ankers; None = loest nicht (mehr) auf (auch: Datei nicht
    parsbar, Endung ohne Ankerart)."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.yaml: dict[str, yaml.Node | None] = {}
        self.ast: dict[str, list[anker_modul.Definition] | None] = {}

    def zeile(self, pfad: str, anker: str, nahe: int) -> int | None:
        datei = self.repo / pfad
        endung = datei.suffix.lower()
        if endung == ".py":
            if pfad not in self.ast:
                self.ast[pfad] = anker_modul.ast_definitionen(datei, "")
            defs = self.ast[pfad]
            return _py_zeile(defs, pfad, anker, nahe) if defs is not None else None
        if endung == ".sql":
            zeilen = _sql_zeilen(datei, anker)
            return min(zeilen, key=lambda z: abs(z - nahe)) if zeilen else None
        if endung in (".yaml", ".yml"):
            if pfad not in self.yaml:
                try:
                    self.yaml[pfad] = yaml.compose(datei.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
                except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError):
                    self.yaml[pfad] = None
            wurzel = self.yaml[pfad]
            return _yaml_zeile(wurzel, anker) if wurzel is not None else None
        if endung == ".toml":
            return _toml_zeile(datei, anker)
        return None


# ---------- Dokument lesen ----------

@dataclass
class _Einheit:
    art: str  # hop | tabelle | prosa
    zeile: int  # 1-basiert (erste Zeile)
    kette: str | None
    zellen: list[str]  # hop: sechs Zellen; sonst [Text]
    gerendert: bool


_KETTE = re.compile(r"^###\s+(K\d+)\b")
_UEBERSCHRIFT = re.compile(r"^(#{1,6})\s")


def _block_zeilen(text: str) -> set[int]:
    zeilen: set[int] = set()
    for b in bloecke_finden(text).values():
        # 1-basiert, Markerzeilen eingeschlossen
        zeilen.update(range(text.count("\n", 0, b.start), text.count("\n", 0, b.ende) + 2))
    return zeilen


def _einheiten(text: str) -> list[_Einheit]:
    gerendert = _block_zeilen(text)
    einheiten: list[_Einheit] = []
    kette: str | None = None
    in_hop = False
    absatz: list[str] = []
    absatz_start = 0

    def schliessen() -> None:
        nonlocal absatz
        if absatz:
            einheiten.append(_Einheit("prosa", absatz_start, kette, [" ".join(absatz)],
                                      absatz_start in gerendert))
        absatz = []

    for nr, roh in enumerate(text.lstrip("﻿").split("\n"), 1):
        zeile = roh.rstrip()
        m = _KETTE.match(zeile)
        if m:
            schliessen()
            kette, in_hop = m.group(1), False
            continue
        u = _UEBERSCHRIFT.match(zeile)
        if u:
            schliessen()
            in_hop = False
            if len(u.group(1)) <= 3:
                kette = None
            continue
        if not zeile.strip() or zeile.lstrip().startswith("<!--"):
            schliessen()
            in_hop = False
            continue
        if zeile.lstrip().startswith("|"):
            schliessen()
            zellen = ks._zellen(zeile.strip())
            if "Von → Nach" in zellen and "Code-Stelle" in zellen:
                in_hop = True
                continue
            if zellen and all(re.fullmatch(r":?-{2,}:?", z.strip()) for z in zellen):
                continue
            if in_hop and len(zellen) == 6:
                einheiten.append(_Einheit("hop", nr, kette, zellen, nr in gerendert))
            else:
                einheiten.append(_Einheit("tabelle", nr, kette, [zeile], nr in gerendert))
            continue
        in_hop = False
        if not absatz:
            absatz_start = nr
        absatz.append(zeile.strip())
    schliessen()
    return einheiten


# ---------- Messen ----------

class _Messung:
    def __init__(self, repo: Path, konfig: Konfig) -> None:
        self.repo = repo
        self.index: Callable[[], Mapping[str, list[str]]] = ks._index_holer(repo, konfig, None)
        self.bekannte: dict[str, str] = {}
        self.dateien: dict[str, list[str]] = {}
        self.anker = _Anker(repo)
        self.befunde: list[BelegBefund] = []

    def zeilen(self, pfad: str) -> list[str]:
        if pfad not in self.dateien:
            self.dateien[pfad] = (self.repo / pfad).read_text(encoding="utf-8", errors="replace").split("\n")
        return self.dateien[pfad]

    def stellen(self, spanne: str, voriger: str | None, code_spalte: bool) -> tuple[list[ks.Stelle], str | None]:
        neu, weiter = ks._aufloesen(f"`{spanne}`", voriger, self.repo, self.index, self.bekannte)
        aus: list[ks.Stelle] = []
        for st in neu:
            if st.pfad is not None and st.form == "voll":
                self.bekannte[st.pfad.rsplit("/", 1)[-1]] = st.pfad
            if st.start is None:
                continue  # Pfad ohne Zeile: Bezug fuer Kurzformen, kein Beleg
            if st.form == "kurzform" and not code_spalte:
                if (st.pfad is None or not anker_modul.pfad_sicher(st.pfad) or not (self.repo / st.pfad).is_file()
                        or st.start > len(self.zeilen(st.pfad))):
                    continue  # Port o. ae.
            aus.append(st)
        return aus, weiter

    def einstufen(self, st: ks.Stelle, e: _Einheit, stufen: Sequence[Sequence[str]], anker: str | None) -> None:
        stelle = ks._stelle_voll(st)
        hat = anker is not None
        if st.pfad is not None and not anker_modul.pfad_sicher(st.pfad):  # Final-Fix 22: nie ausserhalb lesen
            self.befunde.append(BelegBefund(e.zeile, e.kette, stelle, "nicht_aufloesbar", None, anker, e.gerendert,
                                            hat))
            return
        if st.pfad is None:
            klasse = "datei_fehlt" if st.form == "anker" else "nicht_aufloesbar"
            self.befunde.append(BelegBefund(e.zeile, e.kette, stelle, klasse, None, anker, e.gerendert, hat))
            return
        if not (self.repo / st.pfad).is_file():
            self.befunde.append(BelegBefund(e.zeile, e.kette, stelle, "datei_fehlt", None, anker, e.gerendert, hat))
            return
        assert st.start is not None
        if anker is not None:
            zeile = self.anker.zeile(st.pfad, anker, st.start)
            klasse = "token_fehlt" if zeile is None else (
                "exakt" if abs(zeile - st.start) <= FENSTER else "verschoben")
            self.befunde.append(BelegBefund(e.zeile, e.kette, stelle, klasse, zeile, anker, e.gerendert, True))
            return
        klasse, gefunden, token = token_einstufen(stufen, self.zeilen(st.pfad), st.start, st.ende)
        self.befunde.append(BelegBefund(e.zeile, e.kette, stelle, klasse, gefunden, token, e.gerendert))

    def hop(self, e: _Einheit, voriger: str | None) -> str | None:
        ohne_anker = [ks.render_anker_trennen(z)[0] for z in e.zellen]
        mech, von = kandidaten(_spannen(ohne_anker[SPALTE_MECH])), kandidaten(_spannen(ohne_anker[SPALTE_VON]))
        for spalte, zelle in enumerate(e.zellen):
            eigen = kandidaten(_spannen(ohne_anker[spalte])) if spalte != SPALTE_MECH else []
            stufen = (mech, eigen, von) if spalte == SPALTE_CODE else (eigen, mech, von)
            paare = _paare(zelle) if e.gerendert and spalte == SPALTE_CODE else \
                [(s, None) for s in _spannen(zelle)]
            vor = voriger if spalte == SPALTE_CODE else None
            for spanne, anker in paare:
                stellen, vor = self.stellen(spanne, vor, spalte == SPALTE_CODE)
                for st in stellen:
                    self.einstufen(st, e, stufen, anker)
            if spalte == SPALTE_CODE:
                voriger = vor
        return voriger

    def andere(self, e: _Einheit) -> None:
        text = ks.render_anker_trennen(e.zellen[0])[0]
        stufen = (kandidaten(_spannen(text)),)
        voriger: str | None = None
        for spanne in _spannen(text):
            stellen, voriger = self.stellen(spanne, voriger, False)
            for st in stellen:
                self.einstufen(st, e, stufen, None)


def belege_messen(ketten_md: Path, repo: Path, *, konfig: Konfig | None = None) -> list[BelegBefund]:
    """Alle Belege in `ketten_md` einstufen (siehe Moduldoku); wirft nicht wegen einzelner Stellen.

    `konfig` liefert den Ausschluss fuer den Dateiindex (Dateinamen-Anker); ohne sie gilt die Standard-
    Konfiguration mit `repo` als Wurzel."""
    if konfig is None:
        konfig = dataclasses.replace(konfig_modul.laden(), repo=repo)
    text = Path(ketten_md).read_bytes().decode("utf-8")
    messung = _Messung(repo, konfig)
    voriger: str | None = None
    letzte_kette: str | None = None
    for e in _einheiten(text):
        if e.art == "hop":
            if e.kette != letzte_kette:
                letzte_kette, voriger = e.kette, None  # wie im Seed: Kurzformen beziehen sich nur in der Kette
            voriger = messung.hop(e, voriger)
        else:
            messung.andere(e)
    return messung.befunde


# ---------- Bericht ----------

def _zaehlung(befunde: Sequence[BelegBefund]) -> str:
    zahl = Counter(b.klasse for b in befunde)
    return " · ".join(f"{k} {zahl.get(k, 0)}" for k in KLASSEN)


def bericht(befunde: Sequence[BelegBefund]) -> str:
    """Zaehlung je Klasse (rohe und gerenderte Zeilen getrennt, gerenderte zusaetzlich mit/ohne Anker: C.9.1
    zaehlt ueber `hat_anker`), dann je Beleg eine Zeile nach Klasse."""
    roh = [b for b in befunde if not b.gerendert]
    gerendert = [b for b in befunde if b.gerendert]
    mit = [b for b in gerendert if b.hat_anker]
    ohne = [b for b in gerendert if not b.hat_anker]
    zeilen = [
        f"Belege: {len(befunde)} (roh {len(roh)}, gerendert {len(gerendert)})",
        f"roh: {_zaehlung(roh)}",
        f"gerendert: {_zaehlung(gerendert)}",
        f"gerendert mit Anker, C.9.1 ({len(mit)}): {_zaehlung(mit)}",
        f"gerendert ohne Anker ({len(ohne)}): {_zaehlung(ohne)}",
    ]
    rang = {k: i for i, k in enumerate(KLASSEN)}
    for b in sorted(befunde, key=lambda b: (rang.get(b.klasse, len(KLASSEN)), b.zeile, b.stelle)):
        teil = f"  Z{b.zeile:<4} {b.kette or '-':<4} {b.klasse:<16} {b.stelle}"
        if b.gefunden_zeile is not None:
            teil += f" -> {b.gefunden_zeile}"
        if b.token is not None:
            teil += f" token={b.token}"
        zeilen.append(teil + ((" [gerendert, Anker]" if b.hat_anker else " [gerendert]") if b.gerendert
                              else " [roh]"))
    return "\n".join(zeilen) + "\n"
