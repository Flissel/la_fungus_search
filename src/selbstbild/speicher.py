"""SQLite-Speicher: Parse-Cache, Graph, Symbolverlauf, JSON-Export.

`seit` (Beginn der aktuellen Digest-Strecke eines Symbols) haengt vom gemessenen Verlauf
dieses Stands ab. Es steht als Attribut der Symbolknoten in der Tabelle `knoten`, wird
aber nicht exportiert (R17): Der Export ist eine reine Funktion von (Commit, Regeln).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sqlite3
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any

from .sprachen import GRAMMATIK_VERSION, PARSER_VERSION
from .typen import (
    Aufruf, Beleg, DateiErgebnis, Einstieg, EnvLesung, Import, Kante, Knoten,
    SqlObjekt, Symbol,
)

_SURROGAT = re.compile("[\ud800-\udfff]")


def _esc(text: str) -> str:
    """Einzelne Surrogate (Git-Pfade mit surrogateescape) als \\udXXX-Escape
    schreiben: gueltiges UTF-8, von json.loads verlustfrei zurueckgelesen."""
    return _SURROGAT.sub(lambda m: "\\u%04x" % ord(m.group()), text)


def _dumps(wert: Any) -> str:
    return _esc(json.dumps(wert, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _beleg_zu(b: Beleg | None) -> Any:
    return None if b is None else [b.repo, b.pfad, b.zeile, b.anker, b.digest]


def _beleg_von(d: Any) -> Beleg | None:
    return None if d is None else Beleg(d[0], d[1], d[2], d[3], d[4])


def _knoten_zu(k: Knoten) -> Any:
    return {"id": k.id, "art": k.art, "name": k.name,
            "beleg": _beleg_zu(k.beleg), "attr": [list(p) for p in k.attr]}


def _knoten_von(d: Any) -> Knoten:
    return Knoten(d["id"], d["art"], d["name"], _beleg_von(d["beleg"]),
                  tuple((a, b) for a, b in d["attr"]))


def _kante_zu(k: Kante) -> Any:
    return {"art": k.art, "von": k.von, "nach": k.nach,
            "sicher": k.sicher, "beleg": _beleg_zu(k.beleg)}


def _kante_von(d: Any) -> Kante:
    return Kante(d["art"], d["von"], d["nach"], d["sicher"], _beleg_von(d["beleg"]))


def _ergebnis_zu(e: DateiErgebnis) -> Any:
    return {
        "symbole": [[s.qualname, s.art, s.zeile_von, s.zeile_bis, s.digest] for s in e.symbole],
        "importe": [[i.modul, list(i.namen), i.zeile, i.relativ] for i in e.importe],
        "aufrufe": [[a.in_symbol, a.name, a.zeile] for a in e.aufrufe],
        "env": [[x.name, x.zeile, x.in_symbol] for x in e.env],
        "einstiege": [[x.art, x.symbol, x.zeile] for x in e.einstiege],
        "sql": [[s.art, s.name, s.zeile, s.digest] for s in e.sql],
        "tz": [list(t) for t in e.tabellen_zugriffe],
        "ek": [_knoten_zu(k) for k in e.extra_knoten],
        "ekk": [_kante_zu(k) for k in e.extra_kanten],
        "fehlerhaft": e.fehlerhaft,
    }


def _ergebnis_von(d: Any) -> DateiErgebnis:
    return DateiErgebnis(
        symbole=tuple(Symbol(*s) for s in d["symbole"]),
        importe=tuple(Import(i[0], tuple(i[1]), i[2], i[3]) for i in d["importe"]),
        aufrufe=tuple(Aufruf(*a) for a in d["aufrufe"]),
        env=tuple(EnvLesung(*x) for x in d["env"]),
        einstiege=tuple(Einstieg(*x) for x in d["einstiege"]),
        sql=tuple(SqlObjekt(*s) for s in d["sql"]),
        tabellen_zugriffe=tuple((t[0], t[1], t[2]) for t in d["tz"]),
        extra_knoten=tuple(_knoten_von(k) for k in d["ek"]),
        extra_kanten=tuple(_kante_von(k) for k in d["ekk"]),
        fehlerhaft=d["fehlerhaft"],
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache(schluessel TEXT PRIMARY KEY, wert TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS knoten(
    id TEXT PRIMARY KEY, sortierung BLOB NOT NULL, wert TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS knoten_sortierung ON knoten(sortierung);
CREATE TABLE IF NOT EXISTS kanten(
    wert TEXT PRIMARY KEY, s_art BLOB NOT NULL, s_von BLOB NOT NULL, s_nach BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS kanten_sortierung ON kanten(s_art, s_von, s_nach, wert);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS laeufe(
    bereich TEXT NOT NULL, zeit INTEGER NOT NULL, commit_id TEXT NOT NULL,
    PRIMARY KEY(bereich, zeit));
CREATE TABLE IF NOT EXISTS symbolzeit(
    symbol TEXT NOT NULL, bereich TEXT NOT NULL, zeit INTEGER NOT NULL,
    digest TEXT NOT NULL, commit_id TEXT NOT NULL, PRIMARY KEY(symbol, zeit));
CREATE INDEX IF NOT EXISTS symbolzeit_bereich ON symbolzeit(bereich, symbol, zeit);
"""

# Digest-Platzhalter: Symbol war in diesem erfassten Lauf nicht vorhanden.
_ABWESEND = ""


def _sortierschluessel(text: str) -> bytes:
    """UTF-8 mit surrogatepass: Bytevergleich (SQLite BLOB, memcmp) == Codepunkt-Ordnung
    von Python-`str`, auch fuer Nicht-BMP-Zeichen und einzelne Surrogate."""
    return text.encode("utf-8", "surrogatepass")


def _strecke(zeilen_neu_zuerst: Iterable[tuple[int, str]]) -> int | None:
    """Beginn der Strecke des juengsten Digests (Zeilen nach Zeit absteigend), None = abwesend."""
    seit: int | None = None
    aktuell: str | None = None
    for zeit, digest in zeilen_neu_zuerst:
        if aktuell is None:
            if digest == _ABWESEND:
                return None
            aktuell = digest
        elif digest != aktuell:
            break
        seit = zeit
    return seit


def _alte_tabellen_entfernen(c: sqlite3.Connection) -> None:
    """Graph-Tabellen ohne Sortierspalten (vor dem streamenden Export) und den alten,
    nach Einfuegereihenfolge gefuehrten Symbolverlauf (vor R17) verwerfen."""
    spalten = {z[1] for z in c.execute("PRAGMA table_info(knoten)")}
    with c:
        if spalten and "sortierung" not in spalten:
            c.execute("DROP TABLE IF EXISTS knoten")
            c.execute("DROP TABLE IF EXISTS kanten")
        c.execute("DROP TABLE IF EXISTS symbolverlauf")


def _liste_schreiben(f: IO[bytes], objekte: Iterator[Any]) -> None:
    """Eine JSON-Liste auf Tiefe 1 im Format von `json.dumps(..., indent=1)` streamen."""
    erstes = True
    for obj in objekte:
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1)
        f.write(_esc(("[\n  " if erstes else ",\n  ") + text.replace("\n", "\n  ")).encode("utf-8"))
        erstes = False
    f.write(b"[]" if erstes else b"\n ]")


class Speicher:
    def __init__(self, pfad: Path) -> None:
        self.pfad = Path(pfad)
        self._c: sqlite3.Connection | None = None

    def _verbindung(self) -> sqlite3.Connection:
        if self._c is None:
            self.pfad.parent.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(str(self.pfad))
            c.execute("PRAGMA journal_mode=WAL")
            _alte_tabellen_entfernen(c)
            c.executescript(_SCHEMA)
            self._c = c
        return self._c

    def __enter__(self) -> "Speicher":
        self._verbindung()
        return self

    def __exit__(self, *_: object) -> None:
        self.schliessen()

    def schliessen(self) -> None:
        if self._c is not None:
            self._c.close()
            self._c = None

    # -- Parse-Cache: Schluessel = Parser- + Grammatikversion, Blob, Sprache, Pfad --
    # Der Pfad gehoert dazu (R11): Parser verhalten sich pfadabhaengig (test_*.py,
    # app/**/route.ts, *compose*.yml, package.json, *_test.go).
    @staticmethod
    def _cache_schluessel(blob: str, sprache: str, pfad: str) -> str:
        return _dumps([PARSER_VERSION, GRAMMATIK_VERSION, blob, sprache, pfad])

    def cache_holen(self, blob: str, sprache: str, pfad: str) -> DateiErgebnis | None:
        z = self._verbindung().execute(
            "SELECT wert FROM cache WHERE schluessel=?",
            (self._cache_schluessel(blob, sprache, pfad),)).fetchone()
        return None if z is None else _ergebnis_von(json.loads(z[0]))

    def cache_setzen(self, blob: str, sprache: str, ergebnis: DateiErgebnis, pfad: str) -> None:
        c = self._verbindung()
        with c:
            c.execute("INSERT OR REPLACE INTO cache VALUES(?,?)",
                      (self._cache_schluessel(blob, sprache, pfad), _dumps(_ergebnis_zu(ergebnis))))

    # -- Graph -------------------------------------------------------------
    def graph_ersetzen(self, commit: str, knoten: Iterable[Knoten], kanten: Iterable[Kante]) -> None:
        c = self._verbindung()
        with c:
            c.execute("DELETE FROM knoten")
            c.execute("DELETE FROM kanten")
            c.executemany(
                "INSERT OR REPLACE INTO knoten VALUES(?,?,?)",
                ((_dumps(k.id), _sortierschluessel(k.id), _dumps(_knoten_zu(k))) for k in knoten))
            c.executemany(
                "INSERT OR IGNORE INTO kanten VALUES(?,?,?,?)",
                ((_dumps(_kante_zu(k)), _sortierschluessel(k.art), _sortierschluessel(k.von),
                  _sortierschluessel(k.nach)) for k in kanten))
            c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (_dumps("commit"), _dumps(commit)))

    _KNOTEN_SQL = "SELECT wert FROM knoten ORDER BY sortierung"
    # Reihenfolge wie sorted(key=(art, von, nach, _dumps(kante))): BLOB/TEXT binaer = Codepunkte
    _KANTEN_SQL = "SELECT wert FROM kanten ORDER BY s_art, s_von, s_nach, wert"

    def knoten(self) -> list[Knoten]:
        return [_knoten_von(json.loads(r[0])) for r in self._verbindung().execute(self._KNOTEN_SQL)]

    def kanten(self) -> list[Kante]:
        return [_kante_von(json.loads(r[0])) for r in self._verbindung().execute(self._KANTEN_SQL)]

    # -- Symbolverlauf: zeitlich geordnet, je Bereich (Repo) mit Abwesenheit ---
    def symbolverlauf_merken(
        self, commit: str, commit_zeit: int, symbole: Mapping[str, str], bereich: str = "",
    ) -> None:
        """Erfasst einen Lauf von `bereich` zur Zeit `commit_zeit` (Symbol-Id -> Digest).

        Zeilen gibt es nur zu Zeiten erfasster Laeufe, und nur bei einem Wechsel (Digest
        oder Abwesenheit) gegenueber dem vorigen erfassten Lauf. Ein nachgetragener aelterer
        Lauf wird zeitlich einsortiert. Damit spaetere Strecken stimmen, wird der bisherige
        Zustand am naechsten erfassten Lauf festgeschrieben, falls er dort keine Zeile hatte.
        Ein Lauf auf einer schon erfassten Zeit ueberschreibt sie und ist bei gleichen Daten
        idempotent. Fehlt ein Symbol in einem erfassten Lauf, beginnt seine Strecke danach neu.
        """
        c = self._verbindung()
        z = commit_zeit
        with c:
            naechster = c.execute(
                "SELECT zeit, commit_id FROM laeufe WHERE bereich=? AND zeit>? ORDER BY zeit LIMIT 1",
                (bereich, z)).fetchone()
            n_zeit: int | None = naechster[0] if naechster else None
            n_commit: str = naechster[1] if naechster else ""
            neu = {_dumps(s): d for s, d in symbole.items()}
            schreiben: list[tuple[str, str, int, str, str]] = []

            def pruefen(sym: str, zeilen: list[tuple[int, str]]) -> None:
                vor, bei, am_naechsten, zeile_am_naechsten = _ABWESEND, None, _ABWESEND, False
                for t, d in zeilen:
                    if t < z:
                        vor = d
                    elif t == z:
                        bei = d
                    if n_zeit is not None and t <= n_zeit:
                        am_naechsten = d
                        zeile_am_naechsten = t == n_zeit
                ziel = neu.get(sym, _ABWESEND)
                if (bei is None and ziel == vor) or bei == ziel:
                    return
                schreiben.append((sym, bereich, z, ziel, commit))
                if n_zeit is not None and not zeile_am_naechsten and am_naechsten != ziel:
                    schreiben.append((sym, bereich, n_zeit, am_naechsten, n_commit))

            gesehen: set[str] = set()
            zeilen = c.execute(
                "SELECT symbol, zeit, digest FROM symbolzeit WHERE bereich=? ORDER BY symbol, zeit",
                (bereich,))
            for sym, gruppe in itertools.groupby(zeilen, key=lambda r: r[0]):
                gesehen.add(sym)
                pruefen(sym, [(t, d) for _, t, d in gruppe])
            for sym in neu:
                if sym not in gesehen:
                    pruefen(sym, [])
            c.executemany("INSERT OR REPLACE INTO symbolzeit VALUES(?,?,?,?,?)", schreiben)
            c.execute("INSERT OR REPLACE INTO laeufe VALUES(?,?,?)", (bereich, z, commit))

    def symbol_seit(self, symbol_id: str, bis_zeit: int) -> int | None:
        """Beginn der Strecke des bei `bis_zeit` gueltigen Digests (nur Zeilen mit zeit <= bis_zeit)."""
        zeilen = self._verbindung().execute(
            "SELECT zeit, digest FROM symbolzeit WHERE symbol=? AND zeit<=? ORDER BY zeit DESC",
            (_dumps(symbol_id), bis_zeit))
        return _strecke(zeilen)

    def seit_je_symbol(self, bereich: str, bis_zeit: int) -> dict[str, int]:
        """`symbol_seit` fuer alle Symbole eines Bereichs in einem Durchgang (abwesende fehlen)."""
        erg: dict[str, int] = {}
        zeilen = self._verbindung().execute(
            "SELECT symbol, zeit, digest FROM symbolzeit WHERE bereich=? AND zeit<=? "
            "ORDER BY symbol, zeit", (bereich, bis_zeit))
        for sym, gruppe in itertools.groupby(zeilen, key=lambda r: r[0]):
            seit = _strecke(reversed([(t, d) for _, t, d in gruppe]))
            if seit is not None:
                erg[json.loads(sym)] = seit
        return erg

    # -- Meta / Export -----------------------------------------------------
    def meta_setzen(self, k: str, v: str) -> None:
        c = self._verbindung()
        with c:
            c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (_dumps(k), _dumps(v)))

    def meta(self, k: str) -> str | None:
        z = self._verbindung().execute("SELECT v FROM meta WHERE k=?", (_dumps(k),)).fetchone()
        return None if z is None else str(json.loads(z[0]))

    def export_json(self, ziel: Path) -> None:
        """Deterministischer Export, gestreamt (nie das ganze Dokument im Speicher).

        Format wie `json.dumps(doc, sort_keys=True, indent=1)` mit den Schluesseln
        commit, kanten, knoten, version (in dieser, also sortierter, Reihenfolge). Knoten
        nach Id, Kanten nach (art, von, nach, Serialisierung). `seit` fehlt (R17).
        """
        c = self._verbindung()
        ziel = Path(ziel)
        ziel.parent.mkdir(parents=True, exist_ok=True)

        def knoten_ohne_seit() -> Iterator[Any]:
            for (wert,) in c.execute(self._KNOTEN_SQL):
                d = json.loads(wert)
                d["attr"] = [p for p in d["attr"] if p[0] != "seit"]
                yield d

        with open(ziel, "wb", buffering=1 << 20) as f:
            kopf = '{\n "commit": ' + json.dumps(self.meta("commit"), ensure_ascii=False) + ',\n "kanten": '
            f.write(_esc(kopf).encode("utf-8"))
            _liste_schreiben(f, (json.loads(w) for (w,) in c.execute(self._KANTEN_SQL)))
            f.write(b',\n "knoten": ')
            _liste_schreiben(f, knoten_ohne_seit())
            f.write(b',\n "version": 1\n}\n')


def stand_basis() -> Path:
    """Basisordner des lokalen Stands (wird hier nicht angelegt)."""
    home = os.environ.get("SELBSTBILD_HOME")
    if home:
        return Path(home)
    if sys.platform == "win32":
        lokal = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(lokal) / "selbstbild"
    return Path.home() / ".local" / "state" / "selbstbild"


def pfad_kennung(wurzel: Path) -> str:
    return hashlib.sha256(str(Path(wurzel).resolve()).encode("utf-8", "surrogateescape")).hexdigest()[:12]


def stand_ordner(wurzel: Path, name: str) -> Path:
    ordner = stand_basis() / f"{name}-{pfad_kennung(wurzel)}"
    ordner.mkdir(parents=True, exist_ok=True)
    return ordner
