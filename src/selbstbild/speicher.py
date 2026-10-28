"""SQLite-Speicher: Parse-Cache, Graph, Symbolverlauf, JSON-Export."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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
CREATE TABLE IF NOT EXISTS knoten(id TEXT PRIMARY KEY, wert TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kanten(wert TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS symbolverlauf(
    nr INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
    zeit INTEGER NOT NULL, digest TEXT NOT NULL, commit_id TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS symbolverlauf_symbol ON symbolverlauf(symbol, nr);
"""


class Speicher:
    def __init__(self, pfad: Path) -> None:
        self.pfad = Path(pfad)
        self._c: sqlite3.Connection | None = None

    def _verbindung(self) -> sqlite3.Connection:
        if self._c is None:
            self.pfad.parent.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(str(self.pfad))
            c.execute("PRAGMA journal_mode=WAL")
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

    # -- Parse-Cache: Schluessel = Parser- + Grammatikversion, Blob, Sprache --
    @staticmethod
    def _cache_schluessel(blob: str, sprache: str) -> str:
        return _dumps([PARSER_VERSION, GRAMMATIK_VERSION, blob, sprache])

    def cache_holen(self, blob: str, sprache: str) -> DateiErgebnis | None:
        z = self._verbindung().execute(
            "SELECT wert FROM cache WHERE schluessel=?",
            (self._cache_schluessel(blob, sprache),)).fetchone()
        return None if z is None else _ergebnis_von(json.loads(z[0]))

    def cache_setzen(self, blob: str, sprache: str, ergebnis: DateiErgebnis) -> None:
        c = self._verbindung()
        with c:
            c.execute("INSERT OR REPLACE INTO cache VALUES(?,?)",
                      (self._cache_schluessel(blob, sprache), _dumps(_ergebnis_zu(ergebnis))))

    # -- Graph -------------------------------------------------------------
    def graph_ersetzen(self, commit: str, knoten: Iterable[Knoten], kanten: Iterable[Kante]) -> None:
        c = self._verbindung()
        with c:
            c.execute("DELETE FROM knoten")
            c.execute("DELETE FROM kanten")
            c.executemany("INSERT OR REPLACE INTO knoten VALUES(?,?)",
                          [(_dumps(k.id), _dumps(_knoten_zu(k))) for k in knoten])
            c.executemany("INSERT OR IGNORE INTO kanten VALUES(?)",
                          [(_dumps(_kante_zu(k)),) for k in kanten])
            c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (_dumps("commit"), _dumps(commit)))

    def knoten(self) -> list[Knoten]:
        z = self._verbindung().execute("SELECT wert FROM knoten").fetchall()
        return sorted((_knoten_von(json.loads(r[0])) for r in z), key=lambda k: k.id)

    def kanten(self) -> list[Kante]:
        z = self._verbindung().execute("SELECT wert FROM kanten").fetchall()
        return sorted((_kante_von(json.loads(r[0])) for r in z),
                      key=lambda k: (k.art, k.von, k.nach, _dumps(_kante_zu(k))))

    # -- Symbolverlauf: nur Digest-Wechsel werden festgehalten ---------------
    def symbolverlauf_merken(self, commit: str, commit_zeit: int, symbole: Mapping[str, str]) -> None:
        c = self._verbindung()
        with c:
            for sid in sorted(symbole):
                digest = symbole[sid]
                s = _dumps(sid)
                z = c.execute("SELECT digest FROM symbolverlauf WHERE symbol=? "
                              "ORDER BY nr DESC LIMIT 1", (s,)).fetchone()
                if z is None or z[0] != digest:
                    c.execute("INSERT INTO symbolverlauf(symbol,zeit,digest,commit_id) "
                              "VALUES(?,?,?,?)", (s, commit_zeit, digest, commit))

    def symbol_seit(self, symbol_id: str) -> int | None:
        z = self._verbindung().execute(
            "SELECT zeit FROM symbolverlauf WHERE symbol=? ORDER BY nr DESC LIMIT 1",
            (_dumps(symbol_id),)).fetchone()
        return None if z is None else int(z[0])

    # -- Meta / Export -----------------------------------------------------
    def meta_setzen(self, k: str, v: str) -> None:
        c = self._verbindung()
        with c:
            c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (_dumps(k), _dumps(v)))

    def meta(self, k: str) -> str | None:
        z = self._verbindung().execute("SELECT v FROM meta WHERE k=?", (_dumps(k),)).fetchone()
        return None if z is None else str(json.loads(z[0]))

    def export_json(self, ziel: Path) -> None:
        doc = {
            "version": 1,
            "commit": self.meta("commit"),
            "knoten": [_knoten_zu(k) for k in self.knoten()],
            "kanten": [_kante_zu(k) for k in self.kanten()],
        }
        text = _esc(json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1)) + "\n"
        ziel = Path(ziel)
        ziel.parent.mkdir(parents=True, exist_ok=True)
        ziel.write_bytes(text.encode("utf-8"))


def stand_ordner(wurzel: Path, name: str) -> Path:
    home = os.environ.get("SELBSTBILD_HOME")
    if home:
        basis = Path(home)
    elif sys.platform == "win32":
        lokal = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        basis = Path(lokal) / "selbstbild"
    else:
        basis = Path.home() / ".local" / "state" / "selbstbild"
    kurz = hashlib.sha256(str(Path(wurzel).resolve()).encode("utf-8", "surrogateescape")).hexdigest()[:12]
    ordner = basis / f"{name}-{kurz}"
    ordner.mkdir(parents=True, exist_ok=True)
    return ordner
