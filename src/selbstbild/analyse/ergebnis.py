"""Ergebnis-Speicher und kanonischer Export der Analyse.

Speicher: `analyse.sqlite` neben `graph.sqlite` (graph.sqlite wird nie geoeffnet). Tabelle
`lauf` mit einer Zeile je (commit, version, parameter_digest); in `daten` steht das kanonische
JSON der Analyse (dieselben Bytes wie `export_json`). Gleicher Schluessel ersetzt die Zeile.

Zahlen im Export stehen als Strings mit fester Nachkommazahl: kohaesion/aehnlichkeit 4,
Hotspot-wert und -teile 6, pagerank 8, betweenness 6, Kolonie-Werte 6.
"""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from ..speicher import _esc
from .bereiche import Bereich
from .doppelt import Doppelung
from .kolonie import KolonieErgebnis
from .tot import TotFund
from .zentral import Hotspot


@dataclass(frozen=True)
class Analyse:
    commit: str
    version: int
    parameter: tuple[tuple[str, str], ...]
    bereiche: tuple[Bereich, ...]
    pagerank_top: tuple[tuple[str, float], ...]     # die 500 hoechsten
    betweenness_top: tuple[tuple[str, float], ...]  # die 200 hoechsten
    hotspots: tuple[Hotspot, ...]
    tot: tuple[TotFund, ...]
    doppelungen: tuple[Doppelung, ...]
    kolonie: KolonieErgebnis | None
    luecken: tuple[tuple[str, int], ...]
    statistik: tuple[tuple[str, int], ...] = ()


def _z(wert: float, stellen: int) -> str:
    return f"{wert:.{stellen}f}"


def _paare(paare: tuple[tuple[str, float], ...], stellen: int) -> list[list[Any]]:
    return [[k, _z(w, stellen)] for k, w in paare]


def _kolonie(k: KolonieErgebnis) -> dict[str, Any]:
    p = k.parameter
    return {
        "besuche": [list(x) for x in k.besuche],
        "bereiche": _paare(k.bereiche, 6),
        "parameter": {"agenten": p.agenten, "schritte": p.schritte, "verfall": _z(p.verfall, 6),
                      "ablage": _z(p.ablage, 6), "erkundung": _z(p.erkundung, 6)},
        "spuren": _paare(k.spuren, 6),
        "tote_ecken": list(k.tote_ecken),
        "zentrum": _paare(k.zentrum, 6),
    }


def _teile(a: Analyse) -> Iterator[tuple[str, Any]]:
    """Schluessel in sortierter Reihenfolge; grosse Listen als Generatoren (gestreamt)."""
    yield "bereiche", ({"dateien": list(b.dateien), "id": b.id, "kohaesion": _z(b.kohaesion, 4),
                        "name": b.name} for b in a.bereiche)
    yield "betweenness_top", _paare(a.betweenness_top, 6)
    yield "commit", a.commit
    yield "doppelungen", ({"aehnlichkeit": _z(d.aehnlichkeit, 4), "art": d.art,
                           "mitglieder": list(d.mitglieder)} for d in a.doppelungen)
    yield "hotspots", ({"datei": h.datei, "teile": _paare(h.teile, 6), "wert": _z(h.wert, 6)}
                       for h in a.hotspots)
    yield "kolonie", None if a.kolonie is None else _kolonie(a.kolonie)
    yield "luecken", [list(x) for x in sorted(a.luecken)]
    yield "pagerank_top", _paare(a.pagerank_top, 8)
    yield "parameter", [list(x) for x in a.parameter]
    yield "statistik", [list(x) for x in sorted(a.statistik)]
    yield "tot", ({"gruende": list(t.gruende), "stufe": t.stufe, "symbol": t.symbol,
                   "zuletzt": t.zuletzt} for t in a.tot)
    yield "version", a.version


def _dumps(x: Any) -> str:
    return _esc(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _schreiben(a: Analyse, f: BinaryIO) -> None:
    def w(s: str) -> None:
        f.write(s.encode("utf-8"))

    w("{")
    for i, (schluessel, wert) in enumerate(_teile(a)):
        w(("\n" if i == 0 else ",\n") + _dumps(schluessel) + ":")
        if isinstance(wert, list) or hasattr(wert, "__next__"):
            w("[")
            erstes = True
            for el in wert:
                w(("\n " if erstes else ",\n ") + _dumps(el))
                erstes = False
            w("]" if erstes else "\n]")
        else:
            w(_dumps(wert))
    w("\n}\n")


def export_json(a: Analyse, ziel: Path) -> None:
    """Kanonisches, gestreamtes JSON ohne Zeitstempel (gleiche Analyse -> gleiche Bytes)."""
    ziel = Path(ziel)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with open(ziel, "wb", buffering=1 << 20) as f:
        _schreiben(a, f)


def parameter_digest(a: Analyse) -> str:
    return hashlib.sha256(_dumps([list(p) for p in sorted(a.parameter)]).encode("utf-8")).hexdigest()


def speichern(stand: Path, a: Analyse) -> None:
    """Schreibt (ersetzt) den Lauf in `<stand>/analyse.sqlite`. `stand` ist der Ordner mit graph.sqlite."""
    ordner = Path(stand)
    ordner.mkdir(parents=True, exist_ok=True)
    puffer = io.BytesIO()
    _schreiben(a, puffer)
    c = sqlite3.connect(str(ordner / "analyse.sqlite"))
    try:
        with c:
            c.execute("CREATE TABLE IF NOT EXISTS lauf (commit_id TEXT NOT NULL, version INTEGER NOT NULL,"
                      " parameter_digest TEXT NOT NULL, daten BLOB NOT NULL,"
                      " PRIMARY KEY (commit_id, version, parameter_digest))")
            c.execute("INSERT OR REPLACE INTO lauf VALUES (?, ?, ?, ?)",
                      (a.commit, a.version, parameter_digest(a), puffer.getvalue()))
    finally:
        c.close()
