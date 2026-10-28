"""Ergebnis-Speicher und kanonischer Export der Analyse.

Speicher: `analyse.sqlite` neben `graph.sqlite` (graph.sqlite wird nie geoeffnet). Tabelle
`lauf` mit einer Zeile je (commit, version, parameter_digest); in `daten` steht das kanonische
JSON der Analyse (dieselben Bytes wie `export_json`). Gleicher Schluessel ersetzt die Zeile.

Zahlen im Export stehen als Strings mit fester Nachkommazahl: kohaesion/aehnlichkeit 4,
Hotspot-wert und -teile 6, pagerank 8, betweenness 6, Kolonie-Werte 6.

Laufzeiten (`statistik`-Schluessel mit Praefix `laufzeit_`) stehen NICHT im Export: sie aendern sich
von Lauf zu Lauf und wuerden den byte-gleichen Export brechen. `speichern` legt sie in der Nebentabelle
`messung` ab, `lesen` gibt sie in `statistik` zurueck (goldsatz.vergleichen braucht sie).
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
from .kolonie import KolonieErgebnis, KolonieParameter
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


LAUFZEIT_PRAEFIX = "laufzeit_"


def _ist_laufzeit(schluessel: str) -> bool:
    return schluessel.startswith(LAUFZEIT_PRAEFIX)


def _z(wert: float, stellen: int) -> str:
    return f"{wert:.{stellen}f}"


def _paare(paare: tuple[tuple[str, float], ...], stellen: int) -> list[list[Any]]:
    return [[k, _z(w, stellen)] for k, w in paare]


def _kolonie(k: KolonieErgebnis) -> dict[str, Any]:
    p = k.parameter
    return {
        "besuche": [list(x) for x in k.besuche],
        "bereiche": [list(b) for b in k.bereiche],
        "parameter": {"agenten": p.agenten, "schritte": p.schritte, "verfall": _z(p.verfall, 6),
                      "ablage": _z(p.ablage, 6), "erkundung": _z(p.erkundung, 6)},
        "spuren": [[a, b, _z(w, 6)] for a, b, w in k.spuren],
        "tote_ecken": list(k.tote_ecken),
        "zentrum": list(k.zentrum),
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
    yield "statistik", [list(x) for x in sorted(a.statistik) if not _ist_laufzeit(x[0])]
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


def digest_von(parameter: tuple[tuple[str, str], ...]) -> str:
    return hashlib.sha256(_dumps([list(p) for p in sorted(parameter)]).encode("utf-8")).hexdigest()


def parameter_digest(a: Analyse) -> str:
    return digest_von(a.parameter)


def speichern(stand: Path, a: Analyse) -> None:
    """Schreibt (ersetzt) den Lauf in `<stand>/analyse.sqlite`. `stand` ist der Ordner mit graph.sqlite.

    `lauf.daten` = Export-Bytes; die Laufzeiten des Laufs ersetzen die Zeilen in `messung`."""
    ordner = Path(stand)
    ordner.mkdir(parents=True, exist_ok=True)
    puffer = io.BytesIO()
    _schreiben(a, puffer)
    schluessel = (a.commit, a.version, parameter_digest(a))
    c = sqlite3.connect(str(ordner / "analyse.sqlite"))
    try:
        with c:
            c.execute("CREATE TABLE IF NOT EXISTS lauf (commit_id TEXT NOT NULL, version INTEGER NOT NULL,"
                      " parameter_digest TEXT NOT NULL, daten BLOB NOT NULL,"
                      " PRIMARY KEY (commit_id, version, parameter_digest))")
            c.execute("CREATE TABLE IF NOT EXISTS messung (commit_id TEXT NOT NULL, version INTEGER NOT NULL,"
                      " parameter_digest TEXT NOT NULL, schluessel TEXT NOT NULL, wert INTEGER NOT NULL,"
                      " PRIMARY KEY (commit_id, version, parameter_digest, schluessel))")
            c.execute("INSERT OR REPLACE INTO lauf VALUES (?, ?, ?, ?)", (*schluessel, puffer.getvalue()))
            c.execute("DELETE FROM messung WHERE commit_id=? AND version=? AND parameter_digest=?", schluessel)
            c.executemany("INSERT INTO messung VALUES (?, ?, ?, ?, ?)",
                          [(*schluessel, k, int(w)) for k, w in sorted(a.statistik) if _ist_laufzeit(k)])
    finally:
        c.close()


def _f(x: str) -> float:
    return float(x)


def _aus_json(d: dict[str, Any], laufzeiten: list[tuple[str, int]]) -> Analyse:
    k = d["kolonie"]
    kol = None
    if k is not None:
        p = k["parameter"]
        kol = KolonieErgebnis(
            besuche=tuple((str(i), int(n)) for i, n in k["besuche"]),
            spuren=tuple((str(a), str(b), _f(w)) for a, b, w in k["spuren"]),
            bereiche=tuple(tuple(b) for b in k["bereiche"]),
            zentrum=tuple(k["zentrum"]), tote_ecken=tuple(k["tote_ecken"]),
            parameter=KolonieParameter(int(p["agenten"]), int(p["schritte"]), _f(p["verfall"]),
                                       _f(p["ablage"]), _f(p["erkundung"])))
    return Analyse(
        commit=d["commit"], version=int(d["version"]),
        parameter=tuple((str(a), str(b)) for a, b in d["parameter"]),
        bereiche=tuple(Bereich(b["id"], b["name"], tuple(b["dateien"]), _f(b["kohaesion"]))
                       for b in d["bereiche"]),
        pagerank_top=tuple((str(i), _f(w)) for i, w in d["pagerank_top"]),
        betweenness_top=tuple((str(i), _f(w)) for i, w in d["betweenness_top"]),
        hotspots=tuple(Hotspot(h["datei"], _f(h["wert"]), tuple((str(n), _f(w)) for n, w in h["teile"]))
                       for h in d["hotspots"]),
        tot=tuple(TotFund(t["symbol"], t["stufe"], tuple(t["gruende"]), t["zuletzt"]) for t in d["tot"]),
        doppelungen=tuple(Doppelung(x["art"], tuple(x["mitglieder"]), _f(x["aehnlichkeit"]))
                          for x in d["doppelungen"]),
        kolonie=kol,
        luecken=tuple((str(a), int(b)) for a, b in d["luecken"]),
        statistik=tuple(sorted([(str(a), int(b)) for a, b in d["statistik"]] + laufzeiten)))


def lesen(stand: Path, commit: str, version: int, digest: str) -> Analyse | None:
    """Gespeicherten Lauf zurueckholen (Zahlen gerundet wie im Export; erneuter Export = gleiche Bytes).
    None, wenn es `analyse.sqlite` oder den Lauf nicht gibt. Oeffnet nur lesend."""
    db = Path(stand) / "analyse.sqlite"
    if not db.is_file():
        return None
    c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        schluessel = (commit, version, digest)
        z = c.execute("SELECT daten FROM lauf WHERE commit_id=? AND version=? AND parameter_digest=?",
                      schluessel).fetchone()
        if z is None:
            return None
        try:
            laufzeiten = [(str(k), int(w)) for k, w in c.execute(
                "SELECT schluessel, wert FROM messung WHERE commit_id=? AND version=? AND parameter_digest=?"
                " ORDER BY schluessel", schluessel)]
        except sqlite3.OperationalError:  # Stand von vor der Nebentabelle
            laufzeiten = []
    finally:
        c.close()
    return _aus_json(json.loads(bytes(z[0]).decode("utf-8")), laufzeiten)
