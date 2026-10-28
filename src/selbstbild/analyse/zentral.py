"""Zentralitaet und Hotspots: PageRank (Symbolgraph), Betweenness (Dateigraph, E3), Hotspot-Wert."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .lader import GraphDaten, dateigraph, datei_von, symbolgraph

DAEMPFUNG: float = 0.85
BETWEENNESS_CUTOFF: int = 4
_NACHKOMMA: int = 8


def pagerank(d: GraphDaten) -> dict[str, float]:
    """PageRank auf dem Symbolgraphen, gewichtet (Kantenstaerke = Naehe), auf 8 Stellen gerundet."""
    s = symbolgraph(d)
    if not s.ids:
        return {}
    werte = s.graph.pagerank(directed=True, damping=DAEMPFUNG,
                             weights="weight" if s.graph.ecount() else None)
    return {i: round(w, _NACHKOMMA) for i, w in zip(s.ids, werte)}


def betweenness(d: GraphDaten) -> dict[str, float]:
    """Betweenness auf dem Dateigraphen mit cutoff=4 (E3), durch das Maximum auf [0, 1] normiert.

    Bewusst ungewichtet: igraph liest Gewichte als Distanzen, unsere Gewichte sind aber
    Staerken (hoeher = enger). Unter 3 Dateien gibt es keine Zwischenstation: alles 0.
    """
    s = dateigraph(d)
    if len(s.ids) < 3:
        return {i: 0.0 for i in s.ids}
    roh = s.graph.betweenness(directed=True, cutoff=BETWEENNESS_CUTOFF, weights=None)
    spitze = max(roh)
    if spitze <= 0:
        return {i: 0.0 for i in s.ids}
    return {i: round(r / spitze, _NACHKOMMA) for i, r in zip(s.ids, roh)}


@dataclass(frozen=True)
class Hotspot:
    datei: str
    wert: float
    teile: tuple[tuple[str, float], ...]  # sortiert nach Name: aenderung, groesse, zentral


def _perzentile(werte: Mapping[str, float]) -> dict[str, float]:
    """Rangperzentil in [0, 1] mit mittlerem Rang bei Gleichstand: (kleiner + (gleich-1)/2) / (n-1).

    Ein einzelner Wert hat 1.0; lauter gleiche Werte haben 0.5.
    """
    n = len(werte)
    if n == 1:
        return {k: 1.0 for k in werte}
    sortiert = sorted(werte.values())
    erg: dict[str, float] = {}
    for k, w in werte.items():
        kleiner = sum(1 for x in sortiert if x < w)
        gleich = sum(1 for x in sortiert if x == w)
        erg[k] = (kleiner + (gleich - 1) / 2) / (n - 1)
    return erg


def _lebendigkeit(attr: tuple[tuple[str, str], ...]) -> float:
    for k, v in attr:
        if k == "lebendigkeit":
            try:
                return float(v)
            except ValueError:
                return 0.0
    return 0.0


def hotspots(d: GraphDaten, pr: Mapping[str, float], top: int = 50) -> tuple[Hotspot, ...]:
    """Wert = Produkt dreier Perzentile: zentral (PageRank Datei + Symbole), aenderung
    (Attr `lebendigkeit`, fehlend = 0), groesse (Anzahl Symbole, Ruling S1)."""
    dateien = sorted(i for i, k in d.knoten.items() if k.art == "datei")
    if not dateien:
        return ()
    zentral = {f: pr.get(f, 0.0) for f in dateien}
    groesse = {f: 0.0 for f in dateien}
    for i, k in d.knoten.items():
        if k.art == "symbol" and i.startswith("symbol:"):
            f = datei_von(i)
            if f in groesse:
                zentral[f] += pr.get(i, 0.0)
                groesse[f] += 1.0
    aenderung = {f: _lebendigkeit(d.knoten[f].attr) for f in dateien}
    pz, pa, pg = _perzentile(zentral), _perzentile(aenderung), _perzentile(groesse)
    erg = [Hotspot(f, round(pz[f] * pa[f] * pg[f], _NACHKOMMA),
                   (("aenderung", pa[f]), ("groesse", pg[f]), ("zentral", pz[f])))
           for f in dateien]
    erg.sort(key=lambda h: (-h.wert, h.datei))
    return tuple(erg[:top])
