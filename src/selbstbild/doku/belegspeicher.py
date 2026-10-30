"""Gemerkte Belege der Aussagen im Graph (Task 15).

`aussagen_lesen` verankert jedes Mal neu (Digest = heutiger Code); ein Vergleich gegen den heutigen Stand waere immer
`beleg_ok`. Darum merkt sich der Loop die Belege jeder Aussage im Graph: Knoten `aussage:<id>` und je Datei eine
Kante `belegt_durch` mit den Belegen. Ein Signal heisst: ein gemerkter Beleg ist heute geaendert oder verschwunden.
`graph_bauen` uebernimmt diese Knoten/Kanten unveraendert aus der Basis (sie haben kein `space`-Attribut).
"""

from __future__ import annotations

from collections.abc import Sequence

from . import aussagen as au
from .anker import Beleg
from .graph import Graph, Kante, Knoten, kante_id


def knoten_id(aid: str) -> str:
    return f"aussage:{aid}"


def gespeicherte_belege(graph: Graph, aid: str) -> tuple[Beleg, ...] | None:
    nid = knoten_id(aid)
    if nid not in graph.knoten:
        return None
    return tuple(b for e in sorted(graph.kanten.values(), key=lambda e: e.id)
                 if e.art == "belegt_durch" and e.von == nid for b in e.belege)


def aussage_entfernen(graph: Graph, aid: str) -> None:
    nid = knoten_id(aid)
    graph.knoten.pop(nid, None)
    for eid in [e.id for e in graph.kanten.values() if e.von == nid]:
        del graph.kanten[eid]


def aussage_speichern(graph: Graph, aussage: au.Aussage, belege: Sequence[Beleg], quelle: str, *,
                      aid: str | None = None, text: str | None = None) -> None:
    """Knoten `aussage:<id>` und je Datei eine Kante `belegt_durch` (ersetzt den alten Stand; ohne Belege: weg)."""
    aid = aid or aussage.id
    aussage_entfernen(graph, aid)
    if not belege:
        return
    nid = knoten_id(aid)
    graph.knoten_hinzu(Knoten(nid, "aussage", au.erster_satz(text if text is not None else aussage.text)[:160],
                              datei=aussage.datei, attribute={"ueberschrift": aussage.ueberschrift}))
    je_datei: dict[str, dict[tuple[str, str, str], Beleg]] = {}
    for b in belege:
        je_datei.setdefault(b.file, {}).setdefault((b.art, b.file, b.anker), b)
    for datei, bs in sorted(je_datei.items()):
        ziel = f"modul:{datei}"
        graph.kante_hinzu(Kante(kante_id("belegt_durch", nid, ziel), "belegt_durch", nid, ziel, quelle,
                                tuple(bs.values())))


def beleg_dateien(graph: Graph) -> set[str]:
    """Alle Dateien, auf die gemerkte Belege zeigen (Hops und Aussagen)."""
    return {b.file for e in graph.kanten.values() if e.art in ("hop", "belegt_durch") for b in e.belege}
