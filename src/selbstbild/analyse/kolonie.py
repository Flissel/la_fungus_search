"""Kolonie als Herausforderer (E2): MCMP-Ameisenmechanik auf dem Dateigraphen statt im Embedding-Raum.

Uebernommen aus embeddinggemma (nicht importiert): Agenten starten an Einstiegspunkten, laufen
entlang der Kanten, legen Pheromon ab, das Pheromon verfaellt, Besuche werden gezaehlt.

Start (Ruling S11): Die Wurzeldateien werden einmal per `rng.permutation` gemischt. Agent i startet
auf der i-ten Wurzel dieser Reihenfolge; nur wenn es mehr Agenten als Wurzeln gibt, wird rundlaufend
wiederholt. Ein Agent in einer Sackgasse (keine ausgehende Kante) startet neu auf der naechsten Wurzel
eines gemeinsamen Rundlauf-Zaehlers ueber ALLE Wurzeln (nicht auf seiner eigenen). So erreicht die
Kolonie auch bei mehr Wurzeln als Agenten jede Wurzel, sobald es genug Sackgassen gibt.

`budget_schritte` liefert die Schrittzahl fuer einen budgetgleichen Lauf (Agenten-Schritte ~ 10 x
Dateien, hoechstens `BUDGET_MAX_SCHRITTE`), den `lauf` in der Abnahme bzw. mit `--kolonie-budget`
zusaetzlich rechnet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .erreichbar import wurzeln
from .lader import SEED, GraphDaten, datei_von, dateigraph

_SPUREN_MAX = 2000
_ZENTRUM_MAX = 50
_SCHWELLE_ANTEIL = 0.1  # Bereiche: Kanten mit Pheromon >= 10 % des Maximums
_STELLEN = 6
BUDGET_JE_DATEI = 10       # Agenten-Schritte je Datei im budgetgleichen Lauf (Ruling S11)
BUDGET_MAX_SCHRITTE = 5000


@dataclass(frozen=True)
class KolonieParameter:
    agenten: int = 192
    schritte: int = 400
    # Standardwerte aus dem MCMP-Code:
    # verfall: src/embeddinggemma/mcmp_rag.py MCPMRetriever.__init__ `pheromone_decay=0.95`
    #   (Multiplikator je Schritt, wie decay_pheromones in mcmp/simulation.py: trail *= decay).
    # ablage: mcmp/simulation.py deposit_pheromones `amount = energy * trail_strength * 0.1`
    #   mit Agent.energy = 1.0 und trail_strength = 1.0 (mcmp_rag.py class Agent) => 0.1.
    # erkundung: mcmp_rag.py MCPMRetriever.__init__ `exploration_bonus=0.1`.
    verfall: float = 0.95
    ablage: float = 0.1
    erkundung: float = 0.1


@dataclass(frozen=True)
class KolonieErgebnis:
    besuche: tuple[tuple[str, int], ...]            # (Datei, Besuche), nur > 0, sortiert nach Id
    spuren: tuple[tuple[str, str, float], ...]      # staerkste 2000 Kanten (von, nach, Pheromon)
    bereiche: tuple[tuple[str, ...], ...]           # Zusammenhangskomponenten der starken Spuren
    zentrum: tuple[str, ...]                        # die 50 meistbesuchten Dateien
    tote_ecken: tuple[str, ...]                     # nie besuchte Dateien, sortiert
    parameter: KolonieParameter = KolonieParameter()


def budget_schritte(dateien: int, agenten: int = KolonieParameter.agenten) -> int:
    """Schritte je Agent, damit Agenten x Schritte ~ 10 x Dateien (mindestens 1, hoechstens 5000)."""
    if agenten <= 0:
        return 1
    return max(1, min(BUDGET_MAX_SCHRITTE, math.ceil(BUDGET_JE_DATEI * dateien / agenten)))


def _leer(p: KolonieParameter) -> KolonieErgebnis:
    return KolonieErgebnis((), (), (), (), (), p)


def _startdateien(d: GraphDaten, index: dict[str, int]) -> list[int]:
    """Produktivwurzeln (Symbole auf ihre Datei abgebildet) als sortierte, eindeutige Knotenindizes."""
    produktiv, _ = wurzeln(d)
    dateien = {datei_von(i) if i.startswith("symbol:") else i for i in produktiv}
    return sorted(index[i] for i in dateien if i in index)


def _bereiche(ids: tuple[str, ...], paare: list[tuple[int, int]]) -> tuple[tuple[str, ...], ...]:
    eltern: dict[int, int] = {}

    def finde(x: int) -> int:
        eltern.setdefault(x, x)
        while eltern[x] != x:
            eltern[x] = eltern[eltern[x]]
            x = eltern[x]
        return x

    for a, b in paare:
        ra, rb = finde(a), finde(b)
        if ra != rb:
            eltern[max(ra, rb)] = min(ra, rb)
    gruppen: dict[int, list[str]] = {}
    for x in sorted(eltern):
        gruppen.setdefault(finde(x), []).append(ids[x])
    komp = [tuple(sorted(g)) for g in gruppen.values()]
    return tuple(sorted(komp, key=lambda g: (-len(g), g)))


def kolonie_laufen(d: GraphDaten, p: KolonieParameter = KolonieParameter()) -> KolonieErgebnis:
    sicht = dateigraph(d)
    ids, g = sicht.ids, sicht.graph
    start = _startdateien(d, sicht.index)
    if not start or p.agenten <= 0:
        return _leer(p)
    n, m = len(ids), g.ecount()
    # CSR nach Quelle; Kantenindex e = Position in der nach (von, nach) sortierten Kantenliste.
    von = np.array([e[0] for e in g.get_edgelist()], dtype=np.int64)
    nach = np.array([e[1] for e in g.get_edgelist()], dtype=np.int64)
    gew = np.array(g.es["weight"], dtype=np.float64) if m else np.zeros(0)
    ordnung = np.argsort(von, kind="stable")
    ziel, gewicht = nach[ordnung], gew[ordnung]
    zeiger = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(von, minlength=n), out=zeiger[1:])
    pheromon = np.zeros(m, dtype=np.float64)  # indiziert wie `ordnung`-Positionen (CSR-Reihenfolge)
    besuche = np.zeros(n, dtype=np.int64)

    rng = np.random.default_rng(SEED)
    # Ruling S11: gestreute Reihenfolge aller Wurzeln; Agent i startet auf reihe[i mod Anzahl].
    reihe = np.array(start, dtype=np.int64)[rng.permutation(len(start))]
    pos = reihe[np.arange(p.agenten) % len(reihe)]
    zaehler = p.agenten % len(reihe)  # naechste Wurzel fuer einen Neustart (Rundlauf ueber alle)
    np.add.at(besuche, pos, 1)
    for _ in range(p.schritte):
        grad = zeiger[pos + 1] - zeiger[pos]
        lebend = grad > 0
        # Sackgasse (keine ausgehende Kante): Neustart auf den naechsten Wurzeln des Rundlaufs,
        # in Agenten-Reihenfolge (deterministisch).
        if not lebend.all():
            tot_idx = np.flatnonzero(~lebend)
            pos[tot_idx] = reihe[(zaehler + np.arange(len(tot_idx))) % len(reihe)]
            zaehler = (zaehler + len(tot_idx)) % len(reihe)
            np.add.at(besuche, pos[tot_idx], 1)  # der Neustart verbraucht den Schritt
        if not lebend.any():
            continue
        idx_a = np.flatnonzero(lebend)
        gr = grad[idx_a]
        breite = int(gr.max())
        spalte = np.arange(breite)
        maske = spalte[None, :] < gr[:, None]
        kanten = np.minimum(zeiger[pos[idx_a]][:, None] + spalte[None, :], max(m - 1, 0))
        w = np.where(maske, gewicht[kanten] * (1.0 + pheromon[kanten]), 0.0)
        wsum = w.sum(axis=1, keepdims=True)
        prob = w / wsum * (1.0 - p.erkundung) + p.erkundung * maske / gr[:, None]
        wahl = (prob.cumsum(axis=1) < rng.random((len(idx_a), 1)) * prob.sum(axis=1, keepdims=True)).sum(axis=1)
        wahl = np.minimum(wahl, gr - 1)  # Rundungsschutz
        gewaehlt = kanten[np.arange(len(idx_a)), wahl]
        np.add.at(pheromon, gewaehlt, p.ablage)  # Ablage auf der begangenen Kante
        pos[idx_a] = ziel[gewaehlt]
        np.add.at(besuche, pos[idx_a], 1)
        pheromon *= p.verfall  # globaler Verfall nach jedem Schritt

    vonpos = von[ordnung]
    besucht = np.flatnonzero(besuche > 0)
    bes = tuple((ids[i], int(besuche[i])) for i in besucht)  # ids sortiert => Ergebnis nach Id sortiert
    tote = tuple(ids[i] for i in range(n) if besuche[i] == 0)
    zentrum = tuple(i for _, i in sorted(((-int(besuche[i]), ids[i]) for i in besucht))[:_ZENTRUM_MAX])
    kandidaten = sorted(((-float(pheromon[e]), ids[vonpos[e]], ids[ziel[e]]) for e in np.flatnonzero(pheromon > 0)))
    spuren = tuple((a, b, round(-w, _STELLEN)) for w, a, b in kandidaten[:_SPUREN_MAX])
    if kandidaten:
        schwelle = -kandidaten[0][0] * _SCHWELLE_ANTEIL
        paare = [(int(vonpos[e]), int(ziel[e])) for e in np.flatnonzero(pheromon >= schwelle)]
        bereiche = _bereiche(ids, paare)
    else:
        bereiche = ()
    return KolonieErgebnis(bes, spuren, bereiche, zentrum, tote, p)
