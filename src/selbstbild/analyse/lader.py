"""Lader: graph.sqlite (Stufe 1) nur lesend in eine Analysesicht (igraph) bringen."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import igraph

from ..speicher import Speicher, _kante_von, _knoten_von
from ..typen import Kante, Knoten, id_datei

ANALYSE_VERSION: int = 1
SEED: int = 20261028

_GEWICHT_SICHER = 1.0
_GEWICHT_UNSICHER = 0.5


@dataclass(frozen=True)
class GraphDaten:
    commit: str
    commit_zeit: int
    knoten: dict[str, Knoten]
    kanten: tuple[Kante, ...]
    luecken: dict[str, int]


@dataclass(frozen=True)
class Sicht:
    graph: igraph.Graph
    ids: tuple[str, ...]
    index: dict[str, int]


def _meta(c: sqlite3.Connection, k: str) -> str | None:
    # Schluessel und Werte sind wie in Speicher.meta als JSON abgelegt.
    z = c.execute("SELECT v FROM meta WHERE k=?", (json.dumps(k),)).fetchone()
    return None if z is None else str(json.loads(z[0]))


def _nicht_vermessen(stand: Path) -> RuntimeError:
    return RuntimeError(f"Stand {stand} ist nicht vermessen: erst `python -m selbstbild vermessen` ausfuehren")


def laden(stand: Path) -> GraphDaten:
    """`stand` ist der Ordner mit `graph.sqlite`. Nur lesend, eine Lese-Transaktion."""
    db = Path(stand) / "graph.sqlite"
    if not db.is_file():
        raise _nicht_vermessen(stand)
    c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
    try:
        c.execute("BEGIN")
        try:
            commit = _meta(c, "commit")
            if commit is None:
                raise _nicht_vermessen(stand)
            zeit = _meta(c, "commit_zeit")
            if zeit is None:
                bericht = _meta(c, "bericht")
                zeit = None if bericht is None else json.loads(bericht).get("commit_zeit")
            if zeit is None:
                raise RuntimeError(f"Stand {stand}: commit_zeit fehlt in meta und bericht (neu vermessen)")
            knoten = {k.id: k for k in (_knoten_von(json.loads(w))
                                        for (w,) in c.execute(Speicher._KNOTEN_SQL))}
            kanten = tuple(_kante_von(json.loads(w)) for (w,) in c.execute(Speicher._KANTEN_SQL))
        finally:
            c.execute("COMMIT")
    except sqlite3.OperationalError as e:  # fehlende Tabellen: kein vermessener Stand
        raise _nicht_vermessen(stand) from e
    finally:
        c.close()
    ohne = sum(1 for k in kanten if k.von not in knoten or k.nach not in knoten)
    return GraphDaten(commit, int(zeit), knoten, kanten,
                      {"kante_ohne_knoten": ohne})


def datei_von(symbol_id: str) -> str:
    """`symbol:<repo>:<pfad>#q` -> `datei:<repo>:<pfad>`."""
    kopf, _, _ = symbol_id.rpartition("#")  # Qualnamen enthalten nie '#', Pfade manchmal
    return "datei:" + kopf.removeprefix("symbol:")


def _sicht(ids: set[str], gewichte: dict[tuple[str, str], float]) -> Sicht:
    sortiert = tuple(sorted(ids))
    index = {i: n for n, i in enumerate(sortiert)}
    g = igraph.Graph(n=len(sortiert), directed=True)
    paare = sorted(gewichte)
    g.add_edges([(index[a], index[b]) for a, b in paare])
    g.es["weight"] = [gewichte[p] for p in paare]
    return Sicht(g, sortiert, index)


def _gewicht(k: Kante) -> float:
    return _GEWICHT_SICHER if k.sicher else _GEWICHT_UNSICHER


def dateigraph(d: GraphDaten) -> Sicht:
    """Knoten = Dateien; Kante a->b bei `importiert` a->b oder `ruft_auf` aus a nach einem Symbol in b."""
    dateien = {i for i, k in d.knoten.items() if k.art == "datei"}

    def datei(i: str) -> str | None:
        k = d.knoten.get(i)
        if k is None:
            return None
        if k.art == "datei":
            return i
        if k.art == "symbol":
            if k.beleg is not None:
                f = id_datei(k.beleg.repo, k.beleg.pfad)
                if f in dateien:
                    return f
            f = datei_von(i)
            return f if f in dateien else None
        return None

    w: dict[tuple[str, str], float] = {}
    ohne_datei = 0
    for k in d.kanten:
        if k.art == "importiert":
            a, b = (k.von if k.von in dateien else None), (k.nach if k.nach in dateien else None)
        elif k.art == "ruft_auf":
            a = datei(k.von)
            if a is None and k.von in d.knoten and d.knoten[k.von].art == "symbol":
                ohne_datei += 1
            b = datei(k.nach) if d.knoten.get(k.nach, None) and d.knoten[k.nach].art == "symbol" else None
        else:
            continue
        if a is None or b is None or a == b:
            continue
        w[(a, b)] = w.get((a, b), 0.0) + _gewicht(k)
    d.luecken["symbol_ohne_datei"] = ohne_datei  # Zuweisung, nicht Addition: wiederholbar
    return _sicht(dateien, w)


_SYMBOLKANTEN = ("ruft_auf", "importiert", "definiert_route", "registriert")


def symbolgraph(d: GraphDaten) -> Sicht:
    """Knoten = Datei- und Symbolknoten; Kanten = ruft_auf, importiert, definiert_route, registriert."""
    ids = {i for i, k in d.knoten.items() if k.art in ("datei", "symbol")}
    w: dict[tuple[str, str], float] = {}
    for k in d.kanten:
        if k.art not in _SYMBOLKANTEN or k.von not in ids or k.nach not in ids or k.von == k.nach:
            continue
        w[(k.von, k.nach)] = w.get((k.von, k.nach), 0.0) + _gewicht(k)
    return _sicht(ids, w)
