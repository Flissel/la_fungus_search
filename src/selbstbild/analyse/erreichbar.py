"""Erreichbarkeit: was ist von Einstiegspunkten aus erreichbar (sicher / irgendwie / nur Test)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from . import ist_testdatei
from .lader import GraphDaten, datei_von

_KANTEN = ("ruft_auf", "importiert", "definiert_route", "registriert")
_QUELLEN_PRODUKTIV = ("definiert_route", "registriert")


@dataclass(frozen=True)
class Erreichbarkeit:
    sicher: frozenset[str]
    irgendwie: frozenset[str]
    nur_test: frozenset[str]
    wurzeln: frozenset[str]


def _attr(d: GraphDaten, i: str, schluessel: str) -> str | None:
    return dict(d.knoten[i].attr).get(schluessel)


def _pfad(d: GraphDaten, i: str) -> str:
    k = d.knoten[i]
    if k.beleg is not None:
        return k.beleg.pfad
    kopf = i.rpartition("#")[0] if k.art == "symbol" else i
    return kopf.split(":", 2)[2] if kopf.count(":") >= 2 else kopf


def _arten(d: GraphDaten, i: str) -> set[str]:
    return {x for x in (_attr(d, i, "einstieg") or "").split(",") if x}


def wurzeln(d: GraphDaten) -> tuple[frozenset[str], frozenset[str]]:
    """(Produktivwurzeln, Testwurzeln) unter den Datei- und Symbolknoten."""
    ids = {i for i, k in d.knoten.items() if k.art in ("datei", "symbol")}
    produktiv: set[str] = set()
    test: set[str] = set()
    for i in ids:
        arten = _arten(d, i)
        if arten - {"test"}:
            produktiv.add(i)
        if "test" in arten or ist_testdatei(_pfad(d, i)):
            test.add(i)
    for k in d.kanten:
        if k.art in _QUELLEN_PRODUKTIV and k.von in ids:
            produktiv.add(k.von)
    return frozenset(produktiv), frozenset(test)


def _init_dateien(datei_id: str, ids: set[str]) -> list[str]:
    """`import a.b.c` fuehrt `a/__init__.py` und `a/b/__init__.py` aus: Vorfahren-Pakete einer .py-Datei."""
    teile = datei_id.split(":", 2)
    if len(teile) < 3 or not teile[2].endswith(".py"):
        return []
    ordner = teile[2].replace("\\", "/").split("/")[:-1]
    erg = []
    for n in range(1, len(ordner) + 1):
        i = f"datei:{teile[1]}:{'/'.join(ordner[:n])}/__init__.py"
        if i != datei_id and i in ids:
            erg.append(i)
    return erg


def _bfs(start: frozenset[str], nachbarn: dict[str, list[str]],
         verschachtelt: dict[str, list[str]], ids: set[str]) -> frozenset[str]:
    gesehen: set[str] = set(start)
    schlange = deque(sorted(start))
    while schlange:
        n = schlange.popleft()
        folge = list(nachbarn.get(n, ()))
        if n.startswith("symbol:"):
            f = datei_von(n)
            if f in ids:
                folge.append(f)  # Symbol erreicht => Datei erreicht
        elif n.startswith("datei:"):
            folge.extend(_init_dateien(n, ids))  # Paket-__init__ laeuft beim Import mit
        folge.extend(verschachtelt.get(n, ()))  # Symbol erreicht => Verschachteltes (E4 verallgemeinert)
        for m in sorted(folge):
            if m not in gesehen:
                gesehen.add(m)
                schlange.append(m)
    return frozenset(gesehen)


def erreichbarkeit(d: GraphDaten) -> Erreichbarkeit:
    ids = {i for i, k in d.knoten.items() if k.art in ("datei", "symbol")}
    alle: dict[str, list[str]] = {}
    sicher: dict[str, list[str]] = {}
    for k in d.kanten:
        if k.art not in _KANTEN or k.von not in ids or k.nach not in ids or k.von == k.nach:
            continue
        alle.setdefault(k.von, []).append(k.nach)
        if k.sicher:
            sicher.setdefault(k.von, []).append(k.nach)
    # Verschachtelt = gleiche Datei, Qualname beginnt mit `<qualname>.` (Methoden, Closures, Wrapper).
    verschachtelt: dict[str, list[str]] = {}
    for i in sorted(ids):
        if d.knoten[i].art != "symbol":
            continue
        kopf, _, q = i.rpartition("#")
        teile = q.split(".")
        for n in range(1, len(teile)):
            eltern = f"{kopf}#{'.'.join(teile[:n])}"
            if eltern in ids:
                verschachtelt.setdefault(eltern, []).append(i)
    produktiv, test = wurzeln(d)
    s = _bfs(produktiv, sicher, verschachtelt, ids)
    irgendwie = _bfs(produktiv, alle, verschachtelt, ids)
    test_erreicht = _bfs(test, alle, verschachtelt, ids)
    return Erreichbarkeit(s, irgendwie, test_erreicht - irgendwie, produktiv)
