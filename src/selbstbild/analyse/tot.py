"""Toter Code in drei Stufen (Spec 6.2): sicher_tot, wahrscheinlich_tot, verdaechtig.

Ein falsches `sicher_tot` ist der teure Fehler (daraus werden Issues). Im Zweifel wird abgestuft."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ..sprachen import SPRACHEN
from . import ist_testdatei
from .erreichbar import Erreichbarkeit
from .lader import GraphDaten, datei_von

TOT_TAGE: int = 90
_SEKUNDEN = TOT_TAGE * 86400

Stufe = Literal["sicher_tot", "wahrscheinlich_tot", "verdaechtig"]

_TS_ENDUNGEN = tuple(sorted(e for s in ("typescript", "tsx", "javascript") for e in SPRACHEN[s]))


@dataclass(frozen=True)
class TotFund:
    symbol: str
    stufe: Stufe
    gruende: tuple[str, ...]
    zuletzt: int | None


def _attr(d: GraphDaten, i: str) -> dict[str, str]:
    k = d.knoten.get(i)
    return {} if k is None else dict(k.attr)


def _pfad(symbol_id: str) -> str:
    return symbol_id.split(":", 2)[2].rpartition("#")[0]


def _repo(symbol_id: str) -> str:
    return symbol_id.split(":", 2)[1]


def _dokumentiert(symbol_id: str, doku: Mapping[str, frozenset[str]]) -> bool:
    """Ruling S5: Doku des eigenen Repos oder eines Eltern-Repos (Namenspraefix an `/`-Grenze)."""
    repo = _repo(symbol_id)
    name = symbol_id.rpartition("#")[2].rpartition(".")[2]
    teile = repo.split("/")
    return any(name in doku.get("/".join(teile[:n]), frozenset()) for n in range(1, len(teile) + 1))


def _zuletzt(d: GraphDaten, symbol_id: str) -> int | None:
    a = _attr(d, datei_von(symbol_id))
    if a.get("aenderungen") == "0":
        return None  # keine Historie: Alter unbekannt
    try:
        z = None if a.get("zuletzt") is None else int(a["zuletzt"])
    except ValueError:
        return None  # unlesbar: nicht als alt werten (im Zweifel abstufen)
    return z if z is not None and z > 0 else None


def _ausnahmen(d: GraphDaten, e: Erreichbarkeit, i: str) -> set[str]:
    """Gruende, aus denen ein Symbol nie `sicher_tot` sein darf (Global Constraints, E4, Ruling S4)."""
    erg: set[str] = set()
    pfad = _pfad(i)
    qual = i.rpartition("#")[2]
    einfach = qual.rpartition(".")[2]
    art = _attr(d, i).get("symbolart")
    datei = _attr(d, datei_von(i))
    if pfad.endswith(".rs"):
        erg.add("rust_r22")
    if datei.get("fehlerhaft") == "1" or _attr(d, i).get("fehlerhaft") == "1":
        erg.add("fehlerhaft")
    if datei.get("spezial_fehlt") == "1":
        erg.add("spezial_fehlt")
    if len(einfach) > 4 and einfach.startswith("__") and einfach.endswith("__"):
        erg.add("dunder")
    if ist_testdatei(pfad):
        erg.add("testdatei")
    if art == "methode" and "." in qual:
        eltern = f"{i.rpartition('#')[0]}#{qual.rpartition('.')[0]}"
        if eltern in e.irgendwie and _attr(d, eltern).get("symbolart") == "klasse":
            erg.add("methode_lebender_klasse")
    if pfad.endswith(".go") and art == "methode":
        erg.add("go_methode")
    if pfad.endswith(_TS_ENDUNGEN) and art in ("klasse", "methode"):
        erg.add("ts_klasse")
    if pfad.endswith(_TS_ENDUNGEN) and art == "funktion" and einfach[:1].isupper():
        erg.add("jsx_komponente")
    return erg


def tot_finden(d: GraphDaten, e: Erreichbarkeit, doku: Mapping[str, frozenset[str]]) -> tuple[TotFund, ...]:
    """Nur Symbolknoten. Sortiert nach Symbol-Id, ohne Uhr (Stichtag = `d.commit_zeit`)."""
    grenze = d.commit_zeit - _SEKUNDEN
    funde: list[TotFund] = []
    for i in sorted(k for k, n in d.knoten.items() if n.art == "symbol"):
        zuletzt = _zuletzt(d, i)
        pfad = _pfad(i)
        if i in e.nur_test:
            if ist_testdatei(pfad):
                continue  # Testcode ist per Definition nur von Tests erreicht
            funde.append(TotFund(i, "verdaechtig", ("nur_test",), zuletzt))
            continue
        alt = zuletzt is not None and zuletzt <= grenze
        if i in e.irgendwie:
            if i in e.sicher or not alt or pfad.endswith(_TS_ENDUNGEN):
                continue  # TS: unsichere Kante genuegt (R15), kein Fund
            gruende = {"nur_unsicher", "alt_90"}
            ausnahmen = _ausnahmen(d, e, i)
            gruende |= ausnahmen
            if _dokumentiert(i, doku):
                gruende.add("doku_verweis")
            funde.append(TotFund(i, "wahrscheinlich_tot", tuple(sorted(gruende)), zuletzt))
            continue
        verwiesen = _dokumentiert(i, doku)
        ausnahmen = _ausnahmen(d, e, i)
        gruende = {"unerreichbar", "doku_verweis" if verwiesen else "kein_doku_verweis"} | ausnahmen
        if alt:
            gruende.add("alt_90")
        sicher = alt and not verwiesen and not ausnahmen
        funde.append(TotFund(i, "sicher_tot" if sicher else "wahrscheinlich_tot", tuple(sorted(gruende)), zuletzt))
    return tuple(funde)
