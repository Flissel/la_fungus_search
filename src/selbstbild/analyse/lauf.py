"""Analyse-Lauf (Stufe 2): alle Analysen auf einem vermessenen Stand, dann speichern.

Ablauf: `laden`, Erreichbarkeit, Doku-Bezeichner, toter Code, Bereiche, PageRank, Betweenness,
Hotspots, Doppelungen, optional die Kolonie, dann `speichern` (analyse.sqlite neben graph.sqlite).
Nach `tot_finden` stuft der Namensverweis-Waechter (Ruling S8) `sicher_tot` ab, wenn der Name
anderswo im Code vorkommt.

`statistik` traegt `doppelt_eimer_uebersprungen`, `tot_generiert_uebersprungen` (Symbole in
Build-/generierten Dateien, Ruling S9) sowie die Laufzeiten `laufzeit_laden_ms`,
`laufzeit_mathe_ms` (alles nach dem Laden ausser der Kolonie) und `laufzeit_kolonie_ms`. Laufzeiten
stehen nicht im Export (byte-gleicher `analyse.json`, siehe `ergebnis`). Nur mit
`pruefen_reproduzierbar=True` laufen Mathe und Kolonie ein zweites Mal im selben Prozess;
`reproduzierbar_mathe` / `reproduzierbar_kolonie` sind dann 1 bei identischem Ergebnis, sonst 0.
Ohne die Pruefung fehlen beide Schluessel (goldsatz meldet dann None).

`luecken` wird erst nach allen Analysen aus `GraphDaten.luecken` gebaut (Ruling S3): `dateigraph`
traegt dort `symbol_ohne_datei` nach, und `bereiche_finden` ruft `dateigraph` immer auf.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..vermessen import stand_finden
from . import doppelt
from .bereiche import Bereich, bereiche_finden
from .doku_verweise import doku_bezeichner
from .doppelt import Doppelung
from .ergebnis import Analyse, digest_von, lesen, speichern
from .erreichbar import erreichbarkeit
from .kolonie import KolonieParameter, kolonie_laufen
from .lader import ANALYSE_VERSION, SEED, GraphDaten, _meta, datei_von, laden
from .tot import (NAME_MIN, TOT_TAGE, TotFund, generiert_zaehlen, namensverweis_abstufen, namensverweise,
                  tot_finden)
from .zentral import BETWEENNESS_CUTOFF, DAEMPFUNG, Hotspot, betweenness, hotspots, pagerank

PAGERANK_TOP = 500
BETWEENNESS_TOP = 200
HOTSPOTS_TOP = 50
DOPPELT_SCHWELLE = 0.9
AUFLOESUNG = 1.0


class NichtVermessen(RuntimeError):
    pass


class NichtAnalysiert(RuntimeError):
    pass


def parameter(kolonie: bool) -> tuple[tuple[str, str], ...]:
    p = {
        "aufloesung": f"{AUFLOESUNG}", "betweenness_cutoff": str(BETWEENNESS_CUTOFF),
        "betweenness_top": str(BETWEENNESS_TOP), "daempfung": f"{DAEMPFUNG}",
        "doppelt_schwelle": f"{DOPPELT_SCHWELLE}", "hotspots_top": str(HOTSPOTS_TOP),
        "kolonie": "an" if kolonie else "aus", "pagerank_top": str(PAGERANK_TOP),
        "namensverweis_min": str(NAME_MIN), "tot_regeln": "s9", "seed": str(SEED), "tot_tage": str(TOT_TAGE),
    }
    if kolonie:
        k = KolonieParameter()
        p.update({"kolonie_agenten": str(k.agenten), "kolonie_schritte": str(k.schritte),
                  "kolonie_verfall": f"{k.verfall}", "kolonie_ablage": f"{k.ablage}",
                  "kolonie_erkundung": f"{k.erkundung}"})
    return tuple(sorted(p.items()))


@dataclass(frozen=True)
class _Mathe:
    tot: tuple[TotFund, ...]
    bereiche: tuple[Bereich, ...]
    pagerank_top: tuple[tuple[str, float], ...]
    betweenness_top: tuple[tuple[str, float], ...]
    hotspots: tuple[Hotspot, ...]
    doppelungen: tuple[Doppelung, ...]
    eimer_uebersprungen: int
    tot_generiert: int


def _top(werte: dict[str, float], n: int) -> tuple[tuple[str, float], ...]:
    return tuple(sorted(werte.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def _mathe(wurzel: Path, d: GraphDaten) -> _Mathe:
    e = erreichbarkeit(d)
    doku = doku_bezeichner(wurzel, d.commit, d)
    tot = tot_finden(d, e, doku)
    kandidaten = [t.symbol for t in tot if t.stufe == "sicher_tot"]
    tot = namensverweis_abstufen(tot, namensverweise(wurzel, d.commit, kandidaten))  # Ruling S8
    del e, doku  # grosse Mengen vor den naechsten Schritten freigeben
    ber = bereiche_finden(d, AUFLOESUNG)
    pr = pagerank(d)
    pr_top = _top(pr, PAGERANK_TOP)
    hs = hotspots(d, pr, HOTSPOTS_TOP)
    del pr
    bw_top = _top(betweenness(d), BETWEENNESS_TOP)
    dopp = doppelt.doppelungen(wurzel, d, DOPPELT_SCHWELLE)
    eimer = int(doppelt.STATISTIK["eimer_uebersprungen"])  # direkt danach, gleicher Prozess
    return _Mathe(tot, ber, pr_top, bw_top, hs, dopp, eimer, generiert_zaehlen(d))


def _ms(seit: int) -> int:
    return (time.perf_counter_ns() - seit) // 1_000_000


def _stand(wurzel: Path) -> Path:
    pfad = stand_finden(wurzel)
    if pfad is None:
        raise NichtVermessen(f"noch nicht vermessen: {wurzel} "
                             f"(erst vermessen: python -m selbstbild vermessen {wurzel})")
    return pfad.parent


def analysieren(wurzel: Path, *, kolonie: bool = True, pruefen_reproduzierbar: bool = False) -> Analyse:
    wurzel = Path(wurzel)
    stand = _stand(wurzel)
    t = time.perf_counter_ns()
    d = laden(stand)
    stat: dict[str, int] = {"laufzeit_laden_ms": _ms(t)}
    t = time.perf_counter_ns()
    m = _mathe(wurzel, d)
    stat["laufzeit_mathe_ms"] = _ms(t)
    stat["doppelt_eimer_uebersprungen"] = m.eimer_uebersprungen
    stat["tot_generiert_uebersprungen"] = m.tot_generiert
    if pruefen_reproduzierbar:
        stat["reproduzierbar_mathe"] = int(_mathe(wurzel, d) == m)
    kol = None
    if kolonie:
        t = time.perf_counter_ns()
        kol = kolonie_laufen(d)
        stat["laufzeit_kolonie_ms"] = _ms(t)
        if pruefen_reproduzierbar:
            stat["reproduzierbar_kolonie"] = int(kolonie_laufen(d) == kol)
    luecken = tuple(sorted(d.luecken.items()))  # Ruling S3: erst nach allen Analysen
    commit = d.commit
    del d  # Graph vor dem Schreiben freigeben
    a = Analyse(commit=commit, version=ANALYSE_VERSION, parameter=parameter(kolonie),
                bereiche=m.bereiche, pagerank_top=m.pagerank_top, betweenness_top=m.betweenness_top,
                hotspots=m.hotspots, tot=m.tot, doppelungen=m.doppelungen, kolonie=kol,
                luecken=luecken, statistik=tuple(sorted(stat.items())))
    speichern(stand, a)
    return a


def _stand_commit(stand: Path) -> str | None:
    db = Path(stand) / "graph.sqlite"
    c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return _meta(c, "commit")
    except sqlite3.OperationalError:
        return None
    finally:
        c.close()


def gespeicherte_analyse(wurzel: Path) -> Analyse:
    """Der gespeicherte Lauf zum aktuellen Graph-Commit (mit Kolonie bevorzugt, sonst ohne)."""
    wurzel = Path(wurzel)
    stand = _stand(wurzel)
    commit = _stand_commit(stand)
    if commit is not None:
        for mit_kolonie in (True, False):
            a = lesen(stand, commit, ANALYSE_VERSION, digest_von(parameter(mit_kolonie)))
            if a is not None:
                return a
    raise NichtAnalysiert(f"noch nicht analysiert: {wurzel} "
                          f"(erst analysieren: python -m selbstbild analysieren {wurzel})")


def speicher_spitze_mb() -> tuple[float, float]:
    """(Spitze Working Set, Spitze privater Commit) dieses Prozesses in MB; ausserhalb von Windows
    zweimal ru_maxrss."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Zaehler(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        z = _Zaehler()
        z.cb = ctypes.sizeof(_Zaehler)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Zaehler), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(z), z.cb):
            return 0.0, 0.0
        return round(z.PeakWorkingSetSize / 2**20, 1), round(z.PeakPagefileUsage / 2**20, 1)
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mb = rss / 2**20 if sys.platform == "darwin" else rss / 1024
    return round(mb, 1), round(mb, 1)


def _dateien_top(pagerank_top: tuple[tuple[str, float], ...], n: int) -> list[str]:
    erg: list[str] = []
    for i, _ in pagerank_top:
        f = datei_von(i) if i.startswith("symbol:") else i
        if f not in erg:
            erg.append(f)
            if len(erg) == n:
                break
    return erg


def zusammenfassung(a: Analyse) -> dict[str, Any]:
    """Kurzbericht fuer die CLI: Zaehler je Fundart, 10 groesste Bereiche, 10 oberste Hotspots,
    Kolonie gegen Mathe (Ueberlappungen). Dauer und Speicher setzt die CLI dazu."""
    tot = {s: 0 for s in ("sicher_tot", "wahrscheinlich_tot", "verdaechtig")}
    for f in a.tot:
        tot[f.stufe] += 1
    dopp = {s: 0 for s in ("gleiche_datei", "gleiches_symbol", "aehnliche_datei")}
    for x in a.doppelungen:
        dopp[x.art] += 1
    gross = sorted(a.bereiche, key=lambda b: (-len(b.dateien), b.id))[:10]
    erg: dict[str, Any] = {
        "commit": a.commit,
        "tot": tot,
        "doppelungen": dopp,
        "bereiche": len(a.bereiche),
        "bereiche_top": [{"id": b.id, "name": b.name, "dateien": len(b.dateien), "kohaesion": b.kohaesion}
                         for b in gross],
        "hotspots_top": [{"datei": h.datei, "wert": h.wert} for h in a.hotspots[:10]],
        "luecken": dict(a.luecken),
        "statistik": dict(a.statistik),
        "kolonie": None,
    }
    k = a.kolonie
    if k is not None:
        top50 = set(_dateien_top(a.pagerank_top, 50))
        tot_dateien = {datei_von(f.symbol) for f in a.tot}
        sicher_dateien = {datei_von(f.symbol) for f in a.tot if f.stufe == "sicher_tot"}
        ecken = set(k.tote_ecken)
        erg["kolonie"] = {
            "zentrum": len(k.zentrum), "tote_ecken": len(ecken), "bereiche": len(k.bereiche),
            "besuchte_dateien": len(k.besuche),
            "zentrum_in_pagerank_top50_dateien": len(set(k.zentrum) & top50),
            "dateien_mit_totfund": len(tot_dateien),
            "dateien_mit_totfund_in_tote_ecken": len(tot_dateien & ecken),
            "dateien_mit_sicher_tot": len(sicher_dateien),
            "dateien_mit_sicher_tot_in_tote_ecken": len(sicher_dateien & ecken),
        }
    return erg
