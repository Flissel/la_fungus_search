"""Goldsatz: Kandidaten zum Bestaetigen durch den User und Vergleich Mathe gegen Kolonie.

`kandidaten` erzeugt deterministisches YAML (Urteil je Eintrag `offen | ja | nein`). `vergleichen`
liest das ausgefuellte YAML und misst die Mathe-Analyse und die Kolonie dagegen:

- tot: Praezision = ja / (ja + nein) ueber die beurteilten Eintraege. Die Kolonie "sagt tot", wenn die
  Datei des Symbols in `kolonie.tote_ecken` steht; ihre Praezision zaehlt nur beurteilte Eintraege,
  die sie so markiert (n_kolonie).
- doppelt: nur Mathe (die Kolonie kennt keine Doppelungen); Praezision wie bei tot.
- bereiche: Mittel ueber Eintraege mit nicht leerem `erwartet` der Jaccard-Werte zwischen der erwarteten
  Dateimenge (Dateien aus den Bereichen der Analyse, deren Pfad mit einem Praefix beginnt) und dem
  jeweils besten Bereich (Mathe bzw. Kolonie).
- zentrum: nur wenn `zentral` Eintraege hat: Anteil der markierten Dateien, die in den Top-50 stehen
  (Mathe: `pagerank_top`, auf Dateien abgebildet; Kolonie: `kolonie.zentrum`).
- Laufzeit: nur wenn `statistik` die Schluessel `laufzeit_mathe_ms` / `laufzeit_kolonie_ms` traegt,
  Reproduzierbarkeit (1/0) nur mit `reproduzierbar_mathe` / `reproduzierbar_kolonie`; sonst None.
Metriken ohne Daten sind None (keine Division durch null).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import yaml

from .ergebnis import Analyse
from .lader import SEED, datei_von

KOPF = """\
# Goldsatz-Kandidaten (Selbstbild). So fuellst du ihn aus:
#  - tot / doppelt: setze je Eintrag `urteil` auf ja (stimmt: wirklich tot bzw. wirklich doppelt),
#    nein (stimmt nicht) oder lass `offen` (zaehlt nicht).
#  - bereiche: trage unter `erwartet` Pfad-Praefixe ein, die deiner Meinung nach zu diesem
#    Bereich gehoeren (z. B. `src/selbstbild/`).
#  - zentral: optional, Datei-Ids (datei:<repo>:<pfad>), die du fuer zentral haeltst.
"""


def kandidaten(a: Analyse, n_tot: int = 40, n_doppelt: int = 15) -> str:
    tot = sorted((t for t in a.tot if t.stufe == "sicher_tot"), key=lambda t: t.symbol)
    if len(tot) > n_tot:
        tot = sorted(random.Random(SEED).sample(tot, n_tot), key=lambda t: t.symbol)
    dopp = sorted(a.doppelungen, key=lambda d: (-d.aehnlichkeit, d.art, d.mitglieder))[:n_doppelt]
    gross = sorted((b for b in a.bereiche if b.name != "(isoliert)"),
                   key=lambda b: (-len(b.dateien), b.id))[:3]
    doc: dict[str, Any] = {
        "tot": [{"symbol": t.symbol, "gruende": list(t.gruende), "urteil": "offen"} for t in tot],
        "doppelt": [{"art": d.art, "mitglieder": list(d.mitglieder),
                     "aehnlichkeit": f"{d.aehnlichkeit:.4f}", "urteil": "offen"} for d in dopp],
        "bereiche": [{"id": b.id, "name": b.name, "dateien": len(b.dateien), "erwartet": []}
                     for b in sorted(gross, key=lambda b: b.id)],
        "zentral": [],
    }
    return KOPF + yaml.safe_dump(doc, sort_keys=True, allow_unicode=True)


@dataclass(frozen=True)
class Frage:
    mathe: float | None
    kolonie: float | None
    n_mathe: int = 0       # Anzahl Urteile bzw. Eintraege, auf denen der Mathe-Wert beruht
    n_kolonie: int = 0


@dataclass(frozen=True)
class Vergleich:
    tot: Frage
    doppelt: Frage
    bereiche: Frage
    zentrum: Frage
    laufzeit_mathe_ms: int | None = None
    laufzeit_kolonie_ms: int | None = None
    reproduzierbar_mathe: bool | None = None
    reproduzierbar_kolonie: bool | None = None


def _praezision(urteile: list[str]) -> tuple[float | None, int]:
    ja, nein = urteile.count("ja"), urteile.count("nein")
    return (ja / (ja + nein) if ja + nein else None), ja + nein


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _passt(datei_id: str, praefixe: list[str]) -> bool:
    pfad = datei_id.removeprefix("datei:")
    ohne_repo = pfad.partition(":")[2] or pfad
    return any(c.startswith(p) for p in praefixe for c in (datei_id, pfad, ohne_repo))


def _stat(a: Analyse, schluessel: str) -> int | None:
    return next((w for k, w in a.statistik if k == schluessel), None)


def vergleichen(gold_yaml: str, mathe: Analyse) -> Vergleich:
    gold = yaml.safe_load(gold_yaml) or {}
    kol = mathe.kolonie
    tote = set(kol.tote_ecken) if kol else set()

    tot = [e for e in gold.get("tot") or [] if e.get("urteil") in ("ja", "nein")]
    p_m, n_m = _praezision([e["urteil"] for e in tot])
    gemeldet = [e["urteil"] for e in tot if datei_von(e["symbol"]) in tote]
    p_k, n_k = _praezision(gemeldet) if kol else (None, 0)

    p_d, n_d = _praezision([e.get("urteil", "offen") for e in gold.get("doppelt") or []])

    bekannt = sorted({d for b in mathe.bereiche for d in b.dateien})
    b_m: list[float] = []
    b_k: list[float] = []
    for e in gold.get("bereiche") or []:
        praefixe = [str(p) for p in e.get("erwartet") or []]
        erwartet = {d for d in bekannt if _passt(d, praefixe)} if praefixe else set()
        if not erwartet:
            continue
        b_m.append(max((_jaccard(erwartet, set(b.dateien)) for b in mathe.bereiche), default=0.0))
        if kol:
            b_k.append(max((_jaccard(erwartet, set(b)) for b in kol.bereiche), default=0.0))

    zentral = {str(z) for z in gold.get("zentral") or []}
    z_m = z_k = None
    if zentral:
        top = {datei_von(s) if s.startswith("symbol:") else s for s, _ in mathe.pagerank_top[:50]}
        z_m = len(zentral & top) / len(zentral)
        if kol:
            z_k = len(zentral & set(kol.zentrum[:50])) / len(zentral)

    def mittel(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    rm, rk = _stat(mathe, "reproduzierbar_mathe"), _stat(mathe, "reproduzierbar_kolonie")
    return Vergleich(
        tot=Frage(p_m, p_k, n_m, n_k), doppelt=Frage(p_d, None, n_d, 0),
        bereiche=Frage(mittel(b_m), mittel(b_k), len(b_m), len(b_k)),
        zentrum=Frage(z_m, z_k, len(zentral) if zentral else 0, len(zentral) if z_k is not None else 0),
        laufzeit_mathe_ms=_stat(mathe, "laufzeit_mathe_ms"),
        laufzeit_kolonie_ms=_stat(mathe, "laufzeit_kolonie_ms"),
        reproduzierbar_mathe=None if rm is None else bool(rm),
        reproduzierbar_kolonie=None if rk is None else bool(rk))
