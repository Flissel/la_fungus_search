"""Ketten-Waechter-Zustand -> Stufe `gemessen`.

Datei-Form (logs/ketten-waechter-zustand.json): {"stand": {pruefung: bool}, "zuletzt_gemeldet": iso}.
Je Kette: eine zugeordnete Pruefung False -> `bricht`; sonst alle True -> `laeuft`; Kette ohne
(vorhandene) Pruefung -> `keine_probe`. `zeit` = Aenderungszeit der Datei (ISO, UTC).
Fehlende/unlesbare Datei: jede Kette `keine_probe`, zeit None, keine Ausnahme.
Final-Fix I8: mit `max_alter_min` gilt eine Datei, die laenger nicht geschrieben wurde (toter Waechter), wie eine
fehlende -- sonst behielte jede Kette den letzten Wert des toten Waechters.

Ruling 55 (Task 17 Fix-Runde 1):
- Die Zuordnung darf statt einer ganzen Kette (`K11`) auch einen einzelnen Hop nennen (`K11/7`). Ein Hop-Schluessel
  traegt zusaetzlich die Pruefungen seiner Kette (falls es welche gibt); `bauen` nimmt den Hop-Schluessel vor dem
  Ketten-Schluessel. Hops einer Kette, die nur Hop-Zuordnungen hat, haben keine Messung.
- `hysterese`: ein geaenderter `gemessen.wert` eines Hops gilt erst, wenn er in zwei aufeinanderfolgenden messenden
  Laeufen gleich gelesen wurde; die erste Lesung bleibt lokal im Stand (`gemessen_ausstehend`). Ein Hop ohne bisherige
  Messung (committeter Wert ohne `zeit`) wird nicht verzoegert.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .graph import Graph


@dataclass(frozen=True)
class Messung:
    wert: str  # laeuft | bricht | keine_probe
    zeit: str | None


def _stand(zustandsdatei: Path) -> tuple[dict[str, bool], str] | None:
    try:
        datei = Path(zustandsdatei)
        roh = json.loads(datei.read_text(encoding="utf-8"))
        zeit = datetime.fromtimestamp(datei.stat().st_mtime, tz=timezone.utc).isoformat()
    except (OSError, ValueError):
        return None
    stand = roh.get("stand") if isinstance(roh, dict) else None
    if not isinstance(stand, dict):
        return None
    return {str(k): v for k, v in stand.items() if isinstance(v, bool)}, zeit


def veraltet_seit(zustandsdatei: Path, max_alter_min: float | None, *, jetzt_ts: float | None = None) -> str | None:
    """Aenderungszeit (ISO, UTC) der Zustandsdatei, wenn sie aelter als `max_alter_min` ist; sonst (oder ohne Datei,
    ohne Grenze) None."""
    if max_alter_min is None:
        return None
    try:
        mtime = Path(zustandsdatei).stat().st_mtime
    except OSError:
        return None
    jetzt_ts = time.time() if jetzt_ts is None else jetzt_ts
    if jetzt_ts - mtime <= max_alter_min * 60:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def gemessen_laden(
    zustandsdatei: Path,
    zuordnung: Mapping[str, Sequence[str]],
    *,
    alle_ketten: Iterable[str] = (),
    max_alter_min: float | None = None,
) -> tuple[dict[str, Messung], list[str]]:
    """(Messung je Kette, Pruefungsnamen der Datei ohne Zuordnung). Aelter als `max_alter_min` -> wie fehlend."""
    ketten: dict[str, list[str]] = {}
    for pruefung, ids in zuordnung.items():
        for kette in ids:
            ketten.setdefault(kette, []).append(pruefung)
    for schl in [k for k in ketten if "/" in k]:  # Hop-Schluessel: Pruefungen der ganzen Kette gelten mit
        ketten[schl].extend(p for p in ketten.get(schl.split("/", 1)[0], []) if p not in ketten[schl])
    geladen = None if veraltet_seit(zustandsdatei, max_alter_min) is not None else _stand(zustandsdatei)
    if geladen is None:
        ids = set(ketten) | set(alle_ketten)
        return {k: Messung("keine_probe", None) for k in sorted(ids)}, []
    stand, zeit = geladen
    ergebnis: dict[str, Messung] = {}
    for kette, pruefungen in ketten.items():
        werte = [stand[p] for p in pruefungen if p in stand]
        if any(w is False for w in werte):
            ergebnis[kette] = Messung("bricht", zeit)
        elif werte:
            ergebnis[kette] = Messung("laeuft", zeit)
        else:
            ergebnis[kette] = Messung("keine_probe", zeit)
    for kette in alle_ketten:
        ergebnis.setdefault(kette, Messung("keine_probe", zeit))
    unbekannt = sorted(n for n in stand if n not in zuordnung)
    return dict(sorted(ergebnis.items())), unbekannt


def _gemessen(hop_attr: Mapping[str, object]) -> dict[str, object] | None:
    z = hop_attr.get("zustand")
    g = z.get("gemessen") if isinstance(z, dict) else None
    return g if isinstance(g, dict) else None


def hysterese(basis: Graph, neu: Graph, ausstehend: Mapping[str, str]) -> dict[str, str]:
    """Ruling 55a: in `neu` (gerade gebaut) jede Aenderung von `gemessen.wert` gegenueber `basis` (committet)
    zuruecknehmen, solange sie nicht schon im letzten messenden Lauf gleich gelesen wurde (`ausstehend`: Hop-ID ->
    Wert). Veraendert `neu`; Rueckgabe die neuen ausstehenden Lesungen."""
    offen: dict[str, str] = {}
    for hid, hop in neu.kanten.items():
        alt_hop = basis.kanten.get(hid)
        if hop.art != "hop" or alt_hop is None:
            continue
        g_neu, g_alt = _gemessen(hop.attribute), _gemessen(alt_hop.attribute)
        if g_neu is None or g_alt is None:
            continue
        wert = str(g_neu.get("wert"))
        if not g_alt.get("zeit"):  # Fix-Runde 2: nie gemessen (Wert aus dem Dokument) -> die erste Messung gilt sofort
            continue
        if wert == str(g_alt.get("wert")) or ausstehend.get(hid) == wert:
            continue
        offen[hid] = wert
        zustand = hop.attribute["zustand"]
        if isinstance(zustand, dict):
            zustand["gemessen"] = dict(g_alt)
    return offen
