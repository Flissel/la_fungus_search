"""Belege-Abschnitt einer Doku-Datei in drei Formen (Stufe 4, Task 4).

1. Klammerform des Loops (KETTEN.md): Ueberschrift `## Belege` / `### Belege`, Zeilen `[B3] ziel …`, `- [B3]: ziel`
   oder Tabellenzeilen `| B3 | ziel | … |` (auch `[B3]` in der ersten Zelle).
2. Vorlage von `scripts/space_doc_check.py`: Ueberschrift `## 7. Belege`, Zeilen `- B<n>: <art> <ziel>`
   (`RECEIPT_RE`), z. B. `- B1: datei vibemind-os/config/space_agent_registry.yml:1`. `belege_abschnitt` bildet nur
   das Ziel ab; die Art (`datei`, `messung`, …) traegt `Eintrag.art`. Fehlt die Art (`- B2: pfad:4`), ist alles
   hinter dem Doppelpunkt das Ziel. Eine erklaerte `datei`-Quittung bindet die Aussage auch dann an die Datei, wenn
   die feinere Verankerung zu grob waere (`aussagen._Verankerer`).
3. Prosa-Liste der echten Space- und Modul-Doku: `## Belege` (auch mit Zusatz in Klammern, `## Belege (gelesene
   Dateien)`) mit Backtick-Pfaden ohne Nummer. Diese Pfade sind nur Stellen des Abschnitts (`belege_stellen`), keine
   `[Bn]`-Eintraege; an Aussagen gebunden wird weiter nur ueber die Backtick-Spannen im Absatz selbst.

Der Abschnitt reicht bis zur naechsten Ueberschrift gleicher oder hoeherer Ebene; Code-Bloecke zaehlen nicht. Bei
doppelter Nummer gewinnt der erste Eintrag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import ketten_seed as ks

BOM = "﻿"

_KOPF = re.compile(r"^(#{2,3})\s+(?:\d+\.\s*)?Belege(?:\s+\([^()]*\))?\s*$")
_UEBERSCHRIFT = re.compile(r"^(#{1,6})\s")
_ZAUN = re.compile(r"^(`{3,}|~{3,})")
_KLAMMER = re.compile(r"^(?:[-*+]\s+)?\[B(\d+)\]\s*:?\s*(.+)$")
_VORLAGE = re.compile(r"^[-*+]\s+B(\d+):\s+(.+?)\s*$")
_ART = re.compile(r"^[a-z_]+$")
_ZELLE = re.compile(r"^\[?B(\d+)\]?$")
_TABELLEN_TRENN = re.compile(r"^\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?$")
_SPANNE = re.compile(r"`([^`]+)`")


def ist_belege_kopf(zeile: str) -> bool:
    """Ueberschrift eines Belege-Abschnitts (Ebene 2 oder 3; `Belege`, `7. Belege`, `Belege (Zusatz)`)."""
    return bool(_KOPF.match(zeile.strip()))


def kopf_ebene(zeile: str) -> int | None:
    m = _KOPF.match(zeile.strip())
    return len(m.group(1)) if m else None


@dataclass(frozen=True)
class Eintrag:
    nr: int
    ziel: str
    art: str | None = None  # nur Vorlagenform: `datei`, `messung`, ...


def eintrag(klar: str) -> Eintrag | None:
    """Eine (gestutzte) Zeile im Belege-Abschnitt -> Eintrag; None, wenn sie keinen nummerierten Eintrag traegt
    (Prosa, Tabellenkopf oder -trennzeile)."""
    if klar.startswith("|"):
        if _TABELLEN_TRENN.match(klar) and "-" in klar:
            return None
        zellen = ks._zellen(klar)
        m = _ZELLE.match(zellen[0]) if zellen else None
        return Eintrag(int(m.group(1)), " ".join(zellen[1:])) if m else None
    m = _KLAMMER.match(klar)
    if m:
        return Eintrag(int(m.group(1)), m.group(2))
    m = _VORLAGE.match(klar)
    if m:
        rest = m.group(2)
        teile = rest.split(None, 1)
        if len(teile) == 2 and _ART.match(teile[0]):
            return Eintrag(int(m.group(1)), teile[1], teile[0])
        return Eintrag(int(m.group(1)), rest)
    return None


def _abschnitt_zeilen(text: str) -> list[str]:
    """Gestutzte Zeilen innerhalb von Belege-Abschnitten, ohne Ueberschriften, Leerzeilen und Code-Bloecke."""
    aus: list[str] = []
    ebene: int | None = None
    zaun: str | None = None
    for i, roh in enumerate(text.replace("\r\n", "\n").split("\n")):
        klar = (roh[1:] if i == 0 and roh.startswith(BOM) else roh).strip()
        if zaun is not None:
            if klar.startswith(zaun[0] * len(zaun)) and not klar.strip(zaun[0]):
                zaun = None
            continue
        m_zaun = _ZAUN.match(klar)
        if m_zaun:
            zaun = m_zaun.group(1)
            continue
        m = _UEBERSCHRIFT.match(klar)
        if m:
            tiefe = len(m.group(1))
            if ebene is not None and tiefe <= ebene:
                ebene = None
            neu = kopf_ebene(klar)
            if neu is not None and ebene is None:
                ebene = neu
            continue
        if ebene is not None and klar:
            aus.append(klar)
    return aus


def belege_abschnitt(text: str) -> dict[int, str]:
    """`B<n>` -> Zielausdruck aus allen Belege-Abschnitten (Klammer- und Vorlagenform, siehe Moduldoku)."""
    aus: dict[int, str] = {}
    for klar in _abschnitt_zeilen(text):
        e = eintrag(klar)
        if e is not None:
            aus.setdefault(e.nr, e.ziel)
    return aus


def belege_stellen(text: str) -> list[str]:
    """Backtick-Spannen der nicht nummerierten Zeilen (Prosa-Liste) in Lesereihenfolge, ohne Doppelte. Sie sind
    Stellen des Abschnitts, nie `[Bn]`-Ziele und nie Belege einer Aussage."""
    aus: dict[str, None] = {}
    for klar in _abschnitt_zeilen(text):
        if eintrag(klar) is not None or (klar.startswith("|") and _TABELLEN_TRENN.match(klar)):
            continue
        for spanne in _SPANNE.findall(klar):
            aus.setdefault(spanne.strip(), None)
    return [s for s in aus if s]
