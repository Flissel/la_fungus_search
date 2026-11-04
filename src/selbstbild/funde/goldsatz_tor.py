"""Tor fuer toten Code und Uebernahme der Goldsatz-Urteile (Stufe 5, Task 1; D5)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

MIN_BEURTEILT = 30
MIN_QUOTE = 0.9
_ERLAUBT = ("ja", "nein", "offen")
_PRAEFIX = {"tot": "tot", "dop": "doppelt"}
_ITEM = re.compile(r"^- ")
_URTEIL = re.compile(r"^(\s+urteil:[ \t]*)(\S*)([ \t]*)$")


@dataclass(frozen=True)
class ToterCodeTor:
    offen: bool
    beurteilt: int
    ja: int
    grund: str


def tor_tot(goldsatz_pfad: Path) -> ToterCodeTor:
    doc = yaml.safe_load(goldsatz_pfad.read_text(encoding="utf-8")) or {}
    urteile = [str(e.get("urteil", "offen")) for e in (doc.get("tot") or [])]
    beurteilt = sum(1 for u in urteile if u in ("ja", "nein"))
    ja = sum(1 for u in urteile if u == "ja")
    if beurteilt >= MIN_BEURTEILT and ja / beurteilt >= MIN_QUOTE:
        return ToterCodeTor(True, beurteilt, ja, f"offen: {ja} von {beurteilt} bestätigt")
    if beurteilt < MIN_BEURTEILT:
        grund = f"gesperrt: {ja}/{beurteilt} beurteilt, mindestens {MIN_BEURTEILT} nötig"
    else:
        grund = f"gesperrt: {ja}/{beurteilt} beurteilt, mindestens 90 % nötig"
    return ToterCodeTor(False, beurteilt, ja, grund)


def _urteilszeilen(zeilen: list[str], block: str) -> list[int]:
    """Zeilenindex der `urteil:`-Zeile je Listeneintrag im Top-Level-Block `block:`."""
    start = next((i for i, z in enumerate(zeilen) if z.rstrip("\r\n") == f"{block}:"), None)
    if start is None:
        return []
    treffer: list[int] = []
    aktuell: int | None = None
    for i in range(start + 1, len(zeilen)):
        z = zeilen[i].rstrip("\r\n")
        if z and not z[0].isspace() and not _ITEM.match(z) and z != "-":
            break  # naechster Top-Level-Schluessel
        if _ITEM.match(z) or z == "-":
            treffer.append(-1)
            aktuell = len(treffer) - 1
        if aktuell is not None and treffer[aktuell] < 0 and _URTEIL.match(z):
            treffer[aktuell] = i
    return treffer


def urteile_uebernehmen(goldsatz_pfad: Path, urteile: dict[str, str]) -> str:
    """Tragt `tot-NN` / `dop-NN` -> ja/nein/offen in die `urteil:`-Zeilen ein, sonst Byte-gleich.

    Gibt einen Unified-Diff zurueck ("" und keine Schreibung, wenn nichts ausser Kraft aendert)."""
    roh = goldsatz_pfad.read_bytes().decode("utf-8")
    zeilen = roh.splitlines(keepends=True)
    index = {b: _urteilszeilen(zeilen, b) for b in ("tot", "doppelt")}
    neu = list(zeilen)
    for schluessel, wert in urteile.items():
        m = re.fullmatch(r"(tot|dop)-(\d+)", schluessel)
        if not m:
            raise ValueError(f"unbekannte Urteils-Id: {schluessel!r}")
        if wert not in _ERLAUBT:
            raise ValueError(f"ungültiges Urteil für {schluessel}: {wert!r}")
        liste = index[_PRAEFIX[m.group(1)]]
        n = int(m.group(2))
        if n >= len(liste) or liste[n] < 0:
            raise ValueError(f"Index außerhalb der Liste: {schluessel}")
        zeile = zeilen[liste[n]]
        ende = zeile[len(zeile.rstrip("\r\n")):]
        t = _URTEIL.match(zeile.rstrip("\r\n"))
        assert t is not None
        neu[liste[n]] = f"{t.group(1)}{wert}{t.group(3)}{ende}"
    if neu == zeilen:
        return ""
    diff = "".join(difflib.unified_diff(
        [z.rstrip("\r\n") + "\n" for z in zeilen], [z.rstrip("\r\n") + "\n" for z in neu],
        goldsatz_pfad.name, goldsatz_pfad.name, n=0))
    goldsatz_pfad.write_bytes("".join(neu).encode("utf-8"))
    return diff
