"""Tor fuer toten Code und Uebernahme der Goldsatz-Urteile (Stufe 5, Task 1; D5)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MIN_BEURTEILT = 30
MIN_QUOTE = 0.9
_ERLAUBT = ("ja", "nein", "offen", "unklar")
_PRAEFIX = {"tot": "tot", "dop": "doppelt"}
_ITEM = re.compile(r"^- ")
_URTEIL = re.compile(r"^(\s+urteil:[ \t]*)(\S*)([ \t]*)$")
_BLEIBT = re.compile(r"^\s+bleibt:")
_BLEIBT_TRUE = re.compile(r"^\s+bleibt:[ \t]*true[ \t]*$")


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
        neu[liste[n]] = _urteil_setzen(zeilen[liste[n]], wert)
    return _schreiben(goldsatz_pfad, zeilen, neu)


def _ende(zeile: str) -> str:
    return zeile[len(zeile.rstrip("\r\n")):]


def _urteil_setzen(zeile: str, wert: str) -> str:
    t = _URTEIL.match(zeile.rstrip("\r\n"))
    assert t is not None
    return f"{t.group(1)}{wert}{t.group(3)}{_ende(zeile)}"


def _schreiben(goldsatz_pfad: Path, zeilen: list[str], neu: list[str]) -> str:
    """Unified-Diff alt/neu; schreibt nur bei Aenderung ("" sonst, die mtime bleibt)."""
    if neu == zeilen:
        return ""
    diff = "".join(difflib.unified_diff(
        [z.rstrip("\r\n") + "\n" for z in zeilen], [z.rstrip("\r\n") + "\n" for z in neu],
        goldsatz_pfad.name, goldsatz_pfad.name, n=0))
    goldsatz_pfad.write_bytes("".join(neu).encode("utf-8"))
    return diff


def _tot_eintraege(goldsatz_pfad: Path) -> list[dict[str, Any]]:
    doc = yaml.safe_load(goldsatz_pfad.read_text(encoding="utf-8")) or {}
    return [e if isinstance(e, dict) else {} for e in (doc.get("tot") or [])]


def goldsatz_bleibt(goldsatz_pfad: Path) -> frozenset[str]:
    """Symbole der `tot`-Eintraege mit `bleibt: true`."""
    return frozenset(str(e["symbol"]) for e in _tot_eintraege(goldsatz_pfad)
                     if e.get("bleibt") is True and e.get("symbol"))


def unklare_eintraege(goldsatz_pfad: Path) -> list[tuple[str, dict[str, Any]]]:
    """`(tot-NN, eintrag)` fuer jeden `tot`-Eintrag mit Urteil `unklar` (NN 0-basiert wie die Seite)."""
    return [(f"tot-{n:02d}", e) for n, e in enumerate(_tot_eintraege(goldsatz_pfad)) if e.get("urteil") == "unklar"]


def _bleibt_zeile(zeilen: list[str], urteil_i: int) -> int | None:
    """Index einer eingerueckten `bleibt:`-Zeile im selben Eintrag wie die urteil-Zeile, sonst None."""
    def eintrag_grenze(z: str) -> bool:
        return not z[:1].isspace() or _ITEM.match(z) is not None or z == "-"

    for bereich in (range(urteil_i + 1, len(zeilen)), range(urteil_i - 1, -1, -1)):
        for j in bereich:
            z = zeilen[j].rstrip("\r\n")
            if not z.strip():
                continue
            if _BLEIBT.match(z) and not _ITEM.match(z):
                return j
            if eintrag_grenze(z):
                break
    return None


def antworten_uebernehmen(goldsatz_pfad: Path, antworten: dict[str, tuple[str, bool]]) -> str:
    """Traegt je Symbol `(urteil, bleibt)` ein: `urteil:`-Zeile setzen, `bleibt: true` direkt dahinter einfuegen bzw.
    beibehalten, bei `bleibt=False` eine vorhandene `bleibt:`-Zeile entfernen. Sonst Byte-gleich.

    Unified-Diff als Rueckgabe ("" und keine Schreibung ohne Aenderung); unbekanntes Symbol -> ValueError.
    Nicht unterstuetzt: ein `bleibt:` in der `- `-Eintragszeile selbst (nur eingerueckte Zeilen)."""
    zeilen = goldsatz_pfad.read_bytes().decode("utf-8").splitlines(keepends=True)
    urteilszeilen = _urteilszeilen(zeilen, "tot")
    nr_je_symbol: dict[str, list[int]] = {}
    for n, e in enumerate(_tot_eintraege(goldsatz_pfad)):
        if e.get("symbol"):
            nr_je_symbol.setdefault(str(e["symbol"]), []).append(n)
    arbeit: dict[int, tuple[str, bool]] = {}  # Zeilenindex der urteil-Zeile -> (urteil, bleibt)
    for symbol, (urteil, bleibt) in antworten.items():
        if urteil not in _ERLAUBT:
            raise ValueError(f"ungültiges Urteil für {symbol}: {urteil!r}")
        nummern = nr_je_symbol.get(symbol)
        if not nummern:
            raise ValueError(f"unbekanntes Symbol: {symbol!r}")
        for n in nummern:
            if n >= len(urteilszeilen) or urteilszeilen[n] < 0:
                raise ValueError(f"Eintrag ohne urteil-Zeile: {symbol!r}")
            arbeit[urteilszeilen[n]] = (urteil, bleibt)
    ersatz: dict[int, list[str]] = {}  # Zeilenindex -> Ersatzzeilen (leer = Zeile entfernen)
    for i, (urteil, bleibt) in arbeit.items():
        ersatz[i] = [_urteil_setzen(zeilen[i], urteil)]
        einzug = zeilen[i][: len(zeilen[i]) - len(zeilen[i].lstrip())]
        j = _bleibt_zeile(zeilen, i)
        if bleibt and j is None:
            ersatz[i].append(f"{einzug}bleibt: true{_ende(zeilen[i])}")
        elif bleibt and j is not None and not _BLEIBT_TRUE.match(zeilen[j].rstrip("\r\n")):
            ersatz[j] = [f"{einzug}bleibt: true{_ende(zeilen[j])}"]
        elif not bleibt and j is not None:
            ersatz[j] = []
    neu = [z for k, alt in enumerate(zeilen) for z in ersatz.get(k, [alt])]
    return _schreiben(goldsatz_pfad, zeilen, neu)
