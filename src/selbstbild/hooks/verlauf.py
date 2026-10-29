"""Gespraechsverlauf fuer den Einspeisungs-Hook (Stufe 3, T12). NUR Standardbibliothek.

Kurze Fortsetzungs-Prompts ("ja", "weiter") nennen ihre Zieldatei nicht; sie ergibt sich aus dem Verlauf. Dieses
Modul liest dafuer aus der Claude-Code-Transkript-JSONL nur die *Pfade* der zuletzt benutzten Dateien
(`tool_use`-Eingaben von Assistant-Nachrichten), nie Prompt- oder Nachrichtentexte. Gelesen wird nur ein
begrenztes Ende der Datei (Transkripte werden mehrere hundert MB gross), Fehler fuehren zu einer leeren Liste.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

__all__ = ["letzte_dateien", "hauptpfad"]

_FELDER = ("file_path", "notebook_path", "path")
_MUSTER = re.compile(r"[*?\[\]{}]")  # Glob-Muster sind kein Ort
_WORKTREE = re.compile(r"^(.*?)/(?:\.worktrees|\.claude/worktrees)/([^/]+)(?:/(.*))?$", re.IGNORECASE)


def _ort(feld: str, wert: object) -> str | None:
    if not isinstance(wert, str) or not wert or len(wert) > 4096 or "\n" in wert:
        return None
    if not os.path.isabs(wert):
        return None
    if feld == "path":  # Grep/Glob: nur ein konkreter Ordner oder eine Datei, kein Muster und keine Laufwerkswurzel
        if _MUSTER.search(wert) or Path(wert).parent == Path(wert):
            return None
    return wert


def letzte_dateien(transkript: Path, *, ende: int | None = None, max_bytes: int = 262_144,
                   max_dateien: int = 30) -> list[str]:
    """Absolute Dateipfade aus `tool_use`-Eingaben, neueste zuerst, dedupliziert, hoechstens `max_dateien`.

    Gelesen werden hoechstens `max_bytes` vor dem Byte-Offset `ende` (Standard: Dateiende); eine am Anfang
    angeschnittene Zeile wird verworfen. Felder: `file_path`, `notebook_path`, `path` (nur ohne Glob-Muster).
    Subagent-Zeilen (`isSidechain`) zaehlen nicht. Kaputte Zeilen werden uebersprungen; jede Ausnahme -> []."""
    try:
        with open(transkript, "rb") as f:
            groesse = os.fstat(f.fileno()).st_size
            bis = groesse if ende is None else min(int(ende), groesse)
            if bis <= 0 or max_bytes <= 0:
                return []
            von = max(0, bis - int(max_bytes))
            ganz = von == 0
            if not ganz:  # das Byte davor zeigt, ob die erste Zeile vollstaendig ist
                f.seek(von - 1)
                ganz = f.read(1) == b"\n"
            else:
                f.seek(0)
            roh = f.read(bis - von)
        zeilen = roh.split(b"\n")
        if not ganz:
            zeilen = zeilen[1:]
        aus: list[str] = []
        gesehen: set[str] = set()
        for zeile in reversed(zeilen):
            if b'"tool_use"' not in zeile:
                continue  # schneller Vorfilter: nur Zeilen mit Werkzeugaufrufen werden dekodiert
            try:
                eintrag = json.loads(zeile)
            except ValueError:
                continue
            if not isinstance(eintrag, dict) or eintrag.get("type") != "assistant" or eintrag.get("isSidechain") is True:
                continue
            nachricht = eintrag.get("message")
            inhalt = nachricht.get("content") if isinstance(nachricht, dict) else None
            if not isinstance(inhalt, list):
                continue
            for x in reversed(inhalt):
                if not (isinstance(x, dict) and x.get("type") == "tool_use" and isinstance(x.get("input"), dict)):
                    continue
                for feld in _FELDER:
                    pfad = _ort(feld, x["input"].get(feld))
                    if pfad is None:
                        continue
                    schluessel = os.path.normcase(os.path.normpath(pfad))
                    if schluessel in gesehen:
                        continue
                    gesehen.add(schluessel)
                    aus.append(pfad)
                    if len(aus) >= max_dateien:
                        return aus
        return aus
    except Exception:  # noqa: BLE001 - der Verlauf ist eine Zugabe, nie ein Fehlergrund
        return []


def hauptpfad(pfad: str) -> tuple[str, str | None]:
    """`…/.worktrees/<name>/rest` und `…/.claude/worktrees/<name>/rest` -> (`…/rest`, name); sonst (pfad, None).

    Ausgabe immer mit `/` als Trenner. Der Index kennt nur den committeten Hauptbaum, darum zaehlt fuer ihn der
    Pfad im Hauptbaum; der Worktree-Name dient der Warnung "zweiter Checkout"."""
    norm = pfad.replace("\\", "/")
    m = _WORKTREE.match(norm)
    if m is None:
        return norm, None
    basis, name, rest = m.group(1), m.group(2), m.group(3)
    return (f"{basis}/{rest}" if rest else basis), name
