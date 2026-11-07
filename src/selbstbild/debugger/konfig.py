"""Konfiguration des Debugger-Claw: Abschnitt `debugger:` in `selbstbild.yaml`."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGELDATEI = "selbstbild.yaml"


@dataclass(frozen=True)
class DebuggerKonfig:
    """Feste Einstellungen des Wächters. `repos` bildet Repo -> Pfad im Klon ab."""

    issue_repo: str
    repos: dict[str, str] = field(default_factory=dict)
    ausloeser: str = "Flissel"
    helfer_konto: str = "Vibemind-LAB"
    basis: str = "master"
    intervall_s: int = 90
    frist_s: int = 1200
    min_frei_gb: float = 2.0
    tagesgrenze: int = 20


def konfig_laden(wurzel: Path) -> DebuggerKonfig:
    """Liest `wurzel/selbstbild.yaml`, Abschnitt `debugger:`. Fehler ergeben `ValueError`."""
    try:
        daten: Any = yaml.safe_load((wurzel / REGELDATEI).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"selbstbild.yaml nicht lesbar ({type(e).__name__})") from None
    abschnitt = daten.get("debugger") if isinstance(daten, dict) else None
    if not isinstance(abschnitt, dict):
        raise ValueError("Abschnitt `debugger:` fehlt in selbstbild.yaml")
    erlaubt = {f.name for f in dataclasses.fields(DebuggerKonfig)}
    unbekannt = sorted(str(k) for k in abschnitt if k not in erlaubt)
    if unbekannt:
        raise ValueError(f"unbekannte Schlüssel in `debugger:`: {', '.join(unbekannt)}")
    if not abschnitt.get("issue_repo"):
        raise ValueError("`debugger.issue_repo` fehlt")
    repos = abschnitt.get("repos")
    if not isinstance(repos, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in repos.items()):
        raise ValueError("`debugger.repos` muss eine Zuordnung Repo -> Pfad sein")
    if abschnitt["issue_repo"] not in repos:
        raise ValueError("`debugger.issue_repo` steht nicht in `debugger.repos`")
    try:
        return DebuggerKonfig(**abschnitt)
    except TypeError as e:
        raise ValueError(f"`debugger:` ungültig ({type(e).__name__})") from None
