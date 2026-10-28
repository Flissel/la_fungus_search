"""Kolonie: Parameter- und Ergebnistypen (reine Dataclasses; `kolonie_laufen` kommt in Task 9)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KolonieParameter:
    agenten: int = 192
    schritte: int = 400
    verfall: float = 0.1
    ablage: float = 1.0
    erkundung: float = 0.1


@dataclass(frozen=True)
class KolonieErgebnis:
    besuche: tuple[tuple[str, int], ...]     # (Datei, Besuche), sortiert nach Id
    spuren: tuple[tuple[str, float], ...]    # (Datei, Pheromon), sortiert nach Id
    bereiche: tuple[tuple[str, float], ...]  # (Bereich-Id, Besuchsanteil), sortiert nach Id
    zentrum: tuple[tuple[str, float], ...]   # (Datei, Wert), absteigend, dann Id
    tote_ecken: tuple[str, ...]              # nie besuchte Dateien, sortiert
    parameter: KolonieParameter = KolonieParameter()
