"""Gemeinsame Hilfstypen des Doku-Loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pruefpunkt:
    """Ein Punkt der Pruefliste: etwas, das ein Mensch ansehen soll."""

    art: str
    text: str
    datei: str | None = None
    zeile: int | None = None
    aussage_id: str | None = None
