"""Gemeinsames fuer die Doku-Loop-Tests: das echte VibeMind-Repo nur, wenn `SELBSTBILD_VIBEMIND` darauf zeigt.

Ohne die Variable laufen die Tests gegen die Vorlage `fixtures/config/doku_loop.yaml` (Kopie der Loop-Konfiguration);
Tests, die das echte Repo oder Submodul lesen, werden uebersprungen.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from selbstbild.doku import konfig as konfig_modul
from selbstbild.doku.konfig import Konfig

_ROH = os.environ.get("SELBSTBILD_VIBEMIND")
HAT_VM = bool(_ROH) and Path(_ROH).is_dir()
# ohne Variable ein Pfad, den es nicht gibt: jedes `(REPO / ...).is_dir()` ist dann falsch
REPO = Path(_ROH) if HAT_VM else Path(__file__).resolve().parent / "_kein_vibemind"
VORLAGE = Path(__file__).resolve().parent / "fixtures" / "config" / "doku_loop.yaml"
braucht_vm = pytest.mark.skipif(not HAT_VM, reason="SELBSTBILD_VIBEMIND nicht gesetzt")


def laden() -> Konfig:
    """Die echte Loop-Konfiguration des VibeMind-Repos, sonst die Vorlage."""
    return konfig_modul.laden(REPO / "config" / "doku_loop.yaml" if HAT_VM else VORLAGE)


@pytest.fixture()
def standard_konfig(monkeypatch: pytest.MonkeyPatch) -> None:
    """Der Standardpfad von `konfig.laden()` (ohne Pfad) zeigt auf die echte Loop-Konfig oder die Vorlage."""
    ziel = REPO / "config" / "doku_loop.yaml" if HAT_VM else VORLAGE
    monkeypatch.setattr(konfig_modul, "_standard_pfad", lambda: ziel)
