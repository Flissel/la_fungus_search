"""Final-Fix für wachter.py: Aufräumen vor dem Nachlauf (R44) und weitere Härtungen."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from test_wachter import Welt, welt  # noqa: F401  (Fixture `welt`)

import selbstbild.debugger.wachter as wachter
from selbstbild.debugger.wachter import durchgang


def test_claw_aufraeumen_vor_dem_nachlauf(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    ablauf: list[str] = []

    def nachlauf(*args: Any) -> list[str]:
        ablauf.append("nachlauf")
        return []

    monkeypatch.setattr(wachter, "nachlauf", nachlauf)
    welt.kommentar(1)
    w = replace(welt.w, claw_aufraeumen=lambda: ablauf.append("aufraeumen"))
    assert durchgang(welt.tmp, welt.klon, welt.ordner, w, konfig=welt.konfig) == 0
    assert len(welt.helfer.aufrufe) == 1
    assert ablauf == ["aufraeumen", "aufraeumen", "nachlauf"]  # nach dem Helfer (Runde) und vor dem Nachlauf
