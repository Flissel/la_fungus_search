"""Dauerhafter Stand des Debugger-Claw (JSON-Datei, atomar geschrieben)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

MAX_VERARBEITET = 5000


@dataclass
class FundStand:
    """Bearbeitungsstand eines Fundes (Issue); neue Felder haben Standardwerte, alte Stände laden weiter."""

    repo: str
    branch: str
    sitzung: str | None = None
    pr: int | None = None
    zustand: str = "offen"  # "offen" | "pr" | "erledigt"
    versuche: dict[str, int] = field(default_factory=dict)
    runde: int = 1  # R37: zählt neue Anläufe nach `erledigt`; fehlt in alten Ständen


@dataclass
class DebuggerStand:
    """`start` rückt nie weiter, `seit` ist die Lesemarke; Zeiten sind ISO-8601-Texte (UTC)."""

    start: str | None = None
    seit: str | None = None
    verarbeitet: list[int] = field(default_factory=list)
    funde: dict[str, FundStand] = field(default_factory=dict)
    gemeldet: list[str] = field(default_factory=list)

    def verarbeitet_hinzu(self, ids: Iterable[int]) -> None:
        """Hängt Kommentar-Ids hinten an und behält nur die letzten `MAX_VERARBEITET`."""
        self.verarbeitet.extend(ids)
        del self.verarbeitet[:-MAX_VERARBEITET]


def stand_laden(pfad: Path) -> DebuggerStand:
    """Fehlende, unlesbare oder kaputte Datei ergibt einen leeren Stand mit Eintrag in `gemeldet`.

    `stand_fehlt`: keine Datei. `stand_unlesbar`: Lesefehler (OSError), der Inhalt ist evtl. heil.
    `stand_kaputt`: Inhalt nicht deutbar. Die Datei bleibt in jedem Fall.
    """
    try:
        text = pfad.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DebuggerStand(gemeldet=["stand_fehlt"])
    except UnicodeDecodeError:
        return DebuggerStand(gemeldet=["stand_kaputt"])
    except OSError:
        return DebuggerStand(gemeldet=["stand_unlesbar"])
    try:
        roh = json.loads(text)
        funde = {str(k): FundStand(**v) for k, v in roh.get("funde", {}).items()}
        start, seit = roh.get("start"), roh.get("seit")
        verarbeitet = [int(i) for i in roh.get("verarbeitet", [])][-MAX_VERARBEITET:]
        gemeldet = [str(g) for g in roh.get("gemeldet", [])]
        if not all(x is None or isinstance(x, str) for x in (start, seit)):
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        return DebuggerStand(gemeldet=["stand_kaputt"])
    return DebuggerStand(start=start, seit=seit, verarbeitet=verarbeitet, funde=funde, gemeldet=gemeldet)


def stand_speichern(pfad: Path, stand: DebuggerStand) -> None:
    """Schreibt atomar (temporäre Datei, dann replace); kappt `verarbeitet` auf die letzten 5000."""
    daten = asdict(stand)
    daten["verarbeitet"] = daten["verarbeitet"][-MAX_VERARBEITET:]
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps(daten, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, pfad)
