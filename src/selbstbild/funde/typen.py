"""Fund-Modell (Stufe 5, Task 1)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

Schwere = Literal["hoch", "mittel", "niedrig"]
ARTEN: tuple[str, ...] = ("doku_widerspruch", "doppelung", "tot", "konfig")


@dataclass(frozen=True)
class Fund:
    art: str
    schwere: Schwere
    code_repo: str
    titel: str
    text: str
    belege: tuple[str, ...]
    anker: str


def fund_schluessel(f: Fund) -> str:
    """Stabiler Schluessel `<art>:<sha256(anker)[:16]>` (D3)."""
    return f"{f.art}:{hashlib.sha256(f.anker.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class FremdAusgelassen:
    """Doppelungs-Funde, die eine `fremdcode:`-Regel ausnimmt (F1). `abgleichen` schliesst ihre Issues mit eigenem Text.

    `schluessel`: `fund_schluessel` der ausgelassenen Funde (fuer Einzel-Issues, exakt).
    `paare`: ihre (art, code_repo)-Paare (fuer Sammel-Issues; deren Block nennt keine Schluessel).
    """
    schluessel: frozenset[str] = field(default_factory=frozenset)
    paare: frozenset[tuple[str, str]] = field(default_factory=frozenset)
