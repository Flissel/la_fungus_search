"""Fund-Modell (Stufe 5, Task 1)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
