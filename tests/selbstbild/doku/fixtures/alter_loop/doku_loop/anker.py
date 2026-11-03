"""Fixture: nur `Beleg` aus dem alten `scripts/doku_loop/anker.py` (byte-gleich), damit `graph.py` importierbar ist."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Beleg:
    art: str  # symbol | sql | yaml_key | datei
    file: str  # repo-relativ, POSIX
    anker: str
    digest: str
    commit: str
    zeile: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "Beleg":
        zeile = d.get("zeile")
        return cls(
            art=str(d["art"]), file=str(d["file"]), anker=str(d["anker"]),
            digest=str(d["digest"]), commit=str(d["commit"]),
            zeile=int(zeile) if isinstance(zeile, int) else None,
        )
