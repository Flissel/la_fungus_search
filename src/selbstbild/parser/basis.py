"""Dispatcher: Sprache -> Parser-Funktion. Spaetere Tasks haengen sich in REGISTER ein."""
from __future__ import annotations

from collections.abc import Callable

from functools import partial

from selbstbild.parser.python import parse_python
from selbstbild.parser.typescript import parse_ts
from selbstbild.typen import DateiErgebnis

ParserFunktion = Callable[[str, bytes], DateiErgebnis]

REGISTER: dict[str, ParserFunktion] = {
    "python": parse_python,
    "typescript": partial(parse_ts, sprache="typescript"),
    "tsx": partial(parse_ts, sprache="tsx"),
    "javascript": partial(parse_ts, sprache="javascript"),
}


def parsen(sprache: str, pfad: str, quelle: bytes) -> DateiErgebnis:
    """Parst eine Datei; unbekannte Sprache ergibt ein leeres Ergebnis."""
    funktion = REGISTER.get(sprache)
    if funktion is None:
        return DateiErgebnis()
    return funktion(pfad, quelle)
