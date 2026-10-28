"""Dispatcher: Sprache -> Parser-Funktion. Spaetere Tasks haengen sich in REGISTER ein."""
from __future__ import annotations

from collections.abc import Callable

from functools import partial

from selbstbild.parser.daten import (
    parse_bash, parse_dotenv, parse_json, parse_powershell, parse_sql, parse_yaml,
)
from selbstbild.parser.python import parse_python
from selbstbild.parser.typescript import parse_ts
from selbstbild.typen import DateiErgebnis

ParserFunktion = Callable[[str, bytes], DateiErgebnis]

REGISTER: dict[str, ParserFunktion] = {
    "python": parse_python,
    "typescript": partial(parse_ts, sprache="typescript"),
    "tsx": partial(parse_ts, sprache="tsx"),
    "javascript": partial(parse_ts, sprache="javascript"),
    "sql": parse_sql,
    "yaml": parse_yaml,
    "json": parse_json,
    "dotenv": parse_dotenv,
    "bash": parse_bash,
    "powershell": parse_powershell,
}


def parsen(sprache: str, pfad: str, quelle: bytes) -> DateiErgebnis:
    """Parst eine Datei; unbekannte Sprache ergibt ein leeres Ergebnis."""
    funktion = REGISTER.get(sprache)
    if funktion is None:
        return DateiErgebnis()
    return funktion(pfad, quelle)
