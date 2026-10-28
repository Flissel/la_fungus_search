"""Gemeinsame Typen und Id-Schema des Selbstbilds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KNOTENARTEN = (
    "repo", "datei", "symbol", "route", "tabelle",
    "sql_funktion", "mcp_werkzeug", "env", "dienst", "doku_seite",
)
KANTENARTEN = (
    "enthaelt", "importiert", "ruft_auf", "definiert_route", "liest_schreibt",
    "liest_env", "registriert", "verbindet", "nutzt_dienst", "geaendert_durch",
)


@dataclass(frozen=True)
class Beleg:
    repo: str
    pfad: str
    zeile: int
    anker: str
    digest: str  # sha256 des belegten Ausschnitts


@dataclass(frozen=True)
class Knoten:
    id: str
    art: str
    name: str
    beleg: Beleg | None
    attr: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Kante:
    art: str
    von: str
    nach: str
    sicher: bool
    beleg: Beleg | None


def id_repo(repo: str) -> str:
    return f"repo:{repo}"


def id_datei(repo: str, pfad: str) -> str:
    return f"datei:{repo}:{pfad}"


def id_symbol(repo: str, pfad: str, qualname: str) -> str:
    return f"symbol:{repo}:{pfad}#{qualname}"


def id_route(repo: str, methode: str, pfad: str) -> str:
    return f"route:{repo}:{methode.upper()} {pfad}"


def _sql_name(name: str) -> str:
    n = name.strip().replace('"', "").lower()
    return n if "." in n else f"public.{n}"


def id_tabelle(name: str) -> str:
    return f"tabelle:{_sql_name(name)}"


def id_sql_funktion(name: str) -> str:
    return f"sql_funktion:{_sql_name(name)}"


def id_mcp(repo: str, name: str) -> str:
    return f"mcp_werkzeug:{repo}:{name}"


def id_env(name: str) -> str:
    return f"env:{name}"


def id_dienst(name: str) -> str:
    return f"dienst:{name}"


@dataclass(frozen=True)
class Symbol:
    qualname: str
    art: Literal["funktion", "klasse", "methode"]
    zeile_von: int
    zeile_bis: int
    digest: str


@dataclass(frozen=True)
class Import:
    modul: str
    namen: tuple[str, ...]
    zeile: int
    relativ: bool


@dataclass(frozen=True)
class Aufruf:
    in_symbol: str | None
    name: str
    zeile: int


@dataclass(frozen=True)
class EnvLesung:
    name: str
    zeile: int
    in_symbol: str | None


@dataclass(frozen=True)
class Einstieg:
    art: Literal["main", "test", "route", "mcp", "cli_skript", "hook", "compose", "aufgabe", "regel"]
    symbol: str | None
    zeile: int


@dataclass(frozen=True)
class SqlObjekt:
    art: Literal["tabelle", "sql_funktion", "view"]
    name: str
    zeile: int
    digest: str


@dataclass(frozen=True)
class DateiErgebnis:
    symbole: tuple[Symbol, ...] = ()
    importe: tuple[Import, ...] = ()
    aufrufe: tuple[Aufruf, ...] = ()
    env: tuple[EnvLesung, ...] = ()
    einstiege: tuple[Einstieg, ...] = ()
    sql: tuple[SqlObjekt, ...] = ()
    tabellen_zugriffe: tuple[tuple[str, int, str | None], ...] = ()
    extra_knoten: tuple[Knoten, ...] = ()
    extra_kanten: tuple[Kante, ...] = ()
    fehlerhaft: bool = False
