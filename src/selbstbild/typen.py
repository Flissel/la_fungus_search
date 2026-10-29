"""Gemeinsame Typen und Id-Schema des Selbstbilds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KNOTENARTEN = (
    "repo", "datei", "symbol", "route", "tabelle",
    "sql_funktion", "mcp_werkzeug", "env", "dienst", "doku_seite",
)
# nutzt_dienst: Symbol/Datei -> dienst:<name> (aus dem Dienst-Register);
# ruft_http: HTTP-Aufrufstelle -> env:<BASE_URL>, Methode und Pfad im Beleg-Anker (R18).
KANTENARTEN = (
    "enthaelt", "importiert", "ruft_auf", "definiert_route", "liest_schreibt",
    "liest_env", "registriert", "verbindet", "nutzt_dienst", "ruft_http", "geaendert_durch",
)


@dataclass(frozen=True)
class Beleg:
    repo: str
    pfad: str
    zeile: int
    anker: str
    # belegt die Quelle, nicht den Anker (R19): Symbolkoerper (Symbol.digest), auf Modulebene
    # die Blob-SHA, bei Spezialextraktoren sha256 des AST-/YAML-Ausschnitts
    digest: str


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


# Parser-Ergebnisse werden je Blob gecacht und enthalten nie das echte Repo (R10).
PLATZHALTER_REPO = "@"


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


def id_doku_seite(repo: str, pfad: str) -> str:
    return f"doku_seite:{repo}:{pfad}"


@dataclass(frozen=True)
class Symbol:
    qualname: str
    art: Literal["funktion", "klasse", "methode"]
    zeile_von: int
    zeile_bis: int
    digest: str
    # Ruling S10: Decorator-Koepfe ohne Argumente in Quellreihenfolge (`kopf.timer`, `app.get`)
    dekoriert: tuple[str, ...] = ()


@dataclass(frozen=True)
class Import:
    modul: str
    namen: tuple[str, ...]
    zeile: int
    relativ: bool
    # Re-Exporte: (exportierter Name, Quellname in `modul`); Quellname "*" = Namensraum (`export * as ns`)
    umbenennungen: tuple[tuple[str, str], ...] = ()


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
    # Spezialextraktoren (ast) konnten die Datei nicht lesen, obwohl der Parser lief:
    # Routen, MCP-Werkzeuge und HTTP-Aufrufe dieser Datei fehlen (Datei-Attr `spezial_fehlt`).
    spezial_fehlt: bool = False
