"""Daten- und Skript-Parser: SQL, YAML, JSON (package.json), dotenv, Bash, PowerShell.

Gespeichert werden nur Namen (Env-Namen, SQL-Objektnamen, Skriptnamen), nie Werte.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import posixpath
import re
from collections.abc import Iterator
from typing import Any

import yaml

from selbstbild.spezial import spezial_compose
from selbstbild.typen import PLATZHALTER_REPO, DateiErgebnis, Einstieg, EnvLesung, SqlObjekt, id_tabelle

_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")


def _text(quelle: bytes) -> str:
    return quelle.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _env_ergebnis(treffer: dict[str, int]) -> DateiErgebnis:
    env = tuple(EnvLesung(name=n, zeile=z, in_symbol=None)
                for n, z in sorted(treffer.items(), key=lambda kv: (kv[1], kv[0])))
    return DateiErgebnis(env=env)


# --------------------------------------------------------------------------- SQL

_NAME = r'(?:"[^"]+"|\w+)(?:\s*\.\s*(?:"[^"]+"|\w+))?'
_CREATE_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?"
    r"(?:(?:GLOBAL|LOCAL)\s+)?(?:(?:UNLOGGED|TEMP|TEMPORARY)\s+)?"
    r"(?:(?:MATERIALIZED|RECURSIVE)\s+)?"
    r"(TABLE|FUNCTION|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?(" + _NAME + ")",
    re.IGNORECASE,
)
_DOLLAR_RE = re.compile(r"\$(?:[A-Za-z_]\w*)?\$")


def _sql_anweisungen(text: str) -> Iterator[tuple[int, str]]:
    """Zerlegt SQL in (Startoffset, Anweisung ohne Kommentare).

    Beachtet '...'-Strings, "..."-Bezeichner, -- und /* */-Kommentare sowie
    $$...$$ bzw. $tag$...$tag$. Kommentare werden durch Leerraum ersetzt
    (Zeilenumbrueche bleiben), damit Zeilennummern stimmen.
    """
    i, n = 0, len(text)
    puffer: list[str] = []
    start = 0
    while i < n:
        c = text[i]
        if c == "-" and text.startswith("--", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            puffer.append(" " * (j - i))
            i = j
        elif c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            puffer.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c in ("'", '"'):
            j = i + 1
            while j < n:
                if text[j] == c:
                    if j + 1 < n and text[j + 1] == c:
                        j += 2
                        continue
                    break
                j += 1
            j = min(j + 1, n)
            puffer.append(text[i:j])
            i = j
        elif c == "$" and (m := _DOLLAR_RE.match(text, i)):
            ende = text.find(m.group(0), m.end())
            j = n if ende < 0 else ende + len(m.group(0))
            puffer.append(text[i:j])
            i = j
        elif c == ";":
            yield start, "".join(puffer)
            puffer = []
            i += 1
            start = i
        else:
            puffer.append(c)
            i += 1
    if "".join(puffer).strip():
        yield start, "".join(puffer)


def parse_sql(pfad: str, quelle: bytes) -> DateiErgebnis:
    text = _text(quelle)
    objekte: list[SqlObjekt] = []
    for start, anweisung in _sql_anweisungen(text):
        m = _CREATE_RE.match(anweisung)
        if not m:
            continue
        art = {"TABLE": "tabelle", "FUNCTION": "sql_funktion", "VIEW": "view"}[m.group(1).upper()]
        name = id_tabelle(m.group(2).replace(" ", "")).removeprefix("tabelle:")
        zeile = text.count("\n", 0, start + m.start(1)) + 1
        digest = hashlib.sha256(anweisung.strip().encode("utf-8")).hexdigest()
        objekte.append(SqlObjekt(art=art, name=name, zeile=zeile, digest=digest))  # type: ignore[arg-type]
    return DateiErgebnis(sql=tuple(objekte))


# -------------------------------------------------------------------------- YAML

_REF_RE = re.compile(r"(?<!\$)\$\{?([A-Za-z_][A-Za-z0-9_]*)")


def _erste_zeile(text: str, name: str) -> int:
    m = re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text)
    return text.count("\n", 0, m.start()) + 1 if m else 1


def _yaml_namen(knoten: Any, namen: set[str], in_env: bool = False) -> None:
    if isinstance(knoten, dict):
        for k, v in knoten.items():
            if in_env and isinstance(k, str) and _ENV_NAME.fullmatch(k):
                namen.add(k)
            _yaml_namen(v, namen, in_env=False if in_env else k == "environment")
    elif isinstance(knoten, list):
        for v in knoten:
            _yaml_namen(v, namen, in_env)
    elif isinstance(knoten, str):
        wert = knoten
        if in_env:
            name, gleich, rest = knoten.partition("=")
            if _ENV_NAME.fullmatch(name.strip()):
                namen.add(name.strip())
            wert = rest if gleich else ""
        for ref in _REF_RE.findall(wert):
            if _ENV_NAME.fullmatch(ref):
                namen.add(ref)


def parse_yaml(pfad: str, quelle: bytes) -> DateiErgebnis:
    """Env-Namen aus `environment:` (Liste/Mapping) und `${NAME}`-Referenzen.

    Zeilennummern sind Best-Effort: erste Fundstelle des Namens im Rohtext.
    """
    text = _text(quelle)
    try:
        dokumente = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return DateiErgebnis(fehlerhaft=True)
    namen: set[str] = set()
    for dok in dokumente:
        _yaml_namen(dok, namen)
    ergebnis = _env_ergebnis({n: _erste_zeile(text, n) for n in namen})
    if fnmatch.fnmatch(posixpath.basename(pfad.replace("\\", "/")).lower(), "*compose*.y*ml"):
        knoten, kanten = spezial_compose(PLATZHALTER_REPO, pfad, quelle)
        ergebnis = dataclasses.replace(ergebnis, extra_knoten=knoten, extra_kanten=kanten)
    return ergebnis


# ------------------------------------------------------------------------- dotenv

_DOTENV_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def parse_dotenv(pfad: str, quelle: bytes) -> DateiErgebnis:
    treffer: dict[str, int] = {}
    for nr, zeile in enumerate(_text(quelle).split("\n"), start=1):
        m = _DOTENV_RE.match(zeile)
        if m:
            treffer.setdefault(m.group(1), nr)
    return _env_ergebnis(treffer)


# --------------------------------------------------------------------------- Shell

_BASH_RE = re.compile(r"(?<!\\)\$\{?([A-Za-z_][A-Za-z0-9_]*)")
_PS_RE = re.compile(r"\$\{?env:([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def parse_bash(pfad: str, quelle: bytes) -> DateiErgebnis:
    """Nur Grossschreibung (`${NAME}`, `$NAME`); `$1`, `$@` und Kleinbuchstaben entfallen."""
    treffer: dict[str, int] = {}
    for nr, zeile in enumerate(_text(quelle).split("\n"), start=1):
        if zeile.lstrip().startswith("#"):
            continue
        for name in _BASH_RE.findall(zeile):
            if _ENV_NAME.fullmatch(name):
                treffer.setdefault(name, nr)
    return _env_ergebnis(treffer)


def parse_powershell(pfad: str, quelle: bytes) -> DateiErgebnis:
    """`$env:NAME` bzw. `${env:NAME}` per Regex; Name wie geschrieben (Windows: gleichgueltig)."""
    treffer: dict[str, int] = {}
    for nr, zeile in enumerate(_text(quelle).split("\n"), start=1):
        if zeile.lstrip().startswith("#"):
            continue
        for name in _PS_RE.findall(zeile):
            treffer.setdefault(name, nr)
    return _env_ergebnis(treffer)


# --------------------------------------------------------------------------- JSON

def parse_json(pfad: str, quelle: bytes) -> DateiErgebnis:
    """Nur `package.json`: `scripts` und `bin` als Einstiege `cli_skript`."""
    if posixpath.basename(pfad.replace("\\", "/")) != "package.json":
        return DateiErgebnis()
    text = _text(quelle)
    try:
        daten = json.loads(text)
    except json.JSONDecodeError:
        return DateiErgebnis(fehlerhaft=True)
    if not isinstance(daten, dict):
        return DateiErgebnis()
    symbole: list[str] = []
    scripts = daten.get("scripts")
    if isinstance(scripts, dict):
        symbole += [k for k in scripts if isinstance(k, str)]
    bin_ = daten.get("bin")
    if isinstance(bin_, dict):
        symbole += [k for k in bin_ if isinstance(k, str)]
    elif isinstance(bin_, str) and isinstance(daten.get("name"), str):
        symbole.append(daten["name"].rsplit("/", 1)[-1])
    einstiege = []
    for s in symbole:
        m = re.search(re.escape(json.dumps(s)), text)
        zeile = text.count("\n", 0, m.start()) + 1 if m else 1
        einstiege.append(Einstieg(art="cli_skript", symbol=s, zeile=zeile))
    return DateiErgebnis(einstiege=tuple(einstiege))
