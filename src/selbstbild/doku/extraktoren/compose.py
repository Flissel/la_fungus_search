"""Compose-Extraktor: Services, Ports, Env und Service-zu-Service-Kanten der Live-Compose-Dateien."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..konfig import Konfig

_VERBOTEN = (".vibemind-stack-rendered-*.yml", "*partial-*.yml")
_VAR_VORGABE = re.compile(r"\$\{[A-Za-z_]\w*:?-([^}]*)\}")
_VAR_OHNE = re.compile(r"\$\{[A-Za-z_]\w*[^}]*\}|\$[A-Za-z_]\w*")
_NACKT_ENDEN = ("_HOST", "_HOSTNAME", "_ADDR", "_ADDRESS", "_SERVER", "_SERVICE", "_ENDPOINT", "_URL", "_URI")
_URL_HOST = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^/@\s]*@)?([^/:\s?#]+)")


@dataclass(frozen=True)
class ComposeService:
    datei: str  # repo-relativ, POSIX
    name: str
    ports: tuple[int, ...]
    env: dict[str, str]


@dataclass(frozen=True)
class ComposeKante:
    datei: str  # repo-relativ, POSIX
    von_service: str
    env: str
    nach_service: str


def _ersetzen(wert: str) -> str:
    """${VAR:-v} und ${VAR-v} -> v; ${VAR} / $VAR ohne Vorgabe -> ''."""
    wert = _VAR_VORGABE.sub(lambda m: m.group(1), wert)
    return _VAR_OHNE.sub("", wert)


def _port(eintrag: Any) -> int | None:
    if isinstance(eintrag, bool):
        return None
    if isinstance(eintrag, int):
        return eintrag
    if isinstance(eintrag, dict):
        return _port(eintrag.get("published"))
    if not isinstance(eintrag, str):
        return None
    teile = _ersetzen(eintrag).split("/", 1)[0].split(":")
    kandidat = teile[0] if len(teile) <= 2 else teile[-2]  # ip:host:container -> host
    kandidat = kandidat.strip()
    return int(kandidat) if kandidat.isdigit() else None


def _env(roh: Any) -> dict[str, str]:
    ergebnis: dict[str, str] = {}
    if isinstance(roh, dict):
        for k, v in roh.items():
            ergebnis[str(k)] = "" if v is None else _ersetzen(str(v))
    elif isinstance(roh, list):
        for z in roh:
            if not isinstance(z, str):
                continue
            k, _, v = z.partition("=")
            ergebnis[k.strip()] = _ersetzen(v)
    return dict(sorted(ergebnis.items()))


def ziel_service(wert: str, namen: set[str], schluessel: str = "") -> str | None:
    """Servicename, wenn der Wert eine URL mit diesem Host enthaelt oder (nur bei Host/URL-artigem
    Schluessel) nackt der Name ist."""
    wert = wert.strip()
    if wert in namen and schluessel.upper().endswith(_NACKT_ENDEN):
        return wert
    for host in _URL_HOST.findall(wert):
        if host in namen:
            return host
    return None


def _pruefe_pfad(pfad: str) -> None:
    name = pfad.replace("\\", "/").rsplit("/", 1)[-1]
    if any(fnmatch.fnmatch(name, m) for m in _VERBOTEN):
        raise ValueError(f"Compose-Datei nicht erlaubt (gerendert/partial): {pfad}")


def compose_lesen(konfig: Konfig) -> tuple[list[ComposeService], list[ComposeKante]]:
    for p in konfig.compose_live:
        _pruefe_pfad(p)
    services: list[ComposeService] = []
    kanten: list[ComposeKante] = []
    for rel in konfig.compose_live:
        pfad = Path(konfig.repo) / rel
        if not pfad.is_file():
            continue
        roh = yaml.safe_load(pfad.read_text(encoding="utf-8"))
        eintraege = roh.get("services") if isinstance(roh, dict) else None
        if not isinstance(eintraege, dict):
            continue
        datei = rel.replace("\\", "/")
        namen = {str(n) for n in eintraege}
        for name in sorted(namen):
            sv = eintraege[name] if isinstance(eintraege[name], dict) else {}
            ports = tuple(sorted({p for e in (sv.get("ports") or []) if (p := _port(e)) is not None}))
            env = _env(sv.get("environment"))
            services.append(ComposeService(datei, name, ports, env))
            for schluessel, wert in env.items():
                ziel = ziel_service(wert, namen, schluessel)
                if ziel is not None and ziel != name:
                    kanten.append(ComposeKante(datei, name, schluessel, ziel))
    return services, kanten
