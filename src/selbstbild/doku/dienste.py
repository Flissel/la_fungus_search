"""Dienst-Register: Env-Name -> Dienst (Port, Ort, Space). Seed aus Compose und vibemind.config.json."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .extraktoren.compose import ComposeService, ziel_service

ORTE = frozenset({"pc", "vm", "container", "extern"})
_SPACE_COMPOSE = re.compile(r"^vibemind-os/spaces/([^/]+)/")


def space_aus_compose(datei: str, name: str) -> str | None:
    """Ruling 32: Compose-Datei unter `vibemind-os/spaces/<x>/` -> `<x>`; sonst Name `brain*` -> `brain`."""
    m = _SPACE_COMPOSE.match(datei.replace("\\", "/"))
    if m:
        return m.group(1)
    return "brain" if name.startswith("brain") else None


@dataclass(frozen=True)
class Dienst:
    name: str
    port: int | None
    ort: str
    space: str | None
    env: tuple[str, ...]


class DienstRegister:
    def __init__(self, dienste: dict[str, Dienst]) -> None:
        self.dienste: dict[str, Dienst] = dict(sorted(dienste.items()))
        self._env: dict[str, Dienst] = {}
        for d in self.dienste.values():
            if d.ort not in ORTE:
                raise ValueError(f"Dienst {d.name}: unbekannter Ort {d.ort!r}")
            for e in d.env:
                if e in self._env and self._env[e].name != d.name:
                    raise ValueError(f"Env {e} gehoert zu {self._env[e].name} und {d.name}")
                self._env[e] = d

    @classmethod
    def laden(cls, pfad: Path) -> "DienstRegister":
        roh = yaml.safe_load(Path(pfad).read_text(encoding="utf-8")) or {}
        dienste: dict[str, Dienst] = {}
        for name, e in (roh.get("dienste") or {}).items():
            e = e or {}
            port = e.get("port")
            dienste[str(name)] = Dienst(
                str(name), None if port is None else int(port), str(e.get("ort")),
                None if e.get("space") is None else str(e["space"]),
                tuple(str(x) for x in (e.get("env") or ())))
        return cls(dienste)

    def aufloesen(self, env_name: str) -> Dienst | None:
        return self._env.get(env_name)

    def unbekannte(self, env_namen: Iterable[str]) -> list[str]:
        return sorted({n for n in env_namen if n not in self._env})

    def speichern(self, pfad: Path) -> None:
        daten = {"dienste": {
            n: {"port": d.port, "ort": d.ort, "space": d.space, "env": sorted(d.env)}
            for n, d in sorted(self.dienste.items())}}
        text = yaml.safe_dump(daten, sort_keys=False, allow_unicode=True, default_flow_style=False)
        Path(pfad).write_text(text, encoding="utf-8", newline="\n")


def seed_aus_compose(services: list[ComposeService], ports_json: Path) -> DienstRegister:
    """Compose-Services (container) plus ports.json (pc); Compose gewinnt bei gleichem Namen."""
    ports_roh = json.loads(Path(ports_json).read_text(encoding="utf-8")).get("ports") or {}
    alle = {s.name for s in services} | {str(n) for n in ports_roh}
    env_je_dienst: dict[str, set[str]] = {}
    for s in services:
        for schluessel, wert in s.env.items():
            ziel = ziel_service(wert, alle, schluessel)
            if ziel is not None and ziel != s.name:
                env_je_dienst.setdefault(ziel, set()).add(schluessel)
    vergeben: set[str] = set()  # ein Env-Name gehoert genau einem Dienst: alphabetisch erster gewinnt
    dienste: dict[str, Dienst] = {}
    for s in sorted(services, key=lambda x: x.name):
        if s.name in dienste:  # gleicher Name in mehreren Dateien: Port ergaenzen
            alt = dienste[s.name]
            dienste[s.name] = Dienst(s.name, alt.port or (s.ports[0] if s.ports else None),
                                     "container",
                                     alt.space if alt.space == space_aus_compose(s.datei, s.name) else None,
                                     alt.env)
            continue
        env = sorted(env_je_dienst.get(s.name, set()) - vergeben)
        vergeben.update(env)
        dienste[s.name] = Dienst(s.name, s.ports[0] if s.ports else None, "container",
                                 space_aus_compose(s.datei, s.name), tuple(env))
    for name in sorted(ports_roh):
        if name in dienste:
            continue
        env = sorted(env_je_dienst.get(name, set()) - vergeben)
        vergeben.update(env)
        dienste[name] = Dienst(name, int(ports_roh[name]), "pc", None, tuple(env))
    return DienstRegister(dienste)
