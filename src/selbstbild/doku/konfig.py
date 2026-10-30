"""Konfiguration des Doku-Loops laden und validieren (config/doku_loop.yaml)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

def _standard_pfad() -> Path:
    """Ohne --konfig: `config/doku_loop.yaml` unter `SELBSTBILD_VIBEMIND`, sonst unter dem Arbeitsordner."""
    wurzel = os.environ.get("SELBSTBILD_VIBEMIND")
    return (Path(wurzel) if wurzel else Path.cwd()) / "config" / "doku_loop.yaml"
MCP_MUSTER = frozenset({"decorator", "werkzeuge"})


class KonfigFehler(ValueError):
    """Die Konfiguration ist unvollstaendig oder widerspruechlich."""


@dataclass(frozen=True)
class SpaceKonfig:
    name: str
    wurzel: str  # repo-relativ, POSIX
    mcp_muster: str


@dataclass(frozen=True)
class Konfig:
    repo: Path
    spaces: dict[str, SpaceKonfig]
    ausschluss_namen: frozenset[str]
    ausschluss_pfade: frozenset[str]
    ketten_md: str
    weitere_doku: tuple[str, ...]
    compose_live: tuple[str, ...]
    sql_wurzeln: tuple[str, ...]
    dienste_yaml: str
    waechter_zustandsdatei: str
    waechter_zuordnung: dict[str, tuple[str, ...]]
    tick_minuten: int
    marker: Path
    agent: str
    judge_modell: str
    claude_bin: Path
    rowboat_wissen: Path
    rowboat_notizen: str
    rowboat_urteile: str
    # Final-Fix-Welle (Ruling 57), alle optional:
    zweig: str | None = None  # gesetzt -> geschrieben wird nur auf diesem Branch (sonst `falscher_zweig`)
    rowboat_notizen_aktiv: bool = True  # aus -> keine Rowboat-Notizen (Urteile werden trotzdem gelesen)
    tick_deadline_min: int = 10  # nach so vielen Minuten Laufzeit keine neuen Judge-Fragen mehr
    waechter_max_alter_min: int = 60  # aeltere Waechter-Zustandsdatei -> alle Messungen `keine_probe`
    # Selbstbild Stufe 4 (D4): Aussenabhaengigkeiten sind Konfiguration, beide repo-relativ
    budget_modul: str = "scripts/claude_budget.py"  # per Dateipfad geladen; fehlt es, gibt es kein LLM
    torwaechter: tuple[str, ...] = (".venv/Scripts/python.exe", "scripts/docs_drift_waechter.py")


def _pfad(wert: str) -> Path:
    return Path(os.path.expandvars(wert))


def _brauche(roh: dict[str, Any], schluessel: str) -> Any:
    if schluessel not in roh:
        raise KonfigFehler(f"Schluessel fehlt: {schluessel}")
    return roh[schluessel]


def _ganzzahl(roh: dict[str, Any], schluessel: str, vorgabe: int) -> int:
    wert = roh.get(schluessel, vorgabe)
    if isinstance(wert, bool) or not isinstance(wert, int) or wert <= 0:
        raise KonfigFehler(f"{schluessel} muss eine positive ganze Zahl sein")
    return wert


def _aussen(roh: dict[str, Any]) -> dict[str, Any]:
    """Optionale Schluessel `budget_modul` und `torwaechter`; fehlen sie, gelten die Vorgaben von `Konfig`."""
    aus: dict[str, Any] = {}
    if "budget_modul" in roh:
        aus["budget_modul"] = str(roh["budget_modul"])
    if "torwaechter" in roh:
        aus["torwaechter"] = tuple(map(str, roh["torwaechter"] or ()))
    return aus


def laden(pfad: Path | None = None) -> Konfig:
    datei = Path(pfad) if pfad is not None else _standard_pfad()
    roh = yaml.safe_load(datei.read_text(encoding="utf-8"))
    if not isinstance(roh, dict):
        raise KonfigFehler(f"Keine Zuordnung in {datei}")

    spaces: dict[str, SpaceKonfig] = {}
    for name, eintrag in dict(_brauche(roh, "spaces")).items():
        if not isinstance(eintrag, dict):
            raise KonfigFehler(f"Space {name}: keine Zuordnung")
        for k in ("mcp_muster", "wurzel"):
            if k not in eintrag:
                raise KonfigFehler(f"Space {name}: Schluessel {k} fehlt")
        muster = str(eintrag["mcp_muster"])
        if muster not in MCP_MUSTER:
            raise KonfigFehler(f"Space {name}: unbekanntes mcp_muster {muster!r}")
        spaces[str(name)] = SpaceKonfig(str(name), str(eintrag["wurzel"]).strip("/"), muster)

    tick = int(_brauche(roh, "tick_minuten"))
    if tick <= 0:
        raise KonfigFehler("tick_minuten muss positiv sein")
    zweig = roh.get("zweig")
    zweig = str(zweig).strip() if zweig is not None else ""
    notizen_aktiv = roh.get("rowboat_notizen_aktiv", True)
    if not isinstance(notizen_aktiv, bool):
        raise KonfigFehler("rowboat_notizen_aktiv muss true oder false sein")
    deadline = _ganzzahl(roh, "tick_deadline_min", 10)
    max_alter = _ganzzahl(roh, "waechter_max_alter_min", 60)

    return Konfig(
        repo=datei.resolve().parent.parent,
        spaces=spaces,
        ausschluss_namen=frozenset(map(str, _brauche(roh, "ausschluss_namen"))),
        ausschluss_pfade=frozenset(str(p).strip("/") for p in _brauche(roh, "ausschluss_pfade")),
        ketten_md=str(_brauche(roh, "ketten_md")),
        weitere_doku=tuple(map(str, roh.get("weitere_doku") or ())),
        compose_live=tuple(map(str, _brauche(roh, "compose_live"))),
        sql_wurzeln=tuple(map(str, _brauche(roh, "sql_wurzeln"))),
        dienste_yaml=str(_brauche(roh, "dienste_yaml")),
        waechter_zustandsdatei=str(_brauche(roh, "waechter_zustandsdatei")),
        waechter_zuordnung={
            str(k): tuple(map(str, v)) for k, v in dict(_brauche(roh, "waechter_zuordnung")).items()
        },
        tick_minuten=tick,
        marker=_pfad(str(_brauche(roh, "marker"))),
        agent=str(_brauche(roh, "agent")),
        judge_modell=str(_brauche(roh, "judge_modell")),
        claude_bin=_pfad(str(_brauche(roh, "claude_bin"))),
        rowboat_wissen=_pfad(str(_brauche(roh, "rowboat_wissen"))),
        rowboat_notizen=str(_brauche(roh, "rowboat_notizen")),
        rowboat_urteile=str(_brauche(roh, "rowboat_urteile")),
        zweig=zweig or None,
        rowboat_notizen_aktiv=notizen_aktiv,
        tick_deadline_min=deadline,
        waechter_max_alter_min=max_alter,
        **_aussen(roh),
    )
