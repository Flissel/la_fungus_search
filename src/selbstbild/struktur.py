"""Strukturregeln (Stufe 3, T8): Schreibziele pruefen (Waechter) und Pflichtteile pruefen (CI).

NUR Standardbibliothek (der Waechter-Hook importiert dies bei jedem Write/Edit): Regeln kommen aus dem kleinen
Leser `hooks.regeln_lesen`, Muster aus `_glob`, nie aus pydantic/yaml.

Muster-Semantik (`verboten`, `ort`): relativ zum Ordner der jeweiligen `selbstbild.yaml`, mit `/` als Trenner.
- `spaces/*/`   (endet auf `/`): Verzeichnismuster; trifft jeden Pfad, der unter einem solchen Verzeichnis liegt
                (`spaces/neu/x.py`, `spaces/a/b/c.py`), nicht aber `spaces/x.py`.
- `*/datei.yaml`: ohne `/` am Ende ein exaktes Muster; `*` bleibt innerhalb eines Segments, `**` ueberspannt Segmente.
Der Waechter blockiert nur wegen `verboten` (hart); fehlende `pflicht`-Teile sind weiche Verstoesse der CI.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ._glob import glob_regex
from .hooks import REGELDATEI, regeln_lesen, repo_wurzeln

__all__ = ["Verstoss", "pruefe_schreibziel", "regeln_fuer", "pruefen"]

GROSSKLEIN_EGAL = os.name == "nt"
_UEBERSPRUNGEN = {".git", "node_modules", "__pycache__", ".venv", ".venv-selbstbild"}


@dataclass(frozen=True)
class Verstoss:
    regel: str
    pfad: str
    grund: str
    hinweis: str | None
    hart: bool


def _flags() -> int:
    return re.IGNORECASE if GROSSKLEIN_EGAL else 0


def _muster(m: str) -> re.Pattern[str]:
    """Verzeichnismuster (`.../`) = Praefix, sonst exakter Glob."""
    m = m.replace("\\", "/").lstrip("/")
    if m.endswith("/"):  # Praefix: mindestens ein weiteres Segment unterhalb des Verzeichnisses
        return re.compile(glob_regex(m.rstrip("/")).pattern.removesuffix(r"\Z") + r"/.+\Z", _flags())
    return re.compile(glob_regex(m).pattern, _flags())


def _regeln_von(ordner: Path) -> list[dict]:
    try:
        return regeln_lesen((ordner / REGELDATEI).read_text(encoding="utf-8")).get("struktur", [])
    except (OSError, UnicodeDecodeError):
        return []


def _hinweis(r: dict) -> str:
    return r.get("hinweis") or f"richtiger Ort: {r.get('ort', '?')}"


def _aufloesen(ziel: Path, cwd: Path | None) -> Path:
    """Absolut, ohne `..`, Symlinks des vorhandenen Anteils aufgeloest; das Ziel selbst muss nicht existieren."""
    roh = str(ziel)
    if os.name == "nt":
        roh = roh.replace("/", "\\")
    p = Path(os.path.normpath(os.path.join(str(cwd) if cwd else os.getcwd(), roh)))
    rest: list[str] = []
    while not p.exists() and p.parent != p:
        rest.append(p.name)
        p = p.parent
    p = Path(os.path.realpath(p))
    for teil in reversed(rest):
        p = p / teil
    return p


def pruefe_schreibziel(ziel: Path, cwd: Path | None = None) -> tuple[Verstoss, ...]:
    """Harte Verstoesse, wenn `ziel` auf `verboten` einer Strukturregel irgendeines umschliessenden Repos trifft."""
    p = _aufloesen(Path(ziel), cwd)
    gefunden: list[Verstoss] = []
    for wurzel in repo_wurzeln(p.parent):
        try:
            rel = p.relative_to(wurzel).as_posix()
        except ValueError:
            continue
        for r in _regeln_von(wurzel):
            for muster in r.get("verboten", []):
                if _muster(muster).match(rel):
                    gefunden.append(Verstoss(r.get("art", "?"), rel, f"'{rel}' ist laut Regel verboten ({muster})",
                                             _hinweis(r), True))
                    break
    return tuple(gefunden)


def _relativ(wurzel: Path, pfad: str) -> str:
    p = pfad.replace("\\", "/")
    if os.path.isabs(pfad):
        try:
            p = Path(os.path.normpath(pfad)).relative_to(Path(wurzel).resolve()).as_posix()
        except ValueError:
            try:
                p = Path(os.path.normpath(pfad)).relative_to(Path(wurzel)).as_posix()
            except ValueError:
                return p
    return p.removeprefix("./")


def regeln_fuer(wurzel: Path, pfad: str) -> tuple[dict, ...]:
    """Regeln der `selbstbild.yaml` in `wurzel`, deren `ort` den Pfad umfasst oder deren `verboten` ihn trifft."""
    rel = _relativ(wurzel, pfad)
    aus: list[dict] = []
    for r in _regeln_von(Path(wurzel)):
        ort = str(r.get("ort", "")).replace("\\", "/").rstrip("/")
        umfasst = bool(ort) and re.compile(glob_regex(ort + "/**").pattern, _flags()).match(rel) is not None
        if umfasst or any(_muster(m).match(rel) for m in r.get("verboten", [])):
            aus.append({k: v for k, v in r.items()})
    return tuple(aus)


def _hat_pflichtteil(ordner: Path, teil: str) -> bool:
    teil = teil.replace("\\", "/")
    if not any(c in teil for c in "*?"):
        return (ordner / teil).exists()
    rx = re.compile(glob_regex(teil).pattern, _flags())
    for pfad, unterordner, dateien in os.walk(ordner):
        unterordner[:] = [d for d in unterordner if d not in _UEBERSPRUNGEN]
        basis = Path(pfad).relative_to(ordner).as_posix()
        for name in (*dateien, *unterordner):
            rel = name if basis == "." else f"{basis}/{name}"
            if rx.match(rel):
                return True
    return False


def pruefen(wurzel: Path) -> tuple[Verstoss, ...]:
    """CI: Ordner, die auf `ort` einer Regel mit `pflicht` passen, aber Pflichtteile missen (weich; nur lesend)."""
    wurzel = Path(wurzel)
    regeln = [r for r in _regeln_von(wurzel) if r.get("pflicht") and r.get("ort")]
    if not regeln:
        return ()
    muster = [(r, re.compile(glob_regex(str(r["ort"]).replace("\\", "/").rstrip("/")).pattern, _flags()),
               None if "**" in r["ort"] else r["ort"].rstrip("/").count("/") + 1) for r in regeln]
    tiefe_max = None if any(t is None for _, _, t in muster) else max(t for _, _, t in muster)  # type: ignore[type-var]
    verstoesse: list[Verstoss] = []
    for pfad, unterordner, _ in os.walk(wurzel):
        unterordner[:] = sorted(d for d in unterordner if d not in _UEBERSPRUNGEN)
        basis = Path(pfad).relative_to(wurzel).as_posix()
        tiefe = 0 if basis == "." else basis.count("/") + 1
        if tiefe_max is not None and tiefe >= tiefe_max:
            unterordner[:] = []
        if basis == ".":
            continue
        for r, rx, _ in muster:
            if rx.match(basis):
                for teil in r["pflicht"]:
                    if not _hat_pflichtteil(Path(pfad), teil):
                        verstoesse.append(Verstoss(r.get("art", "?"), basis, f"Pflichtteil fehlt: {teil}",
                                                   r.get("hinweis"), False))
    return tuple(verstoesse)
