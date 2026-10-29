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

__all__ = ["Verstoss", "pruefe_schreibziel", "regel_trifft", "regeln_fuer", "pruefen"]

GROSSKLEIN_EGAL = os.name == "nt"
_UEBERSPRUNGEN = {".git", "node_modules", "__pycache__", ".venv", ".venv-selbstbild"}


@dataclass(frozen=True)
class Verstoss:
    regel: str
    pfad: str
    grund: str
    hinweis: str | None
    hart: bool


def _flags(grossklein_egal: bool | None = None) -> int:
    egal = GROSSKLEIN_EGAL if grossklein_egal is None else grossklein_egal
    return re.IGNORECASE if egal else 0


def _muster(m: str, flags: int | None = None) -> re.Pattern[str]:
    """Verzeichnismuster (`.../`) = Praefix, sonst exakter Glob."""
    f = _flags() if flags is None else flags
    m = m.replace("\\", "/").lstrip("/")
    if m.endswith("/"):  # Praefix: mindestens ein weiteres Segment unterhalb des Verzeichnisses
        return re.compile(glob_regex(m.rstrip("/")).pattern.removesuffix(r"\Z") + r"/.+\Z", f)
    return re.compile(glob_regex(m).pattern, f)


def ort_ist_wurzel(ort: object) -> bool:
    """`ort` der Repo-Wurzel (`""`, `.`, `./`): die Regel gilt fuer das ganze Repo."""
    return str(ort).replace("\\", "/").strip().rstrip("/") in ("", ".")


def regel_trifft(regel: dict, rel: str, *, grossklein_egal: bool | None = None) -> bool:
    """Gilt die Regel fuer den repo-relativen Pfad `rel` (mit `/`)? Gemeinsamer Trefferbegriff von Waechter-Umfeld
    (`regeln_fuer`) und MCP-Server (`selbstbild_regeln`), damit beide nicht auseinanderlaufen.

    Ja, wenn `ort/**` den Pfad umfasst (ein Wurzel-`ort` umfasst alles) oder eines ihrer `verboten`-Muster ihn trifft
    (dieselbe Muster-Semantik wie der Waechter)."""
    f = _flags(grossklein_egal)
    ort = str(regel.get("ort", "")).replace("\\", "/").strip()
    if ort_ist_wurzel(ort):
        return True
    ort = ort.removeprefix("./").rstrip("/")
    if re.compile(glob_regex(ort + "/**").pattern, f).match(rel):
        return True
    return any(_muster(str(m), f).match(rel) for m in regel.get("verboten") or [])


def _regeln_von(ordner: Path) -> list[dict]:
    try:
        return regeln_lesen((ordner / REGELDATEI).read_text(encoding="utf-8")).get("struktur", [])
    except (OSError, UnicodeDecodeError):
        return []


def _ungelesen(wurzel: Path) -> tuple[Verstoss, ...]:
    """CI faellt geschlossen: `struktur:` steht in der Regeldatei, der kleine Leser liefert aber keine Regeln."""
    datei = wurzel / REGELDATEI
    try:
        text = datei.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ()
    if re.search(r"^struktur:", text, re.MULTILINE) and not regeln_lesen(text).get("struktur"):
        return (Verstoss("regeldatei", REGELDATEI, f"Regeldatei nicht lesbar: {datei} – Strukturregeln werden nicht geprueft",
                         None, False),)
    return ()


def _hinweis(r: dict) -> str | None:
    """`hinweis` der Regel, sonst der richtige Ort; ein `ort` der Repo-Wurzel ist nur der Geltungsbereich."""
    if r.get("hinweis"):
        return r["hinweis"]
    return None if ort_ist_wurzel(r.get("ort", "")) else f"richtiger Ort: {r.get('ort', '?')}"


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
    """Regeln der `selbstbild.yaml` in `wurzel` (Arbeitsbaum), die laut `regel_trifft` fuer den Pfad gelten."""
    rel = _relativ(wurzel, pfad)
    return tuple(dict(r) for r in _regeln_von(Path(wurzel)) if regel_trifft(r, rel))


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
    ungelesen = _ungelesen(wurzel)
    regeln = [r for r in _regeln_von(wurzel) if r.get("pflicht") and r.get("ort")]
    if not regeln:
        return ungelesen
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
    return (*ungelesen, *verstoesse)
