"""Strukturregeln (Stufe 3, T8): Schreibziele pruefen (Waechter) und Pflichtteile pruefen (CI).

NUR Standardbibliothek (der Waechter-Hook importiert dies bei jedem Write/Edit): Regeln kommen aus dem kleinen
Leser `hooks.regeln_lesen`, Muster aus `_glob`, nie aus pydantic/yaml.

Muster-Semantik (`verboten`, `ort`): relativ zum Ordner der jeweiligen `selbstbild.yaml`, mit `/` als Trenner.
- `spaces/*/`   (endet auf `/`): Verzeichnismuster; trifft jeden Pfad, der unter einem solchen Verzeichnis liegt
                (`spaces/neu/x.py`, `spaces/a/b/c.py`), nicht aber `spaces/x.py`.
- `*/datei.yaml`: ohne `/` am Ende ein exaktes Muster; `*` bleibt innerhalb eines Segments, `**` ueberspannt Segmente.
Der Waechter blockiert nur wegen `verboten` (hart); die CI meldet getrackte Pfade unter `verboten` hart und
fehlende `pflicht`-Teile weich.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._glob import glob_regex
from .hooks import REGELDATEI, regeln_lesen, repo_wurzeln

__all__ = ["Verstoss", "pruefe_schreibziel", "regel_trifft", "regeln_fuer", "regel_ordner", "pruefen"]

GROSSKLEIN_EGAL = os.name == "nt"
_UEBERSPRUNGEN = {".git", "node_modules", "__pycache__", ".venv", ".venv-selbstbild"}
_MAX_TIEFE = 4  # so tief unter der Wurzel wird nach weiteren `selbstbild.yaml` gesucht (Submodule)
_GIT_FRIST_S = 60
_log = logging.getLogger(__name__)


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


_GIT_BASH = re.compile(r"^[\\/]([A-Za-z])(?:[\\/]|$)")


def _git_bash_pfad(roh: str, *, nt: bool | None = None) -> str:
    """Unter Windows Git-Bash-Form `/c/...` (auch als `\\c\\...`) auf `c:/...` abbilden, sonst unveraendert."""
    if not (os.name == "nt" if nt is None else nt):
        return roh
    m = _GIT_BASH.match(roh)
    return roh if m is None else f"{m.group(1)}:/{roh[m.end():]}"


def _aufloesen(ziel: Path, cwd: Path | None) -> Path:
    """Absolut, ohne `..`, Symlinks des vorhandenen Anteils aufgeloest; das Ziel selbst muss nicht existieren.
    Git-Bash-Pfade (`/c/...`) gelten unter Windows als Laufwerkspfade."""
    roh = _git_bash_pfad(str(ziel))
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


def regel_ordner(wurzel: Path) -> list[Path]:
    """Ordner mit `selbstbild.yaml` im Arbeitsbaum (bis Tiefe 4, ohne Punkt-Ordner), Wurzel zuerst, dann sortiert."""
    wurzel = Path(wurzel)
    gefunden: list[Path] = []
    for pfad, unterordner, dateien in os.walk(wurzel):
        p = Path(pfad)
        tiefe = len(p.relative_to(wurzel).parts)
        unterordner[:] = sorted(d for d in unterordner if d not in _UEBERSPRUNGEN and not d.startswith("."))
        if tiefe >= _MAX_TIEFE:
            unterordner[:] = []
        if REGELDATEI in dateien:
            gefunden.append(p)
    return sorted(gefunden, key=lambda p: (len(p.relative_to(wurzel).parts), p.relative_to(wurzel).as_posix()))


def _walk_ueberspringen(name: str) -> bool:
    return name in (".git", "node_modules", "__pycache__") or name.startswith(".venv")


def _getrackte_pfade(ordner: Path) -> list[str]:
    """Getrackte Pfade relativ zu `ordner` (`git ls-files -z`); ohne git ersatzweise der Arbeitsbaum (Log-Warnung)."""
    try:
        erg = subprocess.run(["git", "-C", str(ordner), "ls-files", "-z"], capture_output=True,
                             timeout=_GIT_FRIST_S, check=True)
        return [t for t in erg.stdout.decode("utf-8", "surrogateescape").split("\0") if t]
    except (OSError, subprocess.SubprocessError) as e:
        _log.warning("git ls-files fehlgeschlagen in %s (%s): verboten wird gegen den Arbeitsbaum geprueft",
                     ordner, type(e).__name__)
    aus: list[str] = []
    for pfad, unterordner, dateien in os.walk(ordner):
        unterordner[:] = sorted(d for d in unterordner if not _walk_ueberspringen(d))
        basis = Path(pfad).relative_to(ordner).as_posix()
        aus += [n if basis == "." else f"{basis}/{n}" for n in sorted(dateien)]
    return aus


def _verbotene(wurzel: Path) -> list[Verstoss]:
    """Harte Verstoesse: getrackte Pfade, die ein `verboten`-Muster der Regeldatei ihres Ordners treffen."""
    aus: list[Verstoss] = []
    for ordner in regel_ordner(wurzel):
        regeln = [(r, [(m, _muster(str(m))) for m in r.get("verboten") or []]) for r in _regeln_von(ordner)]
        regeln = [(r, ms) for r, ms in regeln if ms]
        if not regeln:
            continue
        vorn = ordner.relative_to(wurzel).as_posix()
        for rel in _getrackte_pfade(ordner):
            for r, ms in regeln:
                treffer = next((m for m, rx in ms if rx.match(rel)), None)
                if treffer is not None:
                    voll = rel if vorn == "." else f"{vorn}/{rel}"
                    aus.append(Verstoss(r.get("art", "?"), voll, f"'{rel}' ist laut Regel verboten ({treffer})",
                                        _hinweis(r), True))
    return aus


def pruefen(wurzel: Path) -> tuple[Verstoss, ...]:
    """CI (nur lesend): unlesbare Regeldatei (weich), getrackte Pfade unter `verboten` (hart, je Regeldatei im
    Arbeitsbaum relativ zu deren Ordner) und Ordner, die auf `ort` einer Regel mit `pflicht` passen, aber
    Pflichtteile missen (weich)."""
    wurzel = Path(wurzel)
    ungelesen = (*_ungelesen(wurzel), *_verbotene(wurzel))
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
