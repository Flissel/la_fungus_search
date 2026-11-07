"""Git-Arbeit des Debugger-Claw auf dem VM-Klon: nachziehen, Worktrees, aufraeumen.

Alle Aufrufe laufen ohne Shell. Git-Fehler tragen nur Unterbefehl und
Rueckgabecode, nie stderr (darin koennen URLs mit Token stehen).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Sequence

PRAEFIX = "debugger/"
TITEL_PRAEFIX = "[selbstbild] "
MAX_KURZ = 30


class KonfliktFehler(Exception):
    """Lokaler und entfernter Zweig sind auseinandergelaufen."""


class GitFehler(Exception):
    """Ein Git-Aufruf ist gescheitert; nennt nur Unterbefehl und Code."""

    def __init__(self, unterbefehl: str, code: int) -> None:
        super().__init__(f"git {unterbefehl} fehlgeschlagen (Code {code})")
        self.unterbefehl = unterbefehl
        self.code = code


def _git(
    ort: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Fuehrt `git -C <ort> ...` aus; bei check=True nur ein schlanker Fehler."""
    erg = subprocess.run(
        ["git", "-C", str(ort), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and erg.returncode != 0:
        raise GitFehler(args[0] if args else "?", erg.returncode)
    return erg


def _ref_da(ort: Path, ref: str) -> bool:
    return _git(ort, "rev-parse", "--verify", "-q", ref, check=False).returncode == 0


def _debugger_zweig_pruefen(branch: str) -> None:
    if not branch.startswith(PRAEFIX):
        raise ValueError("kein_debugger_zweig")


def nachziehen(klon: Path, basis: str, submodule: Sequence[str]) -> str | None:
    """Zieht den Klon per Fast-Forward auf `origin/<basis>`.

    Rueckgabe None bei Erfolg, sonst "falscher_zweig", "nicht_sauber",
    "kein_fast_forward" oder "submodul_fehler". Bei den ersten beiden Werten
    bleibt der Arbeitsbaum unberuehrt.
    """
    _git(klon, "fetch", "--prune")
    aktuell = _git(klon, "symbolic-ref", "--short", "HEAD", check=False)
    if aktuell.returncode != 0 or aktuell.stdout.strip() != basis:
        return "falscher_zweig"
    if _git(klon, "status", "--porcelain").stdout.strip():
        return "nicht_sauber"
    if _git(klon, "merge", "--ff-only", f"origin/{basis}", check=False).returncode != 0:
        return "kein_fast_forward"
    for pfad in submodule:
        erg = _git(klon, "submodule", "update", "--init", "--", pfad, check=False)
        if erg.returncode != 0:
            return "submodul_fehler"
    return None


def branch_name(issue: int, titel: str) -> str:
    """`debugger/<issue>-<kurz>`; kurz hoechstens 30 Zeichen aus [a-z0-9-]."""
    text = titel.removeprefix(TITEL_PRAEFIX).lower()
    # Zeichenweise ersetzen, auf 30 kuerzen, dann zusammenfassen (so verlangt es
    # der verbindliche Testfall aus dem Brief).
    roh = re.sub(r"[^a-z0-9]", "-", text)[:MAX_KURZ]
    kurz = re.sub(r"-{2,}", "-", roh).strip("-") or "fund"
    return f"{PRAEFIX}{issue}-{kurz}"


def worktree_bereit(
    repo_klon: Path, wt_ordner: Path, branch: str, basis: str
) -> Path:
    """Stellt einen Worktree fuer `branch` bereit.

    Fehlt er, wird er aus `origin/<branch>` oder `origin/<basis>` angelegt
    (ein lokaler Alt-Zweig ohne Worktree wird wiederverwendet, nie geloescht).
    Existiert er und gibt es `origin/<branch>`, wird per ff-only nachgezogen;
    laufen beide Seiten auseinander, folgt KonfliktFehler und nichts aendert sich.
    """
    _debugger_zweig_pruefen(branch)
    _git(repo_klon, "fetch", "--prune", "origin")
    remote = f"origin/{branch}"
    remote_da = _ref_da(repo_klon, f"refs/remotes/{remote}")
    if not wt_ordner.exists():
        wt_ordner.parent.mkdir(parents=True, exist_ok=True)
        _git(repo_klon, "worktree", "prune")
        if _ref_da(repo_klon, f"refs/heads/{branch}"):
            _git(repo_klon, "worktree", "add", str(wt_ordner), branch)
        else:
            start = remote if remote_da else f"origin/{basis}"
            _git(repo_klon, "worktree", "add", "-b", branch, str(wt_ordner), start)
            return wt_ordner
    if not remote_da:
        return wt_ordner
    if _git(wt_ordner, "merge", "--ff-only", remote, check=False).returncode != 0:
        lokal_ist_vorfahr = _git(
            wt_ordner, "merge-base", "--is-ancestor", "HEAD", remote, check=False
        ).returncode
        remote_ist_vorfahr = _git(
            wt_ordner, "merge-base", "--is-ancestor", remote, "HEAD", check=False
        ).returncode
        if lokal_ist_vorfahr != 0 and remote_ist_vorfahr != 0:
            raise KonfliktFehler(branch)
        raise GitFehler("merge", 1)
    return wt_ordner


def voraus(wt: Path, branch: str, *, basis: str = "master") -> int:
    """Zahl der Commits auf HEAD, die nicht auf `origin/<branch>` liegen.

    Gibt es `origin/<branch>` nicht, gilt `origin/<basis>` als Gegenseite.
    """
    ref = f"origin/{branch}"
    if not _ref_da(wt, f"refs/remotes/{ref}"):
        ref = f"origin/{basis}"
    erg = _git(wt, "rev-list", "--count", "HEAD", f"^{ref}")
    return int(erg.stdout.strip())


def aufraeumen(repo_klon: Path, wt: Path, branch: str) -> None:
    """Entfernt Worktree und lokalen Zweig des Helfers (nur `debugger/*`)."""
    _debugger_zweig_pruefen(branch)
    if wt.exists():
        _git(repo_klon, "worktree", "remove", "--force", str(wt))
    _git(repo_klon, "worktree", "prune")
    if _ref_da(repo_klon, f"refs/heads/{branch}"):
        _git(repo_klon, "branch", "-D", branch)
