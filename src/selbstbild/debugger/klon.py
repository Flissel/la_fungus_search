"""Git-Arbeit des Debugger-Claw auf dem VM-Klon: nachziehen, Worktrees, aufraeumen.

Alle Aufrufe laufen ohne Shell. Git-Fehler tragen nur Unterbefehl und
Rueckgabecode, nie stderr (darin koennen URLs mit Token stehen).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

PRAEFIX = "debugger/"
TITEL_PRAEFIX = "[selbstbild] "
MAX_KURZ = 30
GIT_TIMEOUT_S = 300
TIMEOUT_CODE = -1


class KonfliktFehler(Exception):
    """Lokaler und entfernter Zweig sind auseinandergelaufen."""


class GitFehler(Exception):
    """Ein Git-Aufruf ist gescheitert; nennt nur Unterbefehl und Code."""

    def __init__(self, unterbefehl: str, code: int) -> None:
        zusatz = " (Zeitueberschreitung)" if code == TIMEOUT_CODE else f" (Code {code})"
        super().__init__(f"git {unterbefehl} fehlgeschlagen{zusatz}")
        self.unterbefehl = unterbefehl
        self.code = code


class WorktreeFehler(ValueError):
    """`wt` ist kein registrierter Worktree des Klons auf dem erwarteten Zweig."""


def git_lauf(
    ort: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Fuehrt `git -C <ort> ...` aus, ohne Eingabe und ohne Passwort-Abfrage.

    Zeitueberschreitung wird zu GitFehler(code=TIMEOUT_CODE); bei check=True
    gibt es nur einen schlanken Fehler (nie stderr).
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        erg = subprocess.run(
            ["git", "-C", str(ort), *args],
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise GitFehler(args[0] if args else "?", TIMEOUT_CODE) from None
    if check and erg.returncode != 0:
        raise GitFehler(args[0] if args else "?", erg.returncode)
    return erg


def ist_vorfahr(ort: Path, alt: str, neu: str) -> bool:
    """`merge-base --is-ancestor`: 0 ja, 1 nein, alles andere ist ein Fehler."""
    code = git_lauf(ort, "merge-base", "--is-ancestor", alt, neu, check=False).returncode
    if code not in (0, 1):
        raise GitFehler("merge-base", code)
    return code == 0


def _norm(pfad: str | Path) -> str:
    return os.path.normcase(str(Path(pfad).resolve()))


def _worktree_pruefen(repo_klon: Path, wt: Path, branch: str) -> None:
    """Wirft WorktreeFehler, wenn `wt` kein zusaetzlicher Worktree auf `branch` ist."""
    liste = git_lauf(repo_klon, "worktree", "list", "--porcelain").stdout
    pfade = [z[len("worktree "):] for z in liste.splitlines() if z.startswith("worktree ")]
    # Der erste Eintrag ist der Haupt-Checkout selbst und nie ein Helfer-Worktree.
    if _norm(wt) not in {_norm(p) for p in pfade[1:]}:
        raise WorktreeFehler("kein_worktree_des_klons")
    aktuell = git_lauf(wt, "symbolic-ref", "--short", "HEAD", check=False)
    if aktuell.returncode != 0 or aktuell.stdout.strip() != branch:
        raise WorktreeFehler("falscher_zweig_im_worktree")


def _ref_da(ort: Path, ref: str) -> bool:
    return git_lauf(ort, "rev-parse", "--verify", "-q", ref, check=False).returncode == 0


def _debugger_zweig_pruefen(branch: str) -> None:
    if not branch.startswith(PRAEFIX):
        raise ValueError("kein_debugger_zweig")


def nachziehen(klon: Path, basis: str, submodule: Sequence[str]) -> str | None:
    """Zieht den Klon per Fast-Forward auf `origin/<basis>`.

    Rueckgabe None bei Erfolg, sonst "falscher_zweig", "nicht_sauber",
    "kein_fast_forward" oder "submodul_fehler". Bei den ersten beiden Werten
    bleibt der Arbeitsbaum unberuehrt.
    """
    git_lauf(klon, "fetch", "--prune")
    aktuell = git_lauf(klon, "symbolic-ref", "--short", "HEAD", check=False)
    if aktuell.returncode != 0 or aktuell.stdout.strip() != basis:
        return "falscher_zweig"
    if git_lauf(klon, "status", "--porcelain").stdout.strip():
        return "nicht_sauber"
    if git_lauf(klon, "merge", "--ff-only", f"origin/{basis}", check=False).returncode != 0:
        return "kein_fast_forward"
    for pfad in submodule:
        erg = git_lauf(klon, "submodule", "update", "--init", "--", pfad, check=False)
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
    if wt_ordner.exists():
        _worktree_pruefen(repo_klon, wt_ordner, branch)
    git_lauf(repo_klon, "fetch", "--prune", "origin")
    remote = f"origin/{branch}"
    remote_da = _ref_da(repo_klon, f"refs/remotes/{remote}")
    if not wt_ordner.exists():
        wt_ordner.parent.mkdir(parents=True, exist_ok=True)
        git_lauf(repo_klon, "worktree", "prune")
        if _ref_da(repo_klon, f"refs/heads/{branch}"):
            git_lauf(repo_klon, "worktree", "add", str(wt_ordner), branch)
        else:
            start = remote if remote_da else f"origin/{basis}"
            git_lauf(repo_klon, "worktree", "add", "--no-track", "-b", branch, str(wt_ordner), start)
            return wt_ordner
    if not remote_da:
        return wt_ordner
    erg = git_lauf(wt_ordner, "merge", "--ff-only", remote, check=False)
    if erg.returncode != 0:
        if not ist_vorfahr(wt_ordner, "HEAD", remote) and not ist_vorfahr(
            wt_ordner, remote, "HEAD"
        ):
            raise KonfliktFehler(branch)
        raise GitFehler("merge", erg.returncode)
    return wt_ordner


def voraus(wt: Path, branch: str, *, basis: str = "master") -> int:
    """Zahl der Commits auf HEAD, die nicht auf `origin/<branch>` liegen.

    Gibt es `origin/<branch>` nicht, gilt `origin/<basis>` als Gegenseite.
    """
    ref = f"origin/{branch}"
    if not _ref_da(wt, f"refs/remotes/{ref}"):
        ref = f"origin/{basis}"
    erg = git_lauf(wt, "rev-list", "--count", "HEAD", f"^{ref}")
    return int(erg.stdout.strip())


def aufraeumen(repo_klon: Path, wt: Path, branch: str) -> None:
    """Entfernt Worktree und lokalen Zweig des Helfers (nur `debugger/*`)."""
    _debugger_zweig_pruefen(branch)
    if wt.exists():
        _worktree_pruefen(repo_klon, wt, branch)
        git_lauf(repo_klon, "worktree", "remove", "--force", str(wt))
    git_lauf(repo_klon, "worktree", "prune")
    if _ref_da(repo_klon, f"refs/heads/{branch}"):
        git_lauf(repo_klon, "branch", "-D", branch)
