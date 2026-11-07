"""Ziel eines Fundes: Repo (Ruling R1), Pfade im Klon und für Worktrees, Branch je Runde (R37, R38).

Aus `runde.py` ausgelagert; `runde` reicht die Namen weiter, damit Nachlauf und Wächter
dieselben Regeln benutzen.
"""
from __future__ import annotations

import re
from pathlib import Path

from selbstbild.debugger.klon import PRAEFIX as BRANCH_PRAEFIX
from selbstbild.debugger.klon import branch_name
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.stand import FundStand

OEFFENTLICH_KURZ = "fix"  # R38: Branch im öffentlichen Repo ohne Titelteil

_REPO_MARKE = re.compile(r"<!--\s*selbstbild:repo\s*=\s*([^\s>][^>]*?)\s*-->")
_SAMMEL_MARKE = re.compile(r"<!--\s*selbstbild:sammel\s*=\s*[a-z_]+:([^\s>][^>]*?)\s*-->")
_TITEL_KLAMMER = re.compile(r"\(([^()]+)\)\s*$")
_TITEL_STRICH = re.compile(r"\s–\s(\S+)\s*$")


def _code_repo(issue_text: str, issue_titel: str) -> str | None:
    for muster in (_REPO_MARKE, _SAMMEL_MARKE):
        if m := muster.search(issue_text):
            return m.group(1).strip()
    for muster in (_TITEL_KLAMMER, _TITEL_STRICH):
        if m := muster.search(issue_titel):
            return m.group(1).strip()
    return None


def pfad_norm(pfad: str) -> str:
    p = pfad.replace("\\", "/").strip().strip("/")
    while p.startswith("./"):
        p = p[2:]
    return p or "."


def ziel_bestimmen(issue_text: str, issue_titel: str, konfig: DebuggerKonfig) -> tuple[str, bool]:
    """`(repo, nur_einordnen)` aus Repo-Marke, Sammel-Marke oder Titel-Endung des Issues.

    Das erste Pfadsegment (Wurzelname) fällt weg, der Rest muss genau ein Pfad aus
    `konfig.repos` sein. Kein Treffer: `issue_repo`, nur einordnen. Kein code_repo: `issue_repo`.
    """
    code_repo = _code_repo(issue_text, issue_titel)
    if not code_repo:
        return konfig.issue_repo, False
    pfad = pfad_norm(code_repo.partition("/")[2])
    for repo, ort in konfig.repos.items():
        if pfad_norm(ort) == pfad:
            return repo, False
    return konfig.issue_repo, True


def fund_neu(n: int, titel: str, repo: str, konfig: DebuggerKonfig, *, runde: int = 1,
             versuche: dict[str, int] | None = None) -> FundStand:
    """Frischer FundStand ohne PR und Sitzung (R37); im öffentlichen Repo `debugger/<n>-fix[-<runde>]` (R38)."""
    kurz = titel if repo == konfig.issue_repo else OEFFENTLICH_KURZ
    return FundStand(repo=repo, branch=branch_name(n, kurz, runde), runde=runde, versuche=dict(versuche or {}))


def repo_klon_pfad(konfig: DebuggerKonfig, klon: Path, repo: str) -> Path:
    """Ort des Repos im Klon (`.` ist der Klon selbst); `KeyError` bei nicht freigegebenem Repo."""
    ort = pfad_norm(konfig.repos[repo])
    return klon if ort == "." else klon / ort


def wt_pfad(ordner: Path, repo: str, branch: str) -> Path:
    """Worktree eines Branches außerhalb des Klons (R6): `ordner/wt/<repo_name>-<branch ohne debugger/>`.

    Einzige Pfadregel für Runde und Nachlauf; Zeichen außerhalb `[A-Za-z0-9-]` werden zu `-`.
    """
    kurz = re.sub(r"[^A-Za-z0-9-]", "-", branch.removeprefix(BRANCH_PRAEFIX)) or "fund"
    return ordner / "wt" / f"{repo.split('/')[1]}-{kurz}"
