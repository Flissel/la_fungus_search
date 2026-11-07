"""Nachlauf des Debugger-Claw: offene Helfer-PRs prüfen, Pin heben, aufräumen.

Für jeden Fund mit offenem Helfer-PR (`zustand == "pr"`) liest der Nachlauf den PR. Nach einem
Merge in `Flissel/vibemind-os` hebt er per eigenem PR den Submodul-Pin im äußeren Repo
(`Fixes #<n>`); nach Merge oder Schließen räumt er den eigenen `debugger/*`-Branch samt Worktree
auf. Andere Branches fasst er nie an; der Branch-Name kommt immer aus dem `FundStand`, nie aus
PR-Daten. `stand` wird nur im Speicher geändert, speichern muss der Aufrufer.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from selbstbild.debugger.klon import (
    GitFehler,
    KonfliktFehler,
    WorktreeFehler,
    aufraeumen,
    git_lauf,
    ist_vorfahr,
    voraus,
    worktree_bereit,
)
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.runde import (
    MAX_GEMELDET,
    PRAEFIX,
    Werkzeuge,
    pfad_norm,
    pr_finden_oder_anlegen,
    push_inhalt,
    repo_klon_pfad,
    wt_pfad,
    zurueckgehalten,
)
from selbstbild.debugger.stand import DebuggerStand, FundStand
from selbstbild.funde.github import GitHubFehler
from selbstbild.funde.text import pruefen

OS_REPO = "Flissel/vibemind-os"
PIN_ZUSATZ = "-pin"
_SHA = re.compile(r"[0-9a-f]{40}")
_FEHLER = (GitHubFehler, GitFehler, KonfliktFehler, WorktreeFehler, ValueError, KeyError, RuntimeError, OSError)


class _PinPruefung(Exception):
    """Eine Pin-Prüfung (R9) ist gescheitert; trägt nur einen kurzen Grund."""


def pin_branch(issue: int) -> str:
    """Branch des Pin-PR im äußeren Repo: `debugger/<issue>-pin`."""
    return f"debugger/{issue}{PIN_ZUSATZ}"


def _fund_wt(ordner: Path, fund: FundStand, n: int) -> Path:
    """Worktree des Fundes (R6); der Pin-Branch liegt in `<repo_name>-<n>-pin`."""
    zusatz = PIN_ZUSATZ if fund.branch == pin_branch(n) else ""
    return wt_pfad(ordner, fund.repo, n, zusatz)


def _aufraeumen(konfig: DebuggerKonfig, klon: Path, ordner: Path, fund: FundStand, n: int, w: Werkzeuge) -> None:
    """Lokal Worktree und Zweig, dann entfernt den Branch; beides nur für `fund.branch` (`debugger/*`)."""
    aufraeumen(repo_klon_pfad(konfig, klon, fund.repo), _fund_wt(ordner, fund, n), fund.branch)
    w.gh_lab.branch_loeschen(fund.repo, fund.branch)


def _pin_pruefen(os_klon: Path, sha: str, basis: str) -> None:
    """R9: `sha` ist ein Commit, liegt auf `origin/<basis>`, und der Fetch ist aktuell."""
    git_lauf(os_klon, "fetch", "origin")
    art = git_lauf(os_klon, "cat-file", "-t", sha, check=False)
    if art.returncode != 0 or art.stdout.strip() != "commit":
        raise _PinPruefung("kein_commit")
    if not ist_vorfahr(os_klon, sha, f"origin/{basis}"):
        raise _PinPruefung("nicht_auf_basis")
    entfernt = git_lauf(os_klon, "ls-remote", "origin", f"refs/heads/{basis}").stdout.split()
    lokal = git_lauf(os_klon, "rev-parse", "--verify", "-q", f"refs/remotes/origin/{basis}").stdout.strip()
    if not entfernt or entfernt[0] != lokal:
        raise _PinPruefung("basis_veraltet")


def _gitlink(wt: Path, pfad: str) -> str | None:
    """SHA des Gitlinks `pfad` in HEAD, sonst None."""
    zeile = git_lauf(wt, "ls-tree", "HEAD", "--", pfad).stdout.strip()
    kopf, _, name = zeile.partition("\t")
    teile = kopf.split(" ")
    if name != pfad or len(teile) != 3 or teile[:2] != ["160000", "commit"]:
        return None
    return teile[2]


class _Nachlauf:
    """Zustand eines Nachlauf-Durchgangs."""

    def __init__(self, konfig: DebuggerKonfig, stand: DebuggerStand, klon: Path, ordner: Path,
                 w: Werkzeuge) -> None:
        self.konfig, self.stand, self.klon, self.ordner, self.w = konfig, stand, klon, ordner, w
        self.zeilen: list[str] = []

    def lauf(self) -> list[str]:
        for schluessel, fund in list(self.stand.funde.items()):
            if fund.zustand != "pr" or fund.pr is None:
                continue
            try:
                self.fund(int(schluessel), fund, fund.pr)
            except _FEHLER as e:
                self.zeilen.append(f"#{schluessel}: fehler {type(e).__name__}")
        return self.zeilen

    def fund(self, n: int, fund: FundStand, nummer: int) -> None:
        pr = self.w.gh_lesen.pr_lesen(fund.repo, nummer)
        issue_repo = self.konfig.issue_repo
        if pr.get("merged") is True:
            if fund.repo == OS_REPO and fund.repo != issue_repo:
                self.pin(n, fund, nummer, pr.get("merge_commit_sha"))
            elif fund.repo == issue_repo:
                _aufraeumen(self.konfig, self.klon, self.ordner, fund, n, self.w)
                self.stand.funde[str(n)] = replace(fund, zustand="erledigt")
                self.zeilen.append(f"#{n}: erledigt ({fund.repo}#{nummer} gemergt)")
            else:
                raise ValueError("repo_nicht_freigegeben")
        elif pr.get("state") == "closed":
            _aufraeumen(self.konfig, self.klon, self.ordner, fund, n, self.w)
            self.stand.funde[str(n)] = replace(fund, zustand="offen", pr=None)
            self.zeilen.append(f"#{n}: {fund.repo}#{nummer} ohne Merge geschlossen, aufgeräumt")

    def pin(self, n: int, fund: FundStand, os_pr: int, sha: object) -> None:
        """Hebt den Gitlink im äußeren Repo per eigenem PR auf den Merge-Commit (R9)."""
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError("merge_commit_sha_ungueltig")
        konfig, basis, issue_repo = self.konfig, self.konfig.basis, self.konfig.issue_repo
        sub_pfad = pfad_norm(konfig.repos[OS_REPO])
        # (1) eigener vibemind-os-Branch ist gemergt: aufräumen
        _aufraeumen(konfig, self.klon, self.ordner, fund, n, self.w)
        # (2) Pin-Worktree im äußeren Klon aus origin/<basis>
        branch = pin_branch(n)
        wt = wt_pfad(self.ordner, issue_repo, n, PIN_ZUSATZ)
        worktree_bereit(repo_klon_pfad(konfig, self.klon, issue_repo), wt, branch, basis)
        # (3) Prüfungen im Submodul-Klon
        try:
            _pin_pruefen(repo_klon_pfad(konfig, self.klon, OS_REPO), sha, basis)
        except _PinPruefung as e:
            self.zeilen.append(f"#{n}: Pin-Prüfung fehlgeschlagen ({e})")
            return
        # (4) Gitlink setzen und committen (ohne Pfadangabe), danach im Baum prüfen
        sha7 = sha[:7]
        if _gitlink(wt, sub_pfad) != sha:
            git_lauf(wt, "update-index", "--cacheinfo", f"160000,{sha},{sub_pfad}")
            git_lauf(wt, "commit", "-q", "-m", f"chore(vibemind-os): Pin auf {sha7} (Debugger-Claw #{n})")
        if _gitlink(wt, sub_pfad) != sha:
            raise ValueError("gitlink_falsch")
        if int(git_lauf(wt, "rev-list", "--count", "HEAD", f"^origin/{basis}").stdout.strip()) == 0:
            aufraeumen(repo_klon_pfad(konfig, self.klon, issue_repo), wt, branch)
            self.stand.funde[str(n)] = replace(fund, zustand="erledigt")
            self.zeilen.append(f"#{n}: Pin schon auf {basis}")
            return
        # (5) Schlüssel-Prüfung wie in der Runde
        titel = f"{PRAEFIX}vibemind-os auf {sha7} (#{n})"
        koerper = f"Fixes #{n}\n\n- Pin vibemind-os auf {sha} (gemergt in {fund.repo}#{os_pr})"
        neu = voraus(wt, branch, basis=basis)
        inhalt = push_inhalt(wt, branch, basis) if neu > 0 else ""
        for text in (titel, koerper, inhalt):
            if (muster := pruefen(text)) is not None:
                self.halten(n, sha, muster)
                return
        # (6) Push, (7) PR, (8) FundStand ersetzen
        if neu > 0:
            self.w.push(wt, branch)
        nummer = pr_finden_oder_anlegen(self.w, issue_repo, branch, basis, titel, koerper)
        self.stand.funde[str(n)] = FundStand(repo=issue_repo, branch=branch, sitzung=fund.sitzung, pr=nummer,
                                             zustand="pr", versuche=fund.versuche)
        self.zeilen.append(f"#{n}: Pin-PR #{nummer} geöffnet")

    def halten(self, n: int, sha: str, muster: str) -> None:
        """Rückhalte-Satz ins Issue, je Fund und Merge-Commit nur einmal (Schlüssel in `stand.gemeldet`)."""
        schluessel = f"pin_gehalten:{n}:{sha}"
        if schluessel not in self.stand.gemeldet:
            self.w.gh_lab.kommentieren(self.konfig.issue_repo, n, zurueckgehalten(muster))
            self.stand.gemeldet.append(schluessel)
            del self.stand.gemeldet[:-MAX_GEMELDET]
        self.zeilen.append(f"#{n}: Pin zurückgehalten ({muster})")


def nachlauf(konfig: DebuggerKonfig, stand: DebuggerStand, klon: Path, ordner: Path, w: Werkzeuge) -> list[str]:
    """Prüft alle Funde mit offenem Helfer-PR und gibt kurze Log-Zeilen zurück.

    Gemergt in `Flissel/vibemind-os`: aufräumen, Pin-PR im `issue_repo` (`Fixes #<n>`), der Fund
    zeigt danach auf diesen PR. Gemergt im `issue_repo`: `erledigt`, aufräumen. Ohne Merge
    geschlossen: aufräumen, `offen`, `pr=None`. Fehler je Fund ergeben `#<n>: fehler <Typ>`,
    der Fund bleibt dann unverändert und die übrigen laufen weiter.
    """
    return _Nachlauf(konfig, stand, klon, ordner, w).lauf()
