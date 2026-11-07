"""Nachlauf des Debugger-Claw: offene Helfer-PRs prüfen, Pin heben, aufräumen.

Für jeden Fund mit offenem Helfer-PR (`zustand == "pr"`) liest der Nachlauf den PR. Nach einem
Merge in `Flissel/vibemind-os` hebt er per eigenem PR den Submodul-Pin im äußeren Repo
(`Fixes #<n>`), aber nie rückwärts; nach Merge oder Schließen räumt er den eigenen
`debugger/*`-Branch samt Worktree auf. Andere Branches fasst er nie an; der Branch-Name kommt
immer aus dem `FundStand`, nie aus PR-Daten. `stand` wird nur im Speicher geändert, speichern
muss der Aufrufer.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Callable

from selbstbild.debugger.klon import (
    ZU_GROSS,
    GitFehler,
    KonfliktFehler,
    PushZuGross,
    WorktreeFehler,
    aufraeumen,
    git_lauf,
    ist_vorfahr,
    push_inhalt,
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
    repo_klon_pfad,
    wt_pfad,
    zurueckgehalten,
)
from selbstbild.debugger.spiegel import ZurueckgehaltenFehler
from selbstbild.debugger.stand import DebuggerStand, FundStand
from selbstbild.funde.github import GitHubFehler
from selbstbild.funde.text import pruefen

OS_REPO = "Flissel/vibemind-os"
_SHA = re.compile(r"[0-9a-f]{40}")
_FEHLER = (GitHubFehler, GitFehler, KonfliktFehler, WorktreeFehler, ValueError, KeyError, RuntimeError, OSError)


class _PinWartet(Exception):
    """Der Pin kann (noch) nicht gesetzt werden; trägt nur einen kurzen Grund."""


def pin_branch(issue: int) -> str:
    """Branch des Pin-PR im äußeren Repo: `debugger/<issue>-pin`."""
    return f"debugger/{issue}-pin"


def _ist_commit(ort: Path, sha: str) -> bool:
    art = git_lauf(ort, "cat-file", "-t", sha, check=False)
    return art.returncode == 0 and art.stdout.strip() == "commit"


def _pin_pruefen(os_klon: Path, sha: str, basis: str) -> None:
    """R9: `sha` ist ein Commit, liegt auf `origin/<basis>`, und der Fetch ist aktuell."""
    git_lauf(os_klon, "fetch", "origin")
    if not _ist_commit(os_klon, sha):
        raise _PinWartet("kein_commit")
    if not ist_vorfahr(os_klon, sha, f"origin/{basis}"):
        raise _PinWartet("nicht_auf_basis")
    entfernt = git_lauf(os_klon, "ls-remote", "origin", f"refs/heads/{basis}").stdout.split()
    lokal = git_lauf(os_klon, "rev-parse", "--verify", "-q", f"refs/remotes/origin/{basis}").stdout.strip()
    if not entfernt or entfernt[0] != lokal:
        raise _PinWartet("basis_veraltet")


def _schon_enthalten(os_klon: Path, sha: str, aktuell: str | None) -> bool:
    """True, wenn der aktuelle Pin `sha` schon enthält; False, wenn `sha` ihn vorwärts hebt.

    Sonst (kein Pin, unbekannter Pin, auseinandergelaufen) `_PinWartet("nicht_linear")`:
    so hebt der Nachlauf nie rückwärts.
    """
    if aktuell == sha:
        return True
    if aktuell is None or not _ist_commit(os_klon, aktuell):
        raise _PinWartet("nicht_linear")
    if ist_vorfahr(os_klon, sha, aktuell):
        return True
    if ist_vorfahr(os_klon, aktuell, sha):
        return False
    raise _PinWartet("nicht_linear")


def _gitlink(ort: Path, pfad: str, ref: str = "HEAD") -> str | None:
    """SHA des Gitlinks `pfad` in `ref`, sonst None."""
    zeile = git_lauf(ort, "ls-tree", ref, "--", pfad).stdout.strip()
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

    # --- Hilfen ---

    def aufraeumen(self, repo: str, branch: str) -> None:
        """Lokal Worktree und Zweig, dann entfernt den Branch; nur `debugger/*`, 404 zählt als erledigt."""
        aufraeumen(repo_klon_pfad(self.konfig, self.klon, repo), wt_pfad(self.ordner, repo, branch), branch)
        self.w.gh_lab.branch_loeschen(repo, branch)

    def melden(self, n: int, text: str) -> None:
        """Kommentar ins Issue nach der Schlüssel-Prüfung; bei Treffer nur der Rückhalte-Satz."""
        muster = pruefen(text)
        self.w.gh_lab.kommentieren(self.konfig.issue_repo, n, text if muster is None else zurueckgehalten(muster))

    def einmal(self, schluessel: str, senden: Callable[[], None]) -> None:
        """Führt `senden` je Schlüssel nur einmal aus (Schlüssel in `stand.gemeldet`)."""
        if schluessel in self.stand.gemeldet:
            return
        senden()
        self.stand.gemeldet.append(schluessel)
        del self.stand.gemeldet[:-MAX_GEMELDET]

    # --- Ablauf ---

    def fund(self, n: int, fund: FundStand, nummer: int) -> None:
        pr = self.w.gh_lesen.pr_lesen(fund.repo, nummer)
        issue_repo = self.konfig.issue_repo
        if pr.get("merged") is True:
            if fund.repo == OS_REPO and fund.repo != issue_repo:
                self.pin(n, fund, nummer, pr.get("merge_commit_sha"))
            elif fund.repo == issue_repo:
                self.aufraeumen(fund.repo, fund.branch)
                self.stand.funde[str(n)] = replace(fund, zustand="erledigt")
                self.zeilen.append(f"#{n}: erledigt ({fund.repo}#{nummer} gemergt)")
            else:
                raise ValueError("repo_nicht_freigegeben")
        elif pr.get("state") == "closed":
            self.aufraeumen(fund.repo, fund.branch)
            if fund.repo == issue_repo and fund.branch == pin_branch(n):
                # R25: der Fix selbst ist in vibemind-os gemergt, kein neuer Fix-Auftrag auf dem Pin-Branch
                self.stand.funde[str(n)] = replace(fund, zustand="erledigt")
                self.zeilen.append(f"#{n}: Pin-PR {fund.repo}#{nummer} ohne Merge geschlossen, erledigt")
            else:
                self.stand.funde[str(n)] = replace(fund, zustand="offen", pr=None)
                self.zeilen.append(f"#{n}: {fund.repo}#{nummer} ohne Merge geschlossen, aufgeräumt")

    def pin(self, n: int, fund: FundStand, os_pr: int, sha: object) -> None:
        """Hebt den Gitlink im äußeren Repo per eigenem PR vorwärts auf den Merge-Commit (R9)."""
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError("merge_commit_sha_ungueltig")
        konfig, basis, issue_repo = self.konfig, self.konfig.basis, self.konfig.issue_repo
        sub_pfad = pfad_norm(konfig.repos[OS_REPO])
        aussen = repo_klon_pfad(konfig, self.klon, issue_repo)
        os_klon = repo_klon_pfad(konfig, self.klon, OS_REPO)
        # (1) eigener vibemind-os-Branch ist gemergt: aufräumen
        self.aufraeumen(fund.repo, fund.branch)
        # (2) Pin-Worktree im äußeren Klon aus origin/<basis>
        branch = pin_branch(n)
        wt = wt_pfad(self.ordner, issue_repo, branch)
        neu_angelegt = not wt.exists()
        worktree_bereit(aussen, wt, branch, basis)
        # (3) Prüfungen im Submodul-Klon, dann Richtung gegen den Pin auf origin/<basis> (nie rückwärts)
        try:
            _pin_pruefen(os_klon, sha, basis)
            enthalten = _schon_enthalten(os_klon, sha, _gitlink(aussen, sub_pfad, f"origin/{basis}"))
        except _PinWartet as e:
            if neu_angelegt:  # vor jedem Commit gescheitert: der neue Worktree bleibt nicht liegen
                aufraeumen(aussen, wt, branch)
            self.warten(n, str(e))
            return
        if enthalten:  # R25
            self.aufraeumen(issue_repo, branch)
            name = issue_repo.split("/")[1]
            self.melden(n, f"{PRAEFIX}Der Fix aus {fund.repo}#{os_pr} ist in {name} schon enthalten.")
            self.stand.funde[str(n)] = replace(fund, zustand="erledigt")
            self.zeilen.append(f"#{n}: Fix schon in {name} enthalten, erledigt")
            return
        # (4) Gitlink setzen und committen (ohne Pfadangabe), danach im Baum prüfen
        sha7 = sha[:7]
        if _gitlink(wt, sub_pfad) != sha:
            git_lauf(wt, "update-index", "--cacheinfo", f"160000,{sha},{sub_pfad}")
            git_lauf(wt, "commit", "-q", "-m", f"chore(vibemind-os): Pin auf {sha7} (Debugger-Claw #{n})")
        if _gitlink(wt, sub_pfad) != sha:
            raise ValueError("gitlink_falsch")
        # (5) Schlüssel-Prüfung wie in der Runde
        titel = f"{PRAEFIX}vibemind-os auf {sha7} (#{n})"
        koerper = f"Fixes #{n}\n\n- Pin vibemind-os auf {sha} (gemergt in {fund.repo}#{os_pr})"
        neu = voraus(wt, branch, basis=basis)
        try:
            texte = (titel, koerper, push_inhalt(wt, branch, basis) if neu > 0 else "")
            muster = next((m for m in map(pruefen, texte) if m is not None), None)
        except PushZuGross:  # R23: fail-closed
            muster = ZU_GROSS
        if muster is not None:
            self.gehalten(n, muster)
            return
        # (6) Push (maßgebliche Prüfung im Spiegel, R29), (7) PR, (8) FundStand ersetzen
        if neu > 0:
            try:
                self.w.push(wt, branch)
            except ZurueckgehaltenFehler as e:
                self.gehalten(n, e.muster)
                return
        nummer = pr_finden_oder_anlegen(self.w, issue_repo, branch, basis, titel, koerper)
        self.stand.funde[str(n)] = FundStand(repo=issue_repo, branch=branch, sitzung=fund.sitzung, pr=nummer,
                                             zustand="pr", versuche=fund.versuche)
        self.zeilen.append(f"#{n}: Pin-PR #{nummer} geöffnet")

    def gehalten(self, n: int, muster: str) -> None:
        """Schlüssel-Verdacht vor dem Pin-Push: je Fund nur ein Rückhalte-Kommentar."""
        satz = zurueckgehalten(muster)
        issue_repo = self.konfig.issue_repo
        self.einmal(f"pin_wartet:{n}:schluessel", lambda: self.w.gh_lab.kommentieren(issue_repo, n, satz))
        self.zeilen.append(f"#{n}: Pin zurückgehalten ({muster})")

    def warten(self, n: int, grund: str) -> None:
        """R26: der Fund bleibt und wird erneut versucht; ins Issue je Grund nur ein Kommentar."""
        self.einmal(f"pin_wartet:{n}:{grund}",
                    lambda: self.melden(n, f"{PRAEFIX}Pin für vibemind-os wartet ({grund})."))
        self.zeilen.append(f"#{n}: Pin wartet ({grund})")


def nachlauf(konfig: DebuggerKonfig, stand: DebuggerStand, klon: Path, ordner: Path, w: Werkzeuge) -> list[str]:
    """Prüft alle Funde mit offenem Helfer-PR und gibt kurze Log-Zeilen zurück.

    Gemergt in `Flissel/vibemind-os`: aufräumen, Pin-PR im `issue_repo` (`Fixes #<n>`), der Fund
    zeigt danach auf diesen PR; enthält der Pin den Fix schon, ist der Fund erledigt. Gemergt im
    `issue_repo`: `erledigt`, aufräumen. Ohne Merge geschlossen: aufräumen, dann `offen` mit
    `pr=None`, beim Pin-PR `erledigt`. Fehler je Fund ergeben `#<n>: fehler <Typ>`, der Fund
    bleibt dann unverändert und die übrigen laufen weiter.
    """
    return _Nachlauf(konfig, stand, klon, ordner, w).lauf()
