"""Eine Runde des Debugger-Claw: vom Auftrag zur Antwort, zum Fix-PR oder zur Einordnung.

Jeder Text an GitHub läuft vorher durch `pruefen`; geschrieben wird nur über `gh_lab`
(das die Debugger-Marke selbst anhängt). Den Stand verändert die Runde nur im Speicher,
speichern muss der Aufrufer.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from selbstbild.debugger.auftraege import Auftrag, Kommentar
from selbstbild.debugger.budget import budget_buchen, budget_pruefen
from selbstbild.debugger.helfer import HelferErgebnis, HelferFehler, prompt_bauen
from selbstbild.debugger.klon import (
    GIT_TIMEOUT_S,
    TIMEOUT_CODE,
    TITEL_PRAEFIX,
    GitFehler,
    KonfliktFehler,
    WorktreeFehler,
    branch_name,
    voraus,
    worktree_bereit,
)
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.stand import DebuggerStand, FundStand
from selbstbild.funde.github import GitHub, GitHubFehler
from selbstbild.funde.text import pruefen

PRAEFIX = "🤖 Debugger-Claw: "
KEINE_BEGRUENDUNG = "(keine Begründung vom Helfer)"
MAX_PR_TITEL = 256
MAX_SEITEN = 10
SEITE = 100
RUECKFRAGE_LABEL = "selbstbild:rueckfrage"
MAX_DIFF = 60_000
MAX_VERSUCHE = 3
MAX_GEMELDET = 200
NUR_EINORDNEN_TEXT = PRAEFIX + "Fixes in diesem Repo sind nicht freigegeben, ich kann nur einordnen."
HINWEIS_OEFFENTLICH = (
    "Dieses Repo ist öffentlich. Commit-Texte beschreiben nur die Code-Änderung, ohne Inhalte aus dem Issue."
)
HINWEIS_NUR_EINORDNEN = "Fixes in diesem Repo sind nicht freigegeben. Ändere keine Dateien, ordne nur ein."

_REPO_MARKE = re.compile(r"<!--\s*selbstbild:repo\s*=\s*([^\s>][^>]*?)\s*-->")
_SAMMEL_MARKE = re.compile(r"<!--\s*selbstbild:sammel\s*=\s*[a-z_]+:([^\s>][^>]*?)\s*-->")
_TITEL_KLAMMER = re.compile(r"\(([^()]+)\)\s*$")
_TITEL_STRICH = re.compile(r"\s–\s(\S+)\s*$")
_PUSH_BRANCH = re.compile(r"debugger/[A-Za-z0-9][A-Za-z0-9._/-]*")


@dataclass
class Werkzeuge:
    """Alles, was eine Runde von außen braucht (für Tests austauschbar)."""

    gh_lab: GitHub  # Konto Vibemind-LAB, marke=DEBUGGER_MARKE
    gh_lesen: GitHub
    helfer: Callable[..., HelferErgebnis]  # wie helfer_starten(wt, prompt, sitzung, frist_s)
    frei_gb: Callable[[], float | None]
    uhr: Callable[[], datetime]
    push: Callable[[Path, str], None]


# ---------- Ziel-Repo (Ruling R1) ----------

def _code_repo(issue_text: str, issue_titel: str) -> str | None:
    for muster in (_REPO_MARKE, _SAMMEL_MARKE):
        if m := muster.search(issue_text):
            return m.group(1).strip()
    for muster in (_TITEL_KLAMMER, _TITEL_STRICH):
        if m := muster.search(issue_titel):
            return m.group(1).strip()
    return None


def _pfad_norm(pfad: str) -> str:
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
    pfad = _pfad_norm(code_repo.partition("/")[2])
    for repo, ort in konfig.repos.items():
        if _pfad_norm(ort) == pfad:
            return repo, False
    return konfig.issue_repo, True


# ---------- Git ----------

def push_standard(
    wt: Path,
    branch: str,
    *,
    askpass: Path | None = None,
    laufzeit: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """`git push origin HEAD:refs/heads/<branch>` mit `GIT_ASKPASS` (Token nur in der Umgebung).

    Nur Branches `debugger/...`. Ohne `askpass` gilt `DEBUGGER_ASKPASS`; fehlt beides,
    `RuntimeError("kein_askpass")`. Fehler nennen nur Unterbefehl und Code.
    """
    if not _PUSH_BRANCH.fullmatch(branch) or ".." in branch or branch.endswith("/"):
        raise ValueError("kein_debugger_zweig")
    skript = str(askpass) if askpass is not None else os.environ.get("DEBUGGER_ASKPASS", "")
    if not skript:
        raise RuntimeError("kein_askpass")
    env = dict(os.environ)
    env.update({"GIT_ASKPASS": skript, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"})
    argv = ["git", "-C", str(wt), "-c", "credential.helper=", "push", "origin", f"HEAD:refs/heads/{branch}"]
    try:
        erg = laufzeit(argv, check=False, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       env=env, timeout=GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise GitFehler("push", TIMEOUT_CODE) from None
    if erg.returncode != 0:
        raise GitFehler("push", erg.returncode)


def _git_text(wt: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """`git -C <wt> ...` als UTF-8-Text (ohne Shell, mit Timeout, ohne Prompt, nie stderr im Fehler)."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        erg = subprocess.run(
            ["git", "-C", str(wt), *args], check=False, capture_output=True,
            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, env=env, timeout=GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise GitFehler(args[0], TIMEOUT_CODE) from None
    if check and erg.returncode != 0:
        raise GitFehler(args[0], erg.returncode)
    return erg


def _betreffzeilen(wt: Path, gegen: str) -> list[str]:
    """Betreffzeilen von `<gegen>..HEAD`, älteste zuerst."""
    aus = _git_text(wt, "log", "--format=%s", f"{gegen}..HEAD").stdout
    return [z for z in reversed(aus.splitlines()) if z.strip()]


def _push_inhalt(wt: Path, branch: str, basis: str) -> str:
    """Volles `git log -p` (Nachrichten und Diffs) dessen, was ein Push nach `branch` hochlädt (R21)."""
    remote = f"refs/remotes/origin/{branch}"
    da = _git_text(wt, "rev-parse", "--verify", "-q", remote, check=False).returncode == 0
    gegen = f"origin/{branch}" if da else f"origin/{basis}"
    return _git_text(wt, "log", "-p", "--no-color", "--no-ext-diff", "--no-textconv", f"{gegen}..HEAD").stdout


# ---------- Runde ----------

def _zurueckgehalten(muster: str) -> str:
    return f"{PRAEFIX}Antwort zurückgehalten (Schlüssel-Verdacht: {muster})."


def _liste(daten: Any) -> list[dict[str, Any]]:
    return [x for x in daten if isinstance(x, dict)] if isinstance(daten, list) else []


def _autor(roh: dict[str, Any]) -> str:
    user = roh.get("user")
    return str(user.get("login", "")) if isinstance(user, dict) else ""


def _begruendung(text: str) -> str:
    return text.strip() or KEINE_BEGRUENDUNG


def _kappen(titel: str) -> str:
    return titel if len(titel) <= MAX_PR_TITEL else titel[: MAX_PR_TITEL - 1] + "…"


def _seiten(gh: GitHub, pfad: str) -> list[dict[str, Any]]:
    """Liest `pfad` seitenweise (100 je Seite) bis zur ersten kurzen Seite, höchstens 10 Seiten."""
    alle: list[dict[str, Any]] = []
    for seite in range(1, MAX_SEITEN + 1):
        teil = gh._api("GET", f"{pfad}?per_page={SEITE}&page={seite}")
        alle += _liste(teil)
        if not isinstance(teil, list) or len(teil) < SEITE:
            break
    return alle


class _Lauf:
    """Zustand einer laufenden Runde (ein Auftrag)."""

    def __init__(self, auftrag: Auftrag, konfig: DebuggerKonfig, stand: DebuggerStand, klon: Path,
                 ordner: Path, w: Werkzeuge) -> None:
        self.auftrag, self.konfig, self.stand = auftrag, konfig, stand
        self.klon, self.ordner, self.w = klon, ordner, w
        self.n = auftrag.issue
        self.ids = [k.id for k in auftrag.kommentare]
        self.zurueckgehalten = False

    # --- Senden ---

    def senden(self, repo: str, nummer: int, text: str) -> bool:
        """Kommentiert nach der Schlüssel-Prüfung; bei Treffer nur der Rückhalte-Satz ins Issue."""
        muster = pruefen(text)
        if muster is not None:
            self.halten(muster)
            return False
        self.w.gh_lab.kommentieren(repo, nummer, text)
        return True

    def halten(self, muster: str) -> None:
        self.zurueckgehalten = True
        self.w.gh_lab.kommentieren(self.konfig.issue_repo, self.n, _zurueckgehalten(muster))

    def melden(self, text: str) -> None:
        self.senden(self.konfig.issue_repo, self.n, text)

    def antworten(self, fund: FundStand, text: str) -> None:
        """In den PR nur bei PR im `issue_repo` und reinem PR-Auftrag (R17), sonst ins Issue."""
        nur_pr = fund.pr is not None and all(
            (k.repo, k.nummer) == (fund.repo, fund.pr) for k in self.auftrag.kommentare
        )
        if nur_pr and fund.repo == self.konfig.issue_repo and fund.pr is not None:
            self.senden(fund.repo, fund.pr, text)
        else:
            self.melden(text)

    # --- Abschluss ---

    def fertig(self, fund: FundStand | None, ergebnis: str = "ok") -> str:
        self.stand.verarbeitet_hinzu(self.ids)
        if fund is not None:
            for i in self.ids:
                fund.versuche.pop(str(i), None)
        if ergebnis == "ok" and self.zurueckgehalten:
            return "abgelehnt:schluessel"
        return ergebnis

    def fehlschlag(self, fund: FundStand, art: str) -> str:
        """Zählt einen Versuch je Kommentar; ab 3 Versuchen gilt ein Kommentar als verarbeitet."""
        for i in self.ids:
            fund.versuche[str(i)] = fund.versuche.get(str(i), 0) + 1
        aufgegeben = [i for i in self.ids if fund.versuche[str(i)] >= MAX_VERSUCHE]
        if aufgegeben:
            self.stand.verarbeitet_hinzu(aufgegeben)
            for i in aufgegeben:
                fund.versuche.pop(str(i), None)
            text = f"{PRAEFIX}Aufgegeben nach {MAX_VERSUCHE} Versuchen ({art})."
        else:
            text = f"{PRAEFIX}Runde fehlgeschlagen ({art}), ich versuche es erneut."
        try:  # Stand ist schon fortgeschrieben; die Meldung ist nur ein Hinweis
            self.melden(text)
        except GitHubFehler:
            pass
        return f"fehler:{art}"

    # --- Material ---

    def auftrag_texte(self) -> list[str]:
        return [self._auftrag_text(k) for k in self.auftrag.kommentare]

    @staticmethod
    def _auftrag_text(k: Kommentar) -> str:
        if not k.pfad:
            return k.text
        ort = f"{k.pfad}:{k.zeile}" if k.zeile is not None else k.pfad
        return f"[{ort}] {k.text}"

    def _kommentar(self, c: dict[str, Any], ort: str = "") -> str:
        """Kommentar fürs Material; nur Auslöser und Helfer-Konto, alle anderen als Platzhalter (R22)."""
        autor = _autor(c)
        if autor not in (self.konfig.ausloeser, self.konfig.helfer_konto):
            return f"[Kommentar von {autor or '?'} ausgelassen]"
        return f"--- {autor} ({c.get('created_at', '')}){ort}\n{c.get('body') or ''}"

    def material(self, titel: str, text: str, fund: FundStand) -> str:
        lesen = self.w.gh_lesen
        basis = f"/repos/{self.konfig.issue_repo}/issues/{self.n}"
        teile = [f"Issue #{self.n}: {titel}", text, "", "Kommentare des Issues:"]
        teile += [self._kommentar(c) for c in _seiten(lesen, f"{basis}/comments")]
        if fund.pr is not None:
            pr = f"/repos/{fund.repo}/pulls/{fund.pr}"
            diff = "\n".join(
                f"### {d.get('filename', '')}\n{d.get('patch') or ''}" for d in _seiten(lesen, f"{pr}/files")
            )
            if len(diff) > MAX_DIFF:
                diff = diff[:MAX_DIFF] + "\n[… Diff gekürzt]"
            teile += ["", f"Diff des PR {fund.repo}#{fund.pr}:", diff, "", "Review-Kommentare des PR:"]
            for c in _seiten(lesen, f"{pr}/comments"):
                ort = f" [{c.get('path', '')}:{c.get('line') or c.get('original_line') or ''}]"
                teile.append(self._kommentar(c, ort))
        return "\n".join(teile)

    # --- Ablauf ---

    def ausfuehren(self) -> str:
        issue = self.w.gh_lesen._api("GET", f"/repos/{self.konfig.issue_repo}/issues/{self.n}")
        issue = issue if isinstance(issue, dict) else {}
        titel, text = str(issue.get("title") or ""), str(issue.get("body") or "")
        roh_labels = issue.get("labels")
        labels = {x.get("name") if isinstance(x, dict) else x for x in roh_labels or [] if isinstance(roh_labels, list)}
        ziel, nur_einordnen = ziel_bestimmen(text, titel, self.konfig)
        fund = self.stand.funde.get(str(self.n))
        if fund is None:
            fund = FundStand(repo=ziel, branch=branch_name(self.n, titel))
            self.stand.funde[str(self.n)] = fund
        if fund.repo not in self.konfig.repos:
            self.melden(f"{PRAEFIX}Das Repo {fund.repo} ist nicht freigegeben, ich bearbeite den Auftrag nicht.")
            return self.fertig(fund, "abgelehnt:repo")
        ort = _pfad_norm(self.konfig.repos[fund.repo])
        repo_klon = self.klon if ort == "." else self.klon / ort
        wt = self.ordner / "wt" / f"{fund.repo.split('/')[1]}-{self.n}"
        try:
            worktree_bereit(repo_klon, wt, fund.branch, self.konfig.basis)
        except KonfliktFehler:
            self.melden(f"{PRAEFIX}Branch {fund.branch} ist auf GitHub und hier auseinandergelaufen, bitte prüfen.")
            return self.fertig(fund, "fehler:konflikt")
        except WorktreeFehler:
            self.melden(f"{PRAEFIX}Der Arbeitsordner für Branch {fund.branch} passt nicht zu diesem Fund, "
                        "bitte prüfen.")
            return self.fertig(fund, "fehler:worktree")
        except GitFehler:
            return self.fehlschlag(fund, "git")

        oeffentlich = fund.repo != self.konfig.issue_repo
        hinweise = [h for h, an in ((HINWEIS_OEFFENTLICH, oeffentlich), (HINWEIS_NUR_EINORDNEN, nur_einordnen)) if an]
        rueckfrage = RUECKFRAGE_LABEL in labels
        prompt = prompt_bauen(self.auftrag_texte(), self.material(titel, text, fund),
                              art="rueckfrage" if rueckfrage else "fund", hinweise=hinweise)
        erg: HelferErgebnis | None = None
        fehler: str | None = None
        beginn = self.w.uhr()
        try:
            try:
                erg = self.w.helfer(wt, prompt, fund.sitzung, self.konfig.frist_s)
            except HelferFehler as e:
                fehler = f"helfer_{e.art}"
                if e.art != "frist":  # evtl. verbrauchte/kaputte Sitzung: nächster Versuch beginnt neu
                    fund.sitzung = None
            except OSError:
                fehler = "helfer_start"
        finally:
            ende = self.w.uhr()
            budget_buchen(self.ordner, fehler or ("ok" if erg is not None else "abbruch"),
                          (ende - beginn).total_seconds(), jetzt=ende)
        if erg is None:
            return self.fehlschlag(fund, fehler or "helfer_unbekannt")
        fund.sitzung = erg.sitzung
        try:
            return self.aktion(erg, fund, wt, titel, nur_einordnen=nur_einordnen, rueckfrage=rueckfrage)
        except GitHubFehler:  # zählt als Versuch, sonst wiederholte jeder Durchgang die volle Runde
            return self.fehlschlag(fund, "github")

    def aktion(self, erg: HelferErgebnis, fund: FundStand, wt: Path, titel: str, *,
               nur_einordnen: bool, rueckfrage: bool) -> str:
        issue_repo = self.konfig.issue_repo
        grund = _begruendung(erg.text)
        if erg.aktion == "fix" and nur_einordnen:
            self.antworten(fund, f"{NUR_EINORDNEN_TEXT}\n\n{grund}")
        elif erg.aktion == "fix":
            return self.fix(grund, fund, wt, titel)
        elif erg.aktion == "absichtlich":
            kommentar = PRAEFIX + grund
            muster = pruefen(kommentar)
            if muster is not None:
                self.halten(muster)
            else:
                self.w.gh_lab.schliessen(issue_repo, self.n, kommentar, grund="not_planned")
        elif erg.aktion == "rueckfrage" and rueckfrage and erg.antwort:
            self.w.gh_lab.labels_setzen(issue_repo, self.n, [f"antwort:{erg.antwort}"])
        else:
            self.antworten(fund, PRAEFIX + grund)
        return self.fertig(fund)

    def fix(self, grund: str, fund: FundStand, wt: Path, titel: str) -> str:
        """Push und PR; ohne neue Commits (und mit PR oder ohne Commits vor der Basis) nur Antwort."""
        basis = self.konfig.basis
        try:
            neu = voraus(wt, fund.branch, basis=basis)
            betreffe = _betreffzeilen(wt, f"origin/{basis}")
            inhalt = _push_inhalt(wt, fund.branch, basis) if neu > 0 else ""
        except GitFehler:
            return self.fehlschlag(fund, "git")
        if neu == 0 and (fund.pr is not None or not betreffe):
            self.antworten(fund, PRAEFIX + grund)
            return self.fertig(fund)
        liste = "\n".join(f"- {b}" for b in betreffe)
        if fund.repo == self.konfig.issue_repo:
            koerper = f"Fixes #{self.n}\n\n{liste}"
            pr_titel = _kappen(PRAEFIX + titel.removeprefix(TITEL_PRAEFIX))
        else:  # öffentlich (R19): nur Betreffzeilen und Bezug, kein Gespräch, Titel aus ältestem Commit
            koerper = f"{liste}\n\nBezug: {self.konfig.issue_repo}#{self.n}"
            pr_titel = _kappen(PRAEFIX + betreffe[0])
        for t in (pr_titel, koerper, grund, inhalt):  # R21: auch volle Nachrichten und Diffs vor dem Push
            if (muster := pruefen(t)) is not None:
                self.halten(muster)
                return self.fertig(fund)
        if neu > 0:
            try:
                self.w.push(wt, fund.branch)
            except (GitFehler, RuntimeError, ValueError, OSError):
                return self.fehlschlag(fund, "push")
        neuer_pr = fund.pr is None
        if fund.pr is None:
            fund.pr = self._pr_finden_oder_anlegen(fund, pr_titel, koerper)
        fund.zustand = "pr"
        zusatz = f"PR: {fund.repo}#{fund.pr}" if neuer_pr else f"Neuer Stand im PR {fund.repo}#{fund.pr}."
        self.antworten(fund, f"{PRAEFIX}{grund}\n\n{zusatz}")
        return self.fertig(fund)

    def _pr_finden_oder_anlegen(self, fund: FundStand, titel: str, koerper: str) -> int:
        for pr in self.w.gh_lesen.prs_zum_kopf(fund.repo, fund.branch):
            nummer = pr.get("number")
            if pr.get("state") == "open" and isinstance(nummer, int) and not isinstance(nummer, bool):
                return nummer
        return self.w.gh_lab.pr_anlegen(fund.repo, fund.branch, self.konfig.basis, titel, koerper)


def _warten_melden(grund: str, issue: int, konfig: DebuggerKonfig, stand: DebuggerStand, w: Werkzeuge,
                   jetzt: datetime) -> None:
    """Höchstens eine Wartemeldung je Grund und UTC-Kalendertag (Schlüssel in `stand.gemeldet`)."""
    schluessel = f"wartet:{grund}:{jetzt.astimezone(timezone.utc).date().isoformat()}"
    if schluessel in stand.gemeldet:
        return
    text = f"{PRAEFIX}Auftrag wartet ({grund})."
    muster = pruefen(text)
    w.gh_lab.kommentieren(konfig.issue_repo, issue, text if muster is None else _zurueckgehalten(muster))
    stand.gemeldet.append(schluessel)
    del stand.gemeldet[:-MAX_GEMELDET]


def runde_ausfuehren(auftrag: Auftrag, konfig: DebuggerKonfig, stand: DebuggerStand, klon: Path,
                     ordner: Path, w: Werkzeuge) -> str:
    """Führt einen Auftrag aus: `"ok" | "wartet:<grund>" | "fehler:<art>" | "abgelehnt:<grund>"`.

    Bestätigt jeden Kommentar (`eyes`), prüft Not-Aus, Tagesgrenze und Speicher, bereitet den
    Worktree vor, ruft den Helfer und führt dessen Aktion aus. `stand` wird nur im Speicher geändert.
    """
    for k in auftrag.kommentare:
        w.gh_lab.reagieren(k.repo, k.id, review=k.review)
    jetzt = w.uhr()
    grund = budget_pruefen(ordner, konfig.tagesgrenze, jetzt=jetzt)
    if grund is None:
        frei = w.frei_gb()
        if frei is None or frei < konfig.min_frei_gb:
            grund = "speicher"
    if grund is not None:
        _warten_melden(grund, auftrag.issue, konfig, stand, w, jetzt)
        return f"wartet:{grund}"
    return _Lauf(auftrag, konfig, stand, klon, ordner, w).ausfuehren()
