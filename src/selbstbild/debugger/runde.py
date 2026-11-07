"""Eine Runde des Debugger-Claw: vom Auftrag zur Antwort, zum Fix-PR oder zur Einordnung.

Jeder Text an GitHub läuft vorher durch `pruefen`; geschrieben wird nur über `gh_lab`
(das die Debugger-Marke selbst anhängt). Den Stand verändert die Runde nur im Speicher,
speichern muss der Aufrufer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from selbstbild.debugger.auftraege import Auftrag, Kommentar
from selbstbild.debugger.budget import budget_buchen, budget_pruefen
from selbstbild.debugger.helfer import HelferErgebnis, HelferFehler, prompt_bauen
from selbstbild.debugger.klon import (
    TITEL_PRAEFIX,
    ZU_GROSS,
    GitFehler,
    KonfliktFehler,
    PushZuGross,
    WorktreeFehler,
    aufraeumen,
    betreffzeilen,
    push_inhalt,
    voraus,
    worktree_bereit,
)
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.spiegel import ZurueckgehaltenFehler, ist_pin_branch
from selbstbild.debugger.stand import DebuggerStand, FundStand
from selbstbild.debugger.versuche import MAX_VERSUCHE, einmal_reagieren, reagiert_aufraeumen, versuch_zaehlen
from selbstbild.debugger.ziel import fund_neu, pfad_norm, repo_klon_pfad, wt_pfad, ziel_bestimmen
from selbstbild.funde.github import GitHub, GitHubFehler
from selbstbild.funde.text import pruefen

# pfad_norm, repo_klon_pfad, wt_pfad und ziel_bestimmen stehen in ziel.py; hier für Nachlauf und Wächter
__all__ = ["Werkzeuge", "pfad_norm", "pr_finden_oder_anlegen", "repo_klon_pfad", "runde_ausfuehren", "wt_pfad",
           "ziel_bestimmen", "zurueckgehalten"]

LOG = logging.getLogger("selbstbild.debugger")
PRAEFIX = "🤖 Debugger-Claw: "
KEINE_BEGRUENDUNG = "(keine Begründung vom Helfer)"
MAX_PR_TITEL = 256
MAX_SEITEN = 10
SEITE = 100
RUECKFRAGE_LABEL = "selbstbild:rueckfrage"
MAX_DIFF = 60_000
MAX_GEMELDET = 200
NUR_EINORDNEN_TEXT = PRAEFIX + "Fixes in diesem Repo sind nicht freigegeben, ich kann nur einordnen."
HINWEIS_OEFFENTLICH = (
    "Dieses Repo ist öffentlich. Commit-Texte beschreiben nur die Code-Änderung, ohne Inhalte aus dem Issue."
)
HINWEIS_NUR_EINORDNEN = "Fixes in diesem Repo sind nicht freigegeben. Ändere keine Dateien, ordne nur ein."
HINWEIS_FEST = ("Arbeite nur in diesem Worktree und seinem Branch. Keine Einträge in WORKBOARD.md, keine Commits "
                "auf master, kein Push.")  # Punkt 10: immer, als erster Hinweis
PIN_TEXT = PRAEFIX + ("Der Pin-PR enthält nur die Submodul-Aktualisierung. Für einen weiteren Fix bitte nach dem "
                      "Merge kommentieren.")


def _nichts() -> None:
    """Standard für `Werkzeuge.holen` und `Werkzeuge.claw_aufraeumen`: nichts tun (Tests, PC)."""


@dataclass
class Werkzeuge:
    """Alles, was eine Runde von außen braucht (für Tests austauschbar)."""

    gh_lab: GitHub  # Konto Vibemind-LAB, marke=DEBUGGER_MARKE
    gh_lesen: GitHub
    helfer: Callable[..., HelferErgebnis]  # wie helfer_starten(wt, prompt, sitzung, frist_s)
    frei_gb: Callable[[], float | None]
    uhr: Callable[[], datetime]
    push: Callable[..., None]  # (wt, branch, *, nur_gitlink=None) über den Spiegel
    holen: Callable[[], None] = _nichts  # R27: Spiegel von GitHub holen, vor `nachziehen`
    claw_aufraeumen: Callable[[], None] = _nichts  # R44: Prozesse des Helfer-Nutzers beenden
    spiegel: Path | None = None  # Ordner der root-eigenen Spiegel (R39: Pin-Prüfungen)


# ---------- Runde ----------

def zurueckgehalten(muster: str) -> str:
    """Der feste Rückhalte-Satz, wenn ein Text ein Schlüssel-Muster trifft."""
    return f"{PRAEFIX}Antwort zurückgehalten (Schlüssel-Verdacht: {muster})."


def pr_finden_oder_anlegen(w: Werkzeuge, repo: str, kopf: str, basis: str, titel: str, koerper: str) -> int:
    """Nummer eines offenen PR mit Kopf `kopf`, sonst ein neuer PR über `gh_lab`."""
    for pr in w.gh_lesen.prs_zum_kopf(repo, kopf):
        nummer = pr.get("number")
        if pr.get("state") == "open" and isinstance(nummer, int) and not isinstance(nummer, bool):
            return nummer
    return w.gh_lab.pr_anlegen(repo, kopf, basis, titel, koerper)


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
        self.w.gh_lab.kommentieren(self.konfig.issue_repo, self.n, zurueckgehalten(muster))

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
            fund.naechster_versuch = None  # R42: Erfolg hebt die Pause auf
        if ergebnis == "ok" and self.zurueckgehalten:
            return "abgelehnt:schluessel"
        return ergebnis

    def fehlschlag(self, fund: FundStand, art: str) -> str:
        """Zählt einen Versuch je Kommentar (mit Pause, R42); ab 3 Versuchen gilt ein Kommentar als verarbeitet."""
        aufgegeben = versuch_zaehlen(fund, self.ids, self.w.uhr())
        if aufgegeben:
            self.stand.verarbeitet_hinzu(aufgegeben)
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

    # --- Fund (R37) ---

    def pr_offen(self, fund: FundStand) -> bool:
        return fund.pr is not None and self.w.gh_lesen.pr_lesen(fund.repo, fund.pr).get("state") == "open"

    def fund_bestimmen(self, titel: str, ziel: str) -> FundStand:
        """Bestehender FundStand gewinnt (R1); nach `erledigt` oder mit nicht mehr offenem PR ein neuer.

        Der neue Stand (R37) hat kein PR und keine Sitzung, das Repo aus `ziel_bestimmen`, die
        alten Versuche und `runde + 1`, also auch einen neuen Branch-Namen.
        """
        alt = self.stand.funde.get(str(self.n))
        if alt is not None and alt.zustand != "erledigt" and (alt.pr is None or self.pr_offen(alt)):
            return alt
        fund = fund_neu(self.n, titel, ziel, self.konfig, runde=alt.runde + 1 if alt else 1,
                        versuche=alt.versuche if alt else None)
        self.stand.funde[str(self.n)] = fund
        return fund

    # --- Ablauf ---

    def ausfuehren(self) -> str:
        issue = self.w.gh_lesen._api("GET", f"/repos/{self.konfig.issue_repo}/issues/{self.n}")
        issue = issue if isinstance(issue, dict) else {}
        titel, text = str(issue.get("title") or ""), str(issue.get("body") or "")
        roh_labels = issue.get("labels")
        labels = {x.get("name") if isinstance(x, dict) else x for x in roh_labels or [] if isinstance(roh_labels, list)}
        ziel, nur_einordnen = ziel_bestimmen(text, titel, self.konfig)
        fund = self.fund_bestimmen(titel, ziel)
        # R45: auf dem Pin-Branch nur einordnen und nie pushen (der Pin-PR trägt nur den Gitlink)
        pin = fund.repo == self.konfig.issue_repo and ist_pin_branch(fund.branch)
        nur_einordnen = nur_einordnen or pin
        if fund.repo not in self.konfig.repos:
            self.melden(f"{PRAEFIX}Das Repo {fund.repo} ist nicht freigegeben, ich bearbeite den Auftrag nicht.")
            return self.fertig(fund, "abgelehnt:repo")
        repo_klon = repo_klon_pfad(self.konfig, self.klon, fund.repo)
        wt = wt_pfad(self.ordner, fund.repo, fund.branch)
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
        hinweise = [HINWEIS_FEST] + [h for h, an in ((HINWEIS_OEFFENTLICH, oeffentlich),
                                                     (HINWEIS_NUR_EINORDNEN, nur_einordnen)) if an]
        rueckfrage = RUECKFRAGE_LABEL in labels
        prompt = prompt_bauen(self.auftrag_texte(), self.material(titel, text, fund),
                              art="rueckfrage" if rueckfrage else "fund", hinweise=hinweise)
        erg: HelferErgebnis | None = None
        fehler: str | None = None
        beginn = self.w.uhr()
        try:  # R41: vor dem Helfer buchen; geht das nicht, startet er nicht (fail-closed, ungezählt)
            budget_buchen(self.ordner, "laeuft", 0.0, jetzt=beginn)
        except OSError as e:
            LOG.warning("Budget nicht buchbar (%s), keine Runde", type(e).__name__)
            return "fehler:budget"
        try:
            try:
                erg = self.w.helfer(wt, prompt, fund.sitzung, self.konfig.frist_s)
            except HelferFehler as e:
                fehler = f"helfer_{e.art}"
                if e.art not in ("frist", "antwort_fehler"):  # evtl. kaputte Sitzung: nächster Versuch neu
                    fund.sitzung = None
            except OSError:
                fehler = "helfer_start"
        finally:
            self.w.claw_aufraeumen()  # R44: nichts vom Helfer läuft weiter, bevor gepusht wird
            ende = self.w.uhr()
            try:  # R41: das Ergebnis als Nachtrag, der nicht zählt
                budget_buchen(self.ordner, fehler or ("ok" if erg is not None else "abbruch"),
                              (ende - beginn).total_seconds(), jetzt=ende, nachtrag=True)
            except OSError as e:
                LOG.warning("Budget-Nachtrag nicht buchbar (%s)", type(e).__name__)
        if fehler == "helfer_antwort_fehler":  # R42: Abo-Limit o. Ä., kein Versuch, Kommentare bleiben offen
            _warten_melden("limit", self.n, self.konfig, self.stand, self.w, ende)
            return "wartet:limit"
        if erg is None:
            return self.fehlschlag(fund, fehler or "helfer_unbekannt")
        fund.sitzung = erg.sitzung
        try:
            return self.aktion(erg, fund, wt, titel, nur_einordnen=nur_einordnen, rueckfrage=rueckfrage, pin=pin,
                               labels=frozenset(x for x in labels if isinstance(x, str)))
        except GitHubFehler:  # zählt als Versuch, sonst wiederholte jeder Durchgang die volle Runde
            return self.fehlschlag(fund, "github")
        finally:
            if pin:
                self.wt_aufraeumen(fund, wt, pin=True)

    def wt_aufraeumen(self, fund: FundStand, wt: Path, *, pin: bool = False) -> None:
        """R43: ohne PR und ohne Commits voraus Worktree und lokalen Branch entfernen (`sitzung` bleibt).

        R45 (`pin=True`): umgekehrt, Commits des Helfers im Pin-Worktree werden nie gepusht, also weg.
        """
        if fund.pr is not None and not pin:
            return
        try:
            if (voraus(wt, fund.branch, basis=self.konfig.basis) > 0) == pin:
                aufraeumen(repo_klon_pfad(self.konfig, self.klon, fund.repo), wt, fund.branch)
        except (GitFehler, WorktreeFehler, ValueError, OSError) as e:
            LOG.warning("Worktree nicht aufgeräumt (%s)", type(e).__name__)

    def aktion(self, erg: HelferErgebnis, fund: FundStand, wt: Path, titel: str, *,
               nur_einordnen: bool, rueckfrage: bool, pin: bool, labels: frozenset[str]) -> str:
        issue_repo = self.konfig.issue_repo
        grund = _begruendung(erg.text)
        if erg.aktion == "fix" and pin:
            self.melden(PIN_TEXT)
        elif erg.aktion == "fix" and nur_einordnen:
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
            neu = f"antwort:{erg.antwort}"
            for alt in sorted(x for x in labels if x.startswith("antwort:") and x != neu):  # Punkt 7
                self.w.gh_lab.label_entfernen(issue_repo, self.n, alt)
            self.w.gh_lab.labels_setzen(issue_repo, self.n, [neu])
        else:
            self.antworten(fund, PRAEFIX + grund)
        if not pin:
            self.wt_aufraeumen(fund, wt)  # R43
        return self.fertig(fund)

    def fix(self, grund: str, fund: FundStand, wt: Path, titel: str) -> str:
        """Push und PR; ohne neue Commits (und mit PR oder ohne Commits vor der Basis) nur Antwort."""
        basis = self.konfig.basis
        try:
            neu = voraus(wt, fund.branch, basis=basis)
            betreffe = betreffzeilen(wt, f"origin/{basis}")
            inhalt = push_inhalt(wt, fund.branch, basis) if neu > 0 else ""
        except PushZuGross:  # R23: fail-closed, kein Push
            self.halten(ZU_GROSS)
            return self.fertig(fund)
        except GitFehler:
            return self.fehlschlag(fund, "git")
        if neu == 0 and (fund.pr is not None or not betreffe):
            self.antworten(fund, PRAEFIX + grund)
            self.wt_aufraeumen(fund, wt)  # R43
            return self.fertig(fund)
        liste = "\n".join(f"- {b}" for b in betreffe)
        if fund.repo == self.konfig.issue_repo:
            koerper = f"{PRAEFIX}Fixes #{self.n}\n\n{liste}"
            pr_titel = _kappen(PRAEFIX + titel.removeprefix(TITEL_PRAEFIX))
        else:  # öffentlich (R19): nur Betreffzeilen und Bezug, kein Gespräch, Titel aus ältestem Commit
            koerper = f"{PRAEFIX}{liste}\n\nBezug: {self.konfig.issue_repo}#{self.n}"
            pr_titel = _kappen(PRAEFIX + betreffe[0])
        for t in (pr_titel, koerper, grund, inhalt):  # R21/R23: auch Commit-Texte und rohe Blobs vor dem Push
            if (muster := pruefen(t)) is not None:
                self.halten(muster)
                return self.fertig(fund)
        if neu > 0 and fund.pr is not None and not self.pr_offen(fund):
            return "wartet:pr_zu"  # R37: PR während der Runde zu; der Nachlauf räumt ihn ab, dann neu
        if neu > 0:
            try:
                self.w.push(wt, fund.branch)
            except ZurueckgehaltenFehler as e:  # R29: maßgebliche Prüfung im Spiegel
                self.halten(e.muster)
                return self.fertig(fund)
            except (GitFehler, RuntimeError, ValueError, OSError):
                return self.fehlschlag(fund, "push")
        neuer_pr = fund.pr is None
        if fund.pr is None:
            fund.pr = pr_finden_oder_anlegen(self.w, fund.repo, fund.branch, self.konfig.basis, pr_titel, koerper)
        fund.zustand = "pr"
        zusatz = f"PR: {fund.repo}#{fund.pr}" if neuer_pr else f"Neuer Stand im PR {fund.repo}#{fund.pr}."
        self.antworten(fund, f"{PRAEFIX}{grund}\n\n{zusatz}")
        return self.fertig(fund)


def _warten_melden(grund: str, issue: int, konfig: DebuggerKonfig, stand: DebuggerStand, w: Werkzeuge,
                   jetzt: datetime) -> None:
    """Höchstens eine Wartemeldung je Grund und UTC-Kalendertag (Schlüssel in `stand.gemeldet`)."""
    schluessel = f"wartet:{grund}:{jetzt.astimezone(timezone.utc).date().isoformat()}"
    if schluessel in stand.gemeldet:
        return
    text = f"{PRAEFIX}Auftrag wartet ({grund})."
    muster = pruefen(text)
    w.gh_lab.kommentieren(konfig.issue_repo, issue, text if muster is None else zurueckgehalten(muster))
    stand.gemeldet.append(schluessel)
    del stand.gemeldet[:-MAX_GEMELDET]


def runde_ausfuehren(auftrag: Auftrag, konfig: DebuggerKonfig, stand: DebuggerStand, klon: Path,
                     ordner: Path, w: Werkzeuge) -> str:
    """Führt einen Auftrag aus: `"ok" | "wartet:<grund>" | "fehler:<art>" | "abgelehnt:<grund>"`.

    Bestätigt jeden Kommentar einmal (`eyes`, Punkt 2), prüft Not-Aus, Tagesgrenze und Speicher, bereitet den
    Worktree vor, ruft den Helfer und führt dessen Aktion aus. `stand` wird nur im Speicher geändert.
    """
    einmal_reagieren(stand, auftrag.kommentare, lambda k: w.gh_lab.reagieren(k.repo, k.id, review=k.review))
    try:
        return _runde(auftrag, konfig, stand, klon, ordner, w)
    finally:
        reagiert_aufraeumen(stand)


def _runde(auftrag: Auftrag, konfig: DebuggerKonfig, stand: DebuggerStand, klon: Path, ordner: Path,
           w: Werkzeuge) -> str:
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
