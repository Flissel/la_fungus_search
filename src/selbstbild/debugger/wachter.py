"""Wächter des Debugger-Claw: ein Durchgang, die Schleife und die CLI `python -m selbstbild debugger`.

Ein Durchgang: Sperre -> Stand -> (erster Start) -> Spiegel holen -> Klon nachziehen -> Kommentare
lesen -> höchstens eine Runde (der erste Kandidat ohne Pause und ohne Tagessperre) -> Helfer-Prozesse
beenden -> Nachlauf (auch nach einer Ausnahme der Runde) -> Lesemarke `seit` -> Stand speichern ->
Prozesse des Helfer-Nutzers beenden. Der Token liegt nur in der Datei aus `DEBUGGER_TOKEN_DATEI`;
der Wächter legt ihn nie in seine eigene Umgebung. Ausnahmen kommen nur mit ihrem Typ ins Log.
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, NoReturn, Sequence

# `_zeit`: dieselbe Zeitlesung wie die Filter der Aufträge, damit Lesemarke und Filter übereinstimmen
from selbstbild.debugger.auftraege import Auftrag, Kommentar, _zeit, auftraege_bilden, kommentar_aus_api
from selbstbild.debugger.budget import frei_gb
from selbstbild.debugger.helfer import helfer_starten
from selbstbild.debugger.klon import git_praefix_setzen, nachziehen
from selbstbild.debugger.konfig import DebuggerKonfig, konfig_laden
from selbstbild.debugger.nachlauf import nachlauf
from selbstbild.debugger.runde import Werkzeuge, pfad_norm, pr_eintraege_abgleichen, runde_ausfuehren, wt_pfad
from selbstbild.debugger.spiegel import spiegel_bereit, spiegel_holen, spiegel_pfad, spiegel_push
from selbstbild.debugger.stand import DebuggerStand, stand_laden, stand_speichern
from selbstbild.debugger.versuche import gh_fehler_merken, gh_fehler_vergessen, gh_gesperrt, limit_aktiv, pausiert
from selbstbild.freigabe_eingang import FreigabeEingang
from selbstbild.funde.github import DEBUGGER_MARKE, GitHub, GitHubFehler
from selbstbild.sperre import SperreBelegt, sperre

LOG = logging.getLogger("selbstbild.debugger")
STAND_DATEI = "debugger-stand.json"
SPERRE = "debugger.lock"
KERNEL_SPERRE = "debugger.flock"
UEBERLAPPUNG = timedelta(minutes=10)  # wegen der Uhr (VM gegen GitHub)
PKILL_TIMEOUT_S = 30
KLON_GRUND = "klon:"
_FRUEHESTE = datetime.min.replace(tzinfo=timezone.utc)

Laeufer = Callable[..., subprocess.CompletedProcess[bytes]]


def _iso(t: datetime) -> str:
    """UTC im Format, das GitHub für `since` erwartet (Sekunden abgeschnitten, also nie zu spät)."""
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def kernel_sperre(pfad: Path) -> Iterator[None]:
    """R34: exklusiver Kernel-Lock auf `pfad` für den ganzen Durchgang; belegt: `SperreBelegt`.

    POSIX `fcntl.flock(LOCK_EX | LOCK_NB)`: gilt über Container hinweg, die dasselbe Volume
    einbinden (eine PID ist dort keine Identität), und fällt mit dem Prozess. Windows (nur Tests,
    PC): `msvcrt.locking(LK_NBLCK)` auf das erste Byte. Die Datei bleibt liegen, sie trägt nichts.
    """
    pfad.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(pfad, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SperreBelegt(pfad.name) from None
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)  # POSIX: gibt den flock frei


def _verwaist(_pid: int) -> bool:
    """Unter dem Kernel-Lock läuft kein anderer Durchgang: jede vorhandene Dateisperre ist verwaist."""
    return False


def claw_prozesse_beenden(laeufer: Laeufer = subprocess.run) -> None:
    """R32: Ist `DEBUGGER_CLAW_UID` gesetzt, alle Prozesse dieser uid beenden; Code 1 heißt: es gab keine."""
    uid = os.environ.get("DEBUGGER_CLAW_UID", "").strip()
    if not uid:
        return
    if not uid.isdigit():
        LOG.warning("DEBUGGER_CLAW_UID ist keine Zahl, keine Prozesse beendet")
        return
    try:
        erg = laeufer(["pkill", "-KILL", "-U", uid], check=False, capture_output=True,
                      stdin=subprocess.DEVNULL, timeout=PKILL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        LOG.warning("pkill gescheitert (%s)", type(e).__name__)
        return
    if erg.returncode not in (0, 1):
        LOG.warning("pkill gescheitert (Code %d)", erg.returncode)


class StandUnlesbar(Exception):
    """Die Stand-Datei ließ sich nicht lesen (OSError); sie darf nicht überschrieben werden."""


class TokenDateiFehlt(RuntimeError):
    """`DEBUGGER_TOKEN_DATEI` ist nicht gesetzt (Startfehler)."""


class RootOhnePraefix(RuntimeError):
    """Punkt 3: Der Wächter läuft als root, aber `DEBUGGER_PRAEFIX` ist leer (Helfer liefe als root)."""


class AskpassFehlt(RuntimeError):
    """Punkt 3: `DEBUGGER_ASKPASS` fehlt oder ist keine ausführbare Datei (`kein_askpass`)."""

    def __init__(self) -> None:
        super().__init__("kein_askpass")


def _ist_root(name: str | None = None, geteuid: Callable[[], int] | None = None) -> bool:
    """POSIX mit effektiver uid 0; unter Windows nie. Parameter nur für Tests."""
    if (name or os.name) == "nt":
        return False
    euid = geteuid or getattr(os, "geteuid", None)
    return euid is not None and euid() == 0


def _askpass_pruefen() -> None:
    roh = os.environ.get("DEBUGGER_ASKPASS", "").strip()
    if not roh or not os.path.isfile(roh) or not os.access(roh, os.X_OK):
        raise AskpassFehlt()


class _Durchgang:
    """Ein Durchgang unter der Sperre; `lauf` speichert den Stand, sobald er geladen ist."""

    def __init__(self, konfig: DebuggerKonfig, klon: Path, ordner: Path, w: Werkzeuge) -> None:
        self.konfig, self.klon, self.ordner, self.w = konfig, klon, ordner, w
        self.pfad = ordner / STAND_DATEI

    def lauf(self) -> int:
        try:
            stand = self.laden()
        except Exception as e:  # nichts gespeichert: eine unlesbare Datei wird nie überschrieben
            LOG.error("Stand nicht ladbar (%s)", type(e).__name__)
            return 1
        code = 1
        try:
            code = self.schritte(stand)
        except GitHubFehler as e:  # zählt nicht als Versuch; der nächste Durchgang liest erneut
            LOG.warning("GitHub-Fehler (GitHubFehler %s), weiter im nächsten Durchgang", e.art)
            code = 0
        except Exception as e:
            LOG.error("Durchgang abgebrochen (%s)", type(e).__name__)
            code = 1
        finally:
            try:
                stand_speichern(self.pfad, stand)
            except OSError as e:
                LOG.error("Stand nicht gespeichert (%s)", type(e).__name__)
                code = 1
        return code

    def laden(self) -> DebuggerStand:
        """Ein kaputter Stand wird beiseitegelegt (nie überschrieben); danach gilt der erste Start.

        R36: Ein Lesefehler (`stand_unlesbar`) ist kein kaputter Inhalt: `StandUnlesbar`, der
        Durchgang endet mit Exit 1 und speichert nicht.
        """
        stand = stand_laden(self.pfad)
        if stand.start is None and "stand_unlesbar" in stand.gemeldet:
            raise StandUnlesbar()
        if stand.start is None and "stand_kaputt" in stand.gemeldet and self.pfad.exists():
            kopie = self.pfad.with_name(f"{self.pfad.name}.kaputt-{_iso(self.w.uhr()).replace(':', '')}")
            os.replace(self.pfad, kopie)
            LOG.warning("Stand unlesbar, beiseitegelegt als %s; weiter wie beim ersten Start", kopie.name)
        return stand

    def schritte(self, stand: DebuggerStand) -> int:
        k, w = self.konfig, self.w
        beginn = w.uhr()
        if stand.start is None:  # erster Start: alles bis jetzt gilt als verarbeitet
            stand.start = stand.seit = _iso(beginn)
            stand.gemeldet = [g for g in stand.gemeldet if g != "stand_fehlt"]
            LOG.info("erster Start, Lesemarke %s", stand.start)
            return 0
        w.holen()
        submodule = [p for p in (pfad_norm(o) for o in k.repos.values()) if p != "."]
        if self.klon_melden(stand, nachziehen(self.klon, k.basis, submodule)):
            return 0
        kommentare = self.kommentare(self.lesemarke(stand, beginn))
        issues = {i["number"] for i in w.gh_lesen.issues(k.issue_repo)
                  if isinstance(i.get("number"), int) and not isinstance(i.get("number"), bool)}
        pr_zu_issue = {(f.repo, f.pr): int(n) for n, f in stand.funde.items() if f.pr is not None and n.isdigit()}
        # R4: alle Kandidaten (keine laufende Runde); ausgeführt wird nur der erste, der nicht wartet
        kandidaten = auftraege_bilden(kommentare, stand, k, selbstbild_issues=issues, pr_zu_issue=pr_zu_issue,
                                      laufend=set())
        jetzt = w.uhr()
        bereit = [a for a in kandidaten if not pausiert(stand.funde.get(str(a.issue)), jetzt)  # R42
                  and not gh_gesperrt(stand, a.issue, jetzt)]  # Punkt 1
        if bereit and limit_aktiv(stand, jetzt):  # R49: global kein Helfer; Nachlauf und `seit` wie sonst
            LOG.info("Abo-Limit bis %s, keine Runde", stand.limit_bis)
            bereit = []
        fehler = self.runde(bereit[0], stand) if bereit else None
        try:  # Punkt 1: der Nachlauf läuft auch nach einer Ausnahme der Runde
            w.claw_aufraeumen()  # R44: vor dem Nachlauf läuft nichts mehr als claw
            for zeile in nachlauf(k, stand, self.klon, self.ordner, w):
                LOG.info("Nachlauf %s", zeile)
        except Exception as e:
            LOG.error("Nachlauf abgebrochen (%s)", type(e).__name__)
            fehler = fehler or e
        try:  # Freigabe-Eingang: gemergte/geschlossene PRs abhaken; nie ein Grund, den Durchgang zu brechen
            pr_eintraege_abgleichen(w)
        except Exception as e:
            LOG.warning("Freigabe-Abgleich fehlgeschlagen (%s)", type(e).__name__)
        if fehler is not None:
            raise fehler
        stand.seit = _iso(self.neue_marke(stand, beginn, kandidaten))  # auch übersprungene halten `seit`
        return 0

    def runde(self, auftrag: Auftrag, stand: DebuggerStand) -> Exception | None:
        """Eine Runde; ihre Ausnahme kommt zurück statt hoch, damit der Nachlauf trotzdem läuft.

        Ein ungezählter `GitHubFehler` wird je Issue und Tag gezählt (Punkt 1), Erfolg löscht den Zähler.
        """
        try:
            ergebnis = runde_ausfuehren(auftrag, self.konfig, stand, self.klon, self.ordner, self.w)
        except GitHubFehler as e:
            gh_fehler_merken(stand, auftrag.issue, self.w.uhr())
            return e
        except Exception as e:
            LOG.error("Runde #%d abgebrochen (%s)", auftrag.issue, type(e).__name__)
            return e
        gh_fehler_vergessen(stand, auftrag.issue)
        LOG.info("Runde #%d: %s", auftrag.issue, ergebnis)
        return None

    def klon_melden(self, stand: DebuggerStand, grund: str | None) -> bool:
        """Nachziehen-Grund je Grund einmal ins Log und nach `gemeldet`; ist er weg, der Eintrag auch."""
        schluessel = f"{KLON_GRUND}{grund}" if grund else None
        stand.gemeldet = [g for g in stand.gemeldet if not g.startswith(KLON_GRUND) or g == schluessel]
        if schluessel is None:
            return False
        if schluessel not in stand.gemeldet:
            LOG.warning("Klon nicht nachgezogen (%s), keine Runde", grund)
            stand.gemeldet.append(schluessel)
        return True

    def kommentare(self, seit: str) -> list[Kommentar]:
        """Issue- und Review-Zeilenkommentare aller Repos seit `seit` (Review-Zusammenfassungen nicht)."""
        lesen = self.w.gh_lesen
        alle: list[Kommentar] = []
        for repo in self.konfig.repos:
            for review, liste in ((False, lesen.kommentare_seit(repo, seit)),
                                  (True, lesen.review_kommentare_seit(repo, seit))):
                for roh in liste:
                    try:
                        alle.append(kommentar_aus_api(repo, roh, review=review))
                    except (ValueError, KeyError, TypeError) as e:
                        LOG.info("Kommentar übersprungen (%s)", type(e).__name__)
        return alle

    @staticmethod
    def lesemarke(stand: DebuggerStand, beginn: datetime) -> str:
        """`seit` minus Überlappung; ist `seit` unlesbar, dann `start`, sonst der Beginn."""
        for text in (stand.seit, stand.start):
            t = _zeit(text) if text else _FRUEHESTE
            if t != _FRUEHESTE:
                return _iso(t - UEBERLAPPUNG)
        return _iso(beginn - UEBERLAPPUNG)

    @staticmethod
    def neue_marke(stand: DebuggerStand, beginn: datetime, kandidaten: Sequence[Auftrag]) -> datetime:
        """Minimum aus Beginn minus Überlappung und dem ältesten Kandidaten, der unverarbeitet blieb."""
        verarbeitet = set(stand.verarbeitet)
        offen = [_zeit(k.erstellt) for a in kandidaten for k in a.kommentare if k.id not in verarbeitet]
        return min([beginn - UEBERLAPPUNG, *offen])


def durchgang(wurzel: Path, klon: Path, ordner: Path, w: Werkzeuge, *, konfig: DebuggerKonfig | None = None,
              laeufer: Laeufer = subprocess.run) -> int:
    """Ein Durchgang: Exit 0 (auch bei Wartegrund, Klon-Grund oder GitHub-Fehler), 1 bei Ausnahme, 3 bei Sperre.

    Zuerst der Kernel-Lock (R34), darunter die Dateisperre: Eine vorhandene Sperrdatei stammt dann
    sicher aus einem beendeten Lauf und wird übernommen, auch nach einem Neustart mit derselben PID.
    Der Stand wird gespeichert, sobald er geladen ist; danach (R32) werden die Prozesse des
    Helfer-Nutzers beendet, wenn `DEBUGGER_CLAW_UID` gesetzt ist. `laeufer` nur für Tests.
    """
    try:
        konfig = konfig if konfig is not None else konfig_laden(wurzel)
    except ValueError as e:
        LOG.error("Konfiguration ungültig (%s)", type(e).__name__)
        return 1
    try:
        with kernel_sperre(ordner / KERNEL_SPERRE), sperre(ordner / SPERRE, veraltet_s=0.0, pid_lebt=_verwaist):
            try:
                return _Durchgang(konfig, klon, ordner, w).lauf()
            finally:
                claw_prozesse_beenden(laeufer)
    except SperreBelegt:
        LOG.info("Sperre belegt, kein Durchgang")
        return 3
    except OSError as e:
        LOG.error("Sperre nicht anlegbar (%s)", type(e).__name__)
        return 1


def schleife(wurzel: Path, klon: Path, ordner: Path, fabrik: Callable[[DebuggerKonfig], Werkzeuge], *,
             schlafen: Callable[[float], None] = time.sleep) -> NoReturn:
    """`durchgang` alle `konfig.intervall_s` Sekunden; Konfiguration und Werkzeuge je Durchgang neu (R27)."""
    while True:
        intervall: float = DebuggerKonfig.intervall_s
        try:
            konfig = konfig_laden(wurzel)
            intervall = konfig.intervall_s
            durchgang(wurzel, klon, ordner, fabrik(konfig), konfig=konfig)
        except Exception as e:
            LOG.error("Durchgang nicht gestartet (%s)", type(e).__name__)
        schlafen(intervall)


def werkzeuge_standard(konfig: DebuggerKonfig, ordner: Path, spiegel_ordner: Path) -> Werkzeuge:
    """Echte Werkzeuge: gh als `helfer_konto` mit Token-Datei, Helfer und Klon-Git mit `DEBUGGER_PRAEFIX`.

    `push` findet das Repo zum Worktree über dieselbe Pfadregel wie Runde und Nachlauf (`wt_pfad`)
    und pusht über den Spiegel (Pin-Push mit `nur_gitlink`, R39); `holen` legt je Repo den Spiegel
    an und holt ihn von GitHub; `claw_aufraeumen` ist `claw_prozesse_beenden` (R44).

    Start fail-closed (Punkt 3): ohne Token-Datei `TokenDateiFehlt`, ohne ausführbares
    `DEBUGGER_ASKPASS` `AskpassFehlt`, als root ohne `DEBUGGER_PRAEFIX` `RootOhnePraefix`.
    """
    roh = os.environ.get("DEBUGGER_TOKEN_DATEI", "").strip()
    if not roh:
        raise TokenDateiFehlt()
    token_datei = Path(roh)
    _askpass_pruefen()
    praefix = tuple(os.environ.get("DEBUGGER_PRAEFIX", "").split())
    if not praefix and _ist_root():
        raise RootOhnePraefix()
    git_praefix_setzen(praefix)
    repos = tuple(konfig.repos)

    def holen() -> None:
        for repo in repos:
            pfad = spiegel_pfad(spiegel_ordner, repo)
            spiegel_bereit(pfad)
            spiegel_holen(pfad, repo, token_datei)

    def push(wt: Path, branch: str, *, nur_gitlink: tuple[str, str] | None = None) -> None:
        treffer = [r for r in repos if wt_pfad(ordner, r, branch) == wt]
        if len(treffer) != 1:
            raise ValueError("repo_zum_worktree_unklar")
        repo = treffer[0]
        spiegel_push(spiegel_pfad(spiegel_ordner, repo), wt, branch, repo, token_datei, basis=konfig.basis,
                     nur_gitlink=nur_gitlink)

    freigabe_url = os.environ.get("FREIGABE_URL", "").strip()
    freigabe_datei = os.environ.get("FREIGABE_SCHLUESSEL_DATEI", "").strip()
    freigabe = None
    if freigabe_url and freigabe_datei and Path(freigabe_datei).is_file():
        freigabe = FreigabeEingang(freigabe_url, Path(freigabe_datei))

    return Werkzeuge(
        gh_lab=GitHub(konfig.helfer_konto, marke=DEBUGGER_MARKE, token_datei=token_datei),
        gh_lesen=GitHub(konfig.helfer_konto, marke=None, token_datei=token_datei),
        helfer=functools.partial(helfer_starten, praefix=praefix),
        frei_gb=frei_gb,
        uhr=lambda: datetime.now(timezone.utc),
        push=push,
        holen=holen,
        claw_aufraeumen=claw_prozesse_beenden,
        spiegel=spiegel_ordner,
        freigabe=freigabe,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """`selbstbild debugger durchgang|schleife <wurzel> --klon --ordner --spiegel`; Startfehler: Exit 2."""
    p = argparse.ArgumentParser(prog="selbstbild debugger",
                                description="Debugger-Claw: Wächter für Kommentare in Selbstbild-Issues und PRs.")
    arten = p.add_subparsers(dest="art", required=True)
    for name, hilfe in (("durchgang", "ein Durchgang, dann Ende"),
                        ("schleife", "Durchgang alle intervall_s Sekunden, ohne Ende")):
        a = arten.add_parser(name, help=hilfe)
        a.add_argument("wurzel", type=Path, help="Ordner mit selbstbild.yaml (Abschnitt debugger:)")
        a.add_argument("--klon", type=Path, required=True, help="Klon des issue_repo mit Submodulen")
        a.add_argument("--ordner", type=Path, required=True, help="Stand-Ordner (Stand, Budget, STOP, wt/)")
        a.add_argument("--spiegel", type=Path, required=True, help="Ordner der Spiegel (bare Repos)")
    try:
        args = p.parse_args(argv)
    except SystemExit as e:  # --help und Aufruffehler: Code zurück statt Prozessende
        return e.code if isinstance(e.code, int) else 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def fabrik(konfig: DebuggerKonfig) -> Werkzeuge:
        return werkzeuge_standard(konfig, args.ordner, args.spiegel)

    try:
        konfig = konfig_laden(args.wurzel)
        w = fabrik(konfig)  # Startfehler (Konfiguration, Token-Datei) sofort und klar
    except (ValueError, RuntimeError) as e:  # nur der Typ; `TokenDateiFehlt` sagt, was fehlt
        LOG.error("Start nicht möglich (%s)", type(e).__name__)
        return 2
    if args.art == "durchgang":
        return durchgang(args.wurzel, args.klon, args.ordner, w, konfig=konfig)
    schleife(args.wurzel, args.klon, args.ordner, fabrik)
