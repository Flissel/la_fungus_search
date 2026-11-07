"""Tests für den Wächter-Durchgang: Fake-GitHub, Fake-Helfer, echte temporäre Git-Repos."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import selbstbild.debugger.klon as klon_modul
import selbstbild.debugger.wachter as wachter
from selbstbild.debugger.helfer import HelferErgebnis
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.runde import Werkzeuge, wt_pfad
from selbstbild.debugger.spiegel import spiegel_pfad
from selbstbild.debugger.stand import DebuggerStand, stand_laden, stand_speichern
from selbstbild.debugger.wachter import STAND_DATEI, durchgang, schleife, werkzeuge_standard
from selbstbild.funde.github import DEBUGGER_MARKE, GitHub, GitHubFehler

ISSUE_REPO = "Flissel/Vibemind_V1"
OS_REPO = "Flissel/vibemind-os"
START = "2026-11-07T10:00:00Z"
SRC = Path(__file__).resolve().parents[3] / "src"
# aus Teilen gebaut, damit kein Schlüssel-Muster im Repo steht
TOKEN = "ghp" + "_" + "B" * 36


# ---------- Git ----------

def git(ort: Path, *args: str) -> str:
    erg = subprocess.run(["git", "-C", str(ort), *args], check=True, capture_output=True, text=True)
    return erg.stdout.strip()


def _repos(tmp: Path) -> Path:
    """Bare-Origin (steht für den Spiegel) mit einem Commit auf master und ein Klon davon."""
    origin = tmp / "origin.git"
    klon = tmp / "klon"
    for argv in (["git", "init", "-q", "--bare", "-b", "master", str(origin)],
                 ["git", "clone", "-q", "-c", "core.autocrlf=false", str(origin), str(klon)]):
        subprocess.run(argv, check=True, capture_output=True)
    for k, v in (("user.name", "T"), ("user.email", "t@example.invalid"), ("commit.gpgsign", "false")):
        git(klon, "config", k, v)
    git(klon, "symbolic-ref", "HEAD", "refs/heads/master")  # leerer Klon: Zweig nicht vom lokalen Standard
    (klon / "a.txt").write_text("a\n", encoding="utf-8", newline="\n")
    git(klon, "add", "-A")
    git(klon, "commit", "-q", "-m", "a")
    git(klon, "push", "-q", "origin", "master")
    return klon


# ---------- Fakes ----------

class FakeGitHub:
    """Protokolliert alle Aufrufe; Kommentare filtert es nach `since` wie GitHub."""

    def __init__(self, daten: dict[str, Any]) -> None:
        self.daten = daten
        self.aufrufe: list[tuple[Any, ...]] = []
        self.fehler: dict[str, BaseException] = {}

    def _rein(self, name: str, *args: Any) -> None:
        self.aufrufe.append((name, *args))
        if name in self.fehler:
            raise self.fehler[name]

    def namen(self, name: str) -> list[tuple[Any, ...]]:
        return [a for a in self.aufrufe if a[0] == name]

    def _seit(self, art: str, repo: str, seit: str) -> list[dict[str, Any]]:
        return [dict(k) for k in self.daten[art].get(repo, []) if k["created_at"] >= seit]

    def kommentare_seit(self, repo: str, seit: str) -> list[dict[str, Any]]:
        self._rein("kommentare_seit", repo, seit)
        return self._seit("kommentare", repo, seit)

    def review_kommentare_seit(self, repo: str, seit: str) -> list[dict[str, Any]]:
        self._rein("review_kommentare_seit", repo, seit)
        return self._seit("review_kommentare", repo, seit)

    def issues(self, repo: str) -> list[dict[str, Any]]:
        self._rein("issues", repo)
        return list(self.daten["issues"].values())

    def _api(self, methode: str, pfad: str, daten: dict[str, Any] | None = None) -> Any:
        self._rein("api", methode, pfad)
        teile = pfad.split("?")[0].split("/")
        if teile[-2] == "issues":
            return self.daten["issues"][int(teile[-1])]
        return []

    def reagieren(self, repo: str, kommentar_id: int, *, review: bool = False) -> None:
        self._rein("reagieren", repo, kommentar_id, review)

    def kommentieren(self, repo: str, nummer: int, text: str) -> None:
        self._rein("kommentieren", repo, nummer, text)


@dataclass
class FakeHelfer:
    aufrufe: list[Path] = field(default_factory=list)

    def __call__(self, wt: Path, prompt: str, sitzung: str | None, frist_s: float) -> HelferErgebnis:
        self.aufrufe.append(wt)
        return HelferErgebnis("s-1", "Das steht so in der Doku.", "antwort", None)


@dataclass
class Welt:
    tmp: Path
    klon: Path
    ordner: Path
    konfig: DebuggerKonfig
    daten: dict[str, Any]
    lab: FakeGitHub
    lesen: FakeGitHub
    helfer: FakeHelfer = field(default_factory=FakeHelfer)
    jetzt: datetime = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    frei: float | None = 8.0
    geholt: int = 0

    def holen(self) -> None:
        self.geholt += 1

    @property
    def w(self) -> Werkzeuge:
        return Werkzeuge(
            gh_lab=cast(GitHub, self.lab), gh_lesen=cast(GitHub, self.lesen), helfer=self.helfer,
            frei_gb=lambda: self.frei, uhr=lambda: self.jetzt, push=lambda wt, branch: None, holen=self.holen,
        )

    def durchgang(self, **kw: Any) -> int:
        return durchgang(self.tmp, self.klon, self.ordner, self.w, konfig=self.konfig, **kw)

    @property
    def stand_pfad(self) -> Path:
        return self.ordner / STAND_DATEI

    def stand(self) -> DebuggerStand:
        return stand_laden(self.stand_pfad)

    def issue(self, n: int) -> None:
        self.daten["issues"][n] = {"number": n, "title": f"[selbstbild] Fund {n} (vibemind)",
                                   "body": "<!-- selbstbild:repo=vibemind -->", "labels": [{"name": "selbstbild"}]}

    def kommentar(self, i: int, n: int = 318, erstellt: str = "2026-11-07T11:00:00Z", autor: str = "Flissel") -> None:
        self.daten["kommentare"].setdefault(ISSUE_REPO, []).append({
            "id": i, "issue_url": f"https://api.github.com/repos/{ISSUE_REPO}/issues/{n}",
            "user": {"login": autor}, "body": f"bitte prüfen {i}", "created_at": erstellt,
        })


@pytest.fixture
def welt(tmp_path: Path) -> Welt:
    daten: dict[str, Any] = {"issues": {}, "kommentare": {}, "review_kommentare": {}}
    w = Welt(tmp=tmp_path, klon=_repos(tmp_path), ordner=tmp_path / "stand",
             konfig=DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: "."}),
             daten=daten, lab=FakeGitHub(daten), lesen=FakeGitHub(daten))
    w.issue(318)
    w.issue(319)
    stand_speichern(w.stand_pfad, DebuggerStand(start=START, seit=START))
    return w


# ---------- Tests aus dem Brief ----------

def test_erster_durchgang_setzt_seit_ohne_runde(welt: Welt) -> None:
    welt.stand_pfad.unlink()
    welt.kommentar(1, erstellt="2026-11-07T11:30:00Z")
    assert welt.durchgang() == 0
    s = welt.stand()
    assert s.start == s.seit == "2026-11-07T12:00:00Z"
    assert "stand_fehlt" not in s.gemeldet
    assert welt.lesen.aufrufe == [] and welt.lab.aufrufe == [] and welt.helfer.aufrufe == [] and welt.geholt == 0
    # alles vor dem ersten Start gilt als verarbeitet
    welt.jetzt += timedelta(minutes=2)
    assert welt.durchgang() == 0
    assert welt.helfer.aufrufe == [] and welt.lab.aufrufe == [] and welt.geholt == 1


def test_durchgang_fuehrt_hoechstens_eine_runde(welt: Welt) -> None:
    welt.kommentar(1, 318, "2026-11-07T11:00:00Z")
    welt.kommentar(2, 319, "2026-11-07T11:10:00Z")
    assert welt.durchgang() == 0
    assert len(welt.helfer.aufrufe) == 1 and welt.helfer.aufrufe[0].name.startswith("Vibemind_V1-318-")
    s = welt.stand()
    assert s.verarbeitet == [1]
    assert s.seit == "2026-11-07T11:10:00Z"  # Kommentar 2 war nicht an der Reihe
    welt.jetzt += timedelta(minutes=5)
    assert welt.durchgang() == 0
    assert len(welt.helfer.aufrufe) == 2 and welt.helfer.aufrufe[1].name.startswith("Vibemind_V1-319-")
    s = welt.stand()
    assert s.verarbeitet == [1, 2] and s.seit == "2026-11-07T11:55:00Z"
    assert [a[2] for a in welt.lab.namen("kommentieren")] == [318, 319]


def test_durchgang_sperre_belegt_exit_3(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    """R34: Hält ein anderer Durchgang (z. B. ein zweiter Container) den Kernel-Lock, Exit 3."""
    monkeypatch.setenv("DEBUGGER_CLAW_UID", "1000")
    beendet: list[list[str]] = []
    vorher = welt.stand_pfad.read_bytes()
    welt.kommentar(1)
    with wachter.kernel_sperre(welt.ordner / wachter.KERNEL_SPERRE):
        assert welt.durchgang(laeufer=lambda argv, **kw: beendet.append(argv)) == 3
    assert welt.stand_pfad.read_bytes() == vorher and not (welt.ordner / wachter.SPERRE).exists()
    assert welt.lesen.aufrufe == [] and welt.helfer.aufrufe == [] and welt.geholt == 0
    assert beendet == []  # die Prozesse des anderen Laufs bleiben unberührt
    assert welt.durchgang(laeufer=lambda argv, **kw: subprocess.CompletedProcess(argv, 1)) == 0  # frei


def test_durchgang_klon_nicht_sauber_keine_runde(welt: Welt, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="selbstbild.debugger")
    welt.kommentar(1)
    (welt.klon / "fremd.txt").write_text("nicht von mir\n", encoding="utf-8")
    assert welt.durchgang() == 0
    welt.jetzt += timedelta(minutes=2)
    assert welt.durchgang() == 0
    assert welt.helfer.aufrufe == [] and welt.lab.aufrufe == [] and welt.lesen.aufrufe == []
    assert welt.stand().gemeldet == ["klon:nicht_sauber"]
    assert caplog.text.count("nicht_sauber") == 1
    assert (welt.klon / "fremd.txt").read_text(encoding="utf-8") == "nicht von mir\n"  # nichts verworfen
    # Grund verschwindet: Eintrag weg, die Runde läuft
    (welt.klon / "fremd.txt").unlink()
    welt.jetzt += timedelta(minutes=2)
    assert welt.durchgang() == 0
    assert welt.stand().gemeldet == [] and len(welt.helfer.aufrufe) == 1


def test_wartender_kommentar_haelt_seit_zurueck(welt: Welt) -> None:
    welt.kommentar(1, erstellt="2026-11-07T11:00:00Z")
    welt.frei = 1.0
    for stunde in (12, 13):
        welt.jetzt = datetime(2026, 11, 7, stunde, 0, tzinfo=timezone.utc)
        assert welt.durchgang() == 0
        assert welt.helfer.aufrufe == []
        assert welt.stand().seit == "2026-11-07T11:00:00Z"
    welt.frei = 8.0
    welt.jetzt = datetime(2026, 11, 7, 14, 0, tzinfo=timezone.utc)
    assert welt.durchgang() == 0
    assert len(welt.helfer.aufrufe) == 1 and welt.stand().verarbeitet == [1]
    assert welt.stand().seit == "2026-11-07T13:50:00Z"
    gelesen = [a[2] for a in welt.lesen.namen("kommentare_seit")]
    assert gelesen == ["2026-11-07T09:50:00Z", "2026-11-07T10:50:00Z", "2026-11-07T10:50:00Z"]
    wartet = [a for a in welt.lab.namen("kommentieren") if "wartet (speicher)" in a[3]]
    assert len(wartet) == 1  # je Grund und Tag nur eine Meldung


def test_cli_debugger_hilfe() -> None:
    env = dict(os.environ, PYTHONPATH=str(SRC))
    erg = subprocess.run([sys.executable, "-m", "selbstbild", "debugger", "--help"], capture_output=True, env=env,
                         timeout=120)
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    assert b"durchgang" in erg.stdout and b"schleife" in erg.stdout
    assert wachter.main(["durchgang", "--help"]) == 0


# ---------- Entscheidungen ----------

def test_stand_bei_ausnahme_der_runde_gespeichert(welt: Welt, caplog: pytest.LogCaptureFixture) -> None:
    welt.kommentar(1)
    welt.lab.fehler["kommentieren"] = RuntimeError("GEHEIM-DETAIL")
    assert welt.durchgang() == 1
    fund = welt.stand().funde["318"]
    assert fund.sitzung == "s-1"  # was die Runde im Speicher fortgeschrieben hat, ist gespeichert
    assert "RuntimeError" in caplog.text and "GEHEIM" not in caplog.text
    assert welt.stand().seit == START  # Lesemarke rückt nach einem Abbruch nicht vor


def test_github_fehler_beim_lesen_exit_0(welt: Welt, caplog: pytest.LogCaptureFixture) -> None:
    welt.kommentar(1)
    welt.lesen.fehler["kommentare_seit"] = GitHubFehler("http_502")
    assert welt.durchgang() == 0
    assert welt.stand().seit == START and welt.helfer.aufrufe == []
    assert "GitHubFehler" in caplog.text


def test_kaputter_stand_bleibt_als_kopie(welt: Welt) -> None:
    welt.stand_pfad.write_text("{kaputt", encoding="utf-8")
    assert welt.durchgang() == 0
    kopien = list(welt.ordner.glob(f"{STAND_DATEI}.kaputt-*"))
    assert len(kopien) == 1 and kopien[0].read_text(encoding="utf-8") == "{kaputt"
    s = welt.stand()
    assert s.start == s.seit == "2026-11-07T12:00:00Z" and "stand_kaputt" in s.gemeldet


@pytest.mark.parametrize("pid", ["eigene", "lebende_fremde"])
def test_alte_sperrdatei_bei_freiem_kernel_lock_uebernommen(welt: Welt, pid: str) -> None:
    """R34: Nach einem Neustart (oft dieselbe PID) blockiert eine alte Sperrdatei nicht; ohne Kernel-Lock
    läuft kein anderer Durchgang, sie ist also verwaist."""
    zahl = os.getpid() if pid == "eigene" else os.getppid()
    sperrdatei = welt.ordner / wachter.SPERRE
    sperrdatei.write_text(json.dumps({"pid": zahl, "start": "2026-11-07T11:59:00"}), encoding="utf-8")
    welt.kommentar(1)
    assert welt.durchgang() == 0
    assert len(welt.helfer.aufrufe) == 1 and not sperrdatei.exists()


def test_stand_unlesbar_exit_1_ohne_speichern(welt: Welt, caplog: pytest.LogCaptureFixture) -> None:
    """R36: Ein Lesefehler (hier: Ordner statt Datei) legt nichts beiseite und überschreibt nichts."""
    welt.stand_pfad.unlink()
    welt.stand_pfad.mkdir()
    assert welt.durchgang() == 1
    assert welt.stand_pfad.is_dir() and not list(welt.ordner.glob(f"{STAND_DATEI}*.kaputt-*"))
    assert not list(welt.ordner.glob(f"{STAND_DATEI}.tmp"))  # kein Speicherversuch
    assert "StandUnlesbar" in caplog.text and welt.lesen.aufrufe == [] and welt.geholt == 0


def test_holen_vor_nachziehen(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    reihenfolge: list[Any] = []
    welt.konfig = DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: ".", OS_REPO: "vibemind-os"})

    def nachziehen(klon: Path, basis: str, submodule: list[str]) -> str:
        reihenfolge.append(("nachziehen", klon, basis, list(submodule)))
        return "falscher_zweig"

    monkeypatch.setattr(wachter, "nachziehen", nachziehen)
    monkeypatch.setattr(welt, "holen", lambda: reihenfolge.append("holen"))
    assert welt.durchgang() == 0
    assert reihenfolge == ["holen", ("nachziehen", welt.klon, "master", ["vibemind-os"])]
    assert welt.stand().gemeldet == ["klon:falscher_zweig"]


def test_claw_prozesse_nach_dem_speichern_beendet(welt: Welt, monkeypatch: pytest.MonkeyPatch,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    aufrufe: list[tuple[list[str], list[int]]] = []
    code = 1

    def laeufer(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        aufrufe.append((argv, welt.stand().verarbeitet))
        return subprocess.CompletedProcess(argv, code, b"", b"")

    monkeypatch.delenv("DEBUGGER_CLAW_UID", raising=False)
    welt.jetzt = datetime(2026, 11, 7, 10, 30, tzinfo=timezone.utc)
    assert welt.durchgang(laeufer=laeufer) == 0
    assert aufrufe == []
    welt.kommentar(1)
    monkeypatch.setenv("DEBUGGER_CLAW_UID", "1000")
    welt.jetzt = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    assert welt.durchgang(laeufer=laeufer) == 0
    assert aufrufe == [(["pkill", "-KILL", "-U", "1000"], [1])]  # nach dem Speichern
    assert "pkill" not in caplog.text
    code = 2
    welt.jetzt += timedelta(minutes=2)
    assert welt.durchgang(laeufer=laeufer) == 0
    assert len(aufrufe) == 2 and "pkill" in caplog.text


# ---------- Werkzeuge ----------

@pytest.fixture
def token_umgebung(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    datei = tmp_path / "lab_token"
    datei.write_text(TOKEN + "\n", encoding="utf-8")
    monkeypatch.setenv("DEBUGGER_TOKEN_DATEI", str(datei))
    monkeypatch.setattr(klon_modul, "_praefix", ())
    return datei


def test_werkzeuge_standard_praefix_aus_umgebung(tmp_path: Path, token_umgebung: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    praefix = ["setpriv", "--reuid=1000", "--regid=1000", "--init-groups", "--", "env", "HOME=/home/claw"]
    monkeypatch.setenv("DEBUGGER_PRAEFIX", " ".join(praefix))
    konfig = DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: "."})
    w = werkzeuge_standard(konfig, tmp_path / "stand", tmp_path / "spiegel")
    assert klon_modul.git_praefix() == tuple(praefix)
    gesehen: list[list[str]] = []

    def laeufer(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        gesehen.append(argv)
        kw["stdout"].write(json.dumps({"session_id": "s", "result": "ok"}).encode())
        return subprocess.CompletedProcess(argv, 0)

    w.helfer(tmp_path, "prompt", None, 5, laeufer=laeufer)
    assert gesehen[0][:len(praefix)] == praefix and gesehen[0][len(praefix)] == "claude"
    assert (w.gh_lab.konto, w.gh_lab.marke, w.gh_lab.token_datei) == ("Vibemind-LAB", DEBUGGER_MARKE, token_umgebung)
    assert (w.gh_lesen.konto, w.gh_lesen.marke, w.gh_lesen.token_datei) == ("Vibemind-LAB", None, token_umgebung)
    assert TOKEN not in os.environ.values()
    monkeypatch.setenv("DEBUGGER_PRAEFIX", "")
    werkzeuge_standard(konfig, tmp_path / "stand", tmp_path / "spiegel")
    assert klon_modul.git_praefix() == ()


def test_werkzeuge_standard_push_und_holen(tmp_path: Path, token_umgebung: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUGGER_PRAEFIX", raising=False)
    aufrufe: list[tuple[Any, ...]] = []
    monkeypatch.setattr(wachter, "spiegel_bereit", lambda pfad: aufrufe.append(("bereit", pfad)))
    monkeypatch.setattr(wachter, "spiegel_holen", lambda pfad, repo, td: aufrufe.append(("holen", pfad, repo, td)))
    monkeypatch.setattr(wachter, "spiegel_push",
                        lambda *a, basis, nur_gitlink: aufrufe.append(("push", *a, basis, nur_gitlink)))
    ordner, spiegel = tmp_path / "stand", tmp_path / "spiegel"
    konfig = DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: ".", OS_REPO: "vibemind-os"}, basis="main")
    w = werkzeuge_standard(konfig, ordner, spiegel)
    w.holen()
    aussen, os_sp = spiegel_pfad(spiegel, ISSUE_REPO), spiegel_pfad(spiegel, OS_REPO)
    assert aufrufe == [("bereit", aussen), ("holen", aussen, ISSUE_REPO, token_umgebung),
                       ("bereit", os_sp), ("holen", os_sp, OS_REPO, token_umgebung)]
    aufrufe.clear()
    wt = wt_pfad(ordner, OS_REPO, "debugger/5-fix")
    w.push(wt, "debugger/5-fix")
    assert aufrufe == [("push", os_sp, wt, "debugger/5-fix", OS_REPO, token_umgebung, "main", None)]
    aufrufe.clear()
    pin_wt = wt_pfad(ordner, ISSUE_REPO, "debugger/5-pin")
    w.push(pin_wt, "debugger/5-pin", nur_gitlink=("vibemind-os", "a" * 40))  # R39: an spiegel_push durchgereicht
    assert aufrufe == [("push", aussen, pin_wt, "debugger/5-pin", ISSUE_REPO, token_umgebung, "main",
                        ("vibemind-os", "a" * 40))]
    assert w.spiegel == spiegel and w.claw_aufraeumen is wachter.claw_prozesse_beenden  # R39, R44
    with pytest.raises(ValueError):
        w.push(tmp_path / "anderswo", "debugger/5-fix")


def test_ohne_token_datei_klarer_startfehler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("DEBUGGER_TOKEN_DATEI", raising=False)
    with pytest.raises(wachter.TokenDateiFehlt):
        werkzeuge_standard(DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: "."}), tmp_path, tmp_path)
    (tmp_path / "selbstbild.yaml").write_text(
        f"debugger:\n  issue_repo: {ISSUE_REPO}\n  repos:\n    {ISSUE_REPO}: .\n", encoding="utf-8")
    ort = str(tmp_path)
    assert wachter.main(["schleife", ort, "--klon", ort, "--ordner", ort, "--spiegel", ort]) == 2
    assert "TokenDateiFehlt" in caplog.text


class _Halt(Exception):
    pass


def test_schleife_baut_werkzeuge_je_durchgang(welt: Welt, caplog: pytest.LogCaptureFixture) -> None:
    (welt.tmp / "selbstbild.yaml").write_text(
        f"debugger:\n  issue_repo: {ISSUE_REPO}\n  repos:\n    {ISSUE_REPO}: .\n  intervall_s: 7\n", encoding="utf-8")
    welt.stand_pfad.unlink()
    gebaut: list[DebuggerKonfig] = []
    geschlafen: list[float] = []

    def fabrik(konfig: DebuggerKonfig) -> Werkzeuge:
        gebaut.append(konfig)
        if len(gebaut) == 1:
            raise RuntimeError("GEHEIM-DETAIL")
        return welt.w

    def schlafen(s: float) -> None:
        geschlafen.append(s)
        if len(geschlafen) == 2:
            raise _Halt

    with pytest.raises(_Halt):
        schleife(welt.tmp, welt.klon, welt.ordner, fabrik, schlafen=schlafen)
    assert len(gebaut) == 2 and geschlafen == [7, 7]
    assert welt.stand().start == "2026-11-07T12:00:00Z"  # zweiter Durchgang lief (erster Start)
    assert "RuntimeError" in caplog.text and "GEHEIM" not in caplog.text
