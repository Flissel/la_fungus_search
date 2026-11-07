"""Tests fuer klon.py mit echten temporaeren Repos (bares origin + Klone)."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterator, cast

import pytest

from selbstbild.debugger.klon import (
    GitFehler,
    KonfliktFehler,
    WorktreeFehler,
    aufraeumen,
    branch_name,
    nachziehen,
    voraus,
    worktree_bereit,
)

Umgebung = tuple[Path, Path, Path]


def git(ort: Path, *args: str) -> str:
    erg = subprocess.run(
        ["git", "-C", str(ort), *args], check=True, capture_output=True, text=True
    )
    return erg.stdout.strip()


def _konfig(ort: Path) -> None:
    git(ort, "config", "user.name", "T")
    git(ort, "config", "user.email", "t@example.invalid")
    git(ort, "config", "core.autocrlf", "false")
    git(ort, "config", "commit.gpgsign", "false")


def commit(ort: Path, datei: str, inhalt: str) -> None:
    (ort / datei).write_text(inhalt, encoding="utf-8", newline="\n")
    git(ort, "add", "-A")
    git(ort, "commit", "-q", "-m", datei)


def klonen(origin: Path, ziel: Path) -> Path:
    subprocess.run(
        ["git", "clone", "-q", "-c", "core.autocrlf=false", str(origin), str(ziel)], check=True, capture_output=True
    )
    _konfig(ziel)
    return ziel


@pytest.fixture
def umgebung(tmp_path: Path) -> Umgebung:
    """(origin, klon, zweitklon); Basiszweig `master` mit einem Commit."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "master", str(origin)],
        check=True, capture_output=True,
    )
    start = tmp_path / "start"
    start.mkdir()
    git(start, "init", "-q", "-b", "master")
    _konfig(start)
    commit(start, "a.txt", "a\n")
    git(start, "remote", "add", "origin", str(origin))
    git(start, "push", "-q", "origin", "master")
    klon = klonen(origin, tmp_path / "klon")
    zweit = klonen(origin, tmp_path / "zweit")
    return origin, klon, zweit


def _wt_mit_fremdem_commit(klon: Path, zweit: Path) -> Path:
    """Legt Worktree an, pusht den Zweig und schiebt von `zweit` einen Commit nach."""
    wt = klon.parent / "wt" / "x-1"
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    _konfig(wt)
    git(wt, "push", "-q", "origin", "debugger/1-x")
    git(zweit, "fetch", "-q", "origin")
    git(zweit, "checkout", "-q", "-b", "debugger/1-x", "origin/debugger/1-x")
    commit(zweit, "fremd.txt", "f\n")
    git(zweit, "push", "-q", "origin", "debugger/1-x")
    return wt


def test_nachziehen_fast_forward(umgebung: Umgebung) -> None:
    _, klon, zweit = umgebung
    commit(zweit, "b.txt", "b\n")
    git(zweit, "push", "-q", "origin", "master")
    assert nachziehen(klon, "master", []) is None
    assert (klon / "b.txt").exists()


def test_nachziehen_dreckig_meldet_und_laesst(umgebung: Umgebung) -> None:
    _, klon, zweit = umgebung
    commit(zweit, "b.txt", "b\n")
    git(zweit, "push", "-q", "origin", "master")
    (klon / "a.txt").write_text("lokal geaendert\n", encoding="utf-8", newline="\n")
    assert nachziehen(klon, "master", []) == "nicht_sauber"
    assert (klon / "a.txt").read_text(encoding="utf-8") == "lokal geaendert\n"
    assert not (klon / "b.txt").exists()


def test_nachziehen_falscher_zweig(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    git(klon, "checkout", "-q", "-b", "andere")
    assert nachziehen(klon, "master", []) == "falscher_zweig"
    assert git(klon, "symbolic-ref", "--short", "HEAD") == "andere"
    git(klon, "checkout", "-q", "--detach")
    assert nachziehen(klon, "master", []) == "falscher_zweig"


def test_nachziehen_kein_fast_forward(umgebung: Umgebung) -> None:
    _, klon, zweit = umgebung
    commit(zweit, "b.txt", "b\n")
    git(zweit, "push", "-q", "origin", "master")
    commit(klon, "c.txt", "c\n")
    assert nachziehen(klon, "master", []) == "kein_fast_forward"


def test_nachziehen_submodul_fehler(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    assert nachziehen(klon, "master", ["gibt-es-nicht"]) == "submodul_fehler"


def test_branch_name() -> None:
    titel = "[selbstbild] Doku-Widerspruch: schedule.status …"
    name = branch_name(315, titel)
    assert name == "debugger/315-doku-widerspruch-schedule-sta"
    kurz = name.split("-", 1)[1]
    assert len(kurz) <= 30 and not kurz.endswith("-")
    assert branch_name(7, "[selbstbild] !!!") == "debugger/7-fund"
    assert branch_name(8, "Ohne Praefix") == "debugger/8-ohne-praefix"


def test_worktree_neu_aus_basis(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    wt = klon.parent / "wt" / "x-1"
    erg = worktree_bereit(klon, wt, "debugger/1-x", "master")
    assert erg == wt and (wt / "a.txt").exists()
    assert git(wt, "symbolic-ref", "--short", "HEAD") == "debugger/1-x"
    # Zweiter Aufruf: kein Fehler, nichts zu tun.
    assert worktree_bereit(klon, wt, "debugger/1-x", "master") == wt


def test_worktree_neu_aus_remote_zweig(umgebung: Umgebung) -> None:
    _, klon, zweit = umgebung
    git(zweit, "checkout", "-q", "-b", "debugger/2-y")
    commit(zweit, "y.txt", "y\n")
    git(zweit, "push", "-q", "origin", "debugger/2-y")
    wt = klon.parent / "wt" / "y-2"
    worktree_bereit(klon, wt, "debugger/2-y", "master")
    assert (wt / "y.txt").exists()


def test_worktree_altzweig_ohne_worktree_wird_wiederverwendet(
    umgebung: Umgebung,
) -> None:
    _, klon, _ = umgebung
    git(klon, "branch", "debugger/3-z")
    wt = klon.parent / "wt" / "z-3"
    worktree_bereit(klon, wt, "debugger/3-z", "master")
    assert git(wt, "symbolic-ref", "--short", "HEAD") == "debugger/3-z"


def test_worktree_holt_fremden_commit_nach(umgebung: Umgebung) -> None:
    _, klon, zweit = umgebung
    wt = _wt_mit_fremdem_commit(klon, zweit)
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    assert (wt / "fremd.txt").exists()


def test_worktree_konflikt_wirft(umgebung: Umgebung) -> None:
    _, klon, zweit = umgebung
    wt = _wt_mit_fremdem_commit(klon, zweit)
    commit(wt, "eigen.txt", "e\n")
    kopf = git(wt, "rev-parse", "HEAD")
    with pytest.raises(KonfliktFehler):
        worktree_bereit(klon, wt, "debugger/1-x", "master")
    assert git(wt, "rev-parse", "HEAD") == kopf
    assert git(wt, "status", "--porcelain") == ""
    assert not (wt / "fremd.txt").exists()
    gitdir = Path(git(wt, "rev-parse", "--absolute-git-dir"))
    assert not (gitdir / "MERGE_HEAD").exists()


def test_worktree_lokal_voraus_ist_ok(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    wt = klon.parent / "wt" / "x-1"
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    _konfig(wt)
    git(wt, "push", "-q", "origin", "debugger/1-x")
    commit(wt, "eigen.txt", "e\n")
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    assert (wt / "eigen.txt").exists()


def test_worktree_ohne_remote_zweig_unveraendert(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    wt = klon.parent / "wt" / "x-1"
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    _konfig(wt)
    commit(wt, "eigen.txt", "e\n")
    kopf = git(wt, "rev-parse", "HEAD")
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    assert git(wt, "rev-parse", "HEAD") == kopf


def test_worktree_verweigert_fremden_zweig(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    wt = klon.parent / "wt" / "m"
    with pytest.raises(ValueError):
        worktree_bereit(klon, wt, "master", "master")
    assert not wt.exists()


def test_aufraeumen_nur_debugger(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    wt = klon.parent / "wt" / "x-1"
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    with pytest.raises(ValueError):
        aufraeumen(klon, wt, "master")
    assert wt.exists()
    aufraeumen(klon, wt, "debugger/1-x")
    assert not wt.exists()
    assert "debugger/1-x" not in git(klon, "branch", "--list")
    # Zweiter Aufruf (alles schon weg) ist kein Fehler.
    aufraeumen(klon, wt, "debugger/1-x")


def test_aufraeumen_worktree_schon_weg(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    wt = klon.parent / "wt" / "x-1"
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    shutil.rmtree(wt)
    aufraeumen(klon, wt, "debugger/1-x")
    assert "debugger/1-x" not in git(klon, "branch", "--list")
    assert "x-1" not in git(klon, "worktree", "list")


def test_voraus_zaehlt(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    wt = klon.parent / "wt" / "x-1"
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    _konfig(wt)
    assert voraus(wt, "debugger/1-x") == 0
    commit(wt, "e1.txt", "1\n")
    commit(wt, "e2.txt", "2\n")
    assert voraus(wt, "debugger/1-x") == 2  # Rueckfall auf origin/master
    git(wt, "push", "-q", "origin", "debugger/1-x")
    git(wt, "fetch", "-q", "origin")
    assert voraus(wt, "debugger/1-x") == 0
    commit(wt, "e3.txt", "3\n")
    assert voraus(wt, "debugger/1-x") == 1


def test_git_fehler_traegt_nur_unterbefehl(tmp_path: Path) -> None:
    with pytest.raises(GitFehler) as ei:
        voraus(tmp_path / "gibt-es-nicht", "debugger/1-x")
    assert str(ei.value).startswith("git ")


def _konfig_wert(ort: Path, schluessel: str) -> str:
    erg = subprocess.run(
        ["git", "-C", str(ort), "config", "--get", schluessel],
        capture_output=True, text=True,
    )
    return erg.stdout.strip()


def test_worktree_ohne_upstream(umgebung: Umgebung) -> None:
    _, klon, zweit = umgebung
    wt = klon.parent / "wt" / "x-1"
    worktree_bereit(klon, wt, "debugger/1-x", "master")
    assert _konfig_wert(klon, "branch.debugger/1-x.merge") == ""
    git(zweit, "checkout", "-q", "-b", "debugger/2-y")
    commit(zweit, "y.txt", "y\n")
    git(zweit, "push", "-q", "origin", "debugger/2-y")
    wt2 = klon.parent / "wt" / "y-2"
    worktree_bereit(klon, wt2, "debugger/2-y", "master")
    assert _konfig_wert(klon, "branch.debugger/2-y.merge") == ""


def test_worktree_pruefung_normales_verzeichnis(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    wt = klon.parent / "wt" / "plain"
    wt.mkdir(parents=True)
    (wt / "datei.txt").write_text("x", encoding="utf-8")
    with pytest.raises(WorktreeFehler):
        worktree_bereit(klon, wt, "debugger/1-x", "master")
    assert [p.name for p in wt.iterdir()] == ["datei.txt"]
    with pytest.raises(WorktreeFehler):
        aufraeumen(klon, wt, "debugger/1-x")
    assert wt.exists()


def test_worktree_pruefung_haupt_checkout(umgebung: Umgebung) -> None:
    _, klon, _ = umgebung
    with pytest.raises(WorktreeFehler):
        worktree_bereit(klon, klon, "debugger/1-x", "master")
    with pytest.raises(WorktreeFehler):
        aufraeumen(klon, klon, "debugger/1-x")
    assert (klon / "a.txt").exists()


def test_worktree_pruefung_anderer_zweig(umgebung: Umgebung) -> None:
    _, klon, zweit = umgebung
    wt = _wt_mit_fremdem_commit(klon, zweit)
    git(wt, "checkout", "-q", "-b", "debugger/9-anders")
    with pytest.raises(WorktreeFehler):
        worktree_bereit(klon, wt, "debugger/1-x", "master")
    assert not (wt / "fremd.txt").exists()
    with pytest.raises(WorktreeFehler):
        aufraeumen(klon, wt, "debugger/1-x")
    assert wt.exists()
    git(wt, "checkout", "-q", "--detach")
    with pytest.raises(WorktreeFehler):
        worktree_bereit(klon, wt, "debugger/1-x", "master")


class _Ein(io.BytesIO):
    """stdin-Ersatz, der seinen Inhalt nach `close()` behält."""

    def close(self) -> None:
        self.inhalt = self.getvalue()


class FakePopen:
    """Popen-Ersatz: protokolliert argv/kw; `haengt=True` simuliert einen Prozess, der nie endet."""

    aufrufe: list[dict[str, Any]] = []
    haengt = False

    def __init__(self, argv: list[str], **kw: Any) -> None:
        FakePopen.aufrufe.append({"argv": argv, **kw})
        self.pid = 4242
        self.stdout = io.BytesIO(b"")
        self.stdin = _Ein() if kw.get("stdin") == subprocess.PIPE else None
        self.getoetet = False

    def wait(self, timeout: float | None = None) -> int:
        if FakePopen.haengt and not self.getoetet:
            raise subprocess.TimeoutExpired("git", timeout or 0)
        return 0

    def kill(self) -> None:
        self.getoetet = True


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> type[FakePopen]:
    from selbstbild.debugger import klon as modul

    FakePopen.aufrufe, FakePopen.haengt = [], False
    monkeypatch.setattr(modul.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(modul.os, "killpg", lambda pid, sig: None, raising=False)
    return FakePopen


def test_git_timeout_wird_gitfehler(umgebung: Umgebung, fake_popen: type[FakePopen]) -> None:
    from selbstbild.debugger import klon as modul

    _, klon, _ = umgebung
    fake_popen.haengt = True
    with pytest.raises(GitFehler) as ei:
        nachziehen(klon, "master", [])
    assert ei.value.code == modul.TIMEOUT_CODE
    assert ei.value.unterbefehl == "fetch"
    (gesehen,) = fake_popen.aufrufe
    assert gesehen["stdin"] == subprocess.DEVNULL and gesehen["stderr"] == subprocess.DEVNULL
    assert gesehen["start_new_session"] == (os.name != "nt")
    env = gesehen["env"]
    assert isinstance(env, dict) and env["GIT_TERMINAL_PROMPT"] == "0"


# ---------- Präfix (Task 9a) ----------

@pytest.fixture
def ohne_praefix() -> Iterator[None]:
    from selbstbild.debugger import klon as modul

    alt = modul.git_praefix()
    modul.git_praefix_setzen(())
    yield
    modul.git_praefix_setzen(alt)


def test_git_praefix_wird_vorangestellt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ohne_praefix: None, fake_popen: type[FakePopen]
) -> None:
    from selbstbild.debugger import klon as modul

    aufrufe = fake_popen.aufrufe
    for k in ("DEBUGGER_GIT_TOKEN", "DEBUGGER_ASKPASS", "GH_TOKEN", "GITHUB_TOKEN", "GIT_ASKPASS"):
        monkeypatch.setenv(k, "geheim")
    assert modul.git_praefix() == ()
    modul.git_lauf(tmp_path, "status")
    assert aufrufe[-1]["argv"] == ["git", *modul.KLON_HAERTUNG, "-C", str(tmp_path), "status"]
    praefix = ["setpriv", "--reuid=1000", "--regid=1000", "--init-groups", "--"]
    modul.git_praefix_setzen(praefix)
    praefix.append("veraendert")  # der Präfix ist eine Kopie
    assert modul.git_praefix() == ("setpriv", "--reuid=1000", "--regid=1000", "--init-groups", "--")
    modul.git_lauf(tmp_path, "status")
    modul.git_roh(tmp_path, "cat-file", "--batch", eingabe=b"x\n")
    for a in aufrufe[1:]:
        assert a["argv"][:6] == [*modul.git_praefix(), "git"]
        env = a["env"]
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        for k in ("DEBUGGER_GIT_TOKEN", "DEBUGGER_ASKPASS", "GH_TOKEN", "GITHUB_TOKEN", "GIT_ASKPASS"):
            assert k not in env
    assert aufrufe[-1]["stdin"] == subprocess.PIPE
    with (tmp_path / "aus").open("wb") as aus:
        modul.git_in_datei(tmp_path, "bundle", ziel=aus, grenze=10)
    assert aufrufe[-1]["argv"][:6] == [*modul.git_praefix(), "git"]
    assert all(a["start_new_session"] == (os.name != "nt") for a in aufrufe)
    with pytest.raises(TypeError):
        modul.git_praefix_setzen("setpriv --reuid=1000")


@pytest.mark.parametrize("laeufer", ["git_in_datei", "git_lauf", "git_roh"])
def test_git_in_datei_timeout_trotz_enkel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ohne_praefix: None, enkel: Any, laeufer: str
) -> None:
    """S6/R31: ein Enkel hält stdout offen; die Frist greift trotzdem binnen Sekunden."""
    from selbstbild.debugger import klon as modul

    monkeypatch.setattr(modul, "GIT_TIMEOUT_S", 2)
    modul.git_praefix_setzen(enkel.praefix(0))
    beginn = time.monotonic()
    with pytest.raises(GitFehler) as ei, (tmp_path / "aus").open("wb") as aus:
        if laeufer == "git_in_datei":
            modul.git_in_datei(tmp_path, "bundle", ziel=aus, grenze=1000)
        elif laeufer == "git_lauf":
            modul.git_lauf(tmp_path, "status")
        else:
            modul.git_roh(tmp_path, "cat-file", "--batch", eingabe=b"x\n")
    dauer = time.monotonic() - beginn
    assert ei.value.code == modul.TIMEOUT_CODE and dauer < 15, dauer
    enkel.tot_pruefen()


def test_git_in_datei_groesse_trotz_enkel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ohne_praefix: None, enkel: Any
) -> None:
    """Die Größengrenze beendet den Lauf sofort, auch wenn ein Enkel die Pipe offen hält."""
    from selbstbild.debugger import klon as modul

    monkeypatch.setattr(modul, "GIT_TIMEOUT_S", 60)  # nur die Grenze darf hier greifen
    modul.git_praefix_setzen(enkel.praefix(60, ausgabe=5000))
    beginn = time.monotonic()
    with pytest.raises(GitFehler) as ei, (tmp_path / "aus").open("wb") as aus:
        modul.git_in_datei(tmp_path, "bundle", ziel=aus, grenze=1000)
    dauer = time.monotonic() - beginn
    assert ei.value.code == modul.GROESSE_CODE and dauer < 15, dauer
    assert (tmp_path / "aus").stat().st_size <= 1000
    enkel.tot_pruefen()


# Gibt 3 MB in 64-KB-Stücken aus, ignoriert eine geschlossene Pipe und endet immer mit 0.
_VIEL_AUSGABE = (
    "import os, sys\n"
    "try:\n"
    "    for _ in range(48):\n"
    "        sys.stdout.buffer.write(b'x' * 65536)\n"
    "        sys.stdout.buffer.flush()\n"
    "except OSError:\n"
    "    pass\n"
    "os._exit(0)\n"
)


class _KaputtesZiel(io.BytesIO):
    """Ziel, dessen `write` ab dem n-ten Aufruf scheitert (wie eine volle Platte)."""

    def __init__(self, ab: int) -> None:
        super().__init__()
        self.ab, self.n = ab, 0

    def write(self, daten: Any) -> int:
        self.n += 1
        if self.n >= self.ab:
            raise OSError(28, "No space left on device")
        return super().write(daten)


@pytest.mark.parametrize("ab", [1, 2])
def test_git_in_datei_schreibfehler_ist_kein_erfolg(ohne_praefix: None, tmp_path: Path, ab: int) -> None:
    """Fix-Runde 3: ein Schreibfehler im Leser ergibt GitFehler, auch wenn der Prozess mit 0 endet."""
    from selbstbild.debugger import klon as modul

    modul.git_praefix_setzen([sys.executable, "-c", _VIEL_AUSGABE])
    ziel = _KaputtesZiel(ab)
    with pytest.raises(GitFehler) as ei:
        modul.git_in_datei(tmp_path, "bundle", ziel=ziel, grenze=50_000_000)
    assert ei.value.code == modul.LESE_CODE
    assert ziel.n >= ab


def test_gruppe_toeten_faellt_auf_kill_zurueck(monkeypatch: pytest.MonkeyPatch) -> None:
    from selbstbild.debugger import klon as modul

    class P:
        pid = 4242
        getoetet = 0

        def kill(self) -> None:
            P.getoetet += 1
            raise OSError("schon weg")

    def verboten(pid: int, sig: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    # nur das Modul sieht POSIX; das echte os bleibt für pytest unverändert
    monkeypatch.setattr(modul, "os", types.SimpleNamespace(name="posix", killpg=verboten))
    monkeypatch.setattr(modul, "signal", types.SimpleNamespace(SIGKILL=9))
    modul.gruppe_toeten(cast(Any, P()))  # wirft nicht
    assert P.getoetet == 1


def test_git_praefix_echter_umweg(umgebung: Umgebung, ohne_praefix: None) -> None:
    """Ein echter Präfix (Python-Umweg statt setpriv) trägt alle Klon-Aufrufe."""
    from selbstbild.debugger import klon as modul

    _, klon, zweit = umgebung
    marke = klon.parent / "umweg.log"
    code = (
        "import subprocess, sys\n"
        f"open({str(marke)!r}, 'a').write(sys.argv[1] + chr(10))\n"
        "sys.exit(subprocess.call(sys.argv[1:]))"
    )
    modul.git_praefix_setzen([sys.executable, "-c", code])
    commit(zweit, "b.txt", "b\n")
    git(zweit, "push", "-q", "origin", "master")
    assert nachziehen(klon, "master", []) is None
    wt = worktree_bereit(klon, klon.parent / "wt" / "x-1", "debugger/1-x", "master")
    assert (wt / "b.txt").exists() and voraus(wt, "debugger/1-x") == 0
    zeilen = marke.read_text(encoding="utf-8").splitlines()
    assert len(zeilen) >= 6 and set(zeilen) == {"git"}
