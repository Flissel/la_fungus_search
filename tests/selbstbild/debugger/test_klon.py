"""Tests fuer klon.py mit echten temporaeren Repos (bares origin + Klone)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from selbstbild.debugger.klon import (
    GitFehler,
    KonfliktFehler,
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
