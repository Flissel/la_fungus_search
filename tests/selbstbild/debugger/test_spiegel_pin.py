"""Final-Fix Teil A für spiegel.py: root prüft den Pin-Commit beim Push (R39) und liest nur gehärtet."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from selbstbild.debugger.klon import worktree_bereit
from selbstbild.debugger.spiegel import (
    HAERTUNG,
    ZurueckgehaltenFehler,
    spiegel_bereit,
    spiegel_holen,
    spiegel_lesen,
    spiegel_pfad,
    spiegel_push,
)

REPO = "Flissel/Vibemind_V1"
BRANCH = "debugger/7-pin"
SUB = "vibemind-os"
ALT, NEU = "1" * 40, "2" * 40


def git(ort: Path, *args: str) -> str:
    erg = subprocess.run(["git", "-C", str(ort), *args], check=True, capture_output=True, text=True)
    return erg.stdout.strip()


def _ref(ort: Path, ref: str) -> str | None:
    erg = subprocess.run(["git", "-C", str(ort), "rev-parse", "--verify", "-q", ref], capture_output=True, text=True)
    return erg.stdout.strip() if erg.returncode == 0 else None


def _konfig(ort: Path) -> None:
    for k, v in (("user.name", "T"), ("user.email", "t@example.invalid"), ("core.autocrlf", "false"),
                 ("commit.gpgsign", "false")):
        git(ort, "config", k, v)


def _pin(ort: Path, sha: str, betreff: str = "pin") -> None:
    git(ort, "update-index", "--add", "--cacheinfo", f"160000,{sha},{SUB}")
    git(ort, "commit", "-q", "-m", betreff)


class PinWelt:
    """„GitHub“ (bare) mit Gitlink auf ALT, Spiegel davon, Klon aus dem Spiegel, Pin-Worktree."""

    def __init__(self, tmp: Path) -> None:
        self.github = tmp / "github.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(self.github)], check=True,
                       capture_output=True)
        start = tmp / "start"
        start.mkdir()
        git(start, "init", "-q", "-b", "master")
        _konfig(start)
        (start / "a.txt").write_text("a\n", encoding="utf-8", newline="\n")
        git(start, "add", "a.txt")
        _pin(start, ALT, "start")
        git(start, "push", "-q", str(self.github), "master")
        self.token = tmp / "token-gibt-es-nicht"
        self.spiegel = spiegel_pfad(tmp / "spiegel", REPO)
        spiegel_bereit(self.spiegel)
        spiegel_holen(self.spiegel, REPO, self.token, url=str(self.github))
        self.klon = tmp / "klon"
        subprocess.run(["git", "clone", "-q", "--no-checkout", "-c", "core.autocrlf=false", str(self.spiegel),
                        str(self.klon)], check=True, capture_output=True)
        _konfig(self.klon)
        self.basis = git(self.spiegel, "rev-parse", "refs/heads/master")
        self.wt = worktree_bereit(self.klon, tmp / "wt" / "Vibemind_V1-7-pin", BRANCH, "master", start=self.basis)

    def push(self, erwartet: str) -> None:
        spiegel_push(self.spiegel, self.wt, BRANCH, REPO, self.token, url=str(self.github),
                     nur_gitlink=(SUB, erwartet))

    def nichts_gepusht(self) -> None:
        assert _ref(self.github, f"refs/heads/{BRANCH}") is None
        assert _ref(self.spiegel, f"refs/heads/{BRANCH}") is None
        assert _ref(self.spiegel, f"refs/pruefung/{BRANCH}") is None


@pytest.fixture
def pw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PinWelt:
    monkeypatch.delenv("DEBUGGER_ASKPASS", raising=False)
    return PinWelt(tmp_path)


def test_pin_korrekt_wird_gepusht(pw: PinWelt) -> None:
    _pin(pw.wt, NEU)
    pw.push(NEU)
    kopf = git(pw.wt, "rev-parse", "HEAD")
    assert git(pw.github, "rev-parse", f"refs/heads/{BRANCH}") == kopf


def test_pin_mit_zusaetzlicher_datei_gehalten(pw: PinWelt) -> None:
    (pw.wt / "boese.txt").write_text("x\n", encoding="utf-8", newline="\n")
    git(pw.wt, "add", "boese.txt")
    _pin(pw.wt, NEU)
    with pytest.raises(ZurueckgehaltenFehler) as ei:
        pw.push(NEU)
    assert ei.value.muster == "pin_inhalt"
    pw.nichts_gepusht()


def test_pin_mit_falschem_sha_gehalten(pw: PinWelt) -> None:
    _pin(pw.wt, NEU)
    with pytest.raises(ZurueckgehaltenFehler) as ei:
        pw.push("3" * 40)
    assert ei.value.muster == "pin_inhalt"
    pw.nichts_gepusht()


@pytest.mark.parametrize("art", ["zwei_commits", "falscher_elter", "anderer_pfad"])
def test_pin_form_verletzt_gehalten(pw: PinWelt, art: str) -> None:
    if art == "zwei_commits":
        git(pw.wt, "commit", "-q", "--allow-empty", "-m", "vorher")
        _pin(pw.wt, NEU)
    elif art == "falscher_elter":  # ein einzelner neuer Commit mit richtigem Baum, aber ohne Elter
        _pin(pw.wt, NEU)
        waise = git(pw.wt, "commit-tree", git(pw.wt, "rev-parse", "HEAD^{tree}"), "-m", "pin")
        git(pw.wt, "reset", "-q", "--soft", waise)
    else:
        git(pw.wt, "update-index", "--add", "--cacheinfo", f"160000,{NEU},anderswo")
        git(pw.wt, "commit", "-q", "-m", "pin")
    with pytest.raises(ZurueckgehaltenFehler) as ei:
        pw.push(NEU)
    assert ei.value.muster == "pin_inhalt"
    pw.nichts_gepusht()


def test_spiegel_lesen_gehaertet_und_nur_lesend(tmp_path: Path) -> None:
    gesehen: list[dict[str, Any]] = []

    def laeufer(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        gesehen.append({"argv": argv, **kw})
        return subprocess.CompletedProcess(argv, 0, b"commit\n", b"geheim")

    pfad = tmp_path / "x.git"
    assert spiegel_lesen(pfad, "cat-file", "-t", ALT, laeufer=laeufer) == (0, "commit\n")
    argv = gesehen[0]["argv"]
    assert argv[0] == "git" and argv[1:1 + len(HAERTUNG)] == list(HAERTUNG)
    assert f"--git-dir={pfad}" in argv and "-C" not in argv
    assert gesehen[0]["env"]["GIT_CONFIG_NOSYSTEM"] == "1" and "DEBUGGER_GIT_TOKEN" not in gesehen[0]["env"]
    for unter in ("push", "fetch", "update-ref", "config", "gc"):
        with pytest.raises(ValueError):
            spiegel_lesen(pfad, unter, laeufer=laeufer)
    assert len(gesehen) == 1


# ---------- R45: Pin-Branch nie ohne Gitlink-Prüfung ----------

@pytest.mark.parametrize("branch", ["debugger/7-pin", "debugger/7-pin-2", "debugger/12345-pin-10"])
def test_pin_branch_ohne_nur_gitlink_vor_jedem_lauf_gehalten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              branch: str) -> None:
    import selbstbild.debugger.spiegel as spiegel_modul

    gesehen: list[Any] = []

    def laeufer(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        gesehen.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(spiegel_modul, "git_in_datei", lambda *a, **kw: gesehen.append(("bundle", a)))
    with pytest.raises(ZurueckgehaltenFehler) as ei:
        spiegel_push(tmp_path / "x.git", tmp_path / "wt", branch, REPO, tmp_path / "token", laeufer=laeufer)
    assert ei.value.muster == "pin_inhalt" and gesehen == []


def test_pin_branch_echter_push_ohne_nur_gitlink_gehalten(pw: PinWelt) -> None:
    _pin(pw.wt, NEU)
    with pytest.raises(ZurueckgehaltenFehler) as ei:
        spiegel_push(pw.spiegel, pw.wt, BRANCH, REPO, pw.token, url=str(pw.github))
    assert ei.value.muster == "pin_inhalt"
    pw.nichts_gepusht()


def test_aehnliche_branches_sind_keine_pin_branches() -> None:
    from selbstbild.debugger.nachlauf import pin_branch
    from selbstbild.debugger.spiegel import ist_pin_branch

    assert ist_pin_branch(pin_branch(318))
    for branch in ("debugger/7-pinsel", "debugger/7-fix", "debugger/7-pin-x", "debugger/x-pin", "pin"):
        assert not ist_pin_branch(branch), branch


# ---------- R47: Branch-Namen treffen nie das Pin-Muster ----------

@pytest.mark.parametrize("titel", ["[selbstbild] Pin", "Pin", "[selbstbild] Doku zum Pin", "pin!!!",
                                   "[selbstbild] " + "x" * 26 + " pin"])
@pytest.mark.parametrize("runde", [1, 2, 13])
def test_branch_name_nie_pin_muster(titel: str, runde: int) -> None:
    from selbstbild.debugger.klon import MAX_KURZ, branch_name
    from selbstbild.debugger.spiegel import ist_pin_branch

    name = branch_name(7, titel, runde)
    assert not ist_pin_branch(name)
    kurz = name.removeprefix("debugger/7-")
    ohne_runde = kurz.removesuffix(f"-{runde}") if runde > 1 else kurz
    assert len(kurz) <= MAX_KURZ and not ohne_runde.endswith("-pin") and ohne_runde != "pin"
    assert "--" not in kurz


def test_branch_name_pin_beispiele() -> None:
    from selbstbild.debugger.klon import branch_name

    assert branch_name(5, "[selbstbild] Pin") == "debugger/5-pin-fix"
    assert branch_name(5, "[selbstbild] Pin", 2) == "debugger/5-pin-fix-2"
    assert branch_name(5, "[selbstbild] Pinsel") == "debugger/5-pinsel"  # nur das Wortende zählt


def test_fix_branch_mit_titel_pin_laesst_sich_pushen(pw: PinWelt) -> None:
    from selbstbild.debugger.klon import branch_name

    branch = branch_name(7, "[selbstbild] Pin")
    wt = worktree_bereit(pw.klon, pw.wt.parent / "Vibemind_V1-7-pin-fix", branch, "master", start=pw.basis)
    (wt / "fix.txt").write_text("fix\n", encoding="utf-8", newline="\n")
    git(wt, "add", "fix.txt")
    git(wt, "commit", "-q", "-m", "fix: normaler Fix")
    spiegel_push(pw.spiegel, wt, branch, REPO, pw.token, url=str(pw.github))
    assert git(pw.github, "rev-parse", f"refs/heads/{branch}") == git(wt, "rev-parse", "HEAD")
