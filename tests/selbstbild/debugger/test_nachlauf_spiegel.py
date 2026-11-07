"""Final-Fix für nachlauf.py: Pin-Prüfungen im root-Spiegel (R39), Hooks aus, neue Runde nach Merge (R37)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_klon_haertung import hooks_setzen
from test_nachlauf import (  # noqa: F401  (Fixture `welt`)
    AUSSEN_BRANCH,
    ISSUE_REPO,
    N,
    OS_BRANCH,
    OS_REPO,
    PIN_BRANCH,
    Welt,
    _zweig_da,
    commit,
    git,
    welt,
)

from selbstbild.debugger.auftraege import Auftrag, Kommentar
from selbstbild.debugger.helfer import HelferErgebnis
from selbstbild.debugger.runde import runde_ausfuehren
from selbstbild.debugger.stand import FundStand


def _seitencommit(welt: Welt) -> str:
    """Ein Commit in vibemind-os, der auf einem Seitenzweig liegt, nicht auf master."""
    git(welt.os_arbeit, "checkout", "-q", "-b", "seite")
    sha = commit(welt.os_arbeit, "seite.txt")
    git(welt.os_arbeit, "push", "-q", "origin", "seite")
    git(welt.os_arbeit, "checkout", "-q", "master")
    return sha


def _falsche_quelle(welt: Welt, ort: Path, quelle: Path, master: str) -> None:
    """claw biegt `origin` des Klons auf ein eigenes Repo um, dessen master auf `master` zeigt."""
    falsch = welt.tmp / f"falsch-{quelle.name}"
    subprocess.run(["git", "clone", "-q", "--bare", str(quelle), str(falsch)], check=True, capture_output=True)
    git(falsch, "update-ref", "refs/heads/master", master)
    git(ort, "config", "remote.origin.url", str(falsch))
    git(ort, "fetch", "-q", "origin")


def test_pin_push_mit_nur_gitlink_und_basis_aus_spiegel(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    basis = git(welt.origin, "rev-parse", "refs/heads/master")
    assert welt.lauf() == [f"#{N}: Pin-PR #412 geöffnet"]
    assert welt.push_kw == [{"nur_gitlink": ("vibemind-os", sha)}]
    assert git(welt.origin, "rev-parse", f"{PIN_BRANCH}^") == basis


def test_pin_pruefung_im_spiegel_nicht_im_klon(welt: Welt) -> None:
    """claw zeigt `origin/master` im Submodul-Klon auf einen Seiten-Commit; root prüft im Spiegel."""
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = _seitencommit(welt)
    _falsche_quelle(welt, welt.os_klon, welt.os_origin, sha)
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    assert welt.lauf() == [f"#{N}: Pin wartet (nicht_auf_basis)"]
    assert welt.pushes == [] and welt.lab.namen("pr_anlegen") == []
    assert not welt.wt(ISSUE_REPO, PIN_BRANCH).exists() and not _zweig_da(welt.klon, PIN_BRANCH)


def test_pin_basis_veraltet_wartet_ohne_worktree(welt: Welt) -> None:
    """Sieht der äußere Klon eine andere Basis als der Spiegel, wartet der Pin, bevor ein Worktree entsteht."""
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    alt = git(welt.origin, "rev-parse", "master")
    neu_arbeit = welt.tmp / "aussen-neu"
    subprocess.run(["git", "clone", "-q", str(welt.origin), str(neu_arbeit)], check=True, capture_output=True)
    git(neu_arbeit, "-c", "user.name=T", "-c", "user.email=t@example.invalid", "commit", "-q", "--allow-empty",
        "-m", "weiter")
    _falsche_quelle(welt, welt.klon, welt.origin, alt)
    git(neu_arbeit, "push", "-q", "origin", "master")  # Spiegel ist weiter als die Quelle des Klons
    for _ in range(2):
        assert welt.lauf() == [f"#{N}: Pin wartet (basis_veraltet)"]
        assert not welt.wt(ISSUE_REPO, PIN_BRANCH).exists() and not _zweig_da(welt.klon, PIN_BRANCH)
    assert welt.pushes == []


def test_alter_pin_worktree_ohne_push_neu_aus_aktueller_basis(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    echt = welt.push

    def unklar(wt: Path, branch: str, **kw: Any) -> None:
        raise ValueError("repo_zum_worktree_unklar")

    monkeypatch.setattr(welt, "push", unklar)
    assert welt.lauf() == [f"#{N}: Pin wartet (push)"]
    assert welt.wt(ISSUE_REPO, PIN_BRANCH).exists()
    weiter = welt.tmp / "aussen-weiter"
    subprocess.run(["git", "clone", "-q", str(welt.origin), str(weiter)], check=True, capture_output=True)
    git(weiter, "-c", "user.name=T", "-c", "user.email=t@example.invalid", "commit", "-q", "--allow-empty",
        "-m", "weiter")
    git(weiter, "push", "-q", "origin", "master")
    git(welt.klon, "fetch", "-q", "origin")  # wie `nachziehen` im nächsten Durchgang
    monkeypatch.setattr(welt, "push", echt)
    assert welt.lauf() == [f"#{N}: Pin-PR #412 geöffnet"]
    assert git(welt.origin, "rev-parse", f"{PIN_BRANCH}^") == git(welt.origin, "rev-parse", "master")


def test_hooks_laufen_beim_pin_nicht(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("sh") is None:
        pytest.skip("kein sh für die Hook-Skripte")
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    marke = welt.tmp / "hook-lief"
    for ort in (welt.klon, welt.os_klon):
        hooks_setzen(Path(git(ort, "rev-parse", "--absolute-git-dir")), marke)

    def push(wt: Path, branch: str, **kw: Any) -> None:  # der echte Push läuft im Spiegel, nie im Klon
        welt.pushes.append((wt, branch))
        git(wt, "-c", f"core.hooksPath={os.devnull}", "push", "-q", "origin", f"HEAD:refs/heads/{branch}")

    monkeypatch.setattr(welt, "push", push)
    assert welt.lauf() == [f"#{N}: Pin-PR #412 geöffnet"]
    assert not marke.exists(), marke.read_text(encoding="utf-8")


# ---------- R37 über Runde und Nachlauf ----------

def test_erledigt_neuer_fix_wird_vom_nachlauf_nicht_geloescht(welt: Welt) -> None:
    welt.fund_mit_branch(ISSUE_REPO, AUSSEN_BRANCH)
    welt.pr(ISSUE_REPO, 501, merged=True, state="closed", sha="0" * 40)
    assert welt.lauf() == [f"#{N}: erledigt ({ISSUE_REPO}#501 gemergt)"]
    geloescht = list(welt.lab.namen("branch_loeschen"))

    def helfer(wt: Path, prompt: str, sitzung: str | None, frist_s: float) -> HelferErgebnis:
        commit(wt, "neu.txt", "fix: neuer Fund")
        return HelferErgebnis("s-2", "Neu behoben.", "fix", None)

    welt.helfer = helfer
    k = Kommentar(id=21, repo=ISSUE_REPO, nummer=N, autor="Flissel", text="nochmal kaputt",
                  erstellt="2026-11-07T11:55:00Z", review=False)
    assert runde_ausfuehren(Auftrag(N, (k,)), welt.konfig, welt.stand, welt.klon, welt.ordner, welt.w) == "ok"
    fund = welt.stand.funde[str(N)]
    assert fund.branch == f"debugger/{N}-fix-2" and fund.runde == 2 and fund.pr == 412
    assert welt.lab.namen("pr_anlegen")[-1][2] == fund.branch
    welt.pr(OS_REPO, 412)
    assert welt.lauf() == []
    assert welt.lab.namen("branch_loeschen") == geloescht
    assert _zweig_da(welt.os_origin, fund.branch)


def test_alter_stand_ohne_neue_felder_laedt(tmp_path: Path) -> None:
    import json

    from selbstbild.debugger.stand import stand_laden

    datei = tmp_path / "stand.json"
    datei.write_text(json.dumps({"start": "x", "seit": "x", "funde": {"5": {
        "repo": ISSUE_REPO, "branch": "debugger/5-a", "sitzung": None, "pr": 3, "zustand": "pr", "versuche": {}}}}),
        encoding="utf-8")
    assert stand_laden(datei).funde["5"] == FundStand(repo=ISSUE_REPO, branch="debugger/5-a", pr=3, zustand="pr")
    assert stand_laden(datei).funde["5"].runde == 1


# ---------- R45: Runde auf dem Pin-Branch pusht nie ----------

PIN_TEXT = ("🤖 Debugger-Claw: Der Pin-PR enthält nur die Submodul-Aktualisierung. Für einen weiteren Fix "
            "bitte nach dem Merge kommentieren.")


def test_runde_auf_pin_branch_pusht_nie(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    assert welt.lauf() == [f"#{N}: Pin-PR #412 geöffnet"]
    welt.pr(ISSUE_REPO, 412)
    pushes, prs, prompts = list(welt.pushes), len(welt.lab.namen("pr_anlegen")), []

    def helfer(wt: Path, prompt: str, sitzung: str | None, frist_s: float) -> HelferErgebnis:
        prompts.append(prompt)
        commit(wt, "boese.txt", "fix: mehr als der Pin")
        return HelferErgebnis("s-2", "Gleich mit behoben.", "fix", None)

    welt.helfer = helfer
    k = Kommentar(id=31, repo=ISSUE_REPO, nummer=412, autor="Flissel", text="bitte auch noch X",
                  erstellt="2026-11-07T11:58:00Z", review=False)
    assert runde_ausfuehren(Auftrag(N, (k,)), welt.konfig, welt.stand, welt.klon, welt.ordner, welt.w) == "ok"
    assert welt.pushes == pushes and len(welt.lab.namen("pr_anlegen")) == prs
    assert welt.lab.namen("kommentieren")[-1] == ("kommentieren", ISSUE_REPO, N, PIN_TEXT)
    assert prompts[0].count("Fixes in diesem Repo sind nicht freigegeben. Ändere keine Dateien, ordne nur ein.") == 1
    assert "boese.txt" not in git(welt.origin, "ls-tree", "-r", "--name-only", PIN_BRANCH)
    assert not welt.wt(ISSUE_REPO, PIN_BRANCH).exists()  # der fremde Commit bleibt nicht im Pin-Worktree liegen
    assert welt.stand.funde[str(N)].branch == PIN_BRANCH and welt.stand.funde[str(N)].pr == 412
