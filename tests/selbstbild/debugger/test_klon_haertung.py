"""Final-Fix Teil A für klon.py: Hooks aus (R39), Submodul-Update blockiert nicht (R40), Branch-Runden (R37)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from selbstbild.debugger import klon as klon_modul
from selbstbild.debugger.klon import MAX_KURZ, branch_name, debugger_zweig_pruefen, nachziehen, worktree_bereit
from selbstbild.funde.github import _debugger_branch

HOOKS = ("post-merge", "post-checkout", "pre-commit", "commit-msg", "post-commit", "reference-transaction",
         "pre-merge-commit", "post-rewrite")


def git(ort: Path, *args: str) -> str:
    erg = subprocess.run(["git", "-C", str(ort), *args], check=True, capture_output=True, text=True)
    return erg.stdout.strip()


def _konfig(ort: Path) -> None:
    for k, v in (("user.name", "T"), ("user.email", "t@example.invalid"), ("core.autocrlf", "false"),
                 ("commit.gpgsign", "false")):
        git(ort, "config", k, v)


def commit(ort: Path, datei: str) -> str:
    (ort / datei).write_text(f"{datei}\n", encoding="utf-8", newline="\n")
    git(ort, "add", "--", datei)
    git(ort, "commit", "-q", "-m", datei)
    return git(ort, "rev-parse", "HEAD")


def klonen(quelle: Path, ziel: Path) -> Path:
    subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=false", str(quelle), str(ziel)], check=True,
                   capture_output=True)
    _konfig(ziel)
    return ziel


def bare(tmp: Path, name: str) -> Path:
    ort = tmp / f"{name}.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(ort)], check=True, capture_output=True)
    return ort


def hooks_setzen(gitdir: Path, marke: Path) -> None:
    """Jeder Hook schreibt eine Zeile nach `marke` (Pfade mit / auch unter Windows)."""
    hooks = gitdir / "hooks"
    hooks.mkdir(exist_ok=True)
    for name in HOOKS:
        datei = hooks / name
        datei.write_text(f'#!/bin/sh\necho {name} >> "{marke.as_posix()}"\ncat > /dev/null\nexit 0\n',
                         encoding="utf-8", newline="\n")
        datei.chmod(0o755)


@pytest.fixture
def mit_submodul(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Spiegel-Ersatz (bare) für äußeres Repo und Submodul, ein Klon mit initialisiertem Submodul."""
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    sub = bare(tmp_path, "sub")
    sub_arbeit = klonen(sub, tmp_path / "sub-arbeit")
    git(sub_arbeit, "symbolic-ref", "HEAD", "refs/heads/master")
    commit(sub_arbeit, "s.txt")
    git(sub_arbeit, "push", "-q", "origin", "master")
    aussen = bare(tmp_path, "aussen")
    arbeit = klonen(aussen, tmp_path / "arbeit")
    git(arbeit, "symbolic-ref", "HEAD", "refs/heads/master")
    commit(arbeit, "a.txt")
    git(arbeit, "submodule", "add", "-q", sub.as_uri(), "vibemind-os")
    git(arbeit, "commit", "-q", "-m", "submodul")
    git(arbeit, "push", "-q", "origin", "master")
    klon = klonen(aussen, tmp_path / "klon")
    git(klon, "submodule", "update", "--init", "-q")
    return {"sub": sub, "sub_arbeit": sub_arbeit, "aussen": aussen, "arbeit": arbeit, "klon": klon}


# ---------- R40 ----------

def test_submodul_update_blockiert_nicht_dauerhaft(mit_submodul: dict[str, Path]) -> None:
    r = mit_submodul
    neu = commit(r["sub_arbeit"], "neu.txt")  # noch nicht im Spiegel des Submoduls
    git(r["arbeit"], "update-index", "--cacheinfo", f"160000,{neu},vibemind-os")
    git(r["arbeit"], "commit", "-q", "-m", "pin")
    git(r["arbeit"], "push", "-q", "origin", "master")
    assert nachziehen(r["klon"], "master", ["vibemind-os"]) == "submodul_fehler"
    assert nachziehen(r["klon"], "master", ["vibemind-os"]) == "submodul_fehler"  # erneut versucht
    git(r["sub_arbeit"], "push", "-q", "origin", "master")
    assert nachziehen(r["klon"], "master", ["vibemind-os"]) is None
    assert git(r["klon"] / "vibemind-os", "rev-parse", "HEAD") == neu


def test_submodul_mit_eigenen_aenderungen_nicht_sauber(mit_submodul: dict[str, Path]) -> None:
    r = mit_submodul
    datei = r["klon"] / "vibemind-os" / "s.txt"
    datei.write_text("lokal\n", encoding="utf-8", newline="\n")
    assert nachziehen(r["klon"], "master", ["vibemind-os"]) == "nicht_sauber"
    assert datei.read_text(encoding="utf-8") == "lokal\n"


def test_nachziehen_fetch_ohne_submodule_und_gehaertet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gesehen: list[list[str]] = []

    def begrenzt(argv: list[str], unter: str, **kw: object) -> tuple[int, bytes]:
        gesehen.append(argv)
        return 0, b""

    monkeypatch.setattr(klon_modul, "_begrenzt", begrenzt)
    assert nachziehen(tmp_path, "master", []) == "falscher_zweig"
    assert "--no-recurse-submodules" in gesehen[0] and gesehen[0][-3:] == ["fetch", "--prune",
                                                                            "--no-recurse-submodules"]
    for argv in gesehen:
        i = argv.index("git")
        assert argv[i + 1:i + 5] == ["-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false"]


# ---------- R39: Hooks im Klon laufen nie ----------

def test_hooks_im_klon_laufen_nicht(mit_submodul: dict[str, Path], tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("kein sh für die Hook-Skripte")
    r = mit_submodul
    marke = tmp_path / "hook-lief"
    hooks_setzen(Path(git(r["klon"], "rev-parse", "--absolute-git-dir")), marke)
    hooks_setzen(Path(git(r["klon"] / "vibemind-os", "rev-parse", "--absolute-git-dir")), marke)
    git(r["klon"], "update-ref", "refs/kontrolle", "HEAD")  # Kontrolle: ohne Härtung laufen sie
    assert marke.exists()
    marke.unlink()
    commit(r["arbeit"], "b.txt")
    git(r["arbeit"], "push", "-q", "origin", "master")
    assert nachziehen(r["klon"], "master", ["vibemind-os"]) is None
    assert (r["klon"] / "b.txt").exists()
    wt = worktree_bereit(r["klon"], tmp_path / "wt" / "x-1", "debugger/1-x", "master")
    assert (wt / "b.txt").exists()
    klon_modul.git_lauf(wt, "commit", "-q", "--allow-empty", "-m", "pin")
    assert not marke.exists(), marke.read_text(encoding="utf-8")


# ---------- R37/R38: Branch-Namen je Runde ----------

def test_branch_name_runde_suffix() -> None:
    titel = "[selbstbild] Doku-Widerspruch: Schedule stimmt nicht (vibemind)"
    assert branch_name(318, titel) == branch_name(318, titel, 1)
    zwei = branch_name(318, titel, 2)
    assert zwei.startswith("debugger/318-") and zwei.endswith("-2")
    kurz = zwei.removeprefix("debugger/318-")
    assert len(kurz) <= MAX_KURZ and "--" not in kurz
    assert branch_name(318, "x" * 80, 12).removeprefix("debugger/318-") == "x" * (MAX_KURZ - 3) + "-12"
    assert branch_name(318, "!!!", 3) == "debugger/318-fund-3"
    assert branch_name(5, "fix") == "debugger/5-fix" and branch_name(5, "fix", 2) == "debugger/5-fix-2"
    for name in (zwei, "debugger/5-fix-2"):
        debugger_zweig_pruefen(name)
        assert _debugger_branch(name) == name
    with pytest.raises(ValueError):
        branch_name(5, "x", 0)


def test_worktree_bereit_fetch_ohne_submodule(mit_submodul: dict[str, Path], tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    gesehen: list[list[str]] = []
    echt = klon_modul._begrenzt

    def begrenzt(argv: list[str], unter: str, **kw: object) -> tuple[int, bytes]:
        gesehen.append(argv)
        return echt(argv, unter, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(klon_modul, "_begrenzt", begrenzt)
    worktree_bereit(mit_submodul["klon"], tmp_path / "wt" / "x-5", "debugger/5-x", "master")
    holen = [a for a in gesehen if "fetch" in a]
    assert holen and all("--no-recurse-submodules" in a for a in holen)
