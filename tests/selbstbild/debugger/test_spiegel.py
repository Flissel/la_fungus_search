"""Tests für spiegel.py: Fake-Läufer für argv/env, echte temporäre Repos für die Rundreise."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

import pytest

from selbstbild.debugger import klon as klon_modul
from selbstbild.debugger.klon import GitFehler, nachziehen, worktree_bereit
from selbstbild.debugger.spiegel import (
    HAERTUNG,
    SPIEGEL_TIMEOUT_S,
    spiegel_bereit,
    spiegel_holen,
    spiegel_pfad,
    spiegel_push,
    spiegel_url,
)

REPO = "Flissel/Vibemind_V1"
BRANCH = "debugger/1-x"
ERLAUBTE_ENV = {"PATH", "HOME", "LANG", "GIT_TERMINAL_PROMPT", "GCM_INTERACTIVE", "GIT_CONFIG_NOSYSTEM",
                "GIT_CONFIG_GLOBAL", "GIT_ASKPASS", "DEBUGGER_GIT_TOKEN", "SYSTEMROOT"}


def git(ort: Path, *args: str) -> str:
    erg = subprocess.run(["git", "-C", str(ort), *args], check=True, capture_output=True, text=True)
    return erg.stdout.strip()


def _konfig(ort: Path) -> None:
    for k, v in (("user.name", "T"), ("user.email", "t@example.invalid"), ("core.autocrlf", "false"),
                 ("commit.gpgsign", "false")):
        git(ort, "config", k, v)


def commit(ort: Path, datei: str, inhalt: str) -> str:
    (ort / datei).write_text(inhalt, encoding="utf-8", newline="\n")
    git(ort, "add", "-A")
    git(ort, "commit", "-q", "-m", datei)
    return git(ort, "rev-parse", "HEAD")


class Fake:
    """Fake-Läufer: protokolliert argv und Schlüsselwörter, gibt einen festen Code zurück."""

    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.aufrufe: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        self.aufrufe.append({"argv": list(argv), **kw})
        return subprocess.CompletedProcess(argv, self.rc, "", "fatal: https://x:geheim@github.com/boese")


@pytest.fixture
def token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    datei = tmp_path / "token"
    datei.write_text("  tok-wert-123\n", encoding="utf-8")
    monkeypatch.setenv("DEBUGGER_ASKPASS", str(tmp_path / "askpass.sh"))
    monkeypatch.setenv("GH_TOKEN", "eltern-geheim")
    monkeypatch.setenv("HARMLOS_ABER_FREMD", "ja")
    return datei


@pytest.fixture
def praefix() -> Iterator[tuple[str, ...]]:
    alt = klon_modul.git_praefix()
    klon_modul.git_praefix_setzen(("setpriv", "--reuid=1000", "--"))
    yield klon_modul.git_praefix()
    klon_modul.git_praefix_setzen(alt)


def _haertung_in(argv: list[str]) -> bool:
    n = len(HAERTUNG)
    return any(argv[i:i + n] == list(HAERTUNG) for i in range(len(argv) - n + 1))


# ---------- Fake-Läufer ----------

def test_spiegel_url_prueft_repo(tmp_path: Path) -> None:
    assert spiegel_url(REPO) == "https://github.com/Flissel/Vibemind_V1.git"
    assert spiegel_url("Flissel/vibemind-os") == "https://github.com/Flissel/vibemind-os.git"
    assert spiegel_pfad(tmp_path, "Flissel/vibemind-os") == tmp_path / "Flissel__vibemind-os.git"
    for falsch in ("Flissel", "Flissel/a/b", "x@evil.com/y", "Flissel/v1 --upload-pack=x", "-o/x",
                   "Flissel/", "/x", "Flissel/x\n", "Flis_sel/x", "a:b/c", ""):
        with pytest.raises(ValueError):
            spiegel_url(falsch)
        with pytest.raises(ValueError):
            spiegel_pfad(tmp_path, falsch)


def test_haertung_wert() -> None:
    assert HAERTUNG == (
        "-c", "credential.helper=", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false",
        "-c", "protocol.allow=never", "-c", "protocol.https.allow=always", "-c", "protocol.file.allow=always",
    )


def test_spiegel_holen_argv_und_env(tmp_path: Path, token: Path, praefix: tuple[str, ...]) -> None:
    f = Fake()
    pfad = tmp_path / "s.git"
    spiegel_holen(pfad, REPO, token, laeufer=f)
    (a,) = f.aufrufe
    argv = a["argv"]
    assert argv[0] == "git" and "setpriv" not in argv  # Spiegel-Aufrufe nie mit Präfix
    assert _haertung_in(argv)
    assert argv[argv.index("-C") + 1] == str(pfad)
    assert argv[-5:] == ["fetch", "--prune", "--no-tags", spiegel_url(REPO), "+refs/heads/*:refs/heads/*"]
    assert "tok-wert-123" not in " ".join(argv)
    env = a["env"]
    assert env["DEBUGGER_GIT_TOKEN"] == "tok-wert-123"
    assert env["GIT_ASKPASS"] == str(tmp_path / "askpass.sh")
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GCM_INTERACTIVE"] == "never"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1" and env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert "GH_TOKEN" not in env and "HARMLOS_ABER_FREMD" not in env
    assert set(env) <= ERLAUBTE_ENV
    assert a["stdin"] == subprocess.DEVNULL and a["timeout"] == SPIEGEL_TIMEOUT_S == 300


def test_spiegel_holen_ohne_token_oder_askpass(tmp_path: Path, token: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = Fake()
    with pytest.raises(RuntimeError, match="kein_token"):
        spiegel_holen(tmp_path / "s.git", REPO, tmp_path / "fehlt", laeufer=f)
    leer = tmp_path / "leer"
    leer.write_text(" \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="kein_token"):
        spiegel_holen(tmp_path / "s.git", REPO, leer, laeufer=f)
    monkeypatch.delenv("DEBUGGER_ASKPASS")
    with pytest.raises(RuntimeError, match="kein_askpass"):
        spiegel_holen(tmp_path / "s.git", REPO, token, laeufer=f)
    assert f.aufrufe == []


def test_spiegel_fehler_ohne_stderr_und_url(tmp_path: Path, token: Path) -> None:
    with pytest.raises(GitFehler) as ei:
        spiegel_holen(tmp_path / "s.git", REPO, token, laeufer=Fake(rc=128))
    assert ei.value.unterbefehl == "fetch" and ei.value.code == 128
    assert "geheim" not in str(ei.value) and "github.com" not in str(ei.value)

    def zu_lang(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 1)

    with pytest.raises(GitFehler) as ei:
        spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, REPO, token, laeufer=zu_lang)
    assert ei.value.code == klon_modul.TIMEOUT_CODE


def test_spiegel_push_nur_debugger(tmp_path: Path, token: Path) -> None:
    f = Fake()
    for falsch in ("master", "main", "feature/x", "debugger/", "debugger/../master", "debugger/x:refs/heads/master",
                   "debugger/+x", "+debugger/x", "debugger/x y", "refs/heads/debugger/x", "debugger/-x"):
        with pytest.raises(ValueError):
            spiegel_push(tmp_path / "s.git", tmp_path, falsch, REPO, token, laeufer=f)
    with pytest.raises(ValueError):
        spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, "Flissel/a/b", token, laeufer=f)
    assert f.aufrufe == []


def test_spiegel_push_refspec_ohne_plus(tmp_path: Path, token: Path, praefix: tuple[str, ...]) -> None:
    f = Fake()
    pfad, wt = tmp_path / "s.git", tmp_path / "wt"
    spiegel_push(pfad, wt, BRANCH, REPO, token, laeufer=f)
    alt, holen, push = f.aufrufe
    assert alt["argv"][-4:] == ["rev-parse", "--verify", "-q", f"refs/heads/{BRANCH}"]
    assert "DEBUGGER_GIT_TOKEN" not in alt["env"]
    # (2) Objekte aus dem Worktree in den Spiegel: ohne Token, Upload-Pack mit safe.directory, ohne Präfix
    a = holen["argv"]
    assert a[0] == "git" and _haertung_in(a) and a[a.index("-C") + 1] == str(pfad)
    i = a.index("fetch")
    assert a[i:i + 2] == ["fetch", "--no-tags"] and a[i + 2] == "--upload-pack"
    upload = a[i + 3]
    assert "safe.directory=*" in upload and upload.rstrip().endswith("upload-pack")
    assert "setpriv" not in upload and not any("setpriv" in x for x in a)
    assert a[i + 4:] == [str(wt), f"+HEAD:refs/heads/{BRANCH}"]
    assert "DEBUGGER_GIT_TOKEN" not in holen["env"] and "GIT_ASKPASS" not in holen["env"]
    assert set(holen["env"]) <= ERLAUBTE_ENV
    # (3) Push nur dieses Branches, ohne +
    b = push["argv"]
    assert b[0] == "git" and _haertung_in(b) and b[b.index("-C") + 1] == str(pfad)
    assert b[-3:] == ["push", spiegel_url(REPO), f"refs/heads/{BRANCH}:refs/heads/{BRANCH}"]
    assert not any(x.startswith("+") for x in b[b.index("push"):])
    assert "tok-wert-123" not in " ".join(a + b)
    assert push["env"]["DEBUGGER_GIT_TOKEN"] == "tok-wert-123" and "GH_TOKEN" not in push["env"]
    for aufruf in (holen, push):
        assert aufruf["stdin"] == subprocess.DEVNULL and aufruf["timeout"] == SPIEGEL_TIMEOUT_S


def test_spiegel_push_prueft_token_vor_jeder_aenderung(tmp_path: Path, token: Path) -> None:
    f = Fake()
    with pytest.raises(RuntimeError, match="kein_token"):
        spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, REPO, tmp_path / "fehlt", laeufer=f)
    assert f.aufrufe == []


def test_spiegel_push_fehler_setzt_spiegel_zurueck(tmp_path: Path, token: Path) -> None:
    class PushScheitert(Fake):
        def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
            self.aufrufe.append({"argv": list(argv), **kw})
            rc = 1 if "rev-parse" in argv else 128 if "push" in argv else 0
            return subprocess.CompletedProcess(argv, rc, "", "")

    f = PushScheitert()
    with pytest.raises(GitFehler) as ei:
        spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, REPO, token, laeufer=f)
    assert ei.value.unterbefehl == "push"
    assert f.aufrufe[-1]["argv"][-3:] == ["update-ref", "-d", f"refs/heads/{BRANCH}"]
    assert "DEBUGGER_GIT_TOKEN" not in f.aufrufe[-1]["env"]


# ---------- echte Repos ----------

class Rundreise:
    def __init__(self, tmp: Path) -> None:
        self.github = tmp / "github.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(self.github)], check=True,
                       capture_output=True)
        start = tmp / "start"
        start.mkdir()
        git(start, "init", "-q", "-b", "master")
        _konfig(start)
        commit(start, "a.txt", "a\n")
        git(start, "push", "-q", str(self.github), "master")
        self.token = tmp / "token-gibt-es-nicht"  # lokale URL: kein Token, kein Askpass nötig
        self.spiegel = spiegel_pfad(tmp / "spiegel", REPO)
        spiegel_bereit(self.spiegel)
        spiegel_bereit(self.spiegel)  # zweiter Aufruf ändert nichts
        self.holen()
        self.klon = tmp / "klon"
        subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=false", str(self.spiegel), str(self.klon)],
                       check=True, capture_output=True)
        _konfig(self.klon)
        self.wt = worktree_bereit(self.klon, tmp / "wt" / "x-1", BRANCH, "master")
        self.kopf = commit(self.wt, "fix.txt", "fix\n")

    def holen(self) -> None:
        spiegel_holen(self.spiegel, REPO, self.token, url=str(self.github))

    def push(self) -> None:
        spiegel_push(self.spiegel, self.wt, BRANCH, REPO, self.token, url=str(self.github))


@pytest.fixture
def rund(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rundreise:
    monkeypatch.delenv("DEBUGGER_ASKPASS", raising=False)
    return Rundreise(tmp_path)


def test_spiegel_rundreise_lokal(rund: Rundreise) -> None:
    assert git(rund.spiegel, "rev-parse", "--is-bare-repository") == "true"
    rund.push()
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == rund.kopf
    assert git(rund.github, "rev-parse", "refs/heads/master") != rund.kopf
    rund.holen()
    assert nachziehen(rund.klon, "master", []) is None
    assert git(rund.klon, "rev-parse", f"refs/remotes/origin/{BRANCH}") == rund.kopf
    # zweiter Push mit neuem Commit ist ein Fast-Forward; ein auseinandergelaufener Branch wird nie erzwungen
    neu = commit(rund.wt, "fix2.txt", "2\n")
    rund.push()
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == neu
    git(rund.wt, "reset", "-q", "--hard", "HEAD~2")
    commit(rund.wt, "anders.txt", "anders\n")
    with pytest.raises(GitFehler) as ei:
        rund.push()
    assert ei.value.unterbefehl == "push"
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == neu
    # der Spiegel zeigt wieder den Stand von GitHub, der Klon hält den Branch also nicht für gepusht
    assert git(rund.spiegel, "rev-parse", f"refs/heads/{BRANCH}") == neu


def _marken_skript(marke: Path) -> str:
    """Shell-Skript, das bei Ausführung `marke` anlegt (Pfade mit / auch auf Windows)."""
    ziel = marke.as_posix()
    return f'#!/bin/sh\necho lief >> "{ziel}"\ncat > /dev/null\nexit 0\n'


def test_spiegel_ignoriert_hooks_im_klon(rund: Rundreise, tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("kein sh für die Hook-Skripte")
    marke = tmp_path / "hook-lief"
    skript = _marken_skript(marke)
    gitdirs = [Path(git(rund.klon, "rev-parse", "--absolute-git-dir")), rund.spiegel]
    for gitdir in gitdirs:
        hooks = gitdir / "hooks"
        hooks.mkdir(exist_ok=True)
        for name in ("pre-push", "post-merge", "reference-transaction", "pre-receive", "post-update"):
            datei = hooks / name
            datei.write_text(skript, encoding="utf-8", newline="\n")
            datei.chmod(0o755)
    # Kontrolle: ohne Härtung laufen die Hooks hier wirklich
    git(rund.spiegel, "fetch", "-q", str(rund.github), "+refs/heads/master:refs/heads/kontrolle")
    assert marke.exists()
    marke.unlink()
    # Konfiguration des Helfers im Klon, die als root nie greifen darf
    hook_skript = tmp_path / "boese.sh"
    hook_skript.write_text(skript, encoding="utf-8", newline="\n")
    hook_skript.chmod(0o755)
    git(rund.klon, "config", "uploadpack.packObjectsHook", hook_skript.as_posix())
    git(rund.klon, "config", "core.fsmonitor", hook_skript.as_posix())
    git(rund.klon, "config", f"url.{rund.klon.as_posix()}/umleitung.insteadOf", str(rund.github))
    rund.push()
    rund.holen()
    assert not marke.exists()
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == rund.kopf
