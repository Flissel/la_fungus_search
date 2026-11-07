"""Tests für spiegel.py: Fake-Läufer für argv/env, echte temporäre Repos für Rundreise und Prüfung."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import pytest

from selbstbild.debugger import klon as klon_modul
from selbstbild.debugger import spiegel as spiegel_modul
from selbstbild.debugger.klon import GROESSE_CODE, GitFehler, nachziehen, push_inhalt, worktree_bereit
from selbstbild.debugger.spiegel import (
    FSCK,
    HAERTUNG,
    PFAD_POSIX,
    SPIEGEL_TIMEOUT_S,
    ZurueckgehaltenFehler,
    spiegel_bereit,
    spiegel_holen,
    spiegel_pfad,
    spiegel_push,
    spiegel_url,
)

REPO = "Flissel/Vibemind_V1"
BRANCH = "debugger/1-x"
ERLAUBTE_ENV = {"PATH", "HOME", "LANG", "GIT_TERMINAL_PROMPT", "GCM_INTERACTIVE", "GIT_CONFIG_NOSYSTEM",
                "GIT_CONFIG_GLOBAL", "GIT_NO_REPLACE_OBJECTS", "GIT_NO_LAZY_FETCH", "GIT_ASKPASS",
                "DEBUGGER_GIT_TOKEN", "SYSTEMROOT"}
# aus Teilen gebaut, damit kein Schlüssel-Muster im Repo steht
TOKEN = "ghp" + "_" + "A" * 36
TOKEN_MUSTER = "GitHub-Token (klassisch)"
SHA_BASIS, SHA_NEU = "a" * 40, "b" * 40


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


def commit(ort: Path, datei: str, inhalt: str, *, autor: str | None = None) -> str:
    (ort / datei).write_text(inhalt, encoding="utf-8", newline="\n")
    git(ort, "add", "-A")
    vorne = ["-c", f"user.name={autor}"] if autor else []
    subprocess.run(["git", "-C", str(ort), *vorne, "commit", "-q", "-m", "fix: x"], check=True, capture_output=True)
    return git(ort, "rev-parse", "HEAD")


# ---------- Fakes ----------

class Fake:
    """Fake-Läufer: protokolliert argv und Schlüsselwörter, gibt einen festen Code zurück."""

    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.aufrufe: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        self.aufrufe.append({"argv": list(argv), **kw})
        return subprocess.CompletedProcess(argv, self.rc, b"", b"fatal: https://x:geheim@github.com/boese")


class SpiegelFake(Fake):
    """Antwortet je Unterbefehl wie ein Spiegel mit Basis, ohne Branch, mit einem neuen Commit."""

    def __init__(self, push_rc: int = 0) -> None:
        super().__init__()
        self.push_rc = push_rc

    def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        self.aufrufe.append({"argv": list(argv), **kw})
        i = next(n for n, x in enumerate(argv) if x.startswith("--git-dir="))
        unter, rest = argv[i + 1], argv[i + 2:]
        rc, aus = 0, b""
        if unter == "rev-parse":
            aus = {"refs/heads/master": SHA_BASIS, f"refs/pruefung/{BRANCH}": SHA_NEU}.get(rest[-1], "").encode()
            rc = 0 if aus else 1
        elif unter == "cat-file" and rest[0] == "-t":
            aus = b"commit\n"
        elif unter == "rev-list":
            aus = f"{SHA_NEU}\n".encode()
        elif unter == "cat-file" and rest[0] == "--batch-check":
            aus = f"{SHA_NEU} commit 4\n".encode()
        elif unter == "cat-file":
            aus = f"{SHA_NEU} commit 4\nfrei\n".encode()
        elif unter == "push":
            rc = self.push_rc
        return subprocess.CompletedProcess(argv, rc, aus, b"")


class BundleFake:
    def __init__(self) -> None:
        self.aufrufe: list[tuple[Path, tuple[str, ...], int]] = []

    def __call__(self, ort: Path, *args: str, ziel: BinaryIO, grenze: int) -> None:
        self.aufrufe.append((ort, args, grenze))
        ziel.write(b"# v2 git bundle\n")


@pytest.fixture
def token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    datei = tmp_path / "token"
    datei.write_text("  tok-wert-123\n", encoding="utf-8")
    monkeypatch.setenv("DEBUGGER_ASKPASS", str(tmp_path / "askpass.sh"))
    monkeypatch.setenv("GH_TOKEN", "eltern-geheim")
    monkeypatch.setenv("HARMLOS_ABER_FREMD", "ja")
    return datei


@pytest.fixture
def bundle(monkeypatch: pytest.MonkeyPatch) -> BundleFake:
    b = BundleFake()
    monkeypatch.setattr(spiegel_modul, "git_in_datei", b)
    return b


@pytest.fixture
def praefix() -> Iterator[tuple[str, ...]]:
    alt = klon_modul.git_praefix()
    klon_modul.git_praefix_setzen(("setpriv", "--reuid=1000", "--"))
    yield klon_modul.git_praefix()
    klon_modul.git_praefix_setzen(alt)


def _haertung_in(argv: list[str]) -> bool:
    n = len(HAERTUNG)
    return any(argv[i:i + n] == list(HAERTUNG) for i in range(len(argv) - n + 1))


def _spiegel_argv_ok(argv: list[str], pfad: Path) -> bool:
    return (argv[0] == "git" and _haertung_in(argv) and f"--git-dir={pfad}" in argv and "-C" not in argv
            and not any("setpriv" in x or "upload-pack" in x or "safe.directory" in x for x in argv))


def _env_ok(env: dict[str, str]) -> None:
    assert set(env) <= ERLAUBTE_ENV and "GH_TOKEN" not in env and "HARMLOS_ABER_FREMD" not in env
    assert env["GIT_NO_LAZY_FETCH"] == "1" and env["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert env["PATH"] == (os.environ["PATH"] if os.name == "nt" else PFAD_POSIX)


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
    assert FSCK == ("-c", "transfer.fsckObjects=true")
    assert PFAD_POSIX == "/usr/local/bin:/usr/bin:/bin"


def test_spiegel_holen_argv_und_env(tmp_path: Path, token: Path, praefix: tuple[str, ...]) -> None:
    f = Fake()
    pfad = tmp_path / "s.git"
    spiegel_holen(pfad, REPO, token, laeufer=f)
    (a,) = f.aufrufe
    argv = a["argv"]
    assert _spiegel_argv_ok(argv, pfad)  # Spiegel-Aufrufe nie mit Präfix, immer --git-dir
    assert argv[-5:] == ["fetch", "--prune", "--no-tags", spiegel_url(REPO), "+refs/heads/*:refs/heads/*"]
    assert "tok-wert-123" not in " ".join(argv)
    env = a["env"]
    _env_ok(env)
    assert env["DEBUGGER_GIT_TOKEN"] == "tok-wert-123"
    assert env["GIT_ASKPASS"] == str(tmp_path / "askpass.sh")
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GCM_INTERACTIVE"] == "never"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1" and env["GIT_CONFIG_GLOBAL"] == os.devnull
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


def test_spiegel_url_nur_github_oder_lokal(tmp_path: Path, token: Path, bundle: BundleFake) -> None:
    f = Fake()
    for falsch in ("https://evil.example/x.git", "file:///tmp/x", "ssh://github.com/Flissel/Vibemind_V1.git",
                   "-uboese", ""):
        with pytest.raises(ValueError):
            spiegel_holen(tmp_path / "s.git", REPO, token, url=falsch, laeufer=f)
        with pytest.raises(ValueError):
            spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, REPO, token, url=falsch, laeufer=f)
    assert f.aufrufe == [] and bundle.aufrufe == []
    spiegel_holen(tmp_path / "s.git", REPO, token, url=spiegel_url(REPO), laeufer=f)  # GitHub selbst: mit Token
    assert f.aufrufe[0]["env"]["DEBUGGER_GIT_TOKEN"] == "tok-wert-123"


def test_spiegel_fehler_ohne_stderr_und_url(tmp_path: Path, token: Path, bundle: BundleFake) -> None:
    with pytest.raises(GitFehler) as ei:
        spiegel_holen(tmp_path / "s.git", REPO, token, laeufer=Fake(rc=128))
    assert ei.value.unterbefehl == "fetch" and ei.value.code == 128
    assert "geheim" not in str(ei.value) and "github.com" not in str(ei.value)

    def zu_lang(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(argv, 1)

    with pytest.raises(GitFehler) as ei:
        spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, REPO, token, laeufer=zu_lang)
    assert ei.value.code == klon_modul.TIMEOUT_CODE


def test_spiegel_push_nur_debugger(tmp_path: Path, token: Path, bundle: BundleFake) -> None:
    f = Fake()
    for falsch in ("master", "main", "feature/x", "debugger/", "debugger/../master", "debugger/x:refs/heads/master",
                   "debugger/+x", "+debugger/x", "debugger/x y", "refs/heads/debugger/x", "debugger/-x"):
        with pytest.raises(ValueError):
            spiegel_push(tmp_path / "s.git", tmp_path, falsch, REPO, token, laeufer=f)
    with pytest.raises(ValueError):
        spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, "Flissel/a/b", token, laeufer=f)
    for basis in ("", "-x", "a..b", "x y"):
        with pytest.raises(ValueError):
            spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, REPO, token, basis=basis, laeufer=f)
    assert f.aufrufe == [] and bundle.aufrufe == []


def test_spiegel_push_ablauf(tmp_path: Path, token: Path, praefix: tuple[str, ...], bundle: BundleFake) -> None:
    f = SpiegelFake()
    pfad, wt = tmp_path / "s.git", tmp_path / "wt"
    spiegel_push(pfad, wt, BRANCH, REPO, token, laeufer=f)
    for a in f.aufrufe:
        assert _spiegel_argv_ok(a["argv"], pfad)
        _env_ok(a["env"])
    unter = [a["argv"][a["argv"].index(f"--git-dir={pfad}") + 1:] for a in f.aufrufe]
    # (a) Bundle durch claw (Präfix-Läufer), SHAs aus dem Spiegel, nur HEAD
    assert unter[0] == ["rev-parse", "--verify", "-q", "refs/heads/master"]
    assert unter[1] == ["rev-parse", "--verify", "-q", f"refs/heads/{BRANCH}"]
    ((ort, args, grenze),) = bundle.aufrufe
    assert ort == wt and args == ("bundle", "create", "-", "HEAD", f"^{SHA_BASIS}") and grenze == 50_000_000
    # (b) root holt das Bundle mit fsck in den Prüf-Ref
    holen = f.aufrufe[2]["argv"]
    assert holen[:len(HAERTUNG) + 3] == ["git", *HAERTUNG, *FSCK]
    assert unter[2][:2] == ["fetch", "--no-tags"] and unter[2][-1] == f"+HEAD:refs/pruefung/{BRANCH}"
    assert unter[2][2].endswith("push.bundle")
    # (c) Prüfung im Spiegel, (d) Ref = geprüfter SHA, Push ohne +, Prüf-Ref weg
    assert ["rev-list", "--objects", "--no-object-names", SHA_NEU, f"^{SHA_BASIS}"] in unter
    assert ["update-ref", f"refs/heads/{BRANCH}", SHA_NEU, "0" * 40] in unter
    push = next(a for a in f.aufrufe if "push" in a["argv"])
    assert push["argv"][-3:] == ["push", spiegel_url(REPO), f"{SHA_NEU}:refs/heads/{BRANCH}"]
    assert not any(x.startswith("+") for x in push["argv"][push["argv"].index("push"):])
    assert unter[-1] == ["update-ref", "-d", f"refs/pruefung/{BRANCH}"]
    # Token nur im Push-Prozess, nie in argv
    for a in f.aufrufe:
        assert "tok-wert-123" not in " ".join(a["argv"])
        assert ("DEBUGGER_GIT_TOKEN" in a["env"]) == (a is push)
        assert a["timeout"] == SPIEGEL_TIMEOUT_S
    assert push["env"]["DEBUGGER_GIT_TOKEN"] == "tok-wert-123" and push["stdin"] == subprocess.DEVNULL


def test_spiegel_push_prueft_token_vor_jeder_aenderung(tmp_path: Path, token: Path, bundle: BundleFake) -> None:
    f = Fake()
    with pytest.raises(RuntimeError, match="kein_token"):
        spiegel_push(tmp_path / "s.git", tmp_path, BRANCH, REPO, tmp_path / "fehlt", laeufer=f)
    assert f.aufrufe == [] and bundle.aufrufe == []


def test_spiegel_push_fehler_setzt_spiegel_zurueck(tmp_path: Path, token: Path, bundle: BundleFake) -> None:
    f = SpiegelFake(push_rc=128)
    pfad = tmp_path / "s.git"
    with pytest.raises(GitFehler) as ei:
        spiegel_push(pfad, tmp_path, BRANCH, REPO, token, laeufer=f)
    assert ei.value.unterbefehl == "push"
    unter = [a["argv"][a["argv"].index(f"--git-dir={pfad}") + 1:] for a in f.aufrufe]
    assert unter[-2] == ["update-ref", "-d", f"refs/heads/{BRANCH}", SHA_NEU]
    assert unter[-1] == ["update-ref", "-d", f"refs/pruefung/{BRANCH}"]


# ---------- echte Repos ----------

class Rundreise:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
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
        spiegel_bereit(self.spiegel)  # zweiter Aufruf prüft nur
        self.holen()
        self.klon = tmp / "klon"
        subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=false", str(self.spiegel), str(self.klon)],
                       check=True, capture_output=True)
        _konfig(self.klon)
        self.wt = worktree_bereit(self.klon, tmp / "wt" / "x-1", BRANCH, "master")
        self.kopf = commit(self.wt, "fix.txt", "fix\n")

    def holen(self) -> None:
        spiegel_holen(self.spiegel, REPO, self.token, url=str(self.github))

    def push(self, **kw: Any) -> None:
        spiegel_push(self.spiegel, self.wt, BRANCH, REPO, self.token, url=str(self.github), **kw)

    def nichts_gepusht(self) -> None:
        assert _ref(self.github, f"refs/heads/{BRANCH}") is None
        assert _ref(self.spiegel, f"refs/heads/{BRANCH}") is None
        assert _ref(self.spiegel, f"refs/pruefung/{BRANCH}") is None


@pytest.fixture
def rund(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rundreise:
    monkeypatch.delenv("DEBUGGER_ASKPASS", raising=False)
    return Rundreise(tmp_path)


def test_spiegel_rundreise_lokal(rund: Rundreise) -> None:
    assert git(rund.spiegel, "rev-parse", "--is-bare-repository") == "true"
    rund.push()
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == rund.kopf
    assert git(rund.github, "rev-parse", "refs/heads/master") != rund.kopf
    assert _ref(rund.spiegel, f"refs/pruefung/{BRANCH}") is None
    rund.holen()
    assert nachziehen(rund.klon, "master", []) is None
    assert git(rund.klon, "rev-parse", f"refs/remotes/origin/{BRANCH}") == rund.kopf
    # zweiter Push mit neuem Commit ist ein Fast-Forward; ein auseinandergelaufener Branch wird nie erzwungen
    neu = commit(rund.wt, "fix2.txt", "2\n")
    rund.push()
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == neu
    git(rund.wt, "reset", "-q", "--hard", "HEAD~2")
    commit(rund.wt, "anders.txt", "anders\n")
    with pytest.raises(GitFehler):
        rund.push()
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == neu
    assert git(rund.spiegel, "rev-parse", f"refs/heads/{BRANCH}") == neu
    assert _ref(rund.spiegel, f"refs/pruefung/{BRANCH}") is None


def test_spiegel_push_scheitert_auf_github_setzt_zurueck(rund: Rundreise) -> None:
    rund.push()
    # ein anderer Stand landet direkt auf „GitHub“; der Spiegel weiß davon noch nichts
    fremd = rund.tmp / "fremd"
    subprocess.run(["git", "clone", "-q", str(rund.github), str(fremd)], check=True, capture_output=True)
    _konfig(fremd)
    git(fremd, "checkout", "-q", BRANCH)
    fremd_kopf = commit(fremd, "fremd.txt", "f\n")
    git(fremd, "push", "-q", "origin", BRANCH)
    commit(rund.wt, "fix2.txt", "2\n")
    with pytest.raises(GitFehler) as ei:
        rund.push()
    assert ei.value.unterbefehl == "push"
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == fremd_kopf
    assert git(rund.spiegel, "rev-parse", f"refs/heads/{BRANCH}") == rund.kopf  # zurück auf den alten Stand


def test_spiegel_haelt_geheimnis_trotz_leerem_klon_bereich(rund: Rundreise) -> None:
    commit(rund.wt, "konfig.env", f"TOKEN={TOKEN}\n")
    # claw verschiebt seine eigene Prüfbasis: die Vorprüfung im Klon sieht nichts mehr
    git(rund.wt, "update-ref", f"refs/remotes/origin/{BRANCH}", "HEAD")
    assert TOKEN not in push_inhalt(rund.wt, BRANCH, "master")
    with pytest.raises(ZurueckgehaltenFehler) as ei:
        rund.push()
    assert ei.value.muster == TOKEN_MUSTER and TOKEN not in str(ei.value)
    rund.nichts_gepusht()


@pytest.mark.parametrize("ort", ["dateiname", "autor", "blob_mit_attribut"])
def test_spiegel_prueft_alle_objektarten(rund: Rundreise, ort: str) -> None:
    if ort == "dateiname":
        commit(rund.wt, f"{TOKEN}.txt", "harmlos\n")
    elif ort == "autor":
        commit(rund.wt, "x.txt", "harmlos\n", autor=TOKEN)
    else:
        (rund.wt / ".gitattributes").write_text("*.bin -diff\n", encoding="utf-8", newline="\n")
        commit(rund.wt, "w.bin", f"{TOKEN}\n")
    with pytest.raises(ZurueckgehaltenFehler) as ei:
        rund.push()
    assert ei.value.muster == TOKEN_MUSTER
    rund.nichts_gepusht()


def test_spiegel_zu_gross(rund: Rundreise) -> None:
    (rund.wt / "gross.txt").write_bytes(b"x" * 5_100_000)
    commit(rund.wt, "klein.txt", "k\n")
    with pytest.raises(ZurueckgehaltenFehler) as ei:
        rund.push()
    assert ei.value.muster == "zu_gross"
    rund.nichts_gepusht()


def test_spiegel_bundle_grenze(rund: Rundreise, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spiegel_modul, "MAX_BUNDLE", 100)
    with pytest.raises(GitFehler) as ei:
        rund.push()
    assert ei.value.unterbefehl == "bundle" and ei.value.code == GROESSE_CODE
    rund.nichts_gepusht()


def test_spiegel_push_nimmt_geprueften_sha(rund: Rundreise) -> None:
    """HEAD wandert nach der Prüfung weiter: gepusht wird genau der geprüfte Stand."""
    def laeufer(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        if "push" in argv:
            (rund.wt / "spaet.txt").write_text(f"{TOKEN}\n", encoding="utf-8")
            git(rund.wt, "add", "-A")
            git(rund.wt, "commit", "-q", "-m", "spaet")
        return subprocess.run(argv, **kw)

    rund.push(laeufer=laeufer)
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == rund.kopf
    assert git(rund.wt, "rev-parse", "HEAD") != rund.kopf


def test_spiegel_bereit_verlangt_bare(tmp_path: Path) -> None:
    normal = tmp_path / "normal"
    normal.mkdir()
    git(normal, "init", "-q")
    for falsch in (normal, normal / ".git", tmp_path):
        with pytest.raises(ValueError, match="kein_bare_repo"):
            spiegel_bereit(falsch)
    neu = tmp_path / "a" / "b.git"
    spiegel_bereit(neu)
    assert git(neu, "rev-parse", "--is-bare-repository") == "true"


ALS_CLAW = "import os, subprocess, sys; os.environ['ALS_NUTZER'] = 'claw'; sys.exit(subprocess.call(sys.argv[1:]))"


@pytest.fixture
def claw_praefix() -> Iterator[None]:
    """Präfix, der Prozesse als „claw“ kennzeichnet (Ersatz für setpriv unter Windows)."""
    alt = klon_modul.git_praefix()
    klon_modul.git_praefix_setzen([sys.executable, "-c", ALS_CLAW])
    yield
    klon_modul.git_praefix_setzen(alt)


def test_spiegel_push_promisor_im_klon_nie_als_root(rund: Rundreise, tmp_path: Path, claw_praefix: None) -> None:
    """Review-Repro: Promisor-Remote mit eigenem uploadpack-Befehl plus fehlendes Objekt im Klon.

    Ein Nachladen darf höchstens in einem Präfix-Prozess (claw) laufen, nie in einem des Wächters.
    """
    if shutil.which("sh") is None:
        pytest.skip("kein sh für den uploadpack-Befehl")
    marke = tmp_path / "MARKE"
    for k, v in (("core.repositoryformatversion", "1"), ("extensions.partialClone", "boese"),
                 ("remote.boese.url", (tmp_path / "irgendwo").as_posix()), ("remote.boese.promisor", "true"),
                 ("remote.boese.uploadpack", f"echo lief-$ALS_NUTZER >> '{marke.as_posix()}'; false")):
        git(rund.klon, "config", k, v)
    blob = git(rund.wt, "rev-parse", "HEAD:fix.txt")
    objekt = Path(git(rund.klon, "rev-parse", "--absolute-git-dir")) / "objects" / blob[:2] / blob[2:]
    objekt.chmod(0o644)
    objekt.unlink()
    with pytest.raises(GitFehler):
        rund.push()
    zeilen = marke.read_text(encoding="utf-8").split() if marke.exists() else []
    assert all(z == "lief-claw" for z in zeilen), zeilen
    rund.nichts_gepusht()


def _marken_skript(marke: Path) -> str:
    """Shell-Skript, das bei Ausführung `marke` anlegt (Pfade mit / auch auf Windows)."""
    return f'#!/bin/sh\necho lief >> "{marke.as_posix()}"\ncat > /dev/null\nexit 0\n'


def test_spiegel_ignoriert_hooks(rund: Rundreise, tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("kein sh für die Hook-Skripte")
    marke = tmp_path / "hook-lief"
    skript = _marken_skript(marke)
    for gitdir in (Path(git(rund.klon, "rev-parse", "--absolute-git-dir")), rund.spiegel):
        hooks = gitdir / "hooks"
        hooks.mkdir(exist_ok=True)
        for name in ("pre-push", "post-merge", "reference-transaction", "pre-receive", "post-update"):
            datei = hooks / name
            datei.write_text(skript, encoding="utf-8", newline="\n")
            datei.chmod(0o755)
    # Kontrolle: ohne Härtung laufen die Hooks im Spiegel wirklich
    git(rund.spiegel, "fetch", "-q", str(rund.github), "+refs/heads/master:refs/heads/kontrolle")
    assert marke.exists()
    marke.unlink()
    # Konfiguration des Helfers im Klon, die nie greifen darf
    boese = tmp_path / "boese.sh"
    boese.write_text(skript, encoding="utf-8", newline="\n")
    boese.chmod(0o755)
    git(rund.klon, "config", "uploadpack.packObjectsHook", boese.as_posix())
    git(rund.klon, "config", "core.fsmonitor", boese.as_posix())
    rund.push()
    rund.holen()
    assert not marke.exists()
    assert git(rund.github, "rev-parse", f"refs/heads/{BRANCH}") == rund.kopf
