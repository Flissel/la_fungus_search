"""Spiegel des Debugger-Claw: der einzige Ort, an dem git mit GitHub spricht (R16, R20, R29).

Je Repo gibt es ein bare Repo `<spiegel_ordner>/<owner>__<name>.git`, das dem Wächter (root)
gehört. Klon und Worktrees des Helfers (`claw`) haben den Spiegel als `origin` und kommen nie
an den Token. Nur die GitHub-Prozesse hier bekommen ihn, und nur über `GIT_ASKPASS` plus
`DEBUGGER_GIT_TOKEN` in genau ihrer Umgebung; die URL wird aus dem Repo-Namen gebaut, nie
aus Remote-Konfiguration. Kein Prozess des Wächters läuft im Repo des Helfers: Objekte kommen
nur als Bundle, das `claw` über das Präfix erzeugt (R29). Jeder Spiegel-Aufruf läuft mit
`HAERTUNG` und `--git-dir`, ohne System- und globale Konfiguration, ohne Eingabe und mit
Zeitgrenze. Fehler nennen nur Unterbefehl und Code.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from selbstbild.debugger.klon import (
    MAX_PUSH_INHALT,
    TIMEOUT_CODE,
    ZU_GROSS,
    GitFehler,
    debugger_zweig_pruefen,
    git_in_datei,
)
from selbstbild.funde.text import pruefen

SPIEGEL_TIMEOUT_S = 300
MAX_BUNDLE = 50_000_000
PFAD_POSIX = "/usr/local/bin:/usr/bin:/bin"
HAERTUNG: tuple[str, ...] = (
    "-c", "credential.helper=",
    "-c", f"core.hooksPath={os.devnull}",
    "-c", "core.fsmonitor=false",
    "-c", "protocol.allow=never",
    "-c", "protocol.https.allow=always",
    "-c", "protocol.file.allow=always",
)
FSCK = ("-c", "transfer.fsckObjects=true")
BUNDLE_REF = "HEAD"  # das Bundle trägt HEAD des Worktrees, nie einen Branch-Namen aus dem Klon
NULL_SHA = "0" * 40

_REPO = re.compile(r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+")
_BASIS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_SHA = re.compile(r"[0-9a-f]{40}")
LESEN = ("rev-parse", "cat-file", "merge-base", "ls-tree")  # spiegel_lesen: nur diese Unterbefehle
PIN_INHALT = "pin_inhalt"
_DIFF_ZEILE = re.compile(r":(\d{6}) (\d{6}) ([0-9a-f]{40}) ([0-9a-f]{40}) ([A-Z])	(.+)")

Laeufer = Callable[..., subprocess.CompletedProcess[bytes]]


class ZurueckgehaltenFehler(Exception):
    """Die Prüfung im Spiegel hat ein Muster getroffen (oder `zu_gross`); nichts wurde gepusht."""

    def __init__(self, muster: str) -> None:
        super().__init__(muster)
        self.muster = muster


def _repo_pruefen(repo: str) -> tuple[str, str]:
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise ValueError("repo_ungueltig")
    owner, name = repo.split("/")
    if name in (".", "..") or name.endswith(".git") or owner.startswith("-") or name.startswith("-"):
        raise ValueError("repo_ungueltig")
    return owner, name


def spiegel_url(repo: str) -> str:
    """`https://github.com/<repo>.git`; `repo` muss `owner/name` aus `[A-Za-z0-9._-]` sein."""
    _repo_pruefen(repo)
    return f"https://github.com/{repo}.git"


def spiegel_pfad(spiegel_ordner: Path, repo: str) -> Path:
    """Ort des Spiegels: `<spiegel_ordner>/<owner>__<name>.git`."""
    owner, name = _repo_pruefen(repo)
    return spiegel_ordner / f"{owner}__{name}.git"


def _env(token_datei: Path | None) -> dict[str, str]:
    """Minimale Umgebung mit festem PATH; mit `token_datei` zusätzlich Askpass und Token."""
    windows = os.name == "nt"  # nur für die Tests: dort bleibt der PATH des Eltern-Prozesses
    env = {"PATH": os.environ.get("PATH", "") if windows else PFAD_POSIX}
    for k in ("HOME", "LANG") + (("SYSTEMROOT",) if windows else ()):
        if k in os.environ:
            env[k] = os.environ[k]
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
    })
    if token_datei is None:
        return env
    askpass = os.environ.get("DEBUGGER_ASKPASS", "")
    if not askpass:
        raise RuntimeError("kein_askpass")
    try:
        token = token_datei.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        token = ""
    if not token:
        raise RuntimeError("kein_token")
    env["GIT_ASKPASS"] = askpass
    env["DEBUGGER_GIT_TOKEN"] = token
    return env


def _ziel(repo: str, url: str | None) -> tuple[str, bool]:
    """(URL, braucht_token). Ohne `url` GitHub; eine andere `url` nur als lokaler Pfad (Tests)."""
    standard = spiegel_url(repo)
    if url is None or url == standard:
        return standard, True
    if "://" in url or url.startswith("-") or not url:
        raise ValueError("url_nicht_erlaubt")
    return url, False


def _lauf(
    pfad: Path, args: list[str], env: dict[str, str], laeufer: Laeufer, *, check: bool = True,
    eingabe: bytes | None = None, extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[bytes]:
    """`git <HAERTUNG> <extra> --git-dir=<pfad> ...` (Bytes); Fehler nur als GitFehler."""
    ein: dict[str, object] = {"stdin": subprocess.DEVNULL} if eingabe is None else {"input": eingabe}
    try:
        erg = laeufer(
            ["git", *HAERTUNG, *extra, f"--git-dir={pfad}", *args],
            check=False, capture_output=True, env=env, timeout=SPIEGEL_TIMEOUT_S, **ein,
        )
    except subprocess.TimeoutExpired:
        raise GitFehler(args[0], TIMEOUT_CODE) from None
    if check and erg.returncode != 0:
        raise GitFehler(args[0], erg.returncode)
    return erg


def _sha(pfad: Path, ref: str, env: dict[str, str], laeufer: Laeufer, *, pflicht: bool = True) -> str:
    """SHA von `ref` im Spiegel; fehlt er: `""` oder (pflicht) GitFehler."""
    erg = _lauf(pfad, ["rev-parse", "--verify", "-q", ref], env, laeufer, check=False)
    sha = bytes(erg.stdout).decode("ascii", errors="replace").strip() if erg.returncode == 0 else ""
    if sha and not _SHA.fullmatch(sha):
        raise GitFehler("rev-parse", 1)
    if pflicht and not sha:
        raise GitFehler("rev-parse", erg.returncode or 1)
    return sha


def spiegel_lesen(pfad: Path, *args: str, laeufer: Laeufer = subprocess.run) -> tuple[int, str]:
    """Liest im Spiegel als root (R39): `git <HAERTUNG> --git-dir=<pfad> <args>`, ohne Token.

    Nur Unterbefehle aus `LESEN`; alles andere ergibt `ValueError`. Rückgabe: Code und stdout
    (UTF-8 mit Ersatzzeichen); stderr wird nie gezeigt. Zeitgrenze wie alle Spiegel-Aufrufe.
    """
    if not args or args[0] not in LESEN:
        raise ValueError("nicht_lesend")
    erg = _lauf(pfad, list(args), _env(None), laeufer, check=False)
    return erg.returncode, bytes(erg.stdout).decode("utf-8", errors="replace")


def _nur_gitlink_ok(pfad: Path, sha: str, basis_sha: str, ohne: list[str], nur_gitlink: tuple[str, str],
                    env: dict[str, str], laeufer: Laeufer) -> bool:
    """R39: genau ein neuer Commit, einziger Elter `basis_sha`, einzige Änderung der Gitlink am Pfad."""
    sub_pfad, erwartet = nur_gitlink
    neu = bytes(_lauf(pfad, ["rev-list", sha, *(f"^{x}" for x in ohne)], env, laeufer).stdout).split()
    if neu != [sha.encode("ascii")]:
        return False
    eltern = bytes(_lauf(pfad, ["rev-list", "--parents", "-n", "1", sha], env, laeufer).stdout).split()
    if eltern != [sha.encode("ascii"), basis_sha.encode("ascii")]:
        return False
    diff = bytes(_lauf(pfad, ["diff-tree", "-r", "--no-commit-id", basis_sha, sha], env, laeufer).stdout)
    zeilen = [z for z in diff.decode("utf-8", errors="replace").splitlines() if z.strip()]
    m = _DIFF_ZEILE.fullmatch(zeilen[0]) if len(zeilen) == 1 else None
    return m is not None and m.group(2) == "160000" and m.group(6) == sub_pfad and m.group(4) == erwartet


def spiegel_bereit(pfad: Path, *, laeufer: Laeufer = subprocess.run) -> None:
    """Legt den Spiegel als bare Repo an, wenn er fehlt; ein vorhandener Pfad muss bare sein."""
    env = _env(None)
    if pfad.exists():
        erg = _lauf(pfad, ["rev-parse", "--is-bare-repository"], env, laeufer, check=False)
        if erg.returncode != 0 or bytes(erg.stdout).strip() != b"true":
            raise ValueError("kein_bare_repo")
        return
    pfad.parent.mkdir(parents=True, exist_ok=True)
    try:
        erg = laeufer(
            ["git", *HAERTUNG, "init", "-q", "--bare", str(pfad)],
            check=False, capture_output=True, stdin=subprocess.DEVNULL, env=env, timeout=SPIEGEL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise GitFehler("init", TIMEOUT_CODE) from None
    if erg.returncode != 0:
        raise GitFehler("init", erg.returncode)


def spiegel_holen(
    pfad: Path, repo: str, token_datei: Path, *, url: str | None = None, laeufer: Laeufer = subprocess.run
) -> None:
    """Holt alle Branches von GitHub in den Spiegel (`fetch --prune --no-tags`, erzwungen)."""
    ziel, braucht_token = _ziel(repo, url)
    env = _env(token_datei if braucht_token else None)
    _lauf(pfad, ["fetch", "--prune", "--no-tags", ziel, "+refs/heads/*:refs/heads/*"], env, laeufer)


def _objekte_pruefen(pfad: Path, sha: str, ohne: list[str], env: dict[str, str], laeufer: Laeufer) -> str | None:
    """Prüft alle neuen Objekte (Commits mit Kopf, Bäume mit Namen, Blobs, Tags) roh im Spiegel.

    Rückgabe: Mustername, `zu_gross` über `MAX_PUSH_INHALT`, oder None. Antwortet
    `cat-file` mit einer anderen Id als angefragt, folgt GitFehler.
    """
    liste = _lauf(pfad, ["rev-list", "--objects", "--no-object-names", sha, *(f"^{x}" for x in ohne)], env, laeufer)
    ids = [z.strip() for z in bytes(liste.stdout).decode("ascii", errors="replace").splitlines() if z.strip()]
    if not ids or any(not _SHA.fullmatch(i) for i in ids):
        raise GitFehler("rev-list", 1)
    eingabe = "".join(f"{i}\n" for i in ids).encode("ascii")
    kopfzeilen = bytes(_lauf(pfad, ["cat-file", "--batch-check"], env, laeufer, eingabe=eingabe).stdout).splitlines()
    if len(kopfzeilen) != len(ids):
        raise GitFehler("cat-file", 1)
    gesamt = 0
    for i, zeile in zip(ids, kopfzeilen):
        teile = zeile.split()
        if len(teile) != 3 or teile[0].decode("ascii", errors="replace") != i or not teile[2].isdigit():
            raise GitFehler("cat-file", 1)
        gesamt += int(teile[2])
    if gesamt > MAX_PUSH_INHALT:
        return ZU_GROSS
    roh = bytes(_lauf(pfad, ["cat-file", "--batch"], env, laeufer, eingabe=eingabe).stdout)
    inhalte: list[str] = []
    pos = 0
    for i in ids:
        ende = roh.find(b"\n", pos)
        kopf = roh[pos:ende].split() if ende >= 0 else []
        if len(kopf) != 3 or kopf[0].decode("ascii", errors="replace") != i or not kopf[2].isdigit():
            raise GitFehler("cat-file", 1)
        start, groesse = ende + 1, int(kopf[2])
        if start + groesse > len(roh):
            raise GitFehler("cat-file", 1)
        inhalte.append(roh[start:start + groesse].decode("utf-8", errors="replace"))
        pos = start + groesse + 1
    return pruefen("\n".join(inhalte))


_PIN_BRANCH = re.compile(r"debugger/\d+-pin(?:-\d+)?")


def ist_pin_branch(branch: str) -> bool:
    """R45: `debugger/<n>-pin`, auch mit Runden-Suffix `-<runde>` (das Muster von `pin_branch`)."""
    return isinstance(branch, str) and _PIN_BRANCH.fullmatch(branch) is not None


def spiegel_push(
    pfad: Path, wt: Path, branch: str, repo: str, token_datei: Path, *, basis: str = "master",
    url: str | None = None, laeufer: Laeufer = subprocess.run, nur_gitlink: tuple[str, str] | None = None,
) -> None:
    """Pusht `HEAD` des Worktrees als `refs/heads/<branch>` über den Spiegel nach GitHub (R29).

    1. `branch` muss `debugger/...` sein, `url` erlaubt, Token und Askpass da (vor jeder Änderung).
    2. `claw` erzeugt über das Präfix ein Bundle `HEAD ^<basis> [^<branch>]` mit den SHAs aus dem
       Spiegel; root schreibt es in eine eigene Temp-Datei (höchstens `MAX_BUNDLE`).
    3. root holt das Bundle mit fsck nach `refs/pruefung/<branch>` und prüft dort alle neuen
       Objekte roh; Treffer oder `zu_gross` → `ZurueckgehaltenFehler`, nichts wird gepusht.
    4. `refs/heads/<branch>` = geprüfter SHA (nur neu oder Fast-Forward), Push `<sha>:refs/heads/
       <branch>` ohne `+`; scheitert er, zeigt der Spiegel-Branch wieder auf den alten Stand.
    `refs/pruefung/<branch>` wird immer gelöscht.

    `nur_gitlink=(Submodul-Pfad, erwarteter SHA)` (Pin-Push, R39): nach dem Bundle-Fetch muss es
    genau einen neuen Commit geben, dessen einziger Elter der Basis-SHA des Spiegels ist und der
    nur den Gitlink am Pfad auf den erwarteten SHA setzt; sonst `ZurueckgehaltenFehler("pin_inhalt")`.
    Ein Pin-Branch (`ist_pin_branch`) ohne `nur_gitlink` wird vor jedem Prozess abgewiesen (R45).
    """
    debugger_zweig_pruefen(branch)
    if not isinstance(basis, str) or not _BASIS.fullmatch(basis) or ".." in basis:
        raise ValueError("basis_ungueltig")
    if nur_gitlink is not None and (len(nur_gitlink) != 2 or not _SHA.fullmatch(nur_gitlink[1])
                                    or not nur_gitlink[0] or nur_gitlink[0].startswith(("-", "/"))):
        raise ValueError("nur_gitlink_ungueltig")
    if nur_gitlink is None and ist_pin_branch(branch):  # R45: die Grenze, nicht die Aufrufer
        raise ZurueckgehaltenFehler(PIN_INHALT)
    ziel, braucht_token = _ziel(repo, url)
    push_env, env = _env(token_datei if braucht_token else None), _env(None)
    kopf, pruef = f"refs/heads/{branch}", f"refs/pruefung/{branch}"
    basis_sha = _sha(pfad, f"refs/heads/{basis}", env, laeufer)
    alt = _sha(pfad, kopf, env, laeufer, pflicht=False)
    ohne = [basis_sha] + ([alt] if alt else [])
    try:
        with tempfile.TemporaryDirectory(prefix="debugger-bundle-") as tmp:
            datei = Path(tmp) / "push.bundle"
            with datei.open("wb") as aus:
                git_in_datei(wt, "bundle", "create", "-", BUNDLE_REF, *(f"^{x}" for x in ohne),
                             ziel=aus, grenze=MAX_BUNDLE)
            _lauf(pfad, ["fetch", "--no-tags", str(datei), f"+{BUNDLE_REF}:{pruef}"], env, laeufer, extra=FSCK)
        sha = _sha(pfad, pruef, env, laeufer)
        typ = _lauf(pfad, ["cat-file", "-t", sha], env, laeufer)
        if bytes(typ.stdout).strip() != b"commit":
            raise GitFehler("cat-file", 1)
        if nur_gitlink is not None and not _nur_gitlink_ok(pfad, sha, basis_sha, ohne, nur_gitlink, env, laeufer):
            raise ZurueckgehaltenFehler(PIN_INHALT)
        muster = _objekte_pruefen(pfad, sha, ohne, env, laeufer)
        if muster is not None:
            raise ZurueckgehaltenFehler(muster)
        if alt and _lauf(pfad, ["merge-base", "--is-ancestor", alt, sha], env, laeufer, check=False).returncode != 0:
            raise GitFehler("merge-base", 1)  # kein Fast-Forward des Spiegel-Branches
        _lauf(pfad, ["update-ref", kopf, sha, alt or NULL_SHA], env, laeufer)
        try:
            _lauf(pfad, ["push", ziel, f"{sha}:{kopf}"], push_env, laeufer)
        except GitFehler:
            zurueck = ["update-ref", kopf, alt, sha] if alt else ["update-ref", "-d", kopf, sha]
            _lauf(pfad, zurueck, env, laeufer, check=False)
            raise
    finally:
        _lauf(pfad, ["update-ref", "-d", pruef], env, laeufer, check=False)
