"""Spiegel des Debugger-Claw: der einzige Ort, an dem git mit GitHub spricht (Rulings R16, R20).

Je Repo gibt es ein bare Repo `<spiegel_ordner>/<owner>__<name>.git`, das dem Wächter (root)
gehört. Klon und Worktrees des Helfers (`claw`) haben den Spiegel als `origin` und kommen nie
an den Token. Nur die Prozesse hier bekommen ihn, und nur über `GIT_ASKPASS` plus
`DEBUGGER_GIT_TOKEN` in genau ihrer Umgebung; die URL wird aus dem Repo-Namen gebaut, nie
aus Remote-Konfiguration. Jeder Aufruf läuft mit `HAERTUNG`, ohne System- und globale
Konfiguration, ohne Eingabe und mit Zeitgrenze. Fehler nennen nur Unterbefehl und Code.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Callable

from selbstbild.debugger.klon import TIMEOUT_CODE, GitFehler, debugger_zweig_pruefen

SPIEGEL_TIMEOUT_S = 300
HAERTUNG: tuple[str, ...] = (
    "-c", "credential.helper=",
    "-c", f"core.hooksPath={os.devnull}",
    "-c", "core.fsmonitor=false",
    "-c", "protocol.allow=never",
    "-c", "protocol.https.allow=always",
    "-c", "protocol.file.allow=always",
)
# Upload-Pack läuft im fremden Repo (Klon des Helfers) und liest dort nur Objekte; ohne
# `safe.directory` verweigert git als root ein Repo, das `claw` gehört. Kein Präfix: root liest.
UPLOAD_PACK = " ".join(shlex.quote(x) for x in (
    "git", "-c", "safe.directory=*", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false",
    "upload-pack",
))

_REPO = re.compile(r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+")
_ENV_DURCH = ("PATH", "HOME", "LANG", "SYSTEMROOT")  # SYSTEMROOT nur unter Windows (Tests)

Laeufer = Callable[..., subprocess.CompletedProcess[str]]


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
    """Minimale Umgebung; mit `token_datei` zusätzlich Askpass und Token (nur für GitHub)."""
    env = {k: os.environ[k] for k in _ENV_DURCH if k in os.environ and (k != "SYSTEMROOT" or os.name == "nt")}
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
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
    """(URL, braucht_token): ohne `url` GitHub; eine lokale `url` (Tests) braucht keinen Token."""
    standard = spiegel_url(repo)
    if url is None:
        return standard, True
    return url, "://" in url


def _git(
    pfad: Path, args: list[str], env: dict[str, str], laeufer: Laeufer, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        erg = laeufer(
            ["git", *HAERTUNG, "-C", str(pfad), *args],
            check=False, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env,
            timeout=SPIEGEL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise GitFehler(args[0], TIMEOUT_CODE) from None
    if check and erg.returncode != 0:
        raise GitFehler(args[0], erg.returncode)
    return erg


def spiegel_bereit(pfad: Path, *, laeufer: Laeufer = subprocess.run) -> None:
    """Legt den Spiegel als bare Repo an, wenn er fehlt; sonst nichts."""
    if pfad.exists():
        return
    pfad.parent.mkdir(parents=True, exist_ok=True)
    try:
        erg = laeufer(
            ["git", *HAERTUNG, "init", "-q", "--bare", str(pfad)],
            check=False, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=_env(None),
            timeout=SPIEGEL_TIMEOUT_S,
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
    _git(pfad, ["fetch", "--prune", "--no-tags", ziel, "+refs/heads/*:refs/heads/*"], env, laeufer)


def spiegel_push(
    pfad: Path, wt: Path, branch: str, repo: str, token_datei: Path, *, url: str | None = None,
    laeufer: Laeufer = subprocess.run,
) -> None:
    """Pusht `HEAD` des Worktrees als `refs/heads/<branch>` über den Spiegel nach GitHub.

    Nur `debugger/...` (sonst `ValueError` vor jeder Änderung). Erst holt der Spiegel die
    Objekte aus dem Worktree (lokal, ohne Token), dann pusht er genau diesen einen Branch
    ohne `+`. Token und Askpass werden vor dem ersten Schritt geprüft. Scheitert der Push,
    zeigt der Branch im Spiegel wieder auf den alten Stand (oder fehlt wieder), damit der Klon
    ihn beim nächsten Fetch nicht für gepusht hält.
    """
    debugger_zweig_pruefen(branch)
    ziel, braucht_token = _ziel(repo, url)
    push_env, lokal_env = _env(token_datei if braucht_token else None), _env(None)
    ref = f"refs/heads/{branch}"
    alt = _git(pfad, ["rev-parse", "--verify", "-q", ref], lokal_env, laeufer, check=False)
    vorher = alt.stdout.strip() if alt.returncode == 0 else ""
    _git(pfad, ["fetch", "--no-tags", "--upload-pack", UPLOAD_PACK, str(wt), f"+HEAD:{ref}"], lokal_env, laeufer)
    try:
        _git(pfad, ["push", ziel, f"{ref}:{ref}"], push_env, laeufer)
    except GitFehler:
        zurueck = ["update-ref", ref, vorher] if vorher else ["update-ref", "-d", ref]
        _git(pfad, zurueck, lokal_env, laeufer, check=False)
        raise
