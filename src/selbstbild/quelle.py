"""Git-Quelle: liest nur den committeten Baum (inkl. Submodule), nie den Arbeitsbaum."""
from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Literal

from selbstbild._glob import glob_passt as _ausgenommen
from selbstbild._glob import glob_regex as _glob_regex
from selbstbild.regeln import REGELDATEI, STANDARD_AUSGENOMMEN, Regeln, regeln_laden
from selbstbild.sprachen import sprache_fuer

Grund = Literal[
    "binaer", "zu_gross", "lfs", "ausgenommen", "env_datei",
    "submodul_fehlt", "keine_sprache", "symlink",
]

_LFS_KOPF = b"version https://git-lfs.github.com/spec/v1"
_GITLINK = "160000"
_SYMLINK = "120000"
_FREMDE_GIT_VARIABLEN = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES",
    "GIT_NAMESPACE", "GIT_PREFIX",
)


@dataclass(frozen=True)
class Eintrag:
    repo: str
    pfad: str
    blob: str
    groesse: int
    sprache: str | None


@dataclass(frozen=True)
class Uebersprungen:
    repo: str
    pfad: str
    grund: Grund


@dataclass(frozen=True)
class Baum:
    repo: str
    wurzel: Path
    commit: str
    commit_zeit: int
    eintraege: tuple[Eintrag, ...]
    uebersprungen: tuple[Uebersprungen, ...]
    unterbaeume: tuple["Baum", ...]
    regeln: Regeln


# --- Git-Aufrufe -----------------------------------------------------------

_LEERE_HOOKS = tempfile.mkdtemp(prefix="selbstbild-hooks-")


def _git_args(*args: str) -> list[str]:
    return [
        "git", "-c", f"core.hooksPath={_LEERE_HOOKS}", "-c", "core.quotepath=off",
        *args,
    ]


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _FREMDE_GIT_VARIABLEN}
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return env


def _git(wurzel: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        _git_args(*args), cwd=wurzel, env=_git_env(),
        capture_output=True, check=check,
    )


class BlobLeser:
    """Ein `git cat-file --batch`-Prozess je Repo; als Kontextmanager benutzen."""

    def __init__(self, wurzel: Path) -> None:
        self._wurzel = wurzel
        self._prozess: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "BlobLeser":
        self._prozess = subprocess.Popen(
            _git_args("cat-file", "--batch"), cwd=self._wurzel, env=_git_env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        p = self._prozess
        self._prozess = None
        if p is None:
            return
        try:
            if p.stdin:
                p.stdin.close()
            if p.stdout:
                p.stdout.close()
        finally:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def lesen(self, blob: str) -> bytes:
        p = self._prozess
        if p is None or p.stdin is None or p.stdout is None:
            raise RuntimeError("BlobLeser ist nicht geoeffnet")
        stdin: IO[bytes] = p.stdin
        stdout: IO[bytes] = p.stdout
        stdin.write(blob.encode("ascii") + b"\n")
        stdin.flush()
        kopf = stdout.readline().split()
        if len(kopf) != 3:
            raise KeyError(blob)
        groesse = int(kopf[2])
        daten = stdout.read(groesse)
        stdout.read(1)  # abschliessendes LF
        return daten


# --- Baum ------------------------------------------------------------------

def _regeln_aus_commit(wurzel: Path, commit: str) -> Regeln:
    erg = _git(wurzel, "cat-file", "blob", f"{commit}:{REGELDATEI}", check=False)
    if erg.returncode != 0:
        return regeln_laden(None)
    return regeln_laden(erg.stdout.decode("utf-8"))


def _submodul_vorhanden(wurzel: Path, pfad: str, sha: str) -> Path | None:
    ziel = wurzel / pfad
    if not (ziel / ".git").exists():
        return None  # nicht initialisiert; rev-parse wuerde das Elternrepo finden
    if _git(ziel, "rev-parse", "--git-dir", check=False).returncode != 0:
        return None
    if _git(ziel, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode != 0:
        return None
    return ziel


def _commit_sha(wurzel: Path, commit: str) -> str:
    return _git(wurzel, "rev-parse", f"{commit}^{{commit}}").stdout.decode().strip()


def repo_name(wurzel: Path, commit: str = "HEAD") -> str:
    """Repo-Name wie in `baum_lesen` (Regeldatei im Commit, sonst Ordnername), ohne den Baum
    zu lesen; z. B. um den lokalen Stand vor `baum_lesen` zu oeffnen."""
    wurzel = Path(wurzel)
    return _regeln_aus_commit(wurzel, _commit_sha(wurzel, commit)).name or wurzel.resolve().name


# (blob, sprache, pfad) -> ist das Parser-Ergebnis schon bekannt (Cache-Treffer)?
Bekannt = Callable[[str, str, str], bool]


def baum_lesen(
    wurzel: Path, commit: str = "HEAD", name: str | None = None, bekannt: Bekannt | None = None,
) -> Baum:
    """Committeter Baum mit Submodulen. Gelesen wird ein Blob nur, wenn seine Sprache bekannt
    ist und `bekannt` ihn nicht als Cache-Treffer meldet: Ein Treffer war schon einmal als
    Text (nicht binaer, kein LFS) klassifiziert und geparst, sein Inhalt wird nicht gebraucht."""
    wurzel = Path(wurzel)
    sha = _commit_sha(wurzel, commit)
    zeit = int(_git(wurzel, "log", "-1", "--no-show-signature", "--format=%ct", sha).stdout.decode().strip())
    regeln = _regeln_aus_commit(wurzel, sha)
    repo = name or regeln.name or wurzel.resolve().name
    muster = [_glob_regex(m) for m in (*STANDARD_AUSGENOMMEN, *regeln.ausgenommen)]
    max_bytes = regeln.max_dateigroesse_kb * 1024

    roh = _git(wurzel, "ls-tree", "-r", "-l", "-z", sha).stdout
    eintraege: list[Eintrag] = []
    uebersprungen: list[Uebersprungen] = []
    unterbaeume: list[Baum] = []

    with BlobLeser(wurzel) as leser:
        for satz in roh.split(b"\0"):
            if not satz:
                continue
            kopf, _, pfad_roh = satz.partition(b"\t")
            modus, art, blob, groesse_s = kopf.decode("ascii").split()
            pfad = pfad_roh.decode("utf-8", errors="surrogateescape")

            if modus == _GITLINK:
                if _ausgenommen(pfad, muster):
                    uebersprungen.append(Uebersprungen(repo, pfad, "ausgenommen"))
                    continue
                ziel = _submodul_vorhanden(wurzel, pfad, blob)
                if ziel is None:
                    uebersprungen.append(Uebersprungen(repo, pfad, "submodul_fehlt"))
                else:
                    unterbaeume.append(baum_lesen(ziel, blob, name=f"{repo}/{pfad}", bekannt=bekannt))
                continue
            if art != "blob":
                continue
            if _ausgenommen(pfad, muster):
                uebersprungen.append(Uebersprungen(repo, pfad, "ausgenommen"))
                continue
            if modus == _SYMLINK:
                uebersprungen.append(Uebersprungen(repo, pfad, "symlink"))
                continue
            if PurePosixPath(pfad).name == ".env":
                uebersprungen.append(Uebersprungen(repo, pfad, "env_datei"))
                continue
            groesse = int(groesse_s)
            if groesse > max_bytes:
                uebersprungen.append(Uebersprungen(repo, pfad, "zu_gross"))
                continue
            sprache = sprache_fuer(pfad)
            if sprache is None:  # vor dem Lesen: Blobs ohne Sprache werden nie gelesen
                uebersprungen.append(Uebersprungen(repo, pfad, "keine_sprache"))
                continue
            if bekannt is None or not bekannt(blob, sprache, pfad):
                inhalt = leser.lesen(blob)
                if inhalt.startswith(_LFS_KOPF):
                    uebersprungen.append(Uebersprungen(repo, pfad, "lfs"))
                    continue
                if b"\0" in inhalt[:8192]:
                    uebersprungen.append(Uebersprungen(repo, pfad, "binaer"))
                    continue
                del inhalt
            eintraege.append(Eintrag(repo, pfad, blob, groesse, sprache))

    eintraege.sort(key=lambda e: (e.repo, e.pfad))
    uebersprungen.sort(key=lambda u: (u.repo, u.pfad))
    unterbaeume.sort(key=lambda b: b.repo)
    return Baum(
        repo=repo, wurzel=wurzel, commit=sha, commit_zeit=zeit,
        eintraege=tuple(eintraege), uebersprungen=tuple(uebersprungen),
        unterbaeume=tuple(unterbaeume), regeln=regeln,
    )
