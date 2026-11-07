"""Hermetische Git-Fixture fuer die selbstbild-Tests, dazu die Enkel-Hilfe der Debugger-Tests.

Es gibt bewusst nur diese eine conftest.py unter tests/selbstbild: Test-Ordner ohne
`__init__.py` laden jede conftest.py als Modul `conftest`, und `from conftest import GitRepo`
hinge sonst von der Ladereihenfolge ab.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest


class GitRepo:
    def __init__(self, wurzel: Path, env: dict[str, str]) -> None:
        self.wurzel = wurzel
        self._env = env
        wurzel.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str, datum: str | None = None) -> str:
        env = dict(self._env)
        if datum:
            env["GIT_AUTHOR_DATE"] = datum
            env["GIT_COMMITTER_DATE"] = datum
        erg = subprocess.run(
            ["git", *args], cwd=self.wurzel, env=env, check=True,
            capture_output=True,
        )
        return erg.stdout.decode("utf-8", "replace").strip()

    def datei(self, pfad: str, inhalt: str | bytes) -> None:
        ziel = self.wurzel / pfad
        ziel.parent.mkdir(parents=True, exist_ok=True)
        ziel.write_bytes(inhalt.encode("utf-8") if isinstance(inhalt, str) else inhalt)

    def commit(self, nachricht: str, datum: str = "2026-01-01T00:00:00Z") -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", nachricht, datum=datum)
        return self.git("rev-parse", "HEAD")

    def mv(self, alt: str, neu: str) -> None:
        (self.wurzel / neu).parent.mkdir(parents=True, exist_ok=True)
        self.git("mv", alt, neu)

    def submodul(self, pfad: str, anderes: "GitRepo") -> None:
        self.git(
            "-c", "protocol.file.allow=always",
            "submodule", "add", "-q", str(anderes.wurzel), pfad,
        )


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GitRepo:
    leer_config = tmp_path / "leer.gitconfig"
    leer_config.write_text("", encoding="utf-8")
    hooks = tmp_path / "leere-hooks"
    hooks.mkdir()
    env_neu = {
        "GIT_CONFIG_GLOBAL": str(leer_config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Tester",
        "GIT_AUTHOR_EMAIL": "t@example.org",
        "GIT_COMMITTER_NAME": "Tester",
        "GIT_COMMITTER_EMAIL": "t@example.org",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
        "GIT_LFS_SKIP_SMUDGE": "1",
    }
    for k, v in env_neu.items():
        monkeypatch.setenv(k, v)
    env = dict(os.environ)
    repo = GitRepo(tmp_path / "repo", env)
    repo.git("config", "core.hooksPath", str(hooks))
    repo.git("config", "core.autocrlf", "false")
    return repo


@pytest.fixture
def graph_stand(tmp_path: Path):
    """Schreibt ueber `Speicher` einen Graph-Stand und liefert dessen Ordner."""
    from selbstbild.speicher import Speicher

    zaehler = [0]

    def bauen(knoten, kanten, commit: str = "c1", commit_zeit: int = 1_700_000_000) -> Path:
        zaehler[0] += 1
        ordner = tmp_path / f"stand{zaehler[0]}"
        with Speicher(ordner / "graph.sqlite") as s:
            s.graph_ersetzen(commit, knoten, kanten)
            s.meta_setzen("commit_zeit", str(commit_zeit))
        return ordner

    return bauen


# ---------- Debugger: ein Präfix, der einen Enkel mit offenem stdout startet (S6) ----------

# Kind: startet einen Enkel, der das stdout des Kindes erbt und lange schläft, schreibt dessen
# pid nach argv[1], gibt argv[3] Bytes auf stdout aus und schläft dann argv[2] Sekunden
# (0 = sofort beenden).
_ENKEL = (
    "import subprocess, sys, time\n"
    "enkel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=sys.stdout)\n"
    "open(sys.argv[1], 'w').write(str(enkel.pid))\n"
    "sys.stdout.buffer.write(b'x' * int(sys.argv[3]))\n"
    "sys.stdout.buffer.flush()\n"
    "time.sleep(float(sys.argv[2]))\n"
)


@dataclass
class Enkel:
    pid_datei: Path

    def praefix(self, kind_schlaeft_s: float, ausgabe: int = 0) -> list[str]:
        """argv-Präfix; alles, was der Läufer danach anhängt, ignoriert das Kind."""
        return [sys.executable, "-c", _ENKEL, str(self.pid_datei), str(kind_schlaeft_s), str(ausgabe)]

    def pid(self) -> int | None:
        for _ in range(50):
            if self.pid_datei.exists() and self.pid_datei.read_text(encoding="utf-8").strip():
                return int(self.pid_datei.read_text(encoding="utf-8"))
            time.sleep(0.1)
        return None

    def tot_pruefen(self) -> None:
        """POSIX: der Enkel lebt nach dem Timeout nicht mehr (Windows: kein Gruppen-Kill, nur Ende)."""
        if os.name == "nt":
            return
        pid = self.pid()
        assert pid is not None
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        raise AssertionError("Enkel lebt noch")


@pytest.fixture
def enkel(tmp_path: Path) -> Iterator[Enkel]:
    e = Enkel(tmp_path / "enkel.pid")
    yield e
    pid = e.pid() if e.pid_datei.exists() else None
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
