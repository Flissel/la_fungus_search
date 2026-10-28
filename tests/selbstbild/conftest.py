"""Hermetische Git-Fixture fuer die selbstbild-Tests."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
