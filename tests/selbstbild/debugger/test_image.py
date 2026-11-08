"""Drift-Wächter: Das Debugger-Image bringt jedes Programm mit, das der Code im Container aufruft.

Die übrigen Tests ersetzen `laufzeit`/`laeufer` durch Fakes und merken daher nicht, wenn ein Programm
im Image fehlt (so fehlte `gh` im ersten VM-Build, R52 → R53).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

from selbstbild.funde.github import GitHub

DOCKERFILE = Path(__file__).resolve().parents[3] / "deploy" / "debugger" / "Dockerfile"


def _runs() -> str:
    """Nur die RUN-Anweisungen (mit Fortsetzungszeilen), ohne Kommentare."""
    zeilen = [z for z in DOCKERFILE.read_text(encoding="utf-8").splitlines() if not z.lstrip().startswith("#")]
    text = "\n".join(zeilen).replace("\\\n", " ")
    return "\n".join(z for z in text.splitlines() if z.lstrip().startswith("RUN "))


def test_gh_im_image_und_geprueft() -> None:
    assert inspect.signature(GitHub).parameters["gh"].default == "gh"
    runs = _runs()
    assert re.search(r"apt-get install [^&]*/tmp/gh\.deb", runs)
    assert "sha256sum -c" in runs
    assert "gh --version" in runs


def test_uebrige_programme_im_image() -> None:
    runs = _runs()
    for paket in ("git", "util-linux", "procps"):  # git, setpriv, pkill
        assert re.search(rf"apt-get install [^&]*\b{re.escape(paket)}\b", runs), paket
    assert "@anthropic-ai/claude-code@" in runs  # claude
