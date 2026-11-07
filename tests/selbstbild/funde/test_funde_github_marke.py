"""Debugger-Claw Task 1: Automatik-Marke an Kommentaren, Token aus Datei (gefaelschtes gh, nie Netz)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from selbstbild.funde.github import AUTO_MARKE, DEBUGGER_MARKE, GitHub, GitHubFehler


class Aufzeichner:
    """Fake-Laufzeit: merkt sich Aufrufe, antwortet auf `auth token` und `api`."""

    def __init__(self) -> None:
        self.aufrufe: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, args: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        argv = list(args)
        self.aufrufe.append((argv, kw))
        if argv[1:3] == ["auth", "token"]:
            return subprocess.CompletedProcess(argv, 0, b"vomgh\n", b"")
        return subprocess.CompletedProcess(argv, 0, b"{}", b"")

    def api(self) -> list[tuple[str, str, Any]]:
        aus: list[tuple[str, str, Any]] = []
        for argv, kw in self.aufrufe:
            if argv[1] == "api":
                pfad = next(a for a in argv[2:] if a.startswith("/"))
                daten = json.loads(kw["input"].decode("utf-8")) if kw.get("input") is not None else None
                aus.append((argv[argv.index("--method") + 1], pfad, daten))
        return aus


def test_marken_feste_werte() -> None:
    assert AUTO_MARKE == "<!-- selbstbild:auto -->"
    assert DEBUGGER_MARKE == "<!-- selbstbild:debugger -->"


def test_kommentar_traegt_auto_marke() -> None:
    f = Aufzeichner()
    GitHub("Flissel", laufzeit=f).kommentieren("o/r", 5, "Hallo")
    (methode, pfad, daten), = f.api()
    assert (methode, pfad) == ("POST", "/repos/o/r/issues/5/comments")
    assert daten["body"] == "Hallo\n\n<!-- selbstbild:auto -->"


def test_schliessen_traegt_marke_und_grund() -> None:
    f = Aufzeichner()
    GitHub("Flissel", laufzeit=f).schliessen("o/r", 5, "zu", grund="not_planned")
    kommentar, patch = f.api()
    assert kommentar[2]["body"] == "zu\n\n<!-- selbstbild:auto -->"
    assert patch[0] == "PATCH" and patch[2] == {"state": "closed", "state_reason": "not_planned"}


def test_schliessen_standardgrund_completed() -> None:
    f = Aufzeichner()
    GitHub("Flissel", laufzeit=f).schliessen("o/r", 5, "zu")
    assert f.api()[1][2] == {"state": "closed", "state_reason": "completed"}


def test_schliessen_grund_unbekannt() -> None:
    f = Aufzeichner()
    with pytest.raises(ValueError):
        GitHub("Flissel", laufzeit=f).schliessen("o/r", 5, "zu", grund="egal")
    assert f.api() == []  # nichts gesendet, auch kein Kommentar


def test_marke_nicht_doppelt() -> None:
    f = Aufzeichner()
    text = f"Hallo\n\n{DEBUGGER_MARKE}"
    GitHub("Flissel", laufzeit=f, marke=DEBUGGER_MARKE).kommentieren("o/r", 5, text)
    assert f.api()[0][2]["body"] == text
    f2 = Aufzeichner()
    GitHub("Flissel", laufzeit=f2).kommentieren("o/r", 5, f"x {AUTO_MARKE} y")
    assert f2.api()[0][2]["body"] == f"x {AUTO_MARKE} y"


def test_ohne_marke() -> None:
    f = Aufzeichner()
    GitHub("Flissel", laufzeit=f, marke=None).kommentieren("o/r", 5, "Hallo")
    assert f.api()[0][2]["body"] == "Hallo"


def test_token_aus_datei(tmp_path: Path) -> None:
    datei = tmp_path / "token"
    datei.write_text("abc\n", encoding="utf-8")
    f = Aufzeichner()
    GitHub("Flissel", laufzeit=f, token_datei=datei).kommentieren("o/r", 5, "Hallo")
    assert not any(a[1:3] == ["auth", "token"] for a, _ in f.aufrufe)
    (argv, kw), = f.aufrufe
    assert argv[1] == "api" and kw["env"]["GH_TOKEN"] == "abc"


@pytest.mark.parametrize("inhalt", [None, "", "  \n", "a b\n"])
def test_token_datei_leer_oder_fehlt(tmp_path: Path, inhalt: str | None) -> None:
    datei = tmp_path / "token"
    if inhalt is not None:
        datei.write_text(inhalt, encoding="utf-8")
    f = Aufzeichner()
    with pytest.raises(GitHubFehler) as e:
        GitHub("Flissel", laufzeit=f, token_datei=datei).kommentieren("o/r", 5, "Hallo")
    assert e.value.art == "kein_token"
    assert f.aufrufe == []
