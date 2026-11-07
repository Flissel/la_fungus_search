"""Tests fuer `selbstbild.debugger.konfig`."""
from pathlib import Path

import pytest

from selbstbild.debugger.konfig import DebuggerKonfig, konfig_laden
from selbstbild.regeln import regeln_laden

ABSCHNITT = """\
# Debugger-Claw (Plan 1, 07.11.2026): Wächter auf der VM, Helfer schreibt als Vibemind-LAB.
debugger:
  issue_repo: Flissel/Vibemind_V1
  repos:
    Flissel/Vibemind_V1: "."
    Flissel/vibemind-os: "vibemind-os"
  ausloeser: Flissel
  helfer_konto: Vibemind-LAB
"""


def _yaml(wurzel: Path, text: str) -> None:
    (wurzel / "selbstbild.yaml").write_text(text, encoding="utf-8")


def test_konfig_standardwerte(tmp_path: Path) -> None:
    _yaml(tmp_path, ABSCHNITT)
    k = konfig_laden(tmp_path)
    assert isinstance(k, DebuggerKonfig)
    assert k.issue_repo == "Flissel/Vibemind_V1"
    assert k.repos == {"Flissel/Vibemind_V1": ".", "Flissel/vibemind-os": "vibemind-os"}
    assert (k.ausloeser, k.helfer_konto, k.basis) == ("Flissel", "Vibemind-LAB", "master")
    assert (k.intervall_s, k.frist_s, k.min_frei_gb, k.tagesgrenze) == (90, 1200, 2.0, 20)


def test_konfig_unbekannter_schluessel(tmp_path: Path) -> None:
    _yaml(tmp_path, "debugger:\n  issue_repo: a/b\n  repos: {a/b: '.'}\n  quatsch: 1\n")
    with pytest.raises(ValueError):
        konfig_laden(tmp_path)


def test_konfig_issue_repo_nicht_in_repos(tmp_path: Path) -> None:
    _yaml(tmp_path, "debugger:\n  issue_repo: a/b\n  repos: {c/d: '.'}\n")
    with pytest.raises(ValueError):
        konfig_laden(tmp_path)


def test_konfig_ohne_issue_repo_oder_abschnitt(tmp_path: Path) -> None:
    _yaml(tmp_path, "debugger:\n  repos: {a/b: '.'}\n")
    with pytest.raises(ValueError):
        konfig_laden(tmp_path)
    _yaml(tmp_path, "name: x\n")
    with pytest.raises(ValueError):
        konfig_laden(tmp_path)


def test_konfig_ueberschreibt_werte(tmp_path: Path) -> None:
    _yaml(tmp_path, "debugger:\n  issue_repo: a/b\n  repos: {a/b: '.'}\n  frist_s: 60\n  tagesgrenze: 3\n")
    k = konfig_laden(tmp_path)
    assert (k.frist_s, k.tagesgrenze) == (60, 3)


def test_ausgelieferter_abschnitt_wird_von_regeln_akzeptiert() -> None:
    """Das Regelmodell darf den Abschnitt `debugger:` nicht ablehnen (Tageslauf)."""
    regeln_laden("name: x\n" + ABSCHNITT)
