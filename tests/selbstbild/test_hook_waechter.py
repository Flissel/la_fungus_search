"""Waechter-Hook (Stufe 3, T8): deny nur bei `verboten`, sonst still, nie ein Fehler nach aussen."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
REGELN = ("struktur:\n  - art: space\n    ort: vibemind-os/spaces/*/\n    pflicht: [capabilities.yaml]\n"
          "    verboten: ['spaces/*/']\n    hinweis: Spaces gehoeren nach vibemind-os/spaces\n")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    (r / "selbstbild.yaml").write_text(REGELN, encoding="utf-8")
    return r


def _lauf(eingabe: str | bytes, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    roh = eingabe if isinstance(eingabe, bytes) else eingabe.encode("utf-8")
    return subprocess.run([sys.executable, "-m", "selbstbild.hooks.waechter"], input=roh, capture_output=True,
                          env={**os.environ, "PYTHONPATH": str(SRC)}, cwd=cwd, timeout=60)


def _json(tool: str, pfad: Path | str, schluessel: str = "file_path", **extra: object) -> str:
    return json.dumps({"tool_name": tool, "tool_input": {schluessel: str(pfad)}, **extra})


def test_waechter_json_deny(root):
    p = _lauf(_json("Write", root / "spaces" / "neu" / "x.py"))
    assert p.returncode == 0
    aus = json.loads(p.stdout)["hookSpecificOutput"]
    assert aus["hookEventName"] == "PreToolUse" and aus["permissionDecision"] == "deny"
    assert aus["permissionDecisionReason"].startswith("space: ")
    assert "Spaces gehoeren nach vibemind-os/spaces" in aus["permissionDecisionReason"]
    assert p.stdout.isascii()


def test_waechter_andere_werkzeuge_still(root):
    ziel = root / "spaces" / "neu" / "x.py"
    assert _lauf(_json("Read", ziel)).stdout == b""
    assert _lauf(_json("Bash", ziel)).stdout == b""
    assert _lauf(_json("Edit", ziel)).stdout != b""
    assert _lauf(_json("MultiEdit", ziel)).stdout != b""
    assert _lauf(_json("NotebookEdit", ziel, "notebook_path")).stdout != b""


def test_waechter_erlaubt_still(root):
    p = _lauf(_json("Write", root / "vibemind-os" / "spaces" / "neu" / "x.py"))
    assert p.returncode == 0 and p.stdout == b""


def test_waechter_relativ_zu_cwd(root):
    p = _lauf(_json("Write", "spaces/neu/x.py", cwd=str(root)))
    assert json.loads(p.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("eingabe", [b"{kaputt", b"", b"[]", b"\xff\xfe", b'{"tool_name": "Write"}',
                                     b'{"tool_name": "Write", "tool_input": {"file_path": 5}}',
                                     b'{"tool_name": "Write", "tool_input": null}'])
def test_waechter_fehler_blockiert_nicht(eingabe):
    p = _lauf(eingabe)
    assert p.returncode == 0 and p.stdout == b""


def test_waechter_interner_fehler_blockiert_nicht(root, monkeypatch):
    from selbstbild.hooks import waechter

    def boese(*_a: object, **_k: object) -> None:
        raise RuntimeError("kaputt")

    monkeypatch.setattr("selbstbild.struktur.pruefe_schreibziel", boese)
    import io
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(_json("Write", root / "spaces" / "a" / "b").encode())))
    assert waechter.main() == 0
