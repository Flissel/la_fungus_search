"""Hook-Installer (Stufe 3, T9): idempotent, fremdes bleibt, nie eine fremde Struktur reparieren."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from selbstbild.__main__ import main
from selbstbild.hooks.installieren import einspielen, entfernen, pth_anlegen

PY = Path("C:/venv/Scripts/python.exe")
EIN = "-m selbstbild.hooks.einspeisung"
WAE = "-m selbstbild.hooks.waechter"
FREMD = {
    "permissions": {"allow": ["Bash(ls)"]},
    "env": {"A": "1"},
    "hooks": {
        "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo fremd", "timeout": 3}]}],
        "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "fremd.sh"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "stop.sh"}]}],
    },
}


def _schreibe(pfad: Path, daten: object) -> None:
    pfad.write_text(json.dumps(daten, indent=2) + "\n", encoding="utf-8")


def _eigene(daten: dict, ereignis: str) -> list[dict]:
    return [h for g in daten["hooks"][ereignis] for h in g["hooks"]
            if EIN in h["command"] or WAE in h["command"]]


def test_einspielen_idempotent_und_fremdes_bleibt(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    _schreibe(s, FREMD)
    diff1 = einspielen(s, PY)
    assert diff1.startswith("---") and EIN in diff1
    nach1 = s.read_text(encoding="utf-8")
    assert einspielen(s, PY) == ""
    assert s.read_text(encoding="utf-8") == nach1
    d = json.loads(nach1)
    assert len(_eigene(d, "UserPromptSubmit")) == 1 and len(_eigene(d, "PreToolUse")) == 1
    assert _eigene(d, "UserPromptSubmit")[0] == {
        "type": "command", "command": f'"{PY.as_posix()}" {EIN}', "timeout": 5}
    gruppe = d["hooks"]["PreToolUse"][-1]
    assert gruppe["matcher"] == "Write|Edit|MultiEdit|NotebookEdit"
    assert gruppe["hooks"][0]["command"] == f'"{PY.as_posix()}" {WAE}'
    # Fremdes JSON-gleich
    assert d["permissions"] == FREMD["permissions"] and d["env"] == FREMD["env"]
    assert d["hooks"]["Stop"] == FREMD["hooks"]["Stop"]
    assert d["hooks"]["UserPromptSubmit"][0] == FREMD["hooks"]["UserPromptSubmit"][0]
    assert d["hooks"]["PreToolUse"][0] == FREMD["hooks"]["PreToolUse"][0]
    assert (tmp_path / "settings.local.json.selbstbild-sicherung").read_text(encoding="utf-8") == \
        json.dumps(FREMD, indent=2) + "\n"


def test_neue_datei_und_trocken_schreibt_nicht(tmp_path: Path) -> None:
    s = tmp_path / "neu" / "settings.local.json"
    diff = einspielen(s, PY, trocken=True)
    assert EIN in diff and not s.exists() and not s.parent.exists()
    _schreibe(tmp_path / "a.json", FREMD)
    vorher = (tmp_path / "a.json").read_bytes()
    assert "+" in einspielen(tmp_path / "a.json", PY, trocken=True)
    assert (tmp_path / "a.json").read_bytes() == vorher
    assert not (tmp_path / "a.json.selbstbild-sicherung").exists()
    einspielen(s, PY)
    assert EIN in s.read_text(encoding="utf-8")
    assert not list(s.parent.glob("*.selbstbild-sicherung"))  # Datei fehlte: keine Sicherung


def test_entfernen_nur_eigene(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    _schreibe(s, FREMD)
    einspielen(s, PY)
    diff = entfernen(s)
    assert "-" in diff
    assert json.loads(s.read_text(encoding="utf-8")) == FREMD
    assert entfernen(s) == ""


def test_entfernen_raeumt_leere_struktur(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    _schreibe(s, {"env": {"A": "1"}})
    einspielen(s, PY)
    entfernen(s)
    assert json.loads(s.read_text(encoding="utf-8")) == {"env": {"A": "1"}}


def test_entfernen_ohne_eigene_schreibt_nichts(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    _schreibe(s, FREMD)
    vorher = s.read_bytes()
    assert entfernen(s) == ""
    assert s.read_bytes() == vorher and not (tmp_path / "settings.local.json.selbstbild-sicherung").exists()
    assert entfernen(tmp_path / "fehlt.json") == ""
    assert not (tmp_path / "fehlt.json").exists()


@pytest.mark.parametrize("inhalt", ["{kaputt", "[1, 2]", '{"hooks": []}', '{"hooks": {"PreToolUse": {}}}',
                                    '{"hooks": {"PreToolUse": [1]}}',
                                    '{"hooks": {"PreToolUse": [{"hooks": "x"}]}}'])
def test_fremde_struktur_wirft_und_laesst_unberuehrt(tmp_path: Path, inhalt: str) -> None:
    s = tmp_path / "settings.local.json"
    s.write_text(inhalt, encoding="utf-8")
    with pytest.raises(ValueError):
        einspielen(s, PY)
    with pytest.raises(ValueError):
        entfernen(s)
    assert s.read_text(encoding="utf-8") == inhalt
    assert not (tmp_path / "settings.local.json.selbstbild-sicherung").exists()


def test_vorhandene_sicherung_bleibt(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    _schreibe(s, FREMD)
    sich = tmp_path / "settings.local.json.selbstbild-sicherung"
    sich.write_text("ALT", encoding="utf-8")
    einspielen(s, PY)
    assert sich.read_text(encoding="utf-8") == "ALT"
    entfernen(s)
    assert sich.read_text(encoding="utf-8") == "ALT"


def test_alter_python_pfad_wird_ersetzt(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    _schreibe(s, FREMD)
    einspielen(s, Path("D:/alt/python.exe"))
    diff = einspielen(s, PY)
    assert "D:/alt" in diff
    d = json.loads(s.read_text(encoding="utf-8"))
    assert [h["command"] for h in _eigene(d, "UserPromptSubmit")] == [f'"{PY.as_posix()}" {EIN}']
    assert [h["command"] for h in _eigene(d, "PreToolUse")] == [f'"{PY.as_posix()}" {WAE}']
    assert "D:/alt" not in s.read_text(encoding="utf-8")


def test_pth(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    sp = venv / "Lib" / "site-packages"
    sp.mkdir(parents=True)
    src = tmp_path / "src"
    src.mkdir()
    pth = pth_anlegen(venv, src)
    assert pth == sp / "selbstbild-src.pth"
    assert pth.read_text(encoding="utf-8") == str(src.resolve()) + "\n"
    ns = pth.stat().st_mtime_ns
    assert pth_anlegen(venv, src) == pth and pth.stat().st_mtime_ns == ns


def test_pth_linux_layout_und_fehlt(tmp_path: Path) -> None:
    venv = tmp_path / "v"
    with pytest.raises(FileNotFoundError):
        pth_anlegen(venv, tmp_path)
    sp = venv / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    assert pth_anlegen(venv, tmp_path) == sp / "selbstbild-src.pth"


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = tmp_path / "settings.local.json"
    assert main(["hooks", "einspielen", str(s), "--trocken"]) == 0
    assert EIN in capsys.readouterr().out and not s.exists()
    assert main(["hooks", "einspielen", str(s)]) == 0
    assert EIN in s.read_text(encoding="utf-8")
    assert main(["hooks", "entfernen", str(s)]) == 0
    s.write_text("{kaputt", encoding="utf-8")
    assert main(["hooks", "einspielen", str(s)]) == 2
    assert "settings" in capsys.readouterr().err
