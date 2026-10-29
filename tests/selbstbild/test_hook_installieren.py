"""Hook-Installer (Stufe 3, T9): idempotent, fremdes bleibt, nie eine fremde Struktur reparieren."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from selbstbild.__main__ import main
from selbstbild.hooks import installieren
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


@pytest.fixture(autouse=True)
def import_geprueft(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, tuple[str, ...]]]:
    """Die Import-Vorpruefung haengt sonst am echten venv (PY gibt es nicht); hier nur protokolliert."""
    aufrufe: list[tuple[Path, tuple[str, ...]]] = []
    monkeypatch.setattr(installieren, "_import_pruefen", lambda py, module: aufrufe.append((py, tuple(module))))
    return aufrufe


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


def test_cli_ausgabe_nicht_cp1252(tmp_path: Path, capsysbinary: pytest.CaptureFixture[bytes]) -> None:
    s = tmp_path / "settings.local.json"
    _schreibe(s, {"permissions": {"allow": ["Bash(echo \u2192)"]}})
    assert main(["hooks", "einspielen", str(s)]) == 0
    assert EIN.encode() in capsysbinary.readouterr().out
    assert EIN in s.read_text(encoding="utf-8")
    assert main(["hooks", "entfernen", str(s)]) == 0
    assert "\u2192".encode("utf-8") in capsysbinary.readouterr().out


def test_cli_ausgabe_ohne_buffer_und_cp1252(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    class Ohne(io.TextIOBase):
        def __init__(self) -> None:
            self.teile: list[str] = []

        encoding = "cp1252"

        def write(self, t: str) -> int:
            t.encode("cp1252")  # wirft wie echtes cp1252-stdout
            self.teile.append(t)
            return len(t)

    s = tmp_path / "settings.local.json"
    _schreibe(s, {"permissions": {"allow": ["Bash(echo \u2192)"]}})
    aus = Ohne()
    monkeypatch.setattr(sys, "stdout", aus)
    assert main(["hooks", "einspielen", str(s)]) == 0
    assert EIN in "".join(aus.teile) and EIN in s.read_text(encoding="utf-8")


def test_bom_bleibt(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    s.write_bytes(b"\xef\xbb\xbf" + json.dumps(FREMD, indent=2).encode("utf-8") + b"\n")
    einspielen(s, PY)
    roh = s.read_bytes()
    assert roh.startswith(b"\xef\xbb\xbf") and not roh.startswith(b"\xef\xbb\xbf\xef\xbb\xbf")
    assert EIN in json.loads(roh.decode("utf-8-sig"))["hooks"]["UserPromptSubmit"][-1]["hooks"][0]["command"]
    sich = (tmp_path / "settings.local.json.selbstbild-sicherung").read_bytes()
    assert sich.startswith(b"\xef\xbb\xbf")
    entfernen(s)
    assert s.read_bytes().startswith(b"\xef\xbb\xbf")


def test_crlf_diff_nur_hookzeilen_und_rundlauf(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    original = (json.dumps(FREMD, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8")
    s.write_bytes(original)
    diff = einspielen(s, PY)
    zeilen = diff.splitlines()
    assert not [z for z in zeilen if z.startswith("-") and not z.startswith("---")]
    assert any(EIN in z for z in zeilen if z.startswith("+"))
    roh = s.read_bytes()
    assert b"\r\n" in roh and b"\n" not in roh.replace(b"\r\n", b"")
    entfernen(s)
    assert s.read_bytes() == original


def test_relativer_python_wird_absolut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    s = tmp_path / "settings.local.json"
    einspielen(s, Path("venv/python.exe"))
    cmd = json.loads(s.read_text(encoding="utf-8"))["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert cmd == f'"{(tmp_path / "venv" / "python.exe").resolve().as_posix()}" {EIN}'


@pytest.mark.parametrize("vorher", ["fremd", "neu", "weg"])
def test_wettlauf_wirft_und_laesst_fremdes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vorher: str) -> None:
    from selbstbild.hooks import installieren

    s = tmp_path / "settings.local.json"
    if vorher != "neu":
        _schreibe(s, FREMD)
    echt = installieren._text

    def dazwischen(daten: dict) -> str:
        if vorher == "weg":
            s.unlink()
        else:
            s.write_text('{"fremd": true}\n', encoding="utf-8")
        return echt(daten)

    monkeypatch.setattr(installieren, "_text", dazwischen)
    with pytest.raises(RuntimeError, match="waehrend des Einspielens"):
        einspielen(s, PY)
    if vorher == "weg":
        assert not s.exists()
    else:
        assert s.read_text(encoding="utf-8") == '{"fremd": true}\n'
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.selbstbild-sicherung"))


def test_cli_wettlauf_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:
    from selbstbild.hooks import installieren

    def boese(*_a: object, **_k: object) -> str:
        raise RuntimeError("settings waehrend des Einspielens geaendert")

    monkeypatch.setattr(installieren, "einspielen", boese)
    assert main(["hooks", "einspielen", str(tmp_path / "x.json")]) == 2
    assert "waehrend" in capsys.readouterr().err


def test_nur_waechter(tmp_path: Path, import_geprueft) -> None:
    s = tmp_path / "settings.local.json"
    einspielen(s, PY, nur={"waechter"})
    d = json.loads(s.read_text(encoding="utf-8"))
    assert list(d["hooks"]) == ["PreToolUse"] and len(_eigene(d, "PreToolUse")) == 1
    assert WAE in _eigene(d, "PreToolUse")[0]["command"]
    assert import_geprueft == [(PY.resolve(), ("selbstbild.hooks.waechter",))]
    assert einspielen(s, PY, nur={"waechter"}) == ""


def test_nur_waechter_laesst_eigene_einspeisung_stehen(tmp_path: Path) -> None:
    s = tmp_path / "settings.local.json"
    _schreibe(s, FREMD)
    alt = Path("D:/alt/python.exe")
    einspielen(s, alt)
    einspielen(s, PY, nur={"waechter"})
    d = json.loads(s.read_text(encoding="utf-8"))
    assert _eigene(d, "UserPromptSubmit") == [{"type": "command", "command": f'"{alt.resolve().as_posix()}" {EIN}',
                                               "timeout": 5}]
    assert _eigene(d, "PreToolUse")[0]["command"] == f'"{PY.resolve().as_posix()}" {WAE}'


def test_nur_unbekannt_wirft(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        einspielen(tmp_path / "s.json", PY, nur={"gibtsnicht"})


def test_import_fehlschlag_schreibt_nichts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def scheitert(py: Path, module: object) -> None:
        raise RuntimeError(f"Hooks nicht importierbar mit {py}")

    monkeypatch.setattr(installieren, "_import_pruefen", scheitert)
    s = tmp_path / "settings.local.json"
    _schreibe(s, FREMD)
    vorher = s.read_bytes()
    for trocken in (False, True):
        with pytest.raises(RuntimeError, match="nicht importierbar"):
            einspielen(s, PY, trocken=trocken, nur={"waechter"})
    assert s.read_bytes() == vorher and not (tmp_path / "settings.local.json.selbstbild-sicherung").exists()
    neu = tmp_path / "neu.json"
    with pytest.raises(RuntimeError):
        einspielen(neu, PY)
    assert not neu.exists()
    assert main(["hooks", "einspielen", str(s), "--nur", "waechter"]) == 2
    assert s.read_bytes() == vorher


def test_cli_nur_wiederholbar(tmp_path: Path, import_geprueft) -> None:
    s = tmp_path / "settings.local.json"
    assert main(["hooks", "einspielen", str(s), "--nur", "waechter"]) == 0
    d = json.loads(s.read_text(encoding="utf-8"))
    assert list(d["hooks"]) == ["PreToolUse"]
    assert main(["hooks", "einspielen", str(s), "--nur", "waechter", "--nur", "einspeisung"]) == 0
    d = json.loads(s.read_text(encoding="utf-8"))
    assert len(_eigene(d, "UserPromptSubmit")) == 1 and len(_eigene(d, "PreToolUse")) == 1
    assert set(import_geprueft[-1][1]) == {"selbstbild.hooks.waechter", "selbstbild.hooks.einspeisung"}


def test_import_pruefen_echter_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Echter Kindprozess mit diesem Interpreter; PYTHONPATH wird nicht vererbt."""
    import sys

    monkeypatch.undo()  # echte Vorpruefung statt Protokoll
    (tmp_path / "nur_im_pythonpath_xyz.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    py = Path(sys.executable)
    assert installieren._importierbar(py, ["json", "sqlite3"]) is None
    assert installieren._importierbar(py, ["nur_im_pythonpath_xyz"]) is not None
    with pytest.raises(RuntimeError, match="hooks pth"):
        installieren._import_pruefen(py, ["nur_im_pythonpath_xyz"])
