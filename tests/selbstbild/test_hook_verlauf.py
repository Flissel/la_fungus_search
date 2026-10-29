"""Gespraechsverlauf fuer den Einspeisungs-Hook (Stufe 3, T12): letzte Dateien aus dem Transkript, Worktree-Pfade."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from selbstbild.hooks.verlauf import hauptpfad, letzte_dateien

GEHEIM = "GEHEIMER-TEXT-4711"


def _zeile(art: str, **kw: object) -> str:
    return json.dumps({"type": art, **kw})


def _prompt(text: str) -> str:
    return _zeile("user", message={"role": "user", "content": text})


def _werkzeug(name: str, **eingabe: object) -> str:
    return _zeile("assistant", message={"role": "assistant", "content": [
        {"type": "text", "text": GEHEIM}, {"type": "tool_use", "id": "t", "name": name, "input": eingabe}]})


def _schreiben(datei: Path, zeilen: list[str]) -> Path:
    datei.write_bytes(("\n".join(zeilen) + "\n").encode("utf-8"))  # LF wie Claude Code, auch unter Windows
    return datei


def test_neueste_zuerst_dedupliziert_nur_pfade(tmp_path):
    w = tmp_path / "repo"
    a, b, c, nb = (str(w / "app" / "a.py"), str(w / "app" / "b.py"), str(w / "docs" / "c.md"), str(w / "n.ipynb"))
    tr = _schreiben(tmp_path / "s.jsonl", [
        _prompt(GEHEIM),
        _werkzeug("Read", file_path=a),
        _werkzeug("Edit", file_path=b, old_string=GEHEIM, new_string=GEHEIM),
        _werkzeug("NotebookEdit", notebook_path=nb),
        _werkzeug("Grep", pattern=GEHEIM, path=str(w / "app")),
        _werkzeug("Glob", pattern="**/*.py", path=str(w / "src" / "*")),  # Glob-Muster: kein Ort
        _werkzeug("Bash", command=f"cat {c}"),  # Befehle werden nicht ausgewertet
        "kaputte zeile {",
        _werkzeug("Write", file_path=c, content=GEHEIM),
        _werkzeug("Read", file_path=a),
        _zeile("user", message={"role": "user", "content": [{"type": "tool_result", "content": GEHEIM}]}),
    ])
    erg = letzte_dateien(tr)
    assert erg == [a, c, str(w / "app"), nb, b]
    assert all(GEHEIM not in p for p in erg)


def test_relative_und_seitenkette_ignoriert(tmp_path):
    a = str(tmp_path / "a.py")
    sub = json.loads(_werkzeug("Edit", file_path=str(tmp_path / "sub.py")))
    sub["isSidechain"] = True
    tr = _schreiben(tmp_path / "s.jsonl", [
        _werkzeug("Edit", file_path=a), _werkzeug("Edit", file_path="relativ/x.py"), json.dumps(sub),
        _werkzeug("Read", file_path=123), _werkzeug("Read")])
    assert letzte_dateien(tr) == [a]


def test_ende_begrenzt_auf_verlauf_davor(tmp_path):
    a, b = str(tmp_path / "a.py"), str(tmp_path / "b.py")
    zeilen = [_werkzeug("Edit", file_path=a), _prompt(GEHEIM), _werkzeug("Edit", file_path=b)]
    tr = _schreiben(tmp_path / "s.jsonl", zeilen)
    ende = len((zeilen[0] + "\n").encode("utf-8"))  # Byte-Offset der Prompt-Zeile
    assert letzte_dateien(tr, ende=ende) == [a]
    assert letzte_dateien(tr) == [b, a]
    assert letzte_dateien(tr, ende=0) == []


def test_max_bytes_verwirft_angeschnittene_zeile(tmp_path):
    a, b = str(tmp_path / "alt.py"), str(tmp_path / "neu.py")
    zeilen = [_werkzeug("Edit", file_path=a), _werkzeug("Edit", file_path=b)]
    tr = _schreiben(tmp_path / "s.jsonl", zeilen)
    letzte = len((zeilen[1] + "\n").encode("utf-8"))
    assert letzte_dateien(tr, max_bytes=letzte) == [b]  # genau an der Zeilengrenze: Zeile ist ganz
    assert letzte_dateien(tr, max_bytes=letzte + 10) == [b]  # alte Zeile angeschnitten -> verworfen
    assert letzte_dateien(tr, max_bytes=letzte - 1) == []  # auch die neue ist angeschnitten


def test_max_dateien(tmp_path):
    zeilen = [_werkzeug("Read", file_path=str(tmp_path / f"f{i}.py")) for i in range(50)]
    tr = _schreiben(tmp_path / "s.jsonl", zeilen)
    erg = letzte_dateien(tr, max_dateien=30)
    assert len(erg) == 30 and erg[0] == str(tmp_path / "f49.py")


def test_robust(tmp_path):
    assert letzte_dateien(tmp_path / "fehlt.jsonl") == []
    assert letzte_dateien(tmp_path) == []  # Ordner statt Datei
    (tmp_path / "muell.jsonl").write_bytes(b"\xff\xfe\x00 kein json\n[1,2]\n\"tool_use\"\n{\"type\": \"assistant\"}\n")
    assert letzte_dateien(tmp_path / "muell.jsonl") == []
    assert letzte_dateien(tmp_path / "muell.jsonl", ende=-5) == []


def test_grosse_datei_nur_ende_gelesen(tmp_path):
    a = str(tmp_path / "a.py")
    tr = tmp_path / "gross.jsonl"
    fuell = _prompt("x" * 1000) + "\n"
    with tr.open("w", encoding="utf-8", newline="\n") as f:
        f.write(_werkzeug("Edit", file_path=str(tmp_path / "ganz_alt.py")) + "\n")
        for _ in range(20_000):  # ~20 MB
            f.write(fuell)
        f.write(_werkzeug("Edit", file_path=a) + "\n")
    t = time.monotonic()
    assert letzte_dateien(tr) == [a]
    assert time.monotonic() - t < 0.5


def test_hauptpfad_worktrees():
    assert hauptpfad("C:/r/vibemind-os/.worktrees/wt1/spaces/x.py") == ("C:/r/vibemind-os/spaces/x.py", "wt1")
    assert hauptpfad(r"C:\r\.claude\worktrees\agent-7\app\a.py") == ("C:/r/app/a.py", "agent-7")
    assert hauptpfad("/home/u/r/.worktrees/wt") == ("/home/u/r", "wt")
    assert hauptpfad(r"C:\r\app\a.py") == ("C:/r/app/a.py", None)
    assert hauptpfad("C:/r/.Claude/Worktrees/W/a.py") == ("C:/r/a.py", "W")
    assert hauptpfad("C:/r/worktrees/x/a.py") == ("C:/r/worktrees/x/a.py", None)  # ohne Punkt: normaler Ordner


def test_nur_stdlib():
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[2] / "src"
    code = ("import sys; import selbstbild.hooks.verlauf; "
            "bad=[m for m in ('numpy','igraph','yaml','selbstbild.suche','selbstbild.speicher') if m in sys.modules];"
            "sys.exit(1 if bad else 0)")
    erg = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=str(src)),
                         capture_output=True, timeout=60)
    assert erg.returncode == 0
