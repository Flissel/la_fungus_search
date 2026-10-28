"""scripts/secret_probe.py: meldet nur Variablennamen, nie Werte (R16). Nur mit Fake-.env."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SKRIPT = Path(__file__).resolve().parents[2] / "scripts" / "secret_probe.py"


def _probe(tmp_path: Path, env_text: str, graph: bytes, export: bytes, wal: bytes | None = None):
    env = tmp_path / "fake.env"
    env.write_text(env_text, encoding="utf-8")
    g = tmp_path / "graph.sqlite"
    g.write_bytes(graph)
    if wal is not None:
        (tmp_path / "graph.sqlite-wal").write_bytes(wal)
    e = tmp_path / "export.json"
    e.write_bytes(export)
    erg = subprocess.run(
        [sys.executable, str(SKRIPT), "--env", str(env), "--graph", str(g), "--export", str(e)],
        capture_output=True, timeout=60,
    )
    return erg.returncode, erg.stdout.decode("utf-8") + erg.stderr.decode("utf-8")


def test_treffer_nur_name(tmp_path):
    code, aus = _probe(
        tmp_path,
        "# Kommentar\nexport GEHEIM_A='fake-wert-alpha'\nGEHEIM_B=\"fake-wert-beta\"\nKURZ=abc\n",
        graph=b"SQLite format 3\0...fake-wert-alpha...abc",
        export=b'{"x": "nichts"}',
    )
    assert code == 1
    assert "GEHEIM_A" in aus
    assert "GEHEIM_B" not in aus
    assert "KURZ" not in aus  # kuerzer als 6 Zeichen: ignoriert
    assert "fake-wert" not in aus and "alpha" not in aus


def test_treffer_im_export_json_escaped_und_wal(tmp_path):
    code, aus = _probe(
        tmp_path,
        'X_PFAD=C:\\geheim\\pfad\nY_TOKEN=fake-im-wal-123\n',
        graph=b"leer",
        export=b'{"v": "C:\\\\geheim\\\\pfad"}',
        wal=b"...fake-im-wal-123...",
    )
    assert code == 1
    assert "X_PFAD" in aus and "Y_TOKEN" in aus
    assert "geheim" not in aus and "fake-im-wal" not in aus


def test_inline_kommentar_abgeschnitten(tmp_path):
    code, aus = _probe(
        tmp_path,
        "A_WERT=fake-wert-delta # Kommentar dahinter\nB_WERT=\"fake#wert-eps # bleibt\"\n",
        graph=b"...fake-wert-delta...fake#wert-eps # bleibt...",
        export=b"{}",
    )
    assert code == 1
    assert "A_WERT: 1" in aus  # nur der Wert ohne Kommentar wurde gesucht
    assert "B_WERT: 1" in aus  # in Quotes gehoert '#' zum Wert
    assert "delta" not in aus and "Kommentar" not in aus


def _sqlite_utf16(pfad: Path, text: str) -> None:
    import sqlite3

    c = sqlite3.connect(pfad)
    c.execute("PRAGMA encoding='UTF-16le'")
    c.execute("CREATE TABLE t(a TEXT, b BLOB, n INTEGER)")
    c.execute("INSERT INTO t VALUES(?,?,?)", ("x" * 9000 + text + "y" * 9000, b"leer", 1))
    c.commit()
    c.close()


def test_spalten_scan_findet_was_bytesuche_nicht_sieht(tmp_path):
    # UTF-16-Datenbank: Die rohen Dateibytes enthalten den UTF-8-Wert nicht, die
    # Spaltenwerte ueber sqlite3 schon (gleiches gilt fuer Werte ueber Overflow-Seiten).
    db = tmp_path / "echt.sqlite"
    _sqlite_utf16(db, "fake-wert-sechzehn")
    assert b"fake-wert-sechzehn" not in db.read_bytes()
    env = tmp_path / "fake.env"
    env.write_text("U16=fake-wert-sechzehn\n", encoding="utf-8")
    export = tmp_path / "export.json"
    export.write_bytes(b"{}")
    erg = subprocess.run(
        [sys.executable, str(SKRIPT), "--env", str(env), "--graph", str(db), "--export", str(export)],
        capture_output=True, timeout=60,
    )
    aus = erg.stdout.decode("utf-8") + erg.stderr.decode("utf-8")
    assert erg.returncode == 1, aus
    assert "U16" in aus and "sechzehn" not in aus


def test_sauber(tmp_path):
    code, aus = _probe(tmp_path, "A=fake-wert-gamma\n", graph=b"leer", export=b"{}")
    assert code == 0
    assert "fake-wert-gamma" not in aus
