"""scripts/secret_probe.py: meldet nur Variablennamen, nie Werte (R16). Nur mit Fake-.env."""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

SKRIPT = Path(__file__).resolve().parents[2] / "scripts" / "secret_probe.py"


def _db(pfad: Path, *texte: str) -> Path:
    c = sqlite3.connect(pfad)
    c.execute("CREATE TABLE t(a TEXT)")
    c.executemany("INSERT INTO t VALUES(?)", [(t,) for t in texte])
    c.commit()
    c.close()
    return pfad


def _lauf(env: Path, graph: Path, export: Path) -> tuple[int, str]:
    erg = subprocess.run(
        [sys.executable, str(SKRIPT), "--env", str(env), "--graph", str(graph), "--export", str(export)],
        capture_output=True, timeout=60,
    )
    return erg.returncode, erg.stdout.decode("utf-8") + erg.stderr.decode("utf-8")


def _probe(tmp_path: Path, env_text: str, graph_texte: tuple[str, ...], export: bytes) -> tuple[int, str]:
    env = tmp_path / "fake.env"
    env.write_text(env_text, encoding="utf-8")
    g = _db(tmp_path / "graph.sqlite", *graph_texte)
    e = tmp_path / "export.json"
    e.write_bytes(export)
    return _lauf(env, g, e)


def test_treffer_nur_name(tmp_path):
    code, aus = _probe(
        tmp_path,
        "# Kommentar\nexport GEHEIM_A='fake-wert-alpha'\nGEHEIM_B=\"fake-wert-beta\"\nKURZ=abc\n",
        graph_texte=("...fake-wert-alpha...abc",),
        export=b'{"x": "nichts"}',
    )
    assert code == 1
    assert "GEHEIM_A" in aus
    assert "GEHEIM_B" not in aus
    assert "KURZ" not in aus  # kuerzer als 6 Zeichen: ignoriert
    assert "fake-wert" not in aus and "alpha" not in aus


def test_treffer_im_export_json_escaped(tmp_path):
    code, aus = _probe(
        tmp_path,
        'X_PFAD=C:\\geheim\\pfad\n',
        graph_texte=("nichts",),
        export=b'{"v": "C:\\\\geheim\\\\pfad"}',
    )
    assert code == 1
    assert "X_PFAD" in aus and "geheim" not in aus


def test_treffer_im_wal(tmp_path):
    # Echter WAL: Die Verbindung bleibt offen, der Inhalt steht noch nicht in graph.sqlite.
    g = tmp_path / "graph.sqlite"
    c = sqlite3.connect(g)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA wal_autocheckpoint=0")
    c.execute("CREATE TABLE t(a TEXT)")
    c.execute("INSERT INTO t VALUES(?)", ("...fake-im-wal-123...",))
    c.commit()
    try:
        assert (tmp_path / "graph.sqlite-wal").exists()
        env = tmp_path / "fake.env"
        env.write_text("Y_TOKEN=fake-im-wal-123\n", encoding="utf-8")
        e = tmp_path / "export.json"
        e.write_bytes(b"{}")
        code, aus = _lauf(env, g, e)
    finally:
        c.close()
    assert code == 1, aus
    assert "Y_TOKEN" in aus and "fake-im-wal" not in aus


def test_inline_kommentar_abgeschnitten(tmp_path):
    code, aus = _probe(
        tmp_path,
        "A_WERT=fake-wert-delta # Kommentar dahinter\nC_WERT=\"fake-c # teil\"\n",
        graph_texte=("...fake-wert-delta...", "...fake-c # teil..."),
        export=b"{}",
    )
    assert code == 1
    assert "A_WERT" in aus  # gesucht wurde der Wert ohne Kommentar
    assert "C_WERT" in aus  # in Quotes gehoert '#' zum Wert (gekuerzt waere er < 6 Zeichen)
    assert "delta" not in aus and "Kommentar" not in aus and "teil" not in aus


def test_spalten_scan_findet_was_bytesuche_nicht_sieht(tmp_path):
    # UTF-16-Datenbank: Die rohen Dateibytes enthalten den UTF-8-Wert nicht, die
    # Spaltenwerte ueber sqlite3 schon (gleiches gilt fuer Werte ueber Overflow-Seiten).
    db = tmp_path / "echt.sqlite"
    c = sqlite3.connect(db)
    c.execute("PRAGMA encoding='UTF-16le'")
    c.execute("CREATE TABLE t(a TEXT, b BLOB, n INTEGER)")
    c.execute("INSERT INTO t VALUES(?,?,?)", ("x" * 9000 + "fake-wert-sechzehn" + "y" * 9000, b"leer", 1))
    c.commit()
    c.close()
    assert b"fake-wert-sechzehn" not in db.read_bytes()
    env = tmp_path / "fake.env"
    env.write_text("U16=fake-wert-sechzehn\n", encoding="utf-8")
    export = tmp_path / "export.json"
    export.write_bytes(b"{}")
    code, aus = _lauf(env, db, export)
    assert code == 1, aus
    assert "U16" in aus and "sechzehn" not in aus


def test_sauber(tmp_path):
    code, aus = _probe(tmp_path, "A=fake-wert-gamma\n", graph_texte=("leer",), export=b"{}")
    assert code == 0
    assert "fake-wert-gamma" not in aus


def test_blockgrenzen_jede_fundstelle_genau_einmal(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("secret_probe_modul", SKRIPT)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    d = tmp_path / "d.bin"
    d.write_bytes(b"xxfake-abcdefyyfake-abcdef" + b"z" * 7 + b"fake-abcdef" + b"aaaa")
    for block in (1, 2, 3, 5, 7, 11, 13, 1 << 20):
        assert modul._datei_treffer(d, [b"fake-abcdef"], block=block) == 3, block
        assert modul._datei_treffer(d, [b"fake-abcdef", b"aa"], block=block) == 3 + 3, block


# --- fail-closed: fehlende oder unlesbare Eingaben sind nie "0 Treffer" -------------

def _gueltig(tmp_path: Path) -> tuple[Path, Path, Path]:
    env = tmp_path / "fake.env"
    env.write_text("A=fake-wert-omega\n", encoding="utf-8")
    g = _db(tmp_path / "graph.sqlite", "leer")
    e = tmp_path / "export.json"
    e.write_bytes(b"{}")
    return env, g, e


def test_fehlender_graph_exit_2(tmp_path):
    env, _, e = _gueltig(tmp_path)
    code, aus = _lauf(env, tmp_path / "fehlt.sqlite", e)
    assert code == 2
    assert "--graph" in aus and "Treffer: 0" not in aus


def test_fehlender_export_exit_2(tmp_path):
    env, g, _ = _gueltig(tmp_path)
    code, aus = _lauf(env, g, tmp_path / "fehlt.json")
    assert code == 2
    assert "--export" in aus and "Treffer: 0" not in aus


def test_fehlende_env_exit_2(tmp_path):
    _, g, e = _gueltig(tmp_path)
    code, aus = _lauf(tmp_path / "fehlt.env", g, e)
    assert code == 2
    assert "--env" in aus


def test_unlesbare_db_exit_2(tmp_path):
    env, _, e = _gueltig(tmp_path)
    kaputt = tmp_path / "kaputt.sqlite"
    kaputt.write_bytes(b"kein sqlite, aber fake-wert-omega")
    code, aus = _lauf(env, kaputt, e)
    assert code == 2
    assert "Treffer: 0" not in aus and "omega" not in aus


def test_beschaedigte_db_exit_2(tmp_path):
    env, _, e = _gueltig(tmp_path)
    g = _db(tmp_path / "g.sqlite", "x" * 50000)
    roh = bytearray(g.read_bytes())
    roh[100:4096] = b"\xff" * (4096 - 100)  # Schema-Seite zerstoert, Kopf bleibt gueltig
    g.write_bytes(bytes(roh))
    code, aus = _lauf(env, g, e)
    assert code == 2, aus
    assert "Treffer: 0" not in aus
