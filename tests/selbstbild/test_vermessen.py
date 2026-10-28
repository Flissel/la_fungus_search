"""Vermessungslauf: inkrementell == voll, Secrets, Submodule, `seit`, CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from selbstbild.speicher import Speicher
from selbstbild.typen import id_datei, id_dienst, id_env, id_repo, id_symbol
from selbstbild.vermessen import Bericht, vermessen

SRC = Path(__file__).resolve().parents[2] / "src"

PY = 'import os\n\n\ndef f():\n    return os.environ.get("APP_URL")\n\n\ndef g():\n    return f()\n'
TS = 'export function hole(): string {\n  return process.env.APP_URL ?? "";\n}\n'
SQL = "CREATE TABLE kunden (id int);\n"
COMPOSE = "services:\n  api:\n    image: x\n    environment:\n      - APP_URL=http://api:1\n"


def _speicher(tmp_path: Path, name: str) -> Speicher:
    return Speicher(tmp_path / name / "graph.sqlite")


def _export(s: Speicher, ziel: Path) -> bytes:
    s.export_json(ziel)
    return ziel.read_bytes()


def _vier_dateien(git_repo) -> None:
    git_repo.datei("app/a.py", PY)
    git_repo.datei("web/b.ts", TS)
    git_repo.datei("db/schema.sql", SQL)
    git_repo.datei("docker-compose.yml", COMPOSE)


def test_inkrementell_gleich_voll(git_repo, tmp_path):
    _vier_dateien(git_repo)
    git_repo.commit("eins")
    with _speicher(tmp_path, "inkr") as s:
        vermessen(git_repo.wurzel, speicher=s)
        # Gleiches Commit-Datum wie Commit 1 (Fixture-Vorgabe): `seit` haengt vom
        # gemessenen Verlauf ab (test_seit_attribut) und ist dann in beiden Laeufen gleich.
        git_repo.datei("app/a.py", PY + "\n\ndef h():\n    return 2\n")
        git_repo.commit("zwei")
        bericht = vermessen(git_repo.wurzel, speicher=s)
        a = _export(s, tmp_path / "a.json")
    with _speicher(tmp_path, "voll") as s:
        voll = vermessen(git_repo.wurzel, voll=True, speicher=s)
        b = _export(s, tmp_path / "b.json")
    assert a == b
    assert isinstance(bericht, Bericht)
    assert bericht.dateien == voll.dateien == 4
    assert bericht.geparst == 1
    assert bericht.aus_cache == bericht.dateien - 1
    assert voll.geparst == 4 and voll.aus_cache == 0
    assert bericht.knoten == voll.knoten and bericht.kanten == voll.kanten
    assert bericht.commit == git_repo.git("rev-parse", "HEAD")


def test_zweiter_lauf_ohne_aenderung(git_repo, tmp_path):
    _vier_dateien(git_repo)
    git_repo.commit("eins")
    with _speicher(tmp_path, "s") as s:
        erster = vermessen(git_repo.wurzel, speicher=s)
        a = _export(s, tmp_path / "a.json")
        zweiter = vermessen(git_repo.wurzel, speicher=s)
        b = _export(s, tmp_path / "b.json")
        assert s.meta("commit") == erster.commit
        assert s.meta("regeln_digest")
    assert erster.geparst == 4
    assert zweiter.geparst == 0 and zweiter.aus_cache == 4
    assert a == b


def test_keine_secret_werte(git_repo, tmp_path):
    git_repo.datei(".env", "TOKEN=wert-geheim-1\n")
    git_repo.datei(".env.example", "TOKEN=wert-geheim-2\n")
    git_repo.datei("docker-compose.yml",
                   "services:\n  api:\n    image: x\n    environment:\n      - KEY=wert-geheim-3\n")
    git_repo.datei("start.ps1", '$env:X="wert-geheim-4"\n')
    git_repo.datei("app.py", 'import os\nos.environ["Y"] = "wert-geheim-5"\n')
    git_repo.commit("secrets")
    graph = tmp_path / "stand" / "graph.sqlite"
    with Speicher(graph) as s:
        bericht = vermessen(git_repo.wurzel, speicher=s)
        s.export_json(tmp_path / "export.json")
    assert bericht.uebersprungen.get("env_datei") == 1
    roh = graph.read_bytes() + (tmp_path / "export.json").read_bytes()
    wal = graph.with_name("graph.sqlite-wal")
    if wal.exists():
        roh += wal.read_bytes()
    for i in range(1, 6):
        assert f"wert-geheim-{i}".encode() not in roh
    assert b"TOKEN" in roh and b"KEY" in roh  # Namen bleiben erhalten


def test_submodul_im_graph(git_repo, tmp_path):
    sub = type(git_repo)(tmp_path / "subquelle", dict(os.environ))
    sub.datei("s.py", 'import os\n\n\ndef lies():\n    return os.environ["SALES_URL"]\n')
    sub.commit("s")
    git_repo.datei("web/b.ts", "export const u = process.env.SALES_URL;\n")
    git_repo.datei("selbstbild.yaml", "dienste: config/dienste.yaml\n")
    git_repo.datei("config/dienste.yaml", "dienste:\n  sales:\n    port: 1\n    env:\n    - SALES_URL\n")
    git_repo.submodul("sub", sub)
    git_repo.commit("mit sub")
    eltern = git_repo.wurzel.name
    with _speicher(tmp_path, "s") as s:
        bericht = vermessen(git_repo.wurzel, speicher=s)
        ids = {k.id for k in s.knoten()}
        kanten = {(k.art, k.von, k.nach) for k in s.kanten()}
    assert bericht.submodule == (f"{eltern}/sub",)
    assert {id_repo(eltern), id_repo(f"{eltern}/sub")} <= ids
    env = id_env("SALES_URL")
    assert ("liest_env", id_symbol(f"{eltern}/sub", "s.py", "lies"), env) in kanten
    assert ("liest_env", id_datei(eltern, "web/b.ts"), env) in kanten
    assert ("nutzt_dienst", id_symbol(f"{eltern}/sub", "s.py", "lies"), id_dienst("sales")) in kanten
    assert ("nutzt_dienst", id_datei(eltern, "web/b.ts"), id_dienst("sales")) in kanten


def test_seit_attribut(git_repo, tmp_path):
    git_repo.datei("a.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    git_repo.commit("eins", datum="2026-01-01T00:00:00Z")
    t1 = int(git_repo.git("log", "-1", "--format=%ct"))
    with _speicher(tmp_path, "s") as s:
        vermessen(git_repo.wurzel, speicher=s)
        git_repo.datei("a.py", "def f():\n    return 1\n\n\ndef g():\n    return 3\n")
        git_repo.commit("zwei", datum="2026-02-01T00:00:00Z")
        t2 = int(git_repo.git("log", "-1", "--format=%ct"))
        vermessen(git_repo.wurzel, speicher=s)
        attr = {k.id: dict(k.attr) for k in s.knoten()}
        sortiert = all(list(k.attr) == sorted(k.attr) for k in s.knoten())
    repo = git_repo.wurzel.name
    f, g = attr[id_symbol(repo, "a.py", "f")], attr[id_symbol(repo, "a.py", "g")]
    assert t1 != t2
    assert f["seit"] == str(t1)
    assert g["seit"] == str(t2)
    # Symbole erben die Zeitattribute ihrer Datei
    datei = attr[id_datei(repo, "a.py")]
    assert datei["aenderungen"] == "2" and f["aenderungen"] == "2"
    assert datei["zuletzt"] == str(t2) == f["zuletzt"]
    assert "seit" not in datei
    assert sortiert


def test_cli_vermessen(git_repo, tmp_path):
    git_repo.datei("a.py", "def f():\n    return 1\n")
    git_repo.commit("eins")
    env = dict(os.environ, PYTHONPATH=str(SRC), SELBSTBILD_HOME=str(tmp_path / "home"))
    erg = subprocess.run(
        [sys.executable, "-m", "selbstbild", "vermessen", str(git_repo.wurzel)],
        capture_output=True, env=env, timeout=120,
    )
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    bericht = json.loads(erg.stdout)
    assert bericht["knoten"] > 0 and bericht["geparst"] == 1

    ziel = tmp_path / "export.json"
    erg = subprocess.run(
        [sys.executable, "-m", "selbstbild", "export", str(git_repo.wurzel), str(ziel)],
        capture_output=True, env=env, timeout=120,
    )
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    assert json.loads(ziel.read_bytes())["commit"] == bericht["commit"]

    erg = subprocess.run(
        [sys.executable, "-m", "selbstbild", "status", str(git_repo.wurzel)],
        capture_output=True, env=env, timeout=120,
    )
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    status = json.loads(erg.stdout)
    assert status["commit"] == bericht["commit"]
    assert status["knoten"] == bericht["knoten"]
    assert status["alter_s"] >= 0
