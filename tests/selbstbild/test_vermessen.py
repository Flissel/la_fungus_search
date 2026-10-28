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
    git_repo.commit("eins", datum="2026-01-01T00:00:00Z")
    with _speicher(tmp_path, "inkr") as s:
        vermessen(git_repo.wurzel, speicher=s)
        # Verschiedene Commit-Zeiten: `seit` unterscheidet sich zwischen den Staenden,
        # steht aber nicht im Export (R17), der bleibt eine Funktion von (Commit, Regeln).
        git_repo.datei("app/a.py", PY + "\n\ndef h():\n    return 2\n")
        git_repo.commit("zwei", datum="2026-03-01T00:00:00Z")
        bericht = vermessen(git_repo.wurzel, speicher=s)
        a = _export(s, tmp_path / "a.json")
        seit_inkr = {k.id: dict(k.attr).get("seit") for k in s.knoten() if k.art == "symbol"}
    with _speicher(tmp_path, "voll") as s:
        voll = vermessen(git_repo.wurzel, voll=True, speicher=s)
        b = _export(s, tmp_path / "b.json")
        seit_voll = {k.id: dict(k.attr).get("seit") for k in s.knoten() if k.art == "symbol"}
    assert a == b
    assert b'"seit"' not in a
    assert seit_inkr != seit_voll  # der Speicher kennt die Historie, der Export nicht
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


def test_seit_aelterer_commit_nachgemessen(git_repo, tmp_path):
    # R17: erst C2, dann --commit C1, dann wieder C2.
    git_repo.datei("a.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    c1 = git_repo.commit("eins", datum="2026-01-01T00:00:00Z")
    t1 = int(git_repo.git("log", "-1", "--format=%ct"))
    git_repo.datei("a.py", "def f():\n    return 1\n\n\ndef g():\n    return 3\n")
    c2 = git_repo.commit("zwei", datum="2026-02-01T00:00:00Z")
    t2 = int(git_repo.git("log", "-1", "--format=%ct"))
    repo = git_repo.wurzel.name
    f, g = id_symbol(repo, "a.py", "f"), id_symbol(repo, "a.py", "g")

    def seit(s: Speicher) -> dict[str, str]:
        return {k.id: dict(k.attr)["seit"] for k in s.knoten() if k.id in (f, g)}

    with _speicher(tmp_path, "s") as s:
        vermessen(git_repo.wurzel, c2, speicher=s)
        assert seit(s) == {f: str(t2), g: str(t2)}  # frischer Stand kennt nur C2
        vermessen(git_repo.wurzel, c1, speicher=s)
        bei_c1 = seit(s)
        assert int(bei_c1[f]) <= t1 and int(bei_c1[g]) <= t1
        vermessen(git_repo.wurzel, c2, speicher=s)
        bei_c2 = seit(s)
        assert s.meta("commit") == c2
    assert bei_c2[g] == str(t2)  # g hat sich in C2 geaendert
    assert bei_c2[f] == str(t1)  # f unveraendert seit C1, jetzt bekannt


def test_doppelte_qualnames_beleg_und_verlauf_aus_erster_definition(git_repo, tmp_path):
    # @property und @x.setter ergeben zweimal `K.x`: Beleg und Symbolverlauf-Digest
    # stammen aus derselben (der ersten) Definition.
    git_repo.datei("k.py", (
        "class K:\n"
        "    @property\n"
        "    def x(self):\n"
        "        return self._x\n"
        "\n"
        "    @x.setter\n"
        "    def x(self, wert):\n"
        "        self._x = pruefen(wert)\n"
        "\n"
        "\n"
        "def pruefen(w):\n"
        "    return w\n"
    ))
    git_repo.commit("eins")
    from selbstbild.parser.basis import parsen
    zwei = [s for s in parsen("python", "k.py", (git_repo.wurzel / "k.py").read_bytes()).symbole
            if s.qualname == "K.x"]
    assert len(zwei) == 2 and zwei[0].digest != zwei[1].digest
    sid = id_symbol(git_repo.wurzel.name, "k.py", "K.x")
    with _speicher(tmp_path, "s") as s:
        vermessen(git_repo.wurzel, speicher=s)
        knoten = {k.id: k for k in s.knoten()}
        (verlauf,) = s._verbindung().execute(
            "SELECT digest FROM symbolzeit WHERE symbol=?", (json.dumps(sid),)).fetchone()
        rufe = [k for k in s.kanten() if k.art == "ruft_auf" and k.von == sid]
    beleg = knoten[sid].beleg
    assert beleg is not None and beleg.zeile == zwei[0].zeile_von
    assert beleg.digest == verlauf == zwei[0].digest
    # die Aufrufkante aus dem Setter haengt am gemeinsamen Symbol; ihr Digest (R19) ist der
    # Koerper der ersten Definition, wie Beleg und Verlauf
    assert [k.beleg.digest for k in rufe if k.beleg] == [zwei[0].digest]


def test_parser_ausnahme_nicht_gecacht(git_repo, tmp_path, monkeypatch):
    from selbstbild import vermessen as vm

    git_repo.datei("a.py", "def f():\n    return 1\n")
    git_repo.commit("eins")
    echt = vm.parsen

    def kaputt(sprache, pfad, quelle):
        raise RecursionError("zu tief")

    with _speicher(tmp_path, "s") as s:
        monkeypatch.setattr(vm, "parsen", kaputt)
        erster = vermessen(git_repo.wurzel, speicher=s)
        monkeypatch.setattr(vm, "parsen", echt)
        zweiter = vermessen(git_repo.wurzel, speicher=s)
    assert erster.fehlerhaft == 1 and erster.geparst == 1
    assert zweiter.geparst == 1 and zweiter.fehlerhaft == 0  # nicht aus dem Cache


def _cli(env, *args):
    return subprocess.run([sys.executable, "-m", "selbstbild", *args],
                          capture_output=True, env=env, timeout=120)


def test_status_export_ohne_stand_legen_nichts_an(git_repo, tmp_path):
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("eins")
    home = tmp_path / "home"
    env = dict(os.environ, PYTHONPATH=str(SRC), SELBSTBILD_HOME=str(home))
    for args in (("status", str(git_repo.wurzel)),
                 ("export", str(git_repo.wurzel), str(tmp_path / "e.json"))):
        erg = _cli(env, *args)
        assert erg.returncode == 1
        assert b"noch nicht vermessen" in erg.stderr
    assert not home.exists()
    assert not (tmp_path / "e.json").exists()


def test_status_findet_stand_nach_namenswechsel(git_repo, tmp_path):
    git_repo.datei("selbstbild.yaml", "name: alt\n")
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("eins")
    env = dict(os.environ, PYTHONPATH=str(SRC), SELBSTBILD_HOME=str(tmp_path / "home"))
    erg = _cli(env, "vermessen", str(git_repo.wurzel))
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    git_repo.datei("selbstbild.yaml", "name: neu\n")
    git_repo.commit("umbenannt")
    vorher = sorted(p.name for p in (tmp_path / "home").iterdir())
    erg = _cli(env, "status", str(git_repo.wurzel))
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    assert json.loads(erg.stdout)["name"] == "alt"
    assert sorted(p.name for p in (tmp_path / "home").iterdir()) == vorher


def test_alter_stand_ohne_graph_neu_vermessen(git_repo, tmp_path):
    import sqlite3

    from selbstbild.speicher import pfad_kennung

    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("eins")
    home = tmp_path / "home"
    ordner = home / f"{git_repo.wurzel.name}-{pfad_kennung(git_repo.wurzel)}"
    ordner.mkdir(parents=True)
    c = sqlite3.connect(ordner / "graph.sqlite")  # Stand im alten Format (ohne Sortierspalten)
    c.executescript("CREATE TABLE knoten(id TEXT PRIMARY KEY, wert TEXT NOT NULL);"
                    "CREATE TABLE kanten(wert TEXT PRIMARY KEY);"
                    "CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);")
    c.executemany("INSERT INTO meta VALUES(?,?)", [
        ('"commit"', '"' + "a" * 40 + '"'), ('"gemessen_um"', '"1"'),
        ('"bericht"', json.dumps(json.dumps({"commit": "a" * 40}))),
    ])
    c.commit()
    c.close()
    env = dict(os.environ, PYTHONPATH=str(SRC), SELBSTBILD_HOME=str(home))
    ziel = tmp_path / "e.json"
    for args in (("export", str(git_repo.wurzel), str(ziel)), ("status", str(git_repo.wurzel)),
                 ("export", str(git_repo.wurzel), str(ziel))):
        erg = _cli(env, *args)
        assert erg.returncode != 0, args
        assert b"neu vermessen" in erg.stderr, erg.stderr
    assert not ziel.exists()
    erg = _cli(env, "vermessen", str(git_repo.wurzel))
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    assert _cli(env, "export", str(git_repo.wurzel), str(ziel)).returncode == 0
    assert json.loads(ziel.read_bytes())["knoten"]


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
