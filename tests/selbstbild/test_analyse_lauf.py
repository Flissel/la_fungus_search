"""Analyse-Lauf (T11): Ende-zu-Ende ueber vermessen + analysieren, Determinismus, CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from selbstbild.analyse.ergebnis import export_json
from selbstbild.analyse.lauf import NichtAnalysiert, NichtVermessen, analysieren, gespeicherte_analyse
from selbstbild.vermessen import stand_finden, vermessen

SRC = Path(__file__).resolve().parents[2] / "src"

MAIN = 'from app import lauf\n\n\nif __name__ == "__main__":\n    lauf()\n'
APP = ("def helfer():\n    return 1\n\n\ndef lauf():\n    return helfer()\n\n\n"
       "def alt_tot():\n    x = 1\n    y = 2\n    return x + y\n")


def _klein(git_repo) -> None:
    git_repo.datei("main.py", MAIN)
    git_repo.datei("app.py", APP)
    git_repo.commit("alt", datum="2025-01-01T00:00:00Z")
    git_repo.datei("README.md", "Ein kleines Repo.\n")
    git_repo.commit("neu", datum="2026-01-01T00:00:00Z")


@pytest.fixture
def heim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("SELBSTBILD_HOME", str(h))
    return h


def test_analysieren_end_to_end(git_repo, heim, tmp_path):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    a1 = analysieren(git_repo.wurzel)
    export_json(a1, tmp_path / "1.json")
    a2 = analysieren(git_repo.wurzel)
    export_json(a2, tmp_path / "2.json")
    b1, b2 = (tmp_path / "1.json").read_bytes(), (tmp_path / "2.json").read_bytes()
    assert b1 == b2  # zwei Laeufe, gleiche Bytes (Laufzeiten stehen nicht im Export)
    tot = {t.symbol: t.stufe for t in a1.tot}
    assert tot.get("symbol:repo:app.py#alt_tot") == "sicher_tot", tot
    assert "symbol:repo:app.py#lauf" not in tot
    assert "symbol:repo:app.py#helfer" not in tot
    stat = dict(a1.statistik)
    assert "laufzeit_mathe_ms" in stat and "laufzeit_kolonie_ms" in stat
    assert "doppelt_eimer_uebersprungen" in stat
    assert "reproduzierbar_mathe" not in stat  # nur mit pruefen_reproduzierbar
    export = json.loads(b1)
    assert all(not k.startswith("laufzeit_") for k, _ in export["statistik"])
    assert a1.kolonie is not None and export["kolonie"] is not None
    # Gespeichert: analyse.sqlite neben graph.sqlite, gleiche Bytes wie der Export
    stand = stand_finden(git_repo.wurzel)
    assert stand is not None and (stand.parent / "analyse.sqlite").is_file()


def test_luecken_nach_allen_analysen(git_repo, heim):
    """Ruling S3: `luecken` enthaelt auch die Nachtraege von dateigraph (symbol_ohne_datei)."""
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    a = analysieren(git_repo.wurzel, kolonie=False)
    luecken = dict(a.luecken)
    assert "kante_ohne_knoten" in luecken
    assert "symbol_ohne_datei" in luecken
    assert a.kolonie is None and "laufzeit_kolonie_ms" not in dict(a.statistik)


def test_reproduzierbar_geprueft(git_repo, heim):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    a = analysieren(git_repo.wurzel, pruefen_reproduzierbar=True)
    stat = dict(a.statistik)
    assert stat["reproduzierbar_mathe"] == 1 and stat["reproduzierbar_kolonie"] == 1


def test_gespeicherte_analyse_rundweg(git_repo, heim, tmp_path):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    a = analysieren(git_repo.wurzel)
    g = gespeicherte_analyse(git_repo.wurzel)
    export_json(a, tmp_path / "a.json")
    export_json(g, tmp_path / "g.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "g.json").read_bytes()
    # Laufzeiten kommen aus der Nebentabelle zurueck (goldsatz.vergleichen liest sie)
    assert dict(g.statistik)["laufzeit_mathe_ms"] == dict(a.statistik)["laufzeit_mathe_ms"]


def test_nicht_vermessen_wirft(git_repo, heim):
    _klein(git_repo)
    with pytest.raises(NichtVermessen, match="erst vermessen"):
        analysieren(git_repo.wurzel)
    assert not heim.exists()


def test_nicht_analysiert_wirft(git_repo, heim):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    with pytest.raises(NichtAnalysiert, match="erst analysieren"):
        gespeicherte_analyse(git_repo.wurzel)


def _cli(home: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ, PYTHONPATH=str(SRC), SELBSTBILD_HOME=str(home))
    return subprocess.run([sys.executable, "-m", "selbstbild", *args],
                          capture_output=True, env=env, timeout=300)


def test_cli_analysieren(git_repo, heim, tmp_path):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    ziel = tmp_path / "x" / "analyse.json"
    erg = _cli(heim, "analysieren", str(git_repo.wurzel), "--export", str(ziel))
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    z = json.loads(erg.stdout)
    assert z["tot"]["sicher_tot"] >= 1
    assert set(z["tot"]) == {"sicher_tot", "wahrscheinlich_tot", "verdaechtig"}
    assert set(z["doppelungen"]) == {"gleiche_datei", "gleiches_symbol", "aehnliche_datei"}
    assert len(z["bereiche_top"]) <= 10 and len(z["hotspots_top"]) <= 10
    assert z["dauer_ms"] >= 0 and z["speicher_spitze_mb"] > 0
    assert json.loads(ziel.read_bytes())["commit"] == z["commit"]

    erg = _cli(heim, "analysieren", str(git_repo.wurzel), "--ohne-kolonie")
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    assert json.loads(erg.stdout)["kolonie"] is None


def v_commit(git_repo) -> str:
    return git_repo.git("rev-parse", "HEAD")


def test_export_prozessuebergreifend_bytegleich(git_repo, heim, tmp_path):
    """Zwei Prozesse mit verschiedenem PYTHONHASHSEED, einer davon mit Pruef-Schalter: gleiche Bytes."""
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    ziele = []
    for n, (seed, extra) in enumerate((("1", ()), ("2", ("--pruefen-reproduzierbar",)))):
        ziel = tmp_path / f"p{n}.json"
        env = dict(os.environ, PYTHONPATH=str(SRC), SELBSTBILD_HOME=str(heim), PYTHONHASHSEED=seed)
        erg = subprocess.run([sys.executable, "-m", "selbstbild", "analysieren", str(git_repo.wurzel),
                              "--export", str(ziel), *extra], capture_output=True, env=env, timeout=300)
        assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
        ziele.append(ziel.read_bytes())
    assert ziele[0] == ziele[1]


def test_neu_vermessener_graph_verlangt_neue_analyse(git_repo, heim):
    from selbstbild.speicher import Speicher

    _klein(git_repo)
    vermessen(git_repo.wurzel)
    analysieren(git_repo.wurzel, kolonie=False)
    assert gespeicherte_analyse(git_repo.wurzel) is not None
    with Speicher(stand_finden(git_repo.wurzel)) as s:  # wie ein neuer Vermessungslauf am selben Commit
        s.meta_setzen("gemessen_um", "1")
    with pytest.raises(NichtAnalysiert, match="neu vermessen"):
        gespeicherte_analyse(git_repo.wurzel)


def test_cli_stand_ohne_graph(git_repo, tmp_path):
    import sqlite3

    from selbstbild.speicher import pfad_kennung

    _klein(git_repo)
    home = tmp_path / "home"
    ordner = home / f"{git_repo.wurzel.name}-{pfad_kennung(git_repo.wurzel)}"
    ordner.mkdir(parents=True)
    sqlite3.connect(ordner / "graph.sqlite").close()  # leere Datei ohne Tabellen
    erg = _cli(home, "analysieren", str(git_repo.wurzel))
    assert erg.returncode == 3 and b"ohne Graph" in erg.stderr
    erg = _cli(home, "goldsatz", str(git_repo.wurzel), str(tmp_path / "g.yaml"))
    assert erg.returncode == 3 and b"ohne Graph" in erg.stderr


def test_nicht_vermessen(git_repo, tmp_path):
    _klein(git_repo)
    home = tmp_path / "leer"
    erg = _cli(home, "analysieren", str(git_repo.wurzel))
    assert erg.returncode != 0
    assert b"erst vermessen" in erg.stderr
    assert not home.exists()


def test_cli_goldsatz_und_vergleichen(git_repo, heim, tmp_path):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    gold = tmp_path / "g" / "gold.yaml"
    erg = _cli(heim, "goldsatz", str(git_repo.wurzel), str(gold))
    assert erg.returncode != 0 and b"erst analysieren" in erg.stderr  # ohne gespeicherte Analyse
    assert _cli(heim, "analysieren", str(git_repo.wurzel)).returncode == 0
    erg = _cli(heim, "goldsatz", str(git_repo.wurzel), str(gold))
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    doc = yaml.safe_load(gold.read_text(encoding="utf-8"))
    assert [e["symbol"] for e in doc["tot"]] == ["symbol:repo:app.py#alt_tot"]
    doc["tot"][0]["urteil"] = "ja"
    gold.write_text(yaml.safe_dump(doc), encoding="utf-8")
    erg = _cli(heim, "vergleichen", str(git_repo.wurzel), str(gold))
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    v = json.loads(erg.stdout)
    assert v["tot"]["mathe"] == 1.0 and v["tot"]["n_mathe"] == 1
    assert v["laufzeit_mathe_ms"] is not None

    # Ein zweiter goldsatz-Aufruf ueberschreibt eingetragene Urteile nicht
    erg = _cli(heim, "goldsatz", str(git_repo.wurzel), str(gold))
    assert erg.returncode == 2 and "existiert schon".encode() in erg.stderr
    assert yaml.safe_load(gold.read_text(encoding="utf-8"))["tot"][0]["urteil"] == "ja"
    # Goldsatz von einem anderen Commit: Warnung auf stderr, Ergebnis trotzdem
    doc["commit"] = "f" * 40
    gold.write_text(yaml.safe_dump(doc), encoding="utf-8")
    erg = _cli(heim, "vergleichen", str(git_repo.wurzel), str(gold))
    assert erg.returncode == 0 and b"Warnung" in erg.stderr and b"0 tot-Eintraege" in erg.stderr
    erg = _cli(heim, "goldsatz", str(git_repo.wurzel), str(gold), "--ueberschreiben")
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    neu = yaml.safe_load(gold.read_text(encoding="utf-8"))
    assert neu["tot"][0]["urteil"] == "offen" and neu["commit"] == v_commit(git_repo)

    kaputt = tmp_path / "kaputt.yaml"
    kaputt.write_text("- nur\n- eine liste\n", encoding="utf-8")
    erg = _cli(heim, "vergleichen", str(git_repo.wurzel), str(kaputt))
    assert erg.returncode != 0
    assert b"Goldsatz ungueltig" in erg.stderr


def test_cli_anderer_db_fehler_exit_1(git_repo, tmp_path):
    import sqlite3

    from selbstbild.speicher import pfad_kennung

    _klein(git_repo)
    home = tmp_path / "home"
    ordner = home / f"{git_repo.wurzel.name}-{pfad_kennung(git_repo.wurzel)}"
    ordner.mkdir(parents=True)
    c = sqlite3.connect(ordner / "graph.sqlite")
    c.execute("CREATE TABLE meta(falsch TEXT)")
    c.commit()
    c.close()
    for befehl in (("analysieren", str(git_repo.wurzel)), ("goldsatz", str(git_repo.wurzel), str(tmp_path / "g.yaml"))):
        erg = _cli(home, *befehl)
        assert erg.returncode == 1, befehl
        assert b"no such column" in erg.stderr and b"Traceback" not in erg.stderr


@pytest.mark.parametrize("modul, name, wert", [
    ("selbstbild.analyse.tot", "VORLAGEN_ENDUNGEN", (".html",)),
    ("selbstbild.analyse.tot", "INERTE_DEKORATOREN", frozenset({"property"})),
    ("selbstbild.analyse", "TEST_ORDNER", frozenset({"tests"})),
    ("selbstbild.analyse", "CODE_SPRACHEN", frozenset({"python"})),
])
def test_regelkonstanten_im_parameter_digest_s13(monkeypatch, modul, name, wert):
    import importlib
    import re

    from selbstbild.analyse.ergebnis import digest_von
    from selbstbild.analyse.lauf import parameter

    vorher = digest_von(parameter(True))
    assert "tot_regeln" not in dict(parameter(True)) and "regeln" in dict(parameter(True))
    monkeypatch.setattr(importlib.import_module(modul), name, wert)
    assert digest_von(parameter(True)) != vorher
    monkeypatch.undo()
    assert digest_von(parameter(True)) == vorher
    from selbstbild.analyse import tot

    monkeypatch.setattr(tot, "_KONVENTION", re.compile("^x$"))
    assert digest_von(parameter(True)) != vorher


def test_kolonie_budget_lauf_s11(git_repo, heim):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    a = analysieren(git_repo.wurzel, kolonie_budget=True)
    stat = dict(a.statistik)
    assert stat["kolonie_budget_schritte"] == 1  # ceil(10 * 2 Dateien / 192)
    assert {"kolonie_budget_besuchte_dateien", "kolonie_budget_tote_ecken",
            "kolonie_budget_zentrum_in_pagerank_top50_dateien", "laufzeit_kolonie_budget_ms"} <= set(stat)
    assert dict(a.parameter)["kolonie_budget"] == "an"
    gespeichert = gespeicherte_analyse(git_repo.wurzel)  # Lauf mit Budget wird bevorzugt gefunden
    assert dict(gespeichert.parameter)["kolonie_budget"] == "an"
    assert dict(gespeichert.statistik)["kolonie_budget_schritte"] == 1
    from selbstbild.analyse.lauf import zusammenfassung

    assert zusammenfassung(a)["kolonie_budget"]["besuchte_dateien"] == stat["kolonie_budget_besuchte_dateien"]


def test_cli_kolonie_budget_schalter(git_repo, heim, tmp_path):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    erg = _cli(heim, "analysieren", str(git_repo.wurzel), "--kolonie-budget")
    assert erg.returncode == 0, erg.stderr.decode("utf-8", "replace")
    z = json.loads(erg.stdout)
    assert z["kolonie_budget"]["schritte"] == 1 and "besuchte_dateien" in z["kolonie_budget"]
    erg = _cli(heim, "analysieren", str(git_repo.wurzel))
    assert json.loads(erg.stdout)["kolonie_budget"] is None


def test_pagerank_quersprache_gezaehlt(git_repo, heim):
    _klein(git_repo)
    vermessen(git_repo.wurzel)
    assert dict(analysieren(git_repo.wurzel, kolonie=False).statistik)["pagerank_quersprache_ausgelassen"] == 0
