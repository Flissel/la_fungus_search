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

    kaputt = tmp_path / "kaputt.yaml"
    kaputt.write_text("- nur\n- eine liste\n", encoding="utf-8")
    erg = _cli(heim, "vergleichen", str(git_repo.wurzel), str(kaputt))
    assert erg.returncode != 0
    assert b"Goldsatz ungueltig" in erg.stderr
