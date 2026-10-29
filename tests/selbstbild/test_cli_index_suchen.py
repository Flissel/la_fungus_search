"""CLI `index` und `suchen` (Stufe 3, T6 Fix U5): Zaehler/Treffer als JSON, Exit 1 ohne Stand, 3 ohne Graph."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from selbstbild.__main__ import main
from selbstbild.speicher import pfad_kennung
from selbstbild.vermessen import vermessen


@pytest.fixture
def heim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("SELBSTBILD_HOME", str(h))
    return h


def _vermessen(git_repo) -> None:
    git_repo.datei("app/versand.py", "def newsletter_freigeben():\n    return 1\n")
    git_repo.datei("docs/n.md", "# Versand\n\nDer Newsletter wird freigegeben.\n")
    git_repo.commit("init")
    vermessen(git_repo.wurzel)


def test_index_und_suchen(git_repo, heim, capsys):
    _vermessen(git_repo)
    assert main(["index", str(git_repo.wurzel)]) == 0
    zaehler = json.loads(capsys.readouterr().out)
    assert zaehler["eintrag"] > 0 and "kante" in zaehler
    assert main(["suchen", str(git_repo.wurzel), "newsletter freigeben", "--top", "3"]) == 0
    antwort = json.loads(capsys.readouterr().out)
    assert antwort["stand"]["index_aktuell"] is True
    assert 0 < len(antwort["treffer"]) <= 3


def test_suchen_ohne_index_meldet(git_repo, heim, capsys):
    _vermessen(git_repo)
    assert main(["suchen", str(git_repo.wurzel), "newsletter"]) == 1
    assert "Index fehlt" in capsys.readouterr().err


def test_index_ohne_stand_exit_1(git_repo, heim, capsys):
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("eins")
    assert main(["index", str(git_repo.wurzel)]) == 1
    assert "erst vermessen" in capsys.readouterr().err


def test_index_stand_ohne_graph_exit_3(git_repo, heim, capsys):
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("eins")
    ordner = heim / f"{git_repo.wurzel.name}-{pfad_kennung(git_repo.wurzel)}"
    ordner.mkdir(parents=True)
    c = sqlite3.connect(ordner / "graph.sqlite")
    c.executescript("CREATE TABLE knoten(id TEXT PRIMARY KEY, wert TEXT NOT NULL);"
                    "CREATE TABLE kanten(wert TEXT PRIMARY KEY);"
                    "CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);")
    c.commit()
    c.close()
    assert main(["index", str(git_repo.wurzel)]) == 3
    assert "neu vermessen" in capsys.readouterr().err
