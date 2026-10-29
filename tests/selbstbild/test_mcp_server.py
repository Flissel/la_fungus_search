"""MCP-Server (Stufe 3, T6): reine Funktionen mit Stand-Meldung, Fehlertexte und --selftest per Subprozess."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from selbstbild import mcp_server
from selbstbild.analyse.bereiche import Bereich
from selbstbild.analyse.ergebnis import Analyse, speichern
from selbstbild.analyse.lader import ANALYSE_VERSION
from selbstbild.analyse.lauf import parameter
from selbstbild.analyse.tot import TotFund
from selbstbild.suche.index import index_bauen
from selbstbild.typen import Beleg, Kante, Knoten

FREIGEBEN = "symbol:repo:app/versand.py#newsletter_freigeben"
TOT = "symbol:repo:app/a.py#pruefe_rechnung"
DOKU = "doku_seite:repo:docs/newsletter.md"


def _sym(pfad: str, name: str) -> Knoten:
    return Knoten(f"symbol:repo:{pfad}#{name}", "symbol", name, Beleg("repo", pfad, 1, name, "d"))


def _aufbauen(git_repo, graph_stand, monkeypatch, *, index: bool = True) -> Path:
    git_repo.datei("selbstbild.yaml", "struktur:\n  - art: dienst\n    ort: app/\n    pflicht: [README.md]\n    verboten: [tmp]\n")
    git_repo.datei("app/versand.py", "def newsletter_freigeben():\n    return 1\n")
    git_repo.datei("app/a.py", "def pruefe_rechnung():\n    return 3\n")
    git_repo.datei("docs/newsletter.md", "# Versand\n\nDer Newsletter wird freigegeben.\n")
    sha = git_repo.commit("init")
    knoten = [Knoten("datei:repo:app/versand.py", "datei", "app/versand.py", None, (("lebendigkeit", "0.5"),)),
              Knoten("datei:repo:app/a.py", "datei", "app/a.py", None),
              _sym("app/versand.py", "newsletter_freigeben"), _sym("app/a.py", "pruefe_rechnung"),
              Knoten(DOKU, "doku_seite", "docs/newsletter.md", None)]
    kanten = [Kante("importiert", "datei:repo:app/versand.py", "datei:repo:app/a.py", True, None)]
    stand = graph_stand(knoten, kanten, commit=sha)
    speichern(stand, Analyse(
        commit=sha, version=ANALYSE_VERSION, parameter=parameter(False),
        bereiche=(Bereich("b1", "Versand", ("datei:repo:app/a.py", "datei:repo:app/versand.py"), 0.5),),
        pagerank_top=(), betweenness_top=(), hotspots=(),
        tot=(TotFund(TOT, "sicher_tot", ("kein Aufrufer",), None),),
        doppelungen=(), kolonie=None, luecken=()), graph_gemessen_um="")
    if index:
        index_bauen(git_repo.wurzel, stand)
    monkeypatch.setattr(mcp_server, "stand_finden", lambda w: stand / "graph.sqlite")
    return stand


def test_w_suchen_mit_stand(git_repo, graph_stand, monkeypatch):
    _aufbauen(git_repo, graph_stand, monkeypatch)
    erg = mcp_server.w_suchen(git_repo.wurzel, "Newsletter freigegeben")
    assert erg["stand"]["index_aktuell"] is True and erg["stand"]["commit"]
    assert FREIGEBEN in [t["id"] for t in erg["treffer"]]


def test_w_tot_filtert(git_repo, graph_stand, monkeypatch):
    _aufbauen(git_repo, graph_stand, monkeypatch)
    erg = mcp_server.w_tot(git_repo.wurzel, "sicher_tot", "app/")
    assert [t["id"] for t in erg["tot"]] == [TOT]
    assert mcp_server.w_tot(git_repo.wurzel, "sicher_tot", "docs/")["tot"] == []
    assert mcp_server.w_tot(git_repo.wurzel, "sicher_tot", "app/", limit=0)["tot"] == []


def test_w_bereich_lebendigkeit_regeln_nachbarn_doku(git_repo, graph_stand, monkeypatch):
    _aufbauen(git_repo, graph_stand, monkeypatch)
    w = git_repo.wurzel
    assert mcp_server.w_bereich(w, "app/a.py")["bereich"] == "Versand"
    assert mcp_server.w_lebendigkeit(w, "app/versand.py")["lebendigkeit"] == 0.5
    regeln = mcp_server.w_regeln(w, "app/versand.py")["regeln"]
    assert regeln and regeln[0]["verboten"] == ["tmp"]
    assert mcp_server.w_regeln(w, "docs/x.md")["regeln"] == []
    n = mcp_server.w_nachbarn(w, "datei:repo:app/versand.py", "aus")
    assert n["kanten"] == [{"von": "datei:repo:app/versand.py", "nach": "datei:repo:app/a.py", "sicher": True}]
    assert DOKU in [d["id"] for d in mcp_server.w_doku(w, "newsletter_freigeben")["seiten"]] or \
        mcp_server.w_doku(w, "Newsletter")["seiten"]


def test_index_fehlt_meldet(git_repo, graph_stand, monkeypatch):
    _aufbauen(git_repo, graph_stand, monkeypatch, index=False)
    erg = mcp_server.w_suchen(git_repo.wurzel, "newsletter")
    assert "Index fehlt" in erg["fehler"] and "python -m selbstbild index" in erg["fehler"]


def test_index_veraltet_meldet(git_repo, graph_stand, monkeypatch):
    stand = _aufbauen(git_repo, graph_stand, monkeypatch)
    from selbstbild.speicher import Speicher
    with Speicher(stand / "graph.sqlite") as s:
        s.meta_setzen("gemessen_um", "2099-01-01T00:00:00Z")
    erg = mcp_server.w_suchen(git_repo.wurzel, "newsletter")
    assert "Index veraltet" in erg["fehler"]


def test_erst_vermessen_meldet(git_repo, monkeypatch):
    monkeypatch.setattr(mcp_server, "stand_finden", lambda w: None)
    assert "erst vermessen" in mcp_server.w_suchen(git_repo.wurzel, "x")["fehler"]


def test_wurzel_waehlen(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    monkeypatch.setenv("SELBSTBILD_WURZELN", f"{a};{b}")
    assert mcp_server.wurzel_waehlen(None, None) == a
    assert mcp_server.wurzel_waehlen(None, str(b / "x" / "y.py")) == b
    assert mcp_server.wurzel_waehlen(str(b), None) == b
    monkeypatch.delenv("SELBSTBILD_WURZELN")
    assert mcp_server.wurzel_waehlen(None, None) is None


def test_selftest(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
           "SELBSTBILD_WURZELN": str(tmp_path), "SELBSTBILD_HOME": str(tmp_path / "nichts")}
    erg = subprocess.run([sys.executable, "-m", "selbstbild.mcp_server", "--selftest"], env=env,
                         capture_output=True, text=True, timeout=60)
    assert erg.returncode == 0, erg.stderr
    for name in ("selbstbild_suchen", "selbstbild_nachbarn", "selbstbild_bereich", "selbstbild_regeln",
                 "selbstbild_lebendigkeit", "selbstbild_tot", "selbstbild_doku"):
        assert name in erg.stdout
    assert str(tmp_path) in erg.stdout and "erst vermessen" in erg.stdout
