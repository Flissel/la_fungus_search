"""MCP-Server (Stufe 3, T6): reine Funktionen mit Stand-Meldung, Fehlertexte und --selftest per Subprozess."""
from __future__ import annotations

import os
import sqlite3
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


REGELN = ("struktur:\n  - art: dienst\n    ort: app/\n    pflicht: [README.md]\n    verboten: [tmp]\n"
          "  - art: workflow\n    ort: .github/\n    pflicht: [ci.yml]\n")


def _aufbauen(git_repo, graph_stand, monkeypatch, *, index: bool = True, regeln: str = REGELN) -> Path:
    git_repo.datei("selbstbild.yaml", regeln)
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
    assert [d["id"] for d in mcp_server.w_doku(w, "newsletter_freigeben")["seiten"]] == [DOKU]
    assert mcp_server.w_doku(w, "zzzgibtesnicht")["seiten"] == []


def _sub_eintraege(stand: Path) -> None:
    """Ein Submodul-Repo `repo/sub` im Index (Submodule heissen `<wurzelrepo>/<submodulpfad>`)."""
    c = sqlite3.connect(stand / "suche.sqlite")
    c.execute("INSERT INTO eintrag VALUES ('datei:repo/sub:x/y.py','datei','repo/sub','x/y.py',NULL,NULL,0.9,NULL,'Sub',NULL)")
    c.execute("INSERT INTO eintrag VALUES ('symbol:repo/sub:x/y.py#f','symbol','repo/sub','x/y.py',1,NULL,NULL,'sicher_tot',NULL,NULL)")
    c.commit()
    c.close()


def test_absolute_pfade(git_repo, graph_stand, monkeypatch):
    stand = _aufbauen(git_repo, graph_stand, monkeypatch)
    _sub_eintraege(stand)
    w = git_repo.wurzel
    a = str(w / "app" / "a.py")
    assert mcp_server.w_bereich(w, a)["bereich"] == "Versand"
    assert mcp_server.w_bereich(w, a.replace("/", "\\"))["bereich"] == "Versand"
    assert mcp_server.w_lebendigkeit(w, str(w / "app" / "versand.py"))["lebendigkeit"] == 0.5
    assert mcp_server.w_regeln(w, str(w / "app" / "a.py"))["regeln"][0]["art"] == "dienst"
    assert [t["id"] for t in mcp_server.w_tot(w, "sicher_tot", str(w / "app"))["tot"]] == [TOT]
    # Submodul: laengster Repo-Praefix gewinnt
    assert mcp_server.w_bereich(w, str(w / "sub" / "x" / "y.py"))["bereich"] == "Sub"
    assert mcp_server.w_lebendigkeit(w, "sub/x/y.py")["lebendigkeit"] == 0.9
    assert [t["pfad"] for t in mcp_server.w_tot(w, "sicher_tot", "sub")["tot"]] == ["sub/x/y.py"]
    assert len(mcp_server.w_tot(w, "sicher_tot", "")["tot"]) == 2
    assert "ausserhalb" in mcp_server.w_bereich(w, str(w.parent / "fremd" / "a.py"))["fehler"]
    assert "ausserhalb" in mcp_server.w_regeln(w, "../x.py")["fehler"]


def test_absolute_pfade_gemischte_schreibweise(git_repo, graph_stand, monkeypatch):
    _aufbauen(git_repo, graph_stand, monkeypatch)
    monkeypatch.setattr(mcp_server, "PFAD_OHNE_GROSSKLEIN", True)  # Windows-Verhalten auf jeder Plattform
    w = git_repo.wurzel
    misch = str(w / "app" / "a.py").swapcase()
    assert mcp_server.w_bereich(w, misch)["bereich"] == "Versand"
    assert mcp_server.w_regeln(w, misch)["regeln"][0]["art"] == "dienst"
    assert [t["id"] for t in mcp_server.w_tot(w, "sicher_tot", str(w / "APP").swapcase())["tot"]] == [TOT]


def test_punkt_praefix_bleibt_erhalten(git_repo, graph_stand, monkeypatch):
    """`lstrip("./")` frass das fuehrende Punkt-Zeichen von `.github`."""
    _aufbauen(git_repo, graph_stand, monkeypatch)
    w = git_repo.wurzel
    assert [r["art"] for r in mcp_server.w_regeln(w, ".github/x")["regeln"]] == ["workflow"]
    assert [r["art"] for r in mcp_server.w_regeln(w, "./.github/x")["regeln"]] == ["workflow"]
    assert mcp_server.w_regeln(w, "github/x")["regeln"] == []
    assert mcp_server.w_regeln(w, "./app/a.py")["regeln"][0]["art"] == "dienst"


@pytest.mark.parametrize("text", ["struktur: [", "unbekannt: 1\n"])
def test_regeldatei_ungueltig(git_repo, graph_stand, monkeypatch, text):
    _aufbauen(git_repo, graph_stand, monkeypatch)  # Index braucht gueltige Regeln; die Datei kippt danach
    from selbstbild import quelle
    from selbstbild.regeln import regeln_laden

    monkeypatch.setattr(quelle, "_regeln_aus_commit", lambda w, c: regeln_laden(text))
    erg = mcp_server.w_regeln(git_repo.wurzel, "app/a.py")
    assert erg["fehler"].startswith("Regeldatei ungueltig")
    assert "unbekannt:" not in erg["fehler"]


def test_interner_fehler_ohne_meldungstext(git_repo, graph_stand, monkeypatch):
    _aufbauen(git_repo, graph_stand, monkeypatch)
    from selbstbild.suche import abfrage

    def kaputt(*a, **k):
        raise RuntimeError("geheim-wert-123")

    monkeypatch.setattr(abfrage, "suchen", kaputt)
    erg = mcp_server.w_suchen(git_repo.wurzel, "newsletter")
    assert erg == {"fehler": "RuntimeError: interner Fehler"}


def test_import_laedt_kein_igraph():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    erg = subprocess.run([sys.executable, "-c",
                          "import sys, selbstbild.mcp_server; "
                          "assert not {'igraph','leidenalg','numpy'} & set(sys.modules), sorted(sys.modules)"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert erg.returncode == 0, erg.stderr


def test_registrierte_werkzeuge_gleich_werkzeuge():
    import asyncio

    namen = [t.name for t in asyncio.run(mcp_server.server_bauen().list_tools())]
    assert sorted(namen) == sorted(mcp_server.WERKZEUGE) and len(namen) == 7


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
