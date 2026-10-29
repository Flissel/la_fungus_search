"""Suchindex (Stufe 3, T2): suche.sqlite aus Graph und Analyse, nur lesend aus beiden."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from selbstbild.analyse.bereiche import Bereich
from selbstbild.analyse.doppelt import Doppelung
from selbstbild.analyse.ergebnis import Analyse, digest_von, speichern
from selbstbild.analyse.lader import ANALYSE_VERSION
from selbstbild.analyse.lauf import parameter
from selbstbild.analyse.tot import TotFund
from selbstbild.speicher import Speicher
from selbstbild.suche.index import INDEX_VERSION, index_aktuell, index_bauen, index_pfad
from selbstbild.typen import Beleg, Kante, Knoten
from selbstbild.vermessen import stand_finden, vermessen

A = "datei:repo:app/kaufe_laden.py"
B = "datei:repo:app/zwilling.py"
SYM = "symbol:repo:app/kaufe_laden.py#kaufeLaden"
DOKU = "doku_seite:repo:docs/anleitung.md"
CODE = "def kaufeLaden():\n    return 1\n"


def _repo(git_repo) -> str:
    git_repo.datei("app/kaufe_laden.py", CODE)
    git_repo.datei("app/zwilling.py", CODE)
    git_repo.datei("docs/anleitung.md", "# Versand\n\nDer Newsletter wird freigegeben, sobald es passt.\n")
    return git_repo.commit("init")


def _knoten() -> list[Knoten]:
    b = Beleg("repo", "app/kaufe_laden.py", 1, "kaufeLaden", "d")
    return [
        Knoten(A, "datei", "app/kaufe_laden.py", None, (("lebendigkeit", "0.7500"),)),
        Knoten(B, "datei", "app/zwilling.py", None, (("lebendigkeit", "0.2500"),)),
        Knoten(SYM, "symbol", "kaufeLaden", b),
        Knoten(DOKU, "doku_seite", "docs/anleitung.md", None),
        Knoten("repo:repo", "repo", "repo", None),
    ]


def _zeilen(stand: Path, sql: str, *args: object) -> list[tuple]:
    c = sqlite3.connect(f"{index_pfad(stand).as_uri()}?mode=ro", uri=True)
    try:
        return c.execute(sql, args).fetchall()
    finally:
        c.close()


def _treffer(stand: Path, frage: str) -> set[str]:
    return {i for (i,) in _zeilen(stand, "SELECT id FROM volltext WHERE volltext MATCH ?", frage)}


def _analyse(sha: str, stand: Path, gemessen_um: str | None = None) -> None:
    a = Analyse(
        commit=sha, version=ANALYSE_VERSION, parameter=parameter(False),
        bereiche=(Bereich("b1", "Versand", (A, B), 0.5),), pagerank_top=(), betweenness_top=(),
        hotspots=(), tot=(TotFund(SYM, "sicher_tot", ("kein Aufrufer",), None),),
        doppelungen=(Doppelung("gleiche_datei", (A, B), 1.0),), kolonie=None, luecken=())
    speichern(stand, a, graph_gemessen_um=gemessen_um or "")


def test_index_bauen_und_finden(git_repo, graph_stand):
    sha = _repo(git_repo)
    stand = graph_stand(_knoten(), [Kante("importiert", A, B, False, None)], commit=sha)
    zaehler = index_bauen(git_repo.wurzel, stand)
    assert zaehler["eintrag"] == 4 and zaehler["kante"] == 1  # Repo-Knoten wird nicht indiziert
    assert _treffer(stand, "freigegeben") == {DOKU}
    assert SYM in _treffer(stand, "kaufe")  # kaufeLaden -> kaufe laden kaufeladen
    assert SYM in _treffer(stand, "kaufeladen")
    assert A in _treffer(stand, "pfad:kaufe")  # Pfadsegmente, snake zerlegt
    assert A in _treffer(stand, "text:kaufe")  # Datei traegt die Symbolnamen
    assert DOKU not in _treffer(stand, "text:kaufe")
    z = _zeilen(stand, "SELECT art, repo, pfad, zeile, blob, lebendigkeit FROM eintrag WHERE id=?", SYM)
    assert z == [("symbol", "repo", "app/kaufe_laden.py", 1,
                  git_repo.git("rev-parse", "HEAD:app/kaufe_laden.py"), 0.75)]
    assert _zeilen(stand, "SELECT tot, bereich, warnung FROM eintrag WHERE id=?", A) == [(None, None, None)]
    meta = dict(_zeilen(stand, "SELECT k, v FROM meta"))
    assert meta["commit"] == sha and meta["index_version"] == str(INDEX_VERSION) and meta["analyse_digest"] == ""
    assert _zeilen(stand, "SELECT von, nach, art, sicher FROM kante") == [(A, B, "importiert", 0)]


def test_ohne_analyse(git_repo, graph_stand):
    sha = _repo(git_repo)
    stand = graph_stand(_knoten(), [], commit=sha)
    assert not (stand / "analyse.sqlite").exists()
    zaehler = index_bauen(git_repo.wurzel, stand)
    assert zaehler["eintrag"] == 4
    assert _zeilen(stand, "SELECT COUNT(*) FROM eintrag WHERE tot IS NOT NULL OR bereich IS NOT NULL"
                          " OR warnung IS NOT NULL") == [(0,)]
    assert not (stand / "analyse.sqlite").exists()  # nichts angelegt


def test_mit_analyse(git_repo, graph_stand):
    sha = _repo(git_repo)
    stand = graph_stand(_knoten(), [], commit=sha)
    _analyse(sha, stand)
    index_bauen(git_repo.wurzel, stand)
    assert _zeilen(stand, "SELECT tot, bereich FROM eintrag WHERE id=?", SYM) == [("sicher_tot", "Versand")]
    assert _zeilen(stand, "SELECT bereich, warnung FROM eintrag WHERE id=?", A) == [("Versand", f"kopie:{B}")]
    assert _zeilen(stand, "SELECT warnung FROM eintrag WHERE id=?", B) == [(f"kopie:{A}",)]
    assert _zeilen(stand, "SELECT warnung FROM eintrag WHERE id=?", SYM) == [(None,)]
    assert _zeilen(stand, "SELECT v FROM meta WHERE k='analyse_digest'") == [(digest_von(parameter(False)),)]


def test_index_aktuell(git_repo, graph_stand):
    sha = _repo(git_repo)
    stand = graph_stand(_knoten(), [], commit=sha)
    assert not index_aktuell(stand)  # noch kein Index
    index_bauen(git_repo.wurzel, stand)
    assert index_aktuell(stand)
    with Speicher(stand / "graph.sqlite") as s:
        s.meta_setzen("gemessen_um", "2026-10-29T10:00:00Z")
    assert not index_aktuell(stand)  # Graph neu vermessen
    index_bauen(git_repo.wurzel, stand)
    assert index_aktuell(stand)
    _analyse(sha, stand, "2026-10-29T10:00:00Z")
    assert not index_aktuell(stand)  # Analyse ist dazugekommen
    index_bauen(git_repo.wurzel, stand)
    assert index_aktuell(stand)


def test_neubau_ersetzt_und_ist_deterministisch(git_repo, graph_stand):
    sha = _repo(git_repo)
    stand = graph_stand(_knoten(), [Kante("importiert", A, B, True, None)], commit=sha)
    index_bauen(git_repo.wurzel, stand)
    erst = _zeilen(stand, "SELECT * FROM eintrag ORDER BY rowid")
    index_bauen(git_repo.wurzel, stand)
    assert _zeilen(stand, "SELECT * FROM eintrag ORDER BY rowid") == erst
    assert [z[0] for z in erst] == sorted(z[0] for z in erst)  # eingefuegt nach id
    assert not (stand / "suche.sqlite.tmp").exists()


def test_graph_unberuehrt(git_repo, graph_stand):
    sha = _repo(git_repo)
    stand = graph_stand(_knoten(), [], commit=sha)
    _analyse(sha, stand)
    vorher = [(stand / n).read_bytes() for n in ("graph.sqlite", "analyse.sqlite")]
    index_bauen(git_repo.wurzel, stand)
    assert [(stand / n).read_bytes() for n in ("graph.sqlite", "analyse.sqlite")] == vorher
    assert sorted(p.name for p in stand.iterdir() if p.name.startswith("suche")) == ["suche.sqlite"]


def test_keine_env_werte(git_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "home"))
    git_repo.datei("compose.yml", "services:\n  web:\n    image: x\n    environment:\n"
                                  "      DB_URL: geheim-123\n")
    git_repo.datei("app.py", "def f():\n    return 1\n")
    git_repo.commit("init")
    vermessen(git_repo.wurzel)
    stand = stand_finden(git_repo.wurzel).parent
    zaehler = index_bauen(git_repo.wurzel, stand)
    assert zaehler["eintrag"] > 0
    assert any(i.startswith("env:") for (i,) in _zeilen(stand, "SELECT id FROM eintrag"))
    assert b"geheim-123" not in index_pfad(stand).read_bytes()
    assert _treffer(stand, "geheim") == set()


def test_index_bauen_wiederholt_replace_bei_permissionerror(git_repo, graph_stand, monkeypatch):
    """U4: Windows sperrt geoeffnete Dateien; os.replace wird bis zu 5x wiederholt, tmp-Name traegt die pid."""
    import os

    from selbstbild.suche import index as idx
    git_repo.datei("a.py", "def f():\n    return 1\n")
    sha = git_repo.commit("init")
    stand = graph_stand([Knoten("datei:repo:a.py", "datei", "a.py", None)], [], commit=sha)
    echt, versuche, tmps = os.replace, [], []

    def sperre(quelle, ziel):
        versuche.append(1)
        tmps.append(str(quelle))
        if len(versuche) < 3:
            raise PermissionError("gesperrt")
        return echt(quelle, ziel)

    monkeypatch.setattr(idx.os, "replace", sperre)
    monkeypatch.setattr(idx.time, "sleep", lambda s: None)
    idx.index_bauen(git_repo.wurzel, stand)
    assert len(versuche) == 3 and idx.index_pfad(stand).is_file()
    assert str(os.getpid()) in tmps[0]

    monkeypatch.setattr(idx.os, "replace", lambda q, z: (_ for _ in ()).throw(PermissionError("dauerhaft")))
    with pytest.raises(PermissionError):
        idx.index_bauen(git_repo.wurzel, stand)
    assert not list(stand.glob("suche.sqlite.*tmp"))
