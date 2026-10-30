"""doku_loop: Graph-Modell, Speicher und Diff."""

import sys
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku.anker import Beleg  # noqa: E402
from selbstbild.doku.graph import (  # noqa: E402
    Graph, Kante, Knoten, graph_diff, hop_zustand_neu, kante_id, mit_schlank, schlank,
)

ZEIT = "2026-10-27T12:00:00Z"


def _beleg(digest="d1", anker="a.f", file="x/a.py"):
    return Beleg(art="symbol", file=file, anker=anker, digest=digest, commit="abc", zeile=3)


def _graph(reihenfolge=1):
    g = Graph({}, {}, "aaa", "bbb")
    knoten = [Knoten("s:a", "space", "a"), Knoten("d:b", "dienst", "b", space="a", attribute={"port": "1"})]
    if reihenfolge == -1:
        knoten.reverse()
    for k in knoten:
        g.knoten_hinzu(k)
    g.kante_hinzu(Kante(kante_id("ruft_auf", "s:a", "d:b"), "ruft_auf", "s:a", "d:b", "extraktor", (_beleg(),)))
    return g


def test_speichern_laden_roundtrip_und_digest_stabil(tmp_path):
    g = _graph()
    p = tmp_path / "graph.json"
    g.speichern(p, ZEIT)
    g2 = Graph.laden(p)
    assert g2.knoten == g.knoten and g2.kanten == g.kanten
    assert g2.commit_aussen == "aaa" and g2.commit_submodul == "bbb"
    assert g2.digest() == g.digest() == _graph(-1).digest()
    g2.kante_hinzu(Kante(kante_id("ruft_auf", "s:a", "d:b"), "ruft_auf", "s:a", "d:b", "extraktor", (_beleg("neu"),)))
    assert g2.digest() != g.digest()


def test_kante_id_deterministisch_hop():
    assert kante_id("ruft_auf", "a", "b") == "ruft_auf:a->b"
    assert kante_id("hop", "a", "b", kette="K1", nr=2) == "hop:a->b#K1/2"
    z = hop_zustand_neu()
    assert z == {"verdrahtet": "unbestimmt", "konfiguriert": "unbestimmt",
                 "gemessen": {"wert": "keine_probe", "zeit": None}, "aus": False, "nur_doku": False}
    assert hop_zustand_neu() is not hop_zustand_neu()


def test_idempotent_und_validierung():
    g = _graph()
    k = Kante("ruft_auf:s:a->d:b", "ruft_auf", "s:a", "d:b", "seed", (_beleg("z"),))
    g.kante_hinzu(k)
    assert len(g.kanten) == 1 and g.kanten[k.id].quelle == "seed"
    with pytest.raises(ValueError):
        Knoten("x", "unsinn", "x")
    with pytest.raises(ValueError):
        Kante("i", "unsinn", "a", "b", "extraktor")
    with pytest.raises(ValueError):
        Kante("i", "hop", "a", "b", "geraten")


def test_graph_diff_neue_weg_und_geaenderte_belege():
    alt, neu = _graph(), _graph()
    eid = "ruft_auf:s:a->d:b"
    alt.kante_hinzu(Kante("nur_alt", "nutzt_dienst", "s:a", "d:b", "seed"))
    neu.kante_hinzu(Kante("nur_neu", "nutzt_dienst", "s:a", "d:b", "seed"))
    alt.kante_hinzu(Kante(eid, "ruft_auf", "s:a", "d:b", "extraktor",
                          (_beleg("d1"), _beleg("d1", anker="a.g"), _beleg("d1", anker="a.h"))))
    neu.kante_hinzu(Kante(eid, "ruft_auf", "s:a", "d:b", "extraktor",
                          (_beleg("d2"), _beleg("d1", anker="a.g"), _beleg("d1", anker="a.neu"))))
    d = graph_diff(alt, neu)
    assert d.neue_kanten == ["nur_neu"] and d.weg_kanten == ["nur_alt"]
    assert sorted((i, b.anker, art) for i, b, art in d.belege_geaendert) == [
        (eid, "a.f", "geaendert"), (eid, "a.h", "verschwunden")]


def test_speichern_sortiert_fuer_kleine_diffs(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _graph(1).speichern(a, ZEIT)
    _graph(-1).speichern(b, ZEIT)
    assert a.read_bytes() == b.read_bytes()
    _graph(1).speichern(b, ZEIT)
    assert a.read_bytes() == b.read_bytes()


def test_digest_stabil_ignoriert_messzeit_und_beleg_commit():
    """Ruling 45 (C.9.5): Messzeit und Beleg-Commit sind fluechtig; Wert, Digest und Zeile nicht."""
    def mit(zeit="2026-10-27T12:00:00Z", wert="laeuft", commit="abc", digest="d1", zeile=3):
        g = _graph()
        z = hop_zustand_neu()
        z["gemessen"] = {"wert": wert, "zeit": zeit}
        b = Beleg("symbol", "x/a.py", "a.f", digest, commit, zeile)
        g.kante_hinzu(Kante(kante_id("hop", "s:a", "d:b", kette="K1", nr=1), "hop", "s:a", "d:b", "seed", (b,),
                            {"kette": "K1", "nr": "1", "zustand": z}))
        return g

    basis = mit()
    assert mit(zeit="2026-10-28T09:00:00Z").digest_stabil() == basis.digest_stabil()
    assert mit(commit="neu").digest_stabil() == basis.digest_stabil()
    assert mit(zeit="2026-10-28T09:00:00Z").digest() != basis.digest()  # der volle Digest sieht beides
    assert mit(wert="bricht").digest_stabil() != basis.digest_stabil()
    assert mit(digest="d2").digest_stabil() != basis.digest_stabil()
    assert mit(zeile=9).digest_stabil() != basis.digest_stabil()


# ---------- Task 15b (Ruling 49): schlanker Repo-Graph ----------

def _voll(reihenfolge=1):
    """Voller Graph: Hops, Aussage, Compose, plus abgeleitete Symbol-Daten (ruft_auf, gehoert_zu, Routen, ...)."""
    g = Graph({}, {}, "aaa", "bbb")
    knoten = [
        Knoten("space:a", "space", "a", space="a", datei="sp/a"),
        Knoten("dienst:b", "dienst", "b", attribute={"herkunft": "register", "port": "8130"}),
        Knoten("dienst:c", "dienst", "c", attribute={"herkunft": "register"}),
        Knoten("kette:K1", "kette", "Fixture"),
        Knoten("kette:K2", "kette", "ohne Hops"),
        Knoten("komponente:client", "komponente", "Client"),
        Knoten("komponente:frei", "komponente", "ohne Kante"),
        Knoten("mensch:felix", "mensch", "Felix"),
        Knoten("tabelle:marketing.x", "tabelle", "marketing.x"),
        Knoten("symbol:a:client.senden", "symbol", "client.senden", space="a", datei="sp/a/client.py"),
        Knoten("symbol:a:client.helfer", "symbol", "client.helfer", space="a", datei="sp/a/client.py"),
        Knoten("symbol:a:client.ziel", "symbol", "client.ziel", space="a", datei="sp/a/client.py"),
        Knoten("modul:sp/a/client.py", "modul", "sp/a/client.py", space="a", datei="sp/a/client.py"),
        Knoten("modul:sp/a/rest.py", "modul", "sp/a/rest.py", space="a", datei="sp/a/rest.py"),
        Knoten("route:POST /api/x@a", "route", "POST /api/x", space="a", datei="sp/a/api.py"),
        Knoten("mcp_werkzeug:w", "mcp_werkzeug", "w", attribute={"herkunft": "extraktor"}),
        Knoten("aussage:abc", "aussage", "Der Client ruft senden auf.", datei="docs/spaces/KETTEN.md"),
    ]
    kanten = [
        Kante(kante_id("hop", "komponente:client", "dienst:b", kette="K1", nr=1), "hop", "komponente:client",
              "dienst:b", "seed", (_beleg(file="sp/a/client.py", anker="client.senden"),),
              {"kette": "K1", "nr": "1", "zustand": hop_zustand_neu()}),
        Kante(kante_id("hop", "mensch:felix", "tabelle:marketing.x", kette="K1", nr=2), "hop", "mensch:felix",
              "tabelle:marketing.x", "seed", (), {"kette": "K1", "nr": "2", "zustand": hop_zustand_neu()}),
        Kante(kante_id("hop", "symbol:a:client.ziel", "dienst:b", kette="K1", nr=3), "hop", "symbol:a:client.ziel",
              "dienst:b", "seed", (), {"kette": "K1", "nr": "3", "zustand": hop_zustand_neu()}),
        Kante(kante_id("belegt_durch", "aussage:abc", "modul:sp/a/client.py"), "belegt_durch", "aussage:abc",
              "modul:sp/a/client.py", "extraktor", (_beleg(file="sp/a/client.py", anker="client.senden"),)),
        Kante(kante_id("verbindet", "dienst:c", "dienst:b"), "verbindet", "dienst:c", "dienst:b", "extraktor",
              (), {"env": ["B_URL"]}),
        Kante(kante_id("ruft_auf", "symbol:a:client.senden", "symbol:a:client.helfer"), "ruft_auf",
              "symbol:a:client.senden", "symbol:a:client.helfer", "extraktor", (), {"space": "a"}),
        Kante(kante_id("gehoert_zu", "symbol:a:client.senden", "modul:sp/a/client.py"), "gehoert_zu",
              "symbol:a:client.senden", "modul:sp/a/client.py", "extraktor", (), {"space": "a"}),
        Kante(kante_id("gehoert_zu", "modul:sp/a/rest.py", "space:a"), "gehoert_zu", "modul:sp/a/rest.py",
              "space:a", "extraktor", (), {"space": "a"}),
        Kante(kante_id("definiert_route", "symbol:a:client.helfer", "route:POST /api/x@a"), "definiert_route",
              "symbol:a:client.helfer", "route:POST /api/x@a", "extraktor", (_beleg(),), {"space": "a"}),
        Kante(kante_id("registriert", "symbol:a:client.helfer", "mcp_werkzeug:w"), "registriert",
              "symbol:a:client.helfer", "mcp_werkzeug:w", "extraktor", (), {"space": "a"}),
        Kante(kante_id("nutzt_dienst", "symbol:a:client.senden", "dienst:b"), "nutzt_dienst",
              "symbol:a:client.senden", "dienst:b", "extraktor", (_beleg(),), {"space": "a"}),
    ]
    if reihenfolge == -1:
        knoten.reverse()
        kanten.reverse()
    for k in knoten:
        g.knoten_hinzu(k)
    for e in kanten:
        g.kante_hinzu(e)
    return g


def test_schlank_behaelt_hops_aussagen_und_endpunkte():
    voll = _voll()
    s = schlank(voll)
    assert {e.art for e in s.kanten.values()} == {"hop", "belegt_durch", "verbindet"}
    assert set(s.kanten) == {i for i, e in voll.kanten.items() if e.art in ("hop", "belegt_durch", "verbindet")}
    assert set(s.knoten) == {
        "space:a", "dienst:b", "dienst:c", "kette:K1", "kette:K2",  # immer
        "komponente:client", "mensch:felix", "tabelle:marketing.x", "symbol:a:client.ziel",  # Hop-Endpunkte
        "aussage:abc", "modul:sp/a/client.py",  # Endpunkte von belegt_durch
    }
    for i, e in s.kanten.items():  # Zustand und Belege unveraendert
        assert e == voll.kanten[i]
    for i, k in s.knoten.items():
        assert k == voll.knoten[i]
    assert (s.commit_aussen, s.commit_submodul) == ("aaa", "bbb")


def test_schlank_wirft_symbolkanten_weg():
    s = schlank(_voll())
    weg_arten = {"ruft_auf", "gehoert_zu", "definiert_route", "registriert", "nutzt_dienst", "liest_schreibt"}
    assert not any(e.art in weg_arten for e in s.kanten.values())
    for nid in ("symbol:a:client.senden", "symbol:a:client.helfer", "modul:sp/a/rest.py", "route:POST /api/x@a",
                "mcp_werkzeug:w", "komponente:frei"):
        assert nid not in s.knoten
    # nur Symbol-Kanten anders -> schlanker Digest gleich
    anders = _voll()
    anders.kante_hinzu(Kante(kante_id("ruft_auf", "symbol:a:client.helfer", "symbol:a:client.senden"), "ruft_auf",
                             "symbol:a:client.helfer", "symbol:a:client.senden", "extraktor", (), {"space": "a"}))
    assert anders.digest_stabil() != _voll().digest_stabil()
    assert schlank(anders).digest_stabil() == s.digest_stabil()


def test_schlank_deterministisch(tmp_path):
    voll = _voll()
    vorher = voll.digest()
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    schlank(voll).speichern(a, ZEIT)
    schlank(voll).speichern(b, ZEIT)
    schlank(_voll(-1)).speichern(c, ZEIT)
    assert a.read_bytes() == b.read_bytes() == c.read_bytes()
    assert voll.digest() == vorher  # rein: die Eingabe bleibt unveraendert
    assert schlank(schlank(voll)).digest() == schlank(voll).digest()  # idempotent
    s = schlank(voll)
    s.kanten[next(iter(s.kanten))].attribute["neu"] = 1
    assert voll.digest() == vorher  # keine geteilten Objekte


def test_mit_schlank_repo_gewinnt_fuer_hops_aussagen_und_dienste():
    """Der volle lokale Graph liefert die Symbol-Daten, der schlanke Repo-Graph Hops, Aussagen, Dienste."""
    voll = _voll()
    repo = schlank(_voll())
    hop = next(e for e in repo.kanten.values() if e.art == "hop")
    hop.attribute["belegt_durch"] = "042"  # im Repo geaendert (andere Maschine)
    repo.knoten.pop("aussage:abc")  # Aussage im Repo entfernt (samt Endpunkt ihrer Belegkante)
    repo.knoten.pop("modul:sp/a/client.py")
    repo.kanten = {i: e for i, e in repo.kanten.items() if e.art != "belegt_durch"}
    voll.knoten_hinzu(Knoten("aussage:alt", "aussage", "lokal veraltet"))
    voll.knoten_hinzu(Knoten("dienst:veraltet", "dienst", "veraltet"))
    g = mit_schlank(voll, repo)
    assert g.kanten[hop.id].attribute["belegt_durch"] == "042"
    assert not any(k.typ == "aussage" for k in g.knoten.values())
    assert not any(e.art == "belegt_durch" for e in g.kanten.values())
    assert "dienst:veraltet" not in g.knoten
    assert "ruft_auf:symbol:a:client.senden->symbol:a:client.helfer" in g.kanten  # Symbol-Daten bleiben
    assert schlank(g).digest() == repo.digest()
    assert "aussage:alt" in voll.knoten and "dienst:veraltet" in voll.knoten  # rein: Eingaben unveraendert
    assert voll.kanten[hop.id].attribute.get("belegt_durch") is None


def test_mit_schlank_voller_repo_graph_ersetzt_keine_frischen_symbol_daten():
    """Fix-Runde 1: liegt im Repo noch ein voller (veralteter) Graph, gewinnen die frischen lokalen Symbol-Daten."""
    voll = _voll()
    repo = _voll()  # voller Repo-Graph wie der Seed 91bcc5f, aber mit aelteren Symbol-Daten
    nutzt = "nutzt_dienst:symbol:a:client.senden->dienst:b"
    repo.kanten[nutzt].attribute["pfade"] = ["/api/alt"]
    repo.knoten["symbol:a:client.senden"].attribute["zeile"] = "99"
    repo.kante_hinzu(Kante(kante_id("ruft_auf", "symbol:a:client.helfer", "symbol:a:weg"), "ruft_auf",
                           "symbol:a:client.helfer", "symbol:a:weg", "extraktor", (), {"space": "a"}))
    g = mit_schlank(voll, repo)
    assert "pfade" not in g.kanten[nutzt].attribute
    assert "zeile" not in g.knoten["symbol:a:client.senden"].attribute
    assert "ruft_auf:symbol:a:client.helfer->symbol:a:weg" not in g.kanten
    assert schlank(g).digest() == schlank(repo).digest()


ECHTER_GRAPH = REPO / "docs" / "graph" / "graph.json"


@pytest.mark.skipif(not ECHTER_GRAPH.is_file(), reason="kein docs/graph/graph.json im Repo")
def test_schlank_echter_repo_graph_unter_1_mb(tmp_path):
    """Messung am echten Repo, nur lesend (Ruling 49: Zielgroesse unter 1 MB); geschrieben wird nur nach tmp_path."""
    voll = Graph.laden(ECHTER_GRAPH)
    s = schlank(voll)
    ziel = tmp_path / "graph.json"
    s.speichern(ziel, ZEIT)
    assert ziel.stat().st_size < 1_000_000
    assert {e.art for e in s.kanten.values()} <= {"hop", "belegt_durch", "verbindet"}
    assert sum(1 for e in s.kanten.values() if e.art == "hop") == sum(1 for e in voll.kanten.values() if e.art == "hop")
