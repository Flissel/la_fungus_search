from __future__ import annotations

import os
from pathlib import Path

import pytest

from selbstbild.analyse.lader import dateigraph, datei_von, laden, symbolgraph
from selbstbild.speicher import Speicher
from selbstbild.typen import Kante, Knoten, id_datei, id_symbol

A, B = id_datei("r", "a.py"), id_datei("r", "b.py")
SA, SB = id_symbol("r", "a.py", "f"), id_symbol("r", "b.py", "g")


def _knoten() -> list[Knoten]:
    return [Knoten(A, "datei", "a.py", None), Knoten(B, "datei", "b.py", None),
            Knoten(SA, "symbol", "f", None), Knoten(SB, "symbol", "g", None)]


def test_laden_nur_lesend(graph_stand):
    ordner = graph_stand(_knoten(), [Kante("ruft_auf", SA, SB, True, None)])
    db = ordner / "graph.sqlite"
    vorher = os.stat(db).st_mtime_ns
    d = laden(ordner)
    assert d.commit == "c1" and d.commit_zeit == 1_700_000_000
    assert set(d.knoten) == {A, B, SA, SB} and len(d.kanten) == 1
    assert os.stat(db).st_mtime_ns == vorher
    with Speicher(db) as s:  # Schreiber blockiert nicht
        s.graph_ersetzen("c2", _knoten(), [])
    assert laden(ordner).commit == "c2"


def test_dateigraph_gewichte(graph_stand):
    kanten = [Kante("ruft_auf", SA, SB, True, None), Kante("ruft_auf", SA, SB, True, None),
              Kante("ruft_auf", A, SB, False, None),
              Kante("ruft_auf", SA, id_symbol("r", "a.py", "h"), True, None)]
    kn = _knoten() + [Knoten(id_symbol("r", "a.py", "h"), "symbol", "h", None)]
    # zwei verschiedene sichere Kanten brauchen verschiedene Belege
    kanten[1] = Kante("ruft_auf", id_symbol("r", "a.py", "k"), SB, True, None)
    kn.append(Knoten(id_symbol("r", "a.py", "k"), "symbol", "k", None))
    s = dateigraph(laden(graph_stand(kn, kanten)))
    assert s.ids == tuple(sorted(s.ids))
    g = s.graph
    e = g.get_eid(s.index[A], s.index[B])
    assert g.es[e]["weight"] == 2.5
    assert not any(x.source == x.target for x in g.es)
    assert g.ecount() == 1


def test_symbolgraph_und_import(graph_stand):
    s = symbolgraph(laden(graph_stand(_knoten(), [Kante("importiert", A, B, False, None)])))
    assert s.graph.es[0]["weight"] == 0.5 and s.graph.vcount() == 4
    assert datei_von(SA) == A


def test_luecke_gezaehlt(graph_stand):
    d = laden(graph_stand(_knoten(), [Kante("ruft_auf", SA, "symbol:r:fehlt.py#x", True, None)]))
    assert d.luecken["kante_ohne_knoten"] == 1
    assert dateigraph(d).graph.ecount() == 0


def test_nicht_vermessen(tmp_path: Path):
    with pytest.raises(RuntimeError, match="vermessen"):
        laden(tmp_path)


def test_datei_von_mit_raute_im_pfad():
    assert datei_von(id_symbol("r", "a#b.py", "K.m")) == id_datei("r", "a#b.py")


def test_dateigraph_symbol_in_pfad_mit_raute(graph_stand):
    f, s = id_datei("r", "a#b.py"), id_symbol("r", "a#b.py", "f")
    kn = [Knoten(f, "datei", "a#b.py", None), Knoten(s, "symbol", "f", None),
          Knoten(B, "datei", "b.py", None), Knoten(SB, "symbol", "g", None)]
    sicht = dateigraph(laden(graph_stand(kn, [Kante("ruft_auf", s, SB, True, None)])))
    assert sicht.graph.get_eid(sicht.index[f], sicht.index[B]) >= 0


def test_symbol_ohne_datei_gezaehlt(graph_stand):
    kn = [Knoten(B, "datei", "b.py", None), Knoten(SB, "symbol", "g", None),
          Knoten(SA, "symbol", "f", None)]  # Datei a.py fehlt
    d = laden(graph_stand(kn, [Kante("ruft_auf", SA, SB, True, None)]))
    assert dateigraph(d).graph.ecount() == 0
    assert d.luecken["symbol_ohne_datei"] == 1
