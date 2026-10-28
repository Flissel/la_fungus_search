from __future__ import annotations

import random
import time

import pytest

from selbstbild.analyse.lader import laden, symbolgraph
from selbstbild.analyse.zentral import Hotspot, _perzentile, betweenness, hotspots, pagerank
from selbstbild.typen import Kante, Knoten, id_datei, id_symbol


def _d(pfad: str, leben: str | None = None) -> Knoten:
    attr = () if leben is None else (("lebendigkeit", leben),)
    return Knoten(id_datei("r", pfad), "datei", pfad, None, attr)


def _s(pfad: str, name: str) -> Knoten:
    return Knoten(id_symbol("r", pfad, name), "symbol", name, None, ())


def _ruft(a: str, b: str) -> Kante:
    return Kante("ruft_auf", a, b, True, None)


def _imp(a: str, b: str) -> Kante:
    return Kante("importiert", id_datei("r", a), id_datei("r", b), True, None)


def test_pagerank_stern(graph_stand):
    ziel = _s("zentral.py", "f")
    knoten = [_d("zentral.py"), ziel]
    kanten = []
    for i in range(5):
        s = _s(f"a{i}.py", "g")
        knoten += [_d(f"a{i}.py"), s]
        kanten.append(_ruft(s.id, ziel.id))
    pr = pagerank(laden(graph_stand(knoten, kanten)))
    symbole = {i: v for i, v in pr.items() if i.startswith("symbol:")}
    assert max(symbole, key=symbole.get) == ziel.id
    assert len(pr) == len(knoten)


def test_pagerank_zyklus_stabil(graph_stand):
    a, b = _s("a.py", "f"), _s("b.py", "g")
    d = laden(graph_stand([a, b], [_ruft(a.id, b.id), _ruft(b.id, a.id)]))
    pr = pagerank(d)
    assert all(0.0 < v < 1.0 for v in pr.values())
    assert sum(pr.values()) == pytest.approx(1.0, abs=1e-6)
    assert pr == pagerank(d)


def test_betweenness_bruecke(graph_stand):
    gruppe1, gruppe2 = ["a1.py", "a2.py", "a3.py"], ["b1.py", "b2.py", "b3.py"]
    knoten = [_d(p) for p in gruppe1 + gruppe2 + ["bruecke.py"]]
    kanten = [_imp(p, "bruecke.py") for p in gruppe1] + [_imp("bruecke.py", p) for p in gruppe2]
    bt = betweenness(laden(graph_stand(knoten, kanten)))
    assert max(bt, key=bt.get) == id_datei("r", "bruecke.py")
    assert bt[id_datei("r", "bruecke.py")] == 1.0
    assert all(0.0 <= v <= 1.0 for v in bt.values())


def test_betweenness_zu_klein(graph_stand):
    knoten = [_d("a.py"), _d("b.py")]
    bt = betweenness(laden(graph_stand(knoten, [_imp("a.py", "b.py")])))
    assert bt == {id_datei("r", "a.py"): 0.0, id_datei("r", "b.py"): 0.0}


def test_hotspot_braucht_alle_drei(graph_stand):
    # zentral.py: zentral, aber unveraendert; heiss.py: zentral, oft geaendert, gross.
    knoten = [_d("zentral.py", "0.0000"), _d("heiss.py", "5.0000"), _d("rand.py", "1.0000")]
    kanten = []
    for pfad in ("zentral.py", "heiss.py"):
        for i in range(3):
            s = _s(pfad, f"f{i}")
            knoten.append(s)
    knoten += [_s("rand.py", "x")]
    d = laden(graph_stand(knoten, kanten))
    pr = {k: 0.01 for k in d.knoten}
    pr[id_datei("r", "zentral.py")] = 0.5
    pr[id_datei("r", "heiss.py")] = 0.4
    erg = hotspots(d, pr)
    wert = {h.datei: h.wert for h in erg}
    assert erg[0].datei == id_datei("r", "heiss.py")
    assert wert[id_datei("r", "zentral.py")] == 0.0 < wert[id_datei("r", "heiss.py")]
    assert [n for n, _ in erg[0].teile] == ["aenderung", "groesse", "zentral"]
    assert all(0.0 <= h.wert <= 1.0 for h in erg)


def test_hotspot_perzentil_gleichstand_mittlerer_rang(graph_stand):
    knoten = [_d("a.py", "1.0"), _d("b.py", "1.0"), _d("c.py", "2.0")]
    d = laden(graph_stand(knoten, []))
    erg = {h.datei: dict(h.teile) for h in hotspots(d, {})}
    assert erg[id_datei("r", "a.py")]["aenderung"] == 0.25
    assert erg[id_datei("r", "b.py")]["aenderung"] == 0.25
    assert erg[id_datei("r", "c.py")]["aenderung"] == 1.0


def test_hotspot_top_und_ohne_lebendigkeit(graph_stand):
    knoten = [_d("a.py", "1.0"), _d("b.py"), _d("c.py", "3.0")]
    d = laden(graph_stand(knoten, []))
    erg = hotspots(d, {}, top=2)
    assert len(erg) == 2
    assert id_datei("r", "b.py") not in [h.datei for h in erg]
    alle = {h.datei: dict(h.teile) for h in hotspots(d, {})}
    assert alle[id_datei("r", "b.py")]["aenderung"] == 0.0
    assert isinstance(erg[0], Hotspot)


def test_leer(graph_stand):
    d = laden(graph_stand([], []))
    assert pagerank(d) == {}
    assert betweenness(d) == {}
    assert hotspots(d, {}) == ()
    assert symbolgraph(d).ids == ()


def test_perzentile_schnell_und_wie_referenz():
    rng = random.Random(20261028)
    werte = {f"k{i}": float(rng.randint(0, 800)) for i in range(5000)}  # viele Gleichstaende
    t0 = time.perf_counter()
    erg = _perzentile(werte)
    assert time.perf_counter() - t0 < 1.0
    n = len(werte)
    for k in list(werte)[:50]:
        w = werte[k]
        kleiner = sum(1 for x in werte.values() if x < w)
        gleich = sum(1 for x in werte.values() if x == w)
        assert erg[k] == (kleiner + (gleich - 1) / 2) / (n - 1)


def test_lebendigkeit_nicht_endlich_ist_null(graph_stand):
    knoten = [_d("a.py", "nan"), _d("b.py", "inf"), _d("c.py", "2.0")]
    d = laden(graph_stand(knoten, []))
    erg = {h.datei: dict(h.teile)["aenderung"] for h in hotspots(d, {})}
    assert erg[id_datei("r", "a.py")] == erg[id_datei("r", "b.py")] == 0.25
    assert erg[id_datei("r", "c.py")] == 1.0


def test_top_nicht_positiv(graph_stand):
    d = laden(graph_stand([_d("a.py", "1.0")], []))
    assert hotspots(d, {}, top=0) == ()
    assert hotspots(d, {}, top=-3) == ()
