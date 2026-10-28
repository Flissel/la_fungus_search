from __future__ import annotations

import pytest

from selbstbild.analyse.bereiche import Bereich, bereiche_finden
from selbstbild.analyse.lader import laden
from selbstbild.typen import Kante, Knoten, id_datei


def _d(pfad: str, repo: str = "r") -> Knoten:
    return Knoten(id_datei(repo, pfad), "datei", pfad, None, ())


def _imp(a: str, b: str, repo: str = "r") -> Kante:
    return Kante("importiert", id_datei(repo, a), id_datei(repo, b), True, None)


def _zwei_cluster(rueckwaerts: bool = False):
    gruppe_a = [f"spaces/marketing/a{i}.py" for i in range(4)]
    gruppe_b = [f"spaces/learning/b{i}.py" for i in range(4)]
    kanten = []
    for gruppe in (gruppe_a, gruppe_b):
        for i, x in enumerate(gruppe):
            for y in gruppe[i + 1:]:
                kanten.append(_imp(x, y))
    kanten.append(_imp(gruppe_a[0], gruppe_b[0]))
    knoten = [_d(p) for p in gruppe_a + gruppe_b]
    if rueckwaerts:
        knoten.reverse()
        kanten.reverse()
    return gruppe_a, gruppe_b, knoten, kanten


def test_zwei_cluster(graph_stand):
    a, b, knoten, kanten = _zwei_cluster()
    erg = bereiche_finden(laden(graph_stand(knoten, kanten)))
    assert len(erg) == 2
    mengen = {frozenset(x.dateien) for x in erg}
    assert mengen == {frozenset(id_datei("r", p) for p in a), frozenset(id_datei("r", p) for p in b)}
    assert [x.id for x in erg] == ["bereich:1", "bereich:2"]
    assert all(0.0 < x.kohaesion < 1.0 for x in erg)
    assert all(x.dateien == tuple(sorted(x.dateien)) for x in erg)
    # Gleichstand der Groesse: kleinste Datei-Id zuerst (spaces/learning < spaces/marketing)
    assert erg[0].name == "spaces/learning" and erg[1].name == "spaces/marketing"


def test_deterministisch(graph_stand):
    _, _, k1, e1 = _zwei_cluster()
    _, _, k2, e2 = _zwei_cluster(rueckwaerts=True)
    r1 = bereiche_finden(laden(graph_stand(k1, e1)))
    r1b = bereiche_finden(laden(graph_stand(k1, e1)))
    r2 = bereiche_finden(laden(graph_stand(k2, e2)))
    assert r1 == r1b == r2


def test_leer_und_einzeln(graph_stand):
    assert bereiche_finden(laden(graph_stand([], []))) == ()
    erg = bereiche_finden(laden(graph_stand([_d("a.py")], [])))
    assert len(erg) == 1
    assert erg[0] == Bereich("bereich:1", "(isoliert)", (id_datei("r", "a.py"),), 0.0)


def test_isolierte_in_einem_bereich(graph_stand):
    knoten = [_d("x/a.py"), _d("x/b.py"), _d("z1.py"), _d("z2.py"), _d("z3.py")]
    erg = bereiche_finden(laden(graph_stand(knoten, [_imp("x/a.py", "x/b.py")])))
    iso = [b for b in erg if b.name == "(isoliert)"]
    assert len(iso) == 1 and len(iso[0].dateien) == 3
    assert len(erg) == 2 and erg[0].name == "(isoliert)"  # groesser -> bereich:1


def test_name_praefix(graph_stand):
    knoten = [_d("spaces/marketing/api/a.py"), _d("spaces/marketing/b.py"), _d("spaces/marketing/ui/c.py")]
    kanten = [_imp("spaces/marketing/api/a.py", "spaces/marketing/b.py"),
              _imp("spaces/marketing/b.py", "spaces/marketing/ui/c.py")]
    erg = bereiche_finden(laden(graph_stand(knoten, kanten)))
    assert [b.name for b in erg] == ["spaces/marketing"]
    assert erg[0].kohaesion == 1.0


def test_name_haeufigster_ordner(graph_stand):
    knoten = [_d("a/x.py"), _d("a/y.py"), _d("b/z.py")]
    kanten = [_imp("a/x.py", "a/y.py"), _imp("a/y.py", "b/z.py")]
    erg = bereiche_finden(laden(graph_stand(knoten, kanten)))
    assert [b.name for b in erg] == ["a"]


def test_name_ueber_repos(graph_stand):
    knoten = [_d("pkg/x.py", "r1"), _d("pkg/y.py", "r2")]
    kanten = [Kante("importiert", id_datei("r1", "pkg/x.py"), id_datei("r2", "pkg/y.py"), True, None)]
    erg = bereiche_finden(laden(graph_stand(knoten, kanten)))
    assert len(erg) == 1 and erg[0].name in ("r1/pkg", "r2/pkg")


def test_kohaesion_wert(graph_stand):
    # a-b (intern, Gewicht 1) + b-c (abgehend, 1): a,b zusammen -> 1/(1+1)
    knoten = [_d("a.py"), _d("b.py"), _d("c.py")]
    kanten = [_imp("a.py", "b.py"), _imp("b.py", "c.py")]
    erg = bereiche_finden(laden(graph_stand(knoten, kanten)), aufloesung=1.0)
    for b in erg:
        assert 0.0 <= b.kohaesion <= 1.0
        assert b.kohaesion == round(b.kohaesion, 4)
