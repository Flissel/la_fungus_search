from __future__ import annotations

import pytest

from selbstbild.analyse import ist_testdatei
from selbstbild.analyse.erreichbar import erreichbarkeit, wurzeln
from selbstbild.analyse.lader import laden
from selbstbild.typen import Kante, Knoten, id_datei, id_symbol


def _s(pfad: str, q: str, art: str = "funktion", einstieg: str | None = None) -> Knoten:
    attr = [("symbolart", art)]
    if einstieg:
        attr.append(("einstieg", einstieg))
    return Knoten(id_symbol("r", pfad, q), "symbol", q, None, tuple(sorted(attr)))


def _i(pfad: str, q: str) -> str:
    return id_symbol("r", pfad, q)


def _d(pfad: str, einstieg: str | None = None) -> Knoten:
    attr = (("einstieg", einstieg),) if einstieg else ()
    return Knoten(id_datei("r", pfad), "datei", pfad, None, attr)


def _ruft(a: str, b: str, sicher: bool = True) -> Kante:
    return Kante("ruft_auf", a, b, sicher, None)


def _bauen(graph_stand, knoten, kanten):
    return laden(graph_stand(knoten, kanten))


@pytest.mark.parametrize("pfad,erwartet", [
    ("test_a.py", True), ("x/a_test.py", True), ("x/a_test.go", True),
    ("web/a.test.ts", True), ("web/a.test.tsx", True), ("a.test.js", True),
    ("a.spec.ts", True), ("pkg/tests/helfer.py", True), ("web/__tests__/x.ts", True),
    (r"pkg\tests\h.py", True), ("a.py", False), ("pkg/testing.py", False),
    ("contest.py", False), ("tests", False),
])
def test_ist_testdatei(pfad, erwartet):
    assert ist_testdatei(pfad) is erwartet


def test_kette_von_main(graph_stand):
    d = _bauen(graph_stand,
               [_d("a.py"), _s("a.py", "main", einstieg="main"), _s("a.py", "a"), _s("a.py", "b"), _s("a.py", "c")],
               [_ruft(_i("a.py", "main"), _i("a.py", "a")), _ruft(_i("a.py", "a"), _i("a.py", "b"))])
    e = erreichbarkeit(d)
    assert _i("a.py", "b") in e.sicher
    assert _i("a.py", "c") not in e.sicher and _i("a.py", "c") not in e.irgendwie
    assert _i("a.py", "main") in e.wurzeln
    assert id_datei("r", "a.py") in e.sicher  # Symbol erreicht => Datei erreicht


def test_unsicher_nur_irgendwie(graph_stand):
    d = _bauen(graph_stand, [_s("a.py", "main", einstieg="main"), _s("a.py", "x")],
               [_ruft(_i("a.py", "main"), _i("a.py", "x"), sicher=False)])
    e = erreichbarkeit(d)
    assert _i("a.py", "x") in e.irgendwie and _i("a.py", "x") not in e.sicher


def test_nur_test(graph_stand):
    d = _bauen(graph_stand,
               [_s("test_a.py", "test_a", einstieg="test"), _s("h.py", "h"), _s("h.py", "p", einstieg="main")],
               [_ruft(_i("test_a.py", "test_a"), _i("h.py", "h"))])
    e = erreichbarkeit(d)
    assert _i("h.py", "h") in e.nur_test
    assert _i("h.py", "h") not in e.irgendwie
    assert _i("h.py", "p") not in e.nur_test


def test_testdatei_symbole_sind_testwurzeln(graph_stand):
    d = _bauen(graph_stand, [_d("tests/t.py"), _s("tests/t.py", "helfer"), _s("h.py", "h")],
               [_ruft(_i("tests/t.py", "helfer"), _i("h.py", "h"))])
    prod, test = wurzeln(d)
    assert _i("tests/t.py", "helfer") in test and not prod
    assert _i("h.py", "h") in erreichbarkeit(d).nur_test


def test_methode_folgt_klasse(graph_stand):
    d = _bauen(graph_stand,
               [_s("a.py", "main", einstieg="main"), _s("a.py", "K", "klasse"), _s("a.py", "K.m", "methode"),
                _s("a.py", "L", "klasse"), _s("a.py", "L.m", "methode"), _s("a.py", "K.m.inner")],
               [_ruft(_i("a.py", "main"), _i("a.py", "K"))])
    e = erreichbarkeit(d)
    assert _i("a.py", "K.m") in e.sicher
    assert _i("a.py", "L.m") not in e.irgendwie
    assert _i("a.py", "K.m.inner") in e.sicher  # Verschachteltes folgt dem Umschliessenden


def test_methode_unsichere_klasse(graph_stand):
    d = _bauen(graph_stand,
               [_s("a.py", "main", einstieg="main"), _s("a.py", "K", "klasse"), _s("a.py", "K.m", "methode")],
               [_ruft(_i("a.py", "main"), _i("a.py", "K"), sicher=False)])
    e = erreichbarkeit(d)
    assert _i("a.py", "K.m") in e.irgendwie and _i("a.py", "K.m") not in e.sicher


def test_zyklus_endet(graph_stand):
    d = _bauen(graph_stand, [_s("a.py", "a"), _s("a.py", "b")],
               [_ruft(_i("a.py", "a"), _i("a.py", "b")), _ruft(_i("a.py", "b"), _i("a.py", "a"))])
    e = erreichbarkeit(d)
    assert not e.irgendwie and not e.sicher and not e.wurzeln


def test_zyklus_mit_wurzel(graph_stand):
    d = _bauen(graph_stand, [_s("a.py", "a", einstieg="main"), _s("a.py", "b")],
               [_ruft(_i("a.py", "a"), _i("a.py", "b")), _ruft(_i("a.py", "b"), _i("a.py", "a")),
                _ruft(_i("a.py", "a"), _i("a.py", "a"))])
    assert _i("a.py", "b") in erreichbarkeit(d).sicher


def test_regel_einstieg(graph_stand):
    d = _bauen(graph_stand, [_s("a.py", "r", einstieg="regel"), _s("a.py", "z")],
               [_ruft(_i("a.py", "r"), _i("a.py", "z"))])
    assert _i("a.py", "r") in wurzeln(d)[0]
    assert _i("a.py", "z") in erreichbarkeit(d).sicher


def test_gemischter_einstieg(graph_stand):
    d = _bauen(graph_stand, [_s("a.py", "m", einstieg="main,test")], [])
    prod, test = wurzeln(d)
    assert _i("a.py", "m") in prod and _i("a.py", "m") in test


def test_route_und_registriert_quellen(graph_stand):
    d = _bauen(graph_stand, [_s("a.py", "reg"), _s("a.py", "h"), _s("b.py", "q"), _s("b.py", "z")],
               [Kante("definiert_route", _i("a.py", "reg"), _i("a.py", "h"), True, None),
                Kante("registriert", _i("b.py", "q"), _i("b.py", "z"), True, None)])
    prod, _ = wurzeln(d)
    assert {_i("a.py", "reg"), _i("b.py", "q")} <= prod
    assert _i("a.py", "h") in erreichbarkeit(d).sicher


def test_datei_modulebene_nicht_funktionen(graph_stand):
    d = _bauen(graph_stand,
               [_d("m.py", einstieg="cli_skript"), _d("n.py"), _s("m.py", "f"), _s("n.py", "g"), _s("n.py", "unbenutzt")],
               [_ruft(id_datei("r", "m.py"), _i("n.py", "g"))])
    e = erreichbarkeit(d)
    assert id_datei("r", "m.py") in e.wurzeln
    assert _i("n.py", "g") in e.sicher and id_datei("r", "n.py") in e.sicher
    assert _i("m.py", "f") not in e.irgendwie and _i("n.py", "unbenutzt") not in e.irgendwie


def test_luecke_kante_ohne_knoten(graph_stand):
    d = _bauen(graph_stand, [_s("a.py", "main", einstieg="main")],
               [_ruft(_i("a.py", "main"), "symbol:r:fehlt.py#x")])
    assert _i("a.py", "main") in erreichbarkeit(d).sicher


def test_leer(graph_stand):
    e = erreichbarkeit(_bauen(graph_stand, [], []))
    assert not (e.sicher or e.irgendwie or e.nur_test or e.wurzeln)


def test_dekorator_wrapper_folgt_umschliessendem(graph_stand):
    d = _bauen(graph_stand,
               [_s("a.py", "main", einstieg="main"), _s("a.py", "deko"), _s("a.py", "deko.wrapper"),
                _s("a.py", "lose"), _s("a.py", "lose.wrapper")],
               [_ruft(_i("a.py", "main"), _i("a.py", "deko"), sicher=False)])
    e = erreichbarkeit(d)
    assert _i("a.py", "deko.wrapper") in e.irgendwie and _i("a.py", "deko.wrapper") not in e.sicher
    assert _i("a.py", "lose.wrapper") not in e.irgendwie


def test_paket_init_laeuft_mit(graph_stand):
    d = _bauen(graph_stand,
               [_s("a/b/c.py", "main", einstieg="main"), _d("a/b/c.py"), _d("a/__init__.py"), _d("a/b/__init__.py"),
                _s("a/__init__.py", "setup"), _s("a/b/__init__.py", "reg"), _s("a/x.py", "x"),
                _d("z/__init__.py"), _s("z/__init__.py", "fremd")],
               [_ruft(id_datei("r", "a/__init__.py"), _i("a/__init__.py", "setup")),
                _ruft(id_datei("r", "a/b/__init__.py"), _i("a/b/__init__.py", "reg")),
                _ruft(id_datei("r", "z/__init__.py"), _i("z/__init__.py", "fremd"))])
    e = erreichbarkeit(d)
    assert _i("a/__init__.py", "setup") in e.sicher and _i("a/b/__init__.py", "reg") in e.sicher
    assert _i("z/__init__.py", "fremd") not in e.irgendwie
    assert _i("a/x.py", "x") not in e.irgendwie


def test_importkette_zu_modulaufruf(graph_stand):
    d = _bauen(graph_stand,
               [_d("a.py", einstieg="cli_skript"), _d("b.py"), _s("b.py", "ziel"), _s("b.py", "tot")],
               [Kante("importiert", id_datei("r", "a.py"), id_datei("r", "b.py"), True, None),
                _ruft(id_datei("r", "b.py"), _i("b.py", "ziel"))])
    e = erreichbarkeit(d)
    assert _i("b.py", "ziel") in e.sicher and _i("b.py", "tot") not in e.irgendwie


def test_sicher_symbol_ueber_datei_zu_modulaufruf(graph_stand):
    d = _bauen(graph_stand,
               [_s("a.py", "main", einstieg="main"), _s("b.py", "f"), _d("b.py"), _s("b.py", "ziel")],
               [_ruft(_i("a.py", "main"), _i("b.py", "f")),
                _ruft(id_datei("r", "b.py"), _i("b.py", "ziel"))])
    assert _i("b.py", "ziel") in erreichbarkeit(d).sicher


def test_sicher_teilmenge_irgendwie(graph_stand):
    d = _bauen(graph_stand,
               [_s("a.py", "main", einstieg="main"), _s("a.py", "k", "klasse"), _s("a.py", "k.m", "methode"),
                _s("a.py", "x"), _s("a.py", "y")],
               [_ruft(_i("a.py", "main"), _i("a.py", "k")), _ruft(_i("a.py", "main"), _i("a.py", "x"), False),
                _ruft(_i("a.py", "x"), _i("a.py", "y"))])
    e = erreichbarkeit(d)
    assert e.sicher <= e.irgendwie and e.sicher != e.irgendwie
