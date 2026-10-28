from __future__ import annotations

import time

from selbstbild.analyse.kolonie import KolonieErgebnis, KolonieParameter, kolonie_laufen
from selbstbild.analyse.lader import GraphDaten
from selbstbild.typen import Kante, Knoten, id_datei


def _datei(pfad: str, wurzel: bool = False) -> Knoten:
    attr = (("einstieg", "main"),) if wurzel else ()
    return Knoten(id_datei("r", pfad), "datei", pfad, None, attr)


def _imp(a: str, b: str) -> Kante:
    return Kante("importiert", id_datei("r", a), id_datei("r", b), True, None)


def _daten(dateien: list[Knoten], kanten: list[Kante]) -> GraphDaten:
    return GraphDaten("c", 0, {k.id: k for k in dateien}, tuple(kanten), {})


def _stern(n: int = 9) -> GraphDaten:
    nabe = ["nabe.py"] + [f"s{i}.py" for i in range(n)]
    kn = [_datei("start.py", True)] + [_datei(p) for p in nabe]
    ka = [_imp("start.py", "nabe.py")]
    for i in range(n):
        ka += [_imp("nabe.py", f"s{i}.py"), _imp(f"s{i}.py", "nabe.py")]
    return _daten(kn, ka)


def test_deterministisch() -> None:
    d = _stern()
    p = KolonieParameter(agenten=16, schritte=40)
    assert kolonie_laufen(d, p) == kolonie_laufen(d, p)


def test_unerreichbares_nie_besucht() -> None:
    d = _daten([_datei("a.py", True), _datei("b.py"), _datei("x.py"), _datei("y.py")],
               [_imp("a.py", "b.py"), _imp("b.py", "a.py"), _imp("x.py", "y.py")])
    e = kolonie_laufen(d, KolonieParameter(agenten=8, schritte=30))
    assert e.tote_ecken == (id_datei("r", "x.py"), id_datei("r", "y.py"))
    assert all(i not in dict(e.besuche) for i in e.tote_ecken)


def test_stern_zentrum() -> None:
    e = kolonie_laufen(_stern(), KolonieParameter(agenten=16, schritte=60))
    assert e.zentrum[0] == id_datei("r", "nabe.py")
    assert isinstance(e, KolonieErgebnis) and e.spuren and e.bereiche


def test_klein_schnell() -> None:
    kn = [_datei("f0.py", True)] + [_datei(f"f{i}.py") for i in range(1, 10)]
    ka = [_imp(f"f{i}.py", f"f{(i + 1) % 10}.py") for i in range(10)]
    t = time.perf_counter()
    kolonie_laufen(_daten(kn, ka))
    assert time.perf_counter() - t < 2.0


def test_ohne_wurzeln_leer() -> None:
    e = kolonie_laufen(_daten([_datei("a.py"), _datei("b.py")], [_imp("a.py", "b.py")]))
    assert e.besuche == e.spuren == e.bereiche == e.zentrum == ()
