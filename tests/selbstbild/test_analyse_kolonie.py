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


def test_sackgasse_neustart_rundlaufend_s11() -> None:
    """Ruling S11: Ein Agent in einer Sackgasse startet auf der naechsten Wurzel, nicht auf seiner eigenen."""
    kn = [_datei(f"w{i}.py", True) for i in range(3)] + [_datei(f"s{i}.py") for i in range(3)]
    ka = [_imp(f"w{i}.py", f"s{i}.py") for i in range(3)]
    e = kolonie_laufen(_daten(kn, ka), KolonieParameter(agenten=1, schritte=12))
    assert e.tote_ecken == ()
    besuche = dict(e.besuche)
    assert all(besuche[id_datei("r", f"w{i}.py")] >= 2 for i in range(3))


def test_mehr_wurzeln_als_agenten_alle_besucht_s11() -> None:
    """300 Wurzeln, 192 Agenten: Start gestreut, Neustart rundlaufend ueber alle Wurzeln."""
    kn = [_datei(f"w{i:03}.py", True) for i in range(300)] + [_datei("nabe.py")]
    ka = [_imp(f"w{i:03}.py", "nabe.py") for i in range(300)]
    e = kolonie_laufen(_daten(kn, ka))
    wurzeln = {id_datei("r", f"w{i:03}.py") for i in range(300)}
    assert not wurzeln & set(e.tote_ecken)


def test_bereiche_inhalt() -> None:
    kn = [_datei("start.py", True), _datei("a.py"), _datei("b.py"), _datei("x.py"), _datei("y.py")]
    ka = [_imp("start.py", "a.py"), _imp("a.py", "b.py"), _imp("b.py", "start.py"), _imp("x.py", "y.py")]
    e = kolonie_laufen(_daten(kn, ka), KolonieParameter(agenten=6, schritte=60))
    assert e.bereiche == ((id_datei("r", "a.py"), id_datei("r", "b.py"), id_datei("r", "start.py")),)


def test_spuren_gekappt(monkeypatch) -> None:
    from selbstbild.analyse import kolonie

    monkeypatch.setattr(kolonie, "_SPUREN_MAX", 3)
    e = kolonie_laufen(_stern(9), KolonieParameter(agenten=16, schritte=60))
    assert len(e.spuren) == 3
    assert [w for _, _, w in e.spuren] == sorted((w for _, _, w in e.spuren), reverse=True)


def test_2000_dateien_unter_10s() -> None:
    import random

    rnd = random.Random(7)
    kn = [_datei(f"d{i:04}.py", i % 40 == 0) for i in range(2000)]
    ka = sorted({(rnd.randrange(2000), rnd.randrange(2000)) for _ in range(8000)})
    kanten = [_imp(f"d{a:04}.py", f"d{b:04}.py") for a, b in ka if a != b]
    t = time.perf_counter()
    e = kolonie_laufen(_daten(kn, kanten))
    assert time.perf_counter() - t < 10.0
    assert len(e.besuche) > 1000


def test_budget_schritte_s11() -> None:
    from selbstbild.analyse.kolonie import budget_schritte

    assert budget_schritte(19608) == 1022     # ceil(10 * 19608 / 192)
    assert budget_schritte(10) == 1
    assert budget_schritte(10**6) == 5000     # gedeckelt
    assert budget_schritte(0) == 1
