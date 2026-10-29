from pathlib import Path

import pytest
import yaml

from selbstbild.analyse.bereiche import Bereich
from selbstbild.analyse.doppelt import Doppelung
from selbstbild.analyse.ergebnis import Analyse
from selbstbild.analyse.goldsatz import kandidaten, vergleichen
from selbstbild.analyse.kolonie import KolonieErgebnis, KolonieParameter
from selbstbild.analyse.tot import TotFund


def _sym(i: int) -> str:
    return f"symbol:r:pkg/m{i}.py#f{i}"


def _analyse(n_tot: int = 60, statistik: tuple[tuple[str, int], ...] = ()) -> Analyse:
    dateien = tuple(f"datei:r:pkg/m{i}.py" for i in range(n_tot))
    kol = KolonieErgebnis((), (), (dateien[:3],), (dateien[0],), dateien[:30], KolonieParameter())
    return Analyse(
        commit="c", version=1, parameter=(),
        bereiche=(Bereich("b1", "Kern", dateien[:40], 0.5), Bereich("b2", "Rand", dateien[40:], 0.2),
                  Bereich("b3", "(isoliert)", ("datei:r:x.py",) * 1, 0.0)),
        pagerank_top=((_sym(0), 0.5), (_sym(1), 0.4)), betweenness_top=(), hotspots=(),
        tot=tuple(TotFund(_sym(i), "sicher_tot", ("g",), None) for i in range(n_tot)),
        doppelungen=(Doppelung("gleiche_datei", ("a.py", "b.py"), 0.98765),),
        kolonie=kol, luecken=(), statistik=statistik)


def test_kandidaten_deterministisch_und_valide() -> None:
    a = _analyse()
    text = kandidaten(a, n_tot=10, n_doppelt=5)
    assert text == kandidaten(a, n_tot=10, n_doppelt=5)
    doc = yaml.safe_load(text)
    assert len(doc["tot"]) == 10
    assert all(e["urteil"] == "offen" for e in doc["tot"] + doc["doppelt"])
    assert doc["doppelt"][0]["aehnlichkeit"] == "0.9877"
    assert [b["id"] for b in doc["bereiche"]] == ["b1", "b2"]
    assert doc["bereiche"][0]["erwartet"] == [] and doc["zentral"] == []
    ids = [e["symbol"] for e in doc["tot"]]
    assert ids == sorted(ids)


def _gold(ja: int, nein: int, offen: int = 0) -> str:
    urteile = ["ja"] * ja + ["nein"] * nein + ["offen"] * offen
    return yaml.safe_dump({"tot": [{"symbol": _sym(i), "urteil": u} for i, u in enumerate(urteile)],
                           "doppelt": [], "bereiche": [], "zentral": []})


def test_vergleich_praezision() -> None:
    v = vergleichen(_gold(27, 3), _analyse())
    assert v.tot.mathe == 0.9 and v.tot.n_mathe == 30
    # Kolonie flaggt Dateien 0..29 -> alle 30 beurteilten; Praezision identisch
    assert v.tot.kolonie == 0.9 and v.tot.n_kolonie == 30
    assert v.doppelt.mathe is None and v.zentrum.mathe is None and v.bereiche.mathe is None


def test_vergleich_ohne_urteile() -> None:
    v = vergleichen(_gold(0, 0, 5), _analyse())
    assert v.tot.mathe is None and v.tot.kolonie is None and v.tot.n_mathe == 0
    assert v.laufzeit_mathe_ms is None and v.laufzeit_kolonie_ms is None


def test_vergleich_bereiche_zentrum_laufzeit() -> None:
    a = _analyse(statistik=(("laufzeit_mathe_ms", 120),))
    gold = yaml.safe_dump({"tot": [], "doppelt": [{"art": "x", "mitglieder": ["a"], "urteil": "ja"}],
                           "bereiche": [{"id": "b1", "erwartet": ["pkg/m0.py", "pkg/m1.py"]}],
                           "zentral": ["datei:r:pkg/m0.py", "datei:r:pkg/m9.py"]})
    v = vergleichen(gold, a)
    assert v.doppelt.mathe == 1.0
    # erwartet: {m0, m1}; Bereich b1 hat 40
    assert v.bereiche.mathe == 2 / 40 and v.bereiche.kolonie == 2 / 3
    assert v.zentrum.mathe == 0.5 and v.zentrum.kolonie == 0.5
    assert v.laufzeit_mathe_ms == 120 and v.laufzeit_kolonie_ms is None


def test_ungueltiges_yaml_und_listenwurzel() -> None:
    with pytest.raises(ValueError, match="Goldsatz ungueltig"):
        vergleichen("tot: [unclosed", _analyse())
    with pytest.raises(ValueError, match="Goldsatz ungueltig"):
        vergleichen("- a\n- b\n", _analyse())


def test_urteile_tolerant_und_ignoriert() -> None:
    text = (f"tot:\n- {{symbol: '{_sym(0)}', urteil: Ja}}\n- {{symbol: '{_sym(1)}', urteil: yes}}\n"
            f"- {{symbol: '{_sym(2)}', urteil: JA}}\n- {{symbol: '{_sym(3)}', urteil: no}}\n"
            f"- {{symbol: '{_sym(4)}', urteil: vielleicht}}\n- {{symbol: '{_sym(5)}', urteil: ''}}\n"
            "- just a string\n- {urteil: ja}\n")
    v = vergleichen(text, _analyse())
    assert v.tot.n_mathe == 4 and v.tot.mathe == 0.75
    assert v.ignoriert == 3  # vielleicht, Nicht-Dict, ohne Symbol


def test_praefix_nur_an_pfadgrenze() -> None:
    a = _analyse()
    a = Analyse(**{**a.__dict__, "bereiche": (Bereich("b1", "M", ("datei:r:spaces/marketing/x.py",), 0.5),)})
    gold = yaml.safe_dump({"bereiche": [{"id": "b1", "erwartet": ["spaces/m"]}]})
    v = vergleichen(gold, a)
    assert v.bereiche.mathe is None and v.ignoriert == 1
    gold = yaml.safe_dump({"bereiche": [{"id": "b1", "erwartet": "spaces\\marketing/"}]})
    assert vergleichen(gold, a).bereiche.mathe == 1.0  # Skalar, Backslash, Slash am Ende


def test_unbekanntes_tot_symbol_ausgeschlossen() -> None:
    gold = yaml.safe_dump({"tot": [{"symbol": _sym(0), "urteil": "ja"},
                                   {"symbol": "symbol:r:nope.py#x", "urteil": "nein"}]})
    v = vergleichen(gold, _analyse())
    assert v.tot.mathe == 1.0 and v.tot.n_mathe == 1 and v.ignoriert == 1


def test_beispiel_yaml_ladbar() -> None:
    pfad = Path(__file__).resolve().parents[2] / "docs" / "selbstbild" / "goldsatz.beispiel.yaml"
    a = Analyse("c", 1, (), (), (), (), (), (), (), None, ())
    v = vergleichen(pfad.read_text(encoding="utf-8"), a)
    assert v.tot.mathe is None and v.doppelt.mathe == 1.0


def test_kolonie_metrik_uebereinstimmung_s12() -> None:
    """Ruling S12: (ja und tote Ecke) + (nein und keine tote Ecke), geteilt durch ja + nein."""
    gold = yaml.safe_dump({"tot": [{"symbol": _sym(i), "urteil": "ja"} for i in range(10)]
                           + [{"symbol": _sym(i), "urteil": "nein"} for i in range(30, 40)]})
    v = vergleichen(gold, _analyse())
    assert v.tot.mathe == 0.5 and v.tot.n_mathe == 20
    assert v.tot.kolonie == 1.0 and v.tot.n_kolonie == 20
    gold = yaml.safe_dump({"tot": [{"symbol": _sym(i), "urteil": "ja"} for i in range(35, 40)]})
    assert vergleichen(gold, _analyse()).tot.kolonie == 0.0  # kann jetzt verlieren


def test_nur_aktuell_sicher_tot_gewertet_s12() -> None:
    a = _analyse()
    tot = tuple(TotFund(f.symbol, "wahrscheinlich_tot", ("dekorator",), None) if f.symbol == _sym(1) else f
                for f in a.tot)
    a = Analyse(**{**a.__dict__, "tot": tot})
    gold = yaml.safe_dump({"tot": [{"symbol": _sym(0), "urteil": "ja"}, {"symbol": _sym(1), "urteil": "ja"}]})
    v = vergleichen(gold, a)
    assert v.tot.n_mathe == 1 and v.ignoriert == 1


def test_bereiche_unter_spaces_zuerst() -> None:
    """Fix-Welle: die drei groessten Bereiche unter spaces/ (auch Submodul-Repos), sonst aufgefuellt."""
    def b(i: str, name: str, n: int) -> Bereich:
        return Bereich(i, name, tuple(f"datei:r:{i}/{k}.py" for k in range(n)), 0.5)

    a = _analyse()
    a = Analyse(**{**a.__dict__, "bereiche": (
        b("bereich:1", "(isoliert)", 90), b("bereich:2", "v/os/hermes/tests", 80),
        b("bereich:3", "v/os/spaces/captain_cook/minibook", 70), b("bereich:4", "v/os/brain", 60),
        b("bereich:5", "v/os/spaces/marketing/claw", 50), b("bereich:6", "v/os/spaces", 40))})
    doc = yaml.safe_load(kandidaten(a))
    assert [x["id"] for x in doc["bereiche"]] == ["bereich:2", "bereich:3", "bereich:5"]
    assert {x["name"] for x in doc["bereiche"]} == {"v/os/spaces/captain_cook/minibook", "v/os/spaces/marketing/claw",
                                                     "v/os/hermes/tests"}


def test_zentrum_top50_sind_dateien() -> None:
    """Top-50 meint 50 DATEIEN: viele Symbole derselben Datei verdraengen keine andere Datei."""
    a = _analyse()
    top = tuple((f"symbol:r:pkg/m0.py#s{k}", 1.0 - k / 1000) for k in range(60)) + ((_sym(5), 0.001),)
    a = Analyse(**{**a.__dict__, "pagerank_top": top})
    gold = yaml.safe_dump({"zentral": ["datei:r:pkg/m5.py"]})
    assert vergleichen(gold, a).zentrum.mathe == 1.0
