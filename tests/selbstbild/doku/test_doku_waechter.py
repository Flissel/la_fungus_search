"""doku_loop: Waechter-Zustand -> gemessen."""

import json
import sys
from pathlib import Path

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku.graph import Graph, Kante, hop_zustand_neu  # noqa: E402
from selbstbild.doku.waechter import Messung, gemessen_laden, hysterese  # noqa: E402


def _datei(tmp_path, stand):
    p = tmp_path / "z.json"
    p.write_text(json.dumps({"stand": stand, "zuletzt_gemeldet": "2026-10-27T14:25:01+00:00"}),
                 encoding="utf-8")
    return p


def test_gemessen_bricht_gewinnt(tmp_path):
    p = _datei(tmp_path, {"a": True, "b": False, "c": True})
    res, unbekannt = gemessen_laden(p, {"a": ["K1"], "b": ["K1", "K2"], "c": ["K2", "K3"]})
    assert res["K1"].wert == "bricht" and res["K2"].wert == "bricht" and res["K3"].wert == "laeuft"
    assert unbekannt == []
    assert res["K1"].zeit.endswith("+00:00")


def test_gemessen_keine_probe(tmp_path):
    p = _datei(tmp_path, {"x": True, "neu": True})
    res, unbekannt = gemessen_laden(p, {"x": ["K1"], "fehlt": ["K9"]}, alle_ketten=["K1", "K5"])
    assert res["K1"].wert == "laeuft"
    assert res["K9"].wert == "keine_probe" and res["K5"].wert == "keine_probe"
    assert unbekannt == ["neu"]
    res2, _ = gemessen_laden(p, {"x": ["K1"]})
    assert "K5" not in res2


def test_gemessen_datei_fehlt_oder_kaputt(tmp_path):
    res, u = gemessen_laden(tmp_path / "nix.json", {"x": ["K1"]}, alle_ketten=["K2"])
    assert res == {"K1": Messung("keine_probe", None), "K2": Messung("keine_probe", None)} and u == []
    kaputt = tmp_path / "k.json"
    kaputt.write_text("{nicht json", encoding="utf-8")
    res, _ = gemessen_laden(kaputt, {"x": ["K1"]})
    assert res["K1"] == Messung("keine_probe", None)


def test_gemessen_echte_zustandsdatei_form(tmp_path):
    stand = {"laura": True, "laura-dienste": True, "bruecken": True, "rowboat-index": True,
             "katalog": True, "marketing": True, "marketing-waechter": True,
             "openfang-agent": True, "issue_weg": True, "auftrags-ausfuehrer": True}
    p = tmp_path / "ketten-waechter-zustand.json"
    p.write_text(json.dumps({"stand": stand, "zuletzt_gemeldet": "2026-10-27T14:25:01.301878+00:00"},
                            indent=1), encoding="utf-8")
    k = vm_laden()
    res, unbekannt = gemessen_laden(p, k.waechter_zuordnung)
    assert unbekannt == []
    assert set(res) == {"K1", "K2", "K4", "K6", "K7", "K11/7"}  # Ruling 55b: issue_weg deckt nur K11 Hop 7
    assert all(m.wert == "laeuft" and m.zeit for m in res.values())


def test_gemessen_hop_schluessel_traegt_die_pruefungen_der_kette(tmp_path):
    p = _datei(tmp_path, {"hop": True, "kette": False, "nur_hop": True})
    res, _ = gemessen_laden(p, {"hop": ["K11/7"], "kette": ["K11"], "nur_hop": ["K2/1"]})
    assert res["K11/7"].wert == "bricht" and res["K11"].wert == "bricht"  # die Kette faellt auf den Hop durch
    assert res["K2/1"].wert == "laeuft" and "K2" not in res


def _g(wert):
    z = hop_zustand_neu()
    z["gemessen"] = {"wert": wert, "zeit": "t"}
    g = Graph()
    g.kante_hinzu(Kante("h", "hop", "a", "b", "seed", (), {"kette": "K1", "nr": "1", "zustand": z}))
    return g


def test_hysterese_erst_die_zweite_gleiche_lesung_gilt():
    basis = _g("laeuft")
    neu = _g("bricht")
    assert hysterese(basis, neu, {}) == {"h": "bricht"}
    assert neu.kanten["h"].attribute["zustand"]["gemessen"]["wert"] == "laeuft"  # noch der committete Wert
    neu2 = _g("bricht")
    assert hysterese(basis, neu2, {"h": "bricht"}) == {}
    assert neu2.kanten["h"].attribute["zustand"]["gemessen"]["wert"] == "bricht"
    neu3 = _g("keine_probe")  # andere Lesung als die ausstehende: wieder von vorn
    assert hysterese(basis, neu3, {"h": "bricht"}) == {"h": "keine_probe"}
    assert hysterese(basis, _g("laeuft"), {"h": "bricht"}) == {}  # zurueck auf den committeten Wert


def test_hysterese_verzoegert_die_erste_messung_nicht():
    basis = _g("laeuft")
    basis.kanten["h"].attribute["zustand"]["gemessen"]["zeit"] = None  # Wert aus dem Dokument, nie gemessen
    neu = _g("bricht")
    assert hysterese(basis, neu, {}) == {}
    assert neu.kanten["h"].attribute["zustand"]["gemessen"]["wert"] == "bricht"


def test_i8_gemessen_laden_mit_hoechstalter(tmp_path):
    import os
    import time

    p = tmp_path / "w.json"
    p.write_text(json.dumps({"stand": {"a": True}}), encoding="utf-8")
    res, _ = gemessen_laden(p, {"a": ["K1"]}, max_alter_min=60)
    assert res["K1"].wert == "laeuft"
    alt = time.time() - 2 * 3600
    os.utime(p, (alt, alt))
    res, unbekannt = gemessen_laden(p, {"a": ["K1"]}, max_alter_min=60, alle_ketten=["K2"])
    assert res == {"K1": Messung("keine_probe", None), "K2": Messung("keine_probe", None)} and unbekannt == []
    assert gemessen_laden(p, {"a": ["K1"]})[0]["K1"].wert == "laeuft"  # ohne Grenze wie bisher
