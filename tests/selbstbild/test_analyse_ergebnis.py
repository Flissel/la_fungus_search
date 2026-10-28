import dataclasses
import hashlib
import json
import sqlite3
from pathlib import Path

from selbstbild.analyse.bereiche import Bereich
from selbstbild.analyse.doppelt import Doppelung
from selbstbild.analyse.ergebnis import Analyse, export_json, speichern
from selbstbild.analyse.kolonie import KolonieErgebnis, KolonieParameter
from selbstbild.analyse.tot import TotFund
from selbstbild.analyse.zentral import Hotspot


def _analyse(datei: str = "a.py") -> Analyse:
    return Analyse(
        commit="abc", version=1, parameter=(("seed", "1"), ("tot_tage", "90")),
        bereiche=(Bereich("b1", "Kern", (datei, "b.py"), 0.123456789),),
        pagerank_top=((datei, 0.123456789012),), betweenness_top=((datei, 1 / 3),),
        hotspots=(Hotspot(datei, 0.5, (("aenderung", 0.25), ("groesse", 0.5), ("zentral", 1.0))),),
        tot=(TotFund("sym:" + datei, "sicher_tot", ("a", "b"), None),),
        doppelungen=(Doppelung("gleiche_datei", (datei, "b.py"), 1.0),),
        kolonie=KolonieErgebnis(((datei, 3),), ((datei, 0.5),), (("b1", 1.0),), ((datei, 0.2),),
                                ("z.py",), KolonieParameter()),
        luecken=(("fehlende_kanten", 2),), statistik=(("doppelt_eimer_uebersprungen", 1),))


def test_export_bytegleich(tmp_path: Path) -> None:
    export_json(_analyse(), tmp_path / "1.json")
    export_json(_analyse(), tmp_path / "2.json")
    b = (tmp_path / "1.json").read_bytes()
    assert b == (tmp_path / "2.json").read_bytes()
    d = json.loads(b)
    assert d["bereiche"][0]["kohaesion"] == "0.1235"
    assert d["pagerank_top"][0][1] == "0.12345679"
    assert d["kolonie"]["tote_ecken"] == ["z.py"]
    assert d["statistik"] == [["doppelt_eimer_uebersprungen", 1]]


def test_export_ohne_kolonie(tmp_path: Path) -> None:
    a = dataclasses.replace(_analyse(), kolonie=None)
    export_json(a, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_bytes())["kolonie"] is None


def test_export_surrogat_pfad(tmp_path: Path) -> None:
    pfad = "dir/\udcff.py"
    export_json(_analyse(pfad), tmp_path / "s.json")
    b = (tmp_path / "s.json").read_bytes()
    b.decode("utf-8")
    assert json.loads(b)["pagerank_top"][0][0] == pfad


def _zeilen(stand: Path) -> list[tuple[str, int, str]]:
    c = sqlite3.connect(str(stand / "analyse.sqlite"))
    try:
        return c.execute("SELECT commit_id, version, parameter_digest FROM lauf").fetchall()
    finally:
        c.close()


def test_ersetzen(tmp_path: Path) -> None:
    speichern(tmp_path, _analyse())
    speichern(tmp_path, _analyse())
    assert len(_zeilen(tmp_path)) == 1
    speichern(tmp_path, _analyse("c.py"))  # gleicher Schluessel, anderer Inhalt
    assert len(_zeilen(tmp_path)) == 1


def test_graph_unberuehrt(tmp_path: Path) -> None:
    g = tmp_path / "graph.sqlite"
    c = sqlite3.connect(str(g))
    c.execute("CREATE TABLE t (x)")
    c.commit()
    c.close()
    vorher = hashlib.sha256(g.read_bytes()).hexdigest()
    speichern(tmp_path, _analyse())
    assert hashlib.sha256(g.read_bytes()).hexdigest() == vorher
    assert not (tmp_path / "graph.sqlite-wal").exists()
