from __future__ import annotations

import re
from pathlib import Path

import pytest

from selbstbild import speicher as sp
from selbstbild.speicher import Speicher, stand_ordner
from selbstbild.typen import (
    Aufruf, Beleg, DateiErgebnis, Einstieg, EnvLesung, Import, Kante, Knoten,
    SqlObjekt, Symbol, id_datei, id_env, id_tabelle,
)

BELEG = Beleg("r", "a.py", 3, "def f", "ab" * 32)


def _ergebnis(pfad: str = "a.py") -> DateiErgebnis:
    k = Knoten(id_datei("r", pfad), "datei", pfad, BELEG, (("sprache", "python"),))
    return DateiErgebnis(
        symbole=(Symbol("f", "funktion", 1, 3, "d1"),),
        importe=(Import("os", ("path",), 1, False),),
        aufrufe=(Aufruf("f", "g", 2),),
        env=(EnvLesung("HOME", 2, None),),
        einstiege=(Einstieg("main", None, 9),),
        sql=(SqlObjekt("tabelle", "t", 1, "d2"),),
        tabellen_zugriffe=(("t", 4, "f"), ("u", 5, None)),
        extra_knoten=(k,),
        extra_kanten=(Kante("enthaelt", "x", k.id, True, BELEG),),
        fehlerhaft=True,
    )


def test_cache_roundtrip(tmp_path: Path) -> None:
    e = _ergebnis()
    with Speicher(tmp_path / "g.sqlite") as s:
        assert s.cache_holen("b1", "python") is None
        s.cache_setzen("b1", "python", e)
        assert s.cache_holen("b1", "python") == e
    with Speicher(tmp_path / "g.sqlite") as s:
        assert s.cache_holen("b1", "python") == e
        assert s.cache_holen("b1", "sql") is None


def test_cache_parser_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", _ergebnis())
        monkeypatch.setattr(sp, "PARSER_VERSION", sp.PARSER_VERSION + 1)
        assert s.cache_holen("b1", "python") is None


def test_cache_grammatik_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", _ergebnis())
        monkeypatch.setattr(sp, "GRAMMATIK_VERSION", sp.GRAMMATIK_VERSION + "x")
        assert s.cache_holen("b1", "python") is None


def _graph() -> tuple[list[Knoten], list[Kante]]:
    kn = [Knoten(id_datei("r", "a.py"), "datei", "a.py", BELEG),
          Knoten(id_env("X"), "env", "X", None),
          Knoten(id_tabelle("T"), "tabelle", "T", None, (("k", "v"),))]
    ka = [Kante("liest_env", id_datei("r", "a.py"), id_env("X"), False, BELEG),
          Kante("enthaelt", "repo:r", id_datei("r", "a.py"), True, None)]
    return kn, ka


def test_export_deterministisch(tmp_path: Path) -> None:
    kn, ka = _graph()
    with Speicher(tmp_path / "a.sqlite") as s:
        s.graph_ersetzen("c1", kn, ka)
        s.export_json(tmp_path / "a.json")
    with Speicher(tmp_path / "b.sqlite") as s:
        s.graph_ersetzen("c1", reversed(kn), reversed(ka))
        s.export_json(tmp_path / "b.json")
        assert s.knoten() == sorted(kn, key=lambda k: k.id)
    a = (tmp_path / "a.json").read_bytes()
    assert a == (tmp_path / "b.json").read_bytes()
    assert a.endswith(b"\n") and b'"version": 1' in a and b'"commit": "c1"' in a


def test_symbolverlauf(tmp_path: Path) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        assert s.symbol_seit("s") is None
        s.symbolverlauf_merken("c1", 100, {"s": "a"})
        s.symbolverlauf_merken("c2", 200, {"s": "a"})
        assert s.symbol_seit("s") == 100
        s.symbolverlauf_merken("c3", 300, {"s": "b"})
        assert s.symbol_seit("s") == 300
        s.symbolverlauf_merken("c4", 400, {"s": "a"})
        assert s.symbol_seit("s") == 400


def test_meta(tmp_path: Path) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        assert s.meta("x") is None
        s.meta_setzen("x", "y")
        assert s.meta("x") == "y"


def test_surrogat_pfad(tmp_path: Path) -> None:
    pfad = "a\udcff.py"
    e = _ergebnis(pfad)
    kn = [Knoten(id_datei("r", pfad), "datei", pfad, Beleg("r", pfad, 1, "x", "d"))]
    ka = [Kante("enthaelt", "repo:r", id_datei("r", pfad), True, None)]
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", e)
        assert s.cache_holen("b1", "python") == e
        s.graph_ersetzen("c1", kn, ka)
        s.symbolverlauf_merken("c1", 5, {id_datei("r", pfad): "d"})
        assert s.symbol_seit(id_datei("r", pfad)) == 5
        assert s.knoten() == kn and s.kanten() == ka
        s.export_json(tmp_path / "e.json")
        s.export_json(tmp_path / "e2.json")
    roh = (tmp_path / "e.json").read_bytes()
    assert roh == (tmp_path / "e2.json").read_bytes()
    assert b"a\\udcff.py" in roh


def test_verbindung_nach_exit_geschlossen(tmp_path: Path) -> None:
    s = Speicher(tmp_path / "g.sqlite")
    with s:
        s.meta_setzen("a", "b")
    (tmp_path / "g.sqlite").unlink(missing_ok=True)  # waere unter Windows gesperrt


def test_stand_ordner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "home"))
    o = stand_ordner(tmp_path, "proj")
    assert o.parent == tmp_path / "home" and o.is_dir()
    assert re.fullmatch(r"proj-[0-9a-f]{12}", o.name)
    assert stand_ordner(tmp_path, "proj") == o
