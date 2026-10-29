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
        assert s.cache_holen("b1", "python", "a.py") is None
        s.cache_setzen("b1", "python", e, "a.py")
        assert s.cache_holen("b1", "python", "a.py") == e
    with Speicher(tmp_path / "g.sqlite") as s:
        assert s.cache_holen("b1", "python", "a.py") == e
        assert s.cache_holen("b1", "sql", "a.py") is None


def test_cache_pfad_im_schluessel(tmp_path: Path) -> None:
    # R11: Parser sind pfadabhaengig (test_*.py, *compose*.yml, ...): gleicher Blob an
    # einem anderen Pfad darf das Ergebnis nicht erben.
    e = _ergebnis()
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", e, "src/a.py")
        assert s.cache_holen("b1", "python", "src/a.py") == e
        assert s.cache_holen("b1", "python", "tests/test_a.py") is None
        assert s.cache_holen("b1", "python", "a.py") is None


def test_cache_parser_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", _ergebnis(), "a.py")
        monkeypatch.setattr(sp, "PARSER_VERSION", sp.PARSER_VERSION + 1)
        assert s.cache_holen("b1", "python", "a.py") is None


def test_cache_grammatik_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", _ergebnis(), "a.py")
        monkeypatch.setattr(sp, "GRAMMATIK_VERSION", sp.GRAMMATIK_VERSION + "x")
        assert s.cache_holen("b1", "python", "a.py") is None


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


def _referenz_export(commit: str | None, kn: list[Knoten], ka: list[Kante]) -> bytes:
    """Das bisherige (nicht streamende) Exportformat, `seit` entfernt (R17)."""
    import json

    def ohne_seit(k: Knoten) -> Knoten:
        return Knoten(k.id, k.art, k.name, k.beleg, tuple(p for p in k.attr if p[0] != "seit"))

    doc = {
        "version": 1,
        "commit": commit,
        "knoten": [sp._knoten_zu(ohne_seit(k)) for k in sorted(kn, key=lambda k: k.id)],
        "kanten": [sp._kante_zu(k) for k in sorted(
            ka, key=lambda k: (k.art, k.von, k.nach, sp._dumps(sp._kante_zu(k))))],
    }
    return (sp._esc(json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1)) + "\n").encode("utf-8")


def test_export_streamend_bytegleich_mit_referenz(tmp_path: Path) -> None:
    # Sortierfallen: Nicht-BMP (😀) vs. U+FFFD, Surrogate, Quotes, Backslash, Umlaute.
    namen = ["a.py", "z\udcff.py", 'q"x.py', "b\\c.py", "ä.py", "�.py", "\U0001F600.py", "A.py"]
    kn = [Knoten(id_datei("r", n), "datei", n, Beleg("r", n, 1, "x", "d"),
                 (("seit", "5"), ("zuletzt", "7"))) for n in namen]
    kn.append(Knoten(id_env("X"), "env", "X", None))
    ka = [Kante("enthaelt", "repo:r", id_datei("r", n), True, None) for n in namen]
    ka += [Kante("ruft_auf", id_datei("r", namen[i]), id_datei("r", namen[-1 - i]), i % 2 == 0,
                 Beleg("r", namen[i], i, "f", "d")) for i in range(len(namen))]
    ka.append(Kante("liest_env", id_datei("r", "a.py"), id_env("X"), True, None))
    ka.append(Kante("liest_env", id_datei("r", "a.py"), id_env("X"), False, None))  # gleiche Enden
    with Speicher(tmp_path / "g.sqlite") as s:
        s.graph_ersetzen("c1", reversed(kn), reversed(ka))
        s.export_json(tmp_path / "e.json")
    roh = (tmp_path / "e.json").read_bytes()
    assert roh == _referenz_export("c1", kn, ka)
    assert b'"seit"' not in roh and b'"zuletzt"' in roh


def test_export_leer_bytegleich(tmp_path: Path) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        s.export_json(tmp_path / "e.json")
        assert (tmp_path / "e.json").read_bytes() == _referenz_export(None, [], [])
        s.graph_ersetzen("c", [Knoten("repo:r", "repo", "r", None)], [])
        s.export_json(tmp_path / "f.json")
        assert (tmp_path / "f.json").read_bytes() == _referenz_export(
            "c", [Knoten("repo:r", "repo", "r", None)], [])


def test_seit_bleibt_im_speicher(tmp_path: Path) -> None:
    k = Knoten("symbol:r:a.py#f", "symbol", "f", None, (("seit", "5"),))
    with Speicher(tmp_path / "g.sqlite") as s:
        s.graph_ersetzen("c", [k], [])
        assert s.knoten() == [k]


def test_alter_graph_ohne_sortierspalten_wird_ersetzt(tmp_path: Path) -> None:
    import sqlite3
    c = sqlite3.connect(tmp_path / "g.sqlite")
    c.executescript("CREATE TABLE knoten(id TEXT PRIMARY KEY, wert TEXT NOT NULL);"
                    "CREATE TABLE kanten(wert TEXT PRIMARY KEY);"
                    "CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);"
                    "CREATE TABLE symbolverlauf(nr INTEGER PRIMARY KEY, symbol TEXT);")
    c.executemany("INSERT INTO meta VALUES(?,?)", [('"commit"', '"alt"'), ('"bericht"', '"{}"'),
                                                   ('"name"', '"proj"')])
    c.execute("INSERT INTO knoten VALUES(?,?)", ('"x"', "{}"))
    c.commit()
    c.close()
    kn, ka = _graph()
    with Speicher(tmp_path / "g.sqlite") as s:
        # Der alte Graph ist verworfen: Commit und Bericht duerfen ihn nicht mehr behaupten.
        assert s.meta("commit") is None and s.meta("bericht") is None
        assert s.meta("name") == "proj"
        assert not s.hat_graph()
        s.graph_ersetzen("c1", kn, ka)
        assert s.hat_graph()
        assert s.knoten() == sorted(kn, key=lambda k: k.id)


def test_stand_basis_ohne_anlegen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "home"))
    assert sp.stand_basis() == tmp_path / "home"
    assert not (tmp_path / "home").exists()
    assert re.fullmatch(r"[0-9a-f]{12}", sp.pfad_kennung(tmp_path))
    assert stand_ordner(tmp_path, "proj").name == f"proj-{sp.pfad_kennung(tmp_path)}"


def test_cache_pfad_pflicht(tmp_path: Path) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        with pytest.raises(TypeError):
            s.cache_holen("b1", "python")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            s.cache_setzen("b1", "python", _ergebnis())  # type: ignore[call-arg]


def test_symbolverlauf(tmp_path: Path) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        assert s.symbol_seit("s", 999) is None
        s.symbolverlauf_merken("c1", 100, {"s": "a"})
        s.symbolverlauf_merken("c2", 200, {"s": "a"})
        assert s.symbol_seit("s", 200) == 100
        s.symbolverlauf_merken("c3", 300, {"s": "b"})
        assert s.symbol_seit("s", 300) == 300
        s.symbolverlauf_merken("c4", 400, {"s": "a"})
        assert s.symbol_seit("s", 400) == 400
        # bis_zeit schaut in die Vergangenheit
        assert s.symbol_seit("s", 250) == 100
        assert s.symbol_seit("s", 350) == 300
        assert s.symbol_seit("s", 99) is None


def _zeilen(s: Speicher) -> int:
    return s._verbindung().execute("SELECT COUNT(*) FROM symbolzeit").fetchone()[0]


def test_symbolverlauf_aelterer_lauf_nachgetragen(tmp_path: Path) -> None:
    # R17: Ein spaeter nachgetragener aelterer Lauf wird zeitlich einsortiert und
    # verfaelscht die spaeteren Strecken nicht.
    with Speicher(tmp_path / "g.sqlite") as s:
        s.symbolverlauf_merken("c1", 100, {"f": "x", "g": "x"})
        s.symbolverlauf_merken("c3", 300, {"f": "x", "g": "x"})
        assert s.symbol_seit("f", 300) == 100
        s.symbolverlauf_merken("c2", 200, {"f": "y"})  # g fehlt in c2
        assert s.symbol_seit("f", 100) == 100
        assert s.symbol_seit("f", 200) == 200
        assert s.symbol_seit("f", 300) == 300  # x ab c3 wieder, Strecke neu
        assert s.symbol_seit("g", 200) is None  # in c2 abwesend
        assert s.symbol_seit("g", 300) == 300  # Neubeginn nach der Luecke
        assert s.symbol_seit("g", 100) == 100


def test_symbolverlauf_idempotent(tmp_path: Path) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        s.symbolverlauf_merken("c1", 100, {"f": "x", "g": "x"})
        s.symbolverlauf_merken("c2", 200, {"f": "y"})
        vorher = _zeilen(s)
        s.symbolverlauf_merken("c2", 200, {"f": "y"})
        s.symbolverlauf_merken("c1", 100, {"f": "x", "g": "x"})
        assert _zeilen(s) == vorher
        assert s.symbol_seit("f", 200) == 200 and s.symbol_seit("g", 100) == 100


def test_symbolverlauf_verschwinden_und_wiederkehr(tmp_path: Path) -> None:
    # Semantik: Fehlt ein Symbol in einem erfassten Lauf, beginnt seine Strecke danach neu,
    # auch mit gleichem Digest. Ohne erfassten Lauf dazwischen gibt es keine Luecke.
    with Speicher(tmp_path / "g.sqlite") as s:
        s.symbolverlauf_merken("c1", 100, {"f": "x"})
        s.symbolverlauf_merken("c2", 200, {})
        s.symbolverlauf_merken("c3", 300, {"f": "x"})
        assert s.symbol_seit("f", 200) is None
        assert s.symbol_seit("f", 300) == 300
    with Speicher(tmp_path / "h.sqlite") as s:
        s.symbolverlauf_merken("c1", 100, {"f": "x"})
        s.symbolverlauf_merken("c3", 300, {"f": "x"})
        assert s.symbol_seit("f", 300) == 100


def test_symbolverlauf_bereiche_getrennt(tmp_path: Path) -> None:
    # Ein Lauf eines Repos macht die Symbole eines anderen Repos nicht abwesend.
    with Speicher(tmp_path / "g.sqlite") as s:
        s.symbolverlauf_merken("a1", 100, {"symbol:a:x#f": "x"}, bereich="a")
        s.symbolverlauf_merken("b1", 200, {"symbol:b:y#g": "y"}, bereich="b")
        assert s.symbol_seit("symbol:a:x#f", 200) == 100
        assert s.seit_je_symbol("a", 200) == {"symbol:a:x#f": 100}
        assert s.seit_je_symbol("b", 200) == {"symbol:b:y#g": 200}


def test_seit_je_symbol_gleich_einzeln(tmp_path: Path) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        s.symbolverlauf_merken("c1", 100, {"f": "x", "g": "x", "h": "x"})
        s.symbolverlauf_merken("c3", 300, {"f": "x", "g": "z"})
        s.symbolverlauf_merken("c2", 200, {"f": "x", "g": "x"})
        for t in (100, 150, 200, 300, 400):
            alle = s.seit_je_symbol("", t)
            for sym in ("f", "g", "h"):
                assert alle.get(sym) == s.symbol_seit(sym, t), (sym, t)


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
        s.cache_setzen("b1", "python", e, "a.py")
        assert s.cache_holen("b1", "python", "a.py") == e
        s.graph_ersetzen("c1", kn, ka)
        s.symbolverlauf_merken("c1", 5, {id_datei("r", pfad): "d"})
        assert s.symbol_seit(id_datei("r", pfad), 5) == 5
        assert s.seit_je_symbol("", 5) == {id_datei("r", pfad): 5}
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


def test_export_ist_ein_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ein paralleles graph_ersetzen (zweite Verbindung) zwischen den Lesevorgaengen des
    # Exports darf ihn nicht veraendern: commit, kanten und knoten aus einem Stand.
    kn, ka = _graph()
    neu_kn = [Knoten("repo:neu", "repo", "neu", None)]
    with Speicher(tmp_path / "g.sqlite") as s:
        s.graph_ersetzen("c1", kn, ka)
        s.export_json(tmp_path / "vorher.json")
        echt = sp._liste_schreiben
        aufrufe = []

        def mit_schreiber(f, objekte):  # type: ignore[no-untyped-def]
            if not aufrufe:
                with Speicher(tmp_path / "g.sqlite") as zweite:
                    zweite.graph_ersetzen("c2", neu_kn, [])
            aufrufe.append(1)
            echt(f, objekte)

        monkeypatch.setattr(sp, "_liste_schreiben", mit_schreiber)
        s.export_json(tmp_path / "waehrend.json")
        monkeypatch.setattr(sp, "_liste_schreiben", echt)
        assert len(aufrufe) == 2
        assert (tmp_path / "waehrend.json").read_bytes() == (tmp_path / "vorher.json").read_bytes()
        s.export_json(tmp_path / "danach.json")  # danach sieht der Export den neuen Stand
    assert (tmp_path / "danach.json").read_bytes() == _referenz_export("c2", neu_kn, [])


def test_synchronous_normal(tmp_path: Path) -> None:
    with Speicher(tmp_path / "g.sqlite") as s:
        c = s._verbindung()
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL


def test_cache_python_version_im_schluessel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ast-gestuetzte Spezialextraktoren haengen von der Python-Version ab (z. B. PEP 695).
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", _ergebnis(), "a.py")
        assert s.cache_holen("b1", "python", "a.py") is not None
        monkeypatch.setattr(sp.sys, "version_info", (3, 99, 0, "final", 0))
        assert s.cache_holen("b1", "python", "a.py") is None


def test_cache_spezial_fehlt_roundtrip(tmp_path: Path) -> None:
    import dataclasses
    e = dataclasses.replace(_ergebnis(), spezial_fehlt=True)
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", e, "a.py")
        assert s.cache_holen("b1", "python", "a.py") == e
        assert s.cache_hat("b1", "python", "a.py")
        assert not s.cache_hat("b2", "python", "a.py")


def test_stand_ordner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "home"))
    o = stand_ordner(tmp_path, "proj")
    assert o.parent == tmp_path / "home" and o.is_dir()
    assert re.fullmatch(r"proj-[0-9a-f]{12}", o.name)
    assert stand_ordner(tmp_path, "proj") == o


def test_cache_roundtrip_mit_dekoratoren_s10(tmp_path: Path) -> None:
    e = DateiErgebnis(symbole=(Symbol("f", "funktion", 1, 3, "d1", ("app.get", "property")),
                               Symbol("g", "funktion", 4, 5, "d2")))
    with Speicher(tmp_path / "g.sqlite") as s:
        s.cache_setzen("b1", "python", e, "a.py")
        assert s.cache_holen("b1", "python", "a.py") == e
