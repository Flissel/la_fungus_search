"""Auflöser: Kanten über Dateigrenzen aus handgebauten DateiErgebnissen (ohne Git)."""
from __future__ import annotations

from selbstbild.aufloesen import aufloesen
from selbstbild.regeln import Regeln
from selbstbild.typen import (
    PLATZHALTER_REPO, Aufruf, Beleg, DateiErgebnis, Einstieg, EnvLesung, Import, Kante,
    Knoten, SqlObjekt, Symbol, id_datei, id_doku_seite, id_env, id_symbol, id_tabelle,
)


def sym(qualname: str, art: str = "funktion", von: int = 1, bis: int = 2) -> Symbol:
    return Symbol(qualname, art, von, bis, f"d-{qualname}")  # type: ignore[arg-type]


def kanten_von(kanten: list[Kante], art: str) -> dict[tuple[str, str], bool]:
    return {(k.von, k.nach): k.sicher for k in kanten if k.art == art}


def knoten_je_id(knoten: list[Knoten]) -> dict[str, Knoten]:
    return {k.id: k for k in knoten}


def attr(k: Knoten) -> dict[str, str]:
    return dict(k.attr)


def test_import_python():
    erg = {
        ("r", "pkg/__init__.py"): DateiErgebnis(),
        ("r", "pkg/a.py"): DateiErgebnis(
            symbole=(sym("g", von=3, bis=4),),
            importe=(Import("pkg.b", ("f",), 1, False),),
            aufrufe=(Aufruf("g", "f", 4),),
        ),
        ("r", "pkg/b.py"): DateiErgebnis(symbole=(sym("f"),)),
    }
    knoten, kanten = aufloesen(erg, {})
    imp = kanten_von(kanten, "importiert")
    assert imp == {(id_datei("r", "pkg/a.py"), id_datei("r", "pkg/b.py")): True}
    rufe = kanten_von(kanten, "ruft_auf")
    assert rufe[(id_symbol("r", "pkg/a.py", "g"), id_symbol("r", "pkg/b.py", "f"))] is True
    enth = kanten_von(kanten, "enthaelt")
    assert enth[("repo:r", id_datei("r", "pkg/a.py"))] is True
    assert enth[(id_datei("r", "pkg/a.py"), id_symbol("r", "pkg/a.py", "g"))] is True


def test_import_src_relativ_und_ts():
    erg = {
        ("r", "src/app/__init__.py"): DateiErgebnis(),
        ("r", "src/app/kern.py"): DateiErgebnis(
            symbole=(sym("start"),),
            importe=(
                Import("helfer", ("h",), 1, True),          # from .helfer import h
                Import("app.modell", (), 2, False),          # import app.modell
                Import("requests", (), 3, False),            # extern: keine Kante
            ),
            aufrufe=(
                Aufruf("start", "h", 5),
                Aufruf("start", "app.modell.laden", 6),
                Aufruf("start", "requests.get", 7),
            ),
        ),
        ("r", "src/app/helfer.py"): DateiErgebnis(symbole=(sym("h"),)),
        ("r", "src/app/modell.py"): DateiErgebnis(symbole=(sym("laden"), sym("get"))),
        ("r", "web/seite.ts"): DateiErgebnis(
            symbole=(sym("zeigen"),),
            importe=(Import("./util", ("format",), 1, True), Import("react", ("useState",), 2, False)),
            aufrufe=(Aufruf("zeigen", "format", 3),),
        ),
        ("r", "web/util/index.ts"): DateiErgebnis(symbole=(sym("format"),)),
    }
    _, kanten = aufloesen(erg, {})
    imp = kanten_von(kanten, "importiert")
    assert imp == {
        (id_datei("r", "src/app/kern.py"), id_datei("r", "src/app/helfer.py")): True,
        (id_datei("r", "src/app/kern.py"), id_datei("r", "src/app/modell.py")): True,
        (id_datei("r", "web/seite.ts"), id_datei("r", "web/util/index.ts")): True,
    }
    rufe = kanten_von(kanten, "ruft_auf")
    kern = id_symbol("r", "src/app/kern.py", "start")
    assert rufe[(kern, id_symbol("r", "src/app/helfer.py", "h"))] is True
    assert rufe[(kern, id_symbol("r", "src/app/modell.py", "laden"))] is True
    # requests.get: extern importiert, also auch keine unsichere Kante zum einzigen `get`
    assert (kern, id_symbol("r", "src/app/modell.py", "get")) not in rufe
    # TS-Name kann ein Default-Import sein (in `Import` nicht unterscheidbar): nur unsicher
    assert rufe[(id_symbol("r", "web/seite.ts", "zeigen"), id_symbol("r", "web/util/index.ts", "format"))] is False


def test_ts_barrel_eine_ebene():
    erg = {
        ("r", "ui/index.ts"): DateiErgebnis(importe=(
            Import("./knopf", ("Knopf",), 1, True),     # export { Knopf } from "./knopf"
            Import("./feld", ("*",), 2, True),          # export * from "./feld"
            Import("./tiefer", ("*",), 3, True),
        )),
        ("r", "ui/knopf.ts"): DateiErgebnis(symbole=(sym("Knopf"),)),
        ("r", "ui/feld.ts"): DateiErgebnis(symbole=(sym("Feld"),)),
        ("r", "ui/tiefer.ts"): DateiErgebnis(importe=(Import("./ganz", ("*",), 1, True),)),
        ("r", "ui/ganz.ts"): DateiErgebnis(symbole=(sym("Ganz"),)),
        ("r", "seite.ts"): DateiErgebnis(
            symbole=(sym("zeigen"),),
            importe=(Import("./ui/index", ("Knopf", "Feld", "Ganz"), 1, True),),
            aufrufe=(Aufruf("zeigen", "Knopf", 3), Aufruf("zeigen", "Feld", 4), Aufruf("zeigen", "Ganz", 5)),
        ),
    }
    _, kanten = aufloesen(erg, {})
    rufe = kanten_von(kanten, "ruft_auf")
    zeigen = id_symbol("r", "seite.ts", "zeigen")
    assert rufe[(zeigen, id_symbol("r", "ui/knopf.ts", "Knopf"))] is False  # R15: unsicher
    assert rufe[(zeigen, id_symbol("r", "ui/feld.ts", "Feld"))] is False
    # nur eine Ebene: ui/ganz.ts wird über index -> tiefer -> ganz nicht gefunden
    assert (zeigen, id_symbol("r", "ui/ganz.ts", "Ganz")) not in rufe
    assert kanten_von(kanten, "importiert")[(id_datei("r", "ui/index.ts"), id_datei("r", "ui/knopf.ts"))] is True


def test_relativ_hoehere_ebene_bleibt_unsicher():
    # `from .h import f` in p/q/a.py; h.py gibt es nur eine Ebene höher (p/h.py)
    erg = {
        ("r", "p/q/a.py"): DateiErgebnis(
            symbole=(sym("g"),),
            importe=(Import("h", ("f",), 1, True),),
            aufrufe=(Aufruf("g", "f", 2),),
        ),
        ("r", "p/q/b.py"): DateiErgebnis(
            symbole=(sym("k"),),
            importe=(Import("h", ("*",), 1, True), Import("", ("h",), 2, True)),
            aufrufe=(Aufruf("k", "f", 3), Aufruf("k", "h.f", 4)),
        ),
        ("r", "p/h.py"): DateiErgebnis(symbole=(sym("f"),)),
    }
    _, kanten = aufloesen(erg, {})
    assert kanten_von(kanten, "importiert") == {
        (id_datei("r", "p/q/a.py"), id_datei("r", "p/h.py")): False,
        (id_datei("r", "p/q/b.py"), id_datei("r", "p/h.py")): False,
    }
    rufe = kanten_von(kanten, "ruft_auf")
    ziel = id_symbol("r", "p/h.py", "f")
    assert rufe[(id_symbol("r", "p/q/a.py", "g"), ziel)] is False   # importierter Name
    assert rufe[(id_symbol("r", "p/q/b.py", "k"), ziel)] is False   # Stern bzw. Untermodul


def test_zwei_wurzeln_naechste_gewinnt():
    erg = {
        ("r", "spaces/m/app/__init__.py"): DateiErgebnis(),
        ("r", "spaces/m/app/cfg.py"): DateiErgebnis(symbole=(sym("lade"),)),
        ("r", "spaces/s/app/__init__.py"): DateiErgebnis(),
        ("r", "spaces/s/app/cfg.py"): DateiErgebnis(symbole=(sym("lade"),)),
        ("r", "spaces/s/app/x.py"): DateiErgebnis(
            symbole=(sym("lauf"),),
            importe=(Import("app.cfg", ("lade",), 1, False),),
            aufrufe=(Aufruf("lauf", "lade", 2),),
        ),
        ("r", "tools/y.py"): DateiErgebnis(
            symbole=(sym("werkzeug"),),
            importe=(Import("app.cfg", ("lade",), 1, False), Import("json", (), 2, False)),
            aufrufe=(Aufruf("werkzeug", "lade", 3), Aufruf("werkzeug", "json.loads", 4)),
        ),
    }
    _, kanten = aufloesen(erg, {})
    imp = kanten_von(kanten, "importiert")
    rufe = kanten_von(kanten, "ruft_auf")
    x, y = id_datei("r", "spaces/s/app/x.py"), id_datei("r", "tools/y.py")
    assert imp[(x, id_datei("r", "spaces/s/app/cfg.py"))] is True
    assert (x, id_datei("r", "spaces/m/app/cfg.py")) not in imp
    assert rufe[(id_symbol("r", "spaces/s/app/x.py", "lauf"), id_symbol("r", "spaces/s/app/cfg.py", "lade"))] is True
    m_ziele = {id_datei("r", "spaces/m/app/cfg.py"), id_symbol("r", "spaces/m/app/cfg.py", "lade")}
    assert not any(k.sicher and k.nach in m_ziele for k in kanten if k.art != "enthaelt")
    # keine Wurzel über tools/y.py, zwei Kandidaten: geraten, also unsicher
    assert [s for (v, _), s in imp.items() if v == y] == [False]
    assert all(s is False for (v, _), s in rufe.items() if v == id_symbol("r", "tools/y.py", "werkzeug"))


def test_self_methode():
    erg = {
        ("r", "laden.py"): DateiErgebnis(
            symbole=(
                sym("Laden", "klasse", 1, 9),
                sym("Laden.kaufen", "methode", 2, 5),
                sym("Laden.zahlen", "methode", 6, 9),
            ),
            aufrufe=(Aufruf("Laden.kaufen", "self.zahlen", 4),),
        ),
    }
    _, kanten = aufloesen(erg, {})
    rufe = kanten_von(kanten, "ruft_auf")
    assert rufe == {
        (id_symbol("r", "laden.py", "Laden.kaufen"), id_symbol("r", "laden.py", "Laden.zahlen")): True,
    }


def test_eindeutiger_name_unsicher():
    erg = {
        ("r", "a.py"): DateiErgebnis(symbole=(sym("g"),), aufrufe=(Aufruf("g", "rechnen", 2),)),
        ("r", "b.py"): DateiErgebnis(symbole=(sym("rechnen"),)),
    }
    _, kanten = aufloesen(erg, {})
    rufe = kanten_von(kanten, "ruft_auf")
    assert rufe == {(id_symbol("r", "a.py", "g"), id_symbol("r", "b.py", "rechnen")): False}

    erg[("r", "c.py")] = DateiErgebnis(symbole=(sym("rechnen"),))
    _, kanten = aufloesen(erg, {})
    assert kanten_von(kanten, "ruft_auf") == {}


def test_eindeutiger_name_nie_eine_methode():
    # `rechnen()` ist ein schlichter Aufruf; die einzige Methode `Z.rechnen` ist kein Ziel.
    erg = {
        ("r", "a.py"): DateiErgebnis(symbole=(sym("g"),), aufrufe=(Aufruf("g", "rechnen", 2),)),
        ("r", "b.py"): DateiErgebnis(symbole=(sym("Z", "klasse", 1, 9), sym("Z.rechnen", "methode", 2, 3))),
    }
    _, kanten = aufloesen(erg, {})
    assert kanten_von(kanten, "ruft_auf") == {}


def test_skript_ordner_zuerst():
    # sys.path[0]-Semantik: scripts/lauf.py importiert den Nachbarn scripts/helfer.py.
    erg = {
        ("r", "scripts/lauf.py"): DateiErgebnis(
            symbole=(sym("main"),),
            importe=(Import("helfer", ("hilf",), 1, False),),
            aufrufe=(Aufruf("main", "hilf", 4),),
        ),
        ("r", "scripts/helfer.py"): DateiErgebnis(symbole=(sym("hilf"),)),
    }
    lauf = id_datei("r", "scripts/lauf.py")
    main = id_symbol("r", "scripts/lauf.py", "main")
    ziel = (lauf, id_datei("r", "scripts/helfer.py"))
    ruf = (main, id_symbol("r", "scripts/helfer.py", "hilf"))
    _, kanten = aufloesen(erg, {})
    assert kanten_von(kanten, "importiert") == {ziel: True}
    assert kanten_von(kanten, "ruft_auf") == {ruf: True}
    # ein gleichnamiges Modul an der Repo-Wurzel verliert gegen den eigenen Ordner
    erg[("r", "helfer.py")] = DateiErgebnis(symbole=(sym("hilf"),))
    _, kanten = aufloesen(erg, {})
    assert kanten_von(kanten, "importiert") == {ziel: True}
    assert kanten_von(kanten, "ruft_auf") == {ruf: True}


def test_eigener_ordner_nicht_im_paket():
    # Innerhalb eines Pakets gilt kein Skript-Ordner: `import json` bleibt die stdlib.
    erg = {
        ("r", "pkg/__init__.py"): DateiErgebnis(),
        ("r", "pkg/a.py"): DateiErgebnis(
            symbole=(sym("f"),), importe=(Import("json", (), 1, False),),
            aufrufe=(Aufruf("f", "json.loads", 2),)),
        ("r", "pkg/json.py"): DateiErgebnis(symbole=(sym("loads"),)),
    }
    _, kanten = aufloesen(erg, {})
    assert kanten_von(kanten, "importiert") == {}
    assert kanten_von(kanten, "ruft_auf") == {}


def test_gepunktete_namen_ohne_rueckfall():
    erg = {
        ("r", "a.py"): DateiErgebnis(
            symbole=(sym("K", "klasse", 1, 9), sym("K.g", "methode", 2, 5)),
            aufrufe=(Aufruf("K.g", "obj.rechnen", 3), Aufruf("K.g", "np.array", 4), Aufruf("K.g", "self.erbe", 5)),
        ),
        ("r", "b.py"): DateiErgebnis(symbole=(sym("Z.rechnen", "methode"), sym("array"), sym("erbe"))),
    }
    _, kanten = aufloesen(erg, {})
    assert kanten_von(kanten, "ruft_auf") == {}


def test_env_bruecke():
    erg = {
        ("r1", "a.py"): DateiErgebnis(
            symbole=(sym("hole"),), env=(EnvLesung("SALES_API_URL", 2, "hole"),)),
        ("r2", "b.ts"): DateiErgebnis(env=(EnvLesung("SALES_API_URL", 1, None),)),
    }
    knoten, kanten = aufloesen(erg, {}, {"SALES_API_URL": "sales"})
    envs = [k for k in knoten if k.art == "env"]
    assert [k.id for k in envs] == [id_env("SALES_API_URL")]
    assert kanten_von(kanten, "liest_env") == {
        (id_symbol("r1", "a.py", "hole"), id_env("SALES_API_URL")): True,
        (id_datei("r2", "b.ts"), id_env("SALES_API_URL")): True,
    }
    assert kanten_von(kanten, "nutzt_dienst") == {
        (id_symbol("r1", "a.py", "hole"), "dienst:sales"): True,
        (id_datei("r2", "b.ts"), "dienst:sales"): True,
    }
    assert "dienst:sales" in knoten_je_id(knoten)


def test_tabelle():
    erg = {
        ("r", "db/schema.sql"): DateiErgebnis(sql=(SqlObjekt("tabelle", "marketing.kampagnen", 1, "dt"),)),
        ("r", "lade.py"): DateiErgebnis(
            symbole=(sym("lade"),),
            tabellen_zugriffe=(("marketing.kampagnen", 2, "lade"), ("unbekannt", 3, None)),
        ),
    }
    knoten, kanten = aufloesen(erg, {})
    ls = kanten_von(kanten, "liest_schreibt")
    assert ls[(id_symbol("r", "lade.py", "lade"), id_tabelle("marketing.kampagnen"))] is True
    assert ls[(id_datei("r", "lade.py"), id_tabelle("unbekannt"))] is False
    tab = knoten_je_id(knoten)[id_tabelle("marketing.kampagnen")]
    assert tab.art == "tabelle" and tab.beleg is not None and tab.beleg.pfad == "db/schema.sql"


def test_regel_einstieg():
    erg = {
        ("r", "jobs/nacht.py"): DateiErgebnis(symbole=(sym("lauf"), sym("rest"))),
        ("r", "jobs/tag.py"): DateiErgebnis(symbole=(sym("lauf"),)),
    }
    knoten, _ = aufloesen(erg, {"r": Regeln(einstiegspunkte=["jobs/nacht.py::lauf"])})
    k = knoten_je_id(knoten)
    assert attr(k[id_symbol("r", "jobs/nacht.py", "lauf")])["einstieg"] == "regel"
    assert "einstieg" not in attr(k[id_symbol("r", "jobs/nacht.py", "rest")])
    assert "einstieg" not in attr(k[id_symbol("r", "jobs/tag.py", "lauf")])


def test_einstiege_und_fehlerhaft():
    erg = {
        ("r", "t/test_x.py"): DateiErgebnis(
            symbole=(sym("test_a"),),
            einstiege=(Einstieg("test", "test_a", 1), Einstieg("main", None, 5)),
            fehlerhaft=True,
        ),
    }
    knoten, _ = aufloesen(erg, {"r": Regeln(einstiegspunkte=["t/**::test_a"])})
    k = knoten_je_id(knoten)
    s = attr(k[id_symbol("r", "t/test_x.py", "test_a")])
    assert s == {"einstieg": "regel,test", "fehlerhaft": "1", "symbolart": "funktion"}
    assert list(k[id_symbol("r", "t/test_x.py", "test_a")].attr) == sorted(k[id_symbol("r", "t/test_x.py", "test_a")].attr)
    assert attr(k[id_datei("r", "t/test_x.py")])["einstieg"] == "main"


def test_spezial_fehlt_als_datei_attr():
    erg = {
        ("r", "a.py"): DateiErgebnis(symbole=(sym("f"),), fehlerhaft=True, spezial_fehlt=True),
        ("r", "b.py"): DateiErgebnis(symbole=(sym("g"),)),
    }
    k = knoten_je_id(aufloesen(erg, {})[0])
    assert attr(k[id_datei("r", "a.py")]) == {"fehlerhaft": "1", "spezial_fehlt": "1"}
    assert attr(k[id_datei("r", "b.py")]) == {}


def test_hook_und_doku():
    erg = {
        ("r", ".githooks/post-commit"): DateiErgebnis(),
        ("r", "tools/.claude/hooks/vorher.py"): DateiErgebnis(),
        ("r", "docs/a.md"): DateiErgebnis(),
        ("r", "code.py"): DateiErgebnis(),
    }
    knoten, _ = aufloesen(erg, {})
    k = knoten_je_id(knoten)
    assert attr(k[id_datei("r", ".githooks/post-commit")])["einstieg"] == "hook"
    assert attr(k[id_datei("r", "tools/.claude/hooks/vorher.py")])["einstieg"] == "hook"
    assert "einstieg" not in attr(k[id_datei("r", "code.py")])
    assert id_doku_seite("r", "docs/a.md") == "doku_seite:r:docs/a.md"
    assert k["doku_seite:r:docs/a.md"].art == "doku_seite"
    assert not any(i.startswith("doku_seite:r:code.py") for i in k)


def _gemischt() -> dict[tuple[str, str], DateiErgebnis]:
    p = PLATZHALTER_REPO
    route = f"route:{p}:GET /x"
    b = Beleg(p, "api/app.py", 3, "GET /x", "dx")
    return {
        ("r", "api/app.py"): DateiErgebnis(
            symbole=(sym("liste"),),
            aufrufe=(Aufruf("liste", "gibtsnicht", 4), Aufruf(None, "liste", 9)),
            env=(EnvLesung("DB_URL", 4, "liste"),),
            tabellen_zugriffe=(("kunden", 5, "liste"),),
            extra_knoten=(Knoten(route, "route", "GET /x", b), Knoten("env:HTTP_ZIEL", "env", "HTTP_ZIEL", None)),
            extra_kanten=(
                Kante("definiert_route", f"symbol:{p}:api/app.py#liste", route, True, b),
                Kante("definiert_route", f"symbol:{p}:api/app.py#fehlt", route, True, b),
                Kante("nutzt_dienst", f"datei:{p}:api/app.py", "env:HTTP_ZIEL", True, b),
                Kante("liest_env", f"symbol:{p}:api/app.py#fehlt", "env:WAISE", True, b),
            ),
        ),
        ("r", "api/zweit.py"): DateiErgebnis(
            extra_knoten=(Knoten(route, "route", "GET /x", Beleg(p, "api/zweit.py", 1, "GET /x", "dy")),),
            extra_kanten=(Kante("verbindet", "dienst:a", "dienst:b", False, None),
                          Kante("verbindet", "dienst:a", "dienst:b", True, None)),
        ),
        ("s", "api/app.py"): DateiErgebnis(
            extra_knoten=(Knoten(route, "route", "GET /x", b),),
        ),
        ("r", "compose.yml"): DateiErgebnis(env=(EnvLesung("DB_URL", 2, None),)),
    }


def test_keine_haengenden_kanten():
    knoten, kanten = aufloesen(_gemischt(), {}, {"DB_URL": "postgres"})
    ids = {k.id for k in knoten}
    assert len(ids) == len(knoten)
    for k in kanten:
        assert k.von in ids and k.nach in ids, k
    # Kante zu einem nicht existierenden Symbol fällt weg
    assert not any(k.von.endswith("#fehlt") for k in kanten)
    assert {"dienst:a", "dienst:b", "dienst:postgres", "env:DB_URL", "tabelle:public.kunden"} <= ids
    # Bei-Bedarf-Knoten nur für überlebende Kanten: keine Waise aus einer verworfenen Kante
    assert "env:WAISE" not in ids


def test_platzhalter_gebunden_und_dedupliziert():
    knoten, kanten = aufloesen(_gemischt(), {})
    for k in knoten:
        assert f":{PLATZHALTER_REPO}:" not in k.id and k.id != f"repo:{PLATZHALTER_REPO}"
        assert k.beleg is None or k.beleg.repo != PLATZHALTER_REPO
    for k in kanten:
        for i in (k.von, k.nach):
            assert f":{PLATZHALTER_REPO}:" not in i
        assert k.beleg is None or k.beleg.repo != PLATZHALTER_REPO
    k = knoten_je_id(knoten)
    # gleiche Route aus zwei Dateien: erster nach (repo, pfad) gewinnt
    assert k["route:r:GET /x"].beleg is not None and k["route:r:GET /x"].beleg.pfad == "api/app.py"
    assert "route:s:GET /x" in k
    assert kanten_von(kanten, "definiert_route") == {
        (id_symbol("r", "api/app.py", "liste"), "route:r:GET /x"): True,
    }
    verb = [x for x in kanten if x.art == "verbindet"]
    assert len(verb) == 1 and verb[0].sicher is True
    # Modulebene ruft `liste` in derselben Datei auf
    assert kanten_von(kanten, "ruft_auf") == {
        (id_datei("r", "api/app.py"), id_symbol("r", "api/app.py", "liste")): True,
    }


def test_deterministisch():
    erg = _gemischt()
    erg[("r", "pkg/a.py")] = DateiErgebnis(
        symbole=(sym("g"),), importe=(Import("pkg.b", ("f",), 1, False),), aufrufe=(Aufruf("g", "f", 2),))
    erg[("r", "pkg/b.py")] = DateiErgebnis(symbole=(sym("f"),))
    umgekehrt = dict(reversed(list(erg.items())))
    regeln = {"r": Regeln(einstiegspunkte=["pkg/*.py"]), "s": Regeln()}
    eins = aufloesen(erg, regeln, {"DB_URL": "pg"})
    zwei = aufloesen(umgekehrt, dict(reversed(list(regeln.items()))), {"DB_URL": "pg"})
    assert eins == zwei
    knoten, kanten = eins
    assert [k.id for k in knoten] == sorted(k.id for k in knoten)
    assert [(k.art, k.von, k.nach) for k in kanten] == sorted((k.art, k.von, k.nach) for k in kanten)
    assert attr(knoten_je_id(knoten)[id_datei("r", "pkg/a.py")])["einstieg"] == "regel"
