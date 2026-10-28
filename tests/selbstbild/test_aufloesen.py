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
    assert rufe[(id_symbol("r", "web/seite.ts", "zeigen"), id_symbol("r", "web/util/index.ts", "format"))] is True


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
    assert s["einstieg"] == "regel,test" and s["fehlerhaft"] == "1"
    assert list(k[id_symbol("r", "t/test_x.py", "test_a")].attr) == sorted(k[id_symbol("r", "t/test_x.py", "test_a")].attr)
    assert attr(k[id_datei("r", "t/test_x.py")])["einstieg"] == "main"


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
