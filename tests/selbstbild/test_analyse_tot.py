from __future__ import annotations

from selbstbild.analyse import ist_testdatei
from selbstbild.analyse.doku_verweise import bezeichner_aus_text, doku_bezeichner
from selbstbild.analyse.erreichbar import erreichbarkeit
from selbstbild.analyse.lader import laden
from selbstbild.analyse.tot import TotFund, tot_finden
from selbstbild.typen import Kante, Knoten, id_datei, id_doku_seite, id_symbol

STICHTAG = 1_700_000_000
TAG = 86400


def _d(pfad: str, alter_tage: int | None = 120, **extra: str) -> Knoten:
    attr = dict(extra)
    if alter_tage is not None:
        attr["zuletzt"] = str(STICHTAG - alter_tage * TAG)
    return Knoten(id_datei("r", pfad), "datei", pfad, None, tuple(sorted(attr.items())))


def _s(pfad: str, q: str, art: str = "funktion", **extra: str) -> Knoten:
    attr = {"symbolart": art, **extra}
    return Knoten(id_symbol("r", pfad, q), "symbol", q, None, tuple(sorted(attr.items())))


def _i(pfad: str, q: str) -> str:
    return id_symbol("r", pfad, q)


def _ruft(a: str, b: str, sicher: bool = True) -> Kante:
    return Kante("ruft_auf", a, b, sicher, None)


def _finden(graph_stand, knoten, kanten=(), doku=None) -> dict[str, TotFund]:
    d = laden(graph_stand(knoten, list(kanten), commit_zeit=STICHTAG))
    erg = tot_finden(d, erreichbarkeit(d), doku or {})
    assert list(erg) == sorted(erg, key=lambda f: f.symbol)
    return {f.symbol: f for f in erg}


def test_sicher_tot(graph_stand):
    f = _finden(graph_stand, [_d("a.py"), _s("a.py", "alt_fn")])[_i("a.py", "alt_fn")]
    assert f.stufe == "sicher_tot"
    assert f.gruende == ("alt_90", "kein_doku_verweis", "unerreichbar")
    assert f.zuletzt == STICHTAG - 120 * TAG


def test_jung_ist_wahrscheinlich(graph_stand):
    f = _finden(graph_stand, [_d("a.py", 10), _s("a.py", "alt_fn")])[_i("a.py", "alt_fn")]
    assert f.stufe == "wahrscheinlich_tot"
    assert "unerreichbar" in f.gruende and "alt_90" not in f.gruende


def test_doku_verweis_rettet(graph_stand):
    f = _finden(graph_stand, [_d("a.py"), _s("a.py", "rechne_alles")],
                doku={"r": frozenset({"rechne_alles"})})[_i("a.py", "rechne_alles")]
    assert f.stufe == "wahrscheinlich_tot"
    assert "doku_verweis" in f.gruende and "kein_doku_verweis" not in f.gruende


def test_rust_nie_sicher(graph_stand):
    f = _finden(graph_stand, [_d("a.rs"), _s("a.rs", "alt_fn")])[_i("a.rs", "alt_fn")]
    assert f.stufe == "wahrscheinlich_tot" and "rust_r22" in f.gruende


def test_spezial_fehlt_nie_sicher(graph_stand):
    f = _finden(graph_stand, [_d("a.py", spezial_fehlt="1"), _s("a.py", "alt_fn")])[_i("a.py", "alt_fn")]
    assert f.stufe == "wahrscheinlich_tot" and "spezial_fehlt" in f.gruende


def test_fehlerhaft_nie_sicher(graph_stand):
    f = _finden(graph_stand, [_d("a.py", fehlerhaft="1"), _s("a.py", "alt_fn")])[_i("a.py", "alt_fn")]
    assert f.stufe == "wahrscheinlich_tot" and "fehlerhaft" in f.gruende


def test_dunder_nie_sicher(graph_stand):
    f = _finden(graph_stand, [_d("a.py"), _s("a.py", "K", "klasse"), _s("a.py", "K.__repr__", "methode")])
    assert f[_i("a.py", "K.__repr__")].stufe == "wahrscheinlich_tot"
    assert "dunder" in f[_i("a.py", "K.__repr__")].gruende
    assert f[_i("a.py", "K")].stufe == "sicher_tot"


def test_dynamischer_aufruf_regel(graph_stand):
    knoten = [_d("plugins/x.py"), _s("plugins/x.py", "plugin_x")]
    assert _finden(graph_stand, knoten)[_i("plugins/x.py", "plugin_x")].stufe == "sicher_tot"
    knoten = [_d("plugins/x.py"), _s("plugins/x.py", "plugin_x", einstieg="regel")]
    assert _finden(graph_stand, knoten) == {}


def test_nur_test_verdaechtig(graph_stand):
    erg = _finden(graph_stand,
                  [_d("h.py"), _s("h.py", "helfer"), _d("tests/t.py"), _s("tests/t.py", "test_h", einstieg="test")],
                  [_ruft(_i("tests/t.py", "test_h"), _i("h.py", "helfer"))])
    assert erg[_i("h.py", "helfer")].stufe == "verdaechtig"
    assert erg[_i("h.py", "helfer")].gruende == ("nur_test",)
    assert _i("tests/t.py", "test_h") not in erg  # Testcode selbst ist kein Fund


def test_unsicher_alt_python_wahrscheinlich(graph_stand):
    erg = _finden(graph_stand, [_d("a.py"), _s("a.py", "main", einstieg="main"), _s("a.py", "x")],
                  [_ruft(_i("a.py", "main"), _i("a.py", "x"), sicher=False)])
    assert erg[_i("a.py", "x")].stufe == "wahrscheinlich_tot"
    assert "nur_unsicher" in erg[_i("a.py", "x")].gruende
    assert _i("a.py", "main") not in erg


def test_unsicher_ts_ist_kein_fund(graph_stand):
    erg = _finden(graph_stand, [_d("a.ts"), _s("a.ts", "main", einstieg="main"), _s("a.ts", "x")],
                  [_ruft(_i("a.ts", "main"), _i("a.ts", "x"), sicher=False)])
    assert erg == {}


def test_methode_toter_klasse_sicher_lebender_nicht(graph_stand):
    erg = _finden(graph_stand,
                  [_d("a.py"), _s("a.py", "main", einstieg="main"), _s("a.py", "K", "klasse"),
                   _s("a.py", "K.m", "methode"), _s("a.py", "L", "klasse"), _s("a.py", "L.m", "methode")],
                  [_ruft(_i("a.py", "main"), _i("a.py", "K"))])
    assert _i("a.py", "K.m") not in erg
    assert erg[_i("a.py", "L.m")].stufe == "sicher_tot"


def test_s4_go_methode(graph_stand):
    erg = _finden(graph_stand, [_d("a.go"), _s("a.go", "T.Lauf", "methode"), _s("a.go", "alt_fn")])
    assert erg[_i("a.go", "T.Lauf")].stufe == "wahrscheinlich_tot"
    assert "go_methode" in erg[_i("a.go", "T.Lauf")].gruende
    assert erg[_i("a.go", "alt_fn")].stufe == "sicher_tot"


def test_s4_ts_klasse_und_methode(graph_stand):
    erg = _finden(graph_stand, [_d("a.ts"), _s("a.ts", "K", "klasse"), _s("a.ts", "K.m", "methode"),
                                _s("a.ts", "alt_fn")])
    for q in ("K", "K.m"):
        assert erg[_i("a.ts", q)].stufe == "wahrscheinlich_tot"
        assert "ts_klasse" in erg[_i("a.ts", q)].gruende
    assert erg[_i("a.ts", "alt_fn")].stufe == "sicher_tot"


def test_s4_jsx_komponente(graph_stand):
    erg = _finden(graph_stand, [_d("a.tsx"), _s("a.tsx", "Knopf"), _s("a.tsx", "hilfe")])
    assert erg[_i("a.tsx", "Knopf")].stufe == "wahrscheinlich_tot"
    assert "jsx_komponente" in erg[_i("a.tsx", "Knopf")].gruende
    assert erg[_i("a.tsx", "hilfe")].stufe == "sicher_tot"


def test_nur_symbole_und_ausgabe_deterministisch(graph_stand):
    knoten = [_d("a.py"), _s("a.py", "zz"), _s("a.py", "aa")]
    d = laden(graph_stand(knoten, [], commit_zeit=STICHTAG))
    e = erreichbarkeit(d)
    erg = tot_finden(d, e, {})
    assert [f.symbol for f in erg] == [_i("a.py", "aa"), _i("a.py", "zz")]
    assert erg == tot_finden(d, e, {})


def test_doku_bezeichner_committed(git_repo):
    git_repo.datei("docs/a.md", "Siehe `rechne_alles` und das hier_ist_Code sowie MeineKlasse.\n")
    sha = git_repo.commit("doku")
    git_repo.datei("docs/a.md", "nur_im_arbeitsbaum `Geheim` ohne commit\n")
    ids = doku_bezeichner(git_repo.wurzel, sha)["repo"]
    assert {"rechne_alles", "hier_ist_Code", "MeineKlasse"} <= ids
    assert "nur_im_arbeitsbaum" not in ids and "Geheim" not in ids


def test_doku_bezeichner_nur_doku_seiten_aus_graph(git_repo, graph_stand):
    git_repo.datei("docs/a.md", "`in_doku` text\n")
    git_repo.datei("README.md", "`andere_seite`\n")
    sha = git_repo.commit("doku")
    d = laden(graph_stand([Knoten(id_doku_seite("repo", "docs/a.md"), "doku_seite", "docs/a.md", None)],
                          [], commit_zeit=STICHTAG))
    assert doku_bezeichner(git_repo.wurzel, sha, d)["repo"] == frozenset({"in_doku"})
    assert "andere_seite" in doku_bezeichner(git_repo.wurzel, sha)["repo"]  # Regeln: **/*.md


def test_s4_mts_klasse(graph_stand):
    erg = _finden(graph_stand, [_d("a.mts"), _s("a.mts", "K", "klasse"), _s("a.cjs", "x")])
    assert erg[_i("a.mts", "K")].stufe == "wahrscheinlich_tot"
    assert "ts_klasse" in erg[_i("a.mts", "K")].gruende


def test_s4_grossbuchstabe_in_js(graph_stand):
    erg = _finden(graph_stand, [_d("a.js"), _s("a.js", "Konstrukteur"), _s("a.js", "klein")])
    assert "jsx_komponente" in erg[_i("a.js", "Konstrukteur")].gruende
    assert erg[_i("a.js", "Konstrukteur")].stufe == "wahrscheinlich_tot"
    assert erg[_i("a.js", "klein")].stufe == "sicher_tot"


def test_s5_eltern_repo_doku(graph_stand):
    sym = id_symbol("vibemind/vibemind-os", "x.py", "rechne")
    kn = [Knoten(id_datei("vibemind/vibemind-os", "x.py"), "datei", "x.py", None,
                 (("zuletzt", str(STICHTAG - 120 * TAG)),)),
          Knoten(sym, "symbol", "rechne", None, (("symbolart", "funktion"),))]
    d = laden(graph_stand(kn, [], commit_zeit=STICHTAG))
    e = erreichbarkeit(d)
    assert tot_finden(d, e, {})[0].stufe == "sicher_tot"
    assert tot_finden(d, e, {"vibemind": frozenset({"rechne"})})[0].stufe == "wahrscheinlich_tot"
    assert tot_finden(d, e, {"vibemind/vibemind-os": frozenset({"rechne"})})[0].stufe == "wahrscheinlich_tot"
    assert tot_finden(d, e, {"vibemind/andere": frozenset({"rechne"})})[0].stufe == "sicher_tot"
    assert tot_finden(d, e, {"vibemind-x": frozenset({"rechne"})})[0].stufe == "sicher_tot"


def test_unbekanntes_alter_stuft_ab(graph_stand):
    for extra in ({"aenderungen": "0"}, {}):
        erg = _finden(graph_stand, [_d("a.py", None, **extra), _s("a.py", "f")])
        assert erg[_i("a.py", "f")].stufe == "wahrscheinlich_tot"
    k = Knoten(id_datei("r", "b.py"), "datei", "b.py", None, (("zuletzt", "0"),))
    erg = _finden(graph_stand, [k, _s("b.py", "f")])
    assert erg[_i("b.py", "f")].stufe == "wahrscheinlich_tot"
    erg = _finden(graph_stand, [_d("c.py", aenderungen="3"), _s("c.py", "f")])
    assert erg[_i("c.py", "f")].stufe == "sicher_tot"


def test_genau_90_tage(graph_stand):
    assert _finden(graph_stand, [_d("a.py", 90), _s("a.py", "f")])[_i("a.py", "f")].gruende[0] == "alt_90"
    assert "alt_90" not in _finden(graph_stand, [_d("a.py", 89), _s("a.py", "f")])[_i("a.py", "f")].gruende


def test_doku_ende_zu_ende(git_repo, graph_stand):
    git_repo.datei("docs/a.md", "Siehe `rechne_alles` hier.\n")
    sha = git_repo.commit("doku")
    doku = doku_bezeichner(git_repo.wurzel, sha)
    knoten = [Knoten(id_datei("repo", "x.py"), "datei", "x.py", None, (("zuletzt", str(STICHTAG - 200 * TAG)),)),
              Knoten(id_symbol("repo", "x.py", "rechne_alles"), "symbol", "rechne_alles", None,
                     (("symbolart", "funktion"),)),
              Knoten(id_symbol("repo", "x.py", "weg"), "symbol", "weg", None, (("symbolart", "funktion"),))]
    d = laden(graph_stand(knoten, [], commit_zeit=STICHTAG))
    erg = {f.symbol: f for f in tot_finden(d, erreichbarkeit(d), doku)}
    assert erg[id_symbol("repo", "x.py", "rechne_alles")].stufe == "wahrscheinlich_tot"
    assert erg[id_symbol("repo", "x.py", "weg")].stufe == "sicher_tot"


def test_punktketten_in_prosa():
    ids = bezeichner_aus_text("Rufe Klasse.methode auf, dann mein_modul.funktion und modul.tuwas(1). Ende.")
    assert {"Klasse", "methode", "mein_modul", "funktion", "tuwas"} <= ids
    assert "Ende" not in ids and "dann" not in ids


def test_conftest_ist_testdatei():
    assert ist_testdatei("conftest.py") and ist_testdatei("pkg/conftest.py")
    assert not ist_testdatei("contest.py")


def test_ist_testdatei_erweiterung_s2():
    for p in ("pkg/test/helfer.py", "web/spec/a.ts", "e2e/login.ts", "src/__mocks__/api.js",
              "a/b.test.jsx", "a/b.test.mjs", "a/b.test.cjs", "a/b.test.mts", "a/b.test.cts",
              "ui/Button.stories.tsx", "ui/Button.stories.mdx"):
        assert ist_testdatei(p), p
    for p in ("src/testing.py", "src/specs.py", "src/e2e.py", "a/b.test", "src/latest.py", "contest/x.py"):
        assert not ist_testdatei(p), p


def test_doku_bezeichner_unicode():
    ids = bezeichner_aus_text("Die Funktion `größe_berechnen` und straßen_name werden genutzt.")
    assert {"größe_berechnen", "straßen_name"} <= ids
