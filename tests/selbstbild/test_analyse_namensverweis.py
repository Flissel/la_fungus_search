"""Ruling S8: Namensverweis-Waechter und Vererbung der S4-Ausnahmen."""
from __future__ import annotations

from pathlib import Path

import pytest

from selbstbild.analyse.erreichbar import erreichbarkeit
from selbstbild.analyse.lader import laden
from selbstbild.analyse.lauf import analysieren
from selbstbild.analyse.tot import namensverweise, tot_finden
from selbstbild.typen import Knoten, id_datei, id_symbol
from selbstbild.vermessen import vermessen

STICHTAG = 1_700_000_000
TAG = 86400

TYPEN = "class NurTyp:\n    wert: int = 1\n"
NUTZER = "from typen import NurTyp\n\n\ndef nutze_typ(a: NurTyp) -> None:\n    return None\n"
KLASSE = "class Dienst:\n    def methode(self):\n        return 1\n"
RUFER = "def rufe_auf(o):\n    return o.methode()\n"
EINSAM = "def einsam_fn():\n    x = 1\n    return x\n"
BUNDLE = "function verpackt() {\n  return 1;\n}\n"
SELBST = 'def helfer_x():\n    return 1\n\n\nTABELLE = {"h": helfer_x}\n'


@pytest.fixture
def heim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("SELBSTBILD_HOME", str(h))
    return h


def _repo(git_repo) -> None:
    for pfad, text in (("typen.py", TYPEN), ("nutzer.py", NUTZER), ("klasse.py", KLASSE),
                       ("rufer.py", RUFER), ("einsam.py", EINSAM), ("selbst.py", SELBST),
                       ("pkg/dist-embedded/x.js", BUNDLE)):
        git_repo.datei(pfad, text)
    git_repo.commit("alt", datum="2025-01-01T00:00:00Z")
    git_repo.datei("README.md", "Ein Repo.\n")
    git_repo.commit("neu", datum="2026-01-01T00:00:00Z")


def _s(pfad: str, q: str) -> str:
    return id_symbol("repo", pfad, q)


def test_ohne_waechter_alles_sicher_tot(git_repo, heim):
    """Gegenprobe: ohne S8 waeren die Kandidaten sicher_tot (der Test misst also den Waechter)."""
    _repo(git_repo)
    vermessen(git_repo.wurzel)
    from selbstbild.vermessen import stand_finden

    d = laden(stand_finden(git_repo.wurzel).parent)
    roh = {t.symbol: t.stufe for t in tot_finden(d, erreichbarkeit(d), {})}
    for q in (("typen.py", "NurTyp"), ("klasse.py", "Dienst.methode"), ("selbst.py", "helfer_x")):
        assert roh[_s(*q)] == "sicher_tot", q
    gerettet = namensverweise(git_repo.wurzel, d.commit, [s for s, st in roh.items() if st == "sicher_tot"])
    assert _s("typen.py", "NurTyp") in gerettet
    assert _s("einsam.py", "einsam_fn") not in gerettet


def test_waechter_im_lauf(git_repo, heim):
    _repo(git_repo)
    vermessen(git_repo.wurzel)
    a = analysieren(git_repo.wurzel, kolonie=False)
    tot = {t.symbol: t for t in a.tot}
    # Ruling S9: generierte Dateien liefern keinen Fund, werden aber gezaehlt
    assert not any("dist-embedded" in s for s in tot)
    assert dict(a.statistik)["tot_generiert_uebersprungen"] == 1
    # Klasse nur als Typannotation in einer anderen Datei
    f = tot[_s("typen.py", "NurTyp")]
    assert f.stufe == "wahrscheinlich_tot" and "namensverweis" in f.gruende
    # obj.methode() in einer anderen Datei rettet die Methode
    f = tot[_s("klasse.py", "Dienst.methode")]
    assert f.stufe == "wahrscheinlich_tot" and "namensverweis" in f.gruende
    # Verweis in der eigenen Datei ausserhalb der Definition zaehlt
    assert tot[_s("selbst.py", "helfer_x")].stufe == "wahrscheinlich_tot"
    # Name nur an der eigenen Definition: bleibt sicher_tot
    assert tot[_s("einsam.py", "einsam_fn")].stufe == "sicher_tot"
    assert tot[_s("rufer.py", "rufe_auf")].stufe == "sicher_tot"
    assert tot[_s("klasse.py", "Dienst")].stufe == "sicher_tot"


def _d(pfad: str) -> Knoten:
    return Knoten(id_datei("r", pfad), "datei", pfad, None, (("zuletzt", str(STICHTAG - 120 * TAG)),))


def _sym(pfad: str, q: str, art: str) -> Knoten:
    return Knoten(id_symbol("r", pfad, q), "symbol", q, None, (("symbolart", art),))


def test_handler_in_jsx_komponente_erbt_ausnahme(graph_stand):
    knoten = [_d("App.tsx"), _sym("App.tsx", "Panel", "funktion"),
              _sym("App.tsx", "Panel.handleClick", "funktion"),
              _sym("App.tsx", "Panel.handleClick.innen", "funktion"),
              _d("lib.rs"), _sym("lib.rs", "modul", "funktion"), _sym("lib.rs", "modul.innen", "funktion")]
    d = laden(graph_stand(knoten, [], commit_zeit=STICHTAG))
    tot = {t.symbol: t for t in tot_finden(d, erreichbarkeit(d), {})}
    for q in ("Panel.handleClick", "Panel.handleClick.innen"):
        f = tot[id_symbol("r", "App.tsx", q)]
        assert f.stufe == "wahrscheinlich_tot" and "jsx_komponente" in f.gruende, q
    f = tot[id_symbol("r", "lib.rs", "modul.innen")]
    assert f.stufe == "wahrscheinlich_tot" and "rust_r22" in f.gruende


def test_konvention_nie_sicher_tot(graph_stand):
    """Ruling S9: Dispatch per Namenskonvention (http.server ruft do_<METHODE>) ist nie sicher_tot."""
    knoten = [_d("media_server.py"), _sym("media_server.py", "MediaHandler", "klasse"),
              _sym("media_server.py", "MediaHandler.do_OPTIONS", "methode"),
              _sym("media_server.py", "MediaHandler.hilfe", "methode"),
              _sym("media_server.py", "setUp", "funktion")]
    d = laden(graph_stand(knoten, [], commit_zeit=STICHTAG))
    tot = {t.symbol: t for t in tot_finden(d, erreichbarkeit(d), {})}
    f = tot[id_symbol("r", "media_server.py", "MediaHandler.do_OPTIONS")]
    assert f.stufe == "wahrscheinlich_tot" and "konvention" in f.gruende
    assert "konvention" in tot[id_symbol("r", "media_server.py", "setUp")].gruende
    assert tot[id_symbol("r", "media_server.py", "MediaHandler.hilfe")].stufe == "sicher_tot"


@pytest.mark.parametrize("pfad, erwartet", [
    ("pkg/dist-embedded/x.js", True), ("a/dist/b.js", True), ("web/build/x.py", True),
    ("x/out/y.js", True), ("app/.next/z.js", True), ("m/node_modules/p/i.js", True),
    ("lib/foo-dist/a.js", True), ("s/vendor.min.js", True), ("s/app.bundle.js", True),
    ("s/app-bundle.js", True), ("cov/coverage/x.js", True),
    ("src/distance.py", False), ("src/outline.py", False), ("builder/x.py", False), ("s/minimal.js", False),
])
def test_ist_generiert(pfad, erwartet):
    from selbstbild.analyse import ist_generiert

    assert ist_generiert(pfad) is erwartet
