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
BUNDLE = "function verpackt() {\n  return einsam_fn();\n}\n"  # generiert: rettet einsam_fn nicht
SEITE_JS = "function alpine_seite() {\n  return { offen: false };\n}\n"
SEITE_HTML = '<div x-data="alpine_seite"><button onclick="klick_html()">x</button></div>\n'
KLICK_JS = "function klick_html() {\n  return 1;\n}\n"
SELBST = 'def helfer_x():\n    return 1\n\n\nTABELLE = {"h": helfer_x}\n'


@pytest.fixture
def heim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("SELBSTBILD_HOME", str(h))
    return h


def _repo(git_repo) -> None:
    for pfad, text in (("typen.py", TYPEN), ("nutzer.py", NUTZER), ("klasse.py", KLASSE),
                       ("rufer.py", RUFER), ("einsam.py", EINSAM), ("selbst.py", SELBST),
                       ("pkg/dist-embedded/x.js", BUNDLE), ("web/seite.js", SEITE_JS),
                       ("web/seite.html", SEITE_HTML), ("web/klick.js", KLICK_JS)):
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
    # HTML-Vorlagen verweisen per Name (Alpine x-data, onclick): S8-Erweiterung
    for q in (("web/seite.js", "alpine_seite"), ("web/klick.js", "klick_html")):
        f = tot[_s(*q)]
        assert f.stufe == "wahrscheinlich_tot" and "namensverweis" in f.gruende, q


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


def _deko(pfad: str, q: str, art: str, dekoriert: str) -> Knoten:
    return Knoten(id_symbol("r", pfad, q), "symbol", q, None, (("dekoriert", dekoriert), ("symbolart", art)))


def test_dekorator_nie_sicher_tot_ausser_inert_s10(graph_stand):
    """Ruling S10: Framework-Registrierung per Decorator erzeugt keine Kante."""
    knoten = [_d("op.py"), _deko("op.py", "cell_health_timer", "funktion", "kopf.timer"),
              _deko("op.py", "list_installed", "funktion", "marketplace.command,property"),
              _deko("op.py", "nur_inert", "funktion", "functools.lru_cache,property,staticmethod"),
              _deko("op.py", "Daten", "klasse", "dataclass,typing.final"),
              _deko("op.py", "Registriert", "klasse", "registry.register"),
              _sym("op.py", "Registriert.innen", "methode")]
    d = laden(graph_stand(knoten, [], commit_zeit=STICHTAG))
    tot = {t.symbol: t for t in tot_finden(d, erreichbarkeit(d), {})}
    for q in ("cell_health_timer", "list_installed", "Registriert", "Registriert.innen"):
        f = tot[id_symbol("r", "op.py", q)]
        assert f.stufe == "wahrscheinlich_tot" and "dekorator" in f.gruende, q
    for q in ("nur_inert", "Daten"):
        f = tot[id_symbol("r", "op.py", q)]
        assert f.stufe == "sicher_tot" and "dekorator" not in f.gruende, q


@pytest.mark.parametrize("pfad, q, art", [
    ("svc/main.go", "init", "funktion"), ("conftest_plugin.py", "pytest_configure", "funktion"),
    ("alembic/versions/001_x.py", "upgrade", "funktion"), ("alembic/versions/001_x.py", "downgrade", "funktion"),
    ("aws/fn.py", "lambda_handler", "funktion"), ("app/management/commands/sync.py", "Command.handle", "methode"),
])
def test_konvention_ergaenzt(graph_stand, pfad, q, art):
    knoten = [_d(pfad), _sym(pfad, q, art)]
    if "." in q:
        knoten.append(_sym(pfad, q.split(".")[0], "klasse"))
    d = laden(graph_stand(knoten, [], commit_zeit=STICHTAG))
    f = {t.symbol: t for t in tot_finden(d, erreichbarkeit(d), {})}[id_symbol("r", pfad, q)]
    assert f.stufe == "wahrscheinlich_tot" and "konvention" in f.gruende


def test_init_nur_in_go_konvention(graph_stand):
    knoten = [_d("x.py"), _sym("x.py", "init", "funktion")]
    d = laden(graph_stand(knoten, [], commit_zeit=STICHTAG))
    f = {t.symbol: t for t in tot_finden(d, erreichbarkeit(d), {})}[id_symbol("r", "x.py", "init")]
    assert f.stufe == "sicher_tot"


KONFIG = {
    "pyproject.toml": '[project.scripts]\nwerkzeug = "pkg.cli:aus_toml"\n',
    "setup.cfg": "[options.entry_points]\nconsole_scripts =\n    w = pkg.cli:aus_cfg\n",
    "tox.ini": "[testenv]\ncommands = python -c 'import pkg; pkg.aus_ini()'\n",
    "Dockerfile.dev": 'CMD ["python", "-c", "import pkg; pkg.aus_docker()"]\n',
    "Makefile": "lauf:\n\tpython -c 'import pkg; pkg.aus_make()'\n",
    "Procfile": "web: python -c 'import pkg; pkg.aus_procfile()'\n",
    "deploy/job.yaml": "handler: pkg.cli.aus_yaml\n",
    "conf/plugins.json": '{"hook": "aus_json"}\n',
    "web/package-lock.json": '{"x": "aus_lock"}\n',
}
NAMEN = ("aus_toml", "aus_cfg", "aus_ini", "aus_docker", "aus_make", "aus_procfile", "aus_yaml", "aus_json",
         "aus_lock", "größe_berechnen", "nur_hier")
CLI = "".join(f"def {n}():\n    return 1\n\n\n" for n in NAMEN)


def test_s8_liest_konfig_und_unicode_namen(git_repo, heim):
    """Fix-Welle: S8 liest Konfigurations- und Datendateien (ohne Lock-Dateien); Bezeichner Unicode."""
    git_repo.datei("pkg/cli.py", CLI)
    git_repo.datei("pkg/nutzer.py", "from pkg.cli import *\n\n\ndef f():\n    return größe_berechnen()\n")
    for pfad, text in KONFIG.items():
        git_repo.datei(pfad, text)
    git_repo.commit("alt", datum="2025-01-01T00:00:00Z")
    vermessen(git_repo.wurzel)
    from selbstbild.vermessen import stand_finden

    d = laden(stand_finden(git_repo.wurzel).parent)
    kandidaten = [_s("pkg/cli.py", n) for n in NAMEN]
    gerettet = namensverweise(git_repo.wurzel, d.commit, kandidaten)
    erwartet = {_s("pkg/cli.py", n) for n in NAMEN if n not in ("aus_lock", "nur_hier")}
    assert gerettet == erwartet
