"""doku_loop: Beleg-Drift-Messung (Task 12, Abnahme C.9.1)."""

import dataclasses
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden, standard_konfig  # noqa: E402,F401

pytestmark = pytest.mark.usefixtures("standard_konfig")
from selbstbild.doku import ketten_seed as ks  # noqa: E402
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import messen  # noqa: E402
from selbstbild.doku import render_ketten as rk  # noqa: E402
from selbstbild.doku.extraktoren.sql import SqlDefinition  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402

KETTEN = REPO / "docs" / "spaces" / "KETTEN.md"
KOPF = "| # | Von → Nach | Mechanismus | Code-Stelle | Belegt durch | Zustand |"
TRENN = "|---|---|---|---|---|---|"

API = ("def routen():\n    return ziel_aufruf()\n"  # 1-2
       + "\n" * 17  # 3-19
       + "def andere_funktion():\n    return 1\n")  # 20-21


def _fixture(tmp_path):
    (tmp_path / "sp/m").mkdir(parents=True)
    (tmp_path / "sp/m/api.py").write_text(API, encoding="utf-8")
    md = "\n".join([
        "# Ketten", "", "### K1 · Eins", "", KOPF, TRENN,
        "| 1 | A → B | Python `ziel_aufruf` | `sp/m/api.py:2` | 008 | läuft |",
        "| 2 | A → B | Python `andere_funktion` | `sp/m/api.py:1` | 008 | läuft |",
        "| 3 | A → B | Python `gibt_es_nicht` | `sp/m/api.py:2` | 008 | läuft |",
        "| 4 | A → B | Python `ziel_aufruf` | `sp/m/weg.py:2` | 008 | läuft |",
        "| 5 | A → B | SQL `marketing.fn` | `043…sql:297` | 008 | läuft |",
        "", "Der Aufruf `ziel_aufruf` steht in `sp/m/api.py:3`; der Sidecar lauscht auf `:8131`.", "",
    ])
    (tmp_path / "docs").mkdir()
    pfad = tmp_path / "docs/KETTEN.md"
    pfad.write_bytes(md.encode("utf-8"))
    return pfad


def test_belege_messen_fixture_vier_klassen(tmp_path):
    befunde = messen.belege_messen(_fixture(tmp_path), tmp_path)
    assert [(b.zeile, b.kette, b.stelle, b.klasse, b.gefunden_zeile, b.token) for b in befunde] == [
        (7, "K1", "sp/m/api.py:2", "exakt", 2, "ziel_aufruf"),
        (8, "K1", "sp/m/api.py:1", "verschoben", 20, "andere_funktion"),
        (9, "K1", "sp/m/api.py:2", "token_fehlt", None, None),
        (10, "K1", "sp/m/weg.py:2", "datei_fehlt", None, None),
        (11, "K1", "043…sql:297", "nicht_aufloesbar", None, None),
        (13, "K1", "sp/m/api.py:3", "exakt", 2, "ziel_aufruf"),  # Prosa; der Port `:8131` ist kein Beleg
    ]
    assert not any(b.gerendert for b in befunde)
    assert set(messen.KLASSEN) == {"exakt", "verschoben", "token_fehlt", "datei_fehlt", "nicht_aufloesbar"}


def test_bericht_zaehlt_je_klasse_getrennt_nach_roh_und_gerendert(tmp_path):
    befunde = messen.belege_messen(_fixture(tmp_path), tmp_path)
    text = messen.bericht(befunde)
    zeilen = text.splitlines()
    assert zeilen[0] == "Belege: 6 (roh 6, gerendert 0)"
    assert zeilen[1] == "roh: exakt 2 · verschoben 1 · token_fehlt 1 · datei_fehlt 1 · nicht_aufloesbar 1"
    assert zeilen[2] == "gerendert: exakt 0 · verschoben 0 · token_fehlt 0 · datei_fehlt 0 · nicht_aufloesbar 0"
    assert zeilen[3].startswith("gerendert mit Anker, C.9.1 (0): ")
    assert zeilen[4].startswith("gerendert ohne Anker (0): ")
    assert any("Z8" in z and "verschoben" in z and "-> 20" in z and "andere_funktion" in z for z in zeilen)
    assert messen.bericht([]).splitlines()[0] == "Belege: 0 (roh 0, gerendert 0)"


# ---------- gerenderte Bloecke ----------

def _render_repo(tmp_path):
    (tmp_path / "sp/m").mkdir(parents=True)
    (tmp_path / "sp/m/api.py").write_text(
        "def routen():\n    return ziel_aufruf()\n\n\ndef andere():\n    x = 1\n    return x\n", encoding="utf-8")
    (tmp_path / "db").mkdir()
    (tmp_path / "db/001.sql").write_text(
        "-- Kopf\n\nCREATE OR REPLACE FUNCTION marketing.fn()\nRETURNS void AS $$ SELECT 1 $$ LANGUAGE sql;\n",
        encoding="utf-8")
    (tmp_path / "caps.yaml").write_text(
        "# Kopf\ncaps:\n  - capability: send_mail\n    target: x\n  - capability: other_cap\n    target: y\n",
        encoding="utf-8")
    konfig = dataclasses.replace(vm_laden(), repo=tmp_path,
                                 spaces={"m": SpaceKonfig("m", "sp/m", "werkzeuge")}, ketten_md="docs/KETTEN.md")
    docs = (SimpleNamespace(document_id="m.api.routen", file="api.py", start_line=1, end_line=2, symbol="routen",
                            source="def routen():\n    return ziel_aufruf()"),
            SimpleNamespace(document_id="m.api.andere", file="api.py", start_line=5, end_line=7, symbol="andere",
                            source="def andere():\n    x = 1\n    return x"))
    manifeste = {"m": SimpleNamespace(corpus_root=str(tmp_path / "sp/m"), documents=docs)}
    sql_defs = {"marketing.fn": SqlDefinition("db/001.sql", 3, "marketing.fn", "function", "d")}
    md = "\n".join([
        "# Ketten", "", "### K1 · Eins", "", KOPF, TRENN,
        # Zeilennummern im Dokument absichtlich daneben: das Rendern schreibt die aktuellen
        "| 1 | A → B | Python `routen` | `sp/m/api.py:2` | 008 | läuft |",
        "| 2 | A → B | Python `andere` | `sp/m/api.py:7`, `:6` | 008 | läuft |",
        "| 3 | A → B | SQL `marketing.fn` | `db/001.sql:4` | 008 | läuft |",
        "| 4 | A → B | capability `send_mail` | `caps.yaml:4` | 008 | läuft |",
        "", "Prosa danach.", "",
    ])
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/KETTEN.md").write_bytes(md.encode("utf-8"))
    g, _ = ks.seed_graph(konfig, manifeste, sql_defs, "a", "b")
    frisch = rk.belege_auffrischen(g, manifeste, konfig, sql_defs)
    eingebettet = rk.einbetten(md, frisch)
    text = rk.rendern(eingebettet, frisch, {}).text
    ziel = tmp_path / "docs/KETTEN.md"
    ziel.write_bytes(text.encode("utf-8"))
    return ziel, text


def test_belege_messen_gerendert_null_verschoben_null_token_fehlt(tmp_path):
    ziel, text = _render_repo(tmp_path)
    assert "`sp/m/api.py:1` `m.api.routen`" in text and "`db/001.sql:3` `marketing.fn`" in text
    befunde = messen.belege_messen(ziel, tmp_path)
    gerendert = [b for b in befunde if b.gerendert]
    assert len(gerendert) == 4 and len(befunde) == 4
    assert Counter(b.klasse for b in gerendert) == {"exakt": 4} and all(b.hat_anker for b in gerendert)
    assert {(b.stelle, b.token) for b in gerendert} == {
        ("sp/m/api.py:1", "m.api.routen"), ("sp/m/api.py:5", "m.api.andere"),
        ("db/001.sql:3", "marketing.fn"), ("caps.yaml:3", "caps[capability=send_mail]")}
    text = messen.bericht(befunde)
    assert "gerendert: exakt 4 · verschoben 0 · token_fehlt 0" in text
    assert "gerendert mit Anker, C.9.1 (4): exakt 4 · verschoben 0 · token_fehlt 0" in text


def test_belege_messen_gerendert_anker_verschoben_und_verschwunden(tmp_path):
    ziel, _ = _render_repo(tmp_path)
    api = tmp_path / "sp/m/api.py"
    api.write_text("import os\n\n\n\n\n" + api.read_text(encoding="utf-8").replace("def andere", "def anders"),
                   encoding="utf-8")
    befunde = {b.token: b for b in messen.belege_messen(ziel, tmp_path)}
    assert (befunde["m.api.routen"].klasse, befunde["m.api.routen"].gefunden_zeile) == ("verschoben", 6)
    assert (befunde["m.api.andere"].klasse, befunde["m.api.andere"].gefunden_zeile) == ("token_fehlt", None)
    assert befunde["marketing.fn"].klasse == "exakt"


def _block_md(tmp_path, zeilen):
    """KETTEN.md mit einem von Hand markierten Block (gerenderte Zeilen) unter K1."""
    md = "\n".join(["# Ketten", "", "### K1 · Eins", "", "<!-- graph:begin hops K1 -->", KOPF, TRENN,
                    *[f"| {i} | A → B | Python `x` | {code} | 008 | läuft · v:? g:— |"
                      for i, code in enumerate(zeilen, 1)],
                    "<!-- graph:end -->", ""])
    (tmp_path / "docs").mkdir(exist_ok=True)
    ziel = tmp_path / "docs/KETTEN.md"
    ziel.write_bytes(md.encode("utf-8"))
    return ziel


def test_hat_anker_auch_bei_fehlender_datei_kaputtem_py_und_fremder_endung(tmp_path):
    (tmp_path / "sp").mkdir()
    (tmp_path / "sp/kaputt.py").write_text("def f(:\n    pass\n", encoding="utf-8")
    (tmp_path / "daten").mkdir()
    (tmp_path / "daten/x.json").write_text('{"foo": 1}\n', encoding="utf-8")
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf/a.toml").write_text('[[agents]]\nname = "alpha"\nx = 1\n\n[[agents]]\nname = "beta"\n',
                                          encoding="utf-8")
    (tmp_path / "sp/neu.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    ziel = _block_md(tmp_path, ["`sp/weg.py:1` `weg.f`", "`sp/kaputt.py:1` `kaputt.f`", "`daten/x.json:1` `foo`",
                                "`conf/a.toml:5` `agents[name=beta]`", "⚠ `sp/neu.py:1` — ohne Anker"])
    befunde = messen.belege_messen(ziel, tmp_path)
    assert [(b.zeile, b.klasse, b.hat_anker, b.gerendert) for b in befunde] == [
        (8, "datei_fehlt", True, True), (9, "token_fehlt", True, True), (10, "token_fehlt", True, True),
        (11, "exakt", True, True), (12, "token_fehlt", False, True)]
    assert befunde[3].gefunden_zeile == 5
    text = messen.bericht(befunde)
    assert "gerendert mit Anker, C.9.1 (4): exakt 1 · verschoben 0 · token_fehlt 2 · datei_fehlt 1" in text
    assert "gerendert ohne Anker (1): exakt 0 · verschoben 0 · token_fehlt 1" in text


def test_py_anker_kuerzere_endung_nur_eindeutig_mit_zwei_segmenten(tmp_path):
    (tmp_path / "sp").mkdir()
    (tmp_path / "sp/neu.py").write_text(
        "class Klasse:\n    def methode(self):\n        return 1\n\n\ndef routen():\n    return 2\n", encoding="utf-8")
    (tmp_path / "sp/doppelt.py").write_text(
        "class Klasse:\n    def methode(self):\n        return 1\n\n\n"
        "class Klasse:\n    def methode(self):\n        return 2\n", encoding="utf-8")
    (tmp_path / "sp/paket").mkdir()
    (tmp_path / "sp/paket/__init__.py").write_text("\n\ndef sync():\n    pass\n", encoding="utf-8")
    ziel = _block_md(tmp_path, [
        "`sp/paket/__init__.py:3` `sp.paket.__init__.sync`",  # AST-id ausserhalb der Spaces
        "`sp/paket/__init__.py:3` `x.paket.sync`",  # Manifest-Stil: Paketname
        "`sp/neu.py:6` `neu.routen`",  # voll: Modulname + Qualname
        "`sp/neu.py:6` `alt.routen`",  # anderes Modul, nackter Name: kein Anker
        "`sp/neu.py:2` `alt.Klasse.methode`",  # Datei umbenannt, Qualname mit 2 Segmenten eindeutig
        "`sp/doppelt.py:2` `alt.Klasse.methode`",  # mehrdeutig: kein Anker
        "`sp/doppelt.py:7` `doppelt.Klasse.methode`",  # voll, doppelt: der naechste
    ])
    befunde = messen.belege_messen(ziel, tmp_path)
    assert [(b.klasse, b.gefunden_zeile) for b in befunde] == [
        ("exakt", 3), ("exakt", 3), ("exakt", 6),("token_fehlt", None), ("exakt", 2), ("token_fehlt", None), ("exakt", 7)]


# ---------- echte KETTEN.md ----------

def test_belege_messen_echte_datei_mindestens_290_belege():
    if not KETTEN.is_file() or not (REPO / "vibemind-os/spaces/marketing").is_dir():
        pytest.skip("KETTEN.md oder Submodul fehlt")
    if "<!-- graph:begin" in KETTEN.read_bytes().decode("utf-8"):
        pytest.skip("KETTEN.md ist bereits gerendert")
    befunde = messen.belege_messen(KETTEN, REPO)
    zahl = Counter(b.klasse for b in befunde)
    assert len(befunde) >= 290, zahl
    assert zahl["exakt"] >= 190 and zahl["verschoben"] >= 70, zahl
    assert set(zahl) <= set(messen.KLASSEN)
