"""Regeldatei mit `doku:`-Abschnitt (Stufe 4, T2-a): der stdlib-Leser der Hooks darf davon nicht blind werden.

Wuerde `regeln_lesen` bei `doku:` auf `{}` fallen, haette der Waechter keine Regeln mehr (ein Space im Root
wuerde nicht mehr blockiert) und `pruefen` wuerde rot.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from selbstbild.hooks import regeln_lesen
from selbstbild.regeln import regeln_laden
from selbstbild.struktur import pruefe_schreibziel, pruefen

SRC = Path(__file__).resolve().parents[2] / "src"
VORLAGE = Path(__file__).resolve().parent / "doku" / "fixtures" / "config" / "doku_loop.yaml"

# Kopie der aeusseren selbstbild.yaml (Stand vor dem doku-Abschnitt)
AUSSEN = """# Regeln fuer das Selbstbild (Paket `selbstbild` in Flissel/la_fungus_search).
# Gelesen wird nur die committete Fassung dieser Datei. Regeln gelten je Repo (R21):
# Submodule (vibemind-os usw.) nutzen ihre eigene selbstbild.yaml bzw. die Vorgaben.
# Vermessen wird je Repo, die Space-Regel gilt aber fuer den Arbeitsbaum des Waechters und den
# Skill "neuer-space": Sie lebt deshalb hier (vibemind-os hat keine eigene selbstbild.yaml und
# bekommt keine). Sie traegt nur Ort und Hinweis, `pflicht` und `verboten` sind leer, also
# beeinflusst sie die Vermessung und die Pflichtteil-Pruefung nicht.
name: vibemind
doku_pfade:
  - "docs/**/*.md"
struktur:
  # Spaces leben im Submodul vibemind-os; im Root dieses Repos duerfen keine entstehen
  - art: kein_space_im_root
    ort: "./"
    pflicht: []
    verboten:
      - "spaces/*/"
      - "*/capabilities.yaml"
    hinweis: "Spaces gehoeren nach vibemind-os/spaces/<name>/ (Vertrag: vibemind-os/spaces/_contract/<name>.contract.yaml)"
  # Nur Ort und Hinweis fuer den Skill und den Waechter; keine Pflichtteile, keine Verbote
  - art: space
    ort: "vibemind-os/spaces/*/"
    pflicht: []
    verboten: []
    hinweis: "Spaces gehoeren nach vibemind-os/spaces/<name>/ (Vertrag: vibemind-os/spaces/_contract/<name>.contract.yaml)"
dienste: config/dienste.yaml
issue_ziel:
  repo: Flissel/Vibemind_V1
  privat: true
max_dateigroesse_kb: 512
"""

KOMMENTAR = "# doku: ersetzt config/doku_loop.yaml, sobald die Umschaltung (Task 11) erfolgt ist.\n"


def _doku_abschnitt() -> str:
    zeilen = VORLAGE.read_text(encoding="utf-8").splitlines()
    return KOMMENTAR + "doku:\n" + "".join(f"  {z}\n" if z.strip() else "\n" for z in zeilen)


def _mit_doku() -> str:
    return AUSSEN.rstrip("\n") + "\n" + _doku_abschnitt()


def _struktur(text: str) -> list[dict]:
    return regeln_lesen(text)["struktur"]


def test_aussen_hat_struktur_regeln():
    assert len(_struktur(AUSSEN)) == 2


def test_doku_abschnitt_aendert_regeln_lesen_nicht():
    assert regeln_lesen(_mit_doku()) == regeln_lesen(AUSSEN) != {}


def test_doku_abschnitt_aendert_regeln_laden_nicht():
    vorher, nachher = regeln_laden(AUSSEN), regeln_laden(_mit_doku())
    assert [r.model_dump() for r in nachher.struktur] == [r.model_dump() for r in vorher.struktur]
    assert nachher.doku_pfade == vorher.doku_pfade and nachher.dienste == vorher.dienste


def test_regeln_laden_und_lesen_geben_dieselben_regeln():
    text = _mit_doku()
    pyd = [(r.art, r.ort, r.pflicht, r.verboten, r.hinweis) for r in regeln_laden(text).struktur]
    std = [(r["art"], r["ort"], r["pflicht"], r["verboten"], r.get("hinweis")) for r in _struktur(text)]
    assert pyd == std


def test_regeln_laden_nimmt_doku_als_zuordnung():
    assert isinstance(regeln_laden(_mit_doku()).doku, dict)


def test_doku_mitten_in_der_datei():
    # doku vor doku_pfade/struktur: die Regeln dahinter muessen weiter gelesen werden
    text = AUSSEN.replace("doku_pfade:", _doku_abschnitt() + "doku_pfade:", 1)
    assert regeln_lesen(text) == regeln_lesen(AUSSEN)


KNIFFE = [
    ("verschachtelt", "doku:\n  a:\n    b:\n      c: [1, 2]\n    d: {x: 1}\n"),
    ("liste_ohne_einzug", "doku:\n- a\n- b: 1\n  c: 2\n"),
    ("quote_mit_prozent", "doku:\n  marker: \"%LOCALAPPDATA%/x # kein kommentar\"\n  b: '%USERPROFILE%/y'\n"),
    ("kommentare", "doku:  # hinten\n  # mitten\n  a: 1  # rest\n\n  # leer davor\n  b: 2\n"),
    ("blockskalar", "doku:\n  text: |\n    struktur:\n      - art: x\n    name: boese\n  danach: 1\n"),
    ("blockskalar_gefaltet", "doku:\n  text: >-\n    zeile eins\n\n    ordner: bar\n  n: 2\n"),
    ("blockskalar_toplevel", "doku: |\n  struktur:\n    - art: x\nzwei: 1\n"),
    ("flow_mehrzeilig", "doku:\n  l: [a,\n    b,\n    c]\n"),
    ("flow_zuordnung_mehrzeilig", "doku:\n  m: {a: 1,\n    b: 2}\n"),
    ("flow_toplevel_mehrzeilig", "doku: [a,\n  b]\n"),
    ("mehrzeiliges_quote", "doku:\n  q: \"eins\n    zwei\"\n"),
    ("bindestrich_schluessel", "doku:\n  ein-schluessel: 1\nandere-sektion:\n  x: 1\n"),
    ("unbekannter_toplevel_skalar", "weiteres: 5\nnoch_eins: \"a: b\"\n"),
    ("unbekannter_toplevel_mit_ziffer", "v2:\n  a: 1\n"),
    ("anker", "doku:\n  a: &x [1, 2]\n  b: *x\n"),
    ("crlf", "doku:\r\n  a: 1\r\n  b:\r\n    - c\r\n"),
    ("dokument_ende", "doku:\n  a: 1\n...\n"),
]


@pytest.mark.parametrize("name,extra", KNIFFE, ids=[k[0] for k in KNIFFE])
def test_unbekannte_abschnitte_werden_uebersprungen(name, extra):
    erwartet = regeln_lesen(AUSSEN)
    assert erwartet
    # hinten angehaengt und vorn eingefuegt: die Struktur-Regeln bleiben gleich
    assert regeln_lesen(AUSSEN.rstrip("\n") + "\n" + extra) == erwartet
    assert regeln_lesen(extra + AUSSEN) == erwartet


def test_struktur_wird_nicht_von_doku_unterschluckt():
    # ein Eintrag mit art/ort unterhalb von doku gehoert nicht zu struktur
    text = AUSSEN.rstrip("\n") + "\ndoku:\n  struktur:\n    - art: falsch\n      ort: x/\n"
    assert [r["art"] for r in _struktur(text)] == [r["art"] for r in _struktur(AUSSEN)]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    (r / "selbstbild.yaml").write_text(_mit_doku(), encoding="utf-8")
    return r


def test_waechter_blockiert_space_im_root_mit_doku(repo):
    v = pruefe_schreibziel(repo / "spaces" / "foo" / "x.py")
    assert len(v) == 1 and v[0].hart and v[0].regel == "kein_space_im_root"


def test_waechter_prozess_blockiert_mit_doku(repo):
    ein = json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(repo / "spaces" / "foo" / "x.py")}})
    p = subprocess.run([sys.executable, "-m", "selbstbild.hooks.waechter"], input=ein.encode("utf-8"),
                       capture_output=True, env={**os.environ, "PYTHONPATH": str(SRC)}, timeout=60)
    assert p.returncode == 0
    assert json.loads(p.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pruefen_bleibt_gruen_mit_doku(repo):
    assert pruefen(repo) == ()


def test_echte_aussendatei_wenn_vorhanden():
    wurzel = os.environ.get("SELBSTBILD_VIBEMIND")
    if not wurzel or not (Path(wurzel) / "selbstbild.yaml").is_file():
        pytest.skip("SELBSTBILD_VIBEMIND nicht gesetzt")
    text = (Path(wurzel) / "selbstbild.yaml").read_text(encoding="utf-8")
    assert [r["art"] for r in regeln_lesen(text)["struktur"]] == ["kein_space_im_root", "space"]
    assert [r.art for r in regeln_laden(text).struktur] == ["kein_space_im_root", "space"]
    assert pruefen(Path(wurzel)) == ()


@pytest.mark.parametrize("trenner", ["\u2028", "\u2029", "\x85", "\x0b", "\x0c", "\x1c"])
def test_unicode_zeilentrenner_in_quote_lassen_die_regeln_stehen(trenner):
    extra = f'fremd:\n  a: "x{trenner}yz ohne doppelpunkt"\n  b: 1\n'
    erwartet = regeln_lesen(AUSSEN)
    assert erwartet
    assert regeln_lesen(extra + AUSSEN) == erwartet
    assert regeln_lesen(AUSSEN.rstrip("\n") + "\n" + extra) == erwartet
