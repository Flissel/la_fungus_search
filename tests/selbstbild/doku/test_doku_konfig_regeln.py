"""Task 2 (Stufe 4, D5): Doku-Konfiguration aus dem Abschnitt `doku:` der Regeldatei `selbstbild.yaml`."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from doku_vm import VORLAGE

from selbstbild.doku import __main__ as cli
from selbstbild.doku import konfig as konfig_modul
from selbstbild.doku import umschreiben
from selbstbild.doku.konfig import Konfig, KonfigFehler, aus_regeln


def _einruecken(text: str) -> str:
    return "".join(f"  {z}\n" if z.strip() else "\n" for z in text.splitlines())


def _repo(tmp_path: Path, doku_text: str | None = None, *, extra: str = "") -> Path:
    """Fixture-Repo: `config/doku_loop.yaml` (Vorlage) und `selbstbild.yaml` mit `doku:` (leerer Text = ohne)."""
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    vorlage = VORLAGE.read_text(encoding="utf-8")
    (r / "config" / "doku_loop.yaml").write_text(vorlage, encoding="utf-8")
    text = 'name: probe\ndoku_pfade:\n  - "docs/**/*.md"\n'
    if doku_text != "":
        text += "doku:\n" + _einruecken(vorlage if doku_text is None else doku_text)
    (r / "selbstbild.yaml").write_text(text + extra, encoding="utf-8")
    return r


def _vorlage() -> str:
    return VORLAGE.read_text(encoding="utf-8")


def test_aus_regeln_gleich_alter_konfig(tmp_path):
    r = _repo(tmp_path)
    neu = aus_regeln(r)
    alt = konfig_modul.laden(r / "config" / "doku_loop.yaml")
    assert neu == alt
    assert neu.repo == r.resolve() and isinstance(neu.marker, Path) and "%" not in str(neu.marker)
    assert neu.torwaechter == alt.torwaechter and neu.budget_modul == alt.budget_modul


def test_explizit_budget_modul_und_torwaechter(tmp_path):
    t = _vorlage() + ('budget_modul: scripts/claude_budget.py\n'
                      'torwaechter: [".venv/Scripts/python.exe", "scripts/docs_drift_waechter.py"]\n')
    k = aus_regeln(_repo(tmp_path, t))
    assert k.budget_modul == "scripts/claude_budget.py"
    assert k.torwaechter == (".venv/Scripts/python.exe", "scripts/docs_drift_waechter.py")


def test_ohne_doku_abschnitt(tmp_path):
    r = _repo(tmp_path, "")
    k = aus_regeln(r)
    assert k.repo == r.resolve()
    assert k.doku_ziele == ("docs/**/*.md",)
    assert k.ketten_md == "" and k.spaces == {} and k.agent == ""  # agent leer = kein Judge
    assert k.ereignisse == {} and k.space_dokumente == {}
    assert k.rowboat_notizen_aktiv is False and k.rowboat_seiten_aktiv is False
    assert k.torwaechter == () and k.erstbefuellung == {"aktiv": False}
    assert k.waechter_zuordnung == {}


def test_ohne_doku_abschnitt_und_ohne_regeldatei(tmp_path):
    (tmp_path / "leer").mkdir()
    k = aus_regeln(tmp_path / "leer")
    assert k.doku_ziele == ("**/*.md",) and k.agent == ""


def test_ohne_doku_uebernimmt_waechter_zuordnung_und_dienste(tmp_path):
    r = _repo(tmp_path, "", extra="dienste: config/d.yaml\nwaechter_zuordnung:\n  a: [K1]\n")
    k = aus_regeln(r)
    assert k.dienste_yaml == "config/d.yaml" and k.waechter_zuordnung == {"a": ("K1",)}


def test_neue_schluessel_vorgaben(tmp_path):
    k = aus_regeln(_repo(tmp_path))
    assert k.doku_ziele == () and k.doku_nur_lesen == () and k.ereignisse == {} and k.space_dokumente == {}
    assert k.rowboat_seiten_aktiv is False and k.rowboat_seiten_je_lauf == 3
    assert k.erstbefuellung == {"aktiv": False}


def test_neue_schluessel_werte(tmp_path):
    t = _vorlage() + (
        "doku_ziele: [docs/spaces/*.md]\ndoku_nur_lesen: [vibemind-os/spaces/*/README.md]\n"
        "ereignisse: {registry: vibemind-os/config/x.yml}\nspace_dokumente: {verzeichnis: docs/spaces}\n"
        "rowboat_seiten_aktiv: true\nrowboat_seiten_je_lauf: 5\nerstbefuellung: {aktiv: true, agent: x}\n")
    k = aus_regeln(_repo(tmp_path, t))
    assert k.doku_ziele == ("docs/spaces/*.md",) and k.doku_nur_lesen == ("vibemind-os/spaces/*/README.md",)
    assert k.ereignisse == {"registry": "vibemind-os/config/x.yml"}
    assert k.space_dokumente == {"verzeichnis": "docs/spaces"}
    assert k.rowboat_seiten_aktiv is True and k.rowboat_seiten_je_lauf == 5
    assert k.erstbefuellung == {"aktiv": True, "agent": "x"}


@pytest.mark.parametrize("zeile", [
    "ketten_mdd: x", "tick_minuten_: 3", "rowboat_seiten_je_lauf: 0", "rowboat_seiten_je_lauf: true",
    "rowboat_seiten_aktiv: ja", "doku_ziele: docs", "doku_nur_lesen: [1, [2]]", "ereignisse: [a]",
    "space_dokumente: x", "erstbefuellung: [a]", "erstbefuellung: {aktiv: vielleicht}",
])
def test_unbekannte_oder_falsche_schluessel_in_doku(tmp_path, zeile):
    with pytest.raises(KonfigFehler):
        aus_regeln(_repo(tmp_path, _vorlage() + zeile + "\n"))


def test_doku_ist_keine_zuordnung(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    (r / "selbstbild.yaml").write_text("doku: [a, b]\n", encoding="utf-8")
    with pytest.raises(KonfigFehler):
        aus_regeln(r)


@pytest.mark.parametrize("wert", ['"abc"', '[".venv/Scripts/python.exe"]', "[a, 1]", "5", "null", "{a: b}", "[a, b, null]"])
def test_torwaechter_ungueltig(tmp_path, wert):
    t = _vorlage() + f"torwaechter: {wert}\n"
    with pytest.raises(KonfigFehler):
        aus_regeln(_repo(tmp_path, t))
    (tmp_path / "alt.yaml").write_text(t, encoding="utf-8")
    with pytest.raises(KonfigFehler):  # derselbe Pruefer im alten Lader
        konfig_modul.laden(tmp_path / "alt.yaml")


def test_torwaechter_leer_und_drei_elemente(tmp_path):
    assert aus_regeln(_repo(tmp_path, _vorlage() + "torwaechter: []\n")).torwaechter == ()
    (tmp_path / "x").mkdir()
    assert aus_regeln(_repo(tmp_path / "x", _vorlage() + "torwaechter: [a, b, c]\n")).torwaechter == ("a", "b", "c")


@pytest.mark.parametrize("befehl", [("nur-einer",), "string", ["x"]])
def test_drift_fehler_nie_indexerror(tmp_path, befehl):
    with pytest.raises(umschreiben.TorwaechterFehler):
        umschreiben.drift_fehler(tmp_path, befehl)


def test_drift_fehler_leere_liste_ist_kein_torwaechter(tmp_path):
    assert umschreiben.drift_fehler(tmp_path, ()) == set()


def test_drift_fehler_nimmt_das_eigene_skript(tmp_path):
    # gleiche Werte wie der Standard, aber ein anderes Objekt: es zaehlt der Wert, nicht die Identitaet
    with pytest.raises(umschreiben.TorwaechterFehler, match="eigen.py"):
        umschreiben.drift_fehler(tmp_path, [".venv/Scripts/python.exe", "scripts/eigen.py"])
    with pytest.raises(umschreiben.TorwaechterFehler, match="docs_drift_waechter.py"):
        umschreiben.drift_fehler(tmp_path, list(umschreiben.STANDARD_TORWAECHTER))


def test_drift_fuer_gibt_str_nicht_zeichenweise_weiter(tmp_path):
    k = dataclasses.replace(aus_regeln(_repo(tmp_path)), torwaechter="abc")  # type: ignore[arg-type]
    with pytest.raises(umschreiben.TorwaechterFehler):
        umschreiben.drift_fuer(k)(tmp_path)


def test_pfade_repo_relativ_ausser_absolut(tmp_path, monkeypatch):
    t = "\n".join(z for z in _vorlage().splitlines()
                  if not z.startswith(("marker:", "claude_bin:", "rowboat_wissen:")))
    t += ('\nmarker: var/anstoss\nclaude_bin: "%PROBE_BIN%/claude.exe"\nrowboat_wissen: "'
          + (tmp_path / "wissen").as_posix() + '"\n')
    monkeypatch.setenv("PROBE_BIN", (tmp_path / "bin").as_posix())
    r = _repo(tmp_path, t)
    k = aus_regeln(r)
    assert k.marker == r.resolve() / "var" / "anstoss"
    assert k.claude_bin == (tmp_path / "bin" / "claude.exe")
    assert k.rowboat_wissen == tmp_path / "wissen"


# ---- CLI: Quelle der Konfiguration ----

@pytest.fixture
def stub(monkeypatch):
    """`status` durch einen Stummel ersetzen, der die aufgeloeste Konfig festhaelt."""
    gesehen: list[Konfig] = []

    def status(konfig, args, w):
        gesehen.append(konfig)
        return 0

    monkeypatch.setitem(cli.BEFEHLE, "status", status)
    monkeypatch.delenv("SELBSTBILD_VIBEMIND", raising=False)
    return gesehen


def test_cli_wurzel_nutzt_regeldatei(tmp_path, stub, capsys):
    r = _repo(tmp_path)
    assert cli.main(["status", str(r)]) == 0
    assert stub[0].repo == r.resolve()
    err = capsys.readouterr().err
    assert err.count("selbstbild.yaml") == 1 and str(r.resolve()) in err


def test_cli_konfig_nutzt_alten_lader_auch_mit_wurzel(tmp_path, stub, capsys):
    r = _repo(tmp_path)
    datei = str(r / "config" / "doku_loop.yaml")
    assert cli.main(["status", str(tmp_path), "--konfig", datei]) == 0
    assert cli.main(["--konfig", datei, "status"]) == 0
    assert len(stub) == 2 and all(k.repo == r.resolve() for k in stub)
    assert capsys.readouterr().err.count("doku_loop.yaml") == 2


def test_cli_umgebung_als_wurzel(tmp_path, stub, monkeypatch, capsys):
    r = _repo(tmp_path)
    monkeypatch.setenv("SELBSTBILD_VIBEMIND", str(r))
    assert cli.main(["status"]) == 0
    assert stub[0].repo == r.resolve()
    assert "SELBSTBILD_VIBEMIND" in capsys.readouterr().err


def test_cli_ohne_quelle_exit_2_kein_cwd_fallback(tmp_path, stub, monkeypatch, capsys):
    r = _repo(tmp_path)
    monkeypatch.chdir(r)
    assert cli.main(["status"]) == 2
    assert stub == [] and "SELBSTBILD_VIBEMIND" in capsys.readouterr().err


def test_cli_kaputte_regeln_exit_2(tmp_path, stub, capsys):
    r = _repo(tmp_path, _vorlage() + "tippfehler: 1\n")
    assert cli.main(["status", str(r)]) == 2
    assert "tippfehler" in capsys.readouterr().err
