"""doku_loop: Konfiguration und Korpus (Durchlauf mit Ausschluss vor dem Betreten)."""

import os
import sys
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import korpus  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402

braucht_submodul = pytest.mark.skipif(
    not korpus.manifest_verfuegbar(), reason="Submodul vibemind-os nicht ausgecheckt"
)



@pytest.fixture()
def konfig():
    return vm_laden()


def _schreibe(pfad: Path, text: str = "x = 1\n") -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(text, encoding="utf-8")


def test_dateien_betritt_ausgeschlossene_ordner_nicht(tmp_path, konfig):
    _schreibe(tmp_path / "a.py")
    _schreibe(tmp_path / "node_modules" / "b.py")
    _schreibe(tmp_path / ".worktrees" / "x" / "c.py")
    _schreibe(tmp_path / "voice" / "python" / "spaces" / "d.py")
    _schreibe(tmp_path / "spaces" / "captain_cook" / "hermes-agent" / "e.py")
    _schreibe(tmp_path / "readme.md")
    assert [p.name for p in korpus.dateien(tmp_path, konfig, (".py",))] == ["a.py"]


def test_dateien_pfadausschluss_gilt_auch_ab_tieferer_wurzel(tmp_path, konfig):
    wurzel = tmp_path / "spaces" / "captain_cook"
    _schreibe(wurzel / "hermes-agent" / "e.py")
    _schreibe(wurzel / "ok.py")
    assert [p.name for p in korpus.dateien(wurzel, konfig, (".py",))] == ["ok.py"]


def test_dateien_ueberlebt_toten_symlink(tmp_path, konfig):
    _schreibe(tmp_path / "a.py")
    _schreibe(tmp_path / "pkg" / "b.py")
    (tmp_path / "node_modules" / "@x").mkdir(parents=True)
    try:
        os.symlink(tmp_path / "gibt_es_nicht", tmp_path / "node_modules" / "@x" / "preload", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink-Recht fehlt")
    ergebnis = korpus.dateien(tmp_path, konfig, (".py",))
    assert sorted(p.name for p in ergebnis) == ["a.py", "b.py"]


@braucht_submodul
def test_manifest_bauen_liefert_kanten(tmp_path, konfig):
    # tmp_path liegt ausserhalb des Repos, daher Konfig mit passendem repo
    k = konfig_modul.Konfig(**{**konfig.__dict__, "repo": tmp_path.parent})
    _schreibe(tmp_path / "a.py", "def f():\n    return 1\n")
    _schreibe(tmp_path / "b.py", "from a import f\n\ndef g():\n    return f()\n")
    _schreibe(tmp_path / "node_modules" / "c.py", "def f():\n    return 2\n")
    space = SpaceKonfig("t", tmp_path.name, "decorator")
    m = korpus.manifest_bauen(space, k, "abc")
    assert any(m.callees_by_document[d] for d in m.callees_by_document)
    assert not any("node_modules" in d.file for d in m.documents)
    assert m.commit_sha == "abc"


@braucht_submodul
def test_symbol_an_findet_engste_definition(tmp_path, konfig):
    k = konfig_modul.Konfig(**{**konfig.__dict__, "repo": tmp_path.parent})
    _schreibe(
        tmp_path / "m.py",
        "class K:\n"          # 1
        "    def meth(self):\n"  # 2
        "        return 1\n"     # 3
        "\n"                     # 4
        "    def zwei(self):\n"  # 5
        "        return 2\n",    # 6
    )
    m = korpus.manifest_bauen(SpaceKonfig("t", tmp_path.name, "decorator"), k, "abc")
    assert korpus.symbol_an(m, "m.py", 3).symbol == "meth"
    assert korpus.symbol_an(m, "m.py", 1).symbol == "K"
    assert korpus.symbol_an(m, "m.py", 4).symbol == "K"
    assert korpus.symbol_an(m, "m.py", 99) is None
    assert korpus.symbol_an(m, "andere.py", 3) is None


def test_konfig_laden_echte_datei(konfig):
    assert set(konfig.spaces) == {"marketing", "sales-claw", "brain"}
    assert konfig.tick_minuten == 30
    assert konfig.agent == "doku-loop"
    assert konfig.judge_modell == "sonnet"
    assert konfig.spaces["brain"].wurzel == "vibemind-os/brain"
    assert "node_modules" in konfig.ausschluss_namen
    assert "voice/python/spaces" in konfig.ausschluss_pfade
    assert konfig.waechter_zuordnung["marketing"] == ("K1", "K2")
    assert konfig.dienste_yaml == "config/dienste.yaml"
    assert "%" not in str(konfig.marker) and "%" not in str(konfig.claude_bin)
    assert konfig.claude_bin.name == "claude.exe"
    if REPO.is_dir():  # nur mit SELBSTBILD_VIBEMIND: die Vorlage liegt nicht im VibeMind-Repo
        assert konfig.repo == REPO


@braucht_vm
def test_commit_von_liefert_sha():
    sha = korpus.commit_von(REPO)
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_weitere_doku_ist_leer(konfig):
    assert konfig.weitere_doku == ()


@pytest.mark.skipif(
    not (REPO / "vibemind-os" / "spaces" / "marketing" / "db").is_dir(),
    reason="Submodul vibemind-os nicht ausgecheckt",
)
def test_sql_wurzeln_erreichen_marketing_migration(konfig):
    gefunden = {
        p.relative_to(REPO).as_posix()
        for wurzel in konfig.sql_wurzeln
        for p in korpus.dateien(REPO / wurzel, konfig, (".sql",))
    }
    assert "vibemind-os/spaces/marketing/db/043_marketing_versandauftraege.sql" in gefunden


@pytest.mark.parametrize("pfad,erwartet", [
    ("a/tests/x.py", True), ("a/test/x.py", True), ("a/test_x.py", True),
    ("a/foo_test.py", True), ("a/conftest.py", True), ("tests/x.py", True),
    ("a/x.py", False), ("a/testing/x.py", False), ("a/latest.py", False),
    ("a/contest.py", False),
])
def test_ist_testdatei(pfad, erwartet):
    assert korpus.ist_testdatei(pfad) is erwartet
    assert korpus.ist_testdatei(Path(pfad)) is erwartet


def test_ast_lesen_ohne_warnungen(tmp_path):
    import ast
    import warnings

    p = tmp_path / "a.py"
    p.write_text('x = "\\d"\n', encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baum = korpus.ast_lesen(p)
    assert isinstance(baum, ast.Module)
    (tmp_path / "kaputt.py").write_text("def (\n", encoding="utf-8")
    assert korpus.ast_lesen(tmp_path / "kaputt.py") is None
    assert korpus.ast_lesen(tmp_path / "fehlt.py") is None
