"""doku_loop: Umschreiben mit Ledger, Torwaechter und Commit-Regeln (Task 14).

Alle Git-Tests laufen in einem frischen `git init`-Repo unter tmp_path, nie im echten Repo.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import umschreiben as us  # noqa: E402
from selbstbild.doku.anker import Beleg  # noqa: E402

BELEG = Beleg("symbol", "sp/m/api.py", "m.api.routen", "abc123", "c0ffee", 12)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=60,
    ).stdout


def _schreib(repo: Path, rel: str, text: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8"))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.name", "Test")
    _git(r, "config", "user.email", "test@example.invalid")
    _git(r, "config", "core.autocrlf", "false")
    _git(r, "config", "commit.gpgsign", "false")
    _schreib(r, "docs/a.md", "eins\n")
    _schreib(r, "docs/b.md", "zwei\n")
    _schreib(r, "docs/graph/ledger.jsonl", "")
    _git(r, "add", "--", "docs/a.md", "docs/b.md", "docs/graph/ledger.jsonl")
    _git(r, "commit", "-q", "-m", "start")
    return r


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def _namen(repo: Path, rev: str = "HEAD") -> set[str]:
    return {z for z in _git(repo, "show", "--name-only", "--format=", rev).splitlines() if z.strip()}


# --- fremde_aenderungen -------------------------------------------------------------------

def test_fremde_aenderungen_erkennt_uncommitted(repo):
    assert us.fremde_aenderungen(repo, ["docs/a.md", "docs/b.md"]) == []
    _schreib(repo, "docs/a.md", "eins geaendert\n")
    assert us.fremde_aenderungen(repo, ["docs/a.md", "docs/b.md"]) == ["docs/a.md"]


def test_fremde_aenderungen_sieht_staged_unstaged_untracked(repo):
    _schreib(repo, "docs/a.md", "staged\n")
    _git(repo, "add", "--", "docs/a.md")
    _schreib(repo, "docs/b.md", "unstaged\n")
    _schreib(repo, "docs/neu ä.md", "untracked\n")
    gefunden = us.fremde_aenderungen(repo, ["docs/a.md", "docs/b.md", "docs/neu ä.md", "docs/graph/ledger.jsonl"])
    assert sorted(gefunden) == ["docs/a.md", "docs/b.md", "docs/neu ä.md"]


def test_fremde_aenderungen_erkennt_umbenennung(repo):
    _git(repo, "mv", "docs/a.md", "docs/c.md")
    gefunden = us.fremde_aenderungen(repo, ["docs/a.md", "docs/c.md"])
    assert set(gefunden) == {"docs/a.md", "docs/c.md"}


def test_fremde_aenderungen_erkennt_arbeitsbaum_umbenennung(repo):
    """` R neu\\0alt` (Umbenennung im Arbeitsbaum via intent-to-add) liefert beide Pfade."""
    _schreib(repo, "docs/a.md", "eins\nzwei\ndrei\nvier\nfuenf\n")
    _git(repo, "commit", "-q", "-am", "mehr")
    (repo / "docs/a.md").rename(repo / "docs/c.md")
    _git(repo, "add", "-N", "--", "docs/c.md")
    roh = _git(repo, "status", "--porcelain=v1", "--", "docs/a.md", "docs/c.md")
    assert roh.startswith(" R"), roh  # git meldet die Umbenennung im Arbeitsbaum
    gefunden = us.fremde_aenderungen(repo, ["docs/c.md", "docs/a.md", "docs/b.md"])
    # Ohne Sonderfall wuerde der alte Pfad als eigener Eintrag gelesen ("cs/a.md").
    assert gefunden == ["docs/c.md", "docs/a.md"]


def test_fremde_aenderungen_leere_liste(repo):
    _schreib(repo, "docs/a.md", "x\n")
    assert us.fremde_aenderungen(repo, []) == []


# --- committen ----------------------------------------------------------------------------

def test_committen_nur_genannte_dateien(repo):
    _schreib(repo, "docs/a.md", "eins neu\n")
    _schreib(repo, "docs/b.md", "zwei neu\n")
    sha = us.committen(repo, ["docs/a.md"], "docs(loop): Test – 1 Aussagen nachgeführt, 0 in Prüfliste")
    assert sha == _head(repo)
    assert _namen(repo) == {"docs/a.md"}
    status = _git(repo, "status", "--porcelain=v1", "--", "docs/b.md")
    assert status.startswith(" M")  # unstaged geblieben


def test_committen_laesst_fremd_gestagte_datei_gestaged(repo):
    _schreib(repo, "docs/b.md", "fremd gestaged\n")
    _git(repo, "add", "--", "docs/b.md")
    _schreib(repo, "docs/a.md", "loop\n")
    sha = us.committen(repo, ["docs/a.md"], "docs(loop): a")
    assert sha is not None
    assert _namen(repo) == {"docs/a.md"}
    status = _git(repo, "status", "--porcelain=v1", "--", "docs/b.md")
    assert status.startswith("M ")  # weiterhin gestaged, nicht committet
    assert _git(repo, "show", "HEAD:docs/b.md") == "zwei\n"


def test_committen_nicht_ascii_nachricht(repo):
    _schreib(repo, "docs/a.md", "ü\n")
    nachricht = "docs(loop): Prüfliste – Grüße für Ärger"
    sha = us.committen(repo, ["docs/a.md"], nachricht)
    assert sha is not None
    assert _git(repo, "log", "-1", "--format=%s", sha).strip() == nachricht


def test_committen_ohne_aenderung_none(repo):
    vorher = _head(repo)
    assert us.committen(repo, ["docs/a.md"], "docs(loop): nichts") is None
    assert _head(repo) == vorher


def test_committen_neue_datei_und_zwei_dateien(repo):
    _schreib(repo, "docs/neu.md", "neu\n")
    _schreib(repo, "docs/graph/ledger.jsonl", '{"a": 1}\n')
    sha = us.committen(repo, ["docs/neu.md", "docs/graph/ledger.jsonl"], "docs(loop): zwei")
    assert sha is not None
    assert _namen(repo) == {"docs/neu.md", "docs/graph/ledger.jsonl"}


# --- Torwaechter --------------------------------------------------------------------------

def test_torwaechter_ok_regel_teilmenge():
    assert us.torwaechter_ok(set(), set())
    assert us.torwaechter_ok({"a", "b"}, {"a"})
    assert us.torwaechter_ok({"a"}, {"a"})
    assert not us.torwaechter_ok({"a"}, {"a", "b"})
    assert not us.torwaechter_ok(set(), {"neu"})


def _fake_waechter(tmp_path: Path, monkeypatch, code: str) -> None:
    skript = tmp_path / "fake_waechter.py"
    skript.write_text(textwrap.dedent(code), encoding="utf-8")
    monkeypatch.setattr(us, "WAECHTER_SKRIPT", str(skript))


def test_drift_fehler_exit_0_leer(repo, tmp_path, monkeypatch):
    _fake_waechter(tmp_path, monkeypatch, """
        print("ok  docs/spaces/01_x.md  Belege 3  Pins 0  Stand 2026-10-27")
        print("\\n1 Dokumente, 0 Fehler, 0 Warnungen")
    """)
    assert us.drift_fehler(repo) == set()


def test_drift_fehler_exit_1_sammelt_zeilen(repo, tmp_path, monkeypatch):
    _fake_waechter(tmp_path, monkeypatch, """
        import sys
        print("ok  docs/spaces/01_x.md  Belege 3  Pins 0  Stand 2026-10-27")
        print("FEHLER docs/spaces/02_y.md  Belege 4  Pins 1  Stand 2026-10-01")
        print("    FEHLER  Beleg fehlt: `a/b.py`")
        print("    warn    Stand 2026-10-01 ist 26 Tage alt (> 30)")
        print("FEHLER docs/modules/03_z.md  Belege 9  Pins 2  Stand -")
        print("    FEHLER  Beleg fehlt: `a/b.py`")
        print("FEHLER Index: Index verlinkt `04_q.md` nicht")
        print("\\n3 Dokumente, 3 Fehler, 1 Warnungen")
        sys.exit(1)
    """)
    assert us.drift_fehler(repo) == {
        ("docs/spaces/02_y.md", "Beleg fehlt: `a/b.py`"),
        ("docs/modules/03_z.md", "Beleg fehlt: `a/b.py`"),  # gleiche Meldung, anderes Dokument
        ("Index", "Index verlinkt `04_q.md` nicht"),
    }


def test_drift_fehler_zaehler_aendern_die_menge_nicht(repo, tmp_path, monkeypatch):
    """Belege-/Pins-Zaehler in der Kopfzeile sind Rauschen und gehoeren nicht in die Fehlermenge."""
    vorlage = """
        import sys
        print("FEHLER docs/spaces/02_y.md  Belege {n}  Pins 1  Stand 2026-10-01")
        print("    FEHLER  Beleg fehlt: `a/b.py`")
        print("\\n1 Dokumente, 1 Fehler, 0 Warnungen")
        sys.exit(1)
    """
    _fake_waechter(tmp_path, monkeypatch, vorlage.format(n=4))
    vorher = us.drift_fehler(repo)
    _fake_waechter(tmp_path, monkeypatch, vorlage.format(n=5))
    assert us.torwaechter_ok(vorher, us.drift_fehler(repo))


def test_drift_fehler_absturz_wirft(repo, tmp_path, monkeypatch):
    _fake_waechter(tmp_path, monkeypatch, """
        raise RuntimeError("kaputt")
    """)
    with pytest.raises(us.TorwaechterFehler):
        us.drift_fehler(repo)


def test_drift_fehler_unbekannter_exit_wirft(repo, tmp_path, monkeypatch):
    _fake_waechter(tmp_path, monkeypatch, """
        import sys
        sys.exit(2)
    """)
    with pytest.raises(us.TorwaechterFehler):
        us.drift_fehler(repo)


def test_drift_fehler_timeout_wirft(repo, tmp_path, monkeypatch):
    _fake_waechter(tmp_path, monkeypatch, """
        import time
        time.sleep(30)
    """)
    monkeypatch.setattr(us, "WAECHTER_TIMEOUT_S", 1)
    with pytest.raises(us.TorwaechterFehler):
        us.drift_fehler(repo)


def test_drift_fehler_teilausgabe_ohne_summe_wirft(repo, tmp_path, monkeypatch):
    """Exit 1 mit einzelnen Fehlerzeilen, aber ohne Summenzeile = Waechter lief nicht durch."""
    _fake_waechter(tmp_path, monkeypatch, """
        import sys
        print("FEHLER Index: Index verlinkt `04_q.md` nicht")
        sys.exit(1)
    """)
    with pytest.raises(us.TorwaechterFehler):
        us.drift_fehler(repo)


def test_drift_fehler_zaehlt_gegen_summe(repo, tmp_path, monkeypatch):
    """Erkennt der Parser weniger Fehlerzeilen als die Summe nennt, ist das Format unbekannt."""
    _fake_waechter(tmp_path, monkeypatch, """
        import sys
        print("ok  docs/spaces/01_x.md  Belege 3  Pins 0  Stand 2026-10-27")
        print("FEHLER Index: Index verlinkt `04_q.md` nicht")
        print("NEUES FORMAT: irgendwas")
        print("\\n1 Dokumente, 2 Fehler, 0 Warnungen")
        sys.exit(1)
    """)
    with pytest.raises(us.TorwaechterFehler, match="Fehlerzeilen"):
        us.drift_fehler(repo)


def test_drift_fehler_detail_vor_kopfzeile_wirft(repo, tmp_path, monkeypatch):
    _fake_waechter(tmp_path, monkeypatch, """
        import sys
        print("    FEHLER  Beleg fehlt: `a/b.py`")
        print("ok  docs/spaces/01_x.md  Belege 3  Pins 0  Stand 2026-10-27")
        print("\\n1 Dokumente, 1 Fehler, 0 Warnungen")
        sys.exit(1)
    """)
    with pytest.raises(us.TorwaechterFehler, match="vor jeder Dokumentzeile"):
        us.drift_fehler(repo)


def test_drift_fehler_dokumentzahl_gegen_summe(repo, tmp_path, monkeypatch):
    """Fehlt eine Dokument-Kopfzeile (neues Format), passt die Dokumentzahl nicht -> Fehler."""
    _fake_waechter(tmp_path, monkeypatch, """
        print("ok  docs/spaces/01_x.md  Belege 3  Pins 0  Stand 2026-10-27")
        print("OK? docs/spaces/02_y.md  neue Spalten")
        print("\\n2 Dokumente, 0 Fehler, 0 Warnungen")
    """)
    with pytest.raises(us.TorwaechterFehler, match="Dokumentzeilen"):
        us.drift_fehler(repo)


def test_drift_fehler_warnt_bei_fallback_python(repo, tmp_path, monkeypatch, caplog):
    _fake_waechter(tmp_path, monkeypatch, """
        print("\\n0 Dokumente, 0 Fehler, 0 Warnungen")
    """)
    with caplog.at_level("WARNING", logger=us.__name__):
        assert us.drift_fehler(repo) == set()
    assert any("fehlt, nutze" in r.getMessage() for r in caplog.records)


@braucht_vm
def test_drift_fehler_echter_waechter_laeuft(monkeypatch):
    """Der echte Waechter im echten Repo laeuft durch (nur lesend) und liefert eine Menge."""
    repo = REPO
    assert isinstance(us.drift_fehler(repo), set)


def test_drift_fehler_fehlendes_skript_wirft(repo, monkeypatch):
    monkeypatch.setattr(us, "WAECHTER_SKRIPT", "scripts/gibt_es_nicht.py")
    with pytest.raises(us.TorwaechterFehler):
        us.drift_fehler(repo)


# --- revertieren --------------------------------------------------------------------------

def test_revertieren_nur_an_der_spitze(repo):
    _schreib(repo, "docs/a.md", "loop 1\n")
    sha1 = us.committen(repo, ["docs/a.md"], "docs(loop): 1")
    _schreib(repo, "docs/a.md", "loop 2\n")
    sha2 = us.committen(repo, ["docs/a.md"], "docs(loop): 2")
    assert sha1 and sha2
    assert us.revertieren(repo, sha1, "docs/a.md") is False
    assert _head(repo) == sha2
    assert us.revertieren(repo, sha2, "docs/a.md") is True
    assert (repo / "docs/a.md").read_bytes() == b"loop 1\n"
    assert _namen(repo) == {"docs/a.md"}
    assert _git(repo, "log", "-1", "--format=%s").strip() == f"revert(loop): docs/a.md aus {sha2[:7]}"


def test_revertieren_nur_zieldatei_eines_zwei_datei_commits(repo):
    _schreib(repo, "docs/a.md", "loop\n")
    _schreib(repo, "docs/graph/ledger.jsonl", '{"lauf": "1"}\n')
    sha = us.committen(repo, ["docs/a.md", "docs/graph/ledger.jsonl"], "docs(loop): beide")
    assert sha
    assert us.revertieren(repo, sha, "docs/a.md") is True
    assert (repo / "docs/a.md").read_bytes() == b"eins\n"
    assert (repo / "docs/graph/ledger.jsonl").read_bytes() == b'{"lauf": "1"}\n'
    assert _namen(repo) == {"docs/a.md"}
    assert _git(repo, "status", "--porcelain=v1") == ""


def test_revertieren_eines_anlege_commits_loescht_datei(repo):
    _schreib(repo, "docs/neu.md", "neu\n")
    sha = us.committen(repo, ["docs/neu.md"], "docs(loop): neu")
    assert sha
    assert us.revertieren(repo, sha, "docs/neu.md") is True
    assert not (repo / "docs/neu.md").exists()
    assert _namen(repo) == {"docs/neu.md"}
    assert "docs/neu.md" not in _git(repo, "ls-files")


def test_revertieren_verweigert_bei_fremder_aenderung(repo):
    _schreib(repo, "docs/a.md", "loop\n")
    sha = us.committen(repo, ["docs/a.md"], "docs(loop): a")
    assert sha
    _schreib(repo, "docs/a.md", "fremd, nicht committet\n")
    assert us.revertieren(repo, sha, "docs/a.md") is False
    assert (repo / "docs/a.md").read_bytes() == b"fremd, nicht committet\n"
    assert _head(repo) == sha


def test_revertieren_laesst_fremd_gestagte_datei_gestaged(repo):
    _schreib(repo, "docs/a.md", "loop\n")
    sha = us.committen(repo, ["docs/a.md"], "docs(loop): a")
    assert sha
    _schreib(repo, "docs/b.md", "fremd gestaged\n")
    _git(repo, "add", "--", "docs/b.md")
    assert us.revertieren(repo, sha, "docs/a.md") is True
    assert _namen(repo) == {"docs/a.md"}
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/b.md").startswith("M ")


# --- schreiben ----------------------------------------------------------------------------

def test_schreiben_atomar_und_crlf(repo):
    text = "﻿Kopf\r\nZeile ä\r\nLF-Zeile\nEnde"
    us.schreiben(repo, "docs/spaces/KETTEN.md", text)
    ziel = repo / "docs/spaces/KETTEN.md"
    assert ziel.read_bytes() == text.encode("utf-8")
    us.schreiben(repo, "docs/spaces/KETTEN.md", "neu\r\n")
    assert ziel.read_bytes() == b"neu\r\n"
    assert [p.name for p in ziel.parent.iterdir()] == ["KETTEN.md"]  # keine Temp-Reste


@pytest.mark.parametrize("pfad", ["../x", ".git/x", ".git/hooks/post-commit", "docs/../../x", ".GIT/x", ".git"])
def test_schreiben_verweigert_pfade_ausserhalb(repo, pfad):
    with pytest.raises(ValueError):
        us.schreiben(repo, pfad, "boese")
    assert not (repo.parent / "x").exists()
    assert not (repo / ".git" / "x").exists()


def test_schreiben_verweigert_absoluten_pfad_ausserhalb(repo, tmp_path):
    with pytest.raises(ValueError):
        us.schreiben(repo, str(tmp_path / "draussen.md"), "boese")
    assert not (tmp_path / "draussen.md").exists()


# --- Ledger -------------------------------------------------------------------------------

def _eintrag(**kw) -> "us.LedgerEintrag":
    basis = dict(lauf="2026-10-27T10:00:00Z", zeit="2026-10-27T10:00:05Z",
                 aussage_id="docs/spaces/KETTEN.md#K1#abcd1234", datei="docs/spaces/KETTEN.md",
                 alt="alt ä", neu="neu ü", beleg=BELEG, urteil="beleg_geaendert", quelle="judge", commit=None)
    basis.update(kw)
    return us.LedgerEintrag(**basis)


def test_ledger_anhaengen_jsonl(tmp_path):
    pfad = tmp_path / "tief" / "graph" / "ledger.jsonl"
    us.ledger_anhaengen(pfad, _eintrag())
    us.ledger_anhaengen(pfad, _eintrag(commit="deadbeef", urteil="beleg_ok"))
    roh = pfad.read_bytes().decode("utf-8")
    zeilen = roh.split("\n")
    assert zeilen[-1] == "" and len(zeilen) == 3
    assert "alt ä" in zeilen[0]  # ensure_ascii=False
    erste = json.loads(zeilen[0])
    assert list(erste) == sorted(erste)  # sort_keys
    assert erste["beleg"] == BELEG.to_dict()
    assert erste["commit"] is None
    zweite = json.loads(zeilen[1])
    assert zweite["commit"] == "deadbeef" and zweite["urteil"] == "beleg_ok"
    assert us.LedgerEintrag.from_dict(zweite) == _eintrag(commit="deadbeef", urteil="beleg_ok")


# --- Fix-Runde 1: kanonische Pfade ----------------------------------------------------------

def _fremd_stagen(repo: Path) -> None:
    _schreib(repo, "docs/b.md", "fremd gestaged\n")
    _git(repo, "add", "--", "docs/b.md")


def _nur_ziel_committet(repo: Path, vorher: str) -> None:
    assert _head(repo) != vorher
    assert _namen(repo) == {"docs/a.md"}
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/b.md").startswith("M ")
    assert _git(repo, "show", "HEAD:docs/b.md") == "zwei\n"


def test_committen_absoluter_pfad_nimmt_fremd_gestagtes_nicht_mit(repo):
    _fremd_stagen(repo)
    _schreib(repo, "docs/a.md", "loop\n")
    vorher = _head(repo)
    assert us.committen(repo, [str(repo / "docs" / "a.md")], "docs(loop): absolut")
    _nur_ziel_committet(repo, vorher)


def test_committen_punktpunkt_pfad_nimmt_fremd_gestagtes_nicht_mit(repo):
    _fremd_stagen(repo)
    _schreib(repo, "docs/a.md", "loop\n")
    vorher = _head(repo)
    assert us.committen(repo, ["docs/../docs/a.md"], "docs(loop): punktpunkt")
    _nur_ziel_committet(repo, vorher)


def test_committen_backslash_pfad(repo):
    _fremd_stagen(repo)
    _schreib(repo, "docs/a.md", "loop\n")
    vorher = _head(repo)
    assert us.committen(repo, ["docs\\a.md"] if sys.platform == "win32" else ["docs/a.md"], "docs(loop): bs")
    _nur_ziel_committet(repo, vorher)


def _gross_klein_egal(repo: Path) -> bool:
    return (repo / "DOCS" / "A.MD").exists()


def test_committen_falsche_schreibweise(repo):
    if not _gross_klein_egal(repo):
        pytest.skip("Dateisystem unterscheidet Gross-/Kleinschreibung")
    _fremd_stagen(repo)
    _schreib(repo, "docs/a.md", "loop\n")
    vorher = _head(repo)
    assert us.committen(repo, ["Docs/A.MD"], "docs(loop): schreibweise")
    _nur_ziel_committet(repo, vorher)


def test_committen_geloeschte_datei_falsche_schreibweise(repo):
    if not _gross_klein_egal(repo):
        pytest.skip("Dateisystem unterscheidet Gross-/Kleinschreibung")
    (repo / "docs/a.md").unlink()
    assert us.committen(repo, ["DOCS/A.md"], "docs(loop): weg")
    assert _namen(repo) == {"docs/a.md"}
    assert "docs/a.md" not in _git(repo, "ls-files")


def test_committen_teilweise_unveraendert_bricht_vor_commit_ab(repo):
    _fremd_stagen(repo)
    _schreib(repo, "docs/a.md", "loop\n")
    vorher = _head(repo)
    with pytest.raises(us.GitFehler, match="keine gestagte Aenderung"):
        us.committen(repo, ["docs/a.md", "docs/graph/ledger.jsonl"], "docs(loop): teil")
    assert _head(repo) == vorher
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/a.md").startswith(" M")  # eigenes Staging zurueck
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/b.md").startswith("M ")


def test_committen_pfad_ausserhalb_bricht_vor_commit_ab(repo, tmp_path):
    _fremd_stagen(repo)
    vorher = _head(repo)
    with pytest.raises(ValueError):
        us.committen(repo, [str(tmp_path / "draussen.md")], "docs(loop): draussen")
    assert _head(repo) == vorher
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/b.md").startswith("M ")


# --- Fix-Runde 1: Compare-and-Swap ---------------------------------------------------------

def test_schreiben_erwartet_alt_passt(repo):
    us.schreiben(repo, "docs/a.md", "neu\n", erwartet_alt=b"eins\n")
    assert (repo / "docs/a.md").read_bytes() == b"neu\n"


def test_schreiben_erwartet_alt_neue_datei(repo):
    us.schreiben(repo, "docs/neu.md", "neu\n", erwartet_alt=b"")
    assert (repo / "docs/neu.md").read_bytes() == b"neu\n"


def test_schreiben_erwartet_alt_abweichung(repo):
    _schreib(repo, "docs/a.md", "fremd dazwischen\n")
    with pytest.raises(us.FremdeAenderung):
        us.schreiben(repo, "docs/a.md", "loop\n", erwartet_alt=b"eins\n")
    assert (repo / "docs/a.md").read_bytes() == b"fremd dazwischen\n"
    assert sorted(p.name for p in (repo / "docs").iterdir()) == ["a.md", "b.md", "graph"]


def _h(daten: bytes) -> str:
    import hashlib
    return hashlib.sha256(daten).hexdigest()


def test_committen_erwartet_passt(repo):
    us.schreiben(repo, "docs/a.md", "loop\r\n")
    assert us.committen(repo, ["docs/a.md"], "docs(loop): cas", erwartet={"docs/a.md": _h(b"loop\r\n")})
    assert _namen(repo) == {"docs/a.md"}


def test_committen_erwartet_mit_autocrlf_kein_fehlalarm(repo):
    _git(repo, "config", "core.autocrlf", "true")
    us.schreiben(repo, "docs/a.md", "zeile eins\r\nzeile zwei\r\n")
    erwartet = {"docs/a.md": _h(b"zeile eins\r\nzeile zwei\r\n")}
    assert us.committen(repo, ["docs/a.md"], "docs(loop): crlf", erwartet=erwartet)
    assert _git(repo, "show", "HEAD:docs/a.md") == "zeile eins\nzeile zwei\n"  # Index normalisiert


def test_committen_erwartet_arbeitsbaum_fremd_geaendert(repo):
    _fremd_stagen(repo)
    us.schreiben(repo, "docs/a.md", "loop\n")
    _schreib(repo, "docs/a.md", "fremd nach dem Schreiben\n")
    vorher = _head(repo)
    with pytest.raises(us.FremdeAenderung):
        us.committen(repo, ["docs/a.md"], "docs(loop): cas", erwartet={"docs/a.md": _h(b"loop\n")})
    assert _head(repo) == vorher
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/a.md").startswith(" M")
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/b.md").startswith("M ")


def test_committen_erwartet_gestagter_blob_fremd(repo, monkeypatch):
    """Aendert jemand die Datei zwischen Pruefung und `git add`, faellt der gestagte Blob auf."""
    us.schreiben(repo, "docs/a.md", "loop\n")
    echt = us._git

    def dazwischen(r, *args, **kw):
        if args[:1] == ("add",):
            _schreib(repo, "docs/a.md", "fremd beim add\n")
        return echt(r, *args, **kw)

    monkeypatch.setattr(us, "_git", dazwischen)
    vorher = _head(repo)
    with pytest.raises(us.FremdeAenderung, match="gestagter"):
        us.committen(repo, ["docs/a.md"], "docs(loop): cas", erwartet={"docs/a.md": _h(b"loop\n")})
    assert _head(repo) == vorher
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/a.md").startswith(" M")  # nicht mehr gestaged


def test_committen_erwartet_pfad_ausserhalb_von_dateien(repo):
    with pytest.raises(ValueError):
        us.committen(repo, ["docs/a.md"], "x", erwartet={"docs/b.md": "0" * 64})


# --- Fix-Runde 1: Hooks, Timeouts, CommitUnbestaetigt --------------------------------------

def _hook(repo: Path, name: str, rumpf: str) -> None:
    pfad = repo / ".git" / "hooks" / name
    pfad.write_bytes(f"#!/bin/sh\n{rumpf}\n".encode())
    pfad.chmod(0o755)


def test_committen_hook_scheitert_nimmt_staging_zurueck(repo):
    _fremd_stagen(repo)
    _schreib(repo, "docs/a.md", "loop\n")
    _schreib(repo, "docs/neu.md", "neu\n")
    _hook(repo, "pre-commit", "exit 1")
    vorher = _head(repo)
    with pytest.raises(us.GitFehler) as info:
        us.committen(repo, ["docs/a.md", "docs/neu.md"], "docs(loop): hook")
    assert not isinstance(info.value, us.CommitUnbestaetigt)
    assert _head(repo) == vorher
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/a.md").startswith(" M")
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/neu.md").startswith("??")
    assert _git(repo, "status", "--porcelain=v1", "--", "docs/b.md").startswith("M ")


def test_committen_timeout_nach_commit_ist_unbestaetigt(repo, monkeypatch):
    import time
    _schreib(repo, "docs/a.md", "loop\n")
    _hook(repo, "post-commit", "sleep 60")
    monkeypatch.setattr(us, "GIT_COMMIT_TIMEOUT_S", 3)
    start = time.monotonic()
    with pytest.raises(us.CommitUnbestaetigt) as info:
        us.committen(repo, ["docs/a.md"], "docs(loop): langsamer hook")
    assert time.monotonic() - start < 40
    assert info.value.sha == _head(repo)
    assert info.value.dateien_ok is True
    assert _namen(repo) == {"docs/a.md"}


def test_laufen_timeout_beendet_prozessbaum():
    """Ein Enkel, der die Pipe erbt, darf den Timeout nicht endlos verlaengern."""
    import time
    code = ("import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)")
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        us._laufen([sys.executable, "-c", code], timeout=2)
    assert time.monotonic() - start < 30
