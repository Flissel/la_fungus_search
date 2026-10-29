"""Strukturregeln (Stufe 3, T8): Schreibziele, Pfadformen, Verzeichnismuster, CI-Pruefung, leichte Importe."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from selbstbild import struktur
from selbstbild.regeln import regeln_laden
from selbstbild.struktur import pruefe_schreibziel, pruefen, regeln_fuer

SRC = Path(__file__).resolve().parents[2] / "src"
REGELN = (
    "name: aussen\n"
    "struktur:\n"
    "  - art: space\n"
    "    ort: vibemind-os/spaces/*/\n"
    "    pflicht: [capabilities.yaml]\n"
    "    verboten: ['spaces/*/', '*/capabilities.yaml']\n"
    "    hinweis: Spaces gehoeren nach vibemind-os/spaces\n"
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "vibemind-os" / "spaces" / "alt").mkdir(parents=True)
    (r / "vibemind-os" / "spaces" / "alt" / "capabilities.yaml").write_text("x: 1\n", encoding="utf-8")
    (r / "vibemind-os" / "spaces" / "neu").mkdir()
    (r / "selbstbild.yaml").write_text(REGELN, encoding="utf-8")
    return r


def test_space_im_root_blockiert(root):
    v = pruefe_schreibziel(root / "spaces" / "neu" / "x.py")
    assert len(v) == 1 and v[0].hart and v[0].regel == "space"
    assert v[0].hinweis == "Spaces gehoeren nach vibemind-os/spaces"
    assert v[0].pfad == "spaces/neu/x.py"


def test_hinweis_fehlt_dann_richtiger_ort(root):
    (root / "selbstbild.yaml").write_text(REGELN.replace("    hinweis: Spaces gehoeren nach vibemind-os/spaces\n", ""),
                                          encoding="utf-8")
    v = pruefe_schreibziel(root / "spaces" / "neu" / "x.py")
    assert v[0].hinweis == "richtiger Ort: vibemind-os/spaces/*/"


def test_richtiger_ort_frei(root):
    assert pruefe_schreibziel(root / "vibemind-os" / "spaces" / "neu" / "x.py") == ()
    assert pruefe_schreibziel(root / "vibemind-os" / "spaces" / "neu" / "capabilities.yaml") == ()


def test_verzeichnismuster_semantik(root):
    # `spaces/*/` = Praefix: jeder Pfad in einem Verzeichnis unter spaces/, aber nicht tiefer verankert versetzt
    assert pruefe_schreibziel(root / "spaces" / "a" / "b" / "c.py")
    assert pruefe_schreibziel(root / "spaces" / "datei.py") == ()  # kein Verzeichnis dazwischen
    assert pruefe_schreibziel(root / "doku" / "spaces" / "a" / "x.py") == ()  # verankert am Repo-Ordner


def test_ein_segment_muster(root):
    assert pruefe_schreibziel(root / "neuer" / "capabilities.yaml")
    assert pruefe_schreibziel(root / "capabilities.yaml") == ()  # `*/` verlangt genau ein Segment davor
    assert pruefe_schreibziel(root / "a" / "b" / "capabilities.yaml") == ()


def test_pfadformen(root, monkeypatch):
    monkeypatch.setattr(struktur, "GROSSKLEIN_EGAL", True)
    formen = [
        str(root / "doku" / ".." / "spaces" / "neu" / "x.py"),
        str(root / "spaces" / "neu" / "x.py").replace("/", "\\"),
        str(root / "SPACES" / "Neu" / "x.py"),
        str(root / "spaces" / "neu" / "x.py").upper(),
    ]
    for f in formen:
        v = pruefe_schreibziel(Path(f))
        assert len(v) == 1 and v[0].hart, f


def test_relativer_pfad_gegen_cwd(root):
    assert pruefe_schreibziel(Path("spaces/neu/x.py"), cwd=root)
    assert pruefe_schreibziel(Path("vibemind-os/spaces/neu/x.py"), cwd=root) == ()


def test_symlink(root, tmp_path):
    link = tmp_path / "abkuerzung"
    try:
        os.symlink(root / "spaces", link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks nicht erlaubt")
    (root / "spaces").mkdir(exist_ok=True)
    assert pruefe_schreibziel(link / "neu" / "x.py")


def test_ausserhalb_nie_blockiert(root, tmp_path):
    assert pruefe_schreibziel(tmp_path / "fremd" / "spaces" / "neu" / "x.py") == ()


def test_innere_regeln_ohne_aeussere(root):
    """F5: Regeln aller umschliessenden Repos gelten; ein inneres Repo ohne Regeln hebt die aeusseren nicht auf."""
    innen = root / "vibemind-os"
    (innen / "selbstbild.yaml").write_text("name: innen\n", encoding="utf-8")
    assert pruefe_schreibziel(innen / "spaces" / "neu" / "x.py") == ()  # relativ zu innen: nicht verboten
    assert pruefe_schreibziel(root / "spaces" / "neu" / "x.py")


def test_kaputte_regeldatei_blockiert_nicht(root):
    (root / "selbstbild.yaml").write_text("struktur: [oops\n  - ::\n", encoding="utf-8")
    assert pruefe_schreibziel(root / "spaces" / "neu" / "x.py") == ()


def test_regeln_fuer(root):
    r = regeln_fuer(root, "vibemind-os/spaces/neu/x.py")
    assert [x["art"] for x in r] == ["space"]
    r2 = regeln_fuer(root, "spaces/neu/x.py")  # verboten trifft
    assert [x["art"] for x in r2] == ["space"]
    assert regeln_fuer(root, "docs/x.md") == ()


def test_pruefen_pflicht(root):
    v = pruefen(root)
    assert [(x.regel, x.pfad, x.hart) for x in v] == [("space", "vibemind-os/spaces/neu", False)]
    assert "capabilities.yaml" in v[0].grund


def test_pruefen_cli(root):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    p = subprocess.run([sys.executable, "-m", "selbstbild", "pruefen", str(root)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 1 and "vibemind-os/spaces/neu" in p.stdout
    (root / "vibemind-os" / "spaces" / "neu" / "capabilities.yaml").write_text("x: 1\n", encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "selbstbild", "pruefen", str(root)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and p.stdout.strip() == "[]"


def test_hinweis_in_regeln_laden():
    r = regeln_laden("struktur:\n  - art: a\n    ort: x/\n    pflicht: []\n    hinweis: Bitte dort\n")
    assert r.struktur[0].hinweis == "Bitte dort"
    assert regeln_laden("struktur:\n  - art: a\n    ort: x/\n    pflicht: []\n").struktur[0].hinweis is None


def test_waechter_importiert_nichts_schweres():
    code = ("import sys, selbstbild.hooks.waechter, selbstbild.struktur;"
            "bad=[m for m in ('numpy','igraph','yaml','pydantic','tree_sitter') if m in sys.modules];"
            "print(bad); sys.exit(1 if bad else 0)")
    p = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(SRC)},
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr


def test_wurzel_ort_ohne_richtigen_ort(root):
    basis = "struktur:" + chr(10) + '  - art: wurzel' + chr(10) + '    ort: "./"' + chr(10) + "    pflicht: []" + chr(10)
    (root / "selbstbild.yaml").write_text(basis + "    verboten: ['spaces/*/']" + chr(10), encoding="utf-8")
    v = pruefe_schreibziel(root / "spaces" / "neu" / "x.py")
    assert v[0].hinweis is None
    assert "spaces/*/" in v[0].grund and "richtiger Ort" not in v[0].grund
    (root / "selbstbild.yaml").write_text(basis + "    verboten: ['spaces/*/']" + chr(10) + "    hinweis: dort" + chr(10),
                                          encoding="utf-8")
    assert pruefe_schreibziel(root / "spaces" / "neu" / "x.py")[0].hinweis == "dort"


def test_pruefen_unlesbare_regeldatei_faellt_geschlossen(root):
    zeilen = ["struktur:", "  - art: a", "    ort: x/", "    pflicht: [y]", "    hinweis: >", "      lang", "      text", ""]
    (root / "selbstbild.yaml").write_text(chr(10).join(zeilen), encoding="utf-8")
    v = pruefen(root)
    assert len(v) == 1 and v[0].regel == "regeldatei" and "nicht lesbar" in v[0].grund and not v[0].hart
    assert pruefe_schreibziel(root / "x" / "a.py") == ()  # Hook bleibt offen
    p = subprocess.run([sys.executable, "-m", "selbstbild", "pruefen", str(root)],
                       env={**os.environ, "PYTHONPATH": str(SRC)}, capture_output=True, text=True, timeout=60)
    assert p.returncode == 1


def test_pruefen_ohne_struktur_kein_verstoss(root):
    (root / "selbstbild.yaml").write_text("name: x" + chr(10), encoding="utf-8")
    assert pruefen(root) == ()


@pytest.mark.parametrize("ort", ["./", ".", ""])
def test_regel_trifft_wurzel_ort_ganzes_repo(ort):
    r = {"art": "w", "ort": ort, "pflicht": [], "verboten": []}
    assert struktur.regel_trifft(r, "irgendwo/x.py") and struktur.regel_trifft(r, "x.py")


def test_regel_trifft_ort_oder_verboten():
    r = {"art": "space", "ort": "vibemind-os/spaces/*/", "pflicht": [], "verboten": ["spaces/*/"]}
    assert struktur.regel_trifft(r, "vibemind-os/spaces/a/b.py")
    assert struktur.regel_trifft(r, "spaces/a/b.py")  # nur ueber verboten
    assert not struktur.regel_trifft(r, "docs/a.md")
    assert struktur.regel_trifft({"art": "a", "ort": "./app/", "verboten": []}, "app/x.py")  # ./-Praefix
    assert not struktur.regel_trifft(r, "VIBEMIND-OS/spaces/a/b.py", grossklein_egal=False)
    assert struktur.regel_trifft(r, "VIBEMIND-OS/spaces/a/b.py", grossklein_egal=True)


def test_regeln_fuer_wurzel_ort(root):
    (root / "selbstbild.yaml").write_text(
        "struktur:\n  - art: kein_space_im_root\n    ort: \"./\"\n    pflicht: []\n    verboten: ['spaces/*/']\n",
        encoding="utf-8")
    assert [x["art"] for x in regeln_fuer(root, "spaces/x/main.py")] == ["kein_space_im_root"]


VERBOT_REGELN = ("struktur:\n  - art: kein_space_im_root\n    ort: \"./\"\n    pflicht: []\n"
                 "    verboten:\n      - \"spaces/*/\"\n      - \"*/capabilities.yaml\"\n    hinweis: dort\n")


def _pruefen_cli(wurzel: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "selbstbild", "pruefen", str(wurzel)],
                          env={**os.environ, "PYTHONPATH": str(SRC)}, capture_output=True, text=True, timeout=60)


def test_pruefen_verboten_getrackt(git_repo):
    """I2: ein committeter Pfad unter `verboten` ist ein harter Verstoss der CI."""
    git_repo.datei("selbstbild.yaml", VERBOT_REGELN)
    git_repo.datei("spaces/foo/x.py", "x = 1\n")
    git_repo.commit("init")
    v = pruefen(git_repo.wurzel)
    assert [(x.regel, x.pfad, x.hart) for x in v] == [("kein_space_im_root", "spaces/foo/x.py", True)]
    assert "spaces/*/" in v[0].grund and v[0].hinweis == "dort"
    p = _pruefen_cli(git_repo.wurzel)
    assert p.returncode == 1 and "spaces/foo/x.py" in p.stdout


def test_pruefen_verboten_sauber_und_verschachtelt(git_repo):
    git_repo.datei("selbstbild.yaml", VERBOT_REGELN)
    git_repo.datei("vibemind-os/spaces/foo/x.py", "x = 1\n")  # tiefer verschachtelt: Wurzelmuster greifen nicht
    git_repo.datei("vibemind-os/spaces/foo/capabilities.yaml", "a: 1\n")
    git_repo.datei("spaces/datei.py", "x = 1\n")  # kein Verzeichnis unter spaces/
    git_repo.commit("init")
    git_repo.datei("spaces/neu/ungetrackt.py", "x = 1\n")  # nicht getrackt: zaehlt nicht
    assert pruefen(git_repo.wurzel) == ()
    p = _pruefen_cli(git_repo.wurzel)
    assert p.returncode == 0 and p.stdout.strip() == "[]"


def test_pruefen_verboten_ohne_git_rueckfall(root, monkeypatch, caplog):
    """Ohne git: Arbeitsbaum-Durchlauf (ohne .git, node_modules, .venv*, __pycache__) mit Meldung im Log."""
    (root / "selbstbild.yaml").write_text(VERBOT_REGELN, encoding="utf-8")
    (root / "spaces" / "foo").mkdir(parents=True)
    (root / "spaces" / "foo" / "x.py").write_text("x = 1\n", encoding="utf-8")
    (root / "node_modules" / "a").mkdir(parents=True)
    (root / "node_modules" / "a" / "capabilities.yaml").write_text("a: 1\n", encoding="utf-8")
    (root / ".venv-x" / "a").mkdir(parents=True)
    (root / ".venv-x" / "a" / "capabilities.yaml").write_text("a: 1\n", encoding="utf-8")

    def kein_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(struktur.subprocess, "run", kein_git)
    with caplog.at_level("WARNING", logger="selbstbild.struktur"):
        v = pruefen(root)
    assert [(x.pfad, x.hart) for x in v] == [("spaces/foo/x.py", True)]
    assert any("Arbeitsbaum" in r.getMessage() for r in caplog.records)


def test_pruefen_verboten_unter_repo(git_repo, tmp_path):
    """Regeldatei eines Unter-Ordners: Muster relativ zu dessen Ordner, Pfad im Ergebnis relativ zur Wurzel."""
    git_repo.datei("selbstbild.yaml", "name: aussen\n")
    git_repo.datei("sub/selbstbild.yaml", VERBOT_REGELN)
    git_repo.datei("sub/spaces/a/b.py", "x = 1\n")
    git_repo.datei("spaces/a/b.py", "x = 1\n")  # aeussere Datei: Regel von sub gilt hier nicht
    git_repo.commit("init")
    assert [(x.pfad, x.hart) for x in pruefen(git_repo.wurzel)] == [("sub/spaces/a/b.py", True)]
