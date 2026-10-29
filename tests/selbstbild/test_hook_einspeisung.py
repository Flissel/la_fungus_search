"""Einspeisungs-Hook (Stufe 3, T7): Kontextblock, Frische, Frist, Budget, Latenz, leichte Importe."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from selbstbild.hooks import blob_sha, regeln_lesen, repo_wurzeln, stand_fuer
from selbstbild.hooks.einspeisung import kontextblock
from selbstbild.suche.index import index_bauen
from selbstbild.vermessen import stand_finden, vermessen

SRC = Path(__file__).resolve().parents[2] / "src"
REGELN = "struktur:\n  - art: dienst\n    ort: app/\n    pflicht: [README.md]\n    verboten: [tmp]\n"


@pytest.fixture
def heim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("SELBSTBILD_HOME", str(h))
    return h


def _aufbauen(git_repo, *, regeln: str = REGELN) -> Path:
    if regeln:
        git_repo.datei("selbstbild.yaml", regeln)
    git_repo.datei("app/versand.py", "def newsletter_freigeben():\n    return 1\n")
    git_repo.datei("docs/newsletter.md", "# Versand\n\nDer Newsletter wird freigegeben.\n")
    git_repo.commit("init")
    vermessen(git_repo.wurzel)
    stand = stand_fuer(git_repo.wurzel)
    assert stand is not None
    index_bauen(git_repo.wurzel, stand.parent)
    return stand.parent


def test_block_mit_treffer(git_repo, heim):
    _aufbauen(git_repo)
    block = kontextblock(git_repo.wurzel, "newsletter freigeben")
    assert block is not None
    assert block.startswith("Selbstbild (Commit ") and "Index frisch" in block.splitlines()[0]
    assert "app/versand.py:1" in block
    assert "docs/newsletter.md" in block
    assert "dienst" in block and "app/" in block  # Strukturregel


def test_ohne_yaml_schweigt(git_repo, heim):
    _aufbauen(git_repo, regeln="")
    assert kontextblock(git_repo.wurzel, "newsletter freigeben") is None
    assert kontextblock(git_repo.wurzel.parent, "newsletter freigeben") is None


def test_ohne_treffer_schweigt(git_repo, heim):
    _aufbauen(git_repo)
    assert kontextblock(git_repo.wurzel, "zebra xylophon quantum") is None


def test_geaenderte_datei_nur_name(git_repo, heim):
    _aufbauen(git_repo)
    git_repo.datei("app/versand.py", "# fremde Aenderung\ndef newsletter_freigeben():\n    return 2\n")
    block = kontextblock(git_repo.wurzel, "newsletter freigeben")
    assert block is not None
    assert "seit Index geändert" in block
    assert "app/versand.py:1" not in block and "app/versand.py:2" not in block


def test_crlf_arbeitsbaum_gilt_als_frisch(git_repo, heim):
    _aufbauen(git_repo)
    git_repo.datei("app/versand.py", b"def newsletter_freigeben():\r\n    return 1\r\n")
    block = kontextblock(git_repo.wurzel, "newsletter freigeben")
    assert block is not None and "app/versand.py:1" in block


def test_veralteter_index_kopf(git_repo, heim):
    _aufbauen(git_repo)
    git_repo.datei("neu.txt", "x")
    git_repo.commit("weiter")
    block = kontextblock(git_repo.wurzel, "newsletter freigeben")
    assert block is not None and "Index veraltet" in block.splitlines()[0]
    assert "app/versand.py:1" in block  # Datei selbst unveraendert -> Frische-Regel erfuellt


def test_frist(git_repo, heim):
    _aufbauen(git_repo)
    assert kontextblock(git_repo.wurzel, "newsletter freigeben", frist_s=0.0) is None


def test_budget(git_repo, heim):
    regeln = "struktur:\n" + "".join(
        f"  - art: dienst{i}\n    ort: app{i}/\n    pflicht: [README.md]\n" for i in range(600))
    _aufbauen(git_repo, regeln=regeln)
    block = kontextblock(git_repo.wurzel, "newsletter freigeben")
    assert block is not None and len(block) <= 8000


def test_stand_fuer_wie_vermessen(git_repo, heim):
    _aufbauen(git_repo)
    assert stand_fuer(git_repo.wurzel) == stand_finden(git_repo.wurzel)
    assert stand_fuer(git_repo.wurzel.parent) is None


def test_repo_wurzeln_aussen_nach_innen(tmp_path):
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "selbstbild.yaml").write_text("name: a\n", encoding="utf-8")
    (tmp_path / "a" / "b" / "selbstbild.yaml").write_text("name: b\n", encoding="utf-8")
    assert repo_wurzeln(tmp_path / "a" / "b" / "c") == (tmp_path / "a", tmp_path / "a" / "b")
    assert repo_wurzeln(tmp_path) == ()


def test_blob_sha(tmp_path):
    d = tmp_path / "x.txt"
    d.write_bytes(b"a\r\nb\r\n")
    roh, lf = blob_sha(d)
    import hashlib
    assert roh == hashlib.sha1(b"blob 6\0a\r\nb\r\n").hexdigest()
    assert lf == hashlib.sha1(b"blob 4\0a\nb\n").hexdigest()


def test_regeln_lesen():
    text = ("name: demo  # Kommentar\nstruktur:\n  - art: dienst\n    ort: \"app/\"\n    pflicht: [README.md, 'x y']\n"
            "    verboten: [tmp]\n    hinweis: Bitte hier ablegen\n  - art: w\n    ort: .github/\n    pflicht: []\n")
    r = regeln_lesen(text)
    assert r["name"] == "demo"
    assert r["struktur"][0] == {"art": "dienst", "ort": "app/", "pflicht": ["README.md", "x y"],
                                "verboten": ["tmp"], "hinweis": "Bitte hier ablegen"}
    assert r["struktur"][1]["pflicht"] == [] and r["struktur"][1].get("hinweis") is None
    assert regeln_lesen("struktur:\n  - [kaputt\n") == {}
    assert regeln_lesen("") == {}


def _lauf(cwd: Path, prompt: str, heim: Path, **extra: str) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ, PYTHONPATH=str(SRC), SELBSTBILD_HOME=str(heim), **extra)
    return subprocess.run([sys.executable, "-m", "selbstbild.hooks.einspeisung"], cwd=cwd, env=env,
                          input=json.dumps({"prompt": prompt, "cwd": str(cwd)}).encode(),
                          capture_output=True, timeout=60)


def test_main_ausgabe_und_latenz(git_repo, heim):
    _aufbauen(git_repo)
    erg = _lauf(git_repo.wurzel, "newsletter freigeben", heim, SELBSTBILD_HOOK_ZEIT="1")
    assert erg.returncode == 0
    antwort = json.loads(erg.stdout)
    assert list(antwort) == ["hookSpecificOutput"]
    h = antwort["hookSpecificOutput"]
    assert h["hookEventName"] == "UserPromptSubmit" and "app/versand.py:1" in h["additionalContext"]
    ms = float(erg.stderr.decode().split("ms")[0].split()[-1])
    assert ms < 1500
    assert b"newsletter" not in erg.stderr  # Prompt nie echoen


def test_main_schweigt_bei_muell(git_repo, heim):
    erg = subprocess.run([sys.executable, "-m", "selbstbild.hooks.einspeisung"], cwd=git_repo.wurzel,
                         env=dict(os.environ, PYTHONPATH=str(SRC), SELBSTBILD_HOME=str(heim)),
                         input=b"kein json", capture_output=True, timeout=60)
    assert erg.returncode == 0 and erg.stdout == b""


def test_keine_schweren_importe():
    code = ("import sys; import selbstbild.hooks.einspeisung; "
            "bad=[m for m in ('numpy','igraph','yaml','tree_sitter','selbstbild.speicher','selbstbild.suche') "
            "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    erg = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=str(SRC)),
                         capture_output=True, timeout=60)
    assert erg.returncode == 0, erg.stdout


# --- Fix-Runde 1 -----------------------------------------------------------------------------------------------
def test_index_aelter_als_stand_schweigt(git_repo, heim):
    """Ruling U6: Index-Commit != Graph-Commit -> stille Ausgabe."""
    _aufbauen(git_repo)
    git_repo.datei("neu.txt", "x")
    git_repo.commit("weiter")
    vermessen(git_repo.wurzel)  # Graph neu, Index nicht
    assert kontextblock(git_repo.wurzel, "newsletter freigeben") is None


def test_veraltet_geaenderte_stellen_getrennt(git_repo, heim):
    """Index = Graph, aber HEAD weiter: Frisch-Treffer mit Zeile, Geaenderte nur als Name in eigener Liste."""
    _aufbauen(git_repo)
    git_repo.datei("neu.txt", "x")
    git_repo.commit("weiter")
    git_repo.datei("app/versand.py", "# anders\ndef newsletter_freigeben():\n    return 2\n")
    block = kontextblock(git_repo.wurzel, "newsletter freigeben")
    assert block is not None and "Index veraltet" in block.splitlines()[0]
    teile = block.split("Seit Index geändert:")
    assert len(teile) == 2 and "docs/newsletter.md" in teile[0]
    assert "newsletter_freigeben (app/versand.py)" in teile[1]
    assert "app/versand.py:" not in block


def test_veraltet_ohne_frisch_treffer_schweigt(git_repo, heim):
    _aufbauen(git_repo)
    git_repo.datei("neu.txt", "x")
    git_repo.commit("weiter")
    git_repo.datei("app/versand.py", "# anders\n")
    git_repo.datei("docs/newsletter.md", "# anders\n")
    assert kontextblock(git_repo.wurzel, "newsletter freigeben") is None


def test_geaenderte_doku_nicht_als_treffer(git_repo, heim):
    _aufbauen(git_repo)
    git_repo.datei("docs/newsletter.md", "# fremd\n")
    block = kontextblock(git_repo.wurzel, "newsletter freigeben")
    assert block is not None and "Doku:" not in block
    assert "docs/newsletter.md – seit Index geändert" in block


def test_grosser_prompt_schnell(git_repo, heim):
    _aufbauen(git_repo)
    import time
    prompt = "lorem ipsum dolor " * 120_000  # > 2 MB
    t = time.monotonic()
    kontextblock(git_repo.wurzel, prompt)
    assert time.monotonic() - t < 0.5


def test_grosse_datei_wird_nicht_gelesen(git_repo, heim, monkeypatch):
    from selbstbild.hooks import einspeisung

    _aufbauen(git_repo)
    git_repo.datei("app/versand.py", b"x" * (600 * 1024))

    echt = einspeisung.blob_sha

    def nie(datei):
        assert datei.name != "versand.py", "grosse Datei gelesen"
        return echt(datei)

    monkeypatch.setattr(einspeisung, "blob_sha", nie)
    block = kontextblock(git_repo.wurzel, "newsletter freigeben")
    assert block is not None and "seit Index geändert" in block and "app/versand.py:" not in block


def test_fortschrittshandler_bricht_ab(git_repo, heim):
    from selbstbild.hooks import einspeisung
    import sqlite3
    import time

    _aufbauen(git_repo)
    stand = stand_fuer(git_repo.wurzel)
    ablauf = time.monotonic() + 0.1

    def pruefen():
        if time.monotonic() > ablauf:
            raise einspeisung._Frist

    c = einspeisung._ro(stand.parent / "suche.sqlite", pruefen)
    t = time.monotonic()
    with pytest.raises(sqlite3.OperationalError):
        c.execute("WITH RECURSIVE z(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM z) SELECT count(*) FROM z").fetchone()
    c.close()
    assert time.monotonic() - t < 2


def test_main_harter_stopp(monkeypatch, capsysbinary):
    import io
    import time
    from selbstbild.hooks import einspeisung

    def langsam(cwd, prompt, *, frist_s):
        time.sleep(5)
        return "sollte nie erscheinen"

    monkeypatch.setattr(einspeisung, "kontextblock", langsam)
    monkeypatch.setattr(einspeisung, "FRIST_S", 0.3)

    class Std:
        buffer = io.BytesIO(json.dumps({"prompt": "x", "cwd": "."}).encode())

    monkeypatch.setattr(einspeisung.sys, "stdin", Std)
    t = time.monotonic()
    assert einspeisung.main() == 0
    assert time.monotonic() - t < 2
    assert capsysbinary.readouterr().out == b""


SHA = "a" * 40
SHA2 = "b" * 40


def test_git_head_gepackt(tmp_path):
    from selbstbild.hooks.einspeisung import _git_head

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git" / "packed-refs").write_text(f"# pack-refs\n{SHA} refs/heads/main\n{SHA2} refs/heads/x\n")
    assert _git_head(tmp_path) == SHA


def test_git_head_detached_und_unklar(tmp_path):
    from selbstbild.hooks.einspeisung import _git_head

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text(SHA2 + "\n")
    assert _git_head(tmp_path) == SHA2
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/fehlt\n")
    assert _git_head(tmp_path) is None
    assert _git_head(tmp_path / "ohne") is None


def test_git_head_worktree_gitdir_commondir(tmp_path):
    from selbstbild.hooks.einspeisung import _git_head

    haupt = tmp_path / "haupt" / ".git"
    (haupt / "refs" / "heads").mkdir(parents=True)
    (haupt / "refs" / "heads" / "feature").write_text(SHA + "\n")
    wt_git = haupt / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/feature\n")
    (wt_git / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wt_git}\n")
    assert _git_head(wt) == SHA
