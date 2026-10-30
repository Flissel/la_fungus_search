"""doku_loop: Messbaum (Task 15c, Ruling 50) -- detached Worktrees des committeten Stands, auch fuer verschachtelte
Submodule. Alles gegen Fixture-Repos unter tmp_path; der lokale Ordner liegt unter tmp_path/lokal."""

import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import messbaum as mb  # noqa: E402
from selbstbild.doku.schreibvorgang import versionierte_dateien  # noqa: E402



def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "core.quotepath=false", "-c", "protocol.file.allow=always", "-C", str(repo),
                           *args], capture_output=True, text=True, encoding="utf-8", check=True, timeout=60).stdout


def _neu(pfad: Path, dateien: dict[str, str]) -> Path:
    pfad.mkdir(parents=True)
    _git(pfad, "init", "-q")
    for k, v in (("user.name", "Test"), ("user.email", "test@example.invalid"), ("core.autocrlf", "false"),
                 ("commit.gpgsign", "false")):
        _git(pfad, "config", k, v)
    for name, text in dateien.items():
        (pfad / name).parent.mkdir(parents=True, exist_ok=True)
        (pfad / name).write_text(text, encoding="utf-8")
    _git(pfad, "add", "-A")
    _git(pfad, "commit", "-q", "-m", "start")
    return pfad


def _lesen(p: Path) -> str:
    return p.read_text(encoding="utf-8")


@pytest.fixture
def mfx(tmp_path, monkeypatch):
    """aussen -> vibemind-os (Submodul) -> openfang (verschachtelt, initialisiert) + extra (nicht initialisiert)."""
    lokal = tmp_path / "lokal"
    monkeypatch.setenv("LOCALAPPDATA", str(lokal))
    q_openfang = _neu(tmp_path / "q_openfang", {"src/lib.py": "def f():\n    return 1\n"})
    q_extra = _neu(tmp_path / "q_extra", {"x.py": "X = 1\n"})
    q_os = _neu(tmp_path / "q_os", {"spaces/a/client.py": "def senden():\n    return 1\n"})
    _git(q_os, "submodule", "add", "-q", str(q_openfang), "openfang")
    _git(q_os, "submodule", "add", "-q", str(q_extra), "extra")
    _git(q_os, "commit", "-q", "-m", "submodule")
    repo = _neu(tmp_path / "repo", {"README.md": "aussen\n", ".gitignore": "logs/\n"})
    _git(repo, "submodule", "add", "-q", str(q_os), "vibemind-os")
    _git(repo / "vibemind-os", "submodule", "update", "-q", "--init", "openfang")  # extra bleibt leer
    roh = yaml.safe_load((VORLAGE).read_text(encoding="utf-8"))
    roh.update({"spaces": {"a": {"wurzel": "vibemind-os/spaces/a", "mcp_muster": "werkzeuge"}},
                "marker": str(lokal / "VibeMind" / "doku_loop" / "anstoss"), "waechter_zustandsdatei": "logs/w.json"})
    (repo / "config").mkdir()
    (repo / "config" / "doku_loop.yaml").write_text(yaml.safe_dump(roh, allow_unicode=True), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "start")
    konfig = konfig_modul.laden(repo / "config" / "doku_loop.yaml")
    return dataclasses.replace(konfig, repo=repo)


def _worktrees(repo: Path) -> list[str]:
    return [z.split(" ", 1)[1] for z in _git(repo, "worktree", "list", "--porcelain").splitlines()
            if z.startswith("worktree ")]


def test_messbaum_spiegelt_head_nicht_arbeitsbaum(mfx):
    repo = mfx.repo
    (repo / "README.md").write_text("uncommittet\n", encoding="utf-8")
    (repo / "neu.txt").write_text("untracked\n", encoding="utf-8")
    (repo / "vibemind-os/spaces/a/client.py").write_text("def senden():\n    return 2\n", encoding="utf-8")
    (repo / "vibemind-os/openfang/src/lib.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    m = mb.bereitstellen(mfx)
    w = mb.messbaum_pfad(mfx)
    assert m.wurzel == w == mfx.marker.parent / f"messbaum-{mb.repo_schluessel(repo)}"
    assert _lesen(w / "README.md") == "aussen\n" and not (w / "neu.txt").exists()
    assert _lesen(w / "vibemind-os/spaces/a/client.py") == "def senden():\n    return 1\n"
    assert _lesen(w / "vibemind-os/openfang/src/lib.py") == "def f():\n    return 1\n"  # verschachtelt
    assert not any((w / "vibemind-os/extra").iterdir())  # nicht initialisiert -> leer
    assert m.nicht_messbar == ("vibemind-os/extra",)
    assert m.commits == {"": _git(repo, "rev-parse", "HEAD").strip(),
                         "vibemind-os": _git(repo / "vibemind-os", "rev-parse", "HEAD").strip(),
                         "vibemind-os/openfang": _git(repo / "vibemind-os/openfang", "rev-parse", "HEAD").strip()}
    # Konfig: nur repo (und die Waechter-Datei als Laufzeitzustand im echten Arbeitsbaum) anders
    assert m.konfig.repo == w and m.konfig.spaces == mfx.spaces and m.konfig.marker == mfx.marker
    assert Path(m.konfig.waechter_zustandsdatei) == repo / "logs/w.json"
    assert mb.messbaum_bereitstellen(mfx) == m.konfig
    # der echte Arbeitsbaum ist unveraendert, keine Branches angefasst
    assert _lesen(repo / "README.md") == "uncommittet\n" and (repo / "neu.txt").exists()
    assert _git(repo, "branch", "--show-current").strip() in ("master", "main")
    assert "vibemind-os/openfang/src/lib.py" in versionierte_dateien(repo)  # verschachtelt versioniert
    assert not any(f.startswith("vibemind-os/extra/") for f in versionierte_dateien(repo))
    st = mb.messbaum_status(mfx)
    assert st["vorhanden"] and st["worktrees"] == 3 and st["nicht_messbar"] == ["vibemind-os/extra"]


def test_messbaum_inkrementell_nach_commit(mfx):
    repo = mfx.repo
    mb.bereitstellen(mfx)
    w = mb.messbaum_pfad(mfx)
    vorher = sorted(_worktrees(repo) + _worktrees(repo / "vibemind-os"))
    (repo / "README.md").write_text("neu\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "aussen")
    os_ = repo / "vibemind-os"
    (os_ / "spaces/a/client.py").write_text("def senden():\n    return 3\n", encoding="utf-8")
    _git(os_, "commit", "-q", "-am", "submodul (HEAD != Pin im aeusseren Repo)")
    of = os_ / "openfang"
    (of / "src/lib.py").write_text("def f():\n    return 3\n", encoding="utf-8")
    _git(of, "commit", "-q", "-am", "verschachtelt")
    assert _lesen(w / "README.md") == "aussen\n"  # erst nach dem naechsten Bereitstellen
    m = mb.bereitstellen(mfx)
    assert _lesen(w / "README.md") == "neu\n"
    assert _lesen(w / "vibemind-os/spaces/a/client.py") == "def senden():\n    return 3\n"  # Submodul-HEAD, nicht Pin
    assert _lesen(w / "vibemind-os/openfang/src/lib.py") == "def f():\n    return 3\n"
    assert m.commits["vibemind-os"] == _git(os_, "rev-parse", "HEAD").strip()
    assert sorted(_worktrees(repo) + _worktrees(os_)) == vorher  # inkrementell: dieselben Worktrees


def test_messbaum_verwirft_lokale_aenderungen_nur_im_messbaum(mfx):
    m = mb.bereitstellen(mfx)
    w = m.wurzel
    (w / "README.md").write_text("im messbaum veraendert\n", encoding="utf-8")
    (w / "vibemind-os/spaces/a/rest.py").write_text("x = 1\n", encoding="utf-8")
    (mfx.repo / "README.md").write_text("echte Arbeit\n", encoding="utf-8")
    mb.bereitstellen(mfx)
    assert _lesen(w / "README.md") == "aussen\n" and not (w / "vibemind-os/spaces/a/rest.py").exists()
    assert _lesen(mfx.repo / "README.md") == "echte Arbeit\n"  # echter Arbeitsbaum unberuehrt
    assert _git(mfx.repo, "stash", "list").strip() == ""


def test_messbaum_entfernt_nie_fremde_worktrees(mfx, tmp_path):
    repo = mfx.repo
    fremd = tmp_path / "fremder_worktree"
    _git(repo, "worktree", "add", "-q", "--detach", str(fremd), "HEAD")
    weg = tmp_path / "fremd_weg"
    _git(repo, "worktree", "add", "-q", "--detach", str(weg), "HEAD")
    shutil.rmtree(weg)  # verwaist (prunable): ein `git worktree prune` wuerde den Eintrag loeschen
    mb.bereitstellen(mfx)
    mb.bereitstellen(mfx)
    liste = [Path(p).resolve() for p in _worktrees(repo)]
    assert fremd.resolve() in liste and weg.resolve() in liste and (fremd / "README.md").is_file()
    # ein fremder (nicht markierter) Worktree am Messbaum-Pfad wird nicht angefasst -> Fehler statt Uebernahme
    w = mb.messbaum_pfad(mfx)
    shutil.rmtree(mfx.marker.parent)
    _git(repo, "worktree", "add", "-q", "--force", "--detach", str(w), "HEAD")  # wie von Hand angelegt
    with pytest.raises(mb.MessbaumFehler) as fehler:
        mb.bereitstellen(mfx)
    assert fehler.value.grund == "fremder_worktree:."
    assert (w / "README.md").is_file() and w.resolve() in [Path(p).resolve() for p in _worktrees(repo)]


def test_messbaum_fremder_inhalt_am_pfad_ist_fehler(mfx):
    w = mb.messbaum_pfad(mfx)
    w.mkdir(parents=True)
    (w / "privat.txt").write_text("nicht vom Loop\n", encoding="utf-8")
    with pytest.raises(mb.MessbaumFehler) as fehler:
        mb.bereitstellen(mfx)
    assert fehler.value.grund == "fremder_pfad:." and _lesen(w / "privat.txt") == "nicht vom Loop\n"


def test_messbaum_nach_geloeschtem_ordner_neu_angelegt(mfx):
    mb.bereitstellen(mfx)
    w = mb.messbaum_pfad(mfx)
    shutil.rmtree(w)  # Eintraege bleiben registriert (kein prune) -> erneut anlegen ueberschreibt nur unsere Pfade
    m = mb.bereitstellen(mfx)
    assert _lesen(w / "vibemind-os/openfang/src/lib.py") == "def f():\n    return 1\n" and len(m.commits) == 3


def test_messbaum_entfernt_eigenen_worktree_nach_deinit(mfx):
    os_ = mfx.repo / "vibemind-os"
    mb.bereitstellen(mfx)
    w = mb.messbaum_pfad(mfx)
    assert (w / "vibemind-os/openfang/src/lib.py").is_file()
    _git(os_, "submodule", "deinit", "-q", "-f", "openfang")
    m = mb.bereitstellen(mfx)
    assert m.nicht_messbar == ("vibemind-os/extra", "vibemind-os/openfang")
    assert not (w / "vibemind-os/openfang/src/lib.py").exists()  # alter Code wird nicht mehr gemessen
    assert "vibemind-os/openfang" not in m.commits


def test_messbaum_submodul_mit_worktree_config_trifft_nie_den_modulordner(mfx):
    """Wie `openclaw` im echten Repo: Modul-Konfiguration mit `core.worktree` UND `extensions.worktreeConfig=true`.
    Dann loest git in einem Worktree die Arbeitskopie auf den Modul-Ordner selbst auf; der Messbaum darf sich darauf
    nie verlassen (sonst trafen `checkout --force`/`clean -fdx` den Verwaltungsordner)."""
    of = mfx.repo / "vibemind-os/openfang"
    modul = Path(_git(of, "rev-parse", "--absolute-git-dir").strip())
    assert _git(of, "config", "--get", "core.worktree").strip()  # wie von `submodule update --init` angelegt
    _git(of, "config", "extensions.worktreeConfig", "true")
    m = mb.bereitstellen(mfx)
    ziel = m.wurzel / "vibemind-os/openfang"
    oben = subprocess.run(["git", "-C", str(ziel), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    assert Path(oben.stdout.strip()).resolve() != ziel.resolve()  # die Falle ist im Fixture nachgestellt
    vorher = sorted(p.name for p in modul.iterdir())
    (ziel / "src/lib.py").write_text("verdorben\n", encoding="utf-8")
    (ziel / "muell.txt").write_text("x\n", encoding="utf-8")
    m = mb.bereitstellen(mfx)  # zweites Bereitstellen: kein `fremder_worktree`, Aufraeumen nur im Messbaum
    assert _lesen(ziel / "src/lib.py") == "def f():\n    return 1\n" and not (ziel / "muell.txt").exists()
    assert sorted(p.name for p in modul.iterdir()) == vorher  # Modul-Ordner unberuehrt
    assert _lesen(of / "src/lib.py") == "def f():\n    return 1\n"  # echter Checkout unberuehrt
    (of / "src/lib.py").write_text("def f():\n    return 4\n", encoding="utf-8")
    _git(of, "commit", "-q", "-am", "neu")
    mb.bereitstellen(mfx)
    assert _lesen(ziel / "src/lib.py") == "def f():\n    return 4\n"
    assert not (modul / "src").exists() and not (modul / "muell.txt").exists()  # nie in den Modul-Ordner ausgecheckt


# ----- Fix-Runde 1 -----

def test_messbaum_fuehrt_nie_hooks_oder_lfs_filter_aus(mfx, tmp_path):
    """Hooks des echten Repos (core.hooksPath, wie .githooks) und ein LFS-Filter (smudge + process, required)
    duerfen beim Anlegen und Auschecken nie laufen -- sie schrieben Dateien im echten Checkout bzw. gingen ins Netz."""
    repo = mfx.repo
    (repo / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    (repo / "daten.bin").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n",
                                    encoding="utf-8")
    _git(repo, "add", ".gitattributes", "daten.bin")
    _git(repo, "commit", "-q", "-m", "lfs")
    spur = tmp_path / "spur"
    spur.mkdir()
    hooks = tmp_path / "echte_hooks"
    hooks.mkdir()
    for name in ("post-checkout", "post-merge", "reference-transaction"):
        (hooks / name).write_text(f'#!/bin/sh\necho {name} >> "{(spur / "hook").as_posix()}"\n', encoding="utf-8")
    for wo in (repo, repo / "vibemind-os", repo / "vibemind-os/openfang"):
        _git(wo, "config", "core.hooksPath", hooks.as_posix())
    _git(repo, "config", "filter.lfs.smudge", f'echo smudge >> "{(spur / "smudge").as_posix()}"; cat')
    _git(repo, "config", "filter.lfs.process", f'echo process >> "{(spur / "process").as_posix()}"')
    _git(repo, "config", "filter.lfs.required", "true")
    m = mb.bereitstellen(mfx)  # Anlegen
    assert sorted(p.name for p in spur.iterdir()) == []  # Anlegen: weder Hook noch Filter
    (repo / "README.md").write_text("weiter\n", encoding="utf-8")
    # der eigene Commit des Tests ohne den Fake-Filter (er beherrscht das Prozess-Protokoll nicht)
    _git(repo, "-c", "filter.lfs.process=", "-c", "filter.lfs.clean=", "-c", "filter.lfs.required=false",
         "commit", "-q", "-am", "weiter")
    for p in spur.iterdir():  # der Commit des Tests selbst darf Hooks laufen lassen
        p.unlink()
    (m.wurzel / "daten.bin").write_text("verdorben\n", encoding="utf-8")  # erzwingt checkout --force + clean
    mb.bereitstellen(mfx)  # Auschecken (neuer Commit + Verwerfen)
    assert sorted(p.name for p in spur.iterdir()) == []
    assert _lesen(m.wurzel / "daten.bin").startswith("version https://git-lfs")  # Zeiger, nicht geglaettet
    assert _lesen(m.wurzel / "README.md") == "weiter\n"
    leer = mb.hooks_pfad(mfx)
    assert leer.is_dir() and not any(leer.iterdir()) and leer.parent == mfx.marker.parent


def test_messbaum_gescheiterte_entfernung_bleibt_im_zustand(mfx):
    os_ = mfx.repo / "vibemind-os"
    m = mb.bereitstellen(mfx)
    of = m.wurzel / "vibemind-os/openfang"
    (mb._gitdir(of) / mb.MARKER).unlink()  # nicht mehr als eigener erkennbar -> darf nicht entfernt werden
    _git(os_, "submodule", "deinit", "-q", "-f", "openfang")
    mb.bereitstellen(mfx)
    assert (of / "src/lib.py").is_file()  # nicht angefasst
    st = mb.messbaum_status(mfx)
    assert st["entfernen_offen"] == ["vibemind-os/openfang"]
    mb.bereitstellen(mfx)  # wird erneut versucht und bleibt offen
    assert mb.messbaum_status(mfx)["entfernen_offen"] == ["vibemind-os/openfang"]


def test_messbaum_ueberschreibt_keine_fremde_registrierung(mfx):
    repo = mfx.repo
    w = mb.messbaum_pfad(mfx)
    _git(repo, "worktree", "add", "-q", "--detach", str(w), "HEAD")  # fremd, ohne Marker
    shutil.rmtree(w)  # Ordner weg, Registrierung bleibt
    with pytest.raises(mb.MessbaumFehler) as fehler:
        mb.bereitstellen(mfx)
    assert fehler.value.grund == "fremde_registrierung:."
    assert w.resolve() in [Path(p).resolve() for p in _worktrees(repo)]  # nicht ueberschrieben, nicht geprunt


@pytest.mark.parametrize("marker, grund", [
    ("%LOCALAPPDATA%/VibeMind/doku_loop/anstoss", "pfad_nicht_absolut"),
    ("$LOCALAPPDATA/VibeMind/doku_loop/anstoss", "pfad_nicht_absolut"),
    ("IM_REPO", "pfad_im_repo"),
])
def test_messbaum_wurzel_absolut_und_nie_im_repo(mfx, marker, grund):
    pfad = mfx.repo / "lokal" / "anstoss" if marker == "IM_REPO" else Path(marker)
    with pytest.raises(mb.MessbaumFehler) as fehler:
        mb.bereitstellen(dataclasses.replace(mfx, marker=pfad))
    assert fehler.value.grund == grund
    assert not (mfx.repo / "lokal").exists() and not Path("%LOCALAPPDATA%").exists()


def test_messbaum_anlegen_fehler_nennt_git_grund(mfx, tmp_path):
    hooks = mb.hooks_pfad(mfx)
    hooks.mkdir(parents=True)
    w = mb.messbaum_pfad(mfx)
    with pytest.raises(mb.MessbaumFehler) as fehler:
        mb._anlegen(mfx.repo, w, "0" * 40, w, "", hooks)
    assert fehler.value.grund.startswith("anlegen:.:")
    rest = fehler.value.grund[len("anlegen:.:"):]
    assert rest and "\n" not in rest and len(rest) <= 160


def test_messbaum_unlesbare_git_datei_ist_fehler(tmp_path):
    ziel = tmp_path / "wt"
    ziel.mkdir()
    (ziel / ".git").write_bytes(b"gitdir: \xff\xfe\x00kaputt\n")
    with pytest.raises(mb.MessbaumFehler) as fehler:
        mb._gitdir(ziel)
    assert fehler.value.grund == "gitdir_unlesbar:wt"


def test_messbaum_kein_head_ist_fehler(tmp_path, monkeypatch, mfx):
    leer = tmp_path / "leer"
    leer.mkdir()
    _git(leer, "init", "-q")
    with pytest.raises(mb.MessbaumFehler) as fehler:
        mb.bereitstellen(dataclasses.replace(mfx, repo=leer))
    assert fehler.value.grund == "kein_head"
