"""Tests fuer die Git-Quelle."""
from __future__ import annotations

from selbstbild.quelle import BlobLeser, baum_lesen


def _gruende(baum) -> dict[str, str]:
    return {u.pfad: u.grund for u in baum.uebersprungen}


def test_nur_committet(git_repo):
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("a")
    git_repo.datei("a.py", "x = 2\n")
    git_repo.datei("b.py", "y = 1\n")
    baum = baum_lesen(git_repo.wurzel)
    assert [e.pfad for e in baum.eintraege] == ["a.py"]
    with BlobLeser(git_repo.wurzel) as leser:
        assert leser.lesen(baum.eintraege[0].blob) == b"x = 1\n"
    assert baum.eintraege[0].sprache == "python"
    assert baum.eintraege[0].repo == git_repo.wurzel.name
    assert baum.commit_zeit == 1767225600


def test_binaer_und_gross(git_repo):
    git_repo.datei("bild.png", b"\x89PNG\x00\x01\x02")
    git_repo.datei("gross.py", "x = 1\n" * 100_000)
    git_repo.datei("modell.bin.py", "version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 9\n")
    git_repo.commit("x")
    g = _gruende(baum_lesen(git_repo.wurzel))
    assert g["bild.png"] == "binaer"
    assert g["gross.py"] == "zu_gross"
    assert g["modell.bin.py"] == "lfs"


def test_env_nie_gelesen(git_repo):
    git_repo.datei(".env", "GEHEIM=1\n")
    git_repo.datei(".env.example", "GEHEIM=\n")
    git_repo.commit("env")
    baum = baum_lesen(git_repo.wurzel)
    assert _gruende(baum)[".env"] == "env_datei"
    assert [(e.pfad, e.sprache) for e in baum.eintraege] == [(".env.example", "dotenv")]


def test_ausgenommen(git_repo):
    git_repo.datei("node_modules/x.js", "1\n")
    git_repo.datei("alt/y.py", "1\n")
    git_repo.datei("a/b/c.min.js", "1\n")
    git_repo.datei("ok.py", "1\n")
    git_repo.datei("selbstbild.yaml", "ausgenommen: ['alt/**']\n")
    git_repo.commit("x")
    baum = baum_lesen(git_repo.wurzel)
    g = _gruende(baum)
    assert g["node_modules/x.js"] == "ausgenommen"
    assert g["alt/y.py"] == "ausgenommen"
    assert g["a/b/c.min.js"] == "ausgenommen"
    assert "ok.py" in [e.pfad for e in baum.eintraege]


def test_submodul(git_repo, tmp_path):
    sub = type(git_repo)(tmp_path / "subquelle", dict(__import__("os").environ))
    sub.datei("s.py", "z = 1\n")
    sub.commit("s")
    git_repo.datei("a.py", "x = 1\n")
    git_repo.submodul("sub", sub)
    git_repo.commit("mit sub")
    baum = baum_lesen(git_repo.wurzel)
    assert baum.unterbaeume[0].repo == f"{git_repo.wurzel.name}/sub"
    assert [e.pfad for e in baum.unterbaeume[0].eintraege] == ["s.py"]
    assert [e.pfad for e in baum.eintraege] == [".gitmodules", "a.py"] or "a.py" in [e.pfad for e in baum.eintraege]


def test_submodul_fehlt(git_repo, tmp_path):
    sub = type(git_repo)(tmp_path / "subquelle", dict(__import__("os").environ))
    sub.datei("s.py", "z = 1\n")
    sub.commit("s")
    git_repo.datei("a.py", "x = 1\n")
    git_repo.submodul("sub", sub)
    git_repo.git("add", "a.py")
    git_repo.git("update-index", "--cacheinfo", "160000," + "a" * 40 + ",sub")
    git_repo.git("commit", "-q", "-m", "kaputter pin")
    baum = baum_lesen(git_repo.wurzel)
    assert _gruende(baum)["sub"] == "submodul_fehlt"
    assert baum.unterbaeume == ()
    assert "a.py" in [e.pfad for e in baum.eintraege]


def test_regeln_aus_commit(git_repo):
    git_repo.datei("a.py", "x = 1\n")
    git_repo.datei("selbstbild.yaml", "name: mein-name\n")
    git_repo.commit("r")
    git_repo.datei("selbstbild.yaml", "name: anderer\nmax_dateigroesse_kb: 1\n")
    baum = baum_lesen(git_repo.wurzel)
    assert baum.regeln.name == "mein-name"
    assert baum.repo == "mein-name"
    assert baum.regeln.max_dateigroesse_kb == 512
