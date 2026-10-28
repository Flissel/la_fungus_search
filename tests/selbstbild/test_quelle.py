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
    git_repo.datei("daten.py", b"\x89PNG\x00\x01\x02")  # Sprache per Endung, Inhalt binaer
    git_repo.datei("gross.py", "x = 1\n" * 100_000)
    git_repo.datei("modell.bin.py", "version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 9\n")
    git_repo.commit("x")
    g = _gruende(baum_lesen(git_repo.wurzel))
    assert g["bild.png"] == "keine_sprache"  # ohne Sprache wird der Inhalt gar nicht gelesen
    assert g["daten.py"] == "binaer"
    assert g["gross.py"] == "zu_gross"
    assert g["modell.bin.py"] == "lfs"


def _spion(monkeypatch) -> list[str]:
    angefragt: list[str] = []
    original = BlobLeser.lesen

    def spion(self, blob):
        angefragt.append(blob)
        return original(self, blob)

    monkeypatch.setattr(BlobLeser, "lesen", spion)
    return angefragt


def test_keine_sprache_wird_nicht_gelesen(git_repo, monkeypatch):
    git_repo.datei("bild.png", b"\x89PNG\x00\x01\x02")
    git_repo.datei("notiz.txt", "hallo\n")
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("x")
    angefragt = _spion(monkeypatch)
    baum = baum_lesen(git_repo.wurzel)
    assert angefragt == [git_repo.git("rev-parse", "HEAD:a.py")]
    assert _gruende(baum) == {"bild.png": "keine_sprache", "notiz.txt": "keine_sprache"}


def test_bekannte_blobs_werden_nicht_gelesen(git_repo, monkeypatch):
    git_repo.datei("a.py", "x = 1\n")
    git_repo.datei("b.py", "y = 2\n")
    git_repo.commit("x")
    a_blob = git_repo.git("rev-parse", "HEAD:a.py")
    gefragt: list[tuple[str, str, str]] = []

    def bekannt(blob: str, sprache: str, pfad: str) -> bool:
        gefragt.append((blob, sprache, pfad))
        return pfad == "a.py"

    angefragt = _spion(monkeypatch)
    baum = baum_lesen(git_repo.wurzel, bekannt=bekannt)
    assert a_blob not in angefragt and len(angefragt) == 1
    assert [(e.pfad, e.blob, e.sprache) for e in baum.eintraege] == [
        ("a.py", a_blob, "python"), ("b.py", git_repo.git("rev-parse", "HEAD:b.py"), "python")]
    assert ("a.py", "python") in {(p, s) for _, s, p in gefragt}


def test_env_nie_gelesen(git_repo, monkeypatch):
    git_repo.datei(".env", "GEHEIM=1\n")
    git_repo.datei(".env.example", "GEHEIM=\n")
    git_repo.commit("env")
    env_blob = git_repo.git("rev-parse", "HEAD:.env")
    angefragt: list[str] = []
    original = BlobLeser.lesen

    def spion(self, blob):
        angefragt.append(blob)
        return original(self, blob)

    monkeypatch.setattr(BlobLeser, "lesen", spion)
    baum = baum_lesen(git_repo.wurzel)
    assert angefragt and env_blob not in angefragt
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
    assert [e.pfad for e in baum.eintraege] == ["a.py"]


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


def _sub(git_repo, tmp_path):
    sub = type(git_repo)(tmp_path / "subquelle", dict(__import__("os").environ))
    sub.datei("s.py", "z = 1\n")
    sub.commit("s")
    git_repo.datei("a.py", "x = 1\n")
    git_repo.submodul("sub", sub)
    return sub


def test_fremde_git_umgebung_ignoriert(git_repo, tmp_path, monkeypatch):
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("a")
    fremd = type(git_repo)(tmp_path / "fremd", dict(__import__("os").environ))
    fremd.datei("zzz.py", "1\n")
    fremd.commit("f")
    monkeypatch.setenv("GIT_DIR", str(fremd.wurzel / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(fremd.wurzel))
    baum = baum_lesen(git_repo.wurzel)
    assert [e.pfad for e in baum.eintraege] == ["a.py"]


def test_submodul_pin_gilt(git_repo, tmp_path):
    _sub(git_repo, tmp_path)
    pin = git_repo.commit("mit sub")
    sub_wt = git_repo.wurzel / "sub"
    sub_git = type(git_repo).__new__(type(git_repo))
    sub_git.wurzel = sub_wt
    sub_git._env = git_repo._env
    sub_git.datei("s.py", "z = 999\n")
    sub_git.commit("neu im submodul")
    baum = baum_lesen(git_repo.wurzel)
    unter = baum.unterbaeume[0]
    assert unter.commit == pin_gitlink(git_repo, pin)
    with BlobLeser(sub_wt) as leser:
        assert leser.lesen(unter.eintraege[0].blob) == b"z = 1\n"


def pin_gitlink(git_repo, commit):
    return git_repo.git("rev-parse", f"{commit}:sub")


def test_submodul_ausgenommen(git_repo, tmp_path):
    _sub(git_repo, tmp_path)
    git_repo.datei("selbstbild.yaml", "ausgenommen: ['sub/**']\n")
    git_repo.commit("x")
    baum = baum_lesen(git_repo.wurzel)
    assert _gruende(baum)["sub"] == "ausgenommen"
    assert baum.unterbaeume == ()


def test_symlink_uebersprungen(git_repo):
    git_repo.datei("a.py", "x = 1\n")
    import subprocess
    sha = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"], cwd=git_repo.wurzel,
        input=b"a.py", capture_output=True, check=True,
    ).stdout.decode().strip()
    git_repo.git("add", "a.py")
    git_repo.git("update-index", "--add", "--cacheinfo", f"120000,{sha},link.py")
    git_repo.git("commit", "-q", "-m", "link")
    baum = baum_lesen(git_repo.wurzel)
    assert _gruende(baum)["link.py"] == "symlink"
    assert [e.pfad for e in baum.eintraege] == ["a.py"]
