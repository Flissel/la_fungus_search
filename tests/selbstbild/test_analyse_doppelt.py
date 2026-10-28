from __future__ import annotations

import os

from selbstbild.analyse.doppelt import Doppelung, doppelungen
from selbstbild.analyse.lader import GraphDaten
from selbstbild.parser.basis import parsen
from selbstbild.typen import Beleg, Knoten, id_datei, id_symbol


def _daten(commit: str, knoten: list[Knoten] | None = None) -> GraphDaten:
    return GraphDaten(commit, 1_700_000_000, {k.id: k for k in (knoten or [])}, (), {})


def _gross(praefix: str) -> str:
    """Rund 40 Zeilen Python; `praefix` benennt alle Bezeichner um."""
    zeilen = []
    for i in range(10):
        zeilen += [
            f"def {praefix}_fn{i}({praefix}_a, {praefix}_b):",
            f"    {praefix}_c = {praefix}_a + {praefix}_b * {i}",
            f"    if {praefix}_c > {i + 3}:",
            f"        return {praefix}_c - {praefix}_a",
        ]
    return "\n".join(zeilen) + "\n"


def _symbolknoten(repo: str, pfad: str, inhalt: str) -> list[Knoten]:
    erg = parsen("python", pfad, inhalt.encode())
    return [Knoten(id_symbol(repo, pfad, s.qualname), "symbol", s.qualname,
                   Beleg(repo, pfad, s.zeile_von, s.qualname, s.digest)) for s in erg.symbole]


def test_gleiche_datei_zwei_repos(git_repo, tmp_path):
    sub = type(git_repo)(tmp_path / "subquelle", dict(os.environ))
    text = _gross("x")
    sub.datei("kopie.py", text)
    sub.commit("s")
    git_repo.datei("orig.py", text)
    git_repo.submodul("sub", sub)
    sha = git_repo.commit("mit sub")
    erg = doppelungen(git_repo.wurzel, _daten(sha))
    gleich = [x for x in erg if x.art == "gleiche_datei"]
    name = git_repo.wurzel.name
    assert gleich == [Doppelung("gleiche_datei", tuple(sorted((
        id_datei(f"{name}/sub", "kopie.py"), id_datei(name, "orig.py")))), 1.0)]
    assert not [x for x in erg if x.art == "aehnliche_datei"]


def test_gleiches_symbol(git_repo):
    fn = ("def rechne(a, b):\n    c = a + b\n    d = c * 2\n    e = d - a\n"
          "    return e\n")
    kurz = "def kurz(a):\n    return a\n"
    a = fn + kurz + "X = 1\n"
    b = "Y = 2\n" + fn + kurz
    git_repo.datei("a.py", a)
    git_repo.datei("b.py", b)
    sha = git_repo.commit("x")
    repo = git_repo.wurzel.name
    kn = _symbolknoten(repo, "a.py", a) + _symbolknoten(repo, "b.py", b)
    erg = [x for x in doppelungen(git_repo.wurzel, _daten(sha, kn)) if x.art == "gleiches_symbol"]
    assert [x.mitglieder for x in erg] == [(id_symbol(repo, "a.py", "rechne"), id_symbol(repo, "b.py", "rechne"))]
    assert erg[0].aehnlichkeit == 1.0  # `kurz` hat unter 5 Zeilen und zaehlt nicht


def test_aehnlich_umbenannt(git_repo):
    git_repo.datei("alt.py", _gross("p"))
    git_repo.datei("neu.py", _gross("q"))
    git_repo.datei("anders.py", "".join(f"print({i})  # {i * 7}\nclass K{i}: pass\n" for i in range(30)))
    sha = git_repo.commit("x")
    repo = git_repo.wurzel.name
    erg = [x for x in doppelungen(git_repo.wurzel, _daten(sha)) if x.art == "aehnliche_datei"]
    assert [x.mitglieder for x in erg] == [(id_datei(repo, "alt.py"), id_datei(repo, "neu.py"))]
    assert erg[0].aehnlichkeit >= 0.9


def test_klein_ignoriert(git_repo):
    git_repo.datei("a/__init__.py", "")
    git_repo.datei("b/__init__.py", "")
    git_repo.datei("a/k.py", "x = 1\n")
    git_repo.datei("b/k.py", "x = 1\n")
    sha = git_repo.commit("x")
    assert doppelungen(git_repo.wurzel, _daten(sha)) == ()


def test_deterministisch(git_repo):
    git_repo.datei("alt.py", _gross("p"))
    git_repo.datei("neu.py", _gross("q"))
    git_repo.datei("kopie.py", _gross("p"))
    sha = git_repo.commit("x")
    d = _daten(sha)
    e1, e2 = doppelungen(git_repo.wurzel, d), doppelungen(git_repo.wurzel, d)
    assert e1 == e2 and repr(e1) == repr(e2) and len(e1) >= 2
    assert list(e1) == sorted(e1, key=lambda x: (x.art, x.mitglieder))
