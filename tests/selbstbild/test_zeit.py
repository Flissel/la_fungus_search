"""Zeitachse und Lebendigkeit."""
from __future__ import annotations

import time

import pytest

from selbstbild.zeit import Aenderung, lebendigkeit, verlauf, zeit_attribute

TAG = 86400


def test_lebendigkeit_werte() -> None:
    stichtag = 1_800_000_000
    am_tag = Aenderung("a", stichtag, "x")
    assert lebendigkeit([am_tag], stichtag) == pytest.approx(1.0, abs=1e-9)
    vorher = Aenderung("b", stichtag - 30 * TAG, "x")
    assert lebendigkeit([vorher], stichtag) == pytest.approx(0.5, abs=1e-9)
    assert lebendigkeit([am_tag, am_tag], stichtag) == pytest.approx(2.0, abs=1e-9)
    spaeter = Aenderung("c", stichtag + 1, "x")
    assert lebendigkeit([spaeter], stichtag) == 0.0


def test_verlauf_umbenennung(git_repo) -> None:
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("neu", datum="2026-01-01T00:00:00Z")
    git_repo.datei("a.py", "x = 2\n")
    git_repo.commit("aendern", datum="2026-01-05T00:00:00Z")
    git_repo.mv("a.py", "b.py")
    kopf = git_repo.commit("umbenennen", datum="2026-01-10T00:00:00Z")
    v = verlauf(git_repo.wurzel, kopf)
    assert len(v["b.py"]) == 3
    assert "a.py" not in v
    zeiten = [a.zeit for a in v["b.py"]]
    assert zeiten == sorted(zeiten, reverse=True)
    assert all(a.autor == "Tester" for a in v["b.py"])


def test_verlauf_nur_bis_commit(git_repo) -> None:
    git_repo.datei("a.py", "x = 1\n")
    frueh = git_repo.commit("eins", datum="2026-01-01T00:00:00Z")
    git_repo.datei("a.py", "x = 2\n")
    git_repo.datei("neu.py", "y = 1\n")
    git_repo.commit("zwei", datum="2026-01-05T00:00:00Z")
    v = verlauf(git_repo.wurzel, frueh)
    assert len(v["a.py"]) == 1
    assert v["a.py"][0].commit == frueh
    assert "neu.py" not in v


def test_geloeschte_datei_fehlt(git_repo) -> None:
    git_repo.datei("a.py", "x\n")
    git_repo.datei("b.py", "y\n")
    git_repo.commit("eins")
    (git_repo.wurzel / "a.py").unlink()
    kopf = git_repo.commit("weg", datum="2026-01-02T00:00:00Z")
    assert set(verlauf(git_repo.wurzel, kopf)) == {"b.py"}


def test_deterministisch(git_repo, monkeypatch: pytest.MonkeyPatch) -> None:
    git_repo.datei("a.py", "x = 1\n")
    git_repo.commit("eins", datum="2026-01-01T00:00:00Z")
    git_repo.datei("a.py", "x = 2\n")
    kopf = git_repo.commit("zwei", datum="2026-01-05T00:00:00Z")
    v1 = verlauf(git_repo.wurzel, kopf)
    v2 = verlauf(git_repo.wurzel, kopf)
    assert v1 == v2
    stichtag = v1["a.py"][0].zeit
    a1 = zeit_attribute(v1["a.py"], stichtag)
    monkeypatch.setattr(time, "time", lambda: 0.0)
    a2 = zeit_attribute(v2["a.py"], stichtag)
    assert a1 == a2
    d = dict(a1)
    assert d["aenderungen"] == "2"
    assert d["zuletzt"] == str(stichtag)
    assert d["autoren"] == "1"
    assert len(d["lebendigkeit"].split(".")[1]) == 4
    assert [k for k, _ in a1] == sorted(k for k, _ in a1)
