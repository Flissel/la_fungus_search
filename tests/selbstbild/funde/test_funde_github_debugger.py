"""Debugger-Claw Task 2: GitHub-Adapter fuer den Waechter (gefaelschtes gh, nie Netz)."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from selbstbild.funde.github import DEBUGGER_MARKE, GitHub, GitHubFehler

Antwort = Callable[[str, str, Any], bytes | str]


class Fake:
    """Fake-Laufzeit: `auth token` -> Token, `api` -> `antwort(methode, pfad, daten)` (Text oder Fehler)."""

    def __init__(self, antwort: Antwort | None = None) -> None:
        self.antwort = antwort
        self.aufrufe: list[tuple[str, str, Any]] = []

    def __call__(self, args: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        argv = list(args)
        if argv[1:3] == ["auth", "token"]:
            return subprocess.CompletedProcess(argv, 0, b"tok\n", b"")
        methode = argv[argv.index("--method") + 1]
        pfad = next(a for a in argv[2:] if a.startswith("/"))
        daten = json.loads(kw["input"].decode("utf-8")) if kw.get("input") is not None else None
        self.aufrufe.append((methode, pfad, daten))
        roh: bytes | str = self.antwort(methode, pfad, daten) if self.antwort else "{}"
        if isinstance(roh, bytes):  # Fehlertext
            return subprocess.CompletedProcess(argv, 1, b"", roh)
        return subprocess.CompletedProcess(argv, 0, roh.encode("utf-8"), b"")


def _gh(antwort: Antwort | None = None, **kw: Any) -> tuple[GitHub, Fake]:
    fake = Fake(antwort)
    return GitHub("Vibemind-LAB", laufzeit=fake, **kw), fake


def test_kommentare_seit_blaettert() -> None:
    def antwort(m: str, pfad: str, d: Any) -> str:
        seite = int(pfad.rsplit("page=", 1)[1])
        return json.dumps([{"id": seite * 1000 + i} for i in range(100 if seite == 1 else 3)])

    gh, fake = _gh(antwort)
    erg = gh.kommentare_seit("o/r", "2026-11-07T10:00:00Z")
    assert len(erg) == 103
    assert [(m, p) for m, p, _ in fake.aufrufe] == [
        ("GET", "/repos/o/r/issues/comments?since=2026-11-07T10:00:00Z&per_page=100&page=1"),
        ("GET", "/repos/o/r/issues/comments?since=2026-11-07T10:00:00Z&per_page=100&page=2")]


def test_kommentare_seit_einzelne_seite_und_unlesbar() -> None:
    gh, fake = _gh(lambda m, p, d: "[]")
    assert gh.kommentare_seit("o/r", "2026-11-07T10:00:00Z") == [] and len(fake.aufrufe) == 1
    gh2, _ = _gh(lambda m, p, d: '{"x": 1}')
    with pytest.raises(GitHubFehler) as e:
        gh2.kommentare_seit("o/r", "2026-11-07T10:00:00Z")
    assert e.value.art == "antwort_unlesbar"


def test_review_kommentare_seit_blaettert() -> None:
    def antwort(m: str, pfad: str, d: Any) -> str:
        return json.dumps([{"id": i} for i in range(100 if pfad.endswith("page=1") else 3)])

    gh, fake = _gh(antwort)
    assert len(gh.review_kommentare_seit("o/r", "2026-11-07T10:00:00Z")) == 103
    assert fake.aufrufe[0][1] == "/repos/o/r/pulls/comments?since=2026-11-07T10:00:00Z&per_page=100&page=1"
    assert fake.aufrufe[1][1].endswith("&page=2")


def test_reviews_nur_mit_text() -> None:
    daten = [{"id": 1, "body": "Bitte aendern"}, {"id": 2, "body": ""}, {"id": 3, "body": None}, {"id": 4}]
    gh, fake = _gh(lambda m, p, d: json.dumps(daten))
    assert gh.reviews("o/r", 7) == [{"id": 1, "body": "Bitte aendern"}]
    assert fake.aufrufe[0][0] == "GET" and fake.aufrufe[0][1].startswith("/repos/o/r/pulls/7/reviews")


@pytest.mark.parametrize(("review", "pfad"), [
    (False, "/repos/o/r/issues/comments/42/reactions"), (True, "/repos/o/r/pulls/comments/42/reactions")])
def test_reagieren(review: bool, pfad: str) -> None:
    gh, fake = _gh()
    gh.reagieren("o/r", 42, review=review)
    assert fake.aufrufe == [("POST", pfad, {"content": "eyes"})]


def test_labels_setzen() -> None:
    gh, fake = _gh()
    gh.labels_setzen("o/r", 5, ["antwort:weg", "selbstbild"])
    assert fake.aufrufe == [("POST", "/repos/o/r/issues/5/labels", {"labels": ["antwort:weg", "selbstbild"]})]


def test_pr_anlegen() -> None:
    gh, fake = _gh(lambda m, p, d: '{"number": 77}', marke=DEBUGGER_MARKE)
    assert gh.pr_anlegen("o/r", "debugger/315-x", "master", "Titel", "Text") == 77
    (m, p, d), = fake.aufrufe
    assert (m, p) == ("POST", "/repos/o/r/pulls")
    assert d["title"] == "Titel" and d["head"] == "debugger/315-x" and d["base"] == "master"


def test_pr_anlegen_traegt_marke() -> None:
    gh, fake = _gh(lambda m, p, d: '{"number": 1}', marke=DEBUGGER_MARKE)
    gh.pr_anlegen("o/r", "debugger/1-x", "master", "T", "Text")
    assert fake.aufrufe[0][2]["body"] == "Text\n\n" + DEBUGGER_MARKE
    gh.pr_anlegen("o/r", "debugger/1-x", "master", "T", "Text\n\n" + DEBUGGER_MARKE)
    assert fake.aufrufe[1][2]["body"] == "Text\n\n" + DEBUGGER_MARKE  # nicht doppelt
    gh2, fake2 = _gh(lambda m, p, d: '{"number": 1}', marke=None)
    gh2.pr_anlegen("o/r", "debugger/1-x", "master", "T", "Text")
    assert fake2.aufrufe[0][2]["body"] == "Text"


@pytest.mark.parametrize("kopf", ["master", "main", "feature/x", "xdebugger/1", "", "debugger/../x"])
def test_pr_anlegen_nur_debugger_branch(kopf: str) -> None:
    gh, fake = _gh(lambda m, p, d: '{"number": 1}')
    with pytest.raises(ValueError):
        gh.pr_anlegen("o/r", kopf, "master", "T", "Text")
    assert fake.aufrufe == []


def test_pr_anlegen_antwort_unlesbar() -> None:
    gh, _ = _gh(lambda m, p, d: '{"foo": 1}')
    with pytest.raises(GitHubFehler) as e:
        gh.pr_anlegen("o/r", "debugger/1-x", "master", "T", "Text")
    assert e.value.art == "antwort_unlesbar"


def test_pr_lesen() -> None:
    gh, fake = _gh(lambda m, p, d: '{"number": 7, "state": "open", "merged": false}')
    assert gh.pr_lesen("o/r", 7)["state"] == "open"
    assert fake.aufrufe == [("GET", "/repos/o/r/pulls/7", None)]


def test_prs_zum_kopf() -> None:
    gh, fake = _gh(lambda m, p, d: '[{"number": 7}]')
    assert gh.prs_zum_kopf("Flissel/vibemind-os", "debugger/315-x") == [{"number": 7}]
    assert fake.aufrufe == [("GET", "/repos/Flissel/vibemind-os/pulls?head=Flissel:debugger/315-x&state=all", None)]


def test_branch_loeschen() -> None:
    gh, fake = _gh()
    gh.branch_loeschen("o/r", "debugger/315-x")
    assert fake.aufrufe == [("DELETE", "/repos/o/r/git/refs/heads/debugger/315-x", None)]


@pytest.mark.parametrize("name", ["master", "main", "feature/x", "debugger", "debugger/../master", ""])
def test_branch_loeschen_nur_debugger(name: str) -> None:
    gh, fake = _gh()
    with pytest.raises(ValueError):
        gh.branch_loeschen("o/r", name)
    assert fake.aufrufe == []


def test_branch_loeschen_404_ok() -> None:
    gh, fake = _gh(lambda m, p, d: b"gh: Reference does not exist (HTTP 404)")
    gh.branch_loeschen("o/r", "debugger/315-x")  # kein Fehler
    assert len(fake.aufrufe) == 1


def test_branch_loeschen_anderer_fehler_bleibt() -> None:
    gh, _ = _gh(lambda m, p, d: b"gh: Forbidden (HTTP 403) geheim")
    with pytest.raises(GitHubFehler) as e:
        gh.branch_loeschen("o/r", "debugger/315-x")
    assert e.value.art == "http_403" and "geheim" not in str(e.value)


# ---------- Final-Fix Punkt 7 ----------

def test_label_entfernen() -> None:
    gh, fake = _gh(lambda m, p, d: "[]")
    gh.label_entfernen("o/r", 5, "antwort:weg")
    assert fake.aufrufe == [("DELETE", "/repos/o/r/issues/5/labels/antwort%3Aweg", None)]


def test_label_entfernen_404_ok_anderer_fehler_bleibt() -> None:
    gh, fake = _gh(lambda m, p, d: b"gh: Label does not exist (HTTP 404)")
    gh.label_entfernen("o/r", 5, "antwort:weg")  # kein Fehler
    assert len(fake.aufrufe) == 1
    gh, _ = _gh(lambda m, p, d: b"gh: Forbidden (HTTP 403) geheim")
    with pytest.raises(GitHubFehler) as e:
        gh.label_entfernen("o/r", 5, "antwort:weg")
    assert e.value.art == "http_403" and "geheim" not in str(e.value)
    with pytest.raises(GitHubFehler):
        gh.label_entfernen("o/r", 5, "")
