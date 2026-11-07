"""Tests für die Auftragsbildung aus Kommentaren (ohne Netz)."""
from __future__ import annotations

from selbstbild.debugger.auftraege import (
    Auftrag,
    Kommentar,
    auftraege_bilden,
    kommentar_aus_api,
)
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.stand import DebuggerStand
from selbstbild.funde.github import AUTO_MARKE, DEBUGGER_MARKE

ISSUE_REPO = "Flissel/Vibemind_V1"
OS_REPO = "Flissel/vibemind-os"
KONFIG = DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: ".", OS_REPO: "vibemind-os"})
START = "2026-11-07T10:00:00Z"
SEIT = "2026-11-07T12:00:00Z"


def _k(id: int = 1, *, repo: str = ISSUE_REPO, nummer: int = 5, autor: str = "Flissel",
       text: str = "bitte fixen", erstellt: str = "2026-11-07T13:00:00Z", review: bool = False,
       pfad: str | None = None, zeile: int | None = None) -> Kommentar:
    return Kommentar(id=id, repo=repo, nummer=nummer, autor=autor, text=text,
                     erstellt=erstellt, review=review, pfad=pfad, zeile=zeile)


def _stand(verarbeitet: list[int] | None = None) -> DebuggerStand:
    return DebuggerStand(start=START, seit=SEIT, verarbeitet=verarbeitet or [])


def _bilden(kommentare: list[Kommentar], stand: DebuggerStand | None = None, *,
            issues: set[int] | None = None, pr: dict[tuple[str, int], int] | None = None,
            laufend: set[int] | None = None) -> list[Auftrag]:
    return auftraege_bilden(kommentare, stand or _stand(), KONFIG,
                            selbstbild_issues={5} if issues is None else issues,
                            pr_zu_issue=pr or {}, laufend=laufend or set())


def test_nur_ausloeser() -> None:
    assert _bilden([_k(1, autor="Vibemind-LAB"), _k(2, autor="Fremder")]) == []
    assert len(_bilden([_k(3)])) == 1


def test_auto_marke_ignoriert() -> None:
    assert _bilden([_k(text=f"Hinweis\n{AUTO_MARKE}")]) == []


def test_debugger_marke_ignoriert() -> None:
    assert _bilden([_k(text=f"🤖 Debugger-Claw: ok\n\n{DEBUGGER_MARKE}")]) == []


def test_schon_verarbeitet() -> None:
    assert _bilden([_k(7)], _stand([7])) == []


def test_vor_seit_ignoriert() -> None:
    # Regel 4 gilt gegen `start` (Bei-oder-vor), nicht gegen die Lesemarke `seit`.
    assert _bilden([_k(1, erstellt="2026-11-07T09:59:59Z")]) == []
    assert _bilden([_k(2, erstellt=START)]) == []
    zwischen = _bilden([_k(3, erstellt="2026-11-07T11:00:00Z")])
    assert len(zwischen) == 1 and zwischen[0].kommentare[0].id == 3


def test_vor_start_ohne_start_nichts_ausgeschlossen() -> None:
    stand = DebuggerStand(start=None, seit=None)
    assert len(_bilden([_k(erstellt="2020-01-01T00:00:00Z")], stand)) == 1


def test_start_vergleich_als_zeit_nicht_als_text() -> None:
    stand = DebuggerStand(start="2026-11-07T10:00:00+00:00")
    assert _bilden([_k(erstellt="2026-11-07T10:00:00Z")], stand) == []


def test_fremdes_issue_ignoriert() -> None:
    assert _bilden([_k(nummer=99)]) == []


def test_pr_kommentar_landet_beim_issue() -> None:
    zeile = _k(1, repo=OS_REPO, nummer=12, review=True, pfad="a/b.py", zeile=3)
    normal = _k(2, repo=OS_REPO, nummer=12, erstellt="2026-11-07T14:00:00Z")
    res = _bilden([normal, zeile], pr={(OS_REPO, 12): 5})
    assert [a.issue for a in res] == [5]
    assert [k.id for k in res[0].kommentare] == [1, 2]
    assert res[0].kommentare[0].pfad == "a/b.py" and res[0].kommentare[0].zeile == 3


def test_pr_im_issue_repo_ueber_pr_zu_issue() -> None:
    res = _bilden([_k(1, nummer=40)], issues={5}, pr={(ISSUE_REPO, 40): 5})
    assert [a.issue for a in res] == [5]


def test_zwei_kommentare_ein_auftrag() -> None:
    res = _bilden([_k(2, erstellt="2026-11-07T14:00:00Z"), _k(1), _k(1)])
    assert len(res) == 1
    assert [k.id for k in res[0].kommentare] == [1, 2]


def test_zwei_issues_zwei_auftraege() -> None:
    res = _bilden([_k(1, nummer=6, erstellt="2026-11-07T14:00:00Z"), _k(2, nummer=5)], issues={5, 6})
    assert [a.issue for a in res] == [5, 6]


def test_kommentar_waehrend_runde_kommt_in_naechste() -> None:
    assert _bilden([_k()], laufend={5}) == []
    assert len(_bilden([_k()], laufend=set())) == 1


def test_fremdes_repo_ignoriert() -> None:
    assert _bilden([_k(repo="Flissel/anderes")], pr={("Flissel/anderes", 5): 5}) == []


def test_kommentar_aus_api_issue_kommentar() -> None:
    roh = {"id": 11, "issue_url": f"https://api.github.com/repos/{ISSUE_REPO}/issues/5",
           "user": {"login": "Flissel"}, "body": None, "created_at": "2026-11-07T13:00:00Z"}
    k = kommentar_aus_api(ISSUE_REPO, roh, review=False)
    assert (k.id, k.nummer, k.autor, k.text, k.erstellt, k.review, k.pfad, k.zeile) == (
        11, 5, "Flissel", "", "2026-11-07T13:00:00Z", False, None, None)


def test_kommentar_aus_api_review_zeile() -> None:
    roh = {"id": 12, "pull_request_url": f"https://api.github.com/repos/{OS_REPO}/pulls/9",
           "user": {"login": "Flissel"}, "body": "x", "created_at": "2026-11-07T13:00:00Z",
           "path": "src/a.py", "line": None, "original_line": 42}
    k = kommentar_aus_api(OS_REPO, roh, review=True)
    assert (k.nummer, k.review, k.pfad, k.zeile) == (9, True, "src/a.py", 42)
