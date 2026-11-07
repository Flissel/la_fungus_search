"""Final-Fix für runde.py: neue Runde nach erledigtem Fund (R37), Aufräumen der Helfer-Prozesse (R44) u. a."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from test_runde import (  # noqa: F401  (Fixture `welt`)
    BRANCH,
    ISSUE_REPO,
    N,
    OS_REPO,
    Welt,
    fix,
    git,
    kom,
    welt,
)

from selbstbild.debugger.helfer import HelferErgebnis
from selbstbild.debugger.runde import runde_ausfuehren
from selbstbild.debugger.auftraege import Auftrag
from selbstbild.debugger.stand import FundStand

BRANCH_2 = f"debugger/{N}-doku-widerspruch-schedule-s-2"  # branch_name(N, TITEL, 2)


def _wt(welt: Welt, branch: str, repo_name: str = "Vibemind_V1") -> Path:
    return welt.ordner / "wt" / f"{repo_name}-{branch.removeprefix('debugger/')}"


# ---------- R37 ----------

def test_erledigt_neue_runde_neuer_branch_neuer_pr(welt: Welt) -> None:
    welt.stand.funde[str(N)] = FundStand(repo=ISSUE_REPO, branch=BRANCH, sitzung="alt", pr=480, zustand="erledigt",
                                         versuche={"9": 2})
    welt.helfer.schritte = [fix("b.txt", "fix: zweiter Anlauf")]
    assert welt.runde(kom(1, "bitte nochmal")) == "ok"
    fund = welt.stand.funde[str(N)]
    assert (fund.repo, fund.branch, fund.runde, fund.pr, fund.zustand) == (ISSUE_REPO, BRANCH_2, 2, 501, "pr")
    assert fund.versuche == {"9": 2}
    assert welt.helfer.aufrufe[0]["sitzung"] is None and welt.helfer.aufrufe[0]["wt"] == _wt(welt, BRANCH_2)
    assert welt.pushes == [(_wt(welt, BRANCH_2), BRANCH_2)]
    (pr,) = welt.lab.namen("pr_anlegen")
    assert pr[2] == BRANCH_2
    assert BRANCH not in git(welt.origin, "branch", "--list")
    assert welt.lesen.namen("pr_lesen") == []  # erledigt: der alte PR wird nicht mehr gelesen


def test_pr_geschlossen_aber_stand_pr_neuer_pr(welt: Welt) -> None:
    welt.stand.funde[str(N)] = FundStand(repo=ISSUE_REPO, branch=BRANCH, sitzung="alt", pr=480, zustand="pr")
    welt.daten["pr_lesen"] = {(ISSUE_REPO, 480): {"state": "closed", "merged": False}}
    welt.helfer.schritte = [fix("b.txt", "fix: neu")]
    assert welt.runde(kom(1, "bitte fixen", nummer=480)) == "ok"
    assert welt.lesen.namen("pr_lesen")[0] == ("pr_lesen", ISSUE_REPO, 480)
    fund = welt.stand.funde[str(N)]
    assert (fund.branch, fund.runde, fund.pr, fund.zustand, fund.sitzung) == (BRANCH_2, 2, 501, "pr", "s-1")
    assert welt.helfer.aufrufe[0]["sitzung"] is None
    assert [a[2] for a in welt.lab.namen("pr_anlegen")] == [BRANCH_2]
    assert "Diff des PR" not in welt.helfer.aufrufe[0]["prompt"]  # als gäbe es keinen PR


def test_offener_pr_bleibt_und_wird_vor_dem_push_geprueft(welt: Welt) -> None:
    welt.stand.funde[str(N)] = FundStand(repo=ISSUE_REPO, branch=BRANCH, sitzung="s-1", pr=480, zustand="pr")
    welt.helfer.schritte = [fix("b.txt", "fix: weiter")]
    assert welt.runde(kom(1, nummer=480)) == "ok"
    assert len(welt.lesen.namen("pr_lesen")) == 2  # Rundenbeginn und direkt vor dem Push
    assert welt.lab.namen("pr_anlegen") == [] and welt.pushes == [(welt.wt, BRANCH)]
    assert welt.stand.funde[str(N)].branch == BRANCH and welt.stand.funde[str(N)].pr == 480


def test_pr_schliesst_waehrend_der_runde_kein_push(welt: Welt) -> None:
    welt.stand.funde[str(N)] = FundStand(repo=ISSUE_REPO, branch=BRANCH, sitzung="s-1", pr=480, zustand="pr")

    def schritt(wt: Path) -> HelferErgebnis:
        erg = fix("b.txt", "fix: spät")(wt)
        welt.daten["pr_lesen"] = {(ISSUE_REPO, 480): {"state": "closed", "merged": True}}
        return erg

    welt.helfer.schritte = [schritt]
    assert welt.runde(kom(1, nummer=480)) == "wartet:pr_zu"
    assert welt.pushes == [] and welt.lab.namen("pr_anlegen") == []
    assert welt.stand.verarbeitet == [] and welt.stand.funde[str(N)].versuche == {}


def test_oeffentlich_ab_runde_zwei_fix_suffix(welt: Welt) -> None:
    welt.issue(titel="[selbstbild] Port falsch (vibemind/vibemind-os)",
               text="<!-- selbstbild:repo=vibemind/vibemind-os -->")
    welt.stand.funde[str(N)] = FundStand(repo=ISSUE_REPO, branch=f"debugger/{N}-pin", pr=412, zustand="erledigt",
                                         runde=2)
    welt.helfer.schritte = [fix("c.txt", "fix: Port")]
    assert welt.runde(kom(1)) == "ok"
    fund = welt.stand.funde[str(N)]
    assert (fund.repo, fund.branch, fund.runde) == (OS_REPO, f"debugger/{N}-fix-3", 3)
    assert welt.pushes == [(_wt(welt, fund.branch, "vibemind-os"), fund.branch)]


# ---------- R44 ----------

def test_claw_aufraeumen_nach_helfer_vor_push(welt: Welt) -> None:
    ablauf: list[str] = []
    schritt = fix("b.txt", "fix: b")

    def helfer(wt: Path, prompt: str, sitzung: str | None, frist_s: float) -> HelferErgebnis:
        ablauf.append("helfer")
        return schritt(wt)

    def push(wt: Path, branch: str, **kw: object) -> None:
        ablauf.append("push")
        welt.push(wt, branch)

    w = replace(welt.w, helfer=helfer, push=push, claw_aufraeumen=lambda: ablauf.append("aufraeumen"))
    assert runde_ausfuehren(Auftrag(N, (kom(1),)), welt.konfig, welt.stand, welt.klon, welt.ordner, w) == "ok"
    assert ablauf == ["helfer", "aufraeumen", "push"]


def test_claw_aufraeumen_auch_nach_helferfehler(welt: Welt) -> None:
    from selbstbild.debugger.helfer import HelferFehler

    ablauf: list[str] = []

    def helfer(wt: Path, prompt: str, sitzung: str | None, frist_s: float) -> HelferErgebnis:
        ablauf.append("helfer")
        raise HelferFehler("frist")

    w = replace(welt.w, helfer=helfer, claw_aufraeumen=lambda: ablauf.append("aufraeumen"))
    assert runde_ausfuehren(Auftrag(N, (kom(1),)), welt.konfig, welt.stand, welt.klon, welt.ordner, w) == \
        "fehler:helfer_frist"
    assert ablauf == ["helfer", "aufraeumen"]
