"""Final-Fix für runde.py: neue Runde nach erledigtem Fund (R37), Aufräumen der Helfer-Prozesse (R44),
Budget vorab (R41), Abo-Limit und Pause (R42), Aufräumen ohne PR (R43) und die kleineren Punkte."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from test_runde import (  # noqa: F401  (Fixture `welt`)
    BRANCH,
    ISSUE_REPO,
    N,
    OS_REPO,
    PRAEFIX,
    Welt,
    antwort,
    commit,
    fix,
    git,
    kom,
    welt,
)

import selbstbild.debugger.runde as runde_modul
from selbstbild.debugger.helfer import HelferErgebnis, HelferFehler
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
    ablauf: list[str] = []

    def helfer(wt: Path, prompt: str, sitzung: str | None, frist_s: float) -> HelferErgebnis:
        ablauf.append("helfer")
        raise HelferFehler("frist")

    w = replace(welt.w, helfer=helfer, claw_aufraeumen=lambda: ablauf.append("aufraeumen"))
    assert runde_ausfuehren(Auftrag(N, (kom(1),)), welt.konfig, welt.stand, welt.klon, welt.ordner, w) == \
        "fehler:helfer_frist"
    assert ablauf == ["helfer", "aufraeumen"]


# ---------- R41: Budget vor dem Helfer ----------

def _budget(welt: Welt) -> list[dict[str, object]]:
    return [json.loads(z) for d in sorted(welt.ordner.glob("budget-*.jsonl"))
            for z in d.read_text(encoding="utf-8").splitlines()]


def test_budget_vor_dem_helfer_gebucht_ergebnis_als_nachtrag(welt: Welt) -> None:
    gesehen: list[list[dict[str, object]]] = []
    schritt = antwort("Steht so in der Doku.")

    def helfer(wt: Path, prompt: str, sitzung: str | None, frist_s: float) -> HelferErgebnis:
        gesehen.append(_budget(welt))
        return schritt(wt)

    w = replace(welt.w, helfer=helfer)
    assert runde_ausfuehren(Auftrag(N, (kom(1),)), welt.konfig, welt.stand, welt.klon, welt.ordner, w) == "ok"
    assert [z["ergebnis"] for z in gesehen[0]] == ["laeuft"]
    assert [(z["ergebnis"], z.get("nachtrag")) for z in _budget(welt)] == [("laeuft", None), ("ok", True)]


def test_budget_nicht_buchbar_keine_runde(welt: Welt, monkeypatch: pytest.MonkeyPatch,
                                          caplog: pytest.LogCaptureFixture) -> None:
    def kaputt(*args: object, **kw: object) -> None:
        raise PermissionError("GEHEIM-PFAD")

    monkeypatch.setattr(runde_modul, "budget_buchen", kaputt)
    welt.helfer.schritte = [antwort("x")]
    assert welt.runde(kom(1)) == "fehler:budget"
    assert welt.helfer.aufrufe == [] and welt.stand.verarbeitet == []
    assert welt.stand.funde[str(N)].versuche == {} and welt.lab.namen("kommentieren") == []
    assert "PermissionError" in caplog.text and "GEHEIM" not in caplog.text


def test_nachtrag_nicht_buchbar_runde_laeuft_weiter(welt: Welt, monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    echt = runde_modul.budget_buchen

    def nur_laeuft(ordner: Path, ergebnis: str, dauer_s: float, **kw: Any) -> None:
        if kw.get("nachtrag"):
            raise OSError("voll")
        echt(ordner, ergebnis, dauer_s, **kw)

    monkeypatch.setattr(runde_modul, "budget_buchen", nur_laeuft)
    welt.helfer.schritte = [antwort("x")]
    assert welt.runde(kom(1)) == "ok"
    assert [z["ergebnis"] for z in _budget(welt)] == ["laeuft"] and "OSError" in caplog.text


# ---------- R42: Abo-Limit und Pause ----------

def test_antwort_fehler_wartet_limit_ungezaehlt(welt: Welt) -> None:
    welt.helfer.schritte = [HelferFehler("antwort_fehler")]
    for _ in range(2):
        assert welt.runde(kom(1)) == "wartet:limit"
    fund = welt.stand.funde[str(N)]
    assert fund.versuche == {} and fund.naechster_versuch is None and welt.stand.verarbeitet == []
    assert [k[3] for k in welt.lab.namen("kommentieren")] == [PRAEFIX + "Auftrag wartet (limit)."]
    assert "wartet:limit:2026-11-07" in welt.stand.gemeldet


def test_gezaehlter_fehler_setzt_pause(welt: Welt) -> None:
    welt.helfer.schritte = [HelferFehler("antwort_kaputt")]
    assert welt.runde(kom(1)) == "fehler:helfer_antwort_kaputt"
    assert welt.stand.funde[str(N)].naechster_versuch == (welt.jetzt + timedelta(minutes=10)).isoformat()
    welt.jetzt += timedelta(minutes=11)
    assert welt.runde(kom(1)) == "fehler:helfer_antwort_kaputt"
    assert welt.stand.funde[str(N)].naechster_versuch == (welt.jetzt + timedelta(minutes=20)).isoformat()
    welt.helfer.schritte = [antwort("jetzt geht es")]
    assert welt.runde(kom(1)) == "ok"
    assert welt.stand.funde[str(N)].naechster_versuch is None and welt.stand.funde[str(N)].versuche == {}


# ---------- R43: Aufräumen ohne PR ----------

def test_antwort_ohne_pr_raeumt_worktree_auf(welt: Welt) -> None:
    welt.helfer.schritte = [antwort("Steht so in der Doku.")]
    assert welt.runde(kom(1)) == "ok"
    assert not welt.wt.exists()
    assert BRANCH not in git(welt.klon, "branch", "--list")
    assert welt.stand.funde[str(N)].sitzung == "s-1"  # der Pfad ist stabil (R6), die Sitzung bleibt


@pytest.mark.parametrize("art", ["absichtlich", "rueckfrage"])
def test_absichtlich_und_rueckfrage_raeumen_auf(welt: Welt, art: str) -> None:
    if art == "rueckfrage":
        welt.issue(titel="[selbstbild] Rückfrage: Goldsatz", text="Frage",
                   labels=("selbstbild", "selbstbild:rueckfrage"))
    welt.helfer.schritte = [antwort("gewollt", art, "weg" if art == "rueckfrage" else None)]
    assert welt.runde(kom(1)) == "ok"
    branch = welt.stand.funde[str(N)].branch
    assert welt.helfer.aufrufe[0]["wt"] == _wt(welt, branch)
    assert not _wt(welt, branch).exists() and branch not in git(welt.klon, "branch", "--list")


def test_mit_commits_voraus_bleibt_alles(welt: Welt) -> None:
    def schritt(wt: Path) -> HelferErgebnis:
        commit(wt, "halb.txt", "halb\n", "wip")
        return HelferErgebnis("s-1", "Erst einmal nur eine Antwort.", "antwort", None)

    welt.helfer.schritte = [schritt]
    assert welt.runde(kom(1)) == "ok"
    assert welt.wt.exists() and BRANCH in git(welt.klon, "branch", "--list")


def test_antwort_mit_pr_raeumt_nicht_auf(welt: Welt) -> None:
    welt.stand.funde[str(N)] = FundStand(repo=ISSUE_REPO, branch=BRANCH, sitzung="s-1", pr=480, zustand="pr")
    welt.helfer.schritte = [antwort("Antwort im PR.")]
    assert welt.runde(kom(1, nummer=480)) == "ok"
    assert welt.wt.exists()


# ---------- Punkte 2, 6, 7, 10 ----------

def test_reagieren_nur_einmal_je_kommentar(welt: Welt) -> None:
    welt.frei = 1.0
    welt.helfer.schritte = [antwort("x")]
    for _ in range(3):
        assert welt.runde(kom(1)) == "wartet:speicher"
    assert welt.lab.namen("reagieren") == [("reagieren", ISSUE_REPO, 1, False)]
    welt.frei = 8.0
    assert welt.runde(kom(1), kom(2)) == "ok"
    assert [a[2] for a in welt.lab.namen("reagieren")] == [1, 2]
    assert not [g for g in welt.stand.gemeldet if g.startswith("reagiert:")]  # verarbeitet: Schlüssel weg


def test_pr_text_beginnt_mit_praefix(welt: Welt) -> None:
    welt.helfer.schritte = [fix("b.txt", "fix: b")]
    assert welt.runde(kom(1)) == "ok"
    (pr,) = welt.lab.namen("pr_anlegen")
    assert pr[5] == f"{PRAEFIX}Fixes #{N}\n\n- fix: b"


def test_rueckfrage_entfernt_alte_antwort_labels(welt: Welt) -> None:
    welt.issue(titel="[selbstbild] Rückfrage: Goldsatz", text="Frage",
               labels=("selbstbild", "selbstbild:rueckfrage", "antwort:bleibt", "antwort:weg"))
    welt.helfer.schritte = [antwort("weg damit", "rueckfrage", "weg")]
    assert welt.runde(kom(1, "weg")) == "ok"
    assert [a for a in welt.lab.aufrufe if a[0] in ("label_entfernen", "labels_setzen")] == [
        ("label_entfernen", ISSUE_REPO, N, "antwort:bleibt"), ("labels_setzen", ISSUE_REPO, N, ["antwort:weg"])]


def test_fester_hinweis_immer_im_prompt(welt: Welt) -> None:
    satz = ("Arbeite nur in diesem Worktree und seinem Branch. Keine Einträge in WORKBOARD.md, keine Commits auf "
            "master, kein Push.")
    welt.helfer.schritte = [antwort("x")]
    assert welt.runde(kom(1)) == "ok"
    prompt = welt.helfer.aufrufe[0]["prompt"]
    assert prompt.count(satz) == 1 and prompt.index(satz) < prompt.index("<<<AUFTRAG>>>")
