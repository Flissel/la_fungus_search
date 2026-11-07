"""Final-Fix für wachter.py: Aufräumen vor dem Nachlauf (R44), Head-of-Line und GitHub-Fehler (Punkt 1),
Pause je Fund (R42) und Start fail-closed (Punkt 3)."""
from __future__ import annotations

import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from test_wachter import ISSUE_REPO, START, Welt, token_umgebung, welt  # noqa: F401  (Fixtures)

import selbstbild.debugger.wachter as wachter
from selbstbild.debugger.auftraege import Auftrag
from selbstbild.debugger.klon import branch_name
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.stand import DebuggerStand, FundStand, stand_speichern
from selbstbild.debugger.wachter import durchgang, werkzeuge_standard
from selbstbild.funde.github import GitHubFehler


def test_claw_aufraeumen_vor_dem_nachlauf(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    ablauf: list[str] = []

    def nachlauf(*args: Any) -> list[str]:
        ablauf.append("nachlauf")
        return []

    monkeypatch.setattr(wachter, "nachlauf", nachlauf)
    welt.kommentar(1)
    w = replace(welt.w, claw_aufraeumen=lambda: ablauf.append("aufraeumen"))
    assert durchgang(welt.tmp, welt.klon, welt.ordner, w, konfig=welt.konfig) == 0
    assert len(welt.helfer.aufrufe) == 1
    assert ablauf == ["aufraeumen", "aufraeumen", "nachlauf"]  # nach dem Helfer (Runde) und vor dem Nachlauf


# ---------- Punkt 1: Head-of-Line ----------

def test_runde_wirft_nachlauf_laeuft_trotzdem(welt: Welt, monkeypatch: pytest.MonkeyPatch,
                                              caplog: pytest.LogCaptureFixture) -> None:
    ablauf: list[str] = []

    def runde(*args: Any) -> str:
        ablauf.append("runde")
        raise RuntimeError("GEHEIM-INHALT")

    def nachlauf(*args: Any) -> list[str]:
        ablauf.append("nachlauf")
        return []

    monkeypatch.setattr(wachter, "runde_ausfuehren", runde)
    monkeypatch.setattr(wachter, "nachlauf", nachlauf)
    welt.kommentar(1)
    w = replace(welt.w, claw_aufraeumen=lambda: ablauf.append("aufraeumen"))
    assert durchgang(welt.tmp, welt.klon, welt.ordner, w, konfig=welt.konfig) == 1
    assert ablauf == ["runde", "aufraeumen", "nachlauf"]
    assert "RuntimeError" in caplog.text and "GEHEIM" not in caplog.text
    assert welt.stand().seit == START  # die Lesemarke rückt nach einer abgebrochenen Runde nicht vor


def test_nachlauf_wirft_wird_geloggt(welt: Welt, monkeypatch: pytest.MonkeyPatch,
                                     caplog: pytest.LogCaptureFixture) -> None:
    def nachlauf(*args: Any) -> list[str]:
        raise ValueError("GEHEIM-INHALT")

    monkeypatch.setattr(wachter, "nachlauf", nachlauf)
    welt.kommentar(1)
    assert welt.durchgang() == 1
    assert len(welt.helfer.aufrufe) == 1 and welt.stand().verarbeitet == [1]  # die Runde ist gespeichert
    assert "ValueError" in caplog.text and "GEHEIM" not in caplog.text


def test_github_fehler_dreimal_dann_fuer_den_tag_uebersprungen(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    echt = wachter.runde_ausfuehren
    versuche: list[int] = []
    kaputt = {"an": True}

    def runde(auftrag: Auftrag, *args: Any) -> str:
        versuche.append(auftrag.issue)
        if auftrag.issue == 318 and kaputt["an"]:
            raise GitHubFehler("http_502")
        return echt(auftrag, *args)

    monkeypatch.setattr(wachter, "runde_ausfuehren", runde)
    welt.kommentar(1, 318, "2026-11-07T11:00:00Z")
    welt.kommentar(2, 319, "2026-11-07T11:10:00Z")
    for _ in range(3):
        assert welt.durchgang() == 0
        welt.jetzt += timedelta(minutes=5)
    assert versuche == [318, 318, 318] and welt.helfer.aufrufe == []
    assert welt.durchgang() == 0  # 318 ist für heute gesperrt, 319 kommt dran
    assert versuche[-1] == 319 and welt.helfer.aufrufe[0].name.startswith("Vibemind_V1-319-")
    assert welt.stand().seit == "2026-11-07T11:00:00Z"  # 318 hält die Lesemarke
    welt.jetzt += timedelta(days=1)
    kaputt["an"] = False
    assert welt.durchgang() == 0  # neuer Tag: 318 wieder dran, Erfolg löscht den Zähler
    assert versuche[-1] == 318 and welt.stand().verarbeitet == [2, 1]
    assert not [g for g in welt.stand().gemeldet if g.startswith("gh_fehler:")]


def test_github_fehler_zaehler_nur_heute(welt: Welt) -> None:
    gemeldet = ["gh_fehler:318:2026-11-07", "gh_fehler:318:2026-11-07", "gh_fehler:318:2026-11-06"]
    stand_speichern(welt.stand_pfad, DebuggerStand(start=START, seit=START, gemeldet=gemeldet))
    welt.kommentar(1, 318)
    assert welt.durchgang() == 0
    assert len(welt.helfer.aufrufe) == 1  # zwei heute, der gestrige zählt nicht
    assert not [g for g in welt.stand().gemeldet if g.startswith("gh_fehler:")]


# ---------- R42: Pause blockiert die anderen nicht ----------

def test_pausierter_fund_wird_uebersprungen_und_haelt_seit(welt: Welt) -> None:
    fund = FundStand(repo=ISSUE_REPO, branch=branch_name(318, "[selbstbild] Fund 318 (vibemind)"),
                     naechster_versuch=(welt.jetzt + timedelta(minutes=5)).isoformat(), versuche={"1": 1})
    stand_speichern(welt.stand_pfad, DebuggerStand(start=START, seit=START, funde={"318": fund}))
    welt.kommentar(1, 318, "2026-11-07T11:00:00Z")
    welt.kommentar(2, 319, "2026-11-07T11:10:00Z")
    assert welt.durchgang() == 0
    assert len(welt.helfer.aufrufe) == 1 and welt.helfer.aufrufe[0].name.startswith("Vibemind_V1-319-")
    assert welt.stand().seit == "2026-11-07T11:00:00Z" and welt.stand().verarbeitet == [2]
    welt.jetzt += timedelta(minutes=6)
    assert welt.durchgang() == 0
    assert welt.helfer.aufrufe[1].name.startswith("Vibemind_V1-318-") and welt.stand().verarbeitet == [2, 1]
    assert welt.stand().funde["318"].naechster_versuch is None


# ---------- Punkt 3: Start fail-closed ----------

def test_root_ohne_praefix_startet_nicht(tmp_path: Path, token_umgebung: Path,  # noqa: F811
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    konfig = DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: "."})
    monkeypatch.setattr(wachter, "_ist_root", lambda: True)
    monkeypatch.delenv("DEBUGGER_PRAEFIX", raising=False)
    with pytest.raises(wachter.RootOhnePraefix):
        werkzeuge_standard(konfig, tmp_path / "stand", tmp_path / "spiegel")
    assert issubclass(wachter.RootOhnePraefix, RuntimeError)
    monkeypatch.setenv("DEBUGGER_PRAEFIX", "setpriv --reuid=1000 --")
    werkzeuge_standard(konfig, tmp_path / "stand", tmp_path / "spiegel")
    monkeypatch.setattr(wachter, "_ist_root", lambda: False)
    monkeypatch.delenv("DEBUGGER_PRAEFIX")
    werkzeuge_standard(konfig, tmp_path / "stand", tmp_path / "spiegel")


def test_ist_root_nur_posix_mit_euid_null() -> None:
    ergebnisse = [wachter._ist_root(name, lambda e=euid: e) for name, euid in (("posix", 0), ("posix", 1000), ("nt", 0))]
    assert ergebnisse == [True, False, False]
    assert wachter._ist_root("posix", None) is (getattr(os, "geteuid", lambda: 1)() == 0)


@pytest.mark.parametrize("art", ["fehlt", "leer", "gibtsnicht", "ordner", "nicht_ausfuehrbar"])
def test_askpass_fehlt_oder_nicht_ausfuehrbar(tmp_path: Path, token_umgebung: Path,  # noqa: F811
                                               monkeypatch: pytest.MonkeyPatch, art: str) -> None:
    if art == "nicht_ausfuehrbar" and os.name == "nt":
        pytest.skip("unter Windows gibt es kein Ausführungsrecht")
    datei = Path(os.environ["DEBUGGER_ASKPASS"])  # aus `token_umgebung`, ausführbar
    if art == "fehlt":
        monkeypatch.delenv("DEBUGGER_ASKPASS")
    elif art == "leer":
        monkeypatch.setenv("DEBUGGER_ASKPASS", " ")
    elif art == "gibtsnicht":
        monkeypatch.setenv("DEBUGGER_ASKPASS", str(tmp_path / "gibtsnicht"))
    elif art == "ordner":
        monkeypatch.setenv("DEBUGGER_ASKPASS", str(tmp_path))
    else:
        datei.chmod(0o644)
    with pytest.raises(wachter.AskpassFehlt) as ei:
        werkzeuge_standard(DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: "."}), tmp_path, tmp_path)
    assert str(ei.value) == "kein_askpass" and issubclass(wachter.AskpassFehlt, RuntimeError)


def test_main_ohne_askpass_exit_2(tmp_path: Path, token_umgebung: Path,  # noqa: F811
                                  monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("DEBUGGER_ASKPASS")
    (tmp_path / "selbstbild.yaml").write_text(
        f"debugger:\n  issue_repo: {ISSUE_REPO}\n  repos:\n    {ISSUE_REPO}: .\n", encoding="utf-8")
    ort = str(tmp_path)
    assert wachter.main(["durchgang", ort, "--klon", ort, "--ordner", ort, "--spiegel", ort]) == 2
    assert "AskpassFehlt" in caplog.text
