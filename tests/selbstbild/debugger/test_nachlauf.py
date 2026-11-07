"""Tests für den Debugger-Nachlauf: Fake-GitHub, echte temporäre Repos mit Submodul."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from selbstbild.debugger import nachlauf as nachlauf_modul
from selbstbild.debugger.auftraege import Auftrag, Kommentar
from selbstbild.debugger.helfer import HelferErgebnis
from selbstbild.debugger.klon import worktree_bereit
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.nachlauf import nachlauf
from selbstbild.debugger.runde import Werkzeuge, runde_ausfuehren
from selbstbild.debugger.stand import DebuggerStand, FundStand
from selbstbild.funde.github import GitHub, GitHubFehler

ISSUE_REPO = "Flissel/Vibemind_V1"
OS_REPO = "Flissel/vibemind-os"
N = 318
OS_BRANCH = f"debugger/{N}-port-falsch"
AUSSEN_BRANCH = f"debugger/{N}-schedule"
PIN_BRANCH = f"debugger/{N}-pin"
PRAEFIX = "🤖 Debugger-Claw: "


# ---------- Git ----------

def git(ort: Path, *args: str) -> str:
    erg = subprocess.run(["git", "-C", str(ort), *args], check=True, capture_output=True, text=True)
    return erg.stdout.strip()


def _konfig(ort: Path) -> None:
    git(ort, "config", "user.name", "T")
    git(ort, "config", "user.email", "t@example.invalid")
    git(ort, "config", "core.autocrlf", "false")
    git(ort, "config", "commit.gpgsign", "false")


def commit(ort: Path, datei: str, betreff: str | None = None) -> str:
    (ort / datei).write_text(f"{datei}\n", encoding="utf-8", newline="\n")
    git(ort, "add", "--", datei)
    git(ort, "commit", "-q", "-m", betreff or datei)
    return git(ort, "rev-parse", "HEAD")


def _bare(tmp: Path, name: str) -> Path:
    origin = tmp / f"{name}.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(origin)], check=True, capture_output=True)
    return origin


def _arbeit(tmp: Path, name: str, origin: Path) -> Path:
    ort = tmp / name
    subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=false", str(origin), str(ort)],
                   check=True, capture_output=True)
    _konfig(ort)
    return ort


# ---------- Fakes ----------

class FakeGitHub:
    """Protokolliert alle Aufrufe; PR-Daten kommen aus dem gemeinsamen `daten`."""

    def __init__(self, daten: dict[str, Any]) -> None:
        self.daten = daten
        self.aufrufe: list[tuple[Any, ...]] = []
        self.fehler: dict[str, BaseException] = {}

    def _fehler(self, name: str) -> None:
        if name in self.fehler:
            raise self.fehler[name]

    def namen(self, name: str) -> list[tuple[Any, ...]]:
        return [a for a in self.aufrufe if a[0] == name]

    def schreibend(self) -> list[tuple[Any, ...]]:
        return [a for a in self.aufrufe if a[0] not in ("pr_lesen", "prs_zum_kopf", "api")]

    def pr_lesen(self, repo: str, nummer: int) -> dict[str, Any]:
        self.aufrufe.append(("pr_lesen", repo, nummer))
        fehler = self.daten.get("pr_fehler", {}).get((repo, nummer))
        if fehler is not None:
            raise fehler
        return dict(self.daten["prs"][(repo, nummer)])

    def prs_zum_kopf(self, repo: str, kopf: str) -> list[dict[str, Any]]:
        self.aufrufe.append(("prs_zum_kopf", repo, kopf))
        return list(self.daten.get("kopf", {}).get((repo, kopf), []))

    def pr_anlegen(self, repo: str, kopf: str, basis: str, titel: str, text: str) -> int:
        self.aufrufe.append(("pr_anlegen", repo, kopf, basis, titel, text))
        self._fehler("pr_anlegen")
        return 412

    def branch_loeschen(self, repo: str, branch: str) -> None:
        self.aufrufe.append(("branch_loeschen", repo, branch))
        if not branch.startswith("debugger/"):
            raise ValueError("Branch muss debugger/... heißen")
        self._fehler("branch_loeschen")

    def kommentieren(self, repo: str, nummer: int, text: str) -> None:
        self.aufrufe.append(("kommentieren", repo, nummer, text))

    # nur für eine Runde nach dem Nachlauf
    def reagieren(self, repo: str, kommentar_id: int, *, review: bool = False) -> None:
        self.aufrufe.append(("reagieren", repo, kommentar_id, review))

    def _api(self, methode: str, pfad: str, daten: dict[str, Any] | None = None) -> Any:
        self.aufrufe.append(("api", methode, pfad))
        if pfad == f"/repos/{ISSUE_REPO}/issues/{N}":
            return {"number": N, "title": "[selbstbild] Port falsch (vibemind/vibemind-os)", "body": "", "labels": []}
        return []


def _kein_helfer(*args: object) -> HelferErgebnis:
    raise AssertionError("der Nachlauf ruft keinen Helfer")


@dataclass
class Welt:
    tmp: Path
    os_origin: Path
    origin: Path
    os_arbeit: Path
    klon: Path
    ordner: Path
    konfig: DebuggerKonfig
    stand: DebuggerStand
    daten: dict[str, Any]
    lab: FakeGitHub
    lesen: FakeGitHub
    pushes: list[tuple[Path, str]] = field(default_factory=list)
    helfer: Callable[..., HelferErgebnis] = _kein_helfer

    def push(self, wt: Path, branch: str) -> None:
        self.pushes.append((wt, branch))
        git(wt, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")

    @property
    def w(self) -> Werkzeuge:
        return Werkzeuge(
            gh_lab=cast(GitHub, self.lab), gh_lesen=cast(GitHub, self.lesen), helfer=self.helfer,
            frei_gb=lambda: 8.0, uhr=lambda: datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc), push=self.push,
        )

    def lauf(self) -> list[str]:
        return nachlauf(self.konfig, self.stand, self.klon, self.ordner, self.w)

    @property
    def os_klon(self) -> Path:
        return self.klon / "vibemind-os"

    def wt(self, repo: str, branch: str) -> Path:
        return self.ordner / "wt" / f"{repo.split('/')[1]}-{branch.removeprefix('debugger/')}"

    def fund_mit_branch(self, repo: str, branch: str, n: int = N, pr: int = 501) -> FundStand:
        """Legt Worktree und entfernten Branch an wie eine frühere Fix-Runde."""
        repo_klon = self.klon if repo == ISSUE_REPO else self.os_klon
        wt = worktree_bereit(repo_klon, self.wt(repo, branch), branch, "master")
        commit(wt, f"fix-{n}.txt", "fix: etwas")
        git(wt, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")
        fund = FundStand(repo=repo, branch=branch, sitzung="s-1", pr=pr, zustand="pr", versuche={"7": 1})
        self.stand.funde[str(n)] = fund
        return fund

    def pr(self, repo: str, nummer: int, *, merged: bool = False, state: str = "open",
           sha: str | None = None) -> None:
        self.daten["prs"][(repo, nummer)] = {
            "number": nummer, "merged": merged, "state": state, "merge_commit_sha": sha,
            "head": {"ref": "debugger/999-fremd"},  # darf nie benutzt werden
        }

    def os_merge(self) -> str:
        """Ein neuer Commit auf master von vibemind-os (wie ein gemergter PR)."""
        git(self.os_arbeit, "pull", "-q", "--ff-only", "origin", "master")
        sha = commit(self.os_arbeit, "merge.txt", "Merge pull request #77")
        git(self.os_arbeit, "push", "-q", "origin", "master")
        return sha

    def aussen_pin(self, sha: str) -> None:
        """Setzt den Gitlink auf dem äußeren master (wie ein anderer, schon gemergter Pin)."""
        ort = self.tmp / "aussen-pin"
        if not ort.exists():
            _arbeit(self.tmp, ort.name, self.origin)
        git(ort, "pull", "-q", "--ff-only", "origin", "master")
        git(ort, "update-index", "--cacheinfo", f"160000,{sha},vibemind-os")
        git(ort, "commit", "-q", "-m", f"pin {sha[:7]}")
        git(ort, "push", "-q", "origin", "master")

    def kommentare(self) -> list[str]:
        return [a[3] for a in self.lab.namen("kommentieren")]


@pytest.fixture
def welt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Welt:
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    os_origin = _bare(tmp_path, "os")
    os_arbeit = _arbeit(tmp_path, "os-arbeit", os_origin)
    commit(os_arbeit, "a.txt")
    git(os_arbeit, "push", "-q", "origin", "master")

    origin = _bare(tmp_path, "aussen")
    start = _arbeit(tmp_path, "aussen-start", origin)
    commit(start, "a.txt")
    git(start, "submodule", "add", "-q", os_origin.as_uri(), "vibemind-os")
    git(start, "commit", "-q", "-m", "submodul")
    git(start, "push", "-q", "origin", "master")

    klon = _arbeit(tmp_path, "klon", origin)
    git(klon, "submodule", "update", "--init", "-q")
    _konfig(klon / "vibemind-os")

    konfig = DebuggerKonfig(issue_repo=ISSUE_REPO, repos={ISSUE_REPO: ".", OS_REPO: "vibemind-os"})
    daten: dict[str, Any] = {"prs": {}}
    return Welt(tmp=tmp_path, os_origin=os_origin, origin=origin, os_arbeit=os_arbeit, klon=klon,
                ordner=tmp_path / "stand", konfig=konfig,
                stand=DebuggerStand(start="2026-11-01T00:00:00Z", seit="2026-11-07T00:00:00Z"),
                daten=daten, lab=FakeGitHub(daten), lesen=FakeGitHub(daten))


def _zweig_da(ort: Path, branch: str) -> bool:
    erg = subprocess.run(["git", "-C", str(ort), "rev-parse", "--verify", "-q", f"refs/heads/{branch}"],
                         capture_output=True, text=True, check=False)
    return erg.returncode == 0


def _gitlink(ort: Path, ref: str) -> str:
    zeile = git(ort, "ls-tree", ref, "--", "vibemind-os")
    art, typ, rest = zeile.split(" ", 2)
    assert (art, typ) == ("160000", "commit")
    return rest.split("\t")[0]


# ---------- Tests aus dem Brief ----------

def test_gemergt_issue_repo_erledigt_und_raeumt_auf(welt: Welt) -> None:
    welt.fund_mit_branch(ISSUE_REPO, AUSSEN_BRANCH)
    welt.pr(ISSUE_REPO, 501, merged=True, state="closed", sha="0" * 40)
    zeilen = welt.lauf()
    fund = welt.stand.funde[str(N)]
    assert (fund.repo, fund.branch, fund.pr, fund.zustand, fund.sitzung) == (
        ISSUE_REPO, AUSSEN_BRANCH, 501, "erledigt", "s-1")
    assert not welt.wt(ISSUE_REPO, AUSSEN_BRANCH).exists()
    assert not _zweig_da(welt.klon, AUSSEN_BRANCH)
    assert welt.lab.namen("branch_loeschen") == [("branch_loeschen", ISSUE_REPO, AUSSEN_BRANCH)]
    assert welt.lab.namen("pr_anlegen") == [] and welt.pushes == []
    assert welt.lesen.schreibend() == []
    assert zeilen == [f"#{N}: erledigt ({ISSUE_REPO}#501 gemergt)"]


def test_gemergt_vibemind_os_hebt_pin(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    alt = _gitlink(welt.origin, "master")
    zeilen = welt.lauf()
    # vibemind-os-Branch aufgeräumt (lokal und entfernt)
    assert not welt.wt(OS_REPO, OS_BRANCH).exists()
    assert not _zweig_da(welt.os_klon, OS_BRANCH)
    assert welt.lab.namen("branch_loeschen") == [("branch_loeschen", OS_REPO, OS_BRANCH)]
    # Pin-Branch im äußeren Repo: neuer Gitlink = Merge-Commit, eine Commit-Zeile
    pin_wt = welt.wt(ISSUE_REPO, PIN_BRANCH)
    assert welt.pushes == [(pin_wt, PIN_BRANCH)]
    assert _gitlink(welt.origin, PIN_BRANCH) == sha != alt
    assert _gitlink(welt.origin, "master") == alt
    assert git(welt.origin, "log", "--format=%s", f"master..{PIN_BRANCH}").splitlines() == [
        f"chore(vibemind-os): Pin auf {sha[:7]} (Debugger-Claw #{N})"]
    assert git(welt.origin, "diff", "--name-only", "master", PIN_BRANCH).splitlines() == ["vibemind-os"]
    assert welt.lab.namen("pr_anlegen") == [(
        "pr_anlegen", ISSUE_REPO, PIN_BRANCH, "master",
        f"{PRAEFIX}vibemind-os auf {sha[:7]} (#{N})",
        f"Fixes #{N}\n\n- Pin vibemind-os auf {sha} (gemergt in {OS_REPO}#77)",
    )]
    fund = welt.stand.funde[str(N)]
    assert (fund.repo, fund.branch, fund.pr, fund.zustand, fund.sitzung, fund.versuche) == (
        ISSUE_REPO, PIN_BRANCH, 412, "pr", "s-1", {"7": 1})
    assert zeilen == [f"#{N}: Pin-PR #412 geöffnet"]
    assert welt.lab.namen("kommentieren") == []
    # zweiter Durchgang: Pin-PR offen, nichts passiert
    welt.pr(ISSUE_REPO, 412)
    assert welt.lauf() == []
    assert len(welt.lab.namen("pr_anlegen")) == 1 and len(welt.pushes) == 1


@pytest.mark.parametrize("art", ["seitenzweig", "unbekannt"])
def test_pin_nur_wenn_commit_auf_master(welt: Welt, art: str) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    if art == "seitenzweig":  # auf origin vorhanden, aber nicht auf master
        git(welt.os_arbeit, "checkout", "-q", "-b", "seite")
        sha = commit(welt.os_arbeit, "seite.txt")
        git(welt.os_arbeit, "push", "-q", "origin", "seite")
    else:
        sha = "1234567" * 5 + "89abc"
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    vorher = FundStand(**vars(welt.stand.funde[str(N)]))
    grund = "nicht_auf_basis" if art == "seitenzweig" else "kein_commit"
    for _ in range(2):  # R26: wird jedes Mal erneut versucht, aber nur ein Kommentar
        assert welt.lauf() == [f"#{N}: Pin wartet ({grund})"]
        assert welt.stand.funde[str(N)] == vorher
        assert not welt.wt(ISSUE_REPO, PIN_BRANCH).exists() and not _zweig_da(welt.klon, PIN_BRANCH)
    assert welt.kommentare() == [f"{PRAEFIX}Pin für vibemind-os wartet ({grund})."]
    assert welt.pushes == [] and welt.lab.namen("pr_anlegen") == []
    assert not _zweig_da(welt.origin, PIN_BRANCH)


def test_geschlossen_ohne_merge_raeumt_auf(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    welt.pr(OS_REPO, 77, merged=False, state="closed")
    zeilen = welt.lauf()
    fund = welt.stand.funde[str(N)]
    assert (fund.repo, fund.branch, fund.pr, fund.zustand, fund.sitzung, fund.versuche) == (
        OS_REPO, OS_BRANCH, None, "offen", "s-1", {"7": 1})
    assert not welt.wt(OS_REPO, OS_BRANCH).exists()
    assert not _zweig_da(welt.os_klon, OS_BRANCH)
    assert welt.lab.namen("branch_loeschen") == [("branch_loeschen", OS_REPO, OS_BRANCH)]
    assert welt.pushes == [] and welt.lab.namen("pr_anlegen") == []
    assert zeilen == [f"#{N}: {OS_REPO}#77 ohne Merge geschlossen, aufgeräumt"]


def test_fremder_branch_bleibt(welt: Welt) -> None:
    # ein fremder Helfer-Branch mit Worktree und ein Branch ohne debugger/-Präfix
    fremd = welt.fund_mit_branch(ISSUE_REPO, "debugger/999-fremd", n=999, pr=600)
    git(welt.klon, "branch", "feature/x", "origin/master")
    del welt.stand.funde["999"]
    welt.fund_mit_branch(ISSUE_REPO, AUSSEN_BRANCH)
    welt.pr(ISSUE_REPO, 501, merged=True, state="closed", sha="0" * 40)
    # ein Fund, dessen Branch kein debugger/-Branch ist: wird nie angefasst
    welt.stand.funde["400"] = FundStand(repo=ISSUE_REPO, branch="feature/x", pr=700, zustand="pr")
    welt.pr(ISSUE_REPO, 700, merged=False, state="closed")
    zeilen = welt.lauf()
    assert welt.lab.namen("branch_loeschen") == [("branch_loeschen", ISSUE_REPO, AUSSEN_BRANCH)]
    assert welt.wt(ISSUE_REPO, "debugger/999-fremd").exists() and _zweig_da(welt.klon, fremd.branch)
    assert _zweig_da(welt.klon, "feature/x") and _zweig_da(welt.klon, "master")
    assert welt.stand.funde["400"] == FundStand(repo=ISSUE_REPO, branch="feature/x", pr=700, zustand="pr")
    assert zeilen == [f"#{N}: erledigt ({ISSUE_REPO}#501 gemergt)", "#400: fehler ValueError"]


# ---------- Zusätzliche Fälle ----------

def test_offener_pr_nichts(welt: Welt) -> None:
    welt.fund_mit_branch(ISSUE_REPO, AUSSEN_BRANCH)
    welt.pr(ISSUE_REPO, 501)
    welt.stand.funde["5"] = FundStand(repo=ISSUE_REPO, branch="debugger/5-x", pr=None, zustand="offen")
    welt.stand.funde["6"] = FundStand(repo=ISSUE_REPO, branch="debugger/6-x", pr=9, zustand="erledigt")
    assert welt.lauf() == []
    assert welt.lab.aufrufe == []
    assert [a[1:] for a in welt.lesen.aufrufe] == [(ISSUE_REPO, 501)]
    assert welt.wt(ISSUE_REPO, AUSSEN_BRANCH).exists() and welt.stand.funde[str(N)].zustand == "pr"


def test_fehler_bei_einem_fund_stoppt_andere_nicht(welt: Welt) -> None:
    welt.stand.funde["5"] = FundStand(repo=ISSUE_REPO, branch="debugger/5-x", pr=9, zustand="pr")
    welt.daten["pr_fehler"] = {(ISSUE_REPO, 9): GitHubFehler("http_502")}
    welt.fund_mit_branch(ISSUE_REPO, AUSSEN_BRANCH)
    welt.pr(ISSUE_REPO, 501, merged=True, state="closed", sha="0" * 40)
    zeilen = welt.lauf()
    assert zeilen == ["#5: fehler GitHubFehler", f"#{N}: erledigt ({ISSUE_REPO}#501 gemergt)"]
    assert welt.stand.funde["5"] == FundStand(repo=ISSUE_REPO, branch="debugger/5-x", pr=9, zustand="pr")
    assert welt.stand.funde[str(N)].zustand == "erledigt"


def test_branch_auf_github_schon_weg_zaehlt_als_erledigt(welt: Welt) -> None:
    # der echte Adapter schluckt das 404; ein anderer Fehler lässt den Fund unverändert
    welt.fund_mit_branch(ISSUE_REPO, AUSSEN_BRANCH)
    welt.pr(ISSUE_REPO, 501, merged=False, state="closed")
    welt.lab.fehler["branch_loeschen"] = GitHubFehler("http_500")
    assert welt.lauf() == [f"#{N}: fehler GitHubFehler"]
    assert welt.stand.funde[str(N)].pr == 501
    del welt.lab.fehler["branch_loeschen"]
    assert welt.lauf() == [f"#{N}: {ISSUE_REPO}#501 ohne Merge geschlossen, aufgeräumt"]
    assert welt.stand.funde[str(N)].pr is None


def test_pin_nach_pr_fehler_ohne_doppelten_commit(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    welt.lab.fehler["pr_anlegen"] = GitHubFehler("http_422")
    assert welt.lauf() == [f"#{N}: fehler GitHubFehler"]
    assert welt.stand.funde[str(N)].repo == OS_REPO and len(welt.pushes) == 1
    del welt.lab.fehler["pr_anlegen"]
    assert welt.lauf() == [f"#{N}: Pin-PR #412 geöffnet"]
    assert len(welt.pushes) == 1  # schon gepusht, kein zweiter Push
    assert len(git(welt.origin, "log", "--format=%H", f"master..{PIN_BRANCH}").splitlines()) == 1
    assert welt.stand.funde[str(N)].pr == 412


def test_pin_nimmt_offenen_pr_zum_kopf(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    welt.daten["kopf"] = {(ISSUE_REPO, PIN_BRANCH): [{"number": 430, "state": "open"}]}
    assert welt.lauf() == [f"#{N}: Pin-PR #430 geöffnet"]
    assert welt.lab.namen("pr_anlegen") == []
    assert welt.stand.funde[str(N)].pr == 430


def test_pin_pr_gemergt_raeumt_pin_auf(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    welt.lauf()
    welt.pr(ISSUE_REPO, 412, merged=True, state="closed", sha="f" * 40)
    assert welt.lauf() == [f"#{N}: erledigt ({ISSUE_REPO}#412 gemergt)"]
    assert not welt.wt(ISSUE_REPO, PIN_BRANCH).exists()
    assert not _zweig_da(welt.klon, PIN_BRANCH)
    assert welt.lab.namen("branch_loeschen")[-1] == ("branch_loeschen", ISSUE_REPO, PIN_BRANCH)
    assert welt.stand.funde[str(N)].zustand == "erledigt"


def test_pin_schluessel_verdacht_haelt_zurueck(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    monkeypatch.setattr(nachlauf_modul, "pruefen", lambda t: "github_pat" if "Fixes" in t else None)
    vorher = FundStand(**vars(welt.stand.funde[str(N)]))
    assert welt.lauf() == [f"#{N}: Pin zurückgehalten (github_pat)"]
    satz = PRAEFIX + "Antwort zurückgehalten (Schlüssel-Verdacht: github_pat)."
    assert welt.lab.namen("kommentieren") == [("kommentieren", ISSUE_REPO, N, satz)]
    assert welt.pushes == [] and welt.lab.namen("pr_anlegen") == []
    assert welt.stand.funde[str(N)] == vorher
    # nächster Durchgang: kein zweiter Rückhalte-Kommentar
    assert welt.lauf() == [f"#{N}: Pin zurückgehalten (github_pat)"]
    assert len(welt.lab.namen("kommentieren")) == 1


def test_pin_zu_gross_haelt_zurueck(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    from selbstbild.debugger.klon import PushZuGross

    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)

    def zu_gross(*args: object) -> str:
        raise PushZuGross()

    monkeypatch.setattr(nachlauf_modul, "push_inhalt", zu_gross)
    assert welt.lauf() == [f"#{N}: Pin zurückgehalten (zu_gross)"]
    satz = PRAEFIX + "Antwort zurückgehalten (Schlüssel-Verdacht: zu_gross)."
    assert welt.lab.namen("kommentieren") == [("kommentieren", ISSUE_REPO, N, satz)]
    assert welt.pushes == [] and welt.lab.namen("pr_anlegen") == []


def test_pin_spiegel_haelt_beim_push_zurueck(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    from selbstbild.debugger.spiegel import ZurueckgehaltenFehler

    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)

    def halt(wt: Path, branch: str) -> None:
        welt.pushes.append((wt, branch))
        raise ZurueckgehaltenFehler("zu_gross")

    monkeypatch.setattr(welt, "push", halt)
    vorher = FundStand(**vars(welt.stand.funde[str(N)]))
    for _ in range(2):
        assert welt.lauf() == [f"#{N}: Pin zurückgehalten (zu_gross)"]
        assert welt.stand.funde[str(N)] == vorher
    satz = PRAEFIX + "Antwort zurückgehalten (Schlüssel-Verdacht: zu_gross)."
    assert welt.lab.namen("kommentieren") == [("kommentieren", ISSUE_REPO, N, satz)]
    assert len(welt.pushes) == 2 and welt.lab.namen("pr_anlegen") == []


def test_pin_push_ziel_unklar_wartet(welt: Welt, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ein ValueError beim Push (Repo zum Worktree unklar) lässt den Fund warten, ohne PR, je Grund ein Kommentar."""
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)

    def unklar(wt: Path, branch: str) -> None:
        welt.pushes.append((wt, branch))
        raise ValueError("repo_zum_worktree_unklar")

    monkeypatch.setattr(welt, "push", unklar)
    vorher = FundStand(**vars(welt.stand.funde[str(N)]))
    for _ in range(2):
        assert welt.lauf() == [f"#{N}: Pin wartet (push)"]
        assert welt.stand.funde[str(N)] == vorher
    assert welt.lab.namen("kommentieren") == [
        ("kommentieren", ISSUE_REPO, N, PRAEFIX + "Pin für vibemind-os wartet (push).")]
    assert len(welt.pushes) == 2 and welt.lab.namen("pr_anlegen") == []


ENTHALTEN = f"{PRAEFIX}Der Fix aus {OS_REPO}#77 ist in Vibemind_V1 schon enthalten."


def _schon_enthalten_pruefen(welt: Welt) -> None:
    assert welt.lauf() == [f"#{N}: Fix schon in Vibemind_V1 enthalten, erledigt"]
    assert welt.pushes == [] and welt.lab.namen("pr_anlegen") == []
    assert not _zweig_da(welt.origin, PIN_BRANCH)
    fund = welt.stand.funde[str(N)]
    assert (fund.repo, fund.branch, fund.pr, fund.zustand) == (OS_REPO, OS_BRANCH, 77, "erledigt")
    assert not welt.wt(ISSUE_REPO, PIN_BRANCH).exists() and not _zweig_da(welt.klon, PIN_BRANCH)
    assert welt.lab.namen("branch_loeschen") == [
        ("branch_loeschen", OS_REPO, OS_BRANCH), ("branch_loeschen", ISSUE_REPO, PIN_BRANCH)]
    assert welt.kommentare() == [ENTHALTEN]
    assert welt.lauf() == []  # erledigt: kein weiterer Durchgang


def test_pin_schon_auf_basis(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = git(welt.os_origin, "rev-parse", "master")  # schon im äußeren master gepinnt
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    assert _gitlink(welt.origin, "master") == sha
    _schon_enthalten_pruefen(welt)


# ---------- Fix-Runde 1 ----------

def test_neuerer_pin_auf_basis_kein_rueckschritt(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    x = welt.os_merge()
    y = commit(welt.os_arbeit, "spaeter.txt")
    git(welt.os_arbeit, "push", "-q", "origin", "master")
    welt.aussen_pin(y)  # äußerer master pinnt schon y, das x enthält
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=x)
    _schon_enthalten_pruefen(welt)
    assert _gitlink(welt.origin, "master") == y


def test_pin_auseinander_nicht_linear(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    x = welt.os_merge()
    git(welt.os_arbeit, "checkout", "-q", "-b", "seite", "HEAD~1")
    z = commit(welt.os_arbeit, "seite.txt")
    git(welt.os_arbeit, "push", "-q", "origin", "seite")
    welt.aussen_pin(z)  # weder enthält x z noch z x
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=x)
    vorher = FundStand(**vars(welt.stand.funde[str(N)]))
    for _ in range(2):
        assert welt.lauf() == [f"#{N}: Pin wartet (nicht_linear)"]
    assert welt.stand.funde[str(N)] == vorher
    assert welt.kommentare() == [f"{PRAEFIX}Pin für vibemind-os wartet (nicht_linear)."]
    assert welt.pushes == [] and welt.lab.namen("pr_anlegen") == []
    assert not _zweig_da(welt.origin, PIN_BRANCH) and not welt.wt(ISSUE_REPO, PIN_BRANCH).exists()
    assert _gitlink(welt.origin, "master") == z


def test_pin_pr_geschlossen_ohne_merge_erledigt(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    welt.lauf()
    welt.pr(ISSUE_REPO, 412, merged=False, state="closed")
    assert welt.lauf() == [f"#{N}: Pin-PR {ISSUE_REPO}#412 ohne Merge geschlossen, erledigt"]
    fund = welt.stand.funde[str(N)]
    assert (fund.repo, fund.branch, fund.pr, fund.zustand) == (ISSUE_REPO, PIN_BRANCH, 412, "erledigt")
    assert not welt.wt(ISSUE_REPO, PIN_BRANCH).exists() and not _zweig_da(welt.klon, PIN_BRANCH)
    assert welt.lab.namen("branch_loeschen")[-1] == ("branch_loeschen", ISSUE_REPO, PIN_BRANCH)
    assert welt.lauf() == []


def test_runde_nutzt_pin_worktree_des_nachlaufs(welt: Welt) -> None:
    welt.fund_mit_branch(OS_REPO, OS_BRANCH, pr=77)
    sha = welt.os_merge()
    welt.pr(OS_REPO, 77, merged=True, state="closed", sha=sha)
    welt.lauf()
    pin_wt = welt.wt(ISSUE_REPO, PIN_BRANCH)
    assert pin_wt.exists()
    gesehen: list[Path] = []

    def helfer(wt: Path, prompt: str, sitzung: str | None, frist_s: float) -> HelferErgebnis:
        gesehen.append(wt)
        return HelferErgebnis("s-2", "Antwort zum Pin.", "antwort", None)

    welt.helfer = helfer
    k = Kommentar(id=9, repo=ISSUE_REPO, nummer=412, autor="Flissel", text="passt das?",
                  erstellt="2026-11-07T11:50:00Z", review=False)
    assert runde_ausfuehren(Auftrag(N, (k,)), welt.konfig, welt.stand, welt.klon, welt.ordner, welt.w) == "ok"
    assert gesehen == [pin_wt]
    assert welt.lab.namen("kommentieren")[-1] == ("kommentieren", ISSUE_REPO, 412, PRAEFIX + "Antwort zum Pin.")
