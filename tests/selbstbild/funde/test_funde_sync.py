"""Stufe 5, Task 4: GitHub-Adapter und Issue-Abgleich mit Bremsen (gefaelschtes gh, nie Netz)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from selbstbild.funde.github import GitHub, GitHubFehler
from selbstbild.funde.sammeln import UNLESBAR_PRAEFIX
from selbstbild.funde.sync import (
    IssueStand,
    Ziel,
    abgleichen,
    nach_ziel,
    stand_pfad,
    unlesbare_arten,
    ziele_aus_regeln,
)
from selbstbild.funde.text import BEGIN, END
from selbstbild.funde.typen import Fund, fund_schluessel
from selbstbild.hooks import regeln_lesen
from selbstbild.regeln import regeln_laden

TOKEN = "gho_GEHEIMTOKEN0123456789abcdefXYZ"
REPO = "Flissel/Vibemind_V1"
TAG1 = date(2026, 11, 4)
TAG2 = date(2026, 11, 5)
COMMIT = "abc1234def5678"
GEPRUEFT = frozenset({"vibemind"})


# ---------- gefaelschtes gh ----------

class FakeGh:
    """Spielt `gh auth token` und die noetigen REST-Pfade von `gh api` nach. Nie Netz."""

    def __init__(self) -> None:
        self.issues: dict[int, dict[str, Any]] = {}
        self.labels: set[str] = set()
        self.kommentare: list[tuple[int, str]] = []
        self.aufrufe: list[tuple[str, str]] = []
        self.argvs: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self.tag = TAG1
        self.naechste = 1
        self.anlegen_versuche = 0
        self.fehler_bei_anlegen: int | None = None
        self.verloren_bei_anlegen: int | None = None
        self.fehler_text = b"gh: Server Error (HTTP 502) geheimer antworttext"
        self.privat = True
        self.label_verwerfen = False

    # Hilfen fuer Tests
    def issue(self, titel: str, body: str, *, state: str = "open", state_reason: str | None = None,
              labels: tuple[str, ...] = ("selbstbild",), created: date | None = None) -> int:
        n = self.naechste
        self.naechste += 1
        self.issues[n] = {"number": n, "title": titel, "body": body, "state": state, "state_reason": state_reason,
                          "labels": [{"name": x} for x in labels],
                          "created_at": f"{(created or date(2026, 1, 1)).isoformat()}T12:00:00Z"}
        return n

    def methoden(self) -> set[str]:
        return {m for m, _ in self.aufrufe}

    def offen(self) -> list[dict[str, Any]]:
        return [i for i in self.issues.values() if i["state"] == "open"]

    def __call__(self, args: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        argv = list(args)
        self.argvs.append(argv)
        self.envs.append(kw.get("env"))
        assert "timeout" in kw and kw["timeout"]
        if argv[1:3] == ["auth", "token"]:
            return subprocess.CompletedProcess(argv, 0, (TOKEN + "\n").encode(), b"")
        assert argv[1] == "api"
        assert kw["env"]["GH_TOKEN"] == TOKEN
        methode = argv[argv.index("--method") + 1]
        pfad = next(a for a in argv[2:] if a.startswith("/"))
        daten = json.loads(kw["input"].decode("utf-8")) if kw.get("input") is not None else None
        if methode != "GET":
            assert argv[-2:] == ["--input", "-"]
        self.aufrufe.append((methode, pfad))
        return self._route(argv, methode, pfad, daten)

    def _ok(self, argv: list[str], obj: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, json.dumps(obj).encode("utf-8"), b"")

    def _fehler(self, argv: list[str], text: bytes) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 1, b"", text)

    def _route(self, argv: list[str], methode: str, pfad: str, daten: Any) -> subprocess.CompletedProcess[bytes]:
        teile = urlsplit(pfad)
        p = teile.path.split("/")[1:]  # repos, owner, name, ...
        assert p[:3] == ["repos", *REPO.split("/")]
        rest = p[3:]
        if methode == "GET" and rest == []:
            return self._ok(argv, {"full_name": REPO, "private": self.privat})
        if methode == "GET" and rest == ["issues"]:
            q = parse_qs(teile.query)
            assert q["labels"] == ["selbstbild"] and q["state"] == ["all"] and q["per_page"] == ["100"]
            seite = int(q["page"][0])
            alle = sorted((i for i in self.issues.values() if any(x["name"] == "selbstbild" for x in i["labels"])),
                          key=lambda i: -i["number"])
            return self._ok(argv, alle[(seite - 1) * 100: seite * 100])
        if rest == ["labels", "selbstbild"] and methode == "GET":
            if "selbstbild" in self.labels:
                return self._ok(argv, {"name": "selbstbild"})
            return self._fehler(argv, b"gh: Not Found (HTTP 404)")
        if rest == ["labels"] and methode == "POST":
            self.labels.add(daten["name"])
            return self._ok(argv, {"name": daten["name"]})
        if rest == ["issues"] and methode == "POST":
            self.anlegen_versuche += 1
            if self.fehler_bei_anlegen == self.anlegen_versuche:
                return self._fehler(argv, self.fehler_text)
            assert daten["labels"] == ["selbstbild"]
            labels = () if self.label_verwerfen else ("selbstbild",)
            n = self.issue(daten["title"], daten["body"], created=self.tag, labels=labels)
            if self.verloren_bei_anlegen == self.anlegen_versuche:
                return self._fehler(argv, b"gh: Gateway Timeout (HTTP 504)")
            return self._ok(argv, self.issues[n])
        if len(rest) == 2 and rest[0] == "issues" and methode == "PATCH":
            i = self.issues[int(rest[1])]
            for k in ("body", "state", "state_reason"):
                if k in daten:
                    i[k] = daten[k]
            if daten.get("state") == "open":
                i["state_reason"] = "reopened"
            return self._ok(argv, {"number": i["number"]})
        if len(rest) == 3 and rest[0] == "issues" and rest[2] == "comments" and methode == "POST":
            self.kommentare.append((int(rest[1]), daten["body"]))
            return self._ok(argv, {"id": 1})
        raise AssertionError(f"unerwarteter Aufruf {methode} {pfad}")


# ---------- Hilfen ----------

def _fund(i: int, schwere: str = "hoch", art: str = "doku_widerspruch", repo: str = "vibemind",
          text: str = "Widerspruch") -> Fund:
    return Fund(art=art, schwere=schwere, code_repo=repo, titel=f"Fund {i}", text=text,  # type: ignore[arg-type]
                belege=(f"docs/d{i}.md:{i}",), anker=f"{art}|docs/d{i}.md|a{i}")


def _ziel(**kw: Any) -> Ziel:
    return replace(Ziel(repo=REPO, konto="Flissel", oeffentlich=False), **kw)


def _stand(tmp_path: Path) -> IssueStand:
    return IssueStand.laden(tmp_path / "issues-test.json")


def _lauf(fake: FakeGh | None, funde: list[Fund], tmp_path: Path, *, heute: date = TAG1, senden: bool = True,
          ziel: Ziel | None = None, unlesbar: frozenset[str] = frozenset(), commit: str = COMMIT,
          geprueft: frozenset[str] = GEPRUEFT):
    gh = GitHub("Flissel", laufzeit=fake) if fake is not None else None
    return abgleichen(funde, ziel or _ziel(), gh, _stand(tmp_path), heute=heute, senden=senden, commit=commit,
                      geprueft=geprueft, unlesbar=unlesbar)


def _einzel(fake: FakeGh) -> list[dict[str, Any]]:
    return [i for i in fake.issues.values() if "selbstbild:schluessel=" in i["body"]]


def _sammel(fake: FakeGh) -> list[dict[str, Any]]:
    return [i for i in fake.issues.values() if "selbstbild:sammel=" in i["body"]]


def _zelle(f: Fund) -> str:
    return f.anker.replace("|", chr(92) + "|")


def _marke(f: Fund) -> str:
    return f"<!-- selbstbild:schluessel={fund_schluessel(f)} -->"


# ---------- Brief-Tests ----------

def test_keine_dublette_ohne_lokalen_stand(tmp_path: Path) -> None:
    fake = FakeGh()
    f = _fund(1)
    fake.issue("alt", f"{_marke(f)}\n\nirgendwas")
    assert not (tmp_path / "issues-test.json").exists()
    b = _lauf(fake, [f], tmp_path)
    assert ("POST", f"/repos/{REPO}/issues") not in fake.aufrufe
    assert len(fake.issues) == 1 and b.neu == 0 and b.fehler is None


def test_keine_dublette_bei_kaputtem_stand(tmp_path: Path) -> None:
    fake = FakeGh()
    f = _fund(1)
    fake.issue("alt", f"{_marke(f)}\n\nirgendwas")
    (tmp_path / "issues-test.json").write_text("{kaputt", encoding="utf-8")
    b = _lauf(fake, [f], tmp_path)
    assert len(fake.issues) == 1 and b.neu == 0
    assert any("lokaler Stand unlesbar" in n for n in b.notizen)


def test_wontfix_bleibt_zu(tmp_path: Path) -> None:
    fake = FakeGh()
    f1, f2, f3 = _fund(1), _fund(2), _fund(3)
    n1 = fake.issue("a", _marke(f1), state="closed", state_reason="completed", labels=("selbstbild", "wontfix"))
    n2 = fake.issue("b", _marke(f2), state="closed", state_reason="not_planned")
    n3 = fake.issue("c", _marke(f3), state="closed", state_reason="completed")
    b = _lauf(fake, [f1, f2, f3], tmp_path)
    assert fake.issues[n1]["state"] == "closed" and fake.issues[n2]["state"] == "closed"
    assert fake.issues[n3]["state"] == "open"  # ohne wontfix: wieder offen
    assert len(fake.issues) == 3  # nichts neu angelegt, auch kein Sammel-Issue
    assert sorted(b.unterdrueckt) == sorted([fund_schluessel(f1), fund_schluessel(f2)])
    assert b.neu == 0  # Wiederoeffnen zaehlt nicht als neu
    assert IssueStand.laden(tmp_path / "issues-test.json").neu_heute(REPO, TAG1) == 0


def test_tagesgrenze(tmp_path: Path) -> None:
    fake = FakeGh()
    funde = [_fund(i) for i in range(7)]
    b1 = _lauf(fake, funde, tmp_path)
    assert len(_einzel(fake)) == 5 and b1.neu == 5
    assert len(b1.ueber_grenze) == 2
    (sammel,) = _sammel(fake)
    for k in b1.ueber_grenze:
        f = next(f for f in funde if fund_schluessel(f) == k)
        assert _zelle(f) in sammel["body"]
    assert IssueStand.laden(tmp_path / "issues-test.json").neu_heute(REPO, TAG1) == 5
    # gleicher Tag, zweiter Lauf: nichts Neues
    b1b = _lauf(fake, funde, tmp_path)
    assert len(_einzel(fake)) == 5 and b1b.neu == 0
    # naechster Tag: die restlichen 2
    fake.tag = TAG2
    b2 = _lauf(fake, funde, tmp_path, heute=TAG2)
    assert len(_einzel(fake)) == 7 and b2.neu == 2 and b2.ueber_grenze == []
    assert len({fund_schluessel(f) for f in funde}) == len({i["body"].split(" -->")[0] for i in _einzel(fake)})
    # das Sammel-Issue ist jetzt leer und wird geschlossen
    assert sammel["state"] == "closed"


def test_tagesgrenze_ohne_lokalen_stand_zaehlt_github(tmp_path: Path) -> None:
    fake = FakeGh()
    funde = [_fund(i) for i in range(7)]
    _lauf(fake, funde, tmp_path)
    (tmp_path / "issues-test.json").unlink()
    b = _lauf(fake, funde + [_fund(9)], tmp_path)
    assert b.neu == 0 and len(_einzel(fake)) == 5  # 5 heute angelegt laut GitHub: Grenze bleibt


def test_fund_weg_schliesst(tmp_path: Path) -> None:
    fake = FakeGh()
    hoch, mittel = _fund(1), _fund(2, schwere="mittel")
    _lauf(fake, [hoch, mittel], tmp_path)
    (einzel,) = _einzel(fake)
    (sammel,) = _sammel(fake)
    assert einzel["state"] == "open" and sammel["state"] == "open"
    b = _lauf(fake, [], tmp_path)
    assert einzel["state"] == "closed" and sammel["state"] == "closed"
    assert (einzel["number"], "Fund im Code verschwunden (Commit `abc1234`)") in fake.kommentare
    assert {a.art for a in b.aktionen} == {"schliessen", "sammel_schliessen"}


def test_fehler_mitten_drin_kein_doppelt(tmp_path: Path) -> None:
    fake = FakeGh()
    fake.fehler_bei_anlegen = 3
    funde = [_fund(i) for i in range(4)]
    b1 = _lauf(fake, funde, tmp_path)
    assert b1.fehler == "http_502" and len(_einzel(fake)) == 2
    assert IssueStand.laden(tmp_path / "issues-test.json").neu_heute(REPO, TAG1) == 2  # nur Erfolge
    assert any(a.status == "fehlgeschlagen" for a in b1.aktionen)
    assert any(a.status == "abgebrochen" for a in b1.aktionen)
    fake.fehler_bei_anlegen = None
    b2 = _lauf(fake, funde, tmp_path)
    assert b2.fehler is None and b2.neu == 2
    schluessel = [i["body"].split(" -->")[0] for i in _einzel(fake)]
    assert len(schluessel) == 4 and len(set(schluessel)) == 4


def test_antwort_verloren_kein_doppelt(tmp_path: Path) -> None:
    fake = FakeGh()
    fake.verloren_bei_anlegen = 2  # GitHub legt an, die Antwort geht verloren
    funde = [_fund(i) for i in range(3)]
    b1 = _lauf(fake, funde, tmp_path)
    assert b1.fehler == "http_504" and len(_einzel(fake)) == 2
    b2 = _lauf(fake, funde, tmp_path)
    schluessel = [i["body"].split(" -->")[0] for i in _einzel(fake)]
    assert len(schluessel) == 3 and len(set(schluessel)) == 3 and b2.neu == 1
    assert IssueStand.laden(tmp_path / "issues-test.json").neu_heute(REPO, TAG1) == 2  # lokal nur Erfolge


def test_trocken_schreibt_nichts(tmp_path: Path) -> None:
    fake = FakeGh()
    alt = _fund(50)
    fake.issue("weg", _marke(alt) + "\n<!-- selbstbild:repo=vibemind -->")
    funde = [_fund(i) for i in range(7)] + [_fund(8, schwere="mittel")]
    b = _lauf(fake, funde, tmp_path, senden=False)
    assert fake.methoden() == {"GET"}
    # nur Sichtbarkeit und Bestand, auch kein Label-Check
    assert all(p == f"/repos/{REPO}" or p.startswith(f"/repos/{REPO}/issues?") for _, p in fake.aufrufe)
    assert not (tmp_path / "issues-test.json").exists()
    arten = [a.art for a in b.aktionen]
    assert arten.count("anlegen") == 5 and arten.count("sammel_anlegen") == 1 and arten.count("schliessen") == 1
    assert {a.status for a in b.aktionen} == {"trocken"}
    # ohne gh: Bestand gilt als leer, derselbe Plan ohne Schliessen
    b0 = _lauf(None, funde, tmp_path, senden=False)
    assert [a.art for a in b0.aktionen].count("anlegen") == 5
    assert any("Bestand" in n for n in b0.notizen)
    with pytest.raises(ValueError):
        _lauf(None, funde, tmp_path, senden=True)


def test_oeffentlich_kein_konfig(tmp_path: Path) -> None:
    fake = FakeGh()
    konfig = [_fund(1, art="konfig"), _fund(2, art="konfig", schwere="mittel")]
    tot = _fund(3, art="tot")
    ziel = _ziel(oeffentlich=True, arten=("doku_widerspruch", "doppelung", "tot", "konfig"))
    b = _lauf(fake, [*konfig, tot], tmp_path, ziel=ziel)
    bodies = " ".join(i["body"] + i["title"] for i in fake.issues.values())
    assert "Konfig" not in bodies and "konfig" not in bodies
    assert len(_einzel(fake)) == 1
    # ein Alt-Issue der Art konfig (von Hand dort): wird weder geschlossen noch bearbeitet
    alt_nr = fake.issue("alt", _marke(_fund(9, art="konfig")) + "\n<!-- selbstbild:repo=vibemind -->")
    _lauf(fake, [*konfig, tot], tmp_path, ziel=ziel)
    assert fake.issues[alt_nr]["state"] == "open" and fake.kommentare == []
    for f in konfig:
        assert any(fund_schluessel(f) in z for z in b.ausgelassen)
    assert "D6" in b.text()
    # privat geht konfig raus
    fake2 = FakeGh()
    _lauf(fake2, konfig, tmp_path / "p", ziel=_ziel(arten=("konfig",)))
    assert len(_einzel(fake2)) == 2 and len(_sammel(fake2)) == 0  # S5-11: konfig immer einzeln


def test_token_nie_im_bericht_oder_log(tmp_path: Path, caplog: pytest.LogCaptureFixture,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    caplog.set_level(logging.DEBUG)
    vorher = os.environ.get("GH_TOKEN")
    fake = FakeGh()
    fake.fehler_bei_anlegen = 2
    gh = GitHub("Flissel", laufzeit=fake)
    b = abgleichen([_fund(i) for i in range(3)], _ziel(), gh, _stand(tmp_path), heute=TAG1, senden=True,
                   commit=COMMIT, geprueft=GEPRUEFT)
    assert b.fehler == "http_502"
    texte = [repr(b), str(b), b.text(), caplog.text, *capsys.readouterr(),
             (tmp_path / "issues-test.json").read_text(encoding="utf-8"), repr(gh.__dict__)]
    for t in texte:
        assert TOKEN not in t and "geheimer antworttext" not in t
    for argv, env in zip(fake.argvs, fake.envs):
        assert TOKEN not in " ".join(argv)
        if argv[1:3] == ["auth", "token"]:
            assert argv[3:] == ["--hostname", "github.com", "--user", "Flissel"]
            assert env is not None and "GH_TOKEN" not in env and "GITHUB_TOKEN" not in env
        else:
            assert env is not None and env["GH_TOKEN"] == TOKEN
    assert sum(1 for a in fake.argvs if a[1:3] == ["auth", "token"]) == 1  # je Instanz gecacht
    assert os.environ.get("GH_TOKEN") == vorher
    # Fehlerobjekt selbst: nur die Art
    fake.fehler_bei_anlegen = fake.anlegen_versuche + 1
    with pytest.raises(GitHubFehler) as e:
        gh.anlegen(REPO, "t", "x")
    assert e.value.art == "http_502" and TOKEN not in str(e.value) and "geheim" not in repr(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__


# ---------- Ruling S5-7 ----------

def test_unlesbare_quelle_schliesst_nichts(tmp_path: Path) -> None:
    fake = FakeGh()
    doku_hoch, doku_mittel = _fund(1), _fund(2, schwere="mittel")
    doppel = _fund(3, art="doppelung", schwere="mittel")
    _lauf(fake, [doku_hoch, doku_mittel, doppel], tmp_path)
    vorher = {n: dict(i) for n, i in fake.issues.items()}
    assert len(vorher) == 3
    notizen = [f"{UNLESBAR_PRAEFIX}doku: doku-pruefliste fehlt: x"]
    unl = unlesbare_arten(notizen)
    assert unl == frozenset({"doku_widerspruch"})
    b = _lauf(fake, [doppel], tmp_path, unlesbar=unl)
    assert fake.issues == vorher  # nichts geschlossen, Sammel-Block unveraendert
    assert ("PATCH" not in fake.methoden()) and fake.kommentare == []
    assert any("doku_widerspruch" in n and "unlesbar" in n for n in b.notizen)
    # Analyse unlesbar: auch Doppelungen werden nicht geschlossen
    b2 = _lauf(fake, [], tmp_path, unlesbar=frozenset({"doku_widerspruch", "doppelung", "tot"}))
    assert fake.issues == vorher and b2.aktionen == []
    # lesbar und leer: jetzt wird geschlossen
    _lauf(fake, [], tmp_path)
    assert all(i["state"] == "closed" for i in fake.issues.values())


def test_unlesbare_arten_aus_notizen() -> None:
    assert unlesbare_arten([]) == frozenset()
    assert unlesbare_arten(["toter Code: gesperrt (3/30 bestätigt)"]) == frozenset({"tot"})
    assert unlesbare_arten([f"{UNLESBAR_PRAEFIX}analyse: Analyse fehlt"]) == frozenset({"doppelung", "tot"})
    assert unlesbare_arten([f"{UNLESBAR_PRAEFIX}analyse: 2 fehlerhafte Doppelungs-Einträge übersprungen",
                            f"{UNLESBAR_PRAEFIX}doku: x"]) == frozenset({"doppelung", "tot", "doku_widerspruch"})
    # unbekannte Quelle: fail-closed, keine Art darf schliessen
    assert unlesbare_arten([f"{UNLESBAR_PRAEFIX}neu: x"]) == frozenset(
        {"doku_widerspruch", "doppelung", "tot", "konfig"})


# ---------- Weitere Verhaltensregeln ----------

def test_sammel_block_laesst_menschentext_stehen(tmp_path: Path) -> None:
    fake = FakeGh()
    m1, m2 = _fund(1, schwere="mittel"), _fund(2, schwere="mittel")
    _lauf(fake, [m1], tmp_path)
    (s,) = _sammel(fake)
    s["body"] = "Notiz vom Menschen oben\r\n\r\n" + s["body"].replace("\n", "\r\n") + "\r\nUnten auch."
    _lauf(fake, [m1, m2], tmp_path)
    assert s["body"].startswith("Notiz vom Menschen oben\r\n") and s["body"].endswith("\r\nUnten auch.")
    assert _zelle(m2) in s["body"] and s["body"].count(BEGIN) == 1
    # beschaedigter Block: nicht anfassen, im Bericht nennen
    s["body"] = s["body"].replace(END, "")
    davor = s["body"]
    b = _lauf(fake, [m1], tmp_path)
    assert s["body"] == davor and any("beschädigt" in n for n in b.notizen)


def test_belege_werden_aktualisiert(tmp_path: Path) -> None:
    fake = FakeGh()
    f = _fund(1)
    _lauf(fake, [f], tmp_path)
    (i,) = _einzel(fake)
    i["body"] = i["body"].replace("Widerspruch", "Widerspruch (vom Menschen ergänzt)")
    f2 = replace(f, belege=("docs/neu.md:7",))
    b = _lauf(fake, [f2], tmp_path)
    assert "`docs/neu.md:7`" in i["body"] and "`docs/d1.md:1`" not in i["body"]
    assert "(vom Menschen ergänzt)" in i["body"]
    assert [a.art for a in b.aktionen] == ["bearbeiten"]
    b2 = _lauf(fake, [f2], tmp_path)
    assert b2.aktionen == [] and b2.unveraendert == 1


def test_schluesselfilter_sendet_nicht(tmp_path: Path) -> None:
    fake = FakeGh()
    wert = "ghp_" + "A1b2C3d4" * 5
    f = _fund(1, text=f"siehe {wert}")
    b = _lauf(fake, [f], tmp_path)
    assert fake.issues == {} and "POST" not in fake.methoden()
    assert any("GitHub-Token (klassisch)" in g for g in b.gesperrt)
    assert wert not in b.text() and wert not in repr(b)


def test_fremdes_code_repo_wird_nicht_angefasst(tmp_path: Path) -> None:
    fake = FakeGh()
    fremd = _fund(1, repo="vibemind/vibemind-os")
    fake.issue("[selbstbild] Doku-Widerspruch: Fund 1 (vibemind/vibemind-os)", _marke(fremd))
    ziel = _ziel(ausser=frozenset({"vibemind/vibemind-os"}))
    b = _lauf(fake, [_fund(2, repo="vibemind/vibemind-os")], tmp_path, ziel=ziel)
    assert fake.offen() and len(fake.issues) == 1
    assert any("anderes Ziel" in z for z in b.ausgelassen)


# ---------- GitHub-Adapter ----------

def test_issues_paginiert_ohne_pull_requests() -> None:
    fake = FakeGh()
    for i in range(150):
        fake.issue(f"t{i}", "x")
    fake.issues[fake.naechste] = {"number": fake.naechste, "title": "pr", "body": "", "state": "open",
                                  "labels": [{"name": "selbstbild"}], "pull_request": {"url": "u"}, "created_at": "2026-01-01T00:00:00Z"}
    fake.naechste += 1
    alle = GitHub("Flissel", laufzeit=fake).issues(REPO)
    assert len(alle) == 150 and all("pull_request" not in i for i in alle)
    assert [p for _, p in fake.aufrufe] == [
        f"/repos/{REPO}/issues?labels=selbstbild&state=all&per_page=100&page={n}" for n in (1, 2)]


def test_label_sicherstellen_und_methoden() -> None:
    fake = FakeGh()
    gh = GitHub("Flissel", laufzeit=fake)
    gh.label_sicherstellen(REPO)
    gh.label_sicherstellen(REPO)
    assert fake.labels == {"selbstbild"}
    assert fake.aufrufe.count(("POST", f"/repos/{REPO}/labels")) == 1
    n = gh.anlegen(REPO, "t", "b")
    gh.bearbeiten(REPO, n, "neu")
    gh.schliessen(REPO, n, "weg")
    assert fake.issues[n]["body"] == "neu" and fake.issues[n]["state"] == "closed"
    assert fake.issues[n]["state_reason"] == "completed" and fake.kommentare == [(n, "weg")]
    gh.oeffnen(REPO, n)
    assert fake.issues[n]["state"] == "open"


def test_github_fehlerarten() -> None:
    def zeitlimit(args: list[str], **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(args, kw["timeout"], output=TOKEN.encode())

    def fehlt(args: list[str], **kw: Any) -> Any:
        raise FileNotFoundError("gh")

    def leer(args: list[str], **kw: Any) -> Any:
        return subprocess.CompletedProcess(args, 0, b"\n", b"")

    def geprueft_fehler(args: list[str], **kw: Any) -> Any:
        raise subprocess.CalledProcessError(1, args, output=TOKEN.encode(), stderr=b"gh: Bad (HTTP 401) geheim")

    for lauf, art in ((zeitlimit, "zeitlimit"), (fehlt, "gh_fehlt"), (leer, "kein_token"),
                      (geprueft_fehler, "http_401")):
        with pytest.raises(GitHubFehler) as e:
            GitHub("Flissel", laufzeit=lauf).issues(REPO)
        assert e.value.art == art
        # Minor 2: keine angehaengte Ausnahme (mit Teil-Ausgabe oder stderr)
        assert e.value.__context__ is None and e.value.__cause__ is None
    with pytest.raises(GitHubFehler) as e:
        GitHub("Flissel", laufzeit=FakeGh()).issues("../evil")
    assert e.value.art == "repo_ungueltig"
    with pytest.raises(ValueError):
        GitHub("bad user; rm")


def test_bestand_unlesbar_bricht_ab(tmp_path: Path) -> None:
    fake = FakeGh()

    def kaputt(args: list[str], **kw: Any) -> Any:
        if args[1] == "api":
            return subprocess.CompletedProcess(args, 1, b"", b"gh: API rate limit exceeded (HTTP 403)")
        return fake(args, **kw)

    gh = GitHub("Flissel", laufzeit=kaputt)
    with pytest.raises(GitHubFehler) as e:
        abgleichen([_fund(1)], _ziel(), gh, _stand(tmp_path), heute=TAG1, senden=True, commit=COMMIT,
                   geprueft=GEPRUEFT)
    assert e.value.art == "http_403" and fake.issues == {}


# ---------- Konfiguration (Ruling S5-1) ----------

ISSUES_YAML = """
name: vibemind
issues:
  ziel_standard: {repo: Flissel/Vibemind_V1, konto: Flissel, oeffentlich: false}
  ziele:
    vibemind/vibemind-os: {repo: Vibemind-LAB/vibemind-os, konto: Vibemind-LAB, oeffentlich: true}
  arten: [doku_widerspruch, doppelung, tot, konfig]
  max_einzel_je_tag: 3
"""


def test_regeln_issues_abschnitt() -> None:
    r = regeln_laden(ISSUES_YAML)
    assert r.issues is not None and r.issues.max_einzel_je_tag == 3
    standard, os_ziel = ziele_aus_regeln(r)
    assert standard.repo == "Flissel/Vibemind_V1" and standard.konto == "Flissel" and not standard.oeffentlich
    assert standard.max_einzel_je_tag == 3 and standard.zustaendig("vibemind")
    assert not standard.zustaendig("vibemind/vibemind-os")
    assert os_ziel.oeffentlich and os_ziel.konto == "Vibemind-LAB"
    assert os_ziel.zustaendig("vibemind/vibemind-os") and not os_ziel.zustaendig("vibemind")
    gruppen, ohne = nach_ziel([_fund(1), _fund(2, repo="vibemind/vibemind-os")], [standard, os_ziel])
    assert [(z.repo, len(fs)) for z, fs in gruppen] == [("Flissel/Vibemind_V1", 1), ("Vibemind-LAB/vibemind-os", 1)]
    assert ohne == []
    # der stdlib-Leser des Waechters bleibt unberuehrt
    assert regeln_lesen(ISSUES_YAML) == regeln_lesen("name: vibemind\n")


def test_regeln_ohne_issues_nur_bericht() -> None:
    r = regeln_laden("issue_ziel:\n  repo: Flissel/Vibemind_V1\n  privat: true\n")
    (z,) = ziele_aus_regeln(r)
    assert z.repo == "Flissel/Vibemind_V1" and z.konto is None and not z.oeffentlich and z.max_einzel_je_tag == 5
    assert ziele_aus_regeln(regeln_laden(None)) == []


@pytest.mark.parametrize("schlecht", [
    "issues:\n  ziel_standard: {repo: 'kein-slash', konto: x}\n",
    "issues:\n  arten: [unbekannt]\n",
    "issues:\n  max_einzel_je_tag: -1\n",
    "issues:\n  ziel_standard: {repo: a/b, konto: 'x y'}\n",
    "issues:\n  tippfehler: 1\n",
])
def test_regeln_issues_ungueltig(schlecht: str) -> None:
    with pytest.raises(ValueError):
        regeln_laden(schlecht)


def test_oeffentlich_ist_vorgabe() -> None:
    r = regeln_laden("issues:\n  ziel_standard: {repo: a/b, konto: x}\n  ziele: {c: {repo: d/e, konto: y}}\n")
    assert all(z.oeffentlich for z in ziele_aus_regeln(r))  # D6 fail-closed


def test_fund_arten_wie_typen() -> None:
    from selbstbild.funde.typen import ARTEN
    from selbstbild.regeln import FUND_ARTEN
    assert FUND_ARTEN == ARTEN
    assert regeln_laden("issues:\n  max_einzel_je_tag: 20\n").issues.max_einzel_je_tag == 20  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        regeln_laden("issues:\n  max_einzel_je_tag: 21\n")


# ---------- Fix-Runde 1 ----------

def test_nicht_gepruefte_repos_bleiben_offen(tmp_path: Path) -> None:
    fake = FakeGh()
    _lauf(fake, [_fund(1, repo="vibemind"), _fund(2, repo="vibemind", schwere="mittel")], tmp_path)
    ohne_marke = fake.issue("[selbstbild] etwas (siehe #12)", _marke(_fund(3)) + "\n\nText")
    assert len(fake.offen()) == 3
    # ein Lauf, der nur la_fungus geprueft hat (z. B. Worktree ohne Submodule): nichts von vibemind schliessen
    b = _lauf(fake, [_fund(7, repo="la_fungus", schwere="mittel")], tmp_path, geprueft=frozenset({"la_fungus"}))
    vibemind = [i for i in fake.issues.values() if "la_fungus" not in i["body"]]
    assert all(i["state"] == "open" for i in vibemind) and fake.kommentare == []
    assert not any(a.art.endswith("schliessen") for a in b.aktionen)
    assert any("nicht geprüfte" in n and "vibemind" in n for n in b.notizen)
    assert any("ohne Repo-Marke" in n for n in b.notizen)
    # wird vibemind geprueft, schliesst es -- das Issue ohne Repo-Marke bleibt trotzdem offen
    _lauf(fake, [], tmp_path, geprueft=frozenset({"vibemind", "la_fungus"}))
    assert fake.issues[ohne_marke]["state"] == "open"
    assert sum(1 for i in fake.issues.values() if i["state"] == "closed") == 3


SAMMEL_YAML = (
    "name: aussen\ndoku:\n  spaces: {}\n  ausschluss_namen: []\n  ausschluss_pfade: []\n  ketten_md: ''\n"
    "  compose_live: []\n  sql_wurzeln: ['.']\n  dienste_yaml: ''\n  waechter_zustandsdatei: ''\n"
    "  waechter_zuordnung: {}\n  tick_minuten: 5\n  marker: marker\n  agent: ''\n  judge_modell: sonnet\n"
    "  claude_bin: claude\n  rowboat_wissen: w\n  rowboat_notizen: n\n  rowboat_urteile: u\n")


def test_geprueft_repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from selbstbild.doku.konfig import aus_regeln
    from selbstbild.funde.sammeln import geprueft_repos
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "heim"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lokal"))
    w = tmp_path / "repo"
    w.mkdir()
    (w / "selbstbild.yaml").write_text(SAMMEL_YAML, encoding="utf-8")
    (w / ".gitmodules").write_text(
        '[submodule "os"]\n\tpath = vibemind-os\n[submodule "v"]\n\tpath = voice\n', encoding="utf-8")
    (w / "vibemind-os").mkdir()
    (w / "vibemind-os" / ".git").write_text("gitdir: ../.git/modules/os\n", encoding="utf-8")
    (w / "vibemind-os" / ".gitmodules").write_text('[submodule "s"]\n\tpath = spaces/sales\n', encoding="utf-8")
    (w / "vibemind-os" / "spaces" / "sales" / ".git").mkdir(parents=True)
    (w / "voice").mkdir()  # deklariert, Ordner da, aber kein Checkout
    assert geprueft_repos(w, doku_konfig=aus_regeln(w)) == frozenset(
        {"aussen", "aussen/vibemind-os", "aussen/vibemind-os/spaces/sales"})
    anders = tmp_path / "anders"
    anders.mkdir()
    (anders / "selbstbild.yaml").write_text(SAMMEL_YAML, encoding="utf-8")
    assert geprueft_repos(w, doku_konfig=aus_regeln(anders)) == frozenset()  # fail-closed


def test_oeffentlich_laut_github_kein_konfig(tmp_path: Path) -> None:
    fake = FakeGh()
    fake.privat = False  # Konfiguration sagt privat, GitHub sagt oeffentlich
    ziel = _ziel(oeffentlich=False, arten=("konfig", "tot"))
    b = _lauf(fake, [_fund(1, art="konfig"), _fund(2, art="konfig", schwere="mittel"), _fund(3, art="tot")],
              tmp_path, ziel=ziel)
    assert len(fake.issues) == 1 and "konfig" not in json.dumps(list(fake.issues.values()))
    assert any("nicht privat" in n for n in b.notizen) and "D6" in b.text()
    # auch trocken wird gefragt (nur lesend)
    fake2 = FakeGh()
    fake2.privat = False
    b2 = _lauf(fake2, [_fund(1, art="konfig")], tmp_path, ziel=ziel, senden=False)
    assert b2.aktionen == [] and ("GET", f"/repos/{REPO}") in fake2.aufrufe and fake2.methoden() == {"GET"}


def test_label_verworfen_sperrt_ziel(tmp_path: Path) -> None:
    fake = FakeGh()
    fake.label_verwerfen = True
    b1 = _lauf(fake, [_fund(i) for i in range(3)], tmp_path)
    assert b1.fehler == "label_fehlt" and len(fake.issues) == 1
    assert IssueStand.laden(tmp_path / "issues-test.json").sperren == {REPO: "label_fehlt"}
    assert IssueStand.laden(tmp_path / "issues-test.json").neu_heute(REPO, TAG1) == 1
    b2 = _lauf(fake, [_fund(i) for i in range(3)], tmp_path)
    assert len(fake.issues) == 1 and b2.fehler == "ziel_gesperrt"  # keine Dublette des unsichtbaren Issues
    assert all(a.status == "abgebrochen" for a in b2.aktionen)
    with pytest.raises(GitHubFehler) as e:
        GitHub("Flissel", laufzeit=fake).anlegen(REPO, "t", "x")
    assert e.value.art == "label_fehlt"


def test_marke_mit_leerraum(tmp_path: Path) -> None:
    fake = FakeGh()
    f, weg = _fund(1), _fund(2)
    s = fund_schluessel(f).split(":")
    fake.issue("a", f"<!--  selbstbild:schluessel = {s[0]}:{s[1]}  -->\n\nText")
    n = fake.issue("b", f"<!--selbstbild:schluessel={fund_schluessel(weg)}-->\n<!--  selbstbild:repo = vibemind  -->")
    fake.issue("c", "<!--  selbstbild:sammel = doku_widerspruch:vibemind  -->\n" + BEGIN + "\nalt\n" + END)
    b = _lauf(fake, [f, _fund(3, schwere="mittel")], tmp_path)
    assert b.neu == 0 and len(fake.issues) == 3  # kein neues Einzel-, kein neues Sammel-Issue
    assert fake.issues[n]["state"] == "closed"
    assert [a.art for a in b.aktionen] == ["sammel_bearbeiten", "schliessen"]


def test_sammel_verdacht_nur_eigene_zeile(tmp_path: Path) -> None:
    fake = FakeGh()
    wert = "ghp_" + "A1b2C3d4" * 5
    gut = _fund(1, schwere="mittel")
    boese = replace(_fund(2, schwere="mittel"), titel=f"x {wert}")
    b = _lauf(fake, [gut, boese], tmp_path)
    (s,) = _sammel(fake)
    assert "Fund 1" in s["body"] and wert not in s["body"]
    assert b.gesperrt == [f"{fund_schluessel(boese)}: GitHub-Token (klassisch)"]
    # nur noch der verdaechtige Fund: Sammel-Issue bleibt offen und unveraendert
    davor = s["body"]
    _lauf(fake, [boese], tmp_path)
    assert s["state"] == "open" and s["body"] == davor
    # der Einzel-Weg bleibt fail-closed
    b3 = _lauf(fake, [replace(_fund(5), text=wert)], tmp_path)
    assert any(g.startswith(fund_schluessel(_fund(5))) for g in b3.gesperrt) and len(_einzel(fake)) == 0


def test_stand_pfad(tmp_path: Path) -> None:
    from selbstbild.doku.stand import repo_schluessel
    p = stand_pfad(tmp_path / "repo", basis=tmp_path / "basis")
    assert p == tmp_path / "basis" / f"issues-{repo_schluessel(tmp_path / 'repo')}.json"


# ---------- Fix-Runde Task 5: konfig als Einzel-Issue (S5-11), Quellen des Inventars (S5-12) ----------

def _konfig(text: str = "Inventar v1") -> Fund:
    return Fund(art="konfig", schwere="mittel", code_repo="vibemind", titel="Konfig-Inventar", text=text,
                belege=("vibemind/.env.example",), anker="konfig-inventar:vibemind")


def test_konfig_ist_einzel_issue_ohne_tagesgrenze(tmp_path: Path) -> None:
    fake = FakeGh()
    b = _lauf(fake, [_konfig()], tmp_path, ziel=_ziel(max_einzel_je_tag=0))
    assert len(_einzel(fake)) == 1 and _sammel(fake) == []
    body = _einzel(fake)[0]["body"]
    assert _marke(_konfig()) in body and "Inventar v1" in body and "<!-- selbstbild:repo=vibemind -->" in body
    assert b.neu == 0 and b.ueber_grenze == []


def test_konfig_gleicher_text_kein_schreiben_geaendert_ein_edit(tmp_path: Path) -> None:
    fake = FakeGh()
    _lauf(fake, [_konfig()], tmp_path)
    vorher = len(fake.aufrufe)
    b = _lauf(fake, [_konfig()], tmp_path)
    neu = fake.aufrufe[vorher:]
    assert {m for m, _ in neu} == {"GET"} and b.unveraendert == 1 and b.aktionen == []
    vorher = len(fake.aufrufe)
    b2 = _lauf(fake, [_konfig("Inventar v2")], tmp_path)
    schreib = [a for a in fake.aufrufe[vorher:] if a[0] != "GET"]
    assert schreib == [("PATCH", f"/repos/{REPO}/issues/1")]
    assert [a.art for a in b2.aktionen] == ["bearbeiten"] and "Inventar v2" in fake.issues[1]["body"]
    assert len(fake.issues) == 1


def test_konfig_oeffentlich_nichts(tmp_path: Path) -> None:
    fake = FakeGh()
    b = _lauf(fake, [_konfig()], tmp_path, ziel=_ziel(oeffentlich=True))
    assert fake.issues == {} and fake.methoden() == {"GET"} and "D6" in b.text()


def test_konfig_schliessen_nur_lesbar_und_geprueft(tmp_path: Path) -> None:
    fake = FakeGh()
    _lauf(fake, [_konfig()], tmp_path)
    _lauf(fake, [], tmp_path, unlesbar=unlesbare_arten([f"{UNLESBAR_PRAEFIX}graph: Graph fehlt"]))
    assert fake.issues[1]["state"] == "open"
    _lauf(fake, [], tmp_path, geprueft=frozenset())
    assert fake.issues[1]["state"] == "open"
    _lauf(fake, [], tmp_path)
    assert fake.issues[1]["state"] == "closed"


def test_konfig_schluesselverdacht_sendet_nicht(tmp_path: Path) -> None:
    fake = FakeGh()
    b = _lauf(fake, [_konfig("x ghp_" + "a1B2c3D4e5" * 4)], tmp_path)
    assert fake.issues == {} and len(b.gesperrt) == 1 and "ghp_" not in b.text()


def test_unlesbare_arten_inventar_quellen() -> None:
    for q in ("graph", "dienste", "submodul", "dateien"):
        assert unlesbare_arten([f"{UNLESBAR_PRAEFIX}{q}: x"]) == frozenset({"konfig"})
    assert unlesbare_arten([f"{UNLESBAR_PRAEFIX}unbekannt: x"]) == frozenset(
        {"doku_widerspruch", "doppelung", "tot", "konfig"})
