"""Stufe 5, Task 6: `python -m selbstbild funde` und der lokale Bericht (gefaelschtes gh, nie Netz).

Dazu der Schutz des LIVE-Einstiegs `selbstbild.__main__` (Doku-Loop `doku tick` alle 5 min): die Argparse-Struktur
aller bestehenden Unterbefehle ist gegen eine vor Task 6 erzeugte Momentaufnahme gepinnt
(`daten/cli_struktur_vor_funde.json`), die Verteilung je Befehl ist geprueft, und ein kaputter funde-Import darf
`doku` nie brechen.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

import selbstbild
import selbstbild.__main__ as cli
from selbstbild.doku.konfig import aus_regeln
from selbstbild.doku.pruefliste import pruefliste_pfad
from selbstbild.funde.github import GitHub
from selbstbild.funde.sync import stand_pfad
from selbstbild.funde.typen import ARTEN

DATEN = Path(__file__).parent / "daten"
TOKEN = "gho_GEHEIMTOKEN0123456789abcdefXYZ"
ZIEL = "Test/ziel"
HEUTE = date(2026, 11, 4)
WERKZEUG = "X: `ev1` hat je Quelle ein anderes Werkzeug: `a` (q1), `b` (q2)"


# ---------- Argparse-Struktur (Momentaufnahme) ----------

class _Gefangen(Exception):
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__("gefangen")
        self.parser = parser


def _parser_fangen(monkeypatch: pytest.MonkeyPatch) -> argparse.ArgumentParser:
    def fangen(self: argparse.ArgumentParser, *_a: Any, **_k: Any) -> Any:
        raise _Gefangen(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fangen)
    try:
        cli.main(["status", "x"])
    except _Gefangen as g:
        return g.parser
    finally:
        monkeypatch.undo()
    raise AssertionError("main hat parse_args nicht aufgerufen")


def struktur(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    """Alles, was Parsen und Verteilen bestimmt: Optionen, Ziele, nargs, Vorgaben, Auswahl, Pflicht, Typ, Hilfe."""
    aus: list[dict[str, Any]] = []
    for a in parser._actions:
        e: dict[str, Any] = {
            "klasse": type(a).__name__, "optionen": list(a.option_strings), "dest": a.dest,
            "nargs": a.nargs, "default": repr(a.default), "required": a.required, "help": a.help,
            "typ": getattr(a.type, "__name__", repr(a.type)) if a.type is not None else None,
            "metavar": a.metavar if isinstance(a.metavar, (str, type(None))) else list(a.metavar),
        }
        if isinstance(a, argparse._SubParsersAction):
            e["unter"] = {name: struktur(p) for name, p in sorted(a.choices.items())}
            e["unter_hilfe"] = {c.dest: c.help for c in a._choices_actions}
        else:
            e["choices"] = list(map(str, a.choices)) if a.choices is not None else None
        aus.append(e)
    return aus


def test_cli_struktur_bestehender_befehle_unveraendert(monkeypatch: pytest.MonkeyPatch) -> None:
    vorher = json.loads((DATEN / "cli_struktur_vor_funde.json").read_text(encoding="utf-8"))
    jetzt = struktur(_parser_fangen(monkeypatch))
    befehle = next(e for e in jetzt if e["klasse"] == "_SubParsersAction")
    assert "funde" in befehle["unter"]
    befehle["unter"].pop("funde")
    befehle["unter_hilfe"].pop("funde")
    assert jetzt == vorher


_BEFEHLE: list[tuple[list[str], str]] = [
    (["vermessen", "w"], "vermessen"),
    (["export", "w", "z"], "export"),
    (["status", "w"], "_status"),
    (["analysieren", "w"], "_analysieren"),
    (["goldsatz", "w", "z"], "_goldsatz"),
    (["vergleichen", "w", "g"], "_vergleichen"),
    (["index", "w"], "_index"),
    (["suchen", "w", "a"], "_suchen"),
    (["pruefen", "w"], "_pruefen"),
    (["skill", "neuer-space", "w", "z"], "_skill"),
    (["suche-messen", "w"], "_suche_messen"),
    (["hooks", "entfernen", "s"], "_hooks"),
    (["funde", "w"], "_funde"),
]


@pytest.mark.parametrize(("argv", "ziel"), _BEFEHLE)
def test_cli_verteilung(argv: list[str], ziel: str, monkeypatch: pytest.MonkeyPatch) -> None:
    gerufen: list[str] = []
    namen = {z for _, z in _BEFEHLE if z != "export"} | {"_doku"}
    for n in namen:
        def rec(*_a: Any, _n: str = n, **_k: Any) -> Any:
            gerufen.append(_n)
            if _n == "vermessen":
                raise _Gefangen(argparse.ArgumentParser())
            return 17

        monkeypatch.setattr(cli, n, rec)
    if ziel == "vermessen":
        with pytest.raises(_Gefangen):
            cli.main(argv)
    elif ziel == "export":
        monkeypatch.setattr(cli, "stand_finden", lambda w: gerufen.append("export") or None)
        monkeypatch.setattr(cli, "_nicht_vermessen", lambda w: 17)
        assert cli.main(argv) == 17
    else:
        assert cli.main(argv) == 17
    assert gerufen == [ziel]


@pytest.mark.parametrize("argv", [
    ["doku", "tick", "C:/w"], ["-v", "doku", "tick", "C:/w"], ["doku", "tick", "C:/w", "--konfig", "k.yaml"],
    ["doku", "--help"], ["doku"],
])
def test_doku_tick_unveraendert_und_ohne_funde(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """`doku ...` geht vor jedem argparse unveraendert an den Doku-Loop -- auch wenn funde nicht importierbar ist."""
    import selbstbild.doku.__main__ as doku_main

    erhalten: list[list[str]] = []
    monkeypatch.setattr(doku_main, "main", lambda rest: erhalten.append(list(rest)) or 0)
    for mod in [m for m in sys.modules if m == "selbstbild.funde" or m.startswith("selbstbild.funde.")]:
        monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.setitem(sys.modules, "selbstbild.funde", None)  # jeder funde-Import scheitert
    monkeypatch.setitem(sys.modules, "selbstbild.funde.bericht", None)
    assert cli.main(argv) == 0
    assert erhalten == [argv[argv.index("doku") + 1:]]


def test_funde_import_kaputt_bricht_andere_befehle_nicht(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "selbstbild.funde", None)
    monkeypatch.setitem(sys.modules, "selbstbild.funde.bericht", None)
    monkeypatch.setattr(cli, "_status", lambda w: 23)
    assert cli.main(["status", "w"]) == 23
    with pytest.raises(ImportError):
        cli.main(["funde", "w"])


def test_einstieg_importiert_funde_nicht() -> None:
    code = ("import sys, selbstbild.__main__; "
            "print(sorted(m for m in sys.modules if m.startswith('selbstbild.funde')))")
    erg = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert erg.stdout.strip() == "[]"


def test_nur_auswahl_wie_arten() -> None:
    assert tuple(cli.FUND_ARTEN) == ARTEN


# ---------- gefaelschtes gh ----------

class FakeGh:
    """`gh auth token` und die REST-Pfade, die der Abgleich braucht. Nie Netz."""

    def __init__(self, tag: date = HEUTE) -> None:
        self.issues: dict[int, dict[str, Any]] = {}
        self.labels: set[str] = set()
        self.aufrufe: list[tuple[str, str]] = []
        self.tag = tag
        self.naechste = 1
        self.privat = True
        self.fehler_bei: tuple[str, str] | None = None  # (Methode, Pfad-Ende) -> HTTP 502
        self.ausnahme_bei_anlegen: int | None = None  # n-tes POST issues -> RuntimeError (interner Fehler)
        self.anlegen_versuche = 0
        self.kommentare: list[tuple[int, str]] = []

    def issue(self, titel: str, body: str, *, state: str = "open") -> int:
        n = self.naechste
        self.naechste += 1
        self.issues[n] = {"number": n, "title": titel, "body": body, "state": state, "state_reason": None,
                          "labels": [{"name": "selbstbild"}], "created_at": f"{self.tag.isoformat()}T12:00:00Z"}
        return n

    def schreibend(self) -> list[tuple[str, str]]:
        return [a for a in self.aufrufe if a[0] != "GET"]

    def __call__(self, args: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        argv = list(args)
        assert kw.get("timeout")
        if argv[1:3] == ["auth", "token"]:
            return subprocess.CompletedProcess(argv, 0, (TOKEN + "\n").encode(), b"")
        assert argv[1] == "api" and kw["env"]["GH_TOKEN"] == TOKEN
        methode = argv[argv.index("--method") + 1]
        pfad = next(a for a in argv[2:] if a.startswith("/"))
        daten = json.loads(kw["input"].decode("utf-8")) if kw.get("input") is not None else None
        self.aufrufe.append((methode, pfad))
        teile = urlsplit(pfad)
        rest = teile.path.split("/")[1:]
        assert rest[:3] == ["repos", *ZIEL.split("/")]
        rest = rest[3:]

        def ok(obj: Any) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(obj).encode("utf-8"), b"")

        if self.fehler_bei and methode == self.fehler_bei[0] and teile.path.endswith(self.fehler_bei[1]):
            return subprocess.CompletedProcess(argv, 1, b"", b"gh: Server Error (HTTP 502) " + TOKEN.encode())
        if methode == "GET" and rest == []:
            return ok({"full_name": ZIEL, "private": self.privat})
        if methode == "GET" and rest == ["issues"]:
            seite = int(parse_qs(teile.query)["page"][0])
            alle = sorted(self.issues.values(), key=lambda i: -i["number"])
            return ok(alle[(seite - 1) * 100: seite * 100])
        if rest == ["labels", "selbstbild"] and methode == "GET":
            if "selbstbild" in self.labels:
                return ok({"name": "selbstbild"})
            return subprocess.CompletedProcess(argv, 1, b"", b"gh: Not Found (HTTP 404)")
        if rest == ["labels"] and methode == "POST":
            self.labels.add(daten["name"])
            return ok({"name": daten["name"]})
        if rest == ["issues"] and methode == "POST":
            self.anlegen_versuche += 1
            if self.anlegen_versuche == self.ausnahme_bei_anlegen:
                raise RuntimeError(f"kaputt {TOKEN}")
            n = self.issue(daten["title"], daten["body"])
            return ok(self.issues[n])
        if len(rest) == 2 and rest[0] == "issues" and methode == "PATCH":
            i = self.issues[int(rest[1])]
            for k in ("body", "state", "state_reason"):
                if k in daten:
                    i[k] = daten[k]
            return ok({"number": i["number"]})
        if len(rest) == 3 and rest[2] == "comments" and methode == "POST":
            self.kommentare.append((int(rest[1]), daten["body"]))
            return ok({"id": 1})
        raise AssertionError(f"unerwarteter Aufruf {methode} {pfad}")


# ---------- Fixture-Repo ----------

def _regeln(marker: Path, issues: str) -> str:
    return (
        "name: aussen\ndoku:\n  spaces: {}\n  ausschluss_namen: []\n  ausschluss_pfade: []\n  ketten_md: ''\n"
        "  compose_live: []\n  sql_wurzeln: ['.']\n  dienste_yaml: ''\n  waechter_zustandsdatei: ''\n"
        f"  waechter_zuordnung: {{}}\n  tick_minuten: 5\n  marker: '{marker.as_posix()}'\n  agent: ''\n"
        "  judge_modell: sonnet\n  claude_bin: claude\n  rowboat_wissen: w\n  rowboat_notizen: n\n"
        "  rowboat_urteile: u\n" + issues)


ISSUES_MIT_KONTO = f"issues:\n  ziel_standard: {{repo: {ZIEL}, konto: tester, oeffentlich: false}}\n"


def _item(art: str, datei: str | None, aid: str | None, text: str = "x") -> dict[str, Any]:
    return {"art": art, "text": text, "datei": datei, "zeile": 3, "aussage_id": aid, "lauf": "l1"}


def _pruefliste(wurzel: Path, struktur_: list[dict[str, Any]]) -> None:
    pfad = pruefliste_pfad(aus_regeln(wurzel))
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps({"offen": [], "struktur": struktur_}), encoding="utf-8")


ZWEI_DOKU = [
    _item("ereignis_quellen_abweichend", "a/reg.yaml", "ereignis:x:ev1", WERKZEUG),  # hoch
    _item("ereignis_nur_doku", "docs/x.md", "docs/x.md#a.b"),  # mittel
]


class Umgebung:
    def __init__(self, git_repo: Any, tmp_path: Path, issues: str = ISSUES_MIT_KONTO,
                 marker: Path | None = None) -> None:
        self.repo = git_repo
        self.wurzel: Path = git_repo.wurzel
        self.tmp = tmp_path
        self.bericht = tmp_path / "bericht"
        self.basis = tmp_path / "issues-stand"
        self.fake = FakeGh()
        git_repo.datei("selbstbild.yaml", _regeln(marker or tmp_path / "doku" / "marker", issues))
        git_repo.datei("app/.env.example", "APP_MODUS=\n")
        git_repo.commit("init")

    def optionen(self, **extra: Any) -> dict[str, Any]:
        o: dict[str, Any] = {"gh_fabrik": lambda konto: GitHub(konto, laufzeit=self.fake),
                             "bericht_ordner": self.bericht, "stand_basis": self.basis, "heute": HEUTE,
                             "schreib_abstand_s": 0}
        o.update(extra)
        return o

    def lauf(self, capsys: pytest.CaptureFixture[str], *argv: str, **extra: Any) -> tuple[int, dict[str, Any]]:
        code = cli.main(["funde", str(self.wurzel), *argv], funde_optionen=self.optionen(**extra))
        roh = capsys.readouterr()
        self.out, self.err = roh.out, roh.err
        return code, json.loads(roh.out)

    def text(self) -> str:
        return (self.bericht / "funde-bericht.md").read_text(encoding="utf-8")


@pytest.fixture
def umg(git_repo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Umgebung:
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "heim"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lokal"))
    return Umgebung(git_repo, tmp_path)


def _ziel(j: dict[str, Any]) -> dict[str, Any]:
    assert len(j["ziele"]) == 1
    return j["ziele"][0]


# ---------- Brief ----------

def test_cli_trocken(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    _pruefliste(umg.wurzel, ZWEI_DOKU)
    code, j = umg.lauf(capsys, "--goldsatz", str(umg.tmp / "fehlt.yaml"))
    assert code == 0
    assert j["funde"]["doku_widerspruch"] == {"hoch": 1, "mittel": 1, "niedrig": 0}
    assert j["funde"]["konfig"] == {"hoch": 0, "mittel": 1, "niedrig": 0}
    assert j["tor_tot"]["text"] == "toter Code: gesperrt (0/0 bestätigt)" and j["tor_tot"]["offen"] is False
    z = _ziel(j)
    assert z["ziel"] == ZIEL and z["modus"] == "trocken" and z["fehler"] is None
    # hoch -> Einzel, mittel -> Sammel, Inventar -> konfig-Einzel (S5-11)
    assert z["aktionen"] == {"anlegen": 3, "aktualisieren": 0, "schliessen": 0, "unterdrueckt": 0}
    assert z["aktionen_je_art"] == {"anlegen": 2, "sammel_anlegen": 1}
    assert z["neu"] == 1 and z["ueber_grenze"] == 0 and z["gesperrt"] == 0
    # S5-16: trocken mit Konto liest GitHub, schreibt aber nichts; der lokale Stand bleibt unberuehrt
    assert umg.fake.aufrufe and umg.fake.schreibend() == []
    assert not stand_pfad(umg.wurzel, umg.basis).exists()
    assert j["bericht"] == str(umg.bericht / "funde-bericht.md")
    t = umg.text()
    assert "toter Code: gesperrt (0/0 bestätigt)" in t
    assert f"## Issues → {ZIEL} (trocken" in t
    assert "### Inventar" in t and "APP_MODUS" in t  # volle Inventar-Tabelle nur lokal
    assert "Goldsatz fehlt" in t and any("Goldsatz fehlt" in n for n in j["notizen"])
    assert list(j) == sorted(j)


def test_cli_offline_ruft_kein_gh(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    _pruefliste(umg.wurzel, ZWEI_DOKU)

    def nie(konto: str) -> GitHub:
        raise AssertionError("gh im Offline-Lauf")

    code, j = umg.lauf(capsys, "--offline", gh_fabrik=nie)
    assert code == 0 and _ziel(j)["modus"] == "offline"
    assert _ziel(j)["aktionen"]["anlegen"] == 3
    # Fix-Runde 1 (7): Ueberschrift nennt offline; stdout ist reines ASCII (PowerShell)
    assert f"## Issues → {ZIEL} (offline (nichts gesendet))" in umg.text()
    assert umg.out.isascii() and "best\\u00e4tigt" in umg.out


@pytest.mark.parametrize("issues", [
    "",  # weder issues: noch issue_ziel
    "issue_ziel: {repo: Test/ziel}\n",  # S5-1: Fallback ohne Konto
    "issues:\n  ziel_standard: {repo: Test/ziel, oeffentlich: false}\n",  # Ziel ohne Konto
])
def test_cli_senden_ohne_ziel_exit2(issues: str, git_repo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "heim"))
    u = Umgebung(git_repo, tmp_path, issues)

    def nie(konto: str) -> GitHub:
        raise AssertionError("gh trotz Exit 2")

    code = cli.main(["funde", str(u.wurzel), "--senden"], funde_optionen=u.optionen(gh_fabrik=nie))
    aus = capsys.readouterr()
    assert code == 2 and aus.out == "" and "issues" in aus.err
    # ohne --senden ist dieselbe Konfiguration ein normaler Bericht
    assert cli.main(["funde", str(u.wurzel)], funde_optionen=u.optionen(gh_fabrik=nie)) == 0
    j = json.loads(capsys.readouterr().out)
    assert all(z["modus"] == "bericht" for z in j["ziele"])


def test_bericht_ohne_werte(umg: Umgebung, capsys: pytest.CaptureFixture[str],
                            monkeypatch: pytest.MonkeyPatch) -> None:
    ghp = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    umg.repo.datei(".env", "DB_PASSWORT=wert-aus-env-111\n")
    umg.repo.datei(".env.example", "BEISPIEL_NAME=beispielwert222\n")
    umg.repo.datei("docker-compose.yml", "services:\n  a:\n    environment:\n      COMPOSE_NAME: composewert333\n")
    umg.repo.commit("env")
    monkeypatch.setenv("APP_MODUS", "umgebungswert444")
    _pruefliste(umg.wurzel, [
        _item("ereignis_quellen_abweichend", "a/reg.yaml", "ereignis:x:ev9", f"{WERKZEUG} {ghp}"),
        _item("ereignis_nur_doku", f"docs/{ghp}.md", "docs/y.md#a"),
    ])
    code, j = umg.lauf(capsys)
    assert code == 0
    roh = json.dumps(j, ensure_ascii=False) + umg.text()
    for wert in (TOKEN, ghp, "wert-aus-env-111", "beispielwert222", "composewert333", "umgebungswert444"):
        assert wert not in roh
    assert "BEISPIEL_NAME" in umg.text() and "COMPOSE_NAME" in umg.text()  # Namen ja
    assert j["schluessel_treffer"] and all(":" not in m and "ghp_A" not in m for m in j["schluessel_treffer"])
    assert "GitHub-Token (klassisch)" in j["schluessel_treffer"]
    assert _ziel(j)["gesperrt"] >= 1
    assert umg.fake.schreibend() == []


def test_notizen_im_json_geschwaerzt(git_repo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    """Auch Hinweise (hier: der Pfad der fehlenden Pruefliste) zeigen nie einen schluesselartigen Text."""
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "heim"))
    ghp = "ghp_" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3i2"
    u = Umgebung(git_repo, tmp_path, marker=tmp_path / ghp / "marker")  # keine Pruefliste -> Notiz mit Pfad
    code, j = u.lauf(capsys, "--offline")
    assert code == 0
    assert any("doku-pruefliste fehlt" in n for n in j["notizen"])
    assert ghp not in json.dumps(j, ensure_ascii=False) and ghp not in u.text()
    assert "GitHub-Token (klassisch)" in j["schluessel_treffer"]


# ---------- S5-17 Exit-Codes ----------

def test_exit1_bei_schreibfehler_bericht_trotzdem(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    _pruefliste(umg.wurzel, ZWEI_DOKU)
    umg.fake.fehler_bei = ("POST", "/issues")
    code, j = umg.lauf(capsys, "--senden")
    assert code == 1 and _ziel(j)["fehler"] == "http_502" and _ziel(j)["modus"] == "senden"
    assert "Abgebrochen" in umg.text()
    # Fix-Runde 1 (5): der Token (er steht im gefaelschten gh-stderr) erscheint nirgends
    assert TOKEN not in umg.out and TOKEN not in umg.err and TOKEN not in umg.text()


def test_exit1_bei_lesefehler(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    _pruefliste(umg.wurzel, ZWEI_DOKU)
    umg.fake.fehler_bei = ("GET", "/issues")
    code, j = umg.lauf(capsys)  # auch trocken: Bestand nicht lesbar ist ein Fehler
    assert code == 1 and _ziel(j)["fehler"] == "http_502"
    assert (umg.bericht / "funde-bericht.md").is_file()
    assert TOKEN not in umg.out and TOKEN not in umg.err and TOKEN not in umg.text()
    assert any("nichts gesendet (trocken)" in n for n in _ziel(j)["notizen"])


def test_exit1_bei_gesperrtem_ziel(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    _pruefliste(umg.wurzel, ZWEI_DOKU)
    p = stand_pfad(umg.wurzel, umg.basis)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"version": 1, "zaehler": {}, "marken": {}, "sperren": {ZIEL: "label_fehlt"}}),
                 encoding="utf-8")
    code, j = umg.lauf(capsys, "--senden")
    assert code == 1 and _ziel(j)["fehler"] == "ziel_gesperrt"
    assert umg.fake.schreibend() == []


def test_exit1_ohne_commit(git_repo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "heim"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    u = Umgebung(git_repo, tmp_path)
    kein_git = tmp_path / "kein-git"
    kein_git.mkdir()
    (kein_git / "selbstbild.yaml").write_text(_regeln(tmp_path / "doku2" / "marker", ISSUES_MIT_KONTO),
                                             encoding="utf-8")
    code = cli.main(["funde", str(kein_git), "--senden"], funde_optionen=u.optionen())
    j = json.loads(capsys.readouterr().out)
    assert code == 1 and j["commit"] is None and _ziel(j)["fehler"] == "commit_unbekannt"
    assert u.fake.aufrufe == []  # nichts gelesen, nichts gesendet
    assert (u.bericht / "funde-bericht.md").is_file()


@pytest.mark.parametrize("argv", [["--nur", "unbekannt"], ["--senden", "--offline"], ["--nur"]])
def test_exit2_argparse(argv: list[str], umg: Umgebung) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["funde", str(umg.wurzel), *argv], funde_optionen=umg.optionen())
    assert e.value.code == 2


# ---------- Stufe 6a: Sperre im Sendeweg ----------

def test_senden_bei_belegter_sperre_exit_3_ohne_gh_aufruf(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    from selbstbild.funde.bericht import SENDE_SPERRE
    from selbstbild.sperre import sperre

    _pruefliste(umg.wurzel, ZWEI_DOKU)

    def nie(konto: str) -> GitHub:
        raise AssertionError("gh bei belegter Sperre")

    with sperre(umg.bericht / SENDE_SPERRE):
        code = cli.main(["funde", str(umg.wurzel), "--senden"], funde_optionen=umg.optionen(gh_fabrik=nie))
    aus = capsys.readouterr()
    assert code == 3 and aus.out == ""
    assert aus.err == "funde --senden: ein anderer Sendelauf läuft (Sperre)\n"
    assert umg.fake.aufrufe == []
    # nach Freigabe laeuft derselbe Aufruf, und die Sperre bleibt nicht liegen
    code, _ = umg.lauf(capsys, "--senden")
    assert code == 0 and not (umg.bericht / SENDE_SPERRE).exists()


def test_trockenlauf_ignoriert_sperre(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    from selbstbild.funde.bericht import SENDE_SPERRE
    from selbstbild.sperre import sperre

    _pruefliste(umg.wurzel, ZWEI_DOKU)
    with sperre(umg.bericht / SENDE_SPERRE):
        code, j = umg.lauf(capsys)
        assert code == 0 and _ziel(j)["modus"] == "trocken"
        code, j = umg.lauf(capsys, "--offline")
        assert code == 0 and _ziel(j)["modus"] == "offline"


def test_senden_gibt_sperre_auch_bei_fehler_frei(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    from selbstbild.funde.bericht import SENDE_SPERRE

    _pruefliste(umg.wurzel, ZWEI_DOKU)
    umg.fake.fehler_bei = ("POST", "/issues")
    code, _ = umg.lauf(capsys, "--senden")
    assert code == 1 and not (umg.bericht / SENDE_SPERRE).exists()


# ---------- S5-19 --nur ----------

def _einzel_issue(fake: FakeGh, art: str, hash16: str, code_repo: str) -> int:
    return fake.issue(f"[selbstbild] {art} ({code_repo})",
                      f"<!-- selbstbild:schluessel={art}:{hash16} -->\n<!-- selbstbild:repo={code_repo} -->\n\nx\n")


def test_nur_schliesst_nicht_gewaehlte_arten_nie(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    _pruefliste(umg.wurzel, [])  # Doku lesbar, keine Doku-Funde mehr
    alt_doku = _einzel_issue(umg.fake, "doku_widerspruch", "0" * 16, "aussen")
    alt_tot = _einzel_issue(umg.fake, "tot", "1" * 16, "aussen")
    code, j = umg.lauf(capsys, "--senden", "--nur", "doppelung", "--goldsatz", str(umg.tmp / "fehlt.yaml"))
    assert code == 0 and j["nur"] == ["doppelung"]
    assert set(j["funde"]) == {"doppelung"}
    assert umg.fake.issues[alt_doku]["state"] == "open" and umg.fake.issues[alt_tot]["state"] == "open"
    assert ("POST", f"/repos/{ZIEL}/issues") not in umg.fake.aufrufe  # auch kein konfig-Issue
    # Gegenprobe: ohne --nur schliesst derselbe Lauf das verschwundene Doku-Issue
    code, j = umg.lauf(capsys, "--senden")
    assert code == 0 and umg.fake.issues[alt_doku]["state"] == "closed"
    assert umg.fake.issues[alt_tot]["state"] == "open"  # Tor gesperrt -> tot unlesbar


# ---------- Senden: zweiter Lauf legt nichts an (Abnahme Task 7) ----------

def test_senden_zweiter_lauf_ohne_doppel(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    _pruefliste(umg.wurzel, ZWEI_DOKU)
    code, j = umg.lauf(capsys, "--senden")
    assert code == 0
    z = _ziel(j)
    assert z["modus"] == "senden" and z["aktionen"]["anlegen"] == 3 and z["neu"] == 1
    assert len(umg.fake.issues) == 3
    assert stand_pfad(umg.wurzel, umg.basis).is_file()
    vorher = len(umg.fake.aufrufe)
    code, j = umg.lauf(capsys, "--senden")
    assert code == 0
    z = _ziel(j)
    assert z["aktionen"] == {"anlegen": 0, "aktualisieren": 0, "schliessen": 0, "unterdrueckt": 0}
    assert z["neu"] == 0 and len(umg.fake.issues) == 3
    assert all(m == "GET" for m, _ in umg.fake.aufrufe[vorher:])


# ---------- S5-10 geprueft ----------

def test_geprueft_ohne_submodul_mit_fehlendem_pin(git_repo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    from selbstbild.funde.bericht import geprueft_ermitteln
    from selbstbild.funde.sammeln import geprueft_repos
    from selbstbild.vermessen import vermessen

    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "heim"))
    repo_klasse = type(git_repo)
    quellen = {}
    for name in ("sub-a", "sub-b"):
        q = repo_klasse(tmp_path / f"q-{name}", git_repo._env)
        q.datei("m.py", "def f():\n    return 1\n")
        q.commit("q")
        quellen[name] = q
    u = Umgebung(git_repo, tmp_path)
    for name, q in quellen.items():
        git_repo.submodul(name, q)
    git_repo.commit("submodule")
    git_repo.git("update-index", "--cacheinfo", f"160000,{'d' * 40},sub-b")  # Pin, den sub-b nicht hat
    git_repo.git("commit", "-q", "-m", "pin fehlt")
    bericht = vermessen(u.wurzel)
    assert bericht.fehlende_submodule == ("aussen/sub-b",)

    k = aus_regeln(u.wurzel)
    assert "aussen/sub-b" in geprueft_repos(u.wurzel, doku_konfig=k)  # Checkout da -> ohne Abzug "geprueft"
    gep, notizen = geprueft_ermitteln(u.wurzel, doku_konfig=k)
    assert gep == frozenset({"aussen", "aussen/sub-a"}) and notizen == []
    # ohne gespeicherten Stand: nur das aeussere Repo, mit Notiz
    gep, notizen = geprueft_ermitteln(u.wurzel, doku_konfig=k, stand_unter=tmp_path / "leer")
    assert gep == frozenset({"aussen"}) and len(notizen) == 1 and "Submodule" in notizen[0]

    # im Lauf: das verschwundene Issue von sub-a wird geschlossen, das von sub-b bleibt offen
    _pruefliste(u.wurzel, [])
    a = _einzel_issue(u.fake, "doku_widerspruch", "a" * 16, "aussen/sub-a")
    b = _einzel_issue(u.fake, "doku_widerspruch", "b" * 16, "aussen/sub-b")
    code, j = u.lauf(capsys, "--senden")
    assert code == 0
    assert u.fake.issues[a]["state"] == "closed" and u.fake.issues[b]["state"] == "open"
    assert any("aussen/sub-b" in n for n in j["ziele"][0]["notizen"])


# ---------- S5-15 Goldsatz ----------

def test_goldsatz_standard_und_offenes_tor(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    from selbstbild.funde.bericht import goldsatz_standard

    p = goldsatz_standard("vibemind")
    assert p == Path(selbstbild.__file__).resolve().parents[2] / "docs" / "selbstbild" / "goldsatz-vibemind.yaml"
    assert p.is_file()
    code, j = umg.lauf(capsys, "--offline")  # Vorgabe goldsatz-aussen.yaml gibt es nicht -> Tor zu, kein Fehler
    assert code == 0 and j["tor_tot"]["offen"] is False
    assert any("goldsatz-aussen.yaml" in n for n in j["notizen"])
    gold = umg.tmp / "gold.yaml"
    gold.write_text("tot:\n" + "".join(f"- id: t{i}\n  urteil: ja\n" for i in range(30)), encoding="utf-8")
    code, j = umg.lauf(capsys, "--offline", "--goldsatz", str(gold))
    assert code == 0
    assert j["tor_tot"] == {"offen": True, "ja": 30, "beurteilt": 30, "text": "toter Code: offen (30/30 bestätigt)"}
    assert "toter Code: offen (30/30 bestätigt)" in umg.text()
    kaputt = umg.tmp / "kaputt.yaml"
    kaputt.write_text("tot: [unvollstaendig\n", encoding="utf-8")
    code, j = umg.lauf(capsys, "--offline", "--goldsatz", str(kaputt))
    assert code == 0 and j["tor_tot"]["offen"] is False and any("Goldsatz" in n for n in j["notizen"])


# ---------- Fix-Runde 1 ----------

def test_regeldatei_kaputtes_yaml_exit2(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    """(1) yaml.YAMLError aus regeln_laden: Exit 2, eine Zeile, kein YAML-Ausschnitt."""
    (umg.wurzel / "selbstbild.yaml").write_text("name: aussen\ngeheim: [wert-im-yaml-555\n  x: {\n", encoding="utf-8")
    code = cli.main(["funde", str(umg.wurzel), "--offline"], funde_optionen=umg.optionen())
    aus = capsys.readouterr()
    assert code == 2 and aus.out == ""
    assert aus.err.count("\n") == 1 and "wert-im-yaml-555" not in aus.err and "Error" in aus.err


ZWEI_ZIELE = (f"issues:\n  ziel_standard: {{repo: {ZIEL}, konto: tester, oeffentlich: false}}\n"
              "  ziele:\n    aussen/sub: {repo: Test/zwei, oeffentlich: false}\n")


def test_interner_fehler_je_ziel_andere_laufen(git_repo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    """(2) Eine Ausnahme in einem Ziel: `intern:<Typ>`, Exit 1, das andere Ziel laeuft trotzdem."""
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "heim"))
    u = Umgebung(git_repo, tmp_path, ZWEI_ZIELE)
    _pruefliste(u.wurzel, ZWEI_DOKU)

    def kaputt(konto: str) -> GitHub:
        raise RuntimeError(f"fabrik {TOKEN}")

    code, j = u.lauf(capsys, gh_fabrik=kaputt)
    assert code == 1
    erst, zweit = j["ziele"]
    assert erst["ziel"] == ZIEL and erst["fehler"] == "intern:RuntimeError"
    assert any("nichts gesendet (trocken)" in n for n in erst["notizen"])
    assert zweit["ziel"] == "Test/zwei" and zweit["modus"] == "bericht" and zweit["fehler"] is None
    assert "## Issues → Test/zwei" in u.text()
    assert TOKEN not in u.out and TOKEN not in u.err and TOKEN not in u.text()


def test_abbruch_nach_schreiben_sagt_nicht_nichts_gesendet(umg: Umgebung,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    """(3) Wirft der Abgleich nach einem ausgefuehrten Schreibaufruf: ehrliche Notiz, Teil-Aktionen bleiben."""
    _pruefliste(umg.wurzel, ZWEI_DOKU)
    umg.fake.ausnahme_bei_anlegen = 2
    code, j = umg.lauf(capsys, "--senden")
    z = _ziel(j)
    assert code == 1 and z["fehler"] == "intern:RuntimeError"
    assert z["aktionen"]["anlegen"] == 1 and len(umg.fake.issues) == 1  # das erste ist wirklich angelegt
    assert any("nach 1 Schreibaufruf(en), Stand unklar – Bestand per Marke maßgeblich" in n for n in z["notizen"])
    assert not any("nichts gesendet" in n for n in z["notizen"])
    assert "nichts gesendet" not in umg.text().split(f"## Issues → {ZIEL}")[1].split("## Hinweise")[0]
    assert TOKEN not in umg.out and TOKEN not in umg.err and TOKEN not in umg.text()


def test_bericht_nicht_schreibbar_json_trotzdem(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    """(2) Bericht nicht schreibbar: JSON kommt trotzdem, mit `bericht_fehler`, Exit 1."""
    _pruefliste(umg.wurzel, ZWEI_DOKU)
    datei = umg.tmp / "ist-eine-datei"
    datei.write_text("x", encoding="utf-8")
    code, j = umg.lauf(capsys, "--offline", bericht_ordner=datei)
    assert code == 1 and j["bericht"] is None and j["bericht_fehler"]
    assert _ziel(j)["aktionen"]["anlegen"] == 3


def test_stand_nicht_gespeichert_nach_schreiben_exit1(umg: Umgebung, capsys: pytest.CaptureFixture[str],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """(4) Lokaler Stand nach ausgefuehrten Schreibaufrufen nicht speicherbar: Exit 1; ohne Schreiben: Exit 0."""
    from selbstbild.funde.sync import IssueStand

    _pruefliste(umg.wurzel, ZWEI_DOKU)

    def nie_speichern(self: IssueStand) -> None:
        raise PermissionError("gesperrt")

    monkeypatch.setattr(IssueStand, "speichern", nie_speichern)
    code, j = umg.lauf(capsys, "--senden")
    assert code == 1 and _ziel(j)["fehler"] == "stand_nicht_gespeichert:PermissionError"
    assert len(umg.fake.issues) == 3  # gesendet wurde
    # zweiter Lauf schreibt nichts (alles per Marke gefunden): Speicherfehler ohne Schreiben ist kein Fehler
    code, j = umg.lauf(capsys, "--senden")
    assert code == 0 and _ziel(j)["fehler"] is None and len(umg.fake.issues) == 3


def test_analyse_stand_ungleich_head_nur_notiz(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    """(6) Vermessener Commit != HEAD: nur eine Notiz in JSON und Bericht."""
    from selbstbild.vermessen import vermessen

    _pruefliste(umg.wurzel, ZWEI_DOKU)
    gemessen = vermessen(umg.wurzel).commit
    code, j = umg.lauf(capsys, "--offline")
    assert code == 0 and not any(n.startswith("analyse-stand") for n in j["notizen"])
    vorher = j["funde"]
    umg.repo.datei("neu.txt", "x\n")
    head = umg.repo.commit("weiter")
    code, j = umg.lauf(capsys, "--offline")
    erwartet = f"analyse-stand {gemessen[:7]} != HEAD {head[:7]}"
    assert code == 0 and erwartet in j["notizen"] and erwartet in umg.text()
    assert j["funde"] == vorher


# ---------- finale Review-Fixes Stufe 5 ----------

def test_schliess_kommentar_nennt_analyse_commit(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    """Fix 3: der Schliess-Kommentar nennt den vermessenen Commit, nicht HEAD."""
    from selbstbild.vermessen import vermessen

    _pruefliste(umg.wurzel, [])
    gemessen = vermessen(umg.wurzel).commit
    umg.repo.datei("neu.txt", "x\n")
    head = umg.repo.commit("weiter")
    assert gemessen[:7] != head[:7]
    alt = _einzel_issue(umg.fake, "doku_widerspruch", "0" * 16, "aussen")
    code, j = umg.lauf(capsys, "--senden")
    assert code == 0 and umg.fake.issues[alt]["state"] == "closed"
    assert (alt, f"Fund im Code verschwunden (Commit `{gemessen[:7]}`)") in umg.fake.kommentare
    assert not any(n.startswith("quelle-unlesbar:analyse-stand") for n in j["notizen"])  # Vorfahr: lesbar


def test_schliess_kommentar_ohne_analyse_nennt_head(umg: Umgebung, capsys: pytest.CaptureFixture[str]) -> None:
    _pruefliste(umg.wurzel, [])
    head = umg.repo.git("rev-parse", "HEAD")
    alt = _einzel_issue(umg.fake, "doku_widerspruch", "0" * 16, "aussen")
    code, j = umg.lauf(capsys, "--senden")
    assert code == 0
    assert (alt, f"Fund im Code verschwunden (Commit `{head[:7]}`)") in umg.fake.kommentare
    assert any("Analyse-Commit unbekannt" in n and "HEAD" in n for n in j["notizen"])


def test_analyse_commit_kein_vorfahr_doppelung_tot_unlesbar(umg: Umgebung,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    """Fix 3: Analyse-Commit kein Vorfahr von HEAD -> doppelung und tot unlesbar, nichts davon geschlossen."""
    from selbstbild.funde.sync import unlesbare_arten
    from selbstbild.vermessen import vermessen

    _pruefliste(umg.wurzel, [])
    gemessen = vermessen(umg.wurzel).commit
    umg.repo.git("checkout", "-q", "--orphan", "anders")
    head = umg.repo.commit("neue Geschichte")
    alt_doppel = _einzel_issue(umg.fake, "doppelung", "2" * 16, "aussen")
    alt_doku = _einzel_issue(umg.fake, "doku_widerspruch", "0" * 16, "aussen")
    code, j = umg.lauf(capsys, "--senden")
    assert code == 0
    notiz = [n for n in j["notizen"] if n.startswith("quelle-unlesbar:analyse-stand")]
    assert len(notiz) == 1 and gemessen[:7] in notiz[0] and head[:7] in notiz[0]
    assert unlesbare_arten(notiz) == frozenset({"doppelung", "tot"})
    assert umg.fake.issues[alt_doppel]["state"] == "open"
    assert umg.fake.issues[alt_doku]["state"] == "closed"  # andere Arten schliessen weiter
    assert notiz[0] in umg.text()


def test_analyse_stand_pruefen(git_repo: Any) -> None:
    from selbstbild.funde.bericht import analyse_stand_pruefen

    git_repo.datei("a.txt", "a\n")
    a = git_repo.commit("a")
    git_repo.datei("b.txt", "b\n")
    b = git_repo.commit("b")
    assert analyse_stand_pruefen(git_repo.wurzel, a, b) == []
    assert analyse_stand_pruefen(git_repo.wurzel, b, b) == []
    (n,) = analyse_stand_pruefen(git_repo.wurzel, b, a)  # b ist kein Vorfahr von a
    assert n.startswith("quelle-unlesbar:analyse-stand:") and b[:7] in n
    (n,) = analyse_stand_pruefen(git_repo.wurzel, "f" * 40, b)  # unbekannter Commit: fail-closed
    assert n.startswith("quelle-unlesbar:analyse-stand:")


def test_analyse_stand_pruefen_mit_zeitgrenze(git_repo: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    from selbstbild.funde import bericht as modul
    gesehen: list[Any] = []

    def lauf(args: list[str], **kw: Any) -> Any:
        gesehen.append(kw.get("timeout"))
        raise sp.TimeoutExpired(args, 30)

    monkeypatch.setattr(modul.subprocess, "run", lauf)
    (n,) = modul.analyse_stand_pruefen(git_repo.wurzel, "a" * 40, "b" * 40)
    assert gesehen == [modul.GIT_TIMEOUT_S] and n.startswith("quelle-unlesbar:analyse-stand:")
