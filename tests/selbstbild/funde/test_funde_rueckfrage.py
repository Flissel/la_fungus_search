"""Stufe 6a, Task 3: Rueckfragen aus dem Goldsatz als GitHub-Issues (gefaelschtes gh, echtes git im tmp_path)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from selbstbild.funde.github import AUTO_MARKE, GitHub
from selbstbild.funde.rueckfrage import (
    ANTWORTEN,
    MAX_NEU,
    RUECKFRAGE_LABEL,
    code_ausschnitt,
    marke,
    rueckfragen_abgleichen,
)
from test_funde_sync import REPO, FakeGh

COMMIT = "abc1234def5678"
GOLD = "docs/selbstbild/goldsatz.yaml"
MELDUNG = "docs(selbstbild): Rückfrage-Antworten übernehmen"


def _sym(i: int) -> str:
    return f"symbol:vibemind:pkg/a.py#f{i}"


ALLE_TOT = frozenset(_sym(i) for i in range(20))


def _gold_text(eintraege: list[tuple[str, str]]) -> str:
    zeilen = ["commit: abc", "tot:"]
    for sym, urteil in eintraege:
        zeilen += ["- gruende:", "  - alt_90", "  - unerreichbar", f"  symbol: {sym}", f"  urteil: {urteil}"]
    return "\n".join(zeilen) + "\n"


@pytest.fixture
def gold(git_repo: Any) -> Any:
    def bauen(eintraege: list[tuple[str, str]]) -> Path:
        git_repo.datei(GOLD, _gold_text(eintraege))
        git_repo.datei("anderes.txt", "alt\n")
        git_repo.commit("init")
        return git_repo.wurzel / GOLD
    return bauen


def _lauf(fake: FakeGh, gold_pfad: Path, tmp_path: Path, *, senden: bool = True,
          noch_tot: frozenset[str] = ALLE_TOT, stand: dict[str, Any] | None = None, wurzel: Path | None = None,
          **kw: Any):
    gh = GitHub("Flissel", laufzeit=fake)
    return rueckfragen_abgleichen(wurzel or tmp_path / "code", gold_pfad, gh, REPO, senden=senden,
                                  noch_tot=noch_tot, commit=COMMIT, stand={} if stand is None else stand, **kw)


def _rf(fake: FakeGh, sym: str, labels: tuple[str, ...] = (), *, state: str = "open") -> int:
    return fake.issue(f"[selbstbild] Rückfrage: {sym}", f"Frage\n\n{marke(sym)}\n",
                      labels=("selbstbild", RUECKFRAGE_LABEL, *labels), state=state)


def _rueckfragen(fake: FakeGh) -> list[dict[str, Any]]:
    return [i for i in fake.issues.values() if "selbstbild:rueckfrage=" in i["body"]]


def _tot(gold_pfad: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(gold_pfad.read_text(encoding="utf-8"))["tot"]


def _kommentare(fake: FakeGh, n: int) -> list[str]:
    """Kommentartexte ohne die angehaengte Automatik-Marke (die prueft test_funde_github_marke)."""
    return [t.removesuffix("\n\n" + AUTO_MARKE) for k, t in fake.kommentare if k == n]


def _schreibend(fake: FakeGh) -> list[tuple[str, str]]:
    return [(m, p) for m, p in fake.aufrufe if m != "GET"]


# ---------- Konstanten und Marke ----------

def test_konstanten() -> None:
    assert RUECKFRAGE_LABEL == "selbstbild:rueckfrage"
    assert ANTWORTEN == {"antwort:weg": ("ja", False), "antwort:bleibt": ("ja", True),
                         "antwort:benutzt": ("nein", False)}
    assert MAX_NEU == 10


def test_marke_hash_des_symbols() -> None:
    import hashlib

    s = _sym(1)
    assert marke(s) == f"<!-- selbstbild:rueckfrage={hashlib.sha256(s.encode()).hexdigest()[:16]} -->"
    assert marke(s) != marke(_sym(2))


# ---------- Anlegen ----------

def test_neue_rueckfrage_je_unklar_genau_einmal(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar"), (_sym(1), "offen"), (_sym(2), "ja")])
    fake = FakeGh()
    stand: dict[str, Any] = {}
    b1 = _lauf(fake, g, tmp_path, stand=stand)
    b2 = _lauf(fake, g, tmp_path, stand=stand)
    rf = _rueckfragen(fake)
    assert len(rf) == 1 and b1.fehler is None and b2.fehler is None
    i = rf[0]
    assert b1.neu == [i["number"]] and b2.neu == []
    assert marke(_sym(0)) in i["body"]
    assert i["title"] == "[selbstbild] Rückfrage: f0 (vibemind/pkg/a.py)"
    assert "Wird das noch gebraucht?" in i["body"]
    for label in ANTWORTEN:
        assert f"`{label}`" in i["body"]
    assert fake.gesendete_labels == [["selbstbild", RUECKFRAGE_LABEL]]
    assert {RUECKFRAGE_LABEL, *ANTWORTEN, "selbstbild"} <= fake.labels
    assert b1.links[i["number"]] == f"https://github.com/{REPO}/issues/{i['number']}"


def test_hoechstens_zehn_neue_je_lauf(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(i), "unklar") for i in range(12)])
    fake = FakeGh()
    b1 = _lauf(fake, g, tmp_path)
    assert len(b1.neu) == 10 and len(_rueckfragen(fake)) == 10
    assert any("2 weitere" in n for n in b1.notizen)
    b2 = _lauf(fake, g, tmp_path)
    assert len(b2.neu) == 2 and len(_rueckfragen(fake)) == 12


def test_rueckfrage_text_mit_ort_und_ausschnitt(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    code = type(git_repo)(tmp_path / "vibemind", git_repo._env)
    code.datei("pkg/a.py", "import os\n\n\ndef f0():\n    return 1\n")
    code.commit("code")
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    b = _lauf(fake, g, tmp_path, wurzel=code.wurzel)
    body = _rueckfragen(fake)[0]["body"]
    assert b.neu and "vibemind/pkg/a.py:4" in body
    assert "```python\ndef f0():\n    return 1\n```" in body
    assert "unerreichbar" in body


def test_oeffentliches_ziel_ohne_ausschnitt(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    code = type(git_repo)(tmp_path / "vibemind", git_repo._env)
    code.datei("pkg/a.py", "def f0():\n    return 1\n")
    code.commit("code")
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    fake.privat = False
    b = _lauf(fake, g, tmp_path, wurzel=code.wurzel)
    body = _rueckfragen(fake)[0]["body"]
    assert b.neu and "```" not in body and "def f0" not in body


def test_oeffentlich_konfiguriert_ohne_ausschnitt_auch_wenn_github_privat(gold: Any, git_repo: Any,
                                                                         tmp_path: Path) -> None:
    """R17-6 (D6): Konfiguration ODER GitHub sagt oeffentlich -> kein Ausschnitt."""
    code = type(git_repo)(tmp_path / "vibemind", git_repo._env)
    code.datei("pkg/a.py", "def f0():\n    return 1\n")
    code.commit("code")
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    assert fake.privat is True
    b = _lauf(fake, g, tmp_path, wurzel=code.wurzel, oeffentlich=True)
    body = _rueckfragen(fake)[0]["body"]
    assert b.neu and "```" not in body and "def f0" not in body


# ---------- Antworten ----------

def test_antwort_bleibt_wird_ja_und_bleibt_und_schliesst(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar"), (_sym(1), "offen")])
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:bleibt",))
    git_repo.datei("anderes.txt", "fremde Arbeit\n")  # fremde uncommittete Aenderung bleibt liegen
    vorher = git_repo.git("rev-parse", "HEAD")
    b = _lauf(fake, g, tmp_path)
    assert b.fehler is None and b.uebernommen == [n] and b.wartet == []
    tot = _tot(g)
    assert tot[0]["urteil"] == "ja" and tot[0]["bleibt"] is True
    assert tot[1] == {"gruende": ["alt_90", "unerreichbar"], "symbol": _sym(1), "urteil": "offen"}
    assert git_repo.git("rev-parse", "HEAD^") == vorher
    assert git_repo.git("log", "-1", "--format=%s") == MELDUNG
    assert git_repo.git("show", "--name-only", "--format=", "HEAD") == GOLD
    assert git_repo.git("status", "--porcelain") == "M anderes.txt"
    sha7 = git_repo.git("rev-parse", "--short=7", "HEAD")
    assert fake.issues[n]["state"] == "closed"
    assert _kommentare(fake, n) == [f"Antwort übernommen: ja, bleibt (Goldsatz-Commit {sha7})."]
    assert not any(marke(_sym(0)) in i["body"] for i in _rueckfragen(fake) if i["number"] != n)


def test_antwort_benutzt_wird_nein(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:benutzt",))
    b = _lauf(fake, g, tmp_path)
    assert b.uebernommen == [n]
    assert _tot(g)[0]["urteil"] == "nein" and "bleibt" not in _tot(g)[0]
    sha7 = git_repo.git("rev-parse", "--short=7", "HEAD")
    assert _kommentare(fake, n) == [f"Antwort übernommen: nein (Goldsatz-Commit {sha7})."]
    assert fake.issues[n]["state"] == "closed"


def test_antwort_weg_wird_ja(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:weg",))
    b = _lauf(fake, g, tmp_path)
    assert b.uebernommen == [n] and _tot(g)[0]["urteil"] == "ja" and "bleibt" not in _tot(g)[0]


def test_zwei_antwort_labels_kommentiert_einmal(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    roh = g.read_bytes()
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:weg", "antwort:benutzt"))
    vorher = git_repo.git("rev-parse", "HEAD")
    stand: dict[str, Any] = {}
    b1 = _lauf(fake, g, tmp_path, stand=stand)
    b2 = _lauf(fake, g, tmp_path, stand=stand)
    assert _kommentare(fake, n) == [
        "Bitte nur ein Antwort-Label setzen (antwort:weg, antwort:bleibt oder antwort:benutzt)."]
    assert b1.doppelt_beantwortet == [n] and b2.doppelt_beantwortet == [n]
    assert b1.uebernommen == [] and b2.uebernommen == []
    assert g.read_bytes() == roh and git_repo.git("rev-parse", "HEAD") == vorher
    assert fake.issues[n]["state"] == "open"
    assert stand["kommentiert"] == [n]


def test_goldsatz_unsauber_antwort_wartet(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    g.write_bytes(g.read_bytes() + b"# fremde Notiz\n")
    roh = g.read_bytes()
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:weg",))
    vorher = git_repo.git("rev-parse", "HEAD")
    b = _lauf(fake, g, tmp_path)
    assert b.wartet and "unsauber" in b.wartet[0] and f"#{n}" in b.wartet[0]
    assert b.uebernommen == [] and b.fehler is None
    assert git_repo.git("rev-parse", "HEAD") == vorher and g.read_bytes() == roh
    assert fake.issues[n]["state"] == "open" and _kommentare(fake, n) == []


def test_nicht_main_antwort_wartet(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    git_repo.git("checkout", "-q", "-b", "anderer")
    roh = g.read_bytes()
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:weg",))
    b = _lauf(fake, g, tmp_path)
    assert b.wartet and "falscher_zweig" in b.wartet[0]
    assert g.read_bytes() == roh and fake.issues[n]["state"] == "open"


def test_commit_scheitert_datei_zurueck(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    roh = g.read_bytes()
    haken = tmp_path / "haken"
    haken.mkdir()
    (haken / "pre-commit").write_bytes(b"#!/bin/sh\nexit 1\n")
    git_repo.git("config", "core.hooksPath", str(haken))
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:weg",))
    vorher = git_repo.git("rev-parse", "HEAD")
    b = _lauf(fake, g, tmp_path)
    assert b.wartet and "commit" in b.wartet[0]
    assert g.read_bytes() == roh and git_repo.git("rev-parse", "HEAD") == vorher
    assert git_repo.git("status", "--porcelain") == ""
    assert fake.issues[n]["state"] == "open" and b.uebernommen == []


def test_antwort_uebernehmen_idempotent(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "nein")])
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:benutzt",))
    vorher = git_repo.git("rev-parse", "HEAD")
    b = _lauf(fake, g, tmp_path)
    assert git_repo.git("rev-parse", "HEAD") == vorher
    assert b.uebernommen == [n] and fake.issues[n]["state"] == "closed"
    assert _kommentare(fake, n) == ["Antwort übernommen: nein (Goldsatz-Commit schon übernommen)."]


# ---------- Sonderfaelle ----------

def test_symbol_nicht_mehr_tot_schliesst_erledigt(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar"), (_sym(1), "unklar")])
    roh = g.read_bytes()
    fake = FakeGh()
    n0 = _rf(fake, _sym(0))
    n1 = _rf(fake, _sym(1))
    stand: dict[str, Any] = {}
    b = _lauf(fake, g, tmp_path, noch_tot=frozenset({_sym(1)}), stand=stand)
    assert b.erledigt == [n0]
    assert fake.issues[n0]["state"] == "closed" and fake.issues[n1]["state"] == "open"
    assert _kommentare(fake, n0) == [f"Nicht mehr als toter Code erkannt (Commit {COMMIT[:7]}), Rückfrage erledigt."]
    assert g.read_bytes() == roh and b.neu == []
    # spaeter: kein "geschlossen ohne Antwort" und keine neue Rueckfrage
    b2 = _lauf(fake, g, tmp_path, noch_tot=frozenset({_sym(1)}), stand=stand)
    assert b2.geschlossen_ohne_antwort == [] and b2.neu == [] and b2.erledigt == []


def test_geschlossen_ohne_antwort_nie_wieder_offen(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    n = _rf(fake, _sym(0), state="closed")
    stand: dict[str, Any] = {}
    b1 = _lauf(fake, g, tmp_path, stand=stand)
    b2 = _lauf(fake, g, tmp_path, stand=stand)
    assert b1.geschlossen_ohne_antwort == [n] and b2.geschlossen_ohne_antwort == []
    assert not any(m == "PATCH" for m, _ in fake.aufrufe)
    assert fake.issues[n]["state"] == "closed" and b1.neu == [] and b2.neu == []
    assert stand["gemeldet"] == [n]


def test_bestand_fehler_schreibt_nichts(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()

    def kaputt(args: list[str], **kw: Any) -> Any:
        if args[1] == "api" and "/issues?" in args[-1]:
            return subprocess.CompletedProcess(args, 1, b"", b"gh: Server Error (HTTP 502) geheim")
        return fake(args, **kw)

    b = rueckfragen_abgleichen(tmp_path, g, GitHub("Flissel", laufzeit=kaputt), REPO, senden=True,
                               noch_tot=ALLE_TOT, commit=COMMIT, stand={})
    assert b.fehler == "http_502" and b.neu == []
    assert _schreibend(fake) == []


def test_trocken_schreibt_nichts(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar"), (_sym(1), "unklar"), (_sym(2), "unklar"), (_sym(3), "unklar")])
    roh = g.read_bytes()
    fake = FakeGh()
    n1 = _rf(fake, _sym(1), ("antwort:weg",))
    n2 = _rf(fake, _sym(2), ("antwort:weg", "antwort:bleibt"))
    n3 = _rf(fake, _sym(3))
    vorher = git_repo.git("rev-parse", "HEAD")
    stand: dict[str, Any] = {}
    b = _lauf(fake, g, tmp_path, senden=False, noch_tot=ALLE_TOT - {_sym(3)}, stand=stand)
    assert _schreibend(fake) == [] and fake.kommentare == []
    assert g.read_bytes() == roh and git_repo.git("rev-parse", "HEAD") == vorher
    assert stand == {}
    assert b.uebernommen == [n1] and b.doppelt_beantwortet == [n2] and b.erledigt == [n3]
    assert b.neu == [] and any("Rückfrage: f0" in x for x in b.notizen)
    assert all(i["state"] == "open" for i in fake.issues.values())


# ---------- Code-Ausschnitt ----------

def test_code_ausschnitt_mit_schluessel_none(git_repo: Any, tmp_path: Path) -> None:
    code = type(git_repo)(tmp_path / "vibemind", git_repo._env)
    geheim = "ghp_" + "A1b2C3d4" * 5
    code.datei("pkg/a.py", f'def f():\n    token = "{geheim}"\n    return token\n\n\ndef g():\n    return 2\n')
    code.commit("code")
    assert code_ausschnitt(code.wurzel, "symbol:vibemind:pkg/a.py#f") is None
    assert code_ausschnitt(code.wurzel, "symbol:vibemind:pkg/a.py#g") == "def g():\n    return 2"


def test_code_ausschnitt_liest_head_und_untermodul(git_repo: Any, tmp_path: Path) -> None:
    wurzel = tmp_path / "vibemind"
    code = type(git_repo)(wurzel, git_repo._env)
    code.datei("README.md", "x\n")
    code.commit("aussen")
    sub = type(git_repo)(wurzel / "sub", git_repo._env)
    zeilen = "".join(f"        x{i} = {i}\n" for i in range(30))
    sub.datei("m.py", f"class K:\n    @staticmethod\n    def m():\n{zeilen}        return 0\n")
    sub.commit("sub")
    (wurzel / "sub" / "m.py").write_text("class K:\n    def m():\n        pass\n", encoding="utf-8")  # nicht committet
    aus = code_ausschnitt(wurzel, "symbol:vibemind/sub:m.py#K.m", n=5)
    assert aus == "    @staticmethod\n    def m():\n        x0 = 0\n        x1 = 1\n        x2 = 2"
    assert code_ausschnitt(wurzel, "symbol:vibemind/sub:m.py#K.fehlt") is None
    assert code_ausschnitt(wurzel, "symbol:vibemind/sub:../m.py#K.m") is None
    assert code_ausschnitt(wurzel, "symbol:anders/sub:m.py#K.m") is None
    assert code_ausschnitt(wurzel, "symbol:vibemind/fehlt:m.py#K.m") is None
    assert code_ausschnitt(wurzel, "symbol:vibemind:README.md#x") is None


# ---------- Fix-Runde 1 (R6, R7, R8a, R8b) ----------

def test_antwort_ueberschreibt_kein_urteil(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "ja")])
    roh = g.read_bytes()
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:benutzt",))
    vorher = git_repo.git("rev-parse", "HEAD")
    b = _lauf(fake, g, tmp_path)
    assert git_repo.git("rev-parse", "HEAD") == vorher and g.read_bytes() == roh
    assert fake.issues[n]["state"] == "closed"
    assert _kommentare(fake, n) == ["Im Goldsatz inzwischen anders beurteilt (ja), Rückfrage erledigt."]
    assert b.erledigt == [n] and b.uebernommen == [] and b.goldsatz_commit is None


def test_antwort_bleibt_bei_ja_ohne_bleibt_ist_anders(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "ja")])
    roh = g.read_bytes()
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:bleibt",))
    b = _lauf(fake, g, tmp_path)
    assert g.read_bytes() == roh and b.erledigt == [n]
    assert _kommentare(fake, n) == ["Im Goldsatz inzwischen anders beurteilt (ja), Rückfrage erledigt."]


def test_geschlossen_mit_antwort_wird_uebernommen(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:weg",), state="closed")
    stand: dict[str, Any] = {}
    b = _lauf(fake, g, tmp_path, stand=stand)
    sha7 = git_repo.git("rev-parse", "--short=7", "HEAD")
    assert _tot(g)[0]["urteil"] == "ja" and git_repo.git("log", "-1", "--format=%s") == MELDUNG
    assert b.uebernommen == [n] and b.geschlossen_ohne_antwort == [] and b.goldsatz_commit == sha7
    assert _kommentare(fake, n) == [f"Antwort übernommen: ja (Goldsatz-Commit {sha7})."]
    assert fake.issues[n]["state"] == "closed"
    assert not any(m == "PATCH" for m, _ in fake.aufrufe)  # nie wieder geoeffnet, nicht noch einmal geschlossen
    # zweiter Lauf: nichts mehr zu tun
    b2 = _lauf(fake, g, tmp_path, stand=stand)
    assert b2.uebernommen == [] and len(_kommentare(fake, n)) == 1 and b2.goldsatz_commit is None


def test_geschlossen_mit_antwort_wartet_bei_unsauber(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    g.write_bytes(g.read_bytes() + b"# fremde Notiz\n")
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:weg",), state="closed")
    b = _lauf(fake, g, tmp_path)
    assert b.wartet and f"#{n}" in b.wartet[0] and "unsauber" in b.wartet[0]
    assert b.uebernommen == [] and _kommentare(fake, n) == [] and b.geschlossen_ohne_antwort == []


def test_noch_tot_leer_schliesst_nichts_als_erledigt(gold: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    n = _rf(fake, _sym(0))
    b = _lauf(fake, g, tmp_path, noch_tot=frozenset())
    assert b.erledigt == [] and fake.issues[n]["state"] == "open" and _schreibend(fake) == []
    assert "noch_tot leer: keine Rückfrage als erledigt geschlossen" in b.notizen


def test_goldsatz_commit_im_bericht(gold: Any, git_repo: Any, tmp_path: Path) -> None:
    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    _rf(fake, _sym(0), ("antwort:weg",))
    b = _lauf(fake, g, tmp_path)
    assert b.goldsatz_commit == git_repo.git("rev-parse", "--short=7", "HEAD")


def test_kopf_unlesbar_nach_commit_kein_fragezeichen(gold: Any, git_repo: Any, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    import selbstbild.funde.rueckfrage as rf

    g = gold([(_sym(0), "unklar")])
    fake = FakeGh()
    n = _rf(fake, _sym(0), ("antwort:weg",))
    monkeypatch.setattr(rf, "kopf", lambda repo: None)
    b = _lauf(fake, g, tmp_path)
    assert b.wartet and "commit_unbestaetigt" in b.wartet[0] and b.goldsatz_commit is None
    assert fake.issues[n]["state"] == "open" and _kommentare(fake, n) == []
    assert not any("?" in t for _, t in fake.kommentare)
    # naechster Lauf (HEAD wieder lesbar): Antwort steht schon im Goldsatz -> "schon übernommen"
    monkeypatch.undo()
    b2 = _lauf(fake, g, tmp_path)
    assert b2.uebernommen == [n]
    assert _kommentare(fake, n) == ["Antwort übernommen: ja (Goldsatz-Commit schon übernommen)."]


# ---------- Final-Fix-Welle (R17-1) ----------

def test_yaml_kaputt_nach_uebernehmen_datei_zurueck_wartet(gold: Any, git_repo: Any, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Liefert das Eintragen eine Datei, deren Eintrag nicht das erwartete Urteil/bleibt hat (oder die gar kein
    YAML mehr ist), wird nichts committet: Datei auf HEAD, Antwort wartet (`yaml_kaputt`)."""
    import selbstbild.funde.rueckfrage as rf

    g = gold([(_sym(0), "unklar")])
    roh = g.read_bytes()
    vorher = git_repo.git("rev-parse", "HEAD")
    for kaputt in (b"tot: [unvollstaendig\n", roh.replace(b"urteil: unklar", b"urteil: nein")):
        def falsch(pfad: Path, antworten: dict[str, tuple[str, bool]], inhalt: bytes = kaputt) -> str:
            pfad.write_bytes(inhalt)
            return "diff"
        monkeypatch.setattr(rf, "antworten_uebernehmen", falsch)
        fake = FakeGh()
        n = _rf(fake, _sym(0), ("antwort:bleibt",))
        b = _lauf(fake, g, tmp_path)
        assert b.wartet == [f"#{n} f0: yaml_kaputt"] and b.uebernommen == [] and b.fehler is None
        assert g.read_bytes() == roh and git_repo.git("rev-parse", "HEAD") == vorher
        assert git_repo.git("status", "--porcelain") == ""
        assert fake.issues[n]["state"] == "open" and _kommentare(fake, n) == []
