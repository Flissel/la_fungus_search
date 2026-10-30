"""doku_loop: Tick, CLI, Marker-Hook und Windows-Aufgabe (Task 15, mit Fix-Runde 1).

Alle Laeufe gehen gegen ein frisches `git init`-Fixture-Repo unter tmp_path, nie gegen das echte Repo.
Judge (`laufzeit`), Budget, Torwaechter und Manifest-Bau sind Fakes; die Windows-Aufgabenplanung wird nie
angefasst. Der lokale Zustand (Stand, Pruefliste, Urteils-Cache, Sperre) liegt unter tmp_path/lokal.
"""

import ast
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import __main__ as cli  # noqa: E402
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import schreibvorgang as sv  # noqa: E402
from selbstbild.doku import stand as st  # noqa: E402
from selbstbild.doku import tick as tk  # noqa: E402
from selbstbild.doku import umschreiben  # noqa: E402
from selbstbild.doku.graph import Graph, schlank  # noqa: E402

KETTEN = "docs/spaces/KETTEN.md"

CLIENT = (
    "import os\n"
    "import requests\n"
    "\n"
    'B = os.environ.get("B_URL", "http://b:8130")\n'
    "\n"
    "\n"
    "def senden():\n"
    '    return requests.post(f"{B}/api/x", json={})\n'
    "\n"
    "\n"
    "def helfer():\n"
    "    return 1\n"
)
API = (
    "from fastapi import FastAPI\n"
    "\n"
    "app = FastAPI()\n"
    "\n"
    "\n"
    '@app.post("/api/x")\n'
    "def x():\n"
    "    return 1\n"
)
DIENSTE = (
    "dienste:\n"
    "  a:\n"
    "    port: 8120\n"
    "    ort: container\n"
    "    space: a\n"
    "    env: [A_URL]\n"
    "  b:  # pruefen: von Hand gepflegt\n"
    "    port: 8130\n"
    "    ort: container\n"
    "    space: b\n"
    "    env: [B_URL]\n"
)
COMPOSE = (
    "services:\n"
    "  a:\n"
    "    environment:\n"
    "      B_URL: http://b:8130\n"
    "  b:\n"
    "    ports: ['8130:8130']\n"
    "  c:\n"
    "    ports: ['8140:8140']\n"
    "    environment:\n"
    "      B_URL: http://b:8130\n"
)
AUSSAGE = "Der Client ruft `senden` in `sp/a/client.py:7` auf und schickt den Auftrag an Dienst b."
NEUER_TEXT = "Der Client ruft `schicken` in `sp/a/client.py:7` auf und schickt den Auftrag an Dienst b."
KETTEN_MD = "\n".join([
    "# Ketten",
    "",
    "## Übersicht",
    "",
    "| # | Kette | Zustand | Reißt bei |",
    "|---|-------|---------|-----------|",
    "| K1 | Fixture | läuft | — |",
    "",
    "### K1 · Fixture",
    "",
    "| # | Von → Nach | Mechanismus | Code-Stelle | Belegt durch | Zustand |",
    "|---|---|---|---|---|---|",
    "| 1 | Client `senden` → Dienst `b:8130` | HTTP POST `/api/x` | `sp/a/client.py:7` | — | verdrahtet |",
    "",
    AUSSAGE,
    "",
])


# ---------- Fixture ----------

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=60,
    ).stdout


def _schreib(repo: Path, rel: str, text: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8"))


def _commit(repo: Path, nachricht: str, *pfade: str) -> str:
    _git(repo, "add", "--", *pfade)
    _git(repo, "commit", "-q", "-m", nachricht)
    return _git(repo, "rev-parse", "HEAD").strip()


def _konfig_datei(repo: Path, lokal: Path) -> Path:
    roh = yaml.safe_load((VORLAGE).read_text(encoding="utf-8"))
    roh.update({
        "spaces": {"a": {"wurzel": "sp/a", "mcp_muster": "werkzeuge"},
                   "b": {"wurzel": "sp/b", "mcp_muster": "decorator"}},
        "compose_live": ["infra/swarm/vibemind-stack.yml"],
        "ketten_md": KETTEN,
        "weitere_doku": [],
        "sql_wurzeln": ["."],
        "dienste_yaml": "config/dienste.yaml",
        "waechter_zustandsdatei": "logs/w.json",
        "waechter_zuordnung": {"probe": ["K1"]},
        "marker": str(lokal / "VibeMind" / "doku_loop" / "anstoss"),
        "claude_bin": "C:/nirgends/claude.exe",
        "rowboat_wissen": str(lokal / "rowboat"),
        "zweig": None,  # echte Konfig: nur codex/vibemind-ops-baseline; das Fixture-Repo hat einen anderen Branch
        "rowboat_notizen_aktiv": True,  # echte Konfig: aus bis zum Spiegel-Deploy
    })
    pfad = repo / "config" / "doku_loop.yaml"
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(yaml.safe_dump(roh, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return pfad


@pytest.fixture
def fx(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    lokal = tmp_path / "lokal"
    monkeypatch.setenv("LOCALAPPDATA", str(lokal))
    hooks = tmp_path / "keine_hooks"
    hooks.mkdir()
    _git(repo, "init", "-q")
    for k, v in (("user.name", "Test"), ("user.email", "test@example.invalid"), ("core.autocrlf", "false"),
                 ("commit.gpgsign", "false"), ("core.hooksPath", str(hooks))):
        _git(repo, "config", k, v)
    _schreib(repo, "sp/a/client.py", CLIENT)
    _schreib(repo, "sp/b/api.py", API)
    _schreib(repo, "config/dienste.yaml", DIENSTE)
    _schreib(repo, "infra/swarm/vibemind-stack.yml", COMPOSE)
    _schreib(repo, KETTEN, KETTEN_MD)
    _schreib(repo, "README.md", "fixture\n")
    _schreib(repo, ".gitignore", "logs/\n")
    _schreib(repo, "vibemind.config.json", json.dumps({"ports": {"openfang": 4200}}))
    konfig_pfad = _konfig_datei(repo, lokal)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "start")
    konfig = konfig_modul.laden(konfig_pfad)
    return SimpleNamespace(repo=repo, konfig=konfig, konfig_pfad=konfig_pfad, lokal=lokal, marker=konfig.marker)


def _manifest(wurzel: Path, commit: str = "sub1"):
    """Kleines Manifest wie benchmarks.gate2.manifest (Funktionen/Klassen, Methoden in Klassen)."""
    docs = []
    for p in sorted(wurzel.rglob("*.py")):
        rel = p.relative_to(wurzel).as_posix()
        modul = ".".join(p.relative_to(wurzel).with_suffix("").parts)
        text = p.read_text(encoding="utf-8")
        zeilen = text.splitlines()

        def sammeln(knoten, praefix, rel=rel, zeilen=zeilen):
            for k in ast.iter_child_nodes(knoten):
                if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    did = f"{praefix}.{k.name}"
                    docs.append(SimpleNamespace(
                        document_id=did, file=rel, start_line=k.lineno, end_line=k.end_lineno,
                        symbol=k.name, source="\n".join(zeilen[k.lineno - 1:k.end_lineno])))
                    if isinstance(k, ast.ClassDef):
                        sammeln(k, did)

        sammeln(ast.parse(text), modul)
    return SimpleNamespace(corpus_root=str(wurzel), commit_sha=commit, documents=tuple(docs),
                           callees_by_document={})


class FakeManifestBauer:
    def __init__(self):
        self.gebaut: list[str] = []

    def __call__(self, space, konfig, commit):
        self.gebaut.append(space.name)
        return _manifest(konfig.repo / space.wurzel, commit)


class FakeBudget:
    def __init__(self, erlaubt=True, grund="ok"):
        self.erlaubt, self.grund = erlaubt, grund
        self.pruefungen: list[str] = []
        self.buchungen: list[tuple[str, str]] = []

    def pruefen(self, agent, *, jetzt=None, umgebung=None):
        self.pruefungen.append(agent)
        return SimpleNamespace(erlaubt=self.erlaubt, grund=self.grund, agent=agent, heute=1, grenze=30)

    def buchen(self, agent, ergebnis, dauer_s, *, jetzt=None, umgebung=None):
        self.buchungen.append((agent, ergebnis))

    def tagesstand(self, *, jetzt=None, umgebung=None):
        return {"datum": "2026-10-28", "user_aktiv": False, "not_aus": False,
                "agenten": [{"agent": "doku-loop", "heute": 3, "grenze": 30, "abgelehnt": {"tagesgrenze": 1}},
                            {"agent": "anderer", "heute": 9, "grenze": 10, "abgelehnt": {}}]}


class FakeLaufzeit:
    """Fake fuer subprocess.run des Judge: liefert der Reihe nach die vorbereiteten `result`-Texte."""

    def __init__(self, *antworten):
        self.antworten = list(antworten)
        self.aufrufe: list[list[str]] = []

    def __call__(self, argv, **kw):
        self.aufrufe.append(list(argv))
        antwort = self.antworten.pop(0) if self.antworten else ""
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"result": antwort}), stderr="")


def _urteil(urteil="veraltet", text=NEUER_TEXT, anker="client.schicken"):
    if urteil != "veraltet":
        text = anker = None
    return json.dumps({"urteil": urteil, "neuer_text": text, "beleg_anker": anker, "begruendung": "umbenannt"})


_ZEIT = [datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)]


def _jetzt():
    _ZEIT[0] += timedelta(minutes=1)
    return _ZEIT[0]


def _tick(fx, *, laufzeit=None, budget=None, drift=None, bauer=None, **kw):
    return tk.tick(fx.konfig, jetzt=kw.pop("jetzt", None) or _jetzt(), laufzeit=laufzeit or FakeLaufzeit(),
                   budget=budget or FakeBudget(), drift=drift or (lambda repo: set()),
                   manifest_bauer=bauer or FakeManifestBauer(), **kw)


def _seed(fx, drift=None):
    erg = tk.seed(fx.konfig, jetzt=_jetzt(), drift=drift or (lambda repo: set()), manifest_bauer=FakeManifestBauer())
    return erg


def _anstoss(fx):
    fx.marker.parent.mkdir(parents=True, exist_ok=True)
    fx.marker.write_bytes(b"")


def _text(fx):
    return (fx.repo / KETTEN).read_bytes().decode("utf-8")


def _head(repo):
    return _git(repo, "rev-parse", "HEAD").strip()


def _status(repo):
    return [z for z in _git(repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines() if z.strip()]


def _namen(repo, rev="HEAD"):
    return set(_git(repo, "show", "--name-only", "--format=", rev).split())


def _umbenennen(fx):
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("def senden", "def schicken"))
    return _commit(fx.repo, "umbenannt", "sp/a/client.py")


def _code_aenderung(fx, i):
    """Aenderung in Space b ohne Wirkung auf Symbole (Kommentar am Ende)."""
    _schreib(fx.repo, "sp/b/api.py", API + f"\n# Aenderung {i}\n")
    return _commit(fx.repo, f"api {i}", "sp/b/api.py")


def _pruefliste(fx):
    return tk.pruefliste_laden(fx.konfig)


def _start(fx):
    """Geseedetes Fixture (wie nach dem produktiven Seed) + Umbenennung + Marker."""
    _seed(fx)
    geseedet = _text(fx)
    _umbenennen(fx)
    _anstoss(fx)
    return geseedet


# ---------- Anstoss, Stand, geaenderte Dateien ----------

def test_anstoss_noetig(fx):
    jetzt = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
    assert tk.anstoss_noetig(fx.konfig, None, jetzt)
    assert tk.anstoss_noetig(fx.konfig, tk.Stand(letzter_tick=None), jetzt)
    assert not tk.anstoss_noetig(fx.konfig, tk.Stand(letzter_tick=(jetzt - timedelta(minutes=5)).isoformat()), jetzt)
    assert tk.anstoss_noetig(fx.konfig, tk.Stand(letzter_tick=(jetzt - timedelta(minutes=31)).isoformat()), jetzt)
    _anstoss(fx)
    assert tk.anstoss_noetig(fx.konfig, tk.Stand(letzter_tick=(jetzt - timedelta(minutes=5)).isoformat()), jetzt)


def test_geaenderte_dateien_ohne_stand_alle_und_mit_stand_diff(fx):
    assert tk.geaenderte_dateien(fx.repo, None) == (["*"], [])
    alt = _head(fx.repo)
    _umbenennen(fx)
    assert tk.geaenderte_dateien(fx.repo, tk.Stand(letzter_commit_aussen=alt)) == (["sp/a/client.py"], [])
    assert tk.geaenderte_dateien(fx.repo, tk.Stand(letzter_commit_aussen="0" * 40)) == (["*"], [])
    assert tk.betroffene_spaces(fx.konfig, ["sp/a/client.py", "README.md"], []) == ["a"]
    assert tk.betroffene_spaces(fx.konfig, ["README.md"], []) == []
    assert tk.betroffene_spaces(fx.konfig, ["*"], []) == ["a", "b"]
    assert tk.betroffene_spaces(fx.konfig, ["config/dienste.yaml"], []) == ["a", "b"]
    assert tk.betroffene_spaces(fx.konfig, ["config/doku_loop.yaml"], []) == ["a", "b"]


def test_stand_liegt_lokal_je_repo_und_nie_im_repo(fx, tmp_path):
    """Ruling 44: Stand in %LOCALAPPDATA%/VibeMind/doku_loop, je Repo-Pfad ein eigener Hash."""
    _seed(fx)
    pfad = tk.stand_pfad(fx.konfig)
    assert pfad.parent == fx.lokal / "VibeMind" / "doku_loop" and pfad.is_file()
    assert pfad.name == f"stand-{st.repo_schluessel(fx.repo)}.json" and len(st.repo_schluessel(fx.repo)) == 12
    anderes = dataclasses.replace(fx.konfig, repo=tmp_path / "worktree")
    assert tk.stand_pfad(anderes) != pfad
    assert not (fx.repo / "docs/graph/stand.json").exists()
    assert "docs/graph/stand.json" not in _git(fx.repo, "log", "--all", "--name-only", "--format=")


# ---------- Tick ----------

def test_tick_ohne_seed_ueberspringt(fx):
    erg = _tick(fx)
    assert erg.uebersprungen == "kein_seed" and erg.commit is None
    assert _git(fx.repo, "log", "--format=%s").split("\n")[0] == "start" and _status(fx.repo) == []


def test_tick_erster_lauf_baut_alle_spaces(fx):
    _seed(fx)
    tk.stand_pfad(fx.konfig).unlink()  # kein lokaler Stand (neue Maschine, neuer Worktree)
    bauer = FakeManifestBauer()
    erg = _tick(fx, bauer=bauer)
    assert erg.spaces == ["a", "b"] and sorted(bauer.gebaut) == ["a", "b"]
    assert erg.llm_aufrufe == 0 and erg.umgeschrieben == 0
    assert erg.commit is None  # nichts geaendert seit dem Seed: kein Commit
    stand = tk.stand_laden(fx.konfig)
    assert stand.lauf_nr == 1 and stand.letzter_commit_aussen == _head(fx.repo)
    g = Graph.laden(fx.repo / "docs/graph/graph.json")
    assert any(k.typ == "aussage" for k in g.knoten.values())  # Belege der Aussage gemerkt
    assert _status(fx.repo) == []


def test_tick_ohne_anstoss_wird_uebersprungen(fx):
    _seed(fx)
    head = _head(fx.repo)
    erg = _tick(fx)
    assert erg.uebersprungen == "kein_anstoss" and erg.commit is None and _head(fx.repo) == head


def test_tick_ohne_aenderung_kein_llm_kein_commit(fx):
    _seed(fx)
    _schreib(fx.repo, "README.md", "anders\n")
    neu = _commit(fx.repo, "fremd", "README.md")
    _anstoss(fx)
    lauf, budget, bauer = FakeLaufzeit(), FakeBudget(), FakeManifestBauer()
    erg = _tick(fx, laufzeit=lauf, budget=budget, bauer=bauer)
    assert erg.uebersprungen == "keine_aenderung"
    assert erg.llm_aufrufe == 0 and lauf.aufrufe == [] and budget.pruefungen == []
    assert erg.commit is None and _head(fx.repo) == neu and bauer.gebaut == []
    stand = tk.stand_laden(fx.konfig)
    assert stand.letzter_commit_aussen == neu and stand.lauf_nr == 2  # Stand fortgeschrieben (lokal)
    assert not fx.marker.exists() and _status(fx.repo) == []



def test_kein_umschreiben_ohne_signal(fx):
    """C.9.6: ohne mechanisches Signal (Beleg unveraendert) wird der Judge nie gefragt und nichts umgeschrieben,
    auch wenn der Space neu gebaut wird und der Judge `veraltet` antworten wuerde."""
    _seed(fx)
    text, vorher = _text(fx), _head(fx.repo)
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("return 1", "return 2"))  # nur `helfer`
    neu = _commit(fx.repo, "helfer", "sp/a/client.py")
    _anstoss(fx)
    lauf, budget, bauer = FakeLaufzeit(_urteil(), _urteil()), FakeBudget(), FakeManifestBauer()
    erg = _tick(fx, laufzeit=lauf, budget=budget, bauer=bauer)
    assert bauer.gebaut == ["a"]  # der Space wurde neu gebaut ...
    assert erg.signale == 0 and erg.llm_aufrufe == 0 and erg.umgeschrieben == 0  # ... aber ohne Signal
    assert lauf.aufrufe == [] and budget.pruefungen == []
    assert _text(fx) == text and erg.commit is None and _head(fx.repo) == neu != vorher
    assert not (fx.repo / "docs/graph/ledger.jsonl").exists() and _status(fx.repo) == []


def test_tick_mit_signal_schreibt_um_und_committet(fx):
    _start(fx)
    vorher = _head(fx.repo)
    lauf, budget = FakeLaufzeit(_urteil(), _urteil()), FakeBudget()
    erg = _tick(fx, laufzeit=lauf, budget=budget)
    assert erg.spaces == ["a"]
    assert erg.signale >= 1 and erg.llm_aufrufe == 2 and erg.umgeschrieben == 1
    assert budget.pruefungen == ["doku-loop", "doku-loop"]
    assert erg.commit == _head(fx.repo) and erg.uebersprungen is None
    betreff, *rest = _git(fx.repo, "log", "-1", "--format=%B").strip().split("\n")
    assert betreff == f"docs(loop): a – 1 Aussagen nachgeführt, {erg.pruefliste} in Prüfliste"
    assert f"Lauf: {erg.lauf}" in rest
    assert NEUER_TEXT in _text(fx) and AUSSAGE not in _text(fx)
    assert _namen(fx.repo) == {KETTEN, "docs/graph/graph.json", "docs/graph/ledger.jsonl"}
    eintraege = [json.loads(z) for z in (fx.repo / "docs/graph/ledger.jsonl").read_text("utf-8").splitlines()]
    assert len(eintraege) == 1
    e = eintraege[0]
    assert e["urteil"] == "veraltet" and e["quelle"] == "llm" and e["beleg"]["anker"] == "client.schicken"
    assert e["alt"] == AUSSAGE and e["neu"] == NEUER_TEXT and e["lauf"] == erg.lauf
    # Ruling 46: der Eintrag kennt seinen eigenen Commit nicht, wohl aber Basis und Lauf
    assert e["commit"] is None and e["basis_commit"] == vorher
    assert tk.ledger_commit_von(fx.repo, e["lauf"]) == erg.commit
    assert tk.ledger_commit_von(fx.repo, "gibt-es-nicht#9") is None
    assert _status(fx.repo) == []
    assert any(p["art"] == "hop_beleg_verschwunden" for p in _pruefliste(fx)["offen"])
    # naechster Tick: die neue Aussage ist gemerkt, kein neues Signal, kein LLM
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("def senden", "def schicken") + "\n# Kommentar\n")
    _commit(fx.repo, "kommentar", "sp/a/client.py")
    _anstoss(fx)
    lauf2 = FakeLaufzeit()
    erg2 = _tick(fx, laufzeit=lauf2)
    assert lauf2.aufrufe == [] and erg2.umgeschrieben == 0


def test_tick_unklar_kommt_in_die_pruefliste_und_fragt_nicht_erneut(fx):
    _start(fx)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil("unklar"), _urteil("unklar")))
    assert erg.umgeschrieben == 0 and erg.llm_aufrufe == 2
    assert any(p["art"] == "judge_unklar" for p in _pruefliste(fx)["offen"])
    assert AUSSAGE in _text(fx)
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("def senden", "def schicken") + "\n# x\n")
    _commit(fx.repo, "x", "sp/a/client.py")
    _anstoss(fx)
    lauf = FakeLaufzeit()
    _tick(fx, laufzeit=lauf)
    assert lauf.aufrufe == []


def test_tick_budget_abgelehnt_fragt_spaeter_ohne_neubau(fx):
    _start(fx)
    erg = _tick(fx, budget=FakeBudget(erlaubt=False, grund="tagesgrenze"))
    assert erg.umgeschrieben == 0 and erg.llm_aufrufe == 0
    assert tk.stand_laden(fx.konfig).offene_spaces == ["a"]
    _anstoss(fx)  # ohne neue Aenderung: nur das offene Signal wird erneut gefragt
    bauer = FakeManifestBauer()
    erg2 = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()), bauer=bauer)
    assert erg2.llm_aufrufe == 2  # die neue Frage; der Neubau danach nimmt das Urteil aus dem Cache
    assert erg2.spaces == ["a"] and erg2.umgeschrieben == 1 and bauer.gebaut == ["a"]
    assert NEUER_TEXT in _text(fx)
    assert tk.stand_laden(fx.konfig).offene_signale == []


def test_c95_budget_abgelehnt_drei_ticks_ohne_codeaenderung_kein_commit(fx):
    """Ruling 45c: offene Signale ohne neues Urteil -> kein Neubau, kein Render, kein Commit."""
    _start(fx)
    _tick(fx, budget=FakeBudget(erlaubt=False, grund="tagesgrenze"))
    head = _head(fx.repo)
    for _ in range(3):
        _anstoss(fx)
        bauer, budget = FakeManifestBauer(), FakeBudget(erlaubt=False, grund="tagesgrenze")
        erg = _tick(fx, budget=budget, bauer=bauer)
        assert erg.uebersprungen == "offene_signale_ohne_urteil" and erg.commit is None
        assert bauer.gebaut == [] and erg.llm_aufrufe == 0 and budget.pruefungen == ["doku-loop"]
    assert _head(fx.repo) == head and _status(fx.repo) == []


def test_c95_neue_messzeit_ohne_codeaenderung_kein_commit(fx):
    """Ruling 45a/b: eine neue Messung (mtime der Waechter-Datei) aendert weder KETTEN.md noch den stabilen Digest."""
    waechter = fx.repo / "logs" / "w.json"
    _schreib(fx.repo, "logs/w.json", json.dumps({"stand": {"probe": True}}))
    _seed(fx)
    assert "gemessen läuft" in _text(fx) and "g:läuft" in _text(fx)
    head = _head(fx.repo)
    for i in range(2):
        spaeter = time.time() + 3600 * (i + 1)
        os.utime(waechter, (spaeter, spaeter))
        _code_aenderung(fx, i)  # Space b wird neu gebaut, die Messzeit im Graph aendert sich
        _anstoss(fx)
        erg = _tick(fx)
        assert erg.commit is None and erg.spaces == ["b"]
    assert _head(fx.repo) != head  # nur die zwei eigenen Code-Commits ...
    assert _git(fx.repo, "log", "--format=%s", f"{head}..HEAD").split() == ["api", "1", "api", "0"]  # ... kein Loop


def test_tick_ohne_llm_ruft_den_judge_nicht(fx, monkeypatch):
    _start(fx)
    monkeypatch.setitem(sys.modules, "selbstbild.doku.judge", None)  # jeder Import des Judge wuerde scheitern
    lauf, budget = FakeLaufzeit(_urteil(), _urteil()), FakeBudget()
    erg = _tick(fx, laufzeit=lauf, budget=budget, ohne_llm=True)
    assert erg.signale >= 1 and erg.llm_aufrufe == 0 and erg.umgeschrieben == 0
    assert lauf.aufrufe == [] and budget.pruefungen == []
    assert any(p["art"] == "signal_ohne_llm" for p in _pruefliste(fx)["offen"])
    assert AUSSAGE in _text(fx)


def test_tick_fremde_aenderung_ueberspringt_ohne_stash_und_vor_dem_judge(fx):
    geseedet = _start(fx)
    head = _head(fx.repo)
    fremd = geseedet + "\nNotiz einer anderen Session.\n"
    _schreib(fx.repo, KETTEN, fremd)
    lauf = FakeLaufzeit(_urteil(), _urteil())
    erg = _tick(fx, laufzeit=lauf)
    assert erg.uebersprungen == f"fremde_aenderungen:{KETTEN}"
    assert erg.commit is None and lauf.aufrufe == [] and _head(fx.repo) == head
    assert _text(fx) == fremd
    assert _git(fx.repo, "stash", "list").strip() == ""
    _schreib(fx.repo, KETTEN, geseedet)  # aufgeraeumt: die Aenderung wird nachgeholt
    _anstoss(fx)
    erg2 = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg2.spaces == ["a"] and erg2.umgeschrieben == 1


def test_tick_jede_datei_in_docs_graph_ist_fremd(fx):
    """Ruling 44: keine Ausnahme mehr fuer stand.json; alles unter docs/graph wird geprueft."""
    _start(fx)
    _schreib(fx.repo, "docs/graph/stand.json", "{}\n")
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.uebersprungen == "fremde_aenderungen:docs/graph/stand.json"


def test_tick_marker_wird_geloescht_und_fehlender_ordner_stoert_nicht(fx):
    _seed(fx)
    shutil.rmtree(fx.marker.parent)  # Marker-Ordner (samt lokalem Stand) fehlt
    erg = _tick(fx)
    assert erg.uebersprungen is None and erg.commit is None  # alle Spaces gebaut, nichts geaendert
    _anstoss(fx)
    erg = _tick(fx)
    assert not fx.marker.exists() and erg.uebersprungen == "keine_aenderung"


def test_tick_torwaechter_blockiert_commit(fx):
    geseedet = _start(fx)
    head = _head(fx.repo)
    antworten = [set(), {("docs/spaces/KETTEN.md", "neuer Fehler")}]
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()), drift=lambda repo: antworten.pop(0))
    assert erg.commit is None and erg.uebersprungen == "torwaechter:torwaechter_blockiert"
    assert _head(fx.repo) == head and _text(fx) == geseedet and _status(fx.repo) == []
    assert any(p["art"] == "torwaechter_blockiert" for p in _pruefliste(fx)["offen"])


def test_tick_torwaechter_absturz_vor_dem_lauf_fragt_den_judge_nicht(fx):
    _start(fx)

    def kaputt(repo):
        raise umschreiben.TorwaechterFehler("Waechter abgestuerzt")

    lauf = FakeLaufzeit(_urteil(), _urteil())
    erg = _tick(fx, laufzeit=lauf, drift=kaputt)
    assert erg.uebersprungen == "torwaechter:torwaechter_fehler" and lauf.aufrufe == []


def test_tick_torwaechter_absturz_danach_fragt_beim_naechsten_mal_nicht_erneut(fx):
    """Item 3: das Urteil liegt im Cache; der naechste Lauf holt den Commit ohne LLM nach."""
    geseedet = _start(fx)
    aufrufe = [0]

    def zweiter_stuerzt(repo):
        aufrufe[0] += 1
        if aufrufe[0] == 2:
            raise umschreiben.TorwaechterFehler("Waechter abgestuerzt")
        return set()

    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()), drift=zweiter_stuerzt)
    assert erg.commit is None and erg.uebersprungen == "torwaechter:torwaechter_fehler" and erg.llm_aufrufe == 2
    assert _text(fx) == geseedet and _status(fx.repo) == []
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg2 = _tick(fx, laufzeit=lauf)
    assert lauf.aufrufe == [] and erg2.umgeschrieben == 1 and erg2.commit == _head(fx.repo)


def test_tick_committen_bekommt_nur_geschriebene_dateien(fx, monkeypatch):
    _start(fx)
    aufrufe = []
    echt = umschreiben.committen

    def spion(repo, dateien, nachricht, *, erwartet=None):
        aufrufe.append((list(dateien), dict(erwartet or {})))
        return echt(repo, dateien, nachricht, erwartet=erwartet)

    monkeypatch.setattr(umschreiben, "committen", spion)
    _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert len(aufrufe) == 1
    dateien, erwartet = aufrufe[0]
    assert sorted(dateien) == sorted([KETTEN, "docs/graph/graph.json", "docs/graph/ledger.jsonl"])
    assert list(erwartet) == [KETTEN]  # erwartet nur fuer *.md


def test_tick_schreibfehler_ueberspringt_ohne_commit(fx, monkeypatch):
    geseedet = _start(fx)
    head = _head(fx.repo)
    echt = umschreiben.schreiben

    def gesperrt(repo, datei, text, *, erwartet_alt=None):
        if datei == "docs/graph/graph.json":
            raise PermissionError("gesperrt")
        return echt(repo, datei, text, erwartet_alt=erwartet_alt)

    monkeypatch.setattr(umschreiben, "schreiben", gesperrt)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.commit is None and erg.uebersprungen == "schreiben:PermissionError"
    assert _head(fx.repo) == head and _text(fx) == geseedet and _status(fx.repo) == []
    monkeypatch.setattr(umschreiben, "schreiben", echt)
    _anstoss(fx)  # Commits nicht fortgeschrieben -> der naechste Tick holt es nach, Urteil aus dem Cache
    lauf = FakeLaufzeit()
    erg2 = _tick(fx, laufzeit=lauf)
    assert erg2.umgeschrieben == 1 and erg2.commit is not None and lauf.aufrufe == []


def test_tick_gitfehler_vor_dem_commit_setzt_zurueck(fx, monkeypatch):
    geseedet = _start(fx)
    head = _head(fx.repo)

    def scheitert(repo, dateien, nachricht, *, erwartet=None):
        raise umschreiben.GitFehler("index.lock vorhanden")

    monkeypatch.setattr(umschreiben, "committen", scheitert)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.commit is None and erg.uebersprungen == "commit:GitFehler" and _head(fx.repo) == head
    assert _text(fx) == geseedet and _status(fx.repo) == []
    assert tk.stand_laden(fx.konfig).letzter_commit_aussen != head  # nicht fortgeschrieben -> wird nachgeholt


def test_tick_commit_unbestaetigt_vermerkt_und_setzt_nichts_zurueck(fx, monkeypatch):
    _start(fx)
    echt = umschreiben.committen

    def unbestaetigt(repo, dateien, nachricht, *, erwartet=None):
        sha = echt(repo, dateien, nachricht, erwartet=erwartet)
        raise umschreiben.CommitUnbestaetigt("git commit meldete Timeout", sha, True)

    monkeypatch.setattr(umschreiben, "committen", unbestaetigt)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.uebersprungen == "commit_unbestaetigt" and erg.commit == _head(fx.repo)
    assert NEUER_TEXT in _text(fx)  # nichts zurueckgesetzt
    assert any(p["art"] == "commit_unbestaetigt" and erg.commit[:7] in p["text"] for p in _pruefliste(fx)["offen"])
    assert _status(fx.repo) == []  # Item 4: nichts Uncommittetes vom Loop im Repo


def test_tick_untracked_beleg_gibt_kein_signal_und_kommt_einmal_in_die_pruefliste(fx):
    """Item 6: ignorierte/untracked Dateien (wie logs/...) taugen nicht als Beleg.

    Ruling 50 (15c, angepasst): eine nie committete Datei liegt nicht im Messbaum und wird gar nicht erst verankert.
    Der Hinweis `beleg_untracked` (zweite Linie, Ruling 48a/Item 6) kommt fuer gemerkte Belege, deren Datei spaeter
    aus git genommen wurde und nur noch lokal liegt -- im Messbaum fehlt sie, trotzdem kein `verschwunden`-Signal."""
    _schreib(fx.repo, "logs/zustand.json", '{"a": 1}\n')
    _schreib(fx.repo, KETTEN, KETTEN_MD + "\nDen Zustand legt der Dienst in `logs/zustand.json:1` ab und liest ihn"
                                          " beim Start.\n")
    _git(fx.repo, "add", "-f", "logs/zustand.json")
    _commit(fx.repo, "zustand", KETTEN)
    _seed(fx)
    assert any(b.file == "logs/zustand.json" for k in Graph.laden(fx.repo / "docs/graph/graph.json").kanten.values()
               if k.art == "belegt_durch" for b in k.belege)
    _git(fx.repo, "rm", "-q", "--cached", "logs/zustand.json")  # nur noch lokal (ignoriert)
    _git(fx.repo, "commit", "-q", "-m", "zustand nur lokal")
    for i in range(2):
        _schreib(fx.repo, "logs/zustand.json", f'{{"a": {i + 2}}}\n')
        _code_aenderung(fx, i)
        _anstoss(fx)
        lauf = FakeLaufzeit()
        erg = _tick(fx, laufzeit=lauf)
        assert lauf.aufrufe == [] and erg.commit is None
    untracked = [p for p in _pruefliste(fx)["offen"] if p["art"] == "beleg_untracked"]
    assert len(untracked) == 1 and untracked[0]["datei"] == "logs/zustand.json"


def test_tick_geaenderte_belegdatei_ausserhalb_der_spaces_ohne_neubau(fx):
    """Item 7: eine geaenderte Datei ausserhalb der Spaces, auf die ein Beleg zeigt, fuehrt zur Pruefung."""
    _schreib(fx.repo, "tools/helfer.py", "def pruefe():\n    return 1\n")
    _schreib(fx.repo, KETTEN, KETTEN_MD + "\nVor dem Versand läuft `pruefe` aus `tools/helfer.py:1` und stoppt"
                                          " fehlerhafte Aufträge.\n")
    _commit(fx.repo, "helfer", "tools/helfer.py", KETTEN)
    _seed(fx)
    _schreib(fx.repo, "tools/helfer.py", "def pruefe():\n    return 2\n")
    _commit(fx.repo, "helfer geaendert", "tools/helfer.py")
    _anstoss(fx)
    bauer, lauf = FakeManifestBauer(), FakeLaufzeit(_urteil("noch_wahr"), _urteil("noch_wahr"))
    erg = _tick(fx, bauer=bauer, laufzeit=lauf)
    assert erg.spaces == [] and bauer.gebaut == []
    assert erg.signale == 1 and erg.llm_aufrufe == 2 and erg.umgeschrieben == 0
    assert erg.commit is not None and _namen(fx.repo) == {"docs/graph/graph.json"}  # Beleg nachgefuehrt


def test_tick_sperre_verhindert_zweiten_lauf(fx):
    _seed(fx)
    sperre = fx.marker.parent / "tick.lock"
    sperre.write_text("4711 frisch", encoding="utf-8")
    bauer = FakeManifestBauer()
    erg = _tick(fx, bauer=bauer)
    assert erg.uebersprungen == "laeuft_schon" and bauer.gebaut == [] and erg.commit is None
    assert sperre.read_text(encoding="utf-8") == "4711 frisch"  # fremde Sperre bleibt
    assert tk.sperre_zustand(fx.marker.parent) == "belegt"
    alt = time.time() - 31 * 60
    os.utime(sperre, (alt, alt))
    assert tk.sperre_zustand(fx.marker.parent).startswith("veraltet (seit 31 min)")
    erg = _tick(fx, bauer=bauer)  # veraltet -> per Umbenennen uebernommen
    assert erg.uebersprungen != "laeuft_schon" and not sperre.exists()
    assert list(fx.marker.parent.glob("tick.lock*")) == []  # keine Reste der Uebernahme
    assert tk.sperre_zustand(fx.marker.parent) == "frei"


def test_sperre_uebernahme_legt_eine_frische_sperre_zurueck(fx, monkeypatch):
    """Item 8: wer beim Umbenennen eine frische Sperre erwischt (ein anderer war schneller), gibt sie zurueck."""
    ordner = fx.marker.parent
    ordner.mkdir(parents=True)
    pfad = ordner / "tick.lock"
    pfad.write_text("fremd", encoding="utf-8")
    alt = time.time() - 31 * 60
    os.utime(pfad, (alt, alt))
    echt = os.rename

    def dazwischen(quelle, ziel):  # zwischen stat und Umbenennen legt ein anderer Prozess eine frische Sperre an
        if Path(quelle) == pfad:
            pfad.write_text("anderer gewinnt", encoding="utf-8")
            os.utime(pfad, None)
        return echt(quelle, ziel)

    monkeypatch.setattr(st.os, "rename", dazwischen)
    with st.sperre(ordner) as frei:
        assert frei is False
    assert pfad.read_text(encoding="utf-8") == "anderer gewinnt"


def test_tick_sperre_haelt_waehrend_des_laufs(fx):
    _seed(fx)
    _umbenennen(fx)
    _anstoss(fx)
    gesehen = []

    def bauer(space, konfig, commit):
        gesehen.append(tk.tick(fx.konfig, jetzt=_jetzt(), manifest_bauer=FakeManifestBauer(),
                               drift=lambda r: set(), budget=FakeBudget()).uebersprungen)
        return _manifest(konfig.repo / space.wurzel, commit)

    _tick(fx, bauer=bauer, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert gesehen and set(gesehen) == {"laeuft_schon"}


def test_tick_konfig_aenderung_baut_alle_spaces(fx):
    _seed(fx)
    _schreib(fx.repo, "config/dienste.yaml", DIENSTE + "  z:\n    port: 9000\n    ort: pc\n    space: null\n"
                                                      "    env: []\n")
    _commit(fx.repo, "register", "config/dienste.yaml")
    _anstoss(fx)
    bauer = FakeManifestBauer()
    erg = _tick(fx, bauer=bauer)
    assert erg.spaces == ["a", "b"] and sorted(bauer.gebaut) == ["a", "b"]


def test_tick_hand_edit_einmal_je_hash(fx):
    _seed(fx)
    text = _text(fx)
    assert "<!-- graph:begin hops K1 -->" in text
    _schreib(fx.repo, KETTEN, text.replace("HTTP POST `/api/x`", "HTTP POST `/api/x` (von Hand)"))
    _commit(fx.repo, "hand", KETTEN)
    for i in range(2):
        _code_aenderung(fx, i)
        _anstoss(fx)
        _tick(fx)
    hand = [p for p in _pruefliste(fx)["offen"] if p["art"] == "hand_edit"]
    assert len(hand) == 1
    assert "hops K1" in tk.stand_laden(fx.konfig).hand_edit_gemeldet
    assert "`/api/x` (von Hand)" in _text(fx)


def test_tick_raeumt_nur_eigene_alte_temp_reste_auf(fx):
    """Item 9: nur das exakte Temp-Muster des Loops, nur aelter als 30 min."""
    _seed(fx)
    alt = time.time() - 31 * 60
    dateien = {"docs/graph/.graph.json.ab12cd34.tmp": True, "docs/graph/.ledger.jsonl.zz99yy88.tmp": True,
               "docs/spaces/.KETTEN.md.q1w2e3r4.tmp": True, "docs/graph/.graph.json.frisch01.tmp": False,
               "docs/graph/fremd.tmp": False, "docs/graph/.graph.json.tmp": False}
    for rel in dateien:
        _schreib(fx.repo, rel, "x")
        if "frisch" not in rel:
            os.utime(fx.repo / rel, (alt, alt))
    weg = st.temp_aufraeumen(fx.konfig)
    for rel, soll_weg in dateien.items():
        assert (fx.repo / rel).exists() is not soll_weg, rel
    assert len(weg) == 3


def test_tick_nach_tick_hook_bekommt_ergebnis(fx):
    _seed(fx)
    _anstoss(fx)
    gesehen = []
    erg = _tick(fx, nach_tick=gesehen.append)
    assert gesehen == [erg]


def test_tick_meldet_eine_zeile_je_schritt_ohne_inhalte(fx):
    _start(fx)
    zeilen = []
    _tick(fx, melden=zeilen.append, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert zeilen and all("\n" not in z for z in zeilen)
    assert not any("senden" in z or "Dienst b" in z for z in zeilen)  # keine Dateiinhalte


# ---------- Fix-Runde 2 (Ruling 48) ----------

@pytest.mark.parametrize("ziel", ["docs/graph/graph.json", KETTEN])
def test_cas_basis_ist_der_stand_beim_laden(fx, ziel):
    """Item 1: aendert jemand graph.json oder KETTEN.md waehrend Neubau/Judge, wird nichts ueberschrieben."""
    _start(fx)
    head = _head(fx.repo)
    fremd = (fx.repo / ziel).read_bytes() + b"\n"

    def bauer_mit_fremder_aenderung(space, konfig, commit):
        (fx.repo / ziel).write_bytes(fremd)  # nach dem Laden der Basis und der Fremd-Pruefung
        return _manifest(konfig.repo / space.wurzel, commit)

    erg = _tick(fx, bauer=bauer_mit_fremder_aenderung, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.commit is None and erg.uebersprungen == "schreiben:FremdeAenderung"
    assert _head(fx.repo) == head and (fx.repo / ziel).read_bytes() == fremd
    assert [z[3:] for z in _status(fx.repo)] == [ziel]  # nur die fremde Aenderung, nichts vom Loop


def test_unsauberer_beleg_gibt_kein_signal_bis_zum_commit(fx):
    """Ruling 48a: eine nicht committete Aenderung an einer belegten Funktion -> kein Judge, kein Commit."""
    _seed(fx)
    _schreib(fx.repo, "sp/a/notiz.py", "# nur ein Kommentar\n")
    _commit(fx.repo, "notiz", "sp/a/notiz.py")  # baut Space a neu, ohne Symbole zu aendern
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("json={})", 'json={"neu": 1})'))  # uncommittet
    _anstoss(fx)
    head, vorher = _head(fx.repo), Graph.laden(fx.repo / "docs/graph/graph.json")
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf)
    assert erg.spaces == ["a"] and lauf.aufrufe == [] and erg.signale == 0
    assert erg.commit is None and _head(fx.repo) == head
    unsauber = [p for p in _pruefliste(fx)["offen"] if p["art"] == "beleg_unsauber"]
    assert [p["datei"] for p in unsauber] == ["sp/a/client.py"]
    nachher = Graph.laden(fx.repo / "docs/graph/graph.json")
    assert nachher.digest() == vorher.digest()  # gemerkte Belege unveraendert
    _commit(fx.repo, "senden geaendert", "sp/a/client.py")  # jetzt committet -> Signal
    _anstoss(fx)
    lauf2 = FakeLaufzeit(_urteil("noch_wahr"), _urteil("noch_wahr"))
    erg2 = _tick(fx, laufzeit=lauf2)
    assert erg2.signale >= 1 and len(lauf2.aufrufe) == 2 and erg2.commit is not None


def test_eigene_dateien_des_loops_sind_nie_belegziel(fx):
    """Ruling 48b: KETTEN.md und docs/graph/* erzeugen nie ein Signal fuer den Loop selbst."""
    _schreib(fx.repo, "docs/graph/README.md", "# Graph\n\nErklaerung.\n")
    _schreib(fx.repo, KETTEN, KETTEN_MD + "\nDie Erklärung der Dateien steht in `docs/graph/README.md:1` und in "
                                          "`docs/spaces/KETTEN.md:1` oben.\n")
    _commit(fx.repo, "selbstbezug", "docs/graph/README.md", KETTEN)
    _seed(fx)
    g = Graph.laden(fx.repo / "docs/graph/graph.json")
    ziele = {b.file for k in g.kanten.values() if k.art in ("hop", "belegt_durch") for b in k.belege}
    assert not any(z == KETTEN or z.startswith("docs/graph/") for z in ziele)
    _schreib(fx.repo, "docs/graph/README.md", "# Graph\n\nNeu erklaert.\n")
    _commit(fx.repo, "readme", "docs/graph/README.md")
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf)
    assert erg.uebersprungen == "keine_aenderung" and lauf.aufrufe == []
    # ein Loop-Commit (KETTEN.md + graph.json) loest beim naechsten Mal ebenfalls nichts aus
    _umbenennen(fx)
    _anstoss(fx)
    assert _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil())).umgeschrieben == 1
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf)
    assert erg.uebersprungen == "keine_aenderung" and lauf.aufrufe == []


class QuotenBudget(FakeBudget):
    """Erlaubt nur die ersten `n` Pruefungen."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def pruefen(self, agent, *, jetzt=None, umgebung=None):
        self.pruefungen.append(agent)
        erlaubt = len(self.pruefungen) <= self.n
        return SimpleNamespace(erlaubt=erlaubt, grund="ok" if erlaubt else "tagesgrenze", agent=agent, heute=1,
                               grenze=30)


def test_offenes_signal_wird_je_lauf_nur_einmal_gefragt(fx):
    """Item 4: ein offenes Signal, das in `_offene_fragen` erneut scheitert, fragt `_voll` im selben Lauf nicht."""
    zweite = "Auch die Wiederholung nutzt `senden` in `sp/a/client.py:7` für den zweiten Versuch."
    _schreib(fx.repo, KETTEN, KETTEN_MD + "\n" + zweite + "\n")
    _commit(fx.repo, "zweite aussage", KETTEN)
    _start(fx)
    _tick(fx, budget=FakeBudget(erlaubt=False, grund="tagesgrenze"))
    assert len(tk.stand_laden(fx.konfig).offene_signale) == 2
    _anstoss(fx)
    budget = QuotenBudget(2)  # die erste offene Frage geht durch (zwei Befragungen), die zweite nicht
    erg = _tick(fx, budget=budget, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.umgeschrieben == 1 and erg.llm_aufrufe == 2
    assert len(budget.pruefungen) == 3  # ohne die Sperre je Lauf fragte der Neubau die zweite erneut
    offen = tk.stand_laden(fx.konfig).offene_signale
    assert len(offen) == 1 and offen[0]["aussage"]["text"] == zweite


# ---------- Task 15b (Ruling 49): schlanker Repo-Graph, voller Graph lokal ----------

GRAPH = "docs/graph/graph.json"
SYMBOL_ARTEN = {"ruft_auf", "gehoert_zu", "definiert_route", "liest_schreibt", "registriert", "nutzt_dienst"}


class AufrufBauer(FakeManifestBauer):
    """Wie FakeManifestBauer, aber in Space a ruft `senden` jetzt `helfer` auf (nur eine neue `ruft_auf`-Kante)."""

    def __call__(self, space, konfig, commit):
        m = super().__call__(space, konfig, commit)
        if space.name == "a":
            m.callees_by_document = {"client.senden": {"client.helfer"}}
        return m


def test_seed_schreibt_voll_lokal_und_schlank_ins_repo(fx):
    erg = _seed(fx)
    assert erg.commit == _head(fx.repo) and _namen(fx.repo) == {KETTEN, GRAPH}
    pfad = st.vollgraph_pfad(fx.konfig)
    assert pfad.parent == fx.lokal / "VibeMind" / "doku_loop"
    assert pfad.name == f"graph-voll-{st.repo_schluessel(fx.repo)}.json"
    voll, repo_g = Graph.laden(pfad), Graph.laden(fx.repo / GRAPH)
    assert any(k.typ == "symbol" for k in voll.knoten.values())
    assert any(e.art == "gehoert_zu" for e in voll.kanten.values())
    assert {e.art for e in repo_g.kanten.values()} <= {"hop", "belegt_durch", "verbindet"}
    assert any(e.art == "hop" for e in repo_g.kanten.values())
    assert any(k.typ == "aussage" for k in repo_g.knoten.values())  # Aussage-Belege vollstaendig im Repo
    assert repo_g.digest() == schlank(voll).digest()
    assert pfad.stat().st_size > (fx.repo / GRAPH).stat().st_size
    assert list(pfad.parent.glob("*.tmp")) == []  # atomar geschrieben, keine Reste
    assert _status(fx.repo) == []
    assert "graph-voll" not in _git(fx.repo, "log", "--all", "--name-only", "--format=")


def test_tick_commit_nur_bei_aenderung_im_schlanken_graph(fx):
    """Ruling 49 + 45: nur `ruft_auf` geaendert -> der volle Graph aendert sich, der schlanke nicht -> kein Commit."""
    _seed(fx)
    repo_vorher = (fx.repo / GRAPH).read_bytes()
    _schreib(fx.repo, "sp/a/client.py", CLIENT + "\n# nur ein Kommentar\n")
    code = _commit(fx.repo, "kommentar", "sp/a/client.py")
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf, bauer=AufrufBauer())
    assert erg.spaces == ["a"] and lauf.aufrufe == [] and erg.uebersprungen is None
    assert erg.commit is None and _head(fx.repo) == code
    assert (fx.repo / GRAPH).read_bytes() == repo_vorher and _status(fx.repo) == []
    voll = Graph.laden(st.vollgraph_pfad(fx.konfig))
    assert "ruft_auf:symbol:a:client.senden->symbol:a:client.helfer" in voll.kanten  # lokal nachgefuehrt
    assert not any(e.art in SYMBOL_ARTEN for e in Graph.laden(fx.repo / GRAPH).kanten.values())


def test_tick_ohne_lokalen_vollgraph_rekonstruiert_ohne_llm(fx):
    """Neuer Rechner / Datei geloescht: Rekonstruktion aus dem schlanken Graph + Neubau aller Spaces, kein Judge."""
    _seed(fx)
    pfad = st.vollgraph_pfad(fx.konfig)
    pfad.unlink()
    _schreib(fx.repo, "README.md", "anders\n")
    neu = _commit(fx.repo, "fremd", "README.md")  # keine Space-Aenderung: ohne Vollgraph trotzdem alle bauen
    _anstoss(fx)
    lauf, budget, bauer = FakeLaufzeit(), FakeBudget(), FakeManifestBauer()
    erg = _tick(fx, laufzeit=lauf, budget=budget, bauer=bauer)
    assert erg.spaces == ["a", "b"] and sorted(bauer.gebaut) == ["a", "b"]
    assert erg.llm_aufrufe == 0 and lauf.aufrufe == [] and budget.pruefungen == []
    assert erg.commit is None and _head(fx.repo) == neu and _status(fx.repo) == []
    assert pfad.is_file()
    voll = Graph.laden(pfad)
    assert any(k.typ == "symbol" for k in voll.knoten.values())
    assert schlank(voll).digest_stabil() == Graph.laden(fx.repo / GRAPH).digest_stabil()
    # ein Signal danach geht wie gewohnt (Aussage-Belege kommen aus dem schlanken Graph)
    _umbenennen(fx)
    _anstoss(fx)
    erg2 = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg2.umgeschrieben == 1 and erg2.commit == _head(fx.repo)


def test_tick_kaputter_lokaler_vollgraph_gilt_als_fehlend(fx):
    _seed(fx)
    st.vollgraph_pfad(fx.konfig).write_bytes(b"{kaputt")
    _anstoss(fx)
    bauer = FakeManifestBauer()
    erg = _tick(fx, bauer=bauer)
    assert sorted(bauer.gebaut) == ["a", "b"] and erg.commit is None
    assert any(k.typ == "symbol" for k in Graph.laden(st.vollgraph_pfad(fx.konfig)).knoten.values())


def test_tick_schlanker_repo_graph_gewinnt_gegen_veralteten_vollgraph(fx):
    """Eine andere Maschine hat Aussagen im Repo nachgefuehrt: der lokale Vollgraph liefert nur die Symbol-Daten."""
    _seed(fx)
    veraltet = st.vollgraph_pfad(fx.konfig).read_bytes()  # gemerkte Belege mit dem alten Digest von `senden`
    geaendert = CLIENT.replace("json={})", 'json={"neu": 1})')
    _schreib(fx.repo, "sp/a/client.py", geaendert)
    _commit(fx.repo, "senden geaendert", "sp/a/client.py")
    _anstoss(fx)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil("noch_wahr"), _urteil("noch_wahr")))
    assert erg.llm_aufrufe == 2 and erg.commit is not None  # Belege im Repo-Graph nachgefuehrt
    # lokal veraltete Hops/Aussagen (z. B. Lauf auf einer anderen Maschine) bei gueltiger Kennung: nur das Overlay
    # mit dem Repo-Graph schuetzt hier, nicht die Rekonstruktion (Fix-Runde 1: Datei mit fremder Kennung gilt als fehlend)
    kennung = tk.stand_laden(fx.konfig).vollgraph_kennung
    st.vollgraph_schreiben(fx.konfig, Graph.aus_text(veraltet.decode("utf-8")), "2026-10-28T12:00:00+00:00",
                           kennung=kennung)
    st.urteile_pfad(fx.konfig).unlink()  # ohne Urteils-Cache wuerde ein falsches Signal den Judge fragen
    _schreib(fx.repo, "sp/a/client.py", geaendert + "\n# Kommentar\n")
    _commit(fx.repo, "kommentar", "sp/a/client.py")
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg2 = _tick(fx, laufzeit=lauf)
    assert lauf.aufrufe == [] and erg2.signale == 0 and erg2.commit is None
    assert erg2.spaces == ["a"]  # inkrementell (kein Neubau aller Spaces): der Vollgraph galt


# ----- Fix-Runde 1 zu 15b -----

NUTZT = "nutzt_dienst:symbol:a:client.senden->dienst:b"
HOP = "hop:"


def _hop(g):
    return next(e for e in g.kanten.values() if e.art == "hop")


def test_tick_voller_repo_graph_ersetzt_frische_symbol_daten_nicht(fx):
    """Item 1: liegt im Repo noch der volle Graph (Seed 91bcc5f), duerfen seine veralteten Symbol-Daten die frischen
    lokalen nicht ersetzen -- sonst kippt der Hop-Zustand beim Neubau eines anderen Space."""
    _seed(fx)
    voll = Graph.laden(st.vollgraph_pfad(fx.konfig))
    zustand = _hop(voll).attribute["zustand"]
    assert zustand["verdrahtet"] == "ja" and voll.kanten[NUTZT].attribute["pfade"] == ["/api/x"]
    veraltet = Graph.laden(st.vollgraph_pfad(fx.konfig))
    veraltet.kanten[NUTZT].attribute["pfade"] = ["/api/alt"]  # aelterer Code: kein Routentreffer
    veraltet.speichern(fx.repo / GRAPH, "2026-10-28T12:00:00+00:00")
    _commit(fx.repo, "voller graph", GRAPH)
    _code_aenderung(fx, 1)  # nur Space b wird neu gebaut, Space a kommt aus der Basis
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.spaces == ["b"] and erg.commit is None
    nachher = Graph.laden(st.vollgraph_pfad(fx.konfig))
    assert nachher.kanten[NUTZT].attribute["pfade"] == ["/api/x"]  # frisch, nicht aus dem Repo
    assert _hop(nachher).attribute["zustand"]["verdrahtet"] == "ja"


def test_tick_vollgraph_mit_anderer_kennung_wird_rekonstruiert(fx):
    """Item 2: eine lokale Datei, die nicht zum Stand passt (alter Stand, andere Maschine), gilt als fehlend."""
    _seed(fx)
    alt = st.vollgraph_pfad(fx.konfig).read_bytes()
    _code_aenderung(fx, 1)
    _anstoss(fx)
    assert _tick(fx).spaces == ["b"]  # schreibt einen neuen Vollgraph mit neuer Kennung
    st.vollgraph_pfad(fx.konfig).write_bytes(alt)  # alte Datei zurueck (Kennung des Seeds)
    _code_aenderung(fx, 2)
    _anstoss(fx)
    lauf, budget, bauer = FakeLaufzeit(), FakeBudget(), FakeManifestBauer()
    head = _head(fx.repo)
    erg = _tick(fx, laufzeit=lauf, budget=budget, bauer=bauer)
    assert sorted(bauer.gebaut) == ["a", "b"] and erg.spaces == ["a", "b"]
    assert lauf.aufrufe == [] and budget.pruefungen == [] and erg.commit is None and _head(fx.repo) == head
    doc = json.loads(st.vollgraph_pfad(fx.konfig).read_bytes())
    assert doc["kennung"] == tk.stand_laden(fx.konfig).vollgraph_kennung == f"{head}::{erg.lauf}"


def test_tick_schreibfehler_vollgraph_verwirft_die_alte_datei(fx, monkeypatch):
    """Item 2a: beim Schreibfehler bleibt kein alter Vollgraph als gueltig stehen; der naechste Lauf rekonstruiert."""
    _seed(fx)
    _code_aenderung(fx, 1)
    _anstoss(fx)

    def kaputt(*a, **kw):
        raise PermissionError("gesperrt")

    monkeypatch.setattr(tk, "vollgraph_schreiben", kaputt)
    zeilen = []
    erg = _tick(fx, melden=zeilen.append)
    assert erg.commit is None and not st.vollgraph_pfad(fx.konfig).exists()
    assert tk.stand_laden(fx.konfig).vollgraph_kennung is None
    assert any("alte Datei entfernt" in z for z in zeilen)
    monkeypatch.undo()
    _anstoss(fx)
    bauer, lauf = FakeManifestBauer(), FakeLaufzeit()
    erg2 = _tick(fx, bauer=bauer, laufzeit=lauf)
    assert sorted(bauer.gebaut) == ["a", "b"] and lauf.aufrufe == [] and erg2.commit is None
    assert st.vollgraph_pfad(fx.konfig).is_file()


def test_tick_uncommittete_aenderung_kommt_nie_in_den_vollgraph(fx):
    """Ruling 50 (ersetzt 15b Item 2b `unsauber_gebaut`): gebaut wird aus dem Messbaum, eine nicht committete
    Aenderung erreicht den Vollgraph gar nicht erst -- auch nicht nach ihrer Ruecknahme, und ohne Dauer-Neubau."""
    _seed(fx)
    _schreib(fx.repo, "sp/a/notiz.py", "# nur ein Kommentar\n")
    _commit(fx.repo, "notiz", "sp/a/notiz.py")  # baut Space a neu
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace('/api/x"', '/api/y"'))  # uncommittet: kein Routentreffer
    _anstoss(fx)
    head = _head(fx.repo)
    erg = _tick(fx)
    assert erg.spaces == ["a"] and erg.commit is None
    doc = json.loads(st.vollgraph_pfad(fx.konfig).read_bytes())
    assert "unsauber_gebaut" not in doc
    assert Graph.aus_doc(doc).kanten[NUTZT].attribute["pfade"] == ["/api/x"]  # committeter Code, nicht der Arbeitsbaum
    assert _hop(Graph.aus_doc(doc)).attribute["zustand"]["verdrahtet"] == "ja"
    _anstoss(fx)
    bauer = FakeManifestBauer()
    assert _tick(fx, bauer=bauer).uebersprungen == "keine_aenderung" and bauer.gebaut == []  # kein Dauer-Neubau
    _schreib(fx.repo, "sp/a/client.py", CLIENT)  # zurueckgenommen, kein Commit
    _anstoss(fx)
    assert _tick(fx).uebersprungen == "keine_aenderung" and _head(fx.repo) == head
    # Neubau eines anderen Space: die Hop-Bestaetigung nutzt die Daten von Space a aus dem Vollgraph -- sie
    # entsprechen dem committeten Code, der Zustand kippt nicht
    _code_aenderung(fx, 1)
    _anstoss(fx)
    erg3 = _tick(fx)
    assert erg3.spaces == ["b"] and erg3.commit is None
    assert _hop(Graph.laden(st.vollgraph_pfad(fx.konfig))).attribute["zustand"]["verdrahtet"] == "ja"
    assert "verdrahtet" in _text(fx) and "unbestimmt" not in _text(fx).split("### K1", 1)[1]


# ----- Task 15c (Ruling 50): Messbaum -----

def test_tick_uncommittete_aenderung_ohne_belegbezug_kein_commit(fx):
    """Das Szenario aus dem Re-Review 15b: Route in sp/b/api.py umbenannt und NICHT committet (kein Hop-Beleg zeigt
    dorthin), dazu ein unabhaengiger Commit in Space b -> kein Commit, Hop-Zustand unveraendert."""
    _seed(fx)
    graph_vorher, text_vorher = (fx.repo / GRAPH).read_bytes(), _text(fx)
    assert _hop(Graph.laden(fx.repo / GRAPH)).attribute["zustand"]["verdrahtet"] == "ja"
    _schreib(fx.repo, "sp/b/api.py", API.replace('"/api/x"', '"/api/z"'))  # uncommittet
    _schreib(fx.repo, "sp/b/notiz.py", "# unabhaengig\n")
    code = _commit(fx.repo, "notiz", "sp/b/notiz.py")
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf)
    assert erg.spaces == ["b"] and erg.uebersprungen is None and lauf.aufrufe == []
    assert erg.commit is None and _head(fx.repo) == code
    assert (fx.repo / GRAPH).read_bytes() == graph_vorher and _text(fx) == text_vorher
    voll = Graph.laden(st.vollgraph_pfad(fx.konfig))
    assert _hop(voll).attribute["zustand"]["verdrahtet"] == "ja"
    assert "route:POST /api/x@b" in voll.knoten and "route:POST /api/z@b" not in voll.knoten
    assert [z[3:] for z in _status(fx.repo)] == ["sp/b/api.py"]  # die fremde Aenderung bleibt unangetastet


def test_messbaum_fehler_ueberspringt_ohne_commit(fx, monkeypatch, capsys):
    from selbstbild.doku import messbaum

    _start(fx)  # Signal liegt an (Umbenennung): ohne Messbaum darf trotzdem kein Judge fragen
    head = _head(fx.repo)

    def kaputt(konfig):
        raise messbaum.MessbaumFehler("anlegen:.")

    monkeypatch.setattr(messbaum, "bereitstellen", kaputt)
    lauf, budget, bauer = FakeLaufzeit(_urteil(), _urteil()), FakeBudget(), FakeManifestBauer()
    erg = _tick(fx, laufzeit=lauf, budget=budget, bauer=bauer)
    assert erg.uebersprungen == "messbaum:anlegen:." and erg.commit is None and _head(fx.repo) == head
    assert lauf.aufrufe == [] and budget.pruefungen == [] and bauer.gebaut == [] and erg.llm_aufrufe == 0
    assert tk.stand_laden(fx.konfig).letzter_commit_aussen != head  # nicht fortgeschrieben -> wird nachgeholt
    w = {"budget": budget, "laufzeit": lauf, "drift": lambda r: set(), "manifest_bauer": bauer, "jetzt": _jetzt}
    _anstoss(fx)
    assert cli.main(["--konfig", str(fx.konfig_pfad), "tick"], werkzeuge=w) == 0  # Exit 0
    assert "messbaum:anlegen:." in capsys.readouterr().out
    assert cli.main(["--konfig", str(fx.konfig_pfad), "seed"], werkzeuge=w) == 0
    assert "messbaum:anlegen:." in capsys.readouterr().out and _head(fx.repo) == head
    monkeypatch.undo()
    _anstoss(fx)
    erg2 = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg2.umgeschrieben == 1 and erg2.commit == _head(fx.repo)  # danach holt der Loop alles nach


def _submodul(fx, tmp_path, rel, dateien):
    """Echtes Submodul (lokale Quelle) unter `rel` im Fixture-Repo, committet."""
    quelle = tmp_path / ("quelle_" + rel.replace("/", "_"))
    quelle.mkdir()
    _git(quelle, "init", "-q")
    for k, v in (("user.name", "Test"), ("user.email", "test@example.invalid"), ("core.autocrlf", "false"),
                 ("commit.gpgsign", "false")):
        _git(quelle, "config", k, v)
    for name, text in dateien.items():
        _schreib(quelle, name, text)
    _git(quelle, "add", "-A")
    _git(quelle, "commit", "-q", "-m", "start")
    _git(fx.repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(quelle), rel)
    _git(fx.repo, "commit", "-q", "-m", f"submodul {rel}")
    return quelle


def test_tick_belege_in_submodulen_messbar_bzw_nicht_messbar(fx, tmp_path):
    """Nachtrag 15c: initialisierte Submodule liegen als Worktree im Messbaum (Beleg bleibt gueltig, kein
    `verschwunden`); nach `deinit` sind Belege darauf nicht messbar: kein Signal, ein Pruefpunkt je Submodul."""
    _submodul(fx, tmp_path, "ext/lib", {"x.py": "def eins():\n    return 1\n"})
    _schreib(fx.repo, KETTEN, KETTEN_MD + "\nDie Hilfe `eins` in `ext/lib/x.py:1` liefert immer die Eins an den "
                                          "Client.\n")
    _commit(fx.repo, "aussage lib", KETTEN)
    _seed(fx)
    belege = {b.file for k in Graph.laden(fx.repo / GRAPH).kanten.values() if k.art == "belegt_durch"
              for b in k.belege}
    assert "ext/lib/x.py" in belege  # verankert ueber den Submodul-Worktree im Messbaum
    _code_aenderung(fx, 1)
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf, erzwingen=True)
    assert erg.signale == 0 and lauf.aufrufe == [] and erg.commit is None
    assert not [p for p in _pruefliste(fx)["offen"] if p["art"] in ("beleg_untracked", "beleg_nicht_messbar")]
    _git(fx.repo, "submodule", "deinit", "-q", "-f", "ext/lib")  # Checkout leer, Gitlink bleibt
    for i in range(2):
        _code_aenderung(fx, 2 + i)
        _anstoss(fx)
        lauf = FakeLaufzeit()
        erg = _tick(fx, laufzeit=lauf, erzwingen=True)
        assert erg.signale == 0 and lauf.aufrufe == [] and erg.commit is None
    offen = _pruefliste(fx)["offen"]
    assert [p["datei"] for p in offen if p["art"] == "beleg_nicht_messbar"] == ["ext/lib"]
    assert not [p for p in offen if p["art"] == "beleg_untracked"]


def test_cli_messen_stellt_messbaum_nur_unter_der_sperre_bereit(fx, monkeypatch, capsys):
    """15c Fix-Runde 1: `messen` legt/aktualisiert den Messbaum -> nur mit tick.lock; gehalten -> laeuft_schon."""
    from selbstbild.doku import messbaum

    _seed(fx)
    gerufen = []
    echt = messbaum.bereitstellen
    monkeypatch.setattr(messbaum, "bereitstellen", lambda k: gerufen.append(1) or echt(k))
    sperre = fx.marker.parent / "tick.lock"
    sperre.write_text("4711 frisch", encoding="utf-8")
    assert cli.main(["--konfig", str(fx.konfig_pfad), "messen"]) == 0
    assert "laeuft_schon" in capsys.readouterr().out and gerufen == []
    assert sperre.read_text(encoding="utf-8") == "4711 frisch"  # fremde Sperre bleibt
    sperre.unlink()
    assert cli.main(["--konfig", str(fx.konfig_pfad), "messen"]) == 0
    assert gerufen == [1] and "Seed-Deckung" in capsys.readouterr().out and not sperre.exists()


def _space_c_als_submodul(fx, tmp_path):
    """Space c, dessen Wurzel `sp/c` ein Submodul ist (wie `vibemind-os/spaces/sales-claw`), plus eine Aussage."""
    _submodul(fx, tmp_path, "sp/c", {"c.py": "def c():\n    return 1\n"})
    roh = yaml.safe_load(fx.konfig_pfad.read_text(encoding="utf-8"))
    roh["spaces"]["c"] = {"wurzel": "sp/c", "mcp_muster": "werkzeuge"}
    fx.konfig_pfad.write_text(yaml.safe_dump(roh, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _schreib(fx.repo, KETTEN, KETTEN_MD + "\nDer Dienst c rechnet mit `c` in `sp/c/c.py:1` und liefert die Eins "
                                          "zurück.\n")
    _commit(fx.repo, "space c", KETTEN, "config/doku_loop.yaml")
    fx.konfig = konfig_modul.laden(fx.konfig_pfad)


def test_tick_commit_im_verschachtelten_submodul_baut_den_space_neu(fx, tmp_path):
    """15c Fix-Runde 1, Punkt 3: ein Commit IM Submodul einer Space-Wurzel (ohne neuen Pin aussen) loest den Neubau
    dieses Space aus; Belege darin tragen den Commit des Submoduls, nicht den des aeusseren Repos."""
    _space_c_als_submodul(fx, tmp_path)
    _seed(fx)
    c_kopf = _git(fx.repo / "sp/c", "rev-parse", "HEAD").strip()
    assert c_kopf != _head(fx.repo)
    belege = [b for k in Graph.laden(fx.repo / GRAPH).kanten.values() if k.art == "belegt_durch"
              for b in k.belege if b.file == "sp/c/c.py"]
    assert belege and all(b.commit == c_kopf for b in belege)  # Commit des Submoduls
    assert tk.stand_laden(fx.konfig).submodul_commits == {"sp/c": c_kopf}
    _schreib(fx.repo / "sp/c", "neu.py", "def neu():\n    return 2\n")
    _commit(fx.repo / "sp/c", "neu im submodul", "neu.py")  # aussen kein neuer Pin
    neu_kopf = _git(fx.repo / "sp/c", "rev-parse", "HEAD").strip()
    _anstoss(fx)
    bauer, lauf = FakeManifestBauer(), FakeLaufzeit()
    erg = _tick(fx, bauer=bauer, laufzeit=lauf)
    assert erg.spaces == ["c"] and bauer.gebaut == ["c"] and lauf.aufrufe == []
    assert "symbol:c:neu.neu" in Graph.laden(st.vollgraph_pfad(fx.konfig)).knoten  # aus dem Submodul-HEAD gemessen
    assert tk.stand_laden(fx.konfig).submodul_commits == {"sp/c": neu_kopf}
    _anstoss(fx)
    assert _tick(fx).uebersprungen == "keine_aenderung"  # danach ruhig
    # alter Submodul-Commit unbekannt/unerreichbar -> Spaces unter dem Praefix gelten als geaendert
    stand = tk.stand_laden(fx.konfig)
    st.stand_schreiben(fx.konfig, dataclasses.replace(stand, submodul_commits={"sp/c": "0" * 40}))
    _anstoss(fx)
    bauer = FakeManifestBauer()
    erg = _tick(fx, bauer=bauer)
    assert erg.spaces == ["c"] and bauer.gebaut == ["c"]
    assert tk.betroffene_spaces(fx.konfig, [], ["sp/*"]) == ["a", "b", "c"]  # Praefix ueber den Space-Wurzeln
    assert tk.betroffene_spaces(fx.konfig, [], ["sp/c/x/*"]) == ["c"]  # Praefix unter der Space-Wurzel
    assert tk.betroffene_spaces(fx.konfig, [], ["anders/*"]) == []


@pytest.mark.parametrize("inhalt", [b"[]", b"42", b'"text"', b'{"kennung": 1}'])
def test_vollgraph_gueltiges_json_ohne_objekt_gilt_als_fehlend(fx, inhalt):
    _seed(fx)
    kennung = tk.stand_laden(fx.konfig).vollgraph_kennung
    st.vollgraph_pfad(fx.konfig).write_bytes(inhalt)
    assert st.vollgraph_laden(fx.konfig, kennung) is None
    _anstoss(fx)
    bauer = FakeManifestBauer()
    assert _tick(fx, bauer=bauer).commit is None and sorted(bauer.gebaut) == ["a", "b"]


def test_tick_neue_maschine_ohne_stand_und_ohne_vollgraph(fx):
    """Weder lokaler Stand noch Vollgraph (neuer Rechner): alle Spaces, kein Judge, kein Commit, beides danach da."""
    _seed(fx)
    shutil.rmtree(fx.marker.parent)
    head = _head(fx.repo)
    lauf, budget, bauer = FakeLaufzeit(), FakeBudget(), FakeManifestBauer()
    erg = _tick(fx, laufzeit=lauf, budget=budget, bauer=bauer)
    assert sorted(bauer.gebaut) == ["a", "b"] and lauf.aufrufe == [] and budget.pruefungen == []
    assert erg.commit is None and _head(fx.repo) == head and _status(fx.repo) == []
    stand = tk.stand_laden(fx.konfig)
    assert stand.lauf_nr == 1 and st.vollgraph_laden(fx.konfig, stand.vollgraph_kennung) is not None


def test_temp_aufraeumen_entfernt_alte_vollgraph_reste(fx):
    ordner = fx.marker.parent
    ordner.mkdir(parents=True)
    alt = time.time() - 31 * 60
    name = st.vollgraph_pfad(fx.konfig).name
    dateien = {f"{name}.tmp": True, "graph-voll-zzz.json.tmp": False, "fremd.tmp": False, name: False}
    for rel, _ in dateien.items():
        (ordner / rel).write_text("x", encoding="utf-8")
        os.utime(ordner / rel, (alt, alt))
    (ordner / "frisch").mkdir()
    weg = st.temp_aufraeumen(fx.konfig)
    for rel, soll_weg in dateien.items():
        assert (ordner / rel).exists() is not soll_weg, rel
    assert weg == [f"{name}.tmp"]


def test_cli_graph_schlank_schreibfehler_lokal_und_lauf_nr(fx, monkeypatch, capsys):
    _seed(fx)
    voll = Graph.laden(st.vollgraph_pfad(fx.konfig))
    voll.speichern(fx.repo / GRAPH, "2026-10-28T12:00:00+00:00")
    _commit(fx.repo, "voller graph", GRAPH)
    nr = tk.stand_laden(fx.konfig).lauf_nr
    from selbstbild.doku import seed_befehl

    def kaputt(*a, **kw):
        raise PermissionError("gesperrt")

    monkeypatch.setattr(seed_befehl, "vollgraph_schreiben", kaputt)
    w = {"drift": lambda r: set(), "jetzt": _jetzt}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "graph-schlank"], werkzeuge=w) == 0
    aus = capsys.readouterr().out
    assert "nicht geschrieben" in aus and "Commit " in aus
    assert _git(fx.repo, "log", "-1", "--format=%s").strip() == seed_befehl.SCHLANK_BETREFF  # Commit trotzdem
    stand = tk.stand_laden(fx.konfig)
    assert stand.lauf_nr == nr + 1 and stand.vollgraph_kennung is None
    assert not st.vollgraph_pfad(fx.konfig).exists()


def test_cli_graph_schlank_ersetzt_den_vollen_repo_graph(fx, capsys):
    _seed(fx)
    voll = Graph.laden(st.vollgraph_pfad(fx.konfig))
    voll.speichern(fx.repo / GRAPH, "2026-10-28T12:00:00+00:00")  # wie der produktive Seed (91bcc5f)
    _commit(fx.repo, "voller graph", GRAPH)
    st.vollgraph_pfad(fx.konfig).unlink()  # wird vor dem Verschlanken aus dem Repo-Graph lokal gesichert
    w = {"drift": lambda r: set(), "jetzt": _jetzt}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "graph-schlank"], werkzeuge=w) == 0
    betreff, *rest = _git(fx.repo, "log", "-1", "--format=%B").strip().split("\n")
    assert betreff == "docs(loop): graph.json schlank (nur Hops, Aussagen, Dienste)"
    assert any(z.startswith("Lauf: ") for z in rest)
    assert _namen(fx.repo) == {GRAPH} and _status(fx.repo) == []
    assert Graph.laden(fx.repo / GRAPH).digest() == schlank(voll).digest()
    assert Graph.laden(st.vollgraph_pfad(fx.konfig)).digest() == voll.digest()
    head = _head(fx.repo)
    assert cli.main(["--konfig", str(fx.konfig_pfad), "graph-schlank"], werkzeuge=w) == 0  # schon schlank
    assert _head(fx.repo) == head and "schon schlank" in capsys.readouterr().out
    _anstoss(fx)
    bauer = FakeManifestBauer()
    assert _tick(fx, bauer=bauer).uebersprungen == "keine_aenderung" and bauer.gebaut == []


def test_cli_graph_schlank_fremde_aenderung_ueberspringt(fx, capsys):
    _seed(fx)
    _schreib(fx.repo, GRAPH, (fx.repo / GRAPH).read_text(encoding="utf-8") + "\n")
    head = _head(fx.repo)
    assert cli.main(["--konfig", str(fx.konfig_pfad), "graph-schlank"],
                    werkzeuge={"drift": lambda r: set(), "jetzt": _jetzt}) == 0
    assert _head(fx.repo) == head and "fremde_aenderungen" in capsys.readouterr().out


# ---------- Schreibvorgang ----------

def test_zuruecksetzen_mit_cas_laesst_fremde_aenderungen_stehen(fx):
    _schreib(fx.repo, "docs/a.md", "fremd\n")  # nach unserem Schreiben ("neu") hat jemand anderes geaendert
    _schreib(fx.repo, "docs/b.md", "neu\n")
    _schreib(fx.repo, "docs/c.md", "fremd\n")
    meldungen = []
    sv.zuruecksetzen(fx.repo, {"docs/a.md": (b"alt\n", b"neu\n"), "docs/b.md": (None, b"neu\n"),
                               "docs/c.md": (None, b"neu\n")}, meldungen.append)
    assert (fx.repo / "docs/a.md").read_text(encoding="utf-8") == "fremd\n"
    assert not (fx.repo / "docs/b.md").exists()  # neu angelegt und unveraendert -> weg
    assert (fx.repo / "docs/c.md").read_text(encoding="utf-8") == "fremd\n"
    assert len(meldungen) == 2


# ---------- Seed und Render ----------

def test_seed_bettet_ein_und_committet(fx):
    erg = _seed(fx)
    assert erg.commit == _head(fx.repo) and erg.uebersprungen is None
    assert erg.hops == 1 and erg.verankert == 1
    betreff, *rest = _git(fx.repo, "log", "-1", "--format=%B").strip().split("\n")
    assert betreff == "docs(loop): KETTEN.md als Graph-Seed (1 Hops)" and any(z.startswith("Lauf: ") for z in rest)
    text = _text(fx)
    assert "<!-- graph:begin uebersicht -->" in text and "<!-- graph:begin hops K1 -->" in text
    assert AUSSAGE in text
    assert _namen(fx.repo) == {KETTEN, "docs/graph/graph.json"}
    stand = tk.stand_laden(fx.konfig)
    assert set(stand.render_hashes) == {"uebersicht", "hops K1"}
    assert _status(fx.repo) == []
    _anstoss(fx)
    assert _tick(fx).uebersprungen == "keine_aenderung"


def test_seed_fremde_aenderung_ueberspringt(fx):
    _schreib(fx.repo, KETTEN, KETTEN_MD + "\nfremd\n")
    erg = _seed(fx)
    assert erg.uebersprungen == f"fremde_aenderungen:{KETTEN}" and erg.commit is None


def test_seed_commit_fehler_setzt_zurueck(fx, monkeypatch):
    def scheitert(repo, dateien, nachricht, *, erwartet=None):
        raise umschreiben.GitFehler("index.lock vorhanden")

    monkeypatch.setattr(umschreiben, "committen", scheitert)
    erg = _seed(fx)
    assert erg.commit is None and erg.uebersprungen == "commit:GitFehler"
    assert _text(fx) == KETTEN_MD and _status(fx.repo) == [] and tk.stand_laden(fx.konfig) is None


def test_render_schreiben_committet_im_loop_format(fx, capsys):
    _seed(fx)
    g = Graph.laden(fx.repo / "docs/graph/graph.json")
    for k in g.kanten.values():
        if k.art == "hop":
            k.attribute["belegt_durch"] = "042"
    g.speichern(fx.repo / "docs/graph/graph.json", "2026-10-28T12:00:00+00:00")
    _commit(fx.repo, "graph von Hand", "docs/graph/graph.json")
    w = {"drift": lambda r: set(), "jetzt": _jetzt}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "render", "--schreiben"], werkzeuge=w) == 0
    betreff, *rest = _git(fx.repo, "log", "-1", "--format=%B").strip().split("\n")
    assert betreff == "docs(loop): KETTEN.md – 0 Aussagen nachgeführt, 0 in Prüfliste"
    assert any(z.startswith("Lauf: ") for z in rest)
    assert _namen(fx.repo) == {KETTEN} and "| 042 |" in _text(fx) and _status(fx.repo) == []


def test_render_ohne_graph_exit_0(fx, capsys):
    assert cli.main(["--konfig", str(fx.konfig_pfad), "render"]) == 0
    assert "kein_graph" in capsys.readouterr().out


# ---------- Hook ----------

HOOK = REPO / ".githooks" / "post-commit"
MARKER_ZEILE = ('marker_dir="${LOCALAPPDATA:-$HOME}/VibeMind/doku_loop"; '
                '{ mkdir -p "$marker_dir" && : > "$marker_dir/anstoss"; } 2>/dev/null || true')


@braucht_vm
def test_hook_setzt_marker():
    roh = HOOK.read_bytes()
    assert b"\r\n" not in roh  # LF bleibt
    zeilen = roh.decode("utf-8").split("\n")
    assert MARKER_ZEILE in zeilen
    nr = zeilen.index(MARKER_ZEILE)
    davor = [z for z in zeilen[1:nr] if z.strip()]
    assert all(z.lstrip().startswith("#") for z in davor)  # erste ausfuehrbare Anweisung nach dem Kopfkommentar
    erster_exit = next(i for i, z in enumerate(zeilen) if "exit" in z.split("#", 1)[0])
    assert nr < erster_exit


@pytest.mark.skipif(shutil.which("sh") is None, reason="kein sh")
@braucht_vm
def test_hook_laeuft_und_blockiert_nie(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    umg = dict(os.environ, LOCALAPPDATA=str(tmp_path / "lokal"), HOME=str(tmp_path))
    lauf = subprocess.run(["sh", str(HOOK)], cwd=repo, env=umg, capture_output=True, timeout=60)
    assert lauf.returncode == 0
    assert (tmp_path / "lokal" / "VibeMind" / "doku_loop" / "anstoss").is_file()
    (tmp_path / "datei").write_text("x", encoding="utf-8")  # Marker-Ordner nicht anlegbar
    umg["LOCALAPPDATA"] = str(tmp_path / "datei")
    lauf = subprocess.run(["sh", str(HOOK)], cwd=repo, env=umg, capture_output=True, timeout=60)
    assert lauf.returncode == 0 and lauf.stderr == b""


# ---------- CLI ----------

def test_cli_status_zeigt_stand_pruefliste_budget(fx, capsys):
    _seed(fx)
    code = cli.main(["--konfig", str(fx.konfig_pfad), "status"],
                    werkzeuge={"budget": FakeBudget(), "aufgabe_abfragen": lambda: "Bereit"})
    aus = capsys.readouterr().out
    assert code == 0
    assert "Stand: Lauf 1" in aus
    assert f"Prüfliste: {len(_pruefliste(fx)['offen'])} offen" in aus
    assert "Budget doku-loop: heute 3/30" in aus
    assert "Sperre: frei" in aus and "Aufgabe: Bereit" in aus


def test_cli_status_ohne_stand_und_kaputtes_budget(fx, capsys):
    class Kaputt:
        def tagesstand(self, **kw):
            raise ValueError("agent_budget.yaml: 'agenten' fehlt")

    code = cli.main(["--konfig", str(fx.konfig_pfad), "status"],
                    werkzeuge={"budget": Kaputt(), "aufgabe_abfragen": lambda: "nicht registriert"})
    aus = capsys.readouterr().out
    assert code == 0 and "noch kein Lauf" in aus and "Budget: nicht lesbar (ValueError)" in aus


def test_cli_konfigfehler_exit_2(tmp_path, capsys):
    kaputt = tmp_path / "config" / "doku_loop.yaml"
    kaputt.parent.mkdir()
    kaputt.write_text("spaces: {}\n", encoding="utf-8")
    assert cli.main(["--konfig", str(kaputt), "status"]) == 2
    assert cli.main(["--konfig", str(tmp_path / "fehlt.yaml"), "tick"]) == 2
    assert "Konfigfehler" in capsys.readouterr().err


def test_cli_tick_uebersprungen_exit_0(fx, capsys):
    w = {"budget": FakeBudget(), "laufzeit": FakeLaufzeit(), "drift": lambda r: set(),
         "manifest_bauer": FakeManifestBauer(), "jetzt": _jetzt}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "tick"], werkzeuge=w) == 0  # kein_seed
    assert cli.main(["--konfig", str(fx.konfig_pfad), "tick"], werkzeuge=w) == 0  # kein_anstoss
    aus = capsys.readouterr().out
    assert "kein_seed" in aus and "übersprungen" in aus and "kein_anstoss" in aus


def test_cli_tick_ohne_llm_flag(fx, monkeypatch):
    gesehen = {}

    def falsch_tick(konfig, **kw):
        gesehen.update(kw)
        return tk.TickErgebnis("l", [], 0, 0, 0, 0, None, "kein_anstoss")

    monkeypatch.setattr(tk, "tick", falsch_tick)
    assert cli.main(["--konfig", str(fx.konfig_pfad), "tick", "--erzwingen", "--ohne-llm"]) == 0
    assert gesehen["erzwingen"] is True and gesehen["ohne_llm"] is True


def test_cli_messen(fx, capsys):
    _seed(fx)
    assert cli.main(["--konfig", str(fx.konfig_pfad), "messen"]) == 0
    aus = capsys.readouterr().out
    assert "Belege:" in aus and "Seed-Deckung: 1 von 1 Hops verankert (100 %)" in aus
    assert "Ankerarten:" in aus
    assert cli.main(["--konfig", str(fx.konfig_pfad), "messen", "--json"]) == 0
    daten = json.loads(capsys.readouterr().out)
    assert daten["seed"]["hops"] == 1 and isinstance(daten["befunde"], list)


def test_cli_render_ohne_schreiben_aendert_nichts(fx, capsys):
    _seed(fx)
    vorher = (fx.repo / KETTEN).read_bytes()
    assert cli.main(["--konfig", str(fx.konfig_pfad), "render"]) == 0
    assert (fx.repo / KETTEN).read_bytes() == vorher
    assert "Render:" in capsys.readouterr().out


def test_cli_dienste_seed_ergaenzt_nur_fehlende(fx, capsys):
    vorher = (fx.repo / "config/dienste.yaml").read_text(encoding="utf-8")
    assert cli.main(["--konfig", str(fx.konfig_pfad), "dienste-seed"]) == 0
    nachher = (fx.repo / "config/dienste.yaml").read_text(encoding="utf-8")
    assert nachher.startswith(vorher)  # nichts ueberschrieben, Kommentare bleiben
    daten = yaml.safe_load(nachher)["dienste"]
    assert set(daten) == {"a", "b", "c", "openfang"}
    assert daten["b"]["port"] == 8130 and "# pruefen: von Hand gepflegt" in nachher
    assert cli.main(["--konfig", str(fx.konfig_pfad), "dienste-seed"]) == 0  # idempotent
    assert (fx.repo / "config/dienste.yaml").read_text(encoding="utf-8") == nachher
    assert "0 ergänzt" in capsys.readouterr().out


# ---------- Einstieg und Aufgabe ----------

@braucht_vm
def test_cmd_einstieg():
    text = (REPO / "scripts" / "doku-loop.cmd").read_text(encoding="utf-8")
    assert "PYTHONPATH=%~dp0" in text
    assert '"%~dp0..\\.venv\\Scripts\\python.exe" -m doku_loop %*' in text
    assert ">>" not in text  # Umleitung ins Log macht die Aufgabe, nicht der Einstieg


@braucht_vm
def test_aufgabe_skript():
    text = (REPO / "scripts" / "register-doku-loop-task.ps1").read_text(encoding="utf-8")
    for teil in ('"VibeMind Doku-Loop"', "-RepetitionInterval (New-TimeSpan -Minutes 5)", "-Once -At (Get-Date)",
                 "conhost.exe", "--headless", "scripts\\doku-loop.cmd tick", "-ExecutionTimeLimit",
                 "-MultipleInstances IgnoreNew", "logs\\doku-loop.log", "SupportsShouldProcess",
                 "Set-ScheduledTask", "Register-ScheduledTask"):
        assert teil in text, teil
    assert "Start-ScheduledTask" not in text  # registriert nur, startet nicht


# ---------- Final-Fix-Welle (Ruling 57) ----------

def _git_pfad(repo, name):
    p = Path(_git(repo, "rev-parse", "--git-path", name).strip())
    return p if p.is_absolute() else repo / p


def _konfig_mit(fx, **werte):
    roh = yaml.safe_load(fx.konfig_pfad.read_text(encoding="utf-8"))
    roh.update(werte)
    fx.konfig_pfad.write_text(yaml.safe_dump(roh, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _commit(fx.repo, "konfig", "config/doku_loop.yaml")
    fx.konfig = konfig_modul.laden(fx.konfig_pfad)
    return fx.konfig


class _UrteileZaehler:
    def __init__(self, monkeypatch):
        from selbstbild.doku import rowboat
        self.n = 0
        echt = rowboat.urteile_lesen

        def zaehlen(*a, **kw):
            self.n += 1
            return echt(*a, **kw)

        monkeypatch.setattr(rowboat, "urteile_lesen", zaehlen)


@pytest.mark.parametrize("zustand,grund", [
    ("detached", "kopf_detached"),
    ("rebase-merge", "rebase_laeuft"), ("rebase-apply", "rebase_laeuft"),
    ("MERGE_HEAD", "merge_laeuft"), ("CHERRY_PICK_HEAD", "merge_laeuft"), ("REVERT_HEAD", "merge_laeuft"),
    ("BISECT_LOG", "merge_laeuft"),
])
def test_c1_kein_commit_auf_detached_head_oder_laufender_git_operation(fx, monkeypatch, zustand, grund):
    """C1: vor den Urteilen und vor jedem Schreiben (tick, seed, render --schreiben, graph-schlank) ueberspringen."""
    geseedet = _start(fx)
    head = _head(fx.repo)
    if zustand == "detached":
        _git(fx.repo, "checkout", "-q", "--detach")
    else:
        pfad = _git_pfad(fx.repo, zustand)
        if zustand.startswith("rebase"):
            pfad.mkdir()
        else:
            pfad.write_text(head + "\n", encoding="utf-8")
    urteile = _UrteileZaehler(monkeypatch)
    lauf, bauer = FakeLaufzeit(_urteil(), _urteil()), FakeManifestBauer()
    erg = _tick(fx, laufzeit=lauf, bauer=bauer)
    assert erg.uebersprungen == grund and erg.commit is None
    assert lauf.aufrufe == [] and bauer.gebaut == [] and urteile.n == 0  # nichts gemessen, kein Judge, keine Urteile
    assert _head(fx.repo) == head and _text(fx) == geseedet and _status(fx.repo) == []
    assert _seed(fx).uebersprungen == grund
    assert tk.neu_rendern(fx.konfig, jetzt=_jetzt(), schreiben=True, drift=lambda r: set()).uebersprungen in (
        grund, None)  # ohne Aenderung gibt es nichts zu schreiben
    assert tk.graph_schlank(fx.konfig, jetzt=_jetzt(), drift=lambda r: set()).uebersprungen == grund
    assert _head(fx.repo) == head and _status(fx.repo) == []


def test_c1_schreibvorgang_selbst_prueft_den_kopf(fx):
    """Auch ein Schreiber, der den Kopf nicht vorher prueft, schreibt auf detached HEAD nichts."""
    _seed(fx)
    _git(fx.repo, "checkout", "-q", "--detach")
    alt = (fx.repo / KETTEN).read_bytes()
    aus = sv.schreiben_pruefen_committen(fx.repo, [sv.Aenderung(KETTEN, alt, alt.decode("utf-8") + "x\n")], "b", "l",
                                         vorher=set(), drift=lambda r: set(), melden=lambda z: None)
    assert aus.commit is None and aus.uebersprungen == "kopf_detached"
    assert (fx.repo / KETTEN).read_bytes() == alt and _status(fx.repo) == []


def test_c1_zweig_aus_der_konfig(fx):
    _seed(fx)
    zweig = _git(fx.repo, "symbolic-ref", "--short", "HEAD").strip()
    assert fx.konfig.zweig is None  # Vorgabe leer: jeder Branch
    _konfig_mit(fx, zweig="anderer-zweig")
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.uebersprungen == "falscher_zweig" and erg.commit is None
    assert _seed(fx).uebersprungen == "falscher_zweig"
    _konfig_mit(fx, zweig=zweig)
    _anstoss(fx)
    assert _tick(fx).uebersprungen != "falscher_zweig"


def test_c1_echte_konfig_nennt_den_zweig():
    assert vm_laden().zweig == "codex/vibemind-ops-baseline"


def test_c2_unerwartete_ausnahme_wird_pruefpunkt_und_haelt_den_zeiger(fx, monkeypatch, capsys):
    """C2: eine unerwartete Ausnahme -> Pruefpunkt `tick_fehler` (Typ + Ort, ohne Inhalt), Stand mit altem
    Commit-Zeiger gespeichert (der naechste Lauf kommt erst nach tick_minuten), Exit 1 bleibt."""
    from selbstbild.doku import bauen

    _seed(fx)
    alt = tk.stand_laden(fx.konfig)
    _code_aenderung(fx, 1)
    _anstoss(fx)

    def kaputt(*a, **kw):
        raise TypeError("'<' not supported between instances of 'str' and 'int' GEHEIM")

    monkeypatch.setattr(bauen, "graph_bauen", kaputt)
    with pytest.raises(TypeError):
        _tick(fx)
    stand = tk.stand_laden(fx.konfig)
    assert stand.letzter_commit_aussen == alt.letzter_commit_aussen  # Marker-Verbrauch ist kein Fortschritt
    assert stand.lauf_nr == alt.lauf_nr + 1 and stand.letzter_tick != alt.letzter_tick
    fehler = [p for p in _pruefliste(fx)["offen"] if p["art"] == "tick_fehler"]
    assert len(fehler) == 1 and "TypeError" in fehler[0]["text"] and "tick.py:" in fehler[0]["text"]
    assert "GEHEIM" not in fehler[0]["text"]
    _anstoss(fx)
    w = {"budget": FakeBudget(), "laufzeit": FakeLaufzeit(), "drift": lambda r: set(),
         "manifest_bauer": FakeManifestBauer(), "jetzt": _jetzt}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "tick"], werkzeuge=w) == 1
    assert len([p for p in _pruefliste(fx)["offen"] if p["art"] == "tick_fehler"]) == 1  # dedupliziert
    monkeypatch.undo()
    _anstoss(fx)
    assert _tick(fx).spaces == ["b"]  # der naechste Lauf holt die Aenderung nach


class _GitTimeout:
    """subprocess-Ersatz fuer schreibvorgang: `git <befehl>` laeuft in den Timeout, alles andere echt."""

    TimeoutExpired = subprocess.TimeoutExpired
    CompletedProcess = subprocess.CompletedProcess

    def __init__(self, befehl):
        self.befehl = befehl

    def run(self, argv, **kw):
        if self.befehl in argv[argv.index("-C") + 2:argv.index("-C") + 3]:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout") or 1)
        return subprocess.run(argv, **kw)


@pytest.mark.parametrize("befehl", ["status", "ls-files"])
def test_i3_git_fehler_fail_closed(fx, monkeypatch, befehl):
    """I3: ein git-Fehler in den Mengenbildnern (versioniert/unsauber) wird nie zur leeren Menge, der Tick
    ueberspringt mit `git_fehler:<befehl>`; nichts geschrieben, keine Pruefpunkte, Zeiger bleibt."""
    geseedet = _start(fx)
    head = _head(fx.repo)
    vorher = _pruefliste(fx)
    with pytest.raises(sv.GitFehler):
        monkeypatch.setattr(sv, "subprocess", _GitTimeout(befehl))
        (sv.unsaubere_dateien if befehl == "status" else sv.versionierte_dateien)(fx.repo)
    lauf = FakeLaufzeit(_urteil(), _urteil())
    erg = _tick(fx, laufzeit=lauf)
    assert erg.uebersprungen == f"git_fehler:{befehl}" and erg.commit is None and lauf.aufrufe == []
    assert _head(fx.repo) == head and _text(fx) == geseedet and _status(fx.repo) == []
    assert _pruefliste(fx) == vorher  # keine Pruefpunkt-Flut
    assert tk.stand_laden(fx.konfig).letzter_commit_aussen != head
    assert _seed(fx).uebersprungen == f"git_fehler:{befehl}" and _head(fx.repo) == head
    monkeypatch.undo()
    _anstoss(fx)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.umgeschrieben == 1 and erg.commit == _head(fx.repo)


def test_i5_sql_und_compose_aenderungen_bauen_alle_spaces(fx):
    """I5: eine geaenderte *.sql unter sql_wurzeln oder eine Datei aus compose_live loest den Neubau aller aus."""
    k = fx.konfig
    assert tk.betroffene_spaces(k, ["db/migrations/001_x.sql"], []) == ["a", "b"]
    assert tk.betroffene_spaces(k, [], ["vibemind-os/supabase/002_y.sql"]) == ["a", "b"]
    assert tk.betroffene_spaces(k, ["infra/swarm/vibemind-stack.yml"], []) == ["a", "b"]
    assert tk.betroffene_spaces(k, ["db/notiz.md", "README.md"], []) == []
    eng = dataclasses.replace(k, sql_wurzeln=("db",), compose_live=("sp/b/compose.yml",))
    assert tk.betroffene_spaces(eng, ["anders/x.sql"], []) == []
    assert tk.betroffene_spaces(eng, ["db/x.sql"], []) == ["a", "b"]
    assert tk.betroffene_spaces(eng, ["infra/swarm/vibemind-stack.yml"], []) == []
    # Ende zu Ende: neue SQL-Datei ausserhalb der Spaces -> beide Spaces neu gebaut
    _seed(fx)
    _schreib(fx.repo, "db/001_t.sql", "create table b.t (id int);\n")
    _commit(fx.repo, "sql", "db/001_t.sql")
    _anstoss(fx)
    bauer = FakeManifestBauer()
    erg = _tick(fx, bauer=bauer)
    assert erg.spaces == ["a", "b"] and sorted(bauer.gebaut) == ["a", "b"]


def test_i10_unsaubere_dateien_auch_in_verschachtelten_submodulen(fx, tmp_path):
    """I10 (Ruling 48a): nicht committete Aenderungen in jedem initialisierten Submodul, rekursiv, mit Praefix."""
    innen = tmp_path / "quelle_innen"
    innen.mkdir()
    _git(innen, "init", "-q")
    for k, v in (("user.name", "Test"), ("user.email", "test@example.invalid"), ("core.autocrlf", "false")):
        _git(innen, "config", k, v)
    _schreib(innen, "tief.py", "def tief():\n    return 1\n")
    _git(innen, "add", "-A")
    _git(innen, "commit", "-q", "-m", "start")
    quelle = _submodul(fx, tmp_path, "ext/lib", {"x.py": "def eins():\n    return 1\n"})
    _git(quelle, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(innen), "innen")
    _git(quelle, "commit", "-q", "-m", "innen")
    _git(fx.repo / "ext/lib", "pull", "-q")
    _commit(fx.repo, "pin", "ext/lib")
    _git(fx.repo, "-c", "protocol.file.allow=always", "submodule", "update", "-q", "--init", "--recursive")
    _schreib(fx.repo, "ext/lib/x.py", "def eins():\n    return 2\n")
    _schreib(fx.repo, "ext/lib/innen/tief.py", "def tief():\n    return 2\n")
    _schreib(fx.repo, "ext/lib/innen/neu.py", "x = 1\n")
    unsauber = sv.unsaubere_dateien(fx.repo)
    assert {"ext/lib/x.py", "ext/lib/innen/tief.py", "ext/lib/innen/neu.py"} <= unsauber
    assert not any(p.startswith("ext/lib/innen/") and p.count("innen") > 1 for p in unsauber)
    assert "ext/lib/innen/tief.py" in sv.versionierte_dateien(fx.repo)


def test_i6_geaenderter_hop_beleg_wird_sichtbar(fx, monkeypatch):
    """I6: ein geaenderter Hop-Beleg wird nicht mehr still nachgefuehrt: Pruefpunkt `hop_beleg_geaendert` mit
    Kette/Nr, Beleg, altem/neuem Digest und angezeigter Zeile; dedupliziert je Hop + neuem Digest; kein Judge."""
    _seed(fx)
    hop_alt = next(k for k in Graph.laden(fx.repo / GRAPH).kanten.values() if k.art == "hop")
    alt_digest = hop_alt.belege[0].digest
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("json={}", "json={'a': 1}"))
    _commit(fx.repo, "senden geaendert", "sp/a/client.py")
    _anstoss(fx)
    echt = umschreiben.schreiben

    def gesperrt(repo, datei, text, *, erwartet_alt=None):
        raise PermissionError("gesperrt")

    monkeypatch.setattr(umschreiben, "schreiben", gesperrt)  # erster Lauf: nicht committet -> Signal wiederholt
    lauf = FakeLaufzeit()
    _tick(fx, laufzeit=lauf, ohne_llm=True)
    monkeypatch.setattr(umschreiben, "schreiben", echt)
    _anstoss(fx)
    erg = _tick(fx, laufzeit=lauf, ohne_llm=True)
    assert lauf.aufrufe == [] and erg.commit is not None
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "hop_beleg_geaendert"]
    assert len(punkte) == 1  # zwei Laeufe, ein Punkt
    p = punkte[0]
    neu_digest = next(k for k in Graph.laden(fx.repo / GRAPH).kanten.values() if k.art == "hop").belege[0].digest
    assert neu_digest != alt_digest
    assert "K1/1" in p["text"] and "client.senden" in p["text"] and "sp/a/client.py:7" in p["text"]
    assert alt_digest[:12] in p["text"] and neu_digest[:12] in p["text"]
    assert p["datei"] == KETTEN and isinstance(p["zeile"], int) and p["aussage_id"] == f"K1/1#{neu_digest[:12]}"


def test_i8_veralteter_waechter_kommt_einmal_je_tag_in_die_pruefliste(fx):
    _schreib(fx.repo, "logs/w.json", json.dumps({"stand": {"probe": True}}))
    alt = time.time() - 3 * 3600
    os.utime(fx.repo / "logs/w.json", (alt, alt))
    _seed(fx)
    for i in range(2):
        _code_aenderung(fx, i)
        _anstoss(fx)
        _tick(fx)
    veraltet = [p for p in _pruefliste(fx)["offen"] if p["art"] == "waechter_veraltet"]
    assert len(veraltet) == 1


class _Uhr:
    """Monotone Uhr fuer die Tick-Deadline: der erste Aufruf (Start) liefert 0, alle weiteren `spaeter`."""

    def __init__(self, spaeter):
        self.spaeter, self.n = spaeter, 0

    def __call__(self):
        self.n += 1
        return 0.0 if self.n == 1 else float(self.spaeter)


def test_i7a_nach_der_deadline_keine_neuen_judge_fragen(fx):
    """I7a: nach `tick_deadline_min` Laufzeit fragt der Tick den Judge nicht mehr; das Signal bleibt offen und wird
    im naechsten Lauf gefragt."""
    _start(fx)
    assert fx.konfig.tick_deadline_min == 10
    lauf = FakeLaufzeit(_urteil(), _urteil())
    erg = _tick(fx, laufzeit=lauf, uhr=_Uhr(10 * 60 + 1))
    assert lauf.aufrufe == [] and erg.llm_aufrufe == 0 and erg.umgeschrieben == 0 and AUSSAGE in _text(fx)
    assert tk.stand_laden(fx.konfig).offene_spaces == ["a"]
    assert not [p for p in _pruefliste(fx)["offen"] if p["art"].startswith("judge_")]
    _anstoss(fx)
    erg2 = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()), uhr=_Uhr(9 * 60))
    assert erg2.llm_aufrufe == 2 and erg2.umgeschrieben == 1 and NEUER_TEXT in _text(fx)
    assert tk.stand_laden(fx.konfig).offene_signale == []


def _abbruch_beim_commit(monkeypatch):
    def abbruch(repo, dateien, nachricht, *, erwartet=None):
        raise KeyboardInterrupt  # wie ein Abbruch der Aufgabe (ExecutionTimeLimit): kein Ruecksetzen mehr

    monkeypatch.setattr(umschreiben, "committen", abbruch)


def test_i7b_eigene_halbschreibung_wird_zurueckgesetzt(fx, monkeypatch):
    """I7b: vor dem Schreiben steht `schreibt_gerade` (Pfade + Hashes) lokal; ein abgebrochener Lauf hinterlaesst ihn.
    Sind die Dateien noch genau diese Bytes, setzt der naechste Tick sie mit CAS auf HEAD zurueck (Pruefpunkt
    `eigene_halbschreibung_zurueckgesetzt`) und laeuft normal weiter."""
    geseedet = _start(fx)
    head = _head(fx.repo)
    _abbruch_beim_commit(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    vermerk = sv.schreibvermerk_pfad(fx.konfig)
    daten = json.loads(vermerk.read_text(encoding="utf-8"))
    assert set(daten["dateien"]) == {KETTEN, "docs/graph/graph.json", "docs/graph/ledger.jsonl"}
    assert daten["kopf"] == head and _status(fx.repo) != [] and NEUER_TEXT in _text(fx)
    monkeypatch.undo()
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf)
    assert any(p["art"] == "eigene_halbschreibung_zurueckgesetzt" for p in _pruefliste(fx)["offen"])
    assert lauf.aufrufe == []  # das Urteil lag schon im Cache
    assert erg.umgeschrieben == 1 and erg.commit == _head(fx.repo) != head and NEUER_TEXT in _text(fx)
    assert _status(fx.repo) == [] and not vermerk.exists()
    assert geseedet != _text(fx)


def test_i7b_veraenderte_halbschreibung_gilt_als_fremd(fx, monkeypatch):
    _start(fx)
    _abbruch_beim_commit(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    monkeypatch.undo()
    _schreib(fx.repo, KETTEN, _text(fx) + "\nNotiz von Hand.\n")  # jemand arbeitet weiter
    vorher = _text(fx)
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.uebersprungen.startswith("fremde_aenderungen") and _text(fx) == vorher
    assert not any(p["art"] == "eigene_halbschreibung_zurueckgesetzt" for p in _pruefliste(fx)["offen"])
    assert sv.schreibvermerk_pfad(fx.konfig).is_file()


def test_i7b_schreibvermerk_nach_commit_und_ruecksetzen_geloescht(fx, monkeypatch):
    _start(fx)
    vermerk = sv.schreibvermerk_pfad(fx.konfig)
    gesehen = []
    echt = umschreiben.committen

    def spion(repo, dateien, nachricht, *, erwartet=None):
        gesehen.append(vermerk.is_file())
        raise umschreiben.GitFehler("index.lock")

    monkeypatch.setattr(umschreiben, "committen", spion)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert gesehen == [True] and erg.uebersprungen == "commit:GitFehler" and not vermerk.exists()
    monkeypatch.setattr(umschreiben, "committen", echt)
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.commit is not None and not vermerk.exists()


def test_i11_fremder_commit_verschluckt_die_loop_schreibung_nicht(fx, monkeypatch):
    """I11: committet eine andere Session die geschriebenen Bytes, bevor der Loop committet (`committen` -> None),
    gilt das als uebernommen: Pruefpunkt `commit_fremd_uebernommen`, Render-Hashes trotzdem fortgeschrieben (kein
    falscher Hand-Edit danach), Ledger-Eintraege lokal mit `commit_fremd` vermerkt."""
    _start(fx)
    echt = umschreiben.committen
    fremd = []

    def andere_session(repo, dateien, nachricht, *, erwartet=None):
        _git(repo, "add", "--", *dateien)
        _git(repo, "commit", "-q", "-m", "andere Session: alles mitgenommen")
        fremd.append(_head(repo))
        return echt(repo, dateien, nachricht, erwartet=erwartet)

    monkeypatch.setattr(umschreiben, "committen", andere_session)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.commit is None and erg.uebersprungen == "commit_fremd_uebernommen" and _head(fx.repo) == fremd[0]
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "commit_fremd_uebernommen"]
    assert len(punkte) == 1 and fremd[0][:7] in punkte[0]["text"]
    stand = tk.stand_laden(fx.konfig)
    from selbstbild.doku import render_ketten
    bloecke = render_ketten.bloecke_finden(_text(fx))
    assert stand.render_hashes == {b: render_ketten.block_hash(x.inhalt) for b, x in bloecke.items()}
    vermerke = st.fremd_commits_laden(fx.konfig)
    assert vermerke and all(v == fremd[0] for v in vermerke.values()) and erg.lauf in vermerke
    zeilen = [json.loads(z) for z in st.ledger_fremd_pfad(fx.konfig).read_text(encoding="utf-8").splitlines()]
    assert zeilen and all(z["commit_fremd"] == fremd[0] and z["lauf"] == erg.lauf for z in zeilen)
    assert not sv.schreibvermerk_pfad(fx.konfig).exists()
    monkeypatch.undo()
    _code_aenderung(fx, 1)
    _anstoss(fx)
    erg2 = _tick(fx)
    assert not [p for p in _pruefliste(fx)["offen"] if p["art"] == "hand_edit"] and erg2.commit is None
    assert _status(fx.repo) == []
