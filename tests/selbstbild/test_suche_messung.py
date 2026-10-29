"""Messung (Stufe 3, T5): Gold aus KETTEN.md und aus Transkripten, recall@10 und MRR je Suchvariante."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from selbstbild.__main__ import main
from selbstbild.suche.abfrage import suchen
from selbstbild.suche.dicht import Embedder
from selbstbild.suche.index import index_bauen
from selbstbild.suche.messung import gold_aus_ketten, gold_aus_transkripten, messen
from selbstbild.typen import Beleg, Kante, Knoten

KETTEN = """# Ketten

<!-- graph:begin hops K1 -->
| # | Von → Nach | Mechanismus | Code-Stelle | Belegt durch | Zustand |
|---|------------|-------------|-------------|--------------|---------|
| 1 | Pult → Sidecar `:8130` | MCP `versand_beauftragen` | `vibemind-os/spaces/marketing/claw/werkzeuge.py:313` `claw.werkzeuge.versand_beauftragen` | 008 | läuft |
| 2 | Sidecar → API | HTTP `POST /api/x` | ⚠ `app/server.py:5`, `vibemind-os/shared/x.py` `a.b.c` | 008 | läuft |
| 3 | API → DB | SQL | `api.nur.symbol` | 008 | läuft |
<!-- graph:end hops K1 -->
"""
REPOS = ("vibemind", "vibemind/vibemind-os", "vibemind/vibemind-os/shared")

VERSAND = "datei:repo:app/versand.py"
CRON = "datei:repo:app/cron.py"
RECHNUNG = "datei:repo:app/a.py"
FREIGEBEN = "symbol:repo:app/versand.py#newsletter_freigeben"
AUFRUFER = "symbol:repo:app/cron.py#taeglich"


def test_gold_aus_ketten():
    gold = gold_aus_ketten(KETTEN, "vibemind", repos=REPOS)
    assert len(gold) == 2  # Zeile 3 ohne Pfad entfaellt
    q1, ziel1 = gold[0]
    assert "Pult" in q1 and "Sidecar" in q1 and "versand_beauftragen" in q1 and "`" not in q1
    assert ziel1 == frozenset({"datei:vibemind/vibemind-os:spaces/marketing/claw/werkzeuge.py"})
    assert gold[1][1] == frozenset({"datei:vibemind:app/server.py", "datei:vibemind/vibemind-os/shared:x.py"})


def _zeile(art: str, **kw: object) -> str:
    return json.dumps({"type": art, **kw})


def _prompt(text: object) -> str:
    return _zeile("user", message={"role": "user", "content": text})


def _edit(pfad: str, werkzeug: str = "Edit") -> str:
    return _zeile("assistant", message={"role": "assistant", "content": [
        {"type": "text", "text": "ok"}, {"type": "tool_use", "name": werkzeug, "input": {"file_path": pfad}}]})


def test_gold_aus_transkripten(tmp_path, capsys):
    wurzel = tmp_path / "repo"
    (wurzel / "app").mkdir(parents=True)
    geheim = "GEHEIMER-PROMPT-TEXT mit genug Zeichen"
    ordner = tmp_path / "tr"
    ordner.mkdir()
    zeilen = [
        _prompt(geheim),
        _edit(str(wurzel / "app" / "x.py")),
        _edit(str(tmp_path / "anderswo" / "y.py"), "Write"),
        _zeile("user", message={"role": "user", "content": [{"type": "tool_result", "content": "r"}]}),
        _prompt("kurz"),
        _edit(str(wurzel / "app" / "z.py"), "MultiEdit"),
        _prompt([{"type": "text", "text": "zweiter Prompt ohne Dateiaenderung"}]),
        "kaputte zeile {",
    ]
    (ordner / "s1.jsonl").write_text("\n".join(zeilen), encoding="utf-8")
    gold = gold_aus_transkripten(ordner, wurzel, n=5, wurzel_repo="repo")
    assert gold == ((geheim, frozenset({"datei:repo:app/x.py", "datei:repo:app/z.py"})),)
    gleich = gold_aus_transkripten(ordner, wurzel, n=5, wurzel_repo="repo")
    assert gleich == gold
    assert geheim not in capsys.readouterr().out


def test_transkript_stichprobe_deterministisch(tmp_path):
    wurzel = tmp_path / "r"
    wurzel.mkdir()
    ordner = tmp_path / "tr"
    ordner.mkdir()
    zeilen: list[str] = []
    for i in range(10):
        zeilen += [_prompt(f"Aufgabe nummer {i} mit ausreichend Text"), _edit(str(wurzel / f"f{i}.py"))]
    (ordner / "a.jsonl").write_text("\n".join(zeilen), encoding="utf-8")
    a = gold_aus_transkripten(ordner, wurzel, n=3, seed=1, wurzel_repo="r")
    assert len(a) == 3 and a == gold_aus_transkripten(ordner, wurzel, n=3, seed=1, wurzel_repo="r")


def _stand(git_repo, graph_stand) -> Path:
    git_repo.datei("app/versand.py", "def newsletter_freigeben():\n    return 1\n")
    git_repo.datei("app/cron.py", "def taeglich():\n    return 2\n")
    git_repo.datei("app/a.py", "def pruefe_rechnung():\n    return 3\n")
    sha = git_repo.commit("init")

    def sym(p: str, n: str) -> Knoten:
        return Knoten(f"symbol:repo:{p}#{n}", "symbol", n, Beleg("repo", p, 1, n, "d"))

    knoten = [Knoten(VERSAND, "datei", "app/versand.py", None), Knoten(CRON, "datei", "app/cron.py", None),
              Knoten(RECHNUNG, "datei", "app/a.py", None),
              sym("app/versand.py", "newsletter_freigeben"), sym("app/cron.py", "taeglich"),
              sym("app/a.py", "pruefe_rechnung")]
    stand = graph_stand(knoten, [Kante("ruft", AUFRUFER, FREIGEBEN, True, None)], commit=sha)
    index_bauen(git_repo.wurzel, stand)
    return stand


def test_suchen_ohne_struktur(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    assert AUFRUFER in [t.id for t in suchen(stand, "newsletter_freigeben")]
    assert AUFRUFER not in [t.id for t in suchen(stand, "newsletter_freigeben", struktur=False)]


def test_messen_recall(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    gold = (("newsletter freigeben", frozenset({VERSAND})), ("zzzunbekannt", frozenset({RECHNUNG})))
    erg = messen(stand, gold, varianten=["stichwort"])
    assert erg["stichwort"] == {"recall@10": 0.5, "mrr": 0.5}


def test_messen_struktur_hilft(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    gold = (("newsletter_freigeben", frozenset({CRON})),)
    erg = messen(stand, gold, varianten=["stichwort", "struktur"])
    assert erg["stichwort"]["recall@10"] == 0.0
    assert erg["struktur"]["recall@10"] == 1.0 and 0.0 < erg["struktur"]["mrr"] <= 1.0


class _H(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        daten = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        roh = json.dumps({"embeddings": [[0.0, 2.0] if "rechnung" in t.lower() else [3.0, 0.0]
                                         for t in daten["input"]]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(roh)))
        self.end_headers()
        self.wfile.write(roh)

    def log_message(self, *args: object) -> None:
        pass


def test_messen_dicht_und_fusion(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    s = HTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    try:
        e = Embedder("ollama", f"http://127.0.0.1:{s.server_address[1]}", "mod")
        gold = (("buchhaltung rechnung", frozenset({RECHNUNG})),)
        erg = messen(stand, gold, varianten=["stichwort", "dicht:mod", "fusion:mod"], embedder={"mod": e})
    finally:
        s.shutdown()
        s.server_close()
        t.join(5)
    assert erg["stichwort"]["recall@10"] == 1.0
    assert erg["dicht:mod"]["recall@10"] == 1.0 and erg["dicht:mod"]["mrr"] == 1.0
    assert erg["fusion:mod"]["recall@10"] == 1.0


def test_messen_unbekannte_variante(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    with pytest.raises(ValueError):
        messen(stand, (("x", frozenset({VERSAND})),), varianten=["magie"])
    with pytest.raises(ValueError):
        messen(stand, (("x", frozenset({VERSAND})),), varianten=["dicht:ohne"])


def test_cli_ohne_stand(tmp_path, capsys):
    assert main(["suche-messen", str(tmp_path)]) == 1
    assert capsys.readouterr().out == ""
