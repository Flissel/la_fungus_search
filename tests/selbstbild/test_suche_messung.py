"""Messung (Stufe 3, T5): Gold aus KETTEN.md und aus Transkripten, recall@10 und MRR je Suchvariante."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from selbstbild.__main__ import main
from selbstbild.suche.abfrage import suchen
from selbstbild.suche.dicht import Embedder
from selbstbild.suche.index import index_bauen
from selbstbild.speicher import Speicher
from selbstbild.suche.abfrage import suchen_gemeldet
from selbstbild.suche.messung import (_datei_ranking, datei_id_von_pfad, gold_aus_ketten,
                                      gold_aus_transkripten, messen, messen_quellen)
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
    aufrufe: list[int] = []
    nur_aufwaermen = False  # Test: Aufwaermen klappt, jede spaetere Anfrage scheitert

    def do_POST(self) -> None:  # noqa: N802
        daten = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        self.aufrufe.append(len(daten["input"]))
        if self.nur_aufwaermen and daten["input"] != ["aufwaermen"]:
            self.send_response(500)
            self.end_headers()
            return
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
    assert erg["dicht:mod"] == {"recall@10": 1.0, "mrr": 1.0, "dicht_leer": 0}
    assert erg["fusion:mod"]["recall@10"] == 1.0 and erg["fusion:mod"]["dicht_leer"] == 0
    assert "dicht_leer" not in erg["stichwort"]


def test_messen_unbekannte_variante(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    with pytest.raises(ValueError):
        messen(stand, (("x", frozenset({VERSAND})),), varianten=["magie"])
    with pytest.raises(ValueError):
        messen(stand, (("x", frozenset({VERSAND})),), varianten=["dicht:ohne"])


def test_cli_ohne_stand(tmp_path, capsys):
    assert main(["suche-messen", str(tmp_path)]) == 1
    assert capsys.readouterr().out == ""


@contextmanager
def _server(nur_aufwaermen: bool = False) -> Iterator[str]:
    _H.aufrufe = []
    _H.nur_aufwaermen = nur_aufwaermen
    s = HTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{s.server_address[1]}"
    finally:
        s.shutdown()
        s.server_close()
        t.join(5)
        _H.nur_aufwaermen = False


def test_vektoren_einmal_je_modell_ueber_quellen(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    quellen = {"a": (("rechnung", frozenset({RECHNUNG})),), "b": (("newsletter", frozenset({VERSAND})),)}
    with _server() as url:
        erg = messen_quellen(stand, quellen, varianten=["dicht:mod", "fusion:mod"],
                             embedder={"mod": Embedder("ollama", url, "mod")})
        gebaut = [n for n in _H.aufrufe if n == 3]  # alle drei Dateien in einem Batch
    assert len(gebaut) == 1 and set(erg) == {"a", "b"}
    assert erg["a"]["fusion:mod"]["recall@10"] == 1.0


def test_suchen_gemeldet_leerer_dichter_zweig(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    with _server() as url:
        e = Embedder("ollama", url, "mod")
        messen(stand, (("rechnung", frozenset({RECHNUNG})),), varianten=["dicht:mod"], embedder={"mod": e})
        assert suchen_gemeldet(stand, "rechnung", dicht=True, embedder=e)[1] is False
    with _server(nur_aufwaermen=True) as url:  # Vektoren stehen, die Anfrage scheitert -> Zweig leer
        treffer, leer = suchen_gemeldet(stand, "rechnung", dicht=True, embedder=Embedder("ollama", url, "mod"))
        assert leer is True and treffer  # Fusion lieferte still nur Stichwort/Struktur
    assert suchen_gemeldet(stand, "rechnung", dicht=False)[1] is False


def test_aufwaermen_scheitert_variante_fehlgeschlagen(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    gold = (("rechnung", frozenset({RECHNUNG})),)
    tot = Embedder("ollama", "http://127.0.0.1:9", "mod")  # Port 9: niemand hoert zu
    erg = messen(stand, gold, varianten=["stichwort", "dicht:mod", "fusion:mod"], embedder={"mod": tot})
    assert erg["dicht:mod"] == {"fehlgeschlagen": 1.0} and erg["fusion:mod"] == {"fehlgeschlagen": 1.0}
    assert erg["stichwort"]["recall@10"] == 1.0


def test_datei_ranking_dedupliziert_nach_datei():
    ids = [FREIGEBEN, VERSAND, "symbol:repo:app/versand.py#anderes", "env:X", CRON, "doku_seite:repo:app/cron.py"]
    assert _datei_ranking(ids) == [VERSAND, CRON]


def test_datei_id_windows_pfad_und_gross_klein():
    repos = ("vibemind", "vibemind/vibemind-os")
    assert datei_id_von_pfad("Vibemind-OS\\spaces\\x.py", "vibemind", repos) == "datei:vibemind/vibemind-os:spaces/x.py"
    assert datei_id_von_pfad("docs\\a.md", "vibemind", repos) == "datei:vibemind:docs/a.md"
    assert datei_id_von_pfad("vibemind-os-extra/a.py", "vibemind", repos) == "datei:vibemind:vibemind-os-extra/a.py"


def test_transkripte_filtern_keine_prompts(tmp_path):
    wurzel = tmp_path / "r"
    wurzel.mkdir()
    ordner = tmp_path / "tr"
    ordner.mkdir()
    echt = "Echte Aufgabe mit genug Text hier"
    zeilen = [
        _prompt(echt), _edit(str(wurzel / "a.py")),
        _prompt("This session is being continued from a previous conversation that ran out"),
        _prompt("<command-name>/clear</command-name> und mehr Text dahinter"),
        _prompt("<local-command-stdout>Ausgabe lang genug fuer die Grenze</local-command-stdout>"),
        _prompt("[Request interrupted by user for tool use] mit Text"),
        _zeile("user", isSidechain=True, message={"role": "user", "content": "Subagent-Aufgabe mit viel Text"}),
        _zeile("user", isCompactSummary=True, message={"role": "user", "content": "Zusammenfassung mit viel Text"}),
        _edit(str(wurzel / "b.py")),
    ]
    (ordner / "s.jsonl").write_text("\n".join(zeilen), encoding="utf-8")
    (ordner / "agent-abc.jsonl").write_text("\n".join([_prompt("Agenten-Prompt lang genug fuer Grenze"),
                                                      _edit(str(wurzel / "c.py"))]), encoding="utf-8")
    gold = gold_aus_transkripten(ordner, wurzel, n=10, wurzel_repo="r")
    assert gold == ((echt, frozenset({"datei:r:a.py", "datei:r:b.py"})),)


def test_gold_ziele_auf_hauptbaum_und_index(git_repo, graph_stand, tmp_path):
    """T12: Worktree-Ziele zaehlen im Hauptbaum; nur Ziele im Index; Prompts ohne solches Ziel entfallen."""
    stand = _stand(git_repo, graph_stand)
    w = git_repo.wurzel
    ordner = tmp_path / "tr"
    ordner.mkdir()
    zeilen = [
        _prompt("Erste Aufgabe mit genug Text"), _edit(str(w / ".worktrees" / "wt1" / "app" / "versand.py")),
        _edit(str(w / "app" / "nicht_im_index.py")),
        _prompt("Zweite Aufgabe ohne erreichbares Ziel"), _edit(str(w / "docs" / "plan.md")),
        _prompt("Dritte Aufgabe im zweiten Checkout"), _edit(str(w / ".claude" / "worktrees" / "a1" / "app" / "a.py")),
    ]
    (ordner / "s.jsonl").write_bytes(("\n".join(zeilen) + "\n").encode("utf-8"))
    gold = gold_aus_transkripten(ordner, w, n=10, wurzel_repo="repo", stand=stand)
    assert [z for _, z in gold] == [frozenset({VERSAND}), frozenset({RECHNUNG})]
    ohne_index = gold_aus_transkripten(ordner, w, n=10, wurzel_repo="repo")
    assert [z for _, z in ohne_index] == [frozenset({VERSAND, "datei:repo:app/nicht_im_index.py"}),
                                          frozenset({"datei:repo:docs/plan.md"}), frozenset({RECHNUNG})]


def test_stichproben_mit_transkript_und_offset(git_repo, graph_stand, tmp_path, capsys):
    from selbstbild.hooks.verlauf import letzte_dateien
    from selbstbild.suche.messung import transkript_stichproben

    stand = _stand(git_repo, graph_stand)
    w = git_repo.wurzel
    ordner = tmp_path / "tr"
    ordner.mkdir()
    geheim = "GEHEIM-STICHPROBE mit genug Text"
    zeilen = [_prompt("Vorher eine Aufgabe mit Text"), _edit(str(w / "app" / "cron.py")),
              _prompt("kurz"),  # kurze Zurufe bleiben Teil der Aufgabe davor
              _prompt(geheim), _edit(str(w / "app" / "versand.py"))]
    datei = ordner / "s.jsonl"
    datei.write_bytes(("\n".join(zeilen) + "\n").encode("utf-8"))
    (ordner / "ausgeschlossen.jsonl").write_bytes(("\n".join(zeilen) + "\n").encode("utf-8"))
    proben = transkript_stichproben(ordner, w, n=10, wurzel_repo="repo", stand=stand,
                                    ausschliessen=("ausgeschlossen.jsonl",))
    assert len(proben) == 2 and all(p.transkript == datei for p in proben)
    p = proben[1]
    assert p.anfrage == geheim and p.ziele == frozenset({VERSAND})
    roh = datei.read_bytes()
    assert json.loads(roh[p.ende:].split(b"\n", 1)[0])["message"]["content"] == geheim  # Offset = Prompt-Zeile
    assert letzte_dateien(datei, ende=p.ende) == [str(w / "app" / "cron.py")]  # Verlauf nur davor
    assert proben[0].ende == 0
    gleich = gold_aus_transkripten(ordner, w, n=10, wurzel_repo="repo", stand=stand,
                                   ausschliessen=("ausgeschlossen.jsonl",))
    assert gleich == tuple((s.anfrage, s.ziele) for s in proben)
    assert geheim not in capsys.readouterr().out


def test_stichprobe_deterministisch_nach_filter(git_repo, graph_stand, tmp_path):
    from selbstbild.suche.messung import transkript_stichproben

    stand = _stand(git_repo, graph_stand)
    w = git_repo.wurzel
    ordner = tmp_path / "tr"
    ordner.mkdir()
    zeilen: list[str] = []
    for i in range(12):
        zeilen += [_prompt(f"Aufgabe nummer {i} mit ausreichend Text"),
                   _edit(str(w / "app" / ("versand.py" if i % 2 else "fehlt.py")))]
    (ordner / "a.jsonl").write_bytes(("\n".join(zeilen) + "\n").encode("utf-8"))
    a = transkript_stichproben(ordner, w, n=3, wurzel_repo="repo", stand=stand)
    assert len(a) == 3 and a == transkript_stichproben(ordner, w, n=3, wurzel_repo="repo", stand=stand)
    assert all(s.ziele == frozenset({VERSAND}) for s in a)


def _cli_stand(git_repo, graph_stand, monkeypatch, name: str | None = "repo") -> Path:
    stand = _stand(git_repo, graph_stand)
    if name is not None:
        with Speicher(stand / "graph.sqlite") as sp:
            sp.meta_setzen("name", name)
    monkeypatch.setattr("selbstbild.__main__.stand_finden", lambda w: stand / "graph.sqlite")
    return stand


def test_cli_fremdes_repo_im_index(git_repo, graph_stand, monkeypatch, capsys):
    _cli_stand(git_repo, graph_stand, monkeypatch, name="anders")
    assert main(["suche-messen", str(git_repo.wurzel)]) == 2
    aus = capsys.readouterr()
    assert aus.out == "" and "'repo'" in aus.err and "'anders'" in aus.err


def test_cli_ohne_namen_in_meta(git_repo, graph_stand, monkeypatch, capsys):
    _cli_stand(git_repo, graph_stand, monkeypatch, name=None)
    assert main(["suche-messen", str(git_repo.wurzel)]) == 2
    assert "name" in capsys.readouterr().err


def test_cli_keine_prompt_texte_in_ausgaben(git_repo, graph_stand, monkeypatch, capsys, tmp_path):
    _cli_stand(git_repo, graph_stand, monkeypatch)
    marker = "MARKER-7f3a9c-newsletter-freigeben"
    ordner = tmp_path / "tr"
    ordner.mkdir()
    (ordner / "s.jsonl").write_text("\n".join([
        _prompt(f"{marker} bitte erledigen"), _edit(str(git_repo.wurzel / "app" / "versand.py"))]),
        encoding="utf-8")
    assert main(["suche-messen", str(git_repo.wurzel), "--transkripte", str(ordner)]) == 0
    aus = capsys.readouterr()
    assert marker not in aus.out and marker not in aus.err
    daten = json.loads(aus.out)
    assert daten["paare"] == {"transkripte": 1}  # KETTEN fehlt -> Warnung auf stderr, kein Paar
    assert "ketten" in aus.err
    assert set(daten["ergebnis"]["transkripte"]) == {"stichwort", "struktur"}
