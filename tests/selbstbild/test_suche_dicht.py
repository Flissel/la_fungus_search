"""Dichte Suche (Stufe 3, T4): HTTP-Embedder gegen lokalen Test-Server mit festen Vektoren, kein echtes Modell."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from selbstbild.suche.abfrage import suchen
from selbstbild.suche import dicht
from selbstbild.suche.dicht import (Embedder, aehnlichste, datei_text, einbetten, einbetten_stapelweise,
                                    vektoren_bauen)
from selbstbild.suche.index import index_bauen
from selbstbild.typen import Beleg, Knoten

VERSAND = "datei:repo:app/versand.py"
RECHNUNG = "datei:repo:app/a.py"
CRON = "datei:repo:app/cron.py"


def _vek(text: str) -> list[float]:
    t = text.lower()
    if "versand" in t or "zustellung" in t:
        return [3.0, 0.0]
    if "rechnung" in t:
        return [0.0, 2.0]
    return [0.6, 0.8]


class _Handler(BaseHTTPRequestHandler):
    aufrufe: list[tuple[str, int]] = []

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        daten = json.loads(self.rfile.read(n))
        if self.path == "/api/embed":
            self.aufrufe.append((self.path, len(daten["input"])))
            antwort = {"embeddings": [_vek(t) for t in daten["input"]]}
        elif self.path == "/embed":
            self.aufrufe.append((self.path, 1))
            antwort = {"vector": _vek(daten["text"])}
        else:
            self.send_response(404)
            self.end_headers()
            return
        roh = json.dumps(antwort).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(roh)))
        self.end_headers()
        self.wfile.write(roh)

    def log_message(self, *args: object) -> None:  # still
        pass


@contextmanager
def _server() -> Iterator[str]:
    _Handler.aufrufe = []
    s = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{s.server_address[1]}"
    finally:
        s.shutdown()
        s.server_close()
        t.join(5)


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
    stand = graph_stand(knoten, [], commit=sha)
    index_bauen(git_repo.wurzel, stand)
    return stand


def test_einbetten_ollama_format():
    with _server() as url:
        texte = [f"text {i}" for i in range(70)] + ["versand"]
        v = einbetten(Embedder("ollama", url, "m"), texte)
        assert len(v) == 71 and v[-1] == [3.0, 0.0]
        assert _Handler.aufrufe == [("/api/embed", 32), ("/api/embed", 32), ("/api/embed", 7)]


def test_einbetten_dienst_format():
    with _server() as url:
        assert einbetten(Embedder("dienst", url + "/", "m"), ["rechnung", "x"]) == [[0.0, 2.0], [0.6, 0.8]]
        assert [a[0] for a in _Handler.aufrufe] == ["/embed", "/embed"]


def test_datei_text(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    t = datei_text(stand, VERSAND)
    assert t == "app/versand.py\nnewsletter_freigeben"
    assert datei_text(stand, "datei:repo:nix") == ""


def test_vektoren_und_aehnlichste(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    with _server() as url:
        assert vektoren_bauen(stand, Embedder("ollama", url, "mod-a")) == 3
    treffer = aehnlichste(stand, [1.0, 0.0], 2, "mod-a")
    assert [i for i, _ in treffer] == [VERSAND, CRON]
    assert treffer[0][1] == pytest.approx(1.0) and treffer[1][1] == pytest.approx(0.6)
    assert aehnlichste(stand, [1.0, 0.0], 5, "anderes-modell") == ()
    assert aehnlichste(stand, [0.0, 0.0], 5, "mod-a") == ()
    assert aehnlichste(stand, [1.0, 0.0, 0.0], 5, "mod-a") == ()  # falsche Dimension


def test_aehnlichste_gleichstand_nach_id(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    with _server() as url:
        vektoren_bauen(stand, Embedder("ollama", url, "m"))
    # versand und rechnung sind gleich weit von der Diagonale
    treffer = aehnlichste(stand, [1.0, 1.0], 3, "m")
    assert treffer[0][0] == CRON
    assert [i for i, _ in treffer[1:]] == sorted(i for i, _ in treffer[1:])


def test_embedder_aus_meldet_klar():
    s = HTTPServer(("127.0.0.1", 0), _Handler)
    port = s.server_address[1]
    s.server_close()  # Port frei, niemand hoert
    with pytest.raises(RuntimeError) as fehler:
        einbetten(Embedder("ollama", f"http://127.0.0.1:{port}", "m"), ["geheimer Text"], timeout=2.0)
    assert "geheimer" not in str(fehler.value) and "URLError" in str(fehler.value)


def test_fehlerstatus_ohne_inhalt():
    with _server() as url:
        with pytest.raises(RuntimeError) as fehler:
            einbetten(Embedder("ollama", url + "/falsch", "m"), ["geheimer Text"])
    assert "404" in str(fehler.value) and "geheimer" not in str(fehler.value)


def test_suchen_mit_dicht(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    e_aus = Embedder("ollama", "http://127.0.0.1:9", "m")
    with _server() as url:
        e = Embedder("ollama", url, "m")
        vektoren_bauen(stand, e)
        ohne = suchen(stand, "Zustellung Mail")
        assert ohne == ()
        mit = suchen(stand, "Zustellung Mail", dicht=True, embedder=e)
        assert mit and mit[0].id == VERSAND and mit[0].herkunft == ("dicht",)
        # ohne Embedder oder mit anderem Modell bleibt der dichte Zweig leer, ohne Fehler
        assert suchen(stand, "Zustellung Mail", dicht=True) == ()
        assert suchen(stand, "Zustellung Mail", dicht=True, embedder=Embedder("ollama", url, "x")) == ()
    assert suchen(stand, "Zustellung Mail", dicht=True, embedder=e_aus) == ()


def test_stapelweise_normiert_und_je_batch(monkeypatch):
    ereignisse: list[str] = []
    echt_post, echt_norm = dicht._post, dicht._normiert

    def post(url, nutzlast, timeout):
        ereignisse.append(f"post{len(nutzlast['input'])}")
        return echt_post(url, nutzlast, timeout)

    def norm(v):
        ereignisse.append("norm")
        return echt_norm(v)

    monkeypatch.setattr(dicht, "_post", post)
    monkeypatch.setattr(dicht, "_normiert", norm)
    with _server() as url:
        batches = list(einbetten_stapelweise(Embedder("ollama", url, "m"), ["versand", "rechnung", "x"], batch=2))
    assert [len(b) for b in batches] == [2, 1]
    assert batches[0][0].dtype.name == "float32" and float(batches[0][0] @ batches[0][0]) == pytest.approx(1.0)
    # Umwandlung jedes Batches vor dem naechsten Abruf: nie mehr als ein Batch Float-Listen lebendig
    assert ereignisse == ["post2", "norm", "norm", "post1", "norm"]


def test_vektoren_bauen_streamt_stapelweise(git_repo, graph_stand, monkeypatch):
    stand = _stand(git_repo, graph_stand)
    ereignisse: list[str] = []
    echt_post, echt_norm = dicht._post, dicht._normiert
    monkeypatch.setattr(dicht, "BATCH", 2)
    monkeypatch.setattr(dicht, "_post", lambda u, n, t: (ereignisse.append("post"), echt_post(u, n, t))[1])
    monkeypatch.setattr(dicht, "_normiert", lambda v: (ereignisse.append("norm"), echt_norm(v))[1])
    with _server() as url:
        assert vektoren_bauen(stand, Embedder("ollama", url, "m")) == 3
    assert ereignisse == ["post", "norm", "norm", "post", "norm"]


def test_vektoren_bauen_index_ersetzt_schreibt_nichts(git_repo, graph_stand, monkeypatch):
    stand = _stand(git_repo, graph_stand)
    echt_post = dicht._post

    def post_und_ersetzen(url, nutzlast, timeout):
        antwort = echt_post(url, nutzlast, timeout)
        with sqlite3.connect(str(dicht.index_pfad(stand))) as c:  # simuliert index_bauen mit neuem Stand
            c.execute("UPDATE meta SET v='anderer-commit' WHERE k='commit'")
        return antwort

    monkeypatch.setattr(dicht, "_post", post_und_ersetzen)
    with _server() as url:
        with pytest.raises(RuntimeError, match="Index waehrend der Einbettung ersetzt"):
            vektoren_bauen(stand, Embedder("ollama", url, "m"))
    with sqlite3.connect(str(dicht.index_pfad(stand))) as c:
        assert c.execute("SELECT count(*) FROM vektor").fetchone() == (0,)


def test_vektoren_bauen_nur_noch_vorhandene_ids(git_repo, graph_stand, monkeypatch):
    stand = _stand(git_repo, graph_stand)
    echt_post = dicht._post

    def post_und_loeschen(url, nutzlast, timeout):
        antwort = echt_post(url, nutzlast, timeout)
        with sqlite3.connect(str(dicht.index_pfad(stand))) as c:  # Eintrag verschwindet, Fingerabdruck gleich
            c.execute("DELETE FROM eintrag WHERE id=?", (CRON,))
        return antwort

    monkeypatch.setattr(dicht, "_post", post_und_loeschen)
    with _server() as url:
        assert vektoren_bauen(stand, Embedder("ollama", url, "m")) == 2


def test_suchen_dicht_ohne_numpy_ist_leer(git_repo, graph_stand, monkeypatch):
    import builtins
    stand = _stand(git_repo, graph_stand)
    echt = builtins.__import__

    def kein_dicht(name, glob=None, lokal=None, liste=(), stufe=0):
        if stufe == 1 and "dicht" in (liste or ()):
            raise ImportError("numpy fehlt")
        return echt(name, glob, lokal, liste, stufe)

    monkeypatch.setattr(builtins, "__import__", kein_dicht)
    assert suchen(stand, "Zustellung Mail", dicht=True, embedder=Embedder("ollama", "http://127.0.0.1:9", "m")) == ()
