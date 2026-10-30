"""doku_loop: Graph aus den Extraktoren bauen und Seed-Hops bestaetigen (Task 10)."""

import ast
import copy
import dataclasses
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import bauen  # noqa: E402
from selbstbild.doku import ketten_seed as ks  # noqa: E402
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import korpus  # noqa: E402
from selbstbild.doku.anker import Beleg  # noqa: E402
from selbstbild.doku.extraktoren import http_clients, routen  # noqa: E402
from selbstbild.doku.extraktoren import sql as sql_ex  # noqa: E402
from selbstbild.doku.graph import Graph, Kante, Knoten, graph_diff, hop_zustand_neu, kante_id  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402


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
    "def andere():\n"
    '    return requests.get(f"{B}/api/nicht_da")\n'
    "\n"
    "\n"
    "def lesen(db):\n"
    '    return db.query("select * from b.t")\n'
    "\n"
    "\n"
    "def helfer():\n"
    "    return 1\n"
    "\n"
    "\n"
    "def haupt():\n"
    "    return helfer()\n"
    "\n"
    "\n"
    "def fremd():\n"
    '    return requests.get(os.getenv("X_URL") + "/api/y")\n'
    "\n"
    "\n"
    "def fremd2():\n"
    '    return requests.get(os.getenv("X_URL") + "/api/z")\n'
    "\n"
    "\n"
    "def zu_a():\n"
    '    return requests.get(os.environ["A_URL"] + "/api/a")\n'
)
API = (
    "from fastapi import FastAPI\n"
    "\n"
    "app = FastAPI()\n"
    "\n"
    "\n"
    '@app.post("/api/x")\n'
    "def x(db):\n"
    "    return speichern(db)\n"
    "\n"
    "\n"
    "def speichern(db):\n"
    '    return db.execute("insert into b.t values (1)")\n'
)
SERVER = (
    "from mcp.server.fastmcp import FastMCP\n"
    "\n"
    'mcp = FastMCP("b")\n'
    "\n"
    "\n"
    "@mcp.tool()\n"
    "def auftrag_holen():\n"
    "    return 1\n"
)
TEST_API = (
    "import os\n"
    "import requests\n"
    "from api import app\n"
    "from server import mcp\n"
    "\n"
    'B = os.environ.get("B_URL")\n'
    "\n"
    "\n"
    '@app.get("/api/t")\n'
    "def t(db):\n"
    '    requests.get(f"{B}/api/x")\n'
    '    requests.get(os.getenv("NUR_IM_TEST_URL") + "/q")\n'
    '    return db.query("select * from b.t")\n'
    "\n"
    "\n"
    "@mcp.tool()\n"
    "def test_werkzeug():\n"
    "    return 1\n"
)
DIENSTE = (
    "dienste:\n"
    "  a:\n"
    "    port: 8120\n"
    "    ort: container\n"
    "    space: a\n"
    "    env: [A_URL]\n"
    "  b:\n"
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
)
CALLEES = {
    "a": {"client.haupt": {"client.helfer"}},
    "b": {"api.x": {"api.speichern"}, "tests.test_api.t": {"api.x"}},
}


def _schreibe(pfad: Path, text: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(text, encoding="utf-8")


def _zeile(text: str, teil: str) -> int:
    return next(i for i, z in enumerate(text.splitlines(), 1) if teil in z)


def _manifest(wurzel: Path, callees=None, commit="sub1"):
    """Kleines Manifest wie benchmarks.gate2.manifest (Funktionen/Klassen, Methoden in Klassen)."""
    docs = []
    for p in sorted(wurzel.rglob("*.py")):
        rel = p.relative_to(wurzel).as_posix()
        modul = ".".join(p.relative_to(wurzel).with_suffix("").parts)
        text = p.read_text(encoding="utf-8")
        zeilen = text.splitlines()

        def sammeln(knoten, praefix):
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
                           callees_by_document={k: frozenset(v) for k, v in (callees or {}).items()})


def _repo(tmp_path, *, mit_test=True):
    _schreibe(tmp_path / "sp/a/client.py", CLIENT)
    _schreibe(tmp_path / "sp/b/api.py", API)
    _schreibe(tmp_path / "sp/b/server.py", SERVER)
    if mit_test:
        _schreibe(tmp_path / "sp/b/tests/test_api.py", TEST_API)
    _schreibe(tmp_path / "sp/b/db/001_t.sql", "CREATE TABLE b.t (id int);\n")
    _schreibe(tmp_path / "config/dienste.yaml", DIENSTE)
    _schreibe(tmp_path / "infra/swarm/vibemind-stack.yml", COMPOSE)
    _schreibe(tmp_path / "logs/w.json", json.dumps({"stand": {"probe": False, "fremd": True}}))
    return dataclasses.replace(
        vm_laden(), repo=tmp_path,
        spaces={"a": SpaceKonfig("a", "sp/a", "werkzeuge"), "b": SpaceKonfig("b", "sp/b", "decorator")},
        compose_live=("infra/swarm/vibemind-stack.yml",), sql_wurzeln=(".",),
        dienste_yaml="config/dienste.yaml", waechter_zustandsdatei="logs/w.json",
        waechter_zuordnung={"probe": ("K1",)}, ketten_md="docs/KETTEN.md",
    )


def _manifeste(konfig):
    return {n: _manifest(konfig.repo / s.wurzel, CALLEES.get(n)) for n, s in konfig.spaces.items()}


def _sym(datei, anker_):
    return Beleg("symbol", datei, anker_, "d", "c", 1)


def _hop(kette, nr, von, nach, mech, *belege, gemessen="keine_probe", nur_doku=False):
    z = hop_zustand_neu()
    z["gemessen"] = {"wert": gemessen, "zeit": None}
    z["nur_doku"] = nur_doku
    return Kante(kante_id("hop", von, nach, kette=kette, nr=nr), "hop", von, nach, "seed", belege,
                 {"kette": kette, "nr": nr, "mechanismus_typ": mech, "zustand": z})


def _basis(*hops):
    g = Graph({}, {}, "aussen1", "sub1")
    for h in hops:
        for nid in (h.von, h.nach):
            typ, name = nid.split(":", 1)
            g.knoten_hinzu(Knoten(nid, typ, name))
        g.kante_hinzu(h)
    return g


def _bauen(tmp_path, *hops, spaces=("a", "b")):
    konfig = _repo(tmp_path)
    basis = _basis(*hops)
    g, pp = bauen.graph_bauen(konfig, list(spaces), basis, _manifeste(konfig))
    return g, pp, basis


def _zustand(g, hop):
    return g.kanten[hop.id].attribute["zustand"]


# ---------- Kanten aus den Extraktoren ----------

def test_fixture_kanten_aus_allen_extraktoren(tmp_path):
    konfig = _repo(tmp_path)
    manifeste = _manifeste(konfig)
    g, pp = bauen.graph_bauen(konfig, ["a", "b"], None, manifeste)

    nd = g.kanten["nutzt_dienst:symbol:a:client.senden->dienst:b"]
    assert nd.quelle == "extraktor" and nd.attribute["space"] == "a"
    assert nd.attribute["env"] == ["B_URL"] and "/api/x" in nd.attribute["pfade"]
    senden = next(d for d in manifeste["a"].documents if d.document_id == "client.senden")
    assert [(b.art, b.file, b.anker, b.digest, b.commit) for b in nd.belege] == [
        ("symbol", "sp/a/client.py", "client.senden", hashlib.sha256(senden.source.encode()).hexdigest(), "sub1")]

    dr = g.kanten["definiert_route:symbol:b:api.x->route:POST /api/x@b"]
    assert dr.attribute["space"] == "b" and [b.anker for b in dr.belege] == ["api.x"]
    route = g.knoten["route:POST /api/x@b"]
    assert route.typ == "route" and route.space == "b" and route.attribute["generisch"] == "nein"

    ls = g.kanten["liest_schreibt:symbol:b:api.speichern->tabelle:b.t"]
    assert sorted((b.art, b.anker) for b in ls.belege) == [("sql", "b.t"), ("symbol", "api.speichern")]
    sql_beleg = next(b for b in ls.belege if b.art == "sql")
    assert sql_beleg.file == "sp/b/db/001_t.sql" and sql_beleg.zeile == 1
    # Space a liest dieselbe Tabelle: beide Kanten treffen denselben Knoten
    assert "liest_schreibt:symbol:a:client.lesen->tabelle:b.t" in g.kanten
    assert g.knoten["tabelle:b.t"].typ == "tabelle"

    rg = g.kanten["registriert:symbol:b:server.auftrag_holen->mcp_werkzeug:auftrag_holen"]
    assert rg.attribute["space"] == "b" and g.knoten["mcp_werkzeug:auftrag_holen"].typ == "mcp_werkzeug"

    assert "ruft_auf:symbol:a:client.haupt->symbol:a:client.helfer" in g.kanten
    assert "ruft_auf:symbol:b:api.x->symbol:b:api.speichern" in g.kanten
    assert "gehoert_zu:symbol:a:client.senden->modul:sp/a/client.py" in g.kanten
    assert "gehoert_zu:modul:sp/a/client.py->space:a" in g.kanten
    assert g.knoten["symbol:a:client.senden"].datei == "sp/a/client.py"
    assert g.knoten["space:b"].typ == "space"

    vb = g.kanten["verbindet:dienst:a->dienst:b"]
    assert vb.attribute["env"] == ["B_URL"] and "space" not in vb.attribute
    assert g.knoten["dienst:b"].attribute["register"] == "b"
    assert all(k.quelle == "extraktor" for k in g.kanten.values())
    assert g.commit_submodul == "sub1"


def test_commits_aus_argumenten_sonst_aus_manifesten_nie_aus_der_basis(tmp_path):
    konfig = _repo(tmp_path)
    basis = Graph({}, {}, "alt_aussen", "alt_sub")
    g, _ = bauen.graph_bauen(konfig, ["a"], basis, _manifeste(konfig))
    assert (g.commit_aussen, g.commit_submodul) == ("sub1", "sub1")  # sp/* liegt nicht im Submodul
    g, _ = bauen.graph_bauen(konfig, ["a"], basis, _manifeste(konfig), commit_aussen="A", commit_sub="S")
    assert (g.commit_aussen, g.commit_submodul) == ("A", "S")
    g, _ = bauen.graph_bauen(konfig, [], basis, _manifeste(konfig))
    assert (g.commit_aussen, g.commit_submodul) == ("", "")


def test_seed_hop_a_nach_b_http_wird_verdrahtet(tmp_path):
    konfig = _repo(tmp_path)
    z = _zeile(CLIENT, '"{B}/api/x"')
    _schreibe(tmp_path / "docs/KETTEN.md", "\n".join([
        "### K1 · Fixture", "",
        "| # | Von → Nach | Mechanismus | Code-Stelle | Belegt durch | Zustand |",
        "|---|---|---|---|---|---|",
        f"| 1 | Client `senden` → Dienst `b:8130` | HTTP POST `/api/x` | `sp/a/client.py:{z}` | — | verdrahtet |",
        "",
    ]))
    manifeste = _manifeste(konfig)
    defs, _ = sql_ex.sql_definitionen(konfig)
    seed, _ = ks.seed_graph(konfig, manifeste, defs, "aussen1", "sub1")
    hop = next(k for k in seed.kanten.values() if k.art == "hop")
    assert hop.nach == "dienst:Dienst b" and [b.anker for b in hop.belege] == ["client.senden"]

    g, _ = bauen.graph_bauen(konfig, ["a", "b"], seed, manifeste)

    assert g.knoten["dienst:Dienst b"].attribute["register"] == "b"  # Port 8130 verbindet die Schreibweisen
    zustand = g.kanten[hop.id].attribute["zustand"]
    assert zustand["verdrahtet"] == "ja" and zustand["konfiguriert"] == "ja"
    assert zustand["gemessen"]["wert"] == "bricht"  # Waechter-Pruefung `probe` (K1) steht auf False


# ---------- Hop-Bestaetigung je Mechanismus ----------

def test_hop_sql_ja_und_nein(tmp_path):
    ja = _hop("K2", "1", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"))
    nein = _hop("K2", "2", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.helfer"))
    ueber_beleg = _hop("K2", "3", "komponente:A", "komponente:DB", "SQL",
                       _sym("sp/b/api.py", "api.speichern"), Beleg("sql", "sp/b/db/001_t.sql", "b.t", "d", "c", 1))
    g, _, _ = _bauen(tmp_path, ja, nein, ueber_beleg)
    assert _zustand(g, ja)["verdrahtet"] == "ja" and _zustand(g, ja)["nur_doku"] is False
    assert _zustand(g, nein)["verdrahtet"] == "nein" and _zustand(g, nein)["nur_doku"] is True
    assert _zustand(g, ueber_beleg)["verdrahtet"] == "ja"


def test_hop_http_ja_ueber_routentreffer(tmp_path):
    h = _hop("K3", "1", "komponente:A", "dienst:b", "HTTP", _sym("sp/a/client.py", "client.senden"))
    g, _, _ = _bauen(tmp_path, h)
    assert _zustand(g, h)["verdrahtet"] == "ja" and _zustand(g, h)["konfiguriert"] == "ja"


def test_hop_http_nur_konfiguriert_ohne_routentreffer(tmp_path):
    h = _hop("K3", "2", "komponente:A", "dienst:b", "HTTP", _sym("sp/a/client.py", "client.andere"))
    g, _, _ = _bauen(tmp_path, h)
    assert _zustand(g, h)["verdrahtet"] == "unbestimmt" and _zustand(g, h)["konfiguriert"] == "ja"


def test_hop_http_nein_nur_mit_client_der_das_ziel_nicht_erreicht(tmp_path):
    # Client (nutzt_dienst zu a), erreicht b aber nicht -> nein
    h = _hop("K3", "3", "komponente:A", "dienst:b", "HTTP", _sym("sp/a/client.py", "client.zu_a"))
    # kein Beleg ist Client: Helfer ohne HTTP bzw. Routen-Handler des Ziels -> unbestimmt statt falschem nein
    helfer = _hop("K3", "4", "komponente:A", "dienst:b", "HTTP", _sym("sp/a/client.py", "client.helfer"))
    ziel_seite = _hop("K3", "5", "komponente:A", "dienst:b", "HTTP", _sym("sp/b/api.py", "api.x"),
                      _sym("sp/b/api.py", "api.speichern"))
    # Helfer + Client gemischt: der Client entscheidet
    gemischt = _hop("K3", "6", "komponente:A", "dienst:b", "HTTP", _sym("sp/a/client.py", "client.helfer"),
                    _sym("sp/a/client.py", "client.senden"))
    g, _, _ = _bauen(tmp_path, h, helfer, ziel_seite, gemischt)
    assert _zustand(g, h)["verdrahtet"] == "nein" and _zustand(g, h)["nur_doku"] is True
    assert _zustand(g, h)["konfiguriert"] == "unbestimmt"  # nutzt_dienst nur zu einem anderen Dienst
    for x in (helfer, ziel_seite):
        assert _zustand(g, x)["verdrahtet"] == "unbestimmt" and _zustand(g, x)["nur_doku"] is False
    assert _zustand(g, gemischt)["verdrahtet"] == "ja" and _zustand(g, gemischt)["konfiguriert"] == "ja"


def test_hop_http_ziel_als_komponente_ueber_register_namen(tmp_path):
    h = _hop("K3", "7", "komponente:A", "komponente:B", "HTTP", _sym("sp/a/client.py", "client.senden"))
    g, _, _ = _bauen(tmp_path, h)
    assert g.knoten["komponente:B"].attribute["register"] == "b" and g.knoten["komponente:B"].name == "B"
    assert _zustand(g, h)["verdrahtet"] == "ja" and _zustand(g, h)["konfiguriert"] == "ja"


def test_hop_mcp_ja(tmp_path):
    h = _hop("K4", "1", "komponente:A", "mcp_werkzeug:auftrag_holen", "MCP", _sym("sp/a/client.py", "client.helfer"))
    fehlt = _hop("K4", "2", "komponente:A", "mcp_werkzeug:gibt_es_nicht", "MCP",
                 _sym("sp/a/client.py", "client.helfer"))
    # Ruling 32: das Belegsymbol ist selbst die registrierte Werkzeugfunktion -> ja (auch ohne MCP-Endpunkt)
    beleg_ist_werkzeug = _hop("K4", "3", "komponente:A", "dienst:b", "MCP",
                              _sym("sp/b/server.py", "server.auftrag_holen"))
    beleg_statt_name = _hop("K4", "4", "komponente:A", "mcp_werkzeug:gibt_es_nicht", "MCP",
                            _sym("sp/b/server.py", "server.auftrag_holen"))
    ohne_endpunkt = _hop("K4", "5", "komponente:A", "dienst:b", "MCP", _sym("sp/a/client.py", "client.helfer"))
    g, _, _ = _bauen(tmp_path, h, fehlt, beleg_ist_werkzeug, beleg_statt_name, ohne_endpunkt)
    assert _zustand(g, h)["verdrahtet"] == "ja"
    assert _zustand(g, fehlt)["verdrahtet"] == "nein"
    assert _zustand(g, beleg_ist_werkzeug)["verdrahtet"] == "ja"
    assert _zustand(g, beleg_statt_name)["verdrahtet"] == "ja"
    assert _zustand(g, ohne_endpunkt)["verdrahtet"] == "unbestimmt"


def test_hop_python_ja_und_ein_beleg_unbestimmt(tmp_path):
    ja = _hop("K5", "1", "komponente:A", "komponente:B", "Python",
              _sym("sp/a/client.py", "client.haupt"), _sym("sp/a/client.py", "client.helfer"))
    nein = _hop("K5", "2", "komponente:A", "komponente:B", "Python",
                _sym("sp/a/client.py", "client.lesen"), _sym("sp/a/client.py", "client.helfer"))
    einer = _hop("K5", "3", "komponente:A", "komponente:B", "Python", _sym("sp/a/client.py", "client.haupt"))
    # ruft_auf gibt es nur je Manifest: Belege in verschiedenen Spaces (oder ausserhalb) -> unbestimmt
    quer = _hop("K5", "4", "komponente:A", "komponente:B", "Python",
                _sym("sp/a/client.py", "client.haupt"), _sym("sp/b/api.py", "api.x"))
    aussen = _hop("K5", "5", "komponente:A", "komponente:B", "Python", _sym("sp/a/client.py", "client.lesen"),
                  _sym("sp/a/client.py", "client.helfer"), _sym("scripts/x.py", "scripts.x.f"))
    g, _, _ = _bauen(tmp_path, ja, nein, einer, quer, aussen)
    assert _zustand(g, ja)["verdrahtet"] == "ja"
    assert _zustand(g, nein)["verdrahtet"] == "nein"
    for x in (einer, quer, aussen):
        assert _zustand(g, x)["verdrahtet"] == "unbestimmt"


def test_hop_ohne_quellsymbol_oder_anderer_mechanismus_unbestimmt(tmp_path):
    ohne = _hop("K6", "1", "komponente:A", "tabelle:b.t", "SQL")
    aussen = _hop("K6", "2", "komponente:A", "tabelle:b.t", "SQL", _sym("scripts/x.py", "scripts.x.f"))
    datei = _hop("K6", "3", "komponente:A", "komponente:B", "Datei", _sym("sp/a/client.py", "client.helfer"))
    g, _, _ = _bauen(tmp_path, ohne, aussen, datei)
    for h in (ohne, aussen, datei):
        assert _zustand(g, h)["verdrahtet"] == "unbestimmt" and _zustand(g, h)["nur_doku"] is False


def test_nur_doku_aus_nein_wird_zurueckgenommen_seed_nur_doku_bleibt(tmp_path):
    h = _hop("K2", "2", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.helfer"))
    seed_doku = _hop("K2", "9", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"),
                     nur_doku=True)
    g1, _, _ = _bauen(tmp_path, h, seed_doku)
    assert _zustand(g1, h)["nur_doku"] is True and _zustand(g1, seed_doku)["nur_doku"] is True
    # helfer liest jetzt b.t -> ja -> das von bauen gesetzte nur_doku faellt wieder weg
    konfig = _repo(tmp_path)
    _schreibe(tmp_path / "sp/a/client.py", CLIENT.replace("    return 1\n", '    return "from b.t"\n', 1))
    g2, _ = bauen.graph_bauen(konfig, ["a"], g1, _manifeste(konfig))
    assert _zustand(g2, h)["verdrahtet"] == "ja" and _zustand(g2, h)["nur_doku"] is False
    assert _zustand(g2, seed_doku)["nur_doku"] is True


# ---------- konfiguriert / env_unbekannt ----------

def test_env_unbekannt_pruefpunkt(tmp_path):
    h = _hop("K7", "1", "komponente:A", "dienst:x", "HTTP", _sym("sp/a/client.py", "client.fremd"))
    g, pp, _ = _bauen(tmp_path, h)
    unbekannt = [p for p in pp if p.art == "env_unbekannt"]
    assert len(unbekannt) == 1 and "X_URL" in unbekannt[0].text  # zweimal benutzt, einmal gemeldet
    assert unbekannt[0].datei == "sp/a/client.py" and unbekannt[0].zeile == _zeile(CLIENT, '"X_URL"')
    assert _zustand(g, h)["konfiguriert"] == "nein"
    assert not any("NUR_IM_TEST_URL" in p.text for p in pp)  # Testdatei meldet nichts


def test_konfiguriert_nur_fuer_http_und_nur_zum_ziel(tmp_path):
    sql = _hop("K7", "2", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.senden"))
    anderes_ziel = _hop("K7", "3", "komponente:A", "dienst:a", "HTTP", _sym("sp/a/client.py", "client.senden"))
    mit_unbekannt = _hop("K7", "4", "komponente:A", "dienst:b", "HTTP", _sym("sp/a/client.py", "client.senden"),
                         _sym("sp/a/client.py", "client.fremd"))
    g, _, _ = _bauen(tmp_path, sql, anderes_ziel, mit_unbekannt)
    assert _zustand(g, sql)["konfiguriert"] == "unbestimmt"
    assert _zustand(g, anderes_ziel)["konfiguriert"] == "unbestimmt"  # senden nutzt b, nicht a
    assert _zustand(g, mit_unbekannt)["konfiguriert"] == "nein"


# ---------- Testdateien ----------

def test_testdateien_erzeugen_keine_kanten(tmp_path):
    konfig = _repo(tmp_path)
    g, _ = bauen.graph_bauen(konfig, ["a", "b"], None, _manifeste(konfig))
    assert not any("tests/" in (k.datei or "") for k in g.knoten.values())
    assert not any("tests" in k.von or "tests" in k.nach for k in g.kanten.values())
    assert "route:GET /api/t@b" not in g.knoten
    assert "mcp_werkzeug:test_werkzeug" not in g.knoten


# ---------- inkrementell ----------

def _teil(g, space):
    knoten = {i: k.to_dict() for i, k in g.knoten.items() if k.space == space}
    kanten = {i: k.to_dict() for i, k in g.kanten.items() if k.attribute.get("space") == space}
    return knoten, kanten


def test_nur_genannte_spaces_werden_neu_gebaut(tmp_path):
    hop = _hop("K2", "1", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"))
    konfig = _repo(tmp_path)
    g1, _ = bauen.graph_bauen(konfig, ["a", "b"], _basis(hop), _manifeste(konfig))
    # beide Spaces aendern sich; neu gebaut wird nur a
    _schreibe(tmp_path / "sp/b/api.py", API + '\n\n@app.get("/api/neu")\ndef neu():\n    return 1\n')
    _schreibe(tmp_path / "sp/a/client.py", CLIENT + '\n\ndef dazu():\n    return requests.get(f"{B}/api/neu")\n')
    g2, _ = bauen.graph_bauen(konfig, ["a"], g1, _manifeste(konfig))

    assert _teil(g2, "b") == _teil(g1, "b")  # b byte-gleich aus der Basis
    alt_b, neu_b = Graph(), Graph()
    for gg, ziel in ((g1, alt_b), (g2, neu_b)):
        for i, k in gg.kanten.items():
            if k.attribute.get("space") == "b":
                ziel.kante_hinzu(k)
    d = graph_diff(alt_b, neu_b)
    assert not d.neue_kanten and not d.weg_kanten and not d.belege_geaendert
    assert "route:GET /api/neu@b" not in g2.knoten
    assert "nutzt_dienst:symbol:a:client.dazu->dienst:b" in g2.kanten  # a ist neu gebaut
    assert hop.id in g2.kanten and "verbindet:dienst:a->dienst:b" in g2.kanten


def test_inkrementell_entfernte_route_und_symbol_verschwinden(tmp_path):
    hop = _hop("K3", "1", "komponente:A", "dienst:b", "HTTP", _sym("sp/a/client.py", "client.senden"))
    konfig = _repo(tmp_path)
    g1, _ = bauen.graph_bauen(konfig, ["a", "b"], _basis(hop), _manifeste(konfig))
    assert "route:POST /api/x@b" in g1.knoten and "symbol:b:api.speichern" in g1.knoten
    assert _zustand(g1, hop)["verdrahtet"] == "ja"
    # Route /api/x und Funktion speichern entfernt, neu gebaut wird nur b
    ohne = API.replace('@app.post("/api/x")\n', "").split("\n\ndef speichern")[0] + "\n"
    _schreibe(tmp_path / "sp/b/api.py", ohne)
    g2, _ = bauen.graph_bauen(konfig, ["b"], g1, _manifeste(konfig))
    weg_knoten = {"route:POST /api/x@b", "symbol:b:api.speichern"}
    assert not weg_knoten & set(g2.knoten)
    assert not [i for i, k in g2.kanten.items() if k.von in weg_knoten or k.nach in weg_knoten]
    assert "symbol:b:api.x" in g2.knoten and "tabelle:b.t" in g2.knoten  # a liest b.t weiterhin
    assert _teil(g2, "a") == _teil(g1, "a")
    assert _zustand(g2, hop)["verdrahtet"] == "unbestimmt"  # nutzt_dienst bleibt, Routentreffer fehlt


def test_ohne_aenderung_identischer_digest(tmp_path):
    konfig = _repo(tmp_path)
    hop = _hop("K2", "1", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"))
    g1, _ = bauen.graph_bauen(konfig, ["a", "b"], _basis(hop), _manifeste(konfig))
    g2, _ = bauen.graph_bauen(konfig, ["a", "b"], g1, _manifeste(konfig))
    g3, _ = bauen.graph_bauen(konfig, ["b"], g2, _manifeste(konfig))
    assert g1.digest() == g2.digest() == g3.digest()


def test_seed_hop_zustand_neu_rest_unberuehrt_basis_unveraendert(tmp_path):
    k1 = _hop("K1", "1", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"),
              gemessen="laeuft")
    k9 = _hop("K9", "1", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"),
              gemessen="laeuft")
    k9.attribute["belegt_durch"] = "008"
    konfig = _repo(tmp_path)
    basis = _basis(k1, k9)
    vorher = copy.deepcopy({i: k.to_dict() for i, k in basis.kanten.items()})
    g, pp = bauen.graph_bauen(konfig, ["a", "b"], basis, _manifeste(konfig))

    assert {i: k.to_dict() for i, k in basis.kanten.items()} == vorher  # Basis nicht veraendert
    for h in (k1, k9):
        neu = g.kanten[h.id]
        assert (neu.von, neu.nach, neu.quelle, neu.belege) == (h.von, h.nach, h.quelle, h.belege)
        assert neu.attribute["kette"] == h.attribute["kette"] and neu.attribute["nr"] == h.attribute["nr"]
        assert neu.attribute["zustand"]["verdrahtet"] == "ja"
    assert g.kanten[k9.id].attribute["belegt_durch"] == "008"
    assert g.kanten[k1.id].attribute["zustand"]["gemessen"]["wert"] == "bricht"
    assert g.kanten[k1.id].attribute["zustand"]["gemessen"]["zeit"] is not None
    assert g.kanten[k9.id].attribute["zustand"]["gemessen"] == {"wert": "laeuft", "zeit": None}  # keine Probe
    assert [p.text for p in pp if p.art == "waechter_unbekannt"] == ["Waechter-Pruefung ohne Zuordnung: fremd"]


# ---------- Fehler ----------

def test_fehlendes_manifest_oder_space_wurzel_wirft(tmp_path):
    konfig = _repo(tmp_path)
    with pytest.raises(bauen.BauFehler):
        bauen.graph_bauen(konfig, ["a"], None, {})
    with pytest.raises(bauen.BauFehler):
        bauen.graph_bauen(konfig, ["gibt_es_nicht"], None, _manifeste(konfig))
    weg = dataclasses.replace(konfig, spaces={**konfig.spaces, "c": SpaceKonfig("c", "sp/c", "werkzeuge")})
    with pytest.raises(bauen.BauFehler):
        bauen.graph_bauen(weg, ["c"], None, {"c": _manifest(tmp_path / "sp/a")})


def test_extraktor_fehler_und_kaputter_datensatz_werden_pruefpunkte(tmp_path, monkeypatch):
    konfig = _repo(tmp_path)

    def kaputt(*_a, **_k):
        raise RuntimeError("boom")

    echte = http_clients.http_aufrufe

    def mit_kaputtem(space, konfig_, manifest):
        aus = echte(space, konfig_, manifest)
        if space.name != "a":
            return aus
        return aus + [http_clients.HttpAufruf("sp/a/client.py", 1, "GET", "/x", None, None)]  # type: ignore[arg-type]

    monkeypatch.setattr(routen, "routen_extrahieren", kaputt)
    monkeypatch.setattr(http_clients, "http_aufrufe", mit_kaputtem)
    g, pp = bauen.graph_bauen(konfig, ["a", "b"], None, _manifeste(konfig))
    arten = Counter(p.art for p in pp)
    assert arten["extraktor_fehler"] == 2 and arten["datensatz_fehler"] == 1
    assert "nutzt_dienst:symbol:a:client.senden->dienst:b" in g.kanten
    assert not any(k.art == "definiert_route" for k in g.kanten.values())


# ---------- echtes Repo (C.9.3) ----------

@pytest.mark.skipif(not (REPO / "vibemind-os/spaces/marketing").is_dir() or not korpus.manifest_verfuegbar(),
                    reason="Submodul/Manifest fehlt")
def test_echtes_repo_rauchtest_c93():
    konfig = vm_laden()
    commit_aussen = korpus.commit_von(konfig.repo)
    commit_sub = korpus.commit_von(konfig.repo / "vibemind-os")
    t0 = time.perf_counter()
    manifeste = {n: korpus.manifest_bauen(s, konfig, commit_sub) for n, s in konfig.spaces.items()}
    t1 = time.perf_counter()
    defs, _ = sql_ex.sql_definitionen(konfig)
    seed, _ = ks.seed_graph(konfig, manifeste, defs, commit_aussen, commit_sub)
    t2 = time.perf_counter()
    g, pp = bauen.graph_bauen(konfig, list(konfig.spaces), seed, manifeste, sql_defs=defs,
                              commit_aussen=commit_aussen, commit_sub=commit_sub)
    t3 = time.perf_counter()

    kanten = Counter(k.art for k in g.kanten.values())
    knoten = Counter(k.typ for k in g.knoten.values())
    hops = [k for k in g.kanten.values() if k.art == "hop"]
    verteilung: dict[str, Counter] = {}
    for h in hops:
        z = h.attribute["zustand"]
        mech = h.attribute.get("mechanismus_typ", "?")
        verteilung.setdefault(mech, Counter())[f"v={z['verdrahtet']}"] += 1
        verteilung[mech][f"k={z['konfiguriert']}"] += 1
    print("\nKANTEN:", dict(sorted(kanten.items())))
    print("KNOTEN:", dict(sorted(knoten.items())))
    for mech in sorted(verteilung):
        print("HOPS", mech, dict(sorted(verteilung[mech].items())))
    print("PRUEFPUNKTE:", dict(Counter(p.art for p in pp)))
    print(f"ZEIT manifeste={t1 - t0:.1f}s seed={t2 - t1:.1f}s bauen={t3 - t2:.1f}s")
    assert len(hops) >= 95 and kanten["ruft_auf"] > 0 and kanten["definiert_route"] > 0
    assert kanten["liest_schreibt"] > 0 and kanten["registriert"] > 0
    assert not [p for p in pp if p.art == "extraktor_fehler"]


def test_waechter_zuordnung_je_hop_ruling_55b(tmp_path):
    """`K1/2` misst nur Hop 2; Hops ohne eigene Probe in dieser Kette verlieren eine alte Waechter-Messung, ein Wert
    aus dem Dokument (ohne Zeit) bleibt; die Uebersicht fasst nur Hops mit Messung zusammen."""
    from selbstbild.doku import render_ketten
    h1 = _hop("K1", "1", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"))
    h1.attribute["zustand"]["gemessen"] = {"wert": "bricht", "zeit": "2026-10-27T00:00:00+00:00"}
    h2 = _hop("K1", "2", "komponente:B", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"))
    h3 = _hop("K1", "3", "komponente:C", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"),
              gemessen="laeuft")
    konfig = dataclasses.replace(_repo(tmp_path), waechter_zuordnung={"fremd": ("K1/2",)})
    g, _ = bauen.graph_bauen(konfig, ["a", "b"], _basis(h1, h2, h3), _manifeste(konfig))
    assert _zustand(g, h2)["gemessen"]["wert"] == "laeuft" and _zustand(g, h2)["gemessen"]["zeit"]
    assert _zustand(g, h1)["gemessen"] == {"wert": "keine_probe", "zeit": None}
    assert _zustand(g, h3)["gemessen"] == {"wert": "laeuft", "zeit": None}
    assert render_ketten.kette_zusammenfassung(g, "K1").endswith("gemessen läuft")


# ---------- Final-Fix-Welle (Ruling 57) ----------

def test_i9_registerluecke_ist_keine_widerlegung(tmp_path):
    """I9: hat ein Client `env_unbekannt` (Env-Name nicht im Register), ist das HTTP-Ergebnis `unbestimmt`, nie
    `nein` -- der unbekannte Env-Name kann genau das Ziel meinen."""
    nur_unbekannt = _hop("K8", "1", "komponente:A", "dienst:b", "HTTP", _sym("sp/a/client.py", "client.fremd"))
    gemischt = _hop("K8", "2", "komponente:A", "dienst:a", "HTTP", _sym("sp/a/client.py", "client.senden"),
                    _sym("sp/a/client.py", "client.fremd"))
    ohne = _hop("K8", "3", "komponente:A", "dienst:a", "HTTP", _sym("sp/a/client.py", "client.senden"))
    g, _, _ = _bauen(tmp_path, nur_unbekannt, gemischt, ohne)
    assert _zustand(g, nur_unbekannt)["verdrahtet"] == "unbestimmt"
    assert _zustand(g, gemischt)["verdrahtet"] == "unbestimmt"
    assert _zustand(g, ohne)["verdrahtet"] == "nein"  # ohne Registerluecke bleibt die Widerlegung


def test_i8_veralteter_waechter_gibt_keine_probe_und_pruefpunkt(tmp_path):
    """I8: Zustandsdatei aelter als `waechter_max_alter_min` -> alle Messungen `keine_probe` + Pruefpunkt
    `waechter_veraltet` (ein Text je Tag, damit die Pruefliste ihn je Tag einmal fuehrt)."""
    import os

    h = _hop("K1", "1", "komponente:A", "tabelle:b.t", "SQL", _sym("sp/a/client.py", "client.lesen"))
    konfig = _repo(tmp_path)
    g, pp = bauen.graph_bauen(konfig, ["a", "b"], _basis(h), _manifeste(konfig))
    assert _zustand(g, h)["gemessen"]["wert"] == "bricht" and not [p for p in pp if p.art == "waechter_veraltet"]
    alt = time.time() - 61 * 60
    os.utime(tmp_path / "logs/w.json", (alt, alt))
    g, pp = bauen.graph_bauen(konfig, ["a", "b"], _basis(h), _manifeste(konfig))
    assert _zustand(g, h)["gemessen"] == {"wert": "keine_probe", "zeit": None}
    veraltet = [p for p in pp if p.art == "waechter_veraltet"]
    assert len(veraltet) == 1 and time.strftime("%Y-%m-%d") in veraltet[0].text
    assert veraltet[0].datei == "logs/w.json"
    g, pp = bauen.graph_bauen(dataclasses.replace(konfig, waechter_max_alter_min=120), ["a", "b"], _basis(h),
                              _manifeste(konfig))
    assert _zustand(g, h)["gemessen"]["wert"] == "bricht"
