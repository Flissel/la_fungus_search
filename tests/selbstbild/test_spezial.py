"""Spezialextraktoren (Routen, MCP-Werkzeuge, HTTP-Clients, Compose), portiert aus dem Doku-Loop."""
from __future__ import annotations

from selbstbild.parser.basis import parsen
from selbstbild.spezial import repo_binden, spezial_compose, spezial_python
from selbstbild.typen import (
    PLATZHALTER_REPO as P, Beleg, Kante, Knoten, id_dienst, id_env, id_mcp, id_route, id_symbol,
)


def _py(quelle: str, pfad: str = "a.py"):
    return spezial_python(P, pfad, quelle.encode("utf-8"))


def _routen(quelle: str, pfad: str = "a.py"):
    knoten, kanten, einstiege = _py(quelle, pfad)
    return (
        {k.id for k in knoten if k.art == "route"},
        {(k.von, k.nach) for k in kanten if k.art == "definiert_route"},
        [e for e in einstiege if e.art == "route"],
    )


def _r(methode: str, pfad: str) -> str:
    return id_route(P, methode, pfad)


# --------------------------------------------------------------------------- Routen

def test_route_knoten_kante_einstieg():
    knoten, kanten, einstiege = _py('@app.get("/kunden/{id}")\ndef lesen(id): ...\n', "api/k.py")
    (route,) = [k for k in knoten if k.art == "route"]
    assert route.id == "route:@:GET /kunden/{}"
    assert route.id == id_route(P, "GET", "/kunden/{}")
    (kante,) = [k for k in kanten if k.art == "definiert_route"]
    assert kante.von == id_symbol(P, "api/k.py", "lesen") == "symbol:@:api/k.py#lesen"
    assert kante.nach == route.id and kante.sicher
    assert kante.beleg is not None and kante.beleg.repo == P and kante.beleg.zeile == 1
    (e,) = einstiege
    assert (e.art, e.symbol, e.zeile) == ("route", "lesen", 1)


def test_fastapi_router_und_flask():
    ids, _, _ = _routen(
        "from fastapi import FastAPI, APIRouter\n"
        "app = FastAPI()\n"
        "pult_router = APIRouter(prefix='/api/pult')\n"
        "@app.post('/api/x/{iid}/')\n"
        "def x(iid): ...\n"
        "@pult_router.get('/marke/aussehen?v=1')\n"
        "def a(): ...\n"
        "from flask import Blueprint\n"
        "bp = Blueprint('b', __name__)\n"
        "@bp.route('/h', methods=['GET'])\n"
        "def h(): ...\n"
    )
    assert ids == {_r("POST", "/api/x/{}"), _r("GET", "/api/pult/marke/aussehen"), _r("GET", "/h")}


def test_include_router_praefix_in_derselben_datei():
    ids, _, _ = _routen(
        "r_router = APIRouter(prefix='/inner')\n"
        "@r_router.get('/a')\ndef a(): ...\n"
        "app.include_router(r_router, prefix='/v1')\n"
        "app.include_router(r_router, prefix='/v2')\n"
    )
    assert ids == {_r("GET", "/v1/inner/a"), _r("GET", "/v2/inner/a")}


def test_nichtliteral_fremde_objekte_und_kaputtes_wird_ignoriert():
    ids, _, _ = _routen(
        "P = '/x'\n@app.get(P)\ndef a(): ...\n"
        "@app.get(f'/y/{P}')\ndef b(): ...\n"
        "@cache.get('/nein')\ndef n(): ...\n"
        "@app.get('/ok')\ndef c(): ...\n"
    )
    assert ids == {_r("GET", "/ok")}
    assert _py("def (:\n") == ((), (), ())
    assert spezial_python(P, "b.py", b"\xff\xfe\x00\x80") == ((), (), ())


def test_generische_pfade_und_methoden_liste():
    knoten, _, _ = _py(
        "@app.get('/health')\ndef h(): ...\n"
        "@app.api_route('/m', methods=['GET', 'POST'])\ndef m(): ...\n"
    )
    je_id = {k.id: dict(k.attr) for k in knoten if k.art == "route"}
    assert je_id[_r("GET", "/health")].get("generisch") == "1"
    assert _r("GET", "/m") in je_id and _r("POST", "/m") in je_id
    assert "generisch" not in je_id[_r("POST", "/m")]


def test_testdateien_liefern_nichts():
    q = "@app.get('/x')\ndef f(): ...\n@mcp.tool()\ndef t(): ...\n"
    for pfad in ("tests/x.py", "test_x.py", "foo_test.py", "conftest.py"):
        assert _py(q, pfad) == ((), (), ()), pfad


def test_methode_in_klasse_qualname():
    _, kanten, _ = _py("class K:\n    @app.get('/k')\n    def m(self): ...\n")
    (k,) = [x for x in kanten if x.art == "definiert_route"]
    assert k.von == id_symbol(P, "a.py", "K.m")


# --------------------------------------------------------------------------- MCP

def test_mcp_decorator_und_aufrufe():
    knoten, kanten, einstiege = _py(
        "@mcp.tool()\n"
        "def think(message): ...\n"
        "@server.tool(name='suche')\n"
        "async def x(): ...\n"
        "@mcp.tool\n"
        "def nackt(): ...\n"
        "@app.get('/x')\n"
        "def kein_tool(): ...\n"
        "Tool(name='t1', fn=f)\n"
        "register_tool('t2', f)\n"
    )
    namen = {k.name for k in knoten if k.art == "mcp_werkzeug"}
    assert namen == {"think", "suche", "nackt", "t1", "t2"}
    reg = {(k.von, k.nach) for k in kanten if k.art == "registriert"}
    assert (id_symbol(P, "a.py", "think"), id_mcp(P, "think")) in reg
    assert (id_symbol(P, "a.py", "x"), id_mcp(P, "suche")) in reg
    # Aufruf ausserhalb einer Funktion haengt an der Datei
    assert ("datei:@:a.py", id_mcp(P, "t1")) in reg
    assert {e.symbol for e in einstiege if e.art == "mcp"} >= {"think", "x", "nackt"}
    think = next(k for k in knoten if k.id == id_mcp(P, "think"))
    assert think.beleg is not None and think.beleg.zeile == 2


def test_mcp_werkzeuge_liste_name_und_attribut():
    knoten, kanten, _ = _py(
        "from . import werkzeuge\n"
        "def lokal(): ...\n"
        "WERKZEUGE = (\n    werkzeuge.versand, lokal,\n    werkzeuge.weg,\n)\n",
        "claw/server.py",
    )
    namen = {k.name for k in knoten if k.art == "mcp_werkzeug"}
    assert namen == {"versand", "lokal", "weg"}
    reg = {(k.von, k.nach) for k in kanten if k.art == "registriert"}
    assert (id_symbol(P, "claw/server.py", "lokal"), id_mcp(P, "lokal")) in reg
    assert ("datei:@:claw/server.py", id_mcp(P, "versand")) in reg


def test_mcp_werkzeuge_liste_mit_plus_verkettet():
    knoten, _, _ = _py("WERKZEUGE = (a, b) + (c,)\nWEITER: list = [d]\n")
    assert {k.name for k in knoten if k.art == "mcp_werkzeug"} == {"a", "b", "c"}


# --------------------------------------------------------------------------- HTTP-Clients

PULT = (
    "import os\n"
    "import urllib.request\n"
    'BASIS = os.environ.get("MARKETING_PULT_URL", "http://127.0.0.1:5510")\n'
    "def holen(pfad):\n"
    '    req = urllib.request.Request(f"{BASIS}/api/pult/{pfad}")\n'
    "    return urllib.request.urlopen(req)\n"
)


def _http(quelle: str, pfad: str = "a.py") -> list[Kante]:
    return [k for k in _py(quelle, pfad)[1] if k.art == "nutzt_dienst"]


def test_http_pult_fixture():
    (k,) = _http(PULT)
    assert k.von == id_symbol(P, "a.py", "holen")
    assert k.nach == id_env("MARKETING_PULT_URL")
    assert k.beleg is not None and k.beleg.anker == "GET /api/pult/{}" and k.beleg.zeile == 5
    assert "127.0.0.1" not in repr(k)  # nur Namen, nie Werte


def test_http_request_und_urlopen_zaehlen_einmal():
    assert len(_http(PULT)) == 1
    q = (
        "import os, urllib.request\n"
        'B = os.getenv("H_URL")\n'
        'urllib.request.urlopen(urllib.request.Request(B + "/x", data=b"1"))\n'
    )
    (k,) = _http(q)
    assert k.beleg is not None and k.beleg.anker == "POST /x"


def test_http_requests_httpx_und_templates():
    q = (
        "import os, requests, httpx\n"
        "BASE = 'http://h:1'\n"
        "def a(x):\n"
        "    requests.post(os.environ['A_URL'] + '/api/x?y=1')\n"
        "    requests.get('%s/api/%s' % (os.getenv('B_URL'), x))\n"
        "    requests.put('{}/api/{}'.format(os.getenv('C_URL'), x))\n"
        "    requests.delete(BASE + '/api/d')\n"
        "    requests.get(x)\n"
        "async def b():\n"
        "    async with httpx.AsyncClient() as c:\n"
        "        await c.get(f'{_basis()}/api/c')\n"
        "def _basis():\n"
        "    return os.environ.get('D_URL', 'x').rstrip('/')\n"
    )
    je_ziel = {(k.nach, k.beleg.anker if k.beleg else None) for k in _http(q)}
    assert je_ziel == {
        (id_env("A_URL"), "POST /api/x"),
        (id_env("B_URL"), "GET /api/{}"),
        (id_env("C_URL"), "PUT /api/{}"),
        (id_env("D_URL"), "GET /api/c"),
    }  # Aufrufe ohne Env-Namen (BASE-Literal, requests.get(x)) liefern keine Kante


def test_http_request_mit_keyword_url_und_methode():
    q = (
        "import os, requests\n"
        "U = os.getenv('Z_URL')\n"
        "requests.request(method='PUT', url=U + '/a/b')\n"
        "requests.request('GET', url=U + '/c')\n"
    )
    assert sorted(k.beleg.anker for k in _http(q) if k.beleg) == ["GET /c", "PUT /a/b"]


# --------------------------------------------------------------------------- Compose

STACK = """
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
  sales-stt:
    image: x
    ports:
      - "127.0.0.1:2790:2790"
      - {published: 2791, target: 80}
      - 2792
      - "kaputt"
  brain-core:
    depends_on:
      - qdrant
    environment:
      QDRANT_URL: http://qdrant:6333
      OHNE_VORGABE: ${NIX}
      QDRANT_HOST: qdrant
      API_KEY: geheim-wert-123
      EXTERN: https://example.org/x
  sales-mcp:
    depends_on:
      sales-stt:
        condition: service_started
    links:
      - "qdrant:q"
    environment:
      - STT_URL=${STT_URL:-http://sales-stt:2790}
      - ANDERS=${ANDERS-http://qdrant:6333/}
"""


def _compose(text: str, pfad: str = "docker-compose.yml"):
    return spezial_compose(P, pfad, text.encode("utf-8"))


def test_compose_dienste_und_ports():
    knoten, _ = _compose(STACK)
    je = {k.name: dict(k.attr) for k in knoten}
    assert {k.id for k in knoten} == {id_dienst(n) for n in ("qdrant", "sales-stt", "brain-core", "sales-mcp")}
    assert all(k.art == "dienst" for k in knoten)
    assert je["qdrant"]["ports"] == "6333"
    assert je["sales-stt"]["ports"] == "2790,2791,2792"


def test_compose_verbindet_kanten_depends_links_urls():
    _, kanten = _compose(STACK)
    assert all(k.art == "verbindet" for k in kanten)
    paare = {(k.von, k.nach, k.beleg.anker if k.beleg else None) for k in kanten}
    d = id_dienst
    assert paare == {
        (d("brain-core"), d("qdrant"), "depends_on"),
        (d("brain-core"), d("qdrant"), "QDRANT_URL"),
        (d("brain-core"), d("qdrant"), "QDRANT_HOST"),
        (d("sales-mcp"), d("sales-stt"), "depends_on"),
        (d("sales-mcp"), d("qdrant"), "links"),
        (d("sales-mcp"), d("sales-stt"), "STT_URL"),
        (d("sales-mcp"), d("qdrant"), "ANDERS"),
    }


def test_compose_speichert_keine_werte():
    knoten, kanten = _compose(STACK)
    text = repr(knoten) + repr(kanten)
    for wert in ("geheim-wert-123", "example.org", "http://qdrant:6333"):
        assert wert not in text


def test_compose_nackter_name_nur_bei_hostartigem_schluessel_und_keine_selbstkante():
    _, kanten = _compose(
        "services:\n  postgres:\n    environment:\n      SELF_HOST: postgres\n"
        "  app:\n    environment:\n      POSTGRES_PASSWORD: postgres\n      POSTGRES_USER: postgres\n"
        "      DB_HOST: postgres\n      SELF_URL: http://app:1/\n")
    assert [(k.von, k.nach, k.beleg.anker) for k in kanten] == [
        (id_dienst("app"), id_dienst("postgres"), "DB_HOST")]


def test_compose_rendered_kaputt_und_ohne_services():
    assert _compose(STACK, ".vibemind-stack-rendered-1.yml") == ((), ())
    assert _compose(STACK, "x/partial-a.yml") == ((), ())
    assert _compose("a: [", "c.yml") == ((), ())
    assert _compose("version: '3'\n") == ((), ())


# --------------------------------------------------------------------------- repo_binden / Verdrahtung

def test_repo_binden_ersetzt_ids_und_beleg():
    beleg = Beleg(P, "a.py", 1, "x", "d")
    k = Kante("definiert_route", id_symbol(P, "a.py", "f"), id_route(P, "GET", "/a"), True, beleg)
    g = repo_binden(k, "meinrepo")
    assert isinstance(g, Kante)
    assert g.von == "symbol:meinrepo:a.py#f" and g.nach == "route:meinrepo:GET /a"
    assert g.beleg is not None and g.beleg.repo == "meinrepo"
    n = repo_binden(Knoten(id_mcp(P, "t"), "mcp_werkzeug", "t", beleg), "r2")
    assert isinstance(n, Knoten) and n.id == "mcp_werkzeug:r2:t" and n.beleg and n.beleg.repo == "r2"
    # Ids ohne Platzhalter und Knoten ohne Beleg bleiben unberuehrt
    e = repo_binden(Knoten(id_env("X"), "env", "X", None), "r2")
    assert isinstance(e, Knoten) and e.id == "env:X" and e.beleg is None
    d = repo_binden(Knoten(id_dienst("db"), "dienst", "db", beleg), "r2")
    assert isinstance(d, Knoten) and d.id == "dienst:db" and d.beleg and d.beleg.repo == "r2"
    # Doppelpunkt im Pfad-Teil bleibt erhalten
    r = repo_binden(Knoten("route:@:GET /a:b", "route", "x", None), "r2")
    assert isinstance(r, Knoten) and r.id == "route:r2:GET /a:b"


def test_parser_python_haengt_spezial_ein():
    res = parsen("python", "api/k.py", b'@app.get("/kunden")\ndef lesen(): ...\n')
    assert any(k.id == "route:@:GET /kunden" for k in res.extra_knoten)
    assert any(k.art == "definiert_route" for k in res.extra_kanten)
    assert any(e.art == "route" and e.symbol == "lesen" for e in res.einstiege)


def test_parser_yaml_ruft_compose_bei_compose_dateinamen():
    res = parsen("yaml", "infra/docker-compose.prod.yaml", STACK.encode())
    assert any(k.art == "dienst" for k in res.extra_knoten)
    assert any(k.art == "verbindet" for k in res.extra_kanten)
    assert any(e.name == "QDRANT_URL" for e in res.env)
    assert not parsen("yaml", "infra/andere.yml", STACK.encode()).extra_knoten
