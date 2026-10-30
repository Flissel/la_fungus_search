"""doku_loop: Routen-Extraktor (FastAPI/Starlette/Flask per ast)."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import korpus  # noqa: E402
from selbstbild.doku.extraktoren import routen  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402



def _schreibe(pfad: Path, text: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(text, encoding="utf-8")


@pytest.fixture()
def aufbau(tmp_path):
    konfig = dataclasses.replace(vm_laden(), repo=tmp_path)
    space = SpaceKonfig("fix", "sp", "decorator")
    manifest = SimpleNamespace(corpus_root=str(tmp_path / "sp"), documents=[])
    return tmp_path, konfig, space, manifest


def _paare(rs):
    return {(r.methode, r.pfad) for r in rs}


def test_fastapi_router_und_flask(aufbau):
    wurzel, konfig, space, manifest = aufbau
    _schreibe(wurzel / "sp" / "main.py", (
        "from fastapi import FastAPI, APIRouter\n"
        "app = FastAPI()\n"
        "pult_router = APIRouter(prefix='/api/pult')\n"
        "@app.post('/api/x/{iid}/')\n"
        "def x(iid): ...\n"
        "@pult_router.get('/marke/aussehen?v=1')\n"
        "def a(): ...\n"
    ))
    _schreibe(wurzel / "sp" / "web.py", (
        "from flask import Blueprint\n"
        "bp = Blueprint('b', __name__)\n"
        "@bp.route('/h', methods=['GET'])\n"
        "def h(): ...\n"
    ))
    rs = routen.routen_extrahieren(space, konfig, manifest)
    assert _paare(rs) == {
        ("POST", "/api/x/{}"), ("GET", "/api/pult/marke/aussehen"), ("GET", "/h")}
    assert all(r.datei.startswith("sp/") and r.zeile > 0 for r in rs)


def test_include_router_praefix_zusammengesetzt(aufbau):
    wurzel, konfig, space, manifest = aufbau
    _schreibe(wurzel / "sp" / "r.py", (
        "r_router = APIRouter(prefix='/inner')\n"
        "@r_router.get('/a')\ndef a(): ...\n"
    ))
    _schreibe(wurzel / "sp" / "main.py", (
        "from r import r_router\n"
        "app.include_router(r_router, prefix='/v1')\n"
        "app.include_router(r_router, prefix='/v2')\n"
    ))
    rs = routen.routen_extrahieren(space, konfig, manifest)
    assert _paare(rs) == {("GET", "/v1/inner/a"), ("GET", "/v2/inner/a")}


def test_include_ohne_praefix_und_ohne_include(aufbau):
    wurzel, konfig, space, manifest = aufbau
    _schreibe(wurzel / "sp" / "m.py", (
        "x_router = APIRouter()\n"
        "@x_router.delete('/d')\ndef d(): ...\n"
        "app.include_router(x_router)\n"
    ))
    assert _paare(routen.routen_extrahieren(space, konfig, manifest)) == {("DELETE", "/d")}


def test_nichtliteral_und_kaputte_dateien_uebersprungen(aufbau):
    wurzel, konfig, space, manifest = aufbau
    _schreibe(wurzel / "sp" / "m.py", (
        "P = '/x'\n@app.get(P)\ndef a(): ...\n"
        "@app.get(f'/y/{P}')\ndef b(): ...\n"
        "@app.get('/ok')\ndef c(): ...\n"
    ))
    _schreibe(wurzel / "sp" / "kaputt.py", "def (:\n")
    (wurzel / "sp" / "bin.py").write_bytes(b"\xff\xfe\x00\x80")
    rs, uebersprungen, _unaufgeloest = routen.routen_extrahieren_mit_zaehlung(space, konfig, manifest)
    assert _paare(rs) == {("GET", "/ok")}
    assert uebersprungen == 2


def test_andere_objekte_ignoriert(aufbau):
    wurzel, konfig, space, manifest = aufbau
    _schreibe(wurzel / "sp" / "m.py", "@cache.get('/nein')\ndef a(): ...\n")
    assert routen.routen_extrahieren(space, konfig, manifest) == []


def test_symbol_aus_manifest(aufbau):
    wurzel, konfig, space, _ = aufbau
    _schreibe(wurzel / "sp" / "m.py", "@app.get('/s')\ndef holen(): ...\n")
    doc = SimpleNamespace(file="m.py", start_line=2, end_line=2, document_id="m.holen")
    manifest = SimpleNamespace(corpus_root=str(wurzel / "sp"), documents=[doc])
    (r,) = routen.routen_extrahieren(space, konfig, manifest)
    assert r.symbol == "m.holen"


def test_generisch_health(aufbau):
    wurzel, konfig, space, manifest = aufbau
    _schreibe(wurzel / "sp" / "a.py", "@app.get('/health')\ndef h(): ...\n@app.get('/dup')\ndef d(): ...\n@app.get('/solo')\ndef s(): ...\n")
    _schreibe(wurzel / "sp" / "b.py", "@app.post('/dup')\ndef d(): ...\n")
    g = {r.pfad: r.generisch for r in routen.routen_extrahieren(space, konfig, manifest)}
    assert g == {"/health": True, "/dup": True, "/solo": False}


@pytest.mark.skipif(not (REPO / "vibemind-os/spaces/marketing").is_dir()
                    or not korpus.manifest_verfuegbar(), reason="Submodul fehlt")
def test_echt_marketing_mindestens_200_routen():
    konfig = vm_laden()
    space = konfig.spaces["marketing"]
    manifest = korpus.manifest_bauen(space, konfig, korpus.commit_von(konfig.repo))
    assert len(routen.routen_extrahieren(space, konfig, manifest)) >= 200


def test_include_praefix_pro_definierender_datei(aufbau):
    wurzel, konfig, space, manifest = aufbau
    sp = wurzel / "sp"
    _schreibe(sp / "pkg" / "__init__.py", "")
    _schreibe(sp / "pkg" / "a.py", "router = APIRouter()\n@router.get('/x')\ndef f(): ...\n")
    _schreibe(sp / "pkg" / "b.py", "router: APIRouter = APIRouter(prefix='/bb')\n@router.get('/y')\ndef g(): ...\n")
    _schreibe(sp / "pkg" / "c.py", "router = APIRouter()\n@router.get('/z')\ndef h(): ...\n")
    _schreibe(sp / "pkg" / "main.py", (
        "from .a import router as a_router\n"
        "from . import b\n"
        "app.include_router(a_router, prefix='/A')\n"
        "app.include_router(b.router, prefix='/B')\n"
        "app.include_router(fremd.router, prefix='/LECK')\n"
    ))
    rs, _, unaufgeloest = routen.routen_extrahieren_mit_zaehlung(space, konfig, manifest)
    assert _paare(rs) == {("GET", "/A/x"), ("GET", "/B/bb/y"), ("GET", "/z")}
    assert unaufgeloest == 1


def test_import_as_modul_und_absoluter_import(aufbau):
    wurzel, konfig, space, manifest = aufbau
    sp = wurzel / "sp"
    _schreibe(sp / "api" / "u.py", "router = APIRouter()\n@router.get('/u')\ndef f(): ...\n")
    _schreibe(sp / "api" / "v.py", "router = APIRouter()\n@router.get('/v')\ndef f(): ...\n")
    _schreibe(sp / "main.py", (
        "import api.u as m\nfrom api.v import router\n"
        "app.include_router(m.router, prefix='/U')\napp.include_router(router, prefix='/V')\n"
    ))
    assert _paare(routen.routen_extrahieren(space, konfig, manifest)) == {
        ("GET", "/U/u"), ("GET", "/V/v")}
