"""doku_loop: HTTP-Client-Extraktor."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku.extraktoren import http_clients  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402


PULT = (
    "import os\n"
    "import urllib.request\n"
    'BASIS = os.environ.get("MARKETING_PULT_URL", "http://127.0.0.1:5510")\n'
    "def holen(pfad):\n"
    '    req = urllib.request.Request(f"{BASIS}/api/pult/{pfad}")\n'
    "    return urllib.request.urlopen(req)\n"
)


def _lauf(tmp_path, quelle, name="a.py", docs=()):
    (tmp_path / "sp").mkdir(exist_ok=True)
    (tmp_path / "sp" / name).write_text(quelle, encoding="utf-8")
    konfig = dataclasses.replace(vm_laden(), repo=tmp_path)
    manifest = SimpleNamespace(corpus_root=str(tmp_path / "sp"), documents=list(docs))
    return http_clients.http_aufrufe(SpaceKonfig("fix", "sp", "decorator"), konfig, manifest)


def test_pult_fixture(tmp_path):
    doc = SimpleNamespace(file="a.py", start_line=4, end_line=6, document_id="a.holen")
    res = _lauf(tmp_path, PULT, docs=[doc])
    assert len(res) == 1
    a = res[0]
    assert a.env_namen == ("MARKETING_PULT_URL",)
    assert a.pfad_literal == "/api/pult/{}"
    assert a.methode == "GET"
    assert a.datei == "sp/a.py" and a.zeile == 5 and a.symbol == "a.holen"


def test_request_und_urlopen_zaehlen_einmal(tmp_path):
    assert len(_lauf(tmp_path, PULT)) == 1
    q = (
        "import urllib.request\n"
        'urllib.request.urlopen(urllib.request.Request("http://h:1/x", data=b"1"))\n'
    )
    res = _lauf(tmp_path, q)
    assert len(res) == 1 and res[0].methode == "POST" and res[0].pfad_literal == "/x"


def test_requests_httpx_und_templates(tmp_path):
    q = (
        "import os, requests, httpx\n"
        "BASE = 'http://h:1'\n"
        "def a(x):\n"
        "    requests.post(os.environ['A_URL'] + '/api/x?y=1')\n"
        "    requests.get('%s/api/%s' % (os.getenv('B_URL'), x))\n"
        "    requests.put('{}/api/{}'.format(os.getenv('C_URL'), x))\n"
        "    requests.delete(BASE + '/api/d')\n"
        "    requests.get(x)\n"
        "    requests.get(f'{os.getenv(\"K\")}')\n"
        "async def b():\n"
        "    async with httpx.AsyncClient() as c:\n"
        "        await c.get(f'{_basis()}/api/c')\n"
        "def _basis():\n"
        "    return os.environ.get('D_URL', 'x').rstrip('/')\n"
    )
    res = {a.zeile: a for a in _lauf(tmp_path, q)}
    assert (res[4].methode, res[4].env_namen, res[4].pfad_literal) == ("POST", ("A_URL",), "/api/x")
    assert (res[5].env_namen, res[5].pfad_literal) == (("B_URL",), "/api/{}")
    assert (res[6].env_namen, res[6].pfad_literal) == (("C_URL",), "/api/{}")
    assert (res[7].env_namen, res[7].pfad_literal) == ((), "/api/d")
    assert (res[8].env_namen, res[8].pfad_literal) == ((), None)
    assert res[9].pfad_literal is None and res[9].env_namen == ("K",)
    assert (res[12].methode, res[12].env_namen, res[12].pfad_literal) == ("GET", ("D_URL",), "/api/c")


def test_testdateien_liefern_keine_aufrufe(tmp_path):
    q = "import requests\nrequests.get('http://h:1/x')\n"
    for name in ("tests/x.py", "test_x.py", "foo_test.py", "conftest.py"):
        (tmp_path / "sp" / name).parent.mkdir(parents=True, exist_ok=True)
        assert _lauf(tmp_path, q, name=name) == [], name
    assert len(_lauf(tmp_path, q, name="modul.py")) == 1


def test_request_mit_keyword_url_und_methode(tmp_path):
    q = (
        "import requests\n"
        "requests.request(method='PUT', url='http://h:1/a/b')\n"
        "requests.request('GET', url='http://h:1/c')\n"
    )
    res = _lauf(tmp_path, q)
    assert [(a.methode, a.pfad_literal) for a in res] == [("PUT", "/a/b"), ("GET", "/c")]


def test_unparsbar_wird_uebersprungen(tmp_path):
    assert _lauf(tmp_path, "def (:\n") == []


@pytest.mark.skipif(
    not (REPO / "vibemind-os" / "spaces" / "sales-claw").is_dir(), reason="Submodul fehlt"
)
def test_echt_sales_claw_mindestens_22_aufrufe():
    from selbstbild.doku import korpus

    konfig = vm_laden()
    space = konfig.spaces["sales-claw"]
    m = korpus.manifest_bauen(space, konfig, "x")
    assert len(http_clients.http_aufrufe(space, konfig, m)) >= 22  # gemessen 24 ohne Tests
