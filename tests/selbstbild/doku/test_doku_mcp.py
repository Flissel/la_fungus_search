"""doku_loop: MCP-Werkzeug-Extraktor."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import korpus  # noqa: E402
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku.extraktoren import mcp_werkzeuge as mw  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402



def _lauf(tmp_path, dateien, muster, docs=()):
    (tmp_path / "sp").mkdir(exist_ok=True)
    for n, q in dateien.items():
        (tmp_path / "sp" / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "sp" / n).write_text(q, encoding="utf-8")
    konfig = dataclasses.replace(vm_laden(), repo=tmp_path)
    manifest = SimpleNamespace(corpus_root=str(tmp_path / "sp"), documents=list(docs))
    return mw.mcp_werkzeuge(SpaceKonfig("fix", "sp", muster), konfig, manifest)


def test_decorator_muster_findet_mcp_tool_namen(tmp_path):
    q = (
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
    doc = SimpleNamespace(file="a.py", start_line=1, end_line=2, document_id="a.think")
    res = _lauf(tmp_path, {"a.py": q}, "decorator", [doc])
    assert {w.name for w in res} == {"think", "suche", "nackt", "t1", "t2"}
    t = next(w for w in res if w.name == "think")
    assert t.datei == "sp/a.py" and t.zeile == 2 and t.symbol == "a.think"


def test_testdateien_liefern_nichts(tmp_path):
    q = "@mcp.tool()\ndef think(): ...\n"
    assert _lauf(tmp_path, {"tests/test_a.py": q, "conftest.py": q}, "decorator") == []


def test_werkzeuge_muster_name_und_attribut(tmp_path):
    server = (
        "from . import werkzeuge\n"
        "def lokal(): ...\n"
        "WERKZEUGE = (\n    werkzeuge.versand, lokal,\n    werkzeuge.weg,\n)\n"
    )
    wz = "def versand(): ...\n\ndef anderes(): ...\n"
    res = _lauf(tmp_path, {"server.py": server, "werkzeuge.py": wz}, "werkzeuge")
    by = {w.name: w for w in res}
    assert set(by) == {"versand", "lokal", "weg"}
    assert by["versand"].datei == "sp/werkzeuge.py" and by["versand"].zeile == 1
    assert by["lokal"].datei == "sp/server.py" and by["lokal"].zeile == 2
    assert by["weg"].datei == "sp/server.py"  # nicht aufloesbar: Eintrag im Literal


def _echt(wurzel, muster, name):
    if not (REPO / wurzel).is_dir() or not korpus.manifest_verfuegbar():
        pytest.skip("Submodul nicht ausgecheckt")
    konfig = vm_laden()
    space = SpaceKonfig(name, wurzel, muster)
    return mw.mcp_werkzeuge(space, konfig, korpus.manifest_bauen(space, konfig, "x"))


def test_echt_marketing_findet_versand_beauftragen():
    res = _echt("vibemind-os/spaces/marketing", "werkzeuge", "marketing")
    w = [x for x in res if x.name == "versand_beauftragen"]
    assert len(w) == 1 and w[0].datei.endswith("claw/werkzeuge.py") and w[0].symbol
    assert len(res) >= 40


def test_echt_sales_findet_entwurf_freigeben():
    res = _echt("vibemind-os/spaces/sales-claw", "werkzeuge", "sales-claw")
    namen = {x.name for x in res}
    assert {"entwurf_freigeben", "versandauftrag_uebernehmen"} <= namen
    w = next(x for x in res if x.name == "entwurf_freigeben")
    assert w.datei.endswith("sales-mcp/server.py") and w.symbol
    assert len(res) >= 80


def test_echt_brain_findet_think():
    res = _echt("vibemind-os/brain", "decorator", "brain")
    assert "think" in {x.name for x in res}
    assert len(res) >= 60
