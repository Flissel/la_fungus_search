"""doku_loop: SQL-Extraktor (Definitionen und Bezuege)."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import korpus  # noqa: E402
from selbstbild.doku.extraktoren import sql  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402


MIGRATION = (
    "-- kommentar CREATE TABLE marketing.nicht_das;\n"
    "CREATE TABLE IF NOT EXISTS marketing.t (id int);\n"
    "CREATE OR REPLACE FUNCTION marketing.f(p int) RETURNS int AS $$\n"
    "BEGIN RETURN p; END;\n"
    "$$ LANGUAGE plpgsql;\n"
    "create view \"Sales\".\"V_Offen\" as select 1;\n"
)


def _schreibe(pfad: Path, text: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(text, encoding="utf-8")


@pytest.fixture()
def aufbau(tmp_path):
    konfig = dataclasses.replace(vm_laden(), repo=tmp_path)
    return tmp_path, konfig


def test_definitionen_und_bezuege(aufbau):
    wurzel, konfig = aufbau
    _schreibe(wurzel / "db" / "043_x.sql", MIGRATION)
    _schreibe(wurzel / "sp" / "a.py", (
        '"""marketing.versandauftraege_offen im Docstring"""\n'
        "def f():\n"
        '    """sales.docstring_treffer"""\n'
        '    return "select * from marketing.versandauftraege_offen(%s)"\n'
    ))
    defs, doppelte = sql.sql_definitionen(konfig)
    assert set(defs) == {"marketing.t", "marketing.f", "sales.v_offen"}
    assert defs["marketing.t"].art == "table"
    assert defs["marketing.f"].art == "function"
    assert defs["sales.v_offen"].art == "view"
    assert defs["marketing.t"].datei == "db/043_x.sql"
    assert defs["marketing.t"].zeile == 2
    assert doppelte == []
    manifest = SimpleNamespace(corpus_root=str(wurzel / "sp"), documents=[
        SimpleNamespace(file="a.py", start_line=2, end_line=4, document_id="a.f")])
    bez = sql.sql_bezuege(SpaceKonfig("fix", "sp", "decorator"), konfig, manifest)
    assert len(bez) == 1
    assert bez[0].name == "marketing.versandauftraege_offen"
    assert bez[0].datei == "sp/a.py" and bez[0].symbol == "a.f"


def test_digest_aendert_sich_bei_koerperaenderung(aufbau):
    wurzel, konfig = aufbau
    _schreibe(wurzel / "m.sql", MIGRATION)
    d1 = sql.sql_definitionen(konfig)[0]
    _schreibe(wurzel / "m.sql", MIGRATION.replace("RETURN p;", "RETURN p + 1;"))
    d2 = sql.sql_definitionen(konfig)[0]
    assert d1["marketing.f"].digest != d2["marketing.f"].digest
    assert d1["marketing.t"].digest == d2["marketing.t"].digest


def test_semikolon_in_dollar_string_kommentar(aufbau):
    wurzel, konfig = aufbau
    _schreibe(wurzel / "m.sql", (
        "CREATE FUNCTION public.g() RETURNS text AS $tag$ select ';' $tag$ -- ; x\n"
        "/* ; */ LANGUAGE sql;\nCREATE TABLE public.z (a text default 'a;b');\n"))
    defs = sql.sql_definitionen(konfig)[0]
    assert set(defs) == {"public.g", "public.z"}
    a = sql._anweisung("CREATE FUNCTION public.g() AS $tag$ ; $tag$ -- ;\n LANGUAGE sql; rest", 0)
    assert a.endswith("LANGUAGE sql;")


def test_doppelte_hoehere_nummer_gewinnt(aufbau):
    wurzel, konfig = aufbau
    _schreibe(wurzel / "db" / "009_a.sql", "CREATE FUNCTION marketing.f(a int) RETURNS int AS $$ select 1 $$ LANGUAGE sql;")
    _schreibe(wurzel / "db" / "043_a.sql", "CREATE OR REPLACE FUNCTION marketing.f(a int, b int) RETURNS int AS $$ select 2 $$ LANGUAGE sql;")
    _schreibe(wurzel / "db" / "100_a.sql", "CREATE TABLE marketing.f2 (a int);")
    defs, doppelte = sql.sql_definitionen(konfig)
    assert defs["marketing.f"].datei == "db/043_a.sql"
    assert doppelte == ["marketing.f"]


def test_ohne_nummer_null_und_gleichstand_pfad(aufbau):
    wurzel, konfig = aufbau
    _schreibe(wurzel / "a" / "x.sql", "CREATE TABLE public.t (a int);")
    _schreibe(wurzel / "b" / "x.sql", "CREATE TABLE public.t (b int);")
    defs, doppelte = sql.sql_definitionen(konfig)
    assert defs["public.t"].datei == "b/x.sql" and doppelte == ["public.t"]


def test_bezuege_fstring_dreiteilig_schema(aufbau):
    wurzel, konfig = aufbau
    _schreibe(wurzel / "sp" / "b.py", (
        "x = f\"select * from sales.kontakte where id = {i}\"\n"
        "y = 'a.b.c'\n"
        "z = 'other.tabelle'\n"
        "w = 'marketing.a.b'\n"
        "class K:\n    '''public.doc'''\n"
    ))
    _schreibe(wurzel / "sp" / "kaputt.py", "def (:\n")
    manifest = SimpleNamespace(corpus_root=str(wurzel / "sp"), documents=[])
    bez = sql.sql_bezuege(SpaceKonfig("fix", "sp", "decorator"), konfig, manifest)
    assert [b.name for b in bez] == ["sales.kontakte"]
    assert bez[0].symbol is None


@pytest.mark.skipif(not (REPO / "vibemind-os/spaces/sales-claw").is_dir()
                    or not korpus.manifest_verfuegbar(), reason="Submodul fehlt")
def test_echt_sales_claw_alle_marketing_bezuege_definiert():
    konfig = vm_laden()
    space = konfig.spaces["sales-claw"]
    manifest = korpus.manifest_bauen(space, konfig, korpus.commit_von(konfig.repo))
    defs, _ = sql.sql_definitionen(konfig)
    bez = [b for b in sql.sql_bezuege(space, konfig, manifest) if b.name.startswith("marketing.")]
    namen = {b.name for b in bez}
    fehlend = namen - set(defs)
    assert namen and not fehlend, (len(namen), sorted(fehlend))
    assert len(namen) == 12  # 13. Treffer war die URL http://marketing.test, kein SQL-Name
