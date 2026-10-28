import dataclasses
import json

from selbstbild.parser.basis import parsen
from selbstbild.parser.python import parse_python
from selbstbild.typen import DateiErgebnis, Einstieg, Import

SRC = b'''
import os
from .lib import helfer as h
from paket.mod import Ding

class Laden:
    def kaufen(self, x):
        h(x)
        self.zahlen()
        return os.environ["SHOP_KEY"]

def main():
    Laden().kaufen(os.getenv("SHOP_URL", "http://x"))

if __name__ == "__main__":
    main()
'''


def test_hauptquelle():
    e = parse_python("shop/laden.py", SRC)
    assert {s.qualname for s in e.symbole} == {"Laden", "Laden.kaufen", "main"}
    assert Import(modul="lib", namen=("helfer",), relativ=True, zeile=3) in e.importe
    assert Import(modul="paket.mod", namen=("Ding",), relativ=False, zeile=4) in e.importe
    namen = {a.name for a in e.aufrufe if a.in_symbol == "Laden.kaufen"}
    assert {"helfer", "self.zahlen"} <= namen
    assert {(v.name, v.in_symbol) for v in e.env} == {
        ("SHOP_KEY", "Laden.kaufen"), ("SHOP_URL", "main")}
    assert Einstieg(art="main", symbol=None, zeile=15) in e.einstiege
    assert "http://x" not in json.dumps(dataclasses.asdict(e))
    assert e.fehlerhaft is False


def test_alias_modul():
    e = parse_python("a.py", b"import numpy as np\nnp.zeros(3)\n")
    assert "numpy.zeros" in {a.name for a in e.aufrufe}


def test_import_punkt_und_relativ_leer():
    e = parse_python("a.py", b"import a.b\nfrom . import x\n")
    assert Import(modul="a.b", namen=(), zeile=1, relativ=False) in e.importe
    assert Import(modul="", namen=("x",), zeile=2, relativ=True) in e.importe


def test_environ_import_und_verschachtelt():
    e = parse_python("a.py", b"from os import environ\ndef o():\n    def i():\n        return environ.get('K', 'geheim')\n")
    assert {s.qualname for s in e.symbole} == {"o", "o.i"}
    assert [(v.name, v.in_symbol) for v in e.env] == [("K", "o.i")]
    assert "geheim" not in json.dumps(dataclasses.asdict(e))


def test_testdatei_einstieg():
    e = parse_python("tests/test_a.py", b"def test_x():\n    pass\n")
    assert Einstieg(art="test", symbol="test_x", zeile=1) in e.einstiege


def test_syntaxfehler():
    e = parse_python("a.py", b"def f(:\n  pass\nclass K:\n  def g(self): pass\n")
    assert e.fehlerhaft is True
    assert "K.g" in {s.qualname for s in e.symbole}


def test_digest_stabil():
    a = parse_python("a.py", b"x = 1\n\n\ndef f():\n    return 1\n")
    b = parse_python("a.py", b"x = 1\n\n\n\n\ndef f():\n    return 1\n\n\n")
    c = parse_python("a.py", b"def f():\n    return 2\n")
    da = {s.qualname: s.digest for s in a.symbole}["f"]
    assert da == {s.qualname: s.digest for s in b.symbole}["f"]
    assert da != {s.qualname: s.digest for s in c.symbole}["f"]


def test_tabellenzugriff():
    src = b'def f(cur):\n    cur.execute("SELECT * FROM marketing.kampagnen WHERE id=%s")\n'
    e = parse_python("a.py", src)
    assert ("marketing.kampagnen", 2, "f") in e.tabellen_zugriffe


def test_dispatcher():
    assert parsen("python", "a.py", b"def f(): pass\n").symbole
    assert parsen("cobol", "a.cob", b"x") == DateiErgebnis()



def test_docstrings_keine_tabelle():
    e = parse_python("a.py", b"def f():\n    '''select * from docs'''\n")
    assert e.tabellen_zugriffe == ()
    e = parse_python("a.py", b'"""select * from docs"""\nx = 1\n')
    assert e.tabellen_zugriffe == ()


def test_prosa_keine_tabelle():
    e = parse_python("a.py", b"x = 'from the start'\ny = \"update the docs please\"\n")
    assert e.tabellen_zugriffe == ()


def test_spezial_fehlt_bei_ast_fehler_trotz_sauberem_baum():
    # PEP 695 (`type X = int`) liest tree-sitter, ast erst ab Python 3.12.
    import sys
    q = b'type X = int\n\n@app.get("/x")\ndef f():\n    return 1\n'
    e = parse_python("api/a.py", q)
    assert {s.qualname for s in e.symbole} == {"f"}
    if sys.version_info < (3, 12):
        assert e.spezial_fehlt is True and e.fehlerhaft is True
        assert not e.extra_knoten  # die Route fehlt, aber nicht stumm
    else:
        assert e.spezial_fehlt is False and e.extra_knoten
    sauber = parse_python("api/b.py", b'@app.get("/x")\ndef f():\n    return 1\n')
    assert sauber.spezial_fehlt is False and sauber.fehlerhaft is False
    # Testdateien laufen ohne Spezialextraktoren: kein Ausfall
    assert parse_python("tests/test_a.py", b"type X = int\n").spezial_fehlt is False


def test_table_if_not_exists_und_drop_if_exists():
    q = (b'a = "CREATE TABLE IF NOT EXISTS kunden (id int)"\n'
         b'b = "DROP TABLE IF EXISTS x"\n'
         b'c = "create table if not exists s.t (id int)"\n')
    namen = {t[0] for t in parse_python("a.py", q).tabellen_zugriffe}
    assert namen == {"kunden", "x", "s.t"}


def test_extract_und_substring_keine_tabelle():
    q = (b'a = "SELECT EXTRACT(YEAR FROM created_at) FROM t"\n'
         b'b = "SELECT SUBSTRING(name FROM 2 FOR 3), trim(both \' \' from n) FROM u"\n'
         b'c = "SELECT extract(epoch from (now() - x)) FROM v WHERE id IN (SELECT id FROM w)"\n')
    namen = {t[0] for t in parse_python("a.py", q).tabellen_zugriffe}
    assert namen == {"t", "u", "v", "w"}


def test_update_set_und_join():
    q = b'z = "UPDATE a.b SET x=1"\nw = "select 1 from t join u on 1"\n'
    namen = {t[0] for t in parse_python("a.py", q).tabellen_zugriffe}
    assert namen == {"a.b", "t", "u"}
