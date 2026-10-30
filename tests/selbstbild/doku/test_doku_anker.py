"""doku_loop: Belege mit vier Ankerarten und Digest-Pruefung."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import anker  # noqa: E402
from selbstbild.doku.anker import Beleg  # noqa: E402


def _doc(document_id, file, start, end, source="def f(): pass\n"):
    return SimpleNamespace(
        document_id=document_id, file=file, start_line=start, end_line=end,
        symbol=document_id.rsplit(".", 1)[-1], source=source,
    )


def _manifest(docs, corpus_root):
    return SimpleNamespace(corpus_root=str(corpus_root), documents=tuple(docs))


@pytest.fixture()
def repo(tmp_path):
    return tmp_path / "repo"


@pytest.fixture()
def manifest(repo):
    docs = [_doc("spaces.m.api.f", "api.py", 1, 10, "alt")]
    return _manifest(docs, repo / "spaces" / "m")


def test_repo_relativ(manifest, repo):
    assert anker.repo_relativ(manifest, repo, "api.py") == "spaces/m/api.py"


def test_symbol_beleg_ok_geaendert_verschwunden(manifest, repo):
    stand = anker.stand_aus_manifest(manifest, repo)
    schluessel = ("symbol", "spaces/m/api.py", "spaces.m.api.f")
    assert stand[schluessel] == hashlib.sha256(b"alt").hexdigest()
    beleg = Beleg("symbol", "spaces/m/api.py", "spaces.m.api.f", stand[schluessel], "c1", 3)
    assert anker.beleg_pruefen(beleg, stand, repo) == "beleg_ok"
    neu = dict(stand)
    neu[schluessel] = "anders"
    assert anker.beleg_pruefen(beleg, neu, repo) == "beleg_geaendert"
    assert anker.beleg_pruefen(beleg, {}, repo) == "beleg_verschwunden"


def test_beleg_aus_zeile(manifest, repo):
    b = anker.beleg_aus_zeile("spaces/m/api.py", 5, manifest, "c1", repo=repo)
    assert b is not None
    assert (b.art, b.file, b.anker, b.commit, b.zeile) == (
        "symbol", "spaces/m/api.py", "spaces.m.api.f", "c1", 5)
    assert b.digest == hashlib.sha256(b"alt").hexdigest()
    assert anker.beleg_aus_zeile("spaces/m/api.py", 50, manifest, "c1", repo=repo) is None


def test_beleg_aus_zeile_nutzt_document_id_nicht_namen(repo):
    m = _manifest(
        [_doc("pk.a.f", "a.py", 1, 5), _doc("pk.b.f", "b.py", 1, 5)], repo
    )
    b = anker.beleg_aus_zeile("a.py", 2, m, "c", repo=repo)
    assert b is not None and b.anker.endswith(".a.f")


def test_beleg_aus_zeile_mehrdeutig_liefert_none(repo):
    m = _manifest(
        [_doc("pk.a.f", "a.py", 1, 5, "v1"), _doc("pk.a.f", "a.py", 7, 12, "v2")], repo
    )
    assert anker.beleg_aus_zeile("a.py", 2, m, "c", repo=repo) is None
    assert ("symbol", "a.py", "pk.a.f") not in anker.stand_aus_manifest(m, repo)


def test_stand_aus_sql():
    d = SimpleNamespace(name="public.t", datei="x/001.sql", digest="abc")
    assert anker.stand_aus_sql({"public.t": d}) == {("sql", "x/001.sql", "public.t"): "abc"}


def test_yaml_knoten_digest_pfad_und_aenderung(tmp_path):
    f = tmp_path / "c.yml"
    f.write_text("a:\n  b:\n    - x: 1\n    - y: 2\n    - c: {k: v}\n", encoding="utf-8")
    d1 = anker.yaml_knoten_digest(f, "a.b[2].c")
    assert d1 is not None
    f.write_text("a:\n  b:\n    - x: 1\n    - y: 2\n    - c: {k: w}\n", encoding="utf-8")
    assert anker.yaml_knoten_digest(f, "a.b[2].c") != d1
    assert anker.yaml_knoten_digest(f, "a.b[9].c") is None
    assert anker.yaml_knoten_digest(f, "a.zz") is None
    assert anker.yaml_knoten_digest(tmp_path / "fehlt.yml", "a") is None
    f.write_text("a: [unclosed\n", encoding="utf-8")
    assert anker.yaml_knoten_digest(f, "a") is None


def test_toml_knoten_digest(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text('[tool.x]\nitems = [1, {n = "a"}]\n', encoding="utf-8")
    d1 = anker.yaml_knoten_digest(f, "tool.x.items[1].n")
    assert d1 is not None
    f.write_text('[tool.x]\nitems = [1, {n = "b"}]\n', encoding="utf-8")
    assert anker.yaml_knoten_digest(f, "tool.x.items[1].n") != d1
    assert anker.yaml_knoten_digest(f, "tool.nope") is None


def test_datei_beleg_erkennt_jede_aenderung(tmp_path):
    (tmp_path / "d.txt").write_text("eins", encoding="utf-8")
    b = anker.beleg_fuer_datei("d.txt", 1, tmp_path, "c")
    assert b.art == "datei" and b.zeile == 1
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_ok"
    (tmp_path / "d.txt").write_text("eins ", encoding="utf-8")
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_geaendert"
    (tmp_path / "d.txt").unlink()
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_verschwunden"


def test_yaml_key_beleg_pruefen(tmp_path):
    f = tmp_path / "c.yml"
    f.write_text("a:\n  b: 1\n", encoding="utf-8")
    digest = anker.yaml_knoten_digest(f, "a.b")
    b = Beleg("yaml_key", "c.yml", "a.b", digest, "c")
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_ok"
    f.write_text("a:\n  b: 2\n", encoding="utf-8")
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_geaendert"
    f.write_text("a:\n  c: 2\n", encoding="utf-8")
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_verschwunden"


def test_beleg_roundtrip_dict():
    b = Beleg("sql", "x.sql", "public.t", "d", "c", 4)
    assert Beleg.from_dict(b.to_dict()) == b
    assert set(b.to_dict()) == {"art", "file", "anker", "digest", "commit", "zeile"}
    assert Beleg.from_dict({"art": "datei", "file": "f", "anker": "", "digest": "d", "commit": "c"}).zeile is None


# ---------- Ruling 24: Identitaet, Quotes; Ruling 26: Symbol direkt aus der Datei ----------

@pytest.mark.parametrize("pfad, tokens", [
    ("a.b[2].c", ["a", "b", 2, "c"]),
    ("[capability=schedule_create].targets", [("capability", "schedule_create"), "targets"]),
    ('spaces.x.events["agentfarm.status"].execution', ["spaces", "x", "events", "agentfarm.status", "execution"]),
    ('l[name="a]b"]', ["l", ("name", "a]b")]),
    ("l[capability=rose.recommend]", ["l", ("capability", "rose.recommend")]),
])
def test_tokens_und_pfad_text_rundweg(pfad, tokens):
    assert anker._tokens(pfad) == tokens
    assert anker.pfad_text(tokens) == pfad
    assert anker._tokens(anker.pfad_text(tokens)) == tokens


@pytest.mark.parametrize("pfad", ["", "a..b", "a[", "[x", "a[0]b", ".a", 'a["x"', "a[=x]"])
def test_tokens_ungueltig(pfad):
    assert anker._tokens(pfad) is None


def test_yaml_identitaet_stabil_bei_einfuegen_und_mehrdeutig(tmp_path):
    f = tmp_path / "caps.yml"
    f.write_text("caps:\n  - capability: a\n    t: [1]\n  - capability: b\n    t: [2]\n", encoding="utf-8")
    d = anker.yaml_knoten_digest(f, "caps[capability=b]")
    assert d is not None and d != anker.yaml_knoten_digest(f, "caps[capability=a]")
    f.write_text("caps:\n  - capability: neu\n    t: [0]\n  - capability: a\n    t: [1]\n"
                 "  - capability: b\n    t: [2]\n", encoding="utf-8")
    assert anker.yaml_knoten_digest(f, "caps[capability=b]") == d
    assert anker.yaml_knoten_digest(f, "caps[1]") != d  # positional zeigt jetzt auf einen anderen Eintrag
    f.write_text("caps:\n  - capability: b\n  - capability: b\n", encoding="utf-8")
    assert anker.yaml_knoten_digest(f, "caps[capability=b]") is None  # mehrdeutig -> kein Anker


def test_yaml_gequoteter_schluessel(tmp_path):
    f = tmp_path / "r.yml"
    f.write_text('events:\n  "agentfarm.status":\n    tool: x\n  agentfarm: 1\n', encoding="utf-8")
    d = anker.yaml_knoten_digest(f, 'events["agentfarm.status"]')
    assert d is not None and d != anker.yaml_knoten_digest(f, "events.agentfarm")


def test_toml_identitaet_tabellen_array(tmp_path):
    f = tmp_path / "o.toml"
    f.write_text('[[srv]]\nname = "a"\nport = 1\n[[srv]]\nname = "b"\nport = 2\n', encoding="utf-8")
    d = anker.yaml_knoten_digest(f, "srv[name=b]")
    assert d is not None
    f.write_text('[[srv]]\nname = "z"\nport = 0\n[[srv]]\nname = "a"\nport = 1\n[[srv]]\nname = "b"\nport = 2\n',
                 encoding="utf-8")
    assert anker.yaml_knoten_digest(f, "srv[name=b]") == d
    assert anker.yaml_knoten_digest(f, "srv[name=fehlt]") is None


def test_symbol_digest_aus_datei_und_beleg_pruefen(tmp_path):
    (tmp_path / "scripts").mkdir()
    datei = tmp_path / "scripts" / "w.py"
    datei.write_text("class K:\n    def lauf(self):\n        return 1\n\n\ndef g():\n    return 2\n", encoding="utf-8")
    assert anker.modulpfad("scripts/w.py") == "scripts.w"
    digest = anker.symbol_digest_aus_datei(tmp_path, "scripts/w.py", "scripts.w.K.lauf")
    assert digest == hashlib.sha256("    def lauf(self):\n        return 1".encode()).hexdigest()
    assert anker.symbol_digest_aus_datei(tmp_path, "scripts/w.py", "andere.w.g") is None
    assert anker.symbol_digest_aus_datei(tmp_path, "scripts/w.py", "scripts.w.fehlt") is None
    b = Beleg("symbol", "scripts/w.py", "scripts.w.g", anker.symbol_digest_aus_datei(tmp_path, "scripts/w.py",
                                                                                      "scripts.w.g"), "c", 6)
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_ok"  # nicht im Stand -> direkt aus der Datei
    datei.write_text("def g():\n    return 3\n", encoding="utf-8")
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_geaendert"
    datei.write_text("def h():\n    return 3\n", encoding="utf-8")
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_verschwunden"


# ---------- Ruling 29: Stand-Fallback nur ausserhalb der Spaces ----------

def _symboldatei(tmp_path, rel):
    datei = tmp_path / rel
    datei.parent.mkdir(parents=True, exist_ok=True)
    datei.write_text("def g():\n    return 2\n", encoding="utf-8")
    anker_name = f"{anker.modulpfad(rel)}.g"
    return Beleg("symbol", rel, anker_name, anker.symbol_digest_aus_datei(tmp_path, rel, anker_name), "c", 1)


def test_beleg_pruefen_space_symbol_fehlt_ist_verschwunden(tmp_path):
    b = _symboldatei(tmp_path, "sp/m/w.py")
    assert anker.beleg_pruefen(b, {}, tmp_path, space_wurzeln=("sp/m",)) == "beleg_verschwunden"
    assert anker.beleg_pruefen(b, {("symbol", b.file, b.anker): b.digest}, tmp_path,
                               space_wurzeln=("sp/m",)) == "beleg_ok"
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_ok"  # ohne Parameter wie bisher


def test_beleg_pruefen_fremdes_symbol_wird_nachgerechnet(tmp_path):
    b = _symboldatei(tmp_path, "scripts/w.py")
    assert anker.beleg_pruefen(b, {}, tmp_path, space_wurzeln=("sp/m", "sp/mehr/")) == "beleg_ok"
    (tmp_path / "scripts/w.py").write_text("def g():\n    return 3\n", encoding="utf-8")
    assert anker.beleg_pruefen(b, {}, tmp_path, space_wurzeln=("sp/m",)) == "beleg_geaendert"
    b2 = _symboldatei(tmp_path, "sp/mx/w.py")  # Praefix nur ganzsegmentig
    assert anker.beleg_pruefen(b2, {}, tmp_path, space_wurzeln=("sp/m",)) == "beleg_ok"


# ---------- Final-Fix-Welle (Ruling 57) ----------

def test_digest_mit_gemischten_schluesseltypen_wirft_nicht(tmp_path):
    """C2: YAML mit `{1: eins, a: zwei}` legte `sort_keys` lahm (TypeError) und damit jeden Tick."""
    datei = tmp_path / "x.yaml"
    datei.write_text("wurzel:\n  1: eins\n  a: zwei\n  null: drei\n  innen:\n  - {2: x, b: y}\n", encoding="utf-8")
    d1 = anker.yaml_knoten_digest(datei, "wurzel")
    assert d1 is not None and d1 == anker.yaml_knoten_digest(datei, "wurzel")  # deterministisch
    gefunden, wert = anker.yaml_knoten_wert(datei, "wurzel")
    assert gefunden and anker.knoten_text(wert) == anker.knoten_text(wert)
    assert '"eins"' in anker.knoten_text(wert) and '"zwei"' in anker.knoten_text(wert)
    # Kollision "1" (Text) gegen 1 (Zahl): beide bleiben unterscheidbar
    assert anker._digest_json({1: "a", "1": "b"}) != anker._digest_json({1: "b", "1": "a"})


def test_digest_unveraendert_fuer_einheitliche_schluessel():
    """Bestehende Digests (nur Text- bzw. nur Zahl-Schluessel) duerfen sich nicht aendern (sonst Signalflut)."""
    import json as _json
    for wert in ({"b": 1, "a": [1, {"y": 2, "x": 3}]}, {10: "x", 2: "y"}, [1, "a"], None, "t"):
        alt = hashlib.sha256(_json.dumps(wert, sort_keys=True, ensure_ascii=False, default=str)
                             .encode("utf-8")).hexdigest()
        assert anker._digest_json(wert) == alt
        assert anker.knoten_text(wert) == _json.dumps(wert, sort_keys=True, ensure_ascii=False, default=str)
