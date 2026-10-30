"""doku_loop: Seed des Graphen aus den Hop-Tabellen von KETTEN.md."""

import dataclasses
import hashlib
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden, standard_konfig  # noqa: E402,F401

pytestmark = pytest.mark.usefixtures("standard_konfig")
from selbstbild.doku import anker, graph  # noqa: E402
from selbstbild.doku import ketten_seed as ks  # noqa: E402
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import korpus  # noqa: E402
from selbstbild.doku.extraktoren.sql import SqlDefinition  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402

KETTEN = REPO / "docs" / "spaces" / "KETTEN.md"
SCHWELLE_VERANKERT = 91  # gemessen 93 - 2 (Task 9b, Ruling 30)
KOPF = "| # | Von → Nach | Mechanismus | Code-Stelle | Belegt durch | Zustand |"


def _schreibe(pfad: Path, text: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(text, encoding="utf-8")


def _doc(document_id, file, start, end, source="def f():\n    pass\n"):
    return SimpleNamespace(document_id=document_id, file=file, start_line=start, end_line=end,
                           symbol=document_id.rsplit(".", 1)[-1], source=source)


# ---------- Tabellen lesen ----------

@pytest.mark.skipif(not KETTEN.is_file(), reason="KETTEN.md fehlt")
def test_hop_tabellen_lesen_echte_datei_108_zeilen_18_ketten():
    text = KETTEN.read_text(encoding="utf-8")
    if "<!-- graph:begin" in text:
        pytest.skip("KETTEN.md ist bereits gerendert")
    zeilen = ks.hop_tabellen_lesen(text)
    assert len(zeilen) == 108
    assert len({z.kette for z in zeilen}) == 18
    erste = zeilen[0]
    assert (erste.kette, erste.nr, erste.zustand) == ("K1", "1", "läuft")
    assert erste.code_stellen == ("vibemind-os/spaces/marketing/claw/werkzeuge.py:294-297",)
    assert erste.kette_titel == "Marketing → sales-claw (Versandauftrag)"
    assert "3a" in {z.nr for z in zeilen if z.kette == "K1"}
    assert all(z.zeile > 0 for z in zeilen)


def test_hop_tabellen_lesen_crlf_backtick_pipe_und_kaputte_zeile():
    text = "\r\n".join([
        "\ufeff# Titel", "", "### K2 · A → B", "", KOPF, "|---|---|---|---|---|---|",
        "| 1 | X → Y | Python `a | b` | `p/q.py:3` | 008 | läuft |",
        "| 2 | nur | drei |",
        "| 3 | Z → W | SQL | — | 015 | aus |",
        "", "Prosa | mit Pipe", "## Teil B", KOPF, "|---|---|---|---|---|---|",
        "| 9 | ohne Kette | x | y | z | läuft |", "",
    ])
    zeilen, pp = ks.hop_tabellen_lesen_mit_pruefpunkten(text, "docs/K.md")
    assert [z.nr for z in zeilen] == ["1", "3"]
    assert zeilen[0].mechanismus == "Python `a | b`"
    assert zeilen[0].zeile == 7 and zeilen[0].kette_titel == "A → B"
    assert zeilen[1].code_stellen == ("—",)
    assert [(p.art, p.zeile) for p in pp] == [("zeile_unlesbar", 8)]
    assert ks.hop_tabellen_lesen(text) == zeilen


# ---------- Code-Stellen ----------

def test_code_stellen_formen(tmp_path):
    for p in ("vibemind-os/spaces/sales-claw/sales-mcp/dispatch.py",
              "vibemind-os/spaces/sales-claw/sales-mcp/mail_dispatch.py",
              "vibemind-os/brain/core/einzig_hier.py",
              "doppelt/a/zwei.py", "doppelt/b/zwei.py"):
        _schreibe(tmp_path / p, "x = 1\n")
    zelle = ("`vibemind-os/spaces/sales-claw/sales-mcp/dispatch.py:166`, `:282-290` · "
             "E-Mail `mail_dispatch.py:243` ← `043…sql:297` · `einzig_hier.py:12` · `zwei.py:4` "
             "· `.../x.py:9`")
    st = ks.code_stellen_aufloesen(zelle, None, tmp_path)
    voll = "vibemind-os/spaces/sales-claw/sales-mcp/dispatch.py"
    assert [(s.pfad, s.start, s.ende, s.form) for s in st] == [
        (voll, 166, None, "voll"),
        (voll, 282, 290, "kurzform"),
        ("vibemind-os/spaces/sales-claw/sales-mcp/mail_dispatch.py", 243, None, "verzeichnis"),
        (None, 297, None, "auslassung"),
        ("vibemind-os/brain/core/einzig_hier.py", 12, None, "anker"),
        (None, 4, None, "anker"),  # nicht eindeutig -> kein Anker statt eines falschen
        (None, 9, None, "auslassung"),
    ]


def test_code_stellen_kurzform_voriger_pfad_prefix_und_ohne_bezug(tmp_path):
    _schreibe(tmp_path / "vibemind-os/brain/the_brain/Dockerfile", "FROM x\n")
    _schreibe(tmp_path / "vibemind-os/brain/the_brain/data/capabilities.yaml", "a: 1\n")
    st = ks.code_stellen_aufloesen("`:21`, `:42`", "a/b.py", tmp_path)
    assert [(s.pfad, s.start, s.form) for s in st] == [("a/b.py", 21, "kurzform"), ("a/b.py", 42, "kurzform")]
    st = ks.code_stellen_aufloesen("`:7`", None, tmp_path)
    assert [(s.pfad, s.form) for s in st] == [(None, "kurzform")]
    st = ks.code_stellen_aufloesen(
        "`brain/the_brain/Dockerfile:75` · `vibemind-os/brain/the_brain/data/capabilities.yaml` "
        "· `~/.rowboat/x.md:3` · Commit `ec98a0c6`", None, tmp_path)
    assert [(s.pfad, s.start, s.form) for s in st] == [
        ("vibemind-os/brain/the_brain/Dockerfile", 75, "voll"),
        ("vibemind-os/brain/the_brain/data/capabilities.yaml", None, "voll"),
    ]


def test_code_stellen_anker_ueber_bekannte_pfade(tmp_path):
    for p in ("a/capabilities.yaml", "b/capabilities.yaml"):
        _schreibe(tmp_path / p, "a: 1\n")
    st = ks.code_stellen_aufloesen("`capabilities.yaml:1`", None, tmp_path,
                                   bekannte={"capabilities.yaml": "b/capabilities.yaml"})
    assert [(s.pfad, s.form) for s in st] == [("b/capabilities.yaml", "anker")]


# ---------- Knotentyp, Mechanismus, Zustand ----------

def test_knotentyp_dienst_tabelle_funktion_mensch():
    assert ks.knotentyp_erkennen("Marketing-Sidecar `:8130`") == ("dienst", "Marketing-Sidecar")
    assert ks.knotentyp_erkennen("`127.0.0.1:5510`") == ("dienst", ":5510")  # Host statt Name -> Port
    assert ks.knotentyp_erkennen("`localhost:5510`") == ("dienst", ":5510")
    assert ks.knotentyp_erkennen("`:8446`") == ("dienst", ":8446")
    assert ks.knotentyp_erkennen("Marketing → `Marketing.Inhalte`") == ("tabelle", "marketing.inhalte")
    assert ks.knotentyp_erkennen("`public.ideas` / `canvas_nodes`") == ("tabelle", "public.ideas")
    assert ks.knotentyp_erkennen("SQL `marketing.versandauftraege_offen()`") == (
        "sql_funktion", "marketing.versandauftraege_offen")
    assert ks.knotentyp_erkennen("Supabase `marketing.*`") == ("komponente", "Supabase marketing.*")
    assert ks.knotentyp_erkennen("`video.status`") == ("komponente", "video.status")
    assert ks.knotentyp_erkennen("sales-claw `wissensbasis_fragen`", mcp_namen={"wissensbasis_fragen"}) == (
        "mcp_werkzeug", "wissensbasis_fragen")
    assert ks.knotentyp_erkennen("Mensch") == ("mensch", "Mensch")
    assert ks.knotentyp_erkennen("Betreiber") == ("mensch", "Betreiber")
    assert ks.knotentyp_erkennen("**Laura**   (VM)") == ("komponente", "Laura (VM)")


def test_mechanismus_typ():
    f = ks.mechanismus_typ
    assert f("MCP `versand_beauftragen`") == "MCP"
    assert f("MCP-Werkzeug, Adresse") == "MCP"
    assert f("HTTP `POST /api/x`") == "HTTP"
    assert f("PostgREST `GET ideas`") == "HTTP"
    assert f("SQL `marketing.x`") == "SQL"
    assert f("Python `VersandGesperrt`") == "Python"
    assert f("Datei `~/.rowboat/x.md`") == "Datei"
    assert f("`docker exec psql -U x`") == "Docker/Env"
    assert f("SSE `GET :5000/api/events/stream`") == "HTTP"  # Ruling 32: HTTP vor SSE/IPC
    assert f("SSE-Stream `events`") == "SSE/IPC"
    assert f("IPC `wizard_init_from_bubble`") == "SSE/IPC"
    assert f("`build_executor(target)` nach Präfix") == "unbestimmt"


def test_mechanismus_typ_ganze_zelle_mit_vorrang():
    f = ks.mechanismus_typ
    # ganze Zelle, nicht nur das erste Wort
    assert f("`OpenFangExecutor` ruft `mcp__openfang__x`") == "MCP"
    assert f("Capability `truth:supabase_row` liest Supabase") == "SQL"
    assert f("`atomarer` Wechsel per `UPDATE`") == "SQL"
    assert f("Rueckmeldung ueber `marketing.versand_erledigt`") == "SQL"
    assert f("Rueckmeldung ueber `Marketing.x`") == "unbestimmt"  # Schema-Regel mit Gross-/Kleinschreibung
    assert f("Webhook an den Sidecar") == "HTTP"
    assert f("Aufruf `requests.post(... /api/x)`") == "HTTP"
    assert f("Redis-Kanal `events`") == "SSE/IPC"
    assert f("Swarm-Service, Env `X_URL`") == "Docker/Env"
    assert f("schreibt `stand.json`") == "Datei"
    assert f("Aufruf `build_executor(target)`") == "Python"
    assert f("`import x`") == "Python"
    # Vorrang MCP > SQL > HTTP > SSE/IPC > Docker/Env > Datei > Python, gross/klein egal
    assert f("python-Aufruf, dann sql `insert`, dann http") == "SQL"
    assert f("Werkzeug-Aufruf ueber REST") == "MCP"
    assert f("Container-Env ueber HTTP") == "HTTP"
    # Wortanfang: "Klasse"/"Adresse" sind kein SSE, "Katalog" keine Logdatei, "psql" kein SQL
    assert f("Klasse mit Adresse") == "unbestimmt"
    assert f("Katalog") == "unbestimmt"
    # Rueckfall auf die alten Schluesselwoerter am Zellenanfang
    assert f("after-trigger auf ideas") == "SQL"
    assert f("CDP an das Fenster") == "SSE/IPC"


@pytest.mark.parametrize("text, wert, aus, bekannt", [
    ("läuft", "laeuft", False, True),
    ("läuft (Sperre aktiv)", "laeuft", False, True),
    ("**bricht hier**", "bricht", False, True),
    ("läuft nativ · **bricht im Swarm-Image**", "bricht", False, True),
    ("**aus** seit 17.09.", "keine_probe", True, True),
    ("lauffähig, aus", "keine_probe", True, True),          # Ruling 25: aus als letztes Wort
    ("Worker läuft, Ziel aus", "keine_probe", True, True),
    ("nie bewiesen (WORKBOARD:141)", "keine_probe", False, True),
    ("nie E2E bewiesen; x", "keine_probe", False, True),
    ("**nie gelaufen** (Verzeichnis fehlt)", "keine_probe", False, True),
    ("verdrahtet", "keine_probe", False, True),             # bekannt, ohne Pruefpunkt
    ("verdrahtet (gemockt getestet)", "keine_probe", False, True),
    ("gebaut", "keine_probe", False, True),
    ("Code vorhanden", "keine_probe", False, True),
    ("registriert", "keine_probe", False, True),
    ("fail-soft", "keine_probe", False, True),
    ("bewiesen (Teil VI)", "keine_probe", False, False),    # bleibt Pruefpunkt
    ("`tools/list` 7/7 am 18.08. verifiziert", "keine_probe", False, False),
    ("einmal gelaufen", "keine_probe", False, False),
    ("**kein Laufzeitpfad**", "keine_probe", False, False),
    ("**toter Import**, still übersprungen", "keine_probe", False, False),
])
def test_zustand_aus_spalte(text, wert, aus, bekannt):
    z, ok = ks.zustand_aus_spalte(text)
    assert z["gemessen"]["wert"] == wert and z["aus"] is aus and ok is bekannt
    assert z["verdrahtet"] == "unbestimmt" and z["konfiguriert"] == "unbestimmt"  # nie aus dem Dokument


def test_kante_id_nr_als_text():
    assert graph.kante_id("hop", "a", "b", kette="K1", nr="3a") == "hop:a->b#K1/3a"
    assert graph.kante_id("hop", "a", "b", kette="K1", nr=2) == "hop:a->b#K1/2"


# ---------- Schluesselpfad (Ruling 24) ----------

YAML_CAPS = (
    "caps:\n"                       # 1
    "  - capability: a\n"           # 2
    "    targets:\n"                # 3
    "      - t1\n"                  # 4
    "    note: x\n"                 # 5
    "  # Kommentar zu b\n"          # 6
    "  - capability: b\n"           # 7
    "    events:\n"                 # 8
    "      agentfarm.status: 1\n"   # 9
    "      x.y: {k: v}\n"           # 10
    "liste:\n"                      # 11
    "  - {k: 1}\n"                  # 12
    "services:\n"                   # 13
    "  web:\n"                      # 14
    "    image: a\n"                # 15
    "\n"                            # 16
    "  db:\n"                       # 17
    "    image: b\n"                # 18
)


def test_schluessel_anker_yaml_identitaet_bereich_kommentar_quote(tmp_path):
    d = tmp_path / "c.yaml"
    _schreibe(d, YAML_CAPS)
    a = ks.schluessel_anker
    assert a(d, 2) == ("caps[capability=a]", None)               # Identitaetszeile -> Eintrag
    assert a(d, 3) == ("caps[capability=a].targets", None)       # Schluesselzeile einer Liste
    assert a(d, 4) == ("caps[capability=a].targets", None)       # Skalar -> umschliessende Liste
    assert a(d, 5) == ("caps[capability=a]", None)               # Skalarschluessel -> Eintrag
    assert a(d, 2, 5) == ("caps[capability=a]", None)            # ganzer Bereich
    assert a(d, 6) == ("caps[capability=b]", None)               # Kommentar -> naechster Schluessel darunter
    assert a(d, 9) == ("caps[capability=b].events", None)
    assert a(d, 10) == ('caps[capability=b].events["x.y"]', None)  # Punkt im Schluessel -> gequotet
    assert a(d, 12) == ("liste[0]", "anker_positional")
    assert a(d, 15) == ("services.web", None)
    assert a(d, 16) == ("services.db", None)                     # Leerzeile -> naechster Schluessel
    assert a(d, 2, 8) == (None, "anker_zu_grob")                 # nur caps umfasst den Bereich
    assert a(d, 13) == (None, "anker_zu_grob")                   # oberste Sammlung
    assert a(d, 15, 18) == (None, "anker_zu_grob")
    assert a(d, 99) == (None, None)
    assert ks.schluesselpfad_an(d, 4) == "caps[capability=a].targets"
    for pfad in ("caps[capability=a].targets", 'caps[capability=b].events["x.y"]', "liste[0]"):
        assert anker.yaml_knoten_digest(d, pfad) is not None


def test_schluessel_anker_yaml_wurzelliste_und_stabil_bei_einfuegen(tmp_path):
    d = tmp_path / "caps.yaml"
    _schreibe(d, "- capability: x\n  targets: [1]\n- capability: y\n  targets: [2]\n")
    pfad, art = ks.schluessel_anker(d, 3)
    assert (pfad, art) == ("[capability=y]", None)
    assert ks.schluessel_anker(d, 4) == ("[capability=y].targets", None)
    vorher = anker.yaml_knoten_digest(d, pfad)
    _schreibe(d, "- capability: neu\n  targets: [0]\n- capability: x\n  targets: [1]\n"
                 "- capability: y\n  targets: [2]\n")
    assert anker.yaml_knoten_digest(d, pfad) == vorher
    assert ks.schluessel_anker(d, 5) == ("[capability=y]", None)
    assert ks.schluessel_anker(d, 1, 6) == (None, "anker_zu_grob")


def test_schluessel_anker_identitaet_nur_wenn_eindeutig(tmp_path):
    d = tmp_path / "e.yaml"
    _schreibe(d, "a:\n  l:\n    - name: x\n      id: 1\n    - name: x\n      id: 2\n    - k: 3\n")
    assert ks.schluessel_anker(d, 3) == ("a.l[id=1]", None)      # name doppelt -> naechster Schluessel id
    assert ks.schluessel_anker(d, 7) == ("a.l[2]", "anker_positional")


TOML_TEXT = (
    'oben = 1\n'           # 1
    '[srv]\n'              # 2
    'port = 2\n'           # 3
    'liste = [\n'          # 4
    '  "a",\n'             # 5
    '  ["x"],\n'           # 6
    ']\n'                  # 7
    '[[agent]]\n'          # 8
    'name = "a"\n'         # 9
    '[[agent]]\n'          # 10
    'name = "b"\n'         # 11
    'text = """\n'         # 12
    'k = v\n'              # 13
    '"""\n'                # 14
    '\n'                   # 15
    'args = ["x"]\n'       # 16
    '[[werk]]\n'           # 17
    'x = 1\n'              # 18
)


def test_schluessel_anker_toml(tmp_path):
    d = tmp_path / "c.toml"
    _schreibe(d, TOML_TEXT)
    a = ks.schluessel_anker
    assert a(d, 1) == (None, "anker_zu_grob")
    assert a(d, 3) == (None, "anker_zu_grob")                    # Skalar in [srv] -> Tabelle srv (oberste)
    assert a(d, 6) == ("srv.liste", None)
    assert a(d, 4, 7) == ("srv.liste", None)
    assert a(d, 9) == ("agent[name=a]", None)
    assert a(d, 13) == ("agent[name=b]", None)                   # in mehrzeiligem String -> Eintrag
    assert a(d, 15) == ("agent[name=b].args", None)              # Leerzeile -> naechster Schluessel
    assert a(d, 16) == ("agent[name=b].args", None)
    assert a(d, 18) == ("werk[0]", "anker_positional")
    for pfad in ("agent[name=b].args", "agent[name=a]", "werk[0]"):
        assert anker.yaml_knoten_digest(d, pfad) is not None


# ---------- Verankerung ----------

def _konfig(tmp_path, ketten="docs/KETTEN.md"):
    return dataclasses.replace(vm_laden(), repo=tmp_path,
                               spaces={"m": SpaceKonfig("m", "sp/m", "werkzeuge")}, ketten_md=ketten)


def _manifest(tmp_path, *docs):
    return {"m": SimpleNamespace(corpus_root=str(tmp_path / "sp/m"), documents=tuple(docs))}


def test_stelle_verankern_alle_arten(tmp_path):
    _schreibe(tmp_path / "sp/m/api.py", "def f():\n    return 1\n\n\nx = 2\n")
    _schreibe(tmp_path / "sp/m/db/001.sql",
              "CREATE TABLE marketing.t (id int);\n\nCREATE FUNCTION marketing.fn() AS 1;\n")
    _schreibe(tmp_path / "sp/m/c.yaml", "a:\n  b:\n    c: 1\n")
    _schreibe(tmp_path / "sp/m/grob.yaml", "a:\n  b: 1\n")
    _schreibe(tmp_path / "scripts/w.py", "print(1)\n")
    _schreibe(tmp_path / "x.ts", "let a\n")
    konfig = _konfig(tmp_path)
    manifeste = _manifest(tmp_path, _doc("m.api.f", "api.py", 1, 2))
    sql_defs = {"marketing.t": SqlDefinition("sp/m/db/001.sql", 1, "marketing.t", "table", "dt"),
                "marketing.fn": SqlDefinition("sp/m/db/001.sql", 3, "marketing.fn", "function", "df")}

    def v(pfad, start):
        return ks.stelle_verankern(ks.Stelle(pfad, start, None, "voll"), konfig, manifeste, sql_defs, "c1")

    assert (v("sp/m/api.py", 2).art, v("sp/m/api.py", 2).anker) == ("symbol", "m.api.f")
    assert v("sp/m/api.py", 5) is None  # Modulebene im Space, Zeile ohne Namen
    assert v("sp/m/api.py", None).art == "datei"
    b = v("sp/m/db/001.sql", 1)
    assert (b.art, b.anker, b.digest, b.zeile) == ("sql", "marketing.t", "dt", 1)
    assert v("sp/m/db/001.sql", 3).anker == "marketing.fn"
    assert v("sp/m/db/001.sql", 2).art == "datei"  # zwischen den Anweisungen: kein fremdes Objekt
    b = v("sp/m/c.yaml", 3)
    assert (b.art, b.anker) == ("yaml_key", "a.b") and b.digest
    g = ks.stelle_verankern_mit_grund(ks.Stelle("sp/m/grob.yaml", 2, None, "voll"), konfig, manifeste,
                                      sql_defs, "c1")
    assert (g.beleg, g.pruefpunkt) == (None, "anker_zu_grob")
    assert v("scripts/w.py", 1).art == "datei"  # Modulebene ausserhalb der Spaces
    assert v("x.ts", 1).art == "datei"
    assert v("fehlt/nicht.py", 1) is None
    assert ks.stelle_verankern(ks.Stelle(None, 3, None, "auslassung"), konfig, manifeste, sql_defs, "c1") is None


def test_stelle_verankern_bereich_erste_umschlossene_zeile(tmp_path):
    _schreibe(tmp_path / "sp/m/a.py", "X = 1\n\ndef g():\n    pass\n")
    konfig = _konfig(tmp_path)
    manifeste = _manifest(tmp_path, _doc("m.a.g", "a.py", 3, 4))
    b = ks.stelle_verankern(ks.Stelle("sp/m/a.py", 1, 4, "voll"), konfig, manifeste, {}, "c")
    assert (b.art, b.anker, b.zeile) == ("symbol", "m.a.g", 3)
    assert ks.stelle_verankern(ks.Stelle("sp/m/a.py", 1, 2, "voll"), konfig, manifeste, {}, "c") is None


def _zeile(von_nach, mech, code):
    md = "\n".join(["### K1 · T", KOPF, "|---|---|---|---|---|---|",
                    f"| 1 | {von_nach} | {mech} | {code} | 0 | läuft |"])
    return ks.hop_tabellen_lesen(md)[0]


def test_zeilen_namen():
    hz = _zeile("Agent → `versandauftrag_uebernehmen` → `entwurf_erstellen` (sechs Tore)",
                "Python `marketing_pult.anfrage(\"POST\", \"/x\")`, SQL `marketing.versandauftrag_anlegen()`",
                "`vibemind-os/a/server.py:7448`, `:2201` · `swe_design`")
    assert ks.zeilen_namen(hz) == {"versandauftrag_uebernehmen", "entwurf_erstellen", "anfrage", "POST",
                                   "versandauftrag_anlegen", "swe_design"}
    assert ks.zeilen_namen(_zeile("A → B", "Capability-Eintrag", "`a/b.py:3`")) == frozenset()


def _py_fall(tmp_path, quelle, docs, von_nach, mech, code, start, ende=None):
    _schreibe(tmp_path / "sp/m/x.py", quelle)
    konfig = _konfig(tmp_path)
    hz = _zeile(von_nach, mech, code)
    return ks.stelle_verankern_mit_grund(ks.Stelle("sp/m/x.py", start, ende, "voll"), konfig,
                                         _manifest(tmp_path, *docs), {}, "c", namen=ks.zeilen_namen(hz),
                                         tokens=ks.zeilen_tokens(hz))


def test_namensgefuehrt_k1_1_versand_beauftragen_statt_kanal_kann_senden(tmp_path):
    # Replik K1/1: werkzeuge.py:294-297 zeigt heute in `_kanal_kann_senden`, gemeint ist `versand_beauftragen`.
    docs = (_doc("claw.werkzeuge._kanal_kann_senden", "x.py", 290, 300),
            _doc("claw.werkzeuge.versand_beauftragen", "x.py", 320, 360),
            _doc("claw.werkzeuge.anderes", "x.py", 400, 410))
    v = _py_fall(tmp_path, "x = 1\n", docs, "Pult/Agent → Marketing-Sidecar `:8130`",
                 "MCP `versand_beauftragen` (Kanäle `whatsapp, email`)",
                 "`vibemind-os/spaces/marketing/claw/werkzeuge.py:294-297`", 294, 297)
    assert (v.beleg.anker, v.beleg.zeile, v.pruefpunkt) == ("claw.werkzeuge.versand_beauftragen", 320, None)


def test_namensgefuehrt_k1_5_naechstes_benanntes_symbol(tmp_path):
    # Replik K1/5: server.py:7448, :2201 -> `versandauftrag_uebernehmen` bzw. `entwurf_erstellen`, nicht
    # `_termin_einladen`, das heute bei 2201 steht.
    docs = (_doc("s.server.entwurf_erstellen", "x.py", 2100, 2190),
            _doc("s.server._termin_einladen", "x.py", 2195, 2230),
            _doc("s.server.versandauftrag_uebernehmen", "x.py", 7460, 7520),
            _doc("s.server.kontakt_privat_setzen", "x.py", 7440, 7455))
    vn = "Agent → `versandauftrag_uebernehmen` → `entwurf_erstellen` (sechs Tore)"
    v1 = _py_fall(tmp_path, "x = 1\n", docs, vn, "MCP-Werkzeug", "`server.py:7448`, `:2201`", 7448)
    v2 = _py_fall(tmp_path, "x = 1\n", docs, vn, "MCP-Werkzeug", "`server.py:7448`, `:2201`", 2201)
    assert v1.beleg.anker == "s.server.versandauftrag_uebernehmen"
    assert v2.beleg.anker == "s.server.entwurf_erstellen"


def test_namensgefuehrt_k2_2_unbekannte_namen_beleg_unsicher(tmp_path):
    # Replik K2/2: marketing_pult.py:36 liegt heute in `medien_verweise`; die Zeile nennt nur
    # `MARKETING_PULT_URL`/`/api/pult/*`/`X-Pult-Key` -> kein falscher Anker, sondern beleg_unsicher.
    docs = (_doc("p.marketing_pult.anfrage", "x.py", 23, 28), _doc("p.marketing_pult.medien_verweise", "x.py", 31, 45))
    v = _py_fall(tmp_path, "x = 1\n", docs, "Pult-Client → Marketing-API",
                 "HTTP `MARKETING_PULT_URL` + `/api/pult/*`, Header `X-Pult-Key`",
                 "`vibemind-os/spaces/sales-claw/sales-mcp/marketing_pult.py:11`, `:36`", 36)
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"
    assert "p.marketing_pult.medien_verweise" in v.text and "MARKETING_PULT_URL" in v.text
    m = _py_fall(tmp_path, "x = 1\n", docs, "Pult-Client → Marketing-API", "HTTP `MARKETING_PULT_URL`",
                 "`x.py:11`", 11)
    assert m.beleg is None and m.pruefpunkt == "beleg_unsicher"  # Modulebene, Namen ohne Treffer


def test_zeilen_tokens_normalisiert():
    hz = _zeile("Sidecar → Marketing-API `:5510`",
                "HTTP `POST /api/versandauftraege` (`X-API-Key`), `\"MARKETING_PULT_URL\"`, "
                "`marketing.versandauftrag_anlegen()`, `approved`, `Status`, `ok`",
                "`vibemind-os/spaces/marketing/api/server.py:515-530`, `:7` · `swe_design`")
    assert ks.zeilen_tokens(hz) == {"/api/versandauftraege", "X-API-Key", "MARKETING_PULT_URL",
                                    "marketing.versandauftrag_anlegen", "swe_design"}


def test_ruling27_k1_2_route_enthaelt_pfad_literal(tmp_path):
    # Replik K1/2: server.py:515-530, die Zeile nennt `POST /api/versandauftraege`; die umschliessende
    # Route-Funktion (def ab 521) traegt genau dieses Literal in ihrem Dekorator (520) -> verankert, obwohl
    # kein Name passt. manifest.source beginnt erst bei `def`; der Dekorator gehoert zur Definition.
    quelle = "#\n" * 489 + "def l():\n" + "    pass\n" * 9 + "#\n" * 20 + (
        '@app.post("/api/versandauftraege")\ndef versandauftrag_anlegen_route():\n    return 1\n')
    route = _doc("api.server.versandauftrag_anlegen_route", "x.py", 521, 522,
                 "def versandauftrag_anlegen_route():\n    return 1")
    davor = _doc("api.server.list_broadcast_proposals_route", "x.py", 490, 499, "def l(): ...")
    v = _py_fall(tmp_path, quelle, (davor, route), "Sidecar → Marketing-API `:5510`",
                 "HTTP `POST /api/versandauftraege` (`X-API-Key`)",
                 "`vibemind-os/spaces/marketing/api/server.py:515-530`", 515, 530)
    assert (v.beleg.anker, v.beleg.zeile, v.pruefpunkt) == ("api.server.versandauftrag_anlegen_route", 521, None)


def test_ruling27_nur_stoppwort_bleibt_unsicher(tmp_path):
    docs = (_doc("s.dispatch.claim", "x.py", 160, 175, "def claim():\n    status = 'approved'\n"),)
    v = _py_fall(tmp_path, "x = 1\n", docs, "`approved` → Dispatcher", "atomarer Claim `approved`",
                 "`dispatch.py:166`", 166)
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"


def test_ruling27_token_nur_in_der_datei_nicht_im_symbol(tmp_path):
    quelle = "URL = 'MARKETING_PULT_URL'\n"
    docs = (_doc("p.pult.medien_verweise", "x.py", 31, 45, "def medien_verweise():\n    return []\n"),)
    v = _py_fall(tmp_path, quelle, docs, "Pult-Client → Marketing-API", "HTTP `MARKETING_PULT_URL`",
                 "`x.py:36`", 36)
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"


def test_ruling27_auch_ausserhalb_der_spaces(tmp_path):
    quelle = 'def pruefe_laura():\n    return get("http://vm:8765/healthz")\n\n\ndef anderes():\n    return 1\n'
    _schreibe(tmp_path / "scripts/kw.py", quelle)
    konfig = _konfig(tmp_path)
    hz = _zeile("Wächter → Laura (VM)", "`GET http://<VM>:8765/healthz`, `/healthz`", "`scripts/kw.py:2`")
    def v(z):
        return ks.stelle_verankern_mit_grund(ks.Stelle("scripts/kw.py", z, None, "voll"), konfig, {}, {}, "c",
                                             namen=ks.zeilen_namen(hz), tokens=ks.zeilen_tokens(hz))
    assert v(2).beleg.anker == "scripts.kw.pruefe_laura"
    assert (v(6).beleg, v(6).pruefpunkt) == (None, "beleg_unsicher")


def test_namensgefuehrt_ohne_namen_umschliessend(tmp_path):
    docs = (_doc("m.x.f", "x.py", 1, 5),)
    v = _py_fall(tmp_path, "x = 1\n", docs, "A → B", "Capability-Eintrag", "`x.py:3`", 3)
    assert v.beleg.anker == "m.x.f" and v.pruefpunkt is None


def test_ausserhalb_spaces_symbol_per_ast_und_beleg_pruefen(tmp_path):
    quelle = "import os\n\n\nclass K:\n    def lauf(self):\n        return 1\n\n\ndef g():\n    return 2\n"
    _schreibe(tmp_path / "scripts/w.py", quelle)
    konfig = _konfig(tmp_path)
    namen = ks.zeilen_namen(_zeile("A → B", "Python `K.lauf`", "`scripts/w.py:6`"))
    v = ks.stelle_verankern_mit_grund(ks.Stelle("scripts/w.py", 6, None, "voll"), konfig, {}, {}, "c",
                                      namen=namen)
    b = v.beleg
    assert (b.art, b.file, b.anker, b.zeile) == ("symbol", "scripts/w.py", "scripts.w.K.lauf", 6)
    assert b.digest == hashlib.sha256("    def lauf(self):\n        return 1".encode()).hexdigest()
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_ok"
    ohne = ks.stelle_verankern(ks.Stelle("scripts/w.py", 10, None, "voll"), konfig, {}, {}, "c")
    assert ohne.anker == "scripts.w.g"
    assert ks.stelle_verankern(ks.Stelle("scripts/w.py", 1, None, "voll"), konfig, {}, {}, "c").art == "datei"
    _schreibe(tmp_path / "scripts/w.py", quelle.replace("return 1", "return 3"))
    assert anker.beleg_pruefen(b, {}, tmp_path) == "beleg_geaendert"


def test_sql_name_der_umfassenden_anweisung_k1_3(tmp_path):
    # Replik K1/3: 043:151 ist `CREATE OR REPLACE FUNCTION marketing.versandauftrag_anlegen(`; die neueste
    # Definition steht in 093 (sql_defs ist je Name). Nie die Tabelle davor waehlen.
    text = ("CREATE TABLE marketing.versandauftraege (\n  id int\n);\n\n"
            "CREATE OR REPLACE FUNCTION marketing.versandauftrag_anlegen(\n  p int\n) RETURNS int AS $$\n"
            "BEGIN\n  RETURN p; -- ; im Koerper\nEND;\n$$ LANGUAGE plpgsql;\n\nGRANT EXECUTE ON x TO y;\n")
    _schreibe(tmp_path / "sp/m/db/043_v.sql", text)
    konfig = _konfig(tmp_path)
    sql_defs = {
        "marketing.versandauftraege": SqlDefinition("sp/m/db/043_v.sql", 1, "marketing.versandauftraege",
                                                    "table", "dt"),
        "marketing.versandauftrag_anlegen": SqlDefinition("sp/m/db/093_neu.sql", 7,
                                                          "marketing.versandauftrag_anlegen", "function", "dn"),
    }
    assert ks.sql_name_an(tmp_path / "sp/m/db/043_v.sql", 9) == "marketing.versandauftrag_anlegen"
    b = ks.stelle_verankern(ks.Stelle("sp/m/db/043_v.sql", 5, None, "voll"), konfig, {}, sql_defs, "c")
    assert (b.art, b.file, b.anker, b.digest, b.zeile) == (
        "sql", "sp/m/db/093_neu.sql", "marketing.versandauftrag_anlegen", "dn", 7)
    assert ks.stelle_verankern(ks.Stelle("sp/m/db/043_v.sql", 2, None, "voll"), konfig, {}, sql_defs,
                               "c").anker == "marketing.versandauftraege"
    assert ks.stelle_verankern(ks.Stelle("sp/m/db/043_v.sql", 14, None, "voll"), konfig, {}, sql_defs,
                               "c").art == "datei"  # GRANT: kein CREATE umfasst die Zeile


def test_datei_index_ausschluss_aus_konfig(tmp_path):
    _schreibe(tmp_path / "a/einzig.py", "x = 1\n")
    _schreibe(tmp_path / "versteckt/einzig.py", "x = 1\n")
    konfig = dataclasses.replace(_konfig(tmp_path), ausschluss_namen=frozenset({"versteckt"}))
    st = ks.code_stellen_aufloesen("`einzig.py:1`", None, tmp_path, konfig=konfig)
    assert [(s.pfad, s.form) for s in st] == [("a/einzig.py", "anker")]
    ohne = dataclasses.replace(konfig, ausschluss_namen=frozenset())
    assert ks.code_stellen_aufloesen("`einzig.py:1`", None, tmp_path, konfig=ohne)[0].pfad is None


# ---------- Seed ----------

def test_seed_graph_fixture_zwei_hops_mit_belegen(tmp_path):
    _schreibe(tmp_path / "sp/m/api.py", "def versand():\n    x = 1\n    return x\n")
    _schreibe(tmp_path / "sp/m/db/001_x.sql",
              "CREATE TABLE marketing.t (id int);\n\n\n\nCREATE FUNCTION marketing.fn() AS 1;\n\n")
    md = "\r\n".join([
        "# Ketten", "", "### K1 · Test → Kette", "", KOPF, "|---|---|---|---|---|---|",
        "| 1 | Pult → Sidecar `:8130` | MCP `versand` | `sp/m/api.py:2` | 008 | läuft |",
        "| 3a | API → `marketing.fn()` | SQL `marketing.fn` | `sp/m/db/001_x.sql:5`, `043…sql:297` | 008, 015 "
        "| **bricht** |",
        "| 4 | Gate ohne Richtung | Python | — | 015 | bewiesen |",
        "| kaputt |",
        "| 5 | VM `127.0.0.1:8130` → `:9999` | HTTP | — | 015 | verdrahtet |", "",
    ])
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/KETTEN.md").write_bytes(md.encode("utf-8"))
    konfig = _konfig(tmp_path)
    manifeste = _manifest(tmp_path, _doc("m.api.versand", "api.py", 1, 3))
    sql_defs = {"marketing.t": SqlDefinition("sp/m/db/001_x.sql", 1, "marketing.t", "table", "dt"),
                "marketing.fn": SqlDefinition("sp/m/db/001_x.sql", 5, "marketing.fn", "function", "df")}

    g, pp = ks.seed_graph(konfig, manifeste, sql_defs, "aussen1", "sub1")

    assert g.commit_aussen == "aussen1" and g.commit_submodul == "sub1"
    assert g.knoten["kette:K1"].typ == "kette" and g.knoten["kette:K1"].name == "Test → Kette"
    assert g.knoten["dienst:Sidecar"].attribute == {"port": "8130"}
    assert g.knoten["dienst::9999"].attribute == {"port": "9999"}
    assert g.knoten["sql_funktion:marketing.fn"].typ == "sql_funktion"
    hops = {k.attribute["nr"]: k for k in g.kanten.values() if k.art == "hop"}
    assert set(hops) == {"1", "3a", "4", "5"}
    assert (hops["5"].von, hops["5"].nach) == ("dienst:Sidecar", "dienst::9999")  # 127.0.0.1:8130 -> Sidecar

    h1 = hops["1"]
    assert h1.id == "hop:komponente:Pult->dienst:Sidecar#K1/1" and h1.quelle == "seed"
    assert [(b.art, b.file, b.anker, b.commit, b.zeile) for b in h1.belege] == [
        ("symbol", "sp/m/api.py", "m.api.versand", "aussen1", 2)]
    assert h1.attribute["zustand"]["gemessen"] == {"wert": "laeuft", "zeit": None}
    assert h1.attribute["mechanismus_typ"] == "MCP" and h1.attribute["belegt_durch"] == "008"
    assert h1.attribute["kette"] == "K1"

    h3 = hops["3a"]
    assert h3.id.endswith("#K1/3a") and h3.nach == "sql_funktion:marketing.fn"
    assert [(b.art, b.anker, b.digest) for b in h3.belege] == [("sql", "marketing.fn", "df")]
    assert h3.attribute["zustand"]["gemessen"]["wert"] == "bricht"
    assert h3.attribute["belegt_durch"] == "008, 015"

    h4 = hops["4"]
    assert h4.von == h4.nach == "komponente:Gate ohne Richtung" and h4.belege == ()

    arten = Counter(p.art for p in pp)
    assert arten == Counter({"beleg_nicht_aufloesbar": 1, "zustand_unbekannt": 1,
                             "hop_ohne_beleg": 2, "zeile_unlesbar": 1})
    nicht = next(p for p in pp if p.art == "beleg_nicht_aufloesbar")
    assert "043…sql:297" in nicht.text and nicht.datei == "docs/KETTEN.md" and nicht.zeile == 8
    assert nicht.aussage_id == "K1/3a"


def test_seed_graph_meldet_unsicher_positional_zu_grob(tmp_path):
    _schreibe(tmp_path / "sp/m/api.py", "def andere():\n    return 1\n")
    _schreibe(tmp_path / "sp/m/c.yaml", "l:\n  a:\n    - {k: 1}\nwurzel: 1\n")
    md = "\n".join([
        "### K2 · Y", KOPF, "|---|---|---|---|---|---|",
        "| 1 | A → B | Python `versand` | `sp/m/api.py:1` | 0 | läuft |",
        "| 2 | A → B | Datei | `sp/m/c.yaml:3`, `:4` | 0 | läuft |", "",
    ])
    _schreibe(tmp_path / "docs/KETTEN.md", md)
    g, pp = ks.seed_graph(_konfig(tmp_path), _manifest(tmp_path, _doc("m.api.andere", "api.py", 1, 2)),
                          {}, "a", "b")
    assert Counter(p.art for p in pp) == Counter({"beleg_unsicher": 1, "anker_positional": 1, "anker_zu_grob": 1})
    hops = {k.attribute["nr"]: k for k in g.kanten.values()}
    assert hops["1"].belege == ()
    assert [(b.art, b.anker) for b in hops["2"].belege] == [("yaml_key", "l.a[0]")]


def test_seed_graph_ohne_ketten_md_wirft_nicht(tmp_path):
    g, pp = ks.seed_graph(_konfig(tmp_path, "docs/fehlt.md"), {}, {}, "a", "b")
    assert not g.kanten and [p.art for p in pp] == ["ketten_md_fehlt"]


@pytest.mark.skipif(not KETTEN.is_file() or not (REPO / "vibemind-os/spaces/marketing").is_dir()
                    or not korpus.manifest_verfuegbar(), reason="Submodul/Manifest fehlt")
def test_seed_graph_echtes_repo_rauchtest():
    from selbstbild.doku.extraktoren import sql
    konfig = vm_laden()
    commit_aussen = korpus.commit_von(konfig.repo)
    commit_sub = korpus.commit_von(konfig.repo / "vibemind-os")
    manifeste = {n: korpus.manifest_bauen(s, konfig, commit_sub) for n, s in konfig.spaces.items()}
    defs, _ = sql.sql_definitionen(konfig)
    g, pp = ks.seed_graph(konfig, manifeste, defs, commit_aussen, commit_sub)
    hops = [k for k in g.kanten.values() if k.art == "hop"]
    assert len(hops) >= 95
    arten = Counter(b.art for k in hops for b in k.belege)
    verankert = sum(1 for k in hops if k.belege)
    print("\nBELEGE je Art:", dict(arten), "| Hops mit >=1 Beleg:", verankert, "von", len(hops),
          "| Pruefpunkte:", dict(Counter(p.art for p in pp)))
    assert verankert >= SCHWELLE_VERANKERT


# ---------- Ruling 29: YAML namensgefuehrt, Spacenamen kein Beleg ----------

CAPS_K15 = (
    "- capability: minibook.discuss\n"                          # 1
    "  target: direct:spaces.minibook.tools.minibook_tools\n"   # 2
    "- capability: minibook.collaborate\n"                      # 3
    "  target: direct:spaces.minibook.tools.minibook_tools\n"   # 4
    "# ===== ROSE / Flowzen =====\n"                            # 5
    "- capability: rose.recommend\n"                            # 6
    "  target: supabase:rose.recommend\n"                       # 7
)


def _yaml_fall(tmp_path, inhalt, von_nach, mech, zeile, ende=None, name="caps.yaml"):
    _schreibe(tmp_path / f"sp/m/{name}", inhalt)
    hz = _zeile(von_nach, mech, f"`sp/m/{name}:{zeile}`")
    return ks.stelle_verankern_mit_grund(ks.Stelle(f"sp/m/{name}", zeile, ende, "voll"), _konfig(tmp_path), {}, {},
                                         "c", namen=ks.zeilen_namen(hz), tokens=ks.zeilen_tokens(hz))


def test_yaml_anker_kommentarbanner_ohne_namensbezug_ist_unsicher(tmp_path):
    # Replik K15/1: zitierte Zeile ist heute ein Kommentar-Banner; der Knoten darunter ist rose.recommend.
    v = _yaml_fall(tmp_path, CAPS_K15, "`minibook.discuss/collaborate/list_projects` → `direct:spaces.minibook.x`",
                   "Capability, `rule:minibook_verified_result`", 5)
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"
    assert "[capability=rose.recommend]" in v.text


def test_yaml_anker_identitaetswert_gewinnt(tmp_path):
    inhalt = ("- capability: schedule_openclaw_cron\n  target: direct:a\n  x: 1\n"
              "- capability: schedule_create\n  target: direct:b\n  x: 2\n")
    v = _yaml_fall(tmp_path, inhalt, "A → `direct:spaces.schedule.execution:<fn>`",
                   "Capability `schedule_create`", 2)
    assert (v.beleg.anker, v.beleg.zeile, v.pruefpunkt) == ("[capability=schedule_create]", 4, None)
    # liegt die zitierte Zeile schon im benannten Eintrag, bleibt der feinere Knoten
    v = _yaml_fall(tmp_path, inhalt.replace("x: 2", "x: [2]"), "A → B", "Capability `schedule_create`", 6)
    assert v.beleg.anker == "[capability=schedule_create].x"


def test_yaml_anker_bestaetigt_ueber_pfad_oder_knotentext(tmp_path):
    inhalt = "spaces:\n  agentfarm:\n    events:\n      agentfarm.status:\n        tool: captain_codex_status\n"
    v = _yaml_fall(tmp_path, inhalt, "Ledger-Readback `agentfarm.status`", "`captain_codex_status`", 5)
    assert v.beleg.anker == 'spaces.agentfarm.events["agentfarm.status"]'      # Schluessel im Pfad
    v = _yaml_fall(tmp_path, inhalt, "Ledger-Readback", "`captain_codex_status`", 5)
    assert v.beleg.anker == 'spaces.agentfarm.events["agentfarm.status"]'      # Text des Knotens
    v = _yaml_fall(tmp_path, inhalt, "Ledger-Readback", "`etwas_anderes`", 5)
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"
    v = _yaml_fall(tmp_path, inhalt, "Ledger-Readback", "ohne Backticks", 5)
    assert v.beleg.anker == 'spaces.agentfarm.events["agentfarm.status"]'      # Zeile ohne Tokens


def test_yaml_anker_identitaet_rettet_zu_grob_und_toml(tmp_path):
    inhalt = "- capability: a\n  t: 1\n- capability: b\n  t: 2\n"
    v = _yaml_fall(tmp_path, inhalt, "A → B", "`b`-los `capability_b`", 1, 4)
    assert (v.beleg, v.pruefpunkt) == (None, "anker_zu_grob")
    v = _yaml_fall(tmp_path, inhalt.replace("b\n", "bbbbb\n", 1), "A → B", "Capability `bbbbb`", 1, 4)
    assert v.beleg.anker == "[capability=bbbbb]"
    toml = '[[srv]]\nname = "spaces-ideas"\nport = 1\n[[srv]]\nname = "spaces-minibook"\nport = 2\n'
    v = _yaml_fall(tmp_path, toml, "A → B", "MCP `spaces-minibook`", 3, name="o.toml")
    assert (v.beleg.anker, v.beleg.zeile) == ("srv[name=spaces-minibook]", 4)
    v = _yaml_fall(tmp_path, toml, "A → B", "MCP `minibook.status`", 3, name="o.toml")
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"


def test_spacename_token_zaehlt_nicht(tmp_path):
    (tmp_path / "vibemind-os/spaces/ideas").mkdir(parents=True)
    _schreibe(tmp_path / "vibemind-os/spaces/m/x.py", "def lade():\n    return 'ideas'\n")
    konfig = dataclasses.replace(vm_laden(), repo=tmp_path,
                                 spaces={"m": SpaceKonfig("marketing", "vibemind-os/spaces/m", "werkzeuge")})
    schwach = ks.schwache_tokens(konfig)
    assert {"ideas", "marketing", "spaces", "vibemind-os", "m"} <= schwach
    hz = _zeile("A → B", "PostgREST `GET ideas`, `ideas`, `marketing`", "`vibemind-os/spaces/m/x.py:2`")
    assert ks.zeilen_tokens(hz, schwach=schwach) == frozenset()
    assert ks.zeilen_tokens(hz) == {"ideas", "marketing"}
    lade = _doc("m.x.lade", "x.py", 1, 2, "def lade():\n    return 'ideas'")
    manifeste = {"marketing": SimpleNamespace(corpus_root=str(tmp_path / "vibemind-os/spaces/m"), documents=(lade,))}
    v = ks.stelle_verankern_mit_grund(ks.Stelle("vibemind-os/spaces/m/x.py", 2, None, "voll"), konfig, manifeste, {},
                                      "c", namen=ks.zeilen_namen(hz), tokens=ks.zeilen_tokens(hz, schwach=schwach))
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"



# ---------- Ruling 30: Globs, Zeilenknoten vor Nachbareintrag ----------

SCHEDULE = ("- capability: schedule_openclaw_cron\n  target: direct:a\n  x: 1\n"
            "- capability: schedule_create\n  target: direct:b\n  x: 2\n"
            "- capability: anderes_ding\n  target: direct:c\n")


def test_glob_bestaetigt_identitaet(tmp_path):
    v = _yaml_fall(tmp_path, SCHEDULE, "`schedule_*` → `direct:spaces.schedule.execution:<fn>`", "Capability", 2)
    assert (v.beleg.anker, v.pruefpunkt) == ("[capability=schedule_openclaw_cron]", None)
    # benannter Eintrag per Glob, naechster zur zitierten Zeile, wenn der Zeilenknoten nicht passt
    v = _yaml_fall(tmp_path, SCHEDULE, "`schedule_*` → B", "Capability", 8)
    assert (v.beleg.anker, v.beleg.zeile) == ("[capability=schedule_create]", 4)


def test_glob_bestaetigt_nicht_ueber_knotentext(tmp_path):
    inhalt = 'spaces:\n  a:\n    events:\n      x: "schedule_*"\n'
    v = _yaml_fall(tmp_path, inhalt, "`schedule_*` → B", "Capability", 4)
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"


def test_glob_zaehlt_nie_fuer_python(tmp_path):
    quelle = "def f():\n    erlaubt = 'mcp__laura__*'\n    return erlaubt"
    docs = (_doc("m.x.f", "x.py", 1, 3, quelle),)
    v = _py_fall(tmp_path, quelle + "\n", docs, "Claude-CLI → Laura-MCP", "Allow-List `mcp__laura__*`", "`x.py:2`", 2)
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"


def test_zeilenknoten_bestaetigt_schlaegt_benannten_nachbarn(tmp_path):
    # Replik K9/1: `:575` liegt in idea_to_project, dessen Text `idea_to_project_op` nennt; der benannte
    # Nachbar bubble_promote (Token exakt) gewinnt NICHT gegen den bestaetigten Zeilenknoten.
    inhalt = ("- capability: bubble_promote\n  target: supabase:bubble_promote\n  x: 1\n"
              "- capability: idea_to_project\n  validator: {op: idea_to_project_op}\n  x: 2\n")
    v = _yaml_fall(tmp_path, inhalt, "Bubble → `projects`-Zeile",
                   "MCP `bubble_promote` oder Brain `bubble_promote_op`/`idea_to_project_op`", 5)
    assert (v.beleg.anker, v.pruefpunkt) == ("[capability=idea_to_project].validator", None)
    v = _yaml_fall(tmp_path, inhalt, "Bubble → `projects`-Zeile", "MCP `bubble_promote`", 6)
    assert v.beleg.anker == "[capability=bubble_promote]"  # Zeilenknoten unbestaetigt -> benannter Eintrag


# ---------- Task 9b Fix-Runde 1: schwache Tokens befreien YAML nicht ----------

def test_yaml_zeile_nennt_nur_spacename_ist_unsicher(tmp_path):
    inhalt = "spaces:\n  ideas:\n    events:\n      x: 1\n"
    _schreibe(tmp_path / "sp/m/r.yaml", inhalt)
    hz = _zeile("Research → `ideas`", "PostgREST `GET ideas`", "`sp/m/r.yaml:4`")
    tokens = ks.zeilen_tokens(hz, schwach={"ideas"})
    assert tokens == frozenset() and ks.zeilen_namen(hz)
    v = ks.stelle_verankern_mit_grund(ks.Stelle("sp/m/r.yaml", 4, None, "voll"), _konfig(tmp_path), {}, {}, "c",
                                      namen=ks.zeilen_namen(hz), tokens=tokens)
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"
    hz_kurz = _zeile("A → B", "`ok` und `sent`", "`sp/m/r.yaml:4`")  # nur kurze/Stopp-Tokens
    v = ks.stelle_verankern_mit_grund(ks.Stelle("sp/m/r.yaml", 4, None, "voll"), _konfig(tmp_path), {}, {}, "c",
                                      namen=ks.zeilen_namen(hz_kurz), tokens=ks.zeilen_tokens(hz_kurz))
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"


def test_yaml_bestaetigt_aber_ohne_digest_ehrlicher_text(tmp_path, monkeypatch):
    _schreibe(tmp_path / "sp/m/r.yaml", "spaces:\n  a:\n    events:\n      x: wert_hier\n")
    hz = _zeile("A → B", "`wert_hier`", "`sp/m/r.yaml:4`")
    monkeypatch.setattr(ks.anker, "yaml_knoten_digest", lambda datei, pfad: None)
    v = ks.stelle_verankern_mit_grund(ks.Stelle("sp/m/r.yaml", 4, None, "voll"), _konfig(tmp_path), {}, {}, "c",
                                      namen=ks.zeilen_namen(hz), tokens=ks.zeilen_tokens(hz))
    assert v.beleg is None and v.pruefpunkt == "beleg_unsicher"
    assert "bestaetigt" in v.text and "Digest" in v.text and "nennt kein Token" not in v.text


def test_zeilen_namen_ohne_glob_praefixe():
    hz = _zeile("`schedule_*` → `direct:x`", "`mirofish.*`, `rose.*`, `mcp__laura__*`, `echter_name`", "`a/b.py:1`")
    assert ks.zeilen_namen(hz) == {"direct", "echter_name"}


# ---------- Ruling 34: Rohtexte fuer das Rendern ----------

UEBERSICHT_MD = [
    "## Übersicht", "",
    "| # | Kette | Zustand | Reißt bei |", "|---|-------|---------|-----------|",
    "| K1 | Test → Kette (lang, `a|b`) | 🟢 läuft | — |",
    "| K7 | Nur Übersicht | ⚪ nie bewiesen | `x()` wirft |", "",
]


def test_uebersicht_lesen_vier_und_fuenf_spalten():
    vier = ks.uebersicht_lesen("\r\n".join(["﻿# T", "", *UEBERSICHT_MD, "## Teil A",
                                            "| # | Kette | Zustand | Reißt bei |", "|---|---|---|---|",
                                            "| K9 | nicht im Abschnitt | x | y |"]))
    assert vier == {
        "K1": {"titel_roh": "Test → Kette (lang, `a|b`)", "zustand_roh": "🟢 läuft", "reisst_bei_roh": "—"},
        "K7": {"titel_roh": "Nur Übersicht", "zustand_roh": "⚪ nie bewiesen", "reisst_bei_roh": "`x()` wirft"},
    }
    fuenf = ks.uebersicht_lesen("\n".join([
        "## Übersicht", "<!-- graph:begin uebersicht -->",
        "| # | Kette | Zustand | Reißt bei | Graph |", "|---|---|---|---|---|",
        "| K1 | T | 🟢 läuft | — | 🟢 verdrahtet 1/1 · gemessen läuft 27.10. 14:25 |",
        "<!-- graph:end -->"]))
    assert fuenf == {"K1": {"titel_roh": "T", "zustand_roh": "🟢 läuft", "reisst_bei_roh": "—"}}


def test_zellen_escapte_pipe_trennt_nicht():
    assert ks._zellen(r"| 1 | a \| b | `c|d` |") == ["1", r"a \| b", "`c|d`"]


def test_hop_tabelle_render_zusatz_im_zustand_wird_abgeschnitten():
    text = "\n".join(["### K1 · T", KOPF, "|---|---|---|---|---|---|",
                      "| 1 | A → B | MCP | `p/q.py:3` | 008 | läuft (Sperre) · v:ja g:läuft 27.10. 14:25 |",
                      "| 2 | A → B | MCP | `p/q.py:3` | 008 | bricht · v:? g:— |", ""])
    assert [z.zustand for z in ks.hop_tabellen_lesen(text)] == ["läuft (Sperre)", "bricht"]


def test_seed_graph_rohtexte_fuer_render(tmp_path):
    _schreibe(tmp_path / "sp/m/api.py", "def versand():\n    x = 1\n    return x\n")
    md = "\r\n".join([
        "# Ketten", "", *UEBERSICHT_MD, "### K1 · Test → Kette", "", KOPF, "|---|---|---|---|---|---|",
        "| 1 | Pult → Sidecar `:8130` | MCP `versand` | `sp/m/api.py:2`, `043…sql:297` | 008 | läuft (Sperre aktiv) |",
        "| 2 | A → B | Python | — | 015 | **bricht** |", "",
    ])
    _schreibe(tmp_path / "docs/KETTEN.md", "")
    (tmp_path / "docs/KETTEN.md").write_bytes(md.encode("utf-8"))
    g, _ = ks.seed_graph(_konfig(tmp_path), _manifest(tmp_path, _doc("m.api.versand", "api.py", 1, 3)),
                         {}, "a", "b")
    k1 = g.knoten["kette:K1"]
    assert k1.name == "Test → Kette"
    assert k1.attribute == {"titel_roh": "Test → Kette (lang, `a|b`)", "zustand_roh": "🟢 läuft",
                            "reisst_bei_roh": "—"}
    assert g.knoten["kette:K7"].attribute["reisst_bei_roh"] == "`x()` wirft"  # nur in der Uebersicht
    hops = {k.attribute["nr"]: k for k in g.kanten.values() if k.art == "hop"}
    h1 = hops["1"]
    assert h1.attribute["mechanismus_roh"] == h1.attribute["mechanismus"] == "MCP `versand`"
    assert h1.attribute["zustand_roh"] == h1.attribute["zustand_text"] == "läuft (Sperre aktiv)"
    assert h1.attribute["stellen_roh"] == [
        {"roh": "sp/m/api.py:2", "voll": "sp/m/api.py:2", "beleg": ["symbol", "sp/m/api.py", "m.api.versand"]},
        {"roh": "043…sql:297", "voll": "043…sql:297", "beleg": None},
    ]
    assert h1.attribute["code_prosa"] == ""
    assert hops["2"].attribute["stellen_roh"] == [] and hops["2"].attribute["zustand_roh"] == "**bricht**"


def test_render_anker_trennen():
    code, anker_ = ks.render_anker_trennen(
        "`a/b.py:3` `m.b.f`, ⚠ `a/b.py:9`, `c/d.yaml:4` `[datei=x/y.py].ziel`, `e.py:1` `f.py:2`")
    assert code == "`a/b.py:3`, ⚠ `a/b.py:9`, `c/d.yaml:4`, `e.py:1` `f.py:2`"
    assert anker_ == ("m.b.f", "[datei=x/y.py].ziel")  # die YAML-Identitaet mit Pfad wird keine Stelle
    zeile = "| 1 | A → B | Python | `c/d.yaml:4` `[datei=x/y.py].ziel` | 0 | läuft |"
    hz = ks.hop_tabellen_lesen("\n".join(["### K1 · T", KOPF, "|---|---|---|---|---|---|", zeile]))[0]
    assert hz.code_stellen == ("c/d.yaml:4",) and hz.code_anker == ("[datei=x/y.py].ziel",)
    assert "ziel" not in ks.zeilen_namen(hz) and not ks.zeilen_tokens(hz)


def test_code_prosa_menschentext_ohne_stellen_idempotent():
    roh = "`x/disp.py:166`, `:282` · E-Mail `mail.py:243` · Telegram (`x/t.py:1`), laut `abc_def`"
    prosa = ks.code_prosa(roh)
    assert prosa == "E-Mail · Telegram, laut `abc_def`"
    gerendert = f"`x/disp.py:160` `x.disp.f`, ⚠ `x/disp.py:282` — {prosa}"
    assert ks.code_prosa(ks.render_anker_trennen(gerendert)[0]) == prosa
    assert ks.code_prosa("— (laut Doc 008, nicht nachgeprüft)") == "(laut Doc 008, nicht nachgeprüft)"
    assert ks.code_prosa(f"— {ks.code_prosa('— (laut Doc 008)')}") == "(laut Doc 008)"
    assert ks.code_prosa("`a/b.py:3`, `:4`") == "" and ks.code_prosa("—") == ""
