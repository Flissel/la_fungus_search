"""doku_loop: KETTEN.md aus dem Graph rendern, Hand-Edits erkennen (Task 11, Ruling 34)."""

import copy
import dataclasses
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import graph  # noqa: E402
from selbstbild.doku import ketten_seed as ks  # noqa: E402
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import korpus  # noqa: E402
from selbstbild.doku import render_ketten as rk  # noqa: E402
from selbstbild.doku.anker import Beleg  # noqa: E402
from selbstbild.doku.extraktoren.sql import SqlDefinition  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402

KETTEN = REPO / "docs" / "spaces" / "KETTEN.md"
KOPF = "| # | Von → Nach | Mechanismus | Code-Stelle | Belegt durch | Zustand |"
TRENN = "|---|---|---|---|---|---|"
ZEIT = "2026-10-27T13:25:00+00:00"  # 14:25 in Berlin (CET)


def _kette(g, k, titel, zustand, reisst, name=None):
    g.knoten_hinzu(graph.Knoten(f"kette:{k}", "kette", name or titel,
                                attribute={"titel_roh": titel, "zustand_roh": zustand, "reisst_bei_roh": reisst}))


def _hop(g, kette, nr, *, von_nach="A → B", mech="MCP `x`", code="`sp/m/api.py:2`", belegt="008",
         zustand_roh="läuft", belege=(), stellen=None, verdrahtet="unbestimmt", gemessen=None):
    z = graph.hop_zustand_neu()
    z["verdrahtet"] = verdrahtet
    if gemessen is not None:
        z["gemessen"] = gemessen
    von, nach = f"komponente:{kette}-{nr}-von", f"komponente:{kette}-{nr}-nach"
    attr = {"kette": kette, "nr": nr, "von_nach": von_nach, "beidseitig": False,
            "mechanismus": mech, "mechanismus_roh": mech, "mechanismus_typ": "MCP", "code_stelle": code,
            "belegt_durch": belegt, "zustand": z, "zustand_text": zustand_roh, "zustand_roh": zustand_roh,
            "stellen_roh": stellen if stellen is not None else []}
    g.kante_hinzu(graph.Kante(graph.kante_id("hop", von, nach, kette=kette, nr=nr), "hop", von, nach, "seed",
                              tuple(belege), attr))


def _beleg(file="sp/m/api.py", anker="api.versand", zeile=2, art="symbol"):
    return Beleg(art, file, anker, "d", "c", zeile)


def _graph():
    g = graph.Graph()
    _kette(g, "K1", "Eins (lang)", "🟢 läuft", "—", name="Eins")
    _kette(g, "K2", "Zwei", "🔴 bricht", "Sidecar `:8131`")
    b = _beleg()
    _hop(g, "K1", "1", belege=[b], stellen=[{"roh": "sp/m/api.py:2", "beleg": [b.art, b.file, b.anker]}])
    _hop(g, "K1", "3a", mech="SQL `marketing.fn`", belegt="008, 015", zustand_roh="läuft (Sperre aktiv)")
    _hop(g, "K2", "1", von_nach="C → `D`", mech="SQL", belegt="015", zustand_roh="**bricht**")
    return g


def _text(nl="\r\n", bom=True):
    zeilen = [
        "# Ketten", "", "## Übersicht", "",
        "| # | Kette | Zustand | Reißt bei |", "|---|-------|---------|-----------|",
        "| K1 | Eins (lang) | 🟢 läuft | — |", "| K2 | Zwei | 🔴 bricht | Sidecar `:8131` |",
        "", "---", "", "### K1 · Eins", "", "Prosa vor der Tabelle,  mit Leerzeichen am Ende.  ", "",
        KOPF, TRENN,
        "| 1 | A → B | MCP `x` | `sp/m/api.py:2` | 008 | läuft |",
        "| 3a | A → B | SQL `marketing.fn` | `sp/m/api.py:2` | 008, 015 | läuft (Sperre aktiv) |",
        "", "Ein Satz zwischen zwei Tabellen.", "", "### K2 · Zwei", "", KOPF, TRENN,
        "| 1 | C → `D` | SQL | `sp/m/api.py:2` | 015 | **bricht** |", "", "Schluss ohne Zeilenende",
    ]
    return ("﻿" if bom else "") + nl.join(zeilen)


def _ohne_tabellen(text):
    return "".join(z for z in text.splitlines(keepends=True) if not z.lstrip().startswith("|"))


def _ohne_bloecke(text):
    """Alle Zeilen ausserhalb markierter Bloecke (Marker eingeschlossen entfernt)."""
    aus, drin = [], False
    for z in text.splitlines(keepends=True):
        if z.strip().startswith("<!-- graph:begin "):
            drin = True
        elif z.strip() == rk.MARKER_END:
            drin = False
        elif not drin:
            aus.append(z)
    return "".join(aus)


def _block_zeilen(text, block):
    b = rk.bloecke_finden(text)[block]
    return b.inhalt.splitlines()


# ---------- Brief-Tests ----------

def test_einbetten_erhaelt_crlf_und_prosa_byteweise():
    text = _text()
    aus = rk.einbetten(text, _graph())
    assert aus.startswith("﻿# Ketten\r\n")
    assert "\n" not in aus.replace("\r\n", "")  # jede gerenderte Zeile mit CRLF
    assert set(rk.bloecke_finden(aus)) == {"uebersicht", "hops K1", "hops K2"}
    assert "\r\n<!-- graph:begin hops K1 -->\r\n| # | Von → Nach |" in aus
    assert _ohne_bloecke(aus) == _ohne_tabellen(text)  # Prosa byteweise, auch der Satz zwischen den Tabellen
    assert aus.endswith("\r\n\r\nSchluss ohne Zeilenende")
    assert rk.einbetten(aus, _graph()) == aus  # idempotent


def test_einbetten_lf_ohne_bom():
    text = _text(nl="\n", bom=False)
    aus = rk.einbetten(text, _graph())
    assert "\r" not in aus and not aus.startswith("﻿")
    assert _ohne_bloecke(aus) == _ohne_tabellen(text)


def test_rendern_ohne_graphaenderung_erzeugt_keinen_diff():
    g = _graph()
    eingebettet = rk.einbetten(_text(), g)
    erg1 = rk.rendern(eingebettet, g, {})
    assert erg1.text == eingebettet and erg1.pruefpunkte == []
    assert set(erg1.hashes) == {"uebersicht", "hops K1", "hops K2"}
    bloecke = rk.bloecke_finden(eingebettet)
    assert erg1.hashes["hops K1"] == rk.block_hash(bloecke["hops K1"].inhalt)
    erg2 = rk.rendern(erg1.text, copy.deepcopy(g), erg1.hashes)
    assert (erg2.text, erg2.hashes, erg2.pruefpunkte) == (erg1.text, erg1.hashes, [])


def test_hand_edit_bleibt_stehen_und_wird_pruefpunkt():
    g = _graph()
    erg1 = rk.rendern(rk.einbetten(_text(), g), g, {})
    hand = erg1.text.replace("| 008, 015 |", "| 999 |")
    assert hand != erg1.text
    g2 = copy.deepcopy(g)
    for k in g2.kanten.values():
        k.attribute["belegt_durch"] = "042"
    erg2 = rk.rendern(hand, g2, erg1.hashes)
    k1 = "\n".join(_block_zeilen(erg2.text, "hops K1"))
    k2 = "\n".join(_block_zeilen(erg2.text, "hops K2"))
    assert "| 999 |" in k1 and "042" not in k1  # Hand-Edit bleibt stehen
    assert "| 042 |" in k2  # der andere Block wird neu gerendert
    assert [(p.art, p.aussage_id) for p in erg2.pruefpunkte] == [("hand_edit", "hops K1")]
    assert erg2.pruefpunkte[0].zeile == erg2.text.splitlines().index("<!-- graph:begin hops K1 -->") + 1
    # Der alte Hash bleibt, damit der Hand-Edit auch im naechsten Lauf nicht ueberschrieben wird
    assert erg2.hashes["hops K1"] == erg1.hashes["hops K1"]
    erg3 = rk.rendern(erg2.text, g2, erg2.hashes)
    assert erg3.text == erg2.text and [p.art for p in erg3.pruefpunkte] == ["hand_edit"]


def test_uebersicht_zustandsfarben():
    g = graph.Graph()
    faelle = {
        "K1": ([("ja", {"wert": "laeuft", "zeit": ZEIT})] * 2, "🟢 verdrahtet 2/2 · gemessen läuft"),
        "K2": ([("ja", {"wert": "bricht", "zeit": ZEIT})], "🔴 verdrahtet 1/1 · gemessen bricht"),
        "K3": ([("ja", {"wert": "laeuft", "zeit": ZEIT}), ("nein", {"wert": "laeuft", "zeit": ZEIT}),
                ("unbestimmt", {"wert": "laeuft", "zeit": ZEIT})],
               "🟡 verdrahtet 1/3 · gemessen läuft"),
        "K4": ([("ja", {"wert": "keine_probe", "zeit": ZEIT})],
               "⚪ verdrahtet 1/1 · gemessen keine Probe"),
        "K10": ([("nein", {"wert": "laeuft", "zeit": None})], "verdrahtet 0/1 · gemessen —"),  # nur Doku
    }
    for k, (hops, _) in faelle.items():
        _kette(g, k, f"Titel {k}", f"menschlich {k}", f"reisst {k}")
        for i, (v, gem) in enumerate(hops, 1):
            _hop(g, k, str(i), verdrahtet=v, gemessen=gem)
    _kette(g, "K5", "Ohne Hops", "⚪ nie", "—")
    zeilen = rk.uebersicht_rendern(g).splitlines()
    assert zeilen[0] == "| # | Kette | Zustand | Reißt bei | Graph |"
    assert [z.split(" | ")[0] for z in zeilen[2:]] == ["| K1", "| K2", "| K3", "| K4", "| K5", "| K10"]
    for z in zeilen[2:]:
        k = z.split(" | ")[0][2:]
        if k == "K5":
            assert z == "| K5 | Ohne Hops | ⚪ nie | — | verdrahtet 0/0 · gemessen — |"
            continue
        assert z == f"| {k} | Titel {k} | menschlich {k} | reisst {k} | {faelle[k][1]} |"


# ---------- Ruling 34: Inhalte ----------

def test_uebersicht_behaelt_menschentext_und_ergaenzt_graph_spalte():
    zeilen = rk.uebersicht_rendern(_graph(), zeilenende="\r\n").split("\r\n")
    assert zeilen[2] == "| K1 | Eins (lang) | 🟢 läuft | — | verdrahtet 0/2 · gemessen — |"
    assert zeilen[3] == "| K2 | Zwei | 🔴 bricht | Sidecar `:8131` | verdrahtet 0/1 · gemessen — |"
    assert zeilen[-1] == ""


def test_hop_zeile_rohtexte_beleg_mit_aktueller_zeile_und_warnung():
    g = graph.Graph()
    _kette(g, "K1", "T", "z", "r")
    b1 = _beleg("vibemind-os/spaces/marketing/api/server.py", "api.server.versandauftrag_anlegen_route", 520)
    b2 = _beleg("x/caps.yaml", "caps[capability=a|b]", 7, art="yaml_key")
    b3 = _beleg("x/neu.py", "", None, art="datei")  # spaeter dazugekommen, ohne Stelle
    _hop(g, "K1", "2", von_nach="Sidecar → Marketing-API `:5510`", mech="HTTP `POST /api/v` (`X|Y`)",
         belegt="008", zustand_roh="läuft (Sperre aktiv)", belege=[b1, b2, b3], verdrahtet="ja",
         gemessen={"wert": "laeuft", "zeit": ZEIT},
         stellen=[{"roh": "vibemind-os/spaces/marketing/api/server.py:515-530",
                   "beleg": ["symbol", b1.file, b1.anker]},
                  {"roh": "043…sql:297", "voll": "043…sql:297", "beleg": None},
                  {"roh": "x/caps.yaml:7", "beleg": ["yaml_key", b2.file, b2.anker]},
                  {"roh": ":530", "beleg": ["symbol", b1.file, b1.anker]}])
    _hop(g, "K1", "10", zustand_roh="verdrahtet",
         stellen=[{"roh": ":9", "voll": "x/weg.py:9", "beleg": ["symbol", "x/weg.py", "weg"]}])
    _hop(g, "K1", "3", code="—", zustand_roh="nie bewiesen", verdrahtet="nein")
    zeilen = rk.hops_rendern(g, "K1").splitlines()
    assert zeilen[0] == KOPF
    assert [ks._zellen(z)[0] for z in zeilen[2:]] == ["2", "3", "10"]  # natuerliche Reihenfolge
    zellen = ks._zellen(zeilen[2])
    assert len(zellen) == 6
    assert zellen[:3] == ["2", "Sidecar → Marketing-API `:5510`", "HTTP `POST /api/v` (`X|Y`)"]
    assert zellen[3] == ("`vibemind-os/spaces/marketing/api/server.py:520` `api.server.versandauftrag_anlegen_route`"
                         ", ⚠ `043…sql:297`, `x/caps.yaml:7` `caps[capability=a\\|b]`, `x/neu.py`")
    assert zellen[4] == "008"
    assert zellen[5] == "läuft (Sperre aktiv) · v:ja g:läuft"
    assert ks._zellen(zeilen[3])[3:] == ["—", "008", "nie bewiesen · v:nein g:—"]
    assert ks._zellen(zeilen[4])[3] == "⚠ `x/weg.py:9`" and ks._zellen(zeilen[4])[5] == "verdrahtet · v:? g:—"


def test_zelltext_pipe_wird_escaped():
    g = graph.Graph()
    _kette(g, "K1", "a | b", "z", "r")
    _hop(g, "K1", "1", belegt="008 | 009", zustand_roh="läuft")
    z = rk.uebersicht_rendern(g).splitlines()[2]
    assert ks._zellen(z)[1] == r"a \| b" and len(ks._zellen(z)) == 5
    h = rk.hops_rendern(g, "K1").splitlines()[2]
    assert len(ks._zellen(h)) == 6 and ks._zellen(h)[4] == r"008 \| 009"


def test_gerenderte_hop_tabelle_liest_sich_wieder_wie_das_original():
    g = _graph()
    aus = rk.einbetten(_text(), g)
    vorher = ks.hop_tabellen_lesen(_text())
    nachher = ks.hop_tabellen_lesen(aus)
    assert [(z.kette, z.nr, z.von_nach_roh, z.mechanismus, z.belegt_durch, z.zustand) for z in nachher] == \
           [(z.kette, z.nr, z.von_nach_roh, z.mechanismus, z.belegt_durch, z.zustand) for z in vorher]
    assert ks.uebersicht_lesen(aus) == ks.uebersicht_lesen(_text())


def test_kette_ohne_block_und_block_ohne_kette_sind_pruefpunkte():
    g = _graph()
    aus = rk.einbetten(_text(), g)
    _kette(g, "K3", "Drei", "x", "y")
    ohne_k2 = copy.deepcopy(g)
    del ohne_k2.knoten["kette:K2"]
    for eid in [e for e, k in ohne_k2.kanten.items() if k.attribute["kette"] == "K2"]:
        del ohne_k2.kanten[eid]
    erg = rk.rendern(aus, ohne_k2, {})
    assert sorted((p.art, p.aussage_id) for p in erg.pruefpunkte) == [
        ("block_ohne_kette", "hops K2"), ("kette_ohne_block", "hops K3")]
    assert rk.bloecke_finden(erg.text)["hops K2"].inhalt == rk.bloecke_finden(aus)["hops K2"].inhalt


def test_kaputte_marker_werfen_nicht():
    text = "a\n<!-- graph:begin hops K1 -->\n| x |\n\n<!-- graph:end -->\n<!-- graph:begin uebersicht -->\nb\n"
    assert set(rk.bloecke_finden(text)) == {"hops K1"}
    erg = rk.rendern(text, _graph(), {})
    assert "marker_kaputt" in {p.art for p in erg.pruefpunkte}
    assert erg.text.endswith("<!-- graph:begin uebersicht -->\nb\n")


# ---------- echte KETTEN.md ----------

@pytest.fixture(scope="module")
def echtes_repo():
    if not KETTEN.is_file() or not (REPO / "vibemind-os/spaces/marketing").is_dir() \
            or not korpus.manifest_verfuegbar():
        pytest.skip("Submodul/Manifest fehlt")
    from selbstbild.doku.extraktoren import sql
    konfig = vm_laden()
    commit_sub = korpus.commit_von(konfig.repo / "vibemind-os")
    manifeste = {n: korpus.manifest_bauen(s, konfig, commit_sub) for n, s in konfig.spaces.items()}
    defs, _ = sql.sql_definitionen(konfig)
    g, _ = ks.seed_graph(konfig, manifeste, defs, korpus.commit_von(konfig.repo), commit_sub)
    return SimpleNamespace(konfig=konfig, manifeste=manifeste, defs=defs, graph=g,
                           commits=(korpus.commit_von(konfig.repo), commit_sub))


@pytest.fixture(scope="module")
def echter_graph(echtes_repo):
    return echtes_repo.graph


def test_echte_ketten_md_einbetten_idempotent_prosa_byteweise(echter_graph):
    text = KETTEN.read_bytes().decode("utf-8")
    if "<!-- graph:begin" in text:
        pytest.skip("KETTEN.md ist bereits gerendert")
    aus = rk.einbetten(text, echter_graph)
    bloecke = rk.bloecke_finden(aus)
    assert len(bloecke) == 19 and "uebersicht" in bloecke
    assert {f"hops K{i}" for i in range(1, 19)} <= set(bloecke)
    assert _ohne_bloecke(aus) == _ohne_tabellen_im_bereich(text)
    assert rk.einbetten(aus, echter_graph) == aus
    vorher = ks.hop_tabellen_lesen(text)
    nachher = ks.hop_tabellen_lesen(aus)
    assert [(z.kette, z.nr, z.mechanismus, z.belegt_durch, z.zustand) for z in nachher] == \
           [(z.kette, z.nr, z.mechanismus, z.belegt_durch, z.zustand) for z in vorher]
    erg1 = rk.rendern(aus, echter_graph, {})
    assert erg1.text == aus and erg1.pruefpunkte == []
    erg2 = rk.rendern(erg1.text, echter_graph, erg1.hashes)
    assert erg2.text == aus and erg2.hashes == erg1.hashes and erg2.pruefpunkte == []


def _ohne_tabellen_im_bereich(text):
    """Original ohne die 19 ersetzten Tabellen (Uebersicht + Hop-Tabellen); andere Tabellen bleiben."""
    ersetzt = set()
    for b in rk.tabellen_finden(text):
        ersetzt.update(range(b[1], b[2]))
    return "".join(z for i, z in enumerate(text.splitlines(keepends=True)) if i not in ersetzt)


def test_tabellen_finden_echte_datei():
    if not KETTEN.is_file():
        pytest.skip("KETTEN.md fehlt")
    text = KETTEN.read_bytes().decode("utf-8")
    if "<!-- graph:begin" in text:
        pytest.skip("KETTEN.md ist bereits gerendert")
    gefunden = rk.tabellen_finden(text)
    assert [b[0] for b in gefunden] == ["uebersicht", *[f"hops K{i}" for i in range(1, 19)]]
    zeilen = text.splitlines()
    for block, start, ende in gefunden:
        assert all(zeilen[i].lstrip().startswith("|") for i in range(start, ende))
        assert not zeilen[ende].lstrip().startswith("|")


def test_zeitformat_berlin():
    assert rk.zeit_kurz("2026-10-27T13:25:00+00:00") == "27.10. 14:25"
    assert rk.zeit_kurz("2026-07-01T10:00:00Z") == "01.07. 12:00"
    assert rk.zeit_kurz("kaputt") == "" and rk.zeit_kurz(None) == ""


def test_marker_format():
    assert rk.MARKER_BEGIN.format(block="hops K1") == "<!-- graph:begin hops K1 -->"
    assert rk.MARKER_END == "<!-- graph:end -->"
    assert re.fullmatch(r"[0-9a-f]{64}", rk.rendern(rk.einbetten(_text(), _graph()), _graph(), {}).hashes["uebersicht"])


def test_schon_markierter_block_wird_nicht_doppelt_angelegt():
    g = _graph()
    aus = rk.einbetten(_text(nl="\n", bom=False), g)
    nachgetragen = "\n".join([aus, "", "### K1 · Eins nochmal", "", KOPF, TRENN,
                              "| 9 | X → Y | MCP | — | 1 | läuft |", ""])
    assert rk.einbetten(nachgetragen, g) == nachgetragen


# ---------- Fix-Runde 1 ----------

def _stellen_vergleich(g):
    """Je Hop: Belegschluessel und die Stellen in Reihenfolge (verankert -> Belegschluessel, sonst volle Form)."""
    aus = {}
    for k in g.kanten.values():
        if k.art != "hop":
            continue
        folge = []
        for st in k.attribute["stellen_roh"]:
            eintrag = tuple(st["beleg"]) if st["beleg"] else ("⚠", st["voll"])
            if eintrag not in folge:
                folge.append(eintrag)
        aus[(k.attribute["kette"], k.attribute["nr"])] = (sorted((b.art, b.file, b.anker) for b in k.belege), folge)
    return aus


def _seed_repo(tmp_path):
    (tmp_path / "sp/m").mkdir(parents=True)
    (tmp_path / "sp/m/api.py").write_text(
        "def routen():\n    return ziel_aufruf()\n\n\ndef andere():\n    x = 1\n    return x\n", encoding="utf-8")
    konfig = dataclasses.replace(vm_laden(), repo=tmp_path,
                                 spaces={"m": SpaceKonfig("m", "sp/m", "werkzeuge")}, ketten_md="docs/KETTEN.md")
    docs = (SimpleNamespace(document_id="m.api.routen", file="api.py", start_line=1, end_line=2, symbol="routen",
                            source="def routen():\n    return ziel_aufruf()"),
            SimpleNamespace(document_id="m.api.andere", file="api.py", start_line=5, end_line=7, symbol="andere",
                            source="def andere():\n    x = 1\n    return x"))
    return konfig, {"m": SimpleNamespace(corpus_root=str(tmp_path / "sp/m"), documents=docs)}


def test_code_stelle_rundreise_anker_bestaetigt_keine_andere_stelle(tmp_path):
    konfig, manifeste = _seed_repo(tmp_path)
    md = "\r\n".join(["# K", "", "### K2 · Zwei", "", KOPF, TRENN,
                       "| 1 | A → B | Python `ziel_aufruf` | `sp/m/api.py:2`, `:7`, `043…sql:297` (laut `doc_008`) "
                       "| 015 | läuft |",
                       "| 2 | C → D | SQL | — (laut Doc 008, nicht nachgeprüft) | 015 | läuft |", ""])
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/KETTEN.md").write_bytes(md.encode("utf-8"))
    g1, _ = ks.seed_graph(konfig, manifeste, {}, "a", "b")
    aus = rk.einbetten(md, g1)
    zeile = next(z for z in aus.split("\r\n") if z.startswith("| 1 |"))
    assert ks._zellen(zeile)[3] == ("`sp/m/api.py:2` `m.api.routen`, ⚠ `sp/m/api.py:7`, ⚠ `043…sql:297`"
                                    " — (laut `doc_008`)")
    zeile2 = next(z for z in aus.split("\r\n") if z.startswith("| 2 |"))
    assert ks._zellen(zeile2)[3] == "— (laut Doc 008, nicht nachgeprüft)"  # Menschentext unveraendert
    gelesen = ks.hop_tabellen_lesen(aus)[0]
    assert gelesen.code_stellen == ("sp/m/api.py:2", "sp/m/api.py:7", "043…sql:297", "doc_008")
    assert ks.zeilen_namen(gelesen) == ks.zeilen_namen(ks.hop_tabellen_lesen(md)[0])
    assert gelesen.code_anker == ("m.api.routen",)
    assert "routen" not in ks.zeilen_namen(gelesen) and "m.api.routen" not in ks.zeilen_tokens(gelesen)
    (tmp_path / "docs/KETTEN.md").write_bytes(aus.encode("utf-8"))
    g2, _ = ks.seed_graph(konfig, manifeste, {}, "a", "b")
    assert _stellen_vergleich(g2) == _stellen_vergleich(g1)
    assert [k.attribute["code_prosa"] for k in sorted(g2.kanten.values(), key=lambda k: k.attribute["nr"])] == \
           ["(laut `doc_008`)", "(laut Doc 008, nicht nachgeprüft)"]
    assert rk.rendern(aus, g2, {}).text == aus  # der ⚠ und der Menschentext bleiben nach dem Wiederlesen


def test_belege_auffrischen_verschobenes_symbol_neue_zeile(tmp_path):
    konfig, manifeste = _seed_repo(tmp_path)
    (tmp_path / "werkzeug").mkdir()
    (tmp_path / "werkzeug/hilfe.py").write_text("\n\n\ndef hilfe():\n    return 1\n", encoding="utf-8")
    (tmp_path / "caps.yaml").write_text("# Kopf\n\n- capability: a\n  target: x\n- capability: b\n  target: y\n",
                                        encoding="utf-8")
    g = graph.Graph()
    _kette(g, "K1", "T", "z", "r")
    belege = [Beleg("symbol", "sp/m/api.py", "m.api.andere", "d", "c", 6),  # im Space, steht jetzt bei 9
              Beleg("symbol", "werkzeug/hilfe.py", "werkzeug.hilfe.hilfe", "d", "c", 1),  # AST: 4
              Beleg("sql", "db/1.sql", "marketing.fn", "d", "c", 3),  # SqlDefinition: 12
              Beleg("yaml_key", "caps.yaml", "[capability=b].target", "d", "c", 2),  # Schluesselzeile 6
              Beleg("datei", "sp/m/api.py", "", "d", "c", 4),  # bleibt
              Beleg("symbol", "sp/m/api.py", "m.api.weg", "d", "c", 11)]  # nicht aufloesbar: bleibt
    _hop(g, "K1", "1", belege=belege,
         stellen=[{"roh": "sp/m/api.py:6", "voll": "sp/m/api.py:6",
                   "beleg": ["symbol", "sp/m/api.py", "m.api.andere"]}])
    docs = manifeste["m"].documents
    andere_neu = SimpleNamespace(**{**vars(docs[1]), "start_line": 9, "end_line": 11})  # Inhalt gleich
    verschoben = {"m": SimpleNamespace(corpus_root=manifeste["m"].corpus_root, documents=(docs[0], andere_neu))}
    sql_defs = {"marketing.fn": SqlDefinition("db/1.sql", 12, "marketing.fn", "function", "d")}
    vorher = rk.hops_rendern(g, "K1")
    neu = rk.belege_auffrischen(g, verschoben, konfig, sql_defs)
    zeilen = {(b.art, b.anker): b.zeile for b in next(iter(neu.kanten.values())).belege}
    assert zeilen == {("symbol", "m.api.andere"): 9, ("symbol", "werkzeug.hilfe.hilfe"): 4,
                      ("sql", "marketing.fn"): 12, ("yaml_key", "[capability=b].target"): 6,
                      ("datei", ""): 4, ("symbol", "m.api.weg"): 11}
    assert next(iter(g.kanten.values())).belege[0].zeile == 6  # Original unveraendert (Kopie)
    assert "`sp/m/api.py:6` `m.api.andere`" in vorher
    assert "`sp/m/api.py:9` `m.api.andere`" in rk.hops_rendern(neu, "K1")


def test_zeilenenden_wechsel_und_graphaenderung_kein_falscher_hand_edit():
    g = _graph()
    erg1 = rk.rendern(rk.einbetten(_text(), g), g, {})
    lf = erg1.text.replace("\r\n", "\n").replace("|\n", "|   \n")  # Formatierer: LF + Leerraum am Ende
    g2 = copy.deepcopy(g)
    for k in g2.kanten.values():
        k.attribute["belegt_durch"] = "042"
    erg2 = rk.rendern(lf, g2, erg1.hashes)
    assert erg2.pruefpunkte == []
    assert "| 042 |" in rk.bloecke_finden(erg2.text)["hops K1"].inhalt


def test_messzeit_steht_nie_im_markdown():
    """Ruling 45 (C.9.5): eine neue Messung ohne Codeaenderung aendert den gerenderten Text nicht."""
    def bauen(zeit):
        g = graph.Graph()
        _kette(g, "K1", "T", "z", "r")
        _hop(g, "K1", "1", verdrahtet="ja", gemessen={"wert": "laeuft", "zeit": zeit})
        return g

    frueh, spaet = bauen("2026-10-27T13:25:00+00:00"), bauen("2026-10-28T09:00:00+00:00")
    assert rk.kette_zusammenfassung(frueh, "K1") == "🟢 verdrahtet 1/1 · gemessen läuft"
    assert rk.uebersicht_rendern(frueh) == rk.uebersicht_rendern(spaet)
    assert rk.hops_rendern(frueh, "K1") == rk.hops_rendern(spaet, "K1")
    assert not re.search(r"\d\d\.\d\d\. \d\d:\d\d", rk.hops_rendern(spaet, "K1") + rk.uebersicht_rendern(spaet))


def test_echte_ketten_md_rundreise_mit_code_stelle(echtes_repo, tmp_path):
    text = KETTEN.read_bytes().decode("utf-8")
    if "<!-- graph:begin" in text:
        pytest.skip("KETTEN.md ist bereits gerendert")
    r = echtes_repo
    frisch = rk.belege_auffrischen(r.graph, r.manifeste, r.konfig, r.defs)
    aus = rk.einbetten(text, frisch)
    assert all(z.code_anker == () for z in ks.hop_tabellen_lesen(text))  # Original: keine gerenderten Paare
    gelesen = ks.hop_tabellen_lesen(aus)
    anker_ = {b.anker for k in frisch.kanten.values() for b in k.belege if b.anker}
    assert sum(len(z.code_anker) for z in gelesen) > 100
    for z in gelesen:  # gerenderte Anker landen nie in den Stellen, und erkannt wird nur, was ein Anker ist
        assert not set(z.code_stellen) & anker_, (z.kette, z.nr, z.code_stellen)
        assert set(z.code_anker) <= anker_, (z.kette, z.nr, z.code_anker)
    ziel = tmp_path / "KETTEN.md"
    ziel.write_bytes(aus.encode("utf-8"))
    g2, _ = ks.seed_graph(dataclasses.replace(r.konfig, ketten_md=str(ziel)), r.manifeste, r.defs, *r.commits)
    alt, neu = _stellen_vergleich(r.graph), _stellen_vergleich(g2)
    assert set(neu) == set(alt)
    abweichend = {k: (alt[k], neu[k]) for k in alt if alt[k] != neu[k]}
    assert abweichend == {}
    prosa = {(k.attribute["kette"], k.attribute["nr"]): k.attribute["code_prosa"]
             for g_ in (r.graph,) for k in g_.kanten.values() if k.art == "hop"}
    prosa2 = {(k.attribute["kette"], k.attribute["nr"]): k.attribute["code_prosa"]
              for k in g2.kanten.values() if k.art == "hop"}
    assert prosa2 == prosa and sum(1 for p in prosa.values() if p) >= 10
    assert rk.rendern(aus, rk.belege_auffrischen(g2, r.manifeste, r.konfig, r.defs), {}).text == aus
