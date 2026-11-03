"""Stufe 4, Task 5 (D8): Space-Events als generierter, deterministischer Doku-Block (kein LLM).

Quellen: die Registry (`spaces.<id>.events`, Beleg `yaml_key`) und Python-Tabellen `EVENT_TO_TOOL` (Beleg `symbol`).
Abgleich: Abweichungen zwischen den Quellen, Event-Namen nur in der Doku, Spaces ohne Doku-Datei.
"""
from __future__ import annotations

import dataclasses
import difflib
import shutil

from doku_vm import VORLAGE
from test_doku_tick import (  # noqa: F401  (Fixture `fx`)
    FakeLaufzeit, _anstoss, _commit, _git, _head, _pruefliste, _schreib, _seed, _status, _tick, fx,
)
from selbstbild.doku import konfig as konfig_modul
from selbstbild.doku.extraktoren.ereignisse import (
    Ereignis, abgleich, ereignisse_lesen, space_dokumente_zuordnen,
)
from selbstbild.doku.graph import Graph, schlank
from selbstbild.doku.namen import space_schluessel
from selbstbild.doku.render_ereignisse import UEBERSCHRIFT, doku_rendern, ereignisse_rendern

BOM = "﻿"
GRAPH = "docs/graph/graph.json"
REGISTRY = "vibemind-os/config/reg.yml"
AGENT = "vibemind-os/spaces/video/agents/video_agent.py"
QUELLEN = ({"yaml": REGISTRY, "pfad": "spaces.*.events", "werkzeug": "tool"},
           {"python_dict": "EVENT_TO_TOOL", "wurzel": "vibemind-os/spaces"})

REG_VIDEO = """\
version: 1
spaces:
  video:
    prefixes: [video.]
    events:
      video.status: { tool: video_status, required_params: [] }
      video.vision: { tool: db_query, required_params: [geheim_param] }
"""
AGENT_VIDEO = '''\
from typing import Dict


class VideoBackendAgent:
    """Agent."""

    EVENT_TO_TOOL: Dict[str, str] = {
        "video.status": "video_status",
        "video.vision": "vision_generate",
        "video.publish": "publish_videos_to_rowboat",
    }

    def werkzeug(self, ereignis):
        return self.EVENT_TO_TOOL.get(ereignis)
'''


def _konfig(wurzel, quellen=QUELLEN, **mehr):
    (wurzel / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(VORLAGE, wurzel / "config" / "doku_loop.yaml")
    k = konfig_modul.laden(wurzel / "config" / "doku_loop.yaml")
    return dataclasses.replace(k, ereignisse=tuple(quellen), **mehr)


def _video(tmp_path, reg=REG_VIDEO, agent=AGENT_VIDEO):
    _schreib(tmp_path, REGISTRY, reg)
    if agent is not None:
        _schreib(tmp_path, AGENT, agent)
    return _konfig(tmp_path)


# ---------- Step 1 ----------

def test_registry_und_dict_zusammen(tmp_path):
    k = _video(tmp_path)
    evs = ereignisse_lesen(k, tmp_path)
    assert len(evs) == 4  # status in beiden gleich -> einmal; vision zweimal (verschiedene Werkzeuge); publish
    assert [(e.space, e.name, e.werkzeug) for e in evs] == [
        ("video", "video.publish", "publish_videos_to_rowboat"), ("video", "video.status", "video_status"),
        ("video", "video.vision", "db_query"), ("video", "video.vision", "vision_generate")]
    status = evs[1]
    assert status.beleg.art == "yaml_key" and status.beleg.file == REGISTRY
    assert status.beleg.anker == 'spaces.video.events["video.status"]' and status.beleg.zeile == 6
    assert status.beleg.commit == ""  # keine Commits in diesen Knoten (stabiler Digest)
    assert [b.art for _, b in status.weitere] == ["symbol"]  # dieselbe Zeile auch in der Tabelle
    publish = evs[0]
    assert publish.beleg.art == "symbol" and publish.beleg.file == AGENT
    assert publish.beleg.anker == "vibemind-os.spaces.video.agents.video_agent.VideoBackendAgent.EVENT_TO_TOOL"
    assert publish.beleg.zeile == 10 and publish.beleg.commit == ""
    assert ereignisse_lesen(k, tmp_path) == evs  # deterministisch
    pp = abgleich(evs, {}, {"video": "docs/spaces/018_video.md"})
    assert sorted(p.art for p in pp) == ["ereignis_quellen_abweichend"] * 2
    texte = " ".join(p.text for p in pp)
    assert "video.vision" in texte and "video.publish" in texte and "video.status" not in texte
    assert "geheim_param" not in texte and "required_params" not in texte  # nur Namen, nie Inhalte


def test_nur_eine_quelle_je_space_keine_abweichung(tmp_path):
    k = _video(tmp_path, agent=None)
    assert abgleich(ereignisse_lesen(k, tmp_path), {}, {"video": "docs/spaces/018_video.md"}) == []


def test_dict_nur_modul_und_klassenebene_und_keine_tests(tmp_path):
    _schreib(tmp_path, "vibemind-os/spaces/schedule/agent.py",
             'EVENT_TO_TOOL = {"schedule.create": "anlegen"}\n\n\ndef f():\n'
             '    EVENT_TO_TOOL = {"schedule.lokal": "nein"}\n    return EVENT_TO_TOOL\n')
    _schreib(tmp_path, "vibemind-os/spaces/schedule/tests/test_agent.py", 'EVENT_TO_TOOL = {"schedule.test": "x"}\n')
    _schreib(tmp_path, "vibemind-os/spaces/schedule/kaputt.py", 'EVENT_TO_TOOL = {"schedule.x": \n')
    _schreib(tmp_path, "vibemind-os/spaces/schedule/node_modules/m.py", 'EVENT_TO_TOOL = {"schedule.nm": "x"}\n')
    k = _konfig(tmp_path, quellen=QUELLEN[1:])
    pp = []
    evs = ereignisse_lesen(k, tmp_path, pruefpunkte=pp)
    assert [(e.space, e.name, e.werkzeug) for e in evs] == [("schedule", "schedule.create", "anlegen")]
    assert evs[0].beleg.anker == "vibemind-os.spaces.schedule.agent.EVENT_TO_TOOL"
    assert [p.art for p in pp] == ["ereignis_quelle_unlesbar"] and "kaputt.py" in pp[0].datei


# ---------- Step 2 ----------

REG_DREI = REG_VIDEO + """\
  desktop:
    prefixes: [desktop.]
    events:
      desktop.open: { tool: open_app }
  n8n:
    prefixes: [n8n., web.]
    events:
      n8n.run: { tool: n8n_run }
"""


def test_nur_doku(tmp_path):
    k = _video(tmp_path, reg=REG_DREI, agent=None)
    praefixe: set[str] = set()
    evs = ereignisse_lesen(k, tmp_path, praefixe=praefixe)
    assert praefixe == {"video", "desktop", "n8n", "web"}
    doku = ("# Video\n\nDie Spec nennt `video.reel` und `video.status`; die Konfiguration steht in `desktop.json`,\n"
            "`video.yml` und `n8n.py`, der Ordner `spaces/video.reel/x` ist ein Pfad. Unbekannt: `foo.bar`.\n"
            "Der Crawler `web.crawl` laeuft ueber n8n, `video.reel` steht doppelt, `video.reel.alt` ist tiefer.\n")
    pp = abgleich(evs, {"docs/spaces/018_video.md": doku},
                  {"video": "docs/spaces/018_video.md", "desktop": "docs/spaces/004_desktop.md"}, praefixe=praefixe)
    nur = [p for p in pp if p.art == "ereignis_nur_doku"]
    assert sorted(p.aussage_id for p in nur) == ["docs/spaces/018_video.md#video.reel",
                                                 "docs/spaces/018_video.md#web.crawl"]
    assert all(p.datei == "docs/spaces/018_video.md" for p in nur)
    assert next(p for p in nur if "video.reel" in p.text).zeile == 3
    # n8n hat keine Doku-Datei; desktop ist zugeordnet (die Datei muss dafuer nicht gelesen sein)
    assert [p.aussage_id for p in pp if p.art == "space_ohne_doku"] == ["space:n8n"]
    # echte Falschtreffer: Operation `supabase:video.move`, Datei `video.db`, SQL-Objekte (Schema = Praefix)
    doku2 = "Op `supabase:video.move`, Datei `video.db`, Tabelle `video.jobs`, Funktion `video.job_anlegen`.\n"
    pp2 = abgleich(evs, {"docs/spaces/018_video.md": doku2}, {"video": "docs/spaces/018_video.md"},
                   praefixe=praefixe, nicht_ereignisse={"video.jobs", "video.job_anlegen", "public.x"})
    assert [p for p in pp2 if p.art == "ereignis_nur_doku"] == []


# ---------- Step 3 ----------

def _evs(tmp_path):
    return ereignisse_lesen(_video(tmp_path), tmp_path)


def test_rendern_tabelle(tmp_path):
    text = ereignisse_rendern("video", _evs(tmp_path))
    zeilen = text.split("\n")
    assert zeilen[0] == "| Event | Werkzeug | Quelle | Beleg |"
    assert [z.split("|")[1].strip() for z in zeilen[2:-1]] == ["`video.publish`", "`video.status`", "`video.vision`"]
    vision = zeilen[4]
    assert "⚠" in vision and "`db_query`" in vision and "`vision_generate`" in vision
    assert "⚠" not in zeilen[3] and f"`{REGISTRY}:6`" in zeilen[3] and f"`{AGENT}:8`" in zeilen[3]
    assert text.endswith("\n") and "\r" not in text
    assert ereignisse_rendern("anders", _evs(tmp_path)) == "| Event | Werkzeug | Quelle | Beleg |\n|---|---|---|---|\n"


def test_block_einfuegen_crlf_bom(tmp_path):
    block = ereignisse_rendern("video", _evs(tmp_path))
    alt = BOM + "# Video\r\n\r\n## Was hier passiert\r\n\r\nText.\r\n\r\n## Belege\r\n\r\n- `x.py`\r\n"
    res = doku_rendern(alt, {"ereignisse video": block}, {}, datei="docs/spaces/018_video.md")
    neu = res.text
    assert neu.startswith(BOM) and neu.count("\n") == neu.count("\r\n")
    a, b = alt.splitlines(keepends=True), neu.splitlines(keepends=True)
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    assert {op for op, *_ in ops} == {"equal", "insert"}  # nur neue Zeilen, nichts geaendert oder entfernt
    eingefuegt = [z for op, _, _, j1, j2 in ops if op == "insert" for z in b[j1:j2]]
    assert eingefuegt[0] == UEBERSCHRIFT + "\r\n" and eingefuegt[2] == "<!-- graph:begin ereignisse video -->\r\n"
    assert neu.index(UEBERSCHRIFT) < neu.index("## Belege")
    assert res.hashes == {"ereignisse video": res.hashes["ereignisse video"]} and res.pruefpunkte == []
    # idempotent
    zweit = doku_rendern(neu, {"ereignisse video": block}, res.hashes, datei="docs/spaces/018_video.md")
    assert zweit.text == neu and zweit.hashes == res.hashes and zweit.pruefpunkte == []


def test_block_ohne_belege_ans_ende_und_weiterer_space(tmp_path):
    block = ereignisse_rendern("video", _evs(tmp_path))
    alt = "# Ideen\n\nText ohne Zeilenende am Schluss"
    res = doku_rendern(alt, {"ereignisse video": block}, {})
    assert res.text.startswith(alt + "\n\n" + UEBERSCHRIFT + "\n\n<!-- graph:begin ereignisse video -->\n")
    assert res.text.endswith("<!-- graph:end -->\n")
    # ein zweiter Space derselben Datei kommt hinter den vorhandenen Block, keine zweite Ueberschrift
    res2 = doku_rendern(res.text, {"ereignisse video": block, "ereignisse bubbles": block}, res.hashes)
    assert res2.text.count(UEBERSCHRIFT) == 1
    assert res2.text.index("ereignisse video -->") < res2.text.index("ereignisse bubbles -->")
    assert set(res2.hashes) == {"ereignisse video", "ereignisse bubbles"}
    # ohne Bloecke und ohne Marker bleibt die Datei unberuehrt
    assert doku_rendern(alt, {}, {}).text == alt and doku_rendern(alt, {}, {}).pruefpunkte == []


# ---------- Step 4 ----------

def test_block_hand_edit(tmp_path):
    block = ereignisse_rendern("video", _evs(tmp_path))
    erst = doku_rendern("# V\n\n## Belege\n\n- `x.py`\n", {"ereignisse video": block}, {})
    von_hand = erst.text.replace("`db_query`", "`mein_werkzeug`")
    res = doku_rendern(von_hand, {"ereignisse video": block}, erst.hashes, datei="docs/spaces/018_video.md")
    assert res.text == von_hand  # Inhalt bleibt
    assert [p.art for p in res.pruefpunkte] == ["hand_edit"] and res.pruefpunkte[0].aussage_id == "ereignisse video"
    assert res.hashes == erst.hashes  # alter Hash bleibt: der Hand-Edit bleibt auch spaeter stehen
    # ohne gespeicherten Hash wird ueberschrieben; ein verwaister Block bleibt mit Pruefpunkt stehen
    assert doku_rendern(von_hand, {"ereignisse video": block}, {}).text == erst.text
    weg = doku_rendern(erst.text, {}, erst.hashes)
    assert weg.text == erst.text and [p.art for p in weg.pruefpunkte] == ["block_ohne_quelle"]


def test_block_von_hand_entfernt_wird_nicht_wieder_eingefuegt(tmp_path):
    """Wie bei KETTEN.md: wer die Marker entfernt, dem gehoert der Text; nur der Erstlauf fuegt ein."""
    block = ereignisse_rendern("video", _evs(tmp_path))
    erst = doku_rendern("# V\n\n## Belege\n\n- `x.py`\n", {"ereignisse video": block}, {})
    ohne = "# V\n\nEigene Tabelle.\n\n## Belege\n\n- `x.py`\n"
    res = doku_rendern(ohne, {"ereignisse video": block}, erst.hashes)
    assert res.text == ohne and [p.art for p in res.pruefpunkte] == ["block_entfernt"]
    assert res.hashes == erst.hashes


# ---------- Step 5 ----------

def test_namensschreibweisen(tmp_path):
    assert space_schluessel("sales-claw") == space_schluessel("sales_claw") == space_schluessel("Sales_Claw") \
        == "sales_claw"
    dateien = ["docs/spaces/KETTEN.md", "docs/spaces/015_sales_claw.md", "docs/spaces/006_ideas.md",
               "docs/spaces/018_video.md", "docs/modules/01_brain.md", "docs/spaces/n8n/contract.md"]
    zuordnung = space_dokumente_zuordnen(["sales-claw", "bubbles", "n8n", "video", "ideas", "brain"],
                                         {"bubbles": "docs/spaces/006_ideas.md"}, dateien)
    assert zuordnung == {"sales_claw": "docs/spaces/015_sales_claw.md", "bubbles": "docs/spaces/006_ideas.md",
                         "video": "docs/spaces/018_video.md", "ideas": "docs/spaces/006_ideas.md"}
    # zwei Schreibweisen in zwei Quellen sind ein Space und ein Block
    _schreib(tmp_path, REGISTRY, "spaces:\n  sales_claw:\n    events:\n      sales.mail: { tool: mail }\n")
    _schreib(tmp_path, "vibemind-os/spaces/sales-claw/agent.py", 'EVENT_TO_TOOL = {"sales.mail": "mail"}\n')
    evs = ereignisse_lesen(_konfig(tmp_path), tmp_path)
    assert len(evs) == 1 and len(evs[0].weitere) == 1
    assert {space_schluessel(e.space) for e in evs} == {"sales_claw"}
    assert abgleich(evs, {}, zuordnung) == []
    assert "`sales.mail`" in ereignisse_rendern("sales-claw", evs)


def test_ereignis_ist_wertgleich():
    from selbstbild.doku.anker import Beleg
    b = Beleg("yaml_key", "r.yml", 'spaces.a.events["a.b"]', "d", "", 3)
    assert Ereignis("a", "a.b", "t", "r.yml", b) == Ereignis("a", "a.b", "t", "r.yml", b)


# ---------- Konfiguration ----------

def test_konfig_ereignisse_liste(tmp_path):
    from test_doku_konfig_regeln import _repo, _vorlage

    from selbstbild.doku.konfig import aus_regeln
    t = _vorlage() + ('ereignisse:\n  - {yaml: "vibemind-os/config/x.yml", pfad: "spaces.*.events", werkzeug: tool}\n'
                      '  - {python_dict: EVENT_TO_TOOL, wurzel: vibemind-os/spaces}\n'
                      'space_dokumente: {bubbles: docs/spaces/006_ideas.md}\n')
    k = aus_regeln(_repo(tmp_path, t))
    assert k.ereignisse == ({"yaml": "vibemind-os/config/x.yml", "pfad": "spaces.*.events", "werkzeug": "tool"},
                            {"python_dict": "EVENT_TO_TOOL", "wurzel": "vibemind-os/spaces"})
    assert k.space_dokumente == {"bubbles": "docs/spaces/006_ideas.md"}


# ---------- Tick: Block, Graph, Hand-Edit, Messbaum ----------

REG_A = """\
version: 1
spaces:
  a:
    prefixes: [a.]
    events:
      a.start: { tool: starten }
      a.stopp: { tool: stoppen }
"""
DOKU_A = ("# A\n\n## Was hier passiert\n\nSpace a startet `a.start` und kennt `a.fehlt`; die Tabelle `a.jobs` ist SQL.\n\n"
          "## Belege\n\n- `sp/a/client.py`\n")


def _tick_ereignisse(fx):
    fx.konfig = dataclasses.replace(fx.konfig, doku_ziele=("docs/spaces/0*.md",), ereignisse=(
        {"yaml": "config/reg.yml", "pfad": "spaces.*.events", "werkzeug": "tool"},))
    _seed(fx)
    _schreib(fx.repo, "config/reg.yml", REG_A)
    _schreib(fx.repo, "docs/spaces/001_a.md", DOKU_A)
    _schreib(fx.repo, "db/a.sql", "CREATE TABLE a.jobs (id int);\n")
    _commit(fx.repo, "registry und doku", "config/reg.yml", "docs/spaces/001_a.md", "db/a.sql")
    _anstoss(fx)
    erg = _tick(fx, erzwingen=True)
    assert erg.uebersprungen is None and erg.commit == _head(fx.repo)
    return erg


def _ereignis_graph(repo):
    g = Graph.laden(repo / GRAPH)
    return ({k.id: k for k in g.knoten.values() if k.typ == "ereignis"},
            {e.id: e for e in g.kanten.values() if e.art == "ereignis_von"}, g)


def test_tick_block_graph_und_pruefliste(fx):
    erg = _tick_ereignisse(fx)
    assert erg.llm_aufrufe == 0 and erg.signale == 0 and erg.ereignisse == 2
    text = (fx.repo / "docs/spaces/001_a.md").read_text("utf-8")
    assert "<!-- graph:begin ereignisse a -->" in text and text.index(UEBERSCHRIFT) < text.index("## Belege")
    assert text.startswith(DOKU_A.split("## Belege")[0])
    knoten, kanten, g = _ereignis_graph(fx.repo)
    assert sorted(knoten) == ["ereignis:a:a.start", "ereignis:a:a.stopp"]
    assert sorted(kanten) == ["ereignis_von:ereignis:a:a.start->space:a", "ereignis_von:ereignis:a:a.stopp->space:a"]
    assert all(b.commit == "" for e in kanten.values() for b in e.belege)
    assert schlank(g).digest_stabil() == g.digest_stabil()
    offen_struktur = _pruefliste(fx)
    nur = [p for p in offen_struktur["struktur"] if p["art"] == "ereignis_nur_doku"]
    assert [p["aussage_id"] for p in nur] == ["docs/spaces/001_a.md#a.fehlt"]
    # der Block selbst traegt keine Aussagen (keine gemerkten Belege auf die Registry)
    assert not [k for k in g.knoten.values() if k.typ == "aussage" and "Event" in k.name]
    # zweiter Lauf ohne Aenderung: kein Commit
    _anstoss(fx)
    assert _tick(fx, erzwingen=True).commit is None and _status(fx.repo) == []


def test_tick_registry_aenderung_nur_der_block_kein_judge(fx):
    _tick_ereignisse(fx)
    vorher = (fx.repo / "docs/spaces/001_a.md").read_text("utf-8")
    _schreib(fx.repo, "config/reg.yml", REG_A.replace("tool: stoppen", "tool: anhalten"))
    _commit(fx.repo, "registry", "config/reg.yml")
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf)  # ohne erzwingen: die Registry liegt ausserhalb der Spaces
    assert erg.uebersprungen is None and erg.commit == _head(fx.repo)
    assert erg.llm_aufrufe == 0 and lauf.aufrufe == [] and erg.signale == 0
    nachher = (fx.repo / "docs/spaces/001_a.md").read_text("utf-8")
    geaendert = [z for z in difflib.ndiff(vorher.splitlines(), nachher.splitlines()) if z[:1] in "+-"]
    assert geaendert == [f"- {z}" for z in vorher.splitlines() if "`a.stopp`" in z] + \
        [f"+ {z}" for z in nachher.splitlines() if "`a.stopp`" in z]
    assert "`anhalten`" in nachher
    assert set(_git(fx.repo, "show", "--name-only", "--format=", "HEAD").split()) == {
        "docs/spaces/001_a.md", GRAPH}


def test_tick_hand_edit_bleibt(fx):
    _tick_ereignisse(fx)
    von_hand = (fx.repo / "docs/spaces/001_a.md").read_text("utf-8").replace("`starten`", "`starten (von Hand)`")
    _schreib(fx.repo, "docs/spaces/001_a.md", von_hand)
    _commit(fx.repo, "von Hand", "docs/spaces/001_a.md")
    _schreib(fx.repo, "config/reg.yml", REG_A.replace("tool: stoppen", "tool: anhalten"))
    _commit(fx.repo, "registry", "config/reg.yml")
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.llm_aufrufe == 0
    assert (fx.repo / "docs/spaces/001_a.md").read_text("utf-8") == von_hand
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "hand_edit"]
    assert len(punkte) == 1 and punkte[0]["datei"] == "docs/spaces/001_a.md"
    assert punkte[0]["aussage_id"].startswith("ereignisse a#")


def test_tick_liest_die_registry_aus_dem_messbaum(fx):
    """Ruling T5-a: eine nicht committete Aenderung der Quelle ist unsichtbar."""
    _tick_ereignisse(fx)
    vorher = (fx.repo / "docs/spaces/001_a.md").read_bytes()
    _schreib(fx.repo, "config/reg.yml", REG_A.replace("tool: stoppen", "tool: dreckig"))
    _anstoss(fx)
    erg = _tick(fx, erzwingen=True)
    assert erg.commit is None and (fx.repo / "docs/spaces/001_a.md").read_bytes() == vorher
    knoten, _, _ = _ereignis_graph(fx.repo)
    assert "dreckig" not in str(knoten["ereignis:a:a.stopp"].attribute)
