"""Stufe 4, Task 4: weitere Doku-Ziele (`doku_ziele`), Submodul-Doku nur lesend (`doku_nur_lesen`, D7, Ruling T4-a),
fremde Aenderung an einem Doku-Ziel blockiert den Tick nicht (Ruling T4-b), Torwaechter bei Ersetzungen in Space-Doku
und stabiler Digest beim Erstlauf.

Fixture wie `test_doku_tick.py`: git init unter tmp_path, Fake-Manifest-Bauer, Fake-Judge, Torwaechter-Stub.
"""

import dataclasses
import json
import re

from test_doku_anstoss_trocken import _alter_loop, _trocken, _unveraendert, _zustand
from test_doku_tick import (  # noqa: F401  (Fixture `fx`)
    AUSSAGE, CLIENT, KETTEN, NEUER_TEXT, FakeLaufzeit, _anstoss, _code_aenderung, _commit, _git, _head, _jetzt,
    _pruefliste, _schreib, _seed, _status, _submodul, _text, _tick, _umbenennen, _urteil, fx,
)
from selbstbild.doku import schreibvorgang as sv
from selbstbild.doku.graph import Graph, schlank

GRAPH = "docs/graph/graph.json"
WERKZEUG = "def pruefen():\n    return True\n"
README_SUB = ("# Doku-Paket\n\nDie Hilfe `helfer` in `sp/a/client.py:11` liefert immer die Eins an den Aufrufer "
              "zurueck.\n")
README_NEU = "Die Hilfe `helfer` in `sp/a/client.py:11` liefert immer die Zwei an den Aufrufer zurueck."


def _doku(titel: str, *absaetze: str, belege: str = "- `sp/a/client.py`") -> str:
    return "\n".join([f"# {titel}", "", "## Was hier passiert", "", *[z for a in absaetze for z in (a, "")],
                      "## Belege", "", belege, ""])


def _ziele(fx, **werte):
    fx.konfig = dataclasses.replace(fx.konfig, **werte)
    return fx.konfig


def _aussagen(repo, datei=None):
    g = Graph.laden(repo / GRAPH)
    return {k.id: k for k in g.knoten.values() if k.typ == "aussage" and (datei is None or k.datei == datei)}


def _belege_von(repo, knoten_id):
    g = Graph.laden(repo / GRAPH)
    return sorted((e.quelle, b.art, b.file, b.anker, b.digest) for e in g.kanten.values()
                  if e.art == "belegt_durch" and e.von == knoten_id for b in e.belege)


# ---------- doku_dateien ----------

def test_doku_dateien_reihenfolge_ausschluss_und_eigene(fx):
    for rel in ("docs/spaces/002_b.md", "docs/spaces/001_a.md", "docs/spaces/README.md", "docs/spaces/MASSNAHMEN.md",
                "docs/modules/01_m.md", "docs/modules/README.md", "docs/modules/MASSNAHMEN.md",
                "docs/spaces/agentfarm/contract.md", "docs/extra.md", "docs/spaces/003_nur.md"):
        _schreib(fx.repo, rel, "# x\n")
    k = _ziele(fx, weitere_doku=("docs/extra.md",),
               doku_ziele=("docs/spaces/0*.md", "docs/modules/*.md", "docs/spaces/0*.md", "docs/spaces/*.md"),
               doku_nur_lesen=("docs/spaces/003_nur.md",))
    assert sv.doku_dateien(k) == [KETTEN, "docs/extra.md", "docs/modules/01_m.md", "docs/spaces/001_a.md",
                                  "docs/spaces/002_b.md"]
    assert sv.eigene_datei(k, "docs/spaces/001_a.md") and sv.eigene_datei(k, "docs/modules/01_m.md")
    assert sv.eigene_datei(k, "docs/spaces/009_noch_nicht_da.md")  # Muster, nicht Dateisystem
    for nein in ("docs/spaces/README.md", "docs/modules/MASSNAHMEN.md", "docs/spaces/003_nur.md",
                 "docs/spaces/agentfarm/contract.md", "docs/spaces/0x/tief.md", "sp/a/client.py"):
        assert not sv.eigene_datei(k, nein), nein
    # ohne Ziele wie bisher; eine leere `ketten_md` (Vorgabe ohne Abschnitt) kommt nicht in die Liste
    assert sv.doku_dateien(dataclasses.replace(k, doku_ziele=(), doku_nur_lesen=())) == [KETTEN, "docs/extra.md"]
    assert sv.doku_dateien(dataclasses.replace(k, ketten_md="", weitere_doku=(), doku_ziele=())) == []


def test_fremde_aenderungen_nur_ketten_weitere_doku_und_graph(fx):
    """Ruling T4-b: die alte Pruefung (ganzer Tick wartet) umfasst die Doku-Ziele nicht; die eigene Liste schon."""
    _schreib(fx.repo, "docs/spaces/001_a.md", "# a\n")
    _schreib(fx.repo, "docs/spaces/002_b.md", "# b\n")
    _commit(fx.repo, "doku", "docs/spaces")
    k = _ziele(fx, doku_ziele=("docs/spaces/0*.md",))
    _schreib(fx.repo, "docs/spaces/001_a.md", "# a von Hand\n")
    _schreib(fx.repo, "docs/spaces/005_neu.md", "# neu, nicht versioniert\n")
    assert sv.fremde_aenderungen(k) == []
    assert sorted(sv.fremde_doku_ziele(k)) == ["docs/spaces/001_a.md", "docs/spaces/005_neu.md"]
    _schreib(fx.repo, KETTEN, _text(fx) + "von Hand\n")
    assert sv.fremde_aenderungen(k) == [KETTEN]


# ---------- Erstlauf (Step 4) und stabiler Digest (Ruling 5) ----------

def _zwanzig(fx):
    _schreib(fx.repo, "scripts/werkzeug.py", WERKZEUG)
    _commit(fx.repo, "werkzeug", "scripts/werkzeug.py")
    _ziele(fx, doku_ziele=("docs/spaces/0*.md", "docs/modules/0*.md"))
    _seed(fx)
    _anstoss(fx)
    _tick(fx)  # zieht den Stand hinter den Seed-Commit
    for i in range(1, 21):
        _schreib(fx.repo, f"docs/spaces/{i:03d}_s{i}.md", _doku(
            f"S{i}", f"Space {i} prueft seine Doku mit `pruefen` in `scripts/werkzeug.py:1` vor jedem Lauf.",
            belege="- `scripts/werkzeug.py`"))
    _schreib(fx.repo, "docs/spaces/README.md", "# Index\n\nDas Werkzeug `pruefen` in `scripts/werkzeug.py:1`.\n")
    _schreib(fx.repo, "docs/spaces/MASSNAHMEN.md", "# M\n\nDas Werkzeug `pruefen` in `scripts/werkzeug.py:1`.\n")
    _commit(fx.repo, "zwanzig Space-Dokumente", "docs/spaces")
    _anstoss(fx)


def test_erstlauf_keine_signale(fx):
    _zwanzig(fx)
    lauf = FakeLaufzeit()
    erg = _tick(fx, laufzeit=lauf)
    assert erg.llm_aufrufe == 0 and lauf.aufrufe == [] and erg.signale == 0 and erg.umgeschrieben == 0
    assert erg.uebersprungen is None and erg.commit == _head(fx.repo)
    assert erg.doku_gelesen == 21 and erg.aussagen_neu == 20  # KETTEN.md + 20 Ziele; README/MASSNAHMEN nie
    assert _namen_im_commit(fx) == {GRAPH}  # nur gemerkte Belege, keine Doku umgeschrieben
    for i in range(1, 21):
        datei = f"docs/spaces/{i:03d}_s{i}.md"
        knoten = _aussagen(fx.repo, datei)
        assert len(knoten) == 1, datei
        belege = _belege_von(fx.repo, next(iter(knoten)))
        assert [(q, art, f, a) for q, art, f, a, _ in belege] == [("extraktor", "symbol", "scripts/werkzeug.py",
                                                                    "scripts.werkzeug.pruefen")]
    assert not _aussagen(fx.repo, "docs/spaces/README.md") and not _aussagen(fx.repo, "docs/spaces/MASSNAHMEN.md")
    # der schlanke Graph traegt sie (Ruling 49: aussage-Knoten und belegt_durch-Kanten)
    g = Graph.laden(fx.repo / GRAPH)
    assert schlank(g).digest_stabil() == g.digest_stabil()
    # zweiter Lauf ohne Codeaenderung: kein Commit, kein LLM (kein Flattern)
    _anstoss(fx)
    zweiter = _tick(fx, laufzeit=lauf, erzwingen=True)
    assert zweiter.commit is None and zweiter.llm_aufrufe == 0 and zweiter.aussagen_neu == 0
    assert _status(fx.repo) == []


def _namen_im_commit(fx, rev="HEAD"):
    return set(_git(fx.repo, "show", "--name-only", "--format=", rev).split())


def test_zwei_trockenlaeufe_gleicher_digest(fx):
    _zwanzig(fx)
    _alter_loop(fx)
    vorher = _zustand(fx)
    erste, zweite = _trocken(fx), _trocken(fx)
    assert erste.uebersprungen is None and erste.signale == 0 and erste.llm_aufrufe == 0
    assert erste.aussagen_neu == 20 and erste.doku_gelesen == 21
    assert len(erste.graph_digest) == 64 and erste.graph_digest == zweite.graph_digest
    _unveraendert(fx, vorher)
    # der echte Tick schreibt genau den Graph, den der Trockenlauf gemessen hat
    erg = _tick(fx)
    assert erg.commit is not None
    assert Graph.laden(fx.repo / GRAPH).digest_stabil() == erste.graph_digest


# ---------- Ruling T4-b ----------

def _zwei_ziele(fx):
    """KETTEN.md + 001 + 002 tragen dieselbe Aussage (Beleg `senden`); Erstlauf merkt die Belege."""
    _ziele(fx, doku_ziele=("docs/spaces/0*.md",))
    _seed(fx)
    _schreib(fx.repo, "docs/spaces/001_x.md", _doku("X", AUSSAGE))
    _schreib(fx.repo, "docs/spaces/002_y.md", _doku("Y", AUSSAGE))
    _commit(fx.repo, "zwei Ziele", "docs/spaces")
    _anstoss(fx)
    erg = _tick(fx, erzwingen=True)
    assert erg.commit is not None and erg.llm_aufrufe == 0
    assert len(_aussagen(fx.repo, "docs/spaces/001_x.md")) == 1 == len(_aussagen(fx.repo, "docs/spaces/002_y.md"))


def test_fremdes_doku_ziel_blockiert_den_tick_nicht(fx):
    _zwei_ziele(fx)
    alt_001 = {aid: _belege_von(fx.repo, aid) for aid in _aussagen(fx.repo, "docs/spaces/001_x.md")}
    _umbenennen(fx)
    dreckig = _doku("X", AUSSAGE, "Notiz einer anderen Session, noch nicht committet.").encode("utf-8")
    (fx.repo / "docs/spaces/001_x.md").write_bytes(dreckig)
    _anstoss(fx)
    lauf = FakeLaufzeit(*[_urteil()] * 4)  # KETTEN.md und 002 je zweimal; 001 wird nicht gefragt
    erg = _tick(fx, laufzeit=lauf)
    assert erg.uebersprungen is None and erg.commit == _head(fx.repo)
    assert erg.llm_aufrufe == 4 and erg.umgeschrieben == 2
    assert erg.doku_uebersprungen == ["docs/spaces/001_x.md"]
    assert NEUER_TEXT in (fx.repo / "docs/spaces/002_y.md").read_text("utf-8")
    assert (fx.repo / "docs/spaces/001_x.md").read_bytes() == dreckig  # unberuehrt
    assert "docs/spaces/001_x.md" not in _namen_im_commit(fx)
    assert _status(fx.repo) == [" M docs/spaces/001_x.md"]
    # gemerkte Aussagen von 001 bleiben unveraendert im Graph und in graph.json (keine Entfernung)
    assert {aid: _belege_von(fx.repo, aid) for aid in _aussagen(fx.repo, "docs/spaces/001_x.md")} == alt_001
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "doku_ziel_fremd"]
    assert [p["datei"] for p in punkte] == ["docs/spaces/001_x.md"]
    # einmal je Datei und Inhalt: gleicher Inhalt -> kein neuer Punkt, anderer Inhalt -> einer mehr
    _code_aenderung(fx, 1)
    _anstoss(fx)
    _tick(fx)
    assert len([p for p in _pruefliste(fx)["offen"] if p["art"] == "doku_ziel_fremd"]) == 1
    (fx.repo / "docs/spaces/001_x.md").write_bytes(dreckig + b"noch mehr\n")
    _code_aenderung(fx, 2)
    _anstoss(fx)
    _tick(fx)
    assert len([p for p in _pruefliste(fx)["offen"] if p["art"] == "doku_ziel_fremd"]) == 2


def test_fremde_ketten_md_blockiert_weiter_den_ganzen_tick(fx):
    _zwei_ziele(fx)
    _umbenennen(fx)
    head = _head(fx.repo)
    (fx.repo / "docs/spaces/001_x.md").write_bytes(b"# X von Hand\n")
    fremd = _text(fx) + "\nNotiz einer anderen Session.\n"
    _schreib(fx.repo, KETTEN, fremd)
    _anstoss(fx)
    lauf = FakeLaufzeit(*[_urteil()] * 4)
    erg = _tick(fx, laufzeit=lauf)
    assert erg.uebersprungen == f"fremde_aenderungen:{KETTEN}"
    assert erg.commit is None and lauf.aufrufe == [] and _head(fx.repo) == head and _text(fx) == fremd
    assert AUSSAGE in (fx.repo / "docs/spaces/002_y.md").read_text("utf-8")


def test_trocken_meldet_fremde_doku_ziele_lesend(fx):
    _zwei_ziele(fx)
    (fx.repo / "docs/spaces/001_x.md").write_bytes(b"# X von Hand\n")
    _alter_loop(fx)
    vorher = _zustand(fx)
    erg = _trocken(fx)
    assert erg.uebersprungen is None and erg.doku_uebersprungen == ["docs/spaces/001_x.md"]
    assert erg.doku_gelesen == 2  # KETTEN.md + 002
    _unveraendert(fx, vorher)


# ---------- Torwaechter (Ruling 4) ----------

_KOPF_W = re.compile(r"^#{2,4}\s+.*Belege", re.IGNORECASE)


def _waechter_stub(repo):
    """Wie `docs_drift_waechter.belege_tokens`: Backtick-Pfade (mit `/` und Endung) im ersten Abschnitt, dessen
    Ueberschrift `Belege` enthaelt, muessen existieren."""
    fehler = set()
    for doc in sorted((repo / "docs/spaces").glob("0*.md")):
        zeilen = doc.read_text(encoding="utf-8").splitlines()
        start = next((i for i, z in enumerate(zeilen) if _KOPF_W.match(z)), None)
        if start is None:
            continue
        tiefe = len(zeilen[start]) - len(zeilen[start].lstrip("#"))
        for z in zeilen[start + 1:]:
            if re.match(r"^#{2,4}\s", z) and len(z) - len(z.lstrip("#")) <= tiefe:
                break
            for tok in re.findall(r"`([^`\n]+)`", z):
                pfad = re.sub(r":\s*\d[\d,\s\-–]*$", "", tok)
                if "/" in pfad and "." in pfad.rsplit("/", 1)[-1] and not (repo / pfad).exists():
                    fehler.add((doc.relative_to(repo).as_posix(), f"Beleg fehlt: {pfad}"))
    return fehler


def test_ersetzung_bricht_belege_pfad_torwaechter_blockiert(fx):
    """Die Ueberschrift `## Quellen und Belege` ist fuer den Loop kein Belege-Abschnitt (Aussagen), fuer den
    Drift-Waechter schon: ein Ersatztext mit einem nicht vorhandenen Pfad wird zurueckgerollt."""
    _ziele(fx, doku_ziele=("docs/spaces/0*.md",))
    _seed(fx, drift=_waechter_stub)
    doku = "\n".join(["# Y", "", "**Stand:** 2026-10-30", "", "## Quellen und Belege", "", AUSSAGE, ""])
    _schreib(fx.repo, "docs/spaces/002_y.md", doku)
    _commit(fx.repo, "002", "docs/spaces/002_y.md")
    _anstoss(fx)
    assert _tick(fx, erzwingen=True, drift=_waechter_stub).commit is not None  # Erstlauf merkt die Belege
    _umbenennen(fx)
    _anstoss(fx)
    kaputt = NEUER_TEXT.replace("sp/a/client.py:7", "sp/a/kunde.py:7")
    head, ketten = _head(fx.repo), _text(fx)
    lauf = FakeLaufzeit(_urteil(), _urteil(), _urteil(text=kaputt), _urteil(text=kaputt))
    erg = _tick(fx, laufzeit=lauf, drift=_waechter_stub)
    assert erg.llm_aufrufe == 4
    assert erg.commit is None and erg.uebersprungen == "torwaechter:torwaechter_blockiert"
    assert (fx.repo / "docs/spaces/002_y.md").read_text("utf-8") == doku and _text(fx) == ketten
    assert _head(fx.repo) == head and _status(fx.repo) == []
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "torwaechter_blockiert"]
    assert punkte and "sp/a/kunde.py" in punkte[-1]["text"]


# ---------- nur lesend (Step 3, D7, Ruling T4-a) ----------

def _nur_lesend(fx, tmp_path):
    _submodul(fx, tmp_path, "ext/doku", {"README.md": README_SUB})
    _ziele(fx, doku_nur_lesen=("ext/*/README.md",))
    _seed(fx)
    _anstoss(fx)
    erg = _tick(fx, erzwingen=True)  # Erstlauf: gemerkt, nie geschrieben
    assert erg.llm_aufrufe == 0 and erg.nur_lesend_gelesen == 1
    knoten = _aussagen(fx.repo, "ext/doku/README.md")
    assert len(knoten) == 1
    return knoten


def test_nur_lesend_nie_ersetzt(fx, tmp_path):
    _nur_lesend(fx, tmp_path)
    sub = fx.repo / "ext/doku"
    readme, sub_head = (sub / "README.md").read_bytes(), _head(sub)
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("return 1", "return 2"))  # nur `helfer`
    _commit(fx.repo, "helfer", "sp/a/client.py")
    _anstoss(fx)
    antwort = _urteil(text=README_NEU, anker="client.helfer")
    lauf = FakeLaufzeit(antwort, antwort)
    erg = _tick(fx, laufzeit=lauf)
    assert erg.llm_aufrufe == 2 and erg.signale == 1 and erg.umgeschrieben == 0
    assert (sub / "README.md").read_bytes() == readme and _head(sub) == sub_head and _status(sub) == []
    assert _status(fx.repo) == [] and not (fx.repo / "docs/graph/ledger.jsonl").exists()
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "nur_lesend_veraltet"]
    assert len(punkte) == 1 and punkte[0]["datei"] == "ext/doku/README.md"
    assert "veraltet" in punkte[0]["text"] and "die Zwei" in punkte[0]["text"]
    # das Urteil steht im Cache: der naechste Lauf fragt nicht noch einmal und meldet nichts Neues
    _code_aenderung(fx, 1)
    _anstoss(fx)
    lauf2 = FakeLaufzeit()
    erg2 = _tick(fx, laufzeit=lauf2)
    assert lauf2.aufrufe == [] and erg2.umgeschrieben == 0
    assert len([p for p in _pruefliste(fx)["offen"] if p["art"] == "nur_lesend_veraltet"]) == 1


def test_nur_lesend_liest_den_committeten_stand(fx, tmp_path):
    """Ruling T4-a: eine uncommittete Aenderung im Submodul-Arbeitsbaum ist unsichtbar (gelesen wird im Messbaum)."""
    vorher = _nur_lesend(fx, tmp_path)
    sub = fx.repo / "ext/doku"
    dreckig = b"# Doku-Paket\n\nEin ganz anderer Satz ueber `helfer` in `sp/a/client.py:11` ohne Commit.\n"
    (sub / "README.md").write_bytes(dreckig)
    _code_aenderung(fx, 1)
    _anstoss(fx)
    erg = _tick(fx, erzwingen=True)
    assert erg.llm_aufrufe == 0 and erg.nur_lesend_gelesen == 1
    assert set(_aussagen(fx.repo, "ext/doku/README.md")) == set(vorher)
    assert (sub / "README.md").read_bytes() == dreckig


def test_nur_lesend_trigger_bei_commit_im_submodul(fx, tmp_path):
    """Ein Commit, der nur eine nur-lesend-Datei aendert, stoesst die Pruefung an (ohne Neubau der Spaces)."""
    _nur_lesend(fx, tmp_path)
    sub = fx.repo / "ext/doku"
    _schreib(sub, "README.md", README_SUB + "\nDer Dienst b lauscht laut `config/dienste.yaml:8` auf Port 8130.\n")
    _commit(sub, "readme", "README.md")
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.uebersprungen is None and erg.nur_lesend_gelesen == 1 and erg.llm_aufrufe == 0
    assert len(_aussagen(fx.repo, "ext/doku/README.md")) == 2
    assert json.loads((fx.repo / GRAPH).read_text("utf-8"))["version"] == 1


# ---------- Fix-Runde 1, Item 1: Urteile warten nur fuer das fremd geaenderte Ziel ----------

def _urteile_fuer(fx, aids, text):
    from selbstbild.doku.pruefliste import pruefliste_ergaenzen
    from selbstbild.doku.typen import Pruefpunkt

    pruefliste_ergaenzen(fx.konfig, [Pruefpunkt("judge_unklar", "zur Durchsicht", d, 5, aid)
                                     for d, aid in aids.items()], [], voll=False, lauf="2026-10-28T12:00:00#7")
    pfad = fx.konfig.rowboat_wissen / fx.konfig.rowboat_urteile
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_bytes(("\n".join(["# Urteile", *(f"2026-10-28T12:00:00#7 {aid} korrektur: {text}"
                                                for aid in aids.values())]) + "\n").encode("utf-8"))


def test_urteile_warten_nur_fuer_das_dreckige_ziel(fx):
    from selbstbild.doku import rowboat as rb

    _zwei_ziele(fx)
    a, b = "docs/spaces/001_x.md", "docs/spaces/002_y.md"
    aid = {d: next(x.id for x in rb.aussagen_ohne_belege(d, (fx.repo / d).read_text("utf-8")) if x.text == AUSSAGE)
           for d in (a, b)}
    neu = "Ein ganz neuer Satz über den Client."
    _urteile_fuer(fx, aid, neu)
    dreckig = (fx.repo / a).read_bytes() + b"\nfremde Notiz\n"
    (fx.repo / a).write_bytes(dreckig)
    _anstoss(fx)
    _tick(fx, erzwingen=True)
    assert neu in (fx.repo / b).read_text("utf-8")  # B angewandt
    assert (fx.repo / a).read_bytes() == dreckig  # A unberuehrt ...
    erledigt = rb.erledigt_laden(rb.erledigt_pfad(fx.konfig))
    assert [k for k in erledigt if aid[b] in str(k)] and not [k for k in erledigt if aid[a] in str(k)]  # ... offen
    _git(fx.repo, "checkout", "--", a)  # die andere Session raeumt auf: jetzt wird A angewandt
    _anstoss(fx)
    _tick(fx, erzwingen=True)
    assert neu in (fx.repo / a).read_text("utf-8")
    assert [k for k in rb.erledigt_laden(rb.erledigt_pfad(fx.konfig)) if aid[a] in str(k)]
    assert _status(fx.repo) == []
