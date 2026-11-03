"""Stufe 4, Task 6 (D9): ein neuer Space-Ordner ohne Doku bekommt eine generierte Doku-Datei (nur markierte Bloecke)
und eine Zeile im Block `neue-spaces` von `docs/spaces/README.md`. Kein LLM; gemessen wird im Messbaum.
"""
from __future__ import annotations

import dataclasses
import re

from test_doku_tick import (  # noqa: F401  (Fixture `fx`)
    FakeLaufzeit, FakeManifestBauer, _anstoss, _commit, _git, _head, _jetzt, _pruefliste, _schreib, _seed, _status,
    _tick, fx,
)
from test_doku_ziele import _waechter_stub
from selbstbild.doku import neue_spaces as ns
from selbstbild.doku import tick as tk
from selbstbild.doku.graph import Graph, Kante, Knoten, kante_id
from selbstbild.doku.render_ketten import bloecke_finden
from selbstbild.doku.stand import stand_laden

README = "docs/spaces/README.md"
NEU = "docs/spaces/021_testraum.md"

# wie die Regeln in VibeMinds selbstbild.yaml
REGELN = """\
name: fixture
struktur:
  - art: kein_space_im_root
    ort: "./"
    pflicht: []
    verboten:
      - "spaces/*/"
    hinweis: "Spaces gehoeren nach vibemind-os/spaces/<name>/"
  - art: space
    ort: "vibemind-os/spaces/*/"
    pflicht: []
    verboten: []
    hinweis: "Spaces gehoeren nach vibemind-os/spaces/<name>/"
"""
README_HAND = ("# Spaces\n\nHandgepflegter Index.\n\n| # | Space |\n|---|---|\n| 020 | [alt](020_alt.md) |\n\n"
               "## Verwandt\n\n- nichts\n")
DOKU_ALT = ("# 020 alt\n\n**Stand:** 2026-10-30\n\n## Was hier passiert\n\nDer Space alt rechnet.\n\n## Belege\n\n"
            "- `vibemind-os/spaces/alt/main.py`\n")
REG_BASIS = "version: 1\nspaces:\n  alt:\n    prefixes: [alt.]\n    events:\n      alt.ping: { tool: pingen }\n"
REG_TESTRAUM = REG_BASIS + "  testraum:\n    prefixes: [testraum.]\n    events:\n      testraum.start: { tool: starten }\n"
API = 'from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get("/status")\ndef status():\n    return 1\n'
SERVER = ('from mcp.server.fastmcp import FastMCP\n\nmcp = FastMCP("testraum")\n\n\n@mcp.tool()\ndef pruefen():\n'
          '    return True\n')

# docs_drift_waechter: DOC_FILE, INDEX_LINK und STAND wortgleich (echte Doku nutzt dreistellige Nummern)
DOC_FILE = re.compile(r"^\d{2,3}_([a-z0-9_]+)\.md$")
INDEX_LINK = re.compile(r"\]\((\d{2,3}_[a-z0-9_]+\.md)\)")
STAND = re.compile(r"(?:Stand|Datum)[^\d\n]{0,12}(\d{4}-\d{2}-\d{2})")
STAND_PART_RE = re.compile(r"^([A-Za-z0-9_.\-/]+)@([0-9a-f]{7,40})$")  # space_doc_check


def _gitlinks_unter(repo, prefix):
    """`docs_drift_waechter.gitlinks_unter`: (pfad, sha) der Gitlinks direkt unter prefix, `git ls-tree HEAD` im
    Ordner `vibemind-os` (Pfade relativ dazu)."""
    out = _git(repo / "vibemind-os", "ls-tree", "HEAD", prefix)
    treffer = []
    for zeile in out.splitlines():
        meta, _, pfad = zeile.partition("\t")
        mode, typ, sha = meta.split()[:3]
        if mode == "160000" and typ == "commit":
            treffer.append((pfad, sha))
    return treffer


def _pins(repo, doc, text):
    """`docs_drift_waechter.pruefe_pins` fuer `bereich == "spaces"` (`erwartete_pins`, `_space_dir`)."""
    name = DOC_FILE.match(doc.name).group(1)

    def norm(p):
        return p.rsplit("/", 1)[-1].replace("-", "_").lstrip("_")

    pins = [(p, s) for p, s in _gitlinks_unter(repo, "spaces/") if norm(p) == name]
    space_dir = next((v for v in (name, name.replace("_", "-"), "_" + name)
                      if (repo / "vibemind-os/spaces" / v).is_dir()), name)
    pins += _gitlinks_unter(repo, f"spaces/{space_dir}/")
    return {(doc.relative_to(repo).as_posix(), f"Pin-Drift: `{p}` steht in HEAD auf {s[:12]}, im Dokument nicht genannt")
            for p, s in pins if s[:7] not in text}


def _waechter(repo):
    """Torwaechter-Stub wie `docs_drift_waechter`: Belege-Pfade existieren (`_waechter_stub`), jedes Dokument nennt die
    Pins seiner Gitlinks (`pruefe_pins`) und der Index `docs/spaces/README.md` verlinkt genau die vorhandenen
    `NN(N)_name.md` (`pruefe_index`)."""
    fehler = set(_waechter_stub(repo))
    for doc in sorted((repo / "docs/spaces").glob("*.md")):
        if DOC_FILE.match(doc.name):
            fehler |= _pins(repo, doc, doc.read_text(encoding="utf-8", errors="replace"))
    ordner = repo / "docs/spaces"
    index = ordner / "README.md"
    if not index.exists():
        return fehler | {("Index", "docs/spaces/README.md fehlt")}
    verlinkt = set(INDEX_LINK.findall(index.read_text(encoding="utf-8", errors="replace")))
    vorhanden = {p.name for p in ordner.glob("*.md") if DOC_FILE.match(p.name)}
    fehler |= {("Index", f"Index verlinkt `{f}` nicht") for f in vorhanden - verlinkt}
    fehler |= {("Index", f"Index verlinkt `{f}`, Datei fehlt") for f in verlinkt - vorhanden}
    return fehler


def _t(fx, **kw):
    kw.setdefault("drift", _waechter)
    return _tick(fx, **kw)


def _basis(fx):
    """Fixture wie VibeMind: Space-Regel `vibemind-os/spaces/*/`, ein dokumentierter Space `alt` (020), ein
    `_contract`-Ordner, ein Registry-Event; Seed und erster Lauf (Events-Block in 020)."""
    fx.konfig = dataclasses.replace(fx.konfig, doku_ziele=("docs/spaces/0*.md",), ereignisse=(
        {"yaml": "config/reg.yml", "pfad": "spaces.*.events", "werkzeug": "tool"},))
    dateien = {"selbstbild.yaml": REGELN, "vibemind-os/spaces/alt/main.py": "def main():\n    return 1\n",
               "vibemind-os/spaces/_contract/alt.contract.yaml": "name: alt\n",
               "vibemind-os/spaces/__pycache__/x.txt": "kein space\n",
               "docs/spaces/020_alt.md": DOKU_ALT, README: README_HAND, "config/reg.yml": REG_BASIS}
    for rel, text in dateien.items():
        _schreib(fx.repo, rel, text)
    _commit(fx.repo, "basis", *dateien)
    _seed(fx, drift=_waechter)
    _anstoss(fx)
    erg = _t(fx, erzwingen=True)
    assert erg.uebersprungen is None and erg.neue_spaces == [] and not (fx.repo / NEU).exists()
    assert (fx.repo / README).read_text("utf-8") == README_HAND  # ohne generierte Doku bleibt der Index unberuehrt
    assert _waechter(fx.repo) == set() and _status(fx.repo) == []
    return erg


def _testraum(fx, *, registry=True):
    _schreib(fx.repo, "vibemind-os/spaces/testraum/api.py", API)
    _schreib(fx.repo, "vibemind-os/spaces/testraum/server.py", SERVER)
    pfade = ["vibemind-os/spaces/testraum/api.py", "vibemind-os/spaces/testraum/server.py"]
    if registry:
        _schreib(fx.repo, "config/reg.yml", REG_TESTRAUM)
        pfade.append("config/reg.yml")
    return _commit(fx.repo, "neuer space testraum", *pfade)


def _ausserhalb_der_bloecke(text):
    """Zeilen ausserhalb der markierten Bloecke (ohne Marker und Leerzeilen)."""
    aus, drin = [], False
    for z in text.splitlines():
        if z.startswith("<!-- graph:begin "):
            drin = True
        elif z == "<!-- graph:end -->":
            drin = False
        elif not drin and z.strip():
            aus.append(z)
    return aus


def _namen(repo, rev="HEAD"):
    return set(_git(repo, "show", "--name-only", "--format=", rev).split())


# ---------- Step 1 ----------

def test_neuer_space_bekommt_doku(fx):
    _basis(fx)
    sha = _testraum(fx)
    _anstoss(fx)
    lauf = FakeLaufzeit()
    erg = _t(fx, laufzeit=lauf)  # ohne erzwingen: der neue Ordner liegt in keinem konfigurierten Space
    assert erg.uebersprungen is None and erg.commit == _head(fx.repo)
    assert erg.llm_aufrufe == 0 and lauf.aufrufe == []
    assert erg.neue_spaces == ["testraum"] and erg.neue_dateien == [NEU]
    text = (fx.repo / NEU).read_text("utf-8")
    assert text.startswith("# 021 testraum\n")
    assert f"Stand: repo@{sha[:7]}" in text  # Repo des Space (hier das aeussere) @ letzter Commit am Ordner
    assert re.search(r"^Stand:\s*(.+?)\s*$", text, re.M).group(1) == f"repo@{sha[:7]}"  # space_doc_check.STAND_RE
    # M1: `Datum:` mit dem Commit-Datum dieses sha -- docs_drift_waechter.pruefe_stand findet es im Kopf
    datum = _git(fx.repo, "log", "-1", "--format=%cs", sha).strip()
    assert f"Stand: repo@{sha[:7]}\nDatum: {datum}\n" in text
    assert STAND.search(text.split("\n## ", 1)[0]).group(1) == datum
    bloecke = bloecke_finden(text)
    assert set(bloecke) == {"stand testraum", "steckbrief testraum", "ereignisse testraum", "schnittstellen testraum",
                            "belege testraum"}
    assert "`vibemind-os/spaces/testraum/`" in bloecke["steckbrief testraum"].inhalt
    assert "`testraum.start`" in bloecke["ereignisse testraum"].inhalt
    assert "`/status`" in bloecke["schnittstellen testraum"].inhalt
    assert "`pruefen`" in bloecke["schnittstellen testraum"].inhalt
    belege = bloecke["belege testraum"].inhalt
    assert "- [B1] `vibemind-os/spaces/testraum/`" in belege
    assert "`vibemind-os/spaces/testraum/api.py`" in belege and "`config/reg.yml`" in belege
    # nur markierte Bloecke: ausserhalb stehen Kopfzeile, Hinweis und Abschnitts-Ueberschriften
    assert _ausserhalb_der_bloecke(text) == [
        "# 021 testraum", ns.HINWEIS, "## Steckbrief (generiert)", "## Events (generiert)",
        "## Schnittstellen (generiert)", "## Belege"]
    # README: Hand-Inhalt byteweise, Block am Ende mit genau einer Zeile
    readme = (fx.repo / README).read_text("utf-8")
    assert readme.startswith(README_HAND)
    block = bloecke_finden(readme)["neue-spaces"].inhalt
    assert "[testraum](021_testraum.md)" in block and readme.rstrip().endswith("<!-- graph:end -->")
    assert len([z for z in block.splitlines() if "](0" in z]) == 1
    # Torwaechter-Stub (Index + Belege): kein Fehler; Commit nur mit den eigenen Dateien
    assert _waechter(fx.repo) == set() and _status(fx.repo) == []
    assert _namen(fx.repo) == {NEU, README, "docs/graph/graph.json"}
    hashes = stand_laden(fx.konfig).render_hashes
    assert f"{NEU}#steckbrief testraum" in hashes and f"{README}#neue-spaces" in hashes
    # ein weiterer Lauf: nichts mehr zu tun
    _anstoss(fx)
    assert _t(fx, erzwingen=True).commit is None and _status(fx.repo) == []


def test_neuer_space_ohne_events_wird_angestossen(fx):
    """Nur der Ordner kommt dazu (keine Ereignis-Quelle, kein konfigurierter Space): der Lauf startet trotzdem."""
    _basis(fx)
    _testraum(fx, registry=False)
    _anstoss(fx)
    erg = _t(fx)
    assert erg.commit == _head(fx.repo) and erg.neue_dateien == [NEU]
    inhalt = bloecke_finden((fx.repo / NEU).read_text("utf-8"))["ereignisse testraum"].inhalt
    assert "Keine Events" in inhalt
    assert _namen(fx.repo) == {NEU, README}
    assert _waechter(fx.repo) == set()
    struktur = [p for p in _pruefliste(fx)["struktur"] if p.get("aussage_id") == "ereignisse testraum"]
    assert struktur == []  # kein `block_ohne_quelle` fuer den generierten Block


def test_zwei_neue_spaces_fortlaufend_nach_name(fx):
    _basis(fx)
    for name in ("zzz", "aaa"):
        _schreib(fx.repo, f"vibemind-os/spaces/{name}/main.py", "def main():\n    return 2\n")
    _commit(fx.repo, "zwei", "vibemind-os/spaces/zzz/main.py", "vibemind-os/spaces/aaa/main.py")
    _anstoss(fx)
    erg = _t(fx)
    assert erg.neue_spaces == ["aaa", "zzz"]
    assert erg.neue_dateien == ["docs/spaces/021_aaa.md", "docs/spaces/022_zzz.md"]
    readme = (fx.repo / README).read_text("utf-8")
    assert readme.index("021_aaa.md") < readme.index("022_zzz.md")
    assert _waechter(fx.repo) == set() and _status(fx.repo) == []


# ---------- Step 2 ----------

def test_kein_neuer_space_nichts(fx):
    _basis(fx)
    head, readme = _head(fx.repo), (fx.repo / README).read_bytes()
    _anstoss(fx)
    erg = _t(fx, erzwingen=True)
    assert erg.commit is None and erg.neue_spaces == [] and erg.neue_dateien == []
    assert _head(fx.repo) == head and (fx.repo / README).read_bytes() == readme
    assert sorted(p.name for p in (fx.repo / "docs/spaces").glob("0*.md")) == ["020_alt.md"]


# ---------- Step 3 ----------

def test_space_entfernt(fx):
    _basis(fx)
    _testraum(fx)
    _anstoss(fx)
    assert _t(fx).commit is not None
    vorher, readme = (fx.repo / NEU).read_bytes(), (fx.repo / README).read_bytes()
    _git(fx.repo, "rm", "-r", "-q", "vibemind-os/spaces/testraum")
    _git(fx.repo, "commit", "-q", "-m", "testraum weg")
    _anstoss(fx)
    erg = _t(fx)
    assert erg.uebersprungen is None and erg.commit is None
    assert (fx.repo / NEU).read_bytes() == vorher and (fx.repo / README).read_bytes() == readme
    punkte = [p for p in _pruefliste(fx)["struktur"] if p["art"] == "space_doku_ohne_space"]
    assert len(punkte) == 1 and punkte[0]["datei"] == NEU and punkte[0]["aussage_id"] == "space:testraum"
    assert "Datei löschen oder Belege anpassen; der Drift-Waechter meldet bis dahin 'Beleg fehlt'." in punkte[0]["text"]
    assert not [p for p in _pruefliste(fx)["struktur"] if p["art"] in ("block_unbekannt", "block_ohne_quelle")]
    # der Index bleibt vollstaendig; die stehen gebliebenen Belege zeigen auf die geloeschten Dateien -- das hat der
    # Loesch-Commit verursacht (steht im Torwaechter "vorher", blockiert also keinen spaeteren Loop-Commit)
    assert {(d, m.split(":")[0]) for d, m in _waechter(fx.repo)} == {(NEU, "Beleg fehlt")}
    assert _status(fx.repo) == []
    # kommt der Ordner zurueck, verschwindet der Pruefpunkt im naechsten Lauf
    _testraum(fx)
    _anstoss(fx)
    _t(fx)
    assert not [p for p in _pruefliste(fx)["struktur"] if p["art"] == "space_doku_ohne_space"]


# ---------- spaetere Laeufe, Hand-Edit, fremde Arbeit ----------

def test_generierte_doku_wird_spaeter_neu_gerendert_mit_hand_edit_schutz(fx):
    _basis(fx)
    _testraum(fx)
    _anstoss(fx)
    _t(fx)
    # Hand-Edit im Steckbrief-Block, dann eine neue Route
    text = (fx.repo / NEU).read_text("utf-8")
    von_hand = text.replace("| Routen | 1 |", "| Routen | eine (von Hand) |")
    assert von_hand != text
    _schreib(fx.repo, NEU, von_hand)
    _commit(fx.repo, "von Hand", NEU)
    _schreib(fx.repo, "vibemind-os/spaces/testraum/api.py", API + '\n\n@app.post("/neu")\ndef neu():\n    return 2\n')
    _commit(fx.repo, "neue route", "vibemind-os/spaces/testraum/api.py")
    _anstoss(fx)
    erg = _t(fx)
    assert erg.commit == _head(fx.repo) and _namen(fx.repo) == {NEU}
    neu = bloecke_finden((fx.repo / NEU).read_text("utf-8"))
    assert "`/neu`" in neu["schnittstellen testraum"].inhalt  # neu gerendert
    assert "eine (von Hand)" in neu["steckbrief testraum"].inhalt  # Hand-Edit bleibt
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "hand_edit"]
    assert len(punkte) == 1 and punkte[0]["datei"] == NEU and punkte[0]["aussage_id"].startswith("steckbrief testraum#")
    assert _waechter(fx.repo) == set()


def test_fremde_datei_am_ziel_wird_nicht_ueberschrieben(fx):
    """Eine fremde, nicht committete Doku mit dem Namen des neuen Space: sie gilt als Doku (keine zweite Datei) und
    als fremdes Ziel (ausgelassen, Pruefpunkt), wird nie ueberschrieben."""
    _basis(fx)
    fremd = "# 021 testraum\n\nEine andere Session schreibt hier.\n"
    _schreib(fx.repo, NEU, fremd)
    _testraum(fx)
    _anstoss(fx)
    erg = _t(fx)
    assert erg.neue_spaces == [] and erg.neue_dateien == []
    assert (fx.repo / NEU).read_text("utf-8") == fremd
    assert not list((fx.repo / "docs/spaces").glob("022_*.md"))
    assert NEU not in _namen(fx.repo) and README not in _namen(fx.repo)
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "doku_ziel_fremd"]
    assert [p["datei"] for p in punkte] == [NEU]


def test_readme_mit_fremder_aenderung_neue_doku_wartet(fx):
    _basis(fx)
    _schreib(fx.repo, README, README_HAND + "\nfremde Zeile\n")
    _testraum(fx)
    _anstoss(fx)
    erg = _t(fx)
    assert erg.neue_spaces == ["testraum"] and erg.neue_dateien == []
    assert not (fx.repo / NEU).exists()
    assert (fx.repo / README).read_text("utf-8") == README_HAND + "\nfremde Zeile\n"
    punkte = [p for p in _pruefliste(fx)["struktur"] if p["art"] == "space_doku_wartet"]
    assert len(punkte) == 1 and "README" in punkte[0]["text"]
    # sobald die fremde Aenderung weg ist, entsteht die Doku
    _schreib(fx.repo, README, README_HAND)
    _anstoss(fx)
    erg = _t(fx, erzwingen=True)
    assert erg.neue_dateien == [NEU] and (fx.repo / NEU).is_file() and _waechter(fx.repo) == set()
    assert not [p for p in _pruefliste(fx)["struktur"] if p["art"] == "space_doku_wartet"]


def test_halb_angelegter_space_ohne_commit_bekommt_keine_doku(fx):
    """Erkannt wird im Messbaum (committeter Stand): ein nicht committeter neuer Ordner bleibt unsichtbar."""
    _basis(fx)
    _schreib(fx.repo, "vibemind-os/spaces/halb/main.py", "def main():\n    return 3\n")
    _anstoss(fx)
    erg = _t(fx, erzwingen=True)
    assert erg.neue_spaces == [] and not list((fx.repo / "docs/spaces").glob("021_*.md"))


# ---------- Fix-Runde 1, Ruling T6-a: Submodul-Spaces sind Spaces ----------

EXTERN = "docs/spaces/021_extern.md"


def _gitlink(fx, rel, sha):
    """Gitlink ohne Checkout (nicht initialisiertes Submodul); nur dieser Pfad im Commit."""
    (fx.repo / rel).mkdir(parents=True, exist_ok=True)
    _git(fx.repo, "update-index", "--add", "--cacheinfo", f"160000,{sha},{rel}")
    _git(fx.repo, "commit", "-q", "-m", f"gitlink {rel}")
    return _head(fx.repo)


def _stand(text):
    teile = [t.strip() for t in re.search(r"^Stand:\s*(.+?)\s*$", text, re.M).group(1).split(",")]
    assert all(STAND_PART_RE.match(t) for t in teile), teile  # space_doc_check.parse_stand liest jeden Teil
    return teile


def test_neuer_space_als_gitlink_nicht_initialisiert_und_pin_wechsel(fx):
    _basis(fx)
    pin = _head(fx.repo)
    commit = _gitlink(fx, "vibemind-os/spaces/extern", pin)
    _anstoss(fx)
    erg = _t(fx)  # nur der Gitlink im Commit: der Lauf startet trotzdem
    assert erg.commit == _head(fx.repo) and erg.neue_dateien == [EXTERN]
    text = (fx.repo / EXTERN).read_text("utf-8")
    assert _stand(text) == [f"repo@{commit[:7]}", f"vibemind-os/spaces/extern@{pin[:7]}"]
    assert f"Datum: {_git(fx.repo, 'log', '-1', '--format=%cs', commit).strip()}" in text
    bloecke = bloecke_finden(text)
    assert set(bloecke) == {"stand extern", "steckbrief extern", "belege extern"}
    assert "Unter-Repo, im Messbaum nicht vermessen" in bloecke["steckbrief extern"].inhalt
    assert "`vibemind-os/spaces/extern/`" in bloecke["steckbrief extern"].inhalt
    assert bloecke["belege extern"].inhalt == "- [B1] `vibemind-os/spaces/extern/`\n"
    assert _waechter(fx.repo) == set() and _namen(fx.repo) == {EXTERN, README}
    # der Pin-Stub ist echt: ein falscher Pin waere ein Fehler
    falsch = text.replace(pin[:7], "0000000")
    assert _pins(fx.repo, fx.repo / EXTERN, falsch) and not _pins(fx.repo, fx.repo / EXTERN, text)
    # Pin-Wechsel: der Block `stand` wird neu gerendert, sonst nichts
    neu_pin = _git(fx.repo, "rev-parse", "HEAD~1").strip()
    commit2 = _gitlink(fx, "vibemind-os/spaces/extern", neu_pin)
    _anstoss(fx)
    erg = _t(fx)
    assert erg.commit == _head(fx.repo) and _namen(fx.repo) == {EXTERN}
    text2 = (fx.repo / EXTERN).read_text("utf-8")
    assert _stand(text2) == [f"repo@{commit2[:7]}", f"vibemind-os/spaces/extern@{neu_pin[:7]}"]
    assert _waechter(fx.repo) == set() and _status(fx.repo) == []


def test_neuer_space_als_submodul_initialisiert(fx, tmp_path):
    from test_doku_tick import _submodul
    _basis(fx)
    _submodul(fx, tmp_path, "vibemind-os/spaces/extern", {"api.py": API})
    pin = _gitlinks_unter(fx.repo, "spaces/")[0][1]
    _anstoss(fx)
    erg = _t(fx)
    assert erg.commit == _head(fx.repo) and erg.neue_dateien == [EXTERN]
    text = (fx.repo / EXTERN).read_text("utf-8")
    bloecke = bloecke_finden(text)
    assert "`/status`" in bloecke["schnittstellen extern"].inhalt  # aus dem Submodul-Worktree im Messbaum
    assert "`vibemind-os/spaces/extern/api.py`" in bloecke["belege extern"].inhalt
    assert _stand(text)[-1] == f"vibemind-os/spaces/extern@{pin[:7]}"
    assert _waechter(fx.repo) == set() and _status(fx.repo) == []


def test_space_mit_gitlink_darin_misst_eigene_dateien(fx, tmp_path):
    from test_doku_tick import _submodul
    _basis(fx)
    _schreib(fx.repo, "vibemind-os/spaces/mitsub/main.py", "def main():\n    return 4\n")
    _commit(fx.repo, "mitsub", "vibemind-os/spaces/mitsub/main.py")
    _submodul(fx, tmp_path, "vibemind-os/spaces/mitsub/lib", {"api.py": API})
    pin = _gitlinks_unter(fx.repo, "spaces/mitsub/")[0][1]
    _anstoss(fx)
    erg = _t(fx)
    ziel = "docs/spaces/021_mitsub.md"
    assert erg.neue_dateien == [ziel]
    text = (fx.repo / ziel).read_text("utf-8")
    bloecke = bloecke_finden(text)
    assert "`vibemind-os/spaces/mitsub/main.py`" in bloecke["steckbrief mitsub"].inhalt
    assert "/status" not in text and "lib/api.py" not in text  # nur die eigenen Dateien, nicht das Unter-Repo
    assert _stand(text)[-1] == f"vibemind-os/spaces/mitsub/lib@{pin[:7]}"
    assert _waechter(fx.repo) == set() and _status(fx.repo) == []


def test_generierter_space_bekommt_spaeter_ein_unter_repo(fx, tmp_path):
    from test_doku_tick import _submodul
    _basis(fx)
    _testraum(fx)
    _anstoss(fx)
    _t(fx)
    _submodul(fx, tmp_path, "vibemind-os/spaces/testraum/lib", {"x.py": "def x():\n    return 1\n"})
    pin = _gitlinks_unter(fx.repo, "spaces/testraum/")[0][1]
    _anstoss(fx)
    erg = _t(fx)
    assert erg.commit == _head(fx.repo) and _namen(fx.repo) == {NEU}
    assert _stand((fx.repo / NEU).read_text("utf-8"))[-1] == f"vibemind-os/spaces/testraum/lib@{pin[:7]}"
    assert not [p for p in _pruefliste(fx)["struktur"] if p["art"] == "space_doku_ohne_space"]
    assert _waechter(fx.repo) == set() and _status(fx.repo) == []


def test_readme_nicht_utf8_ist_ein_pruefpunkt(fx):
    _basis(fx)
    (fx.repo / README).write_bytes(README_HAND.encode("utf-8") + b"\nkaputt: \xe4\n")
    _commit(fx.repo, "readme latin-1", README)
    _testraum(fx)
    _anstoss(fx)
    erg = _t(fx)
    assert erg.uebersprungen is None and erg.neue_dateien == [] and not (fx.repo / NEU).exists()
    punkte = [p for p in _pruefliste(fx)["struktur"] if p["art"] == "space_doku_wartet"]
    assert len(punkte) == 1 and "UTF-8" in punkte[0]["text"]


def test_trocken_meldet_neue_spaces_ohne_zu_schreiben(fx):
    _basis(fx)
    _testraum(fx)
    # ein echter Lauf ohne Space-Doku-Ziele bringt den Messbaum auf HEAD (Ruling T3-a: trocken stellt ihn nicht bereit)
    aus = dataclasses.replace(fx.konfig, doku_ziele=("docs/modules/0*.md",))
    _anstoss(fx)
    tk.tick(aus, jetzt=_jetzt(), laufzeit=FakeLaufzeit(), drift=_waechter, manifest_bauer=FakeManifestBauer(),
            erzwingen=True)
    head, readme, status = _head(fx.repo), (fx.repo / README).read_bytes(), _status(fx.repo)
    erg = tk.tick(fx.konfig, jetzt=_jetzt(), manifest_bauer=FakeManifestBauer(), trocken=True)
    assert erg.uebersprungen is None
    assert erg.neue_spaces == ["testraum"] and erg.neue_dateien == [NEU]
    assert not (fx.repo / NEU).exists() and (fx.repo / README).read_bytes() == readme
    assert _head(fx.repo) == head and _status(fx.repo) == status


# ---------- Einheiten ----------

def _baum(tmp_path):
    w = tmp_path / "w"
    for rel in ("selbstbild.yaml", "vibemind-os/spaces/alt/a.py", "vibemind-os/spaces/sales-claw/s.py",
                "vibemind-os/spaces/bubbles/b.py", "vibemind-os/spaces/_contract/c.yaml",
                "vibemind-os/spaces/__pycache__/p.pyc", "vibemind-os/spaces/fremd/f.py",
                "vibemind-os/spaces/mitrepo/x.py", "vibemind-os/spaces/neu/n.py", "vibemind-os/spaces/Neu-Zwei/n.py",
                "vibemind-os/spaces/sales_claw/doppelt.py", "docs/spaces/015_sales_claw.md", "docs/spaces/006_ideas.md",
                "docs/spaces/020_alt.md"):
        p = w / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(REGELN if rel == "selbstbild.yaml" else "x\n", encoding="utf-8")
    (w / "vibemind-os/spaces/leer").mkdir()  # leerer Ordner ohne Gitlink: kein Space
    (w / "vibemind-os/spaces/extern").mkdir()  # nicht initialisierter Gitlink im Messbaum: leer, aber ein Space
    (w / "vibemind-os/spaces/_privat").mkdir()  # Gitlink mit `_` am Anfang: kein Space
    return w


def test_neue_spaces_ausschluesse_und_namen(tmp_path, fx):
    """Ruling T6-a: ein Ordner, der ein Gitlink ist oder einen enthaelt, ist ein Space; `_` und `__pycache__` nicht."""
    w = _baum(tmp_path)
    konfig = dataclasses.replace(fx.konfig, repo=w, doku_ziele=("docs/spaces/0*.md",),
                                 space_dokumente={"bubbles": "docs/spaces/006_ideas.md"})
    unter = ("vibemind-os/spaces/fremd", "vibemind-os/spaces/mitrepo/sub", "vibemind-os/spaces/extern",
             "vibemind-os/spaces/_privat")
    assert ns.neue_spaces(konfig, Graph(), w, unter_repos=unter) == ["Neu-Zwei", "extern", "fremd", "mitrepo", "neu"]
    ordner = ns.space_ordner(w, unter_repos=unter)
    assert "sales-claw" in ordner and "sales_claw" not in ordner  # eine Schreibweise je Space-Schluessel
    assert not {"_contract", "__pycache__", "leer", "_privat"} & set(ordner)


def test_dateien_planen_nummer_und_name():
    pp = []
    geplant = ns.dateien_planen(["zwei", "Eins-A", "x.y"], ["docs/spaces/019_contract.md", "docs/spaces/020_nav.md",
                                                           "docs/spaces/KETTEN.md", "docs/spaces/README.md"], pp)
    assert geplant == [("Eins-A", "docs/spaces/021_eins_a.md"), ("zwei", "docs/spaces/022_zwei.md")]
    assert [p.art for p in pp] == ["space_name_ungeeignet"]


def _graph():
    g = Graph()
    g.knoten_hinzu(Knoten("space:t", "space", "t", space="t", datei="vibemind-os/spaces/t"))
    for datei in ("vibemind-os/spaces/t/api.py", "vibemind-os/spaces/t/hilfe.py"):
        g.knoten_hinzu(Knoten(f"modul:{datei}", "modul", datei, space="t", datei=datei))
    g.knoten_hinzu(Knoten("route:GET /a@t", "route", "GET /a", space="t", datei="vibemind-os/spaces/t/api.py",
                          attribute={"methode": "GET", "pfad": "/a", "zeile": "9"}))
    g.kante_hinzu(Kante(kante_id("registriert", "modul:vibemind-os/spaces/t/api.py", "mcp_werkzeug:w"), "registriert",
                        "modul:vibemind-os/spaces/t/api.py", "mcp_werkzeug:w", "extraktor", (), {"space": "t"}))
    return g


def test_doku_entwurf_format():
    text = ns.doku_entwurf("t", _graph(), [], 21, stand="vibemind-os@abcdef1")
    zeilen = text.splitlines()
    assert zeilen[0] == "# 021 t" and "Stand: vibemind-os@abcdef1" in zeilen
    assert text.index("## Steckbrief (generiert)") < text.index("## Events (generiert)") < text.index(
        "## Schnittstellen (generiert)") < text.index("## Belege")
    assert ns.HINWEIS in zeilen and "\r" not in text and text.endswith("\n")
    b = bloecke_finden(text)
    assert "<!-- graph:begin steckbrief t -->" in text
    assert "`vibemind-os/spaces/t/api.py`" in b["steckbrief t"].inhalt  # Einstiegspunkt: Route und Werkzeug
    assert "hilfe.py" not in b["steckbrief t"].inhalt
    assert "| `GET` | `/a` |" in b["schnittstellen t"].inhalt and "`w`" in b["schnittstellen t"].inhalt
    assert ":9" not in text  # keine Zeilennummern (verschobene Zeilen aendern nichts)
    assert b["belege t"].inhalt.splitlines()[:2] == ["- [B1] `vibemind-os/spaces/t/`",
                                                      "- [B2] `vibemind-os/spaces/t/api.py`"]
    assert ns.doku_entwurf("t", _graph(), [], 21, stand="vibemind-os@abcdef1") == text  # deterministisch


def test_readme_crlf_und_bom_nur_angehaengt():
    alt = "﻿# Index\r\n\r\n| # | Space |\r\n|---|---|\r\n| 020 | [alt](020_alt.md) |\r\n"
    res = ns.readme_rendern(alt, [("docs/spaces/021_t.md", "t")], {})
    assert res.text.startswith(alt) and "\n" not in res.text[len(alt):].replace("\r\n", "")
    assert "[t](021_t.md)" in res.text and "neue-spaces" in res.hashes
    # zweiter Lauf: unveraendert; Hand-Inhalt ausserhalb bleibt, Block wird ersetzt
    assert ns.readme_rendern(res.text, [("docs/spaces/021_t.md", "t")], res.hashes).text == res.text
    zwei = ns.readme_rendern(res.text, [("docs/spaces/021_t.md", "t"), ("docs/spaces/022_u.md", "u")], res.hashes)
    assert zwei.text.startswith(alt) and "[u](022_u.md)" in zwei.text and zwei.text.count("graph:begin") == 1
