"""Stufe 4, Final-Fix-Welle (Ruling FINAL-a): C1, C2b, I1-I5, M1, M3, M4.

Alle Schluessel in diesen Tests sind synthetisch (Wiederholungen von Buchstaben/Ziffern), nie echte Werte. Laeufe
gehen gegen das `git init`-Fixture aus `test_doku_tick.py` unter tmp_path; Judge, Budget und Torwaechter sind Fakes.
"""

import json
import re

import pytest

from selbstbild.doku import judge as jg
from selbstbild.doku import schluessel as sl
from selbstbild.doku import tick as tk
from selbstbild.doku.anker import Beleg
from selbstbild.doku.aussagen import Aussage
from selbstbild.doku.graph import Graph
from test_doku_aufgabe import umg  # noqa: E402,F401  (Fixture fuer M1)
from test_doku_erstbefuellung import D1, Abo, _aussage_im_prompt, _nachtlauf  # noqa: E402
from test_doku_erstbefuellung import NOCH_WAHR as ERST_NOCH_WAHR  # noqa: E402
from test_doku_erstbefuellung import _start as _erst_start  # noqa: E402
from test_doku_erstbefuellung import _veraltet  # noqa: E402
from test_doku_tick import (  # noqa: E402,F401  (Fixture `fx`)
    AUSSAGE, FakeBudget, FakeLaufzeit, FakeManifestBauer, _head, _jetzt, _start, _status, _text, _tick, _urteil, fx,
)

# synthetische Fake-Werte (keine echten Schluessel)
GHP = "ghp_" + "A1b2C3d4E5" * 3 + "F6g7H8"  # 36 Zeichen nach dem Praefix
SK_ANT = "sk-ant-api03-" + "Ab1_Cd2-" * 4
SK_PROJ = "sk-proj-" + "Zz9_Yy8-" * 4
TELEGRAM = "123456789:" + "AAbbCCddEEffGGhhIIjjKKllMMnnOOppQQr"  # 9 Ziffern, 35 Zeichen
PEM = "\n".join([
    "-----BEGIN RSA PRIVATE KEY-----",
    "MIIEowIBAAKCAQEAfakefakefakefakefakefakefakefakefakefakefakefake",
    "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2",
    "-----END RSA PRIVATE KEY-----",
])


# ---------- I1: PEM ganz schwaerzen ----------

def test_i1_pem_koerper_und_end_zeile_verschwinden():
    text = f"vorher\n{PEM}\nnachher"
    aus = sl.schwaerzen(text)
    assert "vorher" in aus and "nachher" in aus
    assert "MIIEowIBAAKCAQEA" not in aus and "QUJDREVGR0hJSktM" not in aus
    assert "END RSA PRIVATE KEY" not in aus and "BEGIN" not in aus
    assert "[Schlüssel-Verdacht: privater Schlüssel (PEM)]" in aus
    assert sl.schluessel_muster(text) == "privater Schlüssel (PEM)"


def test_i1_pem_ohne_end_bis_textende():
    text = "Kopf\n" + PEM.rsplit("\n", 1)[0] + "\nund noch eine Zeile"
    aus = sl.schwaerzen(text)
    assert aus.startswith("Kopf\n") and "MIIE" not in aus and "QUJD" not in aus and "noch eine Zeile" not in aus


# ---------- I2: sk-ant/sk-proj, Telegram ----------

@pytest.mark.parametrize("wert", [SK_ANT, SK_PROJ])
def test_i2_sk_mit_bindestrich_und_unterstrich(wert):
    text = f"Schluessel {wert} im Text"
    assert sl.schluessel_muster(text) == "OpenAI-artiger API-Schlüssel"
    aus = sl.schwaerzen(text)
    assert wert[3:12] not in aus and "sk-" not in aus and aus.endswith(" im Text")
    assert sl.schluessel_muster(f"KEY={wert}") is not None and sl.schluessel_muster(f'"{wert}"') is not None


def test_i2_sk_kein_fehltreffer_im_fliesstext():
    for harmlos in ("risk-assessment-report-v2-final", "task-runner-for-the-whole-space", "desk-sk-"):
        assert sl.schluessel_muster(harmlos) is None, harmlos


def test_i2_telegram_bot_token():
    text = f"Bot {TELEGRAM} fertig"
    assert sl.schluessel_muster(text) == "Telegram-Bot-Token"
    aus = sl.schwaerzen(text)
    assert "AAbbCC" not in aus and "123456789" not in aus and aus == "Bot [Schlüssel-Verdacht: Telegram-Bot-Token] fertig"
    # keine Fehltreffer: Uhrzeit, Port, kurze Kennung
    for harmlos in ("um 12:30 Uhr", "localhost:8130", "1234:abc", "12345678:" + "x" * 20):
        assert sl.schluessel_muster(harmlos) is None, harmlos


def test_i2_namen_ohne_wertteile():
    for name, _ in sl.SCHLUESSEL_MUSTER:
        assert not re.search(r"sk-|ghp_|xox|AKIA|eyJ|\d{8}:", name)


# ---------- C1: der Ersatztext des LLM geht durch den Schluessel-Filter ----------

MIT_SCHLUESSEL = f"Der Client ruft `schicken` in `sp/a/client.py:7` mit {GHP} auf."


def _lokal_text(fx):
    """Alles, was der Loop lokal schreibt (Stand, Pruefliste, Cache, Ledger-Vermerke), als ein Text."""
    teile = []
    for p in sorted(fx.lokal.rglob("*")):
        if p.is_file():
            teile.append(p.read_bytes().decode("utf-8", errors="replace"))
    return "\n".join(teile)


def test_c1_antwort_parsen_verwirft_schluessel_im_ersatztext():
    a = Aussage("d.md#1", "d.md", "X", 3, 3, AUSSAGE, ())
    roh = _urteil(text=MIT_SCHLUESSEL)
    assert jg.antwort_parsen(roh, ["client.schicken"], aussage=a) is None
    assert jg.antwort_parsen(roh, ["client.schicken"]) is None  # auch ohne Aussage
    sauber = jg.antwort_parsen(_urteil(), ["client.schicken"], aussage=a)
    assert sauber is not None and sauber.urteil == "veraltet"


def test_c1_signal_veraltet_mit_schluessel_schreibt_und_committet_nichts(fx):
    _start(fx)
    kopf = _head(fx.repo)
    zeilen: list[str] = []
    lauf, budget = FakeLaufzeit(_urteil(text=MIT_SCHLUESSEL), _urteil(text=MIT_SCHLUESSEL)), FakeBudget()
    erg = _tick(fx, laufzeit=lauf, budget=budget, melden=zeilen.append)
    assert erg.umgeschrieben == 0 and erg.llm_aufrufe == 1  # die erste Antwort ist schon ungueltig
    assert budget.buchungen == [("doku-loop", "fehler")]
    assert AUSSAGE in _text(fx) and GHP not in _text(fx)
    assert erg.commit is None or GHP not in (fx.repo / "docs/graph/graph.json").read_text(encoding="utf-8")
    assert not (fx.repo / "docs/graph/ledger.jsonl").exists()
    assert GHP not in _lokal_text(fx) and not any(GHP in z for z in zeilen)
    offen = tk.pruefliste_laden(fx.konfig)["offen"]
    assert any(p["art"] == "judge_ungueltig" for p in offen)
    assert _status(fx.repo) == [] and (_head(fx.repo) == kopf or erg.commit is not None)


def test_c1_ersetzen_lehnt_schluessel_ab_und_nennt_nur_das_muster(fx):
    lauf = tk._Lauf(fx.konfig, jetzt=_jetzt(), erzwingen=False, laufzeit=FakeLaufzeit(), ohne_llm=False, budget=None,
                    drift=lambda r: set(), manifest_bauer=FakeManifestBauer(), nach_tick=None, melden=lambda z: None)
    lauf.lauf = "2026-11-03T01:00:00#1"
    b = Beleg("symbol", "sp/a/client.py", "client.senden", "d" * 64, "c", 7)
    text = f"# K\n\n{AUSSAGE}\n"
    a = Aussage("docs/spaces/KETTEN.md#a", "docs/spaces/KETTEN.md", "K", 3, 3, AUSSAGE, (b,))
    graph = Graph({}, {})
    for nur_lesend in (False, True):
        lauf.offen = []
        neu, getan = lauf._ersetzen(graph, text, [(a, MIT_SCHLUESSEL, b, [b])], nur_lesend=nur_lesend)
        assert neu == text and getan == [] and lauf.umgeschrieben == 0 and lauf.ledger == []
        (pp,) = lauf.offen
        assert pp.art == "ersetzen_abgelehnt" and GHP not in pp.text and "ghp_" not in pp.text
        assert "GitHub-Token (klassisch)" in pp.text and pp.aussage_id == a.id
        assert "aussage:docs/spaces/KETTEN.md#a" in graph.knoten  # Belege bleiben gemerkt


def test_c1_erstbefuellung_veraltet_mit_schluessel(fx):
    _erst_start(fx)
    doku = fx.repo / "docs/spaces/001_e1.md"
    vorher, kopf = doku.read_bytes(), _head(fx.repo)
    abo = Abo(lambda p: _veraltet(D1.replace("Eins", GHP), "client.helfer") if _aussage_im_prompt(p) == D1
              else ERST_NOCH_WAHR)
    zeilen: list[str] = []
    erg = _nachtlauf(fx, laufzeit=abo, melden=zeilen.append)
    assert erg.umgeschrieben == 0 and doku.read_bytes() == vorher
    assert erg.commit is None and _head(fx.repo) == kopf and _status(fx.repo) == []
    assert GHP not in _lokal_text(fx) and not any(GHP in z for z in zeilen)
    offen = tk.pruefliste_laden(fx.konfig)["offen"]
    assert any(p["art"] == "judge_ungueltig" and p["datei"] == "docs/spaces/001_e1.md" for p in offen)
    assert all(GHP not in json.dumps(p, ensure_ascii=False) for p in offen)


# ---------- C2b: ein Lauf nur wegen der Erstbefuellung baut alle Spaces ----------

def _ohne_aussagen_im_repo_graph(fx, datei):
    """Wie nach der Umschaltung: der committete Graph kennt die Aussagen eines Doku-Ziels noch nicht (der alte Loop
    merkte nur KETTEN.md). Nur der Loop-Graph aendert sich -- weder Code noch Doku."""
    pfad = fx.repo / "docs/graph/graph.json"
    doc = json.loads(pfad.read_text(encoding="utf-8"))
    weg = {k["id"] for k in doc["knoten"] if k.get("typ") == "aussage" and k.get("datei") == datei}
    assert weg
    doc["knoten"] = [k for k in doc["knoten"] if k["id"] not in weg]
    doc["kanten"] = [e for e in doc["kanten"] if e["von"] not in weg]
    doc["digest"] = Graph.aus_doc(doc).digest()
    pfad.write_text(json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    _git_commit(fx, "graph ohne Aussagen", "docs/graph/graph.json")


def _git_commit(fx, nachricht, *pfade):
    from test_doku_tick import _commit
    return _commit(fx.repo, nachricht, *pfade)


def test_c2b_erstbefuellungs_lauf_ohne_aenderung_baut_alle_spaces(fx):
    from test_doku_ziele import _aussagen, _belege_von

    _erst_start(fx)
    _ohne_aussagen_im_repo_graph(fx, "docs/spaces/001_e1.md")
    _tick(fx, erzwingen=False)  # der bewegte HEAD wird am Tag nachgezogen (keine Aenderung an Code oder Doku)
    bauer, abo = FakeManifestBauer(), Abo()
    erg = _nachtlauf(fx, bauer=bauer, laufzeit=abo)
    (kid,) = _aussagen(fx.repo, "docs/spaces/001_e1.md")
    belege = _belege_von(fx.repo, kid)
    assert any(art == "symbol" and anker == "client.helfer" for _, art, _, anker, _ in belege), belege
    assert D1 in abo.aussagen()  # mit ihrem Space-Anker signaltragend -> einmal geprueft
    assert erg.erst_faellig and sorted(bauer.gebaut) == ["a", "b"] and erg.spaces == ["a", "b"]


def test_c2b_ohne_erstbefuellung_bleibt_der_lauf_ohne_neubau(fx):
    _erst_start(fx)
    _ohne_aussagen_im_repo_graph(fx, "docs/spaces/001_e1.md")
    bauer = FakeManifestBauer()
    from test_doku_erstbefuellung import _tag
    erg = _tick(fx, jetzt=_tag(), bauer=bauer)  # am Tag: nicht faellig -> kein Neubau, kein LLM
    assert not erg.erst_faellig and bauer.gebaut == [] and erg.llm_aufrufe == 0


# ---------- I4: doku_ziel_fremd einmal je Datei ----------

def test_i4_doku_ziel_fremd_drei_inhalte_ein_eintrag(fx):
    from test_doku_tick import _code_aenderung, _pruefliste
    from test_doku_ziele import _doku, _zwei_ziele

    _zwei_ziele(fx)
    ziel = fx.repo / "docs/spaces/001_x.md"
    for i in range(3):
        ziel.write_bytes(_doku("X", AUSSAGE, f"Fremde Notiz Nummer {i}, noch nicht committet.").encode("utf-8"))
        _code_aenderung(fx, i)
        from test_doku_tick import _anstoss
        _anstoss(fx)
        erg = _tick(fx)
        assert erg.doku_uebersprungen == ["docs/spaces/001_x.md"]
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "doku_ziel_fremd"]
    assert len(punkte) == 1 and punkte[0]["datei"] == "docs/spaces/001_x.md"
    assert punkte[0]["aussage_id"] == "docs/spaces/001_x.md"


# ---------- M4: Budget-Vorpruefung erst nach dem Anstoss ----------

class _StandZaehler:
    def __init__(self):
        self.tagesstand_aufrufe = 0
        self.pruefungen: list[str] = []

    def pruefen(self, agent, *, jetzt=None, umgebung=None):
        self.pruefungen.append(agent)
        from types import SimpleNamespace
        return SimpleNamespace(erlaubt=True, grund="ok", agent=agent, heute=0, grenze=150)

    def buchen(self, agent, ergebnis, dauer_s, *, jetzt=None, umgebung=None):
        pass

    def tagesstand(self, *, jetzt=None, umgebung=None):
        self.tagesstand_aufrufe += 1
        return {"datum": "2026-10-29", "user_aktiv": False, "not_aus": False, "agenten": []}


def test_m4_ohne_anstoss_kein_tagesstand(fx):
    from test_doku_erstbefuellung import Budget, _nacht

    _erst_start(fx)
    erste = _nachtlauf(fx, budget=Budget(erlaubt_bis=2))  # erster Nachtlauf (mit Marker), Budget nach 2 Aufrufen aus
    assert erste.erst_faellig and not tk.stand_laden(fx.konfig).erstbefuellung_fertig  # Erstbefuellung bleibt offen
    budget = _StandZaehler()
    erg = _tick(fx, jetzt=_nacht(1), budget=budget, laufzeit=Abo())  # kein Marker, 1 min spaeter, HEAD gleich
    assert erg.uebersprungen == "kein_anstoss"
    assert budget.tagesstand_aufrufe == 0 and budget.pruefungen == []


def test_m4_mit_anstoss_im_fenster_wird_vorgeprueft(fx):
    _erst_start(fx)
    budget = _StandZaehler()
    erg = _nachtlauf(fx, budget=budget)
    assert erg.erst_faellig and budget.tagesstand_aufrufe >= 1


# ---------- M3: erzaehlte Seiten robust gegen Abbruch und Zeitnot ----------

class _Abbruch(BaseException):
    """Wie ein Kill mitten im Lauf: keine Ausnahme, die irgendein `except Exception` auffaengt."""


def _notizen(tmp_path, version, pruefpunkte):
    from selbstbild.doku import rowboat as rb
    zusatz = [rb.Notiz(f"Spaces/{n}.md", f"---\nerzeugt_von: selbstbild\n---\n# {n}\n\nFassung {version}.\n")
              for n in ("A", "B")]
    return rb.notizen_schreiben(Graph({}, {}), [], [], tmp_path / "wissen", "Doku", f"L{version}",
                                manifest_pfad=tmp_path / "notizen.json", zusatz=zusatz, pruefpunkte=pruefpunkte)


def test_m3_abbruch_nach_einer_seite_macht_sie_nicht_zur_hand_seite(tmp_path, monkeypatch):
    from selbstbild.doku import rowboat as rb

    _notizen(tmp_path, 1, [])
    echt = rb._atomar

    def bricht_bei_b(pfad, daten):
        if pfad.name == "B.md":
            raise _Abbruch()
        echt(pfad, daten)

    monkeypatch.setattr(rb, "_atomar", bricht_bei_b)
    with pytest.raises(_Abbruch):
        _notizen(tmp_path, 2, [])  # A in Fassung 2 geschrieben, dann "Kill"
    monkeypatch.setattr(rb, "_atomar", echt)
    assert "Fassung 2" in (tmp_path / "wissen/Doku/Spaces/A.md").read_text(encoding="utf-8")
    punkte: list = []
    geschrieben = _notizen(tmp_path, 3, punkte)
    assert [p for p in punkte if p.art == "notiz_von_hand"] == []
    assert {p.name for p in geschrieben} == {"A.md", "B.md"}
    assert "Fassung 3" in (tmp_path / "wissen/Doku/Spaces/A.md").read_text(encoding="utf-8")


def test_m3_seite_nur_mit_restzeit_fuer_einen_ganzen_aufruf(fx):
    from test_doku_rowboat_erzaehlt import ErstBudget, TickAbo, _ohne_erst
    from test_doku_tick import _anstoss, _Uhr

    _ohne_erst(fx)
    knapp = fx.konfig.tick_deadline_min * 60 - jg.TIMEOUT_S - 1  # vor der Deadline, aber zu wenig fuer einen Aufruf
    abo = TickAbo()
    _anstoss(fx)
    erg = _tick(fx, erzwingen=True, budget=ErstBudget(), laufzeit=abo, uhr=_Uhr(knapp))
    assert "seite" not in abo.arten() and erg.erzaehlt_aufrufe == 0 and erg.erzaehlt_faellig
    abo2 = TickAbo()
    _anstoss(fx)
    erg2 = _tick(fx, erzwingen=True, budget=ErstBudget(), laufzeit=abo2, uhr=_Uhr(60))
    assert abo2.arten().count("seite") == 2 and erg2.erzaehlt_aufrufe == 2


# ---------- M1: Rueckweg entfernt Aussagen ausserhalb des alten Doku-Satzes ----------

KETTEN_DOC = "docs/spaces/KETTEN.md"
SPACE_DOC = "docs/spaces/001_x.md"


def _graph_mit_space_aussagen() -> Graph:
    from selbstbild.doku.belegspeicher import aussage_speichern
    from selbstbild.doku.graph import Knoten

    g = Graph({}, {}, "c" * 40, "d" * 40)
    g.knoten_hinzu(Knoten("space:a", "space", "a"))
    for datei in ("sp/a/client.py", "sp/b/api.py"):
        g.knoten_hinzu(Knoten(f"modul:{datei}", "modul", datei, datei=datei))
    b_client = Beleg("symbol", "sp/a/client.py", "client.senden", "d" * 64, "c", 7)
    b_api = Beleg("symbol", "sp/b/api.py", "api.x", "e" * 64, "c", 7)
    aussage_speichern(g, Aussage(f"{KETTEN_DOC}#K#1", KETTEN_DOC, "K", 3, 3, "Ketten-Satz.", ()), [b_client],
                      "extraktor")
    aussage_speichern(g, Aussage(f"{SPACE_DOC}#X#1", SPACE_DOC, "X", 3, 3, "Space-Satz.", ()), [b_client, b_api],
                      "extraktor")
    return g


def test_m1_alter_doku_satz_aus_der_alten_konfig(tmp_path):
    from selbstbild.doku import aufgabe as ag

    satz, quelle = ag.alter_doku_satz(tmp_path)
    assert satz == frozenset({"docs/spaces/ketten.md"}) and "Vorgabe" in quelle
    (tmp_path / "config").mkdir()
    (tmp_path / "config/doku_loop.yaml").write_text(
        "ketten_md: docs/spaces/KETTEN.md\nweitere_doku: [docs\\spaces\\README.md, ./docs/x.md]\n", encoding="utf-8")
    satz, quelle = ag.alter_doku_satz(tmp_path)
    assert satz == frozenset({"docs/spaces/ketten.md", "docs/spaces/readme.md", "docs/x.md"})
    assert quelle == "config/doku_loop.yaml"
    (tmp_path / "config/doku_loop.yaml").write_text("ketten_md: [kaputt\n", encoding="utf-8")
    satz, quelle = ag.alter_doku_satz(tmp_path)
    assert satz == frozenset({"docs/spaces/ketten.md"}) and "Vorgabe" in quelle


def test_m1_graph_bereinigen_entfernt_aussagen_ausserhalb_und_verwaiste_module(monkeypatch):
    from selbstbild.doku import aufgabe as ag
    from test_doku_aufgabe import ERZEUGT, _alter_graph

    alt_mod = _alter_graph(monkeypatch)
    text = ag._graph_json_text(_graph_mit_space_aussagen(), ERZEUGT)
    plan = ag.graph_bereinigen_plan(text.encode("utf-8"), ag.ALTE_TYPEN_KOPIE,
                                    doku_satz=frozenset({"docs/spaces/ketten.md"}))
    assert plan.knoten_weg == {"aussage": 1, "modul": 1} and plan.kanten_weg == {"belegt_durch": 2}
    assert plan.aussagen_fremd == 1 and plan.module_verwaist == 1
    g = alt_mod.Graph.aus_text(plan.neu)  # der alte Loop laedt das Ergebnis
    aussagen = {k.id: k.datei for k in g.knoten.values() if k.typ == "aussage"}
    assert aussagen == {f"aussage:{KETTEN_DOC}#K#1": KETTEN_DOC}
    assert "modul:sp/a/client.py" in g.knoten and "modul:sp/b/api.py" not in g.knoten
    assert {e.von for e in g.kanten.values()} == {f"aussage:{KETTEN_DOC}#K#1"}
    # wiederholbar: nichts mehr zu tun; ohne Doku-Satz (alter Aufruf) bleiben Aussagen unangetastet
    assert ag.graph_bereinigen_plan(plan.neu.encode("utf-8"), ag.ALTE_TYPEN_KOPIE,
                                    doku_satz=frozenset({"docs/spaces/ketten.md"})).neu is None
    assert ag.graph_bereinigen_plan(text.encode("utf-8"), ag.ALTE_TYPEN_KOPIE).neu is None


@pytest.fixture()
def umg_aussagen(umg, git_repo):
    from test_doku_aufgabe import ERZEUGT

    import dataclasses
    u = umg
    git_repo.git("checkout", "-q", "-b", u.k.zweig or "main")
    _graph_mit_space_aussagen().speichern(u.repo / "docs" / "graph" / "graph.json", ERZEUGT)
    (u.repo / ".gitignore").write_bytes(b"logs/\n")
    git_repo.commit("seed")
    u.k = dataclasses.replace(u.k, torwaechter=())
    u.git = git_repo
    return u


def test_m1_rueckweg_pruefen_trocken_und_echt(umg_aussagen, monkeypatch):
    from selbstbild.doku import aufgabe as ag
    from test_doku_aufgabe import _alter_graph, _um, _zu

    u = umg_aussagen
    alt_mod = _alter_graph(monkeypatch)
    zeilen: list[str] = []
    assert ag.pruefen(u.k, backend=u.backend, aus=zeilen.append) == 0
    assert "Aussagen außerhalb des alten Doku-Satzes: 1" in "\n".join(zeilen)
    assert _um(u)[0] == 0
    graph = u.repo / "docs" / "graph" / "graph.json"
    vorher = graph.read_bytes()
    rc, text = _zu(u, trocken=True)
    assert rc == 0, text
    assert "Aussagen außerhalb des alten Doku-Satzes: 1" in text and "verwaiste Module: 1" in text
    assert graph.read_bytes() == vorher
    rc, text = _zu(u)
    assert rc == 0, text
    g = alt_mod.Graph.aus_text(graph.read_text(encoding="utf-8"))
    assert {k.datei for k in g.knoten.values() if k.typ == "aussage"} == {KETTEN_DOC}
    assert u.git.git("show", "--name-only", "--format=", "HEAD").splitlines() == ["docs/graph/graph.json"]


# ---------- I5: Struktur-Hinweise in Pruefliste.md gruppiert ----------

def _pl_notiz(struktur, punkte=()):
    from selbstbild.doku import rowboat as rb
    return rb._pruefliste_notiz("Doku", list(punkte), [], commits={}, spitzen={}, vermerke=[], struktur=struktur,
                                lauf="2026-11-03T01:00:00#7", urteile_datei="Doku-Urteile/Urteile.md").text


def _struktur_500():
    arten = {"anker_unsicher": 260, "beleg_unsicher": 140, "env_unbekannt": 70, "symbol_mehrdeutig": 30}
    aus = []
    for art, n in arten.items():
        for i in range(n):
            datei = f"docs/spaces/{i % 23:03d}_space.md" if art != "env_unbekannt" else "docs/spaces/KETTEN.md"
            aus.append({"art": art, "text": f"Hinweis {art} Nummer {i} mit etwas Text", "datei": datei,
                        "zeile": 10 + i, "aussage_id": None, "lauf": "x"})
    return aus, arten


def test_i5_struktur_gruppiert_nach_art_und_datei():
    struktur, arten = _struktur_500()
    offen = [{"art": "judge_unklar", "text": "unklar", "datei": "docs/spaces/KETTEN.md", "zeile": 5,
              "aussage_id": "docs/spaces/KETTEN.md#K#1", "lauf": "2026-11-03T01:00:00#7"}]
    text = _pl_notiz(struktur, offen)
    abschnitt = text.split("## Struktur-Hinweise", 1)[1]
    zeilen = abschnitt.splitlines()
    assert len(zeilen) < 150, len(zeilen)
    assert zeilen[0] == " (500)"
    for art, n in arten.items():
        assert f"### {art} ({n}" in abschnitt, art
    # Summen je Art stimmen auch ueber die Dateizeilen (gezeigte + "weitere Dateien")
    for art, n in arten.items():
        teil = abschnitt.split(f"### {art} (", 1)[1].split("\n### ", 1)[0]
        gezeigt = sum(int(m) for m in re.findall(r"^- `[^`]+` \((\d+)\)", teil, re.M))
        rest = sum(int(m) for m in re.findall(r"weitere Datei\(en\) mit (\d+) Hinweis", teil))
        assert gezeigt + rest == n, art
    # hoechstens 3 Beispiele je Datei
    for z in zeilen:
        if z.startswith("- `"):
            assert z.count("Zeile ") <= 3, z
    # die offenen Punkte bleiben, wie sie sind
    assert "### judge_unklar: docs/spaces/KETTEN.md#K#1" in text


def test_i5_wenig_struktur_bleibt_lesbar_und_geschwaerzt():
    struktur = [{"art": "anker_unsicher", "text": "zu grob", "datei": "docs/spaces/KETTEN.md", "zeile": 3,
                 "aussage_id": None}]
    text = _pl_notiz(struktur)
    assert "## Struktur-Hinweise (1)" in text and "### anker_unsicher (1 in 1 Datei(en))" in text
    assert "- `docs/spaces/KETTEN.md` (1): Zeile 3: zu grob" in text
    assert "Keine." in _pl_notiz([]).split("## Struktur-Hinweise", 1)[1]
    # ein Schluessel kurz vor der Kuerzungsgrenze: erst geschwaerzt, dann gekuerzt -- kein angeschnittener Rest
    lang = [{"art": "anker_unsicher", "text": "x" * 120 + " " + GHP, "datei": "d.md", "zeile": 1, "aussage_id": None}]
    text = _pl_notiz(lang)
    assert "ghp_" not in text and GHP[4:14] not in text
