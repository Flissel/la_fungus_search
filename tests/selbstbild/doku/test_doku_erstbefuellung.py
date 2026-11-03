"""Stufe 4, Task 7: Erstbefuellung (D10, Spec 12a, Rulings T7-a/T7-b, Plan-Ruling P3).

Jede signaltragend gebundene Aussage wird einmal mit `zweimal_einig` unter dem eigenen Budget-Agenten
`selbstbild-erstbefuellung` geprueft -- nur im Fenster (Vorgabe 00:00 bis 06:00 Ortszeit), nach den echten Signalen
im selben Tick, mit Abbruch bei Budget, WIEDERHOLEN-Gruenden und Deadline; `veraltet` mit gueltigem Ersatz geht ueber
denselben Schreibweg wie ein Signal. Fixture wie `test_doku_tick.py`: git init unter tmp_path, Fake-Manifest-Bauer,
Fake-Judge (`laufzeit`), Fake-Budget, Torwaechter-Stub. Nie ein echtes `claude -p`.
"""

import dataclasses
import fnmatch
import json
import subprocess
from datetime import datetime, time, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from doku_vm import REPO, braucht_vm
from test_doku_anstoss_trocken import _alter_loop, _unveraendert, _zustand, ohne_judge  # noqa: F401
from test_doku_tick import (  # noqa: F401  (Fixture `fx`)
    AUSSAGE, CLIENT, KETTEN, FakeManifestBauer, _anstoss, _commit, _git, _head, _jetzt, _pruefliste, _schreib, _seed,
    _status, _submodul, _tick, fx,
)
from test_doku_ziele import README_SUB, _aussagen, _doku, _ziele
from selbstbild.doku import erstbefuellung as eb
from selbstbild.doku import judge as jg
from selbstbild.doku import stand as st
from selbstbild.doku import tick as tk
from selbstbild.doku.anker import Beleg
from selbstbild.doku.aussagen import Aussage
from selbstbild.doku.konfig import KonfigFehler, aus_regeln

AGENT = "selbstbild-erstbefuellung"
ERST = {"aktiv": True, "von": "00:00", "bis": "06:00", "agent": AGENT}
D1 = "Die Hilfe `helfer` in `sp/a/client.py:11` liefert immer die Eins an den Aufrufer zurueck."
D1_NEU = "Die Hilfe `helfer` in `sp/a/client.py:11` liefert immer die Eins an jeden Aufrufer zurueck."
D2 = "Die Hilfe `helfer` in `sp/a/client.py:11` wird von keinem anderen Space aufgerufen."
D3 = "Der Client ruft `senden` in `sp/a/client.py:7` auf und nennt dabei keinen Dienst."
GROB = "Das Modul `sp/a/client.py` enthaelt den ganzen Client des Space a."
NOCH_WAHR = json.dumps({"urteil": "noch_wahr", "neuer_text": None, "beleg_anker": None, "begruendung": "passt"})


def _veraltet(text, anker):
    return json.dumps({"urteil": "veraltet", "neuer_text": text, "beleg_anker": anker, "begruendung": "anders"})


class Abo:
    """Fake fuer `claude -p`: antwortet je Prompt (`antwort(prompt)`), merkt Prompts und Arbeitsordner."""

    def __init__(self, antwort=lambda prompt: NOCH_WAHR):
        self.antwort = antwort
        self.prompts: list[str] = []
        self.cwds: list[str] = []

    def __call__(self, argv, **kw):
        self.prompts.append(kw["input"])
        self.cwds.append(kw.get("cwd"))
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"result": self.antwort(kw["input"])}),
                                           stderr="")

    def aussagen(self):
        return [_aussage_im_prompt(p) for p in self.prompts]


def _aussage_im_prompt(prompt):
    return prompt.split("<<<AUSSAGE\n", 1)[1].split("\nAUSSAGE>>>", 1)[0]


class Budget:
    """Budget-Waechter-Fake: erlaubt `erlaubt_bis` Pruefungen (None = alle); jede erlaubte Pruefung zaehlt wie beim
    echten Waechter als Aufruf des Tages."""

    def __init__(self, erlaubt_bis=None):
        self.grenze = erlaubt_bis
        self.pruefungen: list[str] = []
        self.erlaubt: list[str] = []
        self.buchungen: list[tuple[str, str]] = []

    def pruefen(self, agent, *, jetzt=None, umgebung=None):
        ok = self.grenze is None or len(self.erlaubt) < self.grenze
        self.pruefungen.append(agent)
        if ok:
            self.erlaubt.append(agent)
        return SimpleNamespace(erlaubt=ok, grund="ok" if ok else "tagesgrenze", agent=agent, heute=len(self.erlaubt),
                               grenze=self.grenze or 0)

    def buchen(self, agent, ergebnis, dauer_s, *, jetzt=None, umgebung=None):
        self.buchungen.append((agent, ergebnis))


def _nacht(minute=0):
    """Ortszeit 01:00 + `minute` am 29.10. (nach allen `_jetzt()`-Zeiten des Fixtures), zeitzonenbewusst."""
    return datetime(2026, 10, 29, 1, 0).astimezone() + timedelta(minutes=minute)


def _tag(minute=0):
    return datetime(2026, 10, 29, 12, 0).astimezone() + timedelta(minutes=minute)


def _start(fx, saetze=(D1, D2, D3), *, erst=ERST, **ziele):
    """Doku-Ziele 001.. je mit einem Absatz, Erstbefuellung konfiguriert, Erstlauf am Tag merkt die Belege."""
    _ziele(fx, doku_ziele=("docs/spaces/0*.md",), erstbefuellung=dict(erst), **ziele)
    _seed(fx)
    for i, satz in enumerate(saetze, start=1):
        _schreib(fx.repo, f"docs/spaces/{i:03d}_e{i}.md", _doku(f"E{i}", satz, belege=""))
    if saetze:
        _commit(fx.repo, "ziele", "docs/spaces")
    _anstoss(fx)
    erg = _tick(fx, erzwingen=True, budget=Budget())
    assert erg.llm_aufrufe == 0 and erg.uebersprungen is None


def _nachtlauf(fx, minute=0, **kw):
    _anstoss(fx)
    kw.setdefault("budget", Budget())
    kw.setdefault("laufzeit", Abo())
    return _tick(fx, jetzt=_nacht(minute), **kw)


def _stand(fx):
    return st.stand_laden(fx.konfig)


def _id(fx, datei):
    (aid,) = [k[len("aussage:"):] for k in _aussagen(fx.repo, datei)]
    return aid


# ---------- Step 1: Fenster ----------

def test_nur_im_fenster(fx):
    k = dataclasses.replace(fx.konfig, erstbefuellung=dict(ERST))
    stand = st.Stand()
    tag = datetime(2026, 11, 4)
    assert eb.erstbefuellung_faellig(k, stand, tag.replace(hour=5, minute=59))
    assert eb.erstbefuellung_faellig(k, stand, tag.replace(hour=0, minute=0))
    assert eb.erstbefuellung_faellig(k, None, tag.replace(hour=3))  # noch kein Stand
    assert not eb.erstbefuellung_faellig(k, stand, tag.replace(hour=6, minute=0))
    assert not eb.erstbefuellung_faellig(k, stand, tag.replace(hour=23, minute=59))
    # fertig oder aus: nie
    assert not eb.erstbefuellung_faellig(k, st.Stand(erstbefuellung_fertig=True), tag.replace(hour=3))
    aus = dataclasses.replace(k, erstbefuellung={"aktiv": False})
    assert not eb.erstbefuellung_faellig(aus, stand, tag.replace(hour=3))
    # Vorgaben: ohne von/bis gilt 00:00 bis 06:00, ohne agent selbstbild-erstbefuellung
    vorgabe = dataclasses.replace(k, erstbefuellung={"aktiv": True})
    assert eb.einstellung(vorgabe) == eb.ErstEinstellung(True, time(0, 0), time(6, 0), AGENT)
    assert eb.erstbefuellung_faellig(vorgabe, stand, tag.replace(hour=5, minute=59))
    # ueber Mitternacht
    spaet = dataclasses.replace(k, erstbefuellung={"aktiv": True, "von": "22:30", "bis": "01:00"})
    assert eb.erstbefuellung_faellig(spaet, stand, tag.replace(hour=23))
    assert eb.erstbefuellung_faellig(spaet, stand, tag.replace(hour=0, minute=59))
    assert not eb.erstbefuellung_faellig(spaet, stand, tag.replace(hour=1))
    assert not eb.erstbefuellung_faellig(spaet, stand, tag.replace(hour=22, minute=29))
    # zeitzonenbewusst: es zaehlt die Ortszeit
    ort = tag.replace(hour=5, minute=59).astimezone()
    assert eb.erstbefuellung_faellig(k, stand, ort.astimezone(timezone.utc))
    assert not eb.erstbefuellung_faellig(k, stand, (ort + timedelta(minutes=1)).astimezone(timezone.utc))


@pytest.mark.parametrize("wert", [
    "{aktiv: true, von: '6:00'}", "{aktiv: true, bis: '24:00'}", "{aktiv: true, von: 600}", "{aktiv: true, agent: ''}",
    "{aktiv: true, agent: 3}", "{aktiv: true, fenster: x}", "{aktiv: true, von: '03:00', bis: '03:00'}",
])
def test_konfig_erstbefuellung_ungueltig(tmp_path, wert):
    from test_doku_konfig_regeln import _repo, _vorlage

    with pytest.raises(KonfigFehler):
        aus_regeln(_repo(tmp_path, _vorlage() + f"erstbefuellung: {wert}\n"))


def test_konfig_erstbefuellung_gueltig(tmp_path):
    from test_doku_konfig_regeln import _repo, _vorlage

    k = aus_regeln(_repo(tmp_path, _vorlage() + "erstbefuellung: {aktiv: true, von: '00:00', bis: '06:00', "
                                                "agent: selbstbild-erstbefuellung}\n"))
    assert k.erstbefuellung == ERST


# ---------- Step 2: Budget aus mitten drin ----------

def test_budget_aus_mitten_drin(fx):
    _start(fx)
    budget, abo = Budget(erlaubt_bis=3), Abo()
    kopf = _head(fx.repo)
    erg = _nachtlauf(fx, budget=budget, laufzeit=abo)
    # Aussage 1 mit zwei Aufrufen geprueft; Aussage 2: erster Aufruf erlaubt, zweiter abgelehnt -> nicht geprueft
    assert abo.aussagen() == [D1, D1, D2]
    assert erg.erst_geprueft == 1 and erg.erst_offen == 3 and erg.erst_aufrufe == 3  # D2, D3, KETTEN offen
    assert budget.erlaubt == [AGENT] * 3 and budget.pruefungen == [AGENT] * 4
    assert budget.buchungen == [(AGENT, "ok")] * 3  # die Ablehnung kostet keine Buchung
    assert erg.commit is None and _head(fx.repo) == kopf and _status(fx.repo) == []
    stand = _stand(fx)
    assert stand.erstgeprueft == [_id(fx, "docs/spaces/001_e1.md")] and stand.erstbefuellung_fertig is False
    # naechster Lauf (neuer Tag, Budget frei): weiter mit Aussage 2, Aussage 1 nie wieder (keine Doppelbuchung);
    # Aussage 2 wird zweimal neu gefragt -- der verworfene erste Aufruf von eben kostet genau eine Buchung extra
    budget2, abo2 = Budget(), Abo()
    erg2 = _nachtlauf(fx, 40, budget=budget2, laufzeit=abo2)
    assert abo2.aussagen() == [D2, D2, D3, D3, AUSSAGE, AUSSAGE]
    assert erg2.erst_geprueft == 3 and erg2.erst_offen == 0 and budget2.erlaubt == [AGENT] * 6
    assert _stand(fx).erstbefuellung_fertig is True and erg2.commit is None


def test_budget_aus_mitten_drin_direkt(fx):
    """Die Schnittstelle `erstbefuellung_schritt` allein: dieselbe Reihenfolge, Stand-Fortschritt ueber das Ergebnis."""
    k = dataclasses.replace(fx.konfig, erstbefuellung=dict(ERST))
    posten = [_posten(f"docs/spaces/00{i}_x.md", 3, f"Aussage {i} nennt `f{i}`.") for i in (2, 1, 3)]
    stand = st.Stand()
    budget, abo = Budget(erlaubt_bis=3), Abo()
    erg = eb.erstbefuellung_schritt(k, None, stand, budget=budget, laufzeit=abo, deadline=lambda: False,
                                    posten=posten, quelltext=lambda b: "def f():\n    return 1\n")
    assert [a.split()[1] for a in abo.aussagen()] == ["1", "1", "2"]  # nach Datei und Zeile sortiert
    assert (erg.geprueft, erg.offen, erg.fertig, erg.abbruch) == (1, 2, False, "budget:tagesgrenze")
    assert [u.posten.aussage.id for u in erg.urteile] == ["docs/spaces/001_x.md#a"]
    stand = eb.stand_fortschreiben(stand, erg)
    abo2 = Abo()
    erg2 = eb.erstbefuellung_schritt(k, None, stand, budget=Budget(), laufzeit=abo2, deadline=lambda: False,
                                     posten=posten, quelltext=lambda b: "x")
    assert [a.split()[1] for a in abo2.aussagen()] == ["2", "2", "3", "3"]
    assert (erg2.geprueft, erg2.offen, erg2.fertig) == (2, 0, True)
    # Plan-Ruling P3: offene Rowboat-Seiten (Task 9) halten `fertig` zurueck
    stand2 = eb.stand_fortschreiben(stand, erg2)
    erg3 = eb.erstbefuellung_schritt(k, None, stand2, budget=Budget(), laufzeit=Abo(), deadline=lambda: False,
                                     posten=posten, offene_seiten=2)
    assert (erg3.geprueft, erg3.offen, erg3.fertig) == (0, 0, False)


def _posten(datei, zeile, text):
    b = Beleg("symbol", "sp/a/client.py", f"client.{text.split('`')[1]}", "d" * 64, "c", 1)
    a = Aussage(f"{datei}#a", datei, "X", zeile, zeile, text, (b,))
    return eb.ErstPosten(a, b)


def test_deadline_und_wiederholen_brechen_ab(fx):
    k = dataclasses.replace(fx.konfig, erstbefuellung=dict(ERST))
    posten = [_posten(f"docs/spaces/00{i}_x.md", 3, f"Aussage {i} nennt `f{i}`.") for i in (1, 2)]
    uhr = iter([False, True])
    abo = Abo()
    erg = eb.erstbefuellung_schritt(k, None, st.Stand(), budget=Budget(), laufzeit=abo, deadline=lambda: next(uhr),
                                    posten=posten, quelltext=lambda b: "x")
    assert len(abo.prompts) == 2 and (erg.geprueft, erg.offen, erg.abbruch) == (1, 1, "deadline")

    def kaputt(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 180)

    erg2 = eb.erstbefuellung_schritt(k, None, st.Stand(), budget=Budget(), laufzeit=kaputt, deadline=lambda: False,
                                     posten=posten, quelltext=lambda b: "x")
    # Aussage 1 kommt aus dem Cache (kein Aufruf), Aussage 2 scheitert technisch und bleibt offen
    assert (erg2.geprueft, erg2.offen, erg2.abbruch) == (1, 1, "timeout")
    assert [u.aus_cache for u in erg2.urteile] == [True]


# ---------- Step 3: endet von selbst ----------

def test_endet_von_selbst(fx):
    _start(fx)
    budget, abo = Budget(), Abo()
    erg = _nachtlauf(fx, budget=budget, laufzeit=abo)
    assert abo.aussagen() == [D1, D1, D2, D2, D3, D3, AUSSAGE, AUSSAGE]
    assert erg.erst_geprueft == 4 and erg.erst_offen == 0 and erg.erst_faellig
    stand = _stand(fx)
    assert stand.erstbefuellung_fertig is True and len(stand.erstgeprueft) == 4
    assert not eb.erstbefuellung_faellig(fx.konfig, stand, _nacht(30))
    # danach ruft kein Lauf mehr den Agenten der Erstbefuellung -- auch nicht erzwungen und nach einer neuen Aussage
    _schreib(fx.repo, "docs/spaces/004_e4.md", _doku("E4", "Die Hilfe `helfer` in `sp/a/client.py:11` ist kurz."))
    _commit(fx.repo, "neue Aussage", "docs/spaces")
    for i, erzwingen in enumerate((False, True, True), start=1):
        budget, abo = Budget(), Abo()
        erg = _nachtlauf(fx, 40 * i, budget=budget, laufzeit=abo, erzwingen=erzwingen)
        assert AGENT not in budget.pruefungen and abo.prompts == [] and not erg.erst_faellig


# ---------- Step 4: Ruhe ausserhalb des Fensters ----------

def test_ruhe_ohne_aenderung_ausserhalb_fenster(fx):
    _start(fx)
    kopf = _head(fx.repo)
    budget, abo = Budget(), Abo()
    aufrufe = 0
    for i in range(10):
        if i % 2:
            _anstoss(fx)
        erg = _tick(fx, jetzt=_tag(40 * i), budget=budget, laufzeit=abo, erzwingen=i % 3 == 0)
        aufrufe += erg.llm_aufrufe
        assert erg.commit is None and not erg.erst_faellig
    assert aufrufe == 0 and abo.prompts == [] and budget.pruefungen == []
    assert _head(fx.repo) == kopf and _status(fx.repo) == []
    assert _stand(fx).erstgeprueft == [] and _stand(fx).erstbefuellung_fertig is False


# ---------- Schreibweg, Auswahl, Sonderfaelle ----------

def test_veraltet_ersetzt_wie_ein_signal(fx):
    _start(fx)
    alt_id = _id(fx, "docs/spaces/001_e1.md")
    abo = Abo(lambda p: _veraltet(D1_NEU, "client.helfer") if _aussage_im_prompt(p) == D1 else NOCH_WAHR)
    kopf = _head(fx.repo)
    erg = _nachtlauf(fx, laufzeit=abo)
    assert erg.umgeschrieben == 1 and erg.commit is not None and erg.commit != kopf
    assert D1_NEU in (fx.repo / "docs/spaces/001_e1.md").read_text(encoding="utf-8")
    geaendert = set(_git(fx.repo, "show", "--name-only", "--format=", "HEAD").split())
    assert geaendert == {"docs/spaces/001_e1.md", "docs/graph/graph.json", "docs/graph/ledger.jsonl"}
    (eintrag,) = [json.loads(z) for z in (fx.repo / "docs/graph/ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert eintrag["aussage_id"] == alt_id and eintrag["urteil"] == "veraltet" and eintrag["quelle"] == "llm"
    assert eintrag["neu"] == D1_NEU and eintrag["beleg"]["anker"] == "client.helfer"
    assert "docs(loop)" in _git(fx.repo, "log", "-1", "--format=%s") and _status(fx.repo) == []
    neu_id = _id(fx, "docs/spaces/001_e1.md")
    assert neu_id != alt_id
    stand = _stand(fx)
    assert {alt_id, neu_id} <= set(stand.erstgeprueft) and stand.erstbefuellung_fertig is True
    # der umgeschriebene Satz gilt als geprueft: kein weiterer Aufruf
    abo2 = Abo()
    assert _nachtlauf(fx, 40, laufzeit=abo2).commit is None and abo2.prompts == []


def test_veraltet_ohne_commit_wird_nicht_als_geprueft_gemerkt(fx):
    """Blockiert der Torwaechter den Commit, bleibt die Aussage offen; das Urteil steht im Cache (kein neuer Aufruf)."""
    _start(fx)
    alt_id = _id(fx, "docs/spaces/001_e1.md")
    abo = Abo(lambda p: _veraltet(D1_NEU, "client.helfer") if _aussage_im_prompt(p) == D1 else NOCH_WAHR)
    zaehler = iter(range(100))

    def drift(repo):  # vorher sauber, nachher ein neuer Fehler -> Ruecksetzen, kein Commit
        return set() if next(zaehler) == 0 else {("docs/spaces/001_e1.md", "neu")}

    kopf = _head(fx.repo)
    erg = _nachtlauf(fx, laufzeit=abo, drift=drift)
    assert erg.commit is None and _head(fx.repo) == kopf and _status(fx.repo) == []
    stand = _stand(fx)
    assert alt_id not in stand.erstgeprueft and stand.erstbefuellung_fertig is False
    abo2 = Abo()
    erg2 = _nachtlauf(fx, 40, laufzeit=abo2)
    assert abo2.prompts == [] and erg2.commit is not None  # Urteil aus dem Cache, jetzt geschrieben
    assert alt_id in _stand(fx).erstgeprueft and _stand(fx).erstbefuellung_fertig is True


def test_nur_ganzdatei_anker_auf_code_nicht_geprueft(fx):
    """Ruling T7-a: eine Aussage, die nur ein ganze-Datei-Anker auf Code bindet, wird nicht erstgeprueft."""
    _start(fx, (D1, GROB))
    grob = _id(fx, "docs/spaces/002_e2.md")
    from test_doku_ziele import _belege_von

    assert [art for _, art, *_ in _belege_von(fx.repo, f"aussage:{grob}")] == ["datei"]
    abo = Abo()
    erg = _nachtlauf(fx, laufzeit=abo)
    assert abo.aussagen() == [D1, D1, AUSSAGE, AUSSAGE]
    assert erg.erst_offen == 0 and _stand(fx).erstbefuellung_fertig is True and grob not in _stand(fx).erstgeprueft


def test_nur_lesend_veraltet_ohne_schreiben(fx, tmp_path):
    _submodul(fx, tmp_path, "ext/doku", {"README.md": README_SUB})
    _start(fx, (), doku_nur_lesen=("ext/*/README.md",))
    neu = "Die Hilfe `helfer` in `sp/a/client.py:11` liefert immer genau die Eins an den Aufrufer zurueck."
    abo = Abo(lambda p: _veraltet(neu, "client.helfer") if "helfer" in _aussage_im_prompt(p) else NOCH_WAHR)
    readme = (fx.repo / "ext/doku/README.md").read_bytes()
    erg = _nachtlauf(fx, laufzeit=abo)
    assert len(abo.prompts) == 4 and erg.umgeschrieben == 0 and erg.commit is None
    assert (fx.repo / "ext/doku/README.md").read_bytes() == readme and _status(fx.repo) == []
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "nur_lesend_veraltet"]
    assert len(punkte) == 1 and punkte[0]["datei"] == "ext/doku/README.md" and "genau die Eins" in punkte[0]["text"]
    assert _stand(fx).erstbefuellung_fertig is True and len(_stand(fx).erstgeprueft) == 2


def test_fremd_geaenderte_doku_wird_uebersprungen(fx):
    _start(fx)
    pfad = fx.repo / "docs/spaces/002_e2.md"
    roh = pfad.read_bytes()
    pfad.write_bytes(roh + b"\nVon Hand ergaenzt.\n")
    abo = Abo()
    erg = _nachtlauf(fx, laufzeit=abo)
    assert D2 not in abo.aussagen() and abo.aussagen() == [D1, D1, D3, D3, AUSSAGE, AUSSAGE]
    assert erg.doku_uebersprungen == ["docs/spaces/002_e2.md"]
    assert _stand(fx).erstbefuellung_fertig is False  # die Aussagen der ausgelassenen Datei sind unbekannt
    pfad.write_bytes(roh)
    abo2 = Abo()
    erg2 = _nachtlauf(fx, 40, laufzeit=abo2)
    assert abo2.aussagen() == [D2, D2] and erg2.erst_offen == 0 and _stand(fx).erstbefuellung_fertig is True


def test_uneinig_ist_geprueft_mit_pruefpunkt(fx):
    _start(fx, (D1,))
    antworten = iter([NOCH_WAHR, json.dumps({"urteil": "unklar", "neuer_text": None, "beleg_anker": None,
                                              "begruendung": "x"})])
    abo = Abo(lambda p: next(antworten, NOCH_WAHR))
    erg = _nachtlauf(fx, laufzeit=abo)
    assert len(abo.prompts) == 4 and erg.erst_geprueft == 2
    punkte = [p for p in _pruefliste(fx)["offen"] if p["art"] == "judge_uneinig"]
    assert len(punkte) == 1 and punkte[0]["datei"] == "docs/spaces/001_e1.md" and "Erstprüfung" in punkte[0]["text"]
    assert _stand(fx).erstbefuellung_fertig is True


def test_echte_signale_zuerst_keine_doppelbuchung(fx):
    _start(fx)
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("return 1", "return 2"))  # nur `helfer` (D1, D2)
    _commit(fx.repo, "helfer", "sp/a/client.py")
    budget, abo = Budget(), Abo()
    erg = _nachtlauf(fx, budget=budget, laufzeit=abo)
    # D1 und D2 haben ein echtes Signal: zuerst, unter doku-loop; die Erstbefuellung fragt nur den Rest
    assert budget.pruefungen == ["doku-loop"] * 4 + [AGENT] * 4
    assert abo.aussagen() == [D1, D1, D2, D2, D3, D3, AUSSAGE, AUSSAGE]
    signal, erst = abo.prompts[:4], abo.prompts[4:]
    assert all("nach der Aenderung" in p for p in signal) and not any("nach der Aenderung" in p for p in erst)
    assert all("Stimmt die Aussage zum Code?" in p for p in erst)
    # ein Signal-Urteil prueft die Aussage gegen den heutigen Code: sie gilt als erstgeprueft
    assert erg.erst_offen == 0 and _stand(fx).erstbefuellung_fertig is True


class AgentBudget(Budget):
    """Lehnt nur einen Agenten ab (hier: doku-loop hat sein Tageslimit erreicht)."""

    def __init__(self, gesperrt):
        super().__init__()
        self.gesperrt = gesperrt

    def pruefen(self, agent, *, jetzt=None, umgebung=None):
        if agent == self.gesperrt:
            self.pruefungen.append(agent)
            return SimpleNamespace(erlaubt=False, grund="tagesgrenze", agent=agent, heute=30, grenze=30)
        return super().pruefen(agent, jetzt=jetzt, umgebung=umgebung)


def test_unbeantwortetes_signal_nicht_unter_dem_erst_agenten(fx):
    """Lehnt das Budget das echte Signal ab (doku-loop am Limit), bleibt das Signal offen -- die Erstbefuellung fragt
    dieselbe Aussage im selben Lauf NICHT unter ihrem eigenen Agenten (keine Umgehung des doku-loop-Limits)."""
    _start(fx)
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("return 1", "return 2"))  # Signal an D1 und D2
    _commit(fx.repo, "helfer", "sp/a/client.py")
    budget, abo = AgentBudget("doku-loop"), Abo()
    erg = _nachtlauf(fx, budget=budget, laufzeit=abo)
    assert abo.aussagen() == [D3, D3, AUSSAGE, AUSSAGE]
    assert budget.erlaubt == [AGENT] * 4 and "doku-loop" in budget.pruefungen
    assert erg.erst_offen == 2 and _stand(fx).erstbefuellung_fertig is False
    assert len(_stand(fx).offene_signale) == 2


def test_erstpruefung_im_aufruf_ordner_des_judge(fx):
    _start(fx, (D1,))
    abo = Abo()
    _nachtlauf(fx, laufzeit=abo)
    assert abo.cwds and all(c == str(jg.aufruf_ordner()) for c in abo.cwds)
    assert all(fnmatch.fnmatch(jg.projekt_slug(jg.aufruf_ordner()), "*vibemind-claude-invocation*") for _ in abo.cwds)


def test_ohne_llm_keine_erstbefuellung(fx):
    _start(fx)
    budget, abo = Budget(), Abo()
    erg = _nachtlauf(fx, budget=budget, laufzeit=abo, ohne_llm=True)
    assert abo.prompts == [] and budget.pruefungen == [] and erg.erst_aufrufe == 0
    assert _stand(fx).erstgeprueft == []


# ---------- Urteils-Cache ----------

def test_cache_erstpruefung_getrennt_von_signalen(fx):
    """Ein Erst-Urteil gilt fuer genau den heutigen Code-Stand (alt = neu = heutiger Digest) und liegt unter einem
    eigenen Schluessel: ein spaeteres echtes Signal auf demselben Anker (alt = derselbe Digest, neu = anderer) fragt
    neu, und ein Signal-Eintrag beantwortet nie eine Erstpruefung."""
    k = fx.konfig
    schl = ("symbol", "sp/a/client.py", "client.helfer")
    st.urteil_merken(k, "a#1", schl, "D0", "D0", {"urteil": "noch_wahr"}, fragetyp="erstpruefung")
    assert st.urteil_holen(k, "a#1", schl, "D0", "D0", fragetyp="erstpruefung")["urteil"] == "noch_wahr"
    assert st.urteil_holen(k, "a#1", schl, "D0", "D0") is None  # Signal-Schluessel: anderer Eintrag
    assert st.urteil_holen(k, "a#1", schl, "D0", "D1") is None  # echte Aenderung D0 -> D1: neu fragen
    st.urteil_merken(k, "a#1", schl, "D0", "D1", {"urteil": "veraltet"})
    assert st.urteil_holen(k, "a#1", schl, "D0", "D1")["urteil"] == "veraltet"
    assert st.urteil_holen(k, "a#1", schl, "D0", "D0", fragetyp="erstpruefung")["urteil"] == "noch_wahr"
    # der alte Schluessel des Signal-Wegs bleibt byte-gleich (der alte Loop teilt sich urteile.jsonl)
    zeilen = [json.loads(z) for z in st.urteile_pfad(k).read_text(encoding="utf-8").splitlines()]
    assert zeilen[1]["schluessel"] == json.dumps(["a#1", *schl, "D0"], ensure_ascii=False)
    assert json.loads(zeilen[0]["schluessel"])[0] == "erstpruefung"


# ---------- Stand ----------

def test_stand_fortschritt_additiv():
    alt = {"letzter_commit_aussen": "abc", "lauf_nr": 3, "offene_signale": []}  # Stand ohne die neuen Felder
    s = st.Stand.from_dict(alt)
    assert s.erstgeprueft == [] and s.erstbefuellung_fertig is False
    neu = st.Stand(lauf_nr=4, erstgeprueft=["b", "a"], erstbefuellung_fertig=True).to_dict()
    assert neu["erstgeprueft"] == ["b", "a"] and neu["erstbefuellung_fertig"] is True
    s2 = st.Stand.from_dict({**neu, "erstgeprueft": ["a", 3, None], "erstbefuellung_fertig": "ja"})
    assert s2.erstgeprueft == ["a"] and s2.erstbefuellung_fertig is False  # fremde Typen: verworfen bzw. falsch


# ---------- Trockenlauf ----------

def test_trocken_zaehlt_offene_ohne_judge(fx, ohne_judge):
    _start(fx)
    _alter_loop(fx)
    vorher = _zustand(fx)
    nacht = tk.tick(fx.konfig, jetzt=_nacht(), manifest_bauer=FakeManifestBauer(), trocken=True)
    assert nacht.uebersprungen is None and nacht.erst_faellig and nacht.erst_offen == 4
    assert nacht.llm_aufrufe == 0 and nacht.erst_geprueft == 0 and nacht.erst_aufrufe == 0
    tag = tk.tick(fx.konfig, jetzt=_tag(), manifest_bauer=FakeManifestBauer(), trocken=True)
    assert not tag.erst_faellig and tag.erst_offen == 4
    _unveraendert(fx, vorher)


# ---------- Judge: Fragetyp erstpruefung ----------

def test_judge_erstpruefung_prompt_und_agent(fx):
    k = fx.konfig
    b = Beleg("symbol", "sp/m/api.py", "m.api.routen", "abc", "c", 1)
    a = Aussage("docs/spaces/001_x.md#X#abcd1234", "docs/spaces/001_x.md", "X", 3, 3,
                "Die Route `sp/m/api.py:1` ruft ziel_aufruf auf.", (b,))
    quelle = "def routen():\n    return ziel_aufruf()\n"
    frage = jg.JudgeFrage(a, b, quelle, quelle, (), fragetyp="erstpruefung")
    prompt = jg.prompt_bauen(frage)
    assert "Stimmt die Aussage zum Code?" in prompt and "nach der Aenderung" not in prompt
    assert prompt.count("<<<QUELLTEXT") == 1 and quelle.strip() in prompt
    for teil in ("<<<AUSSAGE", "<<<ANKER", "noch_wahr", "veraltet", "unklar", "beleg_anker", "Anweisung"):
        assert teil in prompt, teil
    # Signal-Prompt unveraendert: beide Quelltexte
    signal = jg.prompt_bauen(dataclasses.replace(frage, quelltext_neu="def routen():\n    return 2\n",
                                                 fragetyp="signal"))
    assert "<<<QUELLTEXT_ALT" in signal and "<<<QUELLTEXT_NEU" in signal
    budget, abo = Budget(), Abo()
    antwort, grund = jg.zweimal_einig(frage, k, laufzeit=abo, budget=budget, agent=AGENT)
    assert grund == "ok" and antwort.urteil == "noch_wahr"
    assert budget.pruefungen == [AGENT, AGENT] and budget.buchungen == [(AGENT, "ok"), (AGENT, "ok")]
    budget2 = Budget()
    jg.befragen(frage, k, laufzeit=Abo(), budget=budget2)
    assert budget2.pruefungen == [k.agent]  # ohne agent: der Agent der Konfig (doku-loop)
    with pytest.raises(ValueError):
        jg.JudgeFrage(a, b, quelle, quelle, (), fragetyp="unbekannt")


# ---------- echtes VibeMind (nur mit SELBSTBILD_VIBEMIND) ----------

@braucht_vm
def test_agent_erstbefuellung_in_budget_config():
    agenten = yaml.safe_load((REPO / "config" / "agent_budget.yaml").read_text(encoding="utf-8"))["agenten"]
    assert agenten[AGENT] == {"bereich": "betrieb", "klasse": "hintergrund", "modell": "sonnet", "tagesgrenze": 150,
                              "wrapper": "python"}


@braucht_vm
def test_regeldatei_erstbefuellung():
    k = aus_regeln(REPO)
    assert k.erstbefuellung == ERST
    assert eb.einstellung(k) == eb.ErstEinstellung(True, time(0, 0), time(6, 0), AGENT)


# ---------- Fix-Runde 1 ----------

PLAN = "Der Plan in `notes/plan.md` nennt die naechsten drei Schritte des Teams."


def test_offenes_signal_bleibt_ohne_erstpruefung_ueber_laeufe(fx):
    """Review I1: ein echtes Signal, das das doku-loop-Budget nicht beantworten durfte, bleibt offen -- auch im
    NAECHSTEN Lauf (ohne Codeaenderung, Space nicht neu gebaut) fragt die Erstbefuellung diese Aussagen nicht."""
    _start(fx)
    _schreib(fx.repo, "sp/a/client.py", CLIENT.replace("return 1", "return 2"))  # Signal an D1 und D2
    _commit(fx.repo, "helfer", "sp/a/client.py")
    b1, a1 = AgentBudget("doku-loop"), Abo()
    _nachtlauf(fx, budget=b1, laufzeit=a1)
    assert a1.aussagen() == [D3, D3, AUSSAGE, AUSSAGE] and len(_stand(fx).offene_signale) == 2
    b2, a2 = AgentBudget("doku-loop"), Abo()
    e2 = _nachtlauf(fx, 40, budget=b2, laufzeit=a2)
    assert a2.prompts == [] and AGENT not in b2.erlaubt and "doku-loop" in b2.pruefungen
    assert e2.erst_offen == 2 and _stand(fx).erstbefuellung_fertig is False
    assert len(_stand(fx).offene_signale) == 2


def test_befragter_anker_nach_art(fx):
    """Review I2: symbol > sql > yaml_key > datei mit Zeile > datei ohne Anker -- nicht der erste in der Liste."""
    k = _ziele(fx, doku_ziele=("docs/spaces/0*.md",))
    md = Beleg("datei", "notes/plan.md", "", "d1", "c", None)
    sym = Beleg("symbol", "sp/a/client.py", "client.helfer", "d2", "c", 11)
    yml = Beleg("yaml_key", "config/dienste.yaml", "dienste.a", "d3", "c", 2)
    a = Aussage("docs/spaces/001_x.md#X#1", "docs/spaces/001_x.md", "X", 3, 3, "Text.", ())
    (p,) = eb.posten_waehlen(k, [(a, (md, yml, sym), True, False)])
    assert p.beleg == sym and p.aussage.belege == (md, yml, sym)
    (p,) = eb.posten_waehlen(k, [(a, (md, yml), True, False)])
    assert p.beleg == yml


def test_nur_leere_anker_nicht_erstgeprueft(fx):
    """Ruling T7-c: traegt eine Aussage nur Belege ohne Ankernamen (ganze Datei, z. B. .md), kann der Judge kein
    gueltiges `veraltet` nennen -- sie wird nicht erstgeprueft und zaehlt nicht als offen."""
    k = _ziele(fx, doku_ziele=("docs/spaces/0*.md",))
    md = Beleg("datei", "notes/plan.md", "", "d1", "c", 4)
    a = Aussage("docs/spaces/001_x.md#X#1", "docs/spaces/001_x.md", "X", 3, 3, "Text.", ())
    assert eb.posten_waehlen(k, [(a, (md,), True, False)]) == []
    # im Tick: die Plan-Aussage ist gebunden (datei auf .md), wird aber nicht gefragt; trotzdem fertig
    _schreib(fx.repo, "notes/plan.md", "# Plan\n\nerster Stand\n")
    _commit(fx.repo, "plan", "notes/plan.md")
    _start(fx, (D1, PLAN))
    from test_doku_ziele import _belege_von

    plan = _id(fx, "docs/spaces/002_e2.md")
    assert [art for _, art, *_ in _belege_von(fx.repo, f"aussage:{plan}")] == ["datei"]
    abo = Abo()
    erg = _nachtlauf(fx, laufzeit=abo)
    assert abo.aussagen() == [D1, D1, AUSSAGE, AUSSAGE]
    assert erg.erst_offen == 0 and _stand(fx).erstbefuellung_fertig is True


class StandBudget(Budget):
    """Budget-Fake mit lesendem `tagesstand` (wie claude_budget.tagesstand)."""

    def __init__(self, *, heute=0, grenze=150, not_aus=False, user_aktiv=False, schalter=True, fehler=False):
        super().__init__()
        self.stand = {"datum": "2026-10-29", "user_aktiv": user_aktiv, "not_aus": not_aus,
                      "pause_aktiv_schalter": schalter,
                      "agenten": [{"agent": AGENT, "klasse": "hintergrund", "heute": heute, "grenze": grenze,
                                   "abgelehnt": {}}]}
        self.fehler = fehler
        self.tagesstaende = 0

    def tagesstand(self, *, jetzt=None, umgebung=None):
        self.tagesstaende += 1
        if self.fehler:
            raise OSError("kaputt")
        return self.stand


@pytest.mark.parametrize("kw", [{"heute": 150}, {"heute": 151}, {"not_aus": True}, {"user_aktiv": True}])
def test_budget_vorpruefung_spart_den_lauf(fx, kw):
    """Review M4: zeigt der lesende Tagesstand, dass der Agent heute nichts mehr darf, wird ohne Aenderung nichts
    gelesen und nichts gefragt (`keine_aenderung`); `pruefen` wird gar nicht erst gerufen."""
    _start(fx)
    budget, abo = StandBudget(**kw), Abo()
    erg = _nachtlauf(fx, budget=budget, laufzeit=abo)
    assert budget.tagesstaende == 1 and budget.pruefungen == [] and abo.prompts == []
    assert erg.uebersprungen == "keine_aenderung" and erg.erst_faellig and _stand(fx).erstgeprueft == []


@pytest.mark.parametrize("kw", [{"heute": 149}, {"user_aktiv": True, "schalter": False}, {"fehler": True}])
def test_budget_vorpruefung_laesst_pruefen_entscheiden(fx, kw):
    _start(fx, (D1,))
    budget, abo = StandBudget(**kw), Abo()
    erg = _nachtlauf(fx, budget=budget, laufzeit=abo)
    assert abo.aussagen() == [D1, D1, AUSSAGE, AUSSAGE] and budget.pruefungen == [AGENT] * 4
    assert erg.erst_geprueft == 2


def test_giftige_aussage_nach_zwei_fehlschlaegen(fx):
    """Review M5: zweimal hintereinander technisch gescheitert (Timeout/Exit) -> Pruefpunkt
    `erstpruefung_gescheitert` ohne Inhalt, als erstgeprueft markiert, weiter mit der naechsten; Budget-Ablehnungen
    zaehlen nicht."""
    _start(fx)

    class Haenger(Abo):
        def __call__(self, argv, **kw):
            if _aussage_im_prompt(kw["input"]) == D1:
                self.prompts.append(kw["input"])
                raise subprocess.TimeoutExpired(argv, 180)
            return super().__call__(argv, **kw)

    for i in range(3):  # Budget-Ablehnungen zaehlen nicht
        _nachtlauf(fx, i * 40, budget=Budget(erlaubt_bis=0))
    assert _stand(fx).erst_fehlversuche == {}
    abo = Haenger()
    erg = _nachtlauf(fx, 200, laufzeit=abo)
    d1 = _id(fx, "docs/spaces/001_e1.md")
    assert abo.aussagen() == [D1] and erg.erst_geprueft == 0 and _stand(fx).erst_fehlversuche == {d1: 1}
    abo2 = Haenger()
    erg2 = _nachtlauf(fx, 240, laufzeit=abo2)
    assert abo2.aussagen() == [D1, D2, D2, D3, D3, AUSSAGE, AUSSAGE]
    assert erg2.erst_offen == 0 and _stand(fx).erstbefuellung_fertig is True
    stand = _stand(fx)
    assert d1 in stand.erstgeprueft and stand.erst_fehlversuche == {}
    (punkt,) = [p for p in _pruefliste(fx)["offen"] if p["art"] == "erstpruefung_gescheitert"]
    assert punkt["aussage_id"] == d1 and D1 not in punkt["text"] and "helfer" not in punkt["text"]


def test_giftige_aussage_direkt_exit_code(fx):
    k = dataclasses.replace(fx.konfig, erstbefuellung=dict(ERST))
    posten = [_posten(f"docs/spaces/00{i}_x.md", 3, f"Aussage {i} nennt `f{i}`.") for i in (1, 2)]

    def exit1(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    erg = eb.erstbefuellung_schritt(k, None, st.Stand(), budget=Budget(), laufzeit=exit1, deadline=lambda: False,
                                    posten=posten, quelltext=lambda b: "x")
    assert (erg.geprueft, erg.abbruch, erg.fehlversuche) == (0, "exit:1", {"docs/spaces/001_x.md#a": 1})
    stand = st.Stand(erst_fehlversuche=dict(erg.fehlversuche))
    erg2 = eb.erstbefuellung_schritt(k, None, stand, budget=Budget(), laufzeit=exit1, deadline=lambda: False,
                                     posten=posten, quelltext=lambda b: "x")
    assert [(u.posten.aussage.id, u.grund) for u in erg2.urteile] == [("docs/spaces/001_x.md#a", "gescheitert:exit:1")]
    assert erg2.fehlversuche == {"docs/spaces/002_x.md#a": 1} and erg2.abbruch == "exit:1"


def test_fenster_endet_mitten_im_lauf(fx):
    """Review M6: schliesst das Fenster waehrend des Laufs, hoert die Erstbefuellung auf (die Uhr laeuft mit jedem
    Aufruf 50 s weiter: Start 05:58:30, nach Aussage 1 ist es 06:00:10)."""
    _start(fx)
    uhr = [1000.0]

    class Langsam(Abo):
        def __call__(self, argv, **kw):
            uhr[0] += 50
            return super().__call__(argv, **kw)

    abo = Langsam()
    _anstoss(fx)
    erg = _tick(fx, jetzt=datetime(2026, 10, 29, 5, 58, 30).astimezone(), laufzeit=abo, budget=Budget(),
                uhr=lambda: uhr[0])
    assert abo.aussagen() == [D1, D1] and erg.erst_geprueft == 1 and erg.erst_offen == 3


def test_hindernis_arten_beim_ersetzen():
    """Review M7: genaue Pruefpunkt-Art, warum ein `veraltet` nicht ersetzt wird."""
    from selbstbild.doku.judge import JudgeAntwort

    b = Beleg("symbol", "sp/a/client.py", "client.helfer", "d", "c", 11)
    ok = JudgeAntwort("veraltet", "Neu.", "client.helfer", "x")
    assert eb.ersatz_hindernis(ok, b, doku_da=True) is None
    assert eb.ersatz_hindernis(ok, None, doku_da=True) == "beleg_nicht_aufloesbar"
    assert eb.ersatz_hindernis(dataclasses.replace(ok, neuer_text=""), b, doku_da=True) == "erstpruefung_ohne_ersatz"
    assert eb.ersatz_hindernis(ok, b, doku_da=False) == "erstpruefung_doku_nicht_gelesen"
