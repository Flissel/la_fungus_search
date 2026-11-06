"""Taeglicher Lauf `python -m selbstbild auto` (Stufe 6a, Task 4). Alle Werkzeuge gefaelscht, echtes git-Repo."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from selbstbild import auto
from selbstbild.__main__ import main
from selbstbild.analyse.lauf import NichtVermessen
from selbstbild.funde.rueckfrage import RueckfrageBericht, marke
from selbstbild.funde.sync import IssueStand, stand_pfad
from selbstbild.sperre import sperre

ZIEL = "Test/ziel"
# Donnerstag, 05.11.2026 (kein Montag)
TAG1 = datetime(2026, 11, 5, 7, 30)
MONTAG = datetime(2026, 11, 9, 7, 30)
SYMBOL = "symbol:aussen:app/a.py#f"
SCHLUESSEL_WERT = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def _regeln(marker: Path, zweig: str = "main") -> str:
    return (
        "name: aussen\ndoku:\n  spaces: {}\n  ausschluss_namen: []\n  ausschluss_pfade: []\n  ketten_md: ''\n"
        "  compose_live: []\n  sql_wurzeln: ['.']\n  dienste_yaml: ''\n  waechter_zustandsdatei: ''\n"
        f"  waechter_zuordnung: {{}}\n  tick_minuten: 5\n  marker: '{marker.as_posix()}'\n  agent: ''\n"
        "  judge_modell: sonnet\n  claude_bin: claude\n  rowboat_wissen: w\n  rowboat_notizen: n\n"
        f"  rowboat_urteile: u\n  zweig: {zweig}\n"
        f"issues:\n  ziel_standard: {{repo: {ZIEL}, konto: tester, oeffentlich: false}}\n")


def _funde_json(je_art: dict[str, int] | None = None, *, tor: str = "toter Code: gesperrt (0/0 bestätigt)",
                modus: str = "senden", fehler: str | None = None) -> dict[str, Any]:
    return {"commit": "abc1234", "modus": modus,
            "tor_tot": {"offen": False, "ja": 0, "beurteilt": 0, "text": tor},
            "ziele": [{"ziel": ZIEL, "modus": modus, "aktionen_je_art": dict(je_art or {}), "fehler": fehler}]}


class FakeGh:
    host = "github.com"

    def __init__(self) -> None:
        self.bestand: list[dict[str, Any]] = []
        self.lesen = 0
        self.fehler: Exception | None = None

    def issues(self, repo: str) -> list[dict[str, Any]]:
        assert repo == ZIEL
        self.lesen += 1
        if self.fehler is not None:
            raise self.fehler
        return [dict(i) for i in self.bestand]


class Umg:
    def __init__(self, git_repo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.repo = git_repo
        self.wurzel: Path = git_repo.wurzel
        self.tmp = tmp_path
        self.ordner = tmp_path / "ordner"
        self.gold = tmp_path / "gold" / "goldsatz-aussen.yaml"
        git_repo.datei("selbstbild.yaml", _regeln(tmp_path / "doku" / "marker"))
        git_repo.datei("app/a.py", "def f():\n    return 1\n")
        git_repo.commit("init")
        self.kinder: list[list[str]] = []
        self.rc: dict[str, int] = {}
        self.funde: dict[str, Any] | None = _funde_json()
        self.beim_kind: dict[str, Any] = {}
        self.texte: list[str] = []
        self.telegram_ok = True
        self.frei: float | None = 16.0
        self.uhr = TAG1
        self.gh = FakeGh()
        self.konten: list[str] = []
        self.rueckfragen: list[dict[str, Any]] = []
        monkeypatch.setattr(auto, "gespeicherte_analyse", self._keine_analyse)

    @staticmethod
    def _keine_analyse(wurzel: Path) -> Any:
        raise NichtVermessen("noch nicht vermessen")

    def analyse(self, monkeypatch: pytest.MonkeyPatch, symbole: tuple[str, ...] = (SYMBOL,)) -> None:
        a = SimpleNamespace(commit="b" * 40,
                            tot=tuple(SimpleNamespace(symbol=s, stufe="wahrscheinlich") for s in symbole))
        monkeypatch.setattr(auto, "gespeicherte_analyse", lambda wurzel: a)

    def rueckfrage_fake(self, monkeypatch: pytest.MonkeyPatch, bericht: RueckfrageBericht | None = None,
                        stand_setzen: dict[str, Any] | None = None) -> None:
        def fake(wurzel: Path, goldsatz_pfad: Path, gh: Any, repo: str, *, senden: bool,
                 noch_tot: frozenset[str], commit: str, stand: dict[str, Any]) -> RueckfrageBericht:
            self.rueckfragen.append({"senden": senden, "noch_tot": noch_tot, "commit": commit, "repo": repo,
                                     "gold": goldsatz_pfad, "gh": gh})
            if stand_setzen and senden:
                stand.update(stand_setzen)
            return bericht if bericht is not None else RueckfrageBericht()
        monkeypatch.setattr(auto, "rueckfragen_abgleichen", fake)

    # ---------- Werkzeuge ----------

    def _kind(self, argv: list[str], log: Path) -> int:
        self.kinder.append(list(argv))
        schritt = argv[3]
        if schritt in self.beim_kind:
            self.beim_kind[schritt](argv, log)
        log.parent.mkdir(parents=True, exist_ok=True)
        if schritt == "funde" and self.funde is not None:
            log.write_text("Warnung: etwas auf stderr\n{nicht json\n" + json.dumps(self.funde, indent=1)
                           + "\nnoch eine Zeile stderr\n", encoding="utf-8")
        else:
            log.write_text("{}\n", encoding="utf-8")
        return self.rc.get(schritt, 0)

    def _telegram(self, text: str) -> bool:
        self.texte.append(text)
        return self.telegram_ok

    def _gh_fabrik(self, konto: str) -> Any:
        self.konten.append(konto)
        return self.gh

    def werkzeuge(self) -> auto.Werkzeuge:
        return auto.Werkzeuge(kind=self._kind, frei_gb=lambda: self.frei, telegram=self._telegram,
                              gh_fabrik=self._gh_fabrik, jetzt=lambda: self.uhr, ordner=self.ordner,
                              goldsatz=self.gold, stand_basis=self.tmp / "issues-basis")

    def lauf(self, *, trocken: bool = False) -> int:
        return auto.lauf(self.wurzel, trocken=trocken, werkzeuge=self.werkzeuge())

    def stand(self) -> dict[str, Any]:
        return json.loads((self.ordner / auto.STAND_DATEI).read_text(encoding="utf-8"))

    def schritte(self) -> list[str]:
        return [a[3] for a in self.kinder]

    def neuer_commit(self, n: int = 1) -> None:
        self.repo.datei(f"app/neu{n}.py", f"X = {n}\n")
        self.repo.commit(f"neu {n}")


@pytest.fixture(autouse=True)
def _hermetisch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELBSTBILD_HOME", str(tmp_path / "selbstbild-home"))


@pytest.fixture
def u(git_repo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Umg:
    return Umg(git_repo, tmp_path, monkeypatch)


# ---------- Brief-Tests ----------

def test_konstanten() -> None:
    assert auto.MIN_FREI_GB == 3.0
    assert auto.AUFGABE == "VibeMind Selbstbild-Funde"
    assert auto.STAND_DATEI == "auto-stand.json"
    assert auto.SPERRE == "auto.lock"


def test_ohne_aenderung_keine_nachricht_kein_kindprozess(u: Umg) -> None:
    u.funde = _funde_json({"anlegen": 1})
    assert u.lauf() == 0
    assert u.schritte() == ["vermessen", "analysieren", "funde"]
    assert len(u.texte) == 1
    u.kinder.clear()
    u.uhr = TAG1 + timedelta(days=1)  # anderer Tag: auch dann nichts, solange sich nichts aendert
    assert u.lauf() == 0
    assert u.kinder == []
    assert len(u.texte) == 1
    assert u.stand()["ergebnis"] == "unveraendert"


def test_kindprozesse_mit_genauen_argumenten(u: Umg) -> None:
    import sys

    assert u.lauf() == 0
    sha = u.repo.git("rev-parse", "HEAD")
    w = str(u.wurzel)
    assert u.kinder == [
        [sys.executable, "-m", "selbstbild", "vermessen", w, "--commit", sha],
        [sys.executable, "-m", "selbstbild", "analysieren", w],
        [sys.executable, "-m", "selbstbild", "funde", w, "--senden"],
    ]
    assert u.stand()["kopf"] == sha


def test_zweiter_lauf_am_selben_tag_sendet_nicht(u: Umg) -> None:
    u.funde = _funde_json({"anlegen": 1})
    assert u.lauf() == 0
    assert len(u.texte) == 1
    u.neuer_commit()
    u.funde = _funde_json({"schliessen": 2})
    u.uhr = TAG1 + timedelta(hours=3)
    assert u.lauf() == 0
    assert len(u.texte) == 1  # schon eine Nachricht heute
    s = u.stand()
    assert s["ausstehend"] and any("2 geschlossen" in z for z in s["ausstehend"])
    assert s["letzte_nachricht_tag"] == TAG1.date().isoformat()
    # Folgetag: ohne Aenderung geht das Ausstehende mit genau einer Nachricht raus
    u.kinder.clear()
    u.uhr = TAG1 + timedelta(days=1)
    assert u.lauf() == 0
    assert u.kinder == []
    assert len(u.texte) == 2 and "2 geschlossen" in u.texte[1]
    s = u.stand()
    assert s["ausstehend"] == [] and s["letzte_nachricht_tag"] == (TAG1 + timedelta(days=1)).date().isoformat()


def test_falscher_zweig_ueberspringt_und_meldet_einmal(u: Umg) -> None:
    u.repo.git("checkout", "-q", "-b", "anderer")
    assert u.lauf() == 0
    assert u.kinder == []
    assert len(u.texte) == 1 and "falscher_zweig" in u.texte[0] and "übersprungen" in u.texte[0]
    assert u.stand()["letzter_fehler"] == "zweig:falscher_zweig"
    u.uhr = TAG1 + timedelta(days=1)
    assert u.lauf() == 0
    assert u.kinder == [] and len(u.texte) == 1
    assert u.stand()["ausstehend"] == []


def test_speicher_knapp_ueberspringt(u: Umg) -> None:
    u.frei = 2.0
    assert u.lauf() == 0
    assert u.kinder == []
    assert len(u.texte) == 1 and "übersprungen: Speicher" in u.texte[0]
    assert u.stand()["letzter_fehler"] == "speicher"


def test_speicher_unbekannt_ueberspringt(u: Umg) -> None:
    u.frei = None
    assert u.lauf() == 0
    assert u.kinder == [] and "übersprungen: Speicher" in u.texte[0]


def test_kindfehler_bricht_ab_und_meldet_typ(u: Umg) -> None:
    u.rc["analysieren"] = 1
    assert u.lauf() == 1
    assert u.schritte() == ["vermessen", "analysieren"]
    assert len(u.texte) == 1 and "Selbstbild-Lauf fehlgeschlagen: analysieren, exit_1" in u.texte[0]
    s = u.stand()
    assert s["letzter_fehler"] == "analysieren:exit_1" and s["kopf"] is None  # Tor nicht weitergeschaltet
    # gleicher Fehler am Folgetag: erneut versucht, aber nicht noch einmal gemeldet
    u.kinder.clear()
    u.uhr = TAG1 + timedelta(days=1)
    assert u.lauf() == 1
    assert u.schritte() == ["vermessen", "analysieren"]
    assert len(u.texte) == 1
    # danach erfolgreich: letzter_fehler geloescht
    u.rc.clear()
    u.uhr = TAG1 + timedelta(days=2)
    assert u.lauf() == 0
    assert u.stand()["letzter_fehler"] is None


def test_ausnahme_im_kind_nennt_nur_den_typ(u: Umg) -> None:
    def kaputt(argv: list[str], log: Path) -> None:
        raise TimeoutError("geheimer Text " + SCHLUESSEL_WERT)
    u.beim_kind["vermessen"] = kaputt
    assert u.lauf() == 1
    assert u.texte == [auto.TEXT_KOPF.format(tag="05.11.2026") + "\n- Selbstbild-Lauf fehlgeschlagen: "
                       "vermessen, TimeoutError"]


def test_telegram_fehler_haelt_ausstehend(u: Umg) -> None:
    u.funde = _funde_json({"anlegen": 1})
    u.telegram_ok = False
    assert u.lauf() == 0
    s = u.stand()
    assert s["ausstehend"] and s["letzte_nachricht_tag"] is None
    assert len(u.texte) == 1
    # naechster Tag, Telegram wieder da: das Ausstehende geht mit
    u.telegram_ok = True
    u.uhr = TAG1 + timedelta(days=1)
    assert u.lauf() == 0
    assert len(u.texte) == 2 and "1 neu" in u.texte[1]
    assert u.stand()["ausstehend"] == []


def test_telegram_ausnahme_gilt_als_fehlschlag(u: Umg) -> None:
    u.funde = _funde_json({"anlegen": 1})

    def kaputt(text: str) -> bool:
        raise OSError("weg")
    w = u.werkzeuge()
    w.telegram = kaputt
    assert auto.lauf(u.wurzel, werkzeuge=w) == 0
    assert u.stand()["ausstehend"]


def test_schluessel_im_text_wird_zurueckgehalten(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.analyse(monkeypatch)
    u.rueckfrage_fake(monkeypatch, RueckfrageBericht(wartet=[f"#3 {SCHLUESSEL_WERT}: goldsatz_unsauber"]))
    assert u.lauf() == 0
    assert u.texte == [auto.ERSATZ_TEXT]
    assert auto.ERSATZ_TEXT == "Selbstbild: Lauf fertig, Details im Bericht (Text zurückgehalten)."
    assert u.stand()["ausstehend"] == []
    assert SCHLUESSEL_WERT not in (u.ordner / auto.STAND_DATEI).read_text(encoding="utf-8")


def test_sperre_belegt_exit_3_still(u: Umg) -> None:
    with sperre(u.ordner / auto.SPERRE):
        assert u.lauf() == 3
    assert u.kinder == [] and u.texte == []
    assert not (u.ordner / auto.STAND_DATEI).exists()


def test_verwaiste_sperre_uebernommen_und_erwaehnt(u: Umg) -> None:
    u.ordner.mkdir(parents=True)
    (u.ordner / auto.SPERRE).write_text(json.dumps({"pid": 999999999, "start": "x"}), encoding="utf-8")
    u.funde = _funde_json({"anlegen": 1})
    assert u.lauf() == 0
    assert "verwaiste Sperre" in u.texte[0]
    assert not (u.ordner / auto.SPERRE).exists()


def test_sperre_oserror_meldet_nur_typ(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    def kaputt(pfad: Path, **kw: Any) -> Any:
        raise PermissionError("Zugriff verweigert C:/geheim")
    monkeypatch.setattr(auto, "sperre", kaputt)
    assert u.lauf() == 1
    assert u.kinder == []
    assert len(u.texte) == 1 and "Selbstbild-Lauf fehlgeschlagen: sperre, PermissionError" in u.texte[0]
    assert "geheim" not in u.texte[0]


def test_trocken_sendet_nicht_und_schreibt_keinen_stand(u: Umg, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.analyse(monkeypatch)
    u.rueckfrage_fake(monkeypatch)
    u.funde = _funde_json({"anlegen": 2}, modus="trocken")
    assert u.lauf(trocken=True) == 0
    assert u.schritte() == ["vermessen", "analysieren", "funde"]
    assert not any("--senden" in a for a in u.kinder)
    assert u.texte == []
    assert not (u.ordner / auto.STAND_DATEI).exists()
    assert [r["senden"] for r in u.rueckfragen] == [False]
    out = capsys.readouterr().out
    assert "2 neu" in out


def test_montag_erinnerung_ohne_aenderung(u: Umg) -> None:
    alt = (MONTAG - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    frisch = (MONTAG - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    u.gh.bestand = [
        {"number": 5, "state": "open", "labels": [{"name": "selbstbild"}, {"name": "selbstbild:rueckfrage"}],
         "body": "x\n" + marke(SYMBOL), "html_url": f"https://github.com/{ZIEL}/issues/5", "created_at": alt},
        {"number": 6, "state": "open", "labels": [{"name": "selbstbild"}, {"name": "selbstbild:rueckfrage"}],
         "body": "x\n" + marke(SYMBOL + "2"), "html_url": f"https://github.com/{ZIEL}/issues/6",
         "created_at": frisch},
    ]
    u.uhr = MONTAG - timedelta(days=3)  # Freitag: erster Lauf (keine Erinnerung)
    assert u.lauf() == 0
    assert not any("Erinnerung" in t for t in u.texte)
    vorher = len(u.texte)
    u.kinder.clear()
    u.uhr = MONTAG
    assert u.lauf() == 0
    assert u.kinder == []
    assert len(u.texte) == vorher + 1
    t = u.texte[-1]
    assert "Erinnerung" in t and "#5" in t and "8 Tagen" in t and "issues/5" in t
    assert "#6" not in t
    # zweiter Lauf am selben Montag: keine zweite Erinnerung, auch nicht ausstehend
    u.uhr = MONTAG + timedelta(hours=2)
    assert u.lauf() == 0
    assert len(u.texte) == vorher + 1 and u.stand()["ausstehend"] == []


def test_montag_keine_erinnerung_fuer_eben_geschlossene(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    alt = (MONTAG - timedelta(days=9)).strftime("%Y-%m-%dT%H:%M:%SZ")
    u.gh.bestand = [{"number": 5, "state": "open", "labels": [{"name": "selbstbild"}], "body": marke(SYMBOL),
                     "html_url": f"https://github.com/{ZIEL}/issues/5", "created_at": alt},
                    {"number": 8, "state": "open", "labels": [{"name": "selbstbild"}],
                     "body": marke(SYMBOL + "x"), "html_url": f"https://github.com/{ZIEL}/issues/8", "created_at": alt}]
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.analyse(monkeypatch)
    u.rueckfrage_fake(monkeypatch, RueckfrageBericht(erledigt=[5], links={5: f"https://github.com/{ZIEL}/issues/5"}))
    u.uhr = MONTAG
    assert u.lauf() == 0
    t = u.texte[0]
    assert "Rückfrage erledigt: #5" in t and "Erinnerung: Rückfrage #8 seit 9 Tagen" in t
    assert "Erinnerung: Rückfrage #5" not in t


def test_cli_auto_trocken(u: Umg, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    gesehen: list[Path] = []

    def standard(wurzel: Path) -> auto.Werkzeuge:
        gesehen.append(Path(wurzel))
        return u.werkzeuge()
    monkeypatch.setattr(auto, "standard_werkzeuge", standard)
    assert main(["auto", str(u.wurzel), "--trocken"]) == 0
    assert gesehen == [u.wurzel]
    assert u.texte == [] and not (u.ordner / auto.STAND_DATEI).exists()
    assert not any("--senden" in a for a in u.kinder)


def test_cli_auto_aufgabe_ist_keine_wurzel(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    def nie(wurzel: Path) -> auto.Werkzeuge:
        raise AssertionError("auto aufgabe darf nie als Lauf starten")
    monkeypatch.setattr(auto, "standard_werkzeuge", nie)
    assert main(["auto", "aufgabe"]) == 2
    assert main(["auto", "aufgabe", str(u.wurzel)]) == 2


def test_cli_auto_ohne_verzeichnis_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def nie(wurzel: Path) -> auto.Werkzeuge:
        raise AssertionError("kein Lauf ohne Verzeichnis")
    monkeypatch.setattr(auto, "standard_werkzeuge", nie)
    assert main(["auto", str(tmp_path / "gibt-es-nicht")]) == 2


# ---------- Rueckfragen ----------

def test_rueckfragen_mit_noch_tot_aus_jeder_stufe_und_stand_sofort_gesichert(
        u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.analyse(monkeypatch, symbole=(SYMBOL, "symbol:aussen:app\\b.py#g"))
    u.rueckfrage_fake(monkeypatch, RueckfrageBericht(neu=[7], links={7: f"https://github.com/{ZIEL}/issues/7"}),
                      stand_setzen={"kommentiert": [4]})
    gesichert: list[dict[str, Any]] = []

    def pruefen(argv: list[str], log: Path) -> None:
        gesichert.append(json.loads((u.ordner / auto.STAND_DATEI).read_text(encoding="utf-8")))
        raise RuntimeError("Absturz nach den Rueckfragen")
    u.beim_kind["funde"] = pruefen
    assert u.lauf() == 1
    r = u.rueckfragen[0]
    assert r["senden"] is True and r["repo"] == ZIEL and r["gold"] == u.gold and r["commit"] == "b" * 40
    assert r["noch_tot"] == frozenset({SYMBOL, "symbol:aussen:app/b.py#g"})
    assert u.konten == ["tester"]
    assert gesichert and gesichert[0]["rueckfrage"] == {"kommentiert": [4]}
    assert "Neue Rückfrage #7" in u.texte[0] and "issues/7" in u.texte[0]
    assert "Selbstbild-Lauf fehlgeschlagen: funde, RuntimeError" in u.texte[0]


def test_noch_tot_leer_ueberspringt_rueckfragen(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.analyse(monkeypatch, symbole=())
    u.rueckfrage_fake(monkeypatch)
    u.funde = _funde_json({"anlegen": 1})
    assert u.lauf() == 0
    assert u.rueckfragen == []
    assert "Rückfragen übersprungen" in u.texte[0]


def test_analyse_unlesbar_ueberspringt_rueckfragen(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.rueckfrage_fake(monkeypatch)
    u.funde = _funde_json({"anlegen": 1})
    assert u.lauf() == 0
    assert u.rueckfragen == []
    assert "Rückfragen übersprungen" in u.texte[0] and "NichtVermessen" in u.texte[0]


def test_doppelt_beantwortet_nur_einmal_gemeldet(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.analyse(monkeypatch)
    u.rueckfrage_fake(monkeypatch, RueckfrageBericht(doppelt_beantwortet=[4],
                                                     links={4: f"https://github.com/{ZIEL}/issues/4"}))
    assert u.lauf() == 0
    assert len(u.texte) == 1 and "#4" in u.texte[0]
    u.neuer_commit()
    u.uhr = TAG1 + timedelta(days=1)
    assert u.lauf() == 0
    assert len(u.texte) == 1  # nichts Neues: keine Nachricht
    assert u.stand()["ausstehend"] == []


def test_antwort_uebernommen_mit_commit_und_tor_aendert_sich_nicht_durch_eigenes_schliessen(
        u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.gh.bestand = [{"number": 4, "state": "open", "body": marke(SYMBOL), "created_at": "2026-11-01T00:00:00Z",
                     "labels": [{"name": "selbstbild"}, {"name": "antwort:weg"}]}]
    u.analyse(monkeypatch)

    def fake(wurzel: Path, goldsatz_pfad: Path, gh: Any, repo: str, *, senden: bool,
             noch_tot: frozenset[str], commit: str, stand: dict[str, Any]) -> RueckfrageBericht:
        u.rueckfragen.append({})
        u.gold.write_text("tot: [] # neu\n", encoding="utf-8")  # wie der Goldsatz-Commit
        u.gh.bestand[0]["state"] = "closed"
        return RueckfrageBericht(uebernommen=[4], goldsatz_commit="1234567",
                                 links={4: f"https://github.com/{ZIEL}/issues/4"})
    monkeypatch.setattr(auto, "rueckfragen_abgleichen", fake)
    assert u.lauf() == 0
    assert "Antwort übernommen: #4" in u.texte[0] and "1234567" in u.texte[0]
    u.kinder.clear()
    u.uhr = TAG1 + timedelta(days=1)
    assert u.lauf() == 0
    assert u.kinder == []  # eigenes Schliessen und eigener Goldsatz-Commit zaehlen nicht als Aenderung


def test_neues_antwort_label_ist_aenderung(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gh.bestand = [{"number": 4, "state": "open", "body": marke(SYMBOL), "created_at": "2026-11-01T00:00:00Z",
                     "labels": [{"name": "selbstbild"}]}]
    assert u.lauf() == 0
    u.kinder.clear()
    u.gh.bestand[0]["labels"].append({"name": "antwort:bleibt"})
    assert u.lauf() == 0
    assert u.schritte() == ["vermessen", "analysieren", "funde"]


def test_github_unlesbar_gilt_als_geaendert(u: Umg) -> None:
    from selbstbild.funde.github import GitHubFehler

    assert u.lauf() == 0
    u.kinder.clear()
    u.gh.fehler = GitHubFehler("http_502")
    assert u.lauf() == 0
    assert u.schritte() == ["vermessen", "analysieren", "funde"]


def test_goldsatz_geaendert_ist_aenderung(u: Umg) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    assert u.lauf() == 0
    u.kinder.clear()
    u.gold.write_text("tot: [] # unklar dazu\n", encoding="utf-8")
    assert u.lauf() == 0
    assert u.schritte() == ["vermessen", "analysieren", "funde"]


def test_label_fehlt_sperrt_rueckfragen_und_sync(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    u.analyse(monkeypatch)
    u.rueckfrage_fake(monkeypatch, RueckfrageBericht(fehler="label_fehlt"))
    assert u.lauf() == 1
    assert "Selbstbild-Lauf fehlgeschlagen: rueckfragen, label_fehlt" in u.texte[0]
    assert u.schritte() == ["vermessen", "analysieren"]
    assert u.stand()["sperren"] == {ZIEL: "label_fehlt"}
    assert IssueStand.laden(stand_pfad(u.wurzel, u.tmp / "issues-basis")).sperren == {ZIEL: "label_fehlt"}
    # naechster Lauf: keine Rueckfragen mehr, funde laeuft (und meldet die Sperre selbst)
    u.neuer_commit()
    u.uhr = TAG1 + timedelta(days=1)
    assert u.lauf() == 0
    assert len(u.rueckfragen) == 1
    assert u.schritte()[-1] == "funde"


def test_sync_sperre_gilt_auch_fuer_rueckfragen(u: Umg, monkeypatch: pytest.MonkeyPatch) -> None:
    u.gold.parent.mkdir(parents=True)
    u.gold.write_text("tot: []\n", encoding="utf-8")
    st = IssueStand.laden(stand_pfad(u.wurzel, u.tmp / "issues-basis"))
    st.sperren[ZIEL] = "label_fehlt"
    st.speichern()
    u.analyse(monkeypatch)
    u.rueckfrage_fake(monkeypatch)
    assert u.lauf() == 0
    assert u.rueckfragen == []


def test_funde_exit_3_kein_fehler(u: Umg) -> None:
    u.rc["funde"] = 3
    u.funde = None
    assert u.lauf() == 0
    assert len(u.texte) == 1 and "anderer Sendelauf" in u.texte[0]
    assert u.stand()["letzter_fehler"] is None


def test_funde_exit_1_ist_fehler_mit_zeilen(u: Umg) -> None:
    u.rc["funde"] = 1
    u.funde = _funde_json({"anlegen": 1}, fehler="http_502")
    assert u.lauf() == 1
    t = u.texte[0]
    assert "Selbstbild-Lauf fehlgeschlagen: funde, exit_1" in t and "1 neu" in t and "http_502" in t


def test_tor_zeile_nur_bei_aenderung_allein_nachricht(u: Umg) -> None:
    assert u.lauf() == 0
    assert len(u.texte) == 1 and "toter Code: gesperrt" in u.texte[0]  # erster Stand des Tors
    u.neuer_commit()
    u.uhr = TAG1 + timedelta(days=1)
    assert u.lauf() == 0
    assert len(u.texte) == 1  # Tor unveraendert, sonst nichts: keine Nachricht
    u.neuer_commit(2)
    u.funde = _funde_json(tor="toter Code: offen (31/33 bestätigt)")
    u.uhr = TAG1 + timedelta(days=2)
    assert u.lauf() == 0
    assert len(u.texte) == 2 and "offen (31/33" in u.texte[1]


def test_stand_kaputt_gilt_als_leer(u: Umg) -> None:
    u.ordner.mkdir(parents=True)
    (u.ordner / auto.STAND_DATEI).write_text("{kaputt", encoding="utf-8")
    u.funde = _funde_json({"anlegen": 1})
    assert u.lauf() == 0
    assert u.schritte() == ["vermessen", "analysieren", "funde"]
    assert "auto-stand.json unlesbar" in u.texte[0]
    assert u.stand()["kopf"]


def test_stand_hat_alle_schluessel(u: Umg) -> None:
    assert u.lauf() == 0
    s = u.stand()
    for k in ("kopf", "goldsatz_hash", "antworten", "letzte_nachricht_tag", "letzter_fehler", "ausstehend",
              "letzter_lauf", "ergebnis", "rueckfrage"):
        assert k in s
    assert s["ergebnis"] == "ok"


# ---------- Bausteine ----------

def test_letztes_json_objekt(tmp_path: Path) -> None:
    p = tmp_path / "log"
    p.write_text('{"a": 1}\nrauschen\n{\n "b": {\n  "c": 2\n }\n}\nWarnung\n', encoding="utf-8")
    assert auto.letztes_json(p) == {"b": {"c": 2}}
    p.write_text("nur rauschen\n", encoding="utf-8")
    assert auto.letztes_json(p) is None
    assert auto.letztes_json(tmp_path / "fehlt") is None


def test_text_wird_gekappt() -> None:
    zeilen = [f"Zeile {i} " + "x" * 100 for i in range(100)]
    text = auto.nachricht(zeilen, TAG1.date())
    assert len(text) <= auto.MAX_TEXT
    assert "weitere Zeilen" in text


def test_frei_gb_standard_liefert_zahl_oder_none() -> None:
    wert = auto.frei_gb_standard()
    assert wert is None or wert > 0


def test_telegram_standard_ohne_skript_false(tmp_path: Path) -> None:
    assert auto.telegram_standard(tmp_path, "hallo") is False
