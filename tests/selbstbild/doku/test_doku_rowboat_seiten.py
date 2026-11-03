"""Rowboat-Seiten ohne LLM (Task 8): Mitbauen aus der Skill-Quelle, Zustand, Schreibschutz, Index, Trockenlauf.

Alles in tmp_path; der echte Wissensordner wird nie angefasst."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import rowboat as rb  # noqa: E402
from selbstbild.doku import rowboat_seiten as rs  # noqa: E402
from selbstbild.doku.graph import Graph, Kante, Knoten, hop_zustand_neu, kante_id  # noqa: E402
from selbstbild.skill import skill_neuer_space  # noqa: E402

from selbstbild.doku.typen import Pruefpunkt  # noqa: E402
from test_doku_rowboat import KETTEN, LAUF, _graph, _ledger_eintrag, _lesen, _punkt  # noqa: E402

REGELN = (
    "name: aussen\n"
    "struktur:\n"
    "  - art: kein_space_im_root\n"
    "    ort: \"./\"\n"
    "    pflicht: []\n"
    "    verboten:\n"
    "      - \"spaces/*/\"\n"
    "    hinweis: \"Spaces gehoeren nach sub/spaces/<name>/\"\n"
    "  - art: space\n"
    "    ort: \"sub/spaces/*/\"\n"
    "    pflicht: [main.py, tests/]\n"
    "    verboten: []\n"
    "    hinweis: \"Spaces gehoeren nach sub/spaces/<name>/\"\n"
)


@pytest.fixture(autouse=True)
def heim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "leer-home"
    monkeypatch.setenv("SELBSTBILD_HOME", str(h))
    return h


@pytest.fixture
def wurzel(tmp_path: Path) -> Path:
    w = tmp_path / "aussen"
    w.mkdir()
    (w / "selbstbild.yaml").write_bytes(REGELN.encode("utf-8"))
    return w


def _abschnitte(text: str) -> dict[str, str]:
    teile = re.split(r"^(## .+)$", text, flags=re.M)
    return {teile[i][3:].strip(): teile[i + 1].strip("\n") for i in range(1, len(teile), 2)}


def _kaputter_graph() -> Graph:
    g = _graph()
    z = hop_zustand_neu()
    z["gemessen"] = {"wert": "bricht", "zeit": "2026-10-28T00:00:00+00:00"}
    g.knoten_hinzu(Knoten("kette:K3", "kette", "Dritte", None, None, {"zustand_roh": "bricht"}))
    g.kante_hinzu(Kante(kante_id("hop", "dienst:c", "dienst:b", "K3", 1), "hop", "dienst:c", "dienst:b", "seed", (),
                        {"kette": "K3", "nr": "1", "mechanismus": "x", "zustand": z, "zustand_text": "bricht"}))
    z2 = hop_zustand_neu()
    z2["aus"] = True
    g.kante_hinzu(Kante(kante_id("hop", "dienst:b", "dienst:c", "K3", 2), "hop", "dienst:b", "dienst:c", "seed", (),
                        {"kette": "K3", "nr": "2", "mechanismus": "y", "zustand": z2, "zustand_text": "aus"}))
    return g


# ---------- Mitbauen ----------

def test_mitbauen_gleich_skill(wurzel):
    skill = skill_neuer_space(wurzel)
    seite = rs.seite_mitbauen(wurzel, None)
    assert seite.startswith("---\nerzeugt_von: selbstbild\ngrundlage: ")
    assert "# Neuen Space erstellen mit Claude Code" in seite
    assert "bau mir einen neuen Space für X" in seite
    soll = _abschnitte(skill)
    for name in ("Wohin", "Nie hier anlegen"):
        assert name in soll, name
        assert f"## {name}\n\n{soll[name]}" in seite, name  # wortgleich
    assert "name: neuer-space" not in seite  # Frontmatter des Skills fehlt
    assert rs.seite_mitbauen(wurzel, None) == seite  # deterministisch


def test_mitbauen_grundlage_folgt_dem_skill(wurzel):
    a = rs.seite_mitbauen(wurzel, None)
    (wurzel / "selbstbild.yaml").write_bytes(REGELN.replace("main.py", "app.py").encode("utf-8"))
    b = rs.seite_mitbauen(wurzel, None)
    assert "## Vorbilder" not in a  # ohne Stand keine Vorbilder
    assert a != b and a.split("\n")[2] != b.split("\n")[2]  # grundlage-Zeile


# ---------- Zustand ----------

def test_zustand_ohne_analyse():
    seite = rs.seite_zustand(_kaputter_graph(), None, {"offen": [_punkt(), _punkt(text="b"), _punkt(art="x")]})
    assert "Toter Code" not in seite and "sicher_tot" not in seite
    assert "| [[Doku/Ketten/K3]] Dritte | 1 | 2 |" in seite
    assert "[[Doku/Ketten/K1]]" not in seite
    assert "| `hop_beleg_verschwunden` | 2 |" in seite and "| `x` | 1 |" in seite
    assert "## Offene Prüfpunkte (3)" in seite
    assert seite.startswith("---\nerzeugt_von: selbstbild\ngrundlage: ")


def test_zustand_mit_analyse_und_deterministisch():
    an = {"commit": "c", "tot": {"sicher_tot": 4, "wahrscheinlich_tot": 7},
          "hotspots_top": [{"datei": f"d{i}.py", "wert": 1.0 / i} for i in range(1, 9)]}
    seite = rs.seite_zustand(_graph(), an, {"offen": []})
    assert "`sicher_tot`: 4" in seite and "`wahrscheinlich_tot`: 7" in seite
    assert "1. `d1.py`" in seite and "5. `d5.py`" in seite and "d6.py" not in seite
    assert "Keine offenen Punkte." in seite and "Keine Kette bricht" in seite
    assert seite == rs.seite_zustand(_graph(), an, {"offen": []})
    assert not re.search(r"\d{4}-\d{2}-\d{2}", seite)  # keine Zeitstempel im Text
    ohne = rs.seite_zustand(_graph(), None, {"offen": []})
    assert ohne.split("\n")[2] != seite.split("\n")[2]  # grundlage haengt von der Analyse ab


def _analyse_db(stand: Path, daten: dict) -> None:
    stand.mkdir(parents=True)
    with closing(sqlite3.connect(stand / "analyse.sqlite")) as c:
        c.execute("CREATE TABLE lauf (commit_id TEXT NOT NULL, version INTEGER NOT NULL, parameter_digest TEXT NOT NULL,"
                  " daten BLOB NOT NULL, PRIMARY KEY (commit_id, version, parameter_digest))")
        c.execute("INSERT INTO lauf VALUES ('c', 1, 'p', ?)", (json.dumps(daten).encode("utf-8"),))
        c.commit()


def test_analyse_lesen_nur_lesend_und_fehlt_ohne_absturz(tmp_path):
    assert rs.analyse_zusammenfassung_lesen(None) is None
    assert rs.analyse_zusammenfassung_lesen(tmp_path / "gibt-es-nicht") is None
    kaputt = tmp_path / "kaputt"
    kaputt.mkdir()
    (kaputt / "analyse.sqlite").write_bytes(b"kein sqlite")
    assert rs.analyse_zusammenfassung_lesen(kaputt) is None
    stand = tmp_path / "stand"
    _analyse_db(stand, {"commit": "abc", "tot": [{"stufe": "sicher_tot"}, {"stufe": "sicher_tot"},
                                                  {"stufe": "wahrscheinlich_tot"}, {"stufe": "verdaechtig"}],
                        "hotspots": [{"datei": f"h{i}.py", "wert": str(i)} for i in range(9, 0, -1)]})
    vorher = (stand / "analyse.sqlite").read_bytes()
    z = rs.analyse_zusammenfassung_lesen(stand)
    assert z == {"commit": "abc", "tot": {"sicher_tot": 2, "wahrscheinlich_tot": 1},
                 "hotspots_top": [{"datei": f"h{i}.py", "wert": float(i)} for i in range(9, 4, -1)]}
    assert (stand / "analyse.sqlite").read_bytes() == vorher


def test_seiten_nehmen_analyse_aus_stand_unter(wurzel, tmp_path):
    stand = tmp_path / "stand"
    _analyse_db(stand, {"commit": "abc", "tot": [{"stufe": "sicher_tot"}], "hotspots": []})
    (stand / "suche.sqlite").write_bytes(b"")  # Ordner gilt nur mit Index als Stand des Skills; Analyse geht auch so
    n = rs.seiten(_graph(), wurzel, {"offen": []}, stand)
    assert [x.rel for x in n] == [rs.MITBAUEN_REL, rs.ZUSTAND_REL]
    assert "`sicher_tot`: 1" in n[1].text
    n2 = rs.seiten(_graph(), wurzel, {"offen": []}, tmp_path / "leer")
    assert "sicher_tot" not in n2[1].text


# ---------- Schreiben ----------

def _schreiben(wissen, zusatz, lauf=LAUF):
    pp: list = []
    geschrieben = rb.notizen_schreiben(
        _graph(), [_punkt()], [], wissen, "Doku", lauf, manifest_pfad=wissen.parent / "m.json", pruefpunkte=pp,
        zusatz=zusatz, zusatz_links=rs.index_links("Doku"))
    return geschrieben, pp


def test_pfade_leerzeichen_und_index(wurzel, tmp_path):
    assert rb.sicherer_name("Neuen Space erstellen mit Claude Code") == "Neuen Space erstellen mit Claude Code"
    wissen = tmp_path / "wissen"
    zusatz = rs.seiten(_graph(), wurzel, {"offen": [_punkt()]}, tmp_path / "leer")
    geschrieben, pp = _schreiben(wissen, zusatz)
    doku = wissen / "Doku"
    assert doku / "Mitbauen" / "Neuen Space erstellen mit Claude Code.md" in geschrieben
    assert doku / "Zustand.md" in geschrieben and pp == []
    index = _lesen(doku / "Index.md")
    assert "[[Doku/Mitbauen/Neuen Space erstellen mit Claude Code]]" in index and "[[Doku/Zustand]]" in index
    # zweiter Lauf, anderer Lauf-Name: nichts neu, nichts geloescht
    assert _schreiben(wissen, zusatz, lauf="2026-10-29T00:00:00#9")[0] == []
    assert (doku / "Zustand.md").is_file()


@pytest.mark.parametrize("rel", [rs.MITBAUEN_REL, rs.ZUSTAND_REL])
def test_hand_edit_schutz(wurzel, tmp_path, rel):
    wissen = tmp_path / "wissen"
    leer = tmp_path / "leer"
    _schreiben(wissen, rs.seiten(_graph(), wurzel, {"offen": []}, leer))
    datei = wissen / "Doku" / rel
    datei.write_bytes(datei.read_bytes() + "\nMeine Notiz.\n".encode("utf-8"))
    hand = datei.read_bytes()
    # die Grundlage aendert sich (neuer Graph, andere Regeln): die Seite soll neu, die Hand-Fassung bleibt
    (wurzel / "selbstbild.yaml").write_bytes(REGELN.replace("main.py", "app.py").encode("utf-8"))
    zusatz2 = rs.seiten(_kaputter_graph(), wurzel, {"offen": [_punkt()]}, leer)
    geschrieben, pp = _schreiben(wissen, zusatz2)
    assert datei.read_bytes() == hand and datei not in geschrieben
    assert [(p.art, p.datei) for p in pp] == [("notiz_von_hand", f"Doku/{rel}")]
    assert _schreiben(wissen, zusatz2)[0] == [] and datei.read_bytes() == hand  # weiterhin geschuetzt


def test_unveraenderte_seite_wird_bei_neuer_grundlage_neu_geschrieben(wurzel, tmp_path):
    wissen = tmp_path / "wissen"
    leer = tmp_path / "leer"
    _schreiben(wissen, rs.seiten(_graph(), wurzel, {"offen": []}, leer))
    geschrieben, pp = _schreiben(wissen, rs.seiten(_kaputter_graph(), wurzel, {"offen": []}, leer))
    assert wissen / "Doku" / "Zustand.md" in geschrieben and pp == []
    assert "Dritte" in _lesen(wissen / "Doku" / "Zustand.md")


def test_trocken_gleich_live(wurzel, tmp_path):
    wissen = tmp_path / "wissen"
    leer = tmp_path / "leer"
    manifest = wissen.parent / "m.json"
    zusatz = rs.seiten(_kaputter_graph(), wurzel, {"offen": [_punkt()]}, leer)

    def lauf(trocken):
        pp: list = []
        return rb.notizen_schreiben(_kaputter_graph(), [_punkt()], [], wissen, "Doku", LAUF, manifest_pfad=manifest,
                                    pruefpunkte=pp, zusatz=zusatz, zusatz_links=rs.index_links("Doku"),
                                    trocken=trocken), pp

    trocken, _ = lauf(True)
    assert not wissen.exists() and not manifest.exists()  # nichts geschrieben
    live, _ = lauf(False)
    assert trocken == live and wissen / "Doku" / "Index.md" in trocken
    assert lauf(True)[0] == [] and lauf(False)[0] == []
    # von Hand geaendert: auch im Trockenlauf gemeldet, nichts angefasst
    (wissen / "Doku" / "Zustand.md").write_bytes(b"von Hand")
    neu = rs.seiten(_graph(), wurzel, {"offen": []}, leer)
    pp: list = []
    rb.notizen_schreiben(_kaputter_graph(), [_punkt()], [], wissen, "Doku", LAUF, manifest_pfad=manifest,
                         pruefpunkte=pp, zusatz=neu, trocken=True)
    assert [p.art for p in pp] == ["notiz_von_hand"] and (wissen / "Doku" / "Zustand.md").read_bytes() == b"von Hand"


def test_seite_fehler_laesst_den_rest_schreiben(wurzel, tmp_path, monkeypatch):
    import selbstbild.skill as skill

    def kaputt(*a, **k):
        raise RuntimeError("geheimer Text")

    monkeypatch.setattr(skill, "skill_neuer_space", kaputt)
    fehler: list[Pruefpunkt] = []
    ausgefallen: list[str] = []
    zusatz = rs.seiten(_graph(), wurzel, {"offen": []}, tmp_path / "leer", fehler=fehler, ausgefallen=ausgefallen)
    assert [z.rel for z in zusatz] == [rs.ZUSTAND_REL] and ausgefallen == [rs.MITBAUEN_REL]
    assert [p.art for p in fehler] == ["seite_fehler"]
    assert "RuntimeError" in fehler[0].text and "geheimer" not in fehler[0].text
    wissen = tmp_path / "wissen"
    geschrieben = rb.notizen_schreiben(_graph(), [_punkt()], [], wissen, "Doku", LAUF,
                                       manifest_pfad=wissen.parent / "m.json", zusatz=zusatz,
                                       zusatz_ausgefallen=ausgefallen)
    doku = wissen / "Doku"
    assert {doku / "Zustand.md", doku / "Index.md", doku / "Ketten" / "K1.md"} <= set(geschrieben)
    assert not (doku / "Mitbauen").exists()


def test_ausgefallene_seite_wird_nicht_geloescht(wurzel, tmp_path, monkeypatch):
    import selbstbild.skill as skill

    wissen = tmp_path / "wissen"
    _schreiben(wissen, rs.seiten(_graph(), wurzel, {"offen": []}, tmp_path / "leer"))
    alt = wissen / "Doku" / rs.MITBAUEN_REL
    assert alt.is_file()
    monkeypatch.setattr(skill, "skill_neuer_space", lambda *a, **k: 1 / 0)
    ausgefallen: list[str] = []
    zusatz = rs.seiten(_graph(), wurzel, {"offen": []}, tmp_path / "leer", fehler=[], ausgefallen=ausgefallen)
    rb.notizen_schreiben(_graph(), [_punkt()], [], wissen, "Doku", LAUF, manifest_pfad=wissen.parent / "m.json",
                         zusatz=zusatz, zusatz_ausgefallen=ausgefallen)
    assert alt.is_file()


def test_kettenlinks_nutzen_echte_notiznamen(tmp_path):
    g = _graph()
    for kid in ("A/1", "A_1"):  # sicherer_name macht aus beiden "A_1"
        g.knoten_hinzu(Knoten(f"kette:{kid}", "kette", f"Kette {kid}", None, None, {"zustand_roh": "bricht"}))
        z = hop_zustand_neu()
        z["gemessen"] = {"wert": "bricht", "zeit": None}
        g.kante_hinzu(Kante(kante_id("hop", "dienst:c", "dienst:b", kid, 1), "hop", "dienst:c", "dienst:b", "seed", (),
                            {"kette": kid, "nr": "1", "mechanismus": "x", "zustand": z, "zustand_text": "bricht"}))
    wissen = tmp_path / "wissen"
    zusatz = [rs.Seite(rs.ZUSTAND_REL, lambda i, p: rs.seite_zustand(g, None, {"offen": []}, index_name=i,
                                                                      pruefliste_name=p))]
    rb.notizen_schreiben(g, [], [], wissen, "Doku", LAUF, manifest_pfad=wissen.parent / "m.json", zusatz=zusatz)
    seite = _lesen(wissen / "Doku" / "Zustand.md")
    ziele = re.findall(r"\[\[Doku/Ketten/([^\]]+)\]\] Kette (A/1|A_1)", seite)
    assert len(ziele) == 2 and len({z for z, _ in ziele}) == 2
    for name, _ in ziele:
        assert (wissen / "Doku" / "Ketten" / f"{name}.md").is_file(), name


def test_mitbauen_ohne_skill_absatz_und_digest_ueber_ganzen_skill(wurzel):
    skill = skill_neuer_space(wurzel)
    seite = rs.seite_mitbauen(wurzel, None)
    assert "Diese Datei wird" not in seite and "Nicht von Hand" not in seite
    assert f"grundlage: {json.dumps(rs._digest(skill))}" in seite
    assert "## Wohin" in seite and "## Nie hier anlegen" in seite


def test_zustand_zeigt_analyse_stand_im_digest():
    an = {"commit": "abcdef1234567890", "tot": {"sicher_tot": 1, "wahrscheinlich_tot": 2}, "hotspots_top": []}
    a = rs.seite_zustand(_graph(), an, {"offen": []})
    b = rs.seite_zustand(_graph(), dict(an, commit="12345678aaaa"), {"offen": []})
    assert "Analyse-Stand: `abcdef12`" in a
    assert a.split("\n")[2] != b.split("\n")[2]


def test_rueckverweise_folgen_der_umleitung(wurzel, tmp_path):
    wissen = tmp_path / "wissen"
    doku = wissen / "Doku"
    doku.mkdir(parents=True)
    (doku / "Index.md").write_bytes(b"von Hand")
    (doku / "Prüfliste.md").write_bytes(b"auch von Hand")
    _schreiben(wissen, rs.seiten(_graph(), wurzel, {"offen": [_punkt()]}, tmp_path / "leer"))
    z = _lesen(doku / "Zustand.md")
    m = _lesen(doku / "Mitbauen" / "Neuen Space erstellen mit Claude Code.md")
    assert "[[Doku/Index (Loop)]]" in m and "[[Doku/Index]]" not in m
    assert "[[Doku/Index (Loop)]]" in z and "[[Doku/Prüfliste (Loop)]]" in z and "[[Doku/Prüfliste]]" not in z


GOLD = {
    "Dienste/b.md": "5a2b94e5276c020aa893f4ce5d006194cf296388250a2fd5c911df385d82352e",
    "Dienste/c.md": "07410ec3bc58564ca59ca177519521fc3d827046e19dcf1976f88e6ac53c4a85",
    "Index.md": "514ee267193093306b2613a84d86e9630a4cfc5b2667a556eb92c33023adaac5",
    "Ketten/K1.md": "8777a7722e40b19574629433eb16796ff4f92f5bb49a48c453224505bbd16e6c",
    "Ketten/K2.md": "578d57d96b20e53a08902e778a6db2970fc98b837a9c6c1a22a20152c05351e6",
    "Prüfliste.md": "1e6bbda22c171bdcc9d1a8a43d064adb49e45e7359b1a367e1b6fe946ea295a5",
    "Spaces/a.md": "dc5486b4821b01b659d1d58de9524cc87fdfa42347d4d6010066e4102ef336f3",
}
GOLD_MANIFEST = "aebbf34e98e248a3623c5acef714190275e3521207874aeb117f555678d5ba9b"


def test_ohne_zusatz_byte_identisch_zum_stand_vor_task_8(tmp_path):
    """Goldwerte (SHA-256) stammen aus `rowboat.py` vor Task 8 (Commit 1240c9b) mit demselben Fixture."""
    wissen = tmp_path / "wissen"
    rb.notizen_schreiben(_graph(), [_punkt()], [_ledger_eintrag()], wissen, "Doku", LAUF,
                         commits={LAUF: "f" * 40}, spitzen={KETTEN: "f" * 40}, manifest_pfad=tmp_path / "m.json",
                         zusatz=[])
    ist = {p.relative_to(wissen / "Doku").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (wissen / "Doku").rglob("*.md")}
    assert ist == GOLD
    assert hashlib.sha256((tmp_path / "m.json").read_bytes()).hexdigest() == GOLD_MANIFEST
