"""Rowboat-Seiten ohne LLM (Task 8): Mitbauen aus der Skill-Quelle, Zustand, Schreibschutz, Index, Trockenlauf.

Alles in tmp_path; der echte Wissensordner wird nie angefasst."""
from __future__ import annotations

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

from test_doku_rowboat import LAUF, _graph, _lesen, _punkt  # noqa: E402

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


def test_ohne_zusatz_unveraendertes_verhalten(tmp_path):
    wissen = tmp_path / "wissen"
    g = rb.notizen_schreiben(_graph(), [_punkt()], [], wissen, "Doku", LAUF, manifest_pfad=wissen.parent / "m.json")
    assert not (wissen / "Doku" / "Zustand.md").exists()
    assert "Zustand" not in _lesen(wissen / "Doku" / "Index.md") and g


def test_trocken_meldet_ohne_zu_schreiben(wurzel, tmp_path):
    wissen = tmp_path / "wissen"
    leer = tmp_path / "leer"
    zusatz = rs.seiten(_graph(), wurzel, {"offen": []}, leer)
    manifest = wissen.parent / "m.json"
    assert rs.voraussichtlich_geschrieben(zusatz, wissen, "Doku", manifest) == ([rs.MITBAUEN_REL, rs.ZUSTAND_REL], [])
    assert not wissen.exists() and not manifest.exists()
    _schreiben(wissen, zusatz)
    assert rs.voraussichtlich_geschrieben(zusatz, wissen, "Doku", manifest) == ([], [])
    (wissen / "Doku" / "Zustand.md").write_bytes(b"von Hand")
    neu = rs.seiten(_kaputter_graph(), wurzel, {"offen": []}, leer)
    schreiben, hand = rs.voraussichtlich_geschrieben(neu, wissen, "Doku", manifest)
    assert hand == [rs.ZUSTAND_REL] and (wissen / "Doku" / "Zustand.md").read_bytes() == b"von Hand"
