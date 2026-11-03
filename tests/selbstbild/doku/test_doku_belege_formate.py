"""Stufe 4, Task 4: beide Belege-Formate (Klammer `[Bn]`, Vorlage `## 7. Belege` / `- B<n>: <art> <ziel>`) und die
Prosa-Liste der echten Space-Doku (`## Belege` mit Backtick-Pfaden, ohne Nummer)."""

from test_doku_aussagen import _repo  # noqa: F401
from selbstbild.doku import aussagen as au
from selbstbild.doku import belege_formate as bf

REGISTRY = "version: 1\nspaces:\n  ideas:\n    agent: ideas-agent\n"


def _lesen(tmp_path, konfig, manifeste, zeilen, name="docs/spaces/019_probe.md"):
    pfad = tmp_path / name
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_bytes("\n".join(zeilen).encode("utf-8"))
    return au.aussagen_lesen(pfad, konfig, manifeste, {}, "c1")


def test_vorlage_7_belege(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    (tmp_path / "vibemind-os/config").mkdir(parents=True)
    (tmp_path / "vibemind-os/config/space_agent_registry.yml").write_text(REGISTRY, encoding="utf-8")
    zeilen = [
        "# Probe",
        "",
        "## 6. Umsetzungsstand und Grenzen",
        "",
        "Die Registry fuehrt den Space ideas mit eigenem Agenten [B1].",
        "",
        "## 7. Belege",
        "",
        "- B1: datei vibemind-os/config/space_agent_registry.yml:1",
        "",
    ]
    assert bf.belege_abschnitt("\n".join(zeilen)) == {1: "vibemind-os/config/space_agent_registry.yml:1"}
    liste = _lesen(tmp_path, konfig, manifeste, zeilen)
    assert [a.text for a in liste] == ["Die Registry fuehrt den Space ideas mit eigenem Agenten [B1]."]
    assert {b.file for b in liste[0].belege} == {"vibemind-os/config/space_agent_registry.yml"}


def test_belege_prosa_liste(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    zeilen = [
        "# Marketing",
        "",
        "## Was hier passiert",
        "",
        "Der Space versendet nichts selbst.",
        "",
        "## Belege",
        "",
        "Gelesen, alle unter `vibemind-os/spaces/marketing/` sofern nicht anders angegeben:",
        "",
        "- `vibemind-os/spaces/marketing/api/main.py:12`",
        "",
    ]
    text = "\n".join(zeilen)
    assert bf.belege_abschnitt(text) == {}
    assert bf.belege_stellen(text) == ["vibemind-os/spaces/marketing/",
                                       "vibemind-os/spaces/marketing/api/main.py:12"]
    liste = _lesen(tmp_path, konfig, manifeste, zeilen)  # kein Absturz, der Abschnitt ist keine Aussage
    assert [a.text for a in liste] == ["Der Space versendet nichts selbst."]


def test_klammer_format_bleibt_unveraendert():
    text = "\n".join([
        "## Belege",
        "",
        "[B1] sp/m/api.py:2 die Route",
        "- [B2]: sp/m/api.py:6",
        "",
        "| Nr | Ziel |",
        "|---|---|",
        "| B3 | `sp/m/api.py:1` |",
        "| [B4] | x.py |",
        "[B1] doppelt, der erste gewinnt",
    ])
    assert bf.belege_abschnitt(text) == {1: "sp/m/api.py:2 die Route", 2: "sp/m/api.py:6", 3: "`sp/m/api.py:1`",
                                        4: "x.py"}
    assert bf.belege_stellen(text) == []  # nummerierte Eintraege sind keine Prosa-Stellen


def test_vorlage_zeile_ohne_art_und_mit_messung():
    text = "\n".join([
        "## 7. Belege",
        "",
        "- B1: datei vibemind-os/x.py:3-9",
        "- B2: vibemind-os/y.py:4",
        "- B3: messung docker service ls",
        "* B4: datei `a/b.py`",
    ])
    assert bf.belege_abschnitt(text) == {1: "vibemind-os/x.py:3-9", 2: "vibemind-os/y.py:4",
                                        3: "docker service ls", 4: "`a/b.py`"}


def test_belege_kopf_varianten():
    assert bf.ist_belege_kopf("## Belege")
    assert bf.ist_belege_kopf("### Belege")
    assert bf.ist_belege_kopf("## 7. Belege")
    assert bf.ist_belege_kopf("## Belege (gelesene Dateien)")
    assert not bf.ist_belege_kopf("# Belege")
    assert not bf.ist_belege_kopf("#### Belege")
    assert not bf.ist_belege_kopf("## Bekannte Fallen (als Regeln, mit Beleg)")
    assert not bf.ist_belege_kopf("## Belegexyz")
    assert not bf.ist_belege_kopf("Belege")


def test_belege_abschnitt_endet_an_gleicher_ebene_und_ignoriert_code():
    text = "\n".join([
        "## Belege",
        "",
        "```",
        "[B7] im Code-Block zaehlt nicht",
        "```",
        "[B1] a.py:1",
        "### Unterabschnitt",
        "[B2] b.py:2",
        "## Danach",
        "[B3] c.py:3",
        "- B4: datei d.py:4",
    ])
    assert bf.belege_abschnitt(text) == {1: "a.py:1", 2: "b.py:2"}


def test_belege_mit_bom_und_crlf():
    text = "﻿## Belege\r\n\r\n- B1: datei a.py:1\r\n- `b/c.py:2`\r\n"
    assert bf.belege_abschnitt(text) == {1: "a.py:1"}
    assert bf.belege_stellen(text) == ["b/c.py:2"]


def test_aussage_mit_verweis_auf_vorlage_und_klammer_gemischt(tmp_path):
    """`[Bn]` loest gegen beide Zeilenformen auf; die Bindung ueber Backticks im Absatz bleibt."""
    konfig, manifeste = _repo(tmp_path)
    zeilen = [
        "# Probe",
        "",
        "Die Route `routen` liefert das Ziel [B1].",
        "",
        "Die Hilfe `andere` rechnet in `sp/m/api.py:6` lokal.",
        "",
        "## 7. Belege",
        "",
        "- B1: datei sp/m/api.py:2",
    ]
    liste = _lesen(tmp_path, konfig, manifeste, zeilen)
    assert [(b.art, b.file, b.anker) for b in liste[0].belege] == [("symbol", "sp/m/api.py", "m.api.routen")]
    assert [(b.art, b.file, b.anker) for b in liste[1].belege] == [("symbol", "sp/m/api.py", "m.api.andere")]


def test_datei_quittung_nur_in_der_vorlagenform(tmp_path):
    """Nur eine erklaerte `datei`-Quittung bindet an die ganze Datei, wo der YAML-Anker zu grob waere; die
    Klammerform bleibt wie bisher (`anker_zu_grob`, kein Beleg)."""
    konfig, manifeste = _repo(tmp_path)
    (tmp_path / "vibemind-os/config").mkdir(parents=True)
    (tmp_path / "vibemind-os/config/space_agent_registry.yml").write_text(REGISTRY, encoding="utf-8")
    klammer = ["# P", "", "Die Registry fuehrt den Space ideas [B1].", "", "## Belege", "",
               "[B1] vibemind-os/config/space_agent_registry.yml:1", ""]
    liste = _lesen(tmp_path, konfig, manifeste, klammer)
    assert liste[0].belege == () and [p.art for p in liste[0].pruefpunkte] == ["anker_zu_grob"]
    vorlage = [*klammer[:4], "## 7. Belege", "", "- B1: datei vibemind-os/config/space_agent_registry.yml:1", ""]
    liste = _lesen(tmp_path, konfig, manifeste, vorlage)
    assert [(b.art, b.file, b.zeile) for b in liste[0].belege] == [
        ("datei", "vibemind-os/config/space_agent_registry.yml", 1)]
    assert liste[0].pruefpunkte == ()
    messung = [*klammer[:4], "## 7. Belege", "", "- B1: messung vibemind-os/config/space_agent_registry.yml:1", ""]
    assert _lesen(tmp_path, konfig, manifeste, messung)[0].belege == ()


def test_datei_quittung_nicht_lesbar_wird_pruefpunkt(tmp_path, monkeypatch):
    """Fix-Runde 1, Item 5: scheitert das Lesen der Datei (OSError), faellt die Quittung nicht still weg."""
    from selbstbild.doku import anker

    konfig, manifeste = _repo(tmp_path)
    (tmp_path / "vibemind-os/config").mkdir(parents=True)
    (tmp_path / "vibemind-os/config/space_agent_registry.yml").write_text(REGISTRY, encoding="utf-8")

    def gesperrt(*a, **kw):
        raise PermissionError("gesperrt")

    monkeypatch.setattr(anker, "beleg_fuer_datei", gesperrt)
    zeilen = ["# P", "", "Die Registry fuehrt den Space ideas [B1].", "", "## 7. Belege", "",
              "- B1: datei vibemind-os/config/space_agent_registry.yml:1", ""]
    liste = _lesen(tmp_path, konfig, manifeste, zeilen)
    assert liste[0].belege == ()
    punkte = [p for p in liste[0].pruefpunkte if p.art == "beleg_nicht_lesbar"]
    assert len(punkte) == 1 and "vibemind-os/config/space_agent_registry.yml" in punkte[0].text
    assert punkte[0].aussage_id == liste[0].id and "gesperrt" not in punkte[0].text  # nur der Typ, nie der Text
