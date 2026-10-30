"""doku_loop: Aussagen-Parser (Task 12, Ruling 36)."""

import dataclasses
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import aussagen as au  # noqa: E402
from selbstbild.doku import ketten_seed as ks  # noqa: E402
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku.konfig import SpaceKonfig  # noqa: E402

API = "def routen():\n    return ziel_aufruf()\n\n\ndef andere():\n    x = 1\n    return x\n"
BOM = "﻿"


def _repo(tmp_path):
    (tmp_path / "sp/m").mkdir(parents=True)
    (tmp_path / "sp/m/api.py").write_text(API, encoding="utf-8")
    konfig = dataclasses.replace(vm_laden(), repo=tmp_path,
                                 spaces={"m": SpaceKonfig("m", "sp/m", "werkzeuge")})
    docs = (SimpleNamespace(document_id="m.api.routen", file="api.py", start_line=1, end_line=2, symbol="routen",
                            source="def routen():\n    return ziel_aufruf()"),
            SimpleNamespace(document_id="m.api.andere", file="api.py", start_line=5, end_line=7, symbol="andere",
                            source="def andere():\n    x = 1\n    return x"))
    return konfig, {"m": SimpleNamespace(corpus_root=str(tmp_path / "sp/m"), documents=docs)}


DOC = [
    "---",
    "titel: Probe",
    "---",
    "# Doku",
    "",
    "## Abschnitt A",
    "",
    "Der Versand laeuft ueber `routen` in `sp/m/api.py:2`. Danach mehr Text,",
    "der in der zweiten Zeile weitergeht.",
    "",
    "| Wer | Wo |",
    "|---|---|",
    "| `andere` rechnet | `sp/m/api.py:6` |",
    "",
    "<!-- graph:begin hops K1 -->",
    "| # | Von → Nach | Mechanismus | Code-Stelle | Belegt durch | Zustand |",
    "|---|---|---|---|---|---|",
    "| 1 | A → B | Python `routen` | `sp/m/api.py:2` `m.api.routen` | 008 | läuft |",
    "",
    "Absatz im Block.",
    "<!-- graph:end -->",
    "",
    "### Unterabschnitt",
    "",
    "<!-- ein Kommentar",
    "ueber zwei Zeilen -->",
    "",
    "```mermaid",
    "graph TD; A-->B",
    "",
    "Kein Absatz `sp/m/api.py:2`",
    "```",
    "",
    "Siehe `sp/m/api.py:6` `m.api.routen` fuer Details.",
    "",
    "Die Route `routen` steht in [B1], [B9] gibt es nicht.",
    "",
    "## Belege",
    "",
    "[B1] sp/m/api.py:2 die Route",
    "",
]


def _schreiben(tmp_path, nl="\n", bom=False):
    pfad = tmp_path / "docs" / "probe.md"
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_bytes(((BOM if bom else "") + nl.join(DOC)).encode("utf-8"))
    return pfad


def _belege(a):
    return [(b.art, b.file, b.anker) for b in a.belege]


def test_aussagen_lesen_absaetze_und_tabellenzeilen_ohne_markierte_bloecke(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    liste = au.aussagen_lesen(_schreiben(tmp_path), konfig, manifeste, {}, "c0")
    assert [(a.zeile_start, a.zeile_ende) for a in liste] == [(8, 9), (13, 13), (34, 34), (36, 36)]
    absatz, zeile, gerendert, verweis = liste
    assert absatz.datei == "docs/probe.md"
    assert absatz.ueberschrift == "Doku > Abschnitt A"
    assert absatz.text == "\n".join(DOC[7:9])
    assert _belege(absatz) == [("symbol", "sp/m/api.py", "m.api.routen")]
    assert absatz.belege[0].commit == "c0"
    assert zeile.text == "| `andere` rechnet | `sp/m/api.py:6` |"
    assert _belege(zeile) == [("symbol", "sp/m/api.py", "m.api.andere")]
    assert gerendert.ueberschrift == "Doku > Abschnitt A > Unterabschnitt"
    assert verweis.id == au.aussage_id("docs/probe.md", verweis.ueberschrift, verweis.text)
    # nichts aus dem markierten Block, dem Kommentar, dem Code-Block, dem Tabellenkopf oder dem Abschnitt Belege
    texte = " ".join(a.text for a in liste)
    assert "Absatz im Block" not in texte and "| 1 |" not in texte and "Kommentar" not in texte
    assert "Kein Absatz" not in texte and "[B1] sp/m" not in texte and "| Wer |" not in texte


def test_gerenderte_anker_spanne_im_absatz_verankert_nichts(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    gerendert = au.aussagen_lesen(_schreiben(tmp_path), konfig, manifeste, {}, "c0")[2]
    # `m.api.routen` ist nur gerenderter Anker: die Stelle :6 liegt in `andere` (umschliessend), nicht `routen`
    assert _belege(gerendert) == [("symbol", "sp/m/api.py", "m.api.andere")]


def test_verweis_auf_abschnitt_belege_und_unaufgeloeste_verweise(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    verweis = au.aussagen_lesen(_schreiben(tmp_path), konfig, manifeste, {}, "c0")[3]
    assert _belege(verweis) == [("symbol", "sp/m/api.py", "m.api.routen")]  # [B9] still ignoriert


def test_belege_tabelle_und_unterabschnitt(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    pfad = tmp_path / "t.md"
    pfad.write_text("# T\n\nRechnet in [B2].\n\n### Belege\n\n| Nr | Stelle |\n|---|---|\n"
                    "| B2 | `sp/m/api.py:6` |\n", encoding="utf-8")
    (a,) = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c")
    assert a.datei == "t.md" and _belege(a) == [("symbol", "sp/m/api.py", "m.api.andere")]


def test_aussage_id_stabil_bei_umschreibung_des_restes():
    eins = au.aussage_id("d.md", "A > B", "Erster Satz. Rest eins.")
    assert eins == "d.md#A > B#" + hashlib.sha1("Erster Satz.".encode("utf-8")).hexdigest()[:8]
    assert au.aussage_id("d.md", "A > B", "Erster\n  Satz. Ganz anderer Rest!") == eins
    assert au.aussage_id("d.md", "A > B", "Zweiter Satz. Rest eins.") != eins
    assert au.aussage_id("d.md", "A", "Erster Satz. Rest eins.") != eins
    abk = au.aussage_id("d.md", "A", "Das ist z. B. wichtig. Rest.")
    assert abk == au.aussage_id("d.md", "A", "Das ist z. B. wichtig. Anderes.")
    assert abk != au.aussage_id("d.md", "A", "Das ist z. B. unwichtig. Rest.")


def test_aussage_ersetzen_erhaelt_crlf(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    pfad = _schreiben(tmp_path, nl="\r\n", bom=True)
    text = pfad.read_bytes().decode("utf-8")
    liste = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c0")
    assert [(a.zeile_start, a.zeile_ende) for a in liste] == [(8, 9), (13, 13), (34, 34), (36, 36)]
    absatz = liste[0]
    assert absatz.text == "\n".join(DOC[7:9])  # Text ohne Zeilenenden
    neu = au.aussage_ersetzen(text, absatz, "Neu: `routen` in `sp/m/api.py:1`.\nZweite Zeile.\nDritte.")
    assert "\n" not in neu.replace("\r\n", "")
    assert neu.startswith(BOM + "---\r\n")
    erwartet = text.replace("\r\n".join(DOC[7:9]), "Neu: `routen` in `sp/m/api.py:1`.\r\nZweite Zeile.\r\nDritte.")
    assert neu == erwartet
    assert "## Abschnitt A\r\n\r\nNeu:" in neu and "Dritte.\r\n\r\n| Wer |" in neu
    pfad.write_bytes(neu.encode("utf-8"))
    letzte = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c0")[-1]
    assert au.aussage_ersetzen(neu, letzte, "Kurz.").endswith(
        "\r\n\r\nKurz.\r\n\r\n## Belege\r\n\r\n[B1] sp/m/api.py:2 die Route\r\n")
    # letzte Zeile der Datei ohne Zeilenende bleibt ohne
    ende = tmp_path / "e.md"
    ende.write_bytes("# E\r\n\r\nSchluss ohne Ende".encode("utf-8"))
    (schluss,) = au.aussagen_lesen(ende, konfig, manifeste, {}, "c0")
    assert au.aussage_ersetzen(ende.read_bytes().decode("utf-8"), schluss, "A\nB") == "# E\r\n\r\nA\r\nB"


def test_aussage_ersetzen_verweigert_veraltete_aussage(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    pfad = _schreiben(tmp_path)
    text = pfad.read_text(encoding="utf-8")
    absatz = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c0")[0]
    with pytest.raises(ValueError):
        au.aussage_ersetzen(text.replace("Danach mehr Text", "Danach anderer Text"), absatz, "Neu.")
    with pytest.raises(ValueError):
        au.aussage_ersetzen(text, absatz, "  \n ")  # ersetzt nur, loescht nie
    with pytest.raises(ValueError):
        au.aussage_ersetzen("kurz\n", absatz, "Neu.")


def test_doppelte_ids_bekommen_suffix_und_pruefpunkt(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    pfad = tmp_path / "d.md"
    pfad.write_text("# T\n\n## H\n\nNicht belegt. Erstens.\n\nNicht belegt. Zweitens.\n\n"
                    "Nicht belegt. Drittens.\n\n## Andere\n\nNicht belegt. Hier nicht doppelt.\n", encoding="utf-8")
    liste = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c")
    basis = au.aussage_id("d.md", "T > H", "Nicht belegt.")
    assert [a.id for a in liste[:3]] == [basis, f"{basis}~2", f"{basis}~3"]
    assert liste[3].id == au.aussage_id("d.md", "T > Andere", "Nicht belegt.")
    assert len({a.id for a in liste}) == 4
    doppelt = [(a.zeile_start, p.art, p.aussage_id, p.zeile) for a in liste for p in a.pruefpunkte]
    assert doppelt == [(7, "aussage_id_doppelt", f"{basis}~2", 7), (9, "aussage_id_doppelt", f"{basis}~3", 9)]
    assert "Zeile 5" in liste[1].pruefpunkte[0].text


def test_erster_satz_zahlen_und_daten():
    assert au.erster_satz("Wir messen 298. Davon sind 202 exakt.") == "Wir messen 298."
    assert au.aussage_id("d", "h", "Wir messen 298. Davon sind 202 exakt.") == \
        au.aussage_id("d", "h", "Wir messen 298. Alles anders.")
    assert au.erster_satz("Seit dem 27.10. Messen wir neu. Rest.") == "Seit dem 27.10. Messen wir neu."
    assert au.erster_satz("Am 5. Okt. kam es. Rest.") == "Am 5. Okt. kam es."
    assert au.erster_satz("Version 3. danach klein. Rest.") == "Version 3. danach klein."


@pytest.mark.parametrize("ersatz", [
    "Erste Zeile.\n\nZweiter Absatz.",  # innere Leerzeile
    "# Ueberschrift", "Text\n## Unter", "```python", "~~~", "<!-- Kommentar -->", "Text\n---",
    "Text mit <!-- graph:begin hops K1 --> drin", "Text graph:end",
    "| a | b |",  # ein Absatz wird keine Tabellenzeile
])
def test_aussage_ersetzen_verweigert_strukturaenderung_im_absatz(tmp_path, ersatz):
    konfig, manifeste = _repo(tmp_path)
    pfad = _schreiben(tmp_path)
    absatz = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c0")[0]
    with pytest.raises(ValueError):
        au.aussage_ersetzen(pfad.read_bytes().decode("utf-8"), absatz, ersatz)


@pytest.mark.parametrize("ersatz", [
    "| `andere` rechnet |\n| `sp/m/api.py:6` |",  # zwei Zeilen
    "`andere` rechnet | `sp/m/api.py:6` |",  # beginnt nicht mit |
    "| `andere` rechnet | `sp/m/api.py:6`",  # endet nicht mit |
    "| a | b | c |",  # andere Zellenzahl
    "| a |",
    "Kein Tabellentext.",
])
def test_aussage_ersetzen_tabellenzeile_nur_durch_gleich_breite_zeile(tmp_path, ersatz):
    konfig, manifeste = _repo(tmp_path)
    pfad = _schreiben(tmp_path)
    text = pfad.read_bytes().decode("utf-8")
    zeile = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c0")[1]
    with pytest.raises(ValueError):
        au.aussage_ersetzen(text, zeile, ersatz)


def test_aussage_ersetzen_tabellenzeile_gleich_breit_mit_escaptem_strich(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    pfad = _schreiben(tmp_path)
    text = pfad.read_bytes().decode("utf-8")
    zeile = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c0")[1]
    neu = au.aussage_ersetzen(text, zeile, r"| `a|b` oder x \| y | `sp/m/api.py:5` |")
    assert "\n| `a|b` oder x \\| y | `sp/m/api.py:5` |\n" in neu


def _nackt(text):
    return au.Aussage("d.md#h#x", "d.md", "h", 1, 1, text, ())


@pytest.mark.parametrize("ersatz", [
    "Text mit\rlosem Wagenruecklauf", "Text\x0bmit VT", "Text\x0cmit FF", "Text mit LS", "Text mit PS",
    "Text\x85mit NEL",
    "Text\n    eingerueckter Code", "Text\n\tTab-Code", "    Code am Anfang", "\tTab am Anfang",
    "Text\n===", "Text\n=", "Text\n--", "Text\n-",
    "Text\n* * *", "Text\n***", "Text\n_ _ _", "Text\n___", "Text\n- - -",
    "Text\n- Punkt", "Text\n+ Punkt", "Text\n* Punkt", "Text\n> Zitat", "Text\n>Zitat", "Text\n1. Erstens",
    "Text\n2) Zweitens", "Text\n   - eingerueckter Punkt",
    "- Liste am Anfang", "+ Liste am Anfang", "> Zitat am Anfang", "1. Nummer am Anfang", "3) Nummer am Anfang",
])
def test_ersatz_pruefen_verweigert_markdown_brueche(ersatz):
    with pytest.raises(ValueError):
        au.ersatz_pruefen(_nackt("Ein Absatz.\nZweite Zeile."), ersatz)


def test_ersatz_pruefen_erste_zeile_darf_form_der_aussage_behalten():
    au.ersatz_pruefen(_nackt("- Punkt eins"), "- Punkt neu")
    au.ersatz_pruefen(_nackt("* Punkt eins"), "* Punkt neu")
    au.ersatz_pruefen(_nackt("1. Schritt"), "2. Neuer Schritt")
    au.ersatz_pruefen(_nackt("> Zitat alt"), "> Zitat neu")
    for alt, neu in (("- Punkt", "1. Nummer"), ("1. Schritt", "- Punkt"), ("> Zitat", "- Punkt"),
                     ("- Punkt eins", "- a\n- b")):
        with pytest.raises(ValueError):
            au.ersatz_pruefen(_nackt(alt), neu)
    # gewoehnlicher Text mit Zahlen, Strichen und Zeichen mitten in der Zeile bleibt erlaubt
    au.ersatz_pruefen(_nackt("Absatz."), "Absatz mit 3. Wert - und > Zeichen\n1.5 Sekunden, -x und *betont*\n  zwei")
    au.ersatz_pruefen(_nackt("| a | b |"), "| - | > |")


def test_aussage_ersetzen_verweigert_markdown_bruch(tmp_path):
    konfig, manifeste = _repo(tmp_path)
    pfad = _schreiben(tmp_path)
    absatz = au.aussagen_lesen(pfad, konfig, manifeste, {}, "c0")[0]
    with pytest.raises(ValueError):
        au.aussage_ersetzen(pfad.read_bytes().decode("utf-8"), absatz, "Neu.\n    eingerueckt")


def test_handgepflegt():
    assert au.handgepflegt("---\npflege: hand\ntitel: x\n---\n# T\n")
    assert au.handgepflegt(BOM + "---\r\npflege: hand\r\n---\r\n# T\r\n")
    assert au.handgepflegt("# T\n\n<!-- handgepflegt -->\n\nText\n")
    assert au.handgepflegt("# T\n\n  <!--  handgepflegt  -->  \n")
    assert not au.handgepflegt("---\npflege: loop\n---\n# T\n")
    assert not au.handgepflegt("# T\n\npflege: hand\n")  # kein Frontmatter
    assert not au.handgepflegt("# T\n\nMit `<!-- handgepflegt -->` markiert man Dateien.\n")
    assert not au.handgepflegt("# T\n\n```\n<!-- handgepflegt -->\n```\n")


def test_namen_und_tokens_wie_im_seed():
    hz = ks.HopZeile("K1", "1", "A", "B", "SQL `marketing.versand_sperren()`", ("x.py:1",), "", "", 1,
                     von_nach_roh="`mail_dispatch` → B", code_stelle_roh="`x.py:1`")
    text = " ".join((hz.von_nach_roh, hz.mechanismus, "`x.py:1` `m.api.routen`"))
    namen, tokens = ks.namen_und_tokens_aus_text(text, schwach={"mail_dispatch"})
    assert namen == ks.zeilen_namen(hz)
    assert tokens == ks.zeilen_tokens(hz, schwach={"mail_dispatch"})
    assert "routen" not in namen and tokens == {"marketing.versand_sperren"}
