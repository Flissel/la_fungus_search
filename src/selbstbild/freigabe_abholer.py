"""Abholer am PC: holt Label-Aufträge aus dem Freigabe-Eingang und setzt das Label auf GitHub.

`python -m selbstbild.freigabe_abholer <wurzel>` macht genau einen Durchgang (die geplante Aufgabe ruft ihn alle
2 Minuten): Aufträge der eigenen Herkunft holen, je Auftrag prüfen, Label setzen, Ergebnis melden. Der Schlüssel
(`FREIGABE_SCHLUESSEL_DATEI`) und GitHub-Schreibrechte bleiben am PC; die VM sieht beides nie.

Geprüft wird alles, auch wenn der Server es schon prüft: Art `selbstbild_label`, Repo = das Issue-Ziel der Regeln,
`nummer` ein echtes `int` (kein `bool`) > 0, Label aus `ANTWORTEN`. Bei einem Fehler geht nur der Typname als Ergebnis
zurück, nie die Meldung (sie kann Schlüssel enthalten). Geloggt wird nie ein Schlüssel.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from selbstbild.freigabe_eingang import FreigabeEingang
from selbstbild.funde.github import GitHub
from selbstbild.funde.rueckfrage import ANTWORTEN
from selbstbild.funde.sammeln import _aussen_name
from selbstbild.funde.sync import ziele_aus_regeln
from selbstbild.regeln import REGELDATEI, regeln_laden

__all__ = ["abholen", "freigabe_aus_umgebung", "main"]

LOG = logging.getLogger("selbstbild.freigabe_abholer")
ART = "selbstbild_label"


def freigabe_aus_umgebung(env: Mapping[str, str] | None = None) -> FreigabeEingang | None:
    """Client aus `FREIGABE_URL` und `FREIGABE_SCHLUESSEL_DATEI`; `None`, wenn eins fehlt oder die Datei nicht da ist."""
    e = os.environ if env is None else env
    url = e.get("FREIGABE_URL", "").strip()
    datei = e.get("FREIGABE_SCHLUESSEL_DATEI", "").strip()
    if not url or not datei or not Path(datei).is_file():
        return None
    return FreigabeEingang(url, Path(datei))


def _ablehnen(freigabe: FreigabeEingang, auftrag_id: str, grund: str) -> None:
    LOG.warning("Auftrag abgelehnt: %s", grund)
    freigabe.ergebnis(auftrag_id, False, grund)


def abholen(freigabe: FreigabeEingang, gh: GitHub, repo: str) -> int:
    """Ein Durchgang; Anzahl der ausgeführten Aufträge. Wirft nie (außer bei Programmierfehlern vor der Schleife)."""
    auftraege = freigabe.auftraege()
    if not auftraege:
        return 0
    getan = 0
    for a in auftraege:
        auftrag_id = a.get("id") if isinstance(a, dict) else None
        if not isinstance(auftrag_id, str) or not auftrag_id:
            LOG.warning("Auftrag ohne Kennung übersprungen")
            continue
        try:
            if a.get("art") != ART:
                _ablehnen(freigabe, auftrag_id, "unbekannte_art")
                continue
            nutzlast = a.get("nutzlast")
            if not isinstance(nutzlast, dict):
                _ablehnen(freigabe, auftrag_id, "nutzlast_unzulaessig")
                continue
            if nutzlast.get("repo") != repo:
                _ablehnen(freigabe, auftrag_id, "repo_unzulaessig")
                continue
            nummer: Any = nutzlast.get("nummer")
            if not isinstance(nummer, int) or isinstance(nummer, bool) or nummer <= 0:
                _ablehnen(freigabe, auftrag_id, "nummer_unzulaessig")
                continue
            label = nutzlast.get("label")
            if not isinstance(label, str) or label not in ANTWORTEN:
                _ablehnen(freigabe, auftrag_id, "label_unzulaessig")
                continue
            for anderes in ANTWORTEN:  # nie zwei Antwort-Labels: der tägliche Lauf würde nachfragen
                if anderes != label:
                    gh.label_entfernen(repo, nummer, anderes)
            gh.labels_setzen(repo, nummer, [label])
            if freigabe.ergebnis(auftrag_id, True):
                getan += 1
        except Exception as e:  # noqa: BLE001 -- nur der Typname geht zurück (Meldungen können Schlüssel tragen)
            LOG.warning("Auftrag fehlgeschlagen (%s)", type(e).__name__)
            freigabe.ergebnis(auftrag_id, False, type(e).__name__)
    return getan


def main(argv: Sequence[str] | None = None) -> int:
    """Ein Durchgang, Exit 0 (auch ohne Umgebung oder ohne Ziel: der Abholer ist dann aus)."""
    p = argparse.ArgumentParser(prog="selbstbild.freigabe_abholer", description=__doc__.splitlines()[0])
    p.add_argument("wurzel", type=Path, help="Wurzel des vermessenen Repos (für Regeln und Issue-Ziel)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    freigabe = freigabe_aus_umgebung()
    if freigabe is None:
        LOG.info("Abholer aus: FREIGABE_URL oder FREIGABE_SCHLUESSEL_DATEI fehlt")
        return 0
    wurzel = Path(args.wurzel)
    datei = wurzel / REGELDATEI
    regeln = regeln_laden(datei.read_text(encoding="utf-8") if datei.is_file() else None)
    aussen = _aussen_name(wurzel)
    ziel = next((z for z in ziele_aus_regeln(regeln) if z.zustaendig(aussen) and z.konto), None)
    if ziel is None or ziel.konto is None:
        LOG.info("Abholer aus: kein Issue-Ziel mit Konto")
        return 0
    n = abholen(freigabe, GitHub(ziel.konto), ziel.repo)
    LOG.info("Abholer: %d Auftrag/Aufträge ausgeführt", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
