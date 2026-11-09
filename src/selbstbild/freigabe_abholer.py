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
from selbstbild.funde.github import GitHub, GitHubFehler
from selbstbild.funde.rueckfrage import ANTWORTEN
from selbstbild.funde.sammeln import _aussen_name
from selbstbild.funde.sync import rueckfrage_ziel

__all__ = ["abholen", "freigabe_aus_umgebung", "main"]

LOG = logging.getLogger("selbstbild.freigabe_abholer")
ART = "selbstbild_label"


def _env_wert(e: Mapping[str, str], name: str) -> str:
    """`SELBSTBILD_<name>` zuerst (der Learning-Claw am selben PC nutzt die unpraefixierten Namen), sonst `<name>`."""
    return e.get(f"SELBSTBILD_{name}", "").strip() or e.get(name, "").strip()


def freigabe_aus_umgebung(env: Mapping[str, str] | None = None) -> FreigabeEingang | None:
    """Client aus `SELBSTBILD_FREIGABE_URL` und `SELBSTBILD_FREIGABE_SCHLUESSEL_DATEI` (je Name Rueckfall auf
    `FREIGABE_URL` / `FREIGABE_SCHLUESSEL_DATEI`); `None`, wenn eins fehlt oder die Datei nicht da ist."""
    e = os.environ if env is None else env
    url = _env_wert(e, "FREIGABE_URL")
    datei = _env_wert(e, "FREIGABE_SCHLUESSEL_DATEI")
    if not url or not datei:
        return None
    try:
        if not Path(datei).is_file():
            return None
    except OSError:  # z. B. PermissionError: nie den ganzen Lauf abbrechen
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
            art = e.art if isinstance(e, GitHubFehler) else type(e).__name__  # `art` ist schluesselfrei (GitHubFehler)
            LOG.warning("Auftrag fehlgeschlagen (%s)", art)
            freigabe.ergebnis(auftrag_id, False, art)
    return getan


def main(argv: Sequence[str] | None = None) -> int:
    """Ein Durchgang, Exit 0 (auch ohne Umgebung oder ohne Ziel: der Abholer ist dann aus)."""
    p = argparse.ArgumentParser(prog="selbstbild.freigabe_abholer", description=__doc__.splitlines()[0])
    p.add_argument("wurzel", type=Path, help="Wurzel des vermessenen Repos (für Regeln und Issue-Ziel)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    freigabe = freigabe_aus_umgebung()
    if freigabe is None:
        # alle 2 Minuten, das Log wird nie rotiert: nur sichtbar, wenn die Umgebung halb gesetzt ist
        halb = bool(_env_wert(os.environ, "FREIGABE_URL")) != bool(
            _env_wert(os.environ, "FREIGABE_SCHLUESSEL_DATEI"))
        LOG.log(logging.INFO if halb else logging.DEBUG,
                "Abholer aus: FREIGABE_URL oder FREIGABE_SCHLUESSEL_DATEI fehlt oder Datei nicht lesbar")
        return 0
    wurzel = Path(args.wurzel)
    ziel = rueckfrage_ziel(wurzel, _aussen_name(wurzel))  # dieselbe Regel wie `auto`: erst das Ziel, dann das Konto
    if ziel is None or not ziel.konto:
        LOG.debug("Abholer aus: kein Issue-Ziel mit Konto")
        return 0
    n = abholen(freigabe, GitHub(ziel.konto), ziel.repo)
    if n:
        LOG.info("Abholer: %d Auftrag/Aufträge ausgeführt", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
