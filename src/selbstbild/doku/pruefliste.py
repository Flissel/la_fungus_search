"""Pruefliste des Doku-Loops (lokal, ausserhalb des Repos; Task 17 macht daraus Rowboat-Notizen).

`pruefliste-<hash>.json` im Marker-Ordner:
- `offen`    Punkte, die ein Mensch entscheiden soll (Urteile, verschwundene Belege, Hand-Edits, Torwaechter, ...);
             dedupliziert nach art/datei/aussage_id/text, nie automatisch entfernt
- `struktur` Verankerungs-, Bau- und Render-Pruefpunkte; bei vollem Bau ersetzt, sonst vereinigt
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .konfig import Konfig
from .stand import lokaler_ordner, repo_schluessel
from .typen import Pruefpunkt


def beleg_untracked(datei: str) -> Pruefpunkt:
    """Item 6: ein Beleg auf eine nicht versionierte Datei (ein Eintrag je Datei, dedupliziert)."""
    return Pruefpunkt("beleg_untracked", f"{datei} ist nicht versioniert; kein Signal", datei, None, None)


def beleg_nicht_messbar(datei: str, submodul: str) -> Pruefpunkt:
    """Ruling 50: Beleg in einem nicht initialisierten Submodul (ein Eintrag je Submodul, dedupliziert)."""
    return Pruefpunkt("beleg_nicht_messbar", f"Submodul {submodul} ist nicht initialisiert; Belege darin sind nicht "
                      "messbar, kein Signal", submodul, None, None)


def pruefliste_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"pruefliste-{repo_schluessel(konfig.repo)}.json"


def _pp_dict(p: Pruefpunkt, lauf: str) -> dict[str, object]:
    return {"art": p.art, "text": p.text, "datei": p.datei, "zeile": p.zeile, "aussage_id": p.aussage_id,
            "lauf": lauf}


# Arten, deren Text wechselnde Einzelheiten traegt (Zahl/Ids der Aussagen): zusammengefuehrt wird ueber die
# `aussage_id` (`<datei>#<digest12>`), der erste Text bleibt (Task 4, Fix-Runde 2); `doku_ziel_fremd` mit der Datei
# als `aussage_id` (Final-Fix I4: ein Eintrag je Datei, nicht je fremdem Inhalt)
OHNE_TEXT_ZUSAMMENFUEHREN = frozenset({"beleg_datei_geaendert", "doku_ziel_fremd"})


def _schluessel(d: Mapping[str, object]) -> str:
    if d.get("art") in OHNE_TEXT_ZUSAMMENFUEHREN:
        return f"{d.get('art')}|{d.get('datei')}|{d.get('aussage_id')}"
    return f"{d.get('art')}|{d.get('datei')}|{d.get('aussage_id')}|{d.get('text')}"


def pruefliste_laden(konfig: Konfig) -> dict[str, list[dict[str, object]]]:
    try:
        roh = json.loads(pruefliste_pfad(konfig).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"offen": [], "struktur": []}
    if not isinstance(roh, dict):
        return {"offen": [], "struktur": []}
    return {k: [e for e in roh.get(k) or [] if isinstance(e, dict)] for k in ("offen", "struktur")}


def neue_offene(konfig: Konfig, offen: Sequence[Pruefpunkt]) -> int:
    """Wie viele der Punkte noch nicht in `offen` stehen (ohne zu schreiben)."""
    bekannt = {_schluessel(e) for e in pruefliste_laden(konfig)["offen"]}
    return len({_schluessel(_pp_dict(p, "")) for p in offen} - bekannt)


def pruefliste_ergaenzen(konfig: Konfig, offen: Sequence[Pruefpunkt], struktur: Sequence[Pruefpunkt], *,
                         voll: bool, lauf: str,
                         ersetzen: Callable[[Mapping[str, object]], bool] | None = None) -> int:
    """Neue offene Punkte anhaengen (dedupliziert); Struktur bei vollem Bau ersetzen, sonst vereinigen.
    `ersetzen` (Task 5, Fix-Runde 1): alte Struktur-Eintraege, fuer die es wahr ist, fallen auch ohne vollen Bau weg
    (Arten, die jeder Lauf vollstaendig neu berechnet, z. B. die Ereignis-Punkte). Das Dateiformat bleibt gleich.
    Rueckgabe: Zahl der neuen offenen Punkte."""
    alt = pruefliste_laden(konfig)
    bekannt = {_schluessel(e) for e in alt["offen"]}
    neu = 0
    for p in offen:
        d = _pp_dict(p, lauf)
        if _schluessel(d) not in bekannt:
            bekannt.add(_schluessel(d))
            alt["offen"].append(d)
            neu += 1
    basis = [] if voll else [e for e in alt["struktur"] if ersetzen is None or not ersetzen(e)]
    gesehen = {_schluessel(e) for e in basis}
    for p in struktur:
        d = _pp_dict(p, lauf)
        if _schluessel(d) not in gesehen:
            gesehen.add(_schluessel(d))
            basis.append(d)
    alt["struktur"] = basis
    pfad = pruefliste_pfad(konfig)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps(alt, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(pfad)
    return neu
