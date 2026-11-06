"""Zeilen und Text der Telegram-Zusammenfassung des taeglichen Laufs (Stufe 6a, Task 4).

Reine Funktionen ohne Seiteneffekte: Klartext, deutsch, kurz, mit Links. Ob eine Zeile eine Nachricht ausloest
(`zeilen`) oder nur mitgeht (`kontext`), entscheidet hier die Rueckgabe; `selbstbild.auto` sendet.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any

from .funde.rueckfrage import _MARKE, RueckfrageBericht
from .funde.sync import _body

TEXT_KOPF = "Selbstbild, {tag}:"
ERSATZ_TEXT = "Selbstbild: Lauf fertig, Details im Bericht (Text zurückgehalten)."
MAX_TEXT = 3500  # Telegram nimmt 4096 Zeichen
ERINNERN_AB_TAGEN = 7
_NEU = ("anlegen", "sammel_anlegen")
_AKTUALISIERT = ("bearbeiten", "sammel_bearbeiten", "oeffnen", "sammel_oeffnen")
_GESCHLOSSEN = ("schliessen", "sammel_schliessen")


def nachricht(zeilen: list[str], tag: date) -> str:
    """Kopfzeile, dann je Zeile `- …`; gekappt bei `MAX_TEXT` mit Hinweis auf die restlichen Zeilen."""
    teile = [TEXT_KOPF.format(tag=f"{tag:%d.%m.%Y}")]
    laenge = len(teile[0])
    for i, z in enumerate(zeilen):
        zeile = f"- {z}"
        if laenge + 1 + len(zeile) > MAX_TEXT - 80:
            teile.append(f"- … und {len(zeilen) - i} weitere Zeilen (Details im Bericht)")
            break
        teile.append(zeile)
        laenge += 1 + len(zeile)
    return "\n".join(teile)


def kurz_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def nummer(issue: dict[str, Any]) -> int | None:
    n = issue.get("number")
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None


def anzahl(wert: object) -> int:
    return wert if isinstance(wert, int) and not isinstance(wert, bool) and wert > 0 else 0


def ziel_fehler(daten: dict[str, Any] | None) -> list[str]:
    """Die Fehlerarten der Ziele (`ziele[*].fehler`), sortiert und ohne Doppelte."""
    if daten is None:
        return []
    return sorted({z["fehler"] for z in daten.get("ziele") or []
                   if isinstance(z, dict) and isinstance(z.get("fehler"), str) and z["fehler"]})


def funde_zeilen(daten: dict[str, Any],
                 tor_alt: str | None) -> tuple[list[str], list[str], list[str], str | None]:
    """(zeilen, fehler_zeilen, kontext, Tor-Text) aus der JSON-Zusammenfassung von `funde`. Der Stand des Tors loest
    nur eine Nachricht aus, wenn er sich gegen `tor_alt` geaendert hat; sonst geht er nur mit. Die Fehlerzeilen je
    Ziel gibt es getrennt: ob sie eine Nachricht ausloesen, entscheidet der Fehlerschluessel (nur beim ersten Mal)."""
    zeilen: list[str] = []
    fehler: list[str] = []
    kontext: list[str] = []
    for z in daten.get("ziele") or []:
        if not isinstance(z, dict):
            continue
        repo = str(z.get("ziel") or "?")
        je = z.get("aktionen_je_art") if isinstance(z.get("aktionen_je_art"), dict) else {}
        neu, akt, zu = (sum(anzahl(je.get(k)) for k in arten) for arten in (_NEU, _AKTUALISIERT, _GESCHLOSSEN))
        if neu or akt or zu:
            trocken = " (trocken)" if z.get("modus") == "trocken" else ""
            zeilen.append(f"Issues in {repo}{trocken}: {neu} neu, {akt} aktualisiert, {zu} geschlossen: "
                          f"https://github.com/{repo}/issues?q=label%3Aselbstbild")
        if isinstance(z.get("fehler"), str) and z["fehler"]:
            fehler.append(f"Issues in {repo}: Fehler {z['fehler']}")
    tor = daten.get("tor_tot")
    text = tor.get("text") if isinstance(tor, dict) else None
    if not isinstance(text, str) or not text:
        return zeilen, fehler, kontext, tor_alt
    (zeilen if text != tor_alt else kontext).append(text)
    return zeilen, fehler, kontext, text


def rueckfrage_zeilen(rb: RueckfrageBericht, gemeldet: dict[str, list[Any]],
                      trocken: bool) -> tuple[list[str], list[str], dict[str, list[Any]]]:
    """(zeilen, kontext, neues `gemeldet`). Mehrfach-Labels und wartende Antworten stehen in jedem Lauf im Bericht;
    gemeldet wird nur, was nicht schon in `gemeldet` steht (wartende nur als Hash, nie mit Text im Stand)."""
    zeilen: list[str] = []

    def link(nr: int) -> str:
        return rb.links.get(nr, "")

    zeilen += [f"Neue Rückfrage #{nr}: {link(nr)}" for nr in rb.neu]
    zeilen += [f"Antwort übernommen: #{nr} {link(nr)}" for nr in rb.uebernommen]
    if rb.goldsatz_commit:
        zeilen.append(f"Goldsatz-Commit {rb.goldsatz_commit} (Rückfrage-Antworten, nicht gepusht)")
    zeilen += [f"Rückfrage erledigt: #{nr} {link(nr)}" for nr in rb.erledigt]
    zeilen += [f"Rückfrage ohne Antwort geschlossen, bleibt unklar: #{nr} {link(nr)}"
               for nr in rb.geschlossen_ohne_antwort]
    alt_doppelt = set(gemeldet.get("doppelt") or [])
    zeilen += [f"Rückfrage #{nr}: mehrere Antwort-Labels, bitte nur eins setzen {link(nr)}"
               for nr in rb.doppelt_beantwortet if nr not in alt_doppelt]
    alt_wartet = set(gemeldet.get("wartet") or [])
    zeilen += [f"Antwort wartet: {w}" for w in rb.wartet if kurz_hash(w) not in alt_wartet]
    doppelt, wartet = set(rb.doppelt_beantwortet), {kurz_hash(w) for w in rb.wartet}
    if rb.fehler:  # R17-5: abgebrochen (z. B. Bestand unlesbar): Bericht unvollstaendig, Gemeldetes nicht vergessen
        doppelt |= alt_doppelt
        wartet |= alt_wartet
    neu = {"doppelt": sorted(doppelt), "wartet": sorted(wartet)}
    kontext = [f"Rückfragen: {n}" for n in rb.notizen] if trocken else []
    return zeilen, kontext, neu


def erinnerungen(bestand: list[dict[str, Any]], heute: date) -> list[str]:
    """Offene Rueckfragen (Marke), die laenger als `ERINNERN_AB_TAGEN` offen sind (`created_at`)."""
    aus = []
    for issue in sorted(bestand, key=lambda i: nummer(i) or 0):
        nr = nummer(issue)
        roh = issue.get("created_at")
        if nr is None or issue.get("state") != "open" or not _MARKE.search(_body(issue)) or not isinstance(roh, str):
            continue
        try:
            angelegt = datetime.fromisoformat(roh.replace("Z", "+00:00")).astimezone().date()
        except ValueError:
            continue
        tage = (heute - angelegt).days
        if tage > ERINNERN_AB_TAGEN:
            url = issue.get("html_url") if isinstance(issue.get("html_url"), str) else ""
            aus.append(f"Erinnerung: Rückfrage #{nr} seit {tage} Tagen offen: {url}".rstrip())
    return aus
