"""Versuche, Pausen und Einmal-Schlüssel des Debugger-Claw (rein, ohne Netz).

Alles hier verändert nur den Stand im Speicher: gezählte Versuche mit Pause (R42), die
Reaktion nur einmal je Kommentar (Punkt 2) und der Tageszähler für ungezählte GitHub-Fehler
einer Runde (Punkt 1). Schlüssel in `stand.gemeldet` werden mit den anderen Meldungen gekappt.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Sequence

from selbstbild.debugger.auftraege import Kommentar, _zeit
from selbstbild.debugger.stand import DebuggerStand, FundStand

MAX_VERSUCHE = 3
PAUSE = timedelta(minutes=10)  # R42: je gezähltem Versuch
MAX_GH_FEHLER = 3  # Punkt 1: so oft in Folge, dann ist der Kandidat für den Tag gesperrt
REAGIERT = "reagiert:"
GH_FEHLER = "gh_fehler:"


def _tag(jetzt: datetime) -> str:
    return jetzt.astimezone(timezone.utc).date().isoformat()


# ---------- R42: gezählte Versuche und Pause ----------

def versuch_zaehlen(fund: FundStand, ids: Sequence[int], jetzt: datetime) -> list[int]:
    """Zählt je Kommentar einen Versuch; gibt die aufgegebenen Ids zurück (ab `MAX_VERSUCHE`).

    Für die übrigen setzt es `fund.naechster_versuch` = jetzt + 10 min × höchster Versuch, sonst None.
    """
    for i in ids:
        fund.versuche[str(i)] = fund.versuche.get(str(i), 0) + 1
    aufgegeben = [i for i in ids if fund.versuche[str(i)] >= MAX_VERSUCHE]
    for i in aufgegeben:
        fund.versuche.pop(str(i), None)
    rest = [fund.versuche[str(i)] for i in ids if str(i) in fund.versuche]
    fund.naechster_versuch = (jetzt + PAUSE * max(rest)).isoformat() if rest else None
    return aufgegeben


def pausiert(fund: FundStand | None, jetzt: datetime) -> bool:
    """True, solange `jetzt < naechster_versuch`; ein unlesbarer Wert hält nichts zurück."""
    if fund is None or not fund.naechster_versuch:
        return False
    return jetzt < _zeit(fund.naechster_versuch)


# ---------- Punkt 2: Reaktion nur einmal je Kommentar ----------

def einmal_reagieren(stand: DebuggerStand, kommentare: Iterable[Kommentar],
                     reagieren: Callable[[Kommentar], None]) -> None:
    """Ruft `reagieren` nur für Kommentare ohne `reagiert:<id>` und merkt sich die Id danach."""
    for k in kommentare:
        schluessel = f"{REAGIERT}{k.id}"
        if schluessel in stand.gemeldet:
            continue
        reagieren(k)
        stand.gemeldet.append(schluessel)


def reagiert_aufraeumen(stand: DebuggerStand) -> None:
    """Entfernt `reagiert:<id>` verarbeiteter Kommentare (sie kommen nie wieder)."""
    fertig = {f"{REAGIERT}{i}" for i in stand.verarbeitet}
    stand.gemeldet = [g for g in stand.gemeldet if g not in fertig]


# ---------- Punkt 1: ungezählte GitHub-Fehler einer Runde ----------

def gh_fehler_merken(stand: DebuggerStand, issue: int, jetzt: datetime) -> None:
    """Ein Eintrag `gh_fehler:<issue>:<tag>` je Fehler; Einträge anderer Tage fallen weg."""
    tag = _tag(jetzt)
    stand.gemeldet = [g for g in stand.gemeldet if not g.startswith(GH_FEHLER) or g.endswith(f":{tag}")]
    stand.gemeldet.append(f"{GH_FEHLER}{issue}:{tag}")


def gh_fehler_vergessen(stand: DebuggerStand, issue: int) -> None:
    """Nach einer Runde ohne GitHub-Fehler beginnt die Zählung „in Folge“ neu."""
    stand.gemeldet = [g for g in stand.gemeldet if not g.startswith(f"{GH_FEHLER}{issue}:")]


def gh_gesperrt(stand: DebuggerStand, issue: int, jetzt: datetime) -> bool:
    """True ab `MAX_GH_FEHLER` Einträgen für `issue` am heutigen Tag."""
    return stand.gemeldet.count(f"{GH_FEHLER}{issue}:{_tag(jetzt)}") >= MAX_GH_FEHLER
