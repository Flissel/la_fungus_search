"""Aufträge des Debugger-Claw aus GitHub-Kommentaren (rein, ohne Netz)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.stand import DebuggerStand
from selbstbild.funde.github import AUTO_MARKE, DEBUGGER_MARKE

_FRUEHESTE = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Kommentar:
    """Ein Kommentar an einem Issue oder PR; `pfad`/`zeile` nur bei Zeilenkommentaren."""

    id: int
    repo: str
    nummer: int
    autor: str
    text: str
    erstellt: str
    review: bool
    pfad: str | None = None
    zeile: int | None = None


@dataclass(frozen=True)
class Auftrag:
    """Alle neuen Kommentare zu einem Fund-Issue, nach `erstellt` sortiert."""

    issue: int
    kommentare: tuple[Kommentar, ...]


def _zeit(text: str) -> datetime:
    """ISO-8601 (auch mit `Z`) als UTC-Zeit; Unlesbares gilt als frühestmöglich."""
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _FRUEHESTE
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def kommentar_aus_api(repo: str, roh: dict[str, Any], *, review: bool) -> Kommentar:
    """Baut einen `Kommentar` aus der GitHub-Antwort; `nummer` aus `issue_url` bzw. `pull_request_url`."""
    url = str(roh.get("issue_url") or roh.get("pull_request_url") or "")
    nummer = int(url.rstrip("/").rsplit("/", 1)[-1])
    zeile = roh.get("line")
    if zeile is None:
        zeile = roh.get("original_line")
    user = roh.get("user") or {}
    return Kommentar(
        id=int(roh["id"]),
        repo=repo,
        nummer=nummer,
        autor=str(user.get("login", "")),
        text=roh.get("body") or "",
        erstellt=str(roh.get("created_at", "")),
        review=review,
        pfad=roh.get("path"),
        zeile=int(zeile) if zeile is not None else None,
    )


def auftraege_bilden(
    kommentare: Sequence[Kommentar],
    stand: DebuggerStand,
    konfig: DebuggerKonfig,
    *,
    selbstbild_issues: set[int],
    pr_zu_issue: dict[tuple[str, int], int],
    laufend: set[int],
) -> list[Auftrag]:
    """Filtert (Auslöser, Marken, verarbeitet, nach `start`, bekanntes Repo), ordnet Issues zu, gruppiert."""
    start = _zeit(stand.start) if stand.start else None
    verarbeitet = set(stand.verarbeitet)
    gesehen: set[int] = set()
    je_issue: dict[int, list[Kommentar]] = {}
    for k in kommentare:
        if k.autor != konfig.ausloeser:
            continue
        if AUTO_MARKE in k.text or DEBUGGER_MARKE in k.text:
            continue
        if k.id in verarbeitet or k.id in gesehen:
            continue
        if start is not None and _zeit(k.erstellt) <= start:
            continue
        if k.repo not in konfig.repos:
            continue
        issue = pr_zu_issue.get((k.repo, k.nummer))
        if issue is None and k.repo == konfig.issue_repo and k.nummer in selbstbild_issues:
            issue = k.nummer
        if issue is None:
            continue
        gesehen.add(k.id)
        if issue in laufend:
            continue
        je_issue.setdefault(issue, []).append(k)
    auftraege = [
        Auftrag(issue=i, kommentare=tuple(sorted(ks, key=lambda c: (_zeit(c.erstellt), c.id))))
        for i, ks in je_issue.items()
    ]
    auftraege.sort(key=lambda a: (_zeit(a.kommentare[0].erstellt), a.issue))
    return auftraege
