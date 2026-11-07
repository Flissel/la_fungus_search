"""Material für den Helfer-Prompt: Issue, Kommentare, PR-Diff und Review-Kommentare (nur lesend).

Ausgelagert aus runde.py. Kommentare anderer Konten als Auslöser und Helfer-Konto erscheinen
nur als Platzhalter (R22); der Diff wird auf `MAX_DIFF` Zeichen gekürzt.
"""
from __future__ import annotations

from typing import Any

from selbstbild.debugger.auftraege import Kommentar
from selbstbild.debugger.konfig import DebuggerKonfig
from selbstbild.debugger.stand import FundStand
from selbstbild.funde.github import GitHub

MAX_SEITEN = 10
SEITE = 100
MAX_DIFF = 60_000


def _liste(daten: Any) -> list[dict[str, Any]]:
    return [x for x in daten if isinstance(x, dict)] if isinstance(daten, list) else []


def _autor(roh: dict[str, Any]) -> str:
    user = roh.get("user")
    return str(user.get("login", "")) if isinstance(user, dict) else ""


def _seiten(gh: GitHub, pfad: str) -> list[dict[str, Any]]:
    """Liest `pfad` seitenweise (100 je Seite) bis zur ersten kurzen Seite, höchstens 10 Seiten."""
    alle: list[dict[str, Any]] = []
    for seite in range(1, MAX_SEITEN + 1):
        teil = gh._api("GET", f"{pfad}?per_page={SEITE}&page={seite}")
        alle += _liste(teil)
        if not isinstance(teil, list) or len(teil) < SEITE:
            break
    return alle


def auftrag_texte(kommentare: tuple[Kommentar, ...]) -> list[str]:
    """Die Texte des Auftrags; Zeilenkommentare mit `[pfad:zeile]` davor."""
    texte: list[str] = []
    for k in kommentare:
        ort = (f"{k.pfad}:{k.zeile}" if k.zeile is not None else k.pfad) if k.pfad else None
        texte.append(f"[{ort}] {k.text}" if ort else k.text)
    return texte


def _kommentar(konfig: DebuggerKonfig, c: dict[str, Any], ort: str = "") -> str:
    """Kommentar fürs Material; nur Auslöser und Helfer-Konto, alle anderen als Platzhalter (R22)."""
    autor = _autor(c)
    if autor not in (konfig.ausloeser, konfig.helfer_konto):
        return f"[Kommentar von {autor or '?'} ausgelassen]"
    return f"--- {autor} ({c.get('created_at', '')}){ort}\n{c.get('body') or ''}"


def material_bauen(lesen: GitHub, konfig: DebuggerKonfig, n: int, titel: str, text: str, fund: FundStand) -> str:
    """Issue mit Kommentaren, bei einem PR dazu Diff und Review-Kommentare."""
    basis = f"/repos/{konfig.issue_repo}/issues/{n}"
    teile = [f"Issue #{n}: {titel}", text, "", "Kommentare des Issues:"]
    teile += [_kommentar(konfig, c) for c in _seiten(lesen, f"{basis}/comments")]
    if fund.pr is not None:
        pr = f"/repos/{fund.repo}/pulls/{fund.pr}"
        diff = "\n".join(
            f"### {d.get('filename', '')}\n{d.get('patch') or ''}" for d in _seiten(lesen, f"{pr}/files")
        )
        if len(diff) > MAX_DIFF:
            diff = diff[:MAX_DIFF] + "\n[… Diff gekürzt]"
        teile += ["", f"Diff des PR {fund.repo}#{fund.pr}:", diff, "", "Review-Kommentare des PR:"]
        for c in _seiten(lesen, f"{pr}/comments"):
            ort = f" [{c.get('path', '')}:{c.get('line') or c.get('original_line') or ''}]"
            teile.append(_kommentar(konfig, c, ort))
    return "\n".join(teile)
