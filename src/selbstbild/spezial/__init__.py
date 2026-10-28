"""Spezialextraktoren: Routen, MCP-Werkzeuge, HTTP-Clients (Python) und Compose-Dienste.

Alle Ergebnisse tragen den Platzhalter-Repo (R10); `repo_binden` setzt das echte Repo ein.
"""
from __future__ import annotations

import dataclasses

from selbstbild.spezial._gemeinsam import baum_lesen, definitionen, ist_testdatei
from selbstbild.spezial.compose import spezial_compose
from selbstbild.spezial.http_clients import http_aufrufe
from selbstbild.spezial.mcp_werkzeuge import mcp_werkzeuge
from selbstbild.spezial.routen import routen
from selbstbild.typen import PLATZHALTER_REPO, Beleg, Einstieg, Kante, Knoten

__all__ = ["spezial_python", "spezial_compose", "repo_binden"]

_LEER: tuple[tuple[Knoten, ...], tuple[Kante, ...], tuple[Einstieg, ...]] = ((), (), ())


def spezial_python(
    repo: str, pfad: str, quelle: bytes,
) -> tuple[tuple[Knoten, ...], tuple[Kante, ...], tuple[Einstieg, ...]]:
    """Routen, MCP-Werkzeuge und HTTP-Aufrufe einer Python-Datei. Testdateien liefern nichts."""
    if ist_testdatei(pfad):
        return _LEER
    baum = baum_lesen(quelle)
    if baum is None:
        return _LEER
    je_knoten, defs = definitionen(baum)
    try:
        rk, rka, re_ = routen(repo, pfad, baum, je_knoten, defs)
        mk, mka, me = mcp_werkzeuge(repo, pfad, baum, je_knoten, defs)
        hk, hka = http_aufrufe(repo, pfad, baum, defs)
    except RecursionError:
        return _LEER
    return rk + mk + hk, rka + mka + hka, re_ + me


def _id_binden(wert: str, repo: str) -> str:
    if wert == f"repo:{PLATZHALTER_REPO}":
        return f"repo:{repo}"
    art, trenner, rest = wert.partition(":")
    if trenner and rest.startswith(f"{PLATZHALTER_REPO}:"):
        return f"{art}:{repo}:{rest[len(PLATZHALTER_REPO) + 1:]}"
    return wert


def _beleg_binden(b: Beleg | None, repo: str) -> Beleg | None:
    if b is not None and b.repo == PLATZHALTER_REPO:
        return dataclasses.replace(b, repo=repo)
    return b


def repo_binden(obj: Knoten | Kante, repo: str) -> Knoten | Kante:
    """Ersetzt den Platzhalter-Repo in Ids (auch `von`/`nach`) und im Beleg durch `repo`."""
    if isinstance(obj, Knoten):
        return dataclasses.replace(obj, id=_id_binden(obj.id, repo), beleg=_beleg_binden(obj.beleg, repo))
    return dataclasses.replace(
        obj, von=_id_binden(obj.von, repo), nach=_id_binden(obj.nach, repo),
        beleg=_beleg_binden(obj.beleg, repo),
    )
