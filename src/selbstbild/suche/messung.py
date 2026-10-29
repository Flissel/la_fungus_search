"""Messung der Suche (Stufe 3, T5): recall@10 und MRR gegen Gold aus KETTEN.md und aus Transkripten.

Gold ist eine Folge von `(anfrage, erwartete_datei_ids)`. Prompt-Texte aus Transkripten leben nur im
Speicher: dieses Modul schreibt sie nie in Ausgaben, Logs oder Meldungen von Ausnahmen.

Datei-Ids heissen `datei:<repo>:<pfad>`. Pfade aus KETTEN.md und Transkripten sind relativ zum aeusseren
Repo und werden per laengstem Praefix auf das Repo (Wurzel oder Submodul) abgebildet.
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Mapping, Sequence

from ..analyse.lader import SEED
from .abfrage import suchen
from .dicht import Embedder, aehnlichste, einbetten, vektoren_bauen
from .index import index_pfad

TOP: int = 10
MIN_PROMPT: int = 20
_SCHREIBER = frozenset({"Write", "Edit", "MultiEdit"})
_HOP_BLOCK = re.compile(r"<!--\s*graph:begin hops[^>]*-->(.*?)<!--\s*graph:end", re.DOTALL)
_ZELLE = re.compile(r"(?<!\\)\|")
_BACKTICK = re.compile(r"`([^`]+)`")
_PFAD_ZEILE = re.compile(r"^([^\s:#`]*/[^\s:#`]*?)(?::\d+(?:-\d+)?)?(?:#\S+)?$")

Gold = Sequence[tuple[str, frozenset[str]]]


# --- Abbildung Pfad -> Datei-Id --------------------------------------------

def _praefix(repo: str, wurzel_repo: str) -> str:
    return "" if repo == wurzel_repo else repo.removeprefix(wurzel_repo + "/")


def datei_id_von_pfad(pfad: str, wurzel_repo: str, repos: Sequence[str] = ()) -> str:
    """Pfad relativ zum aeusseren Repo -> `datei:<repo>:<rest>` (laengstes Praefix der bekannten Repos gewinnt)."""
    pfad = pfad.replace("\\", "/").lstrip("/")
    bester = (wurzel_repo, "")
    for repo in sorted(set(repos) | {wurzel_repo}):
        p = _praefix(repo, wurzel_repo)
        if p and pfad.startswith(p + "/") and len(p) > len(bester[1]):
            bester = (repo, p)
    repo, p = bester
    return f"datei:{repo}:{pfad[len(p) + 1:] if p else pfad}"


# --- Gold aus KETTEN -------------------------------------------------------

def gold_aus_ketten(ketten_md: str, wurzel_repo: str, *, repos: Sequence[str] = ()) -> tuple[tuple[str, frozenset[str]], ...]:
    """Je Hop-Zeile ein Paar: Anfrage = Von→Nach + Mechanismus, Ziele = Pfade der Spalte Code-Stelle.

    Zeilen ohne Pfad (nur Symbolnamen) entfallen."""
    aus: list[tuple[str, frozenset[str]]] = []
    for block in _HOP_BLOCK.findall(ketten_md):
        for zeile in block.splitlines():
            zeile = zeile.strip()
            if not zeile.startswith("|"):
                continue
            zellen = [z.strip() for z in _ZELLE.split(zeile.strip("|"))]
            if len(zellen) < 4 or not zellen[0][:1].isdigit():
                continue
            ziele = set()
            for roh in _BACKTICK.findall(zellen[3]):
                m = _PFAD_ZEILE.match(roh.strip())
                if m:
                    ziele.add(datei_id_von_pfad(m.group(1), wurzel_repo, repos))
            if ziele:
                anfrage = f"{zellen[1]} {zellen[2]}".replace("`", "")
                aus.append((anfrage, frozenset(ziele)))
    return tuple(aus)


# --- Gold aus Transkripten -------------------------------------------------

def _prompt_text(nachricht: object) -> str | None:
    """Text eines echten User-Prompts; `None` bei tool_result-Nachrichten und Nicht-Text."""
    if not isinstance(nachricht, dict):
        return None
    inhalt = nachricht.get("content")
    if isinstance(inhalt, str):
        return inhalt
    if isinstance(inhalt, list):
        texte = [x.get("text") for x in inhalt if isinstance(x, dict) and x.get("type") == "text"]
        texte = [t for t in texte if isinstance(t, str)]
        return "\n".join(texte) if texte else None
    return None


def _relativ(pfad: str, wurzel: Path) -> str | None:
    if not os.path.isabs(pfad):
        return None
    ziel = os.path.normcase(os.path.normpath(pfad))
    basis = os.path.normcase(os.path.normpath(str(wurzel)))
    try:
        rel = os.path.relpath(ziel, basis)
    except ValueError:  # anderes Laufwerk
        return None
    if rel == "." or rel.startswith(".."):
        return None
    # normcase kann Gross-/Kleinschreibung veraendern: Originalschreibweise aus dem Pfad zurueckholen
    teile = os.path.normpath(pfad).replace("\\", "/").split("/")
    n = len(rel.split(os.sep))
    return "/".join(teile[-n:])


def _kandidaten(datei: Path, wurzel: Path, wurzel_repo: str, repos: Sequence[str]) -> list[tuple[str, frozenset[str]]]:
    paare: list[tuple[str, set[str]]] = []
    try:
        zeilen = datei.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for zeile in zeilen:
        try:
            eintrag = json.loads(zeile)
        except ValueError:
            continue
        if not isinstance(eintrag, dict):
            continue
        nachricht = eintrag.get("message")
        if eintrag.get("type") == "user" and not eintrag.get("isMeta"):
            text = _prompt_text(nachricht)
            if text is not None and len(text) >= MIN_PROMPT:  # kurze Zurufe ("ja") beenden keine Aufgabe
                paare.append((text, set()))
        elif eintrag.get("type") == "assistant" and paare and isinstance(nachricht, dict):
            inhalt = nachricht.get("content")
            for x in inhalt if isinstance(inhalt, list) else []:
                if not (isinstance(x, dict) and x.get("type") == "tool_use" and x.get("name") in _SCHREIBER):
                    continue
                eingabe = x.get("input")
                pfad = eingabe.get("file_path") if isinstance(eingabe, dict) else None
                rel = _relativ(pfad, wurzel) if isinstance(pfad, str) else None
                if rel is not None:
                    paare[-1][1].add(datei_id_von_pfad(rel, wurzel_repo, repos))
    return [(t, frozenset(z)) for t, z in paare if z]


def gold_aus_transkripten(ordner: Path, repo_wurzel: Path, n: int = 20, seed: int = SEED, *,
                          wurzel_repo: str | None = None,
                          repos: Sequence[str] = ()) -> tuple[tuple[str, frozenset[str]], ...]:
    """Je User-Prompt (ab 20 Zeichen) die danach bis zum naechsten Prompt geschriebenen Dateien unter `repo_wurzel`.

    Nur lesend; Prompt-Texte stehen ausschliesslich im Rueckgabewert (Speicher). Stichprobe von `n` aus der
    nach (Datei, Position) sortierten Kandidatenliste mit `random.Random(seed)`."""
    wurzel_repo = wurzel_repo or Path(repo_wurzel).name
    alle: list[tuple[str, frozenset[str]]] = []
    for datei in sorted(Path(ordner).glob("*.jsonl"), key=lambda p: p.name):
        alle.extend(_kandidaten(datei, Path(repo_wurzel), wurzel_repo, repos))
    if len(alle) <= n:
        return tuple(alle)
    ausgewaehlt = sorted(random.Random(seed).sample(range(len(alle)), n))
    return tuple(alle[i] for i in ausgewaehlt)


# --- Messen ----------------------------------------------------------------

def _als_datei_id(i: str) -> str | None:
    """Trefferid -> Datei-Id der zugehoerigen Datei; `None` ohne Dateibezug (env, dienst, ...)."""
    teile = i.split(":", 2)
    if len(teile) != 3 or teile[0] not in ("datei", "doku_seite", "symbol"):
        return None
    pfad = teile[2].rpartition("#")[0] if teile[0] == "symbol" else teile[2]
    return f"datei:{teile[1]}:{pfad}"


def _variante(v: str) -> tuple[str, str | None]:
    art, _, modell = v.partition(":")
    if art in ("stichwort", "struktur") and not modell:
        return art, None
    if art in ("dicht", "fusion") and modell:
        return art, modell
    raise ValueError(f"unbekannte Variante: {v}")


def messen(stand: Path, gold: Gold, *, varianten: Sequence[str],
           embedder: Mapping[str, Embedder] | None = None) -> dict[str, dict[str, float]]:
    """`recall@10` (mind. ein erwartetes Ziel unter den Top-10-Treffern) und `mrr` (Rang des ersten Treffers).

    Varianten: `stichwort`, `struktur`, `dicht:<modell>`, `fusion:<modell>`. Vektoren gibt es nur zu einem
    Modell zugleich; sie werden beim Wechsel des Modells neu gebaut."""
    stand = Path(stand)
    embedder = embedder or {}
    plan = [(v, *_variante(v)) for v in varianten]
    for v, art, modell in plan:
        if modell is not None and modell not in embedder:
            raise ValueError(f"kein Embedder fuer Modell: {modell}")
    gebaut: str | None = None
    ergebnis: dict[str, dict[str, float]] = {}
    for name, art, modell in plan:
        e = embedder[modell] if modell is not None else None
        if e is not None and gebaut != modell:
            vektoren_bauen(stand, e)
            gebaut = modell
        treffer_summe = 0
        rr_summe = 0.0
        for anfrage, ziele in gold:
            if art == "dicht" and e is not None:
                vektor = einbetten(e, [anfrage], timeout=30.0)[0]
                ids = [i for i, _ in aehnlichste(stand, vektor, TOP, e.modell)]
            else:
                ids = [t.id for t in suchen(stand, anfrage, TOP, struktur=art in ("struktur", "fusion"),
                                            dicht=art == "fusion", embedder=e)]
            for rang, i in enumerate(ids[:TOP], start=1):
                if _als_datei_id(i) in ziele:
                    treffer_summe += 1
                    rr_summe += 1.0 / rang
                    break
        n = len(gold)
        ergebnis[name] = {"recall@10": treffer_summe / n if n else 0.0, "mrr": rr_summe / n if n else 0.0}
    return ergebnis


def repos_im_index(stand: Path) -> tuple[str, ...]:
    """Repo-Namen aus `suche.sqlite` (nur lesend), sortiert."""
    ziel = index_pfad(Path(stand))
    if not ziel.is_file():
        raise FileNotFoundError(f"kein Suchindex: {ziel}")
    with closing(sqlite3.connect(f"{ziel.resolve().as_uri()}?mode=ro", uri=True)) as c:
        return tuple(sorted(r for (r,) in c.execute("SELECT DISTINCT repo FROM eintrag WHERE repo IS NOT NULL")))
