"""Bezeichner aus Doku-Seiten, gelesen nur am committeten Stand (ueber `quelle`, nie der Arbeitsbaum)."""
from __future__ import annotations

import re
from pathlib import Path

from .._glob import glob_regex
from ..quelle import Baum, BlobLeser, baum_lesen
from .lader import GraphDaten

_WORT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BACKTICK = re.compile(r"`([^`\n]+)`")
_CODEARTIG = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


_KETTE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(\()?")


def _codeartig(w: str) -> bool:
    return _CODEARTIG.fullmatch(w) is not None and ("_" in w or any(c.isupper() for c in w[1:]))


def bezeichner_aus_text(text: str) -> set[str]:
    """Woerter in Backticks (jede Laenge) plus code-artige Woerter (mit `_` oder Binnen-Grossbuchstaben)."""
    erg: set[str] = set()
    for m in _BACKTICK.finditer(text):
        erg.update(_WORT.findall(m.group(1)))
    erg.update(w for w in _WORT.findall(text) if _codeartig(w))
    for m in _KETTE.finditer(text):  # `Klasse.methode`, `modul.funktion(`: Segmente einer Punktkette
        teile = _WORT.findall(m.group(0))
        if m.group(1) or any(_codeartig(t) or t[:1].isupper() for t in teile):
            erg.update(teile)
    return erg


def _seiten_aus_graph(d: GraphDaten) -> set[tuple[str, str]]:
    erg: set[tuple[str, str]] = set()
    for i, k in d.knoten.items():
        if k.art == "doku_seite":
            _, repo, pfad = i.split(":", 2)
            erg.add((repo, pfad))
    return erg


def _sammeln(baum: Baum, seiten: set[tuple[str, str]] | None, erg: dict[str, set[str]]) -> None:
    muster = [glob_regex(m) for m in baum.regeln.doku_pfade]
    gewaehlt = [
        e for e in baum.eintraege
        if ((e.repo, e.pfad) in seiten if seiten is not None else any(rx.match(e.pfad) for rx in muster))
    ]
    if gewaehlt:
        ids = erg.setdefault(baum.repo, set())
        with BlobLeser(baum.wurzel) as leser:
            for e in gewaehlt:  # ein Blob nach dem anderen, behalten werden nur die Bezeichner
                ids |= bezeichner_aus_text(leser.lesen(e.blob).decode("utf-8", errors="replace"))
    for unter in baum.unterbaeume:
        _sammeln(unter, seiten, erg)


def doku_bezeichner(wurzel: Path, commit: str, d: GraphDaten | None = None) -> dict[str, frozenset[str]]:
    """Repo-Name (wie in den Ids) -> Bezeichner seiner Doku-Seiten am Commit.

    Mit `d` zaehlen genau die Dateien, die dort einen `doku_seite`-Knoten haben; ohne `d` die Dateien,
    die auf `Regeln.doku_pfade` des jeweiligen Repos passen."""
    baum = baum_lesen(Path(wurzel), commit)
    erg: dict[str, set[str]] = {}
    _sammeln(baum, None if d is None else _seiten_aus_graph(d), erg)
    return {repo: frozenset(ids) for repo, ids in sorted(erg.items())}
