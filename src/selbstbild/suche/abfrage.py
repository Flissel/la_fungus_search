"""Suche ueber `suche.sqlite`: FTS5-Stichwort (BM25), Struktur-Erweiterung, optional dicht, RRF-Fusion.

`suche.sqlite` wird nur kurzlebig und schreibgeschuetzt (`mode=ro`) geoeffnet. Nutzereingaben gehen nie roh
in `MATCH`: jedes Token steht in doppelten Anfuehrungszeichen, die Tokens sind per OR verbunden. Die Ausgabe
ist deterministisch (Gleichstand nach id).
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .dicht import Embedder

from .index import index_pfad, zerlegen

RRF_K: int = 60
MAX_TOKENS: int = 12
STICHWORT_TOP: int = 50
STRUKTUR_QUELLEN: int = 10
STRUKTUR_PRO_QUELLE: int = 10
TOT_FAKTOR: float = 0.5
STRUKTUR_FAKTOR: float = 0.5

_STOPP_DE = frozenset("""
aber alle allem allen aller alles also auch auf aus bei bin bis bist dann das dass dein deine dem den der des
die dies diese diesem diesen dieser dieses doch dort durch ein eine einem einen einer eines euch euer für
gegen hab habe haben hat hast hatte hier ihr ihre ich ist jede jedem jeden jeder jedes jetzt kann kein keine
man mehr mein meine mit muss nach nicht noch nun nur oder ohne sein seine sich sie sind soll sollen sondern
über und uns unser unter vom von vor war waren was weil wenn wer werden wie wieder will wir wird wirst wo
wollen würde zum zur zwischen
""".split())
_STOPP_EN = frozenset("""
about after all also and any are because been before being both but can could did does doing for from had
has have her here him his how into its just more most not off only other our out over she should some such
than that the their them then there these they this those through very was were what when where which while
who whom why will with would you your
""".split())
_STOPP = _STOPP_DE | _STOPP_EN

_GRENZE = re.compile(r"(?<=[a-zäöüß0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-zäöüß])")
_NICHT_WORT = re.compile(r"[\W_]+")


@dataclass(frozen=True)
class Treffer:
    id: str
    art: str
    pfad: str | None
    zeile: int | None
    wert: float
    herkunft: tuple[str, ...]  # sortiert aus: stichwort, struktur, dicht
    lebendigkeit: float
    tot: str | None
    bereich: str | None
    warnung: str | None


def anfrage_tokens(text: str) -> tuple[str, ...]:
    """Woerter ab 3 Zeichen ohne Stoppwoerter (DE/EN), CamelCase/snake_case zerlegt, hoechstens 12, ohne Dopplungen."""
    aus: list[str] = []
    for wort in _NICHT_WORT.split(_GRENZE.sub(" ", text)):
        w = wort.lower()
        if len(w) >= 3 and w not in _STOPP and w not in aus:
            aus.append(w)
    return tuple(aus[:MAX_TOKENS])


def _fts_anfrage(tokens: tuple[str, ...]) -> str:
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def _offnen(stand: Path) -> sqlite3.Connection:
    ziel = index_pfad(stand)
    if not ziel.is_file():
        raise FileNotFoundError(f"kein Suchindex: {ziel}")
    return sqlite3.connect(f"{ziel.resolve().as_uri()}?mode=ro", uri=True)


def _stichwort(c: sqlite3.Connection, tokens: tuple[str, ...]) -> list[str]:
    zeilen = c.execute("SELECT id FROM volltext WHERE volltext MATCH ? ORDER BY bm25(volltext), id LIMIT ?",
                       (_fts_anfrage(tokens), STICHWORT_TOP)).fetchall()
    return [i for (i,) in zeilen]


def _struktur(c: sqlite3.Connection, stichwort: list[str]) -> dict[str, float]:
    """1-Hop-Nachbarn der besten Treffer (sichere Kanten zuerst) mit halbem Wert des Ausgangstreffers."""
    wert: dict[str, float] = {}
    for rang, quelle in enumerate(stichwort[:STRUKTUR_QUELLEN], start=1):
        halb = STRUKTUR_FAKTOR / (RRF_K + rang)
        zeilen = c.execute(
            "SELECT nach AS n, sicher FROM kante WHERE von=? UNION ALL SELECT von AS n, sicher FROM kante WHERE nach=?",
            (quelle, quelle)).fetchall()
        for n, _sicher in sorted(set(zeilen), key=lambda z: (-z[1], z[0]))[:STRUKTUR_PRO_QUELLE]:
            if n != quelle:
                wert[n] = max(wert.get(n, 0.0), halb)
    return wert


def _dicht(stand: Path, anfrage: str, embedder: "Embedder | None") -> list[str]:
    """Dichter Zweig: Anfrage einmal einbetten, Datei-Vektoren des Modells abfragen.

    Ohne Embedder, ohne Vektoren zu diesem Modell oder bei Embedder-/Lesefehler ist der Zweig leer: kein Fehler,
    keine Ausgabe. `dicht` wird erst hier importiert (numpy nur bei Bedarf).
    """
    if embedder is None:
        return []
    try:
        from . import dicht as _modul  # ImportError (z. B. numpy fehlt) = leerer Zweig
        vektor = _modul.einbetten(embedder, [anfrage], timeout=10.0)[0]
        return [i for i, _ in _modul.aehnlichste(stand, vektor, STICHWORT_TOP, embedder.modell)]
    except (ImportError, RuntimeError, OSError, sqlite3.DatabaseError):
        return []


def suchen(stand: Path, anfrage: str, top_k: int = 10, *, dicht: bool = False,
           embedder: "Embedder | None" = None, struktur: bool = True) -> tuple[Treffer, ...]:
    """Fusion aus Stichwort, Struktur und (bei `dicht=True` mit `embedder`) dichtem Zweig.

    Der dichte Zweig braucht Vektoren aus `dicht.vektoren_bauen` zum Modell des Embedders; fehlen sie oder
    antwortet der Embedder nicht, bleibt er leer und die Fusion ist unveraendert.
    `struktur=False` laesst die Struktur-Erweiterung weg (reine Stichwortsuche, fuer die Messung)."""
    tokens = anfrage_tokens(anfrage)
    if not tokens:
        return ()
    stand = Path(stand)
    with closing(_offnen(stand)) as c:
        stichwort = _stichwort(c, tokens)
        nachbar = _struktur(c, stichwort) if struktur else {}
        dichte = _dicht(stand, anfrage, embedder) if dicht else []
        beitrag: dict[str, dict[str, float]] = {}
        for rang, i in enumerate(stichwort, start=1):
            beitrag.setdefault(i, {})["stichwort"] = 1.0 / (RRF_K + rang)
        for i, w in nachbar.items():
            beitrag.setdefault(i, {})["struktur"] = w
        for rang, i in enumerate(dichte, start=1):
            beitrag.setdefault(i, {})["dicht"] = 1.0 / (RRF_K + rang)
        zeilen = {r[0]: r for r in c.execute(
            f"SELECT id, art, pfad, zeile, lebendigkeit, tot, bereich, warnung FROM eintrag"
            f" WHERE id IN ({','.join('?' * len(beitrag))})", tuple(beitrag))} if beitrag else {}
    treffer: list[Treffer] = []
    for i, zweige in beitrag.items():
        if i not in zeilen:
            continue
        _, art, pfad, zeile, lebendig, tot, bereich, warnung = zeilen[i]
        wert = sum(zweige.values()) * (TOT_FAKTOR if tot == "sicher_tot" else 1.0)
        treffer.append(Treffer(i, art, pfad, zeile, wert, tuple(sorted(zweige)),
                               float(lebendig) if lebendig is not None else 0.0, tot, bereich, warnung))
    treffer.sort(key=lambda t: (-t.wert, t.id))
    return tuple(treffer[:top_k])


def nachbarn(stand: Path, id: str, richtung: Literal["ein", "aus", "beide"] = "beide",
             arten: tuple[str, ...] = ()) -> tuple[tuple[str, str, bool], ...]:
    """Kanten um `id` als (von, nach, sicher), sortiert; `ein` = Kanten auf id, `aus` = Kanten von id."""
    bedingung = {"ein": "nach=?", "aus": "von=?", "beide": "(von=? OR nach=?)"}[richtung]
    args: list[str] = [id] * (2 if richtung == "beide" else 1)
    sql = f"SELECT DISTINCT von, nach, sicher FROM kante WHERE {bedingung}"
    if arten:
        sql += f" AND art IN ({','.join('?' * len(arten))})"
        args += list(arten)
    with closing(_offnen(Path(stand))) as c:
        zeilen = c.execute(sql, args).fetchall()
    return tuple(sorted((v, n, bool(s)) for v, n, s in zeilen))
