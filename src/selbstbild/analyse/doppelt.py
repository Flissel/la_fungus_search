"""Doppelungen, rein textuell und strukturell (E1, keine Embeddings).

Drei Arten: gleiche Blob-SHA (`gleiche_datei`), gleicher Symbol-Digest (`gleiches_symbol`)
und MinHash-Aehnlichkeit ueber 5-Token-Shingles (`aehnliche_datei`).

Zeilenregel fuer `gleiches_symbol` (Ruling S6): gemessen, nicht geschaetzt. Nur Digests, die
bei mindestens 2 Symbolen vorkommen, sind Kandidaten; deren Dateien werden am Commit gelesen
und geparst, die Laenge ist `zeile_bis - zeile_von + 1` des Symbols mit passendem Digest.

Normalisierung der Shingles: sprachunabhaengiger Regex-Tokenizer; Kommentare entfallen,
Bezeichner werden zu `ID`, String- und Zahlenliterale zu `LIT`; Schluesselwoerter und
Satzzeichen bleiben. Datenformate ohne Code-Struktur (json, dotenv) bleiben aussen vor.

`aehnlichkeit` bei `aehnliche_datei` ist der exakte Jaccard-Wert der Shingle-Mengen
(Kandidaten werden aus den gelesenen Blobs nachgerechnet). Bestaetigte Paare werden per Union-Find
zu Clustern; die Cluster-aehnlichkeit ist der kleinste Jaccard-Wert entlang der Baumkanten, auf 4
Stellen gerundet. LSH-Eimer mit mehr als MAX_EIMER Blobs (Boilerplate) werden uebersprungen und
in STATISTIK["eimer_uebersprungen"] gezaehlt.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..parser.basis import parsen
from ..quelle import Baum, BlobLeser, Eintrag, baum_lesen
from ..typen import id_datei
from .lader import SEED, GraphDaten

MIN_BYTES = 200
MIN_ZEILEN_SYMBOL = 5
MIN_ZEILEN_DATEI = 30
SHINGLE = 5
PERMUTATIONEN = 128
BAENDER = 32
ZEILEN_JE_BAND = 4
_PRIM = 2147483629  # Primzahl < 2^31
MAX_EIMER = 50  # groessere LSH-Eimer (Boilerplate) werden uebersprungen und gezaehlt
_CHUNK = 4096
STATISTIK: dict[str, int] = {"eimer_uebersprungen": 0}  # Kennzahlen des letzten Laufs
_VORFILTER = 0.1  # Kandidaten brauchen Signatur-Gleichheit >= schwelle - _VORFILTER
_OHNE_SHINGLES = frozenset({"json", "dotenv"})

Art = Literal["gleiche_datei", "gleiches_symbol", "aehnliche_datei"]


@dataclass(frozen=True)
class Doppelung:
    art: Art
    mitglieder: tuple[str, ...]
    aehnlichkeit: float


_SCHLUESSELWOERTER = frozenset("""
as assert async await break case catch class const continue def default defer del do elif else
enum except export extends false fn finally for from func function go if impl import in interface
is lambda let loop match mod mut new nil none not null of or and package pass pub raise range
return select self static struct super switch this throw true try type typeof use var where while
with yield
""".split())

_KOMMENTAR_HASH = r"#[^\n]*"
_KOMMENTAR_C = r"//[^\n]*|/\*.*?(?:\*/|\Z)"
# Unbeendete Literale/Kommentare laufen genau einmal bis Dateiende (`\Z`): kein quadratisches Suchen.
_TOKEN = (r"(?P<k>{kommentar})|(?P<s>\"\"\"[\s\S]*?(?:\"\"\"|\Z)|'''[\s\S]*?(?:'''|\Z)"
          r"|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`[^`]*(?:`|\Z))|(?P<n>\d[\w.]*)|(?P<i>[A-Za-z_]\w*)|(?P<p>\S)")
_HASH_SPRACHEN = frozenset({"python", "bash", "powershell", "yaml"})
_REGEX = {
    True: re.compile(_TOKEN.format(kommentar=_KOMMENTAR_HASH), re.DOTALL),
    False: re.compile(_TOKEN.format(kommentar=_KOMMENTAR_C), re.DOTALL),
}


def _tokens(text: str, sprache: str) -> list[str]:
    erg: list[str] = []
    for m in _REGEX[sprache in _HASH_SPRACHEN].finditer(text):
        g = m.lastgroup
        if g == "k":
            continue
        if g in ("s", "n"):
            erg.append("LIT")
        elif g == "i":
            w = m.group()
            erg.append(w if w.lower() in _SCHLUESSELWOERTER else "ID")
        else:
            erg.append(m.group())
    return erg


def _shingles(text: str, sprache: str) -> set[int]:
    t = _tokens(text, sprache)
    return {int.from_bytes(hashlib.blake2b(" ".join(t[i:i + SHINGLE]).encode(), digest_size=8).digest(), "big") % _PRIM
            for i in range(max(len(t) - SHINGLE + 1, 0))}


def _permutationen() -> tuple[np.ndarray, np.ndarray]:
    rnd = random.Random(SEED)  # eigene Instanz: kein globaler Zufallszustand
    a = np.array([rnd.randrange(1, _PRIM) for _ in range(PERMUTATIONEN)], dtype=np.uint64)
    b = np.array([rnd.randrange(0, _PRIM) for _ in range(PERMUTATIONEN)], dtype=np.uint64)
    return a, b


def _signatur(hashes: set[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spaltenweises Minimum von (a*x+b) mod p; x, a, b < p < 2^31: kein uint64-Ueberlauf."""
    x = np.fromiter(sorted(hashes), dtype=np.uint64, count=len(hashes))
    erg = np.full(PERMUTATIONEN, _PRIM, dtype=np.uint64)
    for i in range(0, len(x), _CHUNK):
        teil = x[i:i + _CHUNK]
        erg = np.minimum(erg, ((a[:, None] * teil[None, :] + b[:, None]) % np.uint64(_PRIM)).min(axis=1))
    return erg


def _jaccard(a: set[int], b: set[int]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _baeume(b: Baum) -> Iterator[Baum]:
    yield b
    for u in b.unterbaeume:
        yield from _baeume(u)


def _text(leser: BlobLeser, blob: str) -> str:
    return leser.lesen(blob).decode("utf-8", errors="replace")


def _gleiche_dateien(eintraege: list[Eintrag]) -> list[Doppelung]:
    je_blob: dict[str, set[str]] = {}
    for e in eintraege:
        if e.groesse >= MIN_BYTES:
            je_blob.setdefault(e.blob, set()).add(id_datei(e.repo, e.pfad))
    return [Doppelung("gleiche_datei", tuple(sorted(m)), 1.0) for m in je_blob.values() if len(m) >= 2]


def _gleiche_symbole(d: GraphDaten, blob_von: dict[tuple[str, str], tuple[Eintrag, Baum]],
                     leser: dict[str, BlobLeser]) -> list[Doppelung]:
    """Mindestens 2 verschiedene Symbol-Ids mit gleichem Digest und je >= 5 Zeilen. Ausgenommen
    sind nur Gruppen, die aus einem einzigen (Blob, Qualname) bestehen: Das sind Kopien derselben
    Datei und damit `gleiche_datei`. Zwei gleiche Funktionen in einer Datei zaehlen."""
    je_digest: dict[str, list[str]] = {}
    for sid in sorted(d.knoten):
        k = d.knoten[sid]
        if k.art == "symbol" and k.beleg is not None and k.beleg.digest:
            je_digest.setdefault(k.beleg.digest, []).append(sid)
    erg: list[Doppelung] = []
    # Datei -> (qualname, digest) -> Zeilen der ersten Definition; nur fuer Kandidaten-Dateien
    laengen: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for digest, ids in sorted(je_digest.items()):
        if len(ids) < 2:
            continue
        lang: list[str] = []
        herkunft: set[tuple[str, str]] = set()
        for sid in ids:
            k = d.knoten[sid]
            beleg = k.beleg
            if beleg is None:
                continue
            schluessel = (beleg.repo, beleg.pfad)
            ziel = blob_von.get(schluessel)
            if ziel is None:
                continue
            eintrag, baum = ziel
            if schluessel not in laengen:
                p = parsen(eintrag.sprache or "", eintrag.pfad, leser[baum.repo].lesen(eintrag.blob))
                je: dict[tuple[str, str], int] = {}
                for s in p.symbole:
                    je.setdefault((s.qualname, s.digest), s.zeile_bis - s.zeile_von + 1)
                laengen[schluessel] = je
            if laengen[schluessel].get((k.name, digest), 0) >= MIN_ZEILEN_SYMBOL:
                lang.append(sid)
                herkunft.add((eintrag.blob, k.name))
        if len(lang) >= 2 and len(herkunft) >= 2:
            erg.append(Doppelung("gleiches_symbol", tuple(sorted(lang)), 1.0))
    return erg


def _aehnliche_dateien(eintraege: list[Eintrag], leser: dict[str, BlobLeser],
                       schwelle: float) -> list[Doppelung]:
    """Cluster (Zusammenhangskomponenten) bestaetigter Paare. `aehnlichkeit` eines Clusters ist
    der kleinste Jaccard-Wert entlang der Union-Find-Baumkanten (Kanten, die zwei Teile
    verbinden), also eine untere Schranke fuer den Zusammenhang."""
    STATISTIK["eimer_uebersprungen"] = 0
    a_perm, b_perm = _permutationen()
    ids_je_blob: dict[str, set[str]] = {}
    erster: dict[str, Eintrag] = {}
    for e in eintraege:
        ids_je_blob.setdefault(e.blob, set()).add(id_datei(e.repo, e.pfad))
        erster.setdefault(e.blob, e)
    sig: dict[str, np.ndarray] = {}
    for blob in sorted(erster):
        e = erster[blob]
        if e.sprache is None or e.sprache in _OHNE_SHINGLES or e.groesse < MIN_ZEILEN_DATEI:
            continue
        text = _text(leser[e.repo], blob)
        if text.count("\n") + 1 < MIN_ZEILEN_DATEI:
            continue
        h = _shingles(text, e.sprache)
        if h:
            sig[blob] = _signatur(h, a_perm, b_perm)  # nur die Signatur bleibt im Speicher
    baender: dict[tuple[int, bytes], list[str]] = {}
    for blob, s in sig.items():
        for b in range(BAENDER):
            baender.setdefault((b, s[b * ZEILEN_JE_BAND:(b + 1) * ZEILEN_JE_BAND].tobytes()), []).append(blob)
    kandidaten: set[tuple[str, str]] = set()
    for gruppe in baender.values():
        if len(gruppe) > MAX_EIMER:
            STATISTIK["eimer_uebersprungen"] += 1
            continue
        gruppe.sort()
        for i, a in enumerate(gruppe):
            for b in gruppe[i + 1:]:
                kandidaten.add((a, b))
    cache: dict[str, set[int]] = {}  # je Lauf: nur Kandidaten-Blobs, jeder hoechstens einmal gelesen

    def menge(blob: str) -> set[int]:
        if blob not in cache:
            e = erster[blob]
            cache[blob] = _shingles(_text(leser[e.repo], blob), e.sprache or "")
        return cache[blob]

    eltern: dict[str, str] = {}

    def wurzel_von(x: str) -> str:
        while eltern.setdefault(x, x) != x:
            eltern[x] = eltern[eltern[x]]
            x = eltern[x]
        return x

    kanten_min: dict[str, float] = {}  # Cluster-Wurzel -> kleinster Jaccard der Baumkanten
    for a, b in sorted(kandidaten):
        if erster[a].sprache != erster[b].sprache:
            continue
        if np.count_nonzero(sig[a] == sig[b]) / PERMUTATIONEN < schwelle - _VORFILTER:
            continue
        j = _jaccard(menge(a), menge(b))
        if j < schwelle:
            continue
        ra, rb = wurzel_von(a), wurzel_von(b)
        if ra == rb:
            continue  # keine Baumkante
        eltern[rb] = ra
        kanten_min[ra] = min(j, kanten_min.pop(ra, 1.0), kanten_min.pop(rb, 1.0))
    gruppen: dict[str, set[str]] = {}
    for blob in sorted(eltern):
        gruppen.setdefault(wurzel_von(blob), set()).add(blob)
    erg: list[Doppelung] = []
    for w, blobs in gruppen.items():
        ids: set[str] = set()
        for blob in blobs:
            ids |= ids_je_blob[blob]
        erg.append(Doppelung("aehnliche_datei", tuple(sorted(ids)), round(kanten_min.get(w, 1.0), 4)))
    return erg


def doppelungen(wurzel: Path, d: GraphDaten, schwelle: float = 0.9) -> tuple[Doppelung, ...]:
    """Doppelungen am Commit `d.commit`; Blobs nur ueber `quelle`, sortiert nach (art, mitglieder).
    `STATISTIK["eimer_uebersprungen"]` zaehlt die uebergrossen LSH-Eimer des letzten Laufs."""
    baum = baum_lesen(Path(wurzel), d.commit)
    eintraege: list[Eintrag] = []
    baum_je_repo: dict[str, Baum] = {}
    blob_von: dict[tuple[str, str], tuple[Eintrag, Baum]] = {}
    for b in _baeume(baum):
        baum_je_repo[b.repo] = b
        for e in b.eintraege:
            eintraege.append(e)
            blob_von[(e.repo, e.pfad)] = (e, b)
    with ExitStack() as stack:
        leser = {r: stack.enter_context(BlobLeser(b.wurzel)) for r, b in sorted(baum_je_repo.items())}
        erg = (_gleiche_dateien(eintraege)
               + _gleiche_symbole(d, blob_von, leser)
               + _aehnliche_dateien(eintraege, leser, schwelle))
    erg.sort(key=lambda x: (x.art, x.mitglieder))
    return tuple(erg)
