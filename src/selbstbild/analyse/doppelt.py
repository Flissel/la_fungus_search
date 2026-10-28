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
(die Kandidaten werden aus zwei frisch gelesenen Blobs nachgerechnet), auf 4 Stellen gerundet.
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
_PRIM = (1 << 61) - 1
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
_KOMMENTAR_C = r"//[^\n]*|/\*.*?\*/"
_TOKEN = (r"(?P<k>{kommentar})|(?P<s>\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"(?:\\.|[^\"\\\n])*\""
          r"|'(?:\\.|[^'\\\n])*'|`[^`]*`)|(?P<n>\d[\w.]*)|(?P<i>[A-Za-z_]\w*)|(?P<p>\S)")
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
    return {int.from_bytes(hashlib.blake2b(" ".join(t[i:i + SHINGLE]).encode(), digest_size=8).digest(), "big")
            for i in range(max(len(t) - SHINGLE + 1, 0))}


def _permutationen() -> list[tuple[int, int]]:
    rnd = random.Random(SEED)  # eigene Instanz: kein globaler Zufallszustand
    return [(rnd.randrange(1, _PRIM), rnd.randrange(0, _PRIM)) for _ in range(PERMUTATIONEN)]


def _signatur(hashes: set[int], perm: list[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(min((a * x + b) % _PRIM for x in hashes) for a, b in perm)


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
    je_digest: dict[str, list[str]] = {}
    for sid in sorted(d.knoten):
        k = d.knoten[sid]
        if k.art == "symbol" and k.beleg is not None and k.beleg.digest:
            je_digest.setdefault(k.beleg.digest, []).append(sid)
    erg: list[Doppelung] = []
    # Datei -> (qualname, digest) -> Zeilen; nur fuer Dateien mit Kandidaten gefuellt
    laengen: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for digest, ids in sorted(je_digest.items()):
        if len(ids) < 2:
            continue
        lang: list[str] = []
        blobs: set[str] = set()
        for sid in ids:
            beleg = d.knoten[sid].beleg
            if beleg is None:
                continue
            schluessel = (beleg.repo, beleg.pfad)
            ziel = blob_von.get(schluessel)
            if ziel is None:
                continue
            eintrag, baum = ziel
            if schluessel not in laengen:
                p = parsen(eintrag.sprache or "", eintrag.pfad, leser[baum.repo].lesen(eintrag.blob))
                laengen[schluessel] = {(s.qualname, s.digest): s.zeile_bis - s.zeile_von + 1
                                       for s in p.symbole}
            if laengen[schluessel].get((sid.rpartition("#")[2], digest), 0) >= MIN_ZEILEN_SYMBOL:
                lang.append(sid)
                blobs.add(eintrag.blob)
        if len(lang) >= 2 and len(blobs) >= 2:  # gleiche Datei ist schon gleiche_datei
            erg.append(Doppelung("gleiches_symbol", tuple(sorted(lang)), 1.0))
    return erg


def _aehnliche_dateien(eintraege: list[Eintrag], leser: dict[str, BlobLeser],
                       schwelle: float) -> list[Doppelung]:
    perm = _permutationen()
    ids_je_blob: dict[str, set[str]] = {}
    erster: dict[str, Eintrag] = {}
    for e in eintraege:
        ids_je_blob.setdefault(e.blob, set()).add(id_datei(e.repo, e.pfad))
        erster.setdefault(e.blob, e)
    sig: dict[str, tuple[int, ...]] = {}
    for blob in sorted(erster):
        e = erster[blob]
        if e.sprache is None or e.sprache in _OHNE_SHINGLES or e.groesse < MIN_ZEILEN_DATEI:
            continue
        text = _text(leser[e.repo], blob)
        if text.count("\n") + 1 < MIN_ZEILEN_DATEI:
            continue
        h = _shingles(text, e.sprache)
        if h:
            sig[blob] = _signatur(h, perm)  # nur die Signatur bleibt im Speicher
    baender: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for blob, s in sig.items():
        for b in range(BAENDER):
            baender.setdefault((b, s[b * ZEILEN_JE_BAND:(b + 1) * ZEILEN_JE_BAND]), []).append(blob)
    kandidaten: set[tuple[str, str]] = set()
    for gruppe in baender.values():
        gruppe.sort()
        for i, a in enumerate(gruppe):
            for b in gruppe[i + 1:]:
                kandidaten.add((a, b))
    erg: list[Doppelung] = []
    for a, b in sorted(kandidaten):
        gleich = sum(x == y for x, y in zip(sig[a], sig[b])) / PERMUTATIONEN
        if gleich < schwelle - _VORFILTER:
            continue
        ea, eb = erster[a], erster[b]
        if ea.sprache != eb.sprache or ea.sprache is None:
            continue
        j = _jaccard(_shingles(_text(leser[ea.repo], a), ea.sprache),
                     _shingles(_text(leser[eb.repo], b), eb.sprache))
        if j >= schwelle:
            erg.append(Doppelung("aehnliche_datei", tuple(sorted(ids_je_blob[a] | ids_je_blob[b])),
                                 round(j, 4)))
    return erg


def doppelungen(wurzel: Path, d: GraphDaten, schwelle: float = 0.9) -> tuple[Doppelung, ...]:
    """Doppelungen am Commit `d.commit`; Blobs nur ueber `quelle`, sortiert nach (art, mitglieder)."""
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
