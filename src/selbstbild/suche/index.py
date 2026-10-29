"""Suchindex `suche.sqlite` (SQLite FTS5), gebaut nur lesend aus Graph (Stufe 1) und Analyse (Stufe 2).

`graph.sqlite` und `analyse.sqlite` werden nur mit `mode=ro` geoeffnet; `suche.sqlite` gehoert allein
diesem Modul. Der Index wird in `suche.sqlite.<pid>.tmp` gebaut und per `os.replace` ersetzt (nie halbfertig
sichtbar). Zeilen gehen sortiert nach id hinein (deterministisch). Es werden nur Namen, Pfade, Symbolnamen
und committeter Doku-Text indiziert, nie Env-Werte (Env-Knoten tragen nur Namen) und nie Beleg-Anker.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator

from ..analyse.ergebnis import Analyse, digest_von, lesen
from ..analyse.lader import ANALYSE_VERSION, GraphDaten, StandOhneGraph, StandUnlesbar, datei_von, laden
from ..analyse.lauf import _stand_meta, parameter
from ..quelle import Baum, BlobLeser, baum_lesen

INDEX_VERSION: int = 1

_ARTEN = frozenset({"datei", "symbol", "route", "mcp_werkzeug", "tabelle", "dienst", "env", "doku_seite"})
# Reihenfolge wie `gespeicherte_analyse`: mit Kolonie bevorzugt, sonst ohne.
_PARAMETER = ((True, True), (True, False), (False, False))

_SCHEMA = """
CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE eintrag (id TEXT PRIMARY KEY, art TEXT NOT NULL, repo TEXT, pfad TEXT, zeile INTEGER,
                      blob TEXT, lebendigkeit REAL, tot TEXT, bereich TEXT, warnung TEXT);
CREATE VIRTUAL TABLE volltext USING fts5(id UNINDEXED, name, pfad, text,
                                         tokenize='unicode61 remove_diacritics 2');
CREATE TABLE kante (von TEXT NOT NULL, nach TEXT NOT NULL, art TEXT NOT NULL, sicher INTEGER NOT NULL);
CREATE INDEX kante_von ON kante(von);
CREATE INDEX kante_nach ON kante(nach);
CREATE TABLE vektor (datei TEXT PRIMARY KEY, modell TEXT NOT NULL, v BLOB NOT NULL);
"""

_GRENZE = re.compile(r"(?<=[a-zäöüß0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-zäöüß])")
_NICHT_WORT = re.compile(r"[\W_]+")
_PFAD_TEILE = re.compile(r"[/._\-\\]+")


def index_pfad(stand: Path) -> Path:
    return Path(stand) / "suche.sqlite"


def zerlegen(name: str) -> str:
    """Teilwoerter (snake_case, CamelCase, Punkte) plus der ganze Name klein: `kaufeLaden` -> `kaufe laden kaufeladen`."""
    teile = [t.lower() for t in _NICHT_WORT.split(_GRENZE.sub(" ", name)) if t]
    ganz = "".join(t for t in _NICHT_WORT.split(name.lower()) if t)
    return " ".join(teile + ([ganz] if len(teile) > 1 else []))


def _pfad_teile(pfad: str | None) -> str:
    return " ".join(t.lower() for t in _PFAD_TEILE.split(pfad or "") if t)


def _repo_pfad(k_id: str) -> tuple[str | None, str | None]:
    """`datei:<repo>:<pfad>` / `symbol:<repo>:<pfad>#q` / `doku_seite:<repo>:<pfad>` -> (repo, pfad); sonst (None, None)."""
    teile = k_id.split(":", 2)
    if teile[0] in ("datei", "doku_seite") and len(teile) == 3:
        return teile[1], teile[2]
    if teile[0] == "symbol" and len(teile) == 3:
        return teile[1], teile[2].rpartition("#")[0]
    return None, None


# --- Analyse ---------------------------------------------------------------

def _gefundener_digest(stand: Path, commit: str, gemessen_um: str | None) -> str | None:
    """Parameter-Digest des gespeicherten Laufs zu diesem Graphen (leichte Abfrage ohne Dekodieren)."""
    db = Path(stand) / "analyse.sqlite"
    if not db.is_file():
        return None
    c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        for mit_kolonie, mit_budget in _PARAMETER:
            digest = digest_von(parameter(mit_kolonie, mit_budget))
            try:
                z = c.execute("SELECT 1 FROM lauf l JOIN herkunft h ON h.commit_id=l.commit_id AND"
                              " h.version=l.version AND h.parameter_digest=l.parameter_digest"
                              " WHERE l.commit_id=? AND l.version=? AND l.parameter_digest=?"
                              " AND h.graph_gemessen_um=?",
                              (commit, ANALYSE_VERSION, digest, gemessen_um or "")).fetchone()
            except sqlite3.OperationalError:  # Stand von vor der Herkunftstabelle
                z = None
            if z is not None:
                return digest
        return None
    finally:
        c.close()


def _analyse_lesen(stand: Path, commit: str, gemessen_um: str | None) -> tuple[Analyse | None, str]:
    digest = _gefundener_digest(stand, commit, gemessen_um)
    if digest is None:
        return None, ""
    return lesen(stand, commit, ANALYSE_VERSION, digest, graph_gemessen_um=gemessen_um or ""), digest


# --- Bauen -----------------------------------------------------------------

def _baeume(b: Baum) -> Iterator[Baum]:
    yield b
    for u in b.unterbaeume:
        yield from _baeume(u)


def _wert(attr: tuple[tuple[str, str], ...], schluessel: str) -> str | None:
    return next((v for k, v in attr if k == schluessel), None)


def _zahl(text: str | None) -> float | None:
    try:
        return None if text is None else float(text)
    except ValueError:
        return None


def _bauen(d: GraphDaten, wurzel: Path, stand: Path, ziel: Path) -> dict[str, int]:
    analyse, digest = _analyse_lesen(stand, d.commit, d.gemessen_um)
    bereich_von: dict[str, str] = {}
    tot_von: dict[str, str] = {}
    warnung_von: dict[str, set[str]] = {}
    if analyse is not None:
        for b in analyse.bereiche:
            for datei in b.dateien:
                bereich_von[datei] = b.name
        tot_von = {t.symbol: t.stufe for t in analyse.tot}
        for dp in analyse.doppelungen:
            if dp.art == "gleiche_datei":
                for m in dp.mitglieder:
                    warnung_von.setdefault(m, set()).update(x for x in dp.mitglieder if x != m)
        del analyse

    blob_von: dict[tuple[str, str], str] = {}
    doku_blob: dict[tuple[str, str], tuple[Path, str]] = {}
    doku_ids = {i for i, k in d.knoten.items() if k.art == "doku_seite"}
    for baum in _baeume(baum_lesen(wurzel, d.commit)):
        for e in baum.eintraege:
            blob_von[(e.repo, e.pfad)] = e.blob
            if f"doku_seite:{e.repo}:{e.pfad}" in doku_ids:
                doku_blob[(e.repo, e.pfad)] = (baum.wurzel, e.blob)

    symbole_von: dict[str, list[str]] = {}
    for i, k in d.knoten.items():
        if k.art == "symbol":
            symbole_von.setdefault(datei_von(i), []).append(k.name)

    ziel.parent.mkdir(parents=True, exist_ok=True)
    for rest in (ziel, ziel.with_name(ziel.name + "-journal")):
        rest.unlink(missing_ok=True)
    c = sqlite3.connect(str(ziel), isolation_level=None)
    zaehler = {"eintrag": 0, "kante": 0, "doku_text": 0, "mit_analyse": int(bool(digest))}
    try:
        c.execute("PRAGMA journal_mode=OFF")  # Zwischendatei: Atomaritaet kommt von os.replace
        c.execute("PRAGMA synchronous=OFF")
        c.executescript(_SCHEMA)
        c.execute("BEGIN")
        with ExitStack() as stapel:
            leser: dict[Path, BlobLeser] = {}
            for i in sorted(i for i, k in d.knoten.items() if k.art in _ARTEN):
                k = d.knoten[i]
                repo, pfad = _repo_pfad(i)
                zeile = None
                if k.beleg is not None:
                    repo, pfad, zeile = k.beleg.repo, k.beleg.pfad, k.beleg.zeile
                datei = i if k.art == "datei" else (datei_von(i) if k.art == "symbol" else None)
                datei_attr = d.knoten[datei].attr if datei is not None and datei in d.knoten else ()
                blob = blob_von.get((repo, pfad)) if k.art in ("datei", "symbol") and repo and pfad else None
                text = ""
                if k.art == "datei":
                    text = " ".join(zerlegen(n) for n in sorted(symbole_von.get(i, ())))
                elif k.art == "doku_seite" and (repo, pfad) in doku_blob:
                    w, b = doku_blob[(repo, pfad)]
                    if w not in leser:
                        leser[w] = stapel.enter_context(BlobLeser(w))
                    text = leser[w].lesen(b).decode("utf-8", errors="replace")
                    zaehler["doku_text"] += 1
                warnung = warnung_von.get(i) if k.art == "datei" else None
                c.execute("INSERT INTO eintrag VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    i, k.art, repo, pfad, zeile, blob, _zahl(_wert(datei_attr, "lebendigkeit")),
                    tot_von.get(i) if k.art == "symbol" else None,
                    bereich_von.get(datei) if datei is not None else None,
                    "kopie:" + ",".join(sorted(warnung)) if warnung else None))
                c.execute("INSERT INTO volltext VALUES (?,?,?,?)", (i, zerlegen(k.name), _pfad_teile(pfad), text))
                zaehler["eintrag"] += 1
        for kante in sorted(d.kanten, key=lambda x: (x.von, x.nach, x.art, x.sicher)):
            a, b = d.knoten.get(kante.von), d.knoten.get(kante.nach)
            if a is None or b is None or a.art not in _ARTEN or b.art not in _ARTEN:
                continue
            c.execute("INSERT INTO kante VALUES (?,?,?,?)", (kante.von, kante.nach, kante.art, int(kante.sicher)))
            zaehler["kante"] += 1
        c.executemany("INSERT INTO meta VALUES (?,?)", [
            ("analyse_digest", digest), ("commit", d.commit), ("gemessen_um", d.gemessen_um or ""),
            ("index_version", str(INDEX_VERSION))])
        c.execute("COMMIT")
    finally:
        c.close()
    return zaehler


def _ersetzen(tmp: Path, ziel: Path) -> None:
    """`os.replace` mit bis zu 5 Versuchen: Windows sperrt die Zieldatei, solange ein Leser sie offen hat."""
    for versuch in range(1, 6):
        try:
            os.replace(tmp, ziel)
            return
        except PermissionError:
            if versuch == 5:
                raise
            time.sleep(0.1 * versuch)


def index_bauen(wurzel: Path, stand: Path) -> dict[str, int]:
    """Baut `<stand>/suche.sqlite` neu und ersetzt die alte Datei atomar. Liefert Zaehler."""
    stand = Path(stand)
    ziel = index_pfad(stand)
    tmp = ziel.with_name(f"{ziel.name}.{os.getpid()}.tmp")  # eindeutig je Prozess (parallele Bauten)
    d = laden(stand)
    try:
        zaehler = _bauen(d, Path(wurzel), stand, tmp)
        _ersetzen(tmp, ziel)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return zaehler


def index_aktuell(stand: Path) -> bool:
    """Commit und `gemessen_um` des Graphen sowie der Analyse-Digest stimmen mit dem Index ueberein."""
    stand = Path(stand)
    ziel = index_pfad(stand)
    if not ziel.is_file() or not (stand / "graph.sqlite").is_file():
        return False
    try:
        commit, gemessen_um = _stand_meta(stand)
        if commit is None:
            return False
        c = sqlite3.connect(f"{ziel.resolve().as_uri()}?mode=ro", uri=True)
        try:
            meta = dict(c.execute("SELECT k, v FROM meta"))
        finally:
            c.close()
        digest = _gefundener_digest(stand, commit, gemessen_um) or ""
    except (sqlite3.DatabaseError, StandOhneGraph, StandUnlesbar):
        return False
    return (meta.get("commit") == commit and meta.get("gemessen_um") == (gemessen_um or "")
            and meta.get("analyse_digest") == digest and meta.get("index_version") == str(INDEX_VERSION))
