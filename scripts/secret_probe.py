"""Secret-Probe (Spec 12a, R16): Stehen Werte aus einer .env im Selbstbild-Stand?

Aufruf (startet der User selbst, mit seiner echten .env):

    python scripts/secret_probe.py --env <pfad/.env> --graph <graph.sqlite> --export <export.json>

Die Werte bleiben nur im Speicher dieses Prozesses. Gesucht wird jeder Wert (roh und
JSON-escaped):
- als UTF-8-Bytefolge in den Dateibytes von `graph.sqlite`, einer evtl. vorhandenen
  `graph.sqlite-wal` und des Exports (blockweise gelesen), und
- in allen Text- und Blob-Spalten aller Tabellen von `graph.sqlite`, gelesen ueber sqlite3
  (nur lesend). Das findet auch Werte, die in der Datei ueber Overflow-Seiten verteilt oder
  anders kodiert (UTF-16) stehen.
Ausgegeben werden NUR die Namen der Variablen mit Treffer und die Zahl der Fundstellen
(Dateibytes plus Spaltenwerte), nie Werte.

Exit-Codes: 0 = kein Treffer, 1 = mindestens ein Treffer, 2 = Pruefung nicht moeglich
(Datei fehlt oder ist unlesbar, `--graph` ist keine lesbare SQLite-Datenbank). Fail-closed:
Ein Fehler wird nie als "0 Treffer" gemeldet.

Werte kuerzer als 6 Zeichen werden ignoriert (z. B. `true`, `8080`, `1`): Sie kommen als
gewoehnliche Teilstrings ueberall vor und waeren keine aussagekraeftigen Treffer.
Unquotierte Werte enden vor ` #` (Inline-Kommentar); in Quotes gehoert `#` zum Wert.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

MIN_LAENGE = 6
_SQLITE_KOPF = b"SQLite format 3\x00"


class ProbeFehler(Exception):
    """Pruefung nicht moeglich (Exit 2). Die Meldung enthaelt nie Werte aus der .env."""


def env_werte(text: str) -> dict[str, str]:
    """Einfacher .env-Leser: `[export ]NAME=wert`, Kommentare, Quotes, Inline-Kommentare."""
    werte: dict[str, str] = {}
    for zeile in text.splitlines():
        zeile = zeile.strip()
        if not zeile or zeile.startswith("#") or "=" not in zeile:
            continue
        if zeile.startswith("export "):
            zeile = zeile[len("export "):].lstrip()
        name, _, wert = zeile.partition("=")
        name, wert = name.strip(), wert.strip()
        if wert[:1] in ("'", '"'):
            ende = wert.find(wert[0], 1)
            wert = wert[1:ende] if ende > 0 else wert[1:]
        else:
            for trenner in (" #", "\t#"):
                if trenner in wert:
                    wert = wert[: wert.index(trenner)].rstrip()
        if name:
            werte[name] = wert
    return werte


def _formen(wert: str) -> set[str]:
    return {wert, json.dumps(wert, ensure_ascii=False)[1:-1], json.dumps(wert, ensure_ascii=True)[1:-1]}


def _lesbar(pfad: Path, schalter: str) -> None:
    """Datei muss existieren und lesbar sein, sonst ProbeFehler."""
    try:
        with open(pfad, "rb") as f:
            f.read(1)
    except OSError as fehler:
        raise ProbeFehler(f"{schalter} {pfad}: nicht lesbar ({type(fehler).__name__})") from None


def _datei_treffer(pfad: Path, muster: list[bytes], block: int = 16 * 1024 * 1024) -> int:
    """Fundstellen aller Muster in der Datei, blockweise gelesen (konstanter Speicher).

    Kein mmap: `mmap.find` stuerzt unter CPython 3.11.0/Windows mit einer Zugriffsverletzung
    ab. Zwischen den Bloecken ueberlappt der Puffer um (laengstes Muster - 1) Bytes; gezaehlt
    wird nur, was im neuen Block endet, also jede Fundstelle genau einmal.
    """
    if not muster:
        return 0
    rest_laenge = max(len(m) for m in muster) - 1
    anzahl = 0
    rest = b""
    try:
        with open(pfad, "rb") as f:
            while True:
                stueck = f.read(block)
                if not stueck:
                    break
                puffer = rest + stueck
                for mu in muster:
                    pos = puffer.find(mu, max(0, len(rest) - len(mu) + 1))
                    while pos != -1:
                        anzahl += 1
                        pos = puffer.find(mu, pos + 1)
                rest = puffer[-rest_laenge:] if rest_laenge else b""
    except OSError as fehler:
        raise ProbeFehler(f"{pfad}: Bytesuche fehlgeschlagen ({type(fehler).__name__})") from None
    return anzahl


def _spalten_treffer(db: Path, formen: dict[str, set[str]]) -> dict[str, int]:
    """Name -> Fundstellen in allen Text-/Blob-Zellen aller Tabellen (nur lesend)."""
    with open(db, "rb") as f:
        if f.read(len(_SQLITE_KOPF)) != _SQLITE_KOPF:
            raise ProbeFehler(f"--graph {db}: keine SQLite-Datenbank")
    treffer: dict[str, int] = {}
    roh = {n: {f.encode("utf-8") for f in fs} for n, fs in formen.items()}
    c: sqlite3.Connection | None = None
    try:
        c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
        tabellen = [z[0] for z in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for tabelle in tabellen:
            for zeile in c.execute(f'SELECT * FROM "{tabelle.replace(chr(34), chr(34) * 2)}"'):
                for zelle in zeile:
                    if isinstance(zelle, str):
                        for name, fs in formen.items():
                            n = sum(zelle.count(f) for f in fs)
                            if n:
                                treffer[name] = treffer.get(name, 0) + n
                    elif isinstance(zelle, bytes):
                        for name, bs in roh.items():
                            n = sum(zelle.count(b) for b in bs)
                            if n:
                                treffer[name] = treffer.get(name, 0) + n
    except sqlite3.Error as fehler:
        raise ProbeFehler(f"--graph {db}: SQLite nicht lesbar ({type(fehler).__name__})") from None
    finally:
        if c is not None:
            c.close()
    return treffer


def pruefen(werte: dict[str, str], dateien: list[Path], db: Path | None) -> dict[str, int]:
    """Name -> Fundstellen ueber Dateibytes und (falls db) Spaltenwerte, nur Werte ab MIN_LAENGE.

    Wirft ProbeFehler, wenn eine Quelle nicht vollstaendig geprueft werden kann.
    """
    formen = {n: _formen(w) for n, w in werte.items() if len(w) >= MIN_LAENGE}
    treffer: dict[str, int] = {}
    for name, fs in formen.items():
        muster = sorted({f.encode("utf-8") for f in fs})
        anzahl = sum(_datei_treffer(p, muster) for p in dateien)
        if anzahl:
            treffer[name] = anzahl
    if db is not None:
        for name, n in _spalten_treffer(db, formen).items():
            treffer[name] = treffer.get(name, 0) + n
    return dict(sorted(treffer.items()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", required=True, type=Path)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--export", required=True, type=Path)
    args = p.parse_args(argv)

    try:
        for pfad, schalter in ((args.env, "--env"), (args.graph, "--graph"), (args.export, "--export")):
            _lesbar(pfad, schalter)
        try:
            text = args.env.read_text(encoding="utf-8", errors="replace")
        except OSError as fehler:
            raise ProbeFehler(f"--env {args.env}: nicht lesbar ({type(fehler).__name__})") from None
        werte = env_werte(text)
        del text
        wal = args.graph.with_name(args.graph.name + "-wal")
        dateien = [args.graph, args.export] + ([wal] if wal.exists() else [])
        geprueft = sum(1 for w in werte.values() if len(w) >= MIN_LAENGE)
        treffer = pruefen(werte, dateien, args.graph)
        del werte
    except ProbeFehler as fehler:
        sys.stderr.write(f"FEHLER: {fehler}. Pruefung nicht moeglich, Exit 2.\n")
        return 2

    for name, anzahl in treffer.items():
        sys.stdout.write(f"TREFFER {name}: {anzahl}\n")
    sys.stdout.write(f"geprueft: {geprueft} Werte, Treffer: {len(treffer)} Variablen\n")
    return 1 if treffer else 0


if __name__ == "__main__":
    sys.exit(main())
