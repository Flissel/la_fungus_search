"""Secret-Probe (Spec 12a, R16): Stehen Werte aus einer .env im Selbstbild-Stand?

Aufruf (startet der User selbst, mit seiner echten .env):

    python scripts/secret_probe.py --env <pfad/.env> --graph <graph.sqlite> --export <export.json>

Die Werte bleiben nur im Speicher dieses Prozesses. Gesucht wird jeder Wert (roh und
JSON-escaped):
- als UTF-8-Bytefolge in den Dateibytes von `graph.sqlite`, einer evtl. vorhandenen
  `graph.sqlite-wal` und des Exports (per mmap), und
- in allen Text- und Blob-Spalten aller Tabellen von `graph.sqlite`, gelesen ueber sqlite3
  (nur lesend). Das findet auch Werte, die in der Datei ueber Overflow-Seiten verteilt oder
  anders kodiert (UTF-16) stehen.
Ausgegeben werden NUR die Namen der Variablen mit Treffer und die Zahl der Fundstellen
(Dateibytes plus Spaltenwerte), nie Werte. Exit 1 bei mindestens einem Treffer.

Werte kuerzer als 6 Zeichen werden ignoriert (z. B. `true`, `8080`, `1`): Sie kommen als
gewoehnliche Teilstrings ueberall vor und waeren keine aussagekraeftigen Treffer.
Unquotierte Werte enden vor ` #` (Inline-Kommentar); in Quotes gehoert `#` zum Wert.
"""
from __future__ import annotations

import argparse
import json
import mmap
import sqlite3
import sys
from pathlib import Path

MIN_LAENGE = 6


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


def _datei_treffer(pfad: Path, muster: list[bytes]) -> int:
    if not pfad.exists() or pfad.stat().st_size == 0:
        return 0
    anzahl = 0
    with open(pfad, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        for mu in muster:
            pos = m.find(mu)
            while pos != -1:
                anzahl += 1
                pos = m.find(mu, pos + 1)
    return anzahl


def _spalten_treffer(db: Path, formen: dict[str, set[str]]) -> dict[str, int]:
    """Name -> Fundstellen in allen Text-/Blob-Zellen aller Tabellen (nur lesend)."""
    treffer: dict[str, int] = {}
    try:
        c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return treffer
    try:
        tabellen = [z[0] for z in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        roh = {n: {f.encode("utf-8") for f in fs} for n, fs in formen.items()}
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
    except sqlite3.DatabaseError:
        pass  # keine (gueltige) SQLite-Datei: nur die Bytesuche zaehlt
    finally:
        c.close()
    return treffer


def pruefen(werte: dict[str, str], dateien: list[Path], db: Path | None) -> dict[str, int]:
    """Name -> Fundstellen ueber Dateibytes und (falls db) Spaltenwerte, nur Werte ab MIN_LAENGE."""
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

    werte = env_werte(args.env.read_text(encoding="utf-8", errors="replace"))
    wal = args.graph.with_name(args.graph.name + "-wal")
    dateien = [args.graph, args.export] + ([wal] if wal.exists() else [])
    geprueft = sum(1 for w in werte.values() if len(w) >= MIN_LAENGE)
    treffer = pruefen(werte, dateien, args.graph)
    del werte

    for name, anzahl in treffer.items():
        sys.stdout.write(f"TREFFER {name}: {anzahl}\n")
    sys.stdout.write(f"geprueft: {geprueft} Werte, Treffer: {len(treffer)} Variablen\n")
    return 1 if treffer else 0


if __name__ == "__main__":
    sys.exit(main())
