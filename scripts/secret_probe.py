"""Secret-Probe (Spec 12a, R16): Stehen Werte aus einer .env im Selbstbild-Stand?

Aufruf (startet der User selbst, mit seiner echten .env):

    python scripts/secret_probe.py --env <pfad/.env> --graph <graph.sqlite> --export <export.json>

Die Werte bleiben nur im Speicher dieses Prozesses. Gesucht wird jeder Wert als
UTF-8-Bytefolge (roh und JSON-escaped) in den Bytes von `graph.sqlite`, einer evtl.
vorhandenen `graph.sqlite-wal` und des Exports. Ausgegeben werden NUR die Namen der
Variablen mit Treffer und Zähler, nie Werte. Exit 1 bei mindestens einem Treffer.

Werte kürzer als 6 Zeichen werden ignoriert (z. B. `true`, `8080`, `1`): Sie kommen als
gewöhnliche Teilstrings überall vor und wären keine aussagekräftigen Treffer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MIN_LAENGE = 6


def env_werte(text: str) -> dict[str, str]:
    """Einfacher .env-Leser: `[export ]NAME=wert`, Kommentare, umschließende Quotes."""
    werte: dict[str, str] = {}
    for zeile in text.splitlines():
        zeile = zeile.strip()
        if not zeile or zeile.startswith("#") or "=" not in zeile:
            continue
        if zeile.startswith("export "):
            zeile = zeile[len("export "):].lstrip()
        name, _, wert = zeile.partition("=")
        name, wert = name.strip(), wert.strip()
        if len(wert) >= 2 and wert[0] == wert[-1] and wert[0] in "\"'":
            wert = wert[1:-1]
        if name:
            werte[name] = wert
    return werte


def _muster(wert: str) -> set[bytes]:
    roh = wert.encode("utf-8")
    escaped = json.dumps(wert, ensure_ascii=False)[1:-1].encode("utf-8")
    ascii_escaped = json.dumps(wert, ensure_ascii=True)[1:-1].encode("ascii")
    return {roh, escaped, ascii_escaped}


def pruefen(werte: dict[str, str], daten: list[bytes]) -> dict[str, int]:
    """Name -> Anzahl Treffer über alle Datenquellen (nur Werte ab MIN_LAENGE)."""
    treffer: dict[str, int] = {}
    for name in sorted(werte):
        wert = werte[name]
        if len(wert) < MIN_LAENGE:
            continue
        anzahl = sum(blob.count(m) for m in _muster(wert) for blob in daten)
        if anzahl:
            treffer[name] = anzahl
    return treffer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", required=True, type=Path)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--export", required=True, type=Path)
    args = p.parse_args(argv)

    werte = env_werte(args.env.read_text(encoding="utf-8", errors="replace"))
    daten = [args.graph.read_bytes(), args.export.read_bytes()]
    wal = args.graph.with_name(args.graph.name + "-wal")
    if wal.exists():
        daten.append(wal.read_bytes())
    geprueft = sum(1 for w in werte.values() if len(w) >= MIN_LAENGE)
    treffer = pruefen(werte, daten)
    del werte, daten

    for name, anzahl in treffer.items():
        sys.stdout.write(f"TREFFER {name}: {anzahl}\n")
    sys.stdout.write(f"geprueft: {geprueft} Werte, Treffer: {len(treffer)} Variablen\n")
    return 1 if treffer else 0


if __name__ == "__main__":
    sys.exit(main())
