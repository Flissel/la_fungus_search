"""CLI: `python -m selbstbild vermessen|export|status <wurzel> ...`."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from selbstbild.speicher import Speicher
from selbstbild.vermessen import stand_finden, vermessen


def _ausgeben(daten: Any) -> None:
    sys.stdout.write(json.dumps(daten, ensure_ascii=True, sort_keys=True, indent=1) + "\n")


def _nicht_vermessen(wurzel: Path) -> int:
    sys.stderr.write(f"noch nicht vermessen: {wurzel}\n")
    return 1


def _neu_vermessen(wurzel: Path) -> int:
    sys.stderr.write(f"neu vermessen nötig: {wurzel} (Stand ohne Graph, z. B. nach einem Formatwechsel)\n")
    return 3


def _status(wurzel: Path) -> int:
    pfad = stand_finden(wurzel)
    if pfad is None:
        return _nicht_vermessen(wurzel)
    with Speicher(pfad) as s:
        bericht_text = s.meta("bericht")
        commit_zeit = s.meta("commit_zeit")
        gemessen = s.meta("gemessen_um")
        name = s.meta("name")
        hat_graph = s.hat_graph()
    if bericht_text is None or gemessen is None or not hat_graph:
        return _neu_vermessen(wurzel)
    bericht = json.loads(bericht_text)
    jetzt = int(time.time())
    _ausgeben({
        "name": name,
        "commit": bericht["commit"],
        "commit_zeit": int(commit_zeit) if commit_zeit else None,
        "gemessen_um": int(gemessen),
        "alter_s": max(0, jetzt - int(gemessen)),
        **{k: bericht[k] for k in ("dateien", "knoten", "kanten", "kanten_unsicher",
                                   "fehlerhaft", "uebersprungen", "submodule")},
        "stand": str(pfad),
    })
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="selbstbild", description="Vermisst ein Git-Repo als Graph.")
    p.add_argument("-v", "--ausfuehrlich", action="store_true", help="Fortschritt nach stderr")
    befehle = p.add_subparsers(dest="befehl", required=True)
    v = befehle.add_parser("vermessen", help="Commit vermessen, Bericht als JSON")
    v.add_argument("wurzel", type=Path)
    v.add_argument("--commit", default="HEAD")
    v.add_argument("--voll", action="store_true", help="Parse-Cache umgehen")
    e = befehle.add_parser("export", help="Graph als JSON exportieren")
    e.add_argument("wurzel", type=Path)
    e.add_argument("ziel", type=Path)
    s = befehle.add_parser("status", help="letzter Commit, Zähler, Alter")
    s.add_argument("wurzel", type=Path)
    args = p.parse_args(argv)

    if args.ausfuehrlich:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                            format="%(asctime)s %(levelname)s %(message)s")

    if args.befehl == "vermessen":
        bericht = vermessen(args.wurzel, args.commit, voll=args.voll)
        _ausgeben(dataclasses.asdict(bericht))
        return 0
    if args.befehl == "export":
        pfad = stand_finden(args.wurzel)
        if pfad is None:
            return _nicht_vermessen(args.wurzel)
        with Speicher(pfad) as sp:
            if not sp.hat_graph():  # nie einen leeren Graphen unter altem Commit exportieren
                return _neu_vermessen(args.wurzel)
            sp.export_json(args.ziel)
        return 0
    return _status(args.wurzel)


if __name__ == "__main__":
    sys.exit(main())
