"""CLI: `python -m selbstbild vermessen|export|status|analysieren|goldsatz|vergleichen|suche-messen <wurzel> ...`."""
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


def _fehler(meldung: str, code: int) -> int:
    sys.stderr.write(meldung + "\n")
    return code


def _analyse_holen(wurzel: Path) -> Any:
    """Gespeicherte Analyse oder ein Exit-Code (int) samt Meldung auf stderr."""
    from selbstbild.analyse.lader import StandOhneGraph, StandUnlesbar
    from selbstbild.analyse.lauf import NichtAnalysiert, NichtVermessen, gespeicherte_analyse

    try:
        return gespeicherte_analyse(wurzel)
    except NichtVermessen as e:
        return _fehler(str(e), 1)
    except StandOhneGraph as e:
        return _fehler(str(e), 3)
    except (NichtAnalysiert, StandUnlesbar) as e:
        return _fehler(str(e), 1)


def _analysieren(args: argparse.Namespace) -> int:
    from selbstbild.analyse.ergebnis import export_json
    from selbstbild.analyse.lader import StandOhneGraph, StandUnlesbar
    from selbstbild.analyse.lauf import NichtVermessen, analysieren, speicher_spitze_mb, zusammenfassung

    start = time.perf_counter_ns()
    try:
        a = analysieren(args.wurzel, kolonie=not args.ohne_kolonie,
                        pruefen_reproduzierbar=args.pruefen_reproduzierbar,
                        kolonie_budget=args.kolonie_budget)
    except (NichtVermessen, StandUnlesbar) as e:
        return _fehler(str(e), 1)
    except StandOhneGraph as e:
        return _fehler(str(e), 3)
    if args.export is not None:
        try:
            export_json(a, args.export)
        except OSError as e:
            return _fehler(f"Export nicht schreibbar: {args.export} ({type(e).__name__}: {e.strerror})", 2)
    z = zusammenfassung(a)
    ws, privat = speicher_spitze_mb()
    z.update({"dauer_ms": (time.perf_counter_ns() - start) // 1_000_000,
              "speicher_spitze_mb": ws, "speicher_privat_spitze_mb": privat})
    _ausgeben(z)
    return 0


def _goldsatz(args: argparse.Namespace) -> int:
    import yaml

    from selbstbild.analyse.goldsatz import kandidaten

    if args.ziel.exists() and not args.ueberschreiben:
        return _fehler(f"Goldsatz existiert schon: {args.ziel}. Nicht ueberschrieben, damit eingetragene Urteile"
                       " erhalten bleiben. Zum Neu-Erzeugen --ueberschreiben angeben.", 2)
    a = _analyse_holen(args.wurzel)
    if isinstance(a, int):
        return a
    text = kandidaten(a)
    try:
        args.ziel.parent.mkdir(parents=True, exist_ok=True)
        args.ziel.write_bytes(text.encode("utf-8"))  # LF auch unter Windows
    except OSError as e:
        return _fehler(f"Goldsatz nicht schreibbar: {args.ziel} ({type(e).__name__}: {e.strerror})", 2)
    doc = yaml.safe_load(text)
    _ausgeben({"ziel": str(args.ziel), "commit": a.commit,
               **{k: len(doc[k]) for k in ("tot", "doppelt", "bereiche")}})
    return 0


def _vergleichen(args: argparse.Namespace) -> int:
    from selbstbild.analyse.goldsatz import gold_commit_und_fehlend, vergleichen

    a = _analyse_holen(args.wurzel)
    if isinstance(a, int):
        return a
    try:
        text = args.gold.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return _fehler(f"Goldsatz nicht lesbar: {args.gold} ({type(e).__name__})", 2)
    try:
        v = vergleichen(text, a)
    except ValueError as e:
        return _fehler(str(e), 2)
    commit, fehlend = gold_commit_und_fehlend(text, a)
    if commit is not None and commit != a.commit:
        sys.stderr.write(f"Warnung: Goldsatz stammt von Commit {commit[:12]}, die Analyse von {a.commit[:12]}."
                         f" {fehlend} tot-Eintraege haben kein Symbol mehr in der Analyse"
                         " (in `ignoriert` enthalten).\n")
    _ausgeben(dataclasses.asdict(v))
    return 0


def _suche_messen(args: argparse.Namespace) -> int:
    from selbstbild.suche.dicht import Embedder
    from selbstbild.suche.messung import (gold_aus_ketten, gold_aus_transkripten, messen_quellen, repos_im_index,
                                          wurzel_name)

    if len(args.ollama) != len(args.modell):
        return _fehler("--ollama und --modell muessen paarweise angegeben werden", 2)
    pfad = stand_finden(args.wurzel)
    if pfad is None:
        return _nicht_vermessen(args.wurzel)
    try:
        repos = repos_im_index(pfad.parent)
    except FileNotFoundError as e:
        return _fehler(str(e), 1)
    wurzel_repo = wurzel_name(pfad.parent)
    if wurzel_repo is None:
        return _fehler("Wurzel-Repo-Name fehlt in graph.sqlite (meta `name`): neu vermessen", 2)
    fremd = [r for r in repos if r != wurzel_repo and not r.startswith(wurzel_repo + "/")]
    if fremd:
        return _fehler(f"Index und Graph passen nicht zusammen: Repo {fremd[0]!r} liegt nicht unter"
                       f" Wurzel {wurzel_repo!r} (Index neu bauen)", 2)
    gold: dict[str, tuple[tuple[str, frozenset[str]], ...]] = {}
    ketten = args.wurzel / "docs" / "spaces" / "KETTEN.md"
    if ketten.is_file():
        gold["ketten"] = gold_aus_ketten(ketten.read_text(encoding="utf-8"), wurzel_repo, repos=repos)
    else:
        sys.stderr.write(f"Warnung: Gold-Quelle ketten fehlt ({ketten})\n")
    if args.transkripte is not None:
        gold["transkripte"] = gold_aus_transkripten(args.transkripte, args.wurzel, args.n,
                                                    wurzel_repo=wurzel_repo, repos=repos)
    else:
        sys.stderr.write("Warnung: Gold-Quelle transkripte nicht angegeben (--transkripte)\n")
    for q in [q for q, g in gold.items() if not g]:
        sys.stderr.write(f"Warnung: Gold-Quelle {q} ist leer und wird nicht gemessen\n")
        del gold[q]
    embedder = {m: Embedder("ollama", u, m) for u, m in zip(args.ollama, args.modell)}
    varianten = ["stichwort", "struktur"] + [f"{a}:{m}" for m in embedder for a in ("dicht", "fusion")]
    try:
        ergebnis = messen_quellen(pfad.parent, gold, varianten=varianten, embedder=embedder)
    except RuntimeError as e:  # Embedder-Fehler tragen nur Typ und Status
        return _fehler(str(e), 2)
    _ausgeben({"paare": {q: len(g) for q, g in gold.items()}, "ergebnis": ergebnis})  # nie Prompt-Texte
    kaputt = sorted({v for r in ergebnis.values() for v, z in r.items()
                     if v.startswith("dicht:") and "fehlgeschlagen" in z})
    if kaputt:
        return _fehler(f"Embedder nicht erreichbar, Variante(n) fehlgeschlagen: {', '.join(kaputt)}", 2)
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
    an = befehle.add_parser("analysieren", help="Stufe-2-Analyse des vermessenen Stands, Zusammenfassung als JSON")
    an.add_argument("wurzel", type=Path)
    an.add_argument("--ohne-kolonie", action="store_true", help="Kolonie (Herausforderer) auslassen")
    an.add_argument("--export", type=Path, default=None, help="kanonisches analyse.json schreiben")
    an.add_argument("--pruefen-reproduzierbar", action="store_true",
                    help="Mathe und Kolonie ein zweites Mal rechnen und vergleichen (doppelte Laufzeit)")
    an.add_argument("--kolonie-budget", action="store_true",
                    help="zusaetzlich budgetgleiche Kolonie (~10 Agenten-Schritte je Datei, max. 5000 Schritte)")
    g = befehle.add_parser("goldsatz", help="Goldsatz-Kandidaten (YAML) aus der gespeicherten Analyse")
    g.add_argument("wurzel", type=Path)
    g.add_argument("ziel", type=Path)
    g.add_argument("--ueberschreiben", action="store_true",
                   help="vorhandene Zieldatei ersetzen (eingetragene Urteile gehen verloren)")
    vg = befehle.add_parser("vergleichen", help="ausgefuellten Goldsatz gegen Mathe und Kolonie messen")
    vg.add_argument("wurzel", type=Path)
    vg.add_argument("gold", type=Path)
    sm = befehle.add_parser("suche-messen", help="recall@10 und MRR der Suche gegen KETTEN- und Transkript-Gold")
    sm.add_argument("wurzel", type=Path)
    sm.add_argument("--ollama", action="append", default=[], help="Ollama-URL (paarweise mit --modell)")
    sm.add_argument("--modell", action="append", default=[], help="Embedding-Modell (paarweise mit --ollama)")
    sm.add_argument("--transkripte", type=Path, default=None, help="Ordner mit Claude-Code-JSONL-Transkripten")
    sm.add_argument("-n", type=int, default=20, help="Stichprobengroesse der Transkript-Prompts")
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
    if args.befehl == "analysieren":
        return _analysieren(args)
    if args.befehl == "goldsatz":
        return _goldsatz(args)
    if args.befehl == "vergleichen":
        return _vergleichen(args)
    if args.befehl == "suche-messen":
        return _suche_messen(args)
    return _status(args.wurzel)


if __name__ == "__main__":
    sys.exit(main())
