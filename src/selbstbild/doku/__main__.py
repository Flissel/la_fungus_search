"""CLI des Doku-Loops (Task 15): `python -m doku_loop <befehl>`, Einstieg `scripts\\doku-loop.cmd`.

    tick [--erzwingen] [--ohne-llm]   ein Lauf (siehe tick.py); Exit 0 auch bei `uebersprungen`
    seed                              Graph aus KETTEN.md, Tabellen einbetten, rendern, committen
    messen [--json] [--details]       Beleg-Drift (C.9.1) + Seed-Deckung (C.9.2) + Ankerarten (C.9.3)
    render [--schreiben]              KETTEN.md aus graph.json neu rendern (ohne --schreiben nur zaehlen)
    graph-schlank                     vollen docs/graph/graph.json durch die schlanke Fassung ersetzen (committet)
    (voller Graph lokal in %LOCALAPPDATA%/VibeMind/doku_loop/graph-voll-<hash>.json, im Repo nur der schlanke)
    (gemessen wird nur committeter Code: Messbaum = detached Worktrees in %LOCALAPPDATA%/VibeMind/doku_loop/
     messbaum-<hash>/, Ruling 50; tick, seed und messen stellen ihn bereit, status zeigt ihn)
    (Stand, Pruefliste und Urteils-Cache liegen lokal in %LOCALAPPDATA%/VibeMind/doku_loop, nie im Repo)
    dienste-seed                      fehlende Dienste aus Compose + vibemind.config.json in dienste.yaml
    status                            Stand, Pruefliste, Schalter (Notizen, Zweig), Budget, Marker, Sperre, Aufgabe
    aufgabe --pruefen|--umschalten|--zurueck [--trocken]
                                      Windows-Aufgabe "VibeMind Doku-Loop" pruefen, auf diesen Loop umschalten
                                      (mit Sicherung des lokalen Zustands) oder zurueck (siehe aufgabe.py)
    (tick rotiert vorher logs/doku-loop.log ab 5 MB nach .1, auch waehrend die Aufgabe hinein umleitet)

Exit-Codes: 0 ok (auch uebersprungen), 1 Fehler, 2 Konfigfehler. Ausgabe deutsch, eine Zeile je Schritt; es werden
nie Dateiinhalte oder Schluessel ausgegeben.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from . import dienste, messbaum, messen, umschreiben
from . import konfig as konfig_modul
from . import tick as tk
from .extraktoren.compose import compose_lesen
from .konfig import Konfig, KonfigFehler

AUFGABE = "VibeMind Doku-Loop"
PORTS_JSON = "vibemind.config.json"
LOG_DATEI = "logs/doku-loop.log"  # die Windows-Aufgabe haengt hier mit `>> ... 2>&1` an
LOG_MAX_BYTES = 5 * 1024 * 1024
TICK_WERKZEUGE = ("laufzeit", "budget", "drift", "manifest_bauer", "nach_tick")

Werkzeuge = Mapping[str, Any]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="selbstbild doku", description="Doku-Loop: KETTEN.md aus dem Datenfluss-Graphen")
    p.add_argument("--konfig", help="alte Konfigurationsdatei config/doku_loop.yaml (statt der Regeldatei)")
    # `<befehl> [<wurzel>] [--konfig <pfad>]`: beides hinter dem Befehl; SUPPRESS laesst einen Wert vor dem Befehl stehen
    gemeinsam = argparse.ArgumentParser(add_help=False)
    gemeinsam.add_argument("wurzel", nargs="?", help="Repo-Wurzel mit selbstbild.yaml (Abschnitt doku:)")
    gemeinsam.add_argument("--konfig", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    unter = p.add_subparsers(dest="befehl", required=True)
    t = unter.add_parser("tick", parents=[gemeinsam], help="ein Lauf")
    t.add_argument("--erzwingen", action="store_true", help="ohne Anstoss laufen und alle Spaces bauen")
    t.add_argument("--erlaube-vorgabe", action="store_true",
                   help="tick auch auf der universellen Vorgabe ohne Torwaechter erlauben")
    t.add_argument("--ohne-llm", action="store_true", help="Signale nur in die Pruefliste, kein Judge")
    t.add_argument("--trocken", action="store_true",
                   help="nur messen und zaehlen, nichts schreiben; TickErgebnis als JSON auf stdout")
    unter.add_parser("seed", parents=[gemeinsam], help="Graph-Seed aus KETTEN.md (committet)")
    m = unter.add_parser("messen", parents=[gemeinsam], help="Beleg-Drift und Seed-Deckung messen (nur lesend)")
    m.add_argument("--json", action="store_true")
    m.add_argument("--details", action="store_true", help="jede Fundstelle einzeln")
    r = unter.add_parser("render", parents=[gemeinsam], help="KETTEN.md aus graph.json neu rendern")
    r.add_argument("--schreiben", action="store_true", help="schreiben und committen")
    unter.add_parser("graph-schlank", parents=[gemeinsam], help="docs/graph/graph.json verschlanken (committet)")
    unter.add_parser("dienste-seed", parents=[gemeinsam], help="fehlende Eintraege in config/dienste.yaml ergaenzen")
    unter.add_parser("status", parents=[gemeinsam], help="Stand, Pruefliste, Budget")
    a = unter.add_parser("aufgabe", parents=[gemeinsam], help="Windows-Aufgabe pruefen, umschalten oder zurueck")
    modus = a.add_mutually_exclusive_group(required=True)
    modus.add_argument("--pruefen", action="store_true", help="aktuelle Aktion zeigen und einordnen (nur lesend)")
    modus.add_argument("--umschalten", action="store_true",
                       help="Zustand sichern, alte Aktion merken, Aktion auf diesen Loop setzen")
    modus.add_argument("--zurueck", action="store_true",
                       help="alte Aktion setzen, eigene Rowboat-Seiten archivieren, Sicherung zurueckspielen")
    a.add_argument("--trocken", action="store_true", help="jeden Schritt zeigen, nichts schreiben, nichts setzen")
    a.add_argument("--python", help="Python der neuen Aktion (Standard: das feste .venv-selbstbild in "
                                    "wt-fungus-gate1; nie sys.executable); muss selbstbild ohne PYTHONPATH laden")
    return p


def _kurz(fehler: BaseException) -> str:
    if isinstance(fehler, yaml.YAMLError):
        return type(fehler).__name__  # die Meldung zitiert Dateizeilen
    text = " ".join(str(fehler).split())
    return f"{type(fehler).__name__}: {text[:200]}" if text else type(fehler).__name__


def _jetzt(w: Werkzeuge) -> datetime:
    quelle = w.get("jetzt")
    return quelle() if callable(quelle) else datetime.now().astimezone()


def _sha7(sha: str | None) -> str:
    return sha[:7] if sha else "-"


# ---------- Log-Rotation (Minor 13) ----------

def _strom_auf(pfad: Path) -> int | None:
    """Dateideskriptor von stdout/stderr, der auf `pfad` zeigt (Umleitung der Aufgabe), sonst None."""
    try:
        ziel = os.stat(pfad)
    except OSError:
        return None
    for fd in (1, 2):
        try:
            if os.path.samestat(os.fstat(fd), ziel):
                return fd
        except OSError:
            continue
    return None


def log_rotieren(pfad: Path, *, max_bytes: int = LOG_MAX_BYTES) -> str | None:
    """Ist das Log groesser als `max_bytes`, wird es zu `<name>.1` (eine Generation; die vorige `.1` faellt weg).

    Die Aufgabe leitet mit `>> logs\\doku-loop.log 2>&1` um, die Datei ist also waehrend des Laufs offen und laesst
    sich unter Windows nicht umbenennen. Zeigt stdout/stderr auf sie, wird sie nach `.1` kopiert und ueber den
    eigenen Strom auf 0 gekuerzt (Position an den Anfang, damit keine Luecke entsteht) -> `gekuerzt`; sonst
    umbenannt -> `umbenannt`. Klein genug, fehlend oder Fehler -> None (rotiert wird beim naechsten Lauf)."""
    pfad = Path(pfad)
    try:
        if pfad.stat().st_size <= max_bytes:
            return None
    except OSError:
        return None
    alt = pfad.with_name(pfad.name + ".1")
    fd = _strom_auf(pfad)
    try:
        if fd is None:
            os.replace(pfad, alt)
            return "umbenannt"
        for strom in (sys.stdout, sys.stderr):
            try:
                strom.flush()
            except (AttributeError, OSError, ValueError):
                pass
        shutil.copyfile(pfad, alt)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        return "gekuerzt"
    except OSError:
        return None


# ---------- Befehle ----------

def _tick_trocken(konfig: Konfig, w: Werkzeuge) -> int:
    """Trockenlauf: schreibt nichts (auch kein Log, keine Rotation); Meldungen auf stderr, stdout nur das JSON."""
    extra = {k: w[k] for k in ("laufzeit", "budget", "manifest_bauer") if k in w}
    erg = tk.tick(konfig, jetzt=_jetzt(w), melden=lambda zeile: print(zeile, file=sys.stderr), trocken=True, **extra)
    print(json.dumps(dataclasses.asdict(erg), ensure_ascii=False, indent=1, sort_keys=True))
    return 0


def _tick(konfig: Konfig, args: argparse.Namespace, w: Werkzeuge) -> int:
    if args.trocken:
        return _tick_trocken(konfig, w)
    rotiert = log_rotieren(konfig.repo / LOG_DATEI)  # Minor 13: vor dem Lauf, eine Generation `.1`
    if rotiert:
        print(f"Log: {LOG_DATEI} {rotiert} (vorige Fassung in {LOG_DATEI}.1)")
    extra = {k: w[k] for k in TICK_WERKZEUGE if k in w}
    erg = tk.tick(konfig, jetzt=_jetzt(w), erzwingen=args.erzwingen, ohne_llm=args.ohne_llm, melden=print, **extra)
    print(f"Ergebnis: Lauf {erg.lauf}; Spaces {', '.join(erg.spaces) or '-'}; Signale {erg.signale}; "
          f"LLM {erg.llm_aufrufe}; umgeschrieben {erg.umgeschrieben}; Prüfliste +{erg.pruefliste}; "
          f"Commit {_sha7(erg.commit)}" + (f"; übersprungen: {erg.uebersprungen}" if erg.uebersprungen else "")
          + (f"; Erstbefüllung: {erg.erst_geprueft} geprüft ({erg.erst_aufrufe} LLM), {erg.erst_offen} offen"
             if erg.erst_faellig and not args.ohne_llm else "")
          + (f"; Seiten: {len(erg.erzaehlt_geschrieben)} erzählt ({erg.erzaehlt_aufrufe} LLM), "
             f"{len(erg.erzaehlt_faellig)} fällig, {erg.erzaehlt_offen} nie geschrieben"
             if erg.erzaehlt_faellig or erg.erzaehlt_aufrufe else ""))
    return 0


def _seed(konfig: Konfig, args: argparse.Namespace, w: Werkzeuge) -> int:
    extra = {k: w[k] for k in ("drift", "manifest_bauer") if k in w}
    erg = tk.seed(konfig, jetzt=_jetzt(w), melden=print, **extra)
    print(f"Ergebnis: {erg.verankert} von {erg.hops} Hops verankert; Commit {_sha7(erg.commit)}"
          + (f"; übersprungen: {erg.uebersprungen}" if erg.uebersprungen else ""))
    return 0


def _deckung_zeilen(deckung: Mapping[str, Any] | None) -> list[str]:
    if deckung is None:
        return ["Seed-Deckung: kein docs/graph/graph.json (erst `seed`)"]
    hops, verankert = int(deckung["hops"]), int(deckung["verankert"])
    anteil = round(100 * verankert / hops) if hops else 0
    arten: Mapping[str, int] = deckung["ankerarten"]
    summe = sum(arten.values())
    teile = [f"{a} {n} ({round(100 * n / summe) if summe else 0} %)" for a, n in arten.items()]
    return [f"Seed-Deckung: {verankert} von {hops} Hops verankert ({anteil} %)",
            f"Ankerarten: {', '.join(teile) or '-'}"]


def _messen(konfig: Konfig, args: argparse.Namespace, w: Werkzeuge) -> int:
    # Ruling 50: gemessen wird im Messbaum, KETTEN.md kommt aus dem Arbeitsbaum; Bereitstellen nur unter der Sperre
    with tk.sperre(konfig.marker.parent) as frei:
        if not frei:
            print("Sperre: ein anderer Lauf hält tick.lock – übersprungen (laeuft_schon)")
            return 0
        try:
            mess = messbaum.messbaum_bereitstellen(konfig)
        except (messbaum.MessbaumFehler, OSError) as fehler:
            print(f"Messbaum: {getattr(fehler, 'grund', type(fehler).__name__)} – übersprungen (messbaum)")
            return 0
        befunde = messen.belege_messen(konfig.repo / konfig.ketten_md, mess.repo, konfig=mess)
    deckung = tk.seed_deckung(konfig.repo, konfig)
    if args.json:
        print(json.dumps({"befunde": [dataclasses.asdict(b) for b in befunde], "seed": deckung},
                         ensure_ascii=False, indent=1))
        return 0
    zeilen = messen.bericht(befunde).splitlines()
    print("\n".join(zeilen if args.details else zeilen[:5]))
    print("\n".join(_deckung_zeilen(deckung)))
    return 0


def _render(konfig: Konfig, args: argparse.Namespace, w: Werkzeuge) -> int:
    extra = {k: w[k] for k in ("drift",) if k in w}
    tk.neu_rendern(konfig, jetzt=_jetzt(w), schreiben=args.schreiben, melden=print, **extra)
    return 0  # Exit 0 auch bei uebersprungen (z. B. kein_graph)


def _graph_schlank(konfig: Konfig, args: argparse.Namespace, w: Werkzeuge) -> int:
    extra = {k: w[k] for k in ("drift",) if k in w}
    erg = tk.graph_schlank(konfig, jetzt=_jetzt(w), melden=print, **extra)
    print(f"Ergebnis: {erg.bytes_vorher} -> {erg.bytes_nachher} Bytes; Commit {_sha7(erg.commit)}"
          + (f"; übersprungen: {erg.uebersprungen}" if erg.uebersprungen else ""))
    return 0


def _dienste_yaml(eintraege: Mapping[str, dienste.Dienst]) -> str:
    daten = {n: {"port": d.port, "ort": d.ort, "space": d.space, "env": sorted(d.env)} for n, d in eintraege.items()}
    roh = yaml.safe_dump(daten, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return "".join(f"  {z}\n" for z in roh.splitlines())


def _dienste_seed(konfig: Konfig, args: argparse.Namespace, w: Werkzeuge) -> int:
    """Ruling 15: nur fehlende Eintraege anhaengen (markierter Block am Ende), nie vorhandene ueberschreiben."""
    services, _ = compose_lesen(konfig)
    seed = dienste.seed_aus_compose(services, konfig.repo / PORTS_JSON)
    pfad = konfig.repo / konfig.dienste_yaml
    if not pfad.is_file():
        seed.speichern(pfad)
        print(f"dienste-seed: {konfig.dienste_yaml} neu angelegt, {len(seed.dienste)} Einträge")
        return 0
    alt_bytes = pfad.read_bytes()
    alt_text = alt_bytes.decode("utf-8")
    alt = dienste.DienstRegister.laden(pfad)
    vergeben: set[str] = set()
    neu: dict[str, dienste.Dienst] = {}
    for name, d in seed.dienste.items():
        if name in alt.dienste:
            continue
        env = tuple(e for e in d.env if alt.aufloesen(e) is None and e not in vergeben)
        vergeben.update(env)
        neu[name] = dataclasses.replace(d, env=env)
    if not neu:
        print("dienste-seed: 0 ergänzt (alle Dienste stehen schon im Register)")
        return 0
    nl = "\r\n" if alt_text.count("\r\n") > alt_text.count("\n") // 2 else "\n"
    block = (f"  # dienste-seed {_jetzt(w):%Y-%m-%d}: neu ergaenzt, bitte pruefen\n" + _dienste_yaml(neu))
    text = alt_text + ("" if alt_text.endswith("\n") else nl) + block.replace("\n", nl)
    with tempfile.TemporaryDirectory(prefix="doku-loop-") as tmp:  # erst pruefen, dann schreiben
        probe = Path(tmp) / "dienste.yaml"
        probe.write_bytes(text.encode("utf-8"))
        geprueft = dienste.DienstRegister.laden(probe)
        roh = yaml.safe_load(text)
    if set(roh) != {"dienste"} or set(geprueft.dienste) != set(alt.dienste) | set(neu):
        print("dienste-seed: Register lässt sich nicht sicher ergänzen (weitere Schlüssel nach `dienste:`?)",
              file=sys.stderr)
        return 1
    umschreiben.schreiben(konfig.repo, konfig.dienste_yaml, text, erwartet_alt=alt_bytes)
    print(f"dienste-seed: {len(neu)} ergänzt ({', '.join(sorted(neu))}); bitte prüfen und committen")
    return 0


def _aufgabe_abfragen() -> str:
    try:
        lauf = subprocess.run(["schtasks", "/Query", "/TN", AUFGABE, "/FO", "LIST"], capture_output=True,
                              timeout=20)
    except (OSError, subprocess.TimeoutExpired) as fehler:
        return f"unbekannt ({type(fehler).__name__})"
    if lauf.returncode != 0:
        return "nicht registriert"
    for zeile in lauf.stdout.decode("utf-8", errors="replace").splitlines():
        if zeile.strip().lower().startswith("status:"):
            return f"registriert, {zeile.split(':', 1)[1].strip()}"
    return "registriert"


def _budget_zeile(konfig: Konfig, budget: Any) -> str:
    try:
        if budget is None:
            from . import judge
            budget, grund = judge.budget_laden(konfig)
            if budget is None:
                return f"Budget: nicht lesbar ({grund})"
        stand = budget.tagesstand()
        eintrag = next((a for a in stand.get("agenten", []) if a.get("agent") == konfig.agent), None)
    except Exception as fehler:  # noqa: BLE001 - Status zeigt den Rest trotzdem
        return f"Budget: nicht lesbar ({type(fehler).__name__})"
    if eintrag is None:
        return f"Budget {konfig.agent}: nicht in config/agent_budget.yaml"
    abgelehnt = ", ".join(f"{k} {v}" for k, v in sorted((eintrag.get("abgelehnt") or {}).items())) or "keine"
    zusatz = [t for t, an in (("User aktiv", stand.get("user_aktiv")), ("NOT-AUS", stand.get("not_aus"))) if an]
    return (f"Budget {konfig.agent}: heute {eintrag.get('heute')}/{eintrag.get('grenze')}; abgelehnt: {abgelehnt}"
            + (f"; {', '.join(zusatz)}" if zusatz else ""))


def _status(konfig: Konfig, args: argparse.Namespace, w: Werkzeuge) -> int:
    stand = tk.stand_laden(konfig)
    if stand is None:
        print(f"Stand: noch kein Lauf (lokal: {tk.stand_pfad(konfig).name} fehlt)")
    else:
        print(f"Stand: Lauf {stand.lauf_nr}; letzter Tick {stand.letzter_tick or '-'}; Commit außen "
              f"{_sha7(stand.letzter_commit_aussen)}, Submodul {_sha7(stand.letzter_commit_sub)}; offene Spaces: "
              f"{', '.join(stand.offene_spaces) or 'keine'} ({len(stand.offene_signale)} Signal(e))")
    deckung = tk.seed_deckung(konfig.repo, konfig)
    print("Graph: fehlt (erst `seed`)" if deckung is None
          else f"Graph: {deckung['hops']} Hops, {deckung['verankert']} verankert")
    liste = tk.pruefliste_laden(konfig)
    print(f"Prüfliste: {len(liste['offen'])} offen, {len(liste['struktur'])} Struktur")
    print(f"Rowboat-Notizen: {'an' if konfig.rowboat_notizen_aktiv else 'aus'} (rowboat_notizen_aktiv); "
          f"Zweig: {konfig.zweig or 'jeder'}")
    print(_budget_zeile(konfig, w.get("budget")))
    print(f"Marker: {'gesetzt' if konfig.marker.exists() else 'nicht gesetzt'}; Sperre: "
          f"{tk.sperre_zustand(konfig.marker.parent)}")
    mb = messbaum.messbaum_status(konfig)
    print(f"Messbaum: {'vorhanden' if mb['vorhanden'] else 'fehlt (wird beim nächsten Lauf angelegt)'}; "
          f"{mb['worktrees']} Worktree(s); Commit außen {_sha7(mb['commit_aussen'])}; nicht messbar: "
          f"{', '.join(mb['nicht_messbar']) or 'keine'}"
          + (f"; Entfernen offen: {', '.join(mb['entfernen_offen'])}" if mb["entfernen_offen"] else ""))
    abfragen: Callable[[], str] = w.get("aufgabe_abfragen") or _aufgabe_abfragen
    print(f"Aufgabe: {abfragen()}")
    return 0


def _aufgabe(konfig: Konfig, args: argparse.Namespace, w: Werkzeuge) -> int:
    """Task 10: Umschalten der Windows-Aufgabe mit Rueckweg (`aufgabe.py`). Backend, Import-Pruefung und Uhr sind
    fuer Tests injizierbar (`aufgabe_backend`, `import_pruefen`, `jetzt`); das Python kommt nur aus `--python`."""
    from . import aufgabe as ag

    backend = w.get("aufgabe_backend") or ag.PowerShellBackend()
    if args.pruefen:
        return ag.pruefen(konfig, backend=backend)
    jetzt = _jetzt(w).replace(tzinfo=None)
    if args.umschalten:
        return ag.umschalten(konfig, backend=backend, python=args.python, trocken=args.trocken,
                             jetzt=jetzt, import_pruefen=w.get("import_pruefen") or ag.import_pruefen)
    return ag.zurueck(konfig, backend=backend, trocken=args.trocken, jetzt=jetzt)


BEFEHLE: dict[str, Callable[[Konfig, argparse.Namespace, Werkzeuge], int]] = {
    "tick": _tick, "seed": _seed, "messen": _messen, "render": _render, "graph-schlank": _graph_schlank,
    "dienste-seed": _dienste_seed, "status": _status, "aufgabe": _aufgabe,
}


def _konfig_aufloesen(args: argparse.Namespace) -> tuple[Konfig, str]:
    """Quelle der Konfiguration: `--konfig` (alter Lader), sonst `<wurzel>`, sonst `SELBSTBILD_VIBEMIND`
    (jeweils die Regeldatei). Kein stiller Rueckfall auf den Arbeitsordner."""
    if args.konfig:
        return konfig_modul.laden(Path(args.konfig)), f"alte Datei {Path(args.konfig).resolve()}"
    env = os.environ.get("SELBSTBILD_VIBEMIND")
    wurzel = Path(args.wurzel) if args.wurzel else Path(env) if env else None
    if wurzel is not None:
        konfig, art = konfig_modul.aus_regeln_mit_quelle(wurzel)
        herkunft = "" if args.wurzel else " (SELBSTBILD_VIBEMIND)"
        return konfig, f"{art} in {wurzel.resolve()}{herkunft}"
    raise KonfigFehler("keine Konfigurationsquelle: <wurzel> angeben, --konfig <pfad> setzen oder "
                       "SELBSTBILD_VIBEMIND auf die Repo-Wurzel zeigen lassen")


def main(argv: list[str] | None = None, *, werkzeuge: Werkzeuge | None = None) -> int:
    args = _parser().parse_args(argv)
    for strom in (sys.stdout, sys.stderr):
        try:
            # Log-Umleitung der Aufgabe: UTF-8 statt Codepage, nie an Umlauten scheitern
            strom.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    w = dict(werkzeuge or {})
    try:
        konfig, quelle = _konfig_aufloesen(args)
        print(f"Konfig: {quelle}; Repo {konfig.repo}", file=sys.stderr)
        if (args.befehl == "tick" and konfig_modul.ist_vorgabe(quelle) and not konfig.torwaechter
                and not args.erlaube_vorgabe and not args.trocken):  # der Trockenlauf schreibt nichts
            raise KonfigFehler("tick auf der universellen Vorgabe ohne Torwaechter ist gesperrt "
                               "(falsche Wurzel? sonst mit --erlaube-vorgabe)")
        if args.befehl == "aufgabe" and args.umschalten and not quelle.startswith(konfig_modul.QUELLE_DOKU):
            # die Aufgabe ruft danach `doku tick <wurzel>`: ohne Abschnitt doku: liefe er auf der Vorgabe
            raise KonfigFehler("umschalten braucht <wurzel> mit selbstbild.yaml und Abschnitt doku: (kein --konfig)")
    except (KonfigFehler, OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as fehler:
        print(f"Konfigfehler: {_kurz(fehler)}", file=sys.stderr)
        return 2
    try:
        return BEFEHLE[args.befehl](konfig, args, w)
    except KonfigFehler as fehler:
        print(f"Konfigfehler: {_kurz(fehler)}", file=sys.stderr)
        return 2
    except Exception as fehler:  # noqa: BLE001 - eine Zeile statt Traceback im Aufgaben-Log
        print(f"Fehler: {_kurz(fehler)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
