"""Erzeugt die Claude-Code-Skill "neuer-space" aus den Regeln der Repos und dem vermessenen Stand (Stufe 3, T10).

Datengetrieben und deterministisch (keine Zeitstempel, alles sortiert):
- Ort, Pflichtteile, Hinweise: Strukturregeln aller `selbstbild.yaml` im Arbeitsbaum unter der Wurzel, deren `art`
  "space" enthaelt und deren `ort` nicht die Repo-Wurzel ist. Pfade sind relativ zur Wurzel (Regeln von
  Unter-Repos bekommen deren Ordner vorangestellt).
- "Nie hier anlegen": `verboten` der Regeln mit Wurzel-`ort`.
- Vorbilder: Ordner im Suchindex des Stands, die auf `ort` passen, ohne Namen mit `_` am Anfang und ohne
  `__pycache__`, mit allen Pflichtteilen. Rang = SUMME der `lebendigkeit` aller Dateien darunter. `lebendigkeit` ist nach
  oben nicht begrenzt (zerfallsgewichtete Summe der Aenderungen). Die Summe misst, wie stark der Space als Ganzes
  veraendert wird; sie bevorzugt dadurch grosse Spaces, ein Maximum je Datei wuerde das nicht abbilden. Die ersten
  drei, Gleichstand nach Name. Ohne Stand oder ohne Treffer entfaellt der Abschnitt.
"""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from ._glob import glob_regex
from .hooks import REGELDATEI
from . import struktur
from .struktur import _UEBERSPRUNGEN, _flags, _regeln_von

__all__ = ["skill_neuer_space"]

_WURZEL_ORTE = ("", ".")
_MAX_TIEFE = 4  # so tief unter der Wurzel wird nach weiteren `selbstbild.yaml` gesucht (Submodule)
_ANZAHL_VORBILDER = 3
_WERKZEUGE = (
    ("selbstbild_suchen", "Stichwortsuche ueber Code und Doku, um aehnliche Stellen zu finden"),
    ("selbstbild_nachbarn", "Aufrufer, Aufgerufene und Importe eines Knotens"),
    ("selbstbild_bereich", "zu welchem Bereich (Modul-Cluster) eine Datei gehoert"),
    ("selbstbild_regeln", "welche Strukturregeln fuer einen Pfad gelten"),
    ("selbstbild_lebendigkeit", "wie lebendig eine Datei ist (zerfallsgewichtete Aenderungen, hoeher = lebendiger)"),
    ("selbstbild_tot", "Code, den nichts mehr erreicht"),
    ("selbstbild_doku", "Doku-Abschnitte, die zu einem Knoten gehoeren"),
)


def _ort_norm(ort: object) -> str:
    return str(ort).replace("\\", "/").strip().removeprefix("./").rstrip("/")


def _verbinden(vorn: str, hinten: str) -> str:
    return f"{vorn}/{hinten}" if vorn else hinten


def _regeldateien(wurzel: Path) -> list[Path]:
    """Ordner mit `selbstbild.yaml` (Arbeitsbaum), Wurzel zuerst, dann sortiert."""
    gefunden: list[Path] = []
    for pfad, unterordner, dateien in os.walk(wurzel):
        p = Path(pfad)
        tiefe = len(p.relative_to(wurzel).parts)
        unterordner[:] = sorted(d for d in unterordner if d not in _UEBERSPRUNGEN and not d.startswith("."))
        if tiefe >= _MAX_TIEFE:
            unterordner[:] = []
        if REGELDATEI in dateien:
            gefunden.append(p)
    return sorted(gefunden, key=lambda p: (len(p.relative_to(wurzel).parts), p.relative_to(wurzel).as_posix()))


def _regeln_sammeln(wurzel: Path) -> tuple[list[dict], list[dict], list[str]]:
    """(Space-Regeln mit Wurzel-Ort, Wurzel-Verbote je Regel samt Repo, Hinweise)."""
    space: list[dict] = []
    verbote: list[dict] = []
    hinweise: set[str] = set()
    for ordner in _regeldateien(wurzel):
        vorn = ordner.relative_to(wurzel).as_posix()
        vorn = "" if vorn == "." else vorn
        for r in _regeln_von(ordner):
            if "space" not in str(r.get("art", "")).lower():
                continue
            ort = _ort_norm(r.get("ort", ""))
            if ort in _WURZEL_ORTE:
                muster = sorted({str(v) for v in r.get("verboten", [])})
                if muster:  # Muster bleiben relativ zur Wurzel ihres Repos
                    verbote.append({"repo": vorn or wurzel.name, "art": str(r.get("art", "")), "muster": muster,
                                    "hinweis": str(r["hinweis"]) if r.get("hinweis") else None})
                continue
            space.append({"ort": _verbinden(vorn, ort), "pflicht": [str(t) for t in r.get("pflicht", [])]})
            if r.get("hinweis"):
                hinweise.add(str(r["hinweis"]))
    verbote.sort(key=lambda v: (v["repo"], v["art"]))
    return space, verbote, sorted(hinweise)


def _submodule(wurzel: Path, orte: list[str]) -> list[str]:
    """Submodul-Pfade aus `.gitmodules`, unter denen mindestens ein Space-`ort` liegt."""
    try:
        text = (wurzel / ".gitmodules").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    pfade = {m.group(1).strip().replace("\\", "/").strip("/") for m in re.finditer(r"^\s*path\s*=\s*(.+)$", text, re.M)}
    return sorted(p for p in pfade if p and any(o == p or o.startswith(p + "/") for o in orte))


def _schritt_2(orte: list[str], hinweise: list[str], submodule: list[str]) -> str:
    satz = "2. Den Space am oben genannten Ort anlegen, mit den oben genannten Pflichtteilen (falls es welche gibt)."
    if hinweise:
        satz += " Hinweis aus den Regeln: " + " / ".join(hinweise) + "."
    for sub in submodule:
        satz += (f" Der Ort liegt im Submodul `{sub}`: dort im Submodul committen und danach den Submodul-Pin"
                 " im aeusseren Repo aktualisieren.")
    return satz


def _stand_ordner(wurzel: Path, stand_unter: Path | None) -> Path | None:
    from .speicher import pfad_kennung
    from .suche.index import index_pfad
    from .vermessen import stand_finden

    if stand_unter is not None:
        s = Path(stand_unter)
        if index_pfad(s).is_file():
            return s
        if s.is_dir():
            treffer = sorted(o for o in s.glob(f"*-{pfad_kennung(wurzel)}") if index_pfad(o).is_file())
            return treffer[-1] if treffer else None
        return None
    graph = stand_finden(Path(wurzel))
    if graph is not None and index_pfad(graph.parent).is_file():
        return graph.parent
    return None


def _dateien_im_stand(stand: Path) -> list[tuple[str, float]]:
    """(Wurzel-relativer Pfad, Lebendigkeit) aller Dateien im Index; `[]` bei unlesbarem Index."""
    from .suche.index import index_pfad
    from .suche.messung import wurzel_name

    try:
        with closing(sqlite3.connect(f"{index_pfad(stand).resolve().as_uri()}?mode=ro", uri=True)) as c:
            zeilen = c.execute("SELECT repo, pfad, lebendigkeit FROM eintrag"
                               " WHERE art='datei' AND repo IS NOT NULL AND pfad IS NOT NULL").fetchall()
    except sqlite3.DatabaseError:
        return []
    repos = sorted({r for r, _, _ in zeilen}, key=lambda r: (len(r), r))
    if not repos:
        return []
    kopf = wurzel_name(stand) or repos[0]
    aus: list[tuple[str, float]] = []
    for repo, pfad, leb in zeilen:
        if repo == kopf:
            vorn = ""
        elif repo.startswith(kopf + "/"):
            vorn = repo[len(kopf) + 1:]
        else:
            continue
        aus.append((_verbinden(vorn, str(pfad)), float(leb or 0.0)))
    return aus


def _hat_teil(dateien: list[str], teil: str) -> bool:
    teil = teil.replace("\\", "/").removeprefix("./").rstrip("/")
    if not any(c in teil for c in "*?"):
        if struktur.GROSSKLEIN_EGAL:
            teil = teil.lower()
            dateien = [d.lower() for d in dateien]
        return any(d == teil or d.startswith(teil + "/") for d in dateien)
    rx = re.compile(glob_regex(teil).pattern, _flags())
    for d in dateien:
        teile = d.split("/")
        if any(rx.match("/".join(teile[:i])) for i in range(1, len(teile) + 1)):
            return True
    return False


def _vorbilder(stand: Path, space: list[dict]) -> list[tuple[str, float, int]]:
    """(Ordner, Lebendigkeits-Summe, Dateien) der besten Ordner, die auf einen Space-`ort` passen und alle Teile haben."""
    dateien = _dateien_im_stand(stand)
    if not dateien:
        return []
    ergebnis: dict[str, tuple[float, int]] = {}
    for regel in space:
        rx = re.compile(glob_regex(regel["ort"]).pattern, _flags())
        ordner: dict[str, list[tuple[str, float]]] = {}
        for pfad, leb in dateien:
            teile = pfad.split("/")
            for i in range(1, len(teile)):  # echte Vorfahren-Ordner
                kopf = "/".join(teile[:i])
                if rx.match(kopf):
                    ordner.setdefault(kopf, []).append(("/".join(teile[i:]), leb))
        for kopf, inhalt in ordner.items():
            name = kopf.rsplit("/", 1)[-1]
            if name.startswith("_") or name == "__pycache__":
                continue
            rel = [p for p, _ in inhalt]
            if all(_hat_teil(rel, t) for t in regel["pflicht"]):
                ergebnis[kopf + "/"] = (round(sum(l for _, l in inhalt), 6), len(inhalt))
    rang = sorted(ergebnis.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [(k, v[0], v[1]) for k, v in rang[:_ANZAHL_VORBILDER]]


def skill_neuer_space(wurzel: Path, stand_unter: Path | None = None) -> str:
    """Markdown-Inhalt der Skill-Datei `neuer-space` (UTF-8, LF). Ohne vermessenen Stand entfaellt der Abschnitt Vorbilder."""
    wurzel = Path(wurzel)
    space, verbote, hinweise = _regeln_sammeln(wurzel)
    orte = sorted({r["ort"] for r in space})
    pflicht = sorted({t for r in space for t in r["pflicht"]})

    z: list[str] = [
        "---",
        "name: neuer-space",
        "description: Legt einen neuen VibeMind-Space am richtigen Ort mit den Pflichtteilen an. Verwenden bei"
        " \"neuen Space anlegen\", \"Space erstellen\", \"new space\" oder \"create a space\".",
        "---",
        "",
        "# Neuen Space anlegen",
        "",
        "Diese Datei wird aus den `selbstbild.yaml`-Regeln der Repos und dem Suchindex des vermessenen Stands erzeugt"
        " (`python -m selbstbild skill neuer-space <wurzel> <ziel.md>`). Nicht von Hand aendern.",
        "",
        "## Wohin",
        "",
    ]
    if orte:
        z += [f"- Ort: `{o}/`" for o in orte]
        z += [f"- Hinweis: {h}" for h in hinweise]
    else:
        z.append("- Keine Space-Regel gefunden. Vor dem Anlegen mit dem User klaeren, wohin der Space gehoert.")
    z += ["", "## Pflichtteile", ""]
    if pflicht:
        z += [f"- `{t}`" for t in pflicht]
    else:
        z.append("Es gibt keine festen Pflichtdateien. Dem Aufbau der Vorbilder folgen (soweit vorhanden)"
                 " und mit dem User klaeren, was der neue Space braucht.")
    if verbote:
        z += ["", "## Nie hier anlegen", ""]
        for v in verbote:
            z += [f"Regel `{v['art']}`: Die Muster gelten relativ zur Wurzel des Repos `{v['repo']}`, aus dessen"
                  " Regel sie stammen, nicht fuer jeden gleichnamigen Ordner im Baum.", ""]
            z += [f"- `{m}`" for m in v["muster"]]
            if v["hinweis"]:
                z += ["", f"Hinweis: {v['hinweis']}"]
            z.append("")
        z += ["Der Schreib-Waechter blockiert diese Pfade."]

    stand = _stand_ordner(wurzel, stand_unter)
    vorbilder = _vorbilder(stand, space) if stand is not None and space else []
    if vorbilder:
        z += ["", "## Vorbilder", "",
              f"Die {len(vorbilder)} lebendigsten vorhandenen Spaces mit allen Pflichtteilen, nach der Summe der"
              " Lebendigkeit ihrer Dateien (Stand der letzten Vermessung):", ""]
        z += [f"- `{ordner}` (Lebendigkeit {summe:.2f}, {n} Dateien)" for ordner, summe, n in vorbilder]
        z += ["", "Aufbau und Namensgebung dieser Ordner als Muster nehmen."]

    z += [
        "", "## Ablauf", "",
        "1. Mit dem User klaeren: Welches Vorbild passt, und welche Faehigkeiten soll der Space bekommen?",
        _schritt_2(orte, hinweise, _submodule(wurzel, orte)),
        "3. Tests schreiben und ausfuehren.",
        "4. Doku nicht von Hand schreiben: Sie entsteht durch den Doku-Loop.",
        "", "## Hilfen (MCP-Server `selbstbild`)", "",
    ]
    z += [f"- `{name}`: {text}" for name, text in _WERKZEUGE]
    return "\n".join(z) + "\n"
