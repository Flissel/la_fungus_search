"""Rowboat-Seiten ohne LLM (Stufe 4, Task 8, D11): "Neuen Space erstellen mit Claude Code" und `Zustand`.

Beide Seiten sind deterministisch (keine Zeitstempel im Text) und tragen im Kopf `erzeugt_von: selbstbild` und
`grundlage: <digest>` (Digest ihrer Eingaben). Geschrieben werden sie von `rowboat.notizen_schreiben` (Parameter
`zusatz`) -- also durch denselben Manifest-Schreibschutz wie die anderen Notizen: Eine von Hand geaenderte Seite
wird nie ueberschrieben, es entsteht der Pruefpunkt `notiz_von_hand`.

- Mitbauen: gleiche Quelle wie der Skill (`skill.skill_neuer_space`, Spec 6.8): der Skill-Text ohne Frontmatter und
  ohne seine Titelzeile, mit einer Einleitung fuer Menschen davor. `stand_unter=None` heisst: der Skill sucht den
  vermessenen Stand selbst (`vermessen.stand_finden`), genau wie `python -m selbstbild skill`; dann erscheinen die
  Vorbilder, sonst entfaellt der Abschnitt.
- Zustand: Ketten mit Hops im Zustand `bricht`/`aus` (Graph), offene Pruefpunkte nach Art (Pruefliste) und, wenn
  die Stufe-2-Analyse im Stand liegt (`analyse.sqlite`, nur lesend), `sicher_tot`, `wahrscheinlich_tot` und die
  fuenf obersten Hotspots.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from .graph import Graph, Kante
from . import rowboat
from .rowboat import INDEX_NAME, Notiz, sicherer_name

MARKER = "selbstbild"
MITBAUEN_TITEL = "Neuen Space erstellen mit Claude Code"
MITBAUEN_REL = f"Mitbauen/{MITBAUEN_TITEL}.md"
ZUSTAND_TITEL = "Zustand"
ZUSTAND_REL = f"{ZUSTAND_TITEL}.md"
TOP_HOTSPOTS = 5

__all__ = ["MITBAUEN_REL", "ZUSTAND_REL", "seite_mitbauen", "seite_zustand", "analyse_zusammenfassung_lesen",
           "seiten", "index_links", "voraussichtlich_geschrieben"]


def _digest(inhalt: object) -> str:
    roh = inhalt if isinstance(inhalt, str) else json.dumps(inhalt, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(roh.encode("utf-8")).hexdigest()[:16]


def _kopf(grundlage: str) -> str:
    return f"---\nerzeugt_von: {MARKER}\ngrundlage: {json.dumps(grundlage)}\n---\n"


def _ohne_frontmatter(text: str) -> str:
    zeilen = text.replace("\r\n", "\n").split("\n")
    if zeilen and zeilen[0].strip() == "---":
        for j in range(1, len(zeilen)):
            if zeilen[j].strip() == "---":
                zeilen = zeilen[j + 1:]
                break
    return "\n".join(zeilen).lstrip("\n")


def _ohne_titel(text: str) -> str:
    zeilen = text.split("\n")
    if zeilen and zeilen[0].startswith("# "):
        zeilen = zeilen[1:]
    return "\n".join(zeilen).lstrip("\n")


# ---------- Mitbauen ----------

def seite_mitbauen(wurzel: Path, stand_unter: Path | None, *, ordner: str = "Doku") -> str:
    """Markdown der Seite `Neuen Space erstellen mit Claude Code` (mit Kopf `erzeugt_von`/`grundlage`)."""
    from selbstbild.skill import skill_neuer_space

    skill = skill_neuer_space(Path(wurzel), stand_unter)
    inhalt = _ohne_titel(_ohne_frontmatter(skill)).rstrip("\n")
    einleitung = [
        f"# {MITBAUEN_TITEL}", "",
        f"Zurück: [[{ordner}/{INDEX_NAME}]]", "",
        "## So fragst du", "",
        "Öffne Claude Code im VibeMind-Repo und schreib in normalen Worten, was du brauchst, zum Beispiel: "
        "„bau mir einen neuen Space für X“. Claude Code lädt dann den Skill `neuer-space` und folgt dem Ablauf "
        "weiter unten: Es klärt mit dir, welches Vorbild passt, legt den Space am richtigen Ort mit den "
        "Pflichtteilen an und schreibt Tests.", "",
        "## Was der Wächter blockiert und warum", "",
        "Ein Schreib-Wächter (Hook vor jedem Schreiben von Claude Code) prüft jeden Pfad gegen die Strukturregeln "
        "der Repos. Er blockiert Pfade, an denen ein Space nicht entstehen darf (Abschnitt „Nie hier anlegen“), "
        "damit ein neuer Space nicht versehentlich an der falschen Stelle landet und später niemand ihn findet. "
        "Wird ein Schreibvorgang blockiert, nennt der Wächter die Regel; dann den richtigen Ort aus dem Abschnitt "
        "„Wohin“ nehmen.", "",
        "## Wo der Skill liegt", "",
        "Der Skill `neuer-space` ist eine erzeugte Datei `SKILL.md` im Skill-Ordner von Claude Code. Er wird mit "
        "`python -m selbstbild skill neuer-space <wurzel> <ziel>/SKILL.md --ueberschreiben` aus denselben Regeln "
        "und demselben vermessenen Stand erzeugt wie der Text unten: Diese Seite und der Skill haben eine Quelle "
        "und sagen deshalb dasselbe.", "",
        "## Der Skill im Wortlaut", "",
    ]
    # Ueberschriften des Skills eine Ebene tiefer waere eine Aenderung des Wortlauts -> unveraendert uebernehmen
    return _kopf(_digest(skill)) + "\n".join(einleitung) + "\n" + inhalt + "\n"


# ---------- Zustand ----------

def _hop_zustand(e: Kante) -> tuple[str, bool]:
    z = e.attribute.get("zustand")
    if not isinstance(z, dict):
        return "", False
    g = z.get("gemessen")
    wert = str(g.get("wert")) if isinstance(g, dict) and g.get("wert") else ""
    return wert, bool(z.get("aus"))


def _natur(text: str) -> list[object]:
    import re

    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", text)]


def _ketten_befund(graph: Graph) -> tuple[dict[str, dict[str, list[str]]], dict[str, int]]:
    """({Ketten-ID: {"bricht": [Hop-Nr], "aus": [Hop-Nr]}} nur mit Befund, Zahlen: gesamt/laeuft/ohne_probe/defekt)."""
    hops: dict[str, list[Kante]] = {}
    for e in graph.kanten.values():
        if e.art == "hop":
            hops.setdefault(f"kette:{e.attribute.get('kette')}", []).append(e)
    befund: dict[str, dict[str, list[str]]] = {}
    zahlen = {"gesamt": 0, "defekt": 0, "laeuft": 0, "ohne_probe": 0}
    for kid in sorted((i for i, k in graph.knoten.items() if k.typ == "kette"), key=_natur):
        zahlen["gesamt"] += 1
        eintrag: dict[str, list[str]] = {"bricht": [], "aus": []}
        gemessen_laeuft = 0
        liste = sorted(hops.get(kid, []), key=lambda e: _natur(str(e.attribute.get("nr", ""))))
        for e in liste:
            wert, aus = _hop_zustand(e)
            nr = str(e.attribute.get("nr", ""))
            if wert == "bricht":
                eintrag["bricht"].append(nr)
            if aus:
                eintrag["aus"].append(nr)
            if wert == "laeuft":
                gemessen_laeuft += 1
        if eintrag["bricht"] or eintrag["aus"]:
            befund[kid] = eintrag
            zahlen["defekt"] += 1
        elif liste and gemessen_laeuft == len(liste):
            zahlen["laeuft"] += 1
        else:
            zahlen["ohne_probe"] += 1
    return befund, zahlen


def _nrs(nrs: Sequence[str]) -> str:
    return ", ".join(nrs)


def seite_zustand(graph: Graph, analyse_zusammenfassung: Mapping[str, Any] | None,
                  pruefliste: Mapping[str, Sequence[Mapping[str, object]]] | Sequence[Mapping[str, object]],
                  *, ordner: str = "Doku") -> str:
    """Markdown der Seite `Zustand`. `pruefliste`: das Ergebnis von `pruefliste_laden` (Schluessel `offen`) oder
    direkt die Liste offener Punkte. `analyse_zusammenfassung`: siehe `analyse_zusammenfassung_lesen`; None laesst
    den Analyse-Abschnitt weg."""
    offen = pruefliste.get("offen", []) if isinstance(pruefliste, Mapping) else pruefliste
    befund, zahlen = _ketten_befund(graph)
    arten = Counter(str(p.get("art") or "unbekannt") for p in offen)
    namen = {kid: graph.knoten[kid] for kid in befund}

    z: list[str] = [f"# {ZUSTAND_TITEL}", "", f"Zurück: [[{ordner}/{INDEX_NAME}]]", "",
                    "Erzeugt aus dem Datenfluss-Graph, der Prüfliste und (falls vorhanden) der Analyse. "
                    "Von Hand ändern lohnt nicht: Die Seite wird aus diesen Quellen neu geschrieben, außer du hast "
                    "sie geändert; dann lässt der Loop sie stehen.", "",
                    "## Ketten", "",
                    f"- {zahlen['gesamt']} Ketten insgesamt: {zahlen['laeuft']} laufen (alle Hops gemessen), "
                    f"{zahlen['defekt']} brechen oder sind aus, {zahlen['ohne_probe']} ohne vollständige Messung.", ""]
    if befund:
        z += ["| Kette | Bricht bei Hop | Aus bei Hop |", "|---|---|---|"]
        for kid, b in befund.items():
            link = f"[[{ordner}/Ketten/{sicherer_name(kid.split(':', 1)[1])}]]"
            z.append(f"| {link} {namen[kid].name} | {_nrs(b['bricht']) or '—'} | {_nrs(b['aus']) or '—'} |")
    else:
        z.append("Keine Kette bricht, keine ist ausgeschaltet.")
    z += ["", f"## Offene Prüfpunkte ({sum(arten.values())})", ""]
    if arten:
        z += ["| Art | Anzahl |", "|---|---|"]
        z += [f"| `{art}` | {n} |" for art, n in sorted(arten.items(), key=lambda kv: (-kv[1], kv[0]))]
        z += ["", f"Alle Punkte: [[{ordner}/Prüfliste]]"]
    else:
        z.append("Keine offenen Punkte.")
    if analyse_zusammenfassung is not None:
        tot = analyse_zusammenfassung.get("tot") or {}
        z += ["", "## Toter Code und Hotspots", "",
              f"- `sicher_tot`: {int(tot.get('sicher_tot', 0))}",
              f"- `wahrscheinlich_tot`: {int(tot.get('wahrscheinlich_tot', 0))}", "",
              f"Die {TOP_HOTSPOTS} größten Hotspots (Dateien, an denen Änderungen am meisten Wirkung haben):", ""]
        spots = list(analyse_zusammenfassung.get("hotspots_top") or [])[:TOP_HOTSPOTS]
        if spots:
            z += [f"{i}. `{s.get('datei')}` ({float(s.get('wert', 0.0)):.4f})" for i, s in enumerate(spots, 1)]
        else:
            z.append("Keine Hotspots in der Analyse.")
    grundlage = _digest({"ketten": befund, "zahlen": zahlen, "arten": dict(sorted(arten.items())),
                         "analyse": None if analyse_zusammenfassung is None else {
                             "tot": dict(sorted((analyse_zusammenfassung.get("tot") or {}).items())),
                             "hotspots": [[s.get("datei"), s.get("wert")] for s in
                                          list(analyse_zusammenfassung.get("hotspots_top") or [])[:TOP_HOTSPOTS]]}})
    return _kopf(grundlage) + "\n".join(z) + "\n"


def analyse_zusammenfassung_lesen(stand: Path | None) -> dict[str, Any] | None:
    """Zusammenfassung der neuesten Analyse im Stand-Ordner (`<stand>/analyse.sqlite`, nur `mode=ro`) im Format
    `{"commit", "tot": {stufe: n}, "hotspots_top": [{"datei", "wert"}]}`. Fehlt der Stand, die Datei, die Tabelle
    oder ist sie unlesbar: None (der Abschnitt entfaellt, nichts bricht)."""
    if stand is None:
        return None
    db = Path(stand) / "analyse.sqlite"
    try:
        if not db.is_file():
            return None
        with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)) as c:
            zeile = c.execute("SELECT daten FROM lauf ORDER BY rowid DESC LIMIT 1").fetchone()
        if zeile is None:
            return None
        d = json.loads(bytes(zeile[0]).decode("utf-8"))
        tot = {"sicher_tot": 0, "wahrscheinlich_tot": 0}
        for t in d.get("tot") or []:
            if t.get("stufe") in tot:
                tot[t["stufe"]] += 1
        spots = [{"datei": str(h["datei"]), "wert": float(h["wert"])} for h in (d.get("hotspots") or [])[:TOP_HOTSPOTS]]
        return {"commit": str(d.get("commit", "")), "tot": tot, "hotspots_top": spots}
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
        return None


# ---------- Zusammenstellen ----------

def seiten(graph: Graph, wurzel: Path, pruefliste: Mapping[str, Sequence[Mapping[str, object]]],
           stand_unter: Path | None = None, *, ordner: str = "Doku") -> list[Notiz]:
    """Beide Seiten als Notizen fuer `notizen_schreiben(zusatz=...)`.

    Stand-Ordner der Analyse: der des Skills (`stand_unter` oder `vermessen.stand_finden`); fehlt er, entfaellt
    der Analyse-Abschnitt."""
    stand: Path | None = None
    try:
        from selbstbild import skill
        from selbstbild.vermessen import stand_finden

        stand = skill._stand_ordner(Path(wurzel), stand_unter)
        if stand is None and stand_unter is None:
            g = stand_finden(Path(wurzel))
            stand = g.parent if g is not None else None
    except Exception:  # noqa: BLE001 - ohne Stand nur ohne Analyse
        stand = None
    return [Notiz(MITBAUEN_REL, seite_mitbauen(wurzel, stand_unter, ordner=ordner)),
            Notiz(ZUSTAND_REL, seite_zustand(graph, analyse_zusammenfassung_lesen(stand), pruefliste, ordner=ordner))]


def index_links(ordner: str = "Doku") -> list[str]:
    return [f"[[{ordner}/Mitbauen/{MITBAUEN_TITEL}]] – so baust du mit Claude Code einen neuen Space",
            f"[[{ordner}/{ZUSTAND_TITEL}]] – was lebt, was tot ist, was bricht"]


def voraussichtlich_geschrieben(notizen: Sequence[Notiz], wissen: Path, ordner: str,
                                manifest_pfad: Path | None) -> tuple[list[str], list[str]]:
    """Trockenlauf: (Seiten, die geschrieben wuerden; Seiten, die von Hand geaendert sind und stehen blieben).
    Liest nur (Manifest und Dateien); schreibt nichts. Kaputtes Manifest oder umgeleiteter Ordner -> ([], [])."""
    basis = rowboat._unter(Path(wissen), ordner)
    geladen = rowboat._manifest_laden(manifest_pfad)
    if geladen == rowboat.KAPUTT or rowboat._umgeleitet(basis):
        return [], []
    manifest = {} if isinstance(geladen, str) else dict(geladen)
    s = rowboat.Schreiber(basis, manifest, isinstance(geladen, str), ())
    schreiben: list[str] = []
    hand: list[str] = []
    for n in notizen:
        e = s.entscheiden(n.rel, n.text)
        if e == "schreiben":
            schreiben.append(n.rel)
        elif e == "hand":
            hand.append(n.rel)
    return schreiben, hand
