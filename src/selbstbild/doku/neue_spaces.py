"""Doku-Datei fuer neue Spaces (Stufe 4, Task 6, D9). Deterministisch, kein LLM.

Erkennung (im Messbaum, also nur committeter Stand): die Ordner, die auf den `ort` einer Strukturregel mit `art`
"space" passen (gelesen wie der Skill, `skill._regeln_sammeln`; in VibeMind `vibemind-os/spaces/*/`). Nicht dazu
gehoeren Ordner mit `_` am Anfang, `__pycache__`, Ordner, die selbst ein Unter-Repo sind oder eines enthalten
(Gitlinks des Messbaums, `skill._mit_repo`), und leere Ordner (ein nicht initialisiertes Submodul im Messbaum). Je
Space-Schluessel (`namen.space_schluessel`) zaehlt ein Ordner. "Hat Doku" heisst: die Zuordnung aus Task 5
(`space_dokumente_zuordnen`: `doku.space_dokumente`, sonst `docs/spaces/<NNN>_<name>.md`) findet eine vorhandene Datei.

Fuer einen Space ohne Doku entsteht `docs/spaces/<NNN>_<schluessel>.md` (NNN = hoechste vorhandene Nummer + 1, mehrere
neue nach Name fortlaufend). Die Datei besteht aus Kopfzeile, Hinweis, Abschnitts-Ueberschriften und markierten
Bloecken; aller Inhalt steht in Bloecken (`stand`, `steckbrief`, `ereignisse`, `schnittstellen`, `belege` je Space).
Gerendert wird mit dem allgemeinen Block-Kern (`render_ketten.bloecke_rendern`): Hand-Edit-Schutz und Render-Hashes
gelten wie fuer jede andere Doku, spaetere Laeufe rendern die Bloecke neu. Der Block `steckbrief <space>` kennzeichnet
eine generierte Doku.

Der Index `docs/spaces/README.md` bekommt am Ende einen Block `neue-spaces` mit einer Tabellenzeile je generierter Doku
(der Drift-Waechter verlangt, dass der Index genau die vorhandenen `NNN_name.md` verlinkt). Fehlt der Block, wird er
angehaengt; Hand-Inhalt ausserhalb bleibt byteweise. Sonst ist die README nie Schreibziel.

Verschwindet der Ordner wieder, bleibt die Datei (Loeschen macht nur ein Mensch): ihre Bloecke bleiben stehen, dazu der
Pruefpunkt `space_doku_ohne_space`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .extraktoren.ereignisse import Ereignis, space_dokumente_zuordnen
from .graph import Graph
from .konfig import Konfig
from .namen import space_schluessel
from .render_ketten import (
    MARKER_BEGIN, MARKER_END, RenderErgebnis, _code, _scannen, _zeile, block_hash, bloecke_rendern, zeilenende,
)
from .schreibvorgang import NEUE_SPACES_README, doku_dateien
from .typen import Pruefpunkt

ORDNER = "docs/spaces"
README = NEUE_SPACES_README
README_BLOCK = "neue-spaces"
README_UEBERSCHRIFT = "## Neue Spaces (generiert)"
HINWEIS = "<!-- Von Hand ergänzen ausserhalb der graph-Blöcke -->"
STAND, STECKBRIEF, EREIGNISSE, SCHNITTSTELLEN, BELEGE = "stand ", "steckbrief ", "ereignisse ", "schnittstellen ", \
    "belege "
GENERIERT = (STAND, STECKBRIEF, SCHNITTSTELLEN, BELEGE)  # Bloecke dieses Moduls (Events rendert Task 5)
KEINE_EVENTS = "Keine Events in den Quellen.\n"
# Pruefpunkt-Arten dieses Moduls; jeder Lauf berechnet sie neu (in `struktur` je Lauf ersetzt)
ARTEN = frozenset({"space_doku_ohne_space", "space_doku_wartet", "space_name_ungeeignet", "space_doku_fehler"})
NAME_OK = re.compile(r"^[a-z0-9_]+$")  # docs_drift_waechter.DOC_FILE: nur so benannte Dateien zaehlen im Index
INDEX_LINK = re.compile(r"\]\((\d{2,3}_[a-z0-9_]+\.md)\)")  # docs_drift_waechter.INDEX_LINK
_NUMMER = re.compile(r"^(\d{2,3})_[^/]+\.md$", re.IGNORECASE)
EINSTIEG_NAMEN = frozenset({"__main__.py", "main.py", "server.py", "app.py"})


# ---------- Erkennung ----------

def space_orte(wurzel: Path) -> list[str]:
    """`ort`-Globs (ohne `/` am Ende, relativ zur Wurzel) der Space-Regeln, gelesen wie der Skill `neuer-space`."""
    from .. import skill

    space, _, _ = skill._regeln_sammeln(Path(wurzel))
    return sorted({str(r["ort"]).rstrip("/") for r in space if str(r["ort"]).strip("/")})


def space_ordner(wurzel: Path, *, unter_repos: Iterable[str] = (), orte: Sequence[str] | None = None) -> dict[str, str]:
    """Space-Name (Ordnername) -> repo-relativer Ordner unter `wurzel` (siehe Moduldoku); je Space-Schluessel der erste
    Ordner nach Pfad."""
    from .. import skill

    wurzel = Path(wurzel)
    unter = {u.strip("/") for u in unter_repos if u.strip("/")}
    gefunden: list[tuple[str, str]] = []
    for ort in (space_orte(wurzel) if orte is None else orte):
        try:
            treffer = list(wurzel.glob(ort))
        except (OSError, ValueError):
            continue
        for p in treffer:
            try:
                if not p.is_dir() or (p / ".git").exists() or not any(p.iterdir()):
                    continue
            except OSError:
                continue
            rel = p.relative_to(wurzel).as_posix()
            if p.name.startswith("_") or p.name == "__pycache__" or skill._mit_repo(rel, unter):
                continue
            gefunden.append((rel, p.name))
    aus: dict[str, str] = {}
    schon: set[str] = set()
    for rel, name in sorted(gefunden):
        if space_schluessel(name) not in schon:
            schon.add(space_schluessel(name))
            aus[name] = rel
    return aus


def neue_spaces(konfig: Konfig, graph: Graph | None, messbaum_wurzel: Path, *, unter_repos: Iterable[str] = (),
                dateien: Iterable[str] | None = None, ordner: Mapping[str, str] | None = None) -> list[str]:
    """Space-Namen (Ordnernamen, sortiert) ohne Doku-Datei, erkannt in `messbaum_wurzel`. `dateien`: vorhandene
    Doku-Ziele (sonst `doku_dateien(konfig)`); `ordner`: schon ermittelte `space_ordner`. `graph` wird fuer die
    Erkennung nicht gebraucht (Signatur laut Plan)."""
    del graph
    ordner = space_ordner(messbaum_wurzel, unter_repos=unter_repos) if ordner is None else ordner
    zuordnung = space_dokumente_zuordnen(ordner, konfig.space_dokumente,
                                         doku_dateien(konfig) if dateien is None else dateien)
    return sorted(n for n in ordner if space_schluessel(n) not in zuordnung)


def dateien_planen(namen: Iterable[str], vorhandene: Iterable[str], pruefpunkte: list[Pruefpunkt]
                   ) -> list[tuple[str, str]]:
    """(Name, Datei) je neuem Space: Nummer = hoechste vorhandene `NNN` unter `vorhandene` (Pfade oder Dateinamen)
    plus 1, mehrere fortlaufend nach Name. Ein Schluessel, den der Drift-Waechter nicht als Doku erkennt (nur
    `[a-z0-9_]`), wird nicht angelegt: Pruefpunkt `space_name_ungeeignet`."""
    nummer = max((int(m.group(1)) for d in vorhandene if (m := _NUMMER.match(d.rsplit("/", 1)[-1]))), default=0)
    aus: list[tuple[str, str]] = []
    for name in sorted(namen):
        key = space_schluessel(name)
        if not NAME_OK.match(key):
            pruefpunkte.append(Pruefpunkt("space_name_ungeeignet", f"{name}: als Dateiname `{key}` erkennt der "
                                          "Drift-Waechter keine Doku (nur a-z, 0-9, _); keine Datei angelegt", None,
                                          None, f"space:{key}"))
            continue
        nummer += 1
        aus.append((name, f"{ORDNER}/{nummer:03d}_{key}.md"))
    return aus


def generierter_space(text: str) -> str | None:
    """Space-Schluessel einer generierten Doku (Block `steckbrief <space>`), sonst None."""
    for name in _scannen(text)[0]:
        if name.startswith(STECKBRIEF):
            return name[len(STECKBRIEF):]
    return None


# ---------- Inhalt ----------

def _space_knoten(graph: Graph, space: str) -> str | None:
    k = graph.knoten.get(f"space:{space}")
    return k.datei if k is not None and k.datei else None


def bloecke(space: str, graph: Graph, ereignisse: Sequence[Ereignis], *, ort: str | None = None,
            stand: str = "") -> dict[str, str]:
    """Inhalt (LF) der Bloecke `stand`, `steckbrief`, `schnittstellen`, `belege` und `ereignisse` eines Space aus seinem
    Graphen (`bauen.graph_bauen` fuer diesen Space) und den Events (Task 5). Keine Zeilennummern."""
    from .render_ereignisse import ereignisse_rendern

    key = space_schluessel(space)
    ort = (ort or _space_knoten(graph, space) or "").strip("/")
    routen = sorted({(k.attribute.get("pfad", ""), k.attribute.get("methode", ""), k.datei or "")
                     for k in graph.knoten.values() if k.typ == "route" and k.space == space})
    werkzeuge: dict[str, set[str]] = {}
    for e in graph.kanten.values():
        if e.art == "registriert" and e.attribute.get("space") == space and e.nach.startswith("mcp_werkzeug:"):
            quelle = graph.knoten.get(e.von)
            werkzeuge.setdefault(e.nach[len("mcp_werkzeug:"):], set()).update(
                [quelle.datei] if quelle is not None and quelle.datei else [])
    module = {k.datei for k in graph.knoten.values() if k.typ == "modul" and k.space == space and k.datei}
    einstieg = sorted({m for m in module if m.rsplit("/", 1)[-1] in EINSTIEG_NAMEN}
                      | {d for _, _, d in routen if d} | {d for ds in werkzeuge.values() for d in ds})
    evs = [e for e in ereignisse if space_schluessel(e.space) == key]
    event_dateien = {b.file for e in evs for _, b in e.quellen()}
    dateien = sorted(set(einstieg) | event_dateien)
    nummer = {d: i for i, d in enumerate(dateien, start=2)}

    def ref(datei: str) -> str:
        return f"[B{nummer[datei]}]" if datei in nummer else ""

    zeilen = [_zeile(["Feld", "Wert"]), "|---|---|",
              _zeile(["Ort", f"{_code(ort + '/')} [B1]"]),
              _zeile(["Einstiegspunkte", ", ".join(f"{_code(d)} {ref(d)}" for d in einstieg) or "—"]),
              _zeile(["Routen", str(len(routen))]),
              _zeile(["MCP-Werkzeuge", str(len(werkzeuge))]),
              _zeile(["Events", str(len({e.name for e in evs}))])]
    steckbrief = "".join(z + "\n" for z in zeilen)

    sch = ["**Routen**", ""]
    if routen:
        sch += [_zeile(["Methode", "Pfad", "Beleg"]), "|---|---|---|"]
        sch += [_zeile([_code(m), _code(p), f"{_code(d)} {ref(d)}"]) for p, m, d in routen]
    else:
        sch.append("Keine Routen gefunden.")
    sch += ["", "**MCP-Werkzeuge**", ""]
    if werkzeuge:
        sch += [_zeile(["Werkzeug", "Beleg"]), "|---|---|"]
        sch += [_zeile([_code(w), ", ".join(f"{_code(d)} {ref(d)}" for d in sorted(ds)) or "—"])
                for w, ds in sorted(werkzeuge.items())]
    else:
        sch.append("Keine MCP-Werkzeuge gefunden.")
    belege = [f"- [B1] {_code(ort + '/')}"] + [f"- [B{nummer[d]}] {_code(d)}" for d in dateien]
    return {
        STAND + key: f"Stand: {stand}\n",
        STECKBRIEF + key: steckbrief,
        EREIGNISSE + key: ereignisse_rendern(key, evs) if evs else KEINE_EVENTS,
        SCHNITTSTELLEN + key: "".join(z + "\n" for z in sch),
        BELEGE + key: "".join(z + "\n" for z in belege),
    }


def _leer(name: str) -> str:
    return MARKER_BEGIN.format(block=name) + "\n" + MARKER_END


def geruest(space: str, nummer: int) -> str:
    """Datei mit leeren Bloecken (LF); den Inhalt setzt der Block-Kern."""
    key = space_schluessel(space)
    return "\n".join([
        f"# {nummer:03d} {space}", "", _leer(STAND + key), "", HINWEIS, "",
        "## Steckbrief (generiert)", "", _leer(STECKBRIEF + key), "",
        "## Events (generiert)", "", _leer(EREIGNISSE + key), "",
        "## Schnittstellen (generiert)", "", _leer(SCHNITTSTELLEN + key), "",
        "## Belege", "", _leer(BELEGE + key), ""])


def generierte_rendern(text: str, inhalt: Mapping[str, str], render_hashes: Mapping[str, str],
                       datei: str | None = None) -> RenderErgebnis:
    """Vorhandene Bloecke mit `inhalt` neu rendern (Hand-Edit-Schutz wie ueberall); fuegt nichts ein."""
    return bloecke_rendern(text, lambda b, nl: inhalt[b].replace("\n", nl) if b in inhalt else None, render_hashes,
                           datei)


def doku_entwurf(space: str, graph: Graph, ereignisse: Sequence[Ereignis], nummer: int, *, ort: str | None = None,
                 stand: str = "") -> str:
    """Vollstaendige generierte Doku (LF): Kopfzeile `# <NNN> <name>`, Block `stand` (`Stand: <repo>@<sha7>`), Hinweis,
    Abschnitte Steckbrief, Events, Schnittstellen, Belege -- aller Inhalt in markierten Bloecken."""
    return generierte_rendern(geruest(space, nummer), bloecke(space, graph, ereignisse, ort=ort, stand=stand), {}).text


# ---------- README ----------

def readme_inhalt(eintraege: Sequence[tuple[str, str]]) -> str:
    """Tabelle des Blocks `neue-spaces`: eine Zeile je (Datei, Space-Name), nach Datei sortiert."""
    zeilen = [_zeile(["#", "Space", "Doku"]), "|---|---|---|"]
    for datei, name in sorted(eintraege):
        basis = datei.rsplit("/", 1)[-1]
        zeilen.append(_zeile([basis.split("_", 1)[0], f"[{name}]({basis})",
                              "generiert aus dem Code (Selbstbild); von Hand ergänzen außerhalb der Blöcke"]))
    return "".join(z + "\n" for z in zeilen)


def readme_rendern(text: str, eintraege: Sequence[tuple[str, str]], render_hashes: Mapping[str, str],
                   datei: str = README) -> RenderErgebnis:
    """Block `neue-spaces` neu rendern (Hand-Edit-Schutz) oder, wenn er fehlt, mit Ueberschrift am Ende anhaengen.
    Hand-Inhalt bleibt byteweise; angehaengte Zeilen bekommen das Zeilenende der Mehrheit. Hat der Block einen
    gespeicherten Render-Hash, fehlt aber, hat ihn ein Mensch entfernt: nicht wieder anlegen (`block_entfernt`)."""
    inhalt = readme_inhalt(eintraege)
    res = bloecke_rendern(text, lambda b, nl: inhalt.replace("\n", nl) if b == README_BLOCK else None, render_hashes,
                          datei)
    vorhanden, _, _ = _scannen(text)
    if README_BLOCK in vorhanden or any(p.art == "marker_kaputt" for p in res.pruefpunkte):
        return res
    hashes, pp = dict(res.hashes), list(res.pruefpunkte)
    if README_BLOCK in render_hashes:
        hashes[README_BLOCK] = render_hashes[README_BLOCK]
        pp.append(Pruefpunkt("block_entfernt", f"{README_BLOCK}: Block wurde entfernt; wird nicht wieder eingefuegt, "
                                               "bis der Eintrag aus den Render-Hashes entfernt wird", datei, None,
                             README_BLOCK))
        return RenderErgebnis(res.text, hashes, pp)
    nl = zeilenende(text) if text else "\n"
    neu = res.text
    if neu and not neu.endswith("\n"):
        neu += nl
    if neu.strip("﻿") and not neu.endswith(("\n\n", "\r\n\r\n")):
        neu += nl
    neu += (README_UEBERSCHRIFT + nl + nl + MARKER_BEGIN.format(block=README_BLOCK) + nl + inhalt.replace("\n", nl)
            + MARKER_END + nl)
    hashes[README_BLOCK] = block_hash(inhalt)
    return RenderErgebnis(neu, hashes, pp)


def im_index(readme_text: str, datei: str) -> bool:
    """Verlinkt der Index `datei` so, wie der Drift-Waechter es prueft (`INDEX_LINK`)?"""
    return datei.rsplit("/", 1)[-1] in set(INDEX_LINK.findall(readme_text))
