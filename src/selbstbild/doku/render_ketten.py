"""KETTEN.md aus dem Graph rendern: markierte Bloecke, Hand-Edits erkennen (Task 11, Ruling 34).

Bloecke stehen zwischen eigenen Markerzeilen:

    <!-- graph:begin uebersicht -->        Uebersicht: die vier Spalten mit den Rohtexten + Spalte `Graph`
    <!-- graph:begin hops K<n> -->         Hop-Tabelle einer Kette: dieselben sechs Spalten wie von Hand
    <!-- graph:end -->

Menschliche Information bleibt, Graph-Information kommt dazu: `Kette`, `Zustand`, `Reißt bei`, `Von → Nach`,
`Mechanismus`, `Belegt durch` sind die Rohtexte aus dem Seed; neu sind die Spalte `Graph` (verdrahtet x/n,
gemessen), die Code-Stelle aus den aktuellen Belegen (`file:zeile` + Anker, unverankerte Stellen mit `⚠` und
ausgeschriebenem Pfad, danach ` — ` und der Menschentext der Zelle, `code_prosa`) und der Stufen-Zusatz
`· v:<ja|nein|?> g:<läuft|bricht|—>` an der Zustand-Zelle (ohne Zeit, Ruling 45). Beim Wiederlesen schneidet
`ketten_seed.hop_tabellen_lesen` den Zusatz ab (`RENDER_ZUSATZ`) und trennt die Anker-Spannen ab
(`render_anker_trennen`), damit sie nie als Stelle oder als Bestaetigung zaehlen.

Alles ausserhalb der Bloecke bleibt byteweise (BOM, Zeilenenden, Leerzeichen am Zeilenende). Gerenderte
Zeilen bekommen das Zeilenende der Mehrheit des Textes. Das Rendern oeffnet keine Dateien; die Zeilen der
Belege frischt vorher `belege_auffrischen` auf (Manifest, AST, SQL-Definitionen, YAML-Knoten).

Hand-Edits: `rendern` vergleicht den normalisierten Hash des Blockinhalts (`block_hash`: LF, ohne Leerraum am
Zeilenende) mit dem gespeicherten Render-Hash. Weicht er ab
(und ist der Inhalt nicht zufaellig genau das neue Rendering), bleibt der Block stehen, es entsteht ein
Pruefpunkt `hand_edit`, und der alte Hash wird zurueckgegeben -- so bleibt der Hand-Edit auch in spaeteren
Laeufen stehen, bis ein Mensch den Eintrag aus den Render-Hashes entfernt (dann wird neu gerendert) oder die
Marker entfernt (dann gehoert der Text dem Menschen).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from . import anker
from .anker import Beleg
from .graph import Graph, Kante, hop_zustand_neu
from .ketten_seed import _KETTE, _UEBERSICHT, KOPF, UEBERSICHT_KOPF, _space_von, _zellen
from .konfig import Konfig
from .typen import Pruefpunkt

if TYPE_CHECKING:
    from ._manifest_gate2 import Manifest

    from .extraktoren.sql import SqlDefinition

MARKER_BEGIN = "<!-- graph:begin {block} -->"
MARKER_END = "<!-- graph:end -->"
UEBERSICHT = "uebersicht"
UEBERSICHT_SPALTEN = (*UEBERSICHT_KOPF, "Graph")

_BEGIN = re.compile(r"^<!-- graph:begin (?P<block>\S(?:.*\S)?) -->$")
_HOPS = re.compile(r"^hops (K\d+)$")
_TRENN_UEBERSICHT = "|---|-------|---------|-----------|-------|"
_TRENN_HOPS = "|---|------------|-------------|-------------|--------------|---------|"
_GEMESSEN_TEXT = {"laeuft": "läuft", "bricht": "bricht", "keine_probe": "keine Probe"}

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        _ZONE: tzinfo | None = ZoneInfo("Europe/Berlin")
    except (ZoneInfoNotFoundError, ValueError):
        _ZONE = None
except ImportError:  # pragma: no cover
    _ZONE = None


@dataclass(frozen=True)
class Block:
    start: int  # Index des ersten Zeichens nach der Begin-Markerzeile
    ende: int  # Index des ersten Zeichens der End-Markerzeile
    inhalt: str


@dataclass
class RenderErgebnis:
    text: str
    hashes: dict[str, str] = field(default_factory=dict)
    pruefpunkte: list[Pruefpunkt] = field(default_factory=list)


# ---------- Hilfen ----------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def block_hash(inhalt: str) -> str:
    """Hash des normalisierten Blockinhalts (CRLF -> LF, Leerraum am Zeilenende weg), damit Zeilenenden-
    oder Formatierer-Wechsel kein falscher Hand-Edit sind."""
    return _sha("\n".join(z.rstrip() for z in inhalt.replace("\r\n", "\n").split("\n")))


def zeilenende(text: str) -> str:
    """`\\r\\n`, wenn die Mehrheit der Zeilen so endet, sonst `\\n`."""
    crlf = text.count("\r\n")
    return "\r\n" if crlf > text.count("\n") - crlf else "\n"


def _zeilen(text: str) -> list[str]:
    """Zeilen mit Zeilenende; getrennt wird nur an `\\n` (anders als `splitlines`)."""
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def _zeit_dt(zeit: object) -> datetime | None:
    if not isinstance(zeit, str) or not zeit:
        return None
    try:
        dt = datetime.fromisoformat(zeit.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def zeit_kurz(zeit: object) -> str:
    """ISO-Zeit -> `dd.mm. HH:MM` in Berlin; ohne Zone gilt UTC; Unlesbares -> ``."""
    dt = _zeit_dt(zeit)
    if dt is None:
        return ""
    return (dt.astimezone(_ZONE) if _ZONE is not None else dt.astimezone()).strftime("%d.%m. %H:%M")


def _zelle(text: str) -> str:
    """Zelltext tabellenfest: `|` ausserhalb von Backticks wird `\\|`, Zeilenumbrueche werden Leerzeichen."""
    aus: list[str] = []
    in_code = False
    for zeichen in " ".join(text.splitlines()):
        if zeichen == "`":
            in_code = not in_code
        if zeichen == "|" and not in_code and not (aus and aus[-1] == "\\"):
            aus.append("\\")
        aus.append(zeichen)
    return "".join(aus).strip()


def _code(text: str) -> str:
    """Erzeugter Text als Code-Spanne; `|` darin escapt (GitHub trennt sonst auch in Backticks)."""
    return "`" + text.replace("`", "'").replace("|", "\\|") + "`"


def _zeile(zellen: list[str]) -> str:
    return "| " + " | ".join(zellen) + " |"


def _kette_id(kette: str) -> str:
    return kette.split(":", 1)[1] if kette.startswith("kette:") else kette


def _ketten_nr(kette: str) -> tuple[int, str]:
    m = re.match(r"K(\d+)$", kette)
    return (int(m.group(1)), "") if m else (10**9, kette)


def _ketten(graph: Graph) -> list[str]:
    return sorted((k.id.split(":", 1)[1] for k in graph.knoten.values() if k.typ == "kette"), key=_ketten_nr)


def _nr_schluessel(nr: str) -> tuple[int, str]:
    m = re.match(r"(\d+)(.*)$", nr)
    return (int(m.group(1)), m.group(2)) if m else (10**9, nr)


def _hops(graph: Graph, kette: str) -> list[Kante]:
    hops = [k for k in graph.kanten.values() if k.art == "hop" and k.attribute.get("kette") == kette]
    return sorted(hops, key=lambda k: (_nr_schluessel(str(k.attribute.get("nr", ""))), k.id))


def _zustand(hop: Kante) -> dict[str, object]:
    z = hop.attribute.get("zustand")
    return z if isinstance(z, dict) else hop_zustand_neu()


def _messung(hop: Kante) -> tuple[str, str] | None:
    """(wert, zeit) nur fuer echte Messungen (mit Zeit); Werte aus dem Dokument zaehlen nicht."""
    gem = _zustand(hop).get("gemessen")
    if isinstance(gem, dict) and isinstance(gem.get("zeit"), str) and gem.get("zeit"):
        return str(gem.get("wert", "keine_probe")), str(gem["zeit"])
    return None


def _text(attr: Mapping[str, object], *schluessel: str, ersatz: str = "") -> str:
    for s in schluessel:
        wert = attr.get(s)
        if isinstance(wert, str):
            return wert
    return ersatz


# ---------- Uebersicht ----------

def kette_zusammenfassung(graph: Graph, kette: str) -> str:
    """Graph-Spalte: `[Emoji ]verdrahtet x/n · gemessen <läuft|bricht|keine Probe>` bzw. `… —`.

    Ruling 45 (C.9.5): keine Zeiten im Markdown -- die Messzeit steht nur in graph.json, sonst erzeugte jede neue
    Messung ohne Codeaenderung einen Commit."""
    hops = _hops(graph, kette)
    verdrahtet = [str(_zustand(h).get("verdrahtet")) for h in hops]
    teil = f"verdrahtet {verdrahtet.count('ja')}/{len(hops)} · gemessen "
    messungen = [m for m in (_messung(h) for h in hops) if m is not None]
    if not messungen:
        return teil + "—"
    werte = {w for w, _ in messungen}
    wert = "bricht" if "bricht" in werte else "laeuft" if "laeuft" in werte else (
        "keine_probe" if werte == {"keine_probe"} else sorted(werte)[0])
    emoji = {"bricht": "🔴", "keine_probe": "⚪"}.get(wert, "")
    if wert == "laeuft":
        emoji = "🟡" if "nein" in verdrahtet else "🟢"
    text = teil + _GEMESSEN_TEXT.get(wert, wert)
    return f"{emoji} {text}" if emoji else text


def uebersicht_rendern(graph: Graph, zeilenende: str = "\n") -> str:
    """Tabelle `| # | Kette | Zustand | Reißt bei | Graph |`; die ersten vier Spalten sind die Rohtexte."""
    zeilen = [_zeile(list(UEBERSICHT_SPALTEN)), _TRENN_UEBERSICHT]
    for kette in _ketten(graph):
        knoten = graph.knoten[f"kette:{kette}"]
        attr = knoten.attribute
        zeilen.append(_zeile([
            kette, _zelle(_text(attr, "titel_roh", ersatz=knoten.name)),
            _zelle(_text(attr, "zustand_roh", ersatz="—")), _zelle(_text(attr, "reisst_bei_roh", ersatz="—")),
            kette_zusammenfassung(graph, kette),
        ]))
    return "".join(z + zeilenende for z in zeilen)


# ---------- Hop-Tabellen ----------

def _beleg_text(beleg: object) -> str:
    file, zeile, anker = getattr(beleg, "file", ""), getattr(beleg, "zeile", None), getattr(beleg, "anker", "")
    text = _code(f"{file}:{zeile}" if zeile is not None else str(file))
    return f"{text} {_code(str(anker))}" if anker else text


def _code_stelle(hop: Kante) -> str:
    aktuell = {(b.art, b.file, b.anker): b for b in hop.belege}
    stellen = hop.attribute.get("stellen_roh")
    stellen = stellen if isinstance(stellen, list) else []
    teile: list[str] = []
    gezeigt: set[tuple[str, str, str]] = set()
    for st in stellen:
        if not isinstance(st, dict):
            continue
        roh, schl = str(st.get("roh", "")), st.get("beleg")
        key = (str(schl[0]), str(schl[1]), str(schl[2])) if isinstance(schl, (list, tuple)) and len(schl) == 3 \
            else None
        if key is not None and key in aktuell:
            if key not in gezeigt:
                teile.append(_beleg_text(aktuell[key]))
                gezeigt.add(key)
        elif roh:  # Kurzformen ausgeschrieben, damit ein Wiederlesen sie nicht anders aufloest
            voll = st.get("voll")
            teile.append("⚠ " + _code(voll if isinstance(voll, str) and voll else roh))
    rest = sorted((b for k, b in aktuell.items() if k not in gezeigt),
                  key=lambda b: (b.file, b.zeile or 0, b.anker, b.art))
    teile.extend(_beleg_text(b) for b in rest)
    belege_text = ", ".join(teile)
    prosa = hop.attribute.get("code_prosa")
    if isinstance(prosa, str):  # menschlicher Text der Zelle bleibt hinter den Belegen stehen
        prosa = _zelle(prosa)
        if prosa:
            return f"{belege_text} — {prosa}" if belege_text else f"— {prosa}"
    if not belege_text:  # die Zelle nennt keine Stelle (z. B. `—`): Rohtext bleibt
        return _zelle(_text(hop.attribute, "code_stelle")) or "—"
    return belege_text


def _zustand_text(hop: Kante) -> str:
    z = _zustand(hop)
    v = {"ja": "ja", "nein": "nein"}.get(str(z.get("verdrahtet")), "?")
    m = _messung(hop)
    stufe = f"v:{v} g:—"
    if m is not None and m[0] in ("laeuft", "bricht"):
        stufe = f"v:{v} g:{_GEMESSEN_TEXT[m[0]]}"  # ohne Zeit (Ruling 45)
    roh = _zelle(_text(hop.attribute, "zustand_roh", "zustand_text"))
    return f"{roh} · {stufe}".strip()


def hops_rendern(graph: Graph, kette: str, zeilenende: str = "\n") -> str:
    """Hop-Tabelle einer Kette mit den sechs Spalten von heute (Rohtexte + Belege + Stufen)."""
    zeilen = [_zeile(list(KOPF)), _TRENN_HOPS]
    for hop in _hops(graph, _kette_id(kette)):
        a = hop.attribute
        zeilen.append(_zeile([
            _zelle(str(a.get("nr", ""))), _zelle(_text(a, "von_nach")),
            _zelle(_text(a, "mechanismus_roh", "mechanismus")), _code_stelle(hop),
            _zelle(_text(a, "belegt_durch")), _zustand_text(hop),
        ]))
    return "".join(z + zeilenende for z in zeilen)


def _block_rendern(graph: Graph, block: str, zeilenende: str) -> str | None:
    if block == UEBERSICHT:
        return uebersicht_rendern(graph, zeilenende)
    m = _HOPS.match(block)
    if m and f"kette:{m.group(1)}" in graph.knoten:
        return hops_rendern(graph, m.group(1), zeilenende)
    return None


# ---------- Beleg-Zeilen auffrischen ----------

def _yaml_zeile(wurzel: yaml.Node, pfad: str) -> int | None:
    """Startzeile (1-basiert) des Knotens `a.b[2]` / `[capability=x].y`; bei Schluesseln die Zeile des
    Schluessels."""
    tokens = anker._tokens(pfad)
    if not tokens:
        return None
    knoten: yaml.Node = wurzel
    zeile: int | None = None
    for t in tokens:
        if isinstance(knoten, yaml.MappingNode) and isinstance(t, str):
            paar = next(((k, v) for k, v in knoten.value if isinstance(k, yaml.ScalarNode) and k.value == t), None)
            if paar is None:
                return None
            zeile, knoten = paar[0].start_mark.line + 1, paar[1]
        elif isinstance(knoten, yaml.SequenceNode) and isinstance(t, int) and t < len(knoten.value):
            knoten = knoten.value[t]
            zeile = knoten.start_mark.line + 1
        elif isinstance(knoten, yaml.SequenceNode) and isinstance(t, tuple):
            treffer = [v for v in knoten.value if anker.yaml_identitaet(v, t[0]) == t[1]]
            if len(treffer) != 1:
                return None
            knoten = treffer[0]
            zeile = knoten.start_mark.line + 1
        else:
            return None
    return zeile


class _Auffrischer:
    def __init__(self, manifeste: Mapping[str, "Manifest"], konfig: Konfig,
                 sql_defs: Mapping[str, "SqlDefinition"] | None) -> None:
        self.manifeste, self.konfig = manifeste, konfig
        self.sql_defs: Mapping[str, "SqlDefinition"] = sql_defs or {}
        self.yaml: dict[str, yaml.Node | None] = {}
        self.ast: dict[str, list[anker.Definition] | None] = {}

    def zeile(self, b: Beleg) -> int | None:
        if b.art == "symbol":
            return self.symbol(b)
        if b.art == "sql":
            d = self.sql_defs.get(b.anker)
            return d.zeile if d is not None and d.datei == b.file else None
        if b.art == "yaml_key" and Path(b.file).suffix.lower() in (".yaml", ".yml"):
            if b.file not in self.yaml:
                try:
                    self.yaml[b.file] = yaml.compose((self.konfig.repo / b.file).read_text(encoding="utf-8"),
                                                     Loader=yaml.SafeLoader)
                except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError):
                    self.yaml[b.file] = None
            wurzel = self.yaml[b.file]
            return _yaml_zeile(wurzel, b.anker) if wurzel is not None else None
        return None  # datei, toml: unveraendert

    def symbol(self, b: Beleg) -> int | None:
        space = _space_von(b.file, self.konfig)
        if space is not None:
            manifest = self.manifeste.get(space)
            if manifest is None:
                return None
            rel = anker.korpus_relativ(manifest, self.konfig.repo, b.file)
            treffer = [d for d in manifest.documents if d.file == rel and d.document_id == b.anker]
            return int(treffer[0].start_line) if len(treffer) == 1 else None
        if b.file not in self.ast:
            self.ast[b.file] = anker.ast_definitionen(self.konfig.repo / b.file, anker.modulpfad(b.file))
        treffer_ast = [d for d in self.ast[b.file] or [] if d.document_id == b.anker]
        return treffer_ast[0].start_line if len(treffer_ast) == 1 else None


def belege_auffrischen(graph: Graph, manifeste: Mapping[str, "Manifest"], konfig: Konfig,
                       sql_defs: Mapping[str, "SqlDefinition"] | None = None) -> Graph:
    """Kopie des Graphen, in der `Beleg.zeile` der Hop-Belege die aktuelle Startzeile des Ankers ist.

    symbol im Space -> `Document.start_line` des Manifests (Datei + document_id); ausserhalb -> AST;
    sql -> `SqlDefinition.zeile`; yaml_key -> Zeile des aufgeloesten Knotens; datei/toml unveraendert.
    Laesst sich ein Anker nicht mehr aufloesen, bleibt die Zeile (das meldet `beleg_pruefen`)."""
    neu = copy.deepcopy(graph)
    auffrischer = _Auffrischer(manifeste, konfig, sql_defs)
    for kante in neu.kanten.values():
        if kante.art != "hop" or not kante.belege:
            continue
        belege = []
        for b in kante.belege:
            zeile = auffrischer.zeile(b)
            belege.append(dataclasses.replace(b, zeile=zeile) if zeile is not None and zeile != b.zeile else b)
        kante.belege = tuple(belege)
    return neu


# ---------- Bloecke finden ----------

def _scannen(text: str) -> tuple[dict[str, Block], dict[str, int], list[Pruefpunkt]]:
    bloecke: dict[str, Block] = {}
    zeilen_nr: dict[str, int] = {}
    pp: list[Pruefpunkt] = []
    offen: tuple[str, int, int] | None = None  # (Block, Inhaltsbeginn, Zeilennummer des Markers)
    pos = 0
    for nr, zeile in enumerate(_zeilen(text), 1):
        klar = zeile.strip().lstrip("﻿")
        m = _BEGIN.match(klar)
        if m:
            if offen is not None:
                pp.append(Pruefpunkt("marker_kaputt", f"{offen[0]}: Begin ohne End (Zeile {offen[2]})",
                                     None, offen[2], offen[0]))
            offen = (m.group("block"), pos + len(zeile), nr)
        elif klar == MARKER_END:
            if offen is None:
                pp.append(Pruefpunkt("marker_kaputt", f"End-Marker ohne Begin (Zeile {nr})", None, nr, None))
            elif offen[0] in bloecke:
                pp.append(Pruefpunkt("marker_kaputt", f"{offen[0]}: Block doppelt (Zeile {offen[2]})",
                                     None, offen[2], offen[0]))
            else:
                bloecke[offen[0]] = Block(offen[1], pos, text[offen[1]:pos])
                zeilen_nr[offen[0]] = offen[2]
            offen = None
        pos += len(zeile)
    if offen is not None:
        pp.append(Pruefpunkt("marker_kaputt", f"{offen[0]}: Begin ohne End (Zeile {offen[2]})",
                             None, offen[2], offen[0]))
    return bloecke, zeilen_nr, pp


def bloecke_finden(text: str) -> dict[str, Block]:
    """Markierte Bloecke (vollstaendige Begin/End-Paare; bei Doppelten zaehlt der erste)."""
    return _scannen(text)[0]


def tabellen_finden(text: str) -> list[tuple[str, int, int]]:
    """Unmarkierte Tabellen, die `einbetten` ersetzt: (Block, erste Zeile, Zeile nach dem Ende), 0-basiert.

    Gefunden wie in `ketten_seed`: Hop-Kopf unter `### K<n> · …`, Uebersichts-Kopf unter `## Übersicht`;
    die Tabelle reicht bis zur ersten Zeile, die nicht mit `|` beginnt. Markierte Bereiche werden
    uebersprungen; je Block zaehlt die erste Tabelle, und ein schon markierter Block wird nicht doppelt
    angelegt."""
    zeilen = _zeilen(text)
    gefunden: list[tuple[str, int, int]] = []
    gesehen: set[str] = set(bloecke_finden(text))
    kette: str | None = None
    im_uebersicht = im_block = False
    i = 0
    while i < len(zeilen):
        klar = zeilen[i].strip().lstrip("﻿")
        i += 1
        if im_block:
            im_block = klar != MARKER_END
            continue
        if _BEGIN.match(klar):
            im_block = True
            continue
        m = _KETTE.match(klar)
        if m:
            kette, im_uebersicht = m.group(1), False
            continue
        if klar.startswith("#"):
            kette, im_uebersicht = None, bool(_UEBERSICHT.match(klar))
            continue
        if not klar.startswith("|"):
            continue
        zellen = _zellen(klar)
        block = None
        if kette is not None and tuple(zellen) == KOPF:
            block = f"hops {kette}"
        elif im_uebersicht and tuple(zellen[:4]) == UEBERSICHT_KOPF and len(zellen) in (4, 5):
            block = UEBERSICHT
        if block is None or block in gesehen:
            continue
        ende = i
        while ende < len(zeilen) and zeilen[ende].strip().startswith("|"):
            ende += 1
        gefunden.append((block, i - 1, ende))
        gesehen.add(block)
        i = ende
    return gefunden


# ---------- Einbetten und Rendern ----------

def einbetten(text: str, graph: Graph) -> str:
    """Erster Lauf: Uebersicht und Hop-Tabellen an Ort und Stelle durch markierte Bloecke ersetzen.

    Nur Tabellen, deren Kette im Graph steht; alles andere bleibt byteweise. Idempotent."""
    nl = zeilenende(text)
    zeilen = _zeilen(text)
    aus: list[str] = []
    weiter = 0
    for block, start, ende in tabellen_finden(text):
        inhalt = _block_rendern(graph, block, nl)
        if inhalt is None:
            continue
        letzte = zeilen[ende - 1]
        aus.extend(zeilen[weiter:start])
        aus.append(MARKER_BEGIN.format(block=block) + nl)
        aus.append(inhalt)
        aus.append(MARKER_END + letzte[len(letzte.rstrip("\r\n")):])
        weiter = ende
    aus.extend(zeilen[weiter:])
    return "".join(aus)


BlockRenderer = Callable[[str, str], "str | None"]  # (Blockname, Zeilenende) -> Inhalt oder None (kein Renderer)
OhneRenderer = Callable[[str], tuple[str, str]]  # Blockname -> (Pruefpunkt-Art, Text) fuer Bloecke ohne Inhalt


def bloecke_rendern(text_alt: str, renderer: BlockRenderer, render_hashes: Mapping[str, str],
                    datei: str | None = None, *, ohne_renderer: OhneRenderer | None = None) -> RenderErgebnis:
    """Allgemeiner Kern fuer jede Doku-Datei (KETTEN.md, Space-Doku): jeden markierten Block mit `renderer` neu
    rendern, ausser er wurde von Hand geaendert (`hand_edit`, siehe Moduldoku). Ein Block, fuer den `renderer` None
    gibt, bleibt stehen (Pruefpunkt aus `ohne_renderer`, sonst `block_unbekannt`). Alles ausserhalb der Bloecke bleibt
    byteweise; gerenderte Zeilen bekommen das Zeilenende der Mehrheit."""
    nl = zeilenende(text_alt)
    bloecke, zeilen_nr, pp = _scannen(text_alt)
    pp = [Pruefpunkt(p.art, p.text, datei, p.zeile, p.aussage_id) for p in pp]
    hashes: dict[str, str] = {}
    teile: list[str] = []
    pos = 0
    for block, b in sorted(bloecke.items(), key=lambda kv: kv[1].start):
        alt_hash = block_hash(b.inhalt)
        neu = renderer(block, nl)
        if neu is None:
            art, text = (ohne_renderer(block) if ohne_renderer is not None
                         else ("block_unbekannt", f"{block}: kein Renderer; Block bleibt stehen"))
            pp.append(Pruefpunkt(art, text, datei, zeilen_nr[block], block))
            hashes[block] = alt_hash
            continue
        gespeichert = render_hashes.get(block)
        if gespeichert is not None and gespeichert != alt_hash and alt_hash != block_hash(neu):
            pp.append(Pruefpunkt("hand_edit", f"{block}: von Hand geaendert; bleibt stehen, bis der Eintrag "
                                              f"aus den Render-Hashes entfernt wird", datei, zeilen_nr[block], block))
            hashes[block] = gespeichert
            continue
        teile.extend((text_alt[pos:b.start], neu))
        pos = b.ende
        hashes[block] = block_hash(neu)
    teile.append(text_alt[pos:])
    return RenderErgebnis("".join(teile), hashes, pp)


def _ketten_ohne_renderer(block: str) -> tuple[str, str]:
    art = "block_ohne_kette" if _HOPS.match(block) else "block_unbekannt"
    return art, f"{block}: keine passende Kette im Graph; Block bleibt stehen"


def rendern(text_alt: str, graph: Graph, render_hashes: Mapping[str, str],
            datei: str | None = None) -> RenderErgebnis:
    """Jeden markierten Block von KETTEN.md neu rendern, ausser er wurde von Hand geaendert (siehe Moduldoku)."""
    res = bloecke_rendern(text_alt, lambda block, nl: _block_rendern(graph, block, nl), render_hashes, datei,
                          ohne_renderer=_ketten_ohne_renderer)
    bloecke = bloecke_finden(text_alt)
    pp = res.pruefpunkte
    if UEBERSICHT not in bloecke:
        pp.append(Pruefpunkt("block_fehlt", "uebersicht: kein markierter Block", datei, None, UEBERSICHT))
    for kette in _ketten(graph):
        if f"hops {kette}" not in bloecke:
            pp.append(Pruefpunkt("kette_ohne_block", f"{kette}: Kette im Graph ohne markierten Block",
                                 datei, None, f"hops {kette}"))
    return res
