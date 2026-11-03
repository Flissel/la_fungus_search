"""Events-Block einer Space-Doku rendern und einfuegen (Stufe 4, Task 5, D8). Deterministisch, kein LLM.

Block je Space (Schluessel nach `namen.space_schluessel`):

    <!-- graph:begin ereignisse <space> -->
    | Event | Werkzeug | Quelle | Beleg |
    ...
    <!-- graph:end -->

Zeilen nach Event-Name sortiert; weichen die Werkzeuge der Quellen ab, steht `⚠` in der Werkzeug-Zelle mit allen
Werten. Die Beleg-Zelle nennt Datei und Anker ohne Zeile (Fix-Runde 1), damit verschobene Zeilen den Block nicht
aendern. Gerendert und vor Hand-Edits geschuetzt wird mit demselben Kern wie KETTEN.md
(`render_ketten.bloecke_rendern`, Render-Hashes, Pruefpunkt `hand_edit`).

Fehlt ein Block, fuegt der erste Lauf ihn ein: hinter den letzten vorhandenen Events-Block, sonst als Abschnitt
`## Events (generiert)` vor der ersten Belege-Ueberschrift der Ebene 2 (`## Belege`, `## 7. Belege`,
`## Belege (Zusatz)`), sonst am Dateiende. Nur neue Zeilen kommen dazu; BOM und Zeilenenden bleiben. Gibt es fuer den
Block schon einen gespeicherten Render-Hash, hat ein Mensch die Marker entfernt: dann wird nicht wieder eingefuegt
(Pruefpunkt `block_entfernt`), der Text gehoert dem Menschen.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

from .anker import Beleg, modulpfad
from .belege_formate import BOM, kopf_ebene
from .extraktoren.ereignisse import Ereignis
from .namen import space_schluessel
from .render_ketten import (
    MARKER_BEGIN, MARKER_END, RenderErgebnis, _code, _scannen, _zeile, _zeilen, _zelle, block_hash, bloecke_rendern,
    zeilenende,
)
from .typen import Pruefpunkt

BLOCK_PRAEFIX = "ereignisse "
UEBERSCHRIFT = "## Events (generiert)"
KOPF = ("Event", "Werkzeug", "Quelle", "Beleg")
_TRENN = "|---|---|---|---|"
_ZAUN = re.compile(r"^(`{3,}|~{3,})")


def block_name(space: str) -> str:
    return BLOCK_PRAEFIX + space_schluessel(space)


def _werkzeug(w: str) -> str:
    return _code(w) if w else "—"


def _kurzanker(b: Beleg) -> str:
    """Anker ohne Zeile (Fix-Runde 1): YAML-Pfad wie gespeichert, Symbol ohne den Modulpfad der Datei
    (`VideoBackendAgent.EVENT_TO_TOOL`)."""
    modul = modulpfad(b.file) + "."
    return b.anker[len(modul):] if b.art == "symbol" and b.anker.startswith(modul) else b.anker


def ereignisse_rendern(space: str, ereignisse: Sequence[Ereignis]) -> str:
    """Tabelle `| Event | Werkzeug | Quelle | Beleg |` der Events dieses Space (LF-Zeilen, nach Name sortiert)."""
    key = space_schluessel(space)
    je_name: dict[str, list[Ereignis]] = defaultdict(list)
    for e in ereignisse:
        if space_schluessel(e.space) == key:
            je_name[e.name].append(e)
    zeilen = [_zeile(list(KOPF)), _TRENN]
    for name in sorted(je_name):
        evs = sorted(je_name[name], key=lambda e: e.werkzeug)
        werkzeuge = sorted({e.werkzeug for e in evs})
        if len(werkzeuge) == 1:
            werkzeug = _werkzeug(werkzeuge[0])
        else:
            werkzeug = "⚠ " + " / ".join(
                f"{_werkzeug(e.werkzeug)} ({', '.join(sorted({q for q, _ in e.quellen()}))})" for e in evs)
        quellen = sorted({q for e in evs for q, _ in e.quellen()})
        belege = sorted({(b.file, b.anker): b for e in evs for _, b in e.quellen()}.items())
        stellen = ", ".join(f"{_code(b.file)} {_code(_kurzanker(b))}" for _, b in belege)
        zeilen.append(_zeile([_code(name), werkzeug, _zelle(", ".join(quellen)), stellen]))
    return "".join(z + "\n" for z in zeilen)


def _ohne_renderer(block: str) -> tuple[str, str]:
    if block.startswith(BLOCK_PRAEFIX):
        return "block_ohne_quelle", f"{block}: keine Events mehr in den Quellen; Block bleibt stehen"
    return "block_unbekannt", f"{block}: kein Renderer fuer diese Datei; Block bleibt stehen"


def _block_text(name: str, inhalt: str, nl: str) -> str:
    return MARKER_BEGIN.format(block=name) + nl + inhalt.replace("\r\n", "\n").replace("\n", nl) + MARKER_END + nl


def _belege_pos(text: str) -> int | None:
    """Zeichenposition der ersten Belege-Ueberschrift der Ebene 2 ausserhalb von Code-Bloecken."""
    zaun: str | None = None
    pos = 0
    for i, zeile in enumerate(_zeilen(text)):
        klar = (zeile[1:] if i == 0 and zeile.startswith(BOM) else zeile).strip()
        if zaun is not None:
            if klar.startswith(zaun[0] * len(zaun)) and not klar.strip(zaun[0]):
                zaun = None
        elif _ZAUN.match(klar):
            zaun = _ZAUN.match(klar).group(1)  # type: ignore[union-attr]
        elif kopf_ebene(klar) == 2:
            return pos + (1 if i == 0 and zeile.startswith(BOM) else 0)
        pos += len(zeile)
    return None


def _leerzeile_davor(text: str) -> bool:
    return text in ("", BOM) or text.endswith(("\n\n", "\r\n\r\n"))


def _einfuegen(text: str, neu: list[tuple[str, str]]) -> str:
    nl = zeilenende(text)
    stuecke = [_block_text(name, inhalt, nl) for name, inhalt in neu]
    bloecke, _, _ = _scannen(text)
    eigene = [b for name, b in bloecke.items() if name.startswith(BLOCK_PRAEFIX)]
    if eigene:  # hinter den letzten Events-Block, je Block eine Leerzeile davor
        ende = text.find("\n", max(b.ende for b in eigene))
        pos = len(text) if ende < 0 else ende + 1
        davor = "" if ende >= 0 else nl
        return text[:pos] + davor + "".join(nl + s for s in stuecke) + text[pos:]
    abschnitt = UEBERSCHRIFT + nl + nl + nl.join(stuecke)
    pos = _belege_pos(text)
    if pos is not None:
        return text[:pos] + ("" if _leerzeile_davor(text[:pos]) else nl) + abschnitt + nl + text[pos:]
    if text and not text.endswith("\n"):
        text += nl
    return text + ("" if _leerzeile_davor(text) else nl) + abschnitt


def doku_rendern(text: str, bloecke: Mapping[str, str], render_hashes: Mapping[str, str],
                 datei: str | None = None) -> RenderErgebnis:
    """Vorhandene markierte Bloecke neu rendern (Inhalt aus `bloecke`, Name -> LF-Text) mit Hand-Edit-Schutz, dann
    fehlende Bloecke einfuegen (siehe Moduldoku). `render_hashes` gelten nur fuer diese Datei (Blockname -> Hash).
    Ohne Bloecke und ohne Marker bleibt der Text unberuehrt."""
    if not bloecke and "graph:begin" not in text and "graph:end" not in text:
        return RenderErgebnis(text, {}, [])
    res = bloecke_rendern(text, lambda block, nl: bloecke[block].replace("\n", nl) if block in bloecke else None,
                          render_hashes, datei, ohne_renderer=_ohne_renderer)
    vorhanden, _, _ = _scannen(text)
    kaputt = {p.aussage_id for p in res.pruefpunkte if p.art == "marker_kaputt"}
    hashes, pp = dict(res.hashes), list(res.pruefpunkte)
    neu: list[tuple[str, str]] = []
    for name in sorted(bloecke):
        if name in vorhanden or name in kaputt:
            continue
        if name in render_hashes:  # Marker von Hand entfernt: Text gehoert dem Menschen
            hashes[name] = render_hashes[name]
            pp.append(Pruefpunkt("block_entfernt", f"{name}: Block wurde entfernt; wird nicht wieder eingefuegt, bis "
                                                   "der Eintrag aus den Render-Hashes entfernt wird", datei, None, name))
            continue
        neu.append((name, bloecke[name]))
        hashes[name] = block_hash(bloecke[name])
    return RenderErgebnis(_einfuegen(res.text, neu) if neu else res.text, hashes, pp)
