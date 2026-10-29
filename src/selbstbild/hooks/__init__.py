"""Gemeinsame Helfer der Claude-Code-Hooks (Stufe 3). NUR Standardbibliothek.

Die Hooks laufen bei jedem Prompt und muessen schnell starten: kein numpy, igraph, yaml, tree-sitter und
kein `selbstbild.speicher`/`vermessen` (die ziehen schwere Importe nach). Darum steht die Ordnerformel des
lokalen Stands hier bewusst ein zweites Mal; `test_stand_fuer_wie_vermessen` haelt beide gleich.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

__all__ = ["repo_wurzeln", "stand_fuer", "blob_sha", "regeln_lesen"]

REGELDATEI = "selbstbild.yaml"


def repo_wurzeln(pfad: Path) -> tuple[Path, ...]:
    """Alle Ordner ab `pfad` aufwaerts bis zur Laufwerkswurzel mit `selbstbild.yaml`, von aussen nach innen."""
    p = Path(pfad).resolve()
    gefunden = [d for d in (p, *p.parents) if (d / REGELDATEI).is_file()]
    return tuple(reversed(gefunden))


def _stand_basis() -> Path:
    """Spiegel von `speicher.stand_basis`."""
    home = os.environ.get("SELBSTBILD_HOME")
    if home:
        return Path(home)
    if sys.platform == "win32":
        lokal = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(lokal) / "selbstbild"
    return Path.home() / ".local" / "state" / "selbstbild"


def _pfad_kennung(wurzel: Path) -> str:
    """Spiegel von `speicher.pfad_kennung`."""
    return hashlib.sha256(str(Path(wurzel).resolve()).encode("utf-8", "surrogateescape")).hexdigest()[:12]


def stand_fuer(wurzel: Path) -> Path | None:
    """Spiegel von `vermessen.stand_finden`: vorhandenes `graph.sqlite` dieser Wurzel (zuletzt geschriebenes), sonst None."""
    basis = _stand_basis()
    if not basis.is_dir():
        return None
    kandidaten = [o / "graph.sqlite" for o in basis.glob(f"*-{_pfad_kennung(wurzel)}")
                  if (o / "graph.sqlite").is_file()]
    if not kandidaten:
        return None
    return max(kandidaten, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def blob_sha(datei: Path) -> tuple[str, str]:
    """Git-Blob-SHA1 der Datei wie gelesen und nach CRLF->LF."""
    daten = Path(datei).read_bytes()

    def sha(b: bytes) -> str:
        return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()

    return sha(daten), sha(daten.replace(b"\r\n", b"\n"))


_ERLAUBT = ("art", "ort", "pflicht", "verboten", "hinweis")
_LISTEN = ("pflicht", "verboten")


class _Unbekannt(Exception):
    pass


def _skalar(roh: str) -> str:
    s = roh.strip()
    if s[:1] in ("'", '"'):
        ende = s.find(s[0], 1)
        if ende < 0:
            raise _Unbekannt
        return s[1:ende]
    return re.split(r"\s+#", s, maxsplit=1)[0].strip()


def _fluss(roh: str) -> list[str]:
    s = re.split(r"(?<=\])\s+#", roh.strip(), maxsplit=1)[0].strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise _Unbekannt
    innen = s[1:-1].strip()
    return [_skalar(t) for t in innen.split(",")] if innen else []


def regeln_lesen(yaml_text: str) -> dict:
    """Minimaler Leser fuer `name` und `struktur[*].{art,ort,pflicht,verboten,hinweis}`; sonst `{}`.

    Kann nur Skalare, Flow-Listen (`[a, b]`) und Block-Listen. Alles andere (Anker, Mehrzeiler, verschachtelte
    Strukturen innerhalb der Regeln) fuehrt zu `{}` und damit zu keiner Regelzeile im Hook.
    """
    try:
        name: str | None = None
        regeln: list[dict] = []
        im_struktur = False
        akt: dict | None = None
        offen: str | None = None  # Schluessel einer Block-Liste
        for zeile in yaml_text.splitlines():
            if not zeile.strip() or zeile.lstrip().startswith("#"):
                continue
            einzug = len(zeile) - len(zeile.lstrip(" "))
            inhalt = zeile.strip()
            if einzug == 0:
                im_struktur = False
                m = re.match(r"([A-Za-z_]+):(?:\s+(.*))?$", inhalt)
                if m is None:
                    raise _Unbekannt
                if m.group(1) == "name" and m.group(2):
                    name = _skalar(m.group(2))
                elif m.group(1) == "struktur":
                    if m.group(2):
                        raise _Unbekannt
                    im_struktur = True
                continue
            if not im_struktur:
                continue
            if offen is not None and inhalt.startswith("- ") and akt is not None and einzug > akt["_einzug"]:
                akt[offen].append(_skalar(inhalt[2:]))
                continue
            offen = None
            if inhalt.startswith("- "):
                akt = {"_einzug": einzug, "pflicht": [], "verboten": []}
                regeln.append(akt)
                inhalt = inhalt[2:].strip()
                einzug += 2
            if akt is None:
                raise _Unbekannt
            m = re.match(r"([A-Za-z_]+):(?:\s+(.*))?$", inhalt)
            if m is None:
                raise _Unbekannt
            schluessel, wert = m.group(1), m.group(2)
            if schluessel not in _ERLAUBT:
                continue
            if schluessel in _LISTEN:
                if wert:
                    akt[schluessel] = _fluss(wert)
                else:
                    akt[schluessel] = []
                    offen = schluessel
            elif wert:
                akt[schluessel] = _skalar(wert)
            else:
                raise _Unbekannt
        ergebnis: dict = {}
        if name:
            ergebnis["name"] = name
        if regeln:
            if any("art" not in r or "ort" not in r for r in regeln):
                raise _Unbekannt
            ergebnis["struktur"] = [{k: v for k, v in r.items() if k != "_einzug"} for r in regeln]
        return ergebnis
    except _Unbekannt:
        return {}
