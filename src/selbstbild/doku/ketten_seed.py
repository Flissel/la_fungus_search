"""Seed des Datenfluss-Graphen aus den Hop-Tabellen von docs/spaces/KETTEN.md.

Jede Zeile einer Tabelle mit Kopf `| # | Von → Nach | Mechanismus | Code-Stelle | Belegt durch | Zustand |`
unter einer Ueberschrift `### K<n> · <Titel>` wird eine `hop`-Kante (`quelle: seed`). Die Code-Stellen
werden neu verankert (symbol / sql / yaml_key / datei); was sich nicht (sicher) verankern laesst, landet
als Pruefpunkt in der Pruefliste. Nichts hier wirft wegen einer kaputten Zeile.

Formen der Code-Stellen (Zellen trennen mehrere Belege mit `,` `·` `←` oder Leerzeichen):
  voll         `vibemind-os/spaces/x/a.py:12-20` (auch ohne Zeile, auch ohne fuehrendes `vibemind-os/`)
  kurzform     `:2201`                -> Pfad der vorigen Stelle
  verzeichnis  `mail_dispatch.py:243` -> im Verzeichnis der vorigen Stelle
  anker        `einzig.py:7`          -> eindeutig im Repo (sonst frueher im Dokument genannter Pfad, sonst None)
  auslassung   `043…sql:297`          -> pfad None (nicht aufloesbar)

Namen und Tokens einer Zeile (aus den Backtick-Spannen von Von → Nach, Mechanismus, Code-Stelle):
  Namen  (`zeilen_namen`, Ruling 22) Bezeichner, `modul.name(...)` -> `name`; Glob-Praefixe wie
         `schedule_` aus `schedule_*` zaehlen nicht.
  Tokens (`zeilen_tokens`, Rulings 27/29) ganze Spannen, normalisiert, >= 5 Zeichen, ohne Stoppwoerter und
         ohne schwache Tokens (Spacenamen, `schwache_tokens`). Tokens mit `*` sind Globs.
  Nennt eine Zeile gar keinen Namen (ungefiltert leer), gilt der zeilenbasierte Treffer ohne Bestaetigung.

Verankerung:
  .py    (Rulings 22/26/27) ein Symbol der Datei, dessen Name in der Zeile steht (das naechste zur Zeile);
         sonst das umschliessende Symbol, wenn die Zeile keine Namen nennt oder sein Quelltext (mit
         Dekoratoren) ein Nicht-Glob-Token woertlich enthaelt; sonst None + `beleg_unsicher`.
         Ausserhalb der Spaces direkt per AST (`<modulpfad>.<qualname>`).
  .sql   (Ruling 23) Name der CREATE-Anweisung, die die Zeile umfasst -> `sql_defs[name]`.
  yaml/toml  (Rulings 24/29/30) Knoten = kleinste Sammlung ueber den ganzen Bereich; Listeneintraege per
         Identitaet `[capability=x]`, Position nur als Rueckfall (`anker_positional`). Reihenfolge:
         (i) der Zeilenknoten, wenn ein Identitaetswert (exakt oder Glob), ein Schluessel (exakt) oder sein
         serialisierter Text (Nicht-Glob-Token) ihn bestaetigt; (ii) sonst der naechste Listeneintrag, dessen
         Identitaetswert einem Token gleicht (exakt oder Glob); (iii) sonst `beleg_unsicher` bzw.
         `anker_zu_grob` (nur Wurzel oder oberste Sammlung).
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import re
import tomllib
import warnings
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import yaml

from . import anker, korpus
from . import konfig as konfig_modul
from .anker import Beleg, Token
from .extraktoren import sql as sql_extraktor
from .graph import Graph, Kante, Knoten, hop_zustand_neu, kante_id
from .konfig import Konfig
from .typen import Pruefpunkt

if TYPE_CHECKING:
    from ._manifest_gate2 import Manifest

    from .extraktoren.sql import SqlDefinition

KOPF = ("#", "Von → Nach", "Mechanismus", "Code-Stelle", "Belegt durch", "Zustand")
UEBERSICHT_KOPF = ("#", "Kette", "Zustand", "Reißt bei")
DB_SCHEMATA = frozenset({"marketing", "sales", "public"})

_KETTE = re.compile(r"^###\s+(K\d+)\s*·\s*(.*?)\s*$")
_TRENNZEILE = re.compile(r"^:?-{2,}:?$")
_UEBERSICHT = re.compile(r"^##\s+Übersicht\s*$")
_KETTEN_NR = re.compile(r"^K\d+$")
# Ruling 34: Stufen-Zusatz, den render_ketten an die Zustand-Zelle haengt; beim Lesen wieder ab
RENDER_ZUSATZ = re.compile(r"\s*·\s*v:(?:ja|nein|\?) g:(?:läuft|bricht|—)(?: \d\d\.\d\d\. \d\d:\d\d)?\s*$")
_ENDUNG = r"(?:py|pyi|yaml|yml|sql|rs|tsx|ts|mjs|js|json|toml|ps1|psm1|md|sh|cmd)"
_STELLE = re.compile(
    r"(?P<ausl>[\w\-./]*(?:…|\.\.\.)[\w\-./]*):(?P<aa>\d+)(?:-(?P<ab>\d+))?"
    rf"|(?P<pfad>(?:[\w\-.~]+/)*(?:[\w\-.]+\.{_ENDUNG}|Dockerfile)):(?P<a>\d+)(?:-(?P<b>\d+))?"
    r"|(?<![\w.\-/~)]):(?P<ka>\d+)(?:-(?P<kb>\d+))?"
    rf"|(?P<nur>(?:[\w\-.~]+/)+[\w\-.]+\.{_ENDUNG})(?![\w:])"
)
_PORT = re.compile(r":(\d{4,5})(?!\d)")
_HOST = re.compile(r"^(?:\d{1,3}(?:\.\d{1,3}){3}|localhost|host\.docker\.internal|<VM>|VM)$", re.IGNORECASE)
_SQL_NAME = re.compile(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*(\([^)]*\))?$")
_MENSCH = re.compile(r"\b(Mensch|Betreiber|User)\b")
_BEZEICHNER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_PFEIL = ("→", "↔")
_MECH = {
    "http": "HTTP", "https": "HTTP", "post": "HTTP", "get": "HTTP", "head": "HTTP", "put": "HTTP",
    "patch": "HTTP", "delete": "HTTP", "postgrest": "HTTP",
    "sql": "SQL", "after-trigger": "SQL",
    "python": "Python", "import": "Python", "modulimport": "Python",
    "datei": "Datei",
    "env": "Docker/Env",
    "sse": "SSE/IPC", "ipc": "SSE/IPC", "cdp": "SSE/IPC",
}
# Ruling 25: bekannter Wortlaut ohne Messung -> keine_probe ohne Pruefpunkt (verdrahtet setzt nie das Dokument)
_OHNE_PROBE = ("verdrahtet", "gebaut", "code vorhanden", "registriert", "fail-soft")


# ---------- Datentypen ----------

@dataclass(frozen=True)
class HopZeile:
    kette: str
    nr: str
    von: str
    nach: str
    mechanismus: str
    code_stellen: tuple[str, ...]  # Backtick-Spannen der Zelle (ohne Backticks), sonst die ganze Zelle
    belegt_durch: str
    zustand: str
    zeile: int  # 1-basiert in KETTEN.md
    kette_titel: str = ""
    von_nach_roh: str = ""
    code_stelle_roh: str = ""  # ohne gerenderte Anker-Spannen (siehe `render_anker_trennen`)
    code_anker: tuple[str, ...] = ()  # gerenderte Anker-Spannen: nur Hinweis, nie Beleg


@dataclass(frozen=True)
class Stelle:
    pfad: str | None  # repo-relativ, POSIX; None = nicht aufloesbar
    start: int | None
    ende: int | None
    form: str  # voll | kurzform | verzeichnis | anker | auslassung
    roh: str = ""


@dataclass(frozen=True)
class Verankerung:
    """Ergebnis einer Verankerung: Beleg und/oder Pruefpunkt-Art mit Erklaerung."""

    beleg: Beleg | None
    pruefpunkt: str | None = None  # beleg_unsicher | anker_zu_grob | anker_positional | None
    text: str = ""


class _Definition(Protocol):
    document_id: str
    start_line: int
    end_line: int
    source: str


# ---------- Hilfen ----------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ohne_markup(text: str) -> str:
    return " ".join(text.replace("**", "").replace("`", "").split())


def _zellen(zeile: str) -> list[str]:
    """Tabellenzeile in Zellen; `|` innerhalb von Backticks und ein escapter Strich (Backslash + Strich)
    trennen nicht; beides bleibt im Zelltext."""
    zellen: list[str] = []
    aktuell: list[str] = []
    in_code = False
    for zeichen in zeile:
        if zeichen == "`":
            in_code = not in_code
        if zeichen == "|" and not in_code and not (aktuell and aktuell[-1] == "\\"):
            zellen.append("".join(aktuell))
            aktuell = []
        else:
            aktuell.append(zeichen)
    zellen.append("".join(aktuell))
    if zellen and not zellen[0].strip():
        zellen = zellen[1:]
    if zellen and not zellen[-1].strip():
        zellen = zellen[:-1]
    return [z.strip() for z in zellen]


def _spannen(text: str) -> tuple[str, ...]:
    spannen = tuple(s for s in re.findall(r"`([^`]+)`", text) if s.strip())
    if spannen:
        return spannen
    return (text.strip(),) if text.strip() else ()


def _ausserhalb_backticks_teilen(text: str) -> tuple[list[str], bool]:
    teile: list[str] = []
    aktuell: list[str] = []
    in_code, beidseitig = False, False
    for zeichen in text:
        if zeichen == "`":
            in_code = not in_code
        if zeichen in _PFEIL and not in_code:
            beidseitig = beidseitig or zeichen == "↔"
            teile.append("".join(aktuell).strip())
            aktuell = []
        else:
            aktuell.append(zeichen)
    teile.append("".join(aktuell).strip())
    return [t for t in teile if t], beidseitig


# ---------- Tabellen lesen ----------

def render_anker_trennen(code: str) -> tuple[str, tuple[str, ...]]:
    """Gerenderte Paare `` `stelle` `anker` `` (render_ketten) -> Anker-Spanne entfernen.

    Ein Paar ist eine Spanne, die ganz eine Code-Stelle ist, direkt (nur Leerraum dazwischen) gefolgt von
    einer Spanne, die keine Code-Stelle ist. Der Anker darf weder als Stelle gelesen werden (YAML-Identitaet
    mit Dateipfad) noch Namen/Tokens fuer die Bestaetigung liefern. Ergebnis: (Zelle ohne Anker, Anker)."""
    spannen = [(m.start(), m.end(), m.group(1)) for m in re.finditer(r"`([^`]+)`", code)]
    anker_: list[str] = []
    weg: list[tuple[int, int]] = []
    i = 0
    while i + 1 < len(spannen):
        (_, ende, s), (start2, ende2, s2) = spannen[i], spannen[i + 1]
        zwischen = code[ende:start2]
        if (_STELLE.fullmatch(s.strip()) and zwischen and not zwischen.strip()
                and not _STELLE.fullmatch(s2.strip())):
            anker_.append(s2)
            weg.append((ende, ende2))
            i += 2
            continue
        i += 1
    for start, ende in reversed(weg):
        code = code[:start] + code[ende:]
    return code, tuple(anker_)


def hop_tabellen_lesen_mit_pruefpunkten(
    text: str, datei: str | None = None
) -> tuple[list[HopZeile], list[Pruefpunkt]]:
    zeilen: list[HopZeile] = []
    pruefpunkte: list[Pruefpunkt] = []
    kette: str | None = None
    titel = ""
    in_tabelle = False
    for nr, roh in enumerate(text.lstrip("﻿").splitlines(), 1):
        zeile = roh.strip()
        m = _KETTE.match(zeile)
        if m:
            kette, titel, in_tabelle = m.group(1), m.group(2), False
            continue
        if zeile.startswith("#"):
            kette, in_tabelle = None, False
            continue
        if not zeile.startswith("|"):
            in_tabelle = False
            continue
        zellen = _zellen(zeile)
        if tuple(zellen) == KOPF:
            in_tabelle = kette is not None
            continue
        if not in_tabelle or (zellen and all(_TRENNZEILE.match(z) for z in zellen)):
            continue
        if len(zellen) != 6 or not zellen[0] or kette is None:
            pruefpunkte.append(Pruefpunkt("zeile_unlesbar",
                                          f"{kette}: Hop-Zeile mit {len(zellen)} statt 6 Zellen: {zeile[:120]}",
                                          datei, nr, kette))
            continue
        nr_text, von_nach, mech, code, belegt, zustand = zellen
        zustand = RENDER_ZUSATZ.sub("", zustand)
        teile, _ = _ausserhalb_backticks_teilen(von_nach)
        von = teile[0] if teile else von_nach
        nach = teile[-1] if teile else von_nach
        code, code_anker = render_anker_trennen(code)
        zeilen.append(HopZeile(kette, nr_text, von, nach, mech, _spannen(code), belegt, zustand, nr,
                               titel, von_nach, code, code_anker))
    return zeilen, pruefpunkte


def hop_tabellen_lesen(text: str) -> list[HopZeile]:
    return hop_tabellen_lesen_mit_pruefpunkten(text)[0]


def uebersicht_lesen(text: str) -> dict[str, dict[str, str]]:
    """Rohtexte der Tabelle `| # | Kette | Zustand | Reißt bei |` unter `## Übersicht` (Ruling 34).

    Eine gerenderte fuenfte Spalte (`Graph`) wird ignoriert. Ergebnis: Kette -> {titel_roh, zustand_roh,
    reisst_bei_roh}."""
    ergebnis: dict[str, dict[str, str]] = {}
    im_abschnitt = in_tabelle = False
    for roh in text.lstrip("﻿").splitlines():
        zeile = roh.strip()
        if zeile.startswith("#"):
            im_abschnitt, in_tabelle = bool(_UEBERSICHT.match(zeile)), False
            continue
        if not zeile.startswith("|"):
            in_tabelle = False
            continue
        if not im_abschnitt:
            continue
        zellen = _zellen(zeile)
        if tuple(zellen[:4]) == UEBERSICHT_KOPF and len(zellen) in (4, 5):
            in_tabelle = True
            continue
        if not in_tabelle or all(_TRENNZEILE.match(z) for z in zellen):
            continue
        if len(zellen) >= 4 and _KETTEN_NR.match(zellen[0]):
            ergebnis[zellen[0]] = {"titel_roh": zellen[1], "zustand_roh": zellen[2], "reisst_bei_roh": zellen[3]}
    return ergebnis


def namen_aus_text(text: str) -> frozenset[str]:
    """Kandidatennamen (Ruling 22) aus den Backtick-Spannen eines Textes: Bezeichner, Pfade/Ports entfernt,
    `modul.name(...)` -> `name`, Glob-Praefixe (`schedule_*`) zaehlen nicht, mindestens 3 Zeichen."""
    namen: set[str] = set()
    for spanne in re.findall(r"`([^`]+)`", text):
        rein = _STELLE.sub(" ", spanne)
        for m in _BEZEICHNER.finditer(rein):
            if rein.startswith("*", m.end()) or rein.startswith(".*", m.end()):
                continue  # Glob-Praefix (`schedule_*`, `mirofish.*`) ist kein Name
            name = m.group(0).rsplit(".", 1)[-1]
            if len(name) >= 3:
                namen.add(name)
    return frozenset(namen)


def _zeilen_zellen(hz: HopZeile) -> tuple[str, str, str]:
    return (hz.von_nach_roh or f"{hz.von} → {hz.nach}", hz.mechanismus, hz.code_stelle_roh)


def zeilen_namen(hz: HopZeile) -> frozenset[str]:
    """Kandidatennamen einer Zeile (Ruling 22): `namen_aus_text` ueber Von → Nach, Mechanismus und
    Code-Stelle (ohne gerenderte Anker-Spannen)."""
    return frozenset().union(*(namen_aus_text(zelle) for zelle in _zeilen_zellen(hz)))


_STOPPWOERTER = frozenset({
    "approved", "failed", "pending", "sent", "true", "false", "none", "null", "status", "error", "result",
    "events", "execution", "läuft", "laeuft", "bricht",
})
_METHODE_PFAD = re.compile(r"^(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(\S+)")


def _token_normal(spanne: str) -> str:
    t = spanne.strip()
    for _ in range(2):
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
            t = t[1:-1].strip()
        if t.endswith("()"):
            t = t[:-2].strip()
    m = _METHODE_PFAD.match(t)
    return m.group(1) if m else t


def schwache_tokens(konfig: Konfig) -> frozenset[str]:
    """Spacenamen und Segmente der Space-Wurzeln (klein), die nicht als Beleg zaehlen (Ruling 29).

    Dazu gehoeren die Namen aller Geschwister-Spaces, wenn eine Wurzel unter einem Ordner `spaces/` liegt
    (`vibemind-os/spaces/ideas` -> `ideas`)."""
    schwach: set[str] = set()
    for space in konfig.spaces.values():
        schwach.add(space.name.lower())
        teile = [t for t in space.wurzel.strip("/").split("/") if t]
        schwach.update(t.lower() for t in teile)
        if len(teile) >= 2 and teile[-2] == "spaces":
            eltern = konfig.repo.joinpath(*teile[:-1])
            if eltern.is_dir():
                schwach.update(d.name.lower() for d in eltern.iterdir() if d.is_dir())
    return frozenset(schwach)


def tokens_aus_text(text: str, *, schwach: Collection[str] = ()) -> frozenset[str]:
    """Belegtokens (Ruling 27) aus den Backtick-Spannen eines Textes: ganze Spannen, normalisiert (Quotes,
    `()`, `METHODE /pfad` -> Pfad), >= 5 Zeichen, ohne Stoppwoerter, ohne schwache Tokens (Spacenamen,
    Ruling 29) und ohne Spannen, die ganz eine Code-Stelle sind."""
    tokens: set[str] = set()
    for spanne in re.findall(r"`([^`]+)`", text):
        if _STELLE.fullmatch(spanne.strip()):
            continue
        token = _token_normal(spanne)
        if len(token) >= 5 and token.lower() not in _STOPPWOERTER and token.lower() not in schwach:
            tokens.add(token)
    return frozenset(tokens)


def zeilen_tokens(hz: HopZeile, *, schwach: Collection[str] = ()) -> frozenset[str]:
    """Belegtokens einer Zeile (Ruling 27): `tokens_aus_text` ueber Von → Nach, Mechanismus und Code-Stelle
    (ohne gerenderte Anker-Spannen)."""
    return frozenset().union(*(tokens_aus_text(zelle, schwach=schwach) for zelle in _zeilen_zellen(hz)))


def namen_und_tokens_aus_text(
    text: str, konfig: Konfig | None = None, *, schwach: Collection[str] | None = None
) -> tuple[frozenset[str], frozenset[str]]:
    """(Namen, Belegtokens) eines freien Textes (Absatz, Tabellenzeile) nach denselben Regeln wie eine
    Hop-Zeile (Rulings 22/27/29, Ruling 36). Gerenderte Anker-Spannen (`render_anker_trennen`) zaehlen nie;
    schwache Tokens aus `schwach`, sonst aus `konfig` (`schwache_tokens`), sonst keine."""
    ohne_anker, _ = render_anker_trennen(text)
    if schwach is None:
        schwach = schwache_tokens(konfig) if konfig is not None else frozenset()
    return namen_aus_text(ohne_anker), tokens_aus_text(ohne_anker, schwach=schwach)


# ---------- Code-Stellen ----------

def _datei_index(repo: Path, konfig: Konfig) -> dict[str, list[str]]:
    """Dateiname -> repo-relative Pfade; Ausschluss aus der Konfig, vor dem Betreten (korpus.dateien)."""
    index: dict[str, list[str]] = {}
    for p in korpus.dateien(repo, konfig, ("",)):
        try:
            rel = p.relative_to(repo).as_posix()
        except ValueError:
            continue
        index.setdefault(p.name, []).append(rel)
    return index


def _voll_aufloesen(pfad: str, repo: Path, index: Mapping[str, list[str]] | None) -> str:
    if not anker.pfad_sicher(pfad):
        return pfad  # Final-Fix 22: nicht im Dateisystem nachsehen; die Verankerung lehnt ihn ab
    if (repo / pfad).is_file():
        return pfad
    if not pfad.startswith("vibemind-os/") and (repo / "vibemind-os" / pfad).is_file():
        return f"vibemind-os/{pfad}"
    if index is not None:
        treffer = [p for p in index.get(pfad.rsplit("/", 1)[-1], []) if p.endswith("/" + pfad)]
        if len(treffer) == 1:
            return treffer[0]
    return pfad  # bleibt stehen; die Verankerung meldet die fehlende Datei


def _aufloesen(
    zellen_text: str, voriger_pfad: str | None, repo: Path,
    index_holen: Callable[[], Mapping[str, list[str]]], bekannte: Mapping[str, str] | None,
) -> tuple[list[Stelle], str | None]:
    ergebnis: list[Stelle] = []
    voriger = voriger_pfad
    for spanne in _spannen(zellen_text):
        for m in _STELLE.finditer(spanne):
            roh = m.group(0)
            if m.group("ausl") is not None:
                ergebnis.append(Stelle(None, int(m.group("aa")), _int(m.group("ab")), "auslassung", roh))
                voriger = None  # nach einer Auslassung ist der Bezug fuer Kurzformen verloren
                continue
            if m.group("ka") is not None:
                ergebnis.append(Stelle(voriger, int(m.group("ka")), _int(m.group("kb")), "kurzform", roh))
                continue
            pfad = m.group("pfad") or m.group("nur")
            if pfad.startswith("~"):
                continue
            start = _int(m.group("a"))
            ende = _int(m.group("b"))
            if "/" in pfad:
                stelle = Stelle(_voll_aufloesen(pfad, repo, index_holen()), start, ende, "voll", roh)
            else:
                stelle = _name_aufloesen(pfad, start, ende, roh, voriger, repo, index_holen, bekannte)
            ergebnis.append(stelle)
            if stelle.pfad is not None:
                voriger = stelle.pfad
    return ergebnis, voriger


def _name_aufloesen(
    name: str, start: int | None, ende: int | None, roh: str, voriger: str | None, repo: Path,
    index_holen: Callable[[], Mapping[str, list[str]]], bekannte: Mapping[str, str] | None,
) -> Stelle:
    if voriger and "/" in voriger and anker.pfad_sicher(voriger):
        kandidat = f"{voriger.rsplit('/', 1)[0]}/{name}"
        if (repo / kandidat).is_file():
            return Stelle(kandidat, start, ende, "verzeichnis", roh)
    treffer = index_holen().get(name, [])
    if len(treffer) == 1:
        return Stelle(treffer[0], start, ende, "anker", roh)
    frueher = (bekannte or {}).get(name)
    if frueher and anker.pfad_sicher(frueher) and (repo / frueher).is_file():
        return Stelle(frueher, start, ende, "anker", roh)
    return Stelle(None, start, ende, "anker", roh)


def _int(wert: str | None) -> int | None:
    return int(wert) if wert is not None else None


def _index_holer(repo: Path, konfig: Konfig | None,
                 index: Mapping[str, list[str]] | None) -> Callable[[], Mapping[str, list[str]]]:
    gebaut: list[Mapping[str, list[str]]] = [index] if index is not None else []

    def holen() -> Mapping[str, list[str]]:
        if not gebaut:
            gebaut.append(_datei_index(repo, konfig if konfig is not None else konfig_modul.laden()))
        return gebaut[0]

    return holen


def code_stellen_aufloesen(
    zellen_text: str, voriger_pfad: str | None, repo: Path, *,
    index: Mapping[str, list[str]] | None = None, bekannte: Mapping[str, str] | None = None,
    konfig: Konfig | None = None,
) -> list[Stelle]:
    """Alle Belege einer Code-Stelle-Zelle in Reihenfolge; wirft nie.

    `index` (Dateiname -> Pfade) wird sonst bei Bedarf gebaut (Ausschluss aus `konfig`, sonst aus der
    Standard-Konfig); `bekannte` (Dateiname -> frueher im Dokument voll genannter Pfad) hilft, wenn der
    Name im Repo nicht eindeutig ist.
    """
    return _aufloesen(zellen_text, voriger_pfad, repo, _index_holer(repo, konfig, index), bekannte)[0]


# ---------- Knoten, Mechanismus, Zustand ----------

def knotentyp_erkennen(
    text: str, *, mcp_namen: Collection[str] = (), schemata: Collection[str] = DB_SCHEMATA
) -> tuple[str, str]:
    """(Knotentyp, Name) fuer eine Von- bzw. Nach-Zelle. Dienst mit Host/IP statt Namen -> Name `:port`."""
    klar = _ohne_markup(text)
    port = _PORT.search(klar)
    if port:
        name = klar[: port.start()].strip(" -:/")
        if not name or _HOST.match(name.split()[-1].rsplit("/", 1)[-1]):
            return "dienst", f":{port.group(1)}"
        return "dienst", name
    schema_klein = {s.lower() for s in schemata}
    for kandidat in (*_spannen(text), klar):
        m = _SQL_NAME.match(kandidat.strip())
        if m and m.group(1).lower() in schema_klein:
            name = f"{m.group(1)}.{m.group(2)}".lower()
            return ("sql_funktion" if m.group(3) else "tabelle"), name
    for spanne in re.findall(r"`([^`]+)`", text):
        if spanne.strip() in mcp_namen:
            return "mcp_werkzeug", spanne.strip()
    if _MENSCH.search(klar):
        return "mensch", klar
    return "komponente", klar


def _schluessel_muster(woerter: Sequence[str]) -> re.Pattern[str]:
    """Schluesselwoerter, gross/klein egal; beginnt eins mit einem Buchstaben, muss es am Wortanfang stehen
    (`Klasse` ist kein `SSE`, `psql` kein `SQL`, `Katalog` kein `Log`)."""
    teile = [(r"(?<![^\W\d_])" if w[:1].isalpha() else "") + re.escape(w) for w in woerter]
    return re.compile("|".join(teile), re.IGNORECASE)


# Ruling 32: ganze Zelle durchsuchen, Vorrang in dieser Reihenfolge
_MECH_VORRANG: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("MCP", _schluessel_muster(("MCP", "mcp__", "Werkzeug", "tool"))),
    ("SQL", re.compile(
        _schluessel_muster(("SQL", "SECURITY DEFINER", "RPC", "Supabase", "INSERT", "SELECT", "UPDATE")).pattern
        + r"|(?<![^\W\d_])(?-i:(?:" + "|".join(sorted(DB_SCHEMATA)) + r")\.[A-Za-z_])", re.IGNORECASE)),
    ("HTTP", _schluessel_muster(("HTTP", "GET ", "POST ", "PUT ", "DELETE ", "REST", "Webhook", "/api/"))),
    ("SSE/IPC", _schluessel_muster(("SSE", "IPC", "WebSocket", "Event-Bus", "Redis"))),
    ("Docker/Env", _schluessel_muster(("Docker", "Env", "Swarm", "Compose", "Container"))),
    ("Datei", _schluessel_muster(("Datei", "Log", ".json", ".md"))),
    ("Python", _schluessel_muster(("Python", "Import", "Aufruf"))),
)


def mechanismus_typ(text: str) -> str:
    """MCP | SQL | HTTP | SSE/IPC | Docker/Env | Datei | Python | unbestimmt.

    Ruling 32: Schluesselwoerter irgendwo in der Zelle, Vorrang in der Reihenfolge von `_MECH_VORRANG`
    (gross/klein egal, ausser der Schema-Regel `marketing.`/`sales.`/`public.`). Ohne Treffer gilt das
    alte Schluesselwort am Zellenanfang (`after-trigger`, `postgrest`, `cdp`, ...).
    """
    klar = _ohne_markup(text)
    for typ, muster in _MECH_VORRANG:
        if muster.search(klar):
            return typ
    m = re.match(r"[\w./-]+", klar)
    if not m:
        return "unbestimmt"
    wort = m.group(0).lower()
    if wort.startswith("mcp"):
        return "MCP"
    if wort.startswith("docker"):
        return "Docker/Env"
    return _MECH.get(wort, "unbestimmt")


def zustand_aus_spalte(text: str) -> tuple[dict[str, object], bool]:
    """(Hop-Zustand, Wortlaut bekannt). `bricht` irgendwo gewinnt (vorsichtig), dann `läuft` am Anfang,
    dann `aus` (am Anfang oder letztes Wort eines Komma-Abschnitts); `verdrahtet` & Co. und `nie … bewiesen|
    gelaufen` sind bekannt ohne Messung. Gesetzt wird nur `gemessen` bzw. `aus`."""
    zustand = hop_zustand_neu()
    klar = _ohne_markup(text).lower()
    abschnitte = [a.split() for a in klar.split(",")]
    if re.search(r"\bbricht\b", klar):
        zustand["gemessen"] = {"wert": "bricht", "zeit": None}
    elif klar.startswith(("läuft", "laeuft")):
        zustand["gemessen"] = {"wert": "laeuft", "zeit": None}
    elif re.match(r"aus\b", klar) or any(a and a[-1].strip(".;:!") == "aus" for a in abschnitte):
        zustand["aus"] = True
    elif not (re.match(r"nie\b.*\b(bewiesen|gelaufen)\b", klar) or klar.startswith(_OHNE_PROBE)):
        return zustand, False
    return zustand, True


# ---------- Verankerung: Python ----------

def _space_von(pfad: str, konfig: Konfig) -> str | None:
    beste: tuple[int, str] | None = None
    for s in konfig.spaces.values():
        w = s.wurzel.strip("/")
        if pfad == w or pfad.startswith(w + "/"):
            if beste is None or len(w) > beste[0]:
                beste = (len(w), s.name)
    return beste[1] if beste else None


def _datei_beleg(pfad: str, zeile: int | None, repo: Path, commit: str) -> Beleg | None:
    try:
        return anker.beleg_fuer_datei(pfad, zeile, repo, commit)
    except OSError:
        return None


def _abstand(d: _Definition, start: int, ende: int) -> int:
    if d.end_line < start:
        return start - d.end_line
    if d.start_line > ende:
        return d.start_line - ende
    return 0


def _quelltext_mit_dekoratoren(datei: Path) -> Callable[[_Definition], str]:
    """Quelltext einer Definition inklusive ihrer Dekoratoren (`@app.post("/api/x")` gehoert zur Route);
    manifest.source beginnt erst bei `def`. Faellt auf `source` zurueck, wenn die Datei nicht parsbar ist."""
    try:
        text = datei.read_text(encoding="utf-8", errors="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
            baum = ast.parse(text)
    except (OSError, SyntaxError, ValueError, RecursionError):
        return lambda d: d.source
    zeilen = text.splitlines()
    beginn = {
        k.lineno: min([k.lineno, *(dek.lineno for dek in k.decorator_list)])
        for k in ast.walk(baum) if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return lambda d: "\n".join(zeilen[beginn.get(d.start_line, d.start_line) - 1:d.end_line])


def _symbol_waehlen(
    definitionen: Sequence[_Definition], start: int, ende: int, namen: Collection[str] | None,
    tokens: Collection[str] | None = None, quelltext: Callable[[_Definition], str] | None = None,
) -> tuple[_Definition | None, str, str | None]:
    """(Definition, Grund, umschliessende id). Grund: name | umschliessend | beleg_im_symbol | unsicher |
    modulebene. Ruling 27: das umschliessende Symbol gilt auch, wenn sein Quelltext ein Belegtoken der
    Zeile woertlich enthaelt."""
    zaehler = Counter(d.document_id for d in definitionen)
    eindeutig = [d for d in definitionen if zaehler[d.document_id] == 1]
    if namen:
        benannt = [d for d in eindeutig if d.document_id.rsplit(".", 1)[-1] in namen]
        if benannt:
            return min(benannt, key=lambda d: (_abstand(d, start, ende), d.end_line - d.start_line,
                                               d.start_line)), "name", None
    for z in range(start, ende + 1):
        umschl = [d for d in eindeutig if d.start_line <= z <= d.end_line]
        if umschl:
            innerste = min(umschl, key=lambda d: d.end_line - d.start_line)
            if not namen and not tokens:  # Zeile nennt gar nichts (auch keinen Glob)
                return innerste, "umschliessend", None
            text = quelltext(innerste) if quelltext is not None else innerste.source
            if any(tok in text for tok in tokens or () if "*" not in tok):
                return innerste, "beleg_im_symbol", None
            return None, "unsicher", innerste.document_id
    return None, "modulebene", None


def _py_verankern(stelle: Stelle, konfig: Konfig, manifeste: Mapping[str, "Manifest"], commit: str,
                  namen: Collection[str] | None, tokens: Collection[str] | None = None) -> Verankerung:
    assert stelle.pfad is not None
    repo, pfad, start = konfig.repo, stelle.pfad, stelle.start
    if start is None:
        return Verankerung(_datei_beleg(pfad, None, repo, commit))
    ende = max(start, stelle.ende or start)
    space = _space_von(pfad, konfig)
    manifest = manifeste.get(space) if space else None
    definitionen: Sequence[_Definition]
    if manifest is not None:
        rel = anker.korpus_relativ(manifest, repo, pfad)
        alle = Counter(d.document_id for d in manifest.documents)
        definitionen = [d for d in manifest.documents if d.file == rel and alle[d.document_id] == 1]
    elif space is not None:
        return Verankerung(_datei_beleg(pfad, start, repo, commit))  # Space ohne Manifest
    else:
        ast_defs = anker.ast_definitionen(repo / pfad, anker.modulpfad(pfad))
        if ast_defs is None:
            return Verankerung(_datei_beleg(pfad, start, repo, commit))
        definitionen = ast_defs
    wahl, grund, umschl = _symbol_waehlen(definitionen, start, ende, namen, tokens,
                                          _quelltext_mit_dekoratoren(repo / pfad) if tokens else None)
    if wahl is not None:
        zeile = max(start, wahl.start_line) if _abstand(wahl, start, ende) == 0 else wahl.start_line
        return Verankerung(Beleg("symbol", pfad, wahl.document_id, _sha(wahl.source), commit, zeile))
    kandidaten = ", ".join(sorted(namen or ())[:8]) or "-"
    if grund == "unsicher":
        return Verankerung(None, "beleg_unsicher",
                           f"umschliessend {umschl}, aber kein Symbol der Datei heisst wie ein Name der Zeile "
                           f"({kandidaten}) und sein Quelltext enthaelt kein Belegtoken der Zeile")
    if manifest is None:  # Modulebene ausserhalb der Spaces: Dateibeleg (Ruling 26)
        return Verankerung(_datei_beleg(pfad, start, repo, commit))
    if namen or tokens:
        return Verankerung(None, "beleg_unsicher",
                           f"Modulebene, kein Symbol der Datei heisst wie ein Name der Zeile ({kandidaten})")
    return Verankerung(None)


# ---------- Verankerung: SQL ----------

def sql_name_an(datei: Path, zeile: int) -> str | None:
    """Name (`schema.name`, klein) der CREATE-Anweisung, die Zeile `zeile` umfasst; sonst None."""
    try:
        text = Path(datei).read_bytes().decode("utf-8").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError):
        return None
    maske = sql_extraktor._maskieren(text)
    for m in sql_extraktor._CREATE.finditer(maske):
        anfang = text.count("\n", 0, m.start()) + 1
        schluss = anfang + sql_extraktor._anweisung(text, m.start(), maske).count("\n")
        if anfang <= zeile <= schluss:
            return f"{sql_extraktor._entquoten(m.group(2))}.{sql_extraktor._entquoten(m.group(3))}"
    return None


# ---------- Verankerung: YAML/TOML ----------

def _ende(knoten: yaml.Node) -> int:
    """Letzte Zeile (0-basiert) eines Knotens; Blocksammlungen ueber ihre Kinder (ihre end_mark zeigt
    auf das naechste Token)."""
    if isinstance(knoten, yaml.ScalarNode) or getattr(knoten, "flow_style", False):
        zeile = knoten.end_mark.line
        if knoten.end_mark.column == 0 and zeile > knoten.start_mark.line:
            zeile -= 1
        return zeile
    kinder: list[yaml.Node] = []
    if isinstance(knoten, yaml.MappingNode):
        for k, v in knoten.value:
            kinder.extend((k, v))
    elif isinstance(knoten, yaml.SequenceNode):
        kinder = list(knoten.value)
    return max((_ende(k) for k in kinder), default=knoten.start_mark.line)


def _eintrag_token(eintraege: Sequence[object], i: int,
                   identitaet: Callable[[object], Callable[[str], str | None]]) -> tuple[Token, bool]:
    """Identitaet `(schluessel, wert)` des i-ten Listeneintrags, falls eindeutig, sonst Position (positional)."""
    for k in anker.IDENTITAETS_SCHLUESSEL:
        wert = identitaet(eintraege[i])(k)
        if wert is not None and sum(1 for e in eintraege if identitaet(e)(k) == wert) == 1:
            return (k, wert), False
    return i, True


def _yaml_id(knoten: object) -> Callable[[str], str | None]:
    return lambda k: anker.yaml_identitaet(knoten, k) if isinstance(knoten, yaml.Node) else None


def _toml_id(eintrag: object) -> Callable[[str], str | None]:
    return lambda k: anker.toml_identitaet(eintrag, k)


_Kind = tuple[int, int, "Token | None", yaml.Node, bool]


def _yaml_kinder(knoten: yaml.Node) -> list[_Kind]:
    if isinstance(knoten, yaml.MappingNode):
        return [(k.start_mark.line, _ende(v), k.value if isinstance(k, yaml.ScalarNode) else None, v, False)
                for k, v in knoten.value]
    if isinstance(knoten, yaml.SequenceNode):
        kinder: list[_Kind] = []
        for i, v in enumerate(knoten.value):
            token, positional = _eintrag_token(knoten.value, i, _yaml_id)
            kinder.append((v.start_mark.line, _ende(v), token, v, positional))
        return kinder
    return []


def _yaml_zeilen_pfad(wurzel: yaml.Node, zeile: int) -> list[tuple[Token, bool]]:
    """Pfad zur kleinsten Sammlung, die Zeile `zeile` (0-basiert) traegt. Kommentar-/Leerzeilen gehoeren
    zum naechsten Schluessel darunter im selben Elternknoten, sonst zum vorigen Geschwister."""
    knoten, pfad = wurzel, []
    while True:
        kinder = _yaml_kinder(knoten)
        if not kinder:
            return pfad
        kind = next((c for c in kinder if c[0] <= zeile <= c[1]), None)
        if kind is None:
            kind = next((c for c in kinder if c[0] > zeile), None) or [c for c in kinder if c[1] < zeile][-1]
            zeile = kind[0]
        _, _, token, unter, positional = kind
        if token is None or not isinstance(unter, (yaml.MappingNode, yaml.SequenceNode)):
            return pfad  # Skalar: die umschliessende Sammlung ist der Anker
        pfad.append((token, positional))
        if isinstance(knoten, yaml.MappingNode) and unter.start_mark.line > zeile:
            return pfad  # Schluesselzeile einer Sammlung
        knoten = unter


_TOML_KOPF = re.compile(r"^\s*(\[\[?)\s*([^\[\]]+?)\s*\]\]?\s*(?:#.*)?$")
_TOML_TEIL = r"""(?:[A-Za-z0-9_\-]+|"[^"]*"|'[^']*')"""
_TOML_SCHLUESSEL = re.compile(rf"^\s*({_TOML_TEIL}(?:\s*\.\s*{_TOML_TEIL})*)\s*=(.*)$")
_TOML_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'')


def _toml_teile(text: str) -> tuple[str, ...]:
    return tuple(t.strip().strip("\"'") for t in re.split(r"\.(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", text))


def _klammern(text: str) -> int:
    ohne = _TOML_STRING.sub("", text).split("#", 1)[0]
    return ohne.count("[") + ohne.count("{") - ohne.count("]") - ohne.count("}")


@dataclass
class _TomlEintrag:
    art: str  # kopf | schluessel
    start: int
    ende: int
    knoten: tuple[Token, ...]  # Pfad der kleinsten Sammlung (Positionen fuer Tabellen-Arrays)


def _toml_eintraege(text: str) -> list[_TomlEintrag]:
    eintraege: list[_TomlEintrag] = []
    tabelle: tuple[Token, ...] = ()
    aot: dict[tuple[str, ...], int] = {}
    offen: _TomlEintrag | None = None
    tiefe = 0
    mehrzeilig: str | None = None
    for nr, roh in enumerate(text.splitlines(), 1):
        if mehrzeilig is not None or tiefe > 0:
            if offen is not None:
                offen.ende = nr
            if mehrzeilig is not None:
                if roh.count(mehrzeilig) % 2 == 1:
                    mehrzeilig = None
            else:
                tiefe += _klammern(roh)
            continue
        kopf = _TOML_KOPF.match(roh)
        if kopf:
            teile = _toml_teile(kopf.group(2))
            if kopf.group(1) == "[[":
                aot[teile] = aot.get(teile, 0) + 1
                for k in [k for k in aot if len(k) > len(teile) and k[: len(teile)] == teile]:
                    del aot[k]
            tokens: list[Token] = []
            for i, teil in enumerate(teile):
                tokens.append(teil)
                if teile[: i + 1] in aot:
                    tokens.append(aot[teile[: i + 1]] - 1)
            tabelle = tuple(tokens)
            eintraege.append(_TomlEintrag("kopf", nr, nr, tabelle))
            offen = None
            continue
        sm = _TOML_SCHLUESSEL.match(roh)
        if sm:
            teile = _toml_teile(sm.group(1))
            wert = sm.group(2).strip()
            sammlung = wert.startswith(("[", "{"))
            offen = _TomlEintrag("schluessel", nr, nr, tabelle + (teile if sammlung else teile[:-1]))
            eintraege.append(offen)
            for zeichen in ('"""', "'''"):
                if wert.count(zeichen) % 2 == 1:
                    mehrzeilig = zeichen
            if mehrzeilig is None:
                tiefe = max(0, _klammern(wert))
    return eintraege


def _toml_zeilen_pfad(eintraege: list[_TomlEintrag], zeile: int) -> tuple[Token, ...]:
    for e in eintraege:
        if e.start <= zeile <= e.ende:
            return e.knoten
    koepfe = [i for i, e in enumerate(eintraege) if e.art == "kopf" and e.start < zeile]
    tabelle = eintraege[koepfe[-1]].knoten if koepfe else ()
    abschnitt: list[_TomlEintrag] = []
    for e in eintraege[(koepfe[-1] + 1) if koepfe else 0:]:
        if e.art == "kopf":
            break
        abschnitt.append(e)
    unten = next((e for e in abschnitt if e.start > zeile), None)
    if unten is not None:
        return unten.knoten
    oben = [e for e in abschnitt if e.ende < zeile]
    return oben[-1].knoten if oben else tabelle


def _toml_identitaet(daten: object, tokens: tuple[Token, ...]) -> list[tuple[Token, bool]] | None:
    wert, aus = daten, []
    for t in tokens:
        if isinstance(t, int):
            if not isinstance(wert, list) or t >= len(wert):
                return None
            aus.append(_eintrag_token(wert, t, _toml_id))
            wert = wert[t]
        else:
            if not isinstance(wert, dict) or t not in wert:
                return None
            aus.append((t, False))
            wert = wert[t]
    return aus


def _gemeinsam(a: list[tuple[Token, bool]], b: list[tuple[Token, bool]]) -> list[tuple[Token, bool]]:
    aus = []
    for x, y in zip(a, b, strict=False):
        if x[0] != y[0]:
            break
        aus.append(x)
    return aus


def schluessel_anker(datei: Path, start: int, ende: int | None = None) -> tuple[str | None, str | None]:
    """(Pfad, Pruefpunkt-Art) fuer den Bereich `start..ende` (1-basiert) einer YAML-/TOML-Datei.

    Pfad = kleinste Sammlung, die den ganzen Bereich umfasst (nie ein nackter Skalar). Art:
    `anker_positional` (Pfad mit Listenposition), `anker_zu_grob` (Pfad None: Wurzel oder oberste
    Sammlung) oder None. (None, None) heisst: nicht lesbar -> Rueckfall auf `datei`.
    """
    try:
        text = Path(datei).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, None
    schluss = max(start, ende or start)
    if start < 1 or schluss > len(text.splitlines()):
        return None, None
    pfade: list[list[tuple[Token, bool]]] = []
    try:
        if Path(datei).suffix.lower() == ".toml":
            daten = tomllib.loads(text)
            eintraege = _toml_eintraege(text)
            for z in (start, schluss):
                p = _toml_identitaet(daten, _toml_zeilen_pfad(eintraege, z))
                if p is None:
                    return None, None
                pfade.append(p)
            wurzel_ist_liste = False
        else:
            wurzel = yaml.compose(text, Loader=yaml.SafeLoader)
            if not isinstance(wurzel, (yaml.MappingNode, yaml.SequenceNode)):
                return None, None
            pfade = [_yaml_zeilen_pfad(wurzel, z - 1) for z in (start, schluss)]
            wurzel_ist_liste = isinstance(wurzel, yaml.SequenceNode)
    except (yaml.YAMLError, tomllib.TOMLDecodeError, RecursionError, ValueError):
        return None, None
    lca = _gemeinsam(pfade[0], pfade[1])
    if len(lca) < (1 if wurzel_ist_liste else 2):
        return None, "anker_zu_grob"
    pfad = anker.pfad_text([t for t, _ in lca])
    return pfad, ("anker_positional" if any(p for _, p in lca) else None)


def schluesselpfad_an(datei: Path, zeile: int) -> str | None:
    """Kurzform von `schluessel_anker` fuer eine Zeile: nur der Pfad."""
    return schluessel_anker(datei, zeile)[0]


def _benannte_eintraege(datei: Path, tokens: Collection[str]) -> list[tuple[list[Token], int, int]]:
    """Listeneintraege der Datei, deren Identitaetswert einem Zeilen-Token gleicht: (Pfad, Start, Ende),
    Zeilen 1-basiert."""
    gefunden: list[tuple[list[Token], int, int]] = []
    try:
        text = datei.read_text(encoding="utf-8")
        if datei.suffix.lower() == ".toml":
            daten = tomllib.loads(text)
            eintraege = _toml_eintraege(text)
            koepfe = [e for e in eintraege if e.art == "kopf"]
            letzte = len(text.splitlines())
            for i, e in enumerate(koepfe):
                if not e.knoten or not isinstance(e.knoten[-1], int):
                    continue
                pfad = _toml_identitaet(daten, e.knoten)
                if pfad and isinstance(pfad[-1][0], tuple) and _identitaet_passt(pfad[-1][0][1], tokens):
                    ende = koepfe[i + 1].start - 1 if i + 1 < len(koepfe) else letzte
                    gefunden.append(([t for t, _ in pfad], e.start, ende))
            return gefunden
        wurzel = yaml.compose(text, Loader=yaml.SafeLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, tomllib.TOMLDecodeError, RecursionError, ValueError):
        return []

    def laufen(knoten: yaml.Node, pfad: list[Token]) -> None:
        if isinstance(knoten, yaml.MappingNode):
            for kk, vv in knoten.value:
                if isinstance(kk, yaml.ScalarNode):
                    laufen(vv, [*pfad, kk.value])
        elif isinstance(knoten, yaml.SequenceNode):
            for i, vv in enumerate(knoten.value):
                token, _ = _eintrag_token(knoten.value, i, _yaml_id)
                if isinstance(token, tuple) and _identitaet_passt(token[1], tokens):
                    gefunden.append(([*pfad, token], vv.start_mark.line + 1, _ende(vv) + 1))
                laufen(vv, [*pfad, token])

    if wurzel is not None:
        laufen(wurzel, [])
    return gefunden


def _ist_glob(token: str) -> bool:
    return "*" in token


def _identitaet_passt(wert: str, tokens: Collection[str]) -> bool:
    """Identitaetswert gleicht einem Token exakt oder einem Glob-Token per fnmatch (Ruling 30)."""
    return any(fnmatch.fnmatchcase(wert, t) if _ist_glob(t) else wert == t for t in tokens)


def _yaml_bestaetigt(datei: Path, pfad: list[Token], tokens: Collection[str]) -> bool:
    """Ruling 29/30: Identitaetswert im Pfad gleicht einem Token (exakt oder Glob), ein Schluessel im Pfad
    gleicht einem Token exakt, oder der serialisierte Knotentext enthaelt ein Nicht-Glob-Token woertlich."""
    exakt = [t for t in tokens if not _ist_glob(t)]
    for t in pfad:
        if isinstance(t, tuple) and _identitaet_passt(t[1], tokens):
            return True
        if isinstance(t, str) and t in exakt:
            return True
    if not exakt:
        return False
    gefunden, wert = anker.yaml_knoten_wert(datei, anker.pfad_text(pfad))
    if not gefunden:
        return False
    text = anker.knoten_text(wert)
    return any(tok in text for tok in exakt)


def _yaml_verankern(stelle: Stelle, repo: Path, commit: str, tokens: Collection[str] | None,
                    namen: Collection[str] | None = None) -> Verankerung | None:
    """yaml_key namensgefuehrt (Ruling 29/30); None = nicht lesbar, Rueckfall auf `datei`.

    Reihenfolge: (i) der zeilenbasierte Knoten, wenn Pfad oder eigener Text ihn bestaetigen; ohne
    Bestaetigung nur, wenn die Zeile gar nichts nennt (`namen` und `tokens` leer; schwache, kurze oder
    Stopp-Tokens befreien nicht); (ii) sonst der naechste Listeneintrag, dessen Identitaetswert einem Token
    gleicht (exakt oder Glob); (iii) sonst `beleg_unsicher` bzw. `anker_zu_grob`.
    """
    assert stelle.pfad is not None and stelle.start is not None
    datei, pfad, zeile = repo / stelle.pfad, stelle.pfad, stelle.start
    ende = max(zeile, stelle.ende or zeile)
    schluessel, art = schluessel_anker(datei, zeile, stelle.ende)
    lca = anker._tokens(schluessel) if schluessel else None
    nennt_nichts = not namen and not tokens
    ohne_digest = False
    if lca is not None and (nennt_nichts or (tokens and _yaml_bestaetigt(datei, lca, tokens))):
        digest = anker.yaml_knoten_digest(datei, schluessel or "")
        if digest is not None:
            return Verankerung(Beleg("yaml_key", pfad, schluessel or "", digest, commit, zeile), art,
                               f"Listenposition statt Identitaet in {schluessel}" if art else "")
        ohne_digest = True
    if tokens:
        benannt = _benannte_eintraege(datei, tokens)
        if benannt:
            def abstand(e: tuple[list[Token], int, int]) -> tuple[int, int]:
                _, s, en = e
                return (zeile - en if en < zeile else s - ende if s > ende else 0), en - s

            naechster, start, _ = min(benannt, key=abstand)
            text = anker.pfad_text(naechster)
            digest = anker.yaml_knoten_digest(datei, text)
            if digest is not None:
                return Verankerung(Beleg("yaml_key", pfad, text, digest, commit, start))
    if art == "anker_zu_grob":
        return Verankerung(None, art, "nur Wurzel oder oberste Sammlung umfasst die Stelle")
    if lca is None:
        return None
    if ohne_digest:
        return Verankerung(None, "beleg_unsicher",
                           f"Knoten {schluessel} ist bestaetigt, aber nicht verankerbar (kein Digest)")
    return Verankerung(None, "beleg_unsicher",
                       f"Knoten {schluessel} nennt kein Token der Zeile ({', '.join(sorted(tokens or ())[:6])})")


# ---------- Verankerung: Einstieg ----------

def stelle_verankern_mit_grund(
    stelle: Stelle, konfig: Konfig, manifeste: Mapping[str, "Manifest"],
    sql_defs: Mapping[str, "SqlDefinition"], commit: str, *, namen: Collection[str] | None = None,
    tokens: Collection[str] | None = None,
) -> Verankerung:
    """Wie `stelle_verankern`, mit Pruefpunkt-Art (beleg_unsicher, anker_zu_grob, anker_positional, pfad_unsicher)."""
    if stelle.pfad is None:
        return Verankerung(None)
    repo, pfad, zeile = konfig.repo, stelle.pfad, stelle.start
    try:  # Final-Fix 22: `..`/absolut/Link nach aussen -> nicht messbar, vor jedem Lesen
        anker.repo_pfad(repo, pfad)
    except anker.UnsichererPfad:
        return Verankerung(None, "pfad_unsicher", f"Pfad {pfad} zeigt aus dem Repo heraus; nicht messbar")
    if not (repo / pfad).is_file():
        return Verankerung(None)
    endung = Path(pfad).suffix.lower()
    if endung == ".py":
        return _py_verankern(stelle, konfig, manifeste, commit, namen, tokens)
    if endung == ".sql" and zeile is not None:
        name = sql_name_an(repo / pfad, zeile)
        d = sql_defs.get(name) if name else None
        if d is not None:
            return Verankerung(Beleg("sql", d.datei, d.name, d.digest, commit, d.zeile))
    elif endung in (".yaml", ".yml", ".toml") and zeile is not None:
        v = _yaml_verankern(stelle, repo, commit, tokens, namen)
        if v is not None:
            return v
    return Verankerung(_datei_beleg(pfad, zeile, repo, commit))


def stelle_verankern(
    stelle: Stelle, konfig: Konfig, manifeste: Mapping[str, "Manifest"],
    sql_defs: Mapping[str, "SqlDefinition"], commit: str, *, namen: Collection[str] | None = None,
    tokens: Collection[str] | None = None,
) -> Beleg | None:
    """Beleg fuer eine aufgeloeste Stelle; None, wenn sie sich nicht (sicher) verankern laesst.

    `namen` sind die Kandidatennamen der Zeile (Ruling 22); None/leer = die Zeile nennt keine Namen.
    `tokens` sind ihre Belegtokens (Ruling 27, `zeilen_tokens`).
    """
    return stelle_verankern_mit_grund(stelle, konfig, manifeste, sql_defs, commit, namen=namen,
                                      tokens=tokens).beleg


# ---------- Seed ----------

def _mcp_namen(konfig: Konfig, manifeste: Mapping[str, "Manifest"]) -> set[str]:
    from .extraktoren import mcp_werkzeuge

    namen: set[str] = set()
    for name, space in konfig.spaces.items():
        manifest = manifeste.get(name)
        if manifest is not None and (konfig.repo / space.wurzel).is_dir():
            namen.update(w.name for w in mcp_werkzeuge.mcp_werkzeuge(space, konfig, manifest))
    return namen


def _knoten(text: str, mcp_namen: Collection[str], schemata: Collection[str],
            sql_defs: Mapping[str, "SqlDefinition"], dienst_je_port: Mapping[str, str]) -> Knoten:
    typ, name = knotentyp_erkennen(text, mcp_namen=mcp_namen, schemata=schemata)
    if typ == "tabelle" and getattr(sql_defs.get(name), "art", None) == "function":
        typ = "sql_funktion"
    attribute: dict[str, str] = {}
    if typ == "dienst":
        port = _PORT.search(_ohne_markup(text))
        if port:
            attribute["port"] = port.group(1)
            if name.startswith(":"):
                name = dienst_je_port.get(port.group(1), name)
    return Knoten(f"{typ}:{name}", typ, name, attribute=attribute)


_PROSA_RAND = " ,·←→⚠;"


def code_prosa(code: str) -> str:
    """Menschlicher Text der Code-Stelle-Zelle ohne die Stellen selbst (Ruling 34: Menschentext bleibt).

    Spannen, die ganz eine Stelle sind, und leere Klammern fallen weg; Leerraum wird normalisiert; Trenner am
    Rand und ein fuehrendes `—` fallen weg. Idempotent auf gerenderten Zellen (`<Belege> — <Prosa>`), deren
    Anker-Spannen `render_anker_trennen` schon entfernt hat."""
    rest = re.sub(r"`([^`]+)`", lambda m: " " if _STELLE.fullmatch(m.group(1).strip()) else m.group(0), code)
    rest = re.sub(r"\(\s*[,;·]*\s*\)", " ", rest)
    rest = " ".join(rest.split())
    rest = re.sub(r"\s+([,;)])", r"\1", rest)
    rest = re.sub(r"\(\s+", "(", rest)
    rest = rest.lstrip(_PROSA_RAND)
    if rest.startswith("—"):
        rest = rest[1:]
    return rest.strip(_PROSA_RAND)


def _stelle_voll(st: Stelle) -> str:
    """Stelle mit dem aufgeloesten repo-relativen Pfad (Kurzformen ausgeschrieben); ohne Pfad der Rohtext."""
    if st.pfad is None:
        return st.roh
    if st.start is None:
        return st.pfad
    return f"{st.pfad}:{st.start}" + (f"-{st.ende}" if st.ende is not None else "")


def _dienst_je_port(zeilen: list[HopZeile]) -> dict[str, str]:
    """Port -> Dienstname, wenn genau ein benannter Dienst diesen Port traegt."""
    namen: dict[str, set[str]] = {}
    for hz in zeilen:
        for text in (hz.von, hz.nach):
            typ, name = knotentyp_erkennen(text)
            port = _PORT.search(_ohne_markup(text))
            if typ == "dienst" and port and not name.startswith(":"):
                namen.setdefault(port.group(1), set()).add(name)
    return {p: next(iter(n)) for p, n in namen.items() if len(n) == 1}


def seed_graph(
    konfig: Konfig, manifeste: Mapping[str, "Manifest"], sql_defs: Mapping[str, "SqlDefinition"],
    commit_aussen: str, commit_sub: str,
) -> tuple[Graph, list[Pruefpunkt]]:
    """Graph aus den Hop-Tabellen; Belege werden neu verankert. Wirft nicht bei kaputten Zeilen."""
    g = Graph({}, {}, commit_aussen, commit_sub)
    md = konfig.ketten_md
    try:
        text = (konfig.repo / md).read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as fehler:
        return g, [Pruefpunkt("ketten_md_fehlt", f"{md} nicht lesbar: {type(fehler).__name__}", md, None, None)]
    zeilen, pruefpunkte = hop_tabellen_lesen_mit_pruefpunkten(text, md)
    uebersicht = uebersicht_lesen(text)
    mcp_namen = _mcp_namen(konfig, manifeste)
    schemata = DB_SCHEMATA | {n.split(".", 1)[0] for n in sql_defs}
    dienst_je_port = _dienst_je_port(zeilen)
    index = _datei_index(konfig.repo, konfig)
    schwach = schwache_tokens(konfig)
    bekannte: dict[str, str] = {}
    voriger: str | None = None
    letzte_kette: str | None = None

    for hz in zeilen:
        hop_id = f"{hz.kette}/{hz.nr}"
        if hz.kette != letzte_kette:
            letzte_kette, voriger = hz.kette, None
            g.knoten_hinzu(Knoten(f"kette:{hz.kette}", "kette", hz.kette_titel or hz.kette,
                                  attribute=dict(uebersicht.get(hz.kette, {}))))
        von = _knoten(hz.von, mcp_namen, schemata, sql_defs, dienst_je_port)
        nach = _knoten(hz.nach, mcp_namen, schemata, sql_defs, dienst_je_port)
        g.knoten_hinzu(von)
        g.knoten_hinzu(nach)

        namen = zeilen_namen(hz)
        tokens = zeilen_tokens(hz, schwach=schwach)
        stellen: list[Stelle] = []
        for spanne in hz.code_stellen:
            neu, voriger = _aufloesen(f"`{spanne}`", voriger, konfig.repo, lambda: index, bekannte)
            stellen.extend(neu)
        belege: dict[tuple[str, str, str], Beleg] = {}
        stellen_roh: list[dict[str, object]] = []  # Ruling 34: Stelle -> Beleg-Schluessel (oder None)
        for st in stellen:
            if st.pfad is not None and st.form == "voll":
                bekannte[st.pfad.rsplit("/", 1)[-1]] = st.pfad
            commit = commit_sub if (st.pfad or "").startswith("vibemind-os/") else commit_aussen
            v = stelle_verankern_mit_grund(st, konfig, manifeste, sql_defs, commit, namen=namen, tokens=tokens)
            ort = f"`{st.roh}` ({st.form}) -> {st.pfad or '?'}"
            if v.pruefpunkt is not None:
                pruefpunkte.append(Pruefpunkt(v.pruefpunkt, f"{hop_id}: {ort}: {v.text}", md, hz.zeile, hop_id))
            stellen_roh.append({"roh": st.roh, "voll": _stelle_voll(st), "beleg": None if v.beleg is None
                                else [v.beleg.art, v.beleg.file, v.beleg.anker]})
            if v.beleg is None:
                if v.pruefpunkt is None:
                    pruefpunkte.append(Pruefpunkt(
                        "beleg_nicht_aufloesbar", f"{hop_id}: {ort} nicht verankerbar", md, hz.zeile, hop_id))
                continue
            belege.setdefault((v.beleg.art, v.beleg.file, v.beleg.anker), v.beleg)
        if not stellen:
            pruefpunkte.append(Pruefpunkt(
                "hop_ohne_beleg", f"{hop_id}: keine Code-Stelle in `{hz.code_stelle_roh[:80]}`",
                md, hz.zeile, hop_id))

        zustand, bekannt = zustand_aus_spalte(hz.zustand)
        if not bekannt:
            pruefpunkte.append(Pruefpunkt(
                "zustand_unbekannt", f"{hop_id}: Zustand '{hz.zustand}' -> keine_probe", md, hz.zeile, hop_id))
        _, beidseitig = _ausserhalb_backticks_teilen(hz.von_nach_roh)
        g.kante_hinzu(Kante(
            kante_id("hop", von.id, nach.id, kette=hz.kette, nr=hz.nr), "hop", von.id, nach.id, "seed",
            tuple(belege.values()),
            {
                "kette": hz.kette, "nr": hz.nr, "von_nach": hz.von_nach_roh, "beidseitig": beidseitig,
                "mechanismus": hz.mechanismus, "mechanismus_typ": mechanismus_typ(hz.mechanismus),
                "code_stelle": hz.code_stelle_roh, "belegt_durch": hz.belegt_durch,
                "zustand": zustand, "zustand_text": hz.zustand, "zeile": hz.zeile,
                "mechanismus_roh": hz.mechanismus, "zustand_roh": hz.zustand, "stellen_roh": stellen_roh,
                "code_prosa": code_prosa(hz.code_stelle_roh),
            },
        ))
    for kette, roh in sorted(uebersicht.items()):  # Ketten nur in der Uebersicht (ohne Hop-Tabelle)
        if f"kette:{kette}" not in g.knoten:
            g.knoten_hinzu(Knoten(f"kette:{kette}", "kette", roh["titel_roh"] or kette, attribute=dict(roh)))
    return g, pruefpunkte
