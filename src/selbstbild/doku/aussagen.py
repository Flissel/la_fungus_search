"""Doku-Parser: Aussagen einer Markdown-Datei, ihre Belege, Ersetzen (Task 12, Ruling 36).

Eine Aussage ist ein Absatz (durch Leerzeilen getrennt) oder eine Tabellenzeile ausserhalb markierter Bloecke
(`<!-- graph:begin … -->` … `<!-- graph:end -->`, die gehoeren dem Renderer). Keine Aussagen sind:
Ueberschriften, Frontmatter, Code-Bloecke (```/~~~, auch Mermaid), HTML-Kommentare, Markerzeilen,
Trennlinien, Tabellenkoepfe und -trennzeilen sowie der Belege-Abschnitt selbst (`## Belege` / `### Belege`,
`## 7. Belege`, `## Belege (Zusatz)`; Formen siehe `belege_formate`).

Belege einer Aussage:
  - Backtick-Spannen mit Code-Stellen (`pfad:zeile[-zeile]`, Kurz-, Verzeichnis- und Ankerformen wie im Seed,
    `ketten_seed._aufloesen`); eine Kurzform `:n` hinter einem Pfad zaehlt nur, wenn die Datei so viele
    Zeilen hat (sonst ist es ein Port);
  - Verweise `[Bn]` auf Zeilen des Abschnitts Belege (`[B3] pfad:zeile …`, `- [B3] …`, Tabelle
    `| B3 | pfad:zeile | … |` oder die Vorlagenzeile `- B3: datei pfad:zeile`); unaufgeloeste Verweise werden still
    ignoriert. Backtick-Pfade einer Prosa-Liste im Abschnitt binden keine Aussage.
  Verankert wird wie im Seed mit `ketten_seed.stelle_verankern_mit_grund`; Namen und Belegtokens kommen aus
  der Aussage selbst (`ketten_seed.namen_und_tokens_aus_text`, Rulings 22/27/29). Gerenderte Anker-Spannen
  (`` `datei:zeile` `anker` ``) werden mit `render_anker_trennen` abgetrennt und zaehlen nie als Beleg.

`aussage_id` = `<datei>#<Ueberschriftenpfad>#<sha1(erster Satz)[:8]>`; der Ueberschriftenpfad ist die Kette
der umschliessenden Ueberschriften, verbunden mit ` > `. Den Rest einer Aussage umzuschreiben aendert die
id nicht.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from . import anker
from . import belege_formate as bf
from . import ketten_seed as ks
from .anker import Beleg
from .konfig import Konfig
from .render_ketten import bloecke_finden, zeilenende
from .typen import Pruefpunkt

if TYPE_CHECKING:
    from ._manifest_gate2 import Manifest

    from .extraktoren.sql import SqlDefinition

BOM = "﻿"

_UEBERSCHRIFT = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_ZAUN = re.compile(r"^(`{3,}|~{3,})")
_TRENNLINIE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")
_TABELLEN_TRENN = re.compile(r"^\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?$")
_VERWEIS = re.compile(r"\[B(\d+)\]")
_HANDMARKE = re.compile(r"^<!--\s*handgepflegt\s*-->$")
# Abkuerzungen, nach deren Punkt kein Satz endet (dazu: einzelne Buchstaben und Zahlen)
_ABKUERZUNGEN = frozenset({
    "bzw", "ca", "dh", "etc", "evtl", "ggf", "inkl", "max", "min", "nr", "sog", "usw", "vgl", "zb", "bspw",
    "abs", "bzgl", "exkl", "ff", "mind", "zzgl",
    "jan", "feb", "mrz", "apr", "jun", "jul", "aug", "sep", "sept", "okt", "nov", "dez",
})
_MONAT = re.compile(r"(?:Jan|Feb|Mär|Mrz|Mar|Apr|Mai|May|Jun|Jul|Aug|Sep|Okt|Oct|Nov|Dez|Dec)")
_DATUM_ENDE = re.compile(r"(?<!\d)\d{1,2}\.\d{1,2}\.$")
_VERBOTEN_ANFANG = ("#", "```", "~~~", "<!--")
# Strukturregeln fuer Ersatztexte (`_ersatz_pruefen`)
_ZEILENBRUCH_ZEICHEN = ("\r", "\x0b", "\x0c", "\x85", " ", " ")
_SETEXT = re.compile(r"^(?:=+|-+)$")
_TRENN_ERSATZ = re.compile(r"^(?:(?:\*\s*){3,}|(?:_\s*){3,}|(?:-\s*){3,})$")
_OEFFNER = (
    ("liste", re.compile(r"^[-+*](?:\s|$)")),
    ("nummer", re.compile(r"^\d{1,9}[.)](?:\s|$)")),
    ("zitat", re.compile(r"^>")),
)


@dataclass(frozen=True)
class Aussage:
    id: str
    datei: str  # repo-relativ (POSIX), sonst der Pfad wie uebergeben
    ueberschrift: str
    zeile_start: int  # 1-basiert
    zeile_ende: int  # 1-basiert, einschliesslich
    text: str  # Zeilen ohne Zeilenenden (und ohne BOM), mit `\n` verbunden
    belege: tuple[Beleg, ...]
    pruefpunkte: tuple[Pruefpunkt, ...] = ()  # unsichere/zu grobe Verankerungen dieser Aussage


@dataclass(frozen=True)
class _Einheit:
    start: int  # 0-basiert
    ende: int  # 0-basiert, einschliesslich
    ueberschrift: str


# ---------- id ----------

def erster_satz(text: str) -> str:
    """Erster Satz nach Leerraum-Normalisierung: bis zum ersten `.`/`!`/`?` vor Leerraum, ausser hinter einer
    Abkuerzung (`z. B.`, `bzw.`, `Okt.`) oder einem einzelnen Buchstaben. Hinter einer Zahl endet der Satz nur,
    wenn ein Grossbuchstabe folgt und es kein Datum ist (`27.10.`, `5. Okt`); ohne Satzende der ganze Text."""
    klar = " ".join(text.split())
    for m in re.finditer(r"[.!?](?=\s)", klar):
        if m.group(0) == ".":
            wort = re.search(r"(\w+)$", klar[:m.start()])
            if wort and wort.group(1).isdigit():
                nach = klar[m.end():].lstrip()
                if (not nach[:1].isupper() or _DATUM_ENDE.search(klar[:m.end()])
                        or (len(wort.group(1)) <= 2 and _MONAT.match(nach))):
                    continue
            elif wort and (len(wort.group(1)) == 1 or wort.group(1).lower() in _ABKUERZUNGEN):
                continue
        return klar[:m.end()]
    return klar


def aussage_id(datei: str, ueberschrift: str, text: str) -> str:
    return f"{datei}#{ueberschrift}#{hashlib.sha1(erster_satz(text).encode('utf-8')).hexdigest()[:8]}"


# ---------- Zeilen ----------

def _zeilen_mit_ende(text: str) -> list[str]:
    """Zeilen mit Zeilenende; getrennt wird nur an `\\n`."""
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def _inhalt(zeile: str, erste: bool) -> str:
    inhalt = zeile[:-1] if zeile.endswith("\n") else zeile
    inhalt = inhalt[:-1] if inhalt.endswith("\r") else inhalt
    return inhalt[1:] if erste and inhalt.startswith(BOM) else inhalt


def _block_zeilen(text: str) -> set[int]:
    """0-basierte Zeilen markierter Bloecke, Markerzeilen eingeschlossen."""
    zeilen: set[int] = set()
    for b in bloecke_finden(text).values():
        beginn = text.count("\n", 0, b.start) - 1
        ende = text.count("\n", 0, b.ende)
        zeilen.update(range(beginn, ende + 1))
    return zeilen


def _frontmatter_ende(zeilen: list[str]) -> int | None:
    """0-basierte Zeile des schliessenden `---` eines Frontmatters am Dateianfang; sonst None."""
    if not zeilen or zeilen[0].strip() != "---":
        return None
    for j in range(1, len(zeilen)):
        if zeilen[j].strip() in ("---", "..."):
            return j
    return None


def _ist_trennzeile(zeile: str) -> bool:
    return bool(_TABELLEN_TRENN.match(zeile.strip())) and "-" in zeile


def _einheiten(zeilen: list[str], gesperrt: set[int]) -> tuple[list[_Einheit], dict[int, bf.Eintrag]]:
    """Aussage-Einheiten und nummerierte Eintraege des Abschnitts Belege (`belege_formate.eintrag`)."""
    einheiten: list[_Einheit] = []
    belege: dict[int, bf.Eintrag] = {}
    stapel: list[tuple[int, str]] = []
    belege_ebene: int | None = None
    zaun: str | None = None
    im_kommentar = False
    absatz: int | None = None
    fm = _frontmatter_ende(zeilen)

    def pfad() -> str:
        return " > ".join(t for _, t in stapel)

    def schliessen(i: int) -> None:
        nonlocal absatz
        if absatz is not None:
            einheiten.append(_Einheit(absatz, i - 1, pfad()))
            absatz = None

    for i, zeile in enumerate(zeilen):
        klar = zeile.strip()
        if (fm is not None and i <= fm) or i in gesperrt:
            schliessen(i)
            continue
        if zaun is not None:
            if klar.startswith(zaun[0] * len(zaun)) and not klar.strip(zaun[0]):
                zaun = None
            continue
        if im_kommentar:
            im_kommentar = "-->" not in klar
            continue
        m_zaun = _ZAUN.match(klar)
        if m_zaun:
            schliessen(i)
            zaun = m_zaun.group(1)
            continue
        if klar.startswith("<!--"):
            schliessen(i)
            im_kommentar = "-->" not in klar
            continue
        m = _UEBERSCHRIFT.match(klar)
        if m:
            schliessen(i)
            ebene = len(m.group(1))
            while stapel and stapel[-1][0] >= ebene:
                stapel.pop()
            stapel.append((ebene, m.group(2)))
            if belege_ebene is not None and ebene <= belege_ebene:
                belege_ebene = None
            if belege_ebene is None and bf.ist_belege_kopf(klar):
                belege_ebene = ebene
            continue
        if not klar or _TRENNLINIE.match(klar):
            schliessen(i)
            continue
        if belege_ebene is not None:
            e = bf.eintrag(klar)
            if e is not None:
                belege.setdefault(e.nr, e)
            continue
        if klar.startswith("|"):
            schliessen(i)
            kopf = i + 1 < len(zeilen) and i + 1 not in gesperrt and _ist_trennzeile(zeilen[i + 1])
            if not kopf and not _ist_trennzeile(klar):
                einheiten.append(_Einheit(i, i, pfad()))
            continue
        if absatz is None:
            absatz = i
    schliessen(len(zeilen))
    return einheiten, belege


# ---------- Belege ----------

class _Verankerer:
    def __init__(self, konfig: Konfig, manifeste: Mapping[str, "Manifest"],
                 sql_defs: Mapping[str, "SqlDefinition"], commit: str,
                 index_holer: Callable[[], Mapping[str, list[str]]] | None = None) -> None:
        self.konfig, self.manifeste, self.sql_defs, self.commit = konfig, manifeste, sql_defs, commit
        self.index: Callable[[], Mapping[str, list[str]]] = (index_holer if index_holer is not None
                                                             else ks._index_holer(konfig.repo, konfig, None))
        self.schwach = ks.schwache_tokens(konfig)
        self.bekannte: dict[str, str] = {}
        self.laengen: dict[str, int] = {}

    def _laenge(self, pfad: str) -> int:
        if not anker.pfad_sicher(pfad):  # Final-Fix 22: nie ausserhalb des Repos lesen
            return 0
        if pfad not in self.laengen:
            try:
                self.laengen[pfad] = len((self.konfig.repo / pfad).read_bytes().splitlines())
            except OSError:
                self.laengen[pfad] = 0
        return self.laengen[pfad]

    def stellen(self, text: str, ganz: bool) -> list[ks.Stelle]:
        """Stellen aus den Backtick-Spannen von `text` (`ganz`: ohne Spannen der ganze Text, fuer Belege-
        Eintraege); Ports hinter einem Pfad (Kurzform jenseits des Dateiendes) fallen weg."""
        spannen = re.findall(r"`([^`]+)`", text)
        if not spannen:
            spannen = [text] if ganz and text.strip() else []
        voriger: str | None = None
        aus: list[ks.Stelle] = []
        for spanne in spannen:
            neu, voriger = ks._aufloesen(f"`{spanne}`", voriger, self.konfig.repo, self.index, self.bekannte)
            for st in neu:
                if st.form == "kurzform" and (st.pfad is None or (st.start or 0) > self._laenge(st.pfad)):
                    continue
                if st.pfad is not None and st.form == "voll":
                    self.bekannte[st.pfad.rsplit("/", 1)[-1]] = st.pfad
                aus.append(st)
        return aus

    def belege(self, text: str, belege_abschnitt: Mapping[int, bf.Eintrag],
               datei: str, zeile: int, aid: str) -> tuple[tuple[Beleg, ...], tuple[Pruefpunkt, ...]]:
        ohne_anker, _ = ks.render_anker_trennen(text)
        namen, tokens = ks.namen_und_tokens_aus_text(text, schwach=self.schwach)
        stellen = [(st, False) for st in self.stellen(ohne_anker, ganz=False)]
        for nr in _VERWEIS.findall(ohne_anker):
            eintrag = belege_abschnitt.get(int(nr))
            if eintrag is not None and eintrag.ziel:
                quittung = eintrag.art == "datei"  # Vorlage: erklaerte Datei-Quittung
                stellen.extend((st, quittung) for st in self.stellen(ks.render_anker_trennen(eintrag.ziel)[0],
                                                                     ganz=True))
        belege: dict[tuple[str, str, str], Beleg] = {}
        pp: list[Pruefpunkt] = []
        for st, quittung in stellen:
            v = ks.stelle_verankern_mit_grund(st, self.konfig, self.manifeste, self.sql_defs, self.commit,
                                              namen=namen, tokens=tokens)
            if quittung and v.beleg is None and v.pruefpunkt == "anker_zu_grob" and st.pfad is not None:
                # Task 4: eine `- B<n>: datei pfad:zeile`-Quittung bindet an die Datei, auch wo der feinere Anker zu
                # grob waere (z. B. Zeile 1 einer YAML-Datei)
                v = ks.Verankerung(ks._datei_beleg(st.pfad, st.start, self.konfig.repo, self.commit))
            if v.pruefpunkt is not None:
                pp.append(Pruefpunkt(v.pruefpunkt, f"`{st.roh}` -> {st.pfad or '?'}: {v.text}", datei, zeile, aid))
            if v.beleg is not None:
                belege.setdefault((v.beleg.art, v.beleg.file, v.beleg.anker), v.beleg)
        return tuple(belege.values()), tuple(pp)


def _datei_name(pfad: Path, repo: Path) -> str:
    try:
        return pfad.resolve().relative_to(repo.resolve()).as_posix()
    except (ValueError, OSError):
        return pfad.as_posix()


def aussagen_lesen(pfad: Path, konfig: Konfig, manifeste: Mapping[str, "Manifest"],
                   sql_defs: Mapping[str, "SqlDefinition"], commit: str, *, text: str | None = None,
                   index_holer: Callable[[], Mapping[str, list[str]]] | None = None) -> list[Aussage]:
    """Aussagen der Datei in Reihenfolge, Belege neu verankert (siehe Moduldoku).

    `text`: Inhalt statt der Datei (z. B. der eingebettete Text vor dem Schreiben); `pfad` bestimmt dann nur die
    Ids und Dateinamen. `index_holer`: gemeinsamer Datei-Index (Dateiname -> Pfade) mehrerer Aufrufe im selben Lauf;
    sonst baut jeder Aufruf ihn bei Bedarf selbst."""
    if text is None:
        text = Path(pfad).read_bytes().decode("utf-8")
    roh = _zeilen_mit_ende(text)
    zeilen = [_inhalt(z, i == 0) for i, z in enumerate(roh)]
    einheiten, belege_abschnitt = _einheiten(zeilen, _block_zeilen(text))
    datei = _datei_name(Path(pfad), konfig.repo)
    verankerer = _Verankerer(konfig, manifeste, sql_defs, commit, index_holer)
    aus: list[Aussage] = []
    gesehen: dict[str, tuple[int, int]] = {}  # id -> (Anzahl, erste Zeile)
    for e in einheiten:
        inhalt = "\n".join(zeilen[e.start:e.ende + 1])
        aid = aussage_id(datei, e.ueberschrift, inhalt)
        doppelt: Pruefpunkt | None = None
        if aid in gesehen:  # gleicher erster Satz unter derselben Ueberschrift: Suffix in Lesereihenfolge
            anzahl, erste = gesehen[aid]
            gesehen[aid] = (anzahl + 1, erste)
            basis, aid = aid, f"{aid}~{anzahl + 1}"
            doppelt = Pruefpunkt("aussage_id_doppelt", f"{basis}: gleicher erster Satz wie Zeile {erste}; "
                                                       f"id mit Suffix ~{anzahl + 1}", datei, e.start + 1, aid)
        else:
            gesehen[aid] = (1, e.start + 1)
        belege, pp = verankerer.belege(inhalt, belege_abschnitt, datei, e.start + 1, aid)
        if doppelt is not None:
            pp = (doppelt, *pp)
        aus.append(Aussage(aid, datei, e.ueberschrift, e.start + 1, e.ende + 1, inhalt, belege, pp))
    return aus


# ---------- Ersetzen ----------

def _oeffner(klar: str) -> str | None:
    """Art des Listen-/Zitat-Oeffners am Anfang einer (links gestutzten) Zeile, sonst None."""
    for art, muster in _OEFFNER:
        if muster.match(klar):
            return art
    return None


def _ersatz_pruefen(aussage: Aussage, neu: list[str]) -> None:
    """Der Ersatz (vom Judge) darf die Struktur der Datei nicht aendern; sonst ValueError.

    Abgelehnt werden (Fix-Runde 1 zu Task 13): Zeichen, die manche Leser als Zeilenumbruch nehmen (einzelnes
    CR, VT, FF, NEL, U+2028, U+2029); eingerueckter Code (4 Leerzeichen oder ein Tab am Zeilenanfang);
    Setext-Unterstreichungen (`===`, `--`) und Trennlinien (`***`, `___`, `- - -`); Listen- und Zitat-Oeffner
    (`- `, `+ `, `* `, `>`, `1. `, `1) `) am Anfang einer Folgezeile und am Anfang der ersten Zeile, wenn die
    Aussage selbst nicht mit derselben Art begann."""
    if not "".join(neu).strip():
        raise ValueError(f"{aussage.id}: leerer Ersatztext; Aussagen werden ersetzt, nie geloescht")
    alt_oeffner = _oeffner(aussage.text.split("\n", 1)[0].strip())
    for i, zeile in enumerate(neu):
        klar = zeile.strip()
        if not klar:
            raise ValueError(f"{aussage.id}: Ersatz enthaelt eine Leerzeile (wuerde die Aussage teilen)")
        if any(z in zeile for z in _ZEILENBRUCH_ZEICHEN):
            raise ValueError(f"{aussage.id}: Ersatz enthaelt ein Zeichen, das als Zeilenumbruch gilt")
        einzug = zeile[:len(zeile) - len(zeile.lstrip(" \t"))]
        if "\t" in einzug or len(einzug) >= 4:
            raise ValueError(f"{aussage.id}: Ersatz rueckt eine Zeile wie Code ein")
        if klar.startswith(_VERBOTEN_ANFANG) or _SETEXT.match(klar) or _TRENN_ERSATZ.match(klar):
            raise ValueError(f"{aussage.id}: Ersatz beginnt eine Zeile mit Struktur ({klar[:20]!r})")
        oeffner = _oeffner(klar)
        if oeffner is not None and (i > 0 or oeffner != alt_oeffner):
            raise ValueError(f"{aussage.id}: Ersatz beginnt eine Zeile mit einem Listen-/Zitat-Oeffner "
                             f"({klar[:20]!r})")
        if "graph:begin" in zeile or "graph:end" in zeile:
            raise ValueError(f"{aussage.id}: Ersatz enthaelt einen Render-Marker")
    if aussage.text.lstrip().startswith("|"):
        klar = neu[0].strip()
        if len(neu) != 1 or len(klar) < 2 or not (klar.startswith("|") and klar.endswith("|")):
            raise ValueError(f"{aussage.id}: eine Tabellenzeile wird nur durch genau eine Tabellenzeile ersetzt")
        alt_zellen, neu_zellen = len(ks._zellen(aussage.text.strip())), len(ks._zellen(klar))
        if neu_zellen != alt_zellen:
            raise ValueError(f"{aussage.id}: Tabellenzeile mit {neu_zellen} statt {alt_zellen} Zellen")
    elif any(z.lstrip().startswith("|") for z in neu):
        raise ValueError(f"{aussage.id}: ein Absatz wird nicht durch Tabellenzeilen ersetzt")


def _ersatz_zeilen(neuer_text: str) -> list[str]:
    return [z.rstrip("\r") for z in neuer_text.replace("\r\n", "\n").strip("\n").split("\n")]


def ersatz_pruefen(aussage: Aussage, neuer_text: str) -> None:
    """Strukturregeln von `aussage_ersetzen` ohne Datei: ValueError, wenn `neuer_text` dort abgelehnt wuerde
    (der Judge prueft seine Antwort damit vorab)."""
    _ersatz_pruefen(aussage, _ersatz_zeilen(neuer_text))


def aussage_ersetzen(text: str, aussage: Aussage, neuer_text: str) -> str:
    """Genau die Zeilen der Aussage durch `neuer_text` ersetzen; Zeilenenden, BOM und umgebende Leerzeilen
    bleiben.

    ValueError, wenn der Text dort nicht mehr `aussage.text` ist (veraltet: neu lesen), wenn `neuer_text`
    leer ist (der Loop ersetzt nur, er loescht nie) oder die Struktur aendern wuerde (`_ersatz_pruefen`):
    eine innere Leerzeile; eine Zeile, die mit `#`, drei Backticks, `~~~` oder `<!--` beginnt oder nur `---`
    ist; ein Render-Marker (`graph:begin`/`graph:end`). Eine Tabellenzeile wird nur durch genau eine Zeile mit
    `|` am Anfang und Ende und gleicher Zahl unescapeter Zellen ersetzt, ein Absatz nie durch Tabellenzeilen.
    Dazu die Markdown-Brueche aus `_ersatz_pruefen` (Umbruchzeichen, Code-Einzug, Setext, Trennlinien, Listen-
    und Zitat-Oeffner).

    Aufrufer lesen die Datei mit `read_bytes().decode("utf-8")` (nicht `read_text`: das uebersetzt CRLF und
    die Pruefung gegen `aussage.text` bzw. die Zeilenenden stimmen nicht mehr) und wenden mehrere Ersetzungen
    in einer Datei von unten nach oben an, damit die Zeilennummern der uebrigen Aussagen gueltig bleiben.
    """
    neu = _ersatz_zeilen(neuer_text)
    _ersatz_pruefen(aussage, neu)
    zeilen = _zeilen_mit_ende(text)
    start, ende = aussage.zeile_start - 1, aussage.zeile_ende
    if start < 0 or ende > len(zeilen) or start >= ende:
        raise ValueError(f"{aussage.id}: Zeilen {aussage.zeile_start}-{aussage.zeile_ende} gibt es nicht mehr")
    jetzt = "\n".join(_inhalt(zeilen[i], i == 0) for i in range(start, ende))
    if jetzt != aussage.text:
        raise ValueError(f"{aussage.id}: Aussage ist veraltet (Text an Zeile {aussage.zeile_start} geaendert)")
    erste, letzte = zeilen[start], zeilen[ende - 1]
    nl = erste[len(erste.rstrip("\r\n")):] or zeilenende(text)
    schluss = letzte[len(letzte.rstrip("\r\n")):]
    bom = BOM if start == 0 and erste.startswith(BOM) else ""
    ersatz = bom + nl.join(neu) + schluss
    return "".join(zeilen[:start]) + ersatz + "".join(zeilen[ende:])


# ---------- Handpflege ----------

def handgepflegt(text: str) -> bool:
    """Frontmatter mit `pflege: hand` oder eine eigene Zeile `<!-- handgepflegt -->` (nicht in Code-Bloecken,
    nicht als Zitat in einer Code-Spanne): der Loop fasst die Datei nicht an."""
    zeilen = text.lstrip(BOM).replace("\r\n", "\n").split("\n")
    fm = _frontmatter_ende(zeilen)
    if fm is not None:
        try:
            daten = yaml.safe_load("\n".join(zeilen[1:fm]))
        except yaml.YAMLError:
            daten = None
        if isinstance(daten, dict) and str(daten.get("pflege", "")).strip().lower() == "hand":
            return True
    zaun: str | None = None
    for zeile in zeilen:
        klar = zeile.strip()
        if zaun is not None:
            if klar.startswith(zaun[0] * len(zaun)) and not klar.strip(zaun[0]):
                zaun = None
            continue
        m = _ZAUN.match(klar)
        if m:
            zaun = m.group(1)
            continue
        if _HANDMARKE.match(klar):
            return True
    return False
