"""Rowboat-Notizen, Pruefliste und Urteile-Rueckweg (Task 17, Fix-Runde 1).

Notizen (`notizen_schreiben`) unter `<rowboat_wissen>/<ordner>/` (Standard `Doku/`):
- `Ketten/K<n>.md`   Frontmatter `kette, zustand, commit, lauf`; Hops als Tabelle mit `[[Doku/Dienste/<name>]]`- und
                     `[[Doku/Spaces/<name>]]`-Links (Pfad ab der Wissenswurzel, ohne `.md`, wie die uebrigen Notizen)
- `Spaces/<space>.md`, `Dienste/<name>.md`
- `Prüfliste.md`     Umschreibungen des Loops (Ledger), offene Punkte der lokalen Pruefliste, angewandte Urteile und
                     Struktur-Hinweise; je Punkt Lauf, Aussage-ID, alter/neuer Text, Beleg, Commit, Art und die
                     Urteilszeilen, die dort gelten (`zurueck` nur, solange der Loop-Commit der letzte der Datei ist)
- `Index.md`         Links auf alles
Schreibschutz (Fix-Runde 1, Item 2): ein LOKALES Manifest (`notizen-<hash>.json` im Marker-Ordner, wie der Stand) haelt
je geschriebener Notiz den Hash ihres stabilen Texts (ohne die fluechtigen Kopfzeilen `lauf`/`commit`). Ueberschrieben
oder geloescht wird nur eine Datei, deren aktueller stabiler Hash dem Manifest entspricht; jede andere (von Hand
angelegt, kopiert oder geaendert) bleibt stehen und wird einmal als `notiz_von_hand` gemeldet. Fehlt die Manifest-
Datei (erster Lauf mit dieser Fassung), wird nur uebernommen, was dieser Lauf selbst schreiben wuerde (`soll`) UND
entweder schon genau diesen stabilen Text hat ODER den Marker traegt und im Kopf `lauf:` einen bekannten Loop-Lauf
nennt (Ledger, Loop-Commits, letzter Stand); alles andere mit Marker gilt als von Hand, und dieser Lauf loescht nichts.
Ein unlesbares Manifest: keine Notiz wird geschrieben, Pruefpunkt `notizen_manifest_kaputt` (Fail-closed; zum
Neuaufsetzen die Datei loeschen). Sind `Prüfliste.md` oder `Index.md` von Hand geaendert, schreibt der Loop seine
Fassung daneben (`Prüfliste (Loop).md`, `Index (Loop).md`) und nennt das im Kopf der jeweils anderen Datei, damit die
Pruefliste nie einfriert.
Vor jedem Schreiben und Loeschen: der Pfad liegt nach `resolve()` im Notizordner, und kein Teil des Wegs ist ein
Symlink oder Reparse-Point (Junction) -- solchen Wegen folgt der Loop nie. Geschrieben wird nur, was sich geaendert
hat, atomar ueber eine Temp-Datei im selben Ordner. Dateinamen sind fuer Windows und Wiki-Links bereinigt; Kollisionen
(auch nur in Gross-/Kleinschreibung) bekommen einen kurzen Hash-Zusatz.

Urteile (`urteile_lesen`, `urteile_anwenden`): die Datei `<rowboat_wissen>/Doku-Urteile/Urteile.md` gehoert der VM
(Spiegel); der Loop liest sie nur. Je Zeile `<lauf> <aussage_id> ok|zurueck|korrektur: <text>`; Leerzeilen,
Ueberschriften, `>`-Zitate, HTML-Kommentare und ein Frontmatter zaehlen nicht, ein Listenpunkt und Backticks um die
Zeile werden abgestreift. Unlesbare Zeilen -> je ein Pruefpunkt `urteil_unlesbar`.
Ruling 54: ein Urteil braucht einen Bezug -- `zurueck` einen Ledger-Eintrag, `korrektur` einen Ledger-Eintrag oder einen
offenen Punkt mit Aussage-ID, `ok` einen Ledger-Eintrag oder offenen Punkt; sonst `urteil_ohne_bezug`.
- `ok`        -> der Punkt ist erledigt (verschwindet aus `Prüfliste.md`)
- `zurueck`   -> Loop-Commit mit GENAU `Lauf: <lauf>` (`lauf_commits`), `umschreiben.revertieren` fuer die Datei des
                 Ledger-Eintrags; geht das nicht -> `zurueck_nicht_moeglich`. Danach Torwaechter: neue Fehler ->
                 `zurueck_drift` (nicht blockierend). Andere Umschreibungen desselben Laufs in derselben Datei sind
                 damit auch zurueckgenommen und werden so angezeigt.
- `korrektur` -> `aussage_ersetzen` (Strukturwaechter) ueber den gemeinsamen Schreibvorgang (Torwaechter, CAS,
                 pathspec-Commit `docs(loop): Korrektur aus Review <aussage_id>` mit `Lauf: review-...`), Ledger-Eintrag
                 `quelle: review`; abgelehnt -> `korrektur_abgelehnt`, nichts geschrieben
Jedes Urteil wird hoechstens einmal angewandt; der Vermerk liegt lokal (`urteile-erledigt-<hash>.jsonl` im
Marker-Ordner, wie der Stand) und traegt Zeit, Commit und Datei fuer den Abschnitt "Angewandte Urteile". Der Ordner
`Kanal/` wird nie gelesen.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from . import aussagen as au
from . import umschreiben
from .anker import Beleg
from .graph import Graph, Kante, Knoten
from .konfig import Konfig
from .schluessel import schluessel_muster, schwaerzen
from .schreibvorgang import Aenderung, git, kopf, kopf_pruefen, schreiben_pruefen_committen
from .stand import lokaler_ordner, repo_schluessel
from .typen import Pruefpunkt

MARKER_SCHLUESSEL = "erzeugt_von"
MARKER_WERT = "doku-loop"
FLUECHTIG = ("lauf", "commit")  # Kopfzeilen, die fuer "geaendert?" nicht zaehlen
URTEIL_ARTEN = ("ok", "zurueck", "korrektur")
PRUEFLISTE_NAME = "Prüfliste"
INDEX_NAME = "Index"
VERBOTEN_ORDNER = "kanal"  # knowledge/Kanal/ (Telegram) wird nie gelesen
STRUKTUR_MAX = 200  # Struktur-Hinweise in der Pruefliste (Rest nur gezaehlt)
ANGEWANDT_MAX = 50  # angewandte Urteile in der Pruefliste (neueste zuerst)
_GEMESSEN = {"laeuft": "läuft", "bricht": "bricht", "keine_probe": "keine Probe"}

Melden = Callable[[str], None]
Drift = Callable[[Path], "set[umschreiben.DriftFehler]"]


# ---------- Hilfen ----------

def _still(_zeile: str) -> None:
    return None


def _naturlich(text: str) -> list[object]:
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", text)]


_UNSICHER = re.compile(r'[<>:"/\\|?*\x00-\x1f\[\]#^]')
_RESERVIERT = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                         *(f"LPT{i}" for i in range(1, 10))})


def sicherer_name(name: str) -> str:
    """Dateiname ohne `<>:"/\\|?*`, Steuerzeichen und Wiki-Link-Zeichen (`[]#^`), ohne Punkt/Leerzeichen am Ende,
    nie ein reservierter Windows-Name, hoechstens 120 Zeichen."""
    klar = " ".join(_UNSICHER.sub("_", name).split())[:120].rstrip(". ")
    if not klar:
        klar = "_"
    if klar.split(".", 1)[0].upper() in _RESERVIERT:
        klar = "_" + klar
    return klar


def _namen_vergeben(knoten: Iterable[Knoten]) -> dict[str, str]:
    """Knoten-ID -> eindeutiger sicherer Dateiname (ohne `.md`); deterministisch nach ID."""
    vergeben: dict[str, str] = {}
    belegt: set[str] = set()
    for k in sorted(knoten, key=lambda k: k.id):
        name = sicherer_name(k.name)
        if name.casefold() in belegt:
            name = f"{name[:110]} ({hashlib.sha1(k.id.encode('utf-8')).hexdigest()[:6]})"
        belegt.add(name.casefold())
        vergeben[k.id] = name
    return vergeben


def _ketten_namen(ketten: Iterable[str]) -> dict[str, str]:
    """Ketten-ID -> eindeutiger sicherer Dateiname; wie `_namen_vergeben` mit Hash-Zusatz bei Kollision (ohne
    Kollision unveraendert `sicherer_name` der Kurz-ID)."""
    vergeben: dict[str, str] = {}
    belegt: set[str] = set()
    for kid in sorted(ketten):
        name = sicherer_name(kid.split(":", 1)[1])
        if name.casefold() in belegt:
            name = f"{name[:110]} ({hashlib.sha1(kid.encode('utf-8')).hexdigest()[:6]})"
        belegt.add(name.casefold())
        vergeben[kid] = name
    return vergeben


def _zelle(text: object) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _code(text: str) -> str:
    text = " ".join(text.split())
    if "`" not in text:
        return f"`{text}`"
    return f"`` {text} ``"


def _beleg_text(b: Beleg) -> str:
    ort = f"{b.file}:{b.zeile}" if b.zeile else b.file
    return f"{_code(ort)} ({b.art}{' ' + b.anker if b.anker else ''})"


def _kopf(daten: Mapping[str, object]) -> str:
    """Frontmatter; Werte als JSON-Strings (gueltiges YAML), der Marker unverpackt."""
    zeilen = ["---", f"{MARKER_SCHLUESSEL}: {MARKER_WERT}"]
    for k, v in daten.items():
        zeilen.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    zeilen.append("---")
    return "\n".join(zeilen) + "\n"


def _frontmatter(text: str) -> dict[str, object] | None:
    zeilen = text.lstrip("﻿").replace("\r\n", "\n").split("\n")
    if not zeilen or zeilen[0].strip() != "---":
        return None
    for j in range(1, len(zeilen)):
        if zeilen[j].strip() in ("---", "..."):
            try:
                daten = yaml.safe_load("\n".join(zeilen[1:j]))
            except yaml.YAMLError:
                return None
            return daten if isinstance(daten, dict) else None
    return None


def vom_loop(pfad: Path) -> bool:
    """Traegt die Datei den Marker `erzeugt_von: doku-loop` im Frontmatter?"""
    try:
        roh = Path(pfad).read_bytes()[:4096]
    except OSError:
        return False
    daten = _frontmatter(roh.decode("utf-8", errors="replace"))
    return daten is not None and str(daten.get(MARKER_SCHLUESSEL, "")) == MARKER_WERT


def _stabil(text: str) -> str:
    """Text ohne die fluechtigen Kopfzeilen (fuer den Vergleich alt/neu)."""
    zeilen = text.replace("\r\n", "\n").split("\n")
    if not zeilen or zeilen[0].strip() != "---":
        return text
    aus = [zeilen[0]]
    im_kopf = True
    for z in zeilen[1:]:
        if im_kopf and z.strip() in ("---", "..."):
            im_kopf = False
        elif im_kopf and z.split(":", 1)[0].strip() in FLUECHTIG:
            continue
        aus.append(z)
    return "\n".join(aus)


def _atomar(pfad: Path, daten: bytes) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=pfad.parent, prefix=".doku-loop-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(daten)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, pfad)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _umgeleitet(p: Path) -> bool:
    """Symlink oder Reparse-Point (Junction, unter Windows auch ohne Rechte anlegbar)? Fehlt der Pfad -> False."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & _REPARSE)


def _sicher(basis: Path, pfad: Path) -> bool:
    """`pfad` liegt im Notizordner, und weder der Ordner noch ein Teil des Wegs noch die Datei ist umgeleitet."""
    try:
        rel = pfad.relative_to(basis)
    except ValueError:
        return False
    if _umgeleitet(basis):
        return False
    weg = basis
    for teil in rel.parts:
        weg = weg / teil
        if _umgeleitet(weg):
            return False
    try:
        pfad.resolve().relative_to(basis.resolve())
    except (ValueError, OSError):
        return False
    return True


def _stabil_hash(text: str) -> str:
    return hashlib.sha256(_stabil(text).encode("utf-8")).hexdigest()


def _unter(wurzel: Path, rel: str) -> Path:
    """`wurzel/rel` -- nie ausserhalb der Wurzel, nie unter `Kanal/`; sonst ValueError."""
    teile = Path(rel).parts
    if not rel or Path(rel).is_absolute() or ".." in teile:
        raise ValueError(f"ungueltiger Pfad: {rel!r}")
    if teile[0].casefold() == VERBOTEN_ORDNER:
        raise ValueError(f"{rel!r}: der Ordner Kanal/ wird nie gelesen")
    ziel = (wurzel / rel).resolve()
    ziel.relative_to(wurzel.resolve())  # ValueError, wenn ausserhalb
    return wurzel / rel


def urteile_pfad(wurzel: Path, rel: str) -> Path:
    return _unter(Path(wurzel), rel)


# ---------- Pruefliste (Kennung je Punkt) ----------

def punkt_kennung(punkt: Mapping[str, object]) -> str:
    """Kennung eines Punkts fuer die Urteilszeile: seine Aussage-ID, sonst `pp-<hash>` aus art/datei/text."""
    aid = punkt.get("aussage_id")
    if aid:
        return str(aid)
    schl = f"{punkt.get('art')}|{punkt.get('datei')}|{punkt.get('text')}"
    return "pp-" + hashlib.sha1(schl.encode("utf-8")).hexdigest()[:10]


def lauf_text(lauf: object) -> str:
    """Lauf eines Punkts, wie er in Urteilszeilen, Anzeige und Vermerken steht (M5: ohne Lauf immer `-`)."""
    return str(lauf) if lauf else "-"


def _ist_aussage_id(kennung: str) -> bool:
    return kennung.count("#") >= 2  # `<datei>#<ueberschrift>#<hash>`


def _punkt_dict(p: Pruefpunkt | Mapping[str, object]) -> dict[str, object]:
    if isinstance(p, Pruefpunkt):
        return {"art": p.art, "text": p.text, "datei": p.datei, "zeile": p.zeile, "aussage_id": p.aussage_id,
                "lauf": None}
    return dict(p)


# ---------- Notizen ----------

@dataclass(frozen=True)
class _Notiz:
    rel: str  # relativ zum Notizordner, mit .md
    text: str


class _Notizen:
    def __init__(self, graph: Graph, ordner: str, space_vorhanden: Collection[str] | None = None) -> None:
        self.g, self.ordner = graph, ordner.strip("/")
        # Fix-Runde 1 zu Task 9: gehoert `Spaces/` den erzaehlten Seiten, verlinkt nur, was es gibt (None = immer)
        self.space_vorhanden = None if space_vorhanden is None else set(space_vorhanden)
        self.dienste = {i: k for i, k in graph.knoten.items() if k.typ == "dienst"}
        self.spaces = {i: k for i, k in graph.knoten.items() if k.typ == "space"}
        self.ketten = {i: k for i, k in graph.knoten.items() if k.typ == "kette"}
        self.dienst_name = _namen_vergeben(self.dienste.values())
        self.space_name = _namen_vergeben(self.spaces.values())
        self.kette_name = _ketten_namen(self.ketten)
        self.hops: dict[str, list[Kante]] = {}
        for e in graph.kanten.values():
            if e.art == "hop":
                self.hops.setdefault(f"kette:{e.attribute.get('kette')}", []).append(e)
        for liste in self.hops.values():
            liste.sort(key=lambda e: _naturlich(str(e.attribute.get("nr", ""))))

    def link(self, art: str, name: str) -> str:
        return f"[[{self.ordner}/{art}/{name}]]"

    def knoten_link(self, kid: str) -> str:
        if kid in self.dienst_name:
            return self.link("Dienste", self.dienst_name[kid])
        if kid in self.space_name:
            if self.space_vorhanden is not None and f"Spaces/{self.space_name[kid]}.md" not in self.space_vorhanden:
                return self.spaces[kid].name
            return self.link("Spaces", self.space_name[kid])
        k = self.g.knoten.get(kid)
        return k.name if k is not None else kid.split(":", 1)[-1]

    def kette_link(self, kid: str) -> str:
        return self.link("Ketten", self.kette_name[kid]) if kid in self.kette_name else kid.split(":", 1)[-1]

    @staticmethod
    def zustand(e: Kante) -> str:
        z = e.attribute.get("zustand")
        text = str(e.attribute.get("zustand_text") or "")
        if isinstance(z, dict):
            g = z.get("gemessen")
            wert = g.get("wert") if isinstance(g, dict) else None
            if wert and wert != "keine_probe":
                lesbar = _GEMESSEN.get(str(wert), str(wert))
                text = f"{text} (gemessen: {lesbar})" if text else f"gemessen: {lesbar}"
            if z.get("aus"):
                text += " [aus]"
        return text or "—"

    def hop_zeile(self, e: Kante) -> str:
        belege = "<br>".join(_beleg_text(b) for b in sorted(e.belege, key=lambda b: (b.file, b.zeile or 0, b.anker)))
        return "| " + " | ".join([
            _zelle(e.attribute.get("nr", "")), _zelle(self.knoten_link(e.von)), _zelle(self.knoten_link(e.nach)),
            _zelle(e.attribute.get("mechanismus", "") or "—"), _zelle(self.zustand(e)), _zelle(belege or "—"),
        ]) + " |"

    def kette(self, kid: str, lauf: str) -> _Notiz:
        k = self.ketten[kid]
        kurz = kid.split(":", 1)[1]
        zeilen = [_kopf({"kette": kurz, "zustand": k.attribute.get("zustand_roh", ""), "commit": self.g.commit_aussen,
                         "lauf": lauf}).rstrip("\n"),
                  f"# {kurz} · {k.name}", "",
                  f"Zurück: [[{self.ordner}/{INDEX_NAME}]] · Prüfen: [[{self.ordner}/{PRUEFLISTE_NAME}]]", ""]
        for schl, titel in (("titel_roh", "Titel"), ("zustand_roh", "Zustand"), ("reisst_bei_roh", "Reißt bei")):
            if k.attribute.get(schl):
                zeilen.append(f"- {titel}: {' '.join(str(k.attribute[schl]).split())}")
        zeilen += ["", "## Hops", ""]
        hops = self.hops.get(kid, [])
        if hops:
            zeilen += ["| Nr | Von | Nach | Mechanismus | Zustand | Belege |", "|---|---|---|---|---|---|"]
            zeilen += [self.hop_zeile(e) for e in hops]
        else:
            zeilen.append("Keine Hops im Graph.")
        return _Notiz(f"Ketten/{self.kette_name[kid]}.md", "\n".join(zeilen) + "\n")

    def _ketten_von(self, treffer: Callable[[Kante], bool]) -> list[str]:
        aus = []
        for kid in sorted(self.hops, key=lambda i: _naturlich(i)):
            nrs = [str(e.attribute.get("nr", "")) for e in self.hops[kid] if treffer(e)]
            if nrs:
                aus.append(f"- {self.kette_link(kid)} – Hop {', '.join(nrs)}")
        return aus

    def dienst(self, did: str, lauf: str) -> _Notiz:
        k = self.dienste[did]
        zeilen = [_kopf({"dienst": k.name, "commit": self.g.commit_aussen, "lauf": lauf}).rstrip("\n"),
                  f"# Dienst {k.name}", "", f"Zurück: [[{self.ordner}/{INDEX_NAME}]]", ""]
        for schl in sorted(k.attribute):
            zeilen.append(f"- {schl}: {' '.join(str(k.attribute[schl]).split())}")
        if k.space:
            sid = f"space:{k.space}"
            zeilen.append(f"- Space: {self.knoten_link(sid) if sid in self.space_name else k.space}")
        verb = [e for e in self.g.kanten.values() if e.art == "verbindet" and did in (e.von, e.nach)]
        zeilen += ["", "## Verbindungen", ""]
        for e in sorted(verb, key=lambda e: e.id):
            richtung, ander = ("→", e.nach) if e.von == did else ("←", e.von)
            env = e.attribute.get("env")
            zusatz = f" (Env {', '.join(map(str, env))})" if isinstance(env, list) and env else ""
            zeilen.append(f"- {richtung} {self.knoten_link(ander)}{zusatz}")
        if not verb:
            zeilen.append("Keine Verbindungen im Graph.")
        ketten = self._ketten_von(lambda e: did in (e.von, e.nach))
        zeilen += ["", "## In Ketten", "", *(ketten or ["In keiner Kette."])]
        return _Notiz(f"Dienste/{self.dienst_name[did]}.md", "\n".join(zeilen) + "\n")

    def space(self, sid: str, lauf: str) -> _Notiz:
        k = self.spaces[sid]
        wurzel = (k.datei or "").rstrip("/")

        def im_space(e: Kante) -> bool:
            if sid in (e.von, e.nach):
                return True
            if any(self.g.knoten.get(n) is not None and self.g.knoten[n].space == k.name and n in self.dienste
                   for n in (e.von, e.nach)):
                return True
            return bool(wurzel) and any(b.file == wurzel or b.file.startswith(wurzel + "/") for b in e.belege)

        zeilen = [_kopf({"space": k.name, "commit": self.g.commit_aussen, "lauf": lauf}).rstrip("\n"),
                  f"# Space {k.name}", "", f"Zurück: [[{self.ordner}/{INDEX_NAME}]]", ""]
        if wurzel:
            zeilen.append(f"- Wurzel: {_code(wurzel)}")
        ketten = self._ketten_von(im_space)
        dienste = [self.knoten_link(d) for d, n in sorted(self.dienste.items()) if n.space == k.name]
        zeilen += ["", "## Ketten", "", *(ketten or ["In keiner Kette."]),
                   "", "## Dienste", "", *([f"- {d}" for d in dienste] or ["Keine Dienste zugeordnet."])]
        return _Notiz(f"Spaces/{self.space_name[sid]}.md", "\n".join(zeilen) + "\n")

    def index(self, lauf: str, offen: int, *, rel: str = f"{INDEX_NAME}.md", pruefliste: str = PRUEFLISTE_NAME,
              hinweis: str | None = None, zusatz_links: Sequence[str] = ()) -> _Notiz:
        zeilen = [_kopf({"commit": self.g.commit_aussen, "lauf": lauf}).rstrip("\n"),
                  "# Doku-Loop", "", *([f"> {hinweis}", ""] if hinweis else []),
                  "Erzeugt vom Doku-Loop aus dem Datenfluss-Graph. Von Hand angelegte oder geänderte Notizen in "
                  "diesem Ordner lässt der Loop stehen (er aktualisiert sie dann nicht mehr) und meldet sie einmal in "
                  "der Prüfliste.", "",
                  f"- [[{self.ordner}/{pruefliste}]] – {offen} offene Punkt(e)",
                  *(f"- {z}" for z in zusatz_links), "", "## Ketten", ""]
        for kid in sorted(self.ketten, key=_naturlich):
            k = self.ketten[kid]
            zustand = " ".join(str(k.attribute.get("zustand_roh", "")).split())
            zeilen.append(f"- {self.kette_link(kid)} {k.name}" + (f" – {zustand}" if zustand else ""))
        zeilen += ["", "## Spaces", ""]
        zeilen += [f"- {self.knoten_link(s)}" for s in sorted(self.spaces)]
        zeilen += ["", "## Dienste", ""]
        zeilen += [f"- {self.knoten_link(d)}" for d in sorted(self.dienste, key=lambda d: _naturlich(d))]
        return _Notiz(rel, "\n".join(zeilen) + "\n")


@dataclass(frozen=True)
class _Ausblendung:
    erledigt: frozenset[tuple[str, str]]  # (lauf, kennung) mit angewandtem Urteil
    zurueck: frozenset[tuple[str, str]]  # (lauf, datei) mit angewandtem `zurueck` (M1)

    @classmethod
    def aus(cls, vermerke: Sequence[Mapping[str, object]]) -> "_Ausblendung":
        angewandt = [v for v in vermerke if v.get("ergebnis") == "angewandt"]
        return cls(frozenset((lauf_text(v.get("lauf")), str(v.get("aussage_id"))) for v in angewandt),
                   frozenset((lauf_text(v.get("lauf")), str(v.get("datei"))) for v in angewandt
                             if v.get("art") == "zurueck" and v.get("datei")))


def _pruefliste_notiz(ordner: str, punkte: Sequence[Mapping[str, object]], ledger: Sequence[umschreiben.LedgerEintrag],
                      *, commits: Mapping[str, str], spitzen: Mapping[str, str],
                      vermerke: Sequence[Mapping[str, object]], struktur: Sequence[Mapping[str, object]], lauf: str,
                      urteile_datei: str, rel: str = f"{PRUEFLISTE_NAME}.md", index: str = INDEX_NAME,
                      hinweis: str | None = None) -> _Notiz:
    urteile_link = f"[[{urteile_datei[:-3] if urteile_datei.endswith('.md') else urteile_datei}]]"

    def commit(l: object) -> str:
        sha = commits.get(str(l)) if l else None
        return _code(sha[:10]) if sha else "—"

    def urteilszeilen(l: str, kennung: str, zurueck: bool) -> list[str]:
        formen = ["ok", *(["zurueck"] if zurueck else []),
                  *(["korrektur: <neuer Text>"] if _ist_aussage_id(kennung) else [])]
        return [f"  - {_code(f'{l} {kennung} {f}')}" for f in formen]

    def zitat(text: str) -> list[str]:
        return [f"  > {z}" if z.strip() else "  >" for z in text.split("\n")]

    zeilen = [_kopf({"lauf": lauf}).rstrip("\n"), "# Doku-Prüfliste", "", *([f"> {hinweis}", ""] if hinweis else []),
              f"Zurück: [[{ordner}/{index}]]", "",
              f"So urteilst du: je Punkt eine Zeile in {urteile_link}, in der Form "
              f"{_code('<lauf> <aussage_id> ok|zurueck|korrektur: <text>')}. Unter jedem Punkt stehen die Zeilen, "
              "die dort gelten.", "",
              "- `ok`: passt, der Punkt ist erledigt.",
              "- `zurueck`: die Änderung des Loops an der Datei zurücknehmen (nur, solange sein Commit der letzte "
              "der Datei ist).",
              "- `korrektur: <text>`: die Aussage durch deinen Text ersetzen (eine Zeile, keine Überschrift/Liste).",
              ""]
    zeilen += [f"## Umschreibungen des Loops ({len(ledger)})", ""]
    for e in ledger:
        sha = commits.get(e.lauf)
        an_der_spitze = bool(sha) and spitzen.get(e.datei) == sha
        zeilen += [f"### {e.aussage_id}", "",
                   f"- Art: umschreibung ({e.urteil}, Quelle {e.quelle})",
                   f"- Lauf: {_code(e.lauf)} · Commit: {commit(e.lauf)}"
                   + ("" if an_der_spitze else " · nicht mehr der letzte Commit der Datei, `zurueck` geht nicht"),
                   f"- Datei: {_code(e.datei)}",
                   f"- Beleg: {_beleg_text(e.beleg)}",
                   "- Alter Text:", *zitat(e.alt),
                   "- Neuer Text:", *zitat(e.neu),
                   "- Urteil:", *urteilszeilen(e.lauf, e.aussage_id, an_der_spitze), ""]
    if not ledger:
        zeilen += ["Keine offenen Umschreibungen.", ""]
    zeilen += [f"## Offene Punkte ({len(punkte)})", ""]
    for p in punkte:
        kennung, l = punkt_kennung(p), lauf_text(p.get("lauf"))
        ort = str(p.get("datei") or "—") + (f":{p['zeile']}" if p.get("zeile") else "")
        zeilen += [f"### {p.get('art')}: {kennung}", "",
                   f"- Art: {p.get('art')}",
                   f"- Lauf: {_code(l)} · Commit: {commit(p.get('lauf'))}",
                   f"- Aussage-ID: {_code(kennung)}",
                   f"- Ort: {_code(ort)}",
                   f"- Beleg/Befund: {' '.join(str(p.get('text') or '').split())}",
                   "- Urteil:", *urteilszeilen(l, kennung, False), ""]
    if not punkte:
        zeilen += ["Keine offenen Punkte.", ""]
    angewandt = sorted((v for v in vermerke if v.get("ergebnis") == "angewandt"),
                       key=lambda v: str(v.get("zeit") or ""), reverse=True)
    zeilen += [f"## Angewandte Urteile ({len(angewandt)})", ""]
    for v in angewandt[:ANGEWANDT_MAX]:
        sha = v.get("commit")
        zeilen.append(f"- {v.get('zeit') or '—'} · {v.get('art')} · Lauf {_code(lauf_text(v.get('lauf')))} · "
                      f"{_code(str(v.get('aussage_id')))} · Commit {_code(str(sha)[:10]) if sha else '—'}")
        if v.get("text"):
            zeilen.append(f"  - Text: {' '.join(str(v['text']).split())}")
        if v.get("art") == "zurueck":
            mit = sorted({e.aussage_id for e in v.get("_ledger_mit", ())})  # type: ignore[union-attr]
            if mit:
                zeilen.append(f"  - mit zurückgenommen: {', '.join(_code(a) for a in mit)}")
    if len(angewandt) > ANGEWANDT_MAX:
        zeilen.append(f"- … und {len(angewandt) - ANGEWANDT_MAX} ältere")
    if not angewandt:
        zeilen.append("Noch keine.")
    zeilen += ["", f"## Struktur-Hinweise ({len(struktur)})", "",
               "Verankerungs-, Bau- und Render-Hinweise; sie brauchen kein Urteil.", ""]
    for p in struktur[:STRUKTUR_MAX]:
        ort = str(p.get("datei") or "—") + (f":{p['zeile']}" if p.get("zeile") else "")
        zeilen.append(f"- {p.get('art')}: {' '.join(str(p.get('text') or '').split())} ({_code(ort)})")
    if len(struktur) > STRUKTUR_MAX:
        zeilen.append(f"- … und {len(struktur) - STRUKTUR_MAX} weitere")
    if not struktur:
        zeilen.append("Keine.")
    return _Notiz(rel, "\n".join(zeilen) + "\n")


# ---------- Manifest der geschriebenen Notizen (lokal) ----------

def manifest_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"notizen-{repo_schluessel(konfig.repo)}.json"


FEHLT = "fehlt"
KAPUTT = "kaputt"


def _manifest_laden(pfad: Path | None) -> dict[str, str] | str:
    """rel -> Hash des stabilen Texts; `FEHLT` = keine Datei (Uebernahme erlaubt), `KAPUTT` = unlesbar (fail closed)."""
    if pfad is None or not os.path.lexists(pfad):
        return FEHLT
    try:
        roh = json.loads(Path(pfad).read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return KAPUTT
    dateien = roh.get("dateien") if isinstance(roh, dict) else None
    if not isinstance(dateien, dict):
        return KAPUTT
    return {str(k): str(v) for k, v in dateien.items()}


def _manifest_schreiben(pfad: Path | None, manifest: Mapping[str, str]) -> None:
    if pfad is None:
        return
    text = json.dumps({"version": 1, "dateien": dict(sorted(manifest.items()))}, ensure_ascii=False, indent=1)
    _atomar(Path(pfad), (text + "\n").encode("utf-8"))


def _hash_von(pfad: Path) -> str | None:
    try:
        return _stabil_hash(pfad.read_bytes().decode("utf-8", errors="replace"))
    except OSError:
        return None


def _kopf_lauf(pfad: Path) -> str | None:
    try:
        daten = _frontmatter(Path(pfad).read_bytes()[:4096].decode("utf-8", errors="replace"))
    except OSError:
        return None
    return str(daten["lauf"]) if daten and daten.get("lauf") else None


def _markierte(basis: Path) -> list[str]:
    """Notizen mit Marker in den eigenen Unterordnern (keinem umgeleiteten Weg folgend), rel zum Notizordner."""
    aus: list[str] = []
    for ordner in (basis, basis / "Ketten", basis / "Spaces", basis / "Dienste"):
        if not ordner.is_dir() or not _sicher(basis, ordner):
            continue
        for pfad in sorted(ordner.iterdir()):
            if pfad.suffix == ".md" and not _umgeleitet(pfad) and pfad.is_file() and vom_loop(pfad):
                aus.append(pfad.relative_to(basis).as_posix())
    return aus


class Schreiber:
    """Entscheidet je Notiz: schreiben | gleich | hand | umgeleitet (siehe Moduldoku). Oeffentlich (Task 8), damit
    weitere Seiten denselben Manifest-Schreibschutz nutzen; Verhalten unveraendert."""

    def __init__(self, basis: Path, manifest: dict[str, str], uebernahme: bool, bekannte_laeufe: Collection[str]):
        self.basis, self.manifest, self.uebernahme = basis, manifest, uebernahme
        self.bekannte = set(bekannte_laeufe)

    def entscheiden(self, rel: str, text: str) -> str:
        pfad = self.basis / rel
        if not _sicher(self.basis, pfad):
            return "umgeleitet"
        if not os.path.lexists(pfad):
            return "schreiben"
        aktuell = _hash_von(pfad)
        if aktuell is not None and aktuell == _stabil_hash(text):
            return "gleich"
        if rel in self.manifest:
            return "schreiben" if aktuell == self.manifest[rel] else "hand"
        if self.uebernahme and vom_loop(pfad) and _kopf_lauf(pfad) in self.bekannte:
            return "schreiben"
        return "hand"

    def eigen(self, rel: str) -> bool:
        """Task 9: liegt unter `rel` eine Datei, die der Loop ueberschreiben darf (dieselbe Regel wie `entscheiden`
        fuer eine vorhandene Datei mit anderem Inhalt: Manifest-Hash gleich, oder Uebernahme mit Marker und bekanntem
        Lauf)? Fehlt die Datei oder ist der Weg umgeleitet -> False."""
        pfad = self.basis / rel
        if not _sicher(self.basis, pfad) or not os.path.lexists(pfad):
            return False
        if rel in self.manifest:
            return _hash_von(pfad) == self.manifest[rel]
        return self.uebernahme and vom_loop(pfad) and _kopf_lauf(pfad) in self.bekannte

    def text(self, rel: str) -> str | None:
        """Inhalt der Datei unter `rel` (nur auf sicherem Weg), sonst None."""
        pfad = self.basis / rel
        if not _sicher(self.basis, pfad):
            return None
        try:
            return pfad.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return None


def schreiber_fuer(wurzel: Path, ordner: str, manifest_pfad: Path | None,
                   bekannte_laeufe: Collection[str] = ()) -> Schreiber | None:
    """Task 9: derselbe `Schreiber`, den `notizen_schreiben` baut (Manifest, Uebernahme, bekannte Laeufe) -- zum
    Entscheiden VOR dem Schreiben. None, wenn `notizen_schreiben` nichts schriebe (Ordner umgeleitet, Manifest
    unlesbar, Pfad ungueltig)."""
    wurzel = Path(wurzel)
    if _umgeleitet(wurzel / ordner):
        return None
    try:
        basis = _unter(wurzel, ordner)
    except ValueError:
        return None
    geladen = _manifest_laden(manifest_pfad)
    if geladen == KAPUTT:
        return None
    return Schreiber(basis, {} if isinstance(geladen, str) else dict(geladen), geladen == FEHLT, bekannte_laeufe)


_Schreiber = Schreiber  # alter Name
Notiz = _Notiz  # oeffentlicher Name: eine zu schreibende Notiz (`rel` ab Notizordner, `text`)


def offene_ohne_erledigte(pruefliste: Sequence[Pruefpunkt | Mapping[str, object]],
                          vermerke: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Offene Punkte ohne die, zu denen ein Urteil schon angewandt ist (wie `notizen_schreiben` sie zeigt)."""
    aus = _Ausblendung.aus(vermerke)
    return [d for d in map(_punkt_dict, pruefliste) if (lauf_text(d.get("lauf")), punkt_kennung(d)) not in aus.erledigt]


def notizen_schreiben(graph: Graph, pruefliste: Sequence[Pruefpunkt | Mapping[str, object]],
                      ledger_neu: Sequence[umschreiben.LedgerEintrag], wurzel: Path, ordner: str, lauf: str, *,
                      manifest_pfad: Path | None = None, commits: Mapping[str, str] | None = None,
                      spitzen: Mapping[str, str] | None = None, vermerke: Sequence[Mapping[str, object]] = (),
                      struktur: Sequence[Pruefpunkt | Mapping[str, object]] = (),
                      urteile_datei: str = "Doku-Urteile/Urteile.md", bekannte_laeufe: Collection[str] = (),
                      melden: Melden | None = None, pruefpunkte: list[Pruefpunkt] | None = None,
                      zusatz: Sequence[object] = (), zusatz_links: Sequence[str] = (),
                      zusatz_ausgefallen: Collection[str] = (), trocken: bool = False,
                      spaces: bool = True, vorhandene_seiten: Collection[str] | None = None) -> list[Path]:
    """Notizen schreiben (siehe Moduldoku); Rueckgabe die tatsaechlich geschriebenen Dateien.

    `pruefliste`: offene Punkte (Dicts aus `pruefliste_laden` mit `lauf`, oder Pruefpunkte); `ledger_neu`: Ledger-
    Eintraege (Umschreibungen); `commits`: Lauf -> Commit-sha (`lauf_commits`); `spitzen`: Datei -> sha ihres letzten
    Commits (`datei_spitzen`; `zurueck` wird nur angeboten, wenn beide gleich sind); `vermerke`: lokale Urteils-Vermerke
    (`vermerke_laden`); `struktur`: Struktur-Hinweise; `bekannte_laeufe`: Laeufe des Loops (nur fuer die Uebernahme
    ohne Manifest). Von Hand angelegte/geaenderte Notizen (`notiz_von_hand`) und ein kaputtes Manifest
    (`notizen_manifest_kaputt`) kommen in `pruefpunkte`.

    `zusatz`: weitere Notizen (Task 8, `rowboat_seiten`) -- sie laufen durch denselben Schreibschutz und dasselbe
    Manifest wie alle anderen (sonst raeumte der Lauf sie als "nicht mehr erzeugt" wieder weg); `zusatz_links`:
    Wiki-Link-Zeilen, die `Index.md` zusaetzlich auffuehrt. Ein Eintrag von `zusatz` ist eine `Notiz` oder hat
    `mit_namen(index_name, pruefliste_name) -> Notiz` (die Namen sind die der Rueckverweis-Ziele, `... (Loop)` bei
    Hand-Aenderung); `zusatz_ausgefallen`: Pfade von Seiten, die in diesem Lauf nicht gebaut werden konnten -- ihre
    Dateien bleiben stehen (kein Aufraeumen). `trocken`: entscheidet wie ein echter Lauf, schreibt/loescht aber
    nichts (auch nicht das Manifest); Rueckgabe = die Dateien, die ein echter Lauf schreiben wuerde.

    `spaces=False` (Task 9, Ruling T9-a): die deterministischen Space-Notizen `Spaces/<name>.md` entstehen hier nicht
    mehr -- die Pfade gehoeren dann den erzaehlten Seiten (`rowboat_erzaehlt`), die als `zusatz` kommen (oder, solange
    sie fehlen, als `zusatz_ausgefallen` stehen bleiben). `vorhandene_seiten` (Fix-Runde 1): mit `spaces=False`
    verlinken die Notizen einen Space nur, wenn seine Seite in dieser Menge ist (sonst steht nur sein Name da).

    Schluessel-Filter (Fix-Runde 1, global.md): jede Notiz und Seite wird vor dem Schreiben mit `schluessel_muster`
    geprueft. Treffer -> diese Notiz wird nicht geschrieben (eine vorhandene Fassung bleibt) und `notiz_verdacht_schluessel`
    nennt nur das Muster. Pruefliste und Index werden stattdessen geschwaerzt (`schwaerzen`), damit sie nie einfrieren:
    die Pruefliste traegt Texte aus Judge-Urteilen."""
    melden = melden or _still
    gemeldet = pruefpunkte if pruefpunkte is not None else []
    wurzel = Path(wurzel)
    if _umgeleitet(wurzel / ordner):
        melden(f"Notizen: {ordner}/ ist umgeleitet (Symlink/Junction) – nichts geschrieben")
        return []
    basis = _unter(wurzel, ordner)
    geladen = _manifest_laden(manifest_pfad)
    if geladen == KAPUTT:
        gemeldet.append(Pruefpunkt("notizen_manifest_kaputt", f"Manifest {Path(str(manifest_pfad)).name} unlesbar; "
                                   "keine Notiz geschrieben, bis es geloescht oder repariert ist",
                                   str(manifest_pfad), None, None))
        melden("Notizen: Manifest unlesbar – nichts geschrieben (fail closed)")
        return []
    uebernahme = geladen == FEHLT
    manifest: dict[str, str] = {} if isinstance(geladen, str) else dict(geladen)
    s = Schreiber(basis, manifest, uebernahme, bekannte_laeufe)

    commits, spitzen = commits or {}, spitzen or {}
    aus = _Ausblendung.aus(vermerke)
    n = _Notizen(graph, ordner, None if spaces else vorhandene_seiten)
    punkte = [d for d in map(_punkt_dict, pruefliste)
              if (lauf_text(d.get("lauf")), punkt_kennung(d)) not in aus.erledigt]
    alle_ledger = [e for e in ledger_neu if e.quelle != "review"]
    ledger = [e for e in alle_ledger
              if (e.lauf, e.aussage_id) not in aus.erledigt and (e.lauf, e.datei) not in aus.zurueck]
    vermerke_anzeige = []
    for v in vermerke:  # M1: welche Umschreibungen ein `zurueck` mitgenommen hat
        d = dict(v)
        if d.get("art") == "zurueck" and d.get("ergebnis") == "angewandt":
            d["_ledger_mit"] = [e for e in alle_ledger if e.lauf == d.get("lauf") and e.datei == d.get("datei")]
        vermerke_anzeige.append(d)
    offen = len(punkte) + len(ledger)

    def pruefliste_notiz(hand_pl: bool, hand_ix: bool) -> _Notiz:
        hinweis = (f"`{INDEX_NAME}.md` ist von Hand geändert; die Fassung des Loops steht in "
                   f"[[{n.ordner}/{INDEX_NAME} (Loop)]].") if hand_ix else None
        return _pruefliste_notiz(n.ordner, punkte, ledger, commits=commits, spitzen=spitzen,
                                 vermerke=vermerke_anzeige, struktur=[_punkt_dict(p) for p in struktur], lauf=lauf,
                                 urteile_datei=urteile_datei,
                                 rel=f"{PRUEFLISTE_NAME} (Loop).md" if hand_pl else f"{PRUEFLISTE_NAME}.md",
                                 index=f"{INDEX_NAME} (Loop)" if hand_ix else INDEX_NAME, hinweis=hinweis)

    def index_notiz(hand_pl: bool, hand_ix: bool) -> _Notiz:
        hinweis = (f"`{PRUEFLISTE_NAME}.md` ist von Hand geändert; die aktuelle Prüfliste des Loops steht in "
                   f"[[{n.ordner}/{PRUEFLISTE_NAME} (Loop)]].") if hand_pl else None
        return n.index(lauf, offen, rel=f"{INDEX_NAME} (Loop).md" if hand_ix else f"{INDEX_NAME}.md",
                       pruefliste=f"{PRUEFLISTE_NAME} (Loop)" if hand_pl else PRUEFLISTE_NAME, hinweis=hinweis,
                       zusatz_links=zusatz_links)

    # Item 3: von Hand geaenderte Pruefliste/Index -> Fassung des Loops daneben
    hand_pl = s.entscheiden(f"{PRUEFLISTE_NAME}.md", pruefliste_notiz(False, False).text) == "hand"
    hand_ix = s.entscheiden(f"{INDEX_NAME}.md", index_notiz(False, False).text) == "hand"
    von_hand = [r for r, h in ((f"{PRUEFLISTE_NAME}.md", hand_pl), (f"{INDEX_NAME}.md", hand_ix)) if h]

    notizen = [n.kette(k, lauf) for k in sorted(n.ketten, key=_naturlich)]
    if spaces:
        notizen += [n.space(sp, lauf) for sp in sorted(n.spaces)]
    notizen += [n.dienst(d, lauf) for d in sorted(n.dienste)]
    notizen += [pruefliste_notiz(hand_pl, hand_ix), index_notiz(hand_pl, hand_ix)]
    idx_name = f"{INDEX_NAME} (Loop)" if hand_ix else INDEX_NAME
    pl_name = f"{PRUEFLISTE_NAME} (Loop)" if hand_pl else PRUEFLISTE_NAME
    for z in zusatz:
        notizen.append(z.mit_namen(idx_name, pl_name) if hasattr(z, "mit_namen") else z)  # type: ignore[attr-defined]

    geschrieben: list[Path] = []
    umgeleitet = 0
    soll: set[str] = set()
    navigation = {f"{PRUEFLISTE_NAME}.md", f"{PRUEFLISTE_NAME} (Loop).md", f"{INDEX_NAME}.md", f"{INDEX_NAME} (Loop).md"}
    for notiz in notizen:
        soll.add(notiz.rel)
        ort = f"{ordner}/{schwaerzen(notiz.rel)}"
        if notiz.rel in navigation:
            sauber = schwaerzen(notiz.text)
            if sauber != notiz.text:
                gemeldet.append(Pruefpunkt("notiz_verdacht_schluessel", f"{ort}: enthielt etwas, das nach einem "
                                           "Schlüssel aussieht; geschwärzt geschrieben", ort, None, None))
                notiz = _Notiz(notiz.rel, sauber)
        muster = schluessel_muster(notiz.text)
        if muster is not None:  # nie schreiben; die vorhandene Fassung bleibt (rel ist in `soll`, Manifest unveraendert)
            gemeldet.append(Pruefpunkt("notiz_verdacht_schluessel", f"{ort}: sieht nach einem Schlüssel aus (Muster: "
                                       f"{muster}); nicht geschrieben", ort, None, None))
            continue
        entscheid = s.entscheiden(notiz.rel, notiz.text)
        if entscheid == "umgeleitet":
            umgeleitet += 1
        elif entscheid == "gleich":
            manifest[notiz.rel] = _stabil_hash(notiz.text)
        elif entscheid == "hand":
            von_hand.append(notiz.rel)  # nicht (mehr) vom Loop: stehen lassen
            manifest.pop(notiz.rel, None)
        else:
            if not trocken:
                _atomar(basis / notiz.rel, notiz.text.encode("utf-8"))
            manifest[notiz.rel] = _stabil_hash(notiz.text)
            geschrieben.append(basis / notiz.rel)
    soll |= set(zusatz_ausgefallen)
    weg = 0
    if uebernahme:  # die Uebernahme loescht nie; markierte Dateien ausserhalb von `soll` gelten als von Hand
        von_hand += [rel for rel in _markierte(basis) if rel not in soll and rel not in von_hand]
    else:
        for rel in sorted(set(manifest) - soll):
            pfad = basis / rel
            bekannt = manifest.pop(rel)
            if not _sicher(basis, pfad) or not os.path.lexists(pfad):
                continue
            if _hash_von(pfad) != bekannt:
                von_hand.append(rel)
                continue
            if trocken:
                continue
            try:
                pfad.unlink()
                weg += 1
            except OSError:
                pass
    if not trocken:
        _manifest_schreiben(manifest_pfad, manifest)
    for rel in dict.fromkeys(von_hand):
        gemeldet.append(Pruefpunkt("notiz_von_hand", f"{ordner}/{rel} wurde von Hand angelegt oder geändert; "
                                   "der Loop lässt sie stehen und aktualisiert sie nicht mehr",
                                   f"{ordner}/{rel}", None, None))
    melden(f"Notizen: {len(geschrieben)} geschrieben, {weg} entfernt, {len(von_hand)} von Hand, "
           f"{umgeleitet} umgeleitet (nicht angefasst)" + (" – Übernahme ohne Manifest" if uebernahme else ""))
    return geschrieben


# ---------- Urteile lesen ----------

@dataclass(frozen=True)
class Urteil:
    lauf: str
    aussage_id: str
    art: str  # ok | zurueck | korrektur
    text: str | None


_KORREKTUR = re.compile(r"^(\S+)\s+(.+?)\s+korrektur:(.*)$", re.IGNORECASE)
_EINFACH = re.compile(r"^(\S+)\s+(.+?)\s+(ok|zurueck|zurück)$", re.IGNORECASE)


def _zeile_klar(zeile: str) -> str:
    klar = zeile.strip()
    klar = re.sub(r"^[-*+]\s+", "", klar)
    if klar.startswith("``") and klar.endswith("``") and len(klar) > 4:
        klar = klar[2:-2].strip()
    elif klar.startswith("`") and klar.endswith("`") and len(klar) > 2:
        klar = klar[1:-1].strip()
    return klar


def urteile_lesen(pfad: Path, *, unlesbar: list[Pruefpunkt] | None = None, datei: str | None = None) -> list[Urteil]:
    """Urteile aus der Datei (fehlt sie -> []). Unlesbare Zeilen kommen (dedupliziert) als `urteil_unlesbar` in
    `unlesbar`."""
    try:
        text = Path(pfad).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return []
    zeilen = text.lstrip("﻿").replace("\r\n", "\n").split("\n")
    start = 0
    if zeilen and zeilen[0].strip() == "---":
        for j in range(1, len(zeilen)):
            if zeilen[j].strip() in ("---", "..."):
                start = j + 1
                break
    aus: list[Urteil] = []
    gesehen: set[str] = set()
    im_kommentar = False
    for i in range(start, len(zeilen)):
        roh = zeilen[i].strip()
        if im_kommentar:
            im_kommentar = "-->" not in roh
            continue
        if roh.startswith("<!--"):
            im_kommentar = "-->" not in roh
            continue
        if not roh or roh.startswith(("#", ">")):
            continue
        klar = _zeile_klar(roh)
        m = _KORREKTUR.match(klar)
        if m and m.group(3).strip():
            aus.append(Urteil(m.group(1), m.group(2).strip(), "korrektur", m.group(3).strip()))
            continue
        m = _EINFACH.match(klar) if not m else None
        if m:
            art = m.group(3).lower().replace("ü", "ue")
            aus.append(Urteil(m.group(1), m.group(2).strip(), art, None))
            continue
        if unlesbar is not None and klar not in gesehen:
            gesehen.add(klar)
            unlesbar.append(Pruefpunkt("urteil_unlesbar", f"nicht lesbar: {klar[:200]}",
                                       datei or Path(pfad).name, i + 1, None))
    return aus


# ---------- Erledigt (lokal) ----------

def erledigt_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"urteile-erledigt-{repo_schluessel(konfig.repo)}.jsonl"


def _urteil_schluessel(u: Urteil) -> str:
    return json.dumps([u.lauf, u.aussage_id, u.art, u.text or ""], ensure_ascii=False)


def _erledigt_zeilen(pfad: Path) -> list[dict[str, object]]:
    try:
        text = Path(pfad).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return []
    aus = []
    for z in text.splitlines():
        try:
            d = json.loads(z)
        except ValueError:
            continue
        if isinstance(d, dict):
            aus.append(d)
    return aus


def vermerke_laden(pfad: Path) -> list[dict[str, object]]:
    """Alle lokalen Urteils-Vermerke (angewandt und abgelehnt) in Schreibreihenfolge."""
    return _erledigt_zeilen(pfad)


def erledigt_laden(pfad: Path) -> dict[tuple[str, str], str]:
    """(lauf, kennung) -> Art des angewandten Urteils; abgelehnte Urteile blenden nichts aus."""
    return {(lauf_text(d.get("lauf")), str(d.get("aussage_id"))): str(d.get("art"))
            for d in _erledigt_zeilen(pfad) if d.get("ergebnis") == "angewandt"}


def offene_urteile(urteile: Sequence[Urteil], pfad: Path) -> list[Urteil]:
    """Urteile, die noch nie (angewandt oder endgueltig abgelehnt) vermerkt sind; Doppelte nur einmal."""
    bekannt = {str(d.get("schluessel")) for d in _erledigt_zeilen(pfad)}
    aus: list[Urteil] = []
    for u in urteile:
        s = _urteil_schluessel(u)
        if s not in bekannt:
            bekannt.add(s)
            aus.append(u)
    return aus


def _vermerken(pfad: Path, u: Urteil, ergebnis: str, jetzt: datetime, grund: str = "", *,
               commit: str | None = None, datei: str | None = None) -> None:
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    zeile = json.dumps({"schluessel": _urteil_schluessel(u), "lauf": u.lauf, "aussage_id": u.aussage_id,
                        "art": u.art, "text": u.text, "ergebnis": ergebnis, "grund": grund,
                        "zeit": jetzt.isoformat(), "commit": commit, "datei": datei},
                       ensure_ascii=False, sort_keys=True) + "\n"
    with open(pfad, "ab") as f:
        f.write(zeile.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


# ---------- Ledger und Commits ----------

def ledger_lesen(pfad: Path) -> list[umschreiben.LedgerEintrag]:
    try:
        text = Path(pfad).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return []
    aus = []
    for z in text.splitlines():
        try:
            d = json.loads(z)
            if isinstance(d, dict):
                aus.append(umschreiben.LedgerEintrag.from_dict(d))
        except (ValueError, KeyError, TypeError):
            continue
    return aus


def lauf_commits(repo: Path) -> dict[str, str]:
    """Lauf -> sha des juengsten Commits mit der Zeile `Lauf: <lauf>` (ein `git log` fuer alle Laeufe)."""
    aus = git(Path(repo), "log", "-F", "--grep=Lauf: ", "--format=%H%x00%B%x01") or ""
    commits: dict[str, str] = {}
    for block in aus.split("\x01"):
        sha, _, nachricht = block.strip("\n").partition("\x00")
        for z in nachricht.splitlines():
            if z.startswith("Lauf: ") and sha.strip():
                commits.setdefault(z[len("Lauf: "):].strip(), sha.strip())
    return commits


def datei_spitzen(repo: Path, dateien: Iterable[str]) -> dict[str, str]:
    """Datei -> sha des letzten Commits, der sie beruehrt hat (M3: `zurueck` geht nur dort)."""
    aus: dict[str, str] = {}
    for d in sorted(set(dateien)):
        sha = (git(Path(repo), "log", "-1", "--format=%H", "--", d) or "").strip()
        if sha:
            aus[d] = sha
    return aus


# ---------- Aussagen ohne Verankerung ----------

def aussagen_ohne_belege(datei: str, text: str) -> list[au.Aussage]:
    """Aussagen einer Doku-Datei mit denselben ids wie `aussagen_lesen`, aber ohne Verankerung (billig)."""
    roh = au._zeilen_mit_ende(text)
    zeilen = [au._inhalt(z, i == 0) for i, z in enumerate(roh)]
    einheiten, _ = au._einheiten(zeilen, au._block_zeilen(text))
    aus: list[au.Aussage] = []
    gesehen: dict[str, int] = {}
    for e in einheiten:
        inhalt = "\n".join(zeilen[e.start:e.ende + 1])
        aid = au.aussage_id(datei, e.ueberschrift, inhalt)
        if aid in gesehen:
            gesehen[aid] += 1
            aid = f"{aid}~{gesehen[aid]}"
        else:
            gesehen[aid] = 1
        aus.append(au.Aussage(aid, datei, e.ueberschrift, e.start + 1, e.ende + 1, inhalt, ()))
    return aus


def _kandidaten_ids(aid: str, eintrag: umschreiben.LedgerEintrag | None) -> list[str]:
    """Die id selbst und -- nach einer Umschreibung durch den Loop -- die id des neuen Texts."""
    ids = [aid]
    if eintrag is not None and "#" in aid:
        datei, rest = aid.split("#", 1)
        ueberschrift = rest.rsplit("#", 1)[0] if "#" in rest else ""
        ids.append(au.aussage_id(datei, ueberschrift, eintrag.neu))
    return ids


# ---------- Urteile anwenden ----------

KORREKTUR_MAX = 2000
_KORREKTUR_UNSICHER = re.compile(r"<\s*script|javascript\s*:|<\s*iframe|onerror\s*=", re.IGNORECASE)


def korrektur_pruefen(aussage: au.Aussage, text: str) -> str | None:
    """Minor 16: Grund, warum eine `korrektur` aus Urteile.md abgelehnt wird, sonst None. Hoechstens doppelt so lang
    wie die alte Aussage (wie beim Judge) und hoechstens KORREKTUR_MAX Zeichen; keine Zeile mit `<script`,
    `javascript:`, `<iframe` oder `onerror=` (Urteile.md ist nicht authentifiziert, die Doku wird gerendert)."""
    if len(text) > 2 * len(aussage.text) or len(text) > KORREKTUR_MAX:
        return (f"Korrektur zu lang ({len(text)} Zeichen; erlaubt hoechstens {min(2 * len(aussage.text), KORREKTUR_MAX)})")
    if any(_KORREKTUR_UNSICHER.search(z) for z in text.splitlines()):
        return "Korrektur enthaelt unsicheren Inhalt (<script, javascript:, <iframe oder onerror=)"
    return None

class _Anwender:
    def __init__(self, repo: Path, ledger_pfad: Path, erledigt: Path, *, lauf: str, jetzt: datetime, drift: Drift,
                 melden: Melden, doku: Sequence[str], offene_punkte: Sequence[Mapping[str, object]],
                 zweig: str | None = None, vermerk: Path | None = None,
                 fremd_commits: Mapping[str, str] | None = None, fremd: Sequence[str] = ()) -> None:
        self.repo, self.erledigt, self.lauf, self.jetzt = Path(repo), Path(erledigt), lauf, jetzt
        self.fremd = set(fremd)  # Ruling T4-b (Fix-Runde 1): Doku-Ziele mit fremden Aenderungen -> deren Urteile warten
        self.zweig, self.vermerk, self.fremd_commits = zweig, vermerk, dict(fremd_commits or {})
        self.drift, self.melden, self.doku = drift, melden, list(doku)
        self.ledger = {(e.lauf, e.aussage_id): e for e in ledger_lesen(ledger_pfad)}
        self.punkte = {(lauf_text(p.get("lauf")), punkt_kennung(p)): p for p in offene_punkte}
        self.vorher: set[umschreiben.DriftFehler] | None = None
        self.vorher_fehlt = False  # Waechter lief vorher nicht: kein Vergleich nach einem Revert
        self._commits: dict[str, str] | None = None
        self.nr = 0
        self.pp: list[Pruefpunkt] = []

    def punkt(self, art: str, text: str, datei: str | None, u: Urteil) -> None:
        self.pp.append(Pruefpunkt(art, text[:500], datei, None, u.aussage_id))

    def vermerken(self, u: Urteil, ergebnis: str, grund: str = "", **kw: str | None) -> None:
        _vermerken(self.erledigt, u, ergebnis, self.jetzt, grund, **kw)

    def commits(self) -> dict[str, str]:
        if self._commits is None:  # I11: auch Laeufe, deren Schreibung eine andere Session committet hat
            self._commits = {**self.fremd_commits, **lauf_commits(self.repo)}
        return self._commits

    def waechter_vorher(self, u: Urteil, datei: str) -> set[umschreiben.DriftFehler] | None:
        if self.vorher is None and not self.vorher_fehlt:
            try:
                self.vorher = self.drift(self.repo)
            except umschreiben.TorwaechterFehler as fehler:
                self.vorher_fehlt = True
                self.punkt("torwaechter_fehler", f"Wächter vor dem Urteil: {fehler}", datei, u)
        return self.vorher

    def anwenden(self, u: Urteil) -> None:
        schl = (u.lauf, u.aussage_id)
        e, p = self.ledger.get(schl), self.punkte.get(schl)
        if u.art == "ok":
            bezug = e is not None or p is not None
        elif u.art == "zurueck":
            bezug = e is not None
        else:
            bezug = e is not None or (p is not None and _ist_aussage_id(u.aussage_id))
        if not bezug:  # Ruling 54
            self.punkt("urteil_ohne_bezug", f"{u.lauf} {u.aussage_id} {u.art}: kein passender Ledger-Eintrag bzw. "
                                            "offener Punkt in der Prüfliste", None, u)
            self.vermerken(u, "abgelehnt", "ohne_bezug")
        elif u.art == "ok":
            self.vermerken(u, "angewandt")
        elif u.art == "zurueck" and e is not None:
            self.zurueck(u, e)
        else:
            self.korrektur(u, e)

    def wartet(self, u: Urteil, datei: str) -> bool:
        """Fix-Runde 1, Item 1: das Ziel hat fremde, nicht committete Aenderungen -> nicht vermerken, spaeter erneut."""
        if datei not in self.fremd:
            return False
        self.melden(f"Urteile: {u.art} für {datei} wartet (fremde Änderungen an der Datei)")
        return True

    def zurueck(self, u: Urteil, e: umschreiben.LedgerEintrag) -> None:
        datei = e.datei
        if datei not in self.doku:
            self.punkt("zurueck_nicht_moeglich", f"{datei} ist keine Doku-Datei des Loops", datei, u)
            self.vermerken(u, "abgelehnt", "datei")
            return
        if self.wartet(u, datei):
            return
        sha = self.commits().get(e.lauf)  # GENAU `Lauf: <lauf>`, kein Teilstring
        if sha is None:
            self.punkt("zurueck_nicht_moeglich", f"{e.lauf}: kein Loop-Commit mit genau 'Lauf: {e.lauf}'", datei, u)
            self.vermerken(u, "abgelehnt", "kein_commit")
            return
        vorher = self.waechter_vorher(u, datei)
        kopf_grund = kopf_pruefen(self.repo, self.zweig)  # C1 (Final-Fix): nie auf detached HEAD/laufende Operation
        if kopf_grund is not None:
            self.punkt("zurueck_nicht_moeglich", f"Revert {sha[:7]} von {datei}: {kopf_grund} (wird erneut versucht)",
                       datei, u)
            return
        try:
            ok = umschreiben.revertieren(self.repo, sha, datei)
        except umschreiben.CommitUnbestaetigt as fehler:
            self.punkt("commit_unbestaetigt", f"Revert {sha[:7]}: {fehler}", datei, u)
            self.vermerken(u, "angewandt", "unbestaetigt", commit=fehler.sha, datei=datei)
            return
        except (umschreiben.GitFehler, umschreiben.FremdeAenderung, ValueError, OSError) as fehler:
            self.punkt("zurueck_nicht_moeglich", f"Revert {sha[:7]} von {datei}: {type(fehler).__name__} "
                                                 "(wird erneut versucht)", datei, u)
            return
        if not ok:
            self.punkt("zurueck_nicht_moeglich", f"Commit {sha[:7]} ist nicht mehr der letzte Commit von {datei} "
                                                 "(oder nichts zurückzunehmen); bitte von Hand", datei, u)
            self.vermerken(u, "abgelehnt", "nicht_an_der_spitze")
            return
        neu = kopf(self.repo)
        self.melden(f"Urteile: Revert von {datei} aus {sha[:7]}")
        self.vermerken(u, "angewandt", commit=neu, datei=datei)
        if vorher is None:
            return
        try:  # Item 4: Torwaechter nach dem Revert -- nicht blockierend
            nachher = self.drift(self.repo)
        except umschreiben.TorwaechterFehler as fehler:
            self.punkt("zurueck_drift", f"Wächter nach dem Revert {(neu or '')[:7]} lief nicht: {fehler}", datei, u)
            self.vorher, self.vorher_fehlt = None, True
            return
        dazu = sorted(nachher - vorher)
        if dazu:
            self.punkt("zurueck_drift", f"Revert {(neu or '')[:7]} brachte {len(dazu)} neue Drift-Fehler: "
                       + "; ".join(f"{d}: {m}" for d, m in dazu[:5]), datei, u)
        self.vorher = nachher  # Basis fuer weitere Urteile in diesem Lauf

    def korrektur(self, u: Urteil, e: umschreiben.LedgerEintrag | None) -> None:
        datei = e.datei if e is not None else u.aussage_id.split("#", 1)[0]
        if datei not in self.doku:
            self.punkt("korrektur_abgelehnt", f"{datei} ist keine Doku-Datei des Loops", datei, u)
            self.vermerken(u, "abgelehnt", "datei")
            return
        if self.wartet(u, datei):
            return
        try:
            roh = (self.repo / datei).read_bytes()
        except OSError as fehler:
            self.punkt("korrektur_abgelehnt", f"{datei} nicht lesbar ({type(fehler).__name__})", datei, u)
            return
        text = roh.decode("utf-8")
        if au.handgepflegt(text):
            self.punkt("korrektur_abgelehnt", f"{datei} ist handgepflegt", datei, u)
            self.vermerken(u, "abgelehnt", "handgepflegt")
            return
        ids = _kandidaten_ids(u.aussage_id, e)
        liste = aussagen_ohne_belege(datei, text)
        aussage = next((a for i in ids for a in liste if a.id == i), None)
        if aussage is None:
            self.punkt("korrektur_abgelehnt", f"Aussage {u.aussage_id} nicht (mehr) in {datei}", datei, u)
            self.vermerken(u, "abgelehnt", "nicht_gefunden")
            return
        neu_text = u.text or ""
        grund = korrektur_pruefen(aussage, neu_text)
        if grund is not None:  # Minor 16: Laenge und Inhalt wie beim Judge begrenzt
            self.punkt("korrektur_abgelehnt", grund, datei, u)
            self.vermerken(u, "abgelehnt", "inhalt")
            return
        try:
            neu = au.aussage_ersetzen(text, aussage, neu_text)
        except ValueError as fehler:
            self.punkt("korrektur_abgelehnt", f"Strukturwächter: {fehler}", datei, u)
            self.vermerken(u, "abgelehnt", "struktur")
            return
        vorher = self.waechter_vorher(u, datei)
        if vorher is None:
            return  # ohne Waechter kein Commit; naechster Lauf versucht es erneut
        self.nr += 1
        lauf = f"review-{self.lauf}-{self.nr:02d}"  # nie Praefix eines anderen Laufs
        basis = kopf(self.repo)
        beleg = e.beleg if e is not None else Beleg(
            "datei", datei, "", hashlib.sha256(neu.encode("utf-8")).hexdigest(), basis or "", None)
        eintrag = umschreiben.LedgerEintrag(lauf=lauf, zeit=self.jetzt.isoformat(), aussage_id=aussage.id,
                                            datei=datei, alt=aussage.text, neu="\n".join(au._ersatz_zeilen(neu_text)),
                                            beleg=beleg, urteil="korrektur", quelle="review", basis_commit=basis)
        aus = schreiben_pruefen_committen(self.repo, [Aenderung(datei, roh, neu)],
                                          f"docs(loop): Korrektur aus Review {u.aussage_id}", lauf,
                                          vorher=vorher, drift=self.drift, ledger=[eintrag], melden=self.melden,
                                          pp_datei=datei, zweig=self.zweig, vermerk=self.vermerk)
        self.pp.extend(Pruefpunkt(p.art, p.text, p.datei, p.zeile, u.aussage_id) for p in aus.pruefpunkte)
        if aus.commit is not None or aus.fremd_commit is not None:
            sha = aus.commit or aus.fremd_commit or ""
            self.vermerken(u, "angewandt", commit=sha, datei=datei)
            self.melden(f"Urteile: Korrektur committet ({sha[:7]})")
        elif aus.uebersprungen == "torwaechter:torwaechter_blockiert":
            self.vermerken(u, "abgelehnt", "torwaechter")


def urteile_anwenden(urteile: Sequence[Urteil], repo: Path, ledger_pfad: Path, erledigt_pfad: Path, *, lauf: str,
                     jetzt: datetime, drift: Drift, melden: Melden | None = None, doku: Sequence[str] = (),
                     offene_punkte: Sequence[Mapping[str, object]] = (), zweig: str | None = None,
                     vermerk: Path | None = None, fremd_commits: Mapping[str, str] | None = None,
                     fremd: Sequence[str] = ()) -> list[Pruefpunkt]:
    """Noch nicht vermerkte Urteile anwenden (siehe Moduldoku); Rueckgabe die Pruefpunkte.

    `offene_punkte`: die offenen Punkte der lokalen Pruefliste (Bezug fuer `ok`/`korrektur`, Ruling 54). Der Aufrufer
    stellt sicher, dass KETTEN.md, weitere Doku und docs/graph keine fremden Aenderungen haben (sonst warten alle
    Urteile); `fremd` nennt Doku-Ziele mit fremden Aenderungen, nur deren Urteile warten (nicht vermerkt). Vorlaeufige Fehler
    (Torwaechter laeuft nicht, Schreib-/Commitfehler) werden nicht vermerkt und beim naechsten Mal erneut versucht;
    inhaltliche Ablehnungen werden vermerkt und nie wiederholt."""
    a = _Anwender(Path(repo), Path(ledger_pfad), Path(erledigt_pfad), lauf=lauf, jetzt=jetzt, drift=drift,
                  melden=melden or _still, doku=doku, offene_punkte=offene_punkte, zweig=zweig, vermerk=vermerk,
                  fremd_commits=fremd_commits, fremd=fremd)
    for u in offene_urteile(urteile, Path(erledigt_pfad)):
        try:  # Fix-Runde 2, Item 4: ein Urteil, das wirft, haelt die uebrigen nicht auf (nicht vermerkt -> erneut)
            a.anwenden(u)
        except Exception as fehler:  # noqa: BLE001
            a.punkt("urteil_fehler", f"{u.lauf} {u.aussage_id} {u.art}: {type(fehler).__name__}: {fehler}", None, u)
    return a.pp
