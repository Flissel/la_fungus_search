"""Tick des Doku-Loops: ein Lauf von Anstoss bis Commit (Task 15, Fix-Runde 1).

Reihenfolge (C.7 mit den Uebergaben aus Task 9-14 und den Rulings 44-46):

 1. Sperre (`stand.sperre`), eigene Temp-Reste aufraeumen, Anstoss pruefen, Marker loeschen. Ohne
    `docs/graph/graph.json` -> `kein_seed` (erst `seed`).
 2. Geaenderte Dateien seit dem lokalen Stand -> betroffene Spaces (kein Stand -> alle; `config/dienste.yaml`,
    `config/doku_loop.yaml`, eine `*.sql` unter `sql_wurzeln` oder eine `compose_live`-Datei geaendert -> alle,
    Final-Fix I5). Geaenderte Dateien ausserhalb der Spaces, auf die gemerkte
    Belege zeigen -> Pruefphase ohne Neubau (Item 7). Offene Signale (Budget/Timeout im letzten Lauf) werden
    ohne Neubau erneut gefragt; nur ein neues Urteil fuehrt zum Neubau (Ruling 45c). Nichts davon ->
    `keine_aenderung`.
 3. Billige Vorpruefungen VOR jedem Judge-Aufruf: fremde Aenderungen an KETTEN.md/Doku/`docs/graph/*`
    (ueberspringen, nie stashen), Torwaechter vorher.
 4. Manifeste und `graph_bauen` (explizite Commits) nur fuer betroffene Spaces, `belege_auffrischen`.
 5. Belege pruefen. Nur von git verfolgte Dateien taugen als Beleg (Item 6; sonst einmal `beleg_untracked`).
    Hop-Belege: geaendert -> Digest nachgefuehrt + Pruefpunkt `hop_beleg_geaendert` (Final-Fix I6, kein Judge),
    verschwunden -> Pruefliste. Aussagen: gemerkte Belege (`belegspeicher`) geaendert/verschwunden = Signal.
 6. Je Aussage mit Signal das Urteil aus dem Cache (`stand.urteil_holen`) oder `judge.zweimal_einig` (Judge nur
    hier importiert; `--ohne-llm` laedt ihn nie). Urteile kommen sofort in den Cache, ein spaeter scheiternder
    Lauf fragt nie zweimal. `veraltet` + aufloesbarer Beleg -> `aussage_ersetzen` (von unten nach oben) + Ledger.
 7. `rendern` (ohne Zeiten, Ruling 45a); `hand_edit` einmal je Block und Inhalts-Hash.
 8. Commit nur, wenn Aussagen umgeschrieben wurden, `Graph.digest_stabil()` sich geaendert hat oder der Text von
    KETTEN.md (Ruling 45b). Schreiben/Torwaechter/Commit/Ruecksetzen im gemeinsamen `schreibvorgang`.
 9. Rowboat-Notizen (Task 17, `rowboat.notizen_schreiben`) und Hook `nach_tick(ergebnis)`; lokalen Stand speichern
    (Ruling 44).

Task 17: Urteile aus Rowboat (`<rowboat_wissen>/<rowboat_urteile>`, nur gelesen) werden direkt nach Schritt 1
angewandt -- vor dem Messen und nur, wenn die Doku keine fremden Aenderungen hat (sonst warten sie). Ein Revert macht
die Render-Hashes der dabei geaenderten Bloecke ungueltig (kein falscher Hand-Edit, der Block wird neu gerendert).
Notizen schreibt jeder Lauf, der an der Fremd-Pruefung vorbei kam; `keine_aenderung` nur, wenn sich Pruefliste oder
Urteile geaendert haben oder noch keine Notizen da sind; nie bei `laeuft_schon`, `kein_anstoss`, `kein_seed` oder
fremden Aenderungen.

Ruling 49 (Task 15b): Basis ist der volle lokale Graph (`stand.vollgraph_laden`), fuer Hops, Aussagen, Compose und
Dienste ueberlagert vom schlanken Repo-Graph (`graph.mit_schlank`; dort liegen die gemerkten Aussage-Belege
vollstaendig). Fehlt der lokale Graph, wird er aus `docs/graph/graph.json` + Neubau aller Spaces rekonstruiert; ein
Judge-Aufruf entsteht dabei nur fuer Aussagen, deren Belege sich gegenueber dem schlanken Graph wirklich geaendert
haben. Geschrieben wird `schlank(neu)`, die Commit-Entscheidung vergleicht `schlank(neu).digest_stabil()` mit
`schlank(alt).digest_stabil()` (nur Symbol-Kanten anders -> kein Commit). Der volle Graph wird nach jedem Neubau
lokal geschrieben, nie committet.

Fix-Runde 1 zu 15b: der lokale Vollgraph gilt nur, wenn seine Kennung (`<HEAD aussen>:<HEAD Submodul>:<Lauf>`) zur
`vollgraph_kennung` im Stand passt; ein Lauf ohne Neubau traegt die Kennung weiter. Schreibfehler -> alte Datei weg,
Kennung `None` (naechster Lauf rekonstruiert).

Final-Fix-Welle (Ruling 57): vor allem anderen `kopf_pruefen` (nie auf detached HEAD, waehrend Rebase/Merge/
Cherry-Pick/Revert/Bisect oder auf einem anderen als dem konfigurierten `zweig`; Marker bleibt) und
`halbschreibung_aufraeumen` (eigene abgebrochene Schreibung zuruecknehmen). Nach `tick_deadline_min` keine neuen
Judge-Fragen. git-Fehler in den Mengenbildnern -> `git_fehler:<befehl>`; jede andere unerwartete Ausnahme ->
Pruefpunkt `tick_fehler`, Commit-Zeiger bleibt, die Ausnahme geht weiter. Rowboat-Notizen nur mit
`rowboat_notizen_aktiv`.

Ruling 50 (Task 15c): gemessen wird nur im Messbaum (`messbaum.bereitstellen`, committeter Stand: aussen HEAD, je
initialisiertem Submodul dessen HEAD) -- Manifeste, Extraktoren, SQL, Compose, Register, Beleg-Digests,
`belege_auffrischen`, Aussage-Verankerung, Quelltext fuer den Judge. Doku-Text, CAS-Basis, Schreiben, Commit und alle
Git-Abfragen laufen im echten Repo. Messbaum-Fehler -> `messbaum:<grund>` (kein LLM, kein Commit). Belege in nicht
initialisierten Submodulen sind `nicht_messbar` (kein Signal). `unsauber_gebaut` (15b) entfaellt; Ruling 48a bleibt
als zweite Linie.

Stufe 4, Task 5 (D8): nach der Aussagen-Pruefung liest `_ereignisse` die Space-Events aus den Quellen im Messbaum
(Ruling T5-a), legt sie als `ereignis`-Knoten in den Graph und rendert je zugeordneter Space-Doku einen Block
`ereignisse <space>`; `_rendern` behandelt KETTEN.md und alle anderen Doku-Dateien mit demselben Hand-Edit-Schutz
(Render-Hashes anderer Dateien unter `<datei>#<block>`). Ein Commit an einer Ereignis-Quelle stoesst wie ein Doku-Commit
einen Lauf ohne Neubau an. Kein Judge: Events tragen keine gemerkten Aussage-Belege.

Stufe 4, Task 6 (D9): vor den Events erkennt `_neue_spaces` im Messbaum Space-Ordner ohne Doku (`neue_spaces`) und legt
je einen `docs/spaces/<NNN>_<name>.md` nur aus markierten Bloecken an, dazu eine Zeile im Block `neue-spaces` der
`docs/spaces/README.md` (sonst nie Schreibziel). Bereits generierte Dokus (Block `steckbrief <space>`) werden wie jede
Doku neu gerendert; fehlt ihr Ordner, bleiben sie stehen (`space_doku_ohne_space`). Ein Commit unter einem Space-Ordner
ohne Doku oder mit generierter Doku stoesst einen Lauf ohne Neubau an. Neue Dateien: CAS = "gibt es noch nicht".
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import os
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import anker, bauen, ketten_seed, korpus, messbaum, render_ereignisse, render_ketten, rowboat, rowboat_seiten, umschreiben
from . import erstbefuellung as erst_mod
from . import neue_spaces as neue_spaces_mod
from . import aussagen as au
from . import waechter
from .anker import Beleg
from .belegspeicher import aussage_entfernen, aussage_speichern, beleg_dateien, gespeicherte_belege, knoten_id
from .extraktoren import ereignisse as ereignisse_ex
from .extraktoren import sql as sql_ex
from .graph import Graph, graph_diff, mit_schlank, schlank
from .konfig import Konfig, SpaceKonfig
from .namen import space_schluessel
from .pruefliste import (
    beleg_nicht_messbar, beleg_untracked, neue_offene, pruefliste_ergaenzen, pruefliste_laden, pruefliste_pfad,
)
from .schreibvorgang import (
    GRAPH_DIR, GRAPH_JSON, LEDGER, NEUE_SPACES_README, SUBMODUL, Aenderung, Ausgang, doku_dateien, doku_ziel,
    eigene_datei, fremde_aenderungen, fremde_doku_ziele, git, graph_text, halbschreibung_aufraeumen, hat_submodul, kopf,
    kopf_pruefen, ledger_commit_von, muster_trifft, neue_spaces_aktiv, nur_lesend, nur_lesend_dateien, ohne_gross_klein,
    pfad_schluessel, schreiben_pruefen_committen, submodul_kopf, unsaubere_dateien, versionierte_dateien,
)
from .seed_befehl import RenderBericht, SchlankBericht, SeedErgebnis, graph_schlank, neu_rendern, seed, seed_deckung
from .stand import (
    SPERRE_DATEI, Stand, anstoss_noetig, fremd_commits_laden, fremd_commits_merken, lokaler_ordner, marker_loeschen,
    schreibvermerk_pfad, sperre, sperre_frisch, sperre_zustand, stand_laden, stand_pfad, stand_schreiben,
    temp_aufraeumen,
    urteil_holen, urteil_merken, urteile_kuerzen, vollgraph_kennung, vollgraph_laden, vollgraph_pfad,
    vollgraph_schreiben, vollgraph_verwerfen,
)
from .typen import Pruefpunkt

if TYPE_CHECKING:
    from ._manifest_gate2 import Manifest

    from .extraktoren.sql import SqlDefinition

__all__ = [
    "GRAPH_DIR", "GRAPH_JSON", "LEDGER", "SPERRE_DATEI", "RenderBericht", "SchlankBericht", "SeedErgebnis", "Stand",
    "TickErgebnis", "TickFehler", "anstoss_noetig", "betroffene_spaces", "geaenderte_dateien", "graph_schlank",
    "ledger_commit_von", "marker_loeschen", "neu_rendern", "pruefliste_laden", "pruefliste_pfad", "seed",
    "seed_deckung", "sperre", "sperre_zustand",
    "stand_laden", "stand_pfad", "tick", "vollgraph_pfad",
]

KONFIG_DATEI = "config/doku_loop.yaml"
REGEL_DATEI = "selbstbild.yaml"  # Stufe 4: Abschnitt `doku:` (D5); eine Aenderung baut wie die alte Konfig alles neu
# Ruling T4-c: Endungen von Code-Dateien, deren ganze-Datei-Belege in Doku-Zielen kein Judge-Signal tragen
CODE_ENDUNGEN = frozenset({".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".rs", ".go", ".ps1", ".psm1",
                           ".sh", ".bash", ".cmd", ".bat", ".sql"})
DATEI_GEAENDERT_IDS = 20  # so viele Aussage-Ids nennt ein Pruefpunkt `beleg_datei_geaendert`
NACHBARN_MAX = 12
QUELLTEXT_MERKEN = 7000  # Quelltext offener Signale im lokalen Stand (der Judge kuerzt ohnehin auf 6000)
AUSSCHNITT_ZEILEN = 40  # Minor 15: `datei`-Belege zeigt der Judge nur +-40 Zeilen um die angezeigte Zeile
# Judge-Gruende, bei denen spaeter erneut gefragt wird (technisch, nicht inhaltlich); alle anderen kommen in den Cache
# `deadline` (I7a): die Tick-Deadline war erreicht, gefragt wurde nicht -- das Signal bleibt offen
WIEDERHOLEN = ("budget:", "timeout", "exit:", "nicht_gefunden", "oserror:", "deadline")

Melden = Callable[[str], None]
ManifestBauer = Callable[[SpaceKonfig, Konfig, str], "Manifest"]


OFFENE_BAUPUNKTE = ("waechter_veraltet",)  # Bau-Pruefpunkte, die in `offen` statt in `struktur` gehoeren
KOPF_GRUENDE = ("kopf_detached", "rebase_laeuft", "merge_laeuft", "falscher_zweig")


def _im_paket(dateiname: str, paket: Path) -> bool:
    try:
        return Path(dateiname).resolve().is_relative_to(paket)
    except (OSError, ValueError):
        return False


def _fehler_ort(fehler: BaseException) -> str:
    """`datei.py:zeile funktion` des innersten Rahmens im Paket selbstbild.doku (sonst des innersten ueberhaupt)."""
    rahmen = traceback.extract_tb(fehler.__traceback__)
    paket = Path(__file__).resolve().parent
    eigene = [r for r in rahmen if _im_paket(r.filename, paket)]
    r = (eigene or rahmen or [None])[-1]
    return f"{Path(r.filename).name}:{r.lineno} {r.name}" if r is not None else "?"


class TickFehler(RuntimeError):
    """Der Tick kann nicht laufen (z. B. kein Git-HEAD)."""


@dataclass(frozen=True)
class TickErgebnis:
    lauf: str
    spaces: list[str]
    signale: int
    llm_aufrufe: int
    umgeschrieben: int
    pruefliste: int
    commit: str | None
    uebersprungen: str | None
    graph_digest: str = ""  # nur Trockenlauf: `digest_stabil` des schlanken Graphen (Task 11 vergleicht mit graph.json)
    messbaum_commit: str = ""  # nur Trockenlauf: aeusserer Commit, auf dem tatsaechlich gemessen wurde
    hinweis: str = ""  # nur Trockenlauf: z. B. `messbaum_hinter_head_nur_eigene_dateien` (T3-c)
    # Task 4: Zaehler der Aussagen-Pruefung (auch im Trockenlauf)
    doku_gelesen: int = 0  # Doku-Dateien (KETTEN.md, weitere Doku, Doku-Ziele), deren Aussagen geprueft wurden
    nur_lesend_gelesen: int = 0  # nur-lesend-Dateien (D7), gelesen im Messbaum
    aussagen: int = 0  # gelesene Aussagen
    aussagen_belegt: int = 0  # davon mit gemerkten Belegen nach dem Lauf
    aussagen_neu: int = 0  # Belege erstmals gemerkt (Quelle `extraktor`, ohne LLM)
    doku_uebersprungen: list[str] = dataclasses.field(default_factory=list)  # Ruling T4-b: fremd geaenderte Ziele
    ereignisse: int = 0  # Task 5: Space-Events (Space, Name) aus den Quellen im Messbaum
    neue_spaces: list[str] = dataclasses.field(default_factory=list)  # Task 6: Space-Ordner ohne Doku (Messbaum)
    neue_dateien: list[str] = dataclasses.field(default_factory=list)  # Task 6: angelegte (trocken: geplante) Dokus
    # Task 7 (D10): Erstbefuellung -- faellig (aktiv, nicht fertig, im Fenster), offene Aussagen danach, in diesem Lauf
    # geprueft (auch aus dem Cache) und LLM-Aufrufe unter dem Agenten der Erstbefuellung (in `llm_aufrufe` enthalten)
    erst_faellig: bool = False
    erst_offen: int = 0
    erst_geprueft: int = 0
    erst_aufrufe: int = 0
    # Task 8 (D11): Rowboat-Seiten Mitbauen/Zustand -- geschriebene (trocken: voraussichtlich zu schreibende) und von
    # Hand geaenderte (bleiben stehen), Pfade ab dem Notizordner
    seiten_geaendert: list[str] = dataclasses.field(default_factory=list)
    seiten_von_hand: list[str] = dataclasses.field(default_factory=list)
    index_geaendert: bool = False  # trocken: Index.md wuerde neu geschrieben


# ---------- Aenderungen ----------

def _diff(repo: Path, alt: str | None) -> list[str] | None:
    if not alt:
        return None
    aus = git(repo, "diff", "--name-only", "--no-renames", "-z", alt, "HEAD")
    return None if aus is None else [p for p in aus.split("\0") if p]


def geaenderte_dateien(repo: Path, stand: Stand | None,
                       koepfe: Mapping[str, str] | None = None) -> tuple[list[str], list[str]]:
    """(aussen, Submodule) seit dem gespeicherten Stand; Submodul-Pfade mit ihrem Repo-Praefix.

    Ohne gespeicherten (oder nicht mehr aufloesbaren) Commit -> `["*"]` = alle Spaces. Verschachtelte Submodule
    (alle ausser `vibemind-os`, 15c Fix-Runde 1): je geaendertem HEAD `git diff` in seinem echten Checkout; ist der
    alte Commit unbekannt oder nicht erreichbar -> `<rel>/*` (alle Spaces darunter bzw. darum gelten als geaendert).
    `koepfe`: aktuelle HEADs je Submodul (sonst hier ermittelt)."""
    repo = Path(repo)
    if stand is None:
        return ["*"], (["*"] if hat_submodul(repo) else [])
    aussen = _diff(repo, stand.letzter_commit_aussen)
    sub: list[str] = []
    if hat_submodul(repo):
        roh = _diff(repo / SUBMODUL, stand.letzter_commit_sub)
        sub = ["*"] if roh is None else [f"{SUBMODUL}/{p}" for p in roh]
    if koepfe is None:
        koepfe = _koepfe(repo)
    for rel, kopf_jetzt in sorted(koepfe.items()):
        if rel == SUBMODUL or stand.submodul_commits.get(rel) == kopf_jetzt:
            continue
        roh = _diff(repo / rel, stand.submodul_commits.get(rel))
        sub.extend([f"{rel}/*"] if roh is None else [f"{rel}/{p}" for p in roh])
    return (["*"] if aussen is None else aussen), sub


def _koepfe(repo: Path) -> dict[str, str]:
    try:
        return messbaum.submodul_koepfe(repo)
    except messbaum.MessbaumFehler:
        return {}


ANSTOSS_GIT_TIMEOUT_S = 20


def anstoss_koepfe(repo: Path) -> dict[str, str] | None:
    """D6: aeusserer HEAD (`""`) und der echte HEAD des Checkouts je initialisiertem Submodul (nicht der Gitlink-Pin;
    die Gitlinks des committeten Baums dienen nur dazu, die Submodule zu finden). Dieselben Namen und Werte wie
    `Messbaum.commits` und `Stand.submodul_commits`. Ein `git rev-parse` plus `ls-tree`/`rev-parse` je Submodul, jeder
    Aufruf mit Timeout. Scheitert Git -> `None`: dann entscheiden Marker und Intervall allein."""
    try:
        aussen = messbaum._git_text(Path(repo), "rev-parse", "--verify", "-q", "HEAD^{commit}",
                                    timeout=ANSTOSS_GIT_TIMEOUT_S)
        if not aussen:
            return None
        return {"": aussen, **messbaum.submodul_koepfe(Path(repo), ANSTOSS_GIT_TIMEOUT_S)}
    except (messbaum.MessbaumFehler, OSError, ValueError):
        return None


def _unter(datei: str, wurzeln: Sequence[str]) -> bool:
    return any(datei == w or datei.startswith(w.rstrip("/") + "/") for w in wurzeln if w)


def _unter_sql_wurzel(datei: str, wurzeln: Sequence[str]) -> bool:
    return any(w.strip("/") in ("", ".") or _unter(datei, [w.strip("/")]) for w in wurzeln)


def signal_traegt(beleg: Beleg, doc: str, konfig: Konfig) -> bool:
    """Ruling T4-c: traegt eine Aenderung dieses gemerkten Belegs einer Aussage in `doc` ein Judge-Signal?

    Nein nur fuer ganze-Datei-Belege (`datei`) auf Code-Dateien (`CODE_ENDUNGEN`, bei `.sql` nur `datei`, nicht
    `sql`) in den neuen Doku-Zielen und nur-lesend-Dateien; KETTEN.md und alle anderen Belegarten bleiben wie
    bisher. Auch fuer die Erstbefuellung (Task 7)."""
    if beleg.art != "datei" or doc == konfig.ketten_md:
        return True
    if not (doku_ziel(konfig, doc) or nur_lesend(konfig, doc)):
        return True
    return Path(beleg.file).suffix.lower() not in CODE_ENDUNGEN


# Task 5, Fix-Runde 1: Pruefpunkt-Arten, die jeder `_voll`-Lauf vollstaendig neu berechnet; in `struktur` werden sie
# je Lauf ersetzt (auch ohne vollen Bau). Block-Arten nur fuer Events-Bloecke (Blockname `ereignisse <space>`).
EREIGNIS_ARTEN = frozenset({
    "ereignis_quellen_abweichend", "ereignis_nur_doku", "space_ohne_doku", "space_dokument_fehlt",
    "ereignis_quelle_fehlt", "ereignis_quelle_unlesbar", "ereignis_eintrag_unlesbar",
})
EREIGNIS_BLOCK_ARTEN = frozenset({"block_entfernt", "block_ohne_quelle"})


def ereignis_punkt(eintrag: Mapping[str, object]) -> bool:
    """Ein Struktur-Eintrag der Pruefliste, den der naechste `_voll`-Lauf ersetzt (siehe `EREIGNIS_ARTEN`; Task 6:
    auch die Arten aus `neue_spaces.ARTEN`)."""
    art = eintrag.get("art")
    if art in EREIGNIS_ARTEN or art in neue_spaces_mod.ARTEN:
        return True
    return art in EREIGNIS_BLOCK_ARTEN and str(eintrag.get("aussage_id") or "").startswith(
        render_ereignisse.BLOCK_PRAEFIX)


def _doku_schluessel(schluessel: str) -> bool:
    """Task 5: Render-Hash einer anderen Doku als KETTEN.md (`<datei>.md#<block>`); KETTEN-Bloecke heissen ohne Datei."""
    datei, raute, _ = schluessel.partition("#")
    return bool(raute) and datei.lower().endswith(".md")


def ereignis_quelle(konfig: Konfig, datei: str, bekannt: frozenset[str] = frozenset()) -> bool:
    """Task 5: betrifft eine (committete) Aenderung an `datei` eine Ereignis-Quelle? Die YAML-Datei selbst, eine schon
    bekannte Quelldatei (`bekannt`, aus den Belegen der Ereignis-Kanten) oder eine `.py` unter einer `python_dict`-Wurzel,
    die den Tabellennamen enthaelt. Gelesen wird dafuer nur, ob angestossen wird; gemessen wird im Messbaum (T5-a)."""
    for q in konfig.ereignisse:
        if "yaml" in q and datei == q["yaml"].strip("/"):
            return True
        if "python_dict" in q and datei.endswith(".py") and _unter(datei, [q["wurzel"].strip("/")]):
            if datei in bekannt:
                return True
            try:
                if q["python_dict"].encode("utf-8") in (konfig.repo / datei).read_bytes():
                    return True
            except OSError:
                continue
    return False


def betroffene_spaces(konfig: Konfig, aussen: Sequence[str], sub: Sequence[str]) -> list[str]:
    """Spaces, die neu gebaut werden. Alle bei unbekanntem Stand (`*`), geaendertem Dienst-Register oder Loop-Konfig
    (`config/doku_loop.yaml` oder die Regeldatei `selbstbild.yaml`),
    jeder geaenderten `*.sql` unter `sql_wurzeln` und jeder geaenderten `compose_live`-Datei (I5: SQL-Definitionen
    und Compose-Kanten sind global); sonst die Spaces, unter deren Wurzel eine geaenderte Datei liegt."""
    alle = sorted(konfig.spaces)
    if "*" in aussen or "*" in sub:
        return alle
    if any(p in (konfig.dienste_yaml, KONFIG_DATEI, REGEL_DATEI) for p in aussen):
        return alle
    compose = {c.strip("/") for c in konfig.compose_live}
    for p in (*aussen, *sub):
        if p in compose or (p.lower().endswith(".sql") and _unter_sql_wurzel(p, konfig.sql_wurzeln)):
            return alle

    def trifft(p: str, wurzel: str) -> bool:
        if p.endswith("/*"):  # ganzes Submodul unbekannt: Spaces darin und Spaces, in denen es liegt
            praefix = p[:-2]
            return _unter(wurzel, [praefix]) or _unter(praefix, [wurzel])
        return _unter(p, [wurzel])

    return sorted(n for n, s in konfig.spaces.items() if any(trifft(p, s.wurzel) for p in (*aussen, *sub)))


# ---------- offene Signale (lokal gemerkt) ----------

def _aussage_dict(a: au.Aussage) -> dict[str, object]:
    return {"id": a.id, "datei": a.datei, "ueberschrift": a.ueberschrift, "zeile_start": a.zeile_start,
            "zeile_ende": a.zeile_ende, "text": a.text, "belege": [b.to_dict() for b in a.belege]}


def _aussage_aus(d: Mapping[str, Any]) -> au.Aussage:
    return au.Aussage(str(d["id"]), str(d["datei"]), str(d["ueberschrift"]), int(d["zeile_start"]),
                      int(d["zeile_ende"]), str(d["text"]), tuple(Beleg.from_dict(b) for b in d["belege"]))


def _cache_antwort(eintrag: Mapping[str, object], judge: Any) -> tuple[Any, str]:
    if eintrag.get("urteil"):
        def text(k: str) -> str | None:
            v = eintrag.get(k)
            return str(v) if v is not None else None

        return judge.JudgeAntwort(str(eintrag["urteil"]), text("neuer_text"), text("beleg_anker"),
                                  str(eintrag.get("begruendung") or "")), "ok"
    return None, str(eintrag.get("grund") or "unbekannt")


def _cache_eintrag(antwort: Any, grund: str, jetzt: datetime) -> dict[str, object]:
    if antwort is None:
        return {"grund": grund, "zeit": jetzt.isoformat()}
    return {"urteil": antwort.urteil, "neuer_text": antwort.neuer_text, "beleg_anker": antwort.beleg_anker,
            "begruendung": antwort.begruendung, "zeit": jetzt.isoformat()}


# ---------- Lauf ----------

class _Lauf:
    def __init__(self, konfig: Konfig, *, jetzt: datetime, erzwingen: bool, laufzeit: Callable[..., Any],
                 ohne_llm: bool, budget: Any, drift: Callable[[Path], set[umschreiben.DriftFehler]],
                 manifest_bauer: ManifestBauer, nach_tick: Callable[[TickErgebnis], None] | None,
                 melden: Melden, uhr: Callable[[], float] = time.monotonic) -> None:
        self.konfig, self.repo, self.jetzt = konfig, konfig.repo, jetzt
        self.uhr, self.start = uhr, uhr()  # I7a: Tick-Deadline fuer neue Judge-Fragen
        self.deadline_gemeldet = False
        self.erzwingen, self.ohne_llm = erzwingen, ohne_llm
        self.budget, self.drift, self.manifest_bauer = budget, drift, manifest_bauer
        self.nach_tick, self.melden = nach_tick, melden
        self.llm_aufrufe = 0

        def gezaehlt(*a: Any, **kw: Any) -> Any:
            self.llm_aufrufe += 1
            return laufzeit(*a, **kw)

        self.laufzeit = gezaehlt
        self.stand_alt: Stand | None = None
        self.lauf_nr = 1
        self.lauf = ""
        self.spaces: list[str] = []
        self.signale = 0
        self.umgeschrieben = 0
        self.offen: list[Pruefpunkt] = []
        self.struktur: list[Pruefpunkt] = []
        self.offene_signale: list[dict[str, object]] = []
        self.ledger: list[umschreiben.LedgerEintrag] = []
        self.kopf_aussen = ""
        self.kopf_sub = ""
        self.manifeste: dict[str, "Manifest"] = {}
        self.sql_defs: Mapping[str, "SqlDefinition"] = {}
        self.zustand: dict[tuple[str, str, str], str] = {}
        self.alle_wurzeln = [s.wurzel for s in konfig.spaces.values()]
        self.gebaut: list[str] = []
        self.versioniert: frozenset[str] = frozenset()
        self.unsauber: frozenset[str] = frozenset()  # Ruling 48a: Dateien mit nicht committeten Aenderungen
        self.gefragt: dict[tuple[str, str, str, str], str] = {}  # (aussage_id, Beleg) -> Grund, je Lauf nur einmal
        self.doku_roh: dict[str, bytes] = {}  # CAS-Basis der Doku, gelesen direkt nach der Fremd-Pruefung
        self.vollgraph_kennung: str | None = None  # Kennung des lokalen Vollgraphs, die in den Stand kommt
        # Ruling 50: gemessen wird im Messbaum (vor der Bereitstellung wird nichts gemessen)
        self.mess_konfig: Konfig = konfig
        self.mess_repo: Path = konfig.repo
        self.nicht_messbar: tuple[str, ...] = ()
        self.sub_koepfe: dict[str, str] = {}  # HEAD je initialisiertem Submodul (echte Checkouts, Schritt 2)
        self.rowboat_geaendert = False  # Task 17: Urteile angewandt oder neue Pruefpunkte aus den Urteilen
        self.doku_fremd = False  # Task 17: fremde Aenderungen an der Doku gesehen -> keine Urteile, keine Notizen
        self.render_noetig = False  # Task 17 M2: ein Revert hat Bloecke geaendert -> im selben Lauf neu rendern
        self.ausstehend: dict[str, str] | None = None  # Ruling 55a: None = unveraendert aus dem alten Stand
        self.seiten_geaendert: list[str] = []  # Task 8: Rowboat-Seiten (trocken: geplant)
        self.seiten_von_hand: list[str] = []
        self.index_geaendert = False
        self.trocken = False  # Trockenlauf: misst wie ein Tick, schreibt nichts (`trocken_laufen`)
        self.koepfe: dict[str, str] | None = None  # D6: beim Anstoss gelesene Koepfe, im Tick wiederverwendet
        self.gesehen: dict[str, str] | None = None  # T3-b: Koepfe, die dieser Lauf gesehen hat (kommen in den Stand)
        self.messbaum_hinweis = ""
        # Task 4
        self.doku_fremd_ziele: list[str] = []  # Ruling T4-b: Ziele mit fremden Aenderungen, in diesem Lauf ausgelassen
        self.doku_gelesen = 0
        self.nur_lesend_gelesen = 0
        self.n_aussagen = 0
        self.n_belegt = 0
        self.n_neu = 0
        self.index_holer: Callable[[], Mapping[str, list[str]]] | None = None  # Datei-Index, einer je Lauf
        # Ruling T4-c: Code-Datei -> (alter Digest, neuer Digest, Aussage-Ids) geaenderter Datei-Belege ohne Judge
        self.datei_geaendert: dict[str, tuple[str, str, list[str]]] = {}
        # Task 5: Events-Bloecke je Doku-Datei (Blockname -> Inhalt) und Zahl der Events
        self.ereignis_bloecke: dict[str, dict[str, str]] = {}
        self.n_ereignisse = 0
        self.evs: list[ereignisse_ex.Ereignis] = []
        self.ev_praefixe: set[str] = set()
        self.ev_pp: list[Pruefpunkt] = []
        # Task 6: neue Space-Dokus
        self.neue_spaces: list[str] = []
        self.neu_angelegt: list[str] = []  # in diesem Lauf angelegte (trocken: geplante) Doku-Dateien
        self.generiert_bloecke: dict[str, dict[str, str]] = {}  # generierte Doku -> Blockname -> Inhalt (LF)
        self.readme_roh: bytes | None = None  # CAS-Basis von docs/spaces/README.md (None = fehlt)
        self.readme: render_ketten.RenderErgebnis | None = None  # gerenderter Block `neue-spaces`
        self.readme_text = ""  # dekodierte CAS-Basis der README (nur gesetzt, wenn `readme` gesetzt ist)
        self.gitlinks: list[str] = []  # Ruling T6-a: Gitlinks des Messbaums (initialisiert und nicht)
        # Task 7 (D10): Erstbefuellung
        self.erst_faellig = False  # aktiv, nicht fertig, Ortszeit im Fenster (fuer das Ergebnis)
        self.erst_fragen = False  # faellig und weder Trockenlauf noch --ohne-llm: der Judge wird gefragt
        self.erst_sammeln = False  # aktiv und nicht fertig: Aussagen fuer die Erstpruefung sammeln (auch trocken)
        self.erst_pool: list[tuple[au.Aussage, tuple[Beleg, ...], bool, bool]] = []  # (Aussage, Belege, fragbar, nur_l.)
        self.erst_zurueck = False  # Aussagen dieses Laufs nicht vollstaendig gesehen -> nicht `fertig`
        self.erst_umgeschrieben_docs: set[str] = set()  # Doku mit Ersetzung aus einem echten Signal (Zeilen verschoben)
        self.erst_markiert: set[str] = set()  # geprueft ohne Schreiben (oder per Signal-Urteil): sofort in den Stand
        self.erst_ersetzt: set[str] = set()  # Ids alt/neu ersetzter Aussagen: erst nach dem Commit in den Stand
        self.erst_fertig = False
        self.erst_offen = 0
        self.erst_geprueft = 0
        self.erst_aufrufe = 0
        self.erst_umgeschrieben = 0
        self.erst_fehlversuche: dict[str, int] | None = None  # Fix-Runde 1: neuer Zaehler (None = unveraendert)

    # ----- Ergebnis und Stand -----

    def ergebnis(self, *, commit: str | None = None, uebersprungen: str | None = None,
                 pruefliste: int = 0, graph_digest: str = "", messbaum_commit: str = "",
                 hinweis: str = "") -> TickErgebnis:
        return TickErgebnis(self.lauf, list(self.spaces), self.signale, self.llm_aufrufe, self.umgeschrieben,
                            pruefliste, commit, uebersprungen, graph_digest, messbaum_commit, hinweis,
                            doku_gelesen=self.doku_gelesen, nur_lesend_gelesen=self.nur_lesend_gelesen,
                            aussagen=self.n_aussagen, aussagen_belegt=self.n_belegt, aussagen_neu=self.n_neu,
                            doku_uebersprungen=list(self.doku_fremd_ziele), ereignisse=self.n_ereignisse,
                            neue_spaces=list(self.neue_spaces), neue_dateien=list(self.neu_angelegt),
                            erst_faellig=self.erst_faellig, erst_offen=self.erst_offen,
                            erst_geprueft=self.erst_geprueft, erst_aufrufe=self.erst_aufrufe,
                            seiten_geaendert=list(self.seiten_geaendert), seiten_von_hand=list(self.seiten_von_hand),
                            index_geaendert=self.index_geaendert)

    def stand_speichern(self, *, fortschreiben: bool, render_hashes: Mapping[str, str] | None = None,
                        hand_edit: Mapping[str, str] | None = None,
                        offene: Sequence[Mapping[str, object]] | None = None, committet: bool = False) -> None:
        """`committet` (Task 7): die Schreibung dieses Laufs ist committet -- erst dann zaehlen ersetzte Aussagen
        (alte und neue Id) als erstgeprueft; ohne Commit bleiben sie offen (das Urteil steht im Cache)."""
        alt = self.stand_alt or Stand()
        erst = set(alt.erstgeprueft)
        if self.erst_sammeln:  # nur waehrend der Erstbefuellung (sonst waechst die Liste mit jedem Signal)
            erst |= self.erst_markiert | (self.erst_ersetzt if committet else set())
        fertig = alt.erstbefuellung_fertig or (self.erst_fertig and (committet or not self.erst_ersetzt))
        fehl = self.erst_fehlversuche if self.erst_fehlversuche is not None else alt.erst_fehlversuche
        stand_schreiben(self.konfig, Stand(
            letzter_commit_aussen=self.kopf_aussen if fortschreiben else alt.letzter_commit_aussen,
            letzter_commit_sub=(self.kopf_sub or None) if fortschreiben else alt.letzter_commit_sub,
            letzter_tick=self.jetzt.isoformat(),
            render_hashes=dict(render_hashes if render_hashes is not None else alt.render_hashes),
            lauf_nr=self.lauf_nr,
            hand_edit_gemeldet=dict(hand_edit if hand_edit is not None else alt.hand_edit_gemeldet),
            offene_signale=[dict(s) for s in (offene if offene is not None else alt.offene_signale)],
            vollgraph_kennung=self.vollgraph_kennung,
            submodul_commits=dict(self.sub_koepfe if fortschreiben else alt.submodul_commits),
            koepfe_gesehen=dict(self.gesehen if self.gesehen is not None else alt.koepfe_gesehen),
            gemessen_ausstehend=dict(self.ausstehend if self.ausstehend is not None else alt.gemessen_ausstehend),
            erstgeprueft=sorted(erst), erstbefuellung_fertig=fertig, erst_fehlversuche=dict(fehl),
        ))

    def _abschluss(self, ergebnis: TickErgebnis) -> TickErgebnis:
        if not self.konfig.rowboat_notizen_aktiv:  # I4: aus, bis der Wissens-Spiegel deployt ist
            self.melden("Notizen: aus (rowboat_notizen_aktiv: false)")
        elif self._notizen_faellig(ergebnis):
            try:
                self._notizen()
            except Exception as fehler:  # noqa: BLE001 - Rowboat darf den Tick nicht umwerfen
                self.melden(f"Notizen: {type(fehler).__name__}")
        if self.nach_tick is not None:
            try:
                self.nach_tick(ergebnis)
            except Exception as fehler:  # noqa: BLE001 - Rowboat darf den Tick nicht umwerfen
                self.melden(f"nach_tick: {type(fehler).__name__}")
        return ergebnis

    # ----- Rowboat (Task 17) -----

    def _urteile(self) -> None:
        """Urteile anwenden; ein unerwarteter Fehler wird Pruefpunkt `urteile_fehler`, der Tick laeuft weiter (M8)."""
        try:
            self._urteile_anwenden()
        except Exception as fehler:  # noqa: BLE001 - Urteile duerfen den Tick nicht umwerfen
            text = f"{type(fehler).__name__}: {fehler}"[:300]
            self.melden(f"Urteile: Fehler ({type(fehler).__name__}) – Prüfliste, Lauf geht weiter")
            try:
                neu = pruefliste_ergaenzen(self.konfig, [Pruefpunkt("urteile_fehler", text, self.konfig.rowboat_urteile,
                                                                    None, None)], [], voll=False, lauf=self.lauf)
                self.rowboat_geaendert = self.rowboat_geaendert or neu > 0
            except OSError:
                pass

    def _urteile_anwenden(self) -> None:
        konfig = self.konfig
        if fremde_aenderungen(konfig):
            self.doku_fremd = True
            self.melden("Urteile: fremde Änderungen an der Doku – warten")
            return
        unlesbar: list[Pruefpunkt] = []
        try:
            pfad = rowboat.urteile_pfad(konfig.rowboat_wissen, konfig.rowboat_urteile)
        except ValueError:
            self.melden("Urteile: Pfad ungültig (außerhalb des Wissensordners oder unter Kanal/) – nicht gelesen")
            return
        urteile = rowboat.urteile_lesen(pfad, unlesbar=unlesbar, datei=konfig.rowboat_urteile)
        erledigt = rowboat.erledigt_pfad(konfig)
        offen = rowboat.offene_urteile(urteile, erledigt)
        # Ruling T4-b (Fix-Runde 1): nur Urteile zu einem fremd geaenderten Ziel warten (nie fremde Arbeit mitcommitten)
        fremd_ziele = fremde_doku_ziele(konfig) if offen else []
        pps = list(unlesbar)
        if offen:
            md = self.repo / konfig.ketten_md
            vorher = md.read_bytes() if md.is_file() else b""
            self.rowboat_geaendert = True
            try:
                pps += rowboat.urteile_anwenden(offen, self.repo, self.repo / LEDGER, erledigt, lauf=self.lauf,
                                                jetzt=self.jetzt, drift=self.drift, melden=self.melden,
                                                doku=doku_dateien(konfig),
                                                offene_punkte=pruefliste_laden(konfig)["offen"], zweig=konfig.zweig,
                                                vermerk=schreibvermerk_pfad(konfig),
                                                fremd_commits=fremd_commits_laden(konfig), fremd=fremd_ziele)
            finally:  # Fix-Runde 2, Item 4: ein committeter Revert verwirft die Hashes auch, wenn danach etwas wirft
                nachher = md.read_bytes() if md.is_file() else b""
                if nachher != vorher:
                    self._render_hashes_verwerfen(vorher.decode("utf-8"), nachher.decode("utf-8"))
        neu = pruefliste_ergaenzen(konfig, pps, [], voll=False, lauf=self.lauf) if pps else 0
        self.rowboat_geaendert = self.rowboat_geaendert or neu > 0
        self.melden(f"Urteile: {len(urteile)} gelesen, {len(offen)} neu, {len(unlesbar)} unlesbar, Prüfliste +{neu}")

    def _render_hashes_verwerfen(self, vorher: str, nachher: str) -> None:
        """Bloecke, die ein Urteil (Revert) geaendert hat, gelten nicht als Hand-Edit: Hash vergessen und im selben
        Lauf neu rendern (M2)."""
        if self.stand_alt is None:
            return
        alt, neu = render_ketten.bloecke_finden(vorher), render_ketten.bloecke_finden(nachher)
        for block in [b for b in self.stand_alt.render_hashes if b in neu]:
            if block not in alt or alt[block].inhalt != neu[block].inhalt:
                del self.stand_alt.render_hashes[block]
                self.stand_alt.hand_edit_gemeldet.pop(block, None)
                self.render_noetig = True

    def _notizen_faellig(self, ergebnis: TickErgebnis) -> bool:
        u = ergebnis.uebersprungen
        if u is None:
            return True
        if self.doku_fremd or u in ("kein_seed", "kein_anstoss", "laeuft_schon", *KOPF_GRUENDE):
            return False
        if u.startswith("git_fehler"):
            return False
        if u.startswith("fremde_aenderungen"):
            return False
        if u == "keine_aenderung":
            index = self.konfig.rowboat_wissen / self.konfig.rowboat_notizen / f"{rowboat.INDEX_NAME}.md"
            return self.rowboat_geaendert or not index.is_file()
        return True

    def _letzte_laeufe(self) -> list[str]:
        """Lauf-Kennung des letzten gespeicherten Laufs (aus Zeit + Nummer des Stands) -- fuer die Uebernahme."""
        alt = self.stand_alt
        if alt is None or not alt.letzter_tick:
            return []
        try:
            return [f"{datetime.fromisoformat(alt.letzter_tick):%Y-%m-%dT%H:%M:%S}#{alt.lauf_nr}"]
        except ValueError:
            return []

    def _notizen(self) -> None:
        konfig = self.konfig
        repo_graph = Graph.laden(self.repo / GRAPH_JSON)
        voll = vollgraph_laden(konfig, self.vollgraph_kennung)
        graph = repo_graph if voll is None else mit_schlank(voll.graph, repo_graph)
        for _ in range(2):  # neue `notiz_von_hand`-Punkte erscheinen sofort in der Pruefliste
            von_hand = self._notizen_aufruf(graph, trocken=False)[1]
            if not von_hand or not pruefliste_ergaenzen(konfig, von_hand, [], voll=False, lauf=self.lauf):
                break

    def _notizen_aufruf(self, graph: Graph, *, trocken: bool) -> tuple[list[Path], list[Pruefpunkt]]:
        """Ein Aufruf von `rowboat.notizen_schreiben` samt Seiten (Task 8, D11). Live und Trockenlauf gehen durch
        genau diesen Weg (gleiche Eingaben, gleiche Filter); `trocken` schreibt nichts. Rueckgabe: (geschriebene
        bzw. zu schreibende Dateien, Pruefpunkte: `notiz_von_hand`, `seite_fehler`, ...)."""
        konfig = self.konfig
        ledger = rowboat.ledger_lesen(self.repo / LEDGER)
        spitzen = rowboat.datei_spitzen(self.repo, {e.datei for e in ledger})
        commits = {**fremd_commits_laden(konfig), **rowboat.lauf_commits(self.repo)}  # I11: auch fremd uebernommene
        vermerke = rowboat.vermerke_laden(rowboat.erledigt_pfad(konfig))
        bekannt = set(commits) | {e.lauf for e in ledger} | set(self._letzte_laeufe())
        liste = pruefliste_laden(konfig)
        punkte: list[Pruefpunkt] = []
        ausgefallen: list[str] = []
        # Mitbauen und Zustand, deterministisch; Zustand zaehlt dieselben offenen Punkte wie die Pruefliste
        zusatz = rowboat_seiten.seiten(
            graph, self.repo, {"offen": rowboat.offene_ohne_erledigte(liste["offen"], vermerke)},
            ordner=konfig.rowboat_notizen, fehler=punkte, ausgefallen=ausgefallen)
        geschrieben = rowboat.notizen_schreiben(
            graph, liste["offen"], ledger, konfig.rowboat_wissen, konfig.rowboat_notizen, self.lauf,
            manifest_pfad=rowboat.manifest_pfad(konfig), commits=commits, spitzen=spitzen, vermerke=vermerke,
            struktur=liste["struktur"], urteile_datei=konfig.rowboat_urteile, bekannte_laeufe=bekannt,
            melden=self.melden, pruefpunkte=punkte, zusatz=zusatz, zusatz_ausgefallen=ausgefallen,
            zusatz_links=rowboat_seiten.index_links(konfig.rowboat_notizen), trocken=trocken)
        return geschrieben, punkte

    def _seiten_trocken(self, graph: Graph) -> None:
        """Task 8: was ein echter Lauf an Rowboat-Seiten und `Index.md` schreiben wuerde (Entscheidung wie live,
        aber ohne zu schreiben: weder Notizen noch Manifest noch Pruefliste)."""
        konfig = self.konfig
        if not konfig.rowboat_notizen_aktiv:
            return
        try:
            geschrieben, punkte = self._notizen_aufruf(graph, trocken=True)
        except Exception as fehler:  # noqa: BLE001 - der Trockenlauf meldet, statt zu fallen
            self.melden(f"Trockenlauf: Rowboat-Seiten nicht geprüft ({type(fehler).__name__})")
            return
        basis = (konfig.rowboat_wissen / konfig.rowboat_notizen).resolve()
        rels = {p.resolve().relative_to(basis).as_posix() for p in geschrieben}
        seiten = (rowboat_seiten.MITBAUEN_REL, rowboat_seiten.ZUSTAND_REL)
        self.seiten_geaendert = [r for r in seiten if r in rels]
        self.index_geaendert = f"{rowboat.INDEX_NAME}.md" in rels
        pfade = {f"{konfig.rowboat_notizen}/{r}" for r in seiten}
        self.seiten_von_hand = [str(p.datei).removeprefix(f"{konfig.rowboat_notizen}/") for p in punkte
                                if p.art == "notiz_von_hand" and p.datei in pfade]
        self.melden(f"Trockenlauf: Rowboat-Seiten würden geschrieben: {', '.join(self.seiten_geaendert) or 'keine'}; "
                    f"Index.md {'ändert sich' if self.index_geaendert else 'bleibt'}"
                    + (f"; von Hand geändert (bleiben): {', '.join(self.seiten_von_hand)}" if self.seiten_von_hand
                       else ""))

    # ----- Hilfen fuer Belege -----

    def _submodul_von(self, datei: str) -> str | None:
        """Innerstes Submodul (repo-relativ), in dem `datei` liegt; None = aeusseres Repo."""
        treffer = [rel for rel in self.sub_koepfe if datei == rel or datei.startswith(rel + "/")]
        if not treffer and self.kopf_sub and datei.startswith(SUBMODUL + "/"):
            return SUBMODUL
        return max(treffer, key=len) if treffer else None

    def commit_fuer(self, datei: str) -> str:
        """Commit des Repos, in dem `datei` liegt (15c Fix-Runde 1: auch verschachtelte Submodule)."""
        rel = self._submodul_von(datei)
        if rel is None:
            return self.kopf_aussen
        return self.sub_koepfe.get(rel) or (self.kopf_sub if rel == SUBMODUL else self.kopf_aussen)

    def _space_von(self, datei: str) -> str | None:
        return ketten_seed._space_von(datei, self.konfig)

    def pruefbar(self, b: Beleg) -> bool:
        """Symbol-Belege in nicht gebauten Spaces haben keinen Stand; ihre Dateien sind unveraendert."""
        return not (b.art == "symbol" and _unter(b.file, self.alle_wurzeln) and not _unter(b.file, self.gebaut))

    def mergebar(self, b: Beleg) -> bool:
        """Frisch verankerte Belege taugen nur ausserhalb der Spaces oder in gebauten Spaces (sonst Dateibeleg
        mangels Manifest), nur in versionierten, sauberen Dateien und nie in den eigenen Dateien des Loops."""
        return (b.file in self.versioniert and b.file not in self.unsauber and not eigene_datei(self.konfig, b.file)
                and (not _unter(b.file, self.alle_wurzeln) or _unter(b.file, self.gebaut)))

    def urteil(self, b: Beleg) -> str:
        """beleg_ok | beleg_geaendert | beleg_verschwunden | unsauber.

        Eigene Dateien des Loops (Ruling 48b), nicht versionierte (Item 6) und nicht pruefbare Belege -> ok;
        Dateien mit nicht committeten Aenderungen -> `unsauber` (Ruling 48a: kein Signal, nichts nachfuehren).
        Ein unsicherer Pfad (`..`, absolut; Final-Fix 22) wird nie gelesen: Pruefpunkt `beleg_pfad_unsicher`, ok."""
        if not anker.pfad_sicher(b.file):
            self._pfad_unsicher(b)
            return "beleg_ok"
        if eigene_datei(self.konfig, b.file):
            return "beleg_ok"
        if self.nicht_messbar_hinweis(b.file):  # Ruling 50: nicht initialisiertes Submodul -> kein Signal
            return "beleg_ok"
        if b.file not in self.versioniert:  # Item 6: untracked/ignorierte Dateien geben nie ein Signal
            self.offen.append(beleg_untracked(b.file))
            return "beleg_ok"
        if b.file in self.unsauber:
            try:
                stand = anker.datei_digest(self.repo / b.file)[:12]
            except OSError:
                stand = "fehlt"
            self.offen.append(Pruefpunkt("beleg_unsauber", f"{b.file} hat nicht committete Aenderungen (Stand "
                                                           f"{stand}); kein Signal bis zum Commit", b.file, None, None))
            return "unsauber"
        if not self.pruefbar(b):
            return "beleg_ok"
        try:
            return anker.beleg_pruefen(b, self.zustand, self.mess_repo, space_wurzeln=self.alle_wurzeln)
        except anker.UnsichererPfad:  # z. B. ein Link im Repo, der nach aussen zeigt
            self._pfad_unsicher(b)
            return "beleg_ok"

    def _pfad_unsicher(self, b: Beleg) -> None:
        self.offen.append(Pruefpunkt("beleg_pfad_unsicher", f"{b.file}: Belegpfad zeigt aus dem Repo heraus "
                                                            "(`..`, absolut oder Link); nicht messbar, kein Signal",
                                     b.file, None, None))

    def nicht_messbar_hinweis(self, datei: str) -> bool:
        sub = next((s for s in self.nicht_messbar if _unter(datei, [s])), None)
        if sub is None:
            return False
        self.offen.append(beleg_nicht_messbar(datei, sub))
        return True

    def digest_jetzt(self, b: Beleg) -> str | None:
        if b.art in ("symbol", "sql"):
            d = self.zustand.get((b.art, b.file, b.anker))
            if d is None and b.art == "symbol" and not _unter(b.file, self.alle_wurzeln):
                d = anker.symbol_digest_aus_datei(self.mess_repo, b.file, b.anker)
            return d
        if b.art == "yaml_key":
            return anker.yaml_knoten_digest(self.mess_repo / b.file, b.anker)
        try:
            return anker.datei_digest(self.mess_repo / b.file)
        except OSError:
            return None

    def nachgefuehrt(self, b: Beleg) -> Beleg | None:
        d = self.digest_jetzt(b)
        return None if d is None else dataclasses.replace(b, digest=d, commit=self.commit_fuer(b.file))

    # ----- Quelltext fuer den Judge -----

    def _datei_text(self, datei: str, commit: str | None) -> str | None:
        if not anker.pfad_sicher(datei):  # Final-Fix 22
            return None
        if commit is None:
            try:
                return (self.mess_repo / datei).read_bytes().decode("utf-8", errors="replace")
            except OSError:
                return None
        sub = self._submodul_von(datei)
        if sub is not None:
            wo, rel = self.repo / sub, datei[len(sub) + 1:]
        else:
            wo, rel = self.repo, datei
        return git(wo, "show", f"{commit}:{rel}") if commit else None

    def _module(self, datei: str) -> list[str]:
        module = [anker.modulpfad(datei)]
        space = self._space_von(datei)
        if space is not None:
            wurzel = self.konfig.spaces[space].wurzel
            module.insert(0, ".".join(Path(datei[len(wurzel) + 1:]).with_suffix("").parts))
        return module

    def _symbol_text(self, text: str, b: Beleg) -> str | None:
        with tempfile.TemporaryDirectory(prefix="doku-loop-") as tmp:
            pfad = Path(tmp) / "quelle.py"
            pfad.write_text(text, encoding="utf-8")
            for modul in self._module(b.file):
                treffer = [d for d in anker.ast_definitionen(pfad, modul) or [] if d.document_id == b.anker]
                if len(treffer) == 1:
                    return treffer[0].source
        return None

    def quelltext(self, b: Beleg, *, alt: bool) -> str:
        text = self._datei_text(b.file, b.commit if alt else None)
        if text is None:
            return "(alter Stand nicht lesbar)" if alt else "(Datei fehlt)"
        if b.art == "symbol":
            return self._symbol_text(text, b) or (text if alt else "(Symbol fehlt)")
        if b.art == "yaml_key":  # Minor 15: nur der Knoten, nicht die ganze Datei
            gefunden, wert = anker.yaml_knoten_wert_text(text, Path(b.file).suffix, b.anker)
            if not gefunden:
                return "(alter Knoten nicht lesbar)" if alt else "(Knoten fehlt)"
            return f"Knoten {b.anker} in {b.file}:\n{anker.knoten_text(wert)}"
        if b.art == "datei" and b.zeile is not None:  # Minor 15: +-40 Zeilen um die angezeigte Zeile
            zeilen = text.splitlines(keepends=True)
            von, bis = max(1, b.zeile - AUSSCHNITT_ZEILEN), min(len(zeilen), b.zeile + AUSSCHNITT_ZEILEN)
            return f"Zeilen {von}-{bis} von {b.file}:\n" + "".join(zeilen[von - 1:bis])
        return text

    def kandidat_an_zeile(self, b: Beleg) -> Beleg | None:
        """Symbol, das heute die alte Zeile umschliesst (Umbenennung): Kandidat fuer den neuen Anker."""
        if b.art != "symbol" or b.zeile is None:
            return None
        space = self._space_von(b.file)
        if space is not None:
            manifest = self.manifeste.get(space)
            if manifest is None:
                return None
            return anker.beleg_aus_zeile(b.file, b.zeile, manifest, self.commit_fuer(b.file), repo=self.mess_repo)
        defs = anker.ast_definitionen(self.mess_repo / b.file, anker.modulpfad(b.file)) or []
        um = [d for d in defs if d.start_line <= b.zeile <= d.end_line]
        if not um:
            return None
        d = min(um, key=lambda d: d.end_line - d.start_line)
        if sum(1 for x in defs if x.document_id == d.document_id) > 1:
            return None
        return Beleg("symbol", b.file, d.document_id, anker._sha(d.source), self.commit_fuer(b.file), d.start_line)

    def nachbarn(self, graph: Graph, b: Beleg) -> tuple[str, ...]:
        space = self._space_von(b.file)
        ids = {f"modul:{b.file}"} | ({f"symbol:{space}:{b.anker}"} if space else set())
        aus = [f"{e.art}: {e.von} -> {e.nach}" for e in sorted(graph.kanten.values(), key=lambda e: e.id)
               if e.art != "belegt_durch" and (e.von in ids or e.nach in ids)]
        return tuple(aus[:NACHBARN_MAX])

    # ----- Ablauf -----

    def laufen(self) -> TickErgebnis:
        """`_laufen` mit Fehlerbehandlung (Final-Fix C2): eine unerwartete Ausnahme wird Pruefpunkt `tick_fehler`
        (Typ und Ort, nie der Text: er kann Dateiinhalte zitieren), der Stand wird mit unveraendertem Commit-Zeiger
        gespeichert (der verbrauchte Marker ist kein Fortschritt; der naechste Lauf kommt nach `tick_minuten`) und
        die Ausnahme geht weiter (Exit 1)."""
        try:
            return self._laufen()
        except umschreiben.GitFehler as fehler:  # I3: fail-closed, ohne Pruefpunkt (keine Flut bei Dauerfehler)
            befehl = getattr(fehler, "befehl", None) or "git"
            self.melden(f"Git: {befehl} gescheitert – übersprungen, nichts geschrieben (git_fehler)")
            try:
                self.stand_speichern(fortschreiben=False)
            except OSError:
                pass
            return self._abschluss(self.ergebnis(uebersprungen=f"git_fehler:{befehl}"))
        except Exception as fehler:
            ort = _fehler_ort(fehler)
            self.melden(f"Fehler: {type(fehler).__name__} in {ort} – Prüfliste, Commit-Zeiger bleibt")
            try:
                pruefliste_ergaenzen(self.konfig, [Pruefpunkt("tick_fehler", f"{type(fehler).__name__} in {ort}",
                                                              None, None, None)], [], voll=False, lauf=self.lauf)
            except OSError:
                pass
            try:
                self.stand_speichern(fortschreiben=False)
            except OSError:
                pass
            raise

    def _laufen(self) -> TickErgebnis:
        konfig = self.konfig
        weg = temp_aufraeumen(konfig)
        if weg:
            self.melden(f"Aufräumen: {len(weg)} alte Temp-Datei(en) des Loops entfernt")
        gekuerzt = urteile_kuerzen(konfig)  # Minor 13
        if gekuerzt:
            self.melden(f"Urteils-Cache: {gekuerzt} alte Zeile(n) verworfen")
        self.stand_alt = stand_laden(konfig)
        self.lauf_nr = (self.stand_alt.lauf_nr if self.stand_alt else 0) + 1
        self.lauf = f"{self.jetzt:%Y-%m-%dT%H:%M:%S}#{self.lauf_nr}"
        self.vollgraph_kennung = self.stand_alt.vollgraph_kennung if self.stand_alt else None
        self._erst_einstellen()
        self.erst_fragen = self.erst_faellig and not self.ohne_llm and not self._erst_budget_erschoepft()

        # 1. Anstoss
        try:
            marker = konfig.marker.exists()
        except OSError:
            marker = False
        if not self.erzwingen and not self._anstoss():
            self.melden(f"Anstoß: kein Marker, letzter Lauf jünger als {konfig.tick_minuten} min – übersprungen "
                        "(kein_anstoss)")
            return self.ergebnis(uebersprungen="kein_anstoss")
        # C1 (Final-Fix): nie auf detached HEAD, in eine laufende Git-Operation oder auf einen fremden Branch; der
        # Marker bleibt liegen, der Stand unveraendert -> der naechste Lauf versucht es wieder
        kopf_grund = kopf_pruefen(self.repo, konfig.zweig)
        if kopf_grund is not None:
            self.melden(f"Kopf: {kopf_grund} – übersprungen, nichts angefasst")
            return self._abschluss(self.ergebnis(uebersprungen=kopf_grund))
        marker_loeschen(konfig)
        self._halbschreibung()  # I7b: eigene abgebrochene Schreibung vor allem anderen zuruecknehmen
        grund = ("erzwungen" if self.erzwingen else "Marker" if marker else "erster Lauf" if self.stand_alt is None
                 else f"älter als {konfig.tick_minuten} min")
        self.melden(f"Anstoß: {grund}; Marker gelöscht")
        if not (self.repo / GRAPH_JSON).is_file():
            self.stand_speichern(fortschreiben=False)
            self.melden("Graph: kein docs/graph/graph.json – übersprungen (kein_seed; erst `seed`)")
            return self._abschluss(self.ergebnis(uebersprungen="kein_seed"))

        # 1b. Urteile aus Rowboat (Task 17): vor dem Messen, nur ohne fremde Aenderungen an der Doku
        self._urteile()

        # 2. Was ist zu tun?
        if self.rowboat_geaendert or self.render_noetig:  # ein Urteil kann committet haben: Koepfe neu lesen
            self.koepfe = None
        if self.koepfe is not None:  # beim Anstoss schon gelesen (D6): nicht noch einmal
            kopf_aussen = self.koepfe[""]
            self.kopf_aussen, self.kopf_sub = kopf_aussen, self.koepfe.get(SUBMODUL, "")
            self.sub_koepfe = {rel: k for rel, k in self.koepfe.items() if rel}
        else:
            kopf_aussen = kopf(self.repo)
            if kopf_aussen is None:
                raise TickFehler("kein Git-HEAD im Repo")
            self.kopf_aussen, self.kopf_sub = kopf_aussen, submodul_kopf(self.repo) or ""
            self.sub_koepfe = _koepfe(self.repo)
        self.gesehen = {"": self.kopf_aussen, **self.sub_koepfe}
        aussen, sub = geaenderte_dateien(self.repo, self.stand_alt, self.sub_koepfe)
        basis_roh = (self.repo / GRAPH_JSON).read_bytes()  # Item 1: dieselben Bytes sind die CAS-Basis beim Schreiben
        repo_graph = Graph.aus_text(basis_roh.decode("utf-8"))
        # Ruling 49: Symbol-Daten lokal, Hops/Aussagen aus dem Repo-Graph; nur ein zum Stand passender Vollgraph
        voll = vollgraph_laden(konfig, self.vollgraph_kennung)
        self.vollgraph_kennung = voll.kennung if voll is not None else None
        basis = repo_graph if voll is None else mit_schlank(voll.graph, repo_graph)
        if voll is None:
            self.melden("Graph: kein passender lokaler Vollgraph – Rekonstruktion aus docs/graph/graph.json, "
                        "alle Spaces")
        if self.erzwingen or self.stand_alt is None or voll is None:
            spaces = sorted(konfig.spaces)
        else:
            spaces = betroffene_spaces(konfig, aussen, sub)
        geaendert = [p for p in (*aussen, *sub) if p != "*"]
        belegdateien = {p for p in geaendert if not _unter(p, self.alle_wurzeln) and not eigene_datei(konfig, p)} \
            & beleg_dateien(basis)
        # Task 4: ein Commit an einem Doku-Ziel oder einer nur-lesend-Datei -> Aussagen pruefen (ohne Neubau)
        doku_geaendert = [p for p in geaendert if doku_ziel(konfig, p) or nur_lesend(konfig, p)]
        # Task 5: ein Commit an einer Ereignis-Quelle -> Events neu lesen und Bloecke rendern (ohne Neubau)
        quellen_bekannt = frozenset(b.file for e in basis.kanten.values() if e.art == "ereignis_von" for b in e.belege)
        ereignis_geaendert = [p for p in geaendert if ereignis_quelle(konfig, p, quellen_bekannt)]
        doku_geaendert += [p for p in ereignis_geaendert if p not in doku_geaendert]
        # Task 6: ein Commit unter einem Space-Ordner ohne Doku oder mit generierter Doku -> Lauf ohne Neubau
        space_geaendert = self._space_anstoss(geaendert)
        doku_geaendert += [p for p in space_geaendert if p not in doku_geaendert]
        offene = list(self.stand_alt.offene_signale) if self.stand_alt else []
        self.spaces = spaces

        def zahl(liste: Sequence[str]) -> str:
            return "alle" if "*" in liste else str(len(liste))

        self.melden(f"Änderungen: {zahl(aussen)} Dateien außen, {zahl(sub)} im Submodul -> Spaces: "
                    f"{', '.join(spaces) or 'keine'}; Beleg-Dateien außerhalb: {len(belegdateien)}; "
                    f"Doku geändert: {len(doku_geaendert)} (Ereignis-Quellen {len(ereignis_geaendert)}, "
                    f"Space-Ordner {len(space_geaendert)}); "
                    f"offene Signale: {len(offene)}")
        if self.erst_fragen:
            self.melden("Erstbefüllung: fällig (im Fenster, nicht fertig) – Lauf auch ohne Änderung")
        if not spaces and not belegdateien and not offene and not self.render_noetig and not doku_geaendert \
                and not self.erst_fragen:
            self.stand_speichern(fortschreiben=True)
            self.melden(f"Stand: Lauf {self.lauf_nr} fortgeschrieben – übersprungen (keine_aenderung)")
            return self._abschluss(self.ergebnis(uebersprungen="keine_aenderung"))

        # 3. Vorpruefungen vor jedem Judge-Aufruf
        fremd = fremde_aenderungen(konfig)
        if fremd:
            self.stand_speichern(fortschreiben=False)
            self.melden(f"Fremde Änderungen: {len(fremd)} Datei(en) – übersprungen, nichts angefasst")
            return self._abschluss(self.ergebnis(uebersprungen="fremde_aenderungen:" + ",".join(fremd)))
        self.melden("Fremde Änderungen: keine")
        self._fremde_ziele(melden_pruefliste=True)
        self._doku_roh_lesen()
        try:
            vorher = self.drift(self.repo)
        except umschreiben.TorwaechterFehler as fehler:
            self.stand_speichern(fortschreiben=False)
            pruefliste_ergaenzen(konfig, [Pruefpunkt("torwaechter_fehler", f"Waechter vorher: {fehler}"[:500],
                                                     konfig.ketten_md, None, self.lauf)], [], voll=False,
                                 lauf=self.lauf)
            self.melden("Torwächter: Fehler vor dem Lauf – übersprungen")
            return self._abschluss(self.ergebnis(uebersprungen="torwaechter:torwaechter_fehler"))

        # Ruling 50: Messbaum (committeter Stand) vor jedem Messen und vor jedem Judge-Aufruf
        grund_mb = self._messbaum()
        if grund_mb is not None:
            self.stand_speichern(fortschreiben=False)
            self.melden(f"Messbaum: {grund_mb} – übersprungen, kein Commit")
            return self._abschluss(self.ergebnis(uebersprungen=f"messbaum:{grund_mb}"))

        # Ruling 45c: nur offene Signale -> ohne Neubau erneut fragen; neu gebaut wird nur nach einem Urteil
        # (Task 17 M2: nach einem Revert wird ohne Neubau gerendert, `_voll` mit leerer Space-Liste)
        # Task 7: die echten (offenen) Signale zuerst; die Erstbefuellung laeuft danach in `_voll` mit der Rest-Deadline
        if not spaces and not belegdateien and not self.render_noetig and not doku_geaendert:
            beantwortet, rest = self._offene_fragen(offene)
            self.signale = len(offene)
            if not beantwortet and not self.erst_fragen:
                self.stand_speichern(fortschreiben=True, offene=rest)
                self.melden(f"Offene Signale: {len(offene)}, kein neues Urteil – kein Neubau, kein Commit")
                return self._abschluss(self.ergebnis(uebersprungen="offene_signale_ohne_urteil"))
            for s in beantwortet:
                space = s.get("space")
                if isinstance(space, str) and space in konfig.spaces:
                    spaces.append(space)
                else:
                    belegdateien.add(str(s.get("datei")))
            self.spaces = spaces = sorted(set(spaces))
            if beantwortet:
                self.melden(f"Offene Signale: {len(beantwortet)} beantwortet -> Spaces: {', '.join(spaces) or 'keine'}")
                uebrig = rest  # Item 4: in diesem Lauf schon gefragt -> bleiben offen, `_voll` fragt sie nicht erneut
            else:  # Task 7: nur wegen der Erstbefuellung weiter -- wie unten: was `_voll` neu findet, nicht doppelt
                uebrig = [s for s in rest if isinstance(s.get("space"), str) and s["space"] not in spaces]
        else:
            uebrig = [s for s in offene if isinstance(s.get("space"), str) and s["space"] not in spaces]
        return self._voll(basis, basis_roh, schlank(repo_graph), vorher, uebrig)

    def _space_anstoss(self, geaendert: Sequence[str]) -> list[str]:
        """Task 6: geaenderte Dateien unter einem Space-Ordner (`ort` der Space-Regeln), der keine Doku hat oder eine
        generierte (Block `steckbrief`). Regeln und Doku werden dafuer im Arbeitsbaum gelesen -- nur fuer die Frage, ob
        angestossen wird; erkannt und gemessen wird im Messbaum."""
        konfig = self.konfig
        if not geaendert or not neue_spaces_aktiv(konfig):
            return []
        orte = neue_spaces_mod.space_orte(self.repo)
        ic = ohne_gross_klein(self.repo)
        je_ordner: dict[str, list[str]] = {}
        for p in geaendert:
            teile = p.split("/")
            for ort in orte:
                n = len(ort.split("/"))
                # T6-a: auch der Ordner selbst (ein Gitlink kommt dazu oder bekommt einen neuen Pin)
                if len(teile) >= n and muster_trifft(ort, "/".join(teile[:n]), ignorecase=ic):
                    je_ordner.setdefault(teile[n - 1], []).append(p)
        je_ordner = {n: ps for n, ps in je_ordner.items() if not n.startswith("_") and n != "__pycache__"}
        if not je_ordner:
            return []
        zuordnung = ereignisse_ex.space_dokumente_zuordnen(list(je_ordner), konfig.space_dokumente,
                                                           doku_dateien(konfig))
        treffer: list[str] = []
        for name, pfade in sorted(je_ordner.items()):
            doc = zuordnung.get(space_schluessel(name))
            if doc is not None:
                try:
                    text = (self.repo / doc).read_bytes().decode("utf-8", errors="replace")
                except OSError:
                    text = ""
                if neue_spaces_mod.generierter_space(text) is None:
                    continue  # handgeschriebene Doku: Sache von Task 4/5
            treffer.extend(pfade)
        return treffer

    def _anstoss(self) -> bool:
        """Marker und Intervall zuerst (ohne Git); erst wenn beide nicht anstossen, werden die Koepfe gelesen (D6)."""
        konfig = self.konfig
        if anstoss_noetig(konfig, self.stand_alt, self.jetzt):
            return True
        self.koepfe = anstoss_koepfe(self.repo)
        if self.koepfe is not None:
            self.gesehen = dict(self.koepfe)
        return anstoss_noetig(konfig, self.stand_alt, self.jetzt, koepfe=self.koepfe)

    def trocken_laufen(self) -> TickErgebnis:
        """Trockenlauf mit Fehlerbehandlung: eine unerwartete Ausnahme wird `tick_fehler:<Typ>` (nie der Text, er kann
        Dateiinhalte zitieren), geschrieben wird dabei nichts."""
        try:
            return self._trocken_laufen()
        except Exception as fehler:  # noqa: BLE001 - Ergebnis statt Traceback
            self.melden(f"Trockenlauf: Fehler {type(fehler).__name__} in {_fehler_ort(fehler)}")
            return self.ergebnis(uebersprungen=f"tick_fehler:{type(fehler).__name__}")

    def _trocken_laufen(self) -> TickErgebnis:
        """Trockenlauf (Ruling T3-a/T3-b): misst alle Spaces im VORHANDENEN Messbaum und zaehlt die Signale, schreibt
        aber nichts. Weder Repo, Stand, Pruefliste, Urteils-Cache, Vollgraph, Marker, Sperre, Notizen noch Budget; der
        Messbaum wird nur gelesen (`messbaum.vorhanden_lesend`, nie `bereitstellen`), der Judge nie importiert.
        Der Anstoss wird nicht gefragt."""
        konfig = self.konfig
        self.trocken = self.ohne_llm = True
        self.stand_alt = stand_laden(konfig)
        self.lauf_nr = (self.stand_alt.lauf_nr if self.stand_alt else 0) + 1
        self.lauf = f"{self.jetzt:%Y-%m-%dT%H:%M:%S}#trocken"
        self._erst_einstellen()  # Task 7: nur zaehlen und melden, nie fragen (`erst_fragen` bleibt falsch)
        if sperre_frisch(lokaler_ordner(konfig)):
            self.melden("Trockenlauf: tick.lock ist frisch – der alte Loop läuft (alter_loop_laeuft)")
            return self.ergebnis(uebersprungen="alter_loop_laeuft")
        if not (self.repo / GRAPH_JSON).is_file():
            self.melden("Trockenlauf: kein docs/graph/graph.json (kein_seed)")
            return self.ergebnis(uebersprungen="kein_seed")
        fremd = fremde_aenderungen(konfig)  # lesend wie im echten Tick
        if fremd:
            self.melden(f"Trockenlauf: fremde Änderungen an {len(fremd)} Datei(en) (fremde_aenderungen)")
            return self.ergebnis(uebersprungen="fremde_aenderungen:" + ",".join(fremd))
        mb = messbaum.vorhanden_lesend(konfig)
        if mb is None:
            self.melden("Trockenlauf: kein Messbaum auf dem aktuellen Stand (messbaum_veraltet) – nichts angefasst")
            return self.ergebnis(uebersprungen="messbaum_veraltet")
        self.mess_konfig, self.mess_repo, self.nicht_messbar = mb.konfig, mb.wurzel, mb.nicht_messbar
        self.kopf_aussen = mb.commits[""]
        self.messbaum_hinweis = mb.hinweis
        self.kopf_sub = submodul_kopf(self.repo) or ""
        self.sub_koepfe = {rel: k for rel, k in mb.commits.items() if rel}
        self.spaces = sorted(konfig.spaces)
        basis_roh = (self.repo / GRAPH_JSON).read_bytes()
        repo_graph = Graph.aus_text(basis_roh.decode("utf-8"))
        voll = vollgraph_laden(konfig, self.stand_alt.vollgraph_kennung if self.stand_alt else None)
        basis = repo_graph if voll is None else mit_schlank(voll.graph, repo_graph)
        self._fremde_ziele(melden_pruefliste=False)  # lesend: nur gezaehlt und im Ergebnis genannt
        self._doku_roh_lesen()
        try:
            return self._voll(basis, basis_roh, schlank(repo_graph), set(), [])
        except umschreiben.GitFehler as fehler:
            befehl = getattr(fehler, "befehl", None) or "git"
            return self.ergebnis(uebersprungen=f"git_fehler:{befehl}")

    def _halbschreibung(self) -> None:
        h = halbschreibung_aufraeumen(self.repo, schreibvermerk_pfad(self.konfig), self.melden)
        if h.zustand == "zurueckgesetzt":
            pruefliste_ergaenzen(self.konfig, [Pruefpunkt(
                "eigene_halbschreibung_zurueckgesetzt", f"Lauf {h.lauf or '?'} brach zwischen Schreiben und Commit ab; "
                f"{', '.join(h.dateien)} mit CAS auf HEAD zurückgesetzt", None, None, h.lauf)], [], voll=False,
                lauf=self.lauf)
        elif h.zustand == "fremd":
            self.melden(f"Halbschreibung: {len(h.dateien)} Datei(en) weichen ab – gilt als fremde Änderung")

    def _fremde_ziele(self, *, melden_pruefliste: bool) -> None:
        """Ruling T4-b: Doku-Ziele mit fremden (nicht committeten) Aenderungen werden in diesem Lauf ausgelassen --
        keine Aussagen-Pruefung, kein Ersetzen, ihre gemerkten Aussagen bleiben unveraendert im Graph. Pruefpunkt
        `doku_ziel_fremd` einmal je Datei und Inhalt (die Pruefliste dedupliziert nach Art/Datei/Id/Text)."""
        self.doku_fremd_ziele = sorted(fremde_doku_ziele(self.konfig))
        if not self.doku_fremd_ziele:
            return
        self.melden(f"Doku-Ziele: {len(self.doku_fremd_ziele)} mit fremden Änderungen – in diesem Lauf ausgelassen")
        if not melden_pruefliste:
            return
        for datei in self.doku_fremd_ziele:
            try:
                stand = hashlib.sha256((self.repo / datei).read_bytes()).hexdigest()[:12]
            except OSError:
                stand = "fehlt"
            self.offen.append(Pruefpunkt(
                "doku_ziel_fremd", f"{datei} hat nicht committete fremde Änderungen (Inhalt {stand}); vom Loop "
                "ausgelassen, gemerkte Aussagen bleiben, bis die Änderung committet ist", datei, None,
                f"{datei}#{stand}"))

    def _fremd(self, doc: str) -> bool:
        """Ruling T4-b/Item 4: `doc` ist ein fremd geaendertes Ziel (ohne Gross-/Kleinschreibung, wo sie egal ist)."""
        return pfad_schluessel(self.repo, doc) in {pfad_schluessel(self.repo, d) for d in self.doku_fremd_ziele}

    def _doku_roh_lesen(self) -> None:
        """CAS-Basis der Doku jetzt festhalten (nicht erst nach dem Judge); fremd geaenderte Ziele nie."""
        for doc in doku_dateien(self.konfig):
            if self._fremd(doc):
                continue
            try:
                self.doku_roh[doc] = (self.repo / doc).read_bytes()
            except OSError:
                pass
        if neue_spaces_aktiv(self.konfig):  # Task 6: CAS-Basis des Index (nur Block `neue-spaces`)
            try:
                self.readme_roh = (self.repo / NEUE_SPACES_README).read_bytes()
            except OSError:
                self.readme_roh = None

    def _deadline(self) -> bool:
        """I7a: nach `tick_deadline_min` Laufzeit keine neuen Judge-Fragen mehr (die Signale bleiben offen)."""
        if self.uhr() - self.start < self.konfig.tick_deadline_min * 60:
            return False
        if not self.deadline_gemeldet:
            self.deadline_gemeldet = True
            self.melden(f"Deadline: {self.konfig.tick_deadline_min} min erreicht – keine neuen Judge-Fragen, "
                        "Signale bleiben offen")
        return True

    def _messbaum(self) -> str | None:
        """Messbaum bereitstellen; Rueckgabe der Grund, wenn nicht (dann wird nichts gemessen)."""
        try:
            mb = messbaum.bereitstellen(self.konfig)
        except messbaum.MessbaumFehler as fehler:
            return fehler.grund
        except OSError as fehler:
            return type(fehler).__name__
        # gemessen wird genau, was Schritt 2 gesehen hat (aussen + jedes Submodul); sonst holt es der naechste Lauf nach
        if mb.commits != {"": self.kopf_aussen, **self.sub_koepfe}:
            return "head_bewegt"
        self.mess_konfig, self.mess_repo, self.nicht_messbar = mb.konfig, mb.wurzel, mb.nicht_messbar
        self.melden(f"Messbaum: {len(mb.commits)} Worktree(s) auf HEAD, {len(mb.nicht_messbar)} Submodul(e) nicht "
                    "messbar")
        return None

    def _offene_fragen(self, offene: Sequence[Mapping[str, Any]]
                       ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        """Offene Signale erneut fragen (Cache zuerst). Rueckgabe (beantwortet, offen geblieben)."""
        if self.ohne_llm or not offene:
            return [], list(offene)
        from . import judge  # nur hier und in `_urteilen`: `--ohne-llm` laedt den Judge nie

        beantwortet: list[Mapping[str, Any]] = []
        rest: list[Mapping[str, Any]] = []
        for s in offene:
            try:
                aussage, beleg = _aussage_aus(s["aussage"]), Beleg.from_dict(s["beleg"])
            except (KeyError, TypeError, ValueError):
                continue  # kaputter Eintrag: verwerfen, der naechste Neubau findet das Signal wieder
            schl = (beleg.art, beleg.file, beleg.anker)
            neu_digest = str(s.get("neu_digest") or "")
            if not signal_traegt(beleg, aussage.datei, self.konfig):
                # Ruling T4-c: kein Judge; als erledigt fuehren -> der Neubau ohne Judge fuehrt den Beleg nach und
                # meldet ihn als `beleg_datei_geaendert`
                beantwortet.append(s)
                continue
            treffer = urteil_holen(self.konfig, aussage.id, schl, beleg.digest, neu_digest)
            if treffer is not None:
                beantwortet.append(s)
                continue
            if self._deadline():  # I7a
                rest.append(s)
                continue
            frage = judge.JudgeFrage(aussage, beleg, str(s.get("quelltext_alt", "")), str(s.get("quelltext_neu", "")),
                                     tuple(str(n) for n in s.get("nachbarn") or ()))
            antwort, grund = judge.zweimal_einig(frage, self.konfig, **self._judge_kw())
            self.gefragt[(aussage.id, *schl)] = grund
            if antwort is None and grund.startswith(WIEDERHOLEN):
                rest.append(s)
                continue
            urteil_merken(self.konfig, aussage.id, schl, beleg.digest, neu_digest,
                          _cache_eintrag(antwort, grund, self.jetzt))
            beantwortet.append(s)
        return beantwortet, rest

    def _judge_kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"laufzeit": self.laufzeit}
        if self.budget is not None:
            kw["budget"] = self.budget
        return kw

    def _voll(self, basis: Graph, basis_roh: bytes, alt_schlank: Graph, vorher: set[umschreiben.DriftFehler],
              uebrig: Sequence[Mapping[str, object]]) -> TickErgebnis:
        konfig, repo = self.konfig, self.repo
        self.versioniert = versionierte_dateien(repo)
        self.unsauber = unsaubere_dateien(repo)
        mess = self.mess_konfig  # Ruling 50: alles Gemessene aus dem Messbaum
        self.sql_defs = sql_ex.sql_definitionen(mess)[0]
        self.manifeste = {n: self.manifest_bauer(mess.spaces[n], mess, self.commit_fuer(mess.spaces[n].wurzel))
                          for n in self.spaces}
        if self.spaces:
            graph, bau_pp = bauen.graph_bauen(mess, self.spaces, basis, self.manifeste, sql_defs=self.sql_defs,
                                              commit_aussen=self.kopf_aussen, commit_sub=self.kopf_sub)
            self.offen.extend(p for p in bau_pp if p.art in OFFENE_BAUPUNKTE)  # I8: fuer den Menschen, je Tag
            self.struktur.extend(p for p in bau_pp if p.art not in OFFENE_BAUPUNKTE)
            # Ruling 55a: eine geaenderte Messung gilt erst bei der zweiten gleichen Lesung
            alt_ausstehend = self.stand_alt.gemessen_ausstehend if self.stand_alt else {}
            self.ausstehend = waechter.hysterese(basis, graph, alt_ausstehend)
            if self.ausstehend:
                self.melden(f"Messung: {len(self.ausstehend)} Hop(s) mit geänderter Lesung – gilt erst beim "
                            "nächsten gleichen Wert")
        else:  # Item 7: nur Belegdateien ausserhalb der Spaces geaendert -> pruefen ohne Neubau
            graph = copy.deepcopy(basis)
        diff = graph_diff(basis, graph)
        vor_auffrischen = {k.id: k.belege for k in graph.kanten.values() if k.art == "hop"}
        graph = render_ketten.belege_auffrischen(graph, self.manifeste, mess, self.sql_defs)
        for k in graph.kanten.values():  # Ruling 48a: Zeilen aus nicht committeten Dateien nicht uebernehmen
            if k.id in vor_auffrischen:
                k.belege = tuple(alt if alt.file in self.unsauber else neu
                                 for neu, alt in zip(k.belege, vor_auffrischen[k.id]))
                # Fix-Runde 1 zu 15b: auch der Zustand eines Hops mit Beleg in einer nicht committeten Datei bleibt
                # der committete (sonst rendert und committet KETTEN.md einen Zustand aus dem Arbeitsbaum)
                vorher_hop = basis.kanten.get(k.id)
                if vorher_hop is not None and any(b.file in self.unsauber for b in vorher_hop.belege):
                    k.attribute = copy.deepcopy(vorher_hop.attribute)
        self.melden(f"Graph: {len(self.spaces)} Space(s) gebaut, +{len(diff.neue_kanten)}/-{len(diff.weg_kanten)} "
                    "Kanten")

        # 5. Belege pruefen
        self.gebaut = [konfig.spaces[n].wurzel for n in self.spaces]
        for n in self.spaces:
            self.zustand.update(anker.stand_aus_manifest(self.manifeste[n], self.mess_repo))
        self.zustand.update(anker.stand_aus_sql(self.sql_defs))
        hop_signale = self._hops_pruefen(graph)

        # 6. Aussagen + Judge
        texte: dict[str, tuple[bytes, str]] = {}
        aussagen_signale = 0
        for doc in doku_dateien(konfig):
            if self._fremd(doc):  # Ruling T4-b: gemerkte Aussagen bleiben unveraendert im Graph
                continue
            pfad = repo / doc
            roh = self.doku_roh.get(doc)
            if roh is None:
                continue
            text = roh.decode("utf-8")
            if au.handgepflegt(text):
                self.melden(f"Aussagen: {doc} ist handgepflegt – nicht angefasst")
                continue
            neu, n = self._aussagen(graph, doc, pfad, text)
            aussagen_signale += n
            if neu != text:  # Task 7: Ersetzung aus einem echten Signal -> Zeilen verschoben, keine Erstpruefung jetzt
                self.erst_umgeschrieben_docs.add(doc)
            self.doku_gelesen += 1
            texte[doc] = (roh, neu)
        aussagen_signale += self._nur_lesend_pruefen(graph)
        self._verwaiste_aussagen(graph)
        self._datei_geaendert_melden()
        self._erstbefuellung(graph, texte, uebrig)  # Task 7: nach allen echten Signalen, mit der Rest-Deadline
        self.melden(f"Aussagen: {self.doku_gelesen} Doku-Datei(en), {self.nur_lesend_gelesen} nur lesend, "
                    f"{len(self.doku_fremd_ziele)} ausgelassen; {self.n_aussagen} Aussagen, {self.n_belegt} belegt, "
                    f"{self.n_neu} neu gemerkt")
        self.signale = hop_signale + aussagen_signale
        self.melden(f"Belege: {self.signale} Signal(e) (Hops {hop_signale}, Aussagen {aussagen_signale}); "
                    f"Judge {self.llm_aufrufe} Aufruf(e), {self.umgeschrieben} umgeschrieben")

        # 6b. Space-Events (Task 5, D8): aus dem Messbaum, deterministisch, kein Judge; vorher (Task 6, D9) neue
        # Space-Dokus anlegen und generierte Dokus mit Inhalt versorgen (sie brauchen die Events)
        self._ereignisse_lesen()
        self._neue_spaces(texte)
        self._ereignisse(graph, texte)

        # 7. Rendern
        render_hashes, hand_edit = self._rendern(graph, texte)
        if self.trocken:  # nichts schreiben: weder Vollgraph noch Pruefliste noch Stand noch Commit
            self._seiten_trocken(graph)
            return self.ergebnis(graph_digest=schlank(graph).digest_stabil(), messbaum_commit=self.kopf_aussen,
                                 hinweis=self.messbaum_hinweis)
        # Ruling 49: voller Graph lokal; seine Hops/Aussagen ueberlagert beim naechsten Lauf der Repo-Graph
        kennung = vollgraph_kennung(self.kopf_aussen, self.kopf_sub, self.lauf)
        try:
            vollgraph_schreiben(konfig, graph, self.jetzt.isoformat(), kennung=kennung)
            self.vollgraph_kennung = kennung
        except OSError as fehler:
            self.vollgraph_kennung = None  # passt zu keiner Datei mehr -> naechster Lauf rekonstruiert
            weg = vollgraph_verwerfen(konfig)
            self.melden(f"Vollgraph: lokal nicht geschrieben ({type(fehler).__name__}), alte Datei "
                        f"{'entfernt' if weg else 'nicht entfernbar, gilt als veraltet'} – "
                        "nächster Lauf baut alle neu")

        # 8. Commit nur bei echter Aenderung (Ruling 45b; Ruling 49: nur der schlanke Graph zaehlt)
        offene = [*uebrig, *self.offene_signale]
        voll = set(self.spaces) == set(konfig.spaces)
        # Task 6: neu angelegte Dateien (und ein fehlender Index) mit CAS "gibt es noch nicht"
        ohne_alt = {*self.neu_angelegt, *([NEUE_SPACES_README] if self.readme_roh is None else [])}
        aenderungen = [Aenderung(doc, None if doc in ohne_alt else roh, neu) for doc, (roh, neu) in texte.items()
                       if neu.encode("utf-8") != roh]
        # Ruling 48a: Unterschiede, die nur aus nicht committeten Dateien kommen, zaehlen nicht
        neu_schlank = schlank(graph)
        graph_geaendert = neu_schlank.digest_stabil(ohne_dateien=self.unsauber) != alt_schlank.digest_stabil(
            ohne_dateien=self.unsauber)
        if graph_geaendert:  # Item 1: CAS gegen die Bytes, aus denen die Basis geladen wurde
            aenderungen.append(Aenderung(GRAPH_JSON, basis_roh, graph_text(neu_schlank, self.jetzt.isoformat())))
        if not aenderungen and not self.umgeschrieben:
            neu_offen = pruefliste_ergaenzen(konfig, self.offen, self.struktur, voll=voll, lauf=self.lauf,
                                             ersetzen=ereignis_punkt)
            self.stand_speichern(fortschreiben=True, render_hashes=render_hashes, hand_edit=hand_edit, offene=offene)
            self.melden(f"Schreiben: nichts geändert (stabiler Digest gleich); Prüfliste +{neu_offen}; "
                        f"Stand: Lauf {self.lauf_nr}")
            return self._abschluss(self.ergebnis(pruefliste=neu_offen))

        m = neue_offene(konfig, self.offen)
        was = ', '.join(self.spaces) or ('Erstbefüllung' if self.erst_umgeschrieben else 'Belege')
        betreff = f"docs(loop): {was} – {self.umgeschrieben} Aussagen nachgeführt, {m} in Prüfliste"
        if self.neu_angelegt:
            betreff += f"; neue Space-Doku {', '.join(self.neu_angelegt)}"
        aus: Ausgang = schreiben_pruefen_committen(repo, aenderungen, betreff, self.lauf, vorher=vorher,
                                                   drift=self.drift, ledger=self.ledger, melden=self.melden,
                                                   pp_datei=konfig.ketten_md, zweig=konfig.zweig,
                                                   vermerk=schreibvermerk_pfad(konfig))
        neu_offen = pruefliste_ergaenzen(konfig, [*self.offen, *aus.pruefpunkte], self.struktur, voll=voll,
                                         lauf=self.lauf, ersetzen=ereignis_punkt)
        if aus.fremd_commit is not None:  # I11: lokal vermerken, welcher fremde Commit die Ledger-Eintraege traegt
            try:
                fremd_commits_merken(konfig, self.ledger, aus.fremd_commit)
            except OSError as fehler:
                self.melden(f"Ledger-Vermerk: {type(fehler).__name__}")
        if aus.commit is not None or aus.fremd_commit is not None:
            self.stand_speichern(fortschreiben=True, render_hashes=render_hashes, hand_edit=hand_edit, offene=offene,
                                 committet=True)
        else:
            self.stand_speichern(fortschreiben=aus.fortschreiben, hand_edit=hand_edit, offene=offene)
        self.melden(f"Stand: Lauf {self.lauf_nr}" + (" (Commit-Zeiger nicht fortgeschrieben)"
                                                     if not aus.fortschreiben else ""))
        return self._abschluss(self.ergebnis(commit=aus.commit, uebersprungen=aus.uebersprungen,
                                             pruefliste=neu_offen))

    def _hops_pruefen(self, graph: Graph) -> int:
        signale = 0
        for kante in graph.kanten.values():
            if kante.art != "hop" or not kante.belege:
                continue
            neue: list[Beleg] = []
            for b in kante.belege:
                u = self.urteil(b)  # `unsauber` faellt unten durch: Beleg bleibt wie er ist
                if u == "beleg_geaendert":
                    signale += 1
                    neu = self.nachgefuehrt(b) or b
                    neue.append(neu)
                    self._hop_beleg_geaendert(kante, b, neu)
                    continue
                if u == "beleg_verschwunden":
                    signale += 1
                    hop = f"{kante.attribute.get('kette')}/{kante.attribute.get('nr')}"
                    zeile = kante.attribute.get("zeile")
                    self.offen.append(Pruefpunkt(
                        "hop_beleg_verschwunden", f"{hop}: {b.art} {b.anker or '-'} in {b.file} ist verschwunden",
                        self.konfig.ketten_md, zeile if isinstance(zeile, int) else None, hop))
                neue.append(b)
            kante.belege = tuple(neue)
        return signale

    def _hop_beleg_geaendert(self, kante: Any, alt: Beleg, neu: Beleg) -> None:
        """I6: der Digest eines Hop-Belegs wird nachgefuehrt, aber nicht still: Pruefpunkt mit Kette/Nr, Beleg, altem
        und neuem Digest und der angezeigten Zeile; dedupliziert je Hop + neuem Digest (`aussage_id`). Kein Judge."""
        hop = f"{kante.attribute.get('kette')}/{kante.attribute.get('nr')}"
        zeile = kante.attribute.get("zeile")
        stelle = f"{alt.file}:{alt.zeile}" if alt.zeile is not None else alt.file
        self.offen.append(Pruefpunkt(
            "hop_beleg_geaendert", f"{hop}: {alt.art} {alt.anker or '-'} in {stelle} geändert (Digest "
            f"{alt.digest[:12]} -> {neu.digest[:12]}); Hop-Zeile prüfen", self.konfig.ketten_md,
            zeile if isinstance(zeile, int) else None, f"{hop}#{neu.digest[:12]}"))

    def _stumm_nachfuehren(self, a: au.Aussage, gespeichert: Sequence[Beleg], stumm: Sequence[Beleg]
                           ) -> tuple[Beleg, ...]:
        """Ruling T4-c: Datei-Belege auf Code nachfuehren (neuer Digest, damit sie im naechsten Lauf nicht wieder
        anschlagen) und je Datei die Aussage vormerken; Rueckgabe die gemerkten Belege mit den nachgefuehrten."""
        neu: dict[tuple[str, str, str], Beleg] = {}
        for b in stumm:
            n = self.nachgefuehrt(b) or b
            neu[(b.art, b.file, b.anker)] = n
            eintrag = self.datei_geaendert.setdefault(b.file, (b.digest, n.digest, []))
            if a.id not in eintrag[2]:
                eintrag[2].append(a.id)
        return tuple(neu.get((x.art, x.file, x.anker), x) for x in gespeichert)

    def _datei_geaendert_melden(self) -> None:
        """Ruling T4-c: ein Pruefpunkt `beleg_datei_geaendert` je Code-Datei (nicht je Aussage), ohne LLM;
        dedupliziert ueber `<datei>#<neuer Digest>`."""
        for datei, (alt, neu, aids) in sorted(self.datei_geaendert.items()):
            liste = ", ".join(aids[:DATEI_GEAENDERT_IDS]) + (" …" if len(aids) > DATEI_GEAENDERT_IDS else "")
            self.offen.append(Pruefpunkt(
                "beleg_datei_geaendert", f"{datei} geändert (Digest {alt[:12]} -> {neu[:12]}); {len(aids)} Aussage(n) "
                f"mit Datei-Beleg darauf prüfen (kein Judge): {liste}"[:2000], datei, None, f"{datei}#{neu[:12]}"))
        if self.datei_geaendert:
            self.melden(f"Datei-Belege auf Code: {len(self.datei_geaendert)} Datei(en) geändert – Prüfliste, kein Judge")

    def _verwaiste_aussagen(self, graph: Graph) -> None:
        """Fix-Runde 1, Item 2: gemerkte Aussagen eines geloeschten oder umbenannten Doku-Ziels (bzw. einer nur-lesend-
        Datei) entfernen -- nur wenn die Loeschung committet ist: die Datei trifft ein Muster, ist weder Ziel noch
        nur-lesend-Datei dieses Laufs, nicht versioniert, nicht unsauber (ein fremdes `git rm` ohne Commit ist
        unsauber) und auf der Platte weder im Arbeitsbaum noch im Messbaum."""
        konfig = self.konfig
        schl = functools.partial(pfad_schluessel, self.repo)  # Item 4: ohne Gross-/Kleinschreibung vergleichen
        bekannt = {schl(d) for d in (*doku_dateien(konfig), *nur_lesend_dateien(konfig, self.mess_repo),
                                     *self.doku_fremd_ziele, *self.versioniert, *self.unsauber)}
        weg: list[str] = []
        for k in list(graph.knoten.values()):
            d = k.datei
            if k.typ != "aussage" or not d or not (doku_ziel(konfig, d) or nur_lesend(konfig, d)):
                continue
            if schl(d) in bekannt:
                continue
            if _unter(d, self.nicht_messbar):  # Fix-Runde 2: Submodul nur nicht messbar -> Aussagen/Basis bleiben
                continue
            if (self.repo / d).exists() or (self.mess_repo / d).exists():
                continue
            aussage_entfernen(graph, k.id[len("aussage:"):])
            weg.append(d)
        if weg:
            self.melden(f"Aussagen: {len(weg)} verwaist (Datei committet gelöscht/umbenannt: "
                        f"{', '.join(sorted(set(weg)))}) – entfernt")

    def _nur_lesend_pruefen(self, graph: Graph) -> int:
        """D7/Ruling T4-a: nur-lesend-Dateien aus dem Messbaum (committeter Stand, nie der Arbeitsbaum eines
        Submoduls) pruefen wie Doku, aber nie ersetzen; Rueckgabe die Zahl der Signale."""
        signale = 0
        for rel in nur_lesend_dateien(self.konfig, self.mess_repo):
            try:
                text = (self.mess_repo / rel).read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError) as fehler:
                self.melden(f"Aussagen: {rel} (nur lesend) nicht lesbar ({type(fehler).__name__})")
                self.erst_zurueck = True
                continue
            if au.handgepflegt(text):
                continue
            _, n = self._aussagen(graph, rel, self.mess_repo / rel, text, nur_lesend=True)
            signale += n
            self.nur_lesend_gelesen += 1
        return signale

    def _aussagen(self, graph: Graph, doc: str, pfad: Path, text: str, *,
                  nur_lesend: bool = False) -> tuple[str, int]:
        """Aussagen einer Datei pruefen; Rueckgabe (neuer Text, Zahl der Signale). `nur_lesend` (D7): ein Urteil
        `veraltet` wird Pruefpunkt `nur_lesend_veraltet` mit dem Vorschlag, nie eine Ersetzung."""
        # Ruling 50: Text aus dem echten Arbeitsbaum, verankert wird gegen den Messbaum
        if self.index_holer is None:
            self.index_holer = ketten_seed._index_holer(self.mess_konfig.repo, self.mess_konfig, None)
        liste = au.aussagen_lesen(self.mess_repo / doc, self.mess_konfig, self.manifeste, self.sql_defs,
                                  self.kopf_aussen, text=text, index_holer=self.index_holer)
        signale = 0
        ersetzungen: list[tuple[au.Aussage, str, Beleg, list[Beleg]]] = []
        ids = {a.id for a in liste}
        for a in liste:
            self.struktur.extend(a.pruefpunkte)
            for b in a.belege:
                if eigene_datei(self.konfig, b.file) or self.nicht_messbar_hinweis(b.file):
                    continue
                if b.file not in self.versioniert:
                    self.offen.append(beleg_untracked(b.file))
            aktuell = [dataclasses.replace(b, commit=self.commit_fuer(b.file)) for b in a.belege if self.mergebar(b)]
            gespeichert = gespeicherte_belege(graph, a.id)
            if gespeichert is None:
                aussage_speichern(graph, a, aktuell, "extraktor")
                self.n_neu += 1 if aktuell else 0
                self._erst_merken(a, aktuell, fragbar=True, nur_lesend=nur_lesend)
                continue
            urteile = [(b, self.urteil(b)) for b in gespeichert]
            if any(u == "unsauber" for _, u in urteile):
                self._erst_merken(a, gespeichert, fragbar=False, nur_lesend=nur_lesend)  # Task 7: offen, nicht jetzt
                continue  # Ruling 48a: nicht urteilen, gemerkte Belege bleiben unveraendert
            treffer = [(b, u) for b, u in urteile if u != "beleg_ok"]
            # Ruling T4-c: geaenderte Datei-Belege auf Code in Doku-Zielen sind kein Judge-Signal -- nachfuehren,
            # je Datei sammeln (ein Pruefpunkt nach der Doku-Schleife)
            stumm = [(b, u) for b, u in treffer if u == "beleg_geaendert" and not signal_traegt(b, doc, self.konfig)]
            if stumm:
                treffer = [t for t in treffer if t not in stumm]
                gespeichert = self._stumm_nachfuehren(a, gespeichert, [b for b, _ in stumm])
            if not treffer:
                schluessel = {(b.art, b.file, b.anker) for b in gespeichert}
                dazu = [b for b in aktuell if (b.art, b.file, b.anker) not in schluessel]
                if dazu or stumm:
                    aussage_speichern(graph, a, [*gespeichert, *dazu], "extraktor")
                self._erst_merken(a, [*gespeichert, *dazu], fragbar=True, nur_lesend=nur_lesend)
                continue
            signale += len(treffer)
            self._erst_merken(a, gespeichert, fragbar=False, nur_lesend=nur_lesend)  # Task 7: das Signal geht vor
            if self.ohne_llm:
                b, u = treffer[0]
                self.offen.append(Pruefpunkt("signal_ohne_llm", f"{u}: {b.art} {b.anker or '-'} in {b.file}",
                                             a.datei, a.zeile_start, a.id))
                continue
            ergebnis = self._urteilen(graph, a, gespeichert, treffer, aktuell)
            if ergebnis is not None:
                ersetzungen.append(ergebnis)
        text, getan = self._ersetzen(graph, text, ersetzungen, nur_lesend=nur_lesend)
        for alt_id, neu_id in getan:
            ids.discard(alt_id)
            ids.add(neu_id)
            self.erst_ersetzt.add(neu_id)  # Task 7: ein frisch umgeschriebener Satz gilt nach dem Commit als geprueft
        weg = [k.id[len("aussage:"):] for k in graph.knoten.values()
               if k.typ == "aussage" and k.datei == doc and k.id[len("aussage:"):] not in ids]
        for aid in weg:
            aussage_entfernen(graph, aid)
        self.n_aussagen += len(liste)
        self.n_belegt += sum(1 for aid in ids if knoten_id(aid) in graph.knoten)
        return text, signale

    def _ersetzen(self, graph: Graph, text: str, ersetzungen: Sequence[tuple[au.Aussage, str, Beleg, list[Beleg]]],
                  *, nur_lesend: bool) -> tuple[str, list[tuple[str, str]]]:
        """Der gemeinsame Schreibweg fuer `veraltet` (echtes Signal und Erstbefuellung): von unten nach oben ersetzen,
        neue Id, Belege mit dem gewaehlten Anker vorn (Quelle `llm`), Ledger-Eintrag; CAS, Torwaechter und pathspec-
        Commit folgen in `_voll`. `nur_lesend` (D7): nie ersetzen, Pruefpunkt `nur_lesend_veraltet` mit dem Vorschlag.
        Rueckgabe (neuer Text, [(alte Id, neue Id)] der ausgefuehrten Ersetzungen)."""
        getan: list[tuple[str, str]] = []
        for a, neuer_text, gewaehlt, belege in sorted(ersetzungen, key=lambda e: e[0].zeile_start, reverse=True):
            if nur_lesend:  # D7: nie ersetzen; der Vorschlag geht in die Pruefliste, die Belege werden nachgefuehrt
                self.offen.append(Pruefpunkt(
                    "nur_lesend_veraltet", f"Urteil veraltet ({gewaehlt.art} {gewaehlt.anker or '-'} in "
                    f"{gewaehlt.file}); Vorschlag: {' '.join(neuer_text.split())}"[:500], a.datei, a.zeile_start,
                    a.id))
                aussage_speichern(graph, a, belege, "extraktor")
                continue
            try:
                text = au.aussage_ersetzen(text, a, neuer_text)
            except ValueError as fehler:
                self.offen.append(Pruefpunkt("ersetzen_abgelehnt", str(fehler)[:300], a.datei, a.zeile_start, a.id))
                aussage_speichern(graph, a, belege, "extraktor")
                continue
            neu_id = au.aussage_id(a.datei, a.ueberschrift, neuer_text)
            aussage_entfernen(graph, a.id)
            rest = [b for b in belege if (b.art, b.file, b.anker) != (gewaehlt.art, gewaehlt.file, gewaehlt.anker)]
            aussage_speichern(graph, a, [gewaehlt, *rest], "llm", aid=neu_id, text=neuer_text)
            self.umgeschrieben += 1
            self.ledger.append(umschreiben.LedgerEintrag(
                lauf=self.lauf, zeit=self.jetzt.isoformat(), aussage_id=a.id, datei=a.datei, alt=a.text,
                neu="\n".join(au._ersatz_zeilen(neuer_text)), beleg=gewaehlt, urteil="veraltet", quelle="llm",
                basis_commit=self.kopf_aussen))
            getan.append((a.id, neu_id))
        return text, getan

    # ----- Erstbefuellung (Task 7, D10) -----

    @staticmethod
    def _signal_ids(signale: Sequence[Mapping[str, object]]) -> set[str]:
        ids: set[str] = set()
        for s in signale:
            aussage = s.get("aussage")
            if isinstance(aussage, Mapping) and isinstance(aussage.get("id"), str):
                ids.add(str(aussage["id"]))
        return ids

    def _erst_einstellen(self) -> None:
        """Faellig (fuer das Ergebnis) und ob gesammelt wird; gefragt wird nur im echten Tick ohne `--ohne-llm`."""
        e = erst_mod.einstellung(self.konfig)
        self.erst_sammeln = e.aktiv and not (self.stand_alt is not None and self.stand_alt.erstbefuellung_fertig)
        self.erst_faellig = erst_mod.erstbefuellung_faellig(self.konfig, self.stand_alt, self.jetzt)

    def _erst_budget_erschoepft(self) -> bool:
        """Fix-Runde 1 (M4): lesende Vorpruefung ueber `tagesstand()` des Budget-Waechters (gleicher Lader wie der
        Judge), damit ein Lauf ohne Aenderung nicht alle Dokus liest, wenn der Agent heute sicher nichts mehr darf:
        Tagesgrenze erreicht, Not-Aus, oder User aktiv bei eingeschalteter Pause (Klasse hintergrund). Nur eine Abkuerzung:
        `pruefen` bleibt die einzige Instanz; jeder Fehler hier heisst `False` (weiter wie bisher). Nur bei
        `erst_faellig`, also hoechstens einmal je Lauf im Fenster."""
        if not self.erst_faellig or self.ohne_llm:
            return False
        try:
            budget = self.budget
            if budget is None:
                from . import judge

                budget, _ = judge.budget_laden(self.konfig)
            stand_fn = getattr(budget, "tagesstand", None)
            if stand_fn is None:
                return False
            tag = stand_fn()
            agent = erst_mod.einstellung(self.konfig).agent
            eintrag = next((x for x in tag.get("agenten", []) if x.get("agent") == agent), None)
            grund = None
            if eintrag is not None and int(eintrag.get("heute", 0)) >= int(eintrag.get("grenze", 0)):
                grund = f"Tagesgrenze {eintrag.get('heute')}/{eintrag.get('grenze')}"
            elif tag.get("not_aus") is True:
                grund = "Not-Aus"
            elif (tag.get("user_aktiv") is True and tag.get("pause_aktiv_schalter", True) is True
                  and (eintrag is None or eintrag.get("klasse", "hintergrund") == "hintergrund")):
                grund = "User aktiv (Pause)"
        except Exception:  # noqa: BLE001 - die Vorpruefung darf nie etwas verhindern, das `pruefen` erlauben wuerde
            return False
        if grund is None:
            return False
        self.melden(f"Erstbefüllung: Budget heute erschöpft oder pausiert ({grund}) – in diesem Lauf nicht gefragt")
        return True

    def _erst_merken(self, a: au.Aussage, belege: Sequence[Beleg], *, fragbar: bool, nur_lesend: bool) -> None:
        if self.erst_sammeln and belege:
            self.erst_pool.append((a, tuple(belege), fragbar, nur_lesend))

    def _erstbefuellung(self, graph: Graph, texte: dict[str, tuple[bytes, str]],
                        uebrig: Sequence[Mapping[str, object]] = ()) -> None:
        """Task 7: die noch nicht erstgeprueften, signaltragend gebundenen Aussagen dieses Laufs (Ruling T7-a) unter dem
        Agenten der Erstbefuellung pruefen -- nach allen echten Signalen, mit der Rest-Deadline. Ohne `erst_fragen`
        (Trockenlauf, `--ohne-llm`, ausserhalb des Fensters) nur zaehlen. `veraltet` mit aufloesbarem Anker geht ueber
        `_ersetzen` (derselbe Weg wie ein Signal); sonst Pruefpunkt. Aussagen mit einem echten Signal in diesem Lauf
        und Dateien, die ein Signal schon umgeschrieben hat, bleiben offen (naechster Lauf)."""
        if not self.erst_sammeln:
            return
        alt = self.stand_alt or Stand()
        bekannt = set(alt.erstgeprueft) | self.erst_markiert | self.erst_ersetzt
        # Fix-Runde 1 (I1): nie eine Erstpruefung fuer eine Aussage, deren echtes Signal offen ist -- in diesem Lauf
        # gefragt (auch abgelehnt), aus dem Stand uebernommen (`uebrig`, auch ohne Neubau nicht wiedergefunden) oder neu
        # offen; sonst umginge die Erstbefuellung das doku-loop-Limit
        gesperrt = {k[0] for k in self.gefragt} | self._signal_ids(uebrig) | self._signal_ids(self.offene_signale) \
            | self._signal_ids(alt.offene_signale)
        posten = erst_mod.posten_waehlen(self.konfig, [
            (a, belege, fragbar and a.datei not in self.erst_umgeschrieben_docs and a.id not in gesperrt, nur)
            for a, belege, fragbar, nur in self.erst_pool])
        vollstaendig = not self.doku_fremd_ziele and not self.erst_zurueck
        if not self.erst_fragen:
            self.erst_offen = sum(1 for p in posten if p.aussage.id not in bekannt)
            self.melden(f"Erstbefüllung: {self.erst_offen} Aussage(n) offen; "
                        f"{'fällig' if self.erst_faellig else 'nicht fällig'}, hier nicht gefragt")
            return
        vorher = self.llm_aufrufe
        e = erst_mod.einstellung(self.konfig)

        def schluss() -> bool:  # Fix-Runde 1 (M6): Tick-Deadline oder das Fenster schliesst waehrend des Laufs
            jetzt = self.jetzt + timedelta(seconds=self.uhr() - self.start)
            return self._deadline() or not erst_mod.im_fenster(e, jetzt)

        erg = erst_mod.erstbefuellung_schritt(
            self.konfig, graph, dataclasses.replace(alt, erstgeprueft=sorted(bekannt)), budget=self.budget,
            laufzeit=self.laufzeit, deadline=schluss, posten=posten,
            quelltext=lambda b: self.quelltext(b, alt=False), digest=self.digest_jetzt,
            nachbarn=lambda b: self.nachbarn(graph, b), vollstaendig=vollstaendig, jetzt=self.jetzt)
        self.erst_aufrufe = self.llm_aufrufe - vorher
        self.erst_geprueft, self.erst_offen, self.erst_fertig = erg.geprueft, erg.offen, erg.fertig
        self.erst_fehlversuche = dict(erg.fehlversuche)
        ersetzungen: dict[tuple[str, bool], list[tuple[au.Aussage, str, Beleg, list[Beleg]]]] = {}
        for u in erg.urteile:
            a, b, antwort = u.posten.aussage, u.posten.beleg, u.antwort
            ort = f"Erstprüfung {b.art} {b.anker or '-'} in {b.file}"
            if antwort is not None and antwort.urteil == "veraltet":
                gewaehlt = next((x for x in a.belege if x.anker == antwort.beleg_anker and self.urteil(x) == "beleg_ok"),
                                None)
                hindernis = erst_mod.ersatz_hindernis(antwort, gewaehlt,
                                                      doku_da=u.posten.nur_lesend or a.datei in texte)
                if hindernis is None and gewaehlt is not None and antwort.neuer_text:
                    ersetzungen.setdefault((a.datei, u.posten.nur_lesend), []).append(
                        (a, antwort.neuer_text, gewaehlt, list(a.belege)))
                    if u.posten.nur_lesend:
                        self.erst_markiert.add(a.id)
                    continue
                text = {"beleg_nicht_aufloesbar": f"Judge nennt {antwort.beleg_anker!r}, nicht aufloesbar",
                        "erstpruefung_ohne_ersatz": "Urteil veraltet ohne Ersatztext",
                        "erstpruefung_doku_nicht_gelesen": "Urteil veraltet, Doku in diesem Lauf nicht gelesen"
                        }.get(str(hindernis), str(hindernis))
                self.offen.append(Pruefpunkt(str(hindernis), f"{ort}: {text}", a.datei, a.zeile_start, a.id))
            elif antwort is None and u.grund.startswith(erst_mod.GESCHEITERT):
                # Fix-Runde 1 (M5): ohne Inhalt der Aussage oder des Quelltexts, nur Ort und Grund
                self.offen.append(Pruefpunkt(
                    "erstpruefung_gescheitert", f"Erstprüfung {erst_mod.GIFT_GRENZE}x in Folge technisch gescheitert "
                    f"({u.grund[len(erst_mod.GESCHEITERT):]}); als geprüft markiert, bitte von Hand prüfen",
                    a.datei, a.zeile_start, a.id))
            elif antwort is None:
                self.offen.append(Pruefpunkt(f"judge_{u.grund.split(':', 1)[0]}", f"{ort}: {u.grund}", a.datei,
                                             a.zeile_start, a.id))
            elif antwort.urteil == "unklar":
                self.offen.append(Pruefpunkt("judge_unklar", f"{ort}: {antwort.begruendung[:200]}", a.datei,
                                             a.zeile_start, a.id))
            self.erst_markiert.add(a.id)
        for (doc, nur), liste in sorted(ersetzungen.items()):
            if nur:
                self._ersetzen(graph, "", liste, nur_lesend=True)
                continue
            roh, text = texte[doc]
            neu, getan = self._ersetzen(graph, text, liste, nur_lesend=False)
            texte[doc] = (roh, neu)
            self.erst_umgeschrieben += len(getan)
            for alt_id, neu_id in getan:
                self.erst_ersetzt |= {alt_id, neu_id}
        self.melden(f"Erstbefüllung: {erg.geprueft} geprüft ({self.erst_aufrufe} Aufruf(e) unter "
                    f"{erst_mod.einstellung(self.konfig).agent}), {self.erst_umgeschrieben} umgeschrieben, "
                    f"{erg.offen} offen" + (f"; Abbruch: {erg.abbruch}" if erg.abbruch else "")
                    + ("; fertig" if erg.fertig else ""))

    def _urteilen(self, graph: Graph, a: au.Aussage, gespeichert: Sequence[Beleg],
                  treffer: Sequence[tuple[Beleg, str]], aktuell: Sequence[Beleg]
                  ) -> tuple[au.Aussage, str, Beleg, list[Beleg]] | None:
        from . import judge  # nur hier und in `_offene_fragen`: `--ohne-llm` laedt den Judge nie

        b, _ = treffer[0]
        kandidaten: dict[tuple[str, str, str], Beleg] = {}
        k = self.kandidat_an_zeile(b)
        for x in (*aktuell, *([k] if k is not None else [])):
            kandidaten.setdefault((x.art, x.file, x.anker), x)
        verschwunden = {(x.art, x.file, x.anker) for x, u in treffer if u == "beleg_verschwunden"}
        signal_schl = {(y.art, y.file, y.anker) for y, _ in treffer}
        erneuert: list[Beleg] = []
        for x in gespeichert:
            schl = (x.art, x.file, x.anker)
            if schl in verschwunden:
                continue
            erneuert.append((self.nachgefuehrt(x) or x) if schl in signal_schl else x)
        schluessel = {(x.art, x.file, x.anker) for x in erneuert}
        erneuert.extend(x for x in aktuell if (x.art, x.file, x.anker) not in schluessel)

        frage_aussage = dataclasses.replace(a, belege=tuple(kandidaten.values()))
        schl_b = (b.art, b.file, b.anker)
        neu_digest = self.digest_jetzt(b) or ""
        cache = urteil_holen(self.konfig, a.id, schl_b, b.digest, neu_digest)
        if cache is not None:
            antwort, grund = _cache_antwort(cache, judge)
            self.melden(f"Urteil aus dem Cache: {antwort.urteil if antwort else grund}")
        elif (a.id, *schl_b) in self.gefragt:  # Item 4: in diesem Lauf schon gefragt; bleibt offen (uebrig)
            antwort, grund = None, self.gefragt[(a.id, *schl_b)]
        else:
            frage = judge.JudgeFrage(frage_aussage, b, self.quelltext(b, alt=True), self.quelltext(k or b, alt=False),
                                     self.nachbarn(graph, b))
            if self._deadline():  # I7a: nicht fragen, offen merken (naechster Lauf ohne Neubau)
                antwort, grund = None, "deadline"
            else:
                antwort, grund = judge.zweimal_einig(frage, self.konfig, **self._judge_kw())
                self.gefragt[(a.id, *schl_b)] = grund
            if antwort is not None or not grund.startswith(WIEDERHOLEN):
                urteil_merken(self.konfig, a.id, schl_b, b.digest, neu_digest,
                              _cache_eintrag(antwort, grund, self.jetzt))
            elif antwort is None:  # offen: alles merken, was der Judge fuer eine neue Frage ohne Neubau braucht
                self.offene_signale.append({
                    "space": self._space_von(b.file), "datei": b.file, "aussage": _aussage_dict(frage_aussage),
                    "beleg": b.to_dict(), "quelltext_alt": frage.quelltext_alt[:QUELLTEXT_MERKEN],
                    "quelltext_neu": frage.quelltext_neu[:QUELLTEXT_MERKEN], "nachbarn": list(frage.nachbarn),
                    "neu_digest": neu_digest})
        ort = f"{b.art} {b.anker or '-'} in {b.file}"
        if antwort is not None or not grund.startswith(WIEDERHOLEN):
            self.erst_markiert.add(a.id)  # Task 7: ein Signal-Urteil prueft die Aussage gegen den heutigen Code
        if antwort is None:
            if grund != "deadline":  # I7a: kein Pruefpunkt, das Signal ist nur verschoben
                self.offen.append(Pruefpunkt(f"judge_{grund.split(':', 1)[0]}", f"{ort}: {grund}", a.datei,
                                             a.zeile_start, a.id))
            if not grund.startswith(WIEDERHOLEN):
                aussage_speichern(graph, a, erneuert, "extraktor")
            return None  # bei WIEDERHOLEN bleiben die gemerkten Belege: das Signal bleibt offen
        if antwort.urteil == "veraltet":
            gewaehlt = next((x for x in kandidaten.values() if x.anker == antwort.beleg_anker
                             and self.urteil(x) == "beleg_ok"), None)
            if gewaehlt is None or not antwort.neuer_text:
                self.offen.append(Pruefpunkt("beleg_nicht_aufloesbar",
                                             f"{ort}: Judge nennt {antwort.beleg_anker!r}, nicht aufloesbar",
                                             a.datei, a.zeile_start, a.id))
                aussage_speichern(graph, a, erneuert, "extraktor")
                return None
            return a, antwort.neuer_text, gewaehlt, erneuert
        if antwort.urteil == "unklar":
            self.offen.append(Pruefpunkt("judge_unklar", f"{ort}: {antwort.begruendung[:200]}", a.datei,
                                         a.zeile_start, a.id))
        aussage_speichern(graph, a, erneuert, "llm" if antwort.urteil == "noch_wahr" else "extraktor")
        return None

    # ----- neue Space-Dokus (Task 6, D9) -----

    def _neue_spaces(self, texte: dict[str, tuple[bytes, str]]) -> None:
        """Space-Ordner im Messbaum erkennen; generierte Dokus mit Bloecken versorgen (fehlt ihr Ordner: Bloecke bleiben,
        `space_doku_ohne_space`); fuer Spaces ohne Doku `docs/spaces/<NNN>_<name>.md` anlegen (in `texte`, alt = b"")
        und den Index-Block `neue-spaces` rendern. Wartet, solange die README fremde Aenderungen hat oder ihr Block die
        neue Datei nicht verlinken kann (Hand-Edit, entfernt) -- sonst blockierte der Torwaechter jeden Commit."""
        konfig = self.konfig
        if not neue_spaces_aktiv(konfig):
            return
        pp: list[Pruefpunkt] = []
        # Ruling T6-a: Gitlinks des Messbaums (initialisiert und nicht) machen einen Ordner zum Space, nie zum Ausschluss
        self.gitlinks = sorted({*self.sub_koepfe, *self.nicht_messbar})
        ordner = neue_spaces_mod.space_ordner(self.mess_repo, unter_repos=self.gitlinks)
        je_key = {space_schluessel(n): (n, rel) for n, rel in ordner.items()}
        dateien = doku_dateien(konfig)
        self.neue_spaces = neue_spaces_mod.neue_spaces(konfig, None, self.mess_repo, dateien=dateien, ordner=ordner)
        for doc, (_, text) in sorted(texte.items()):  # 1. vorhandene generierte Dokus
            key = neue_spaces_mod.generierter_space(text)
            if key is None:
                continue
            inhalt = None
            if key in je_key:
                inhalt = self._space_bloecke(*je_key[key], pp)
            else:
                pp.append(Pruefpunkt("space_doku_ohne_space", f"{doc}: generierte Doku, aber kein Space-Ordner `{key}` "
                                     "mehr im committeten Stand; die Datei bleibt (Löschen nur von Hand). Datei löschen "
                                     "oder Belege anpassen; der Drift-Waechter meldet bis dahin 'Beleg fehlt'.", doc,
                                     None, f"space:{key}"))
            # Bloecke ohne neuen Inhalt (Ordner weg, Fehler, Space nicht mehr vermessen) bleiben, wie sie sind
            stehen = {b: x.inhalt.replace("\r\n", "\n") for b, x in render_ketten.bloecke_finden(text).items()
                      if b.startswith((*neue_spaces_mod.GENERIERT, neue_spaces_mod.EREIGNISSE))}
            self.generiert_bloecke[doc] = {**stehen, **(inhalt or {})}
        readme_fremd = bool(umschreiben.fremde_aenderungen(self.repo, [NEUE_SPACES_README]))
        if self.neue_spaces and readme_fremd:
            pp.append(self._wartet(f"{NEUE_SPACES_README} hat nicht committete fremde Änderungen"))
        elif self.neue_spaces:  # 2. neue Dokus
            vorhanden = {d for d in (*dateien, *self.versioniert) if d.startswith(neue_spaces_mod.ORDNER + "/")}
            vorhanden |= {p.name for p in (self.repo / neue_spaces_mod.ORDNER).glob("*.md")}
            for name, doc in neue_spaces_mod.dateien_planen(self.neue_spaces, vorhanden, pp):
                inhalt = self._space_bloecke(name, ordner[name], pp)
                if inhalt is None:
                    continue
                if (self.repo / doc).exists():  # nie ueberschreiben
                    pp.append(self._wartet(f"{doc} existiert schon"))
                    continue
                nummer = int(doc.rsplit("/", 1)[-1].split("_", 1)[0])
                vermessen = neue_spaces_mod.SCHNITTSTELLEN + space_schluessel(name) in inhalt  # sonst T6-a (c)
                geruest = neue_spaces_mod.geruest(name, nummer, vermessen=vermessen)
                texte[doc] = (b"", neue_spaces_mod.generierte_rendern(geruest, inhalt, {}, doc).text)
                self.generiert_bloecke[doc] = inhalt
                self.neu_angelegt.append(doc)
        if not readme_fremd:  # 3. Index
            self._readme(texte, pp)
        self.struktur.extend(pp)
        self.melden(f"Neue Spaces: {len(ordner)} Space-Ordner, ohne Doku {', '.join(self.neue_spaces) or 'keine'}; "
                    f"angelegt {', '.join(self.neu_angelegt) or 'nichts'}; generierte Dokus {len(self.generiert_bloecke)}")

    def _wartet(self, grund: str) -> Pruefpunkt:
        folge = (f"; neue Space-Doku für {', '.join(self.neue_spaces)} wartet" if self.neue_spaces
                 else "; Index nicht gerendert")
        return Pruefpunkt("space_doku_wartet", grund + folge, NEUE_SPACES_README, None, "space_doku_wartet")

    def _neue_verwerfen(self, texte: dict[str, tuple[bytes, str]]) -> None:
        """Die in diesem Lauf angelegten Dateien zuruecknehmen (sie lagen nur in `texte`, nichts ist geschrieben)."""
        for doc in self.neu_angelegt:
            texte.pop(doc, None)
            self.generiert_bloecke.pop(doc, None)
        self.neu_angelegt = []

    def _readme(self, texte: dict[str, tuple[bytes, str]], pp: list[Pruefpunkt]) -> None:
        """Block `neue-spaces` mit einer Zeile je generierter Doku (auch fremd geaendert ausgelassene: die Datei gibt es).
        Kann er eine neu angelegte Datei nicht verlinken, werden die neuen Dateien in diesem Lauf verworfen."""
        alt = self.stand_alt or Stand()
        praefix = f"{NEUE_SPACES_README}#"
        alte = {k[len(praefix):]: v for k, v in alt.render_hashes.items() if k.startswith(praefix)}
        try:
            text = (self.readme_roh or b"").decode("utf-8")
        except UnicodeDecodeError:  # M4: Pruefpunkt statt Abbruch; ohne Index-Zeile keine neue Doku (Torwaechter)
            pp.append(self._wartet(f"{NEUE_SPACES_README} ist kein UTF-8"))
            self._neue_verwerfen(texte)
            return

        def eintraege() -> list[tuple[str, str]]:
            aus: list[tuple[str, str]] = []
            for doc in sorted({*doku_dateien(self.konfig), *self.neu_angelegt}):
                if not doc.startswith(neue_spaces_mod.ORDNER + "/") or doc.count("/") != 2:
                    continue
                try:
                    inhalt = texte[doc][1] if doc in texte else (self.repo / doc).read_bytes().decode("utf-8", "replace")
                except OSError:
                    continue
                key = neue_spaces_mod.generierter_space(inhalt)
                if key is not None:
                    kopf = inhalt.lstrip("﻿").split("\n", 1)[0].strip()
                    teile = kopf.split(" ", 2)
                    aus.append((doc, teile[2] if len(teile) == 3 and teile[0] == "#" else key))
            return aus

        liste = eintraege()
        if not liste and neue_spaces_mod.README_BLOCK not in render_ketten.bloecke_finden(text) \
                and neue_spaces_mod.README_BLOCK not in alte:
            return
        res = neue_spaces_mod.readme_rendern(text, liste, alte)
        if [d for d in self.neu_angelegt if not neue_spaces_mod.im_index(res.text, d)]:
            pp.append(self._wartet(f"Block `{neue_spaces_mod.README_BLOCK}` in {NEUE_SPACES_README} ist von Hand "
                                   "geändert oder entfernt und kann die neue Datei nicht verlinken"))
            self._neue_verwerfen(texte)
            res = neue_spaces_mod.readme_rendern(text, eintraege(), alte)
        self.readme, self.readme_text = res, text

    def _space_bloecke(self, name: str, ort: str, pp: list[Pruefpunkt]) -> dict[str, str] | None:
        """Bloecke einer generierten Doku: Graph nur dieses Space (`bauen.graph_bauen`, Messbaum), Events, Stand. Ein
        Fehler wird `space_doku_fehler` (nur der Typ) und die Doku in diesem Lauf nicht (neu) gerendert.

        Ruling T6-a: (a) enthaelt der Ordner Gitlinks, werden nur seine eigenen Dateien vermessen (die Unter-Repos sind
        fuer Korpus und Extraktoren ausgeschlossen); (b) ist er ein initialisierter Gitlink, wird sein Worktree im
        Messbaum vermessen; (c) ist er ein nicht initialisierter Gitlink, gibt es nur Steckbrief (Ort, Hinweis) und
        Belege. Der Stand nennt jeden Gitlink am oder unter dem Ordner mit seinem Pin."""
        try:
            links = neue_spaces_mod.gitlinks_von(ort, self.gitlinks)
            stand, datum = self._stand(ort, links)
            if any(neue_spaces_mod.pfad_gleich(ort, g) for g in self.nicht_messbar):  # (c)
                return neue_spaces_mod.bloecke_unvermessen(name, ort, stand=stand, datum=datum)
            mess = self.mess_konfig
            # MCP-Muster eines konfigurierten Space mit diesem Ordner, sonst `decorator` (`@mcp.tool`)
            muster = next((s.mcp_muster for s in mess.spaces.values() if s.wurzel.strip("/") == ort), "decorator")
            sk = SpaceKonfig(name, ort, muster)
            innen = frozenset(g for g in links if not neue_spaces_mod.pfad_gleich(g, ort))  # (a): nur eigene Dateien
            k = dataclasses.replace(mess, spaces={sk.name: sk}, ausschluss_pfade=mess.ausschluss_pfade | innen)
            manifest = self.manifest_bauer(sk, k, self.commit_fuer(ort + "/"))
            graph, bau_pp = bauen.graph_bauen(k, [sk.name], None, {sk.name: manifest}, sql_defs=self.sql_defs,
                                              commit_aussen=self.kopf_aussen, commit_sub=self.kopf_sub)
            pp.extend(p for p in bau_pp if p.art in ("extraktor_fehler", "datensatz_fehler"))
            inhalt = neue_spaces_mod.bloecke(sk.name, graph, self.evs, ort=ort, stand=stand, datum=datum)
        except Exception as fehler:  # noqa: BLE001 - ein Space darf den Lauf nicht umwerfen
            pp.append(Pruefpunkt("space_doku_fehler", f"{name}: {type(fehler).__name__} in {_fehler_ort(fehler)}; "
                                 "Space-Doku in diesem Lauf nicht gerendert", None, None, f"space:{space_schluessel(name)}"))
            return None
        return inhalt

    def _eltern(self, pfad: str) -> tuple[str | None, Path, str, str]:
        """(Repo, echter Checkout, Pfad darin, gemessener Commit) des Repos, das `pfad` enthaelt -- bei einem Gitlink
        das Eltern-Repo, nicht der Gitlink selbst."""
        oben = pfad.rsplit("/", 1)[0] if "/" in pfad else ""
        sub = self._submodul_von(oben) if oben else None
        if sub is None:
            return None, self.repo, pfad, self.kopf_aussen
        commit = self.sub_koepfe.get(sub) or (self.kopf_sub if sub == SUBMODUL else "")
        return sub, self.repo / sub, pfad[len(sub) + 1:], commit

    def _stand(self, ort: str, links: Sequence[str] = ()) -> tuple[str, str]:
        """(`<repo>@<sha7>[, <gitlink>@<pin7> ...]`, Datum): Repo, in dem der Space-Ordner liegt (Submodul-Pfad bzw.
        Name des aeusseren Repos), mit dem letzten Commit am Ordner bis zum gemessenen Commit (aendert sich nur, wenn sich
        der Space aendert, bei einem Gitlink also beim Pin-Wechsel); dazu jeder Gitlink am oder unter dem Ordner mit
        seinem Pin (`docs_drift_waechter.pruefe_pins` verlangt dessen sha7 im Text). Datum: `%cs` des ersten sha."""
        sub, wo, pfad, commit = self._eltern(ort)
        sha = (git(wo, "log", "-1", "--format=%H", commit, "--", pfad) or "").strip() if commit else ""
        basis = sha or commit
        teile = [f"{sub or self.repo.name}@{basis[:7]}"]
        for g in links:
            _, g_wo, g_pfad, g_commit = self._eltern(g)
            zeile = (git(g_wo, "ls-tree", g_commit, "--", g_pfad) or "") if g_commit else ""
            meta = zeile.split("\t", 1)[0].split()
            if len(meta) == 3 and meta[0] == "160000":
                teile.append(f"{g}@{meta[2][:7]}")
        datum = (git(wo, "log", "-1", "--format=%cs", basis) or "").strip() if basis else ""
        return ", ".join(teile), datum

    def _ereignisse_lesen(self) -> None:
        """Task 5: Events aus den Quellen im Messbaum lesen (T5-a, committeter Stand); einmal je Lauf."""
        self.evs, self.ev_praefixe, self.ev_pp = [], set(), []
        if self.konfig.ereignisse:
            self.evs = ereignisse_ex.ereignisse_lesen(self.mess_konfig, self.mess_repo, praefixe=self.ev_praefixe,
                                                      pruefpunkte=self.ev_pp)

    def _ereignisse(self, graph: Graph, texte: Mapping[str, tuple[bytes, str]]) -> None:
        """Task 5 (D8): die gelesenen Events (`_ereignisse_lesen`) als `ereignis`-Knoten in den Graph, je zugeordneter
        Doku-Datei ein Block je Space (Task 6: auch die in diesem Lauf angelegten), Abgleich in die Pruefliste
        (`struktur`). Ohne Quellen werden nur alte Ereignis-Knoten entfernt."""
        konfig = self.konfig
        evs, pp, praefixe = self.evs, list(self.ev_pp), self.ev_praefixe
        bauen.ereignisse_einbauen(graph, evs)
        if not konfig.ereignisse:
            return
        zuordnung = ereignisse_ex.space_dokumente_zuordnen([e.space for e in evs], konfig.space_dokumente,
                                                           [*doku_dateien(konfig), *self.neu_angelegt], pruefpunkte=pp)
        bloecke: dict[str, dict[str, str]] = {}
        for key, doc in sorted(zuordnung.items()):
            bloecke.setdefault(doc, {})[render_ereignisse.block_name(key)] = render_ereignisse.ereignisse_rendern(
                key, evs)
        self.ereignis_bloecke = bloecke
        # bekannte SQL-Objekte (`schema.name`) sind keine Events, auch wenn das Schema wie ein Event-Praefix heisst
        pp += ereignisse_ex.abgleich(evs, {doc: neu for doc, (_, neu) in texte.items()}, zuordnung, praefixe=praefixe,
                                     nicht_ereignisse=set(self.sql_defs))
        self.struktur.extend(pp)
        self.n_ereignisse = len({(space_schluessel(e.space), e.name) for e in evs})
        arten: dict[str, int] = {}
        for p in pp:
            arten[p.art] = arten.get(p.art, 0) + 1
        self.melden(f"Ereignisse: {self.n_ereignisse} in {len({space_schluessel(e.space) for e in evs})} Space(s), "
                    f"{len(zuordnung)} mit Doku; Prüfliste " + (", ".join(f"{a} {n}" for a, n in sorted(arten.items()))
                                                                 or "nichts"))

    def _rendern(self, graph: Graph, texte: dict[str, tuple[bytes, str]]) -> tuple[dict[str, str], dict[str, str]]:
        """Markierte Bloecke aller gelesenen Doku-Dateien rendern (Task 5: verallgemeinert). KETTEN.md: Uebersicht und
        Hops, Render-Hashes unter dem Blocknamen (wie bisher). Andere Doku: Events-Bloecke (`render_ereignisse`), Hashes
        unter `<datei>#<block>`. Hand-Edit-Schutz und `hand_edit` (einmal je Block und Inhalts-Hash) fuer alle gleich.
        Hashes nicht gelesener Dateien bleiben stehen."""
        alt = self.stand_alt or Stand()
        md = self.konfig.ketten_md
        hashes, hand_edit = dict(alt.render_hashes), dict(alt.hand_edit_gemeldet)
        n_bloecke = geaendert = neu_hand = 0
        readme = [NEUE_SPACES_README] if self.readme is not None else []
        for doc in [*texte, *readme]:
            praefix = "" if doc == md else f"{doc}#"

            def eigen(k: str, praefix: str = praefix) -> bool:
                return k.startswith(praefix) if praefix else not _doku_schluessel(k)

            alte = {k[len(praefix):]: v for k, v in alt.render_hashes.items() if eigen(k)}
            if doc in readme and self.readme is not None:  # Task 6: in `_neue_spaces` gerendert (nur `neue-spaces`)
                roh, text, res = self.readme_roh or b"", self.readme_text, self.readme
            elif doc == md:
                roh, text = texte[doc]
                res = render_ketten.rendern(text, graph, alte, datei=md)
            else:
                roh, text = texte[doc]
                # Task 6: Bloecke generierter Dokus (nur vorhandene; Events rendert Task 5, sonst der Platzhalter)
                vorhanden = render_ketten.bloecke_finden(text)
                inhalt = {**{b: v for b, v in self.generiert_bloecke.get(doc, {}).items() if b in vorhanden},
                          **self.ereignis_bloecke.get(doc, {})}
                if not inhalt and not alte and "graph:" not in text:
                    continue
                res = render_ereignisse.doku_rendern(text, inhalt, alte, datei=doc)
            for k in [k for k in hashes if eigen(k)]:
                del hashes[k]
            for k in [k for k in hand_edit if eigen(k)]:
                del hand_edit[k]
            hashes.update({praefix + b: h for b, h in res.hashes.items()})
            bloecke = render_ketten.bloecke_finden(text)
            for p in res.pruefpunkte:
                if p.art == "hand_edit" and p.aussage_id in bloecke:
                    h = render_ketten.block_hash(bloecke[p.aussage_id].inhalt)
                    hand_edit[praefix + p.aussage_id] = h
                    if alt.hand_edit_gemeldet.get(praefix + p.aussage_id) != h:
                        neu_hand += 1
                        self.offen.append(Pruefpunkt(p.art, p.text, p.datei, p.zeile, f"{p.aussage_id}#{h[:12]}"))
                elif p.art == "marker_kaputt":
                    self.offen.append(p)
                else:
                    self.struktur.append(p)
            texte[doc] = (roh, res.text)
            n_bloecke += len(res.hashes)
            geaendert += sum(1 for b, h in res.hashes.items() if alte.get(b) != h)
        if n_bloecke:
            self.melden(f"Render: {n_bloecke} Block/Blöcke, {geaendert} neu, {neu_hand} neue Hand-Edits")
        return hashes, hand_edit


def _melder(melden: Melden | None) -> Melden:
    return melden if melden is not None else (lambda _zeile: None)


def tick(konfig: Konfig, *, jetzt: datetime, erzwingen: bool = False, laufzeit: Callable[..., Any] = subprocess.run,
         ohne_llm: bool = False, budget: Any = None,
         drift: Callable[[Path], set[umschreiben.DriftFehler]] | None = None,
         manifest_bauer: ManifestBauer | None = None, nach_tick: Callable[[TickErgebnis], None] | None = None,
         melden: Melden | None = None, uhr: Callable[[], float] = time.monotonic,
         trocken: bool = False) -> TickErgebnis:
    """Ein Lauf des Doku-Loops (siehe Moduldoku). `budget` None -> `claude_budget` (erst im Judge geladen). `uhr`:
    monotone Uhr fuer die Tick-Deadline (I7a). `trocken`: nur messen und zaehlen (`_Lauf.trocken_laufen`), schreibt
    nichts, nimmt keine Sperre, ruft kein LLM; `erzwingen` und `ohne_llm` sind dann ohne Wirkung."""
    melden = _melder(melden)
    lauf = _Lauf(konfig, jetzt=jetzt, erzwingen=erzwingen, laufzeit=laufzeit, ohne_llm=ohne_llm, budget=budget,
                 drift=drift or umschreiben.drift_fuer(konfig), manifest_bauer=manifest_bauer or korpus.manifest_bauen,
                 nach_tick=nach_tick, melden=melden, uhr=uhr)
    if trocken:
        vorher = os.environ.get("GIT_OPTIONAL_LOCKS")
        os.environ["GIT_OPTIONAL_LOCKS"] = "0"  # `git status` darf den Index nicht auffrischen (schreiben)
        try:
            return lauf.trocken_laufen()
        finally:
            if vorher is None:
                os.environ.pop("GIT_OPTIONAL_LOCKS", None)
            else:
                os.environ["GIT_OPTIONAL_LOCKS"] = vorher
    with sperre(lokaler_ordner(konfig)) as frei:
        if not frei:
            melden("Sperre: ein anderer Lauf hält tick.lock – übersprungen (laeuft_schon)")
            return TickErgebnis(f"{jetzt:%Y-%m-%dT%H:%M:%S}", [], 0, 0, 0, 0, None, "laeuft_schon")
        return lauf.laufen()
