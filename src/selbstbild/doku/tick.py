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
"""

from __future__ import annotations

import copy
import dataclasses
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import anker, bauen, ketten_seed, korpus, messbaum, render_ketten, rowboat, umschreiben, waechter
from . import aussagen as au
from .anker import Beleg
from .belegspeicher import aussage_entfernen, aussage_speichern, beleg_dateien, gespeicherte_belege
from .extraktoren import sql as sql_ex
from .graph import Graph, graph_diff, mit_schlank, schlank
from .konfig import Konfig, SpaceKonfig
from .pruefliste import (
    beleg_nicht_messbar, beleg_untracked, neue_offene, pruefliste_ergaenzen, pruefliste_laden, pruefliste_pfad,
)
from .schreibvorgang import (
    GRAPH_DIR, GRAPH_JSON, LEDGER, SUBMODUL, Aenderung, Ausgang, doku_dateien, eigene_datei, fremde_aenderungen, git,
    graph_text, halbschreibung_aufraeumen, hat_submodul, kopf, kopf_pruefen, ledger_commit_von,
    schreiben_pruefen_committen, submodul_kopf, unsaubere_dateien, versionierte_dateien,
)
from .seed_befehl import RenderBericht, SchlankBericht, SeedErgebnis, graph_schlank, neu_rendern, seed, seed_deckung
from .stand import (
    SPERRE_DATEI, Stand, anstoss_noetig, fremd_commits_laden, fremd_commits_merken, lokaler_ordner, marker_loeschen,
    schreibvermerk_pfad, sperre, sperre_zustand, stand_laden, stand_pfad, stand_schreiben, temp_aufraeumen,
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


def _unter(datei: str, wurzeln: Sequence[str]) -> bool:
    return any(datei == w or datei.startswith(w.rstrip("/") + "/") for w in wurzeln if w)


def _unter_sql_wurzel(datei: str, wurzeln: Sequence[str]) -> bool:
    return any(w.strip("/") in ("", ".") or _unter(datei, [w.strip("/")]) for w in wurzeln)


def betroffene_spaces(konfig: Konfig, aussen: Sequence[str], sub: Sequence[str]) -> list[str]:
    """Spaces, die neu gebaut werden. Alle bei unbekanntem Stand (`*`), geaendertem Dienst-Register oder Loop-Konfig,
    jeder geaenderten `*.sql` unter `sql_wurzeln` und jeder geaenderten `compose_live`-Datei (I5: SQL-Definitionen
    und Compose-Kanten sind global); sonst die Spaces, unter deren Wurzel eine geaenderte Datei liegt."""
    alle = sorted(konfig.spaces)
    if "*" in aussen or "*" in sub:
        return alle
    if any(p in (konfig.dienste_yaml, KONFIG_DATEI) for p in aussen):
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

    # ----- Ergebnis und Stand -----

    def ergebnis(self, *, commit: str | None = None, uebersprungen: str | None = None,
                 pruefliste: int = 0) -> TickErgebnis:
        return TickErgebnis(self.lauf, list(self.spaces), self.signale, self.llm_aufrufe, self.umgeschrieben,
                            pruefliste, commit, uebersprungen)

    def stand_speichern(self, *, fortschreiben: bool, render_hashes: Mapping[str, str] | None = None,
                        hand_edit: Mapping[str, str] | None = None,
                        offene: Sequence[Mapping[str, object]] | None = None) -> None:
        alt = self.stand_alt or Stand()
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
            gemessen_ausstehend=dict(self.ausstehend if self.ausstehend is not None else alt.gemessen_ausstehend),
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
                                                fremd_commits=fremd_commits_laden(konfig))
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
        ledger = rowboat.ledger_lesen(self.repo / LEDGER)
        spitzen = rowboat.datei_spitzen(self.repo, {e.datei for e in ledger})
        commits = {**fremd_commits_laden(konfig), **rowboat.lauf_commits(self.repo)}  # I11: auch fremd uebernommene
        vermerke = rowboat.vermerke_laden(rowboat.erledigt_pfad(konfig))
        bekannt = set(commits) | {e.lauf for e in ledger} | set(self._letzte_laeufe())
        for _ in range(2):  # neue `notiz_von_hand`-Punkte erscheinen sofort in der Pruefliste
            liste = pruefliste_laden(konfig)
            von_hand: list[Pruefpunkt] = []
            rowboat.notizen_schreiben(
                graph, liste["offen"], ledger, konfig.rowboat_wissen, konfig.rowboat_notizen, self.lauf,
                manifest_pfad=rowboat.manifest_pfad(konfig), commits=commits, spitzen=spitzen, vermerke=vermerke,
                struktur=liste["struktur"], urteile_datei=konfig.rowboat_urteile, bekannte_laeufe=bekannt,
                melden=self.melden, pruefpunkte=von_hand)
            if not von_hand or not pruefliste_ergaenzen(konfig, von_hand, [], voll=False, lauf=self.lauf):
                break

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

        # 1. Anstoss
        try:
            marker = konfig.marker.exists()
        except OSError:
            marker = False
        if not self.erzwingen and not anstoss_noetig(konfig, self.stand_alt, self.jetzt):
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
        kopf_aussen = kopf(self.repo)
        if kopf_aussen is None:
            raise TickFehler("kein Git-HEAD im Repo")
        self.kopf_aussen, self.kopf_sub = kopf_aussen, submodul_kopf(self.repo) or ""
        self.sub_koepfe = _koepfe(self.repo)
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
        offene = list(self.stand_alt.offene_signale) if self.stand_alt else []
        self.spaces = spaces

        def zahl(liste: Sequence[str]) -> str:
            return "alle" if "*" in liste else str(len(liste))

        self.melden(f"Änderungen: {zahl(aussen)} Dateien außen, {zahl(sub)} im Submodul -> Spaces: "
                    f"{', '.join(spaces) or 'keine'}; Beleg-Dateien außerhalb: {len(belegdateien)}; "
                    f"offene Signale: {len(offene)}")
        if not spaces and not belegdateien and not offene and not self.render_noetig:
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
        for doc in doku_dateien(konfig):  # CAS-Basis der Doku jetzt festhalten (nicht erst nach dem Judge)
            try:
                self.doku_roh[doc] = (self.repo / doc).read_bytes()
            except OSError:
                pass
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
        if not spaces and not belegdateien and not self.render_noetig:
            beantwortet, rest = self._offene_fragen(offene)
            self.signale = len(offene)
            if not beantwortet:
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
            self.melden(f"Offene Signale: {len(beantwortet)} beantwortet -> Spaces: {', '.join(spaces) or 'keine'}")
            uebrig = rest  # Item 4: in diesem Lauf schon gefragt -> bleiben offen, `_voll` fragt sie nicht erneut
        else:
            uebrig = [s for s in offene if isinstance(s.get("space"), str) and s["space"] not in spaces]
        return self._voll(basis, basis_roh, schlank(repo_graph), vorher, uebrig)

    def _halbschreibung(self) -> None:
        h = halbschreibung_aufraeumen(self.repo, schreibvermerk_pfad(self.konfig), self.melden)
        if h.zustand == "zurueckgesetzt":
            pruefliste_ergaenzen(self.konfig, [Pruefpunkt(
                "eigene_halbschreibung_zurueckgesetzt", f"Lauf {h.lauf or '?'} brach zwischen Schreiben und Commit ab; "
                f"{', '.join(h.dateien)} mit CAS auf HEAD zurückgesetzt", None, None, h.lauf)], [], voll=False,
                lauf=self.lauf)
        elif h.zustand == "fremd":
            self.melden(f"Halbschreibung: {len(h.dateien)} Datei(en) weichen ab – gilt als fremde Änderung")

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
            texte[doc] = (roh, neu)
        self.signale = hop_signale + aussagen_signale
        self.melden(f"Belege: {self.signale} Signal(e) (Hops {hop_signale}, Aussagen {aussagen_signale}); "
                    f"Judge {self.llm_aufrufe} Aufruf(e), {self.umgeschrieben} umgeschrieben")

        # 7. Rendern
        render_hashes, hand_edit = self._rendern(graph, texte)
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
        aenderungen = [Aenderung(doc, roh, neu) for doc, (roh, neu) in texte.items() if neu.encode("utf-8") != roh]
        # Ruling 48a: Unterschiede, die nur aus nicht committeten Dateien kommen, zaehlen nicht
        neu_schlank = schlank(graph)
        graph_geaendert = neu_schlank.digest_stabil(ohne_dateien=self.unsauber) != alt_schlank.digest_stabil(
            ohne_dateien=self.unsauber)
        if graph_geaendert:  # Item 1: CAS gegen die Bytes, aus denen die Basis geladen wurde
            aenderungen.append(Aenderung(GRAPH_JSON, basis_roh, graph_text(neu_schlank, self.jetzt.isoformat())))
        if not aenderungen and not self.umgeschrieben:
            neu_offen = pruefliste_ergaenzen(konfig, self.offen, self.struktur, voll=voll, lauf=self.lauf)
            self.stand_speichern(fortschreiben=True, render_hashes=render_hashes, hand_edit=hand_edit, offene=offene)
            self.melden(f"Schreiben: nichts geändert (stabiler Digest gleich); Prüfliste +{neu_offen}; "
                        f"Stand: Lauf {self.lauf_nr}")
            return self._abschluss(self.ergebnis(pruefliste=neu_offen))

        m = neue_offene(konfig, self.offen)
        betreff = f"docs(loop): {', '.join(self.spaces) or 'Belege'} – {self.umgeschrieben} Aussagen nachgeführt, " \
                  f"{m} in Prüfliste"
        aus: Ausgang = schreiben_pruefen_committen(repo, aenderungen, betreff, self.lauf, vorher=vorher,
                                                   drift=self.drift, ledger=self.ledger, melden=self.melden,
                                                   pp_datei=konfig.ketten_md, zweig=konfig.zweig,
                                                   vermerk=schreibvermerk_pfad(konfig))
        neu_offen = pruefliste_ergaenzen(konfig, [*self.offen, *aus.pruefpunkte], self.struktur, voll=voll,
                                         lauf=self.lauf)
        if aus.fremd_commit is not None:  # I11: lokal vermerken, welcher fremde Commit die Ledger-Eintraege traegt
            try:
                fremd_commits_merken(konfig, self.ledger, aus.fremd_commit)
            except OSError as fehler:
                self.melden(f"Ledger-Vermerk: {type(fehler).__name__}")
        if aus.commit is not None or aus.fremd_commit is not None:
            self.stand_speichern(fortschreiben=True, render_hashes=render_hashes, hand_edit=hand_edit, offene=offene)
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

    def _aussagen(self, graph: Graph, doc: str, pfad: Path, text: str) -> tuple[str, int]:
        """Aussagen einer Datei pruefen; Rueckgabe (neuer Text, Zahl der Signale)."""
        # Ruling 50: Text aus dem echten Arbeitsbaum, verankert wird gegen den Messbaum
        liste = au.aussagen_lesen(self.mess_repo / doc, self.mess_konfig, self.manifeste, self.sql_defs,
                                  self.kopf_aussen, text=text)
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
                continue
            urteile = [(b, self.urteil(b)) for b in gespeichert]
            if any(u == "unsauber" for _, u in urteile):
                continue  # Ruling 48a: nicht urteilen, gemerkte Belege bleiben unveraendert
            treffer = [(b, u) for b, u in urteile if u != "beleg_ok"]
            if not treffer:
                schluessel = {(b.art, b.file, b.anker) for b in gespeichert}
                dazu = [b for b in aktuell if (b.art, b.file, b.anker) not in schluessel]
                if dazu:
                    aussage_speichern(graph, a, [*gespeichert, *dazu], "extraktor")
                continue
            signale += len(treffer)
            if self.ohne_llm:
                b, u = treffer[0]
                self.offen.append(Pruefpunkt("signal_ohne_llm", f"{u}: {b.art} {b.anker or '-'} in {b.file}",
                                             a.datei, a.zeile_start, a.id))
                continue
            ergebnis = self._urteilen(graph, a, gespeichert, treffer, aktuell)
            if ergebnis is not None:
                ersetzungen.append(ergebnis)
        for a, neuer_text, gewaehlt, belege in sorted(ersetzungen, key=lambda e: e[0].zeile_start, reverse=True):
            try:
                text = au.aussage_ersetzen(text, a, neuer_text)
            except ValueError as fehler:
                self.offen.append(Pruefpunkt("ersetzen_abgelehnt", str(fehler)[:300], a.datei, a.zeile_start, a.id))
                aussage_speichern(graph, a, belege, "extraktor")
                continue
            neu_id = au.aussage_id(a.datei, a.ueberschrift, neuer_text)
            aussage_entfernen(graph, a.id)
            ids.discard(a.id)
            ids.add(neu_id)
            rest = [b for b in belege if (b.art, b.file, b.anker) != (gewaehlt.art, gewaehlt.file, gewaehlt.anker)]
            aussage_speichern(graph, a, [gewaehlt, *rest], "llm", aid=neu_id, text=neuer_text)
            self.umgeschrieben += 1
            self.ledger.append(umschreiben.LedgerEintrag(
                lauf=self.lauf, zeit=self.jetzt.isoformat(), aussage_id=a.id, datei=a.datei, alt=a.text,
                neu="\n".join(au._ersatz_zeilen(neuer_text)), beleg=gewaehlt, urteil="veraltet", quelle="llm",
                basis_commit=self.kopf_aussen))
        weg = [k.id[len("aussage:"):] for k in graph.knoten.values()
               if k.typ == "aussage" and k.datei == doc and k.id[len("aussage:"):] not in ids]
        for aid in weg:
            aussage_entfernen(graph, aid)
        return text, signale

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

    def _rendern(self, graph: Graph, texte: dict[str, tuple[bytes, str]]) -> tuple[dict[str, str], dict[str, str]]:
        alt = self.stand_alt or Stand()
        md = self.konfig.ketten_md
        if md not in texte:
            return dict(alt.render_hashes), dict(alt.hand_edit_gemeldet)
        roh, text = texte[md]
        res = render_ketten.rendern(text, graph, alt.render_hashes, datei=md)
        bloecke = render_ketten.bloecke_finden(text)
        hand_edit: dict[str, str] = {}
        neu_hand = 0
        for p in res.pruefpunkte:
            if p.art == "hand_edit" and p.aussage_id in bloecke:
                h = render_ketten.block_hash(bloecke[p.aussage_id].inhalt)
                hand_edit[p.aussage_id] = h
                if alt.hand_edit_gemeldet.get(p.aussage_id) != h:
                    neu_hand += 1
                    self.offen.append(Pruefpunkt(p.art, p.text, p.datei, p.zeile, f"{p.aussage_id}#{h[:12]}"))
            elif p.art == "marker_kaputt":
                self.offen.append(p)
            else:
                self.struktur.append(p)
        texte[md] = (roh, res.text)
        geaendert = sum(1 for b, h in res.hashes.items() if alt.render_hashes.get(b) != h)
        self.melden(f"Render: {len(res.hashes)} Block/Blöcke, {geaendert} neu, {neu_hand} neue Hand-Edits")
        return res.hashes, hand_edit


def _melder(melden: Melden | None) -> Melden:
    return melden if melden is not None else (lambda _zeile: None)


def tick(konfig: Konfig, *, jetzt: datetime, erzwingen: bool = False, laufzeit: Callable[..., Any] = subprocess.run,
         ohne_llm: bool = False, budget: Any = None,
         drift: Callable[[Path], set[umschreiben.DriftFehler]] | None = None,
         manifest_bauer: ManifestBauer | None = None, nach_tick: Callable[[TickErgebnis], None] | None = None,
         melden: Melden | None = None, uhr: Callable[[], float] = time.monotonic) -> TickErgebnis:
    """Ein Lauf des Doku-Loops (siehe Moduldoku). `budget` None -> `claude_budget` (erst im Judge geladen). `uhr`:
    monotone Uhr fuer die Tick-Deadline (I7a)."""
    melden = _melder(melden)
    lauf = _Lauf(konfig, jetzt=jetzt, erzwingen=erzwingen, laufzeit=laufzeit, ohne_llm=ohne_llm, budget=budget,
                 drift=drift or umschreiben.drift_fuer(konfig), manifest_bauer=manifest_bauer or korpus.manifest_bauen,
                 nach_tick=nach_tick, melden=melden, uhr=uhr)
    with sperre(lokaler_ordner(konfig)) as frei:
        if not frei:
            melden("Sperre: ein anderer Lauf hält tick.lock – übersprungen (laeuft_schon)")
            return TickErgebnis(f"{jetzt:%Y-%m-%dT%H:%M:%S}", [], 0, 0, 0, 0, None, "laeuft_schon")
        return lauf.laufen()
