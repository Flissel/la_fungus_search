"""Lokaler Zustand des Doku-Loops (Task 15, Rulings 44/45): Stand, Sperre, Marker, Temp-Reste, Urteils-Cache.

Alles hier liegt AUSSERHALB des Repos im Marker-Ordner (`%LOCALAPPDATA%/VibeMind/doku_loop`), damit der Loop zwischen
seinen Commits keinen Dreck im Arbeitsbaum hinterlaesst:

- `stand-<hash>.json`  letzter verarbeiteter Stand je Repo (Hash des Repo-Pfads, damit Worktrees sich nicht stoeren)
- `tick.lock`          Prozess-Sperre (exklusiv angelegt, nach 30 min veraltet, Uebernahme per Umbenennen)
- `anstoss`            Marker aus dem post-commit-Hook
- `urteile.jsonl`      Urteils-Cache: (aussage_id, Beleg, alter Digest) -> (neuer Digest, Urteil); ein Tick, der
                       nach dem Judge scheitert, fragt fuer dasselbe Signal nie wieder das LLM
- `graph-voll-<hash>.json`  voller Graph je Repo (Ruling 49): Basis inkrementeller Laeufe, fuer Rendern, Status und
                       Messen; im Repo liegt nur `schlank(voll)`. Fehlt er, rekonstruiert der Tick ihn aus dem
                       schlanken Repo-Graph und baut alle Spaces neu (kein Fehler). Wird nie committet.
                       Er ist an den Stand gebunden (Fix-Runde 1): seine Kennung `<HEAD aussen>:<HEAD Submodul>:<Lauf>`
                       steht in seinen Metadaten und als `vollgraph_kennung` im Stand; passt sie nicht (alte Datei,
                       Schreibfehler, anderer Stand), gilt er als fehlend. Gebaut wird er nur aus dem Messbaum
                       (Ruling 50, `messbaum.py`), nie aus dem Arbeitsbaum.
- `messbaum-<hash>/`   detached Worktrees des committeten Stands (Ruling 50), `messbaum-<hash>.json` ihr Zustand
- `schreibt-<hash>.json`  Vermerk `schreibt_gerade` (Final-Fix I7b): Lauf, HEAD und sha256 je Datei, die der
                       Schreibvorgang gleich schreibt; nach Commit oder Ruecksetzen geloescht. Liegt er beim naechsten
                       Tick noch da, war der Lauf mittendrin abgebrochen (`schreibvorgang.halbschreibung_aufraeumen`).
                       Eine eigene Datei statt eines Felds im Stand: der Stand wird von den Schreibern als Ganzes neu
                       geschrieben und darf dabei den Vermerk eines laufenden Schreibvorgangs nicht verlieren.
- `ledger-fremd-<hash>.jsonl`  Final-Fix I11: Ledger-Eintraege, deren Schreibung eine andere Session committet hat
                       (`{"lauf", "aussage_id", "commit_fremd"}`); nie im Repo
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import secrets
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .graph import Graph, mit_schlank
from .konfig import Konfig

SPERRE_DATEI = "tick.lock"
SPERRE_VERALTET_S = 30 * 60
TEMP_VERALTET_S = 30 * 60
URTEILE_DATEI = "urteile.jsonl"
GRAPH_DIR = "docs/graph"


def lokaler_ordner(konfig: Konfig) -> Path:
    return konfig.marker.parent


def repo_schluessel(repo: Path) -> str:
    """Kurzer Hash des Repo-Pfads (gross/klein egal), z. B. fuer `stand-<hash>.json`."""
    return hashlib.sha256(str(Path(repo).resolve()).casefold().encode("utf-8")).hexdigest()[:12]


# ---------- Stand ----------

@dataclass
class Stand:
    letzter_commit_aussen: str | None = None
    letzter_commit_sub: str | None = None
    letzter_tick: str | None = None
    render_hashes: dict[str, str] = field(default_factory=dict)
    lauf_nr: int = 0
    hand_edit_gemeldet: dict[str, str] = field(default_factory=dict)  # Block -> gemeldeter Inhalts-Hash
    # Signale ohne Urteil (Budget, Timeout, ...): alles, was der Judge braucht, um ohne Neubau erneut zu fragen
    offene_signale: list[dict[str, object]] = field(default_factory=list)
    # Kennung des lokalen Vollgraphs, der zu diesem Stand passt (None = keiner -> Rekonstruktion)
    vollgraph_kennung: str | None = None
    # 15c Fix-Runde 1: HEAD je initialisiertem Submodul (repo-relativ, rekursiv), damit Commits in verschachtelten
    # Submodulen (z. B. Space-Wurzel `spaces/sales-claw`) einen Neubau ausloesen, auch ohne neuen Pin
    submodul_commits: dict[str, str] = field(default_factory=dict)
    # Ruling 55a: erste Lesung eines geaenderten `gemessen.wert` je Hop (gilt erst bei der zweiten gleichen Lesung)
    gemessen_ausstehend: dict[str, str] = field(default_factory=dict)
    # T3-b: aeusserer HEAD (`""`) und Submodul-Koepfe, die der letzte Lauf gesehen hat -- auch wenn er den Commit-Zeiger
    # nicht fortgeschrieben hat. D6 vergleicht nur damit (sonst loeste ein haengender Zeiger alle 5 min einen Tick aus).
    # Additiv: der alte Loop liest den Stand mit `d.get` und ignoriert das Feld; fehlt es, gelten die Zeiger.
    koepfe_gesehen: dict[str, str] = field(default_factory=dict)
    # Stufe 4, Task 7 (D10): Fortschritt der Erstbefuellung -- Ids der einmal geprueften Aussagen und ob sie fertig ist.
    # Additiv wie `koepfe_gesehen`: der alte Loop liest nur seine Felder (`d.get`) und ignoriert diese beiden.
    erstgeprueft: list[str] = field(default_factory=list)
    erstbefuellung_fertig: bool = False

    @property
    def offene_spaces(self) -> list[str]:
        return sorted({str(s["space"]) for s in self.offene_signale if s.get("space")})

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "Stand":
        def text(k: str) -> str | None:
            v = d.get(k)
            return str(v) if v is not None else None

        def zuordnung(k: str) -> dict[str, str]:
            v = d.get(k)
            return {str(a): str(b) for a, b in v.items()} if isinstance(v, dict) else {}

        offen = d.get("offene_signale")
        nr = d.get("lauf_nr")
        erst = d.get("erstgeprueft")
        return cls(text("letzter_commit_aussen"), text("letzter_commit_sub"), text("letzter_tick"),
                   zuordnung("render_hashes"), int(nr) if isinstance(nr, int) else 0, zuordnung("hand_edit_gemeldet"),
                   [dict(s) for s in offen if isinstance(s, dict)] if isinstance(offen, list) else [],
                   text("vollgraph_kennung"), zuordnung("submodul_commits"), zuordnung("gemessen_ausstehend"),
                   zuordnung("koepfe_gesehen"),
                   [s for s in erst if isinstance(s, str)] if isinstance(erst, list) else [],
                   d.get("erstbefuellung_fertig") is True)


def stand_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"stand-{repo_schluessel(konfig.repo)}.json"


def stand_laden(konfig: Konfig) -> Stand | None:
    """Lokaler Stand dieses Repos; fehlt oder unlesbar -> None (alle Spaces gelten als geaendert)."""
    try:
        roh = json.loads(stand_pfad(konfig).read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return Stand.from_dict(roh) if isinstance(roh, dict) else None


def _atomar(pfad: Path, text: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(f"{pfad.name}.{os.getpid()}.tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(pfad)


def stand_schreiben(konfig: Konfig, stand: Stand) -> None:
    _atomar(stand_pfad(konfig), json.dumps(stand.to_dict(), sort_keys=True, indent=1, ensure_ascii=False) + "\n")


# ---------- voller Graph (Ruling 49) ----------

def vollgraph_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"graph-voll-{repo_schluessel(konfig.repo)}.json"


@dataclass(frozen=True)
class Vollgraph:
    graph: Graph
    kennung: str


def vollgraph_kennung(kopf_aussen: str, kopf_sub: str | None, lauf: str) -> str:
    return f"{kopf_aussen}:{kopf_sub or ''}:{lauf}"


def vollgraph_laden(konfig: Konfig, kennung: str | None) -> Vollgraph | None:
    """Voller lokaler Graph dieses Repos, nur wenn er zur Kennung im Stand passt; fehlt, unlesbar, kein Objekt oder
    andere Kennung -> None (dann Rekonstruktion)."""
    if not kennung:
        return None
    try:
        doc = json.loads(vollgraph_pfad(konfig).read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("kennung") != kennung:
        return None
    try:
        graph = Graph.aus_doc(doc)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return Vollgraph(graph, kennung)


def vollgraph_schreiben(konfig: Konfig, graph: Graph, erzeugt: str, *, kennung: str) -> None:
    """Atomar im lokalen Ordner, nie im Repo (`Graph.speichern` schreibt temp + replace; die Schreiber laufen unter
    der Sperre). Die Kennung muss danach in den Stand."""
    graph.speichern(vollgraph_pfad(konfig), erzeugt, zusatz={"kennung": kennung})


def vollgraph_verwerfen(konfig: Konfig) -> bool:
    """Nach einem Schreibfehler: alte Datei entfernen (best effort). True, wenn keine mehr da ist. Auch wenn das
    scheitert, passt ihre Kennung nicht mehr zum Stand (der Stand bekommt `None`)."""
    try:
        vollgraph_pfad(konfig).unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def arbeitsgraph(konfig: Konfig, schlank_repo: Graph) -> Graph:
    """Voller lokaler Graph (passend zum Stand) mit dem schlanken Repo-Graph als Wahrheit fuer Hops und Aussagen;
    sonst der schlanke (Rueckfall fuer `render`, `status`, `messen`)."""
    stand = stand_laden(konfig)
    voll = vollgraph_laden(konfig, stand.vollgraph_kennung if stand else None)
    return schlank_repo if voll is None else mit_schlank(voll.graph, schlank_repo)


# ---------- Anstoss ----------

def _zeitstempel(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def koepfe_weichen_ab(stand: Stand, koepfe: Mapping[str, str]) -> bool:
    """D6: `koepfe` (`""` = aeusserer HEAD, sonst repo-relativer Submodul-Pfad -> HEAD, wie `messbaum.Messbaum.commits`)
    weichen von dem ab, was der Stand zuletzt gesehen hat (`koepfe_gesehen`; fehlt das Feld: den Commit-Zeigern).
    Ein Submodul, das der Stand nicht kennt, zaehlt."""
    if stand.koepfe_gesehen:  # T3-b: nur gegen das, was der letzte Lauf gesehen hat
        return any(k and stand.koepfe_gesehen.get(rel) != k for rel, k in koepfe.items())
    aussen = koepfe.get("")
    if stand.letzter_commit_aussen is None:  # nie ein Commit verarbeitet (z. B. kein_seed): nur Intervall und Marker
        return False
    if aussen and aussen != stand.letzter_commit_aussen:
        return True
    return any(k and stand.submodul_commits.get(rel) != k for rel, k in koepfe.items() if rel)


def anstoss_noetig(konfig: Konfig, stand: Stand | None, jetzt: datetime, *,
                   koepfe: Mapping[str, str] | None = None) -> bool:
    """Marker da, oder letzter Tick aelter als `tick_minuten`, oder noch nie gelaufen -- oder (D6, ohne Git-Hook)
    `koepfe` weichen vom Stand ab. `koepfe=None` (Git nicht lesbar) verhaelt sich wie vorher."""
    try:
        if konfig.marker.exists():
            return True
    except OSError:
        pass
    if stand is None:
        return True
    if koepfe is not None and koepfe_weichen_ab(stand, koepfe):
        return True
    letzter = _zeitstempel(stand.letzter_tick)
    if letzter is None:
        return True
    return jetzt.timestamp() - letzter >= konfig.tick_minuten * 60


def marker_loeschen(konfig: Konfig) -> bool:
    """Marker entfernen; fehlender Marker oder Ordner stoert nicht. True, wenn einer da war."""
    try:
        konfig.marker.unlink()
        return True
    except OSError:
        return False


# ---------- Sperre ----------

def sperre_frisch(ordner: Path) -> bool:
    """Nur lesend: haelt gerade ein (anderer) Lauf `tick.lock`, d. h. die Datei ist juenger als die Veraltungsgrenze?
    Nie anlegen, nie uebernehmen, nie entfernen (Trockenlauf)."""
    try:
        return time.time() - (Path(ordner) / SPERRE_DATEI).stat().st_mtime <= SPERRE_VERALTET_S
    except OSError:
        return False


def _veraltet_uebernehmen(pfad: Path) -> bool:
    """Veraltete Sperre per Umbenennen wegnehmen; True = neu versuchen. Nur einer gewinnt das Umbenennen; wer dabei
    eine frische Sperre erwischt (ein anderer war schneller), legt sie zurueck."""
    try:
        alter = time.time() - pfad.stat().st_mtime
    except FileNotFoundError:
        return True
    except OSError:  # z. B. PermissionError waehrend ein anderer loescht
        return False
    if alter <= SPERRE_VERALTET_S:
        return False
    weg = pfad.with_name(f"{pfad.name}.{os.getpid()}.{secrets.token_hex(4)}.alt")
    try:
        os.rename(pfad, weg)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        frisch = time.time() - weg.stat().st_mtime <= SPERRE_VERALTET_S
    except OSError:
        frisch = False
    if frisch:
        try:
            os.rename(weg, pfad)
        except OSError:
            pass
        return False
    try:
        weg.unlink()
    except OSError:
        pass
    return True


@contextmanager
def sperre(ordner: Path) -> Iterator[bool]:
    """Prozess-Sperre `<ordner>/tick.lock`. Liefert False, wenn ein anderer Lauf sie haelt (die fremde Sperre bleibt
    unberuehrt); eine Sperre aelter als 30 min wird uebernommen."""
    ordner = Path(ordner)
    pfad = ordner / SPERRE_DATEI
    ordner.mkdir(parents=True, exist_ok=True)
    inhalt = f"{os.getpid()} {time.time():.3f} {secrets.token_hex(4)}"
    for versuch in range(2):
        try:
            fd = os.open(pfad, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            if versuch == 0 and _veraltet_uebernehmen(pfad):
                continue
            yield False
            return
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(inhalt)
        break
    else:  # pragma: no cover - zweimal verloren
        yield False
        return
    try:
        yield True
    finally:
        try:
            if pfad.read_text(encoding="utf-8") == inhalt:
                pfad.unlink()
        except OSError:
            pass


def sperre_zustand(ordner: Path) -> str:
    """`frei`, `belegt` oder `veraltet (seit N min)` (fuer `status`)."""
    try:
        alter = time.time() - (Path(ordner) / SPERRE_DATEI).stat().st_mtime
    except FileNotFoundError:
        return "frei"
    except OSError:
        return "belegt"
    return f"veraltet (seit {int(alter // 60)} min)" if alter > SPERRE_VERALTET_S else "belegt"


# ---------- Temp-Reste ----------

def temp_aufraeumen(konfig: Konfig, *, jetzt_ts: float | None = None) -> list[str]:
    """Eigene Temp-Reste von `umschreiben.schreiben` (`.<name>.<zufall>.tmp`) in docs/graph und neben KETTEN.md sowie
    des lokalen Vollgraphs (`graph-voll-<hash>.json.tmp` im lokalen Ordner) entfernen, wenn aelter als 30 min. Andere
    Dateien werden nie angefasst."""
    jetzt_ts = time.time() if jetzt_ts is None else jetzt_ts
    md = Path(konfig.ketten_md)
    muster: list[tuple[Path, re.Pattern[str]]] = [
        (konfig.repo / GRAPH_DIR, re.compile(r"^\.(?:graph\.json|ledger\.jsonl)\.[A-Za-z0-9_]+\.tmp$")),
        (konfig.repo / md.parent, re.compile(rf"^\.{re.escape(md.name)}\.[A-Za-z0-9_]+\.tmp$")),
        (lokaler_ordner(konfig), re.compile(r"^graph-voll-[0-9a-f]{12}\.json\.tmp$")),
    ]
    weg: list[str] = []
    for ordner, regel in muster:
        try:
            eintraege = list(ordner.iterdir())
        except OSError:
            continue
        for p in eintraege:
            try:
                if regel.match(p.name) and p.is_file() and jetzt_ts - p.stat().st_mtime > TEMP_VERALTET_S:
                    p.unlink()
                    weg.append(p.name)
            except OSError:
                continue
    return weg


# ---------- Schreibvermerk (I7b) und fremd uebernommene Commits (I11) ----------

def schreibvermerk_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"schreibt-{repo_schluessel(konfig.repo)}.json"


def ledger_fremd_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"ledger-fremd-{repo_schluessel(konfig.repo)}.jsonl"


def fremd_commits_merken(konfig: Konfig, eintraege: Sequence[object], sha: str) -> None:
    """Je Ledger-Eintrag (`lauf`, `aussage_id`) den fremden Commit vermerken, der seine Schreibung enthaelt."""
    zeilen = [json.dumps({"lauf": getattr(e, "lauf", None), "aussage_id": getattr(e, "aussage_id", None),
                          "commit_fremd": sha}, ensure_ascii=False, sort_keys=True) for e in eintraege]
    if not zeilen:
        return
    pfad = ledger_fremd_pfad(konfig)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "a", encoding="utf-8") as f:
        f.write("".join(z + "\n" for z in zeilen))


def fremd_commits_laden(konfig: Konfig) -> dict[str, str]:
    """Lauf -> sha des fremden Commits (der letzte Vermerk gewinnt); fehlt/unlesbar -> leer."""
    try:
        zeilen = ledger_fremd_pfad(konfig).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    aus: dict[str, str] = {}
    for z in zeilen:
        try:
            e = json.loads(z)
        except ValueError:
            continue
        if isinstance(e, dict) and isinstance(e.get("lauf"), str) and isinstance(e.get("commit_fremd"), str):
            aus[e["lauf"]] = e["commit_fremd"]
    return aus


# ---------- Urteils-Cache ----------

def _urteil_schluessel(aussage_id: str, art: str, datei: str, anker: str, alt_digest: str,
                       fragetyp: str = "signal") -> str:
    """Schluessel im Urteils-Cache. Signale (wie im alten Loop, byte-gleich): `[aussage_id, art, datei, anker,
    alt_digest]`. Task 7: andere Fragetypen (`erstpruefung`) stehen vorn im Schluessel -- ein Erst-Urteil (alt = neu =
    heutiger Digest) und ein Signal-Urteil teilen sich nie einen Eintrag."""
    teile = [aussage_id, art, datei, anker, alt_digest]
    return json.dumps(teile if fragetyp == "signal" else [fragetyp, *teile], ensure_ascii=False)


def urteile_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / URTEILE_DATEI


def urteil_holen(konfig: Konfig, aussage_id: str, beleg: Sequence[str], alt_digest: str,
                 neu_digest: str, *, fragetyp: str = "signal") -> dict[str, object] | None:
    """Gespeichertes Urteil, wenn es fuer genau diesen Code-Stand (`neu_digest`) gilt; sonst None."""
    schluessel = _urteil_schluessel(aussage_id, *beleg, alt_digest, fragetyp=fragetyp)
    treffer: dict[str, object] | None = None
    try:
        zeilen = urteile_pfad(konfig).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for z in zeilen:
        try:
            e = json.loads(z)
        except ValueError:
            continue
        if isinstance(e, dict) and e.get("schluessel") == schluessel:
            treffer = e  # der letzte Eintrag gewinnt
    if treffer is None or treffer.get("neu_digest") != neu_digest:
        return None
    return treffer


URTEILE_MAX_BYTES = 2 * 1024 * 1024


def urteile_kuerzen(konfig: Konfig, *, max_bytes: int = URTEILE_MAX_BYTES) -> int:
    """Minor 13: `urteile.jsonl` begrenzen. Rueckgabe: Zahl der verworfenen Zeilen (0 = nichts zu tun).

    Regel, nur wenn die Datei groesser als `max_bytes` ist:
    1. Je Schluessel zaehlt nur der letzte Eintrag (`urteil_holen` liest ohnehin nur ihn); aeltere Eintraege desselben
       Schluessels und unlesbare Zeilen fallen weg.
    2. Von den uebrigen wird die aeltere Haelfte verworfen -- ausser Eintraegen mit einem Urteil (`urteil` gesetzt):
       sie sparen einen Judge-Aufruf, sobald derselbe Code-Stand wiederkommt, und werden also noch gebraucht.
       Verworfen werden dort Fehlschlaege ohne Urteil (`grund`: uneinig, ungueltig, ...); kommt ihr Signal wieder,
       wird neu gefragt.
    3. Die juengere Haelfte bleibt ganz. Geschrieben wird atomar (temp + replace)."""
    pfad = urteile_pfad(konfig)
    try:
        if pfad.stat().st_size <= max_bytes:
            return 0
        zeilen = pfad.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return 0
    letzte: dict[str, tuple[int, dict[str, object]]] = {}
    for i, z in enumerate(zeilen):
        try:
            e = json.loads(z)
        except ValueError:
            continue
        if isinstance(e, dict) and isinstance(e.get("schluessel"), str):
            letzte[e["schluessel"]] = (i, e)
    reihe = [e for _, e in sorted(letzte.values(), key=lambda x: x[0])]
    haelfte = len(reihe) // 2
    behalten = [e for e in reihe[:haelfte] if e.get("urteil")] + reihe[haelfte:]
    _atomar(pfad, "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in behalten))
    return len(zeilen) - len(behalten)


def urteil_merken(konfig: Konfig, aussage_id: str, beleg: Sequence[str], alt_digest: str, neu_digest: str,
                  eintrag: Mapping[str, object], *, fragetyp: str = "signal") -> None:
    daten = {**eintrag, "schluessel": _urteil_schluessel(aussage_id, *beleg, alt_digest, fragetyp=fragetyp),
             "neu_digest": neu_digest}
    pfad = urteile_pfad(konfig)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "a", encoding="utf-8") as f:
        f.write(json.dumps(daten, ensure_ascii=False, sort_keys=True) + "\n")
