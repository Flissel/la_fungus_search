"""Gemeinsamer Schreibvorgang von tick, seed und render (Task 15, Fix-Runde 1, Item 13).

`schreiben_pruefen_committen`: Dateien mit Compare-and-Swap schreiben (Ledger eingeschlossen) -> Torwaechter
nachher -> pathspec-begrenzter Commit nur der geschriebenen Dateien (`erwartet` nur fuer *.md, aus den geschriebenen
Bytes). Jeder Fehler vor dem Commit setzt alle geschriebenen Dateien mit CAS zurueck (neu angelegte werden nur
geloescht, wenn sie noch genau den geschriebenen Inhalt haben), so bleibt nichts Uncommittetes vom Loop liegen.
`CommitUnbestaetigt` bzw. eine Abweichung NACH dem eigenen Commit setzt nichts zurueck und wird als Pruefpunkt
gemeldet.

Ruling 46: Ledger-Eintraege liegen im Loop-Commit selbst und kennen dessen sha nicht. Sie tragen `basis_commit` und
`lauf`; die Commit-Nachricht hat die Zeile `Lauf: <lauf>`, `ledger_commit_von` findet den Commit daraus.

Hier liegen auch die Git-Helfer des Loops (HEAD, Submodul, versionierte Dateien, fremde Aenderungen).

Final-Fix C1 (Ruling 57): geschrieben und committet wird nur auf einem Branch ohne laufende Git-Operation
(`kopf_pruefen`): detached HEAD -> `kopf_detached`; `rebase-merge`/`rebase-apply` im Git-Verzeichnis ->
`rebase_laeuft`; `MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD` oder `BISECT_LOG` -> `merge_laeuft`; mit
`zweig:` in der Konfig ein anderer Branch -> `falscher_zweig`. Die Einstiege (tick, seed, render --schreiben,
graph-schlank) pruefen vor jeder Arbeit, `schreiben_pruefen_committen` noch einmal unmittelbar vor dem Schreiben.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import umschreiben
from .graph import Graph
from .konfig import Konfig
from .messbaum import MessbaumFehler, submodule_finden
from .stand import schreibvermerk_pfad  # noqa: F401 - Wiederexport: der Vermerk gehoert zum Schreibvorgang (I7b)
from .typen import Pruefpunkt

GRAPH_DIR = "docs/graph"
GRAPH_JSON = "docs/graph/graph.json"
LEDGER = "docs/graph/ledger.jsonl"
SUBMODUL = "vibemind-os"
GIT_TIMEOUT_S = 60

Melden = Callable[[str], None]
Drift = Callable[[Path], "set[umschreiben.DriftFehler]"]


# ---------- Git ----------

def _git_lauf(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(["git", "-c", "core.quotepath=false", "-C", str(repo), *args],
                              capture_output=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None


def git(repo: Path, *args: str) -> str | None:
    lauf = _git_lauf(repo, *args)
    if lauf is None or lauf.returncode != 0:
        return None
    return lauf.stdout.decode("utf-8", errors="replace")


# Marker laufender Git-Operationen im Git-Verzeichnis (per `git rev-parse --git-path`, auch in Worktrees)
LAUFENDE_OPERATIONEN = (("rebase-merge", "rebase_laeuft"), ("rebase-apply", "rebase_laeuft"),
                        ("MERGE_HEAD", "merge_laeuft"), ("CHERRY_PICK_HEAD", "merge_laeuft"),
                        ("REVERT_HEAD", "merge_laeuft"), ("BISECT_LOG", "merge_laeuft"))


def kopf_pruefen(repo: Path, zweig: str | None = None) -> str | None:
    """None, wenn der Loop hier schreiben und committen darf; sonst der Grund (siehe Moduldoku). Git-Fehler ->
    `git_fehler:<befehl>` (fail-closed)."""
    repo = Path(repo)
    args: list[str] = []
    for name, _ in LAUFENDE_OPERATIONEN:
        args += ["--git-path", name]
    aus = git(repo, "rev-parse", *args)
    pfade = aus.splitlines() if aus is not None else []
    if len(pfade) != len(LAUFENDE_OPERATIONEN):
        return "git_fehler:rev-parse"
    for (_, grund), roh in zip(LAUFENDE_OPERATIONEN, pfade):
        pfad = Path(roh.strip())
        if (pfad if pfad.is_absolute() else repo / pfad).exists():
            return grund
    lauf = _git_lauf(repo, "symbolic-ref", "-q", "HEAD")
    if lauf is None or lauf.returncode not in (0, 1):
        return "git_fehler:symbolic-ref"
    if lauf.returncode == 1:
        return "kopf_detached"
    ref = lauf.stdout.decode("utf-8", errors="replace").strip()
    if zweig and ref.removeprefix("refs/heads/") != zweig:
        return "falscher_zweig"
    return None


def kopf(repo: Path) -> str | None:
    aus = git(repo, "rev-parse", "--verify", "-q", "HEAD")
    return (aus.strip() or None) if aus is not None else None


def hat_submodul(repo: Path) -> bool:
    # Ohne eigenes .git liefe `git -C vibemind-os` ins aeussere Repo hoch.
    return (Path(repo) / SUBMODUL / ".git").exists()


def submodul_kopf(repo: Path) -> str | None:
    return kopf(Path(repo) / SUBMODUL) if hat_submodul(repo) else None


class GitFehler(umschreiben.GitFehler):
    """Final-Fix I3: eine Git-Abfrage, aus der eine Menge wird (versioniert, unsauber), ist gescheitert. Nie eine leere
    Menge daraus machen (fail-closed); der Tick ueberspringt mit `git_fehler:<befehl>`."""

    def __init__(self, befehl: str) -> None:
        super().__init__(f"git {befehl} gescheitert")
        self.befehl = befehl


def git_pflicht(repo: Path, *args: str) -> str:
    """Wie `git`, aber ein Fehler (Exit-Code, Timeout, nicht startbar) wirft `GitFehler(<befehl>)`."""
    aus = git(repo, *args)
    if aus is None:
        raise GitFehler(args[0] if args else "git")
    return aus


def _alle_submodule(repo: Path) -> dict[str, Path]:
    """Repo-relativer Pfad -> echter Checkout jedes initialisierten Submoduls (rekursiv, Ruling 50)."""
    try:
        return {rel: pfad for rel, (pfad, _) in submodule_finden(Path(repo))[0].items()}
    except MessbaumFehler:
        return {SUBMODUL: Path(repo) / SUBMODUL} if hat_submodul(repo) else {}


def versionierte_dateien(repo: Path) -> frozenset[str]:
    """Alle von git verfolgten Dateien (aussen + jedes initialisierte Submodul, rekursiv, mit Pfad-Praefix); nur sie
    taugen als Beleg. Ein git-Fehler wirft `GitFehler` (nie eine leere Menge, I3)."""
    def liste(wo: Path, praefix: str) -> set[str]:
        aus = git_pflicht(wo, "ls-files", "-z")
        return {praefix + p for p in aus.split("\0") if p}

    dateien = liste(Path(repo), "")
    for rel, pfad in _alle_submodule(repo).items():  # Ruling 50: auch verschachtelte (sie liegen im Messbaum)
        dateien |= liste(pfad, rel + "/")
    return frozenset(dateien)


def _porcelain_pfade(ausgabe: str) -> set[str]:
    """Pfade aus `git status --porcelain=v1 -z` (bei Umbenennungen beide)."""
    tokens = ausgabe.split("\0")
    pfade: set[str] = set()
    i = 0
    while i < len(tokens):
        eintrag = tokens[i]
        i += 1
        if len(eintrag) < 4:
            continue
        pfade.add(eintrag[3:])
        if ("R" in eintrag[:2] or "C" in eintrag[:2]) and i < len(tokens):
            if tokens[i]:
                pfade.add(tokens[i])
            i += 1
    return pfade


def unsaubere_dateien(repo: Path) -> frozenset[str]:
    """Dateien mit nicht committeten Aenderungen (gestaged, ungestaged, untracked; aussen + jedes initialisierte
    Submodul, rekursiv, mit Praefix -- I10). Ruling 48a: Belege auf solche Dateien geben in diesem Lauf kein Signal und
    werden nicht nachgefuehrt. Ein git-Fehler wirft `GitFehler` (nie eine leere Menge, I3)."""
    def liste(wo: Path, praefix: str) -> set[str]:
        aus = git_pflicht(wo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=all")
        return {praefix + p for p in _porcelain_pfade(aus)}

    pfade = liste(Path(repo), "")
    for rel, pfad in _alle_submodule(repo).items():
        pfade |= liste(pfad, rel + "/")
    return frozenset(pfade)


def eigene_datei(konfig: Konfig, datei: str) -> bool:
    """Ruling 48b: Dateien, die der Loop selbst schreibt (KETTEN.md, Doku-Pfade, alles unter docs/graph), sind nie
    Belegziel -- ein Loop-Commit darf kein Signal fuer sich selbst erzeugen."""
    return datei in doku_dateien(konfig) or datei == GRAPH_DIR or datei.startswith(GRAPH_DIR + "/")


def doku_dateien(konfig: Konfig) -> list[str]:
    return list(dict.fromkeys([konfig.ketten_md, *konfig.weitere_doku]))


def fremde_aenderungen(konfig: Konfig) -> list[str]:
    """Nicht committete Aenderungen an KETTEN.md, weiterer Doku oder irgendwo in docs/graph (Ruling 44: keine
    Ausnahme mehr; der Loop hat dort zwischen seinen Commits keine eigenen offenen Dateien)."""
    return umschreiben.fremde_aenderungen(konfig.repo, [*doku_dateien(konfig), GRAPH_DIR])


def graph_text(graph: Graph, erzeugt: str) -> str:
    with tempfile.TemporaryDirectory(prefix="doku-loop-") as tmp:
        pfad = Path(tmp) / "graph.json"
        graph.speichern(pfad, erzeugt)
        return pfad.read_bytes().decode("utf-8")


def ledger_commit_von(repo: Path, lauf: str) -> str | None:
    """sha des Loop-Commits mit der Zeile `Lauf: <lauf>` (Ruling 46); None, wenn keiner da ist."""
    aus = git(repo, "log", "-F", f"--grep=Lauf: {lauf}", "--format=%H")
    zeilen = [z.strip() for z in (aus or "").splitlines() if z.strip()]
    return zeilen[0] if zeilen else None


def nachricht(betreff: str, lauf: str) -> str:
    return f"{betreff}\n\nLauf: {lauf}\n"


# ---------- Schreibvorgang ----------

@dataclass(frozen=True)
class Aenderung:
    datei: str  # repo-relativ
    alt: bytes | None  # gelesene Bytes (None = Datei gab es nicht)
    neu: str


@dataclass(frozen=True)
class Ausgang:
    commit: str | None
    uebersprungen: str | None  # None = committet
    fortschreiben: bool  # Commit-Zeiger im Stand fortschreiben? (nein = naechster Lauf holt nach)
    pruefpunkte: tuple[Pruefpunkt, ...] = ()
    dateien: tuple[str, ...] = ()
    fremd_commit: str | None = None  # I11: eine andere Session hat genau die geschriebenen Bytes committet


def _ledger_text(alt: bytes | None, eintraege: Sequence[umschreiben.LedgerEintrag | Mapping[str, object]]) -> str:
    text = (alt or b"").decode("utf-8")
    if text and not text.endswith("\n"):
        text += "\n"  # eine abgebrochene Vorzeile nicht mit dem neuen Eintrag verschmelzen
    for e in eintraege:
        daten = e.to_dict() if isinstance(e, umschreiben.LedgerEintrag) else dict(e)
        text += json.dumps(daten, ensure_ascii=False, sort_keys=True) + "\n"
    return text


def zuruecksetzen(repo: Path, geschrieben: Mapping[str, tuple[bytes | None, bytes]], melden: Melden) -> None:
    """Geschriebene Dateien mit CAS auf den alten Stand bringen; was inzwischen jemand anderes geaendert hat, bleibt."""
    for datei, (alt, neu) in geschrieben.items():
        pfad = repo / datei
        try:
            if alt is None:
                if pfad.is_file() and pfad.read_bytes() == neu:
                    pfad.unlink()
                elif pfad.exists():
                    melden(f"Zurücksetzen: {datei} inzwischen fremd geändert – bleibt")
            else:
                umschreiben.schreiben(repo, datei, alt.decode("utf-8"), erwartet_alt=neu)
        except umschreiben.FremdeAenderung:
            melden(f"Zurücksetzen: {datei} inzwischen fremd geändert – bleibt")
        except (OSError, ValueError) as fehler:
            melden(f"Zurücksetzen: {datei} gescheitert ({type(fehler).__name__})")


def _sha256(daten: bytes) -> str:
    return hashlib.sha256(daten).hexdigest()


def schreiben_pruefen_committen(
    repo: Path, aenderungen: Sequence[Aenderung], betreff: str, lauf: str, *, vorher: "set[umschreiben.DriftFehler]",
    drift: Drift, ledger: Sequence[umschreiben.LedgerEintrag | Mapping[str, object]] = (), melden: Melden,
    pp_datei: str | None = None, zweig: str | None = None, vermerk: Path | None = None,
) -> Ausgang:
    """Siehe Moduldoku. `vorher` = Torwaechter-Ergebnis vor jedem Schreiben (und vor jedem Judge-Aufruf).
    `zweig`: nur auf diesem Branch schreiben (C1, aus der Konfig). `vermerk`: Pfad des Vermerks `schreibt_gerade`
    (I7b; `schreibvermerk_pfad(konfig)`): vor dem ersten Schreiben angelegt, nach Commit oder Ruecksetzen geloescht --
    nur ein abgebrochener Lauf hinterlaesst ihn."""
    repo = Path(repo)
    grund = kopf_pruefen(repo, zweig)
    if grund is not None:  # C1: nie auf detached HEAD oder in eine laufende Git-Operation hinein
        melden(f"Kopf: {grund} – nichts geschrieben, kein Commit")
        return Ausgang(None, grund, False)
    alle = list(aenderungen)
    if ledger:
        pfad = repo / LEDGER
        alt = pfad.read_bytes() if pfad.is_file() else None
        alle.append(Aenderung(LEDGER, alt, _ledger_text(alt, ledger)))
    kopf_start = kopf(repo)  # I11: Bezug fuer "HEAD ist inzwischen gewandert"
    if vermerk is not None:
        try:
            _vermerk_schreiben(vermerk, lauf, kopf_start, {a.datei: _sha256(a.neu.encode("utf-8")) for a in alle})
        except OSError as fehler:
            melden(f"Schreibvermerk: {type(fehler).__name__} – nichts geschrieben")
            return Ausgang(None, f"schreibvermerk:{type(fehler).__name__}", False)
    aus = _schreiben_committen(repo, alle, betreff, lauf, vorher=vorher, drift=drift, melden=melden,
                               pp_datei=pp_datei, kopf_start=kopf_start)
    if vermerk is not None:  # Commit oder Ruecksetzen ist durch; eine Ausnahme oben laesst den Vermerk stehen
        _vermerk_loeschen(vermerk)
    return aus


def _vermerk_schreiben(vermerk: Path, lauf: str, kopf_start: str | None, dateien: Mapping[str, str]) -> None:
    vermerk.parent.mkdir(parents=True, exist_ok=True)
    tmp = vermerk.with_name(vermerk.name + ".tmp")
    tmp.write_bytes(json.dumps({"lauf": lauf, "kopf": kopf_start, "dateien": dict(dateien)}, ensure_ascii=False,
                               sort_keys=True).encode("utf-8"))
    tmp.replace(vermerk)


def _vermerk_loeschen(vermerk: Path) -> None:
    try:
        vermerk.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass  # bleibt stehen: der naechste Tick findet saubere Dateien und raeumt ihn weg


def _schreiben_committen(repo: Path, alle: Sequence[Aenderung], betreff: str, lauf: str, *,
                         vorher: "set[umschreiben.DriftFehler]", drift: Drift, melden: Melden, pp_datei: str | None,
                         kopf_start: str | None) -> Ausgang:
    geschrieben: dict[str, tuple[bytes | None, bytes]] = {}
    try:
        for a in alle:
            umschreiben.schreiben(repo, a.datei, a.neu, erwartet_alt=a.alt if a.alt is not None else b"")
            geschrieben[a.datei] = (a.alt, a.neu.encode("utf-8"))
    except (OSError, umschreiben.FremdeAenderung, ValueError) as fehler:
        zuruecksetzen(repo, geschrieben, melden)
        melden(f"Schreiben: {type(fehler).__name__} – zurückgesetzt, übersprungen")
        return Ausgang(None, f"schreiben:{type(fehler).__name__}", False)

    try:
        nachher = drift(repo)
    except umschreiben.TorwaechterFehler as fehler:
        zuruecksetzen(repo, geschrieben, melden)
        melden("Torwächter: Fehler – zurückgesetzt, kein Commit")
        return Ausgang(None, "torwaechter:torwaechter_fehler", False,
                       (Pruefpunkt("torwaechter_fehler", f"Waechter nachher: {fehler}"[:500], pp_datei, None, lauf),))
    if not umschreiben.torwaechter_ok(vorher, nachher):
        neu = sorted(nachher - vorher)
        zuruecksetzen(repo, geschrieben, melden)
        melden(f"Torwächter: {len(neu)} neue Fehler – zurückgesetzt, kein Commit")
        text = f"{len(neu)} neue Drift-Fehler: " + "; ".join(f"{d}: {m}" for d, m in neu[:5])
        return Ausgang(None, "torwaechter:torwaechter_blockiert", True,
                       (Pruefpunkt("torwaechter_blockiert", text[:500], pp_datei, None, lauf),))
    melden("Torwächter: keine neuen Fehler")

    dateien = list(geschrieben)
    erwartet = {d: _sha256(neu) for d, (_, neu) in geschrieben.items() if d.endswith(".md")}
    kopf_vorher = kopf(repo)
    try:
        sha = umschreiben.committen(repo, dateien, nachricht(betreff, lauf), erwartet=erwartet)
    except umschreiben.CommitUnbestaetigt as fehler:
        return _unbestaetigt(str(fehler), fehler.sha, lauf, pp_datei, dateien, melden)
    except (umschreiben.GitFehler, umschreiben.FremdeAenderung, ValueError, OSError) as fehler:
        jetzt = kopf(repo)
        if (isinstance(fehler, umschreiben.FremdeAenderung) and jetzt and jetzt != kopf_vorher
                and jetzt[:7] in str(fehler)):  # Abweichung NACH dem eigenen Commit
            return _unbestaetigt(str(fehler), jetzt, lauf, pp_datei, dateien, melden)
        zuruecksetzen(repo, geschrieben, melden)
        melden(f"Commit: {type(fehler).__name__} – zurückgesetzt, übersprungen")
        return Ausgang(None, f"commit:{type(fehler).__name__}", False)
    if sha is None:
        fremd = _fremd_uebernommen(repo, geschrieben, kopf_start, dateien)
        if fremd is not None:
            melden(f"Commit: eine andere Session hat die geschriebenen Dateien committet ({fremd[:7]}) – übernommen")
            text = f"{fremd[:7]}: eine andere Session hat {', '.join(dateien)} mit genau den Bytes des Loops committet"
            return Ausgang(None, "commit_fremd_uebernommen", True,
                           (Pruefpunkt("commit_fremd_uebernommen", text[:500], pp_datei, None, lauf),),
                           tuple(dateien), fremd)
    melden(f"Commit: {(sha or '-')[:7]} ({len(dateien)} Datei(en))")
    return Ausgang(sha, None, True, (), tuple(dateien))


def _fremd_uebernommen(repo: Path, geschrieben: Mapping[str, tuple[bytes | None, bytes]], kopf_start: str | None,
                       dateien: Sequence[str]) -> str | None:
    """I11: `committen` fand nichts zu committen. HEAD ist seit dem Schreiben gewandert und der Arbeitsbaum hat genau
    die geschriebenen Bytes -> sha des fremden Commits (letzter Commit dieser Dateien), sonst None."""
    jetzt = kopf(repo)
    if not jetzt or jetzt == kopf_start:
        return None
    for datei, (_, neu) in geschrieben.items():
        try:
            if (repo / datei).read_bytes() != neu:
                return None
        except OSError:
            return None
    aus = git(repo, "log", "-1", "--format=%H", "--", *dateien)
    return (aus or "").strip() or jetzt


@dataclass(frozen=True)
class Halbschreibung:
    zustand: str  # keiner | sauber | unlesbar | fremd | zurueckgesetzt
    lauf: str | None = None
    dateien: tuple[str, ...] = ()


def halbschreibung_aufraeumen(repo: Path, vermerk: Path, melden: Melden) -> Halbschreibung:
    """I7b: steht noch ein Vermerk `schreibt_gerade`, ist ein Lauf zwischen Schreiben und Commit/Ruecksetzen
    abgebrochen (z. B. Laufzeitgrenze der Aufgabe). Haben alle Dateien mit Aenderungen noch genau die vermerkten Bytes,
    ist es die eigene Halbschreibung: eigenes Staging zuruecknehmen und jede Datei mit CAS auf den HEAD-Stand bringen
    (`git cat-file --filters`, also wie ein Checkout; nicht in HEAD -> loeschen, solange noch genau die Bytes). Weicht
    eine ab, gehoert der Stand inzwischen jemand anderem (`fremd`): nichts anfassen, der Vermerk bleibt, der Tick
    ueberspringt wie bei jeder fremden Aenderung. Ohne Aenderungen (`sauber`) oder unlesbar wird er geloescht."""
    repo = Path(repo)
    try:
        roh = json.loads(Path(vermerk).read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return Halbschreibung("keiner")
    except (OSError, ValueError):
        _vermerk_loeschen(vermerk)
        return Halbschreibung("unlesbar")
    dateien = roh.get("dateien") if isinstance(roh, dict) else None
    lauf = roh.get("lauf") if isinstance(roh, dict) and isinstance(roh.get("lauf"), str) else None
    if not isinstance(dateien, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in dateien.items()):
        _vermerk_loeschen(vermerk)
        return Halbschreibung("unlesbar", lauf)
    offen = umschreiben.fremde_aenderungen(repo, list(dateien))
    if not offen:
        _vermerk_loeschen(vermerk)
        return Halbschreibung("sauber", lauf)
    eigene: dict[str, bytes] = {}
    for datei in offen:
        try:
            daten = (repo / datei).read_bytes()
        except OSError:
            return Halbschreibung("fremd", lauf, tuple(offen))
        if datei not in dateien or _sha256(daten) != dateien[datei]:
            return Halbschreibung("fremd", lauf, tuple(offen))
        eigene[datei] = daten
    for datei in eigene:  # eigenes Staging (abgebrochen nach `git add`) zuruecknehmen; nichts gestaged -> egal
        git(repo, "restore", "--staged", "--", datei)
    try:
        for datei, daten in eigene.items():
            kopf_inhalt = _git_lauf(repo, "cat-file", "--filters", f"HEAD:{datei}")
            if kopf_inhalt is None:
                raise GitFehler("cat-file")
            if kopf_inhalt.returncode == 0:
                umschreiben.schreiben(repo, datei, kopf_inhalt.stdout.decode("utf-8"), erwartet_alt=daten)
                continue
            vorhanden = _git_lauf(repo, "cat-file", "-e", f"HEAD:{datei}")
            if vorhanden is None or vorhanden.returncode == 0:
                raise GitFehler("cat-file")
            pfad = repo / datei  # in HEAD nicht vorhanden: neu angelegt -> loeschen, solange genau unsere Bytes
            if pfad.read_bytes() == daten:
                pfad.unlink()
    except (umschreiben.FremdeAenderung, OSError, ValueError):
        melden("Halbschreibung: Datei inzwischen fremd geändert – bleibt, Vermerk bleibt")
        return Halbschreibung("fremd", lauf, tuple(offen))
    _vermerk_loeschen(vermerk)
    melden(f"Halbschreibung: {len(eigene)} eigene Datei(en) aus Lauf {lauf or '?'} auf HEAD zurückgesetzt")
    return Halbschreibung("zurueckgesetzt", lauf, tuple(sorted(eigene)))


def _unbestaetigt(grund: str, sha: str, lauf: str, pp_datei: str | None, dateien: Sequence[str],
                  melden: Melden) -> Ausgang:
    """Handover 11 + Item 4: nichts zuruecksetzen; der Vermerk geht in die (lokale) Pruefliste statt als
    uncommitteter Ledger-Eintrag ins Repo."""
    melden(f"Commit unbestätigt ({sha[:7]}) – nichts zurückgesetzt, Prüfliste-Eintrag")
    return Ausgang(sha, "commit_unbestaetigt", True,
                   (Pruefpunkt("commit_unbestaetigt", f"{sha[:7]}: {grund}"[:500], pp_datei, None, lauf),),
                   tuple(dateien))
