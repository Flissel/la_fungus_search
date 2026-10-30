"""Umschreiben: Ledger, fremde Aenderungen, Torwaechter, Schreiben, Commit und Revert (Task 14).

Regeln (C.3):
- Der Loop schreibt nur nach `fremde_aenderungen(...) == []` auf seine Zieldateien.
- Schreiben ist atomar (Temp-Datei im selben Ordner + `os.replace`), byteweise, ohne
  Zeilenende-Uebersetzung; nie ausserhalb des Repos, nie in `.git`. Mit `erwartet_alt`
  wird unmittelbar vor dem Ersetzen verglichen (Compare-and-Swap).
- Commits nehmen ausschliesslich die genannten Pfade auf (pathspec-begrenzter Commit),
  auch wenn eine andere Session gerade etwas gestaged hat. Jeder Pfad wird vorher auf die
  Schreibweise gebracht, die git meldet; passt ein Pfad nicht, wird VOR dem Commit
  abgebrochen. Kein `git add -A`, kein Stash, kein Branch-Wechsel, kein Push.
- Revert betrifft nur die Aenderung EINER Datei eines Loop-Commits, nie den ganzen Commit
  (der Commit enthaelt auch das append-only `docs/graph/ledger.jsonl`).
- Der Torwaechter (`docs_drift_waechter.py`) darf durch den Loop keine neuen Fehler bekommen;
  bestehende fremde Fehler blockieren nicht. Laeuft der Waechter nicht sauber durch, wird
  `TorwaechterFehler` geworfen und der Tick committet nicht.

Ausser dem vom Aufrufer genannten Ledger-Pfad schreibt keine Funktion ausserhalb des Repos.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .anker import Beleg

log = logging.getLogger(__name__)

GIT_TIMEOUT_S = 60
GIT_COMMIT_TIMEOUT_S = 300  # Hooks (post-commit-Marker) laufen im Commit mit
WAECHTER_SKRIPT = "scripts/docs_drift_waechter.py"
WAECHTER_TIMEOUT_S = 120

DriftFehler = tuple[str, str]  # (dokument, meldung); Index-Fehler: ("Index", meldung)


class GitFehler(RuntimeError):
    """Ein Git-Aufruf ist fehlgeschlagen oder hat nicht das erwartete Ergebnis geliefert."""


class CommitUnbestaetigt(GitFehler):
    """`git commit` meldete einen Fehler/Timeout, HEAD hat sich aber bewegt."""

    def __init__(self, text: str, sha: str, dateien_ok: bool) -> None:
        super().__init__(text)
        self.sha = sha
        self.dateien_ok = dateien_ok


class FremdeAenderung(RuntimeError):
    """Eine Zieldatei hat nicht (mehr) den Inhalt, den der Loop erwartet bzw. geschrieben hat."""


class TorwaechterFehler(RuntimeError):
    """Der Drift-Waechter lief nicht sauber durch (Absturz, Timeout, unlesbare Ausgabe)."""


# --- Ledger -------------------------------------------------------------------------------

@dataclass(frozen=True)
class LedgerEintrag:
    lauf: str
    zeit: str
    aussage_id: str
    datei: str
    alt: str
    neu: str
    beleg: Beleg
    urteil: str
    quelle: str
    commit: str | None = None  # Ruling 46: der Eintrag liegt im Loop-Commit selbst, kennt dessen sha also nicht
    basis_commit: str | None = None  # HEAD vor dem Loop-Commit; den Commit selbst findet `lauf` (Nachricht)

    def to_dict(self) -> dict[str, object]:
        return {
            "lauf": self.lauf, "zeit": self.zeit, "aussage_id": self.aussage_id, "datei": self.datei,
            "alt": self.alt, "neu": self.neu, "beleg": self.beleg.to_dict(), "urteil": self.urteil,
            "quelle": self.quelle, "commit": self.commit, "basis_commit": self.basis_commit,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "LedgerEintrag":
        beleg = d["beleg"]
        if not isinstance(beleg, Mapping):
            raise ValueError("Ledger-Eintrag ohne Beleg-Objekt")
        commit = d.get("commit")
        basis = d.get("basis_commit")
        return cls(
            lauf=str(d["lauf"]), zeit=str(d["zeit"]), aussage_id=str(d["aussage_id"]),
            datei=str(d["datei"]), alt=str(d["alt"]), neu=str(d["neu"]),
            beleg=Beleg.from_dict(beleg), urteil=str(d["urteil"]), quelle=str(d["quelle"]),
            commit=str(commit) if commit is not None else None,
            basis_commit=str(basis) if basis is not None else None,
        )


def ledger_anhaengen(pfad: Path, eintrag: LedgerEintrag | Mapping[str, object]) -> None:
    """Eine JSON-Zeile anhaengen (ensure_ascii=False, sort_keys), danach flush + fsync."""
    daten = eintrag.to_dict() if isinstance(eintrag, LedgerEintrag) else dict(eintrag)
    zeile = json.dumps(daten, ensure_ascii=False, sort_keys=True) + "\n"
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "ab") as f:
        # Eine abgebrochene Vorzeile nicht mit dem neuen Eintrag verschmelzen.
        if f.tell() > 0:
            with open(pfad, "rb") as lesen:
                lesen.seek(-1, os.SEEK_END)
                if lesen.read(1) != b"\n":
                    f.write(b"\n")
        f.write(zeile.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


# --- Prozesse -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _Ergebnis:
    code: int
    aus: bytes
    err: bytes

    @property
    def text(self) -> str:
        return self.aus.decode("utf-8", errors="replace")

    @property
    def fehlertext(self) -> str:
        return self.err.decode("utf-8", errors="replace")


def _baum_beenden(proc: subprocess.Popen[bytes]) -> None:
    """Prozess samt Kindern beenden; geerbte Pipes von Enkeln sollen nicht ewig offen bleiben."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _laufen(befehl: Sequence[str], *, eingabe: bytes | None = None, timeout: float,
            cwd: Path | None = None, env: Mapping[str, str] | None = None) -> _Ergebnis:
    """Ohne Shell starten; bei Timeout den ganzen Prozessbaum beenden und TimeoutExpired werfen."""
    proc = subprocess.Popen(
        list(befehl), stdin=subprocess.PIPE if eingabe is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None, env=dict(env) if env is not None else None,
    )
    try:
        aus, err = proc.communicate(eingabe, timeout=timeout)
    except subprocess.TimeoutExpired:
        _baum_beenden(proc)
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            for strom in (proc.stdout, proc.stderr):
                if strom is not None:
                    strom.close()
        raise subprocess.TimeoutExpired(list(befehl), timeout) from None
    return _Ergebnis(proc.returncode, aus, err)


# --- Git ----------------------------------------------------------------------------------

def _git(repo: Path, *args: str, eingabe: str | bytes | None = None, pruefen: bool = True,
         timeout: float | None = None) -> _Ergebnis:
    befehl = ["git", "-c", "core.quotepath=false", "--literal-pathspecs", "-C", str(repo), *args]
    roh = eingabe.encode("utf-8") if isinstance(eingabe, str) else eingabe
    grenze = GIT_TIMEOUT_S if timeout is None else timeout
    try:
        ergebnis = _laufen(befehl, eingabe=roh, timeout=grenze)
    except subprocess.TimeoutExpired as exc:
        raise GitFehler(f"git {args[0]}: Timeout nach {grenze} s") from exc
    except OSError as exc:
        raise GitFehler(f"git nicht startbar: {type(exc).__name__}") from exc
    if pruefen and ergebnis.code != 0:
        raise GitFehler(f"git {' '.join(args[:2])} -> exit {ergebnis.code}: {ergebnis.fehlertext.strip()[:500]}")
    return ergebnis


def _nul_liste(text: str) -> list[str]:
    return [t for t in text.split("\0") if t]


def _ignorecase(repo: Path) -> bool:
    return _git(repo, "config", "--bool", "core.ignorecase", pruefen=False).text.strip() == "true"


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "--verify", "-q", "HEAD", pruefen=False).text.strip()


def _kanonisch(repo: Path, datei: str, ignorecase: bool) -> str:
    """Repo-relativer POSIX-Pfad in der Schreibweise des Dateisystems bzw. von git.

    `resolve()` loest `..`, absolute Pfade und (unter Windows) die Gross-/Kleinschreibung
    vorhandener Teile auf. Fuer nicht (mehr) vorhandene Dateien wird bei core.ignorecase die
    Schreibweise aus Index/HEAD uebernommen, wenn sie eindeutig ist.
    """
    ziel = _im_repo(repo, datei)
    rel = ziel.relative_to(repo.resolve()).as_posix()
    if ignorecase and not ziel.exists():
        bekannt = set(_nul_liste(_git(repo, "ls-files", "-z").text))
        bekannt |= set(_nul_liste(_git(repo, "ls-tree", "-r", "--name-only", "-z", "HEAD", pruefen=False).text))
        treffer = [p for p in bekannt if p.casefold() == rel.casefold()]
        if len(treffer) == 1:
            rel = treffer[0]
    return rel


def fremde_aenderungen(repo: Path, dateien: Sequence[str]) -> list[str]:
    """Alle Pfade unter `dateien` mit irgendeiner Aenderung (gestaged, ungestaged, untracked).

    Der Loop ruft das VOR dem Schreiben auf; alles, was hier auftaucht, gehoert jemand anderem.
    Umbenennungen (im Index wie im Arbeitsbaum) liefern beide Pfade.
    """
    if not dateien:
        return []
    ic = _ignorecase(repo)
    pfade = [_kanonisch(repo, d, ic) for d in dateien]
    ausgabe = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *pfade).text
    tokens = ausgabe.split("\0")
    gefunden: dict[str, None] = {}
    i = 0
    while i < len(tokens):
        eintrag = tokens[i]
        i += 1
        if len(eintrag) < 4:
            continue
        xy, pfad = eintrag[:2], eintrag[3:]
        gefunden.setdefault(pfad, None)
        if ("R" in xy or "C" in xy) and i < len(tokens):  # -z: "XY neu\0alt\0"
            if tokens[i]:
                gefunden.setdefault(tokens[i], None)
            i += 1
    return list(gefunden)


def _unstagen(repo: Path, pfade: Sequence[str]) -> None:
    """Eigenes Staging des Loops zuruecknehmen (Arbeitsbaum bleibt unberuehrt)."""
    for pfad in pfade:
        _git(repo, "restore", "--staged", "--", pfad, pruefen=False)


def _dateien_im_commit(repo: Path, sha: str) -> set[str]:
    return set(_nul_liste(_git(repo, "show", "--no-renames", "--name-only", "--format=", "-z", sha).text))


def _sha256(daten: bytes) -> str:
    return hashlib.sha256(daten).hexdigest()


def committen(repo: Path, dateien: Sequence[str], nachricht: str, *,
              erwartet: Mapping[str, str] | None = None) -> str | None:
    """Genau `dateien` committen (Inhalt aus dem Arbeitsbaum); Rueckgabe SHA, ohne Aenderung `None`.

    - Pfade werden kanonisiert und auf die von git gemeldeten Aenderungen abgebildet; fehlt
      einer (unveraendert, falsche Schreibweise, ausserhalb), wird VOR dem Commit abgebrochen.
    - `erwartet`: Pfad -> sha256 der Bytes, die der Loop geschrieben hat. Geprueft werden die
      Datei vor `git add`, der gestagte Blob nach `git add` und der committete Blob; Abweichung
      -> `FremdeAenderung`.
    - Pathspec-begrenzter Commit: fremde gestagte Eintraege bleiben gestaged und kommen nicht mit.
    - Scheitert der Commit, wird das eigene Staging zurueckgenommen; hat sich HEAD trotzdem
      bewegt, kommt `CommitUnbestaetigt`.
    """
    if not dateien:
        return None
    ic = _ignorecase(repo)
    pfade = list(dict.fromkeys(_kanonisch(repo, d, ic) for d in dateien))
    erwartet_k = {_kanonisch(repo, k, ic): v for k, v in (erwartet or {}).items()}
    if not set(erwartet_k) <= set(pfade):
        raise ValueError(f"erwartet nennt Pfade ausserhalb von dateien: {sorted(set(erwartet_k) - set(pfade))}")

    blobs: dict[str, str] = {}
    try:
        for pfad in pfade:
            datei = repo / pfad
            if pfad in erwartet_k:
                daten = datei.read_bytes() if datei.is_file() else None
                if daten is None or _sha256(daten) != erwartet_k[pfad]:
                    raise FremdeAenderung(f"{pfad}: Inhalt weicht vom geschriebenen Stand ab (vor git add)")
                # Blob-ID der geschriebenen Bytes nach den Filtern dieses Pfads (autocrlf, eol, ...)
                blobs[pfad] = _git(repo, "hash-object", f"--path={pfad}", "--stdin", eingabe=daten).text.strip()
            if datei.exists():
                _git(repo, "add", "--", pfad)
            else:  # geloeschte Datei: Loeschung vormerken
                _git(repo, "rm", "--cached", "-q", "--ignore-unmatch", "--", pfad)
        for pfad, blob in blobs.items():
            gestaged = _git(repo, "rev-parse", "--verify", "-q", f":{pfad}", pruefen=False).text.strip()
            if gestaged != blob:
                raise FremdeAenderung(f"{pfad}: gestagter Inhalt weicht vom geschriebenen Stand ab")

        alle = set(_nul_liste(_git(repo, "diff", "--cached", "--no-renames", "--name-only", "-z", "HEAD").text))
        zuordnung: dict[str, str] = {}
        for pfad in pfade:
            if pfad in alle:
                zuordnung[pfad] = pfad
            elif ic:
                treffer = [a for a in alle if a.casefold() == pfad.casefold()]
                if len(treffer) == 1:
                    zuordnung[pfad] = treffer[0]
        if not zuordnung:
            return None
        fehlend = [p for p in pfade if p not in zuordnung]
        if fehlend:
            raise GitFehler(f"keine gestagte Aenderung fuer {fehlend}; Commit abgebrochen")
        ziele = list(dict.fromkeys(zuordnung[p] for p in pfade))
        if not ziele:  # nie `git commit --` ohne Pfade: das wuerde den ganzen Index committen
            raise GitFehler("leere Pfadliste; Commit abgebrochen")
    except BaseException:
        _unstagen(repo, pfade)
        raise

    vorher = _head(repo)
    try:
        _git(repo, "commit", "-q", "-F", "-", "--", *ziele, eingabe=nachricht, timeout=GIT_COMMIT_TIMEOUT_S)
    except GitFehler as exc:
        nachher = _head(repo)
        if nachher and nachher != vorher:
            ok = _dateien_im_commit(repo, nachher) == set(ziele)
            raise CommitUnbestaetigt(f"git commit meldete '{exc}', HEAD steht aber auf {nachher[:7]} "
                                     f"(Dateiliste {'passt' if ok else 'passt NICHT'})", nachher, ok) from exc
        _unstagen(repo, pfade)
        raise
    sha = _head(repo)
    im_commit = _dateien_im_commit(repo, sha)
    if im_commit != set(ziele):
        raise CommitUnbestaetigt(f"Commit {sha[:7]} enthaelt {sorted(im_commit)}, erwartet {sorted(ziele)}",
                                 sha, False)
    for pfad, blob in blobs.items():
        im_head = _git(repo, "rev-parse", "--verify", "-q", f"HEAD:{zuordnung[pfad]}", pruefen=False).text.strip()
        if im_head != blob:
            raise FremdeAenderung(f"{pfad}: Commit {sha[:7]} enthaelt nicht den geschriebenen Stand")
    return sha


def revertieren(repo: Path, sha: str, datei: str) -> bool:
    """Nur die Aenderung von `datei` aus `sha` zuruecknehmen und als eigenen Commit buchen.

    Voraussetzungen, sonst `False`: `sha` ist der letzte Commit, der `datei` beruehrt hat,
    und `datei` hat keine nicht committeten (fremden) Aenderungen. Hat `sha` die Datei
    angelegt, wird sie geloescht.
    """
    pfad = _kanonisch(repo, datei, _ignorecase(repo))
    aufgeloest = _git(repo, "rev-parse", "--verify", "-q", f"{sha}^{{commit}}", pruefen=False)
    if aufgeloest.code != 0:
        return False
    voll = aufgeloest.text.strip()
    spitze = _git(repo, "log", "-1", "--format=%H", "--", pfad).text.strip()
    if spitze != voll:
        return False
    if fremde_aenderungen(repo, [pfad]):
        return False
    im_vorgaenger = _git(repo, "rev-parse", "--verify", "-q", f"{voll}^:{pfad}", pruefen=False).code == 0
    try:
        if im_vorgaenger:
            _git(repo, "checkout", f"{voll}^", "--", pfad)
        else:
            _git(repo, "rm", "-q", "--", pfad)
        neu = committen(repo, [pfad], f"revert(loop): {pfad} aus {voll[:7]}")
    except CommitUnbestaetigt:
        raise
    except Exception:
        # Datei auf HEAD zuruecksetzen, damit kein halber Revert im Arbeitsbaum bleibt.
        _git(repo, "checkout", "HEAD", "--", pfad, pruefen=False)
        raise
    return neu is not None


# --- Torwaechter --------------------------------------------------------------------------

_KOPF = re.compile(r"^(?:ok|FEHLER)\s+(\S+)\s+Belege\s+\d+")
_DETAIL = re.compile(r"^\s+FEHLER\s+(.*\S)\s*$")
_INDEX = re.compile(r"^FEHLER Index:\s*(.*\S)\s*$")
_SUMME = re.compile(r"^(\d+) Dokumente, (\d+) Fehler, \d+ Warnungen\s*$")


STANDARD_TORWAECHTER = (".venv/Scripts/python.exe", WAECHTER_SKRIPT)


def _waechter_python(repo: Path, interpreter: str = STANDARD_TORWAECHTER[0]) -> str:
    kandidat = Path(interpreter)
    if not kandidat.is_absolute():
        kandidat = repo / kandidat
    if kandidat.is_file():
        return str(kandidat)
    log.warning("Torwaechter: %s fehlt, nutze %s", kandidat, sys.executable)
    return sys.executable


@dataclass(frozen=True)
class _Auswertung:
    fehler: set[DriftFehler]
    roh: int  # Anzahl Fehlerzeilen
    koepfe: int  # Anzahl Dokument-Kopfzeilen
    summe_dokumente: int | None
    summe_fehler: int | None


def _auswerten(ausgabe: str) -> _Auswertung:
    fehler: set[DriftFehler] = set()
    roh = koepfe = 0
    summe_dok: int | None = None
    summe_fehl: int | None = None
    doc: str | None = None
    for zeile in ausgabe.splitlines():
        if m := _INDEX.match(zeile):
            fehler.add(("Index", m.group(1)))
            roh += 1
        elif m := _KOPF.match(zeile):
            doc = m.group(1)  # Zaehler in der Kopfzeile sind Rauschen
            koepfe += 1
        elif m := _DETAIL.match(zeile):
            if doc is None:
                raise TorwaechterFehler(f"Waechter: Fehlerzeile vor jeder Dokumentzeile: {zeile.strip()}")
            fehler.add((doc, m.group(1)))
            roh += 1
        elif m := _SUMME.match(zeile.strip()):
            summe_dok, summe_fehl = int(m.group(1)), int(m.group(2))
    return _Auswertung(fehler, roh, koepfe, summe_dok, summe_fehl)


def drift_fehler(repo: Path, torwaechter: Sequence[str] | None = None) -> set[DriftFehler]:
    """Fehler des Drift-Waechters (Exit 1) als Menge von `(dokument, meldung)`; Exit 0 -> leer.

    Kopfzeilen mit ihren Zaehlern fallen weg; gleiche Meldungen in verschiedenen Dokumenten
    bleiben getrennt. Absturz, Timeout, unbekannter Exit-Code oder eine Ausgabe, die nicht zur
    Summenzeile passt (Dokumente oder Fehler) -> `TorwaechterFehler`.

    `torwaechter=None` = der Standardbefehl (mit dem aktuellen `WAECHTER_SKRIPT`); `()` = kein Torwaechter; sonst
    gilt der Befehl wortwoertlich (Interpreter, Skript), ein Eintrag oder eine Zeichenkette ist ein Fehler.
    """
    if torwaechter is None:
        torwaechter = (STANDARD_TORWAECHTER[0], WAECHTER_SKRIPT)
    if not torwaechter:
        return set()  # leere Befehlsliste = kein Torwaechter (D4)
    if isinstance(torwaechter, str) or len(torwaechter) < 2:
        raise TorwaechterFehler("Waechter: Befehl braucht Interpreter und Skript (mindestens zwei Eintraege)")
    skript = Path(torwaechter[1])
    if not skript.is_absolute():
        skript = repo / skript
    if not skript.is_file():
        raise TorwaechterFehler(f"Waechter fehlt: {skript}")
    umgebung = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        lauf = _laufen([_waechter_python(repo, torwaechter[0]), str(skript)], cwd=repo, env=umgebung, timeout=WAECHTER_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise TorwaechterFehler(f"Waechter: Timeout nach {WAECHTER_TIMEOUT_S} s") from exc
    except OSError as exc:
        raise TorwaechterFehler(f"Waechter nicht startbar: {type(exc).__name__}") from exc
    if lauf.code not in (0, 1) or "Traceback (most recent call last)" in lauf.fehlertext:
        raise TorwaechterFehler(f"Waechter abgestuerzt (exit {lauf.code}): {lauf.fehlertext.strip()[-300:]}")
    a = _auswerten(lauf.text)
    if a.summe_fehler is None or a.summe_dokumente is None:
        raise TorwaechterFehler(f"Waechter ohne Summenzeile (exit {lauf.code})")
    if a.koepfe != a.summe_dokumente:
        raise TorwaechterFehler(f"Waechter-Ausgabe unlesbar: {a.koepfe} Dokumentzeilen, Summe {a.summe_dokumente}")
    if lauf.code == 0:
        return set()
    if a.summe_fehler == 0 or a.roh != a.summe_fehler:
        raise TorwaechterFehler(f"Waechter-Ausgabe unlesbar: {a.roh} Fehlerzeilen, Summe {a.summe_fehler}")
    return a.fehler


def drift_fuer(konfig: object) -> Callable[[Path], set[DriftFehler]]:
    """`drift_fehler` mit dem Torwaechter aus der Konfiguration (`konfig.torwaechter`)."""
    roh = getattr(konfig, "torwaechter", STANDARD_TORWAECHTER)
    befehl = roh if isinstance(roh, str) else tuple(roh)  # eine Zeichenkette nicht in Buchstaben zerlegen
    return lambda repo: drift_fehler(repo, befehl)


def torwaechter_ok(vorher: set[DriftFehler], nachher: set[DriftFehler]) -> bool:
    """Der Loop darf keine neuen Fehler erzeugen; bestehende fremde Fehler blockieren nicht."""
    return nachher <= vorher


# --- Schreiben ----------------------------------------------------------------------------

def _im_repo(repo: Path, datei: str) -> Path:
    """Zielpfad aufloesen; ausserhalb des Repos oder in `.git` -> ValueError."""
    roh = Path(datei)
    ziel = (roh if roh.is_absolute() else repo / roh).resolve()
    wurzel = repo.resolve()
    try:
        rel = ziel.relative_to(wurzel)
    except ValueError as exc:
        raise ValueError(f"Pfad ausserhalb des Repos: {datei}") from exc
    if not rel.parts:
        raise ValueError(f"Pfad ist das Repo selbst: {datei}")
    if any(teil.lower() == ".git" for teil in rel.parts):
        raise ValueError(f"Pfad in .git: {datei}")
    return ziel


def schreiben(repo: Path, datei: str, text: str, *, erwartet_alt: bytes | None = None) -> None:
    """`text` atomar und byteweise (UTF-8, Zeilenenden wie Eingabe) nach `repo/datei` schreiben.

    `erwartet_alt`: die Bytes, die der Loop gelesen hat; unmittelbar vor `os.replace` wird
    erneut gelesen und verglichen (fehlende Datei zaehlt als b""). Abweichung -> `FremdeAenderung`.
    """
    ziel = _im_repo(repo, datei)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=ziel.parent, prefix=f".{ziel.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        if erwartet_alt is not None:
            aktuell = ziel.read_bytes() if ziel.is_file() else b""
            if aktuell != erwartet_alt:
                raise FremdeAenderung(f"{datei}: seit dem Lesen von jemand anderem geaendert")
        os.replace(temp, ziel)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise
