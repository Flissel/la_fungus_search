"""Stufe 4, Task 10: die Windows-Aufgabe `VibeMind Doku-Loop` auf den neuen Loop umschalten -- und zurueck.

`python -m selbstbild doku aufgabe <wurzel> --pruefen | --umschalten | --zurueck [--trocken]`

- `--pruefen` (nur lesend): aktuelle Aktion, Einordnung `alt` | `neu` | `unbekannt`, gemerkte alte Aktion, neueste
  Sicherung, Archiv der erzaehlten Seiten, Zustand von `tick.lock`, und welche Arten im Repo-Graph der alte Loop
  nicht kennt (die `--zurueck` entfernen wuerde).
- Python der neuen Aktion: genau `STANDARD_PYTHON` oder `--python <pfad>`; es muss existieren und `selbstbild.doku`
  ohne PYTHONPATH laden. Nie stillschweigend `sys.executable`.
- `--umschalten` (nur von `alt`), unter der Sperre `tick.lock` (frisch gehalten -> Exit 2, nichts geaendert):
  1. lokalen Ordner (`%LOCALAPPDATA%\\VibeMind\\doku_loop\\`) ohne `messbaum-*` nach
     `...\\doku_loop-sicherung-<JJJJMMTT-hhmmss>\\` sichern (mtimes bleiben; erst unter `.tmp`, dann umbenannt);
  2. alte Aktion als JSON in `...\\doku_loop\\aufgabe-alt.json` merken;
  3. neue Aktion setzen (Trigger, Limits, Principal bleiben), danach nachlesen;
  4. Seiten aus `...\\doku_loop\\erzaehlt-archiv\\` (von einem frueheren Rueckweg) samt Manifest-Eintrag zuruecklegen;
  5. Zusammenfassung.
  Scheitert Schritt 3, bleibt die Sicherung liegen und sonst aendert sich nichts (aufgabe-alt.json wird entfernt,
  wenn die Aktion nachweislich noch die alte ist; sonst bleibt sie fuer `--zurueck`).
- `--zurueck` (Ruling T10-a/b/d), ebenso unter der Sperre, in der Reihenfolge c, b, d, ZULETZT a (Fix-Runde 1): bis a
  bleibt der neue Loop geplant, jeder Fehler davor laesst einen stimmigen Stand, ein erneutes `--zurueck` macht fertig.
  Vorher (Exit 2, nichts geaendert): `aufgabe-alt.json` lesbar, Aktion `neu` oder schon die gemerkte, Sicherung da
  (die in `aufgabe-alt.json` gemerkte; weicht die neueste ab -> Warnung), und falls d etwas zu tun hat: keine fremde,
  nicht committete Aenderung an `docs/graph/graph.json`, Kopf schreibbar (Zweig, keine Git-Operation).
  c. VOR b: jede eigene Rowboat-Seite des neuen Loops (`Mitbauen/...`, `Zustand.md`, `Was ist VibeMind.md`,
     `Spaces/*`, `Abläufe/*` -- laut aktuellem Notizen-Manifest UND Frontmatter `erzeugt_von: selbstbild`):
     unveraendert (Hash = Manifest) -> mit Pfad und Manifest-Eintrag ins Archiv, Eintrag weg, Datei weg (Seite fuer
     Seite, Manifeste nach jeder Seite); von Hand geaendert -> bleibt. Leer gewordene `Mitbauen/` und `Abläufe/`
     werden entfernt (nur diese);
  b. die Sicherung ueber den Ordner zurueckspielen (ohne `messbaum-*`); was jetzt da ist, aber nicht in der
     Sicherung, wandert nach `...\\doku_loop-nachher-<zeit>\\` (nie geloescht). Danach Manifest-Abgleich: jede
     deterministische Notiz, die der neue Loop unveraendert hinterlassen hat (Datei-Hash = aktueller Manifest-Eintrag),
     behaelt ihren jetzigen Hash im zurueckgespielten Manifest -- sonst hielte der alte Loop sie fuer von Hand
     geaendert (`notiz_von_hand`-Flut, Review-Fokus 4) und schriebe `Index (Loop).md` daneben;
     `seiten-fehlversuche-<hash>.json` bleibt (Task 9 haelt ihn bewusst ausserhalb des Stands);
  d. (T10-d) aus `docs/graph/graph.json` des aeusseren Repos alle Knotentypen, Kantenarten und -quellen entfernen, die
     der alte Loop nicht kennt (sein `graph.py` wirft sonst `ValueError` -> jeder alte Tick stirbt), samt Kanten an
     entfernten Knoten. Die bekannten Arten liest `alte_typen` per `ast` aus `scripts/doku_loop/graph.py` (Kopie als
     Rueckfall). Committet wird nur diese Datei ueber den gemeinsamen Schreibvorgang (Zweig, CAS, Torwaechter,
     pathspec): `docs(loop): Rueckweg – Knotenarten des neuen Loops entfernt`;
  a. alte Aktion aus `aufgabe-alt.json` setzen und nachlesen. Danach wandert `aufgabe-alt.json` in den
     Nachher-Ordner.
- `--trocken` zeigt jeden Schritt (Dateien, Aktionen, Archiv-Eintraege) und schreibt nichts.

Nie gesichert oder zurueckgespielt: `messbaum-*` (Worktrees des Loops), `tick.lock*`, `erzaehlt-archiv`,
`aufgabe-alt.json` und `seiten-fehlversuche-*.json`.

Das Backend (`PowerShellBackend`) ruft `powershell -NoProfile -NonInteractive -Command` mit `Get-ScheduledTask` bzw.
`Set-ScheduledTask -Action (New-ScheduledTaskAction ...)`. Die Werte gehen nur ueber Umgebungsvariablen hinein (keine
Anfuehrungszeichen in der Befehlszeile), Ausgaben nur ueber Dateien (keine Pipe, die ein Kindprozess erben koennte),
mit Zeitlimit. Tests injizieren ein eigenes Backend.

Exit-Codes: 0 ok, 1 Fehler mitten im Ablauf (Meldung sagt, was steht), 2 abgelehnt (nichts geaendert).
"""

from __future__ import annotations

import ast
import filecmp
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Protocol

from . import rowboat, schreibvorgang, umschreiben
from .graph import Graph
from .konfig import Konfig
from .rowboat_erzaehlt import ABLAEUFE, SPACES, WAS_IST_REL
from .rowboat_seiten import MARKER as SEITEN_MARKER
from .rowboat_seiten import MITBAUEN_REL, ZUSTAND_REL
from .stand import SPERRE_DATEI, lokaler_ordner, schreibvermerk_pfad, sperre, sperre_frisch, sperre_zustand

AUFGABE = "VibeMind Doku-Loop"
ALT_DATEI = "aufgabe-alt.json"
ARCHIV = "erzaehlt-archiv"
ARCHIV_MANIFEST = "manifest.json"
ARCHIV_SEITEN = "seiten"
MITBAUEN = MITBAUEN_REL.split("/", 1)[0]
LOG_REL = ("logs", "doku-loop.log")  # wie `__main__.LOG_DATEI`; dort rotiert `tick` vor jedem Lauf
ZEIT_FORMAT = "%Y%m%d-%H%M%S"
PS_TIMEOUT_S = 60
IMPORT_TIMEOUT_S = 120
# Fix-Runde 1 (Minor 4): das Python der neuen Aktion -- genau dieses, oder ausdruecklich `--python <pfad>`
STANDARD_PYTHON = r"C:\Users\User\ClaudeWork\wt-fungus-gate1\.venv-selbstbild\Scripts\python.exe"
# Fix-Runde 1 (Ruling T10-d): Knoten-/Kantenarten, die der ALTE Loop kennt (`scripts/doku_loop/graph.py`, Stand
# 40e05ec0). Gelesen wird per `ast` aus dem aeusseren Repo; diese Kopie gilt nur, wenn das nicht geht.
ALTER_GRAPH_PY = ("scripts", "doku_loop", "graph.py")
ALTE_TYPEN_KOPIE: dict[str, frozenset[str]] = {
    "KNOTEN_TYPEN": frozenset({"space", "dienst", "modul", "symbol", "route", "tabelle", "sql_funktion",
                               "mcp_werkzeug", "mensch", "aussage", "kette", "komponente"}),
    "KANTEN_ARTEN": frozenset({"ruft_auf", "definiert_route", "liest_schreibt", "registriert", "verbindet",
                               "nutzt_dienst", "gehoert_zu", "hop", "belegt_durch", "geaendert_durch"}),
    "KANTEN_QUELLEN": frozenset({"extraktor", "seed", "llm", "review"}),
}
ALTE_GRAPH_VERSION = 1
GRAPH_BETREFF = "docs(loop): Rueckweg – Knotenarten des neuen Loops entfernt"
GRAPH_JSON = schreibvorgang.GRAPH_JSON
_CMD_SONDER = set('"%^&|<>!')  # in cmd.exe auch innerhalb von Anfuehrungszeichen heikel (%, !) oder bruchgefaehrlich
_NEU = re.compile(r"(?:^|\s)-m\s+selbstbild\s+doku\s+tick(?:\s|$)", re.IGNORECASE)
_ALT = re.compile(r"scripts\\doku-loop\.cmd\s+tick(?:\s|$)", re.IGNORECASE)

Aus = Callable[[str], None]
ImportPruefen = Callable[[str, Path], "str | None"]


class AufgabeFehler(RuntimeError):
    """Die Aufgabe liess sich nicht lesen oder setzen."""


@dataclass(frozen=True)
class Aktion:
    execute: str
    arguments: str
    working_directory: str

    def als_dict(self) -> dict[str, str]:
        return {"Execute": self.execute, "Arguments": self.arguments, "WorkingDirectory": self.working_directory}

    @classmethod
    def aus_dict(cls, d: Mapping[str, Any]) -> "Aktion":
        if not isinstance(d, Mapping) or not isinstance(d.get("Execute"), str) or not d["Execute"]:
            raise ValueError("Aktion ohne Execute")
        return cls(str(d["Execute"]), str(d.get("Arguments") or ""), str(d.get("WorkingDirectory") or ""))

    def zeile(self) -> str:
        return f"{self.execute} {self.arguments}".strip() + f"  (WorkingDirectory {self.working_directory or '-'})"


class Backend(Protocol):
    def lesen(self, aufgabe: str) -> list[Aktion]: ...

    def setzen(self, aufgabe: str, aktion: Aktion) -> None: ...


# ---------- Aktion ----------

def _cmd_sicher(wert: str, was: str) -> str:
    schlecht = sorted(set(wert) & _CMD_SONDER)
    if schlecht:
        raise ValueError(f"{was} enthält Zeichen, die cmd.exe umdeutet: {' '.join(schlecht)}")
    return wert


def neue_argumente(python: str | Path, wurzel: str | Path) -> str:
    """`--headless cmd /c ""<python>" -m selbstbild doku tick "<wurzel>" >> "<wurzel>\\logs\\doku-loop.log" 2>&1"`.

    cmd /c mit mehr als zwei Anfuehrungszeichen entfernt nur das erste und das letzte; uebrig bleibt
    `"<python>" -m selbstbild doku tick "<wurzel>" >> "<log>" 2>&1` (getestet mit dem echten cmd)."""
    py = _cmd_sicher(str(python), "Python-Pfad")
    w = _cmd_sicher(str(PureWindowsPath(str(wurzel))), "Wurzel")
    log = str(PureWindowsPath(w, *LOG_REL))
    return f'--headless cmd /c ""{py}" -m selbstbild doku tick "{w}" >> "{log}" 2>&1"'


def neue_aktion(python: str | Path, wurzel: str | Path, working_directory: str | None = None) -> Aktion:
    """Die Aktion des neuen Loops; WorkingDirectory ist die der alten Aktion (sonst die Wurzel)."""
    return Aktion("conhost.exe", neue_argumente(python, wurzel),
                  working_directory if working_directory else str(PureWindowsPath(str(wurzel))))


def einordnen(aktionen: Sequence[Aktion]) -> str:
    """`alt` (scripts\\doku-loop.cmd tick), `neu` (-m selbstbild doku tick) oder `unbekannt` (auch: nicht genau eine
    Aktion oder nicht conhost.exe)."""
    if len(aktionen) != 1:
        return "unbekannt"
    a = aktionen[0]
    if PureWindowsPath(a.execute).name.casefold() != "conhost.exe":
        return "unbekannt"
    if _NEU.search(a.arguments):
        return "neu"
    if _ALT.search(a.arguments):
        return "alt"
    return "unbekannt"


def _gleicher_pfad(a: str, b: str | Path) -> bool:
    def norm(p: str | Path) -> str:
        try:
            return str(Path(p).resolve()).casefold().rstrip("\\/")
        except OSError:
            return str(p).casefold().rstrip("\\/")
    return norm(a) == norm(b)


# ---------- PowerShell-Backend ----------

_PS_LESEN = (
    "$ErrorActionPreference = 'Stop'; try { "
    "$t = Get-ScheduledTask -TaskName $env:VM_AUFGABE; "
    "$a = @($t.Actions | ForEach-Object { [ordered]@{ Execute = [string]$_.Execute; "
    "Arguments = [string]$_.Arguments; WorkingDirectory = [string]$_.WorkingDirectory } }); "
    "$j = ConvertTo-Json -InputObject $a -Depth 3 -Compress; "
    "[IO.File]::WriteAllText($env:VM_AUS, $j, (New-Object System.Text.UTF8Encoding $false)) "
    "} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }"
)
_PS_SETZEN = (
    "$ErrorActionPreference = 'Stop'; try { "
    "$p = @{ Execute = $env:VM_EXECUTE }; "
    "if ($env:VM_ARGUMENTS) { $p.Argument = $env:VM_ARGUMENTS }; "
    "if ($env:VM_WD) { $p.WorkingDirectory = $env:VM_WD }; "
    "$neu = New-ScheduledTaskAction @p; "
    "Set-ScheduledTask -TaskName $env:VM_AUFGABE -Action $neu | Out-Null "
    "} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }"
)


class PowerShellBackend:
    """`powershell -NoProfile -NonInteractive -Command` mit Get-/Set-ScheduledTask; `laufzeit` fuer Tests."""

    def __init__(self, *, laufzeit: Callable[..., Any] = subprocess.run, powershell: str = "powershell",
                 timeout_s: int = PS_TIMEOUT_S) -> None:
        self.laufzeit, self.powershell, self.timeout_s = laufzeit, powershell, timeout_s

    def _ps(self, skript: str, werte: Mapping[str, str]) -> str:
        try:
            return self._ps_in_temp(skript, werte)
        except AufgabeFehler:
            raise
        except OSError as e:  # Fix-Runde 1 (Minor 5): z. B. der Temp-Ordner laesst sich nicht aufraeumen
            raise AufgabeFehler(f"Temp-Ordner: {type(e).__name__}") from None

    def _ps_in_temp(self, skript: str, werte: Mapping[str, str]) -> str:
        with tempfile.TemporaryDirectory(prefix="selbstbild-aufgabe-") as tmp:
            aus, fehler = Path(tmp) / "aus.json", Path(tmp) / "fehler.txt"
            env = dict(os.environ)
            env.update(werte)
            env["VM_AUS"] = str(aus)
            # Ausgaben nur in Dateien: keine Pipe, die ein Kindprozess erben und offen halten koennte
            with open(fehler, "wb") as f_fehler:
                try:
                    lauf = self.laufzeit([self.powershell, "-NoProfile", "-NonInteractive", "-Command", skript],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=f_fehler,
                                         env=env, timeout=self.timeout_s,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                except subprocess.TimeoutExpired:
                    raise AufgabeFehler(f"powershell: Zeitlimit {self.timeout_s} s") from None
                except OSError as e:
                    raise AufgabeFehler(f"powershell: {type(e).__name__}") from None
            if lauf.returncode != 0:
                meldung = " ".join(fehler.read_bytes().decode("utf-8", errors="replace").split())[:300]
                raise AufgabeFehler(f"powershell Exit {lauf.returncode}: {meldung or '-'}")
            try:
                return aus.read_bytes().decode("utf-8-sig") if aus.exists() else ""
            except OSError as e:
                raise AufgabeFehler(f"powershell-Ausgabe: {type(e).__name__}") from None

    def lesen(self, aufgabe: str) -> list[Aktion]:
        roh = self._ps(_PS_LESEN, {"VM_AUFGABE": aufgabe})
        try:
            daten = json.loads(roh)
            if isinstance(daten, dict):
                daten = [daten]
            if not isinstance(daten, list):
                raise ValueError("keine Liste")
            return [Aktion.aus_dict(d) for d in daten]
        except ValueError as e:
            raise AufgabeFehler(f"Aktionen nicht lesbar: {type(e).__name__}") from None

    def setzen(self, aufgabe: str, aktion: Aktion) -> None:
        self._ps(_PS_SETZEN, {"VM_AUFGABE": aufgabe, "VM_EXECUTE": aktion.execute,
                              "VM_ARGUMENTS": aktion.arguments, "VM_WD": aktion.working_directory})


def import_pruefen(python: str, wurzel: Path) -> str | None:
    """Laeuft `import selbstbild.doku.__main__` mit diesem Python ohne PYTHONPATH (wie in der Aufgabe, Arbeitsordner
    = Wurzel)? None = ja, sonst Grund. Nur lesend."""
    if not Path(python).is_file():
        return "Datei fehlt"
    env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}
    try:
        lauf = subprocess.run([python, "-c", "import selbstbild.doku.__main__"], cwd=wurzel, env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=IMPORT_TIMEOUT_S, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return "Zeitlimit"
    except OSError as e:
        return type(e).__name__
    return None if lauf.returncode == 0 else f"exit:{lauf.returncode}"


# ---------- lokaler Ordner: Sicherung und Zurueckspielen ----------

def _ausgenommen(name: str) -> bool:
    """Oberste Ebene des lokalen Ordners, die nie gesichert und nie zurueckgespielt wird."""
    n = name.casefold()
    return (n.startswith("messbaum-") or n == SPERRE_DATEI or n.startswith(SPERRE_DATEI + ".") or n == ARCHIV
            or n == ALT_DATEI or (n.startswith("seiten-fehlversuche-") and n.endswith(".json")))


def _inhalt(ordner: Path) -> tuple[list[str], list[str]]:
    """(Dateien, Ordner) relativ (POSIX), ohne die Ausnahmen der obersten Ebene; umgeleiteten Wegen folgt es nie."""
    dateien: list[str] = []
    ordnerliste: list[str] = []
    if not ordner.is_dir():
        return dateien, ordnerliste
    for wurzel, unter, namen in os.walk(ordner):
        w = Path(wurzel)
        oben = w == ordner
        unter[:] = sorted(d for d in unter if not (oben and _ausgenommen(d)) and not rowboat._umgeleitet(w / d))
        for d in unter:
            ordnerliste.append((w / d).relative_to(ordner).as_posix())
        for n in sorted(namen):
            if oben and _ausgenommen(n):
                continue
            if rowboat._umgeleitet(w / n):
                continue
            dateien.append((w / n).relative_to(ordner).as_posix())
    return dateien, ordnerliste


def _sicherungs_muster(ordner: Path) -> re.Pattern[str]:
    return re.compile(re.escape(ordner.name) + r"-sicherung-(\d{8}-\d{6})(?:_(\d+))?", re.IGNORECASE)


def sicherungen(ordner: Path) -> list[Path]:
    """Fertige Sicherungen neben dem Ordner, aelteste zuerst (`.tmp`-Reste zaehlen nicht)."""
    muster = _sicherungs_muster(ordner)
    treffer: list[tuple[tuple[str, int], Path]] = []
    try:
        eintraege = list(ordner.parent.iterdir())
    except OSError:
        return []
    for p in eintraege:
        m = muster.fullmatch(p.name)
        if m and p.is_dir():
            treffer.append(((m.group(1), int(m.group(2) or 0)), p))
    return [p for _, p in sorted(treffer)]


def _freier_name(eltern: Path, basis: str) -> Path:
    ziel, n = eltern / basis, 2
    while ziel.exists() or Path(str(ziel) + ".tmp").exists():
        ziel, n = eltern / f"{basis}_{n}", n + 1
    return ziel


def _sichern(ordner: Path, ziel: Path) -> int:
    """Kopie ohne Ausnahmen, mtimes bleiben; erst `<ziel>.tmp`, fertig umbenannt. Rueckgabe: Anzahl Dateien."""
    dateien, ordnerliste = _inhalt(ordner)
    tmp = Path(str(ziel) + ".tmp")
    tmp.mkdir(parents=True)
    for d in ordnerliste:
        (tmp / d).mkdir(parents=True, exist_ok=True)
    for f in dateien:
        shutil.copy2(ordner / f, tmp / f)
    for d in reversed(ordnerliste):
        shutil.copystat(ordner / d, tmp / d)
    os.rename(tmp, ziel)
    return len(dateien)


def _gleich(a: Path, b: Path) -> bool:
    """Gleiche Bytes UND gleiche mtime (sonst wird zurueckgespielt, damit auch die mtime der Sicherung gilt)."""
    try:
        return a.stat().st_mtime_ns == b.stat().st_mtime_ns and filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def _ersetzen(quelle: Path, ziel: Path) -> None:
    """`quelle` atomar nach `ziel` (Temp-Datei im Zielordner, copy2 = mtime bleibt, dann replace)."""
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=ziel.parent, prefix=".selbstbild-aufgabe-", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copy2(quelle, temp)
        os.replace(temp, ziel)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


@dataclass
class _Rueckspiel:
    ueberschreiben: list[str]
    gleich: list[str]
    verschieben: list[str]
    ordner_neu: list[str]
    ordner_weg: list[str]


def _rueckspiel_plan(ordner: Path, sicherung: Path) -> _Rueckspiel:
    s_dateien, s_ordner = _inhalt(sicherung)
    o_dateien, o_ordner = _inhalt(ordner)
    s_menge, so_menge = set(s_dateien), set(s_ordner)
    ueberschreiben = [f for f in s_dateien if not (ordner / f).is_file() or not _gleich(sicherung / f, ordner / f)]
    gleich = [f for f in s_dateien if f not in set(ueberschreiben)]
    return _Rueckspiel(ueberschreiben, gleich, [f for f in o_dateien if f not in s_menge],
                       [d for d in s_ordner if not (ordner / d).is_dir()],
                       [d for d in o_ordner if d not in so_menge])


def _zurueckspielen(ordner: Path, sicherung: Path, plan: _Rueckspiel, nachher: Path) -> None:
    for f in plan.verschieben:  # zuerst: eine Datei kann einem Ordner der Sicherung im Weg stehen
        ziel = nachher / f
        ziel.parent.mkdir(parents=True, exist_ok=True)
        os.replace(ordner / f, ziel)
    for d in sorted(plan.ordner_weg, key=lambda x: x.count("/"), reverse=True):
        p = ordner / d
        try:
            if p.is_dir() and not any(p.iterdir()):
                (nachher / d).mkdir(parents=True, exist_ok=True)
                p.rmdir()
        except OSError:
            pass
    for d in plan.ordner_neu:
        (ordner / d).mkdir(parents=True, exist_ok=True)
    for f in plan.ueberschreiben:
        _ersetzen(sicherung / f, ordner / f)


# ---------- Rowboat: eigene Seiten, Archiv, Manifest-Abgleich ----------

def _notizordner(konfig: Konfig) -> Path | None:
    """`<rowboat_wissen>/<rowboat_notizen>`, None wenn umgeleitet oder ungueltig."""
    wurzel = Path(konfig.rowboat_wissen)
    if rowboat._umgeleitet(wurzel / konfig.rowboat_notizen):
        return None
    try:
        return rowboat._unter(wurzel, konfig.rowboat_notizen)
    except ValueError:
        return None


def _eigener_bereich(rel: str) -> bool:
    return (rel in (ZUSTAND_REL, WAS_IST_REL) or rel.startswith(f"{MITBAUEN}/") or rel.startswith(f"{SPACES}/")
            or rel.startswith(f"{ABLAEUFE}/"))


def _selbstbild_markiert(pfad: Path) -> bool:
    try:
        daten = rowboat._frontmatter(pfad.read_bytes()[:4096].decode("utf-8", errors="replace"))
    except OSError:
        return False
    return daten is not None and str(daten.get(rowboat.MARKER_SCHLUESSEL, "")) == SEITEN_MARKER


@dataclass
class _Seitenplan:
    archivieren: dict[str, str]  # rel -> Manifest-Hash
    von_hand: list[str]
    hinweis: str | None = None


def _seitenplan(basis: Path | None, manifest: dict[str, str] | str) -> _Seitenplan:
    if basis is None:
        return _Seitenplan({}, [], "Rowboat-Notizordner umgeleitet oder ungültig – Seiten bleiben")
    if not isinstance(manifest, dict):
        return _Seitenplan({}, [], f"Notizen-Manifest {manifest} – eigene Seiten nicht bestimmbar, sie bleiben")
    archivieren: dict[str, str] = {}
    von_hand: list[str] = []
    for rel, h in sorted(manifest.items()):
        pfad = basis / rel
        if not _eigener_bereich(rel) or not rowboat._sicher(basis, pfad) or not pfad.is_file():
            continue
        if not _selbstbild_markiert(pfad):
            continue  # z. B. die alte Space-Notiz des alten Loops: gehoert ihm
        if rowboat._hash_von(pfad) == h:
            archivieren[rel] = h
        else:
            von_hand.append(rel)
    return _Seitenplan(archivieren, von_hand)


def _leere_eigene_ordner(basis: Path, weg: Sequence[str]) -> list[str]:
    """`Mitbauen/` und `Abläufe/` (samt Unterordnern), die nach dem Entfernen leer waeren."""
    weg_menge = set(weg)
    aus: list[str] = []
    for name in (MITBAUEN, ABLAEUFE):
        p = basis / name
        if not p.is_dir() or rowboat._umgeleitet(p):
            continue
        rest = [q for q in p.rglob("*") if q.is_file() and q.relative_to(basis).as_posix() not in weg_menge]
        if not rest:
            aus.append(name)
    return aus


def _archiv_laden(ordner: Path) -> dict[str, str]:
    geladen = rowboat._manifest_laden(ordner / ARCHIV / ARCHIV_MANIFEST)
    return geladen if isinstance(geladen, dict) else {}


def _seiten_archivieren(ordner: Path, basis: Path, mpfad: Path, manifest: dict[str, str], plan: _Seitenplan) -> None:
    """Ruling T10-b (c), Fix-Runde 1 (Minor 2): Seite fuer Seite -- Kopie ins Archiv und Archiv-Eintrag, dann den
    Eintrag aus dem Notizen-Manifest nehmen (geschrieben), dann die Datei loeschen. Scheitert das Loeschen, kommt der
    Eintrag zurueck. So bleibt nie eine geloeschte Seite mit Eintrag oder eine archivierte Seite ohne Archiv-Eintrag."""
    if not plan.archivieren:
        return
    archiv = ordner / ARCHIV
    eintraege = _archiv_laden(ordner)
    for rel, h in plan.archivieren.items():
        _ersetzen(basis / rel, archiv / ARCHIV_SEITEN / rel)
        eintraege[rel] = h
        rowboat._manifest_schreiben(archiv / ARCHIV_MANIFEST, eintraege)
        alt = manifest.pop(rel, None)
        rowboat._manifest_schreiben(mpfad, manifest)
        try:
            (basis / rel).unlink()
        except BaseException:
            if alt is not None:
                manifest[rel] = alt
                rowboat._manifest_schreiben(mpfad, manifest)
            raise
    for name in _leere_eigene_ordner(basis, []):
        for p in sorted((basis / name).rglob("*"), key=lambda q: len(q.parts), reverse=True):
            p.rmdir()
        (basis / name).rmdir()


def _abgleich_kandidaten(basis: Path | None, manifest: dict[str, str] | str,
                         ohne: Sequence[str] = ()) -> dict[str, str]:
    """Notizen, die der Loop unveraendert hinterlassen hat (Datei-Hash = aktueller Manifest-Eintrag)."""
    if basis is None or not isinstance(manifest, dict):
        return {}
    aus: dict[str, str] = {}
    for rel, h in manifest.items():
        pfad = basis / rel
        if rel in ohne or not rowboat._sicher(basis, pfad) or not pfad.is_file():
            continue
        if rowboat._hash_von(pfad) == h:
            aus[rel] = h
    return aus


def _abgleich(zurueck: dict[str, str] | str, kandidaten: Mapping[str, str]) -> list[str]:
    """Pfade, deren Eintrag im zurueckgespielten Manifest auf den jetzigen Hash gesetzt wird."""
    if not isinstance(zurueck, dict):
        return []
    return sorted(rel for rel, h in kandidaten.items() if zurueck.get(rel) != h)


@dataclass
class _Archivplan:
    zuruecklegen: dict[str, str]  # rel -> Hash
    ueber_loop_notiz: list[str]
    von_hand: list[str]
    defekt: list[str]
    hinweis: str | None = None


def _archivplan(ordner: Path, basis: Path | None, manifest: dict[str, str] | str) -> _Archivplan:
    eintraege = _archiv_laden(ordner)
    if not eintraege:
        return _Archivplan({}, [], [], [])
    if basis is None:
        return _Archivplan({}, [], [], [], "Rowboat-Notizordner umgeleitet oder ungültig – Archiv bleibt")
    if not isinstance(manifest, dict):
        return _Archivplan({}, [], [], [], f"Notizen-Manifest {manifest} – Archiv bleibt")
    plan = _Archivplan({}, [], [], [])
    for rel, h in sorted(eintraege.items()):
        quelle, ziel = ordner / ARCHIV / ARCHIV_SEITEN / rel, basis / rel
        if not quelle.is_file() or rowboat._hash_von(quelle) != h or not rowboat._sicher(basis, ziel):
            plan.defekt.append(rel)
            continue
        if os.path.lexists(ziel):
            jetzt = rowboat._hash_von(ziel)
            if jetzt != h and not (rel in manifest and manifest[rel] == jetzt):
                plan.von_hand.append(rel)
                continue
            if jetzt != h:
                plan.ueber_loop_notiz.append(rel)
        plan.zuruecklegen[rel] = h
    return plan


def _archiv_zuruecklegen(ordner: Path, basis: Path, mpfad: Path, manifest: dict[str, str], plan: _Archivplan) -> None:
    """Fix-Runde 1 (Minor 1): Seite fuer Seite -- Eintrag ins Notizen-Manifest (geschrieben), dann die Seite; scheitert
    das Kopieren, kommt der Eintrag wieder heraus. Danach aus dem Archiv: erst der Archiv-Eintrag, dann die Kopie."""
    if not plan.zuruecklegen:
        return
    archiv = ordner / ARCHIV
    rest = _archiv_laden(ordner)
    for rel, h in plan.zuruecklegen.items():
        vorher = manifest.get(rel)
        manifest[rel] = h
        rowboat._manifest_schreiben(mpfad, manifest)
        try:
            _ersetzen(archiv / ARCHIV_SEITEN / rel, basis / rel)
        except BaseException:
            if vorher is None:
                manifest.pop(rel, None)
            else:
                manifest[rel] = vorher
            rowboat._manifest_schreiben(mpfad, manifest)
            raise
        rest.pop(rel, None)
        rowboat._manifest_schreiben(archiv / ARCHIV_MANIFEST, rest)
        (archiv / ARCHIV_SEITEN / rel).unlink()
    if not rest:
        shutil.rmtree(archiv)  # nur noch leere Ordner und das eigene Manifest


def _archiv_zeilen(plan: _Archivplan, aus: Aus, *, trocken: bool) -> None:
    wird = "würde" if trocken else "wird"
    if plan.hinweis:
        aus(f"   {plan.hinweis}")
    for rel in plan.zuruecklegen:
        art = "über die unveränderte Notiz des alten Loops" if rel in plan.ueber_loop_notiz else "neu"
        aus(f"   zurücklegen: {rel} ({art}) mit Manifest-Eintrag")
    for rel in plan.von_hand:
        aus(f"   bleibt im Archiv: {rel} (die Datei dort ist von Hand angelegt oder geändert; nichts überschrieben)")
    for rel in plan.defekt:
        aus(f"   bleibt im Archiv: {rel} (Archivkopie fehlt, passt nicht zum Eintrag oder Weg unsicher)")
    if not (plan.zuruecklegen or plan.von_hand or plan.defekt or plan.hinweis):
        aus(f"   kein Archiv – nichts {wird} zurückgelegt")


# ---------- T10-d: Graph fuer den alten Loop lesbar machen ----------

def _literal_mengen(text: str, namen: Sequence[str]) -> dict[str, frozenset[str]]:
    """`NAME = frozenset({...})` / `set(...)` / `{...}` auf oberster Ebene, nur Literale aus Zeichenketten."""
    aus: dict[str, frozenset[str]] = {}
    for knoten in ast.parse(text).body:
        if not (isinstance(knoten, ast.Assign) and len(knoten.targets) == 1
                and isinstance(knoten.targets[0], ast.Name) and knoten.targets[0].id in namen):
            continue
        wert = knoten.value
        if (isinstance(wert, ast.Call) and isinstance(wert.func, ast.Name) and wert.func.id in ("frozenset", "set")
                and len(wert.args) == 1 and not wert.keywords):
            wert = wert.args[0]
        menge = ast.literal_eval(wert)  # ValueError bei allem, was kein Literal ist
        if not isinstance(menge, (set, frozenset, list, tuple)) or not all(isinstance(x, str) for x in menge):
            raise ValueError(f"{knoten.targets[0].id}: keine Menge aus Zeichenketten")
        aus[knoten.targets[0].id] = frozenset(menge)
    return aus


def alte_typen(repo: Path) -> tuple[dict[str, frozenset[str]], str]:
    """Knotentypen, Kantenarten und -quellen des alten Loops: per `ast` aus `<repo>/scripts/doku_loop/graph.py`
    (nie importiert), sonst die eingebaute Kopie. Rueckgabe `(typen, quelle)`."""
    pfad = Path(repo).joinpath(*ALTER_GRAPH_PY)
    try:
        mengen = _literal_mengen(pfad.read_text(encoding="utf-8"), tuple(ALTE_TYPEN_KOPIE))
    except FileNotFoundError:
        return dict(ALTE_TYPEN_KOPIE), "Kopie (scripts/doku_loop/graph.py fehlt)"
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError) as e:
        return dict(ALTE_TYPEN_KOPIE), f"Kopie ({'/'.join(ALTER_GRAPH_PY)} nicht lesbar: {type(e).__name__})"
    fehlt = [n for n in ALTE_TYPEN_KOPIE if n not in mengen]
    if fehlt:
        return dict(ALTE_TYPEN_KOPIE), f"Kopie (in {'/'.join(ALTER_GRAPH_PY)} kein Literal fuer {', '.join(fehlt)})"
    return mengen, f"ast aus {'/'.join(ALTER_GRAPH_PY)}"


def _graph_json_text(graph: Graph, erzeugt: str) -> str:
    """Genau das Format, das der Loop schreibt (`Graph.speichern`)."""
    return schreibvorgang.graph_text(graph, erzeugt)


@dataclass
class GraphBereinigung:
    neu: str | None  # None = nichts zu entfernen
    knoten_weg: dict[str, int]  # Knotentyp -> Anzahl
    kanten_weg: dict[str, int]  # Kantenart -> Anzahl (unbekannte Art/Quelle oder ein Endpunkt faellt weg)

    def zeile(self) -> str:
        k = ", ".join(f"{t} ({n})" for t, n in sorted(self.knoten_weg.items())) or "-"
        e = ", ".join(f"{t} ({n})" for t, n in sorted(self.kanten_weg.items())) or "-"
        return f"Knoten {k}; Kanten {e}"


def graph_bereinigen_plan(alt: bytes, typen: Mapping[str, frozenset[str]]) -> GraphBereinigung:
    """Rein: entfernt Knoten unbekannter Typen, Kanten unbekannter Art oder Quelle und Kanten, deren Endpunkt dabei
    wegfaellt; Digest neu, alles andere (auch `erzeugt`, Commits, Zusatzschluessel) bleibt. ValueError, wenn die Datei
    kein Graph der Version 1 ist (dann kann es der alte Loop ohnehin nicht lesen)."""
    doc = json.loads(alt.decode("utf-8"))
    if not isinstance(doc, dict) or doc.get("version") != ALTE_GRAPH_VERSION:
        raise ValueError("kein Graph der Version 1")
    knoten, kanten = doc.get("knoten", []), doc.get("kanten", [])
    if not isinstance(knoten, list) or not isinstance(kanten, list):
        raise ValueError("knoten/kanten keine Listen")
    knoten_weg: dict[str, int] = {}
    weg_ids: set[str] = set()
    bleiben_k = []
    for k in knoten:
        if not isinstance(k, dict):
            raise ValueError("Knoten ist kein Objekt")
        if str(k.get("typ")) in typen["KNOTEN_TYPEN"]:
            bleiben_k.append(k)
        else:
            knoten_weg[str(k.get("typ"))] = knoten_weg.get(str(k.get("typ")), 0) + 1
            weg_ids.add(str(k.get("id")))
    kanten_weg: dict[str, int] = {}
    bleiben_e = []
    for e in kanten:
        if not isinstance(e, dict):
            raise ValueError("Kante ist kein Objekt")
        if (str(e.get("art")) in typen["KANTEN_ARTEN"] and str(e.get("quelle")) in typen["KANTEN_QUELLEN"]
                and str(e.get("von")) not in weg_ids and str(e.get("nach")) not in weg_ids):
            bleiben_e.append(e)
        else:
            kanten_weg[str(e.get("art"))] = kanten_weg.get(str(e.get("art")), 0) + 1
    if not knoten_weg and not kanten_weg:
        return GraphBereinigung(None, {}, {})
    neu_doc = {**doc, "knoten": bleiben_k, "kanten": bleiben_e}
    neu_doc["digest"] = Graph.aus_doc(neu_doc).digest()
    return GraphBereinigung(json.dumps(neu_doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n",
                            knoten_weg, kanten_weg)


@dataclass
class _Graphvorpruefung:
    plan: GraphBereinigung | None  # None = keine Datei oder nichts zu tun
    quelle: str
    fehler: str | None = None  # gesetzt -> Rueckweg verweigert (Exit 2)


def _graph_vorpruefung(konfig: Konfig, *, git_pruefen: bool = True) -> _Graphvorpruefung:
    """Nur lesend. Ist etwas zu entfernen, muss die Datei sauber sein (keine fremde, nicht committete Aenderung) und
    der Kopf schreibbar (Zweig, keine laufende Git-Operation) -- sonst verweigert der Rueckweg VOR jedem Schritt."""
    repo = konfig.repo
    pfad = repo / GRAPH_JSON
    typen, quelle = alte_typen(repo)
    if not pfad.is_file():
        return _Graphvorpruefung(None, quelle)
    try:
        plan = graph_bereinigen_plan(pfad.read_bytes(), typen)
    except (OSError, ValueError) as e:
        return _Graphvorpruefung(None, quelle, f"{GRAPH_JSON} nicht lesbar ({type(e).__name__})")
    if plan.neu is None:
        return _Graphvorpruefung(None, quelle)
    if git_pruefen:
        try:
            fremd = umschreiben.fremde_aenderungen(repo, [GRAPH_JSON])
        except (umschreiben.GitFehler, OSError) as e:
            return _Graphvorpruefung(plan, quelle, f"git status für {GRAPH_JSON} gescheitert ({type(e).__name__})")
        if fremd:
            return _Graphvorpruefung(plan, quelle, f"{GRAPH_JSON} hat eine fremde, nicht committete Änderung – "
                                                   "erst klären (committen oder verwerfen), dann erneut")
        grund = schreibvorgang.kopf_pruefen(repo, konfig.zweig)
        if grund is not None:
            return _Graphvorpruefung(plan, quelle, f"Kopf: {grund} – {GRAPH_JSON} ließe sich nicht committen")
    return _Graphvorpruefung(plan, quelle)


def _graph_bereinigen(konfig: Konfig, jetzt: datetime, aus: Aus) -> bool:
    """Ruling T10-d: unbekannte Arten aus `docs/graph/graph.json` entfernen und nur diese Datei ueber den gemeinsamen
    Schreibvorgang committen (Zweig, CAS, Torwaechter, pathspec). True = erledigt oder nichts zu tun."""
    vor = _graph_vorpruefung(konfig)  # noch einmal unmittelbar davor: die CAS-Basis darf nichts Fremdes tragen
    if vor.fehler is not None:
        aus(f"d. Graph: {vor.fehler}")
        return False
    if vor.plan is None or vor.plan.neu is None:
        aus("d. Graph: nichts zu entfernen")
        return True
    repo = konfig.repo
    alt = (repo / GRAPH_JSON).read_bytes()
    plan = graph_bereinigen_plan(alt, alte_typen(repo)[0])
    if plan.neu is None:
        aus("d. Graph: nichts zu entfernen")
        return True
    drift = umschreiben.drift_fuer(konfig)
    try:
        vorher = drift(repo)
    except umschreiben.TorwaechterFehler:
        aus("d. Graph: Torwächter vorher gescheitert – nichts geschrieben")
        return False
    lauf = f"rueckweg-{jetzt:{ZEIT_FORMAT}}"
    erg = schreibvorgang.schreiben_pruefen_committen(
        repo, [schreibvorgang.Aenderung(GRAPH_JSON, alt, plan.neu)], GRAPH_BETREFF, lauf, vorher=vorher, drift=drift,
        melden=lambda z: aus(f"   {z}"), pp_datei=GRAPH_JSON, zweig=konfig.zweig, vermerk=schreibvermerk_pfad(konfig))
    if erg.commit is None and erg.uebersprungen not in (None, "commit_fremd_uebernommen"):
        aus(f"d. Graph: nicht committet ({erg.uebersprungen}); {GRAPH_JSON} zurückgesetzt")
        return False
    aus(f"d. Graph: entfernt {plan.zeile()} ({vor.quelle}); Commit {(erg.commit or erg.fremd_commit or '-')[:7]} "
        f"„{GRAPH_BETREFF}“")
    return True


def _graph_zeile(vor: _Graphvorpruefung) -> str:
    if vor.plan is None or vor.plan.neu is None:
        return (f"Graph ({GRAPH_JSON}): " + (vor.fehler or "nur Arten, die der alte Loop kennt")
                + f" (Typen: {vor.quelle})")
    return (f"Graph ({GRAPH_JSON}): Arten, die der alte Loop nicht kennt: {vor.plan.zeile()} – `--zurueck` entfernt "
            f"sie und committet nur diese Datei (Typen: {vor.quelle})" + (f"; Achtung: {vor.fehler}" if vor.fehler
                                                                            else ""))


# ---------- gemerkte alte Aktion ----------

def _alt_laden(ordner: Path) -> tuple[Aktion | None, str, str | None]:
    """`(alte Aktion, Grund, Name der gemerkten Sicherung)`."""
    pfad = ordner / ALT_DATEI
    try:
        roh = json.loads(pfad.read_bytes().decode("utf-8"))
        sicherung = roh.get("sicherung")
        return Aktion.aus_dict(roh["aktion"]), "ok", sicherung if isinstance(sicherung, str) and sicherung else None
    except FileNotFoundError:
        return None, "fehlt", None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None, "unlesbar", None


def _alt_schreiben(ordner: Path, alt: Aktion, jetzt: datetime, sicherung: Path) -> None:
    daten = {"version": 1, "aufgabe": AUFGABE, "gemerkt_um": jetzt.isoformat(timespec="seconds"),
             "sicherung": sicherung.name, "aktion": alt.als_dict()}
    from .stand import _atomar

    _atomar(ordner / ALT_DATEI, json.dumps(daten, ensure_ascii=False, indent=1) + "\n")


# ---------- Befehle ----------

def _lesen(backend: Backend, aus: Aus) -> list[Aktion] | None:
    try:
        return backend.lesen(AUFGABE)
    except AufgabeFehler as e:
        aus(f"Aufgabe {AUFGABE}: nicht lesbar ({e})")
        return None


def _aktion_zeigen(aus: Aus, wann: str, aktionen: Sequence[Aktion]) -> None:
    if not aktionen:
        aus(f"Aktion {wann}: keine")
    for a in aktionen:
        aus(f"Aktion {wann}: {a.zeile()}")


def _sperre_text(ordner: Path) -> str:
    z = sperre_zustand(ordner)
    return "frisch (ein Lauf hält tick.lock)" if z == "belegt" else z


def pruefen(konfig: Konfig, *, backend: Backend, aus: Aus = print) -> int:
    """Nur lesend: Aktion, Einordnung, gemerkte Aktion, Sicherung, Archiv, Sperre."""
    ordner = lokaler_ordner(konfig)
    aktionen = _lesen(backend, aus)
    if aktionen is None:
        return 1
    aus(f"Aufgabe: {AUFGABE}")
    _aktion_zeigen(aus, "jetzt", aktionen)
    aus(f"Loop: {einordnen(aktionen)}")
    alt, grund, gemerkt = _alt_laden(ordner)
    aus(f"Gemerkte alte Aktion ({ALT_DATEI}): " + (f"vorhanden (Sicherung {gemerkt or '-'})" if alt else grund))
    s = sicherungen(ordner)
    aus(f"Sicherung: {s[-1].name} (neueste von {len(s)})" if s else "Sicherung: keine")
    archiv = _archiv_laden(ordner)
    aus(f"Archiv erzählter Seiten ({ARCHIV}): {len(archiv)} Seite(n) vorhanden" if archiv
        else f"Archiv erzählter Seiten ({ARCHIV}): keins")
    aus(f"Sperre tick.lock: {_sperre_text(ordner)}")
    aus(_graph_zeile(_graph_vorpruefung(konfig)))
    return 0


def umschalten(konfig: Konfig, *, backend: Backend, python: str | None = None, trocken: bool = False,
               jetzt: datetime | None = None, aus: Aus = print,
               import_pruefen: ImportPruefen = import_pruefen) -> int:
    """`python` None = `STANDARD_PYTHON` (nie `sys.executable`); ein angegebener Pfad (`--python`) muss existieren und
    `selbstbild.doku` ohne PYTHONPATH importieren (`import_pruefen`)."""
    python = python or STANDARD_PYTHON
    ordner = lokaler_ordner(konfig)
    jetzt = jetzt or datetime.now()
    aktionen = _lesen(backend, aus)
    if aktionen is None:
        return 1
    _aktion_zeigen(aus, "jetzt", aktionen)
    art = einordnen(aktionen)
    aus(f"Loop: {art}")
    if art != "alt":
        aus("Abbruch: umgeschaltet wird nur von der Aktion des alten Loops – nichts geändert")
        return 2
    alt = aktionen[0]
    wurzel = konfig.repo
    if alt.working_directory and not _gleicher_pfad(alt.working_directory, wurzel):
        aus(f"Abbruch: WorkingDirectory der Aufgabe ist nicht die Wurzel {wurzel} – nichts geändert")
        return 2
    try:
        neu = neue_aktion(python, wurzel, alt.working_directory or None)
    except ValueError as e:
        aus(f"Abbruch: {e} – nichts geändert")
        return 2
    if not ordner.is_dir():
        aus(f"Abbruch: lokaler Ordner {ordner} fehlt – nichts geändert")
        return 2
    if not (wurzel / LOG_REL[0]).is_dir():
        aus(f"Abbruch: {wurzel / LOG_REL[0]} fehlt (cmd könnte das Log nicht öffnen) – nichts geändert")
        return 2
    fehler = import_pruefen(python, wurzel)
    if fehler is not None:
        aus(f"Abbruch: {python} kann selbstbild ohne PYTHONPATH nicht importieren ({fehler}) – nichts geändert")
        return 2
    aus(f"Python: {python} (import selbstbild.doku ohne PYTHONPATH: ok)")
    ziel = _freier_name(ordner.parent, f"{ordner.name}-sicherung-{jetzt:{ZEIT_FORMAT}}")
    basis = _notizordner(konfig)
    mpfad = rowboat.manifest_pfad(konfig)

    if trocken:
        if sperre_frisch(ordner):
            aus("Sperre tick.lock: frisch – ein echter Aufruf bräche jetzt mit Exit 2 ab, ohne etwas zu ändern")
        else:
            aus(f"Sperre tick.lock: {_sperre_text(ordner)} – ein echter Aufruf hielte sie während des Umschaltens")
        dateien, ordnerliste = _inhalt(ordner)
        groesse = sum((ordner / f).stat().st_size for f in dateien)
        aus(f"1. Sicherung: {ordner} -> {ziel} ({len(dateien)} Dateien, {len(ordnerliste)} Ordner, {groesse} Bytes; "
            f"ohne messbaum-*, tick.lock, {ARCHIV}, {ALT_DATEI}, seiten-fehlversuche-*)")
        for d in ordnerliste:
            aus(f"   sichern: {d}/")
        for f in dateien:
            aus(f"   sichern: {f}")
        aus(f"2. Alte Aktion merken: {ordner / ALT_DATEI}")
        aus(f"3. Neue Aktion setzen: {neu.zeile()}")
        aus(f"   Execute: {neu.execute}")
        aus(f"   Arguments: {neu.arguments}")
        aus(f"   WorkingDirectory: {neu.working_directory}")
        aus("   Trigger, Limits und Principal bleiben unverändert")
        aus(f"4. Archiv erzählter Seiten zurücklegen ({ordner / ARCHIV}):")
        _archiv_zeilen(_archivplan(ordner, basis, rowboat._manifest_laden(mpfad)), aus, trocken=True)
        aus("Trocken: nichts geschrieben, Aufgabe unverändert")
        return 0

    with sperre(ordner) as frei:
        if not frei:
            aus("Abbruch: tick.lock ist frisch (ein Lauf arbeitet gerade) – nichts geändert")
            return 2
        n = _sichern(ordner, ziel)
        aus(f"1. Sicherung: {ziel} ({n} Dateien, ohne messbaum-*)")
        _alt_schreiben(ordner, alt, jetzt, ziel)
        aus(f"2. Alte Aktion gemerkt: {ordner / ALT_DATEI}")
        try:
            backend.setzen(AUFGABE, neu)
            danach = backend.lesen(AUFGABE)
            if danach != [neu]:
                raise AufgabeFehler("nach dem Setzen steht eine andere Aktion in der Aufgabe")
        except AufgabeFehler as e:
            aus(f"Fehler beim Setzen der Aktion: {e}")
            try:
                stand = backend.lesen(AUFGABE)
            except AufgabeFehler:
                stand = None
            if stand == [alt]:
                (ordner / ALT_DATEI).unlink()
                aus(f"Aufgabe unverändert (alte Aktion); {ALT_DATEI} entfernt; die Sicherung {ziel.name} bleibt")
            else:
                aus(f"Aktion der Aufgabe unklar; {ALT_DATEI} und die Sicherung {ziel.name} bleiben – "
                    "mit `aufgabe --pruefen` nachsehen, sonst `aufgabe --zurueck`")
            return 1
        aus(f"3. Neue Aktion gesetzt: {neu.zeile()}")
        aus(f"4. Archiv erzählter Seiten ({ARCHIV}):")
        manifest = rowboat._manifest_laden(mpfad)
        plan = _archivplan(ordner, basis, manifest)
        _archiv_zeilen(plan, aus, trocken=False)
        try:
            if basis is not None and isinstance(manifest, dict):
                _archiv_zuruecklegen(ordner, basis, mpfad, manifest, plan)
        except OSError as e:
            aus(f"Fehler beim Zurücklegen des Archivs ({type(e).__name__}); die Aufgabe ist umgeschaltet, der Rest "
                "des Archivs bleibt – der neue Loop erzählt fehlende Seiten neu")
            return 1
    aus(f"Fertig: Aufgabe auf den neuen Loop umgeschaltet; zurück mit `doku aufgabe {wurzel} --zurueck`")
    return 0


def _sicherung_waehlen(ordner: Path, gemerkt: str | None, aus: Aus) -> Path | None:
    """Fix-Runde 1 (Minor 3): die in aufgabe-alt.json gemerkte Sicherung; weicht die neueste ab -> Warnung. Fehlt die
    gemerkte, die neueste (mit Warnung). Keine -> None."""
    alle = sicherungen(ordner)
    neueste = alle[-1] if alle else None
    if gemerkt is not None:
        p = ordner.parent / gemerkt
        if p in alle:
            if neueste is not None and neueste != p:
                aus(f"Warnung: es gibt eine neuere Sicherung {neueste.name}; zurückgespielt wird die beim Umschalten "
                    f"gemerkte {p.name}")
            return p
        if neueste is not None:
            aus(f"Warnung: die gemerkte Sicherung {gemerkt} fehlt – zurückgespielt wird die neueste {neueste.name}")
    elif neueste is not None:
        aus(f"Warnung: {ALT_DATEI} nennt keine Sicherung – zurückgespielt wird die neueste {neueste.name}")
    return neueste


def zurueck(konfig: Konfig, *, backend: Backend, trocken: bool = False, jetzt: datetime | None = None,
            aus: Aus = print) -> int:
    """Fix-Runde 1 (Item 2): Reihenfolge c (Seiten archivieren), b (Sicherung zurueckspielen), d (Graph bereinigen,
    T10-d), ZULETZT a (alte Aktion). Bis a bleibt der neue Loop geplant; jeder Schritt ist wiederholbar, ein erneutes
    `--zurueck` macht fertig."""
    ordner = lokaler_ordner(konfig)
    jetzt = jetzt or datetime.now()
    alt, grund, gemerkt = _alt_laden(ordner)
    if alt is None:
        aus(f"Abbruch: {ordner / ALT_DATEI} {grund} – nichts geändert")
        return 2
    if einordnen([alt]) != "alt":
        aus(f"Abbruch: {ALT_DATEI} enthält nicht die Aktion des alten Loops – nichts geändert")
        return 2
    sicherung = _sicherung_waehlen(ordner, gemerkt, aus)
    if sicherung is None:
        aus(f"Abbruch: keine Sicherung {ordner.name}-sicherung-* neben {ordner} – nichts geändert")
        return 2
    aktionen = _lesen(backend, aus)
    if aktionen is None:
        return 1
    _aktion_zeigen(aus, "jetzt", aktionen)
    art = einordnen(aktionen)
    aus(f"Loop: {art}")
    setzen = aktionen != [alt]
    if setzen and art != "neu":
        aus("Abbruch: die Aufgabe hat weder die neue noch die gemerkte alte Aktion – nichts geändert")
        return 2
    vor = _graph_vorpruefung(konfig)
    if vor.fehler is not None:  # T10-d: VOR jedem Schritt, laut
        aus(f"Abbruch: {vor.fehler} – nichts geändert, die Aufgabe bleibt beim neuen Loop")
        return 2
    basis = _notizordner(konfig)
    mpfad = rowboat.manifest_pfad(konfig)
    nachher = _freier_name(ordner.parent, f"{ordner.name}-nachher-{jetzt:{ZEIT_FORMAT}}")

    if not trocken:
        return _zurueck_echt(konfig, backend, alt, setzen, sicherung, basis, mpfad, nachher, jetzt, aus)

    aus(f"Sperre tick.lock: {_sperre_text(ordner)}" + (" – ein echter Aufruf bräche jetzt mit Exit 2 ab"
                                                         if sperre_frisch(ordner) else ""))
    manifest = rowboat._manifest_laden(mpfad)
    plan = _seitenplan(basis, manifest)
    aus(f"c. Eigene Rowboat-Seiten archivieren (Archiv {ordner / ARCHIV}):")
    if plan.hinweis:
        aus(f"   {plan.hinweis}")
    for rel, h in plan.archivieren.items():
        aus(f"   archivieren und entfernen: {rel} (unverändert; Manifest-Eintrag {h[:12]}… ins Archiv)")
    for rel in plan.von_hand:
        aus(f"   bleibt: {rel} (von Hand geändert)")
    if basis is not None:
        for name in _leere_eigene_ordner(basis, list(plan.archivieren)):
            aus(f"   leeren Ordner entfernen: {name}/")
    if not (plan.archivieren or plan.von_hand or plan.hinweis):
        aus("   keine eigenen Seiten")
    rest = dict(manifest) if isinstance(manifest, dict) else manifest
    if isinstance(rest, dict):
        for rel in plan.archivieren:
            rest.pop(rel, None)
    kandidaten = _abgleich_kandidaten(basis, rest)
    rp = _rueckspiel_plan(ordner, sicherung)
    aus(f"b. Sicherung zurückspielen: {sicherung} -> {ordner} (ohne messbaum-*; behalten: seiten-fehlversuche-*, "
        f"{ARCHIV})")
    for f in rp.ueberschreiben:
        aus(f"   zurückspielen: {f}")
    if rp.gleich:
        aus(f"   schon gleich: {len(rp.gleich)} Datei(en)")
    for f in rp.verschieben:
        aus(f"   nach {nachher.name}: {f} (nicht in der Sicherung; nie gelöscht)")
    zurueck_manifest = rowboat._manifest_laden(sicherung / mpfad.name)
    for rel in _abgleich(zurueck_manifest, kandidaten):
        aus(f"   Manifest-Abgleich: {rel} behält den jetzigen Hash (vom Loop unverändert hinterlassen)")
    if vor.plan is not None and vor.plan.neu is not None:
        aus(f"d. Graph bereinigen: aus {GRAPH_JSON} entfernen {vor.plan.zeile()} (Typen: {vor.quelle}); nur diese "
            f"Datei committen: „{GRAPH_BETREFF}“")
    else:
        aus(f"d. Graph: nichts zu entfernen (Typen: {vor.quelle})")
    aus(f"a. Zuletzt die alte Aktion setzen: {alt.zeile()}" if setzen
        else "a. Aktion ist schon die alte – nichts zu setzen")
    if setzen:
        aus(f"   Execute: {alt.execute}")
        aus(f"   Arguments: {alt.arguments}")
        aus(f"   WorkingDirectory: {alt.working_directory}")
    aus(f"   {ALT_DATEI} -> {nachher.name}")
    aus("Trocken: nichts geschrieben, Aufgabe unverändert")
    return 0


def _zurueck_echt(konfig: Konfig, backend: Backend, alt: Aktion, setzen: bool, sicherung: Path, basis: Path | None,
                  mpfad: Path, nachher: Path, jetzt: datetime, aus: Aus) -> int:
    ordner = lokaler_ordner(konfig)
    erneut = "die Aufgabe bleibt beim neuen Loop – `aufgabe --zurueck` erneut ausführen (wiederholbar)"
    with sperre(ordner) as frei:
        if not frei:
            aus("Abbruch: tick.lock ist frisch (ein Lauf arbeitet gerade) – nichts geändert")
            return 2
        try:
            # c. (vor b: das Manifest der Sicherung kennt die neuen Seiten nicht)
            manifest = rowboat._manifest_laden(mpfad)
            plan = _seitenplan(basis, manifest)
            if plan.hinweis:
                aus(f"c. {plan.hinweis}")
            if basis is not None and isinstance(manifest, dict):
                leer = _leere_eigene_ordner(basis, list(plan.archivieren))
                _seiten_archivieren(ordner, basis, mpfad, manifest, plan)
                aus(f"c. Eigene Seiten: {len(plan.archivieren)} archiviert und entfernt, {len(plan.von_hand)} von "
                    f"Hand geändert (bleiben); leere Ordner entfernt: {', '.join(leer) or '-'}")
                for rel in plan.archivieren:
                    aus(f"   archiviert: {rel}")
                for rel in plan.von_hand:
                    aus(f"   bleibt (von Hand geändert): {rel}")
            kandidaten = _abgleich_kandidaten(basis, manifest)
            # b.
            rp = _rueckspiel_plan(ordner, sicherung)
            _zurueckspielen(ordner, sicherung, rp, nachher)
            aus(f"b. Sicherung {sicherung.name} zurückgespielt: {len(rp.ueberschreiben)} Datei(en) ersetzt, "
                f"{len(rp.gleich)} schon gleich, {len(rp.verschieben)} nach {nachher.name} verschoben")
            zurueck_manifest = rowboat._manifest_laden(mpfad)
            abgleich = _abgleich(zurueck_manifest, kandidaten)
            if abgleich and isinstance(zurueck_manifest, dict):
                for rel in abgleich:
                    zurueck_manifest[rel] = kandidaten[rel]
                rowboat._manifest_schreiben(mpfad, zurueck_manifest)
            aus(f"   Manifest-Abgleich: {len(abgleich)} Notiz(en) behalten ihren jetzigen Hash"
                + (f" ({', '.join(abgleich)})" if abgleich else ""))
            # d. (T10-d)
            if not _graph_bereinigen(konfig, jetzt, aus):
                aus(f"Fehler im Graph-Schritt; {erneut}")
                return 1
        except OSError as e:
            aus(f"Fehler im Rückweg ({type(e).__name__}); {erneut}")
            return 1
        # a. zuletzt
        if setzen:
            try:
                backend.setzen(AUFGABE, alt)
                if backend.lesen(AUFGABE) != [alt]:
                    raise AufgabeFehler("nach dem Setzen steht eine andere Aktion in der Aufgabe")
            except AufgabeFehler as e:
                aus(f"Fehler beim Setzen der alten Aktion: {e}; Zustand, Seiten und Graph sind schon zurück – "
                    f"{ALT_DATEI} bleibt, `aufgabe --zurueck` erneut ausführen")
                return 1
            aus(f"a. Alte Aktion gesetzt: {alt.zeile()}")
        else:
            aus("a. Aktion war schon die alte")
        try:
            nachher.mkdir(parents=True, exist_ok=True)
            os.replace(ordner / ALT_DATEI, nachher / ALT_DATEI)
        except OSError as e:
            aus(f"Hinweis: {ALT_DATEI} nicht verschoben ({type(e).__name__}); der Rückweg ist trotzdem fertig")
    aus(f"seiten-fehlversuche-* behalten; {ALT_DATEI} liegt in {nachher.name}")
    aus("Fertig: Aufgabe, Zustand, Seiten und Graph auf den alten Loop zurückgesetzt")
    return 0
