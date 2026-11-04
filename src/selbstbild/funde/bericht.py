"""Befehl `python -m selbstbild funde <wurzel>` und der lokale Bericht (Stufe 5, Task 6).

Ablauf: Funde sammeln (Doku-Pruefliste, Analyse, Goldsatz-Tor) plus Konfig-Inventar, nach Ziel verteilen und je Ziel
`sync.abgleichen`. Ausgaben: der Markdown-Bericht `<Selbstbild-Stand>/funde-bericht.md` (atomar geschrieben) und eine
JSON-Zusammenfassung auf stdout (sonst nichts auf stdout). Beide enthalten nur Namen, nie Werte; alles, was nach einem
Schluessel aussieht, wird vor der Ausgabe geschwaerzt und nur als Mustername gezaehlt.

Modi je Ziel: `senden` (Abgleich mit Schreiben), `trocken` (Konto da: GitHub wird nur gelesen, Ruling S5-16),
`offline` (`--offline`: nie gh) und `bericht` (Ziel ohne Konto, S5-1). Exit (Ruling S5-17): 0 ok; 2 bei `--senden`
ohne `issues:`-Ziel oder mit einem Ziel ohne Konto (vor jedem Lesen), bei ungueltiger Regeldatei; 1 bei jedem
`Bericht.fehler` (auch `ziel_gesperrt`), `GitHubFehler` oder unbekanntem Commit -- Bericht und JSON entstehen trotzdem.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import IO, Any

import yaml

from ..doku.konfig import Konfig, KonfigFehler, aus_regeln
from ..doku.schluessel import schluessel_muster, schwaerzen
from ..quelle import _git_args, _git_env
from ..regeln import REGELDATEI, Regeln, regeln_laden
from ..speicher import pfad_kennung, stand_basis
from ..vermessen import stand_finden
from .github import GitHub, GitHubFehler
from .goldsatz_tor import ToterCodeTor, tor_tot
from .inventar import _graph_db, inventar, inventar_fund, inventar_tabelle
from .sammeln import UNLESBAR_PRAEFIX, _aussen_name, funde_sammeln, geprueft_repos
from .sync import (
    RATE_LIMIT_ARTEN,
    RATE_LIMIT_HINWEIS,
    SCHREIB_ABSTAND_S,
    Bericht,
    IssueStand,
    Ziel,
    abgleichen,
    nach_ziel,
    stand_pfad,
    unlesbare_arten,
    ziele_aus_regeln,
)
from .text import _art_name, _zelle
from .typen import ARTEN, Fund

BERICHT_NAME = "funde-bericht.md"
GIT_TIMEOUT_S = 30
SCHWEREN = ("hoch", "mittel", "niedrig")
_SHA = re.compile(r"[0-9a-f]{7,64}")
# Aktion.art -> Art der Zusammenfassung (anlegen, aktualisieren, schliessen; unterdrueckt kommt aus dem Bericht)
_AKTION_ART = {
    "anlegen": "anlegen", "sammel_anlegen": "anlegen",
    "bearbeiten": "aktualisieren", "sammel_bearbeiten": "aktualisieren",
    "oeffnen": "aktualisieren", "sammel_oeffnen": "aktualisieren",
    "schliessen": "schliessen", "sammel_schliessen": "schliessen",
}
_GEPLANT = ("ausgefuehrt", "trocken")


# ---------- Bausteine ----------

def goldsatz_standard(aussen: str) -> Path:
    """`docs/selbstbild/goldsatz-<aussen>.yaml` im Repo, das das Paket `selbstbild` enthaelt (Ruling S5-15)."""
    import selbstbild

    return Path(selbstbild.__file__).resolve().parents[2] / "docs" / "selbstbild" / f"goldsatz-{aussen}.yaml"


def tor_laden(pfad: Path) -> tuple[ToterCodeTor, list[str]]:
    """Tor fuer toten Code; fehlt oder ist der Goldsatz kaputt, ist das Tor zu (0/0), mit Notiz, kein Fehler."""
    if not pfad.is_file():
        return ToterCodeTor(False, 0, 0, "gesperrt: Goldsatz fehlt"), [
            f"Goldsatz fehlt ({pfad.name}): Tor für toten Code geschlossen"]
    try:
        return tor_tot(pfad), []
    except (OSError, UnicodeDecodeError, yaml.YAMLError, AttributeError, TypeError, ValueError) as e:
        return ToterCodeTor(False, 0, 0, "gesperrt: Goldsatz unlesbar"), [
            f"Goldsatz unlesbar ({pfad.name}, {type(e).__name__}): Tor für toten Code geschlossen"]


def tor_text(tor: ToterCodeTor) -> str:
    return f"toter Code: {'offen' if tor.offen else 'gesperrt'} ({tor.ja}/{tor.beurteilt} bestätigt)"


def commit_holen(wurzel: Path) -> str | None:
    """`git rev-parse --short=7 HEAD` von `wurzel` mit Zeitgrenze; None, wenn es scheitert."""
    try:
        erg = subprocess.run(_git_args("rev-parse", "--short=7", "HEAD"), cwd=wurzel, env=_git_env(),
                             capture_output=True, stdin=subprocess.DEVNULL, timeout=GIT_TIMEOUT_S, check=False,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    sha = erg.stdout.decode("ascii", "replace").strip().lower()
    return sha if erg.returncode == 0 and _SHA.fullmatch(sha) else None


def _vermessungsbericht(wurzel: Path, stand_unter: Path | None) -> dict[str, Any] | None:
    """Der gespeicherte Vermessungsbericht (`meta.bericht` in graph.sqlite), nur lesend mit `mode=ro`; None, wenn
    es keinen lesbaren gibt."""
    db = _graph_db(wurzel, stand_unter)
    if db is None:
        return None
    try:
        c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
        try:
            z = c.execute("SELECT v FROM meta WHERE k=?", (json.dumps("bericht"),)).fetchone()
        finally:
            c.close()
        if z is None:
            return None
        bericht = json.loads(json.loads(z[0]))
    except (sqlite3.Error, OSError, ValueError, TypeError):
        return None
    return bericht if isinstance(bericht, dict) else None


def _gemessene_submodule(wurzel: Path, stand_unter: Path | None) -> tuple[frozenset[str], frozenset[str]] | None:
    """(vermessene, fehlende) Submodule aus dem gespeicherten Vermessungsbericht; None ohne lesbaren Bericht."""
    bericht = _vermessungsbericht(wurzel, stand_unter)
    if bericht is None or "fehlende_submodule" not in bericht:
        return None  # kein Bericht oder alter Bericht: fehlende Pins nicht erkennbar
    try:
        return (frozenset(map(str, bericht.get("submodule") or ())),
                frozenset(map(str, bericht["fehlende_submodule"] or ())))
    except TypeError:
        return None


def stand_abweichung(wurzel: Path, commit: str | None, stand_unter: Path | None = None) -> str | None:
    """Notiz `analyse-stand <sha7> != HEAD <sha7>`, wenn der vermessene Commit nicht HEAD ist (sonst None).
    Nur ein Hinweis, aendert nichts am Verhalten."""
    bericht = _vermessungsbericht(wurzel, stand_unter)
    gemessen = str(bericht.get("commit") or "").lower() if bericht else ""
    if not commit or not _SHA.fullmatch(gemessen) or gemessen.startswith(commit.lower()):
        return None
    return f"analyse-stand {gemessen[:7]} != HEAD {commit[:7]}"


def analyse_commit_lesen(wurzel: Path, stand_unter: Path | None = None) -> str | None:
    """Der vermessene Commit (`meta.bericht.commit`, nur lesend); None, wenn es keinen lesbaren gibt."""
    bericht = _vermessungsbericht(Path(wurzel), stand_unter)
    gemessen = str(bericht.get("commit") or "").lower() if bericht else ""
    return gemessen if _SHA.fullmatch(gemessen) else None


def analyse_stand_pruefen(wurzel: Path, analyse: str, head: str) -> list[str]:
    """Notiz `quelle-unlesbar:analyse-stand: ...`, wenn der vermessene Commit weder HEAD noch ein Vorfahr davon ist
    (`git merge-base --is-ancestor`, mit Zeitgrenze); sonst leer. Scheitert die Pruefung, gilt der Stand ebenfalls
    als unlesbar (fail-closed). `sync.unlesbare_arten` macht daraus doppelung und tot: kein Schliessen."""
    a, h = analyse.lower(), head.lower()
    if a.startswith(h) or h.startswith(a):
        return []
    grund: str
    try:
        erg = subprocess.run(_git_args("merge-base", "--is-ancestor", a, h), cwd=wurzel, env=_git_env(),
                             capture_output=True, stdin=subprocess.DEVNULL, timeout=GIT_TIMEOUT_S, check=False,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        grund = "Prüfung überschritt die Zeitgrenze"
    except (OSError, subprocess.SubprocessError) as e:
        grund = f"Prüfung gescheitert ({type(e).__name__})"
    else:
        if erg.returncode == 0:
            return []
        grund = "kein Vorfahr" if erg.returncode == 1 else f"Prüfung gescheitert (git exit {erg.returncode})"
    return [f"{UNLESBAR_PRAEFIX}analyse-stand: Analyse-Commit {a[:7]} vs. HEAD {h[:7]}: {grund}; "
            "Doppelungen und toter Code werden nicht geschlossen"]


def geprueft_ermitteln(wurzel: Path, *, doku_konfig: Konfig,
                       stand_unter: Path | None = None) -> tuple[frozenset[str], list[str]]:
    """Code-Repos, deren verschwundene Funde geschlossen werden duerfen (Ruling S5-10).

    `geprueft_repos` (Checkout da) minus die Submodule, die die Vermessung als fehlend meldet (Pin-Commit nicht im
    Submodul), und nur, was die Vermessung wirklich vermessen hat. Ohne lesbaren Vermessungsbericht gilt jedes
    Submodul als nicht geprueft: nur das aeussere Repo, mit Notiz."""
    basis = geprueft_repos(wurzel, doku_konfig=doku_konfig)
    aussen = _aussen_name(Path(wurzel))
    gemessen = _gemessene_submodule(Path(wurzel), stand_unter)
    if gemessen is None:
        return basis & {aussen}, [
            "Submodule nicht als geprüft gezählt: kein lesbarer Vermessungsbericht (erst `python -m selbstbild "
            "vermessen`); ihre Issues bleiben offen"]
    vermessen_, fehlend = gemessen
    return frozenset(r for r in basis if r == aussen or (r in vermessen_ and r not in fehlend)), []


def bericht_ordner_standard(wurzel: Path) -> Path:
    """Der Selbstbild-Stand der Wurzel (Ordner von graph.sqlite), sonst `<stand_basis>/<aussen>-<kennung>`."""
    graph = stand_finden(wurzel)
    if graph is not None:
        return graph.parent
    return stand_basis() / f"{_aussen_name(wurzel)}-{pfad_kennung(wurzel)}"


def atomar_schreiben(pfad: Path, text: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=pfad.name + ".", suffix=".tmp", dir=pfad.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        os.replace(tmp, pfad)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------- Ergebnis je Ziel ----------

@dataclass
class ZielErgebnis:
    ziel: Ziel
    modus: str  # senden | trocken | offline | bericht
    funde: int
    senden: bool = False
    bericht: Bericht | None = None
    fehler: str | None = None
    notizen: list[str] = field(default_factory=list)

    def zusammenfassung(self) -> dict[str, Any]:
        b = self.bericht
        zaehler = Counter({"anlegen": 0, "aktualisieren": 0, "schliessen": 0})
        je_art: Counter[str] = Counter()
        nicht_ausgefuehrt = 0
        for a in (b.aktionen if b else []):
            if a.status in _GEPLANT:
                zaehler[_AKTION_ART.get(a.art, a.art)] += 1
                je_art[a.art] += 1
            else:
                nicht_ausgefuehrt += 1
        zaehler["unterdrueckt"] = len(b.unterdrueckt) if b else 0
        return {
            "ziel": self.ziel.repo, "modus": self.modus, "funde": self.funde,
            "aktionen": dict(zaehler), "aktionen_je_art": dict(sorted(je_art.items())),
            "nicht_ausgefuehrt": nicht_ausgefuehrt,
            "neu": b.neu if b else 0, "unveraendert": b.unveraendert if b else 0,
            "ueber_grenze": len(b.ueber_grenze) if b else 0, "gesperrt": len(b.gesperrt) if b else 0,
            "ausgelassen": len(b.ausgelassen) if b else 0,
            "fehler": self.fehler, "notizen": [*self.notizen, *(b.notizen if b else [])],
        }

    def text(self) -> str:
        if self.bericht is not None:
            text = self.bericht.text()
            if self.modus == "offline":
                text = text.replace("(trocken (nichts gesendet))", "(offline (nichts gesendet))", 1)
            if self.fehler and not self.bericht.fehler:
                text += f"\n**Abgebrochen:** `{self.fehler}`\n"
            text += "".join(f"- Hinweis: {n}\n" for n in self.notizen)
            return text
        z = [f"## Issues → {self.ziel.repo} ({self.modus})", ""]
        if self.fehler:
            z += [f"**Abgebrochen:** `{self.fehler}`", ""]
        z += [f"- Hinweis: {n}" for n in self.notizen]
        return "\n".join(z) + "\n"


def _abgleich_je_ziel(ziel: Ziel, fs: list[Fund], *, senden: bool, offline: bool, commit: str | None,
                      stand: IssueStand, heute: date, geprueft: frozenset[str], unlesbar: frozenset[str],
                      gh_fabrik: Callable[[str], GitHub], schliess_commit: str | None = None,
                      pause: Callable[[float], None] | None = None,
                      schreib_abstand_s: float = SCHREIB_ABSTAND_S) -> ZielErgebnis:
    """`commit` (HEAD) entscheidet, ob abgeglichen wird; `schliess_commit` (der vermessene Commit, sonst HEAD)
    steht in den Schliess-Kommentaren."""
    if senden:
        modus = "senden"
    elif offline:
        modus = "offline"
    else:
        modus = "trocken" if ziel.konto else "bericht"
    erg = ZielErgebnis(ziel, modus, len(fs), senden=senden)
    if commit is None:
        erg.fehler = "commit_unbekannt"
        erg.notizen.append("HEAD-Commit nicht lesbar (git rev-parse): kein Abgleich, nichts gesendet")
        return erg
    gh: GitHub | None = None
    teil = Bericht(ziel=ziel.repo, senden=senden)  # wird von abgleichen gefuellt, auch wenn es mittendrin wirft
    try:
        if ziel.konto and not offline:
            gh = gh_fabrik(ziel.konto)
        erg.bericht = abgleichen(fs, ziel, gh, stand, heute=heute, senden=senden, commit=schliess_commit or commit,
                                 geprueft=geprueft, unlesbar=unlesbar, bericht=teil,
                                 pause=pause if pause is not None else time.sleep,
                                 schreib_abstand_s=schreib_abstand_s)
    except GitHubFehler as e:
        erg.fehler = e.art
        _abbruch(erg, teil, f"GitHub-Fehler `{e.art}`")
        if e.art in RATE_LIMIT_ARTEN:
            erg.notizen.append(RATE_LIMIT_HINWEIS)
        return erg
    except ValueError as e:  # ungueltiges Konto o. ae.; nie ein Wert im Text
        erg.fehler = "ungueltig"
        _abbruch(erg, teil, f"Abgleich nicht möglich ({type(e).__name__})")
        return erg
    except Exception as e:  # noqa: BLE001 -- ein Ziel darf die anderen nie mitreissen (KeyboardInterrupt nicht)
        erg.fehler = f"intern:{type(e).__name__}"
        _abbruch(erg, teil, f"interner Fehler ({type(e).__name__})")
        return erg
    erg.fehler = erg.bericht.fehler or (f"stand_nicht_gespeichert:{erg.bericht.stand_fehler}"
                                        if erg.bericht.stand_fehler else None)
    return erg


def _abbruch(erg: ZielErgebnis, teil: Bericht, grund: str) -> None:
    """Abbruch eines Ziels: ehrlich sagen, ob schon geschrieben wurde. Nur der Typ/die Art, nie ein Fehlertext."""
    geschrieben = sum(1 for a in teil.aktionen if a.status == "ausgefuehrt")
    if teil.aktionen:
        erg.bericht = teil  # bis zum Abbruch ausgefuehrte bzw. geplante Aktionen bleiben sichtbar
    if not erg.senden:
        erg.notizen.append(f"{grund}: Abbruch, nichts gesendet (trocken)")
    elif geschrieben:
        erg.notizen.append(f"{grund}: Abbruch nach {geschrieben} Schreibaufruf(en), Stand unklar – "
                           "Bestand per Marke maßgeblich")
    else:
        erg.notizen.append(f"{grund}: Abbruch, Stand unklar – Bestand per Marke maßgeblich")


# ---------- Bericht und Zusammenfassung ----------

def _zaehlen(funde: Sequence[Fund], arten: Sequence[str]) -> dict[str, dict[str, int]]:
    z: dict[str, dict[str, int]] = {a: {s: 0 for s in SCHWEREN} for a in arten}
    for f in funde:
        z.setdefault(f.art, {s: 0 for s in SCHWEREN})[f.schwere] += 1
    return z


def bericht_text(*, aussen: str, commit: str | None, modus: str, nur: Sequence[str] | None,
                 funde: Sequence[Fund], zaehler: dict[str, dict[str, int]], tor: ToterCodeTor,
                 ziele: Sequence[ZielErgebnis], ohne_ziel: int, notizen: Sequence[str], inventar_text: str) -> str:
    """Der volle lokale Bericht (vor dem Schwaerzen): Uebersicht, alle Funde je Art und Schwere, je Ziel der
    Abgleich, Hinweise und die volle Inventar-Tabelle."""
    z = [f"# Selbstbild-Funde: {aussen}", "",
         f"Commit `{commit or 'unbekannt'}` · Modus: {modus}"
         + (f" · nur: {', '.join(nur)}" if nur else ""), "",
         "## Übersicht", "", "| Art | hoch | mittel | niedrig | gesamt |", "|---|---|---|---|---|"]
    for art, je in zaehler.items():
        z.append(f"| {_art_name(art)} | {je['hoch']} | {je['mittel']} | {je['niedrig']} | {sum(je.values())} |")
    z += ["", f"- {tor_text(tor)}", f"- Funde ohne Ziel (nur in diesem Bericht): {ohne_ziel}", ""]
    for art in zaehler:
        eigene = [f for f in funde if f.art == art]
        if not eigene:
            continue
        z += [f"## {_art_name(art)}", ""]
        for s in SCHWEREN:
            teil = [f for f in eigene if f.schwere == s]
            if not teil:
                continue
            z += [f"### {s} ({len(teil)})", "", "| Code-Repo | Anker | Titel |", "|---|---|---|"]
            z += [f"| {_zelle(f.code_repo)} | `{_zelle(f.anker)}` | {_zelle(f.titel)} |" for f in teil]
            z.append("")
    for e in ziele:
        z.append(e.text())
    z += ["## Hinweise", ""]
    z += [f"- {n}" for n in notizen] or ["- keine"]
    z += ["", "## Konfig-Inventar (nur lokal)", "", inventar_text]
    return "\n".join(z).rstrip("\n") + "\n"


def schwaerzen_zaehlen(text: str, treffer: Counter[str]) -> str:
    """Jede Zeile mit Schluessel-Verdacht schwaerzen (nur der Mustername bleibt) und das Muster zaehlen."""
    aus = []
    for zeile in text.split("\n"):
        m = schluessel_muster(zeile)
        if m is not None:
            treffer[m] += 1
            zeile = schwaerzen(zeile)
            rest = schluessel_muster(zeile)
            if rest is not None:  # fail-closed: was das Schwaerzen nicht erwischt, kostet die ganze Zeile
                treffer[rest] += 1
                zeile = f"[Zeile ausgelassen: Schlüssel-Verdacht: {rest}]"
        aus.append(zeile)
    return "\n".join(aus)


def _schwaerzen_tief(obj: Any) -> Any:
    """Jeden Text in der JSON-Zusammenfassung schwaerzen (je Wert, damit das JSON gueltig bleibt)."""
    if isinstance(obj, str):
        if schluessel_muster(obj) is None:
            return obj
        neu = schwaerzen(obj)
        m = schluessel_muster(neu)
        return neu if m is None else f"[ausgelassen: Schlüssel-Verdacht: {m}]"
    if isinstance(obj, dict):
        return {_schwaerzen_tief(k): _schwaerzen_tief(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_schwaerzen_tief(v) for v in obj]
    return obj


def _muster_aus_gesperrt(eintrag: str) -> str:
    """`<schluessel>: <Mustername>` -> Mustername (Muster sind fest benannt, ohne Wert)."""
    return eintrag.split(": ", 1)[1] if ": " in eintrag else eintrag


def _json_ausgeben(daten: dict[str, Any], ausgabe: IO[str] | None) -> None:
    # ASCII wie `_ausgeben` im Einstieg: PowerShell liest stdout sonst in der Konsolen-Codepage falsch
    text = json.dumps(daten, ensure_ascii=True, sort_keys=True, indent=1) + "\n"
    if ausgabe is not None:
        ausgabe.write(text)
        return
    puffer = getattr(sys.stdout, "buffer", None)
    if puffer is not None:
        sys.stdout.flush()
        puffer.write(text.encode("utf-8"))
        puffer.flush()
    else:
        sys.stdout.write(text)


# ---------- Befehl ----------

def _regeln(wurzel: Path) -> Regeln:
    datei = wurzel / REGELDATEI
    return regeln_laden(datei.read_text(encoding="utf-8") if datei.is_file() else None)


def befehl(wurzel: Path, *, senden: bool = False, offline: bool = False, nur: Sequence[str] | None = None,
           goldsatz: Path | None = None, gh_fabrik: Callable[[str], GitHub] | None = None,
           bericht_ordner: Path | None = None, stand_basis: Path | None = None, stand_unter: Path | None = None,
           heute: date | None = None, ausgabe: IO[str] | None = None,
           schreib_pause: Callable[[float], None] | None = None, schreib_abstand_s: float = SCHREIB_ABSTAND_S) -> int:
    """`python -m selbstbild funde`. Alle Schluesselargumente ausser den CLI-Optionen sind fuer Tests injizierbar
    (`gh_fabrik(konto) -> GitHub`, Bericht-Ordner, Basis des Issue-Stands, Analyse-Stand, Datum, stdout, Pause und
    Abstand zwischen Schreibaufrufen).

    Schliess-Kommentare nennen den vermessenen Commit (`meta.bericht.commit`), ohne lesbaren Vermessungsbericht HEAD
    (mit Notiz). Ist der vermessene Commit kein Vorfahr von HEAD, gelten doppelung und tot als unlesbar."""
    wurzel = Path(wurzel)
    if senden and offline:
        raise ValueError("--senden und --offline schliessen sich aus")
    fabrik: Callable[[str], GitHub] = gh_fabrik if gh_fabrik is not None else GitHub
    try:
        regeln = _regeln(wurzel)
        ziele = ziele_aus_regeln(regeln)
        k = aus_regeln(wurzel)
    except (OSError, UnicodeDecodeError, KonfigFehler, ValueError, yaml.YAMLError) as e:
        # nur der Typ: weder YAML-Ausschnitt noch Fehlertext (koennte einen Wert aus der Datei zeigen)
        sys.stderr.write(f"funde: Regeldatei oder Konfiguration ungültig ({type(e).__name__})\n")
        return 2
    if senden:
        if not ziele:
            sys.stderr.write("funde --senden: kein Ziel (Abschnitt `issues:` in selbstbild.yaml fehlt)\n")
            return 2
        ohne_konto = [z.repo for z in ziele if not z.konto]
        if ohne_konto:
            sys.stderr.write("funde --senden: Ziel ohne Konto (`issues:` mit `konto` nötig): "
                             + ", ".join(ohne_konto) + "\n")
            return 2

    commit = commit_holen(wurzel)
    aussen = _aussen_name(wurzel)
    tor, notizen = tor_laden(Path(goldsatz) if goldsatz is not None else goldsatz_standard(aussen))
    funde, n_sammeln = funde_sammeln(wurzel, doku_konfig=k, stand_unter=stand_unter, tor=tor)
    notizen += n_sammeln
    inv = inventar(wurzel, stand_unter=stand_unter)
    funde.append(inventar_fund(inv))
    notizen += inv.notizen
    analyse = analyse_commit_lesen(wurzel, stand_unter)
    if commit is not None:
        if analyse is None:
            notizen.append("Analyse-Commit unbekannt (kein lesbarer Vermessungsbericht): Schliess-Kommentare "
                           "nennen HEAD")
        else:
            notizen += analyse_stand_pruefen(wurzel, analyse, commit)

    unlesbar = unlesbare_arten(notizen)  # immer aus allen Notizen (S5-19)
    if nur:
        gewaehlt = frozenset(nur)
        funde = [f for f in funde if f.art in gewaehlt]
        unlesbar = unlesbar | (frozenset(ARTEN) - gewaehlt)  # nicht gewaehlte Arten nie schliessen
    geprueft, n_geprueft = geprueft_ermitteln(wurzel, doku_konfig=k, stand_unter=stand_unter)
    notizen += n_geprueft
    if commit is None:
        notizen.append("HEAD-Commit nicht lesbar: kein Abgleich")
    abweichung = stand_abweichung(wurzel, commit, stand_unter)
    if abweichung:
        notizen.append(abweichung)

    stand = IssueStand.laden(stand_pfad(wurzel, stand_basis))
    tag = heute if heute is not None else date.today()
    gruppen, ohne = nach_ziel(funde, ziele)
    ergebnisse = [
        _abgleich_je_ziel(z, fs, senden=senden, offline=offline, commit=commit, stand=stand, heute=tag,
                          geprueft=geprueft, unlesbar=unlesbar, gh_fabrik=fabrik, schliess_commit=analyse,
                          pause=schreib_pause, schreib_abstand_s=schreib_abstand_s)
        for z, fs in gruppen]

    arten = [a for a in ARTEN if not nur or a in nur]
    zaehler = _zaehlen(funde, arten)
    modus = "senden" if senden else ("offline" if offline else "trocken")
    treffer: Counter[str] = Counter()
    for e in ergebnisse:
        for g in (e.bericht.gesperrt if e.bericht else []):
            treffer[_muster_aus_gesperrt(g)] += 1
    text = bericht_text(aussen=aussen, commit=commit, modus=modus, nur=nur, funde=funde, zaehler=zaehler, tor=tor,
                        ziele=ergebnisse, ohne_ziel=len(ohne), notizen=notizen, inventar_text=inventar_tabelle(inv))
    text = schwaerzen_zaehlen(text, treffer)
    ordner = Path(bericht_ordner) if bericht_ordner is not None else bericht_ordner_standard(wurzel)
    pfad = ordner / BERICHT_NAME
    bericht_fehler: str | None = None
    try:
        atomar_schreiben(pfad, text)
    except OSError as e:  # das JSON kommt trotzdem; nur der Typ, kein Pfad aus dem Fehlertext
        bericht_fehler = type(e).__name__

    zusammenfassung: dict[str, Any] = {
        "bericht": None if bericht_fehler else str(pfad), "bericht_fehler": bericht_fehler,
        "commit": commit, "modus": modus, "nur": sorted(set(nur)) if nur else None,
        "funde": zaehler, "funde_gesamt": len(funde), "ohne_ziel": len(ohne),
        "tor_tot": {"offen": tor.offen, "ja": tor.ja, "beurteilt": tor.beurteilt, "text": tor_text(tor)},
        "schluessel_treffer": sorted(treffer),
        "ziele": [e.zusammenfassung() for e in ergebnisse],
        "notizen": list(notizen),
    }
    _json_ausgeben(_schwaerzen_tief(zusammenfassung), ausgabe)
    return 1 if commit is None or bericht_fehler or any(e.fehler for e in ergebnisse) else 0
