"""Taeglicher Lauf `python -m selbstbild auto <wurzel> [--trocken]` (Stufe 6a, Task 4; Spec 4.1 und 4.4).

Ablauf: Sperre `auto.lock` -> Stand `auto-stand.json` laden -> Zweig pruefen (`kopf_pruefen`, `doku.zweig`) ->
Aenderungs-Tor (HEAD, Hash des Goldsatzes, Antwort-Labels der Rueckfragen aus genau einem `gh.issues`) ->
Speicher pruefen (mindestens `MIN_FREI_GB` freier Commit-Speicher) -> Kindprozesse `vermessen` und `analysieren` ->
`rueckfragen_abgleichen` -> Kindprozess `funde [--senden]` -> Zusammenfassung -> hoechstens eine Telegram-Nachricht
je Kalendertag -> Stand schreiben (atomar, nie mit `--trocken`).

Kindprozesse laufen als `[sys.executable, "-m", "selbstbild", <befehl>, ...]` mit stdout und stderr in eine Datei
`auto-<befehl>.log` im Ordner (nie ueber Pipes: Enkel erben Pipes und haengen). Telegram nur ueber
`scripts/notify_telegram.py` mit dem Text per stdin; dessen stdout und stderr werden verworfen, gewertet wird nur der
Exit-Code. Die `.env` wird hier nie geoeffnet. Fehler erscheinen nur als Typ (`type(e).__name__`, `exit_<n>`,
`GitHubFehler.art`), nie mit Werten.

Exit: 0 bei Erfolg, beim Ueberspringen und bei „nichts zu tun“; 1 bei Fehler; 3, wenn die Sperre belegt ist (still).

Festlegungen dieses Moduls (Builder, Spec 5):
- Zeilen, die eine Nachricht ausloesen (`zeilen`), und Begleitzeilen (`kontext`: Stand des Tors ohne Aenderung,
  uebernommene Sperre, uebersprungene Rueckfragen), die nur mit anderen Zeilen mitgehen.
- Ausstehende Zeilen gehen am naechsten Tag auch ohne Aenderung mit (sonst warteten sie bis zur naechsten Aenderung).
- Das Tor schaltet nur nach einem erfolgreichen Lauf weiter; die eigenen Schliessungen und der eigene Goldsatz-Commit
  zaehlen beim naechsten Lauf nicht als Aenderung.
- `label_fehlt` bei den Rueckfragen sperrt das Ziel fuer beide Wege (`auto-stand.json` und `issues-*.json` wie
  `sync`), bis ein Mensch den Eintrag unter `sperren` loescht.
- Gesamtfrist `FRIST_S` je Lauf (R15): jedes Kind bekommt die Restzeit als Timeout; Ablauf = Fehler
  `<schritt>, TimeoutExpired`. So bleibt der Lauf unter dem stillen Abbruch der Aufgabenplanung (PT2H).
- Ein Fehler oder Ueberspringen wird einmal gemeldet; haelt er laenger als `ERINNERN_AB_TAGEN` Tage an, nennt ihn
  jeder Montag noch einmal (R16, `fehler_seit`, `fehler_erinnert_tag`).
- `funde` mit Exit 3 (anderer Sendelauf): das Tor bleibt stehen, der naechste Lauf versucht es erneut (R17-3).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .analyse.lauf import gespeicherte_analyse
from .auto_aufgabe import AUFGABE, aufgabe, aufgabe_xml, standard_schtasks
from .auto_text import (ERINNERN_AB_TAGEN, ERSATZ_TEXT, MAX_TEXT, TEXT_KOPF, erinnerungen, funde_zeilen, nachricht,
                        nummer, rueckfrage_zeilen, ziel_fehler)
from .doku.konfig import aus_regeln
from .doku.schluessel import sieht_aus_wie_schluessel
from .doku.schreibvorgang import kopf, kopf_pruefen
from .funde.bericht import SENDE_SPERRE, _aussen_name, atomar_schreiben, bericht_ordner_standard, goldsatz_standard
from .funde.github import GitHub, GitHubFehler
from .funde.rueckfrage import _MARKE, ANTWORTEN, RueckfrageBericht, rueckfragen_abgleichen
from .funde.sammeln import _posix
from .funde.sync import IssueStand, Ziel, _body, _labels, stand_pfad, ziele_aus_regeln
from .regeln import REGELDATEI, regeln_laden
from .sperre import SperreBelegt, sperre

MIN_FREI_GB = 3.0
STAND_DATEI = "auto-stand.json"
SPERRE = "auto.lock"
STAND_VERSION = 1
FRIST_S = 6000.0  # R15: Gesamtfrist je Lauf (100 min) unter ExecutionTimeLimit PT2H; jedes Kind bekommt die Restzeit
TELEGRAM_TIMEOUT_S = 60.0
MAX_AUSSTEHEND = 60
ZURUECKGEHALTEN = "[Zeile zurückgehalten: Schlüssel-Verdacht]"
__all__ = ["AUFGABE", "ERSATZ_TEXT", "FRIST_S", "MAX_TEXT", "MIN_FREI_GB", "SPERRE", "STAND_DATEI", "TEXT_KOPF",
           "Werkzeuge",
           "antworten_signatur", "aufgabe", "aufgabe_xml", "frei_gb_standard", "kind_standard", "lauf",
           "letztes_json", "nachricht", "stand_laden", "standard_schtasks", "standard_werkzeuge",
           "telegram_standard"]
_KEIN_FENSTER = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------- Werkzeuge ----------

@dataclass
class Werkzeuge:
    """Alles, was der Lauf nach aussen tut; fuer Tests injizierbar. `kind(argv, log, timeout_s)`: `timeout_s` ist die
    Restzeit der Gesamtfrist; laeuft sie ab, wird das Kind beendet (`subprocess.TimeoutExpired`)."""
    kind: Callable[[list[str], Path, float], int]
    frei_gb: Callable[[], float | None]
    telegram: Callable[[str], bool]
    gh_fabrik: Callable[[str], GitHub]
    jetzt: Callable[[], datetime]
    ordner: Path
    goldsatz: Path | None = None  # Vorgabe: goldsatz_standard(<aussen>)
    stand_basis: Path | None = None  # Basis des Issue-Stands von `sync` (Vorgabe: stand_basis())
    uhr: Callable[[], float] = time.monotonic  # Sekunden, fuer die Gesamtfrist (R15)


def kind_standard(argv: list[str], log: Path, timeout_s: float) -> int:
    """Kindprozess mit stdout und stderr in `log`, stdin leer, ohne Fenster; liefert den Exit-Code. Nach `timeout_s`
    beendet `subprocess.run` das Kind und wirft `TimeoutExpired` (der Lauf meldet `<schritt>, TimeoutExpired`)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as f:
        lauf_ = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               creationflags=_KEIN_FENSTER, timeout=timeout_s, check=False)
    return lauf_.returncode


def frei_gb_standard() -> float | None:
    """Freier Commit-Speicher in GB (`GlobalMemoryStatusEx().ullAvailPageFile`); None bei jedem Fehler."""
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = _Status()
        s.dwLength = ctypes.sizeof(_Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):  # type: ignore[attr-defined]
            return None
        return float(s.ullAvailPageFile) / 2**30
    except Exception:  # noqa: BLE001 -- ohne Messung wird uebersprungen
        return None


def telegram_standard(wurzel: Path, text: str) -> bool:
    """`scripts/notify_telegram.py` der Wurzel, Text per stdin (UTF-8); stdout und stderr verworfen (ein Traceback
    zeigte die URL mit dem Token). Erfolg nur bei Exit 0."""
    skript = Path(wurzel) / "scripts" / "notify_telegram.py"
    if not skript.is_file():
        return False
    try:
        lauf_ = subprocess.run([sys.executable, str(skript)], input=text.encode("utf-8"), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=_KEIN_FENSTER, timeout=TELEGRAM_TIMEOUT_S,
                               check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return lauf_.returncode == 0


def standard_werkzeuge(wurzel: Path) -> Werkzeuge:
    wurzel = Path(wurzel)
    return Werkzeuge(kind=kind_standard, frei_gb=frei_gb_standard,
                     telegram=lambda text: telegram_standard(wurzel, text), gh_fabrik=GitHub,
                     jetzt=datetime.now, ordner=bericht_ordner_standard(wurzel))


# ---------- Bausteine ----------

def letztes_json(pfad: Path) -> dict[str, Any] | None:
    """Das letzte JSON-Objekt, das am Zeilenanfang beginnt (stderr kann davor und danach stehen); sonst None."""
    try:
        text = Path(pfad).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    starts = [i for i, z in enumerate(text) if z == "{" and (i == 0 or text[i - 1] == "\n")]
    dekoder = json.JSONDecoder()
    for i in reversed(starts):
        try:
            obj, _ = dekoder.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _hash_datei(pfad: Path) -> str | None:
    try:
        return hashlib.sha256(Path(pfad).read_bytes()).hexdigest()
    except OSError:
        return None


def antworten_signatur(bestand: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Je OFFENE Rueckfrage (Marke) mit mindestens einem Antwort-Label die Antwort-Labels (R17-2). Geschlossene zaehlen
    nicht: sonst hielte ein laengst beantwortetes Issue das Tor offen."""
    sig: dict[str, dict[str, Any]] = {}
    for issue in bestand:
        nr = nummer(issue)
        if nr is None or issue.get("state") != "open" or not _MARKE.search(_body(issue)):
            continue
        gesetzt = sorted(x for x in _labels(issue) if x in ANTWORTEN)
        if gesetzt:
            sig[str(nr)] = {"offen": True, "labels": gesetzt}
    return sig


def _signatur_nach_lauf(sig: dict[str, dict[str, Any]] | None,
                        rb: RueckfrageBericht) -> dict[str, dict[str, Any]] | None:
    """Was der naechste Lauf sehen wird, wenn niemand etwas tut: selbst geschlossene Issues sind dann zu. Wartende
    Antworten: None (unbekannt = geaendert), damit der naechste Lauf es erneut versucht (Spec 4.4.2)."""
    if sig is None or rb.wartet:
        return None
    neu = {k: dict(v) for k, v in sig.items()}
    for nr in [*rb.uebernommen, *rb.erledigt]:
        neu.pop(str(nr), None)  # jetzt geschlossen: zaehlt nicht mehr (R17-2)
    return neu


# ---------- Stand ----------

def _leerer_stand() -> dict[str, Any]:
    return {"version": STAND_VERSION, "kopf": None, "goldsatz_hash": None, "antworten": None,
            "letzte_nachricht_tag": None, "letzter_fehler": None, "ausstehend": [], "letzter_lauf": None,
            "ergebnis": None, "rueckfrage": {}, "tor": None, "sperren": {}, "gemeldet": {"doppelt": [], "wartet": []},
            "erinnert_tag": None, "fehler_seit": None, "fehler_erinnert_tag": None}


def stand_laden(pfad: Path) -> tuple[dict[str, Any], str | None]:
    """(Stand, Notiz). Fehlt die Datei: leerer Stand ohne Notiz; kaputt: leerer Stand mit Notiz. Ein Schluessel mit
    falschem Typ (auch innen, z. B. `gemeldet.doppelt`) gilt als leer bzw. Vorgabe, mit Notiz (R11); nie eine
    Ausnahme."""
    s = _leerer_stand()
    try:
        roh = json.loads(Path(pfad).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return s, None
    except (OSError, ValueError) as e:
        return s, f"{STAND_DATEI} unlesbar ({type(e).__name__}): neu angelegt"
    if not isinstance(roh, dict):
        return s, f"{STAND_DATEI} unlesbar (kein Objekt): neu angelegt"
    falsch: list[str] = []

    def nehmen(k: str, ok: Callable[[Any], bool], wandeln: Callable[[Any], Any] = lambda v: v) -> None:
        if k not in roh or roh[k] is None:
            return
        if ok(roh[k]):
            s[k] = wandeln(roh[k])
        else:
            falsch.append(k)

    def ist_liste_von(art: type) -> Callable[[Any], bool]:
        return lambda v: isinstance(v, list) and all(isinstance(x, art) and not isinstance(x, bool) for x in v)

    for k in ("kopf", "goldsatz_hash", "letzte_nachricht_tag", "letzter_fehler", "letzter_lauf", "ergebnis", "tor",
              "erinnert_tag", "fehler_seit", "fehler_erinnert_tag"):
        nehmen(k, lambda v: isinstance(v, str))
    nehmen("antworten", lambda v: isinstance(v, dict))
    nehmen("ausstehend", ist_liste_von(str))
    nehmen("rueckfrage", lambda v: isinstance(v, dict))
    nehmen("sperren", lambda v: isinstance(v, dict) and all(isinstance(g, str) for g in v.values()))
    gemeldet = roh.get("gemeldet")
    if isinstance(gemeldet, dict):
        for k, art in (("doppelt", int), ("wartet", str)):
            wert = gemeldet.get(k)
            if wert is None:
                continue
            if ist_liste_von(art)(wert):
                s["gemeldet"][k] = list(wert)
            else:
                falsch.append(f"gemeldet.{k}")
    elif gemeldet is not None:
        falsch.append("gemeldet")
    if falsch:
        return s, f"{STAND_DATEI}: falscher Typ bei {', '.join(falsch)}, leer angenommen"
    return s, None


# ---------- Lauf ----------

class _Abbruch(Exception):
    def __init__(self, schritt: str, typ: str) -> None:
        super().__init__(f"{schritt}, {typ}")
        self.schritt, self.typ = schritt, typ


def _ausgeben(text: str) -> None:
    puffer = getattr(sys.stdout, "buffer", None)
    if puffer is not None:
        sys.stdout.flush()
        puffer.write((text + "\n").encode("utf-8"))
        puffer.flush()
    else:
        sys.stdout.write(text + "\n")


class _Lauf:
    def __init__(self, wurzel: Path, w: Werkzeuge, trocken: bool) -> None:
        """Wirft nie (R11): ein Fehler hier wird in `ausfuehren` als `start, <Typ>` gemeldet."""
        self.wurzel, self.w, self.trocken = wurzel, w, trocken
        self.zeilen: list[str] = []  # loesen eine Nachricht aus
        self.kontext: list[str] = []  # gehen nur mit anderen Zeilen mit
        self.schritt = "start"
        self.ergebnis = "ok"
        self.geschlossen: set[int] = set()  # in diesem Lauf geschlossene Rueckfragen (keine Erinnerung)
        self.start_fehler: str | None = None
        self.frist = 0.0  # Ende der Gesamtfrist (`w.uhr`), gesetzt beim Start des Ablaufs
        self.stand = _leerer_stand()
        try:
            zeit = w.jetzt()
            self.heute = zeit.date()
            self.zeit = zeit
        except Exception as e:  # noqa: BLE001
            self.start_fehler = type(e).__name__
            self.zeit = datetime.now()
            self.heute = self.zeit.date()
        self.pfad = Path(w.ordner) / STAND_DATEI
        try:
            self.stand, notiz = stand_laden(self.pfad)
        except Exception as e:  # noqa: BLE001 -- stand_laden wirft nicht; nur die letzte Sicherung
            self.stand, notiz = _leerer_stand(), f"{STAND_DATEI} unlesbar ({type(e).__name__}): neu angelegt"
        if notiz:
            self.kontext.append(notiz)

    # ---------- Ende ----------

    def fehler(self, schritt: str, typ: str) -> int:
        self._melden(f"{schritt}:{typ}", f"Selbstbild-Lauf fehlgeschlagen: {schritt}, {typ}", "fehler")
        return 1

    def ueberspringen(self, schluessel: str, zeile: str) -> int:
        self._melden(schluessel, zeile, "übersprungen")
        return 0

    def _melden(self, schluessel: str, zeile: str, art: str) -> None:
        if schluessel != self.stand["letzter_fehler"]:  # derselbe Fehler wird nicht noch einmal gemeldet
            self.zeilen.append(zeile)
            self.stand["fehler_seit"] = self.heute.isoformat()
        else:
            self._dauer_melden(zeile)
        self.stand["letzter_fehler"] = schluessel
        self.ergebnis = f"{art}:{schluessel}"
        self.abschliessen()

    def _dauer_melden(self, zeile: str) -> None:
        """R16: montags (einmal je Montag) erneut nennen, was seit mehr als `ERINNERN_AB_TAGEN` Tagen anhaelt."""
        try:
            seit = date.fromisoformat(str(self.stand["fehler_seit"]))
        except ValueError:  # fehlt (alter Stand) oder kaputt: ab heute zaehlen
            self.stand["fehler_seit"] = self.heute.isoformat()
            return
        tage = (self.heute - seit).days
        tag = self.heute.isoformat()
        if self.zeit.weekday() == 0 and tage > ERINNERN_AB_TAGEN and self.stand["fehler_erinnert_tag"] != tag:
            self.zeilen.append(f"{zeile} (hält seit {tage} Tagen an)")
            self.stand["fehler_erinnert_tag"] = tag

    def abschliessen(self) -> None:
        neu = self.zeilen + (self.kontext if self.zeilen else [])
        alt = list(self.stand["ausstehend"])
        alle = alt + neu
        if self.trocken:
            text = nachricht(alle, self.heute) if alle else "Selbstbild (trocken): keine Nachricht"
            _ausgeben(ERSATZ_TEXT if sieht_aus_wie_schluessel(text) else text)
            return
        tag = self.heute.isoformat()
        if alle and self.stand["letzte_nachricht_tag"] != tag:
            text = nachricht(alle, self.heute)
            if sieht_aus_wie_schluessel(text):
                text = ERSATZ_TEXT
            try:
                ok = bool(self.w.telegram(text))
            except Exception:  # noqa: BLE001 -- Telegram darf den Lauf nie brechen
                ok = False
            if ok:
                self.stand["letzte_nachricht_tag"] = tag
                self.stand["ausstehend"] = []
            else:
                self.stand["ausstehend"] = self._ausstehend(alt, neu)
        elif neu:
            self.stand["ausstehend"] = self._ausstehend(alt, neu)
        self.schreiben()

    def _ausstehend(self, alt: list[str], neu: list[str]) -> list[str]:
        """Neue Zeilen mit ihrem Tag; nie ein Schluesselwert in der Datei; hoechstens `MAX_AUSSTEHEND`."""
        vorsatz = f"({self.heute:%d.%m.}) "
        alle = alt + [vorsatz + (ZURUECKGEHALTEN if sieht_aus_wie_schluessel(z) else z) for z in neu]
        if len(alle) > MAX_AUSSTEHEND:
            alle = ["(ältere Zeilen ausgelassen)", *alle[-(MAX_AUSSTEHEND - 1):]]
        return alle

    def schreiben(self) -> None:
        if self.trocken:
            return
        self.stand["letzter_lauf"] = self.zeit.isoformat(timespec="seconds")
        self.stand["ergebnis"] = self.ergebnis
        try:
            atomar_schreiben(self.pfad, json.dumps(self.stand, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        except OSError as e:
            sys.stderr.write(f"auto: {STAND_DATEI} nicht schreibbar ({type(e).__name__})\n")

    # ---------- Ablauf ----------

    def ausfuehren(self, vorab: tuple[str, str] | None = None) -> int:
        """Nichts kommt hier als Ausnahme heraus (R11). `vorab`: ein Fehler vor dem Lauf (z. B. die Sperre)."""
        try:
            try:
                if vorab is not None:
                    return self.fehler(*vorab)
                if self.start_fehler is not None:
                    return self.fehler("start", self.start_fehler)
                return self._ablauf()
            except _Abbruch as a:
                return self.fehler(a.schritt, a.typ)
            except Exception as e:  # noqa: BLE001 -- nur der Typ, nie der Text
                return self.fehler(self.schritt, type(e).__name__)
        except Exception as e:  # noqa: BLE001 -- Fehler in der Fehlerbehandlung selbst: nur noch stderr
            sys.stderr.write(f"Selbstbild-Lauf fehlgeschlagen: abschluss, {type(e).__name__}\n")
            return 1

    def _ablauf(self) -> int:
        self.frist = float(self.w.uhr()) + FRIST_S  # R15: Gesamtfrist, die Kinder teilen sich die Restzeit
        self.schritt = "konfig"
        zweig = aus_regeln(self.wurzel).zweig
        aussen = _aussen_name(self.wurzel)
        ziel = self._rueckfrage_ziel(aussen)
        grund = kopf_pruefen(self.wurzel, zweig)
        if grund:
            return self.ueberspringen(f"zweig:{grund}", f"übersprungen: Zweig nicht messbar ({grund})")

        self.schritt = "tor"  # vor dem Speicher: das Tor liest nur Git und GitHub (Abnahme 07.11.)
        kopf_jetzt = kopf(self.wurzel)
        gold = Path(self.w.goldsatz) if self.w.goldsatz is not None else goldsatz_standard(aussen)
        gh, bestand, sig = self._antworten_lesen(ziel)
        montag = self.zeit.weekday() == 0
        if (kopf_jetzt is not None and kopf_jetzt == self.stand["kopf"] and sig is not None
                and sig == self.stand["antworten"] and _hash_datei(gold) == self.stand["goldsatz_hash"]):
            self.stand.update(letzter_fehler=None, fehler_seit=None)
            self.ergebnis = "unveraendert"
            if montag:
                self._erinnern(bestand)
            self.abschliessen()
            return 0

        self.schritt = "speicher"
        try:
            frei = self.w.frei_gb()
        except Exception:  # noqa: BLE001
            frei = None
        if frei is None or frei < MIN_FREI_GB:
            wert = "unbekannt" if frei is None else f"{frei:.1f} GB"
            return self.ueberspringen("speicher", f"übersprungen: Speicher (frei {wert}, nötig {MIN_FREI_GB:g} GB)")

        for schritt in ("vermessen", "analysieren"):
            self.schritt = schritt
            extra = ["--commit", kopf_jetzt] if schritt == "vermessen" and kopf_jetzt else []
            rc = self._kind(schritt, extra)
            if rc != 0:
                raise _Abbruch(schritt, f"exit_{rc}")

        self.schritt = "rueckfragen"
        sig_nachher = self._rueckfragen(ziel, gh, gold, sig)
        gold_hash = _hash_datei(gold)  # R17-4: direkt nach den Rueckfragen (eigener Commit ja, Aenderungen danach nein)

        self.schritt = "funde"
        rc = self._kind("funde", [] if self.trocken else ["--senden"])
        daten = letztes_json(Path(self.w.ordner) / "auto-funde.log")
        # Fehlerschluessel mit den Fehlerarten der Ziele: derselbe Ziel-Fehler an mehreren Tagen wird nur beim
        # ersten Mal gemeldet (Spec 4.4.6, R10); seine Zeilen gehen danach nur noch als Begleitzeilen mit
        arten = ziel_fehler(daten)
        typ = f"exit_{rc}" + (f":{','.join(arten)}" if arten else "")
        schon_gemeldet = rc not in (0, 3) and f"funde:{typ}" == self.stand["letzter_fehler"]
        if daten is not None:
            self._funde_zeilen(daten, fehler_als_kontext=schon_gemeldet)
        if rc == 3:
            self.zeilen.append("Funde nicht gesendet: ein anderer Sendelauf lief (Sperre)")
        elif rc != 0:
            raise _Abbruch("funde", typ)
        elif daten is None:
            raise _Abbruch("funde", "json_fehlt")

        if rc != 3:  # R17-3: bei Exit 3 bleibt das Tor stehen (alter kopf), der naechste Lauf versucht es erneut
            self.stand.update(kopf=kopf_jetzt, goldsatz_hash=gold_hash, antworten=sig_nachher)
        self.stand.update(letzter_fehler=None, fehler_seit=None)
        if montag:
            self._erinnern(bestand)
        self.ergebnis = "ok"
        self.abschliessen()
        return 0

    def _kind(self, schritt: str, extra: list[str]) -> int:
        """Ein Kind mit der Restzeit der Gesamtfrist; ist sie schon aufgebraucht, startet es nicht (R15)."""
        rest = self.frist - float(self.w.uhr())
        if rest <= 0:
            raise _Abbruch(schritt, "TimeoutExpired")
        argv = [sys.executable, "-m", "selbstbild", schritt, str(self.wurzel), *extra]
        return int(self.w.kind(argv, Path(self.w.ordner) / f"auto-{schritt}.log", rest))

    def _rueckfrage_ziel(self, aussen: str) -> Ziel | None:
        datei = self.wurzel / REGELDATEI
        regeln = regeln_laden(datei.read_text(encoding="utf-8") if datei.is_file() else None)
        return next((z for z in ziele_aus_regeln(regeln) if z.zustaendig(aussen)), None)

    def _antworten_lesen(self, ziel: Ziel | None) -> tuple[GitHub | None, list[dict[str, Any]] | None,
                                                           dict[str, dict[str, Any]] | None]:
        """(gh, Bestand, Signatur). Ohne Ziel mit Konto ist die Signatur leer (fest); ist GitHub nicht lesbar: None
        (unbekannt = geaendert)."""
        if ziel is None or not ziel.konto:
            return None, None, {}
        try:
            gh = self.w.gh_fabrik(ziel.konto)
        except ValueError as e:
            self.kontext.append(f"Rückfragen: Konto nicht nutzbar ({type(e).__name__})")
            return None, None, None
        try:
            bestand = gh.issues(ziel.repo)
        except GitHubFehler as f:
            self.kontext.append(f"Rückfragen: GitHub nicht lesbar ({f.art})")
            return gh, None, None
        return gh, bestand, antworten_signatur(bestand)

    # ---------- Rueckfragen ----------

    def _sync_sperre(self, repo: str) -> str | None:
        return IssueStand.laden(stand_pfad(self.wurzel, self.w.stand_basis)).sperren.get(repo)

    def _rueckfragen(self, ziel: Ziel | None, gh: GitHub | None, gold: Path,
                     sig: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]] | None:
        """Rueckfragen abgleichen; liefert die Signatur fuer den Stand nach einem erfolgreichen Lauf."""
        if ziel is None or not ziel.konto or gh is None:
            if ziel is None or not ziel.konto:
                self.kontext.append("Rückfragen: kein Issue-Ziel mit Konto")
            return sig
        repo = ziel.repo
        gesperrt = self.stand["sperren"].get(repo) or self._sync_sperre(repo)
        if gesperrt:
            self.kontext.append(f"Rückfragen gesperrt ({gesperrt}): nach Prüfung den Eintrag unter `sperren` in "
                                f"{STAND_DATEI} und issues-*.json löschen")
            return sig
        uebersprungen = None if sig else sig  # nichts Offenes: Signatur behalten, sonst erneut versuchen
        if not gold.is_file():
            self.kontext.append(f"Rückfragen übersprungen: Goldsatz fehlt ({gold.name})")
            return uebersprungen
        try:
            a = gespeicherte_analyse(self.wurzel)
            noch_tot = frozenset(_posix(str(t.symbol)) for t in a.tot)
            commit = str(a.commit)
        except Exception as e:  # noqa: BLE001 -- R2: ohne lesbare Analyse keine Rueckfragen
            self.kontext.append(f"Rückfragen übersprungen: Analyse nicht lesbar ({type(e).__name__})")
            return uebersprungen
        if not noch_tot:  # R2: nie mit leerer Menge (sonst wirkte alles „nicht mehr tot“)
            self.kontext.append("Rückfragen übersprungen: Analyse ohne toten Code")
            return uebersprungen
        teil = self.stand["rueckfrage"] if not self.trocken else dict(self.stand["rueckfrage"])
        rb = rueckfragen_abgleichen(self.wurzel, gold, gh, repo, senden=not self.trocken, noch_tot=noch_tot,
                                    commit=commit, stand=teil, oeffentlich=ziel.oeffentlich)
        self.ergebnis = "unterbrochen:nach_rueckfragen"  # bleibt nur stehen, wenn der Prozess hart abbricht
        self.schreiben()  # sofort sichern: ein spaeterer Absturz wiederholt keine Kommentare
        self.geschlossen |= {*rb.uebernommen, *rb.erledigt}
        self._rueckfrage_zeilen(rb)
        if rb.fehler:
            if rb.fehler == "label_fehlt" and not self.trocken:
                self._ziel_sperren(repo)
            raise _Abbruch("rueckfragen", rb.fehler)
        return _signatur_nach_lauf(sig, rb)

    def _ziel_sperren(self, repo: str) -> None:
        """Wie `sync`: GitHub hat das Label verworfen, das Issue ist im Bestand unsichtbar. Beide Wege sperren."""
        self.stand["sperren"][repo] = "label_fehlt"
        self.zeilen.append("Rückfragen: GitHub hat das Label selbstbild verworfen (Rechte?). Ziel gesperrt, das "
                           "angelegte Issue von Hand prüfen")
        try:
            with sperre(Path(self.w.ordner) / SENDE_SPERRE):
                st = IssueStand.laden(stand_pfad(self.wurzel, self.w.stand_basis))
                st.sperren[repo] = "label_fehlt"
                st.speichern()
        except (SperreBelegt, OSError) as e:
            self.kontext.append(f"Sperre für funde nicht gesetzt ({type(e).__name__})")
        self.schreiben()

    def _rueckfrage_zeilen(self, rb: RueckfrageBericht) -> None:
        zeilen, kontext, self.stand["gemeldet"] = rueckfrage_zeilen(rb, self.stand["gemeldet"], self.trocken)
        self.zeilen += zeilen
        self.kontext += kontext

    # ---------- Zusammenfassung ----------

    def _funde_zeilen(self, daten: dict[str, Any], *, fehler_als_kontext: bool) -> None:
        zeilen, fehler, kontext, self.stand["tor"] = funde_zeilen(daten, self.stand["tor"])
        self.zeilen += zeilen
        (self.kontext if fehler_als_kontext else self.zeilen).extend(fehler)
        self.kontext += kontext

    def _erinnern(self, bestand: list[dict[str, Any]] | None) -> None:
        """Montags, einmal je Montag: offene Rueckfragen, die laenger als eine Woche offen sind."""
        tag = self.heute.isoformat()
        if bestand is None or self.stand["erinnert_tag"] == tag:
            return
        zeilen = erinnerungen([i for i in bestand if nummer(i) not in self.geschlossen], self.heute)
        if zeilen:
            self.zeilen += zeilen
            self.stand["erinnert_tag"] = tag


def lauf(wurzel: Path, *, trocken: bool = False, werkzeuge: Werkzeuge | None = None) -> int:
    """Der taegliche Lauf (siehe Moduldoku). Exit 0, 1 oder 3."""
    wurzel = Path(wurzel)
    w = werkzeuge if werkzeuge is not None else standard_werkzeuge(wurzel)
    with contextlib.ExitStack() as stapel:
        try:
            uebernommen = stapel.enter_context(sperre(Path(w.ordner) / SPERRE))
        except SperreBelegt:
            return 3
        except OSError as e:  # Ordner oder Sperrdatei nicht anlegbar: nur der Typ
            return _Lauf(wurzel, w, trocken).ausfuehren(vorab=("sperre", type(e).__name__))
        lauf_ = _Lauf(wurzel, w, trocken)
        if uebernommen:
            lauf_.kontext.append("Hinweis: verwaiste Sperre übernommen (ein früherer Lauf brach ab)")
        return lauf_.ausfuehren()

