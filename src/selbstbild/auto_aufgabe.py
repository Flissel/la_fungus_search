"""Windows-Aufgabe fuer den taeglichen Lauf `auto` (Stufe 6a, Task 5): einrichten, pruefen, entfernen.

`python -m selbstbild auto aufgabe <wurzel> [--pruefen | --zurueck] [--trocken]`. schtasks wird eingespeist
(`schtasks`), damit Tests nie die echte Aufgabenplanung anfassen. Exit 0 = getan bzw. bereit, 1 = Fehler, fehlt oder
Aktion weicht ab. Das Einrichten ist ein Live-Schalter und braucht das OK des Users.
"""
from __future__ import annotations

import codecs
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

from .doku.aufgabe import STANDARD_PYTHON, import_pruefen
from .funde.bericht import bericht_ordner_standard

AUFGABE = "VibeMind Selbstbild-Funde"
__all__ = ["AUFGABE", "aufgabe", "aufgabe_xml", "standard_schtasks"]
_KEIN_FENSTER = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
_ARBEITSORDNER = Path(__file__).resolve().parents[2]  # Wurzel des selbstbild-Repos
SCHTASKS_TIMEOUT_S = 30.0
MODUL = "selbstbild.auto"  # das muss das Python der Aufgabe ohne PYTHONPATH im Arbeitsordner laden
NICHT_GEFUNDEN = re.compile(r"nicht finden|nicht gefunden|not find|cannot find|does not exist|0x80070002",
                            re.IGNORECASE)
FEHLER_START = 255  # schtasks selbst lief nicht (Rueckgabe von standard_schtasks)


def aufgabe_xml(python: Path, wurzel: Path, arbeitsordner: Path, *, heute: date | None = None) -> str:
    """Task-Scheduler-XML: taeglich 07:30, verpassten Start nachholen, nur bei Anmeldung (InteractiveToken)."""
    tag = (heute or date.today()).isoformat()
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="{_NS}">
  <RegistrationInfo><Description>{escape(AUFGABE)}: taeglicher Lauf von selbstbild auto</Description></RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{tag}T07:30:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(str(python))}</Command>
      <Arguments>{escape(f'-m selbstbild auto "{wurzel}"')}</Arguments>
      <WorkingDirectory>{escape(str(arbeitsordner))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _dekodieren(roh: bytes) -> str:
    if roh[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return roh.decode("utf-16", errors="replace")
    try:
        return roh.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return roh.decode("oem", errors="replace")
        except LookupError:
            return roh.decode("cp850", errors="replace")


def standard_schtasks(argv: list[str]) -> tuple[int, str]:
    """`schtasks <argv>` -> (Exit, Ausgabe). capture_output ist hier ok: schtasks startet keine Enkelprozesse."""
    try:
        lauf_ = subprocess.run(["schtasks", *argv], capture_output=True, stdin=subprocess.DEVNULL,
                               creationflags=_KEIN_FENSTER, timeout=SCHTASKS_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return FEHLER_START, f"schtasks nicht ausfuehrbar: {type(e).__name__}"
    if lauf_.returncode == 0:  # nur stdout: eine Warnung auf stderr darf das XML nicht zerreissen
        return 0, _dekodieren(lauf_.stdout)
    return lauf_.returncode, _dekodieren(lauf_.stdout) + _dekodieren(lauf_.stderr)


def _aktion(xml: str) -> tuple[str, str, str] | None:
    """(Command, Arguments, WorkingDirectory) der ersten Exec-Aktion; None, wenn nicht lesbar."""
    text = xml.lstrip("\ufeff \r\n\t")
    if text.startswith("<?xml"):  # die Deklaration (UTF-16) passt nicht zu einem str
        text = text.split("?>", 1)[1] if "?>" in text else text
    try:
        exe = ET.fromstring(text).find(f".//{{{_NS}}}Exec")
    except ET.ParseError:
        return None
    if exe is None:
        return None
    def text_von(name: str) -> str:
        return (exe.findtext(f"{{{_NS}}}{name}") or "").strip()
    return text_von("Command"), text_von("Arguments"), text_von("WorkingDirectory")


def aufgabe(wurzel: Path, modus: Literal["einrichten", "pruefen", "zurueck"], *, trocken: bool,
            schtasks: Callable[[list[str]], tuple[int, str]], python: Path | None = None,
            arbeitsordner: Path | None = None, ordner: Path | None = None,
            probe: Callable[[str, Path], str | None] | None = None) -> int:
    """Windows-Aufgabe einrichten, pruefen oder entfernen. Exit 0 = getan bzw. bereit, 1 = Fehler/fehlt/weicht ab.

    Python wie bei `doku aufgabe`: `STANDARD_PYTHON` oder ausdruecklich `python`, nie `sys.executable`. Es muss
    existieren und `selbstbild.auto` im Arbeitsordner ohne PYTHONPATH importieren (`probe`, sonst `import_pruefen`)."""
    wurzel = Path(wurzel)
    py = Path(python) if python is not None else Path(STANDARD_PYTHON)
    pruefer = probe if probe is not None else (lambda p, w: import_pruefen(p, w, MODUL))
    wd = Path(arbeitsordner) if arbeitsordner is not None else _ARBEITSORDNER
    soll = aufgabe_xml(py, wurzel, wd)
    if modus == "pruefen":
        befehl = ["/Query", "/TN", AUFGABE, "/XML"]
        if trocken:
            print("trocken: schtasks " + " ".join(befehl))
            return 0
        rc, aus = schtasks(befehl)
        if rc != 0:
            if rc == 1 and NICHT_GEFUNDEN.search(aus):
                print(f"{AUFGABE}: fehlt")
            else:  # schtasks lief nicht oder verweigerte etwas anderes: nicht als "fehlt" ausgeben
                grund = aus.split(":", 1)[1].strip() if rc == FEHLER_START and ":" in aus else f"exit_{rc}"
                print(f"{AUFGABE}: fehler: {grund}")
            return 1
        ist, erwartet = _aktion(aus), _aktion(soll)
        if (ist is None or erwartet is None or ist[1] != erwartet[1]
                or os.path.normcase(ist[0]) != os.path.normcase(erwartet[0])):
            print(f"{AUFGABE}: aktion weicht ab (soll: {erwartet}, ist: {ist})")
            return 1
        grund_py = pruefer(ist[0], Path(ist[2]) if ist[2] else wd)
        if grund_py is not None:
            print(f"{AUFGABE}: python importiert selbstbild nicht ({grund_py})")
            return 1
        print(f"{AUFGABE}: bereit")
        return 0
    if modus == "zurueck":
        befehl = ["/Delete", "/TN", AUFGABE, "/F"]
        if trocken:
            print("trocken: schtasks " + " ".join(befehl))
            return 0
        rc, aus = schtasks(befehl)
        print(f"{AUFGABE}: " + ("entfernt" if rc == 0 else f"nicht entfernt ({aus.strip()[:200]})"))
        return 0 if rc == 0 else 1
    ziel_ordner = Path(ordner) if ordner is not None else bericht_ordner_standard(wurzel)
    if not trocken:
        grund_py = pruefer(str(py), wd)
        if grund_py is not None:
            print(f"{AUFGABE}: nicht eingerichtet: {py} importiert {MODUL} ohne PYTHONPATH nicht ({grund_py})")
            return 1
    if trocken:
        datei = ziel_ordner / "aufgabe.xml"
        print(f"trocken: Python {py} (Import-Probe entfaellt)")
        print(f"trocken: schtasks /Create /TN {AUFGABE} /XML {datei} /F\n{soll}")
        return 0
    ziel_ordner.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=ziel_ordner, prefix="aufgabe-", suffix=".xml")
    datei = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(soll.encode("utf-16"))  # mit BOM: das erwartet schtasks /XML
        rc, aus = schtasks(["/Create", "/TN", AUFGABE, "/XML", str(datei), "/F"])
    finally:
        datei.unlink(missing_ok=True)
    print(f"{AUFGABE}: " + ("eingerichtet" if rc == 0 else f"nicht eingerichtet ({aus.strip()[:200]})"))
    return 0 if rc == 0 else 1
