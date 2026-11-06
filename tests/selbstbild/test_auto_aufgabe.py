"""Windows-Aufgabe fuer den taeglichen Lauf (Stufe 6a, Task 5). schtasks ist immer gefaelscht, nie echt."""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pytest

from selbstbild import auto
from selbstbild.__main__ import main

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
PY = Path("C:/Users/Büro User/venv/Scripts/python.exe")
REPO = Path("C:/Users/User/ClaudeWork/wt-fungus-gate1")


class Schtasks:
    def __init__(self, antworten: dict[str, tuple[int, str]] | None = None) -> None:
        self.aufrufe: list[list[str]] = []
        self.antworten = antworten or {}
        self.xml_inhalt: str | None = None

    def __call__(self, argv: list[str]) -> tuple[int, str]:
        self.aufrufe.append(list(argv))
        if "/XML" in argv and argv[0] == "/Create":
            self.xml_inhalt = Path(argv[argv.index("/XML") + 1]).read_bytes().decode("utf-16")
        return self.antworten.get(argv[0], (0, ""))


def _xml(wurzel: Path) -> str:
    return auto.aufgabe_xml(PY, wurzel, REPO, heute=date(2026, 11, 6))


def test_xml_hat_taeglich_0730_und_nachholen(tmp_path: Path) -> None:
    wurzel = tmp_path / "Mein Repo"
    r = ET.fromstring(_xml(wurzel))
    assert r.findtext(".//t:StartWhenAvailable", namespaces=NS) == "true"
    assert (r.findtext(".//t:StartBoundary", namespaces=NS) or "").endswith("T07:30:00")
    assert r.findtext(".//t:DaysInterval", namespaces=NS) == "1"
    assert r.findtext(".//t:LogonType", namespaces=NS) == "InteractiveToken"
    assert r.findtext(".//t:MultipleInstancesPolicy", namespaces=NS) == "IgnoreNew"
    assert r.findtext(".//t:ExecutionTimeLimit", namespaces=NS) == "PT2H"
    assert r.findtext(".//t:Exec/t:Command", namespaces=NS) == str(PY)
    assert r.findtext(".//t:Exec/t:Arguments", namespaces=NS) == f'-m selbstbild auto "{wurzel}"'
    assert r.findtext(".//t:Exec/t:WorkingDirectory", namespaces=NS) == str(REPO)


def test_xml_maskiert_sonderzeichen(tmp_path: Path) -> None:
    r = ET.fromstring(auto.aufgabe_xml(PY, tmp_path / "a&b<c>", REPO))
    assert "a&b<c>" in (r.findtext(".//t:Exec/t:Arguments", namespaces=NS) or "")


def test_einrichten_ruft_create_mit_xml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = Schtasks()
    assert auto.aufgabe(tmp_path, "einrichten", trocken=False, schtasks=s, ordner=tmp_path / "bericht") == 0
    assert len(s.aufrufe) == 1
    a = s.aufrufe[0]
    assert a[:3] == ["/Create", "/TN", auto.AUFGABE] and a[-1] == "/F" and "/XML" in a
    assert Path(a[a.index("/XML") + 1]).parent == tmp_path / "bericht"
    assert s.xml_inhalt is not None and "InteractiveToken" in s.xml_inhalt
    assert auto.AUFGABE in capsys.readouterr().out


def test_einrichten_fehler_gibt_exit_1(tmp_path: Path) -> None:
    s = Schtasks({"/Create": (1, "FEHLER: Zugriff verweigert")})
    assert auto.aufgabe(tmp_path, "einrichten", trocken=False, schtasks=s, ordner=tmp_path) == 1


@pytest.mark.parametrize("modus", ["einrichten", "pruefen", "zurueck"])
def test_trocken_ruft_schtasks_nicht(tmp_path: Path, capsys: pytest.CaptureFixture[str], modus: str) -> None:
    s = Schtasks()
    assert auto.aufgabe(tmp_path, modus, trocken=True, schtasks=s, ordner=tmp_path / "b") == 0  # type: ignore[arg-type]
    assert s.aufrufe == []
    assert not (tmp_path / "b").exists()
    out = capsys.readouterr().out
    assert {"einrichten": "/Create", "pruefen": "/Query", "zurueck": "/Delete"}[modus] in out
    assert ("StartWhenAvailable" in out) == (modus == "einrichten")


def test_pruefen_fehlt_bei_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = Schtasks({"/Query": (1, "FEHLER: Das System kann die Datei nicht finden.")})
    assert auto.aufgabe(tmp_path, "pruefen", trocken=False, schtasks=s) == 1
    assert s.aufrufe == [["/Query", "/TN", auto.AUFGABE, "/XML"]]
    assert capsys.readouterr().out.strip().endswith("fehlt")


def test_pruefen_bereit_auch_mit_utf16_deklaration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ausgabe = _xml(tmp_path).replace('encoding="UTF-8"', 'encoding="UTF-16"')
    assert "UTF-16" in ausgabe
    s = Schtasks({"/Query": (0, ausgabe)})
    assert auto.aufgabe(tmp_path, "pruefen", trocken=False, schtasks=s, python=PY, arbeitsordner=REPO) == 0
    assert capsys.readouterr().out.strip().endswith("bereit")


def test_pruefen_aktion_weicht_ab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    anders = auto.aufgabe_xml(Path("C:/anderes/python.exe"), tmp_path, REPO)
    s = Schtasks({"/Query": (0, anders)})
    assert auto.aufgabe(tmp_path, "pruefen", trocken=False, schtasks=s, python=PY, arbeitsordner=REPO) == 1
    assert "aktion weicht ab" in capsys.readouterr().out


def test_pruefen_unlesbares_xml_weicht_ab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = Schtasks({"/Query": (0, "kein xml")})
    assert auto.aufgabe(tmp_path, "pruefen", trocken=False, schtasks=s, python=PY, arbeitsordner=REPO) == 1
    assert "aktion weicht ab" in capsys.readouterr().out


def test_zurueck_ruft_delete(tmp_path: Path) -> None:
    s = Schtasks()
    assert auto.aufgabe(tmp_path, "zurueck", trocken=False, schtasks=s) == 0
    assert s.aufrufe == [["/Delete", "/TN", auto.AUFGABE, "/F"]]


def test_zurueck_fehler_gibt_exit_1(tmp_path: Path) -> None:
    assert auto.aufgabe(tmp_path, "zurueck", trocken=False, schtasks=Schtasks({"/Delete": (1, "FEHLER")})) == 1


@pytest.mark.parametrize(("flags", "modus"), [(["--trocken"], "einrichten"), (["--pruefen"], "pruefen"),
                                              (["--zurueck", "--trocken"], "zurueck")])
def test_cli_auto_aufgabe_abgezweigt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str],
                                     modus: str) -> None:
    gesehen: list[tuple[Path, str, bool, object]] = []

    def fake(wurzel: Path, m: str, *, trocken: bool, schtasks: object, **_: object) -> int:
        gesehen.append((wurzel, m, trocken, schtasks))
        return 0

    def nie(*_: object, **__: object) -> object:
        raise AssertionError("auto aufgabe darf nie als Lauf starten")

    def echt(argv: list[str]) -> tuple[int, str]:
        raise AssertionError("echtes schtasks")
    monkeypatch.setattr(auto, "aufgabe", fake)
    monkeypatch.setattr(auto, "lauf", nie)
    monkeypatch.setattr(auto, "standard_werkzeuge", nie)
    monkeypatch.setattr(auto, "standard_schtasks", echt)
    assert main(["auto", "aufgabe", str(tmp_path), *flags]) == 0
    assert gesehen == [(tmp_path, modus, "--trocken" in flags, echt)]


def test_cli_auto_aufgabe_fehler_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def nie(*_: object, **__: object) -> int:
        raise AssertionError("nicht aufrufen")
    monkeypatch.setattr(auto, "aufgabe", nie)
    assert main(["auto", "aufgabe"]) == 2
    assert main(["auto", "aufgabe", str(tmp_path / "gibt-es-nicht")]) == 2
    assert main(["auto", "aufgabe", str(tmp_path), "--pruefen", "--zurueck"]) == 2


def test_standard_schtasks_dekodiert_und_faengt_fehler(monkeypatch: pytest.MonkeyPatch) -> None:
    gesehen: dict[str, object] = {}

    def run(argv: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        gesehen["argv"], gesehen["kw"] = argv, kw
        return subprocess.CompletedProcess(argv, 0, "Büro".encode(), b"")
    monkeypatch.setattr(subprocess, "run", run)
    assert auto.standard_schtasks(["/Query"]) == (0, "Büro")
    assert gesehen["argv"] == ["schtasks", "/Query"]
    assert gesehen["kw"]["stdin"] == subprocess.DEVNULL  # type: ignore[index]

    def kaputt(argv: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError("x")
    monkeypatch.setattr(subprocess, "run", kaputt)
    assert auto.standard_schtasks(["/Query"])[0] == 1
