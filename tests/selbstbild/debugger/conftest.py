"""Gemeinsame Hilfen der Debugger-Tests: ein Präfix, der einen Enkel mit offenem stdout startet (S6)."""
from __future__ import annotations

import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

# Kind: startet einen Enkel, der das stdout des Kindes erbt und lange schläft, schreibt dessen
# pid nach argv[1], gibt argv[3] Bytes auf stdout aus und schläft dann argv[2] Sekunden
# (0 = sofort beenden).
_ENKEL = (
    "import subprocess, sys, time\n"
    "enkel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=sys.stdout)\n"
    "open(sys.argv[1], 'w').write(str(enkel.pid))\n"
    "sys.stdout.buffer.write(b'x' * int(sys.argv[3]))\n"
    "sys.stdout.buffer.flush()\n"
    "time.sleep(float(sys.argv[2]))\n"
)


@dataclass
class Enkel:
    pid_datei: Path

    def praefix(self, kind_schlaeft_s: float, ausgabe: int = 0) -> list[str]:
        """argv-Präfix; alles, was der Läufer danach anhängt, ignoriert das Kind."""
        return [sys.executable, "-c", _ENKEL, str(self.pid_datei), str(kind_schlaeft_s), str(ausgabe)]

    def pid(self) -> int | None:
        for _ in range(50):
            if self.pid_datei.exists() and self.pid_datei.read_text(encoding="utf-8").strip():
                return int(self.pid_datei.read_text(encoding="utf-8"))
            time.sleep(0.1)
        return None

    def tot_pruefen(self) -> None:
        """POSIX: der Enkel lebt nach dem Timeout nicht mehr (Windows: kein Gruppen-Kill, nur Ende)."""
        if os.name == "nt":
            return
        pid = self.pid()
        assert pid is not None
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        raise AssertionError("Enkel lebt noch")


@pytest.fixture
def enkel(tmp_path: Path) -> Iterator[Enkel]:
    e = Enkel(tmp_path / "enkel.pid")
    yield e
    pid = e.pid() if e.pid_datei.exists() else None
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
