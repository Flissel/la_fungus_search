"""Dateisperre fuer Selbstbild-Laeufe (Stufe 6a, Task 1).

Die Sperrdatei enthaelt nur `{"pid": int, "start": iso}` -- nie etwas Geheimes. Angelegt wird sie exklusiv
(`O_CREAT|O_EXCL`). Verwaist ist sie, wenn ihre PID nicht mehr lebt oder ihre mtime aelter als `veraltet_s` ist
(Ruling R3: gemessen an der mtime, nicht am Feld `start`). Eine unlesbare Sperrdatei mit frischer mtime gilt als
belegt. Die Uebernahme geschieht per Umbenennen (wie `doku.stand._veraltet_uebernehmen`): nur einer gewinnt.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class SperreBelegt(Exception):
    """Ein anderer Lauf haelt die Sperre."""


def pid_lebt_standard(pid: int) -> bool:
    """Lebt der Prozess? `psutil.pid_exists`, sonst unter Windows `OpenProcess` ueber ctypes, sonst `os.kill(pid, 0)`."""
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:
        pass
    else:
        return bool(psutil.pid_exists(pid))
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return bool(kernel32.GetLastError() == 5)  # ERROR_ACCESS_DENIED: lebt, nur nicht oeffnbar
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                return code.value == 259  # STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _pid_lesen(text: str) -> int | None:
    try:
        daten = json.loads(text)
        pid = daten["pid"]
    except (ValueError, TypeError, KeyError):
        return None
    return pid if isinstance(pid, int) and not isinstance(pid, bool) else None


def _anlegen(pfad: Path, inhalt: str) -> None:
    fd = os.open(pfad, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, inhalt.encode("utf-8"))
    finally:
        os.close(fd)


def _verwaist_uebernehmen(pfad: Path, veraltet_s: float, pid_lebt: Callable[[int], bool]) -> bool:
    """True, wenn die vorhandene Sperre verwaist war und wir sie weggenommen haben (jetzt neu anlegen). False, wenn
    sie gehalten wird oder ein anderer schneller war. Verschwindet sie zwischendurch: True (neu versuchen)."""
    try:
        alter = time.time() - pfad.stat().st_mtime
        text = pfad.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return True
    pid = _pid_lesen(text)
    verwaist = alter > veraltet_s or (pid is not None and not pid_lebt(pid))
    if not verwaist:
        return False
    weg = pfad.with_name(f"{pfad.name}.verwaist.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        os.rename(pfad, weg)
    except FileNotFoundError:
        return True  # ein anderer hat sie schon weggenommen; neu versuchen (O_EXCL entscheidet)
    except OSError:
        return False
    try:
        gleich = weg.read_text(encoding="utf-8", errors="replace") == text
    except OSError:
        gleich = False
    if not gleich:  # wir haben eine frische Sperre eines anderen erwischt: zurueckgeben
        try:
            if not pfad.exists():
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
def sperre(pfad: Path, *, veraltet_s: float = 7200.0,
           pid_lebt: Callable[[int], bool] = pid_lebt_standard) -> Iterator[bool]:
    """Sperre `pfad` halten. Liefert True, wenn eine verwaiste Sperre uebernommen wurde, sonst False. Haelt ein
    anderer Lauf sie: `SperreBelegt` (die fremde Sperre bleibt unberuehrt)."""
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    inhalt = json.dumps({"pid": os.getpid(), "start": datetime.now().isoformat(timespec="seconds")})
    uebernommen = False
    try:
        _anlegen(pfad, inhalt)
    except FileExistsError:
        if not _verwaist_uebernehmen(pfad, veraltet_s, pid_lebt):
            raise SperreBelegt(pfad.name) from None
        try:
            _anlegen(pfad, inhalt)
        except FileExistsError:
            raise SperreBelegt(pfad.name) from None
        uebernommen = True
    try:
        yield uebernommen
    finally:
        try:
            if pfad.read_text(encoding="utf-8", errors="replace") == inhalt:
                pfad.unlink()
        except OSError:
            pass
