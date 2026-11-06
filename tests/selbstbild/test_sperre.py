"""Stufe 6a, Task 1: Dateisperre (exklusiv, verwaist per PID oder mtime, Uebernahme nur durch einen)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from selbstbild.sperre import SperreBelegt, pid_lebt_standard, sperre


def test_sperre_exklusiv(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    with sperre(p) as uebernommen:
        assert uebernommen is False and p.exists()
        with pytest.raises(SperreBelegt):
            with sperre(p):
                pass
        assert p.exists()  # die fremde Sperre bleibt unberuehrt
    assert not p.exists()


def test_sperre_inhalt_nur_pid_und_start(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    with sperre(p):
        daten = json.loads(p.read_text(encoding="utf-8"))
    assert set(daten) == {"pid", "start"} and daten["pid"] == os.getpid()


def test_verwaiste_sperre_tote_pid_wird_uebernommen(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    p.write_text('{"pid": 999999, "start": "2026-11-06T07:30:00"}', encoding="utf-8")
    with sperre(p, pid_lebt=lambda pid: False) as uebernommen:
        assert uebernommen is True
    assert not p.exists()
    assert [f.name for f in tmp_path.iterdir()] == []  # kein Umbenennungsrest


def test_frische_sperre_lebende_pid_ist_belegt(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    p.write_text('{"pid": 1, "start": "2026-11-06T07:30:00"}', encoding="utf-8")
    with pytest.raises(SperreBelegt):
        with sperre(p, pid_lebt=lambda pid: True):
            pass
    assert p.exists()


def test_alte_sperre_lebende_pid_wird_uebernommen(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    p.write_text('{"pid": 1, "start": "2026-11-06T07:30:00"}', encoding="utf-8")
    os.utime(p, (time.time() - 7300, time.time() - 7300))
    with sperre(p, pid_lebt=lambda pid: True) as uebernommen:
        assert uebernommen is True


def test_alter_wird_an_mtime_gemessen_nicht_am_startfeld(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    p.write_text('{"pid": 1, "start": "2000-01-01T00:00:00"}', encoding="utf-8")  # uralt im Feld, frisch per mtime
    with pytest.raises(SperreBelegt):
        with sperre(p, pid_lebt=lambda pid: True):
            pass


def test_kaputte_sperrdatei_gilt_als_verwaist_erst_nach_frist(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    p.write_text("kein json", encoding="utf-8")
    with pytest.raises(SperreBelegt):
        with sperre(p, pid_lebt=lambda pid: True):
            pass
    os.utime(p, (time.time() - 7300, time.time() - 7300))
    with sperre(p, pid_lebt=lambda pid: True) as uebernommen:
        assert uebernommen is True


def test_veraltet_s_einstellbar(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    p.write_text('{"pid": 1, "start": "x"}', encoding="utf-8")
    os.utime(p, (time.time() - 10, time.time() - 10))
    with sperre(p, veraltet_s=5.0, pid_lebt=lambda pid: True) as uebernommen:
        assert uebernommen is True


def test_uebernahme_nur_einer_gewinnt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verliert die Uebernahme das Umbenennen (ein anderer war schneller), ist die Sperre belegt."""
    p = tmp_path / "x.lock"
    p.write_text('{"pid": 999999, "start": "x"}', encoding="utf-8")
    echt = os.rename

    def verloren(src: str | Path, dst: str | Path) -> None:
        echt(src, dst)  # der andere hat schon umbenannt ...
        p.write_text('{"pid": 4242, "start": "neu"}', encoding="utf-8")  # ... und eine frische Sperre angelegt
        raise FileNotFoundError

    monkeypatch.setattr(os, "rename", verloren)
    with pytest.raises(SperreBelegt):
        with sperre(p, pid_lebt=lambda pid: False):
            pass
    assert json.loads(p.read_text(encoding="utf-8"))["pid"] == 4242


def test_freigabe_loescht_fremde_sperre_nicht(tmp_path: Path) -> None:
    p = tmp_path / "x.lock"
    with sperre(p):
        p.write_text('{"pid": 4242, "start": "fremd"}', encoding="utf-8")
    assert p.exists()


def test_pid_lebt_standard() -> None:
    assert pid_lebt_standard(os.getpid()) is True
    assert pid_lebt_standard(0) is False
