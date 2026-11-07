"""Tests fuer `selbstbild.debugger.budget`."""
import json
from datetime import datetime, timezone
from pathlib import Path

from selbstbild.debugger.budget import budget_buchen, budget_pruefen, frei_gb

JETZT = datetime(2026, 11, 7, 12, 0, 0, tzinfo=timezone.utc)


def test_budget_stop(tmp_path: Path) -> None:
    assert budget_pruefen(tmp_path, 20, jetzt=JETZT) is None
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    assert budget_pruefen(tmp_path, 20, jetzt=JETZT) == "stop"


def test_budget_tagesgrenze(tmp_path: Path) -> None:
    gestern = tmp_path / "budget-2026-11-06.jsonl"
    gestern.write_text("{}\n" * 30, encoding="utf-8")
    heute = tmp_path / "budget-2026-11-07.jsonl"
    heute.write_text("{}\n" * 19, encoding="utf-8")
    assert budget_pruefen(tmp_path, 20, jetzt=JETZT) is None
    heute.write_text("{}\n" * 20, encoding="utf-8")
    assert budget_pruefen(tmp_path, 20, jetzt=JETZT) == "tagesgrenze"


def test_budget_unlesbar_ist_tagesgrenze(tmp_path: Path) -> None:
    (tmp_path / "budget-2026-11-07.jsonl").mkdir()
    assert budget_pruefen(tmp_path, 20, jetzt=JETZT) == "tagesgrenze"


def test_budget_buchen(tmp_path: Path) -> None:
    budget_buchen(tmp_path, "ok", 12.5, jetzt=JETZT)
    budget_buchen(tmp_path, "fehler", 1.0, jetzt=JETZT)
    zeilen = (tmp_path / "budget-2026-11-07.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(zeilen) == 2
    assert json.loads(zeilen[0]) == {"zeit": JETZT.isoformat(), "ergebnis": "ok", "dauer_s": 12.5}


def test_frei_gb_liest_memavailable(tmp_path: Path) -> None:
    m = tmp_path / "meminfo"
    m.write_text("MemTotal:       16000000 kB\nMemAvailable:    2097152 kB\n", encoding="utf-8")
    assert frei_gb(m) == 2.0


def test_frei_gb_ohne_zeile_none(tmp_path: Path) -> None:
    m = tmp_path / "meminfo"
    m.write_text("MemTotal:       16000000 kB\n", encoding="utf-8")
    assert frei_gb(m) is None
    assert frei_gb(tmp_path / "gibtsnicht") is None
