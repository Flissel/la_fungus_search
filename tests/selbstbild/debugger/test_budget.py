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


# ---------- R41: Buchung vor dem Helfer, Nachtrag zählt nicht ----------

def test_budget_zaehlt_nur_laeuft_nachtrag_nicht(tmp_path: Path) -> None:
    for _ in range(2):
        budget_buchen(tmp_path, "laeuft", 0.0, jetzt=JETZT)
        budget_buchen(tmp_path, "ok", 3.0, jetzt=JETZT, nachtrag=True)
    zeilen = [json.loads(z) for z in (tmp_path / "budget-2026-11-07.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [z["ergebnis"] for z in zeilen] == ["laeuft", "ok", "laeuft", "ok"]
    assert zeilen[1]["nachtrag"] is True and "nachtrag" not in zeilen[0]
    assert budget_pruefen(tmp_path, 3, jetzt=JETZT) is None  # 2 gezählt, nicht 4
    assert budget_pruefen(tmp_path, 2, jetzt=JETZT) == "tagesgrenze"


def test_budget_alte_datei_ohne_laeuft_zaehlt_je_zeile(tmp_path: Path) -> None:
    alt = "".join(json.dumps({"zeit": "x", "ergebnis": e, "dauer_s": 1.0}) + "\n" for e in ("ok", "fehler", "ok"))
    (tmp_path / "budget-2026-11-07.jsonl").write_text(alt + "kaputt\n", encoding="utf-8")
    assert budget_pruefen(tmp_path, 5, jetzt=JETZT) is None
    assert budget_pruefen(tmp_path, 4, jetzt=JETZT) == "tagesgrenze"  # auch die kaputte Zeile zählt
