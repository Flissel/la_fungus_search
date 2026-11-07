"""Not-Aus (Datei STOP), Tagesgrenze der Helfer-Runden und Speicherprüfung."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _budgetdatei(ordner: Path, jetzt: datetime) -> Path:
    return ordner / f"budget-{jetzt.date().isoformat()}.jsonl"


def budget_pruefen(ordner: Path, tagesgrenze: int, *, jetzt: datetime) -> str | None:
    """`"stop"` bei Datei STOP, `"tagesgrenze"` ab `tagesgrenze` Zeilen heute (auch bei Lesefehler), sonst `None`."""
    if (ordner / "STOP").exists():
        return "stop"
    datei = _budgetdatei(ordner, jetzt)
    if not datei.exists():
        return None
    try:
        with datei.open("r", encoding="utf-8") as f:
            zeilen = sum(1 for z in f if z.strip())
    except (OSError, UnicodeDecodeError):
        return "tagesgrenze"
    return "tagesgrenze" if zeilen >= tagesgrenze else None


def budget_buchen(ordner: Path, ergebnis: str, dauer_s: float, *, jetzt: datetime) -> None:
    """Hängt eine Zeile `{"zeit", "ergebnis", "dauer_s"}` an die Tagesdatei an."""
    ordner.mkdir(parents=True, exist_ok=True)
    zeile = json.dumps({"zeit": jetzt.isoformat(), "ergebnis": ergebnis, "dauer_s": dauer_s}, ensure_ascii=False)
    with _budgetdatei(ordner, jetzt).open("a", encoding="utf-8") as f:
        f.write(zeile + "\n")


def frei_gb(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    """`MemAvailable` in GB; fehlt der Wert oder schlägt das Lesen fehl, `None`."""
    try:
        text = meminfo.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for zeile in text.splitlines():
        if zeile.startswith("MemAvailable:"):
            teile = zeile.split()
            try:
                return int(teile[1]) / 1024 / 1024
            except (IndexError, ValueError):
                return None
    return None
