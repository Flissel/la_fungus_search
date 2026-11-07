"""Not-Aus (Datei STOP), Tagesgrenze der Helfer-Runden und Speicherprüfung."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _budgetdatei(ordner: Path, jetzt: datetime) -> Path:
    return ordner / f"budget-{jetzt.date().isoformat()}.jsonl"


def _nachtrag(zeile: str) -> bool:
    try:
        roh = json.loads(zeile)
    except ValueError:
        return False
    return isinstance(roh, dict) and roh.get("nachtrag") is True


def budget_pruefen(ordner: Path, tagesgrenze: int, *, jetzt: datetime) -> str | None:
    """`"stop"` bei Datei STOP, `"tagesgrenze"` ab `tagesgrenze` gezählten Zeilen heute, sonst `None`.

    R41: Gezählt wird jede nicht leere Zeile außer den Nachträgen (`"nachtrag": true`), also je
    Runde die vorab gebuchte `laeuft`-Zeile. Abwärtskompatibel: alte Dateien ohne `laeuft` (je
    Runde eine Ergebniszeile, kein Nachtrag) zählen wie bisher je Zeile; unlesbare Zeilen zählen
    mit. Ein Lesefehler der Datei gilt als `"tagesgrenze"` (fail-closed).
    """
    if (ordner / "STOP").exists():
        return "stop"
    datei = _budgetdatei(ordner, jetzt)
    if not datei.exists():
        return None
    try:
        with datei.open("r", encoding="utf-8") as f:
            zeilen = sum(1 for z in f if z.strip() and not _nachtrag(z))
    except (OSError, UnicodeDecodeError):
        return "tagesgrenze"
    return "tagesgrenze" if zeilen >= tagesgrenze else None


def budget_buchen(ordner: Path, ergebnis: str, dauer_s: float, *, jetzt: datetime, nachtrag: bool = False) -> None:
    """Hängt eine Zeile `{"zeit", "ergebnis", "dauer_s"}` an die Tagesdatei an.

    `nachtrag=True` (R41: das Ergebnis nach der vorab gebuchten `laeuft`-Zeile) setzt zusätzlich
    `"nachtrag": true`; solche Zeilen zählt `budget_pruefen` nicht.
    """
    ordner.mkdir(parents=True, exist_ok=True)
    daten: dict[str, object] = {"zeit": jetzt.isoformat(), "ergebnis": ergebnis, "dauer_s": dauer_s}
    if nachtrag:
        daten["nachtrag"] = True
    zeile = json.dumps(daten, ensure_ascii=False)
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
