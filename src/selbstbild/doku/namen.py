"""Space-Namen normalisieren (Stufe 4, Plan P2): `sales-claw`, `sales_claw`, `Sales_Claw` und der Dateiname
`015_sales_claw.md` meinen denselben Space. Task 5 (Events) und Task 6 (neue Space-Dokus) benutzen dieselbe Regel."""

from __future__ import annotations


def space_schluessel(name: str) -> str:
    """Vergleichsschluessel eines Space-Namens: klein, `-` gleich `_`, ohne Leerraum am Rand."""
    return name.strip().lower().replace("-", "_")
