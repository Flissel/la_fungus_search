"""Gemeinsamer Client für den Freigabe-Eingang von sales-claw (`/api/freigaben/*`).

Erzeuger (Debugger-Claw, Selbstbild, Werkstatt) melden hier, was auf den Betreiber wartet. Die Herkunft leitet
der Server aus dem Schlüssel ab; der Rumpf trägt keine. Jeder Fehler (Netz, HTTP, JSON, Schlüsseldatei) wird
mit Fehlertyp (und HTTP-Status) geloggt und ergibt `None`/`False`; es kommt nie eine Ausnahme nach außen.
Geloggt wird nie der Schlüssel, der Dateiinhalt, ein Treffer des Schlüsselmusters oder ein Rumpf.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from selbstbild.funde.text import pruefen

__all__ = ["FreigabeEingang"]

LOG = logging.getLogger("selbstbild.freigabe")
TIMEOUT_S = 10
_PFAD = "/api/freigaben"


class FreigabeEingang:
    def __init__(self, url: str, schluessel_datei: Path, *,
                 oeffnen: Callable[..., Any] = urllib.request.urlopen) -> None:
        self.url = url.rstrip("/")
        self.schluessel_datei = Path(schluessel_datei)
        self._oeffnen = oeffnen

    def __repr__(self) -> str:
        return "FreigabeEingang()"

    def _anfrage(self, methode: str, pfad: str, rumpf: dict[str, Any] | None = None) -> Any | None:
        """Eine Anfrage; die Antwort als JSON oder `None` bei jedem Fehler."""
        try:
            schluessel = self.schluessel_datei.read_text(encoding="utf-8").strip()  # pro Anfrage, CRLF tolerant
            if not schluessel:
                raise ValueError("schluessel_leer")
            kopf = {"Authorization": f"Bearer {schluessel}"}
            daten = None
            if rumpf is not None:
                daten = json.dumps(rumpf).encode("utf-8")
                kopf["Content-Type"] = "application/json"
            req = urllib.request.Request(f"{self.url}{_PFAD}{pfad}", data=daten, headers=kopf, method=methode)
            with self._oeffnen(req, timeout=TIMEOUT_S) as antwort:
                return json.loads(antwort.read())
        except urllib.error.HTTPError as e:
            LOG.warning("Freigabe-Eingang %s %s: HTTP %d", methode, pfad.split("/")[1] if "/" in pfad else pfad, e.code)
        except Exception as e:
            LOG.warning("Freigabe-Eingang %s %s: %s", methode, pfad.split("/")[1] if "/" in pfad else pfad,
                        type(e).__name__)
        return None

    def eintragen(self, art: str, quelle_ref: str, titel: str, link: str | None = None,
                  details: dict[str, Any] | None = None) -> str | None:
        """Meldet einen Eintrag; gibt die Kennung zurück. Bei Schlüsselmuster in Titel/Details wird nichts gesendet."""
        texte = [titel, link or "", json.dumps(details, ensure_ascii=False) if details else ""]
        for t in texte:
            if (muster := pruefen(t)) is not None:
                LOG.warning("Freigabe-Eingang eintrag: nicht gesendet (Schlüssel-Verdacht: %s)", muster)
                return None
        rumpf: dict[str, Any] = {"art": art, "quelle_ref": quelle_ref, "titel": titel}
        if link is not None:
            rumpf["link"] = link
        if details is not None:
            rumpf["details"] = details
        antwort = self._anfrage("POST", "/eintrag", rumpf)
        kennung = antwort.get("kennung") if isinstance(antwort, dict) else None
        return kennung if isinstance(kennung, str) else None

    def schliessen(self, quelle_ref: str, status: str = "erledigt") -> int | None:
        antwort = self._anfrage("POST", "/schliessen", {"quelle_ref": quelle_ref, "status": status})
        n = antwort.get("geschlossen") if isinstance(antwort, dict) else None
        return n if isinstance(n, int) and not isinstance(n, bool) else None

    def _liste(self, pfad: str, name: str) -> list[dict[str, Any]] | None:
        antwort = self._anfrage("GET", pfad)
        zeilen = antwort.get(name) if isinstance(antwort, dict) else None
        return [z for z in zeilen if isinstance(z, dict)] if isinstance(zeilen, list) else None

    def offene(self) -> list[dict[str, Any]] | None:
        return self._liste("/offen", "eintraege")

    def auftraege(self) -> list[dict[str, Any]] | None:
        return self._liste("/auftraege", "auftraege")

    def ergebnis(self, auftrag_id: str, ok: bool, fehler: str | None = None) -> bool:
        rumpf: dict[str, Any] = {"ok": ok}
        if fehler is not None:
            rumpf["fehler"] = fehler
        antwort = self._anfrage("POST", f"/auftraege/{auftrag_id}", rumpf)
        return isinstance(antwort, dict) and antwort.get("ok") is True
