"""Waechter-Hook (PreToolUse, Stufe 3, T8): verhindert Schreibzugriffe auf `verboten`-Orte der Strukturregeln.

NUR Standardbibliothek. Blockiert hart nur bei eindeutigen Regeln (`verboten`); bei jedem eigenen Fehler, bei
anderen Werkzeugen und ausserhalb von Repos mit `selbstbild.yaml` gibt er nichts aus. Exit immer 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from selbstbild import struktur

SCHREIBER = {"Write": "file_path", "Edit": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}


def _entscheiden(eingabe: object) -> str | None:
    if not isinstance(eingabe, dict):
        return None
    schluessel = SCHREIBER.get(str(eingabe.get("tool_name")))
    werkzeug = eingabe.get("tool_input")
    if schluessel is None or not isinstance(werkzeug, dict):
        return None
    ziel = werkzeug.get(schluessel)
    if not isinstance(ziel, str) or not ziel:
        return None
    cwd = eingabe.get("cwd")
    harte = [v for v in struktur.pruefe_schreibziel(Path(ziel), Path(cwd) if isinstance(cwd, str) and cwd else None)
             if v.hart]
    if not harte:
        return None
    v = harte[0]
    return f"{v.regel}: {v.grund}. {v.hinweis or ''}".rstrip()


def main() -> int:
    try:
        roh = sys.stdin.buffer.read().decode("utf-8")
        grund = _entscheiden(json.loads(roh))
        if grund:
            sys.stdout.write(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "permissionDecision": "deny",
                "permissionDecisionReason": grund}}))
            sys.stdout.flush()
    except Exception:  # noqa: BLE001 - der Waechter darf nie wegen eigener Fehler blockieren
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
