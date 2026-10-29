"""Hook-Installer (Stufe 3, T9): schreibt/entfernt die zwei Hooks in einer `settings.local.json`.

Fremde Eintraege bleiben unberuehrt; eine Struktur, die nicht dem Settings-Schema entspricht, wird nie repariert
(ValueError, nichts geschrieben).
"""
from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any

SICHERUNG = ".selbstbild-sicherung"
PTH_NAME = "selbstbild-src.pth"
SCHREIB_MATCHER = "Write|Edit|MultiEdit|NotebookEdit"
# (Ereignis, Modul, matcher)
_HOOKS: tuple[tuple[str, str, str | None], ...] = (
    ("UserPromptSubmit", "selbstbild.hooks.einspeisung", None),
    ("PreToolUse", "selbstbild.hooks.waechter", SCHREIB_MATCHER),
)
_MARKER = tuple(f"-m {modul}" for _, modul, _ in _HOOKS)


def _gewollt(python: Path, modul: str) -> dict[str, Any]:
    return {"type": "command", "command": f'"{python.as_posix()}" -m {modul}', "timeout": 5}


def _eigen(hook: dict[str, Any]) -> bool:
    cmd = hook.get("command")
    return isinstance(cmd, str) and any(m in cmd for m in _MARKER)


def _lesen(settings: Path) -> tuple[str, dict[str, Any]]:
    if not settings.exists():
        return "", {}
    text = settings.read_bytes().decode("utf-8")
    try:
        daten = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"settings kein gueltiges JSON: {settings} ({e.msg})") from e
    if not isinstance(daten, dict):
        raise ValueError(f"settings ist kein JSON-Objekt: {settings}")
    return text, daten


def _ereignis(daten: dict[str, Any], name: str, anlegen: bool) -> list[Any] | None:
    """Ereignisliste mit Typpruefung; None, wenn nicht vorhanden (und nicht angelegt)."""
    hooks = daten.get("hooks")
    if hooks is None:
        if not anlegen:
            return None
        hooks = daten["hooks"] = {}
    if not isinstance(hooks, dict):
        raise ValueError("settings: `hooks` ist kein Objekt")
    liste = hooks.get(name)
    if liste is None:
        if not anlegen:
            return None
        liste = hooks[name] = []
    if not isinstance(liste, list):
        raise ValueError(f"settings: hooks.{name} ist keine Liste")
    for gruppe in liste:
        if not isinstance(gruppe, dict):
            raise ValueError(f"settings: Eintrag in hooks.{name} ist kein Objekt")
        inner = gruppe.get("hooks")
        if inner is not None and (not isinstance(inner, list) or not all(isinstance(h, dict) for h in inner)):
            raise ValueError(f"settings: hooks.{name}[].hooks ist keine Liste von Objekten")
    return liste


def _eigene_entfernen(liste: list[Any]) -> bool:
    """Entfernt eigene Hooks; dadurch leere Gruppen fallen weg. True bei Aenderung."""
    geaendert = False
    for gruppe in list(liste):
        inner = gruppe.get("hooks")
        if not inner:
            continue
        rest = [h for h in inner if not _eigen(h)]
        if len(rest) == len(inner):
            continue
        geaendert = True
        if rest:
            gruppe["hooks"] = rest
        else:
            liste.remove(gruppe)
    return geaendert


def _erfuellt(liste: list[Any], soll: dict[str, Any], matcher: str | None) -> bool:
    """Genau ein eigener Hook, gleich dem Soll, in einer Gruppe mit richtigem matcher."""
    treffer = [(g, h) for g in liste for h in g.get("hooks") or [] if _eigen(h)]
    if len(treffer) != 1:
        return False
    gruppe, hook = treffer[0]
    return hook == soll and gruppe.get("matcher") == matcher


def _diff(alt: str, neu: str, name: str) -> str:
    return "".join(difflib.unified_diff(alt.splitlines(keepends=True), neu.splitlines(keepends=True),
                                        fromfile=f"{name} (alt)", tofile=f"{name} (neu)"))


def _schreiben(settings: Path, alt: str, neu: str, vorhanden: bool) -> None:
    settings.parent.mkdir(parents=True, exist_ok=True)
    if vorhanden:
        sich = settings.with_name(settings.name + SICHERUNG)
        if not sich.exists():
            sich.write_bytes(alt.encode("utf-8"))
    tmp = settings.with_name(f"{settings.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(neu.encode("utf-8"))
        os.replace(tmp, settings)
    finally:
        tmp.unlink(missing_ok=True)


def _text(daten: dict[str, Any]) -> str:
    return json.dumps(daten, indent=2, ensure_ascii=False) + "\n"


def einspielen(settings: Path, python: Path, *, trocken: bool = False) -> str:
    alt, daten = _lesen(settings)
    vorhanden = settings.exists()
    geaendert = False
    for ereignis, modul, matcher in _HOOKS:
        soll = _gewollt(python, modul)
        liste = _ereignis(daten, ereignis, anlegen=False)
        if liste is not None and _erfuellt(liste, soll, matcher):
            continue
        liste = _ereignis(daten, ereignis, anlegen=True)
        assert liste is not None
        _eigene_entfernen(liste)
        gruppe: dict[str, Any] = {"hooks": [soll]}
        if matcher is not None:
            gruppe = {"matcher": matcher, **gruppe}
        liste.append(gruppe)
        geaendert = True
    if not geaendert:
        return ""
    neu = _text(daten)
    if not trocken:
        _schreiben(settings, alt, neu, vorhanden)
    return _diff(alt, neu, settings.name)


def entfernen(settings: Path) -> str:
    alt, daten = _lesen(settings)
    geaendert = False
    for ereignis, _, _ in _HOOKS:
        liste = _ereignis(daten, ereignis, anlegen=False)
        if liste is None or not _eigene_entfernen(liste):
            continue
        geaendert = True
        if not liste:
            del daten["hooks"][ereignis]
    if not geaendert:
        return ""
    if not daten["hooks"]:
        del daten["hooks"]
    neu = _text(daten)
    _schreiben(settings, alt, neu, True)
    return _diff(alt, neu, settings.name)


def pth_anlegen(venv: Path, src: Path) -> Path:
    """`selbstbild-src.pth` in site-packages des venv (F7), idempotent."""
    ziel = venv / "Lib" / "site-packages"
    if not ziel.is_dir():
        kandidaten = sorted(p for p in (venv / "lib").glob("python*/site-packages") if p.is_dir())
        if not kandidaten:
            raise FileNotFoundError(f"kein site-packages in {venv}")
        ziel = kandidaten[0]
    pth = ziel / PTH_NAME
    inhalt = str(src.resolve()) + "\n"
    if not pth.is_file() or pth.read_bytes().decode("utf-8") != inhalt:
        pth.write_bytes(inhalt.encode("utf-8"))
    return pth
