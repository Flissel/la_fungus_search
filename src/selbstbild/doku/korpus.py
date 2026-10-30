"""Korpus: Durchlauf mit Ausschluss VOR dem Betreten, Manifest, Symbol-Suche.

`Path.rglob` betritt jeden Ordner und stolpert z. B. ueber den toten Symlink
`node_modules/@x/preload`. Hier wird mit os.walk gelaufen und ausgeschlossene
Ordner werden aus der Liste der Unterordner genommen, bevor sie betreten werden.
"""

from __future__ import annotations

import ast
import os
import subprocess
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

from typing import TYPE_CHECKING

from .konfig import Konfig, SpaceKonfig

if TYPE_CHECKING:  # nur fuer Typpruefer; zur Laufzeit lazy
    from ._manifest_gate2 import Document, Manifest

def _manifest_modul() -> ModuleType:
    """Das gepinnte Manifest-Modul `_manifest_gate2` (nur stdlib; byte-gleich zur Fassung aus la-fungus-search, D2)."""
    from . import _manifest_gate2

    return _manifest_gate2


_OEFFENTLICH = frozenset(
    {"Document", "Manifest", "build_manifest", "load_manifest", "manifest_digest", "save_manifest"}
)


def manifest_verfuegbar() -> bool:
    """Das Manifest liegt im Paket (D2): immer verfuegbar."""
    return True


def __getattr__(name: str) -> object:
    """Manifest-Namen erst beim ersten Zugriff laden; der Import des Moduls bleibt ohne Submodul heil."""
    if name in _OEFFENTLICH:
        return getattr(_manifest_modul(), name)
    raise AttributeError(name)


def ast_lesen(pfad: Path) -> ast.Module | None:
    """Fremden Python-Quelltext parsen, ohne Warnungen (z. B. ungueltige Escapes) auszugeben; None bei Fehlern."""
    try:
        text = Path(pfad).read_text(encoding="utf-8")
        # catch_warnings ist nicht thread-sicher; der Loop laeuft einfaedig.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
            return ast.parse(text)
    except (SyntaxError, UnicodeDecodeError, ValueError, OSError, RecursionError):
        return None


def ist_testdatei(pfad: "Path | str") -> bool:
    """Testcode: Ordner `tests`/`test`, `test_*.py`, `*_test.py`, `conftest.py`."""
    teile = Path(pfad).as_posix().split("/")
    if any(t in {"tests", "test"} for t in teile[:-1]):
        return True
    name = teile[-1]
    return name == "conftest.py" or (
        name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))
    )


def _pfad_ausgeschlossen(ordner: str, suffixe: list[tuple[str, ...]]) -> bool:
    segmente = tuple(Path(ordner).as_posix().strip("/").split("/"))
    return any(len(s) <= len(segmente) and segmente[len(segmente) - len(s):] == s for s in suffixe)


def dateien(wurzel: Path, konfig: Konfig, endungen: tuple[str, ...]) -> list[Path]:
    """Alle Dateien mit passender Endung unter `wurzel`, sortiert; Ausschluss vor dem Betreten."""
    suffixe = [tuple(p.split("/")) for p in konfig.ausschluss_pfade]
    ergebnis: list[Path] = []
    for ordner, unterordner, namen in os.walk(
        wurzel, topdown=True, onerror=lambda _fehler: None, followlinks=False
    ):
        unterordner[:] = sorted(
            u
            for u in unterordner
            if u not in konfig.ausschluss_namen
            and not _pfad_ausgeschlossen(os.path.join(ordner, u), suffixe)
        )
        for name in namen:
            if name.endswith(endungen):
                ergebnis.append(Path(ordner, name))
    return sorted(ergebnis)


def _korpus_pfad_klasse(konfig: Konfig) -> type[Path]:
    basis = type(Path())

    class KorpusPfad(basis):  # type: ignore[valid-type,misc]
        def rglob(self, pattern: str) -> Iterator[Path]:  # type: ignore[override]
            endung = pattern[1:] if pattern.startswith("*") else pattern
            return iter(dateien(Path(self), konfig, (endung,)))

    return KorpusPfad


def manifest_bauen(space: SpaceKonfig, konfig: Konfig, commit: str) -> Manifest:
    wurzel = _korpus_pfad_klasse(konfig)(konfig.repo / space.wurzel)
    # manifest.py (Submodul, nicht aenderbar) parst fremden Quelltext selbst und warnt dabei.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
        return _manifest_modul().build_manifest(
            wurzel, commit, f"doku-loop/{space.name}", exclude_dirs=konfig.ausschluss_namen
        )


def symbol_an(manifest: "Manifest", file: str, zeile: int) -> "Document | None":
    """Engste Definition in `file`, die `zeile` umschliesst (kleinste Zeilenspanne)."""
    beste = None
    for d in manifest.documents:
        if d.file == file and d.start_line <= zeile <= d.end_line:
            if beste is None or (d.end_line - d.start_line) < (beste.end_line - beste.start_line):
                beste = d
    return beste


def commit_von(repo: Path) -> str:
    aus = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return aus.stdout.strip()
