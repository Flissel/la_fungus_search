"""Stufe 2 des Selbstbilds: Mathe auf dem Codegraphen (nur lesend auf Stufe 1)."""
import re

_TESTNAME = re.compile(
    r"^(test_.*\.py|.*_test\.py|.*_test\.go|.*\.test\.(ts|tsx|js)|.*\.spec\..*)$")


def ist_testdatei(pfad: str) -> bool:
    """Testdatei nach Namensmuster oder liegt unter einem `tests/`- bzw. `__tests__/`-Ordner (Ruling S2)."""
    teile = pfad.replace("\\", "/").split("/")
    if any(t in ("tests", "__tests__") for t in teile[:-1]):
        return True
    return teile[-1] == "conftest.py" or _TESTNAME.match(teile[-1]) is not None


_GENERIERT_ORDNER = frozenset({"dist", "build", "out", ".next", ".nuxt", "coverage", "node_modules"})
_GENERIERT_NAME = re.compile(r"^(.*\.min\..*|.*\.bundle\..*|.*-bundle\..*)$")


def ist_generiert(pfad: str) -> bool:
    """Build- oder generierte Ausgabe (Ruling S9): Ordnersegment dist/build/out/.next/.nuxt/coverage/
    node_modules, `dist-*` oder `*-dist`, oder Dateiname `*.min.*`, `*.bundle.*`, `*-bundle.*`."""
    teile = pfad.replace("\\", "/").split("/")
    for t in teile[:-1]:
        if t in _GENERIERT_ORDNER or t.startswith("dist-") or t.endswith("-dist"):
            return True
    return _GENERIERT_NAME.match(teile[-1]) is not None
