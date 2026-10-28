"""Stufe 2 des Selbstbilds: Mathe auf dem Codegraphen (nur lesend auf Stufe 1)."""
import re

_TESTNAME = re.compile(
    r"^(test_.*\.py|.*_test\.py|.*_test\.go|.*\.test\.(ts|tsx|js)|.*\.spec\..*)$")


def ist_testdatei(pfad: str) -> bool:
    """Testdatei nach Namensmuster oder liegt unter einem `tests/`- bzw. `__tests__/`-Ordner (Ruling S2)."""
    teile = pfad.replace("\\", "/").split("/")
    if any(t in ("tests", "__tests__") for t in teile[:-1]):
        return True
    return _TESTNAME.match(teile[-1]) is not None
