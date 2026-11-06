"""Fremdcode-Pruefer: Globs relativ zur aeusseren Wurzel (gemeinsam fuer Quelle und Funde)."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from selbstbild._glob import glob_passt, glob_regex


def fremd_pruefer(muster: Sequence[str], aussen: str) -> Callable[[str, str], bool]:
    """Liefert `(repo_id, pfad) -> bool`; `False` fuer fremde Aussenwurzeln und bei leerer Liste."""
    regexe = [glob_regex(m) for m in muster]

    def pruefe(repo_id: str, pfad: str) -> bool:
        if not regexe:
            return False
        if repo_id == aussen:
            voll = pfad
        elif repo_id.startswith(aussen + "/"):
            voll = repo_id[len(aussen) + 1 :] + "/" + pfad
        else:
            return False
        return glob_passt(voll, regexe)

    return pruefe


def ist_fremd_id(pruefer: Callable[[str, str], bool], graph_id: str) -> bool:
    """Gilt fuer `symbol:<repo>:<pfad>#<q>` und `datei:<repo>:<pfad>`; andere Ids sind nie fremd."""
    teile = graph_id.split(":", 2)
    if len(teile) != 3 or teile[0] not in ("symbol", "datei"):
        return False
    # wie tot._pfad und sammeln._id_teile: der Qualifizierer steht hinter dem LETZTEN `#`; eine Datei-Id hat keinen
    pfad = teile[2].rpartition("#")[0] if teile[0] == "symbol" and "#" in teile[2] else teile[2]
    return pruefer(teile[1], pfad)
