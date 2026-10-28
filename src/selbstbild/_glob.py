"""Glob mit `**` fuer Repo-Pfade (geteilt von Quelle und Aufloeser)."""
from __future__ import annotations

import re
from collections.abc import Iterable


def glob_regex(muster: str) -> re.Pattern[str]:
    """`**` = beliebig viele Pfadsegmente, `*` = innerhalb eines Segments."""
    ende = r"\Z"
    if muster.endswith("/**"):  # `X/**` trifft auch `X` selbst (z. B. Gitlinks)
        muster = muster[:-3]
        ende = r"(?:/.*)?\Z"
    teile: list[str] = []
    i = 0
    while i < len(muster):
        if muster.startswith("**/", i):
            teile.append("(?:.*/)?")
            i += 3
        elif muster.startswith("**", i):
            teile.append(".*")
            i += 2
        elif muster[i] == "*":
            teile.append("[^/]*")
            i += 1
        elif muster[i] == "?":
            teile.append("[^/]")
            i += 1
        else:
            teile.append(re.escape(muster[i]))
            i += 1
    return re.compile("".join(teile) + ende)


def glob_passt(pfad: str, muster: Iterable[re.Pattern[str]]) -> bool:
    return any(m.match(pfad) for m in muster)
