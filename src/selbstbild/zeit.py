"""Zeitachse: Aenderungsverlauf je Datei und daraus abgeleitete Lebendigkeit."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from selbstbild.quelle import _git_args, _git_env

_KOPF = re.compile(rb"^[0-9a-f]{40,64}\t\d+\t")
_SEKUNDEN_PRO_TAG = 86400


@dataclass(frozen=True)
class Aenderung:
    commit: str
    zeit: int
    autor: str


def _pfad(roh: bytes) -> str:
    return roh.decode("utf-8", errors="surrogateescape")


def verlauf(wurzel: Path, bis_commit: str) -> dict[str, tuple[Aenderung, ...]]:
    """Pfad am Stand `bis_commit` -> Aenderungen, neueste zuerst.

    Ein einziger `git log`-Lauf (neueste zuerst). Rueckwaerts wird eine Abbildung
    "Pfad im damaligen Stand -> Pfad am Ende" gefuehrt: Umbenennungen (R) schreiben
    die Historie des alten Pfads auf den neuen; Kopien (C) gelten als Neuanlage.
    Merges tragen ihren Diff gegen den ersten Elternteil bei (`-m --first-parent`).
    """
    wurzel = Path(wurzel)
    baum = subprocess.run(
        _git_args("ls-tree", "-r", "--name-only", "-z", bis_commit),
        cwd=wurzel, env=_git_env(), capture_output=True, check=True,
    ).stdout
    abbildung: dict[str, str] = {
        p: p for p in (_pfad(s) for s in baum.split(b"\0") if s)
    }
    roh = subprocess.run(
        _git_args(
            "-c", "log.showSignature=false", "log", "--no-show-signature",
            "-m", "--first-parent", "--format=%x00%H%x09%ct%x09%an",
            "--name-status", "-M", "-z", bis_commit,
        ),
        cwd=wurzel, env=_git_env(), capture_output=True, check=True,
    ).stdout

    erg: dict[str, list[Aenderung]] = {}
    token = roh.split(b"\0")
    # Je Commit: (Aenderung, [(status, pfad, alter_pfad | None)])
    commits: list[tuple[Aenderung, list[tuple[str, str, str | None]]]] = []
    i = 0
    while i < len(token):
        t = token[i].lstrip(b"\n")
        i += 1
        if not t:
            continue
        if _KOPF.match(t):
            sha, zeit, autor = t.split(b"\t", 2)
            commits.append((
                Aenderung(sha.decode("ascii"), int(zeit), autor.decode("utf-8", "replace")),
                [],
            ))
            continue
        status = t.decode("ascii", "replace")[:1]
        if status in ("R", "C"):
            alt, neu = _pfad(token[i]), _pfad(token[i + 1])
            i += 2
            commits[-1][1].append((status, neu, alt))
        else:
            commits[-1][1].append((status, _pfad(token[i]), None))
            i += 1

    for aenderung, eintraege in commits:
        neue_abb = dict(abbildung)
        for status, pfad, alt in eintraege:
            ziel = abbildung.get(pfad)
            if ziel is None:
                continue  # Datei existiert am Ende nicht (mehr) unter diesem Namen
            erg.setdefault(ziel, []).append(aenderung)
            if status == "R" and alt is not None:
                neue_abb.pop(pfad, None)
                neue_abb[alt] = ziel
            elif status in ("A", "C"):
                neue_abb.pop(pfad, None)  # davor war es eine andere Datei
        abbildung = neue_abb
    return {p: tuple(a) for p, a in sorted(erg.items())}


def lebendigkeit(
    aenderungen: Sequence[Aenderung], stichtag: int, halbwertszeit_tage: float = 30.0,
) -> float:
    """Summe 0.5^((stichtag - zeit) / halbwertszeit) ueber Aenderungen bis zum Stichtag."""
    hwz = halbwertszeit_tage * _SEKUNDEN_PRO_TAG
    return sum(0.5 ** ((stichtag - a.zeit) / hwz) for a in aenderungen if a.zeit <= stichtag)


def zeit_attribute(
    verlauf_eintrag: Sequence[Aenderung], stichtag: int,
) -> tuple[tuple[str, str], ...]:
    bis = [a for a in verlauf_eintrag if a.zeit <= stichtag]
    attribute = {
        "aenderungen": str(len(bis)),
        "zuletzt": str(max((a.zeit for a in bis), default=0)),
        "autoren": str(len({a.autor for a in bis})),
        "lebendigkeit": f"{lebendigkeit(bis, stichtag):.4f}",
    }
    return tuple(sorted(attribute.items()))
