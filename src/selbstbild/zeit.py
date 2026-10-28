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


def _lauf(wurzel: Path, *args: str) -> bytes:
    erg = subprocess.run(_git_args(*args), cwd=wurzel, env=_git_env(), capture_output=True)
    if erg.returncode != 0:
        raise RuntimeError(
            f"git {args[-1] if args else ''} fehlgeschlagen (Exit {erg.returncode}): "
            + erg.stderr.decode("utf-8", "replace").strip()
        )
    return erg.stdout


def verlauf(wurzel: Path, bis_commit: str) -> dict[str, tuple[Aenderung, ...]]:
    """Pfad am Stand `bis_commit` -> Aenderungen, neueste zuerst.

    Ein einziger `git log`-Lauf (neueste zuerst). Rueckwaerts wird eine Abbildung
    "Pfad im damaligen Stand -> Pfad am Ende" gefuehrt: Umbenennungen (R) schreiben
    die Historie des alten Pfads auf den neuen; Kopien (C) gelten als Neuanlage.
    Grenzen von git selbst: Die Umbenennungserkennung unterliegt `diff.renameLimit`
    (bei sehr vielen geaenderten Dateien je Commit erscheint R als D+A und die
    Historie bricht dort ab); wird eine Datei im selben Commit umbenannt und unter
    dem alten Namen neu angelegt, ist die Zuordnung nicht eindeutig.
    Merges tragen ihren Diff gegen den ersten Elternteil bei (`-m --first-parent`).
    """
    wurzel = Path(wurzel)
    baum = _lauf(wurzel, "ls-tree", "-r", "--name-only", "-z", bis_commit)
    abbildung: dict[str, str] = {
        p: p for p in (_pfad(s) for s in baum.split(b"\0") if s)
    }
    roh = _lauf(
        wurzel, "-c", "log.showSignature=false", "-c", "i18n.logOutputEncoding=UTF-8",
        "log", "--no-show-signature", "-m", "--first-parent",
        "--format=%x00%H%x09%ct%x09%an", "--name-status", "-M", "-z", bis_commit,
    )

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
        if not commits:
            raise RuntimeError(f"git log: unerwartete Ausgabe vor dem ersten Commit: {t[:40]!r}")
        if status in ("R", "C"):
            if i + 1 >= len(token):
                raise RuntimeError("git log: Ausgabe abgeschnitten (Umbenennung ohne Pfade)")
            alt, neu = _pfad(token[i]), _pfad(token[i + 1])
            i += 2
            commits[-1][1].append((status, neu, alt))
        else:
            if i >= len(token):
                raise RuntimeError("git log: Ausgabe abgeschnitten (Status ohne Pfad)")
            commits[-1][1].append((status, _pfad(token[i]), None))
            i += 1

    for aenderung, eintraege in commits:
        if not eintraege:
            continue
        zu_entfernen: list[str] = []
        zu_setzen: dict[str, str] = {}
        for status, pfad, alt in eintraege:
            ziel = abbildung.get(pfad)
            if ziel is None:
                continue  # Datei existiert am Ende nicht (mehr) unter diesem Namen
            erg.setdefault(ziel, []).append(aenderung)
            if status == "R" and alt is not None:
                zu_entfernen.append(pfad)
                zu_setzen[alt] = ziel
            elif status in ("A", "C"):
                zu_entfernen.append(pfad)  # davor war es eine andere Datei
        for pfad in zu_entfernen:
            abbildung.pop(pfad, None)
        abbildung.update(zu_setzen)
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
