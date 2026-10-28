"""Compose-Extraktor: Dienste, Ports und Dienst-zu-Dienst-Kanten einer Compose-Datei.

Übernommen aus Vibemind_V1 scripts/doku_loop/extraktoren/compose.py @ 96b05d24; der Doku-Loop behält bis Stufe 4 seine Fassung.

Anpassung: liest Bytes statt Dateisystem, ohne Konfig. Zusaetzlich zu den URL-/Host-Kanten aus `environment:`
entstehen `verbindet`-Kanten aus `depends_on` und `links`. Gespeichert werden nur Dienstnamen, Ports und
Env-**Namen** (als Beleg-Anker); Env-Werte werden nur fluechtig zur Zielerkennung gelesen.
"""
from __future__ import annotations

import fnmatch
import posixpath
import re
from typing import Any

import yaml

from selbstbild.spezial._gemeinsam import beleg
from selbstbild.typen import Kante, Knoten, id_dienst

_VERBOTEN = (".vibemind-stack-rendered-*.yml", "*partial-*.yml")
_VAR_VORGABE = re.compile(r"\$\{[A-Za-z_]\w*:?-([^}]*)\}")
_VAR_OHNE = re.compile(r"\$\{[A-Za-z_]\w*[^}]*\}|\$[A-Za-z_]\w*")
_NACKT_ENDEN = ("_HOST", "_HOSTNAME", "_ADDR", "_ADDRESS", "_SERVER", "_SERVICE", "_ENDPOINT", "_URL", "_URI")
_URL_HOST = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^/@\s]*@)?([^/:\s?#]+)")


def _ersetzen(wert: str) -> str:
    """${VAR:-v} und ${VAR-v} -> v; ${VAR} / $VAR ohne Vorgabe -> ''."""
    wert = _VAR_VORGABE.sub(lambda m: m.group(1), wert)
    return _VAR_OHNE.sub("", wert)


def _port(eintrag: Any) -> int | None:
    if isinstance(eintrag, bool):
        return None
    if isinstance(eintrag, int):
        return eintrag
    if isinstance(eintrag, dict):
        return _port(eintrag.get("published"))
    if not isinstance(eintrag, str):
        return None
    teile = _ersetzen(eintrag).split("/", 1)[0].split(":")
    kandidat = teile[0] if len(teile) <= 2 else teile[-2]  # ip:host:container -> host
    kandidat = kandidat.strip()
    return int(kandidat) if kandidat.isdigit() else None


def _env(roh: Any) -> dict[str, str]:
    ergebnis: dict[str, str] = {}
    if isinstance(roh, dict):
        for k, v in roh.items():
            ergebnis[str(k)] = "" if v is None else _ersetzen(str(v))
    elif isinstance(roh, list):
        for z in roh:
            if not isinstance(z, str):
                continue
            k, _, v = z.partition("=")
            ergebnis[k.strip()] = _ersetzen(v)
    return dict(sorted(ergebnis.items()))


def ziel_service(wert: str, namen: set[str], schluessel: str = "") -> str | None:
    """Servicename, wenn der Wert eine URL mit diesem Host enthaelt oder (nur bei Host/URL-artigem
    Schluessel) nackt der Name ist."""
    wert = wert.strip()
    if wert in namen and schluessel.upper().endswith(_NACKT_ENDEN):
        return wert
    for host in _URL_HOST.findall(wert):
        if host in namen:
            return host
    return None


def _verboten(pfad: str) -> bool:
    name = posixpath.basename(pfad.replace("\\", "/"))
    return any(fnmatch.fnmatch(name, m) for m in _VERBOTEN)


def _namen_liste(roh: Any) -> list[str]:
    """Dienstnamen aus `depends_on` (Liste oder Mapping) bzw. `links` (`dienst[:alias]`)."""
    if isinstance(roh, dict):
        return [str(k) for k in roh]
    if isinstance(roh, list):
        return [str(e).split(":", 1)[0] for e in roh if isinstance(e, str)]
    return []


def _zeile(text: str, name: str) -> int:
    m = re.search(rf"^[ \t]*['\"]?{re.escape(name)}['\"]?[ \t]*:", text, re.M)
    return text.count("\n", 0, m.start()) + 1 if m else 1


def spezial_compose(
    repo: str, pfad: str, quelle: bytes,
) -> tuple[tuple[Knoten, ...], tuple[Kante, ...]]:
    if _verboten(pfad):  # gerenderte bzw. partielle Compose-Dateien sind keine Wahrheit
        return (), ()
    text = quelle.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    try:
        dokumente = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return (), ()
    knoten: dict[str, Knoten] = {}
    kanten: dict[tuple[str, str, str], Kante] = {}
    for roh in dokumente:
        eintraege = roh.get("services") if isinstance(roh, dict) else None
        if not isinstance(eintraege, dict):
            continue
        namen = {str(n) for n in eintraege}
        for name in sorted(namen):
            sv = eintraege[name] if isinstance(eintraege[name], dict) else {}
            zeile = _zeile(text, name)
            ports = sorted({p for e in (sv.get("ports") or []) if (p := _port(e)) is not None})
            attr = (("ports", ",".join(str(p) for p in ports)),) if ports else ()
            did = id_dienst(name)
            knoten.setdefault(did, Knoten(did, "dienst", name, beleg(repo, pfad, zeile, name), attr))

            def verbinde(ziel: str, anker: str, sicher: bool) -> None:
                if ziel == name or ziel not in namen:
                    return
                kanten.setdefault((did, id_dienst(ziel), anker), Kante(
                    "verbindet", did, id_dienst(ziel), sicher, beleg(repo, pfad, zeile, anker)))

            for art in ("depends_on", "links"):
                for ziel in _namen_liste(sv.get(art)):
                    verbinde(ziel, art, True)
            for schluessel, wert in _env(sv.get("environment")).items():
                ziel = ziel_service(wert, namen, schluessel)
                if ziel is not None:
                    # nackter Hostname ist eine Heuristik, eine URL mit Dienst-Host nicht
                    verbinde(ziel, schluessel, ziel != wert.strip())
    return tuple(knoten.values()), tuple(kanten.values())
