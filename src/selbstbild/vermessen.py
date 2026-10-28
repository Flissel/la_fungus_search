"""Vermessungslauf: committeten Baum lesen, parsen (blob- und pfadgecacht), auflösen,
Zeitachse anhängen und den Graphen im lokalen Stand ersetzen.

Speicherschonend: Blobs werden einzeln gelesen und sofort geparst, nie gesammelt; je
Repo ist höchstens ein `BlobLeser` offen. Im Speicher bleiben nur die Parser-Ergebnisse.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from selbstbild.aufloesen import aufloesen
from selbstbild.parser.basis import parsen
from selbstbild.quelle import Baum, BlobLeser, _git, baum_lesen
from selbstbild.speicher import Speicher, pfad_kennung, stand_basis, stand_ordner
from selbstbild.typen import DateiErgebnis, Knoten, id_datei, id_symbol
from selbstbild.zeit import verlauf, zeit_attribute

__all__ = ["Bericht", "vermessen", "stand_finden"]

log = logging.getLogger("selbstbild")


@dataclass(frozen=True)
class Bericht:
    commit: str
    dateien: int
    geparst: int
    aus_cache: int
    uebersprungen: dict[str, int]  # Grund -> Anzahl
    knoten: int
    kanten: int
    kanten_unsicher: int
    fehlerhaft: int
    submodule: tuple[str, ...]
    dauer_s: float
    # Ergänzung zum Plan-Interface: Namen der Submodule mit Grund `submodul_fehlt`
    fehlende_submodule: tuple[str, ...] = ()


def _alle(baum: Baum) -> Iterator[Baum]:
    yield baum
    for unter in baum.unterbaeume:
        yield from _alle(unter)


def _dienste(baum: Baum) -> dict[str, str]:
    """Dienst-Register aus dem Commit (nicht dem Arbeitsbaum): ENV-Name -> Dienstname.

    Format: `dienste: {<name>: {env: [ENV, ...], ...}}`. Fehlt die Datei oder passt die
    Form nicht, gibt es kein Register (Warnung), der Lauf geht weiter.
    """
    pfad = baum.regeln.dienste
    if not pfad:
        return {}
    erg = _git(baum.wurzel, "cat-file", "blob", f"{baum.commit}:{pfad}", check=False)
    if erg.returncode != 0:
        log.warning("%s: Dienst-Register %s fehlt im Commit", baum.repo, pfad)
        return {}
    try:
        daten = yaml.safe_load(erg.stdout.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as fehler:
        log.warning("%s: Dienst-Register %s unlesbar (%s)", baum.repo, pfad, type(fehler).__name__)
        return {}
    dienste = daten.get("dienste") if isinstance(daten, dict) else None
    if not isinstance(dienste, dict):
        return {}
    abbildung: dict[str, str] = {}
    for name in sorted(dienste, key=str):
        eintrag = dienste[name]
        envs = eintrag.get("env") if isinstance(eintrag, dict) else None
        if isinstance(envs, list):
            for env in envs:
                if isinstance(env, str) and env:
                    abbildung.setdefault(env, str(name))
    return abbildung


def _digest(werte: Any) -> str:
    text = json.dumps(werte, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def _mit_attr(k: Knoten, zusatz: Mapping[str, str]) -> Knoten:
    werte = dict(k.attr)
    werte.update(zusatz)
    return dataclasses.replace(k, attr=tuple(sorted(werte.items())))


def _parsen_sicher(sprache: str, pfad: str, quelle: bytes, repo: str) -> tuple[DateiErgebnis, bool]:
    """(Ergebnis, ok). Bei einer Parser-Ausnahme: fehlerhaft nur fuer diesen Lauf, nicht cachen."""
    try:
        return parsen(sprache, pfad, quelle), True
    except Exception as fehler:  # ein Parserfehler darf den Lauf nicht abbrechen
        log.warning("%s:%r: Parser-Ausnahme %s, Datei als fehlerhaft markiert",
                    repo, pfad, type(fehler).__name__)
        return DateiErgebnis(fehlerhaft=True), False


def stand_finden(wurzel: Path) -> Path | None:
    """Vorhandenes `graph.sqlite` dieser Wurzel, ohne etwas anzulegen.

    Gesucht wird ueber die Pfadkennung (`<name>-<kennung>`), also unabhaengig vom aktuellen
    Namen in der Regeldatei: ein Stand, der unter einem frueheren Namen vermessen wurde, wird
    gefunden. Gibt es mehrere, gewinnt der zuletzt geschriebene.
    """
    basis = stand_basis()
    if not basis.is_dir():
        return None
    kandidaten = [o / "graph.sqlite" for o in basis.glob(f"*-{pfad_kennung(wurzel)}")
                  if (o / "graph.sqlite").is_file()]
    if not kandidaten:
        return None
    return max(kandidaten, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def vermessen(
    wurzel: Path,
    commit: str = "HEAD",
    *,
    voll: bool = False,
    speicher: Speicher | None = None,
) -> Bericht:
    """Vermisst `commit` (inkl. initialisierter Submodule); `voll=True` umgeht den Parse-Cache."""
    start = time.monotonic()
    wurzel = Path(wurzel)
    baum = baum_lesen(wurzel, commit)
    baeume = list(_alle(baum))
    log.info("%s @ %s: Baum gelesen, %d Repos, %d Dateien",
             baum.repo, baum.commit[:12], len(baeume), sum(len(b.eintraege) for b in baeume))

    eigen = speicher is None
    s = speicher if speicher is not None else Speicher(stand_ordner(wurzel, baum.repo) / "graph.sqlite")
    try:
        # --- Parsen (Cache je Blob, Sprache und Pfad) ------------------------
        ergebnisse: dict[tuple[str, str], DateiErgebnis] = {}
        geparst = aus_cache = 0
        for b in baeume:
            t0 = time.monotonic()
            offen = []
            for e in b.eintraege:
                assert e.sprache is not None
                erg = None if voll else s.cache_holen(e.blob, e.sprache, e.pfad)
                if erg is None:
                    offen.append(e)
                else:
                    ergebnisse[(e.repo, e.pfad)] = erg
                    aus_cache += 1
            if offen:
                with BlobLeser(b.wurzel) as leser:
                    for e in offen:
                        assert e.sprache is not None
                        erg, ok = _parsen_sicher(e.sprache, e.pfad, leser.lesen(e.blob), e.repo)
                        if ok:
                            s.cache_setzen(e.blob, e.sprache, erg, e.pfad)
                        ergebnisse[(e.repo, e.pfad)] = erg
                        geparst += 1
            log.info("%s: %d Dateien, %d geparst, %d aus Cache (%.1f s)", b.repo,
                     len(b.eintraege), len(offen), len(b.eintraege) - len(offen),
                     time.monotonic() - t0)

        # --- Auflösen -------------------------------------------------------
        dienste: dict[str, str] = {}
        for b in baeume:  # Wurzel zuerst: bei Doppelungen gewinnt das äußere Register
            for env, name in _dienste(b).items():
                dienste.setdefault(env, name)
        regeln = {b.repo: b.regeln for b in baeume}
        t0 = time.monotonic()
        knoten, kanten = aufloesen(ergebnisse, regeln, dienste)
        log.info("aufgelöst: %d Knoten, %d Kanten (%.1f s)", len(knoten), len(kanten),
                 time.monotonic() - t0)

        # --- Zeitachse: Stichtag ist die Committer-Zeit des vermessenen Commits ---
        stichtag = baum.commit_zeit
        zeit_je_datei: dict[str, tuple[tuple[str, str], ...]] = {}
        symbol_datei: dict[str, str] = {}
        seit_je: dict[str, int] = {}
        for b in baeume:
            t0 = time.monotonic()
            aenderungen = verlauf(b.wurzel, b.commit)
            digests: dict[str, str] = {}
            for e in b.eintraege:
                did = id_datei(e.repo, e.pfad)
                zeit_je_datei[did] = zeit_attribute(aenderungen.get(e.pfad, ()), stichtag)
                for sym in ergebnisse[(e.repo, e.pfad)].symbole:
                    sid = id_symbol(e.repo, e.pfad, sym.qualname)
                    digests[sid] = sym.digest
                    symbol_datei[sid] = did
            del aenderungen
            # je (Sub-)Repo mit dessen Commit-Zeit, zeitlich einsortiert (R17)
            s.symbolverlauf_merken(b.commit, b.commit_zeit, digests, bereich=b.repo)
            seit_je.update(s.seit_je_symbol(b.repo, b.commit_zeit))
            log.info("%s: Zeitachse und Symbolverlauf (%.1f s)", b.repo, time.monotonic() - t0)

        fehlerhaft = sum(1 for erg in ergebnisse.values() if erg.fehlerhaft)
        dateien = len(ergebnisse)
        del ergebnisse

        neu: list[Knoten] = []
        for k in knoten:
            if k.art == "datei" and k.id in zeit_je_datei:
                k = _mit_attr(k, dict(zeit_je_datei[k.id]))
            elif k.art == "symbol" and k.id in symbol_datei:
                zusatz = dict(zeit_je_datei.get(symbol_datei[k.id], ()))
                seit = seit_je.get(k.id)
                if seit is not None:  # nur im Speicher, nicht im Export (R17)
                    zusatz["seit"] = str(seit)
                k = _mit_attr(k, zusatz)
            neu.append(k)

        del knoten, seit_je, symbol_datei, zeit_je_datei
        anzahl_knoten, anzahl_kanten = len(neu), len(kanten)
        kanten_unsicher = sum(1 for k in kanten if not k.sicher)
        s.graph_ersetzen(baum.commit, neu, kanten)
        del neu, kanten  # vor dem Export freigeben
        s.meta_setzen("regeln_digest", _digest(
            [[b.repo, b.regeln.model_dump(mode="json")] for b in baeume] + [sorted(dienste.items())]
        ))

        gruende = Counter(u.grund for b in baeume for u in b.uebersprungen)
        bericht = Bericht(
            commit=baum.commit,
            dateien=dateien,
            geparst=geparst,
            aus_cache=aus_cache,
            uebersprungen=dict(sorted(gruende.items())),
            knoten=anzahl_knoten,
            kanten=anzahl_kanten,
            kanten_unsicher=kanten_unsicher,
            fehlerhaft=fehlerhaft,
            submodule=tuple(sorted(b.repo for b in baeume[1:])),
            dauer_s=round(time.monotonic() - start, 3),
            fehlende_submodule=tuple(sorted(
                f"{u.repo}/{u.pfad}" for b in baeume for u in b.uebersprungen
                if u.grund == "submodul_fehlt"
            )),
        )
        # nur im lokalen Stand (nicht im Export): für `status`
        s.meta_setzen("name", baum.repo)
        s.meta_setzen("wurzel", str(wurzel.resolve()))
        s.meta_setzen("commit_zeit", str(baum.commit_zeit))
        s.meta_setzen("gemessen_um", str(int(time.time())))
        s.meta_setzen("bericht", json.dumps(dataclasses.asdict(bericht), sort_keys=True))
        log.info("%s fertig in %.1f s", baum.repo, bericht.dauer_s)
        return bericht
    finally:
        if eigen:
            s.schliessen()
