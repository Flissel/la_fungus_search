"""Auflöser: macht aus den Ergebnissen je Datei den Repo-Graphen (Kanten über Dateigrenzen).

Eingabe sind die blob-gecachten `DateiErgebnis`-Werte je `(repo, pfad)`. Ausgabe sind
deduplizierte, sortierte Knoten und Kanten ohne hängende Enden.
"""
from __future__ import annotations

import dataclasses
import posixpath
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

from selbstbild._glob import glob_regex
from selbstbild.regeln import Regeln
from selbstbild.spezial import repo_binden
from selbstbild.spezial._gemeinsam import beleg
from selbstbild.typen import (
    Beleg, DateiErgebnis, Import, Kante, Knoten, Symbol, id_datei, id_dienst, id_doku_seite,
    id_env, id_repo, id_sql_funktion, id_symbol, id_tabelle,
)

__all__ = ["aufloesen"]

_PY_ENDUNGEN = (".py", ".pyi")
_TS_ENDUNGEN = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
_TS_AUFLOESUNG = (".ts", ".tsx", ".js", ".jsx", ".mjs")
_HOOK_ORDNER = ("/.githooks/", "/.claude/hooks/")
# Knotenarten, die bei Bedarf aus der Id entstehen (geteilte Ressourcen ohne eigene Datei)
_BEI_BEDARF = {"env": "env", "tabelle": "tabelle", "dienst": "dienst", "sql_funktion": "sql_funktion"}


@dataclass
class _Importe:
    """Was die Importe einer Datei auflösen: Ziele und Namensraum für Aufrufe, je mit Sicherheit."""

    ziele: list[tuple[str, bool, int, str]] = field(default_factory=list)  # (pfad, sicher, zeile, anker)
    namen: dict[str, tuple[str, str, bool]] = field(default_factory=dict)  # Name -> (pfad, Präfix, sicher)
    module: dict[str, tuple[str, bool]] = field(default_factory=dict)  # Modul/Spezifizierer -> (pfad, sicher)
    stern: list[tuple[str, bool]] = field(default_factory=list)  # Dateien aus `from x import *`
    extern: set[str] = field(default_factory=set)  # Köpfe nicht auflösbarer Importe
    reihenfolge: list[str] = field(default_factory=list)  # Module, längste zuerst (einmal je Datei)

    def abschluss(self) -> _Importe:
        self.reihenfolge = sorted(self.module, key=lambda m: (-len(m), m))
        return self


def _verbinden(*teile: str) -> str:
    return "/".join(t for t in teile if t)


def _eltern(pfad: str) -> str:
    return posixpath.dirname(pfad)


def _ist_vorfahr(ordner: str, pfad: str) -> bool:
    return ordner == "" or pfad.startswith(ordner + "/")


class _Repo:
    """Dateibestand eines Repos für die Modul- und Pfadauflösung."""

    def __init__(self, pfade: Iterable[str]) -> None:
        self.pfade = set(pfade)
        pakete = {_eltern(p) for p in self.pfade if posixpath.basename(p) == "__init__.py"}
        wurzeln: set[str] = set()
        for paket in pakete:
            oben = paket
            while _eltern(oben) in pakete and _eltern(oben) != oben:
                oben = _eltern(oben)
            if oben:  # Paket auf Repo-Ebene: Wurzel ist das Repo selbst
                wurzeln.add(_eltern(oben))
        # Reihenfolge (nur für Gleichstand): Repo-Root, src/, dann die Ordner über Paketketten
        self.py_wurzeln = ["", "src", *sorted(wurzeln - {"", "src"})]
        # oberster Modulname -> Wurzeln, unter denen er existiert (stdlib/Fremdpakete: keine)
        self._koepfe: dict[str, set[str]] = {}
        for p in self.pfade:
            for w in self.py_wurzeln:
                if not _ist_vorfahr(w, p):
                    continue
                rest = p[len(w) + 1:] if w else p
                kopf, schraeg, _ = rest.partition("/")
                if not schraeg:
                    if not kopf.endswith(_PY_ENDUNGEN):
                        continue
                    kopf = posixpath.splitext(kopf)[0]
                self._koepfe.setdefault(kopf, set()).add(w)

    def wurzeln_fuer(self, modul: str) -> list[str]:
        treffer = self._koepfe.get(modul.split(".")[0], set())
        return [w for w in self.py_wurzeln if w in treffer]

    def py_modul(self, basis: str, modul: str) -> str | None:
        stamm = _verbinden(basis, *modul.split(".")) if modul else basis
        kandidaten = [f"{stamm}.py", _verbinden(stamm, "__init__.py")] if modul \
            else [_verbinden(stamm, "__init__.py")]
        return next((k for k in kandidaten if k in self.pfade), None)

    def ts_modul(self, von: str, spezifizierer: str) -> str | None:
        stamm = posixpath.normpath(posixpath.join(_eltern(von), spezifizierer))
        if stamm == ".." or stamm.startswith("../") or stamm.startswith("/"):
            return None
        kandidaten = [stamm, *(stamm + e for e in _TS_AUFLOESUNG)]
        kopf, endung = posixpath.splitext(stamm)
        if endung in (".js", ".jsx", ".mjs"):  # ESM-Importe nennen die gebaute Endung
            kandidaten += [kopf + ".ts", kopf + ".tsx"]
        kandidaten += [_verbinden(stamm, "index" + e) for e in _TS_AUFLOESUNG]
        return next((k for k in kandidaten if k in self.pfade), None)


def _py_bei(repo: _Repo, basis: str, imp: Import, namen: list[str]) -> tuple[str | None, dict[str, str]] | None:
    """(Moduldatei, Untermodule je Name) unter `basis`, oder None, wenn nichts davon existiert."""
    moddatei = repo.py_modul(basis, imp.modul)
    unter = {n: d for n in namen if (d := repo.py_modul(basis, _punkt(imp.modul, n))) is not None}
    if moddatei is None and not unter:
        return None
    return moddatei, unter


def _py_importe(repo: _Repo, pfad: str, importe: Iterable[Import]) -> _Importe:
    erg = _Importe()
    vorfahren = [_eltern(pfad)]
    while vorfahren[-1]:
        vorfahren.append(_eltern(vorfahren[-1]))
    for imp in importe:
        namen = [n for n in imp.namen if n != "*"]
        wahl: tuple[str, str | None, dict[str, str], bool] | None = None
        if imp.relativ:
            # Der Parser kennt die Punktzahl nicht mehr: Ebene 1 ist sicher, höhere nicht.
            for i, basis in enumerate(vorfahren):
                if (t := _py_bei(repo, basis, imp, namen)) is not None:
                    wahl = (basis, t[0], t[1], i == 0)
                    break
        else:
            treffer = [(w, t) for w in repo.wurzeln_fuer(imp.modul)
                       if (t := _py_bei(repo, w, imp, namen)) is not None]
            nah = [x for x in treffer if _ist_vorfahr(x[0], pfad)]
            if nah:  # die nächste Wurzel über der importierenden Datei gewinnt
                w, t = max(nah, key=lambda x: len(x[0]))
                wahl = (w, t[0], t[1], True)
            elif treffer:  # nur fremde Wurzeln: bei mehreren geraten, also unsicher
                w, t = treffer[0]
                wahl = (w, t[0], t[1], len(treffer) == 1)
        if wahl is None:
            erg.extern.update(namen or [imp.modul.split(".")[0]])
            continue
        basis, moddatei, unter, sicher = wahl
        if moddatei is not None and moddatei != pfad and (imp.modul or not unter):
            erg.ziele.append((moddatei, sicher, imp.zeile, imp.modul or "."))
        for n, d in sorted(unter.items()):
            if d != pfad:
                erg.ziele.append((d, sicher, imp.zeile, _punkt(imp.modul, n)))
        for n in namen:
            if n in unter:
                erg.namen[n] = (unter[n], "", sicher)
            elif moddatei is not None:
                erg.namen[n] = (moddatei, n, sicher)
        if "*" in imp.namen and moddatei is not None:
            erg.stern.append((moddatei, sicher))
        if not imp.namen and not imp.relativ:  # `import a.b` bindet a und a.b
            teile = imp.modul.split(".")
            for i in range(1, len(teile) + 1):
                teil = ".".join(teile[:i])
                d = repo.py_modul(basis, teil)
                if d is not None:
                    erg.module[teil] = (d, sicher)
    return erg.abschluss()


def _punkt(modul: str, name: str) -> str:
    return f"{modul}.{name}" if modul else name


def _ts_importe(repo: _Repo, pfad: str, importe: Iterable[Import]) -> _Importe:
    erg = _Importe()
    for imp in importe:
        namen = [n for n in imp.namen if n != "*"]
        ziel = repo.ts_modul(pfad, imp.modul) if imp.modul.startswith(".") else None
        if ziel is None or ziel == pfad:
            if ziel is None:
                erg.extern.update([*namen, imp.modul])
            continue
        erg.ziele.append((ziel, True, imp.zeile, imp.modul))
        erg.module[imp.modul] = (ziel, True)  # `import * as m` wird im Parser zu `<modul>.f`
        for n in namen:
            # `Import` unterscheidet `import x from` (Default, lokaler Name) nicht von
            # `import { x }`: Auflösung über den Namen ist deshalb nur unsicher.
            erg.namen[n] = (ziel, n, False)
    return erg.abschluss()


def _attr(werte: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(werte.items()))


def _einstieg_attr(arten: set[str], fehlerhaft: bool, grund: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    werte = dict(grund)
    if arten:
        werte["einstieg"] = ",".join(sorted(arten))
    if fehlerhaft:
        werte["fehlerhaft"] = "1"
    return _attr(werte)


def _regel_muster(regeln: Regeln) -> tuple[list[tuple[re.Pattern[str], str | None]], list[re.Pattern[str]]]:
    einstiege: list[tuple[re.Pattern[str], str | None]] = []
    for eintrag in regeln.einstiegspunkte:
        pfad, trenner, symbol = eintrag.partition("::")
        einstiege.append((glob_regex(pfad), symbol if trenner else None))
    return einstiege, [glob_regex(m) for m in regeln.doku_pfade]


class _Graph:
    def __init__(self) -> None:
        self.knoten: dict[str, Knoten] = {}
        self.kanten: dict[tuple[str, str, str], Kante] = {}

    def knoten_neu(self, k: Knoten) -> None:
        self.knoten.setdefault(k.id, k)  # erster nach (repo, pfad) gewinnt

    def kante_neu(self, k: Kante) -> None:
        schluessel = (k.art, k.von, k.nach)
        alt = self.kanten.get(schluessel)
        if alt is None:
            self.kanten[schluessel] = k
        elif k.sicher and not alt.sicher:
            self.kanten[schluessel] = dataclasses.replace(alt, sicher=True)

    def abschliessen(self) -> tuple[list[Knoten], list[Kante]]:
        def haltbar(ende: str) -> bool:
            return ende in self.knoten or ende.partition(":")[0] in _BEI_BEDARF

        kanten: list[Kante] = []
        for k in sorted(self.kanten.values(), key=lambda k: (k.art, k.von, k.nach)):
            if not (haltbar(k.von) and haltbar(k.nach)):
                continue  # hängendes Ende: Kante fällt weg, ohne Waisen-Knoten zu erzeugen
            for ende in (k.von, k.nach):
                if ende not in self.knoten:
                    art, _, name = ende.partition(":")
                    self.knoten[ende] = Knoten(ende, _BEI_BEDARF[art], name, None)
            kanten.append(k)
        return (
            sorted(self.knoten.values(), key=lambda k: k.id),
            sorted(kanten, key=lambda k: (k.art, k.von, k.nach)),
        )


def aufloesen(
    ergebnisse: Mapping[tuple[str, str], DateiErgebnis],
    regeln: Mapping[str, Regeln],
    dienste: Mapping[str, str] | None = None,
) -> tuple[list[Knoten], list[Kante]]:
    """Knoten und Kanten aller Dateien; deterministisch und ohne hängende Kanten."""
    dienste = dienste or {}
    schluessel = sorted(ergebnisse)
    repos: dict[str, _Repo] = {}
    for name in sorted({r for r, _ in schluessel}):
        repos[name] = _Repo(p for r, p in schluessel if r == name)
    muster = {name: _regel_muster(regeln.get(name) or Regeln()) for name in repos}

    symbole: dict[tuple[str, str], dict[str, Symbol]] = {}
    letzte: dict[str, dict[str, list[str]]] = {name: {} for name in repos}  # letztes Segment -> Ids
    sql_tabellen: set[str] = set()
    for repo, pfad in schluessel:
        erg = ergebnisse[(repo, pfad)]
        symbole[(repo, pfad)] = {s.qualname: s for s in erg.symbole}
        for s in erg.symbole:
            letzte[repo].setdefault(s.qualname.rpartition(".")[2], []).append(id_symbol(repo, pfad, s.qualname))
        sql_tabellen.update(id_tabelle(o.name) for o in erg.sql if o.art in ("tabelle", "view"))

    g = _Graph()
    for repo in repos:
        g.knoten_neu(Knoten(id_repo(repo), "repo", repo, None))

    # --- Knoten je Datei ---------------------------------------------------
    for repo, pfad in schluessel:
        erg = ergebnisse[(repo, pfad)]
        eigene = symbole[(repo, pfad)]
        einstiege, doku = muster[repo]
        datei_arten: set[str] = set()
        symbol_arten: dict[str, set[str]] = {}
        for e in erg.einstiege:
            if e.symbol is not None and e.symbol in eigene:
                symbol_arten.setdefault(e.symbol, set()).add(e.art)
            else:
                datei_arten.add(e.art)
        for rx, qual in einstiege:
            if rx.match(pfad):
                if qual is None:
                    datei_arten.add("regel")
                elif qual in eigene:
                    symbol_arten.setdefault(qual, set()).add("regel")
        if any(o in f"/{pfad}" for o in _HOOK_ORDNER):
            datei_arten.add("hook")

        did = id_datei(repo, pfad)
        g.knoten_neu(Knoten(did, "datei", pfad, None, _einstieg_attr(datei_arten, erg.fehlerhaft, {})))
        g.kante_neu(Kante("enthaelt", id_repo(repo), did, True, None))
        for s in erg.symbole:
            sid = id_symbol(repo, pfad, s.qualname)
            attr = _einstieg_attr(symbol_arten.get(s.qualname, set()), erg.fehlerhaft,
                                  {"symbolart": s.art})
            g.knoten_neu(Knoten(sid, "symbol", s.qualname, Beleg(repo, pfad, s.zeile_von, s.qualname, s.digest), attr))
            g.kante_neu(Kante("enthaelt", did, sid, True, None))
        for o in erg.sql:
            b = Beleg(repo, pfad, o.zeile, o.name, o.digest)
            if o.art == "sql_funktion":
                fid = id_sql_funktion(o.name)
                g.knoten_neu(Knoten(fid, "sql_funktion", fid.partition(":")[2], b))
            else:
                tid = id_tabelle(o.name)
                g.knoten_neu(Knoten(tid, "tabelle", tid.partition(":")[2], b,
                                    _attr({"sql_art": "view"}) if o.art == "view" else ()))
        if any(rx.match(pfad) for rx in doku):
            g.knoten_neu(Knoten(id_doku_seite(repo, pfad), "doku_seite", pfad, None))
        for k in erg.extra_knoten:
            g.knoten_neu(cast(Knoten, repo_binden(k, repo)))

    # --- Kanten je Datei ---------------------------------------------------
    for repo, pfad in schluessel:
        erg = ergebnisse[(repo, pfad)]
        eigene = symbole[(repo, pfad)]
        did = id_datei(repo, pfad)

        def quelle(qual: str | None) -> str:
            return id_symbol(repo, pfad, qual) if qual is not None and qual in eigene else did

        if pfad.endswith(_PY_ENDUNGEN):
            imp = _py_importe(repos[repo], pfad, erg.importe)
        elif pfad.endswith(_TS_ENDUNGEN):
            imp = _ts_importe(repos[repo], pfad, erg.importe)
        else:
            imp = _Importe()
        for ziel, sicher, zeile, anker in imp.ziele:
            g.kante_neu(Kante("importiert", did, id_datei(repo, ziel), sicher, beleg(repo, pfad, zeile, anker)))

        for a in erg.aufrufe:
            von = quelle(a.in_symbol)
            treffer = _aufruf_ziel(repo, pfad, a.in_symbol, a.name, eigene, symbole, imp, letzte[repo], von)
            if treffer is not None:
                nach, sicher = treffer
                g.kante_neu(Kante("ruft_auf", von, nach, sicher, beleg(repo, pfad, a.zeile, a.name)))

        for e in erg.env:
            von = quelle(e.in_symbol)
            b = beleg(repo, pfad, e.zeile, e.name)
            g.kante_neu(Kante("liest_env", von, id_env(e.name), True, b))
            if e.name in dienste:
                g.kante_neu(Kante("nutzt_dienst", von, id_dienst(dienste[e.name]), True, b))

        for name, zeile, qual in erg.tabellen_zugriffe:
            tid = id_tabelle(name)
            g.kante_neu(Kante("liest_schreibt", quelle(qual), tid, tid in sql_tabellen,
                              beleg(repo, pfad, zeile, name)))

        for k in erg.extra_kanten:
            g.kante_neu(cast(Kante, repo_binden(k, repo)))

    return g.abschliessen()


def _aufruf_ziel(
    repo: str,
    pfad: str,
    in_symbol: str | None,
    name: str,
    eigene: Mapping[str, Symbol],
    symbole: Mapping[tuple[str, str], Mapping[str, Symbol]],
    imp: _Importe,
    letzte: Mapping[str, list[str]],
    von: str,
) -> tuple[str, bool] | None:
    """Ziel-Id und Sicherheit eines Aufrufs, oder None."""
    def in_datei(ziel: str, qual: str) -> str | None:
        return id_symbol(repo, ziel, qual) if qual in symbole.get((repo, ziel), {}) else None

    kopf, punkt, rest = name.partition(".")
    # 1) self.x / this.x: Methode derselben (innersten umschließenden) Klasse
    if kopf in ("self", "this") and punkt and in_symbol is not None:
        teile = in_symbol.split(".")
        for i in range(len(teile) - 1, 0, -1):
            klasse = ".".join(teile[:i])
            if klasse in eigene and eigene[klasse].art == "klasse":
                if f"{klasse}.{rest}" in eigene:
                    return id_symbol(repo, pfad, f"{klasse}.{rest}"), True
                break
    # 2) dieselbe Datei
    if name in eigene:
        return id_symbol(repo, pfad, name), True
    # 3) importierter Name (ggf. mit Attributkette, z. B. Klasse.methode)
    # Die Sicherheit kommt aus dem Import (R13): unsicher aufgelöst bleibt unsicher.
    if kopf in imp.namen:
        ziel, praefix, sicher = imp.namen[kopf]
        treffer = in_datei(ziel, ".".join(t for t in (praefix, rest) if t))
        return (treffer, sicher) if treffer is not None else None
    # 4) importiertes Modul: `mod.f` bzw. `a.b.f`
    for modul in imp.reihenfolge:
        if name.startswith(modul + "."):
            ziel, sicher = imp.module[modul]
            treffer = in_datei(ziel, name[len(modul) + 1:])
            return (treffer, sicher) if treffer is not None else None
    for ziel, sicher in imp.stern:
        treffer = in_datei(ziel, name)
        if treffer is not None:
            return treffer, sicher
    # 5) unsicher: nur schlichte Namen (ohne Punkt), repo-weit genau einmal, nie aus fremden Paketen
    if punkt or name in imp.extern:
        return None
    kandidaten = letzte.get(name, [])
    if len(kandidaten) == 1 and kandidaten[0] != von:
        return kandidaten[0], False
    return None
