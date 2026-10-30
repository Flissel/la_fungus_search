"""CLI-Befehle `seed`, `render` und `graph-schlank` des Doku-Loops sowie die Seed-Deckung fuer `messen` (Task 15).

Alle schreiben ueber den gemeinsamen Schreibvorgang (`schreibvorgang.schreiben_pruefen_committen`): Torwaechter
vorher/nachher, Commit nur der geschriebenen Dateien, Ruecksetzen mit CAS bei jedem Fehler vor dem Commit. Der Stand
liegt lokal (Ruling 44) und wird nie committet.

Ruling 49 (Task 15b): `seed` schreibt den vollen Graph lokal (`stand.vollgraph_schreiben`) und committet
`schlank(...)`. `render`, `status` und `messen` lesen den vollen lokalen Graph mit dem schlanken Repo-Graph als
Wahrheit fuer Hops und Aussagen (`stand.arbeitsgraph`), ohne lokalen Graph den schlanken. `graph-schlank` ersetzt
einen vollen `docs/graph/graph.json` (Seed `91bcc5f`) durch die schlanke Fassung und sichert ihn vorher lokal.

Ruling 50 (Task 15c): `seed` misst im Messbaum (`messbaum.bereitstellen`: Manifeste, Extraktoren, SQL, Hop-Anker,
`belege_auffrischen`, Aussage-Verankerung); KETTEN.md wird aus dem echten Arbeitsbaum gelesen und dort geschrieben.
`render` und `graph-schlank` messen nichts (sie lesen nur Graphen) und brauchen keinen Messbaum.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import aussagen as au
from . import bauen, ketten_seed, korpus, messbaum, render_ketten, umschreiben
from .belegspeicher import aussage_speichern
from .extraktoren import sql as sql_ex
from .graph import Graph, mit_schlank, schlank
from .konfig import Konfig
from .pruefliste import beleg_nicht_messbar, beleg_untracked, pruefliste_ergaenzen
from .schreibvorgang import (
    GRAPH_JSON, SUBMODUL, Aenderung, Drift, GitFehler, Melden, eigene_datei, fremde_aenderungen, graph_text, kopf, kopf_pruefen,
    schreiben_pruefen_committen, submodul_kopf, unsaubere_dateien, versionierte_dateien,
)
from .stand import (
    Stand, arbeitsgraph, lokaler_ordner, schreibvermerk_pfad, sperre, stand_laden, stand_schreiben, temp_aufraeumen,
    vollgraph_kennung, vollgraph_laden, vollgraph_schreiben, vollgraph_verwerfen,
)
from .typen import Pruefpunkt

RENDER_BETREFF = "docs(loop): KETTEN.md – 0 Aussagen nachgeführt, {m} in Prüfliste"
SCHLANK_BETREFF = "docs(loop): graph.json schlank (nur Hops, Aussagen, Dienste)"


class SeedFehler(RuntimeError):
    """Seed kann nicht laufen (z. B. kein Git-HEAD)."""


@dataclass(frozen=True)
class SeedErgebnis:
    hops: int
    verankert: int
    commit: str | None
    uebersprungen: str | None


@dataclass(frozen=True)
class SchlankBericht:
    bytes_vorher: int
    bytes_nachher: int
    commit: str | None
    uebersprungen: str | None


@dataclass(frozen=True)
class RenderBericht:
    bloecke: int
    geaendert: bool
    pruefpunkte: int
    commit: str | None
    uebersprungen: str | None


def _melder(melden: Melden | None) -> Melden:
    return melden if melden is not None else (lambda _zeile: None)


def _lauf(jetzt: datetime, stand: Stand | None) -> tuple[int, str]:
    nr = (stand.lauf_nr if stand else 0) + 1
    return nr, f"{jetzt:%Y-%m-%dT%H:%M:%S}#{nr}"


def seed(konfig: Konfig, *, jetzt: datetime, drift: Drift | None = None,
         manifest_bauer: Callable[..., object] | None = None, melden: Melden | None = None) -> SeedErgebnis:
    """Graph aus KETTEN.md (Task 9) + Bau aller Spaces, Tabellen einbetten, rendern, Aussagen merken, committen
    (`docs(loop): KETTEN.md als Graph-Seed (<n> Hops)`)."""
    melden = _melder(melden)
    drift = drift or umschreiben.drift_fuer(konfig)
    bauer = manifest_bauer or korpus.manifest_bauen
    repo, md = konfig.repo, konfig.ketten_md
    with sperre(lokaler_ordner(konfig)) as frei:
        if not frei:
            melden("Sperre: ein anderer Lauf hält tick.lock – übersprungen (laeuft_schon)")
            return SeedErgebnis(0, 0, None, "laeuft_schon")
        temp_aufraeumen(konfig)
        grund_kopf = kopf_pruefen(repo, konfig.zweig)  # C1 (Final-Fix)
        if grund_kopf is not None:
            melden(f"Kopf: {grund_kopf} – übersprungen, nichts angefasst")
            return SeedErgebnis(0, 0, None, grund_kopf)
        fremd = fremde_aenderungen(konfig)
        if fremd:
            melden(f"Fremde Änderungen: {len(fremd)} Datei(en) – übersprungen, nichts angefasst")
            return SeedErgebnis(0, 0, None, "fremde_aenderungen:" + ",".join(fremd))
        kopf_aussen = kopf(repo)
        if kopf_aussen is None:
            raise SeedFehler("kein Git-HEAD im Repo")
        kopf_sub = submodul_kopf(repo) or ""
        stand_alt = stand_laden(konfig)
        lauf_nr, lauf = _lauf(jetzt, stand_alt)
        try:
            vorher = drift(repo)
        except umschreiben.TorwaechterFehler as fehler:
            melden(f"Torwächter: Fehler ({type(fehler).__name__}) – kein Seed")
            return SeedErgebnis(0, 0, None, "torwaechter:torwaechter_fehler")

        try:  # Ruling 50: gemessen wird nur der committete Stand
            koepfe = messbaum.submodul_koepfe(repo)  # HEAD je Submodul, wie vom Seed gesehen
            mb = messbaum.bereitstellen(konfig)
        except (messbaum.MessbaumFehler, OSError) as fehler:
            grund = getattr(fehler, "grund", type(fehler).__name__)
            melden(f"Messbaum: {grund} – kein Seed")
            return SeedErgebnis(0, 0, None, f"messbaum:{grund}")
        if mb.commits != {"": kopf_aussen, **koepfe} or (kopf_sub and koepfe.get(SUBMODUL) != kopf_sub):
            melden("Messbaum: HEAD hat sich bewegt – kein Seed")
            return SeedErgebnis(0, 0, None, "messbaum:head_bewegt")

        def commit_fuer(datei: str) -> str:
            """Commit des innersten Repos, in dem `datei` liegt (auch verschachtelte Submodule)."""
            treffer = [r for r in koepfe if datei == r or datei.startswith(r + "/")]
            if treffer:
                return koepfe[max(treffer, key=len)]
            return kopf_sub if datei.startswith(SUBMODUL + "/") else kopf_aussen
        mess = mb.konfig
        melden(f"Messbaum: {len(mb.commits)} Worktree(s) auf HEAD, {len(mb.nicht_messbar)} Submodul(e) nicht messbar")

        def nicht_messbar(datei: str) -> str | None:
            return next((s for s in mb.nicht_messbar if datei == s or datei.startswith(s + "/")), None)

        sql_defs, _ = sql_ex.sql_definitionen(mess)
        manifeste = {n: bauer(s, mess, commit_fuer(s.wurzel)) for n, s in mess.spaces.items()}
        # KETTEN.md im Messbaum = committet = Arbeitsbaum (fremde Aenderungen hat die Pruefung oben ausgeschlossen)
        basis, seed_pp = ketten_seed.seed_graph(mess, manifeste, sql_defs, kopf_aussen, kopf_sub)
        graph, bau_pp = bauen.graph_bauen(mess, sorted(mess.spaces), basis, manifeste, sql_defs=sql_defs,
                                          commit_aussen=kopf_aussen, commit_sub=kopf_sub)
        graph = render_ketten.belege_auffrischen(graph, manifeste, mess, sql_defs)
        hops = [k for k in graph.kanten.values() if k.art == "hop"]
        verankert = sum(1 for k in hops if k.belege)
        melden(f"Seed: {len(hops)} Hops, {verankert} verankert; {len(konfig.spaces)} Space(s) gebaut")

        alt = (repo / md).read_bytes()
        res = render_ketten.rendern(render_ketten.einbetten(alt.decode("utf-8"), graph), graph, {}, datei=md)
        melden(f"Render: {len(res.hashes)} Block/Blöcke eingebettet")
        struktur: list[Pruefpunkt] = [*seed_pp, *(p for p in bau_pp if p.art != "waechter_veraltet"),
                                      *res.pruefpunkte]
        # Aussagen aus dem eingebetteten Text: die Tabellenzeilen sind dann Bloecke, keine Aussagen
        try:  # I3: git-Fehler fail-closed
            versioniert, unsauber = versionierte_dateien(repo), unsaubere_dateien(repo)
        except GitFehler as fehler:
            melden(f"Git: {fehler.befehl} gescheitert – kein Seed (git_fehler)")
            return SeedErgebnis(0, 0, None, f"git_fehler:{fehler.befehl}")
        offen: list[Pruefpunkt] = [p for p in bau_pp if p.art == "waechter_veraltet"]  # I8: fuer den Menschen
        for a in au.aussagen_lesen(mb.wurzel / md, mess, manifeste, sql_defs, kopf_aussen, text=res.text):
            struktur.extend(a.pruefpunkte)
            for b in a.belege:
                sub = nicht_messbar(b.file)
                if sub is not None:
                    offen.append(beleg_nicht_messbar(b.file, sub))
                elif b.file not in versioniert and not eigene_datei(konfig, b.file):
                    offen.append(beleg_untracked(b.file))
            # nur versionierte, saubere Dateien und nie die eigenen Dateien des Loops (Item 6, Ruling 48)
            aussage_speichern(graph, a, [dataclasses.replace(b, commit=commit_fuer(b.file)) for b in a.belege
                                         if b.file in versioniert and b.file not in unsauber
                                         and not eigene_datei(konfig, b.file)], "extraktor")
        graph_pfad = repo / GRAPH_JSON
        aenderungen = [Aenderung(GRAPH_JSON, graph_pfad.read_bytes() if graph_pfad.is_file() else None,
                                 graph_text(schlank(graph), jetzt.isoformat()))]  # Ruling 49: nur schlank ins Repo
        if res.text.encode("utf-8") != alt:
            aenderungen.insert(0, Aenderung(md, alt, res.text))
        aus = schreiben_pruefen_committen(
            repo, aenderungen, f"docs(loop): KETTEN.md als Graph-Seed ({len(hops)} Hops)", lauf, vorher=vorher,
            drift=drift, melden=melden, pp_datei=md, zweig=konfig.zweig, vermerk=schreibvermerk_pfad(konfig))
        pruefliste_ergaenzen(konfig, [*offen, *aus.pruefpunkte], struktur, voll=True, lauf=lauf)
        if aus.commit is not None or aus.fremd_commit is not None:
            kennung = _vollgraph_merken(konfig, graph, jetzt, vollgraph_kennung(kopf_aussen, kopf_sub, lauf), melden)
            stand_schreiben(konfig, Stand(kopf_aussen, kopf_sub or None, jetzt.isoformat(), dict(res.hashes), lauf_nr,
                                          {}, [], kennung, koepfe))
        return SeedErgebnis(len(hops), verankert, aus.commit, aus.uebersprungen)


def _vollgraph_merken(konfig: Konfig, graph: Graph, jetzt: datetime, kennung: str, melden: Melden) -> str | None:
    """Vollen Graph lokal schreiben; Rueckgabe die Kennung fuer den Stand, bei einem Schreibfehler None (alte Datei
    wird best effort entfernt; der naechste Tick rekonstruiert)."""
    try:
        vollgraph_schreiben(konfig, graph, jetzt.isoformat(), kennung=kennung)
        return kennung
    except OSError as fehler:
        weg = vollgraph_verwerfen(konfig)
        melden(f"Vollgraph: lokal nicht geschrieben ({type(fehler).__name__}), alte Datei "
               f"{'entfernt' if weg else 'nicht entfernbar, gilt als veraltet'} – nächster Tick baut alle neu")
        return None


def neu_rendern(konfig: Konfig, *, jetzt: datetime, schreiben: bool = False, drift: Drift | None = None,
                melden: Melden | None = None) -> RenderBericht:
    """KETTEN.md aus `docs/graph/graph.json` neu rendern, ohne neu zu bauen. Ohne `schreiben` nur zaehlen; mit
    `schreiben` unter der Sperre ueber den gemeinsamen Schreibvorgang."""
    melden = _melder(melden)
    repo, md = konfig.repo, konfig.ketten_md
    if not (repo / GRAPH_JSON).is_file():
        melden("Render: kein docs/graph/graph.json – übersprungen (kein_graph; erst `seed`)")
        return RenderBericht(0, False, 0, None, "kein_graph")
    graph = arbeitsgraph(konfig, Graph.laden(repo / GRAPH_JSON))
    stand_alt = stand_laden(konfig) or Stand()
    alt = (repo / md).read_bytes()
    res = render_ketten.rendern(alt.decode("utf-8"), graph, stand_alt.render_hashes, datei=md)
    geaendert = res.text.encode("utf-8") != alt
    melden(f"Render: {len(res.hashes)} Block/Blöcke, {'Änderung' if geaendert else 'keine Änderung'}, "
           f"{len(res.pruefpunkte)} Prüfpunkte")
    if not schreiben or not geaendert:
        return RenderBericht(len(res.hashes), geaendert, len(res.pruefpunkte), None, None)
    drift = drift or umschreiben.drift_fuer(konfig)
    with sperre(lokaler_ordner(konfig)) as frei:
        if not frei:
            melden("Sperre: ein anderer Lauf hält tick.lock – übersprungen (laeuft_schon)")
            return RenderBericht(len(res.hashes), True, len(res.pruefpunkte), None, "laeuft_schon")
        temp_aufraeumen(konfig)
        grund_kopf = kopf_pruefen(repo, konfig.zweig)  # C1 (Final-Fix)
        if grund_kopf is not None:
            melden(f"Kopf: {grund_kopf} – übersprungen, nichts angefasst")
            return RenderBericht(len(res.hashes), True, len(res.pruefpunkte), None, grund_kopf)
        fremd = fremde_aenderungen(konfig)
        if fremd:
            melden(f"Fremde Änderungen: {len(fremd)} Datei(en) – übersprungen")
            return RenderBericht(len(res.hashes), True, len(res.pruefpunkte), None,
                                 "fremde_aenderungen:" + ",".join(fremd))
        try:
            vorher = drift(repo)
        except umschreiben.TorwaechterFehler:
            melden("Torwächter: Fehler – übersprungen")
            return RenderBericht(len(res.hashes), True, len(res.pruefpunkte), None, "torwaechter:torwaechter_fehler")
        lauf_nr, lauf = _lauf(jetzt, stand_laden(konfig))
        aus = schreiben_pruefen_committen(repo, [Aenderung(md, alt, res.text)], RENDER_BETREFF.format(m=0), lauf,
                                          vorher=vorher, drift=drift, melden=melden, pp_datei=md, zweig=konfig.zweig,
                                          vermerk=schreibvermerk_pfad(konfig))
        if aus.pruefpunkte:
            pruefliste_ergaenzen(konfig, list(aus.pruefpunkte), [], voll=False, lauf=lauf)
        if aus.commit is not None or aus.fremd_commit is not None:
            stand_schreiben(konfig, dataclasses.replace(stand_alt, render_hashes=dict(res.hashes),
                                                        letzter_tick=jetzt.isoformat(), lauf_nr=lauf_nr))
        return RenderBericht(len(res.hashes), True, len(res.pruefpunkte), aus.commit, aus.uebersprungen)


def graph_schlank(konfig: Konfig, *, jetzt: datetime, drift: Drift | None = None,
                  melden: Melden | None = None) -> SchlankBericht:
    """`docs/graph/graph.json` durch `schlank(...)` ersetzen (Bereinigung des vollen Seed-Graphs, kein History-
    Rewrite). Vorher wird der volle Graph lokal gesichert (vorhandener lokaler Graph + Repo-Graph), damit der naechste
    Tick nicht alles neu bauen muss. Schon schlank -> nichts zu tun."""
    melden = _melder(melden)
    repo = konfig.repo
    pfad = repo / GRAPH_JSON
    if not pfad.is_file():
        melden("graph-schlank: kein docs/graph/graph.json – übersprungen (kein_graph; erst `seed`)")
        return SchlankBericht(0, 0, None, "kein_graph")
    drift = drift or umschreiben.drift_fuer(konfig)
    with sperre(lokaler_ordner(konfig)) as frei:
        if not frei:
            melden("Sperre: ein anderer Lauf hält tick.lock – übersprungen (laeuft_schon)")
            return SchlankBericht(0, 0, None, "laeuft_schon")
        temp_aufraeumen(konfig)
        grund_kopf = kopf_pruefen(repo, konfig.zweig)  # C1 (Final-Fix)
        if grund_kopf is not None:
            melden(f"Kopf: {grund_kopf} – übersprungen, nichts angefasst")
            return SchlankBericht(0, 0, None, grund_kopf)
        fremd = fremde_aenderungen(konfig)
        if fremd:
            melden(f"Fremde Änderungen: {len(fremd)} Datei(en) – übersprungen (fremde_aenderungen)")
            return SchlankBericht(0, 0, None, "fremde_aenderungen:" + ",".join(fremd))
        alt = pfad.read_bytes()  # CAS-Basis
        repo_graph = Graph.aus_text(alt.decode("utf-8"))
        duenn = schlank(repo_graph)
        if duenn.digest() == repo_graph.digest():
            melden(f"graph-schlank: docs/graph/graph.json ist schon schlank ({len(alt)} Bytes) – nichts zu tun")
            return SchlankBericht(len(alt), len(alt), None, "schon_schlank")
        neu_text = graph_text(duenn, jetzt.isoformat())
        try:
            vorher = drift(repo)
        except umschreiben.TorwaechterFehler:
            melden("Torwächter: Fehler – übersprungen")
            return SchlankBericht(len(alt), len(alt), None, "torwaechter:torwaechter_fehler")
        stand_alt = stand_laden(konfig)
        lauf_nr, lauf = _lauf(jetzt, stand_alt)
        # volle Fassung lokal sichern: passender lokaler Vollgraph + Repo-Graph, sonst der volle Repo-Graph selbst
        lokal = vollgraph_laden(konfig, stand_alt.vollgraph_kennung if stand_alt else None)
        voll = repo_graph if lokal is None else mit_schlank(lokal.graph, repo_graph)
        kennung = _vollgraph_merken(konfig, voll, jetzt, vollgraph_kennung(kopf(repo) or "", submodul_kopf(repo), lauf),
                                    melden)
        stand_schreiben(konfig, dataclasses.replace(stand_alt or Stand(), lauf_nr=lauf_nr, vollgraph_kennung=kennung))
        melden(f"graph-schlank: voller Graph lokal {'gesichert' if kennung else 'nicht gesichert'}; "
               f"{len(alt)} -> {len(neu_text.encode('utf-8'))} Bytes")
        aus = schreiben_pruefen_committen(repo, [Aenderung(GRAPH_JSON, alt, neu_text)], SCHLANK_BETREFF, lauf,
                                          vorher=vorher, drift=drift, melden=melden, pp_datei=GRAPH_JSON,
                                          zweig=konfig.zweig, vermerk=schreibvermerk_pfad(konfig))
        if aus.pruefpunkte:
            pruefliste_ergaenzen(konfig, list(aus.pruefpunkte), [], voll=False, lauf=lauf)
        return SchlankBericht(len(alt), len(neu_text.encode("utf-8")), aus.commit, aus.uebersprungen)


def seed_deckung(repo: Path, konfig: Konfig | None = None) -> dict[str, object] | None:
    """Hops mit mindestens einem Beleg (C.9.2) und Anteil je Ankerart (C.9.3) aus graph.json; None ohne Graph.
    Mit `konfig` aus dem vollen lokalen Graph (Ruling 49; die Hops kommen ohnehin aus dem schlanken Repo-Graph)."""
    pfad = Path(repo) / GRAPH_JSON
    if not pfad.is_file():
        return None
    g = Graph.laden(pfad) if konfig is None else arbeitsgraph(konfig, Graph.laden(pfad))
    hops = [k for k in g.kanten.values() if k.art == "hop"]
    arten: dict[str, int] = {}
    for k in hops:
        for b in k.belege:
            arten[b.art] = arten.get(b.art, 0) + 1
    return {"hops": len(hops), "verankert": sum(1 for k in hops if k.belege), "ankerarten": dict(sorted(arten.items()))}
