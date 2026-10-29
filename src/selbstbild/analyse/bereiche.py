"""Bereiche: Leiden-Gemeinschaften auf dem Dateigraphen (ungerichtet, gewichtet)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import leidenalg

from .lader import SEED, GraphDaten, dateigraph

ISOLIERT = "(isoliert)"


@dataclass(frozen=True)
class Bereich:
    id: str
    name: str
    dateien: tuple[str, ...]
    kohaesion: float


def _repo_pfad(datei_id: str) -> tuple[str, str]:
    """`datei:<repo>:<pfad>` -> (repo, pfad)."""
    repo, _, pfad = datei_id.removeprefix("datei:").partition(":")
    return repo, pfad


def _gemeinsam(folgen: list[tuple[str, ...]]) -> list[str]:
    erg: list[str] = []
    for stufe in zip(*folgen):
        if len(set(stufe)) != 1:
            break
        erg.append(stufe[0])
    return erg


def _haeufigster(folgen: list[tuple[str, ...]]) -> str | None:
    haeufig = Counter("/".join(f) for f in folgen if f)
    if not haeufig:
        return None
    n = max(haeufig.values())
    return min(o for o, c in haeufig.items() if c == n)


def _name(dateien: tuple[str, ...]) -> str:
    """Name `<repo>/<ordner>`, immer mit Repo-Praefix (Fix-Welle).

    Ein Repo: gemeinsamer Ordner, sonst der haeufigste Ordner (bei Gleichstand der kleinste), ohne
    Ordner das Repo selbst. Mehrere Repos: Repo-Segmente zaehlen wie Ordner (Submodul `v/os/sub` teilt
    `v/os` mit seinem Elternrepo); gemeinsames Praefix, sonst der haeufigste volle Pfad."""
    teile = [_repo_pfad(i) for i in dateien]
    repos = {r for r, _ in teile}
    if len(repos) == 1:
        repo = next(iter(repos))
        ordner = [tuple(p.split("/")[:-1]) for _, p in teile]
        rest = "/".join(_gemeinsam(ordner)) or _haeufigster(ordner)
        return f"{repo}/{rest}" if rest else repo
    folgen = [(*r.split("/"), *p.split("/")[:-1]) for r, p in teile]
    return "/".join(_gemeinsam(folgen)) or _haeufigster(folgen) or "."


def _kohaesionen(label: list[int], anzahl: int, kanten: list[tuple[int, int, float]]) -> list[float]:
    """Kohaesion je Gruppe in einem Durchlauf ueber alle Kanten (innen / (innen + Schnitt))."""
    innen = [0.0] * anzahl
    aussen = [0.0] * anzahl
    for a, b, w in kanten:
        la, lb = label[a], label[b]
        if la == lb:
            innen[la] += w
        else:
            aussen[la] += w
            aussen[lb] += w
    return [0.0 if i + o == 0 else round(i / (i + o), 4) for i, o in zip(innen, aussen)]


def bereiche_finden(d: GraphDaten, aufloesung: float = 1.0) -> tuple[Bereich, ...]:
    """Leiden (RBConfiguration, fester Seed) auf dem ungerichteten Dateigraphen.

    Dateien ohne Kanten werden in EINEM Bereich `(isoliert)` gesammelt. Nummeriert wird
    `bereich:<n>` nach absteigender Groesse, bei Gleichstand nach kleinster Datei-Id;
    `(isoliert)` wird dabei wie jeder andere Bereich einsortiert.
    """
    s = dateigraph(d)
    if not s.ids:
        return ()
    g = s.graph
    kanten = [(e.source, e.target, float(e["weight"])) for e in g.es]
    grad = [0] * len(s.ids)
    for a, b, _ in kanten:
        grad[a] += 1
        grad[b] += 1
    aktiv = [v for v, n in enumerate(grad) if n > 0]
    gruppen: list[tuple[set[int], bool]] = []
    if aktiv:
        sub = g.induced_subgraph(aktiv)  # behaelt die sortierte Knotenreihenfolge
        ung = sub.as_undirected(mode="collapse", combine_edges={"weight": "sum"})
        teil = leidenalg.find_partition(
            ung, leidenalg.RBConfigurationVertexPartition, weights=ung.es["weight"],
            resolution_parameter=aufloesung, seed=SEED, n_iterations=-1)
        for mitglieder in teil:
            gruppen.append(({aktiv[m] for m in mitglieder}, False))
    iso = {v for v, n in enumerate(grad) if n == 0}
    if iso:
        gruppen.append((iso, True))
    label = [0] * len(s.ids)
    for n, (mitglieder, _) in enumerate(gruppen):
        for v in mitglieder:
            label[v] = n
    koh = _kohaesionen(label, len(gruppen), kanten)
    fertig = []
    for n, (mitglieder, ist_iso) in enumerate(gruppen):
        ids = tuple(sorted(s.ids[v] for v in mitglieder))
        fertig.append((ids, ISOLIERT if ist_iso else _name(ids), 0.0 if ist_iso else koh[n]))
    fertig.sort(key=lambda t: (-len(t[0]), t[0][0]))
    return tuple(Bereich(f"bereich:{n}", name, ids, k)
                 for n, (ids, name, k) in enumerate(fertig, 1))
