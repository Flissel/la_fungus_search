"""Datenfluss-Graph: Knoten, Kanten (mit Belegen), Speicher und Diff."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .anker import Beleg

KNOTEN_TYPEN = frozenset({
    "space", "dienst", "modul", "symbol", "route", "tabelle", "sql_funktion",
    "mcp_werkzeug", "mensch", "aussage", "kette", "komponente",
    "ereignis",  # Stufe 4, Task 5: Space-Event aus Registry/EVENT_TO_TOOL
})
KANTEN_ARTEN = frozenset({
    "ruft_auf", "definiert_route", "liest_schreibt", "registriert", "verbindet",
    "nutzt_dienst", "gehoert_zu", "hop", "belegt_durch", "geaendert_durch",
    "ereignis_von",  # Stufe 4, Task 5: ereignis -> space
})
KANTEN_QUELLEN = frozenset({"extraktor", "seed", "llm", "review"})
VERSION = 1


def hop_zustand_neu() -> dict[str, object]:
    """Standard-Zustand eines Hops (drei Stufen: verdrahtet, konfiguriert, gemessen)."""
    return {
        "verdrahtet": "unbestimmt",
        "konfiguriert": "unbestimmt",
        "gemessen": {"wert": "keine_probe", "zeit": None},
        "aus": False,
        "nur_doku": False,
    }


def kante_id(art: str, von: str, nach: str, kette: str | None = None, nr: int | str | None = None) -> str:
    """Deterministische Kanten-ID; Hop-Nummern duerfen Text sein (`3a`) und werden unveraendert uebernommen."""
    basis = f"{art}:{von}->{nach}"
    if art == "hop" and kette is not None:
        basis += f"#{kette}/{nr}"
    return basis


@dataclass
class Knoten:
    id: str
    typ: str
    name: str
    space: str | None = None
    datei: str | None = None
    attribute: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.typ not in KNOTEN_TYPEN:
            raise ValueError(f"unbekannter Knotentyp: {self.typ!r}")

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "typ": self.typ, "name": self.name, "space": self.space,
                "datei": self.datei, "attribute": dict(self.attribute)}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "Knoten":
        attr = d.get("attribute")
        space, datei = d.get("space"), d.get("datei")
        return cls(id=str(d["id"]), typ=str(d["typ"]), name=str(d["name"]),
                   space=str(space) if space is not None else None,
                   datei=str(datei) if datei is not None else None,
                   attribute={str(k): str(v) for k, v in attr.items()} if isinstance(attr, dict) else {})


@dataclass
class Kante:
    id: str
    art: str
    von: str
    nach: str
    quelle: str
    belege: tuple[Beleg, ...] = ()
    attribute: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.art not in KANTEN_ARTEN:
            raise ValueError(f"unbekannte Kantenart: {self.art!r}")
        if self.quelle not in KANTEN_QUELLEN:
            raise ValueError(f"unbekannte Kantenquelle: {self.quelle!r}")
        self.belege = tuple(self.belege)

    def to_dict(self) -> dict[str, object]:
        belege = sorted((b.to_dict() for b in self.belege),
                        key=lambda b: json.dumps(b, sort_keys=True))
        return {"id": self.id, "art": self.art, "von": self.von, "nach": self.nach,
                "quelle": self.quelle, "belege": belege, "attribute": dict(self.attribute)}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "Kante":
        belege = d.get("belege")
        attr = d.get("attribute")
        return cls(id=str(d["id"]), art=str(d["art"]), von=str(d["von"]), nach=str(d["nach"]),
                   quelle=str(d["quelle"]),
                   belege=tuple(Beleg.from_dict(b) for b in belege) if isinstance(belege, list) else (),
                   attribute=dict(attr) if isinstance(attr, dict) else {})


def _ohne_zeit(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _ohne_zeit(v) for k, v in obj.items() if k != "zeit"}
    if isinstance(obj, list):
        return [_ohne_zeit(v) for v in obj]
    return obj


def _kanonisch(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Graph:
    def __init__(self, knoten: dict[str, Knoten] | None = None, kanten: dict[str, Kante] | None = None,
                 commit_aussen: str = "", commit_submodul: str = "") -> None:
        self.knoten: dict[str, Knoten] = knoten if knoten is not None else {}
        self.kanten: dict[str, Kante] = kanten if kanten is not None else {}
        self.commit_aussen = commit_aussen
        self.commit_submodul = commit_submodul

    def knoten_hinzu(self, k: Knoten) -> None:
        self.knoten[k.id] = k  # idempotent, letzter Schreiber gewinnt

    def kante_hinzu(self, k: Kante) -> None:
        self.kanten[k.id] = k  # idempotent, letzter Schreiber gewinnt

    def _inhalt(self) -> dict[str, object]:
        return {
            "knoten": [self.knoten[i].to_dict() for i in sorted(self.knoten)],
            "kanten": [self.kanten[i].to_dict() for i in sorted(self.kanten)],
        }

    def digest(self) -> str:
        """Unabhaengig von Einfuegereihenfolge und Zeitstempel."""
        return hashlib.sha256(_kanonisch(self._inhalt()).encode("utf-8")).hexdigest()

    def digest_stabil(self, *, ohne_dateien: Collection[str] = (),
                      still: Callable[[Mapping[str, object], str | None], bool] | None = None) -> str:
        """Wie `digest`, aber ohne fluechtige Felder (Ruling 45, C.9.5): `Beleg.commit` und jeder Schluessel `zeit`
        (z. B. `zustand.gemessen.zeit` aus der Aenderungszeit der Waechter-Datei). Eine neue Messung oder ein neuer
        HEAD ohne Codeaenderung aendert diesen Digest nicht; der Tick committet nur, wenn er sich aendert.

        `ohne_dateien` (Ruling 48a): Knoten mit `datei` darin und Kanten mit einem Beleg darin zaehlen nicht; so
        erzeugen nicht committete Aenderungen keinen Commit.

        `still(beleg, datei)` (Ruling T4-c3): fuer jeden Beleg einer `belegt_durch`-Kante, deren Ausgangsknoten die
        Doku-Datei `datei` traegt; wahr -> sein `digest` zaehlt nicht (nicht signaltragende Ganzdatei-Anker: ihre
        Nachfuehrung allein ist kein Commit, sie faehrt mit dem naechsten faelligen Commit mit)."""
        inhalt = self._inhalt()
        if ohne_dateien:
            weg = set(ohne_dateien)
            inhalt["knoten"] = [k for k in inhalt["knoten"] if k.get("datei") not in weg]  # type: ignore[union-attr]
            inhalt["kanten"] = [k for k in inhalt["kanten"]  # type: ignore[union-attr]
                                if not any(b["file"] in weg for b in k["belege"])]
        datei_von = ({k["id"]: k.get("datei") for k in inhalt["knoten"]}  # type: ignore[union-attr]
                     if still is not None else {})
        for k in inhalt["kanten"]:  # type: ignore[union-attr]
            doc = datei_von.get(k["von"]) if still is not None and k["art"] == "belegt_durch" else None
            for b in k["belege"]:
                ohne_digest = doc is not None and still is not None and still(b, doc)  # vor dem Entfernen pruefen
                b.pop("commit", None)
                if ohne_digest:
                    b.pop("digest", None)
        return hashlib.sha256(_kanonisch(_ohne_zeit(inhalt)).encode("utf-8")).hexdigest()

    def speichern(self, pfad: Path, erzeugt: str, *, zusatz: Mapping[str, object] | None = None) -> None:
        """`zusatz`: weitere Metadaten (z. B. die Kennung des lokalen Vollgraphs); ueberschreiben keine eigenen
        Schluessel und zaehlen nicht zum Digest."""
        doc = {**dict(zusatz or {}), "version": VERSION, "commit_aussen": self.commit_aussen,
               "commit_submodul": self.commit_submodul, "erzeugt": erzeugt,
               **self._inhalt(), "digest": self.digest()}
        text = json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
        pfad = Path(pfad)
        pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = pfad.with_name(pfad.name + ".tmp")
        tmp.write_bytes(text.encode("utf-8"))  # LF, unabhaengig von der Plattform
        tmp.replace(pfad)

    @classmethod
    def laden(cls, pfad: Path) -> "Graph":
        return cls.aus_text(Path(pfad).read_text(encoding="utf-8"))

    @classmethod
    def aus_text(cls, text: str) -> "Graph":
        """Graph aus dem Inhalt von graph.json (z. B. aus denselben Bytes, die als CAS-Basis dienen)."""
        return cls.aus_doc(json.loads(text))

    @classmethod
    def aus_doc(cls, doc: object) -> "Graph":
        """Graph aus dem geparsten Inhalt von graph.json; kein Objekt -> ValueError."""
        if not isinstance(doc, dict):
            raise ValueError("graph.json ist kein Objekt")
        if doc.get("version") != VERSION:
            raise ValueError(f"unbekannte Graph-Version: {doc.get('version')!r}")
        g = cls({}, {}, str(doc.get("commit_aussen", "")), str(doc.get("commit_submodul", "")))
        for d in doc.get("knoten", []):
            g.knoten_hinzu(Knoten.from_dict(d))
        for d in doc.get("kanten", []):
            g.kante_hinzu(Kante.from_dict(d))
        return g


# ---------- Ruling 49: schlanker Repo-Graph, voller Graph lokal ----------

SCHLANK_KANTEN = frozenset({"hop", "belegt_durch", "verbindet", "ereignis_von"})
SCHLANK_KNOTEN = frozenset({"space", "dienst", "kette", "ereignis"})
# Knotentypen, fuer die der schlanke Repo-Graph die alleinige Wahrheit ist (`mit_schlank`)
_REPO_KNOTEN = SCHLANK_KNOTEN | {"aussage"}


def schlank(voll: Graph) -> Graph:
    """Was die Doku und das Review tragen (Ruling 49), rein und deterministisch: alle `hop`-, `belegt_durch`- und
    `verbindet`-Kanten (mit Zustand und Belegen), alle `space`-, `dienst`- und `kette`-Knoten und jeder Knoten, der
    Endpunkt einer behaltenen Kante ist (Symbole nur dann). Die abgeleiteten Symbol-Daten (`gehoert_zu`, `ruft_auf`,
    Routen, ...) berechnet jeder Lauf neu; sie liegen nur im vollen lokalen Graph."""
    kanten = {i: copy.deepcopy(e) for i, e in voll.kanten.items() if e.art in SCHLANK_KANTEN}
    enden = {n for e in kanten.values() for n in (e.von, e.nach)}
    knoten = {i: copy.deepcopy(k) for i, k in voll.knoten.items() if k.typ in SCHLANK_KNOTEN or i in enden}
    return Graph(knoten, kanten, voll.commit_aussen, voll.commit_submodul)


def mit_schlank(voll: Graph, repo: Graph) -> Graph:
    """Arbeitsgraph aus dem vollen lokalen Graph und dem schlanken Repo-Graph (rein): fuer Hops, Aussagen, Compose,
    Spaces, Dienste und Ketten gilt nur der Repo-Graph (eine andere Maschine kann ihn nachgefuehrt haben), die
    Symbol-Daten kommen aus dem lokalen. Commits aus dem Repo-Graph.

    Der Repo-Graph wird zuerst verschlankt: solange im Repo noch ein voller Graph liegt (Seed `91bcc5f`), duerfen
    seine veralteten Symbol-Daten die frischen lokalen nicht ersetzen."""
    repo = schlank(repo)
    g = Graph({i: copy.deepcopy(k) for i, k in voll.knoten.items() if k.typ not in _REPO_KNOTEN},
              {i: copy.deepcopy(e) for i, e in voll.kanten.items() if e.art not in SCHLANK_KANTEN},
              repo.commit_aussen, repo.commit_submodul)
    for k in repo.knoten.values():
        g.knoten_hinzu(copy.deepcopy(k))
    for e in repo.kanten.values():
        g.kante_hinzu(copy.deepcopy(e))
    return g


@dataclass
class GraphDiff:
    neue_kanten: list[str] = field(default_factory=list)
    weg_kanten: list[str] = field(default_factory=list)
    belege_geaendert: list[tuple[str, Beleg, str]] = field(default_factory=list)


def graph_diff(alt: Graph, neu: Graph) -> GraphDiff:
    d = GraphDiff(
        neue_kanten=sorted(set(neu.kanten) - set(alt.kanten)),
        weg_kanten=sorted(set(alt.kanten) - set(neu.kanten)),
    )
    for eid in sorted(set(alt.kanten) & set(neu.kanten)):
        neu_by_key = {(b.art, b.file, b.anker): b for b in neu.kanten[eid].belege}
        for b in alt.kanten[eid].belege:
            gegenueber = neu_by_key.get((b.art, b.file, b.anker))
            if gegenueber is None:
                d.belege_geaendert.append((eid, b, "verschwunden"))
            elif gegenueber.digest != b.digest:
                d.belege_geaendert.append((eid, b, "geaendert"))
    return d
