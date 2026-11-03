"""Graph aus den Extraktoren bauen und die Seed-Hops bestaetigen.

`graph_bauen(konfig, spaces, basis, manifeste)`:

1. Knoten `space`, `modul`, `symbol` und Kanten `gehoert_zu`, `ruft_auf` aus den Manifesten.
2. `definiert_route` (Routen), `liest_schreibt` (SQL-Bezuege, Beleg `symbol` + `sql`), `registriert`
   (MCP-Werkzeuge), `verbindet` (Compose), `nutzt_dienst` (HTTP-Aufrufe + Dienst-Register; unbekannte
   Env-Namen -> Pruefpunkt `env_unbekannt`).
3. Hop-Bestaetigung je Seed-Hop (`verdrahtet`, `konfiguriert`) ueber die Kanten aus 1./2. Quellseite sind
   die Symbolbelege des Hops. SQL: `liest_schreibt` zu Ziel-Tabelle/-Funktion bzw. `sql`-Beleg. HTTP: nur
   Client-Symbole (nutzt_dienst/env_unbekannt) zaehlen; `nutzt_dienst` zum Zieldienst + Routentreffer im
   Space des Dienstes. MCP: `registriert` zum Werkzeug des Hops oder ab einem Belegsymbol (Ruling 32).
   Python: `ruft_auf` zwischen Belegsymbolen, nur wenn alle im selben Space liegen. `konfiguriert` nur
   fuer HTTP (Ruling 32).
4. `gemessen` aus dem Ketten-Waechter (`waechter.gemessen_laden`).

Knoten-IDs (Ruling 31): `symbol:<space>:<document_id>`, `modul:<repo-relative Datei>`,
`route:<METHODE> <pfad>@<space>`, `tabelle:<schema.name>` / `sql_funktion:<schema.name>` (wie im Seed),
`mcp_werkzeug:<name>`, `dienst:<Registername>`, `space:<name>`.

Inkrementell: nur die Spaces aus `spaces` werden neu gebaut. Space-eigene Knoten (`Knoten.space`) und
Kanten (Attribut `space` = Space der Quelle) anderer Spaces kommen unveraendert aus `basis`. Global und
bei jedem Aufruf neu: `verbindet`-Kanten, Register-Dienstknoten (`herkunft: register`) und gemeinsame
Endknoten der Extraktoren (`herkunft: extraktor`, ohne Space; aus den Kanten abgeleitet). Seed-Hops
(`quelle: seed`) kommen immer aus `basis`; neu gerechnet wird nur ihr `zustand`.

Space-Events (Stufe 4, Task 5): `ereignisse_einbauen` ersetzt nach jedem Bau (auch ohne Neubau) alle `ereignis`-Knoten
und `ereignis_von`-Kanten durch die aus dem Messbaum gelesenen; beide gehoeren zum schlanken Repo-Graph.

Testdateien (`korpus.ist_testdatei`) erzeugen weder Knoten noch Kanten. Ein kaputter Datensatz oder ein
werfender Extraktor wird Pruefpunkt; ein fehlendes Manifest oder eine fehlende Space-Wurzel wirft.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter, defaultdict
from datetime import date
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import korpus, waechter
from .anker import Beleg, repo_relativ
from .dienste import DienstRegister
from .extraktoren import compose as compose_ex
from .extraktoren import http_clients, mcp_werkzeuge, routen
from .extraktoren import sql as sql_ex
from .graph import Graph, Kante, Knoten, hop_zustand_neu, kante_id
from .konfig import Konfig, SpaceKonfig
from .typen import Pruefpunkt

if TYPE_CHECKING:
    from ._manifest_gate2 import Manifest

    from .extraktoren.ereignisse import Ereignis
    from .extraktoren.sql import SqlDefinition

GEMEINSAME_TYPEN = frozenset({"tabelle", "sql_funktion", "mcp_werkzeug", "dienst"})
ABGELEITET = frozenset({"extraktor", "register"})  # Knoten-Attribut `herkunft`; Seed-Knoten haben keins
PRUEF_MECHANISMEN = frozenset({"SQL", "HTTP", "MCP", "Python"})
_NUR_DOKU_DURCH_BAUEN = "nur_doku_durch_bauen"  # Hop-Attribut: nur_doku wurde hier (verdrahtet=nein) gesetzt
_PLATZHALTER = "{}"


class BauFehler(RuntimeError):
    """Harter Fehler: Space unbekannt, Manifest fehlt oder Space-Wurzel (Submodul) fehlt."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normal(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _pfad_passt(literal: str, route: str) -> bool:
    """Gleich viele Segmente; `{}` passt auf ein beliebiges Segment; mindestens ein woertliches Segment gleich."""
    a, b = literal.strip("/").split("/"), route.strip("/").split("/")
    if len(a) != len(b):
        return False
    woertlich = False
    for x, y in zip(a, b):
        if x == _PLATZHALTER or y == _PLATZHALTER:
            continue
        if x != y:
            return False
        woertlich = True
    return woertlich


def _liste(*werte: Iterable[object]) -> list[str]:
    return sorted({str(w) for ws in werte for w in ws if w is not None})


class _Space:
    """Ein neu zu bauender Space: Manifest, eindeutige Dokumente, Testdokumente."""

    def __init__(self, sk: SpaceKonfig, manifest: "Manifest", repo: Path) -> None:
        self.name = sk.name
        self.wurzel = sk.wurzel.strip("/")
        self.manifest = manifest
        self.commit = str(getattr(manifest, "commit_sha", "") or "")
        zaehler = Counter(d.document_id for d in manifest.documents)
        self.mehrdeutig = {i for i, n in zaehler.items() if n > 1}
        self.test_docs = {d.document_id for d in manifest.documents if korpus.ist_testdatei(d.file)}
        self.docs: dict[str, tuple[str, str]] = {}  # document_id -> (repo-relative Datei, sha256(source))
        for d in manifest.documents:
            if d.document_id not in self.mehrdeutig:
                self.docs[d.document_id] = (repo_relativ(manifest, repo, d.file), _sha(d.source))

    def ist_test(self, datei: str) -> bool:
        rel = datei[len(self.wurzel) + 1:] if datei.startswith(self.wurzel + "/") else datei
        return korpus.ist_testdatei(rel)

    def symbol_id(self, document_id: str) -> str:
        return f"symbol:{self.name}:{document_id}"

    def beleg(self, document_id: str | None, zeile: int | None) -> Beleg | None:
        """Symbol-Beleg wie `anker.stand_aus_manifest` (mehrdeutige document_ids bekommen keinen)."""
        if document_id is None or document_id not in self.docs:
            return None
        datei, digest = self.docs[document_id]
        return Beleg("symbol", datei, document_id, digest, self.commit, zeile)


class _Bau:
    def __init__(self, konfig: Konfig, basis: Graph | None, commit_aussen: str, commit_sub: str) -> None:
        self.konfig = konfig
        self.g = Graph({}, {}, commit_aussen, commit_sub)
        self.basis = basis
        self.pp: list[Pruefpunkt] = []
        self.env_gemeldet: set[str] = set()
        self.register = DienstRegister({})
        self.sql_defs: Mapping[str, "SqlDefinition"] = {}

    # ---------- Hilfen ----------

    def pruef(self, art: str, text: str, datei: str | None = None, zeile: int | None = None,
              aussage_id: str | None = None) -> None:
        self.pp.append(Pruefpunkt(art, text, datei, zeile, aussage_id))

    def sicher(self, art: str, wo: str, fn: Callable[[], object], datei: str | None = None,
               zeile: int | None = None) -> object:
        try:
            return fn()
        except Exception as fehler:  # noqa: BLE001 - ein Datensatz/Extraktor darf den Bau nicht abbrechen
            self.pruef(art, f"{wo}: {type(fehler).__name__}: {fehler}", datei, zeile)
            return None

    def kante(self, art: str, von: str, nach: str, space: str | None, belege: Iterable[Beleg | None] = (),
              listen: Mapping[str, Iterable[object]] | None = None) -> Kante:
        """Kante anlegen oder zusammenfuehren (Belege je (art, file, anker) einmal, Listen vereinigt)."""
        eid = kante_id(art, von, nach)
        neu = [b for b in belege if b is not None]
        alt = self.g.kanten.get(eid)
        if alt is None:
            attr: dict[str, object] = {} if space is None else {"space": space}
            for k, v in (listen or {}).items():
                attr[k] = _liste(v)
            alt = Kante(eid, art, von, nach, "extraktor", (), attr)
            self.g.kante_hinzu(alt)
        else:
            for k, v in (listen or {}).items():
                vorher = alt.attribute.get(k)
                alt.attribute[k] = _liste(vorher if isinstance(vorher, list) else (), v)
        schluessel = {(b.art, b.file, b.anker) for b in alt.belege}
        dazu = []
        for b in neu:
            if (b.art, b.file, b.anker) not in schluessel:
                schluessel.add((b.art, b.file, b.anker))
                dazu.append(b)
        if dazu:
            alt.belege = tuple(alt.belege) + tuple(dazu)
        return alt

    def commit_fuer(self, datei: str) -> str:
        return self.g.commit_submodul if datei.startswith("vibemind-os/") else self.g.commit_aussen

    # ---------- Basis ----------

    def aus_basis(self, neu: set[str]) -> None:
        if self.basis is None:
            return
        for k in self.basis.knoten.values():
            if k.space in neu:
                continue
            if k.space is None and k.attribute.get("herkunft") in ABGELEITET:
                continue  # gemeinsame/Register-Knoten werden neu abgeleitet
            kk = copy.deepcopy(k)
            if kk.typ in ("dienst", "komponente"):
                kk.attribute.pop("register", None)
            self.g.knoten_hinzu(kk)
        for e in self.basis.kanten.values():
            if e.art != "hop":
                if e.art == "verbindet" and e.quelle == "extraktor":
                    continue
                if e.attribute.get("space") in neu:
                    continue
            self.g.kante_hinzu(copy.deepcopy(e))

    # ---------- Register und Compose (global) ----------

    def register_laden(self) -> None:
        pfad = self.konfig.repo / self.konfig.dienste_yaml
        geladen = self.sicher("dienste_register_fehler", f"Dienst-Register {self.konfig.dienste_yaml}",
                              lambda: DienstRegister.laden(pfad), self.konfig.dienste_yaml)
        if isinstance(geladen, DienstRegister):
            self.register = geladen

    def register_knoten(self) -> None:
        dienste = self.register.dienste
        for d in dienste.values():
            nid = f"dienst:{d.name}"
            if nid in self.g.knoten:  # Seed-Knoten gleicher Schreibweise: nur verbinden
                self.g.knoten[nid].attribute["register"] = d.name
                continue
            attr = {"herkunft": "register", "register": d.name, "ort": d.ort}
            if d.port is not None:
                attr["port"] = str(d.port)
            if d.space is not None:
                attr["dienst_space"] = d.space
            self.g.knoten_hinzu(Knoten(nid, "dienst", d.name, attribute=attr))
        # Seed-Knoten anderer Schreibweise (Ruling 31; Ruling 32: auch `komponente`): gleicher Name
        # normalisiert oder eindeutiger Port -> Attribut `register`, der Knoten wird nicht umbenannt
        je_name: dict[str, list[str]] = defaultdict(list)
        je_port: dict[str, list[str]] = defaultdict(list)
        for d in dienste.values():
            je_name[_normal(d.name)].append(d.name)
            if d.port is not None:
                je_port[str(d.port)].append(d.name)
        for k in self.g.knoten.values():
            if k.typ not in ("dienst", "komponente") or "herkunft" in k.attribute or "register" in k.attribute:
                continue
            treffer = je_name.get(_normal(k.name), [])
            if len(treffer) != 1:
                treffer = je_port.get(k.attribute.get("port", ""), [])
            if len(treffer) == 1:
                k.attribute["register"] = treffer[0]

    def compose_kanten(self) -> None:
        gelesen = self.sicher("compose_fehler", "Compose", lambda: compose_ex.compose_lesen(self.konfig))
        if not isinstance(gelesen, tuple):
            return
        for ck in gelesen[1]:
            self.sicher("datensatz_fehler", f"Compose {ck}", lambda ck=ck: self.kante(
                "verbindet", f"dienst:{ck.von_service}", f"dienst:{ck.nach_service}", None,
                listen={"env": [ck.env], "dateien": [ck.datei]}), getattr(ck, "datei", None))

    # ---------- Space ----------

    def quelle(self, sp: _Space, symbol: str | None, datei: str) -> str:
        if symbol is not None and sp.symbol_id(symbol) in self.g.knoten:
            return sp.symbol_id(symbol)
        return self.modul(sp, datei)

    def modul(self, sp: _Space, datei: str) -> str:
        mid = f"modul:{datei}"
        if mid not in self.g.knoten:
            self.g.knoten_hinzu(Knoten(mid, "modul", datei, space=sp.name, datei=datei))
            self.kante("gehoert_zu", mid, f"space:{sp.name}", sp.name)
        return mid

    def space_bauen(self, sp: _Space) -> None:
        self.g.knoten_hinzu(Knoten(f"space:{sp.name}", "space", sp.name, space=sp.name, datei=sp.wurzel))
        repo = self.konfig.repo
        for d in sp.manifest.documents:
            if d.document_id in sp.test_docs:
                continue
            self.sicher("datensatz_fehler", f"{sp.name}/symbol {d.document_id}",
                        lambda d=d: self.symbol(sp, d, repo_relativ(sp.manifest, repo, d.file)))
        for aufrufer, ziele in sorted(sp.manifest.callees_by_document.items()):
            if aufrufer in sp.test_docs or sp.symbol_id(aufrufer) not in self.g.knoten:
                continue
            for ziel in sorted(ziele):
                if ziel not in sp.test_docs and ziel != aufrufer and sp.symbol_id(ziel) in self.g.knoten:
                    self.kante("ruft_auf", sp.symbol_id(aufrufer), sp.symbol_id(ziel), sp.name)

        space_konfig = self.konfig.spaces[sp.name]
        self.je_datensatz(sp, "routen", lambda: routen.routen_extrahieren(space_konfig, self.konfig, sp.manifest),
                          self.route)
        schemata = tuple(sorted(set(sql_ex.STANDARD_SCHEMATA) | {n.split(".", 1)[0] for n in self.sql_defs}))
        self.je_datensatz(sp, "sql", lambda: sql_ex.sql_bezuege(space_konfig, self.konfig, sp.manifest, schemata),
                          self.sql_bezug)
        self.je_datensatz(sp, "mcp", lambda: mcp_werkzeuge.mcp_werkzeuge(space_konfig, self.konfig, sp.manifest),
                          self.werkzeug)
        self.je_datensatz(sp, "http", lambda: http_clients.http_aufrufe(space_konfig, self.konfig, sp.manifest),
                          self.http)

    def symbol(self, sp: _Space, d: object, datei: str) -> None:
        document_id = str(getattr(d, "document_id"))
        sid = sp.symbol_id(document_id)
        if sid not in self.g.knoten:
            attr = {"document_id": document_id, "zeile": str(getattr(d, "start_line"))}
            if document_id in sp.mehrdeutig:
                attr["mehrdeutig"] = "ja"
            self.g.knoten_hinzu(Knoten(sid, "symbol", document_id, space=sp.name, datei=datei, attribute=attr))
        self.kante("gehoert_zu", sid, self.modul(sp, datei), sp.name)

    def je_datensatz(self, sp: _Space, art: str, extrahieren: Callable[[], object],
                     verarbeiten: Callable[[_Space, object], None]) -> None:
        datensaetze = self.sicher("extraktor_fehler", f"{sp.name}/{art}",
                                  lambda: list(extrahieren()))  # type: ignore[call-overload]
        if not isinstance(datensaetze, list):
            return
        for r in datensaetze:
            datei, zeile = getattr(r, "datei", None), getattr(r, "zeile", None)
            if isinstance(datei, str) and sp.ist_test(datei):
                continue  # Ruling 18: Testdateien erzeugen keine Kanten
            self.sicher("datensatz_fehler", f"{sp.name}/{art} {datei}:{zeile}",
                        lambda r=r: verarbeiten(sp, r), datei if isinstance(datei, str) else None,
                        zeile if isinstance(zeile, int) else None)

    def route(self, sp: _Space, r: object) -> None:
        r_: routen.Route = r  # type: ignore[assignment]
        pfad = routen.normalisieren(r_.pfad)
        rid = f"route:{r_.methode} {pfad}@{sp.name}"
        if rid not in self.g.knoten:
            self.g.knoten_hinzu(Knoten(rid, "route", f"{r_.methode} {pfad}", space=sp.name, datei=r_.datei,
                                       attribute={"methode": r_.methode, "pfad": pfad, "zeile": str(r_.zeile),
                                                  "generisch": "ja" if r_.generisch else "nein"}))
        elif r_.generisch:
            self.g.knoten[rid].attribute["generisch"] = "ja"
        self.kante("definiert_route", self.quelle(sp, r_.symbol, r_.datei), rid, sp.name,
                   [sp.beleg(r_.symbol, r_.zeile)])

    def sql_ziel(self, name: str) -> str:
        d = self.sql_defs.get(name)
        if d is not None:
            return f"{'sql_funktion' if d.art == 'function' else 'tabelle'}:{name}"
        if f"sql_funktion:{name}" in self.g.knoten and f"tabelle:{name}" not in self.g.knoten:
            return f"sql_funktion:{name}"
        return f"tabelle:{name}"

    def sql_bezug(self, sp: _Space, b: object) -> None:
        b_: sql_ex.SqlBezug = b  # type: ignore[assignment]
        name = b_.name.lower()
        d = self.sql_defs.get(name)
        sql_beleg = Beleg("sql", d.datei, d.name, d.digest, self.commit_fuer(d.datei), d.zeile) if d else None
        self.kante("liest_schreibt", self.quelle(sp, b_.symbol, b_.datei), self.sql_ziel(name), sp.name,
                   [sp.beleg(b_.symbol, b_.zeile), sql_beleg])

    def werkzeug(self, sp: _Space, w: object) -> None:
        w_: mcp_werkzeuge.McpWerkzeug = w  # type: ignore[assignment]
        self.kante("registriert", self.quelle(sp, w_.symbol, w_.datei), f"mcp_werkzeug:{w_.name}", sp.name,
                   [sp.beleg(w_.symbol, w_.zeile)])

    def http(self, sp: _Space, a: object) -> None:
        a_: http_clients.HttpAufruf = a  # type: ignore[assignment]
        quelle = self.quelle(sp, a_.symbol, a_.datei)
        unbekannt: list[str] = []
        for env in list(a_.env_namen):
            dienst = self.register.aufloesen(env)
            if dienst is None:
                unbekannt.append(env)
                if env not in self.env_gemeldet:
                    self.env_gemeldet.add(env)
                    self.pruef("env_unbekannt", f"{env}: Env-Name nicht im Dienst-Register "
                               f"({self.konfig.dienste_yaml})", a_.datei, a_.zeile)
                continue
            self.kante("nutzt_dienst", quelle, f"dienst:{dienst.name}", sp.name, [sp.beleg(a_.symbol, a_.zeile)],
                       listen={"env": [env], "pfade": [a_.pfad_literal]})
        if unbekannt:
            k = self.g.knoten[quelle]
            alt = [e for e in k.attribute.get("env_unbekannt", "").split(",") if e]
            k.attribute["env_unbekannt"] = ",".join(_liste(alt, unbekannt))

    # ---------- Endknoten ----------

    def endknoten_ableiten(self) -> None:
        for e in list(self.g.kanten.values()):
            for nid in (e.von, e.nach):
                if nid in self.g.knoten:
                    continue
                typ, _, name = nid.partition(":")
                if typ in GEMEINSAME_TYPEN and name:
                    self.g.knoten_hinzu(Knoten(nid, typ, name, attribute={"herkunft": "extraktor"}))

    # ---------- Hop-Bestaetigung ----------

    def _space_von(self, datei: str) -> str | None:
        beste: tuple[int, str] | None = None
        for s in self.konfig.spaces.values():
            w = s.wurzel.strip("/")
            if datei == w or datei.startswith(w + "/"):
                if beste is None or len(w) > beste[0]:
                    beste = (len(w), s.name)
        return beste[1] if beste else None

    def quell_symbole(self, hop: Kante) -> list[str]:
        ids: list[str] = []
        for b in hop.belege:
            if b.art != "symbol":
                continue
            space = self._space_von(b.file)
            sid = f"symbol:{space}:{b.anker}" if space else None
            if sid is not None and sid in self.g.knoten and sid not in ids:
                ids.append(sid)
        return ids

    def register_name(self, knoten_id: str) -> str | None:
        k = self.g.knoten.get(knoten_id)
        if k is None or k.typ not in ("dienst", "komponente"):
            return None
        name = k.attribute.get("register") or (
            k.name if k.typ == "dienst" and k.name in self.register.dienste else None)
        return name if name in self.register.dienste else None

    def clients(self, quellen: list[str], aus: Mapping[tuple[str, str], set[str]]) -> list[str]:
        """HTTP-Quellseite: nur Belegsymbole, die wirklich Client sind (nutzt_dienst oder env_unbekannt)."""
        return [q for q in quellen
                if aus.get(("nutzt_dienst", q)) or self.g.knoten[q].attribute.get("env_unbekannt")]

    def ein_space(self, hop: Kante) -> bool:
        """Alle Symbolbelege des Hops haben einen Knoten, und alle liegen im selben Space."""
        spaces: set[str] = set()
        for b in hop.belege:
            if b.art != "symbol":
                continue
            space = self._space_von(b.file)
            if space is None or f"symbol:{space}:{b.anker}" not in self.g.knoten:
                return False
            spaces.add(space)
        return len(spaces) == 1

    def hops_bestaetigen(self) -> None:
        aus: dict[tuple[str, str], set[str]] = defaultdict(set)
        nach_art: dict[str, set[str]] = defaultdict(set)
        nutzt: dict[tuple[str, str], Kante] = {}
        for e in self.g.kanten.values():
            if e.art == "hop":
                continue
            aus[(e.art, e.von)].add(e.nach)
            nach_art[e.art].add(e.nach)
            if e.art == "nutzt_dienst":
                nutzt[(e.von, e.nach)] = e
        routen_je_space: dict[str, list[str]] = defaultdict(list)
        for k in self.g.knoten.values():
            if k.typ == "route" and k.space and k.attribute.get("generisch") == "nein":
                routen_je_space[k.space].append(k.attribute.get("pfad", ""))

        zustandsdatei = self.konfig.repo / self.konfig.waechter_zustandsdatei
        max_alter = self.konfig.waechter_max_alter_min
        messungen, unbekannt = waechter.gemessen_laden(zustandsdatei, self.konfig.waechter_zuordnung,
                                                       max_alter_min=max_alter)
        seit = waechter.veraltet_seit(zustandsdatei, max_alter)
        if seit is not None:  # I8: der Text traegt das Datum -> die Pruefliste fuehrt den Punkt einmal je Tag
            self.pruef("waechter_veraltet", f"{date.today().isoformat()}: Waechter-Zustand seit {seit} nicht "
                       f"geschrieben (Grenze {max_alter} min) – alle Messungen keine_probe",
                       self.konfig.waechter_zustandsdatei)
        for name in unbekannt:
            self.pruef("waechter_unbekannt", f"Waechter-Pruefung ohne Zuordnung: {name}",
                       self.konfig.waechter_zustandsdatei)

        hop_ketten = {k.split("/", 1)[0] for k in messungen if "/" in k}  # Ruling 55b: Ketten mit Hop-Zuordnung
        for eid in sorted(self.g.kanten):
            hop = self.g.kanten[eid]
            if hop.art != "hop" or hop.quelle != "seed":
                continue
            alt = hop.attribute.get("zustand")
            zustand: dict[str, object] = dict(alt) if isinstance(alt, dict) else hop_zustand_neu()
            quellen = self.quell_symbole(hop)
            mech = str(hop.attribute.get("mechanismus_typ", "unbestimmt"))
            verdrahtet = self.verdrahtet(hop, mech, quellen, aus, nach_art, nutzt, routen_je_space)
            zustand["verdrahtet"] = verdrahtet
            zustand["konfiguriert"] = self.konfiguriert(hop, mech, quellen, aus) if mech == "HTTP" else "unbestimmt"
            if verdrahtet == "nein":
                if not zustand.get("nur_doku"):
                    zustand["nur_doku"] = True
                    hop.attribute[_NUR_DOKU_DURCH_BAUEN] = True
            elif hop.attribute.pop(_NUR_DOKU_DURCH_BAUEN, False):
                zustand["nur_doku"] = False
            kette = str(hop.attribute.get("kette", ""))
            messung = messungen.get(f"{kette}/{hop.attribute.get('nr', '')}") or messungen.get(kette)
            if messung is not None:
                zustand["gemessen"] = {"wert": messung.wert, "zeit": messung.zeit}
            elif kette in hop_ketten:  # Hop ohne eigene Probe in einer Kette mit Hop-Zuordnung: keine Messung mehr
                gem = zustand.get("gemessen")
                if isinstance(gem, dict) and gem.get("zeit"):
                    zustand["gemessen"] = {"wert": "keine_probe", "zeit": None}
            hop.attribute["zustand"] = zustand

    def verdrahtet(self, hop: Kante, mech: str, quellen: list[str], aus: Mapping[tuple[str, str], set[str]],
                   nach_art: Mapping[str, set[str]], nutzt: Mapping[tuple[str, str], Kante],
                   routen_je_space: Mapping[str, list[str]]) -> str:
        if mech not in PRUEF_MECHANISMEN or not quellen:
            return "unbestimmt"
        if mech == "SQL":
            ziele = {n for n in (hop.von, hop.nach) if n.split(":", 1)[0] in ("tabelle", "sql_funktion")}
            for b in hop.belege:
                if b.art == "sql":
                    ziele |= {f"tabelle:{b.anker.lower()}", f"sql_funktion:{b.anker.lower()}"}
            if not ziele:
                return "unbestimmt"
            return "ja" if any(aus.get(("liest_schreibt", q), set()) & ziele for q in quellen) else "nein"
        if mech == "HTTP":
            # Quellseite = nur Client-Symbole; Helfer und Handler der Zielseite widerlegen nichts
            clients = self.clients(quellen, aus)
            ziel = self.register_name(hop.nach)
            if ziel is None or not clients:
                return "unbestimmt"  # Ziel nicht im Register (Luecke, keine Widerlegung) bzw. kein Client
            kanten = [nutzt[(q, f"dienst:{ziel}")] for q in clients if (q, f"dienst:{ziel}") in nutzt]
            if not kanten:
                if any(self.g.knoten[q].attribute.get("env_unbekannt") for q in clients):
                    return "unbestimmt"  # I9: Registerluecke -- der unbekannte Env-Name kann das Ziel meinen
                return "nein"  # Clients vorhanden, keiner erreicht das Ziel
            space = self.register.dienste[ziel].space
            routen_ziel = routen_je_space.get(space or "", [])
            for k in kanten:
                pfade = k.attribute.get("pfade")
                for p in pfade if isinstance(pfade, list) else []:
                    if any(_pfad_passt(str(p), r) for r in routen_ziel):
                        return "ja"
            return "unbestimmt"  # nutzt_dienst ohne Routentreffer: nur konfiguriert
        if mech == "MCP":
            # Ruling 32: das Belegsymbol ist selbst die registrierte Werkzeugfunktion (registriert-Kante ab ihm)
            if any(aus.get(("registriert", q)) for q in quellen):
                return "ja"
            namen = [n for n in (hop.nach, hop.von) if n.startswith("mcp_werkzeug:")]
            if not namen:
                return "unbestimmt"
            return "ja" if any(n in nach_art.get("registriert", set()) for n in namen) else "nein"
        # Python: ruft_auf gibt es nur innerhalb eines Manifests
        if len(quellen) < 2 or not self.ein_space(hop):
            return "unbestimmt"
        for q in quellen:
            if aus.get(("ruft_auf", q), set()) & (set(quellen) - {q}):
                return "ja"
        return "nein"

    def konfiguriert(self, hop: Kante, mech: str, quellen: list[str],
                     aus: Mapping[tuple[str, str], set[str]]) -> str:
        """Nur HTTP (Ruling 32): Env-Namen der nutzt_dienst-Kanten der Clients zum Zieldienst plus deren
        env_unbekannt. Unbekannt -> nein; alle im aktuellen Register -> ja; nichts davon -> unbestimmt."""
        if mech != "HTTP":
            return "unbestimmt"
        ziel = self.register_name(hop.nach)
        envs: list[str] = []
        unbekannt = False
        for q in self.clients(quellen, aus):
            if self.g.knoten[q].attribute.get("env_unbekannt"):
                unbekannt = True
            if ziel is not None and f"dienst:{ziel}" in aus.get(("nutzt_dienst", q), set()):
                env = self.g.kanten[kante_id("nutzt_dienst", q, f"dienst:{ziel}")].attribute.get("env")
                if isinstance(env, list):
                    envs.extend(str(e) for e in env)
        if unbekannt or self.register.unbekannte(envs):
            return "nein"
        return "ja" if envs else "unbestimmt"


def ereignisse_einbauen(graph: Graph, ereignisse: Sequence["Ereignis"]) -> None:
    """Stufe 4, Task 5 (D8): alle `ereignis`-Knoten und `ereignis_von`-Kanten durch die aktuell gelesenen ersetzen.

    Je (Space-Schluessel, Event-Name) ein Knoten `ereignis:<space>:<name>` (Attribute `werkzeug` und `quellen`, bei
    Abweichung mit ` | ` verbunden) und eine Kante `ereignis_von` -> `space:<space>` mit den Belegen aller Quellen.
    Global und bei jedem Lauf neu (wie die Compose-Kanten): die Quellen liegen meist ausserhalb der Space-Wurzeln.
    Kein Zeitstempel, kein Commit (Belege tragen `commit=""`), sortiert: `digest_stabil` bleibt deterministisch.
    Die Endknoten `space:<space>` muessen nicht existieren (wie `modul:` bei `belegt_durch`)."""
    from .namen import space_schluessel

    for eid in [e.id for e in graph.kanten.values() if e.art == "ereignis_von"]:
        del graph.kanten[eid]
    for nid in [k.id for k in graph.knoten.values() if k.typ == "ereignis"]:
        del graph.knoten[nid]
    je: dict[tuple[str, str], list["Ereignis"]] = defaultdict(list)
    for e in ereignisse:
        je[(space_schluessel(e.space), e.name)].append(e)
    for (space, name), evs in sorted(je.items()):
        quellen = [(q, b) for e in evs for q, b in e.quellen()]
        belege = {(b.art, b.file, b.anker): b for _, b in sorted(quellen, key=lambda qb: (qb[1].file, qb[1].anker))}
        nid = f"ereignis:{space}:{name}"
        graph.knoten_hinzu(Knoten(nid, "ereignis", name, space=space, datei=evs[0].beleg.file, attribute={
            "werkzeug": " | ".join(sorted({e.werkzeug for e in evs})),
            "quellen": " | ".join(sorted({q for q, _ in quellen}))}))
        graph.kante_hinzu(Kante(kante_id("ereignis_von", nid, f"space:{space}"), "ereignis_von", nid,
                                f"space:{space}", "extraktor", tuple(belege.values())))


def graph_bauen(
    konfig: Konfig,
    spaces: Sequence[str],
    basis: Graph | None,
    manifeste: Mapping[str, "Manifest"],
    *,
    sql_defs: Mapping[str, "SqlDefinition"] | None = None,
    commit_aussen: str | None = None,
    commit_sub: str | None = None,
) -> tuple[Graph, list[Pruefpunkt]]:
    """Graph aus den Extraktoren der `spaces` auf `basis` bauen; Seed-Hops bestaetigen.

    `sql_defs` (sonst `sql.sql_definitionen(konfig)`) und die Commits lassen sich vorgeben; ohne Angabe
    kommen die Commits aus den Manifesten der neu gebauten Spaces (Submodul: Spaces unter `vibemind-os/`,
    sonst irgendeiner; aussen: Spaces ausserhalb), nie aus `basis`. Wirft `BauFehler`, wenn ein Space unbekannt ist, sein
    Manifest fehlt oder seine Wurzel nicht existiert.
    """
    neu = sorted(set(spaces))
    for name in neu:
        if name not in konfig.spaces:
            raise BauFehler(f"Space nicht konfiguriert: {name}")
        if name not in manifeste:
            raise BauFehler(f"Manifest fehlt fuer Space {name}")
        if not (konfig.repo / konfig.spaces[name].wurzel).is_dir():
            raise BauFehler(f"Space-Wurzel fehlt (Submodul?): {konfig.spaces[name].wurzel}")
    # Ruling 32: Commits aus den Argumenten, sonst aus den frisch gebauten Manifesten, nie aus `basis`
    def _commit(im_submodul: bool | None) -> str:
        for n in neu:
            if im_submodul is None or konfig.spaces[n].wurzel.startswith("vibemind-os/") == im_submodul:
                c = str(getattr(manifeste[n], "commit_sha", "") or "")
                if c:
                    return c
        return ""

    sub = commit_sub if commit_sub is not None else (_commit(True) or _commit(None))
    aussen = commit_aussen if commit_aussen is not None else _commit(False)

    bau = _Bau(konfig, basis, aussen, sub)
    if sql_defs is None:
        geladen = bau.sicher("sql_fehler", "SQL-Definitionen", lambda: sql_ex.sql_definitionen(konfig))
        sql_defs = geladen[0] if isinstance(geladen, tuple) else {}
    bau.sql_defs = sql_defs
    bau.aus_basis(set(neu))
    bau.register_laden()
    bau.register_knoten()
    for name in neu:
        bau.space_bauen(_Space(konfig.spaces[name], manifeste[name], konfig.repo))
    bau.compose_kanten()
    bau.endknoten_ableiten()
    bau.hops_bestaetigen()
    return bau.g, bau.pp
