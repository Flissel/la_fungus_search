"""Space-Events aus dem Code lesen und mit der Doku abgleichen (Stufe 4, Task 5, D8). Deterministisch, kein LLM.

Quellen (`doku.ereignisse`, Liste, gelesen unter `wurzel` = Messbaum, also der committete Stand, Ruling T5-a):

- `{yaml: <repo-rel>, pfad: "spaces.*.events", werkzeug: "tool"}`: `*` ist die Space-ID, jeder Schluessel darunter
  ein Event-Name, `werkzeug` der Feldname des Werkzeugs. Beleg `yaml_key` (`spaces.<id>.events["<name>"]`, Digest wie
  `anker.yaml_knoten_digest`, keine Zeile). Die Geschwister-Liste `prefixes` (gleiche Ebene wie `events`)
  liefert bekannte Event-Praefixe (`video.` -> `video`).
- `{python_dict: "EVENT_TO_TOOL", wurzel: "vibemind-os/spaces"}`: Dict-Literale dieses Namens, String auf String,
  auf Modulebene oder direkt im Rumpf einer Klasse auf Modulebene (die echten Tabellen sind Klassenattribute wie
  `VideoBackendAgent.EVENT_TO_TOOL`). Der Space ist der erste Pfadteil unter `wurzel`. Testdateien und
  `ausschluss_namen`/`ausschluss_pfade` zaehlen nicht. Beleg `symbol` (`<modulpfad>.[<Klasse>.]<Name>`, Digest des
  Eintragswerts, keine Zeile). Nicht lesbare Eintraege (kein Zeichenketten-Literal, `**x`) werden je Tabelle gezaehlt
  (`ereignis_eintrag_unlesbar`).

Belege tragen weder Commit (`commit=""`) noch Zeile (`zeile=None`, Fix-Runde 1): Ereignis-Knoten gehen in den schlanken
Graph; eine Zeilenverschiebung in Registry oder Agent aendert weder Block noch `digest_stabil`.

`Ereignis.quelle` ist die konfigurierte Quelle (Dateiname der Registry bzw. der Tabellenname `EVENT_TO_TOOL`): zwei
Tabellen derselben Quelle in einem Space werden vereinigt, nicht gegeneinander verglichen (Fix-Runde 1, Minor 5).

Gleiche Events (Space-Schluessel, Name, Werkzeug) aus mehreren Quellen werden ein `Ereignis`; die weiteren Quellen
stehen in `weitere`. Texte von Pruefpunkten tragen nur Namen, nie Dateiinhalte.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .. import anker, korpus
from ..anker import Beleg
from ..konfig import Konfig
from ..namen import space_schluessel
from ..typen import Pruefpunkt

SPACE_DOKU = re.compile(r"^docs/spaces/\d+_(?P<name>[^/]+)\.md$", re.IGNORECASE)
# Endungen, die ein Token `<praefix>.<name>` als Dateinamen ausweisen (`desktop.json`, `research.yml`)
DATEI_ENDUNGEN = frozenset({
    "json", "yml", "yaml", "py", "pyi", "md", "ts", "tsx", "js", "jsx", "mjs", "cjs", "toml", "txt", "sh", "ps1",
    "bat", "cmd", "html", "css", "sql", "env", "lock", "log", "csv", "ini", "cfg", "rs", "go", "xml", "db", "sqlite",
    "sqlite3",
})
# Punktkette `<praefix>.<teil>[.<teil>...]` (Fix-Runde 1: auch dreiteilige Namen wie `desktop.task.create`): nicht
# Teil eines Pfads (`/`), eines Wortes oder eines Bezeichners mit Vorsatz (`supabase:idea.move` ist eine Operation)
_TOKEN = re.compile(r"(?<![\w./\-:])[a-z][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+(?![\w/\-]|\.\w)")
_BLOCK_BEGINN = "<!-- graph:begin ereignisse "
_BLOCK_ENDE = "<!-- graph:end -->"


@dataclass(frozen=True)
class Ereignis:
    space: str  # wie in der Quelle geschrieben; verglichen wird ueber `namen.space_schluessel`
    name: str
    werkzeug: str
    quelle: str  # konfigurierte Quelle: Dateiname der Registry bzw. Tabellenname (`EVENT_TO_TOOL`)
    beleg: Beleg
    weitere: tuple[tuple[str, Beleg], ...] = ()  # dasselbe Event (gleiches Werkzeug) in weiteren Quellen

    def quellen(self) -> list[tuple[str, Beleg]]:
        return [(self.quelle, self.beleg), *self.weitere]


def _pp(liste: list[Pruefpunkt] | None, art: str, text: str, datei: str | None) -> None:
    if liste is not None:
        liste.append(Pruefpunkt(art, text, datei, None, datei))


# ---------- YAML-Quelle ----------

def _abbilden(knoten: Any, schluessel: str) -> Any:
    if isinstance(knoten, yaml.MappingNode):
        return next((v for k, v in knoten.value if isinstance(k, yaml.ScalarNode) and k.value == schluessel), None)
    return None


def _skalar(knoten: Any) -> str | None:
    return str(knoten.value) if isinstance(knoten, yaml.ScalarNode) and knoten.value is not None else None


def _yaml_lesen(q: Mapping[str, Any], wurzel: Path, praefixe: set[str] | None,
                pp: list[Pruefpunkt] | None) -> list[Ereignis]:
    rel = str(q["yaml"]).strip("/")
    teile = str(q.get("pfad", "spaces.*.events")).split(".")
    feld = str(q.get("werkzeug", "tool"))
    stern = teile.index("*")
    try:
        text = (wurzel / rel).read_bytes().decode("utf-8")
        wurzel_knoten = yaml.compose(text, Loader=yaml.SafeLoader)
        daten = yaml.safe_load(text)
    except FileNotFoundError:
        _pp(pp, "ereignis_quelle_fehlt", f"{rel}: Ereignis-Quelle fehlt im committeten Stand", rel)
        return []
    except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError) as fehler:
        _pp(pp, "ereignis_quelle_unlesbar", f"{rel}: Ereignis-Quelle nicht lesbar ({type(fehler).__name__})", rel)
        return []
    knoten, wert = wurzel_knoten, daten
    for t in teile[:stern]:
        knoten = _abbilden(knoten, t)
        wert = wert.get(t) if isinstance(wert, dict) else None
    if not isinstance(knoten, yaml.MappingNode) or not isinstance(wert, dict):
        return []
    quelle = Path(rel).name
    aus: list[Ereignis] = []
    for k_space, v_space in knoten.value:
        space = _skalar(k_space)
        if space is None:
            continue
        sk, sw = v_space, wert.get(space)
        for t in teile[stern + 1:-1]:
            sk = _abbilden(sk, t)
            sw = sw.get(t) if isinstance(sw, dict) else None
        if praefixe is not None and stern + 1 < len(teile):
            for p in (sw.get("prefixes") if isinstance(sw, dict) else None) or ():
                if isinstance(p, str) and p.strip(" ."):
                    praefixe.add(p.strip().rstrip("."))
        if stern + 1 < len(teile):
            sk = _abbilden(sk, teile[-1])
            sw = sw.get(teile[-1]) if isinstance(sw, dict) else None
        if not isinstance(sk, yaml.MappingNode) or not isinstance(sw, dict):
            continue
        for k_ev, v_ev in sk.value:
            name = _skalar(k_ev)
            if name is None or name not in sw:
                continue
            roh = sw[name]
            werkzeug = roh.get(feld) if isinstance(roh, dict) else None
            pfad = anker.pfad_text([*teile[:stern], space, *teile[stern + 1:], name])
            beleg = Beleg("yaml_key", rel, pfad, anker._digest_json(roh), "", None)
            aus.append(Ereignis(space, name, str(werkzeug) if isinstance(werkzeug, (str, int)) else "", quelle,
                                beleg))
    return aus


# ---------- Python-Quelle ----------

def _ziel_name(knoten: ast.stmt) -> tuple[str, ast.expr] | None:
    if isinstance(knoten, ast.Assign) and len(knoten.targets) == 1 and isinstance(knoten.targets[0], ast.Name):
        return knoten.targets[0].id, knoten.value
    if isinstance(knoten, ast.AnnAssign) and isinstance(knoten.target, ast.Name) and knoten.value is not None:
        return knoten.target.id, knoten.value
    return None


def _dicts(baum: ast.Module, name: str) -> list[tuple[str, ast.stmt, ast.Dict]]:
    """(Qualname-Rest, Zuweisung, Dict) auf Modulebene und im Rumpf von Klassen auf Modulebene."""
    aus: list[tuple[str, ast.stmt, ast.Dict]] = []

    def sammeln(rumpf: Sequence[ast.stmt], praefix: str) -> None:
        for knoten in rumpf:
            ziel = _ziel_name(knoten)
            if ziel is not None and ziel[0] == name and isinstance(ziel[1], ast.Dict):
                aus.append((praefix + name, knoten, ziel[1]))

    sammeln(baum.body, "")
    for knoten in baum.body:
        if isinstance(knoten, ast.ClassDef):
            sammeln(knoten.body, f"{knoten.name}.")
    return aus


def _python_lesen(q: Mapping[str, Any], konfig: Konfig, wurzel: Path,
                  pp: list[Pruefpunkt] | None) -> list[Ereignis]:
    name = str(q["python_dict"])
    basis = str(q["wurzel"]).strip("/")
    ordner = wurzel / basis
    if not ordner.is_dir():
        _pp(pp, "ereignis_quelle_fehlt", f"{basis}: Ordner der Ereignis-Quelle fehlt im committeten Stand", basis)
        return []
    suche = name.encode("utf-8")
    aus: list[Ereignis] = []
    for datei in korpus.dateien(ordner, konfig, (".py",)):
        unter = datei.relative_to(ordner).as_posix()
        if "/" not in unter or korpus.ist_testdatei(unter):
            continue
        rel = f"{basis}/{unter}"
        try:
            roh = datei.read_bytes()
            if suche not in roh:
                continue
            text = roh.decode("utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
                baum = ast.parse(text)
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError, RecursionError) as fehler:
            _pp(pp, "ereignis_quelle_unlesbar", f"{rel}: nicht lesbar ({type(fehler).__name__})", rel)
            continue
        space = unter.split("/", 1)[0]
        for qual, _, d in _dicts(baum, name):
            symbol = f"{anker.modulpfad(rel)}.{qual}"
            unlesbar = 0
            for k, v in zip(d.keys, d.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)
                        and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    unlesbar += 1  # `**basis` (Schluessel None), Aufrufe, Namen, Zahlen
                    continue
                beleg = Beleg("symbol", rel, symbol, anker._digest_json(v.value), "", None)
                aus.append(Ereignis(space, k.value, v.value, name, beleg))
            if unlesbar and pp is not None:
                pp.append(Pruefpunkt("ereignis_eintrag_unlesbar", f"{rel}: {unlesbar} Eintrag/Einträge in {qual} sind "
                                     "keine Zeichenketten-Literale und fehlen in den Events", rel, None, f"{rel}#{qual}"))
    return aus


# ---------- oeffentlich ----------

def ereignisse_lesen(konfig: Konfig, wurzel: Path, *, praefixe: set[str] | None = None,
                     pruefpunkte: list[Pruefpunkt] | None = None) -> list[Ereignis]:
    """Alle Events der Quellen in `konfig.ereignisse`, gelesen unter `wurzel` (Messbaum), sortiert nach Space-Schluessel,
    Name und Werkzeug. `praefixe` (optional) bekommt die bekannten Praefixe aus den Quellen (`prefixes` der Registry),
    `pruefpunkte` fehlende oder unlesbare Quellen (nur Pfad und Fehlertyp)."""
    wurzel = Path(wurzel)
    roh: list[Ereignis] = []
    for q in konfig.ereignisse:
        if "yaml" in q:
            roh.extend(_yaml_lesen(q, wurzel, praefixe, pruefpunkte))
        elif "python_dict" in q:
            roh.extend(_python_lesen(q, konfig, wurzel, pruefpunkte))
    gemischt: dict[tuple[str, str, str], Ereignis] = {}
    for e in roh:
        schl = (space_schluessel(e.space), e.name, e.werkzeug)
        alt = gemischt.get(schl)
        if alt is None:
            gemischt[schl] = e
        elif (e.quelle, e.beleg.file, e.beleg.anker) not in {(q, b.file, b.anker) for q, b in alt.quellen()}:
            gemischt[schl] = Ereignis(alt.space, alt.name, alt.werkzeug, alt.quelle, alt.beleg,
                                      (*alt.weitere, (e.quelle, e.beleg)))
    if praefixe is not None:
        praefixe.update(e.name.split(".", 1)[0] for e in gemischt.values() if "." in e.name)
    return [gemischt[s] for s in sorted(gemischt)]


def space_dokumente_zuordnen(spaces: Iterable[str], explizit: Mapping[str, Any], dateien: Iterable[str], *,
                             pruefpunkte: list[Pruefpunkt] | None = None) -> dict[str, str]:
    """Space-Schluessel -> Doku-Datei: zuerst `doku.space_dokumente`, sonst `docs/spaces/<NNN>_<name>.md` mit gleichem
    normalisiertem Namen (bei mehreren Dateien die erste nach Namen). Ohne Treffer fehlt der Space.

    `dateien` sind die vorhandenen Doku-Ziele. Ein explizites Ziel, das nicht darunter ist (fehlt oder liegt ausserhalb
    der `doku_ziele`), gibt `space_dokument_fehlt` (Fix-Runde 1); die Zuordnung bleibt, damit nicht zusaetzlich
    `space_ohne_doku` entsteht."""
    expl = {space_schluessel(str(k)): str(v) for k, v in explizit.items() if isinstance(v, str) and v.strip()}
    dateien = sorted(dateien)
    vorhanden = {d.casefold() for d in dateien}
    for s, ziel in sorted(expl.items()):
        if ziel.strip("/").casefold() not in vorhanden and pruefpunkte is not None:
            pruefpunkte.append(Pruefpunkt("space_dokument_fehlt", f"{s}: doku.space_dokumente nennt {ziel}, das ist "
                                          "keine vorhandene Doku-Datei der doku_ziele", ziel, None, f"space:{s}"))
    je_name: dict[str, str] = {}
    for d in dateien:
        m = SPACE_DOKU.match(d)
        if m:
            je_name.setdefault(space_schluessel(m.group("name")), d)
    aus: dict[str, str] = {}
    for s in sorted({space_schluessel(s) for s in spaces}):
        ziel = expl.get(s) or je_name.get(s)
        if ziel:
            aus[s] = ziel
    return aus


def _ohne_eigene_bloecke(text: str) -> list[str]:
    """Zeilen des Texts; der Inhalt eigener Events-Bloecke ist leer (Fix-Runde 1): ein Event, das aus der Quelle
    verschwunden ist, aber noch im alten generierten Block steht, ist kein `ereignis_nur_doku`."""
    aus: list[str] = []
    drin = False
    for zeile in text.replace("\r\n", "\n").split("\n"):
        klar = zeile.strip().lstrip("\ufeff")
        if klar.startswith(_BLOCK_BEGINN):
            drin = True
        aus.append("" if drin else zeile)
        if drin and klar == _BLOCK_ENDE:
            drin = False
    return aus


def _namen(quellen: Iterable[tuple[str, Beleg]]) -> str:
    return ", ".join(sorted({q for q, _ in quellen}))


def abgleich(ereignisse: Sequence[Ereignis], doku_texte: Mapping[str, str], space_dokumente: Mapping[str, str], *,
             praefixe: Collection[str] = (), nicht_ereignisse: Collection[str] = ()) -> list[Pruefpunkt]:
    """Pruefpunkte (nur Namen im Text):

    - `ereignis_quellen_abweichend`: in einem Space mit mehr als einer konfigurierten Quelle hat ein Name verschiedene Werkzeuge
      oder fehlt in einer Quelle;
    - `ereignis_nur_doku`: Punktkette `<praefix>.<teil>[.<teil>...]` in einer zugeordneten Doku (`doku_texte`, ohne
      den Inhalt eigener Events-Bloecke), die als Ganzes in keiner Quelle steht, deren erster Teil ein bekannter
      Praefix ist (aus den Event-Namen und `praefixe`), deren letzter Teil keine Dateiendung ist und die nicht in
      `nicht_ereignisse` steht (z. B. bekannte SQL-Objekte `schema.name`: `marketing.inhalte` ist eine Tabelle);
    - `space_ohne_doku`: Space mit Events, aber ohne Doku-Datei in `space_dokumente` (Schluessel normalisiert)."""
    pp: list[Pruefpunkt] = []
    zuordnung = {space_schluessel(k): v for k, v in space_dokumente.items()}
    je_space: dict[str, list[Ereignis]] = defaultdict(list)
    for e in ereignisse:
        je_space[space_schluessel(e.space)].append(e)
    for key in sorted(je_space):
        evs = je_space[key]
        anzeige = evs[0].space
        alle = {q for e in evs for q, _ in e.quellen()}  # je konfigurierter Quelle, nicht je Datei
        if len(alle) > 1:
            je_name: dict[str, list[Ereignis]] = defaultdict(list)
            for e in evs:
                je_name[e.name].append(e)
            for name in sorted(je_name):
                eintraege = je_name[name]
                werkzeuge = sorted({e.werkzeug for e in eintraege})
                da = {q for e in eintraege for q, _ in e.quellen()}
                if len(werkzeuge) > 1:
                    text = f"{anzeige}: `{name}` hat je Quelle ein anderes Werkzeug: " + ", ".join(
                        f"`{w or '—'}` ({_namen(q for e in eintraege if e.werkzeug == w for q in e.quellen())})"
                        for w in werkzeuge)
                elif da != alle:
                    fehlt = sorted(alle - da)
                    text = f"{anzeige}: `{name}` steht nur in {', '.join(sorted(da))}, nicht in " \
                           f"{', '.join(fehlt)}"
                else:
                    continue
                b = eintraege[0].beleg
                pp.append(Pruefpunkt("ereignis_quellen_abweichend", text, b.file, b.zeile, f"ereignis:{key}:{name}"))
        if key not in zuordnung:
            pp.append(Pruefpunkt(
                "space_ohne_doku", f"{anzeige}: Events aus {_namen(q for e in evs for q in e.quellen())}, aber keine "
                "Space-Doku "
                f"(docs/spaces/<NNN>_{key}.md oder doku.space_dokumente)", None, None, f"space:{key}"))
    namen = {e.name for e in ereignisse}
    andere = {n.lower() for n in nicht_ereignisse}
    bekannt = {e.name.split(".", 1)[0] for e in ereignisse if "." in e.name}
    bekannt |= {p.strip().rstrip(".") for p in praefixe if p.strip(" .")}
    for doc in sorted(set(zuordnung.values())):
        text = doku_texte.get(doc)
        if text is None:
            continue
        gesehen: set[str] = set()
        for nr, zeile in enumerate(_ohne_eigene_bloecke(text), 1):
            for m in _TOKEN.finditer(zeile):
                token = m.group(0)
                praefix, rest = token.split(".", 1)[0], token.rsplit(".", 1)[1]
                if token in namen or token in gesehen or praefix not in bekannt or rest in DATEI_ENDUNGEN \
                        or token in andere:
                    continue
                gesehen.add(token)
                pp.append(Pruefpunkt("ereignis_nur_doku", f"{doc}: `{token}` steht in der Doku, aber in keiner "
                                     f"Ereignis-Quelle (Präfix `{praefix}.` ist bekannt)", doc, nr, f"{doc}#{token}"))
    return pp
