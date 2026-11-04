"""Konfig-Inventar (Stufe 5, Task 5): Namen von Env-Variablen, Schluesseln, Ports und LLM-Clients.

Nur Namen, nie Werte (D6). Der Lauf schreibt nichts und liest nur:

- versionierte Dateien (R-a): `git ls-files` im aeusseren Repo und in jedem ausgecheckten Submodul, Repo-Namen wie
  in Task 2 (`<aussen>` bzw. `<aussen>/<pfad>`), dieselben Ausschluesse wie Stufe 1 (`STANDARD_AUSGENOMMEN` und
  `ausgenommen` der Regeldatei des jeweiligen Repos);
- den Graphen des Selbstbild-Stands (`graph.sqlite`, nur `mode=ro`): Kanten `liest_env` auf Knoten `env:<NAME>`.

Quellen und ihre Bedeutung:

| Quelle | gesetzt | gelesen |
|---|---|---|
| Graph `liest_env` (Python, TS/JS, Bash, Rust, Go, YAML ausser Compose) | – | ja |
| `.env.example`/`.env.sample`/`.env.template` (auch `*.example.env` usw.) | links vom ersten `=` | – |
| Compose `services.*.environment` (Schluessel bzw. links vom `=`) | ja | – |
| Compose `${NAME}` | – | ja |
| PowerShell `$env:NAME =` (auch `+=` usw.) | ja | – |
| PowerShell `$env:NAME` sonst | – | ja |
| `config/dienste.yaml` (`dienste.<name>.port`) | Port | – |
| `ast` auf `.py`: `Anthropic(`, `OpenAI(` usw. | LLM-Client | – |

Aus dem Graphen werden Kanten aus Dotenv-, Compose- und `.ps1`/`.psm1`-Dateien nicht genommen: Stufe 1 legt dort
Setzen und Lesen gleichermassen als `liest_env` ab; diese Dateien liest das Inventar selbst mit der Unterscheidung.

Geheimnis-Regel (D6, R-b): von den Env-artigen Dateien wird nur geoeffnet, was `darf_oeffnen` erlaubt. Jede
Dateilesung geht durch `_lesen`, das die Regel vorher prueft. Aus erlaubten Dateien wird je Zeile nur der Text links
vom ersten `=` behalten. Namen, die wie ein Schluessel aussehen (`pruefen`), fallen ganz heraus; die Notiz nennt nur
das Muster.

Notizen (`Inventar.notizen`): Ist der Graph nicht da oder nicht lesbar, beginnt die Notiz mit
`quelle-unlesbar:graph`; das Inventar entsteht trotzdem aus den Dateien, nur ohne Befund `tot` (ohne Lesestellen
waere jeder gesetzte Name "tot"). Ein unlesbares Dienst-Register gibt `quelle-unlesbar:dienste`. Alles andere sind
Zaehl-Notizen ohne Praefix.
"""
from __future__ import annotations

import ast
import fnmatch
import json
import posixpath
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from .._glob import glob_passt, glob_regex
from ..analyse.lauf import _stand_meta
from ..parser.daten import _PS_RE, _REF_RE
from ..quelle import _git
from ..regeln import REGELDATEI, STANDARD_AUSGENOMMEN, Regeln, regeln_laden
from ..speicher import _kante_von, _sortierschluessel, pfad_kennung
from ..spezial.compose import _verboten as _compose_verboten
from ..sprachen import sprache_fuer
from ..vermessen import stand_finden
from .sammeln import UNLESBAR_PRAEFIX, _aussen_name, _id_teile, _submodul_pfade
from .text import KAPPEN_RESERVE, _zelle, kappen, kappen_zeile, pruefen
from .typen import Fund

InventarArt = Literal["env", "key", "port", "llm_client"]
INVENTAR_ARTEN: tuple[str, ...] = ("env", "key", "port", "llm_client")
BEFUNDE: tuple[str, ...] = ("doppelt", "widerspruechlich", "tot", "ungesetzt")
KEY_ENDEN: tuple[str, ...] = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
DIENSTE_STANDARD = "config/dienste.yaml"

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ERLAUBT_GENAU = frozenset({".env.example", ".env.sample", ".env.template"})
_ERLAUBT_ENDE = (".example.env", ".sample.env", ".template.env")
# Betriebssystem- und Laufzeitnamen: werden gelesen, aber nie im Projekt gesetzt -> kein Befund `ungesetzt`
SYSTEM_NAMEN = frozenset({
    "APPDATA", "CI", "COMPUTERNAME", "HOME", "HOSTNAME", "LANG", "LOCALAPPDATA", "PATH", "PROGRAMDATA",
    "PROGRAMFILES", "PWD", "SHELL", "SYSTEMROOT", "TEMP", "TERM", "TMP", "TMPDIR", "USER", "USERNAME",
    "USERPROFILE", "WINDIR",
})
_PS_ENDUNGEN = (".ps1", ".psm1")
_PS_ZUWEISUNG = re.compile(r"\}?[ \t]*[-+*/]?=(?!=)")
_LLM_KLASSEN = {
    "Anthropic": "anthropic", "AsyncAnthropic": "anthropic", "ChatAnthropic": "anthropic",
    "OpenAI": "openai", "AsyncOpenAI": "openai", "ChatOpenAI": "openai",
}
_LLM_VORFILTER = (b"Anthropic", b"OpenAI", b"ollama", b"get_client")
GET_CLIENT_ANBIETER = "vibemind_shared"
_ZEILE_ORT = re.compile(r"^(.*):\d+(?: .*)?$")
_ZELLE_KURZ = 3


@dataclass(frozen=True)
class Name:
    art: InventarArt
    name: str
    gelesen: list[str] = field(default_factory=list)
    gesetzt: list[str] = field(default_factory=list)
    befund: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LlmClient:
    anbieter: str  # anthropic | openai | ollama | vibemind_shared
    modell: str | None  # nur ein String-Literal `model=`
    fundstelle: str  # `<repo>/<posix-pfad>:<zeile>`


@dataclass(frozen=True)
class Inventar:
    code_repo: str  # aeusserer Repo-Name (Task 2)
    namen: list[Name]  # sortiert nach Art (INVENTAR_ARTEN), dann Name (Ports numerisch)
    clients: list[LlmClient]  # sortiert
    notizen: list[str]  # sortiert; `quelle-unlesbar:graph...` / `quelle-unlesbar:dienste...` siehe Modul-Doc
    uebersprungen: int  # Dateien, die nicht lesbar oder nicht parsebar waren (gezaehlt, nie Absturz)
    dienste_pfad: str = DIENSTE_STANDARD


# ---------------------------------------------------------------- Geheimnis-Regel

def darf_oeffnen(dateiname: str) -> bool:
    """Darf eine Datei mit diesem Basisnamen ueberhaupt geoeffnet werden (D6, R-b)?

    Erlaubt sind von den Env-artigen nur `.env.example`, `.env.sample`, `.env.template` und `*.example.env`,
    `*.sample.env`, `*.template.env`. Nie: `.env`, `.env.*`, `*.env` (auch `secrets.env`) und `*.pem`.
    """
    n = dateiname.lower()
    if n in _ERLAUBT_GENAU or n.endswith(_ERLAUBT_ENDE):
        return True
    return not (n == ".env" or n.startswith(".env.") or n.endswith(".env") or n.endswith(".pem"))


def ist_env_beispiel(dateiname: str) -> bool:
    n = dateiname.lower()
    return n in _ERLAUBT_GENAU or n.endswith(_ERLAUBT_ENDE)


def _lesen(pfad: Path, max_bytes: int) -> bytes | None:
    """Der einzige Lesepfad fuer Repo-Dateien: prueft die Geheimnis-Regel vor jedem Oeffnen."""
    if not darf_oeffnen(pfad.name):
        return None
    try:
        if pfad.is_symlink() or not pfad.is_file() or pfad.stat().st_size > max_bytes:
            return None
        return pfad.read_bytes()
    except OSError:
        return None


def _text(quelle: bytes) -> str:
    return quelle.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def env_namen(quelle: bytes) -> list[tuple[str, int]]:
    """(Name, Zeile) aus einer erlaubten Env-Beispieldatei: nur der Text links vom ersten `=`.

    Kommentarzeilen (`#`) und Zeilen ohne `=` entfallen, ein `export `-Praefix wird entfernt. Die rechte Seite
    wird nie als eigener Wert angelegt (nur ein Slice bis zum `=`)."""
    aus: list[tuple[str, int]] = []
    for nr, zeile in enumerate(_text(quelle).split("\n"), start=1):
        gleich = zeile.find("=")
        if gleich < 0:
            continue
        links = zeile[:gleich].strip()
        if links.startswith("#"):
            continue
        if links.startswith("export ") or links.startswith("export\t"):
            links = links[len("export"):].strip()
        if _NAME.fullmatch(links):
            aus.append((links, nr))
    return aus


# ---------------------------------------------------------------- Dateien (R-a)

@dataclass(frozen=True)
class _Datei:
    repo: str
    pfad: str  # POSIX, repo-relativ
    voll: Path
    max_bytes: int


def _regeln(ordner: Path) -> Regeln:
    try:
        return regeln_laden((ordner / REGELDATEI).read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return regeln_laden(None)


def _dateien(wurzel: Path, aussen: str, notizen: list[str]) -> list[_Datei]:
    repos = [(aussen, wurzel)] + [
        (f"{aussen}/{s}", wurzel / s) for s in sorted(_submodul_pfade(wurzel)) if (wurzel / s / ".git").exists()]
    aus: list[_Datei] = []
    for repo, ordner in repos:
        regeln = _regeln(ordner)
        muster = [glob_regex(m) for m in (*STANDARD_AUSGENOMMEN, *regeln.ausgenommen)]
        erg = _git(ordner, "ls-files", "-z", check=False)
        if erg.returncode != 0:
            notizen.append(f"git ls-files fehlgeschlagen: {repo} (nicht inventarisiert)")
            continue
        for roh in erg.stdout.split(b"\0"):
            if not roh:
                continue
            pfad = roh.decode("utf-8", errors="surrogateescape").replace("\\", "/")
            if glob_passt(pfad, muster):
                continue
            aus.append(_Datei(repo, pfad, ordner / pfad, regeln.max_dateigroesse_kb * 1024))
    return sorted(aus, key=lambda d: (d.repo, d.pfad))


def _ort(repo: str, pfad: str, zeile: int | None) -> str:
    return f"{repo}/{pfad}:{zeile}" if zeile else f"{repo}/{pfad}"


def _ist_compose(basis: str) -> bool:
    return fnmatch.fnmatch(basis.lower(), "*compose*.y*ml")


# ---------------------------------------------------------------- Graph

def _graph_db(wurzel: Path, stand_unter: Path | None) -> Path | None:
    if stand_unter is None:
        return stand_finden(wurzel)
    s = Path(stand_unter)
    if (s / "graph.sqlite").is_file():
        return s / "graph.sqlite"
    if s.is_dir():
        treffer = sorted(o / "graph.sqlite" for o in s.glob(f"*-{pfad_kennung(wurzel)}")
                         if (o / "graph.sqlite").is_file())
        return treffer[-1] if treffer else None
    return None


def _graph_quelle_ok(pfad: str) -> bool:
    """Kanten aus diesen Dateiarten vermischen Setzen und Lesen (Stufe 1); das Inventar liest sie selbst."""
    basis = posixpath.basename(pfad).lower()
    if sprache_fuer(pfad) == "dotenv" or not darf_oeffnen(basis) or ist_env_beispiel(basis):
        return False
    return not (basis.endswith(_PS_ENDUNGEN) or _ist_compose(basis))


_NICHT_DA = f"{UNLESBAR_PRAEFIX}graph: Graph fehlt (erst `python -m selbstbild vermessen`): Lesestellen nur aus " \
            "PowerShell und Compose, kein Befund tot"


def _graph_lesestellen(wurzel: Path, stand_unter: Path | None,
                       lesen: dict[str, set[str]]) -> list[str]:
    db = _graph_db(wurzel, stand_unter)
    if db is None:
        return [_NICHT_DA]
    try:
        commit, _ = _stand_meta(db.parent)
        if commit is None:
            return [f"{UNLESBAR_PRAEFIX}graph: Graph nicht lesbar (Stand ohne Graph): kein Befund tot"]
        c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
        try:
            c.execute("BEGIN")
            try:
                zeilen = c.execute("SELECT wert FROM kanten WHERE s_art = ?",
                                   (_sortierschluessel("liest_env"),)).fetchall()
            finally:
                c.execute("COMMIT")
        finally:
            c.close()
    except (RuntimeError, sqlite3.Error, OSError, ValueError) as fehler:
        return [f"{UNLESBAR_PRAEFIX}graph: Graph nicht lesbar ({type(fehler).__name__}): kein Befund tot"]
    kaputt = 0
    for (wert,) in zeilen:
        try:
            k = _kante_von(json.loads(wert))
            if not k.nach.startswith("env:"):
                continue
            name = k.nach[len("env:"):]
            if not _NAME.fullmatch(name):
                continue
            if k.beleg is not None:
                repo, pfad, zeile = k.beleg.repo, k.beleg.pfad.replace("\\", "/"), int(k.beleg.zeile)
            else:
                repo, pfad = _id_teile(k.von)
                zeile = 0
            if not _graph_quelle_ok(pfad):
                continue
            lesen[name].add(_ort(repo, pfad, zeile))
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            kaputt += 1
    return [f"{UNLESBAR_PRAEFIX}graph: {kaputt} fehlerhafte liest_env-Kanten übersprungen"] if kaputt else []


# ---------------------------------------------------------------- YAML-Knoten (mit Zeilen, ohne Werte zu bauen)

def _items(knoten: yaml.Node | None, tiefe: int = 0) -> list[tuple[yaml.Node, yaml.Node]]:
    """Schluessel/Wert-Paare eines Mapping-Knotens, Merge-Schluessel `<<` aufgeloest."""
    if not isinstance(knoten, yaml.MappingNode) or tiefe > 10:
        return []
    aus: list[tuple[yaml.Node, yaml.Node]] = []
    for k, v in knoten.value:
        if isinstance(k, yaml.ScalarNode) and k.tag == "tag:yaml.org,2002:merge":
            for quelle in (v.value if isinstance(v, yaml.SequenceNode) else [v]):
                aus += _items(quelle, tiefe + 1)
        else:
            aus.append((k, v))
    return aus


def _holen(knoten: yaml.Node | None, schluessel: str) -> yaml.Node | None:
    for k, v in _items(knoten):
        if isinstance(k, yaml.ScalarNode) and k.value == schluessel:
            return v
    return None


def _env_schluessel(env: yaml.Node | None) -> list[tuple[str, int]]:
    aus: list[tuple[str, int]] = []
    if isinstance(env, yaml.MappingNode):
        for k, _ in _items(env):
            if isinstance(k, yaml.ScalarNode) and _NAME.fullmatch(k.value.strip()):
                aus.append((k.value.strip(), k.start_mark.line + 1))
    elif isinstance(env, yaml.SequenceNode):
        for e in env.value:
            if isinstance(e, yaml.ScalarNode):
                gleich = e.value.find("=")
                links = (e.value if gleich < 0 else e.value[:gleich]).strip()
                if _NAME.fullmatch(links):
                    aus.append((links, e.start_mark.line + 1))
    return aus


def _compose(text: str, ort: str, setzen: dict[str, set[str]], lesen: dict[str, set[str]]) -> None:
    for dok in yaml.compose_all(text, Loader=yaml.SafeLoader):
        for _, dienst in _items(_holen(dok, "services")):
            for name, zeile in _env_schluessel(_holen(dienst, "environment")):
                setzen[name].add(f"{ort}:{zeile}")
    for nr, zeile in enumerate(text.split("\n"), start=1):
        if zeile.lstrip().startswith("#"):
            continue
        for name in _REF_RE.findall(zeile):
            if _NAME.fullmatch(name):
                lesen[name].add(f"{ort}:{nr}")


def _powershell(text: str, ort: str, setzen: dict[str, set[str]], lesen: dict[str, set[str]]) -> None:
    for nr, zeile in enumerate(text.split("\n"), start=1):
        if zeile.lstrip().startswith("#"):
            continue
        for m in _PS_RE.finditer(zeile):
            ziel = setzen if _PS_ZUWEISUNG.match(zeile, m.end()) else lesen
            ziel[m.group(1)].add(f"{ort}:{nr}")


def _ports(text: str, ort: str) -> list[tuple[int, str, str]]:
    """(Port, Dienst, Fundstelle) aus `dienste: {<name>: {port: <int>|[<int>, ...]|null}}`; doppelte Dienst-
    Schluessel bleiben erhalten (Knotenbaum statt dict)."""
    aus: list[tuple[int, str, str]] = []
    for dok in yaml.compose_all(text, Loader=yaml.SafeLoader):
        for k, v in _items(_holen(dok, "dienste")):
            if not isinstance(k, yaml.ScalarNode):
                continue
            p = _holen(v, "port")
            for pn in (p.value if isinstance(p, yaml.SequenceNode) else [p]):
                if isinstance(pn, yaml.ScalarNode) and pn.value.strip().isdigit():
                    aus.append((int(pn.value.strip()), k.value, f"{ort}:{pn.start_mark.line + 1} ({k.value})"))
    return aus


# ---------------------------------------------------------------- LLM-Clients (ast)

def _punktname(knoten: ast.expr) -> str | None:
    teile: list[str] = []
    while isinstance(knoten, ast.Attribute):
        teile.append(knoten.attr)
        knoten = knoten.value
    if not isinstance(knoten, ast.Name):
        return None
    teile.append(knoten.id)
    return ".".join(reversed(teile))


def _wurzelmodul(modul: str | None) -> str:
    return (modul or "").split(".", 1)[0]


def _anbieter(func: ast.expr, importe: dict[str, tuple[str, str | None]]) -> str | None:
    if isinstance(func, ast.Name):
        modul, orig = importe.get(func.id, (None, func.id))
        orig = orig or func.id
        if orig in _LLM_KLASSEN:
            return _LLM_KLASSEN[orig]
        if orig == "Client" and _wurzelmodul(modul) == "ollama":
            return "ollama"
        if orig == "get_client" and _wurzelmodul(modul) == GET_CLIENT_ANBIETER:
            return GET_CLIENT_ANBIETER
        return None
    if isinstance(func, ast.Attribute):
        if func.attr in _LLM_KLASSEN:
            return _LLM_KLASSEN[func.attr]
        basis = _punktname(func.value)
        if basis is None:
            return None
        erster = basis.split(".", 1)[0]
        modul, orig = importe.get(erster, (erster, None))
        if orig is not None:  # `from x import y; y.Client()`: y ist kein Modul-Alias, sondern ein Name aus x
            modul = f"{modul}.{orig}"
        if func.attr == "Client" and _wurzelmodul(modul) == "ollama":
            return "ollama"
        if func.attr == "get_client" and _wurzelmodul(modul) == GET_CLIENT_ANBIETER:
            return GET_CLIENT_ANBIETER
    return None


def llm_clients(quelle: bytes, repo: str, pfad: str) -> list[LlmClient] | None:
    """LLM-Client-Erzeugungen einer Python-Datei; None, wenn `ast` die Datei nicht lesen kann."""
    if not any(m in quelle for m in _LLM_VORFILTER):
        return []
    try:
        baum = ast.parse(quelle)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    importe: dict[str, tuple[str, str | None]] = {}
    knoten = list(ast.walk(baum))
    for n in knoten:
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    importe[a.asname] = (a.name, None)
                else:
                    importe[a.name.split(".", 1)[0]] = (a.name.split(".", 1)[0], None)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                importe[a.asname or a.name] = (n.module, a.name)
    aus: list[LlmClient] = []
    for n in knoten:
        if not isinstance(n, ast.Call):
            continue
        anbieter = _anbieter(n.func, importe)
        if anbieter is None:
            continue
        modell = next((kw.value.value for kw in n.keywords if kw.arg == "model"
                       and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)), None)
        aus.append(LlmClient(anbieter, modell, _ort(repo, pfad, n.lineno)))
    return aus


# ---------------------------------------------------------------- Inventar

def _datei_von(ort: str) -> str:
    m = _ZEILE_ORT.match(ort)
    return m.group(1) if m else ort


def _sortierschluessel_name(n: Name) -> tuple[int, str]:
    return INVENTAR_ARTEN.index(n.art), (n.name.zfill(6) if n.art == "port" else n.name)


def _env_eintraege(setzen: dict[str, set[str]], lesen: dict[str, set[str]], *, mit_tot: bool) -> list[Name]:
    aus: list[Name] = []
    for name in sorted(set(setzen) | set(lesen)):
        gesetzt, gelesen = sorted(setzen.get(name, ())), sorted(lesen.get(name, ()))
        befund: list[str] = []
        if len({_datei_von(o) for o in gesetzt}) >= 2:
            befund.append("doppelt")
        if gesetzt and not gelesen and mit_tot:
            befund.append("tot")
        if gelesen and not gesetzt and name.upper() not in SYSTEM_NAMEN:
            befund.append("ungesetzt")
        art: InventarArt = "key" if name.upper().endswith(KEY_ENDEN) else "env"
        aus.append(Name(art, name, gelesen, gesetzt, sorted(befund)))
    return aus


def _port_eintraege(ports: list[tuple[int, str, str]]) -> list[Name]:
    dienste_je_port: dict[int, set[str]] = defaultdict(set)
    ports_je_dienst: dict[str, set[int]] = defaultdict(set)
    orte: dict[int, set[str]] = defaultdict(set)
    for port, dienst, ort in ports:
        dienste_je_port[port].add(dienst)
        ports_je_dienst[dienst].add(port)
        orte[port].add(ort)
    aus: list[Name] = []
    for port in sorted(orte):
        befund: set[str] = set()
        if len(dienste_je_port[port]) > 1:
            befund.add(f"widerspruechlich: Port {port} für {', '.join(sorted(dienste_je_port[port]))}")
        for dienst in sorted(dienste_je_port[port]):
            if len(ports_je_dienst[dienst]) > 1:
                liste = ", ".join(str(p) for p in sorted(ports_je_dienst[dienst]))
                befund.add(f"widerspruechlich: Dienst {dienst} mit Ports {liste}")
        aus.append(Name("port", str(port), [], sorted(orte[port]), sorted(befund)))
    return aus


def _client_eintraege(clients: list[LlmClient]) -> list[Name]:
    je: dict[str, set[str]] = defaultdict(set)
    for c in clients:
        je[c.anbieter].add(c.fundstelle + (f" model={c.modell}" if c.modell else ""))
    return [Name("llm_client", a, [], sorted(o), []) for a, o in sorted(je.items())]


def inventar(wurzel: Path, *, stand_unter: Path | None = None) -> Inventar:
    """Konfig-Inventar ueber `wurzel` (nur lesend, nur Namen). `stand_unter` wie in `funde_sammeln`: ein Stand-
    Ordner mit `graph.sqlite` oder ein Basisordner mit `<name>-<kennung>`-Staenden; None = `stand_finden`."""
    wurzel = Path(wurzel)
    aussen = _aussen_name(wurzel)
    notizen: list[str] = []
    setzen: dict[str, set[str]] = defaultdict(set)
    lesen: dict[str, set[str]] = defaultdict(set)

    graph_notizen = _graph_lesestellen(wurzel, stand_unter, lesen)
    notizen += graph_notizen
    graph_ok = not any(n.startswith(f"{UNLESBAR_PRAEFIX}graph: Graph") for n in graph_notizen)

    dienste_pfad = (_regeln(wurzel).dienste or DIENSTE_STANDARD).replace("\\", "/")
    dateien = _dateien(wurzel, aussen, notizen)
    ports: list[tuple[int, str, str]] = []
    clients: list[LlmClient] = []
    nicht_lesbar: Counter[str] = Counter()
    dienste_da = False
    for d in dateien:
        basis = posixpath.basename(d.pfad)
        ort = f"{d.repo}/{d.pfad}"
        if ist_env_beispiel(basis):
            art = "env-beispiel"
        elif _ist_compose(basis) and not _compose_verboten(d.pfad):
            art = "compose"
        elif basis.lower().endswith(_PS_ENDUNGEN):
            art = "powershell"
        elif d.repo == aussen and d.pfad == dienste_pfad:
            art = "dienste"
        elif basis.endswith(".py"):
            art = "python"
        else:
            continue
        quelle = _lesen(d.voll, d.max_bytes)
        if quelle is None:
            if d.voll.is_dir():  # Gitlink eines Submoduls
                continue
            nicht_lesbar[art] += 1
            continue
        try:
            if art == "env-beispiel":
                for name, zeile in env_namen(quelle):
                    setzen[name].add(f"{ort}:{zeile}")
            elif art == "compose":
                _compose(_text(quelle), ort, setzen, lesen)
            elif art == "powershell":
                _powershell(_text(quelle), ort, setzen, lesen)
            elif art == "dienste":
                dienste_da = True
                ports += _ports(_text(quelle), ort)
            else:
                gefunden = llm_clients(quelle, d.repo, d.pfad)
                if gefunden is None:
                    nicht_lesbar[art] += 1
                else:
                    clients += gefunden
        except (yaml.YAMLError, RecursionError, ValueError):
            if art == "dienste":
                notizen.append(f"{UNLESBAR_PRAEFIX}dienste: {dienste_pfad} nicht lesbar: keine Ports")
            nicht_lesbar[art] += 1
        del quelle
    if not dienste_da and not any(n.startswith(f"{UNLESBAR_PRAEFIX}dienste") for n in notizen):
        notizen.append(f"Dienst-Register {dienste_pfad} nicht versioniert oder nicht lesbar: keine Ports")
    namen_text = {"python": ".py-Datei(en)", "compose": "Compose-Datei(en)", "powershell": "PowerShell-Datei(en)",
                  "env-beispiel": "Env-Beispieldatei(en)", "dienste": "Dienst-Register"}
    for art, n in sorted(nicht_lesbar.items()):
        notizen.append(f"{n} {namen_text[art]} nicht lesbar oder nicht parsebar, übersprungen")

    roh = (_env_eintraege(setzen, lesen, mit_tot=graph_ok) + _port_eintraege(ports)
           + _client_eintraege(sorted(clients, key=lambda c: (c.anbieter, c.fundstelle, c.modell or ""))))
    namen, verdacht = [], Counter[tuple[str, str]]()
    for n in roh:
        muster = pruefen("\n".join([n.name, *n.gelesen, *n.gesetzt, *n.befund]))
        if muster:
            verdacht[(muster, n.art)] += 1
        else:
            namen.append(n)
    sauber: list[LlmClient] = []
    for c in clients:
        muster = pruefen(f"{c.fundstelle}\n{c.modell or ''}")
        if muster:
            verdacht[(muster, "llm_client")] += 1
        else:
            sauber.append(c)
    for (muster, art), k in sorted(verdacht.items()):
        notizen.append(f"schluessel-verdacht: {muster}: {k} Einträge ({art}) ausgelassen")
    return Inventar(
        code_repo=aussen,
        namen=sorted(namen, key=_sortierschluessel_name),
        clients=sorted(set(sauber), key=lambda c: (c.anbieter, c.fundstelle, c.modell or "")),
        notizen=sorted(notizen),
        uebersprungen=sum(nicht_lesbar.values()),
        dienste_pfad=dienste_pfad,
    )


# ---------------------------------------------------------------- Text

def _zaehlen(inv: Inventar) -> tuple[Counter[str], Counter[str]]:
    arten = Counter(n.art for n in inv.namen)
    befunde = Counter(b.split(":", 1)[0] for n in inv.namen for b in n.befund)
    return arten, befunde


def konsolidierungsplan(inv: Inventar) -> list[str]:
    """Je Gruppe ein Vorschlag, feste Reihenfolge; leer, wenn nichts zu tun ist."""
    def namen_mit(befund: str) -> int:
        return sum(1 for n in inv.namen if befund in n.befund)

    widersprueche = len({b for n in inv.namen for b in n.befund if b.startswith("widerspruechlich")})
    direkt = sum(1 for c in inv.clients if c.anbieter != GET_CLIENT_ANBIETER)
    plan: list[str] = []
    if (k := namen_mit("doppelt")):
        plan.append(f"{k} doppelt gesetzte Namen in eine Quelle zusammenführen (je Name eine `.env.example` "
                    "oder ein Compose-Block).")
    if widersprueche:
        plan.append(f"{widersprueche} Port-Widersprüche in `{inv.dienste_pfad}` auflösen (ein Port je Dienst, "
                    "ein Dienst je Port).")
    if (k := namen_mit("ungesetzt")):
        plan.append(f"{k} gelesene, aber nirgends gesetzte Namen in eine `.env.example` aufnehmen (nur der Name).")
    if (k := namen_mit("tot")):
        plan.append(f"{k} gesetzte, aber nie gelesene Namen prüfen und streichen.")
    if direkt:
        plan.append(f"{direkt} LLM-Clients ohne `get_client(role)`: auf `vibemind_shared.get_client` umstellen.")
    return plan


def _orte(orte: list[str], voll: bool) -> str:
    if not orte:
        return "–"
    zeigen = orte if voll else orte[:_ZELLE_KURZ]
    rest = len(orte) - len(zeigen)
    return "<br>".join(_zelle(o) for o in zeigen) + (f" (+{rest})" if rest else "")


def _zeile(n: Name, voll: bool) -> str:
    befund = "<br>".join(_zelle(b) for b in n.befund) or "–"
    return f"| {n.art} | `{_zelle(n.name)}` | {_orte(n.gelesen, voll)} | {_orte(n.gesetzt, voll)} | {befund} |"


_TABELLENKOPF = "### Inventar\n\n| Art | Name | gelesen | gesetzt | Befund |\n|---|---|---|---|---|"


def _kopf(inv: Inventar) -> str:
    arten, befunde = _zaehlen(inv)
    plan = konsolidierungsplan(inv)
    plan_text = "\n".join(f"{i}. {p}" for i, p in enumerate(plan, start=1)) or "Keine Konsolidierung nötig."
    kopf = (f"**Konfig-Inventar {inv.code_repo}**: nur Namen, nie Werte (D6).\n\n"
            "Namen: " + ", ".join(f"{a} {arten[a]}" for a in INVENTAR_ARTEN)
            + ". Befunde: " + ", ".join(f"{b} {befunde[b]}" for b in BEFUNDE)
            + f". Übersprungene Dateien: {inv.uebersprungen}.\n\n"
            f"### Konsolidierungsplan\n\n{plan_text}\n\n")
    if inv.notizen:
        kopf += "### Hinweise\n\n" + "\n".join(f"- {_zelle(n)}" for n in inv.notizen) + "\n\n"
    return kopf


def inventar_tabelle(inv: Inventar) -> str:
    """Der volle Bericht ohne Kappe (fuer den lokalen Bericht, Task 6): Kopf, Plan, Hinweise, alle Zeilen mit allen
    Fundstellen, dazu die LLM-Clients einzeln."""
    zeilen = [_zeile(n, True) for n in inv.namen]
    clients = "\n".join(f"| {c.anbieter} | {_zelle(c.modell) if c.modell else '–'} | {_zelle(c.fundstelle)} |"
                        for c in inv.clients)
    teil_clients = ("\n\n### LLM-Clients\n\n| Anbieter | Modell | Fundstelle |\n|---|---|---|\n" + clients
                    if clients else "")
    return _kopf(inv) + _TABELLENKOPF + "".join("\n" + z for z in zeilen) + teil_clients + "\n"


def _text_reihenfolge(namen: list[Name]) -> list[Name]:
    """Im Issue zuerst die Eintraege mit Befund, sonst in Inventar-Reihenfolge."""
    return sorted(namen, key=lambda n: (0 if n.befund else 1, *_sortierschluessel_name(n)))


def inventar_fund(inv: Inventar, *, max_zeichen: int = 60000) -> Fund:
    """Der eine `konfig`-Fund (Schwere mittel, Anker `konfig-inventar:<aussen>`, R-f). Text = Kopf mit Plan und
    Tabelle "Name -> wo gelesen, wo gesetzt, Befund", gekappt wie in Task 3."""
    kopf = _kopf(inv)
    zeilen = [_zeile(n, False) for n in _text_reihenfolge(inv.namen)]
    budget = max_zeichen - len(kopf) - len(_TABELLENKOPF) - KAPPEN_RESERVE - 1
    behalten, k = kappen(zeilen, budget)
    text = kopf + _TABELLENKOPF + "".join("\n" + z for z in behalten)
    if k:
        text += "\n\n" + kappen_zeile(k)
    text += "\n"
    _, befunde = _zaehlen(inv)
    belege = sorted({_datei_von(o) for n in inv.namen if n.art != "llm_client" for o in n.gesetzt})
    return Fund(
        art="konfig", schwere="mittel", code_repo=inv.code_repo,
        titel=f"Konfig-Inventar: {len(inv.namen)} Namen, {sum(befunde.values())} Befunde",
        text=text, belege=tuple(belege[:50]), anker=f"konfig-inventar:{inv.code_repo}")
