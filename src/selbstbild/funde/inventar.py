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
| Compose `services.*.environment` (Schluessel bzw. links vom `=`; explizit vor `<<`) | ja | – |
| Compose `${NAME}`, `${NAME:-…}`, `${NAME?…}`; `$NAME` nur als ganzer Wert | – | ja |
| PowerShell `$env:NAME =` (auch `+=`, `??=` …), `[Environment]::SetEnvironmentVariable("NAME"`, `Set-Item`/`New-Item env:NAME` | ja | – |
| PowerShell `$env:NAME` sonst (ohne `#`- und `<# #>`-Kommentare) | – | ja |
| `config/dienste.yaml` (`dienste.<name>.port`) | Port | – |
| `ast` auf `.py`: `Anthropic(`, `OpenAI(` usw. | LLM-Client | – |

Env-Namen werden ohne Gross-/Kleinschreibung zusammengefuehrt (Windows-Env) und gross geschrieben angezeigt.
Aus dem Graphen werden Kanten aus Dotenv-, Compose- und `.ps1`/`.psm1`-Dateien nicht genommen: Stufe 1 legt dort
Setzen und Lesen gleichermassen als `liest_env` ab; diese Dateien liest das Inventar selbst mit der Unterscheidung.
`ungesetzt` zaehlt nur Lesestellen in produktivem Code (S5-13).

Geheimnis-Regel (D6, R-b): `darf_oeffnen` verweigert standardmaessig jede Geheimnis-artige Datei; jede Dateilesung
geht durch `_lesen`, das die Regel vorher prueft. Aus erlaubten Env-Beispieldateien wird je Zeile nur der Text links
vom ersten `=` behalten; mehrzeilige Werte in Anfuehrungszeichen werden bis zu ihrem Ende uebersprungen. Namen, die
wie ein Schluessel aussehen (`pruefen`), fallen ganz heraus; die Notiz nennt nur das Muster.

Notizen (`Inventar.notizen`): Praefix `quelle-unlesbar:` fuer `graph` (Graph fehlt, kaputt), `dienste` (Register
konfiguriert, aber fehlt/zu gross/kaputt), `submodul` (deklariert, nicht ausgecheckt) und `dateien` (`git ls-files`
gescheitert); alle bilden in `sync.unlesbare_arten` auf `konfig` ab. Ohne Graph gibt es keinen Befund `tot` (ohne
Lesestellen waere jeder gesetzte Name "tot"). Alles andere sind Zaehl-Notizen ohne Praefix.
"""
from __future__ import annotations

import ast
import fnmatch
import json
import posixpath
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from .._glob import glob_passt, glob_regex
from ..analyse.lauf import _stand_meta
from ..fremdcode import fremd_pruefer
from ..parser.daten import _PS_RE
from ..quelle import _git_args, _git_env
from ..regeln import REGELDATEI, STANDARD_AUSGENOMMEN, Regeln, regeln_laden
from ..speicher import _kante_von, _sortierschluessel, pfad_kennung
from ..spezial.compose import _verboten as _compose_verboten
from ..sprachen import sprache_fuer
from ..vermessen import stand_finden
from .sammeln import UNLESBAR_PRAEFIX, _aussen_name, _id_teile, _submodul_pfade, ist_nicht_produktiv
from .text import KAPPEN_RESERVE, _zelle, kappen, kappen_zeile, pruefen
from .typen import Fund

InventarArt = Literal["env", "key", "port", "llm_client"]
INVENTAR_ARTEN: tuple[str, ...] = ("env", "key", "port", "llm_client")
BEFUNDE: tuple[str, ...] = ("doppelt", "widerspruechlich", "tot", "ungesetzt")
KEY_ENDEN: tuple[str, ...] = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
DIENSTE_STANDARD = "config/dienste.yaml"
GIT_TIMEOUT_S = 60

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ERLAUBT_GENAU = frozenset({".env.example", ".env.sample", ".env.template"})
_ERLAUBT_ENDE = (".example.env", ".sample.env", ".template.env")
GEHEIM_MUSTER: tuple[str, ...] = (
    ".env*", "env.*", "*.env", ".envrc", "*.pem", "*.key", "id_rsa*", "credentials*", "secrets*")
# Betriebssystem- und Laufzeitnamen: werden gelesen, aber nie im Projekt gesetzt -> kein Befund `ungesetzt`
SYSTEM_NAMEN = frozenset({
    "APPDATA", "CI", "COMPUTERNAME", "HOME", "HOSTNAME", "LANG", "LOCALAPPDATA", "PATH", "PROGRAMDATA",
    "PROGRAMFILES", "PWD", "SHELL", "SYSTEMROOT", "TEMP", "TERM", "TMP", "TMPDIR", "USER", "USERNAME",
    "USERPROFILE", "WINDIR",
})
# S5-13: zusaetzlich zu `ist_nicht_produktiv`
_FREMD_SEGMENTE = frozenset({"vendor", "third_party", "node_modules", "site-packages"})
_PS_ENDUNGEN = (".ps1", ".psm1")
_PS_ZUWEISUNG = re.compile(r"\}?[ \t]*(?:\?\?|[-+*/])?=(?!=)")
_PS_SETENV = re.compile(
    r"\[(?:System\.)?Environment\]::SetEnvironmentVariable\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", re.I)
_PS_ITEM = re.compile(r"\b(?:Set|New)-Item\s+(?:-Path\s+)?['\"]?env:\\?([A-Za-z_][A-Za-z0-9_]*)", re.I)
_COMPOSE_KLAMMER = re.compile(r"(?<!\$)\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?[-?+][^}]*)?\}")
_COMPOSE_GANZ = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
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
    notizen: list[str]  # sortiert; Praefixe `quelle-unlesbar:graph|dienste|submodul|dateien` siehe Modul-Doc
    uebersprungen: int  # Dateien, die nicht lesbar oder nicht parsebar waren (gezaehlt, nie Absturz)
    dienste_pfad: str = DIENSTE_STANDARD
    fehlende_submodule: list[str] = field(default_factory=list)  # deklariert, nicht ausgecheckt (Repo-Namen)
    nicht_geoeffnet: int = 0  # Dateien, die die Geheimnis-Regel nicht oeffnen liess


# ---------------------------------------------------------------- Geheimnis-Regel

def ist_env_beispiel(dateiname: str) -> bool:
    n = dateiname.lower()
    return n in _ERLAUBT_GENAU or n.endswith(_ERLAUBT_ENDE)


def darf_oeffnen(dateiname: str) -> bool:
    """Darf eine Datei mit diesem Basisnamen ueberhaupt geoeffnet werden (D6, R-b; Fix-Runde 1: Verbot zuerst)?

    Verweigert (ohne Gross-/Kleinschreibung) jeden Namen auf `GEHEIM_MUSTER`: `.env*` (auch `.env`, `.envrc`),
    `env.*`, `*.env` (auch `secrets.env`), `*.pem`, `*.key`, `id_rsa*`, `credentials*`, `secrets*`. Ausnahme sind
    nur die Beispieldateien `.env.example`, `.env.sample`, `.env.template` und `*.example.env`, `*.sample.env`,
    `*.template.env`. Alle anderen Namen (Quelltext, YAML, …) sind erlaubt.
    """
    if ist_env_beispiel(dateiname):
        return True
    n = dateiname.lower()
    return not any(fnmatch.fnmatchcase(n, m) for m in GEHEIM_MUSTER)


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


def _quote_ende(zeile: str, start: int, q: str) -> int:
    """Index des schliessenden Anfuehrungszeichens ab `start` oder -1; in `"…"` zaehlt `\\"` nicht."""
    i = start
    while i < len(zeile):
        c = zeile[i]
        if c == "\\" and q == '"':
            i += 2
            continue
        if c == q:
            return i
        i += 1
    return -1


def env_namen(quelle: bytes) -> tuple[list[tuple[str, int]], bool]:
    """((Name, Zeile), ..., offen_am_ende) aus einer erlaubten Env-Beispieldatei: nur der Text links vom ersten `=`.

    Kommentarzeilen (`#`) und Zeilen ohne `=` entfallen, ein `export `-Praefix wird entfernt. Beginnt ein Wert mit
    `"`, `'` oder `` ` `` und schliesst nicht in derselben Zeile, werden alle Folgezeilen bis zum schliessenden Zeichen
    uebersprungen (Fix-Runde 1: mehrzeilige Werte liefern keine Namen). Bleibt es bis zum Dateiende offen, ist
    `offen_am_ende` True. Die rechte Seite wird nie als eigener Wert angelegt, nur Zeichen fuer Zeichen geprueft."""
    aus: list[tuple[str, int]] = []
    offen: str | None = None
    for nr, zeile in enumerate(_text(quelle).split("\n"), start=1):
        if offen is not None:
            if _quote_ende(zeile, 0, offen) >= 0:
                offen = None
            continue
        if zeile.lstrip().startswith("#"):
            continue
        gleich = zeile.find("=")
        if gleich < 0:
            continue
        links = zeile[:gleich].strip()
        if links.startswith("export ") or links.startswith("export\t"):
            links = links[len("export"):].strip()
        if _NAME.fullmatch(links):
            aus.append((links, nr))
        i = gleich + 1
        while i < len(zeile) and zeile[i] in " \t":
            i += 1
        if i < len(zeile) and zeile[i] in "\"'`" and _quote_ende(zeile, i + 1, zeile[i]) < 0:
            offen = zeile[i]
    return aus, offen is not None


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


def _ls_files(ordner: Path) -> tuple[list[bytes] | None, str]:
    """`git ls-files -z` mit Zeitgrenze; (None, Grund) bei fehlendem git, Zeitueberschreitung oder Fehlercode."""
    try:
        erg = subprocess.run(_git_args("ls-files", "-z"), cwd=ordner, env=_git_env(), capture_output=True,
                             timeout=GIT_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return None, "Zeitüberschreitung"
    except (OSError, subprocess.SubprocessError) as fehler:
        return None, type(fehler).__name__
    if erg.returncode != 0:
        return None, f"Exit {erg.returncode}"
    return [p for p in erg.stdout.split(b"\0") if p], ""


def _dateien(wurzel: Path, aussen: str, notizen: list[str]) -> tuple[list[_Datei], list[str]]:
    """(versionierte Dateien, fehlende Submodule)."""
    sub = sorted(_submodul_pfade(wurzel))
    fehlend = [f"{aussen}/{s}" for s in sub if not (wurzel / s / ".git").exists()]
    for name in fehlend:
        notizen.append(f"{UNLESBAR_PRAEFIX}submodul: {name} nicht ausgecheckt")
    repos = [(aussen, wurzel)] + [(f"{aussen}/{s}", wurzel / s) for s in sub if (wurzel / s / ".git").exists()]
    aus: list[_Datei] = []
    for repo, ordner in repos:
        regeln = _regeln(ordner)
        muster = [glob_regex(m) for m in (*STANDARD_AUSGENOMMEN, *regeln.ausgenommen)]
        liste, grund = _ls_files(ordner)
        if liste is None:
            notizen.append(f"{UNLESBAR_PRAEFIX}dateien: git ls-files fehlgeschlagen für {repo} ({grund}): "
                           "nicht inventarisiert")
            continue
        for roh in liste:
            pfad = roh.decode("utf-8", errors="surrogateescape").replace("\\", "/")
            if glob_passt(pfad, muster):
                continue
            aus.append(_Datei(repo, pfad, ordner / pfad, regeln.max_dateigroesse_kb * 1024))
    return sorted(aus, key=lambda d: (d.repo, d.pfad)), fehlend


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
    if sprache_fuer(pfad) == "dotenv" or basis == ".env" or basis.endswith(".env") or ist_env_beispiel(basis):
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
    """Schluessel/Wert-Paare eines Mapping-Knotens wie YAML sie meint: explizite Schluessel gewinnen vor `<<`,
    unter mehreren Merge-Quellen die fruehere (Fix-Runde 1)."""
    if not isinstance(knoten, yaml.MappingNode) or tiefe > 10:
        return []
    explizit: list[tuple[yaml.Node, yaml.Node]] = []
    quellen: list[yaml.Node] = []
    for k, v in knoten.value:
        if isinstance(k, yaml.ScalarNode) and k.tag == "tag:yaml.org,2002:merge":
            quellen += v.value if isinstance(v, yaml.SequenceNode) else [v]
        else:
            explizit.append((k, v))
    gesehen = {k.value for k, _ in explizit if isinstance(k, yaml.ScalarNode)}
    aus = list(explizit)
    for q in quellen:
        for k, v in _items(q, tiefe + 1):
            if isinstance(k, yaml.ScalarNode):
                if k.value in gesehen:
                    continue
                gesehen.add(k.value)
            aus.append((k, v))
    return aus


def _holen(knoten: yaml.Node | None, schluessel: str) -> yaml.Node | None:
    for k, v in _items(knoten):
        if isinstance(k, yaml.ScalarNode) and k.value == schluessel:
            return v
    return None


def _env_schluessel(env: yaml.Node | None) -> list[tuple[str, int, yaml.ScalarNode | None]]:
    """(Name, Zeile, Listeneintrag oder None) der `environment`-Schluessel."""
    aus: list[tuple[str, int, yaml.ScalarNode | None]] = []
    if isinstance(env, yaml.MappingNode):
        for k, _ in _items(env):
            if isinstance(k, yaml.ScalarNode) and _NAME.fullmatch(k.value.strip()):
                aus.append((k.value.strip(), k.start_mark.line + 1, None))
    elif isinstance(env, yaml.SequenceNode):
        for e in env.value:
            if isinstance(e, yaml.ScalarNode):
                gleich = e.value.find("=")
                links = (e.value if gleich < 0 else e.value[:gleich]).strip()
                if _NAME.fullmatch(links):
                    aus.append((links, e.start_mark.line + 1, e))
    return aus


def _wert_skalare(knoten: yaml.Node, gesehen: set[int], tiefe: int = 0) -> Iterator[yaml.ScalarNode]:
    """Alle Skalare in Wert-Position (keine Mapping-Schluessel); Aliase nur einmal."""
    if id(knoten) in gesehen or tiefe > 50:
        return
    gesehen.add(id(knoten))
    if isinstance(knoten, yaml.ScalarNode):
        yield knoten
    elif isinstance(knoten, yaml.SequenceNode):
        for v in knoten.value:
            yield from _wert_skalare(v, gesehen, tiefe + 1)
    elif isinstance(knoten, yaml.MappingNode):
        for _, v in knoten.value:
            yield from _wert_skalare(v, gesehen, tiefe + 1)


def _compose(text: str, ort: str, setzen: dict[str, set[str]], lesen: dict[str, set[str]]) -> None:
    for dok in yaml.compose_all(text, Loader=yaml.SafeLoader):
        if dok is None:
            continue
        listeneintraege: set[int] = set()
        for _, dienst in _items(_holen(dok, "services")):
            for name, zeile, eintrag in _env_schluessel(_holen(dienst, "environment")):
                setzen[name].add(f"{ort}:{zeile}")
                if eintrag is not None:
                    listeneintraege.add(id(eintrag))
        for s in _wert_skalare(dok, set()):
            basis = s.start_mark.line + 1 + (1 if s.style in ("|", ">") else 0)
            for m in _COMPOSE_KLAMMER.finditer(s.value):
                lesen[m.group(1)].add(f"{ort}:{basis + s.value.count(chr(10), 0, m.start())}")
            kandidaten = [s.value.strip()]
            if id(s) in listeneintraege and "=" in s.value:
                kandidaten.append(s.value[s.value.find("=") + 1:].strip())
            for kandidat in kandidaten:
                m = _COMPOSE_GANZ.fullmatch(kandidat)
                if m:
                    lesen[m.group(1)].add(f"{ort}:{basis}")


def _ps_zeilen(text: str) -> list[str]:
    """Zeilen ohne `#`-Zeilenkommentare und `<# … #>`-Blockkommentare (Strings bleiben, `#` darin zaehlt nicht)."""
    aus: list[str] = []
    block = False
    quote: str | None = None
    for zeile in text.split("\n"):
        teile: list[str] = []
        i = 0
        while i < len(zeile):
            c = zeile[i]
            if block:
                if zeile.startswith("#>", i):
                    block = False
                    i += 2
                else:
                    i += 1
                continue
            if quote is not None:
                teile.append(c)
                if c == "`" and quote == '"' and i + 1 < len(zeile):
                    teile.append(zeile[i + 1])
                    i += 2
                    continue
                if c == quote:
                    quote = None
                i += 1
                continue
            if zeile.startswith("<#", i):
                block = True
                i += 2
                continue
            if c == "#":
                break
            if c in "'\"":
                quote = c
            teile.append(c)
            i += 1
        aus.append("".join(teile))
    return aus


def _powershell(text: str, ort: str, setzen: dict[str, set[str]], lesen: dict[str, set[str]]) -> None:
    for nr, zeile in enumerate(_ps_zeilen(text), start=1):
        for m in _PS_RE.finditer(zeile):
            ziel = setzen if _PS_ZUWEISUNG.match(zeile, m.end()) else lesen
            ziel[m.group(1)].add(f"{ort}:{nr}")
        for muster in (_PS_SETENV, _PS_ITEM):
            for m in muster.finditer(zeile):
                setzen[m.group(1)].add(f"{ort}:{nr}")


def _ports(text: str, ort: str) -> list[tuple[int, str, str]]:
    """(Port, Dienst, Fundstelle) aus `dienste: {<name>: {port: <int>|[<int>, ...]|null}}`; doppelte Dienst-
    Schluessel bleiben erhalten (Knotenbaum statt dict)."""
    aus: list[tuple[int, str, str]] = []
    for dok in yaml.compose_all(text, Loader=yaml.SafeLoader):
        dienste = _holen(dok, "dienste")
        if not isinstance(dienste, yaml.MappingNode):
            continue
        for k, v in dienste.value:  # roh: doppelte Schluessel sollen sichtbar bleiben
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


def _repo_pfad(ort: str, repos: Sequence[str]) -> tuple[str, str] | None:
    """`<repo>/<pfad>` -> (Repo, Pfad) mit dem laengsten passenden Repo-Praefix; None ohne Treffer."""
    treffer = [r for r in repos if ort.startswith(r + "/")]
    if not treffer:
        return None
    repo = max(treffer, key=len)
    return repo, ort[len(repo) + 1:]


def produktiv(ort: str, fremd: Callable[[str, str], bool] | None = None, repos: Sequence[str] = ()) -> bool:
    """S5-13: Lesestelle in produktivem Code (nicht Test/Beispiel/Vendor, kein `site-packages`, kein Fremdcode).
    `ort` ist `<repo>/<pfad>[:zeile]`; `repos` sind die bekannten Repo-Namen fuer die Zerlegung."""
    pfad = _datei_von(ort)
    if fremd is not None:
        teile = _repo_pfad(pfad, repos)
        if teile is not None and fremd(*teile):
            return False
    return not ist_nicht_produktiv(pfad) and not any(t in _FREMD_SEGMENTE for t in pfad.split("/")[:-1])


def _sortierschluessel_name(n: Name) -> tuple[int, str]:
    return INVENTAR_ARTEN.index(n.art), (n.name.zfill(6) if n.art == "port" else n.name)


def _env_eintraege(setzen: dict[str, set[str]], lesen: dict[str, set[str]], *,
                   mit_tot: bool, fremd: Callable[[str, str], bool] | None = None,
                   repos: Sequence[str] = ()) -> list[tuple[Name, tuple[str, ...]]]:
    """(Eintrag, Schreibweisen): Namen ohne Gross-/Kleinschreibung zusammengefuehrt (Windows-Env), angezeigt gross;
    die Schreibweisen braucht nur der Schluesselfilter (ein `ghp_…` waere gross geschrieben unsichtbar fuer ihn)."""
    gruppen: dict[str, set[str]] = defaultdict(set)
    for schreibweise in set(setzen) | set(lesen):
        gruppen[schreibweise.upper()].add(schreibweise)
    aus: list[tuple[Name, tuple[str, ...]]] = []
    for name in sorted(gruppen):
        schreib = tuple(sorted(gruppen[name]))
        gesetzt = sorted({o for s in schreib for o in setzen.get(s, ())})
        gelesen = sorted({o for s in schreib for o in lesen.get(s, ())})
        befund: list[str] = []
        if len({_datei_von(o) for o in gesetzt}) >= 2:
            befund.append("doppelt")
        if gesetzt and not gelesen and mit_tot:
            befund.append("tot")
        if not gesetzt and name not in SYSTEM_NAMEN and any(produktiv(o, fremd, repos) for o in gelesen):
            befund.append("ungesetzt")
        art: InventarArt = "key" if name.endswith(KEY_ENDEN) else "env"
        aus.append((Name(art, name, gelesen, gesetzt, sorted(befund)), schreib))
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


_DATEI_TEXT = {"python": ".py-Datei(en)", "compose": "Compose-Datei(en)", "powershell": "PowerShell-Datei(en)",
               "env-beispiel": "Env-Beispieldatei(en)", "dienste": "Dienst-Register"}


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

    regeln_aussen = _regeln(wurzel)
    dienste_konfiguriert = regeln_aussen.dienste is not None
    dienste_pfad = (regeln_aussen.dienste or DIENSTE_STANDARD).replace("\\", "/")
    dateien, fehlende = _dateien(wurzel, aussen, notizen)
    repo_namen = [aussen, *(f"{aussen}/{s}" for s in _submodul_pfade(wurzel))]
    fremd = fremd_pruefer(regeln_aussen.fremdcode, aussen) if regeln_aussen.fremdcode else None
    ports: list[tuple[int, str, str]] = []
    clients: list[LlmClient] = []
    nicht_lesbar: Counter[str] = Counter()
    gesperrt = 0
    offen_am_ende = 0
    dienste_da = False
    dienste_unlesbar = False
    for d in dateien:
        basis = posixpath.basename(d.pfad)
        ort = f"{d.repo}/{d.pfad}"
        if ist_env_beispiel(basis):
            art = "env-beispiel"
        elif d.repo == aussen and d.pfad == dienste_pfad:
            art = "dienste"
        elif _ist_compose(basis) and not _compose_verboten(d.pfad):
            art = "compose"
        elif basis.lower().endswith(_PS_ENDUNGEN):
            art = "powershell"
        elif basis.endswith(".py"):
            art = "python"
        else:
            continue
        if art == "dienste":
            dienste_da = True
        if not darf_oeffnen(basis):
            gesperrt += 1
            dienste_unlesbar = dienste_unlesbar or art == "dienste"
            continue
        quelle = _lesen(d.voll, d.max_bytes)
        if quelle is None:
            if d.voll.is_dir():  # Gitlink eines Submoduls
                continue
            nicht_lesbar[art] += 1
            dienste_unlesbar = dienste_unlesbar or art == "dienste"
            continue
        try:
            if art == "env-beispiel":
                gefunden_env, offen = env_namen(quelle)
                offen_am_ende += offen
                for name, zeile in gefunden_env:
                    setzen[name].add(f"{ort}:{zeile}")
            elif art == "compose":
                _compose(_text(quelle), ort, setzen, lesen)
            elif art == "powershell":
                _powershell(_text(quelle), ort, setzen, lesen)
            elif art == "dienste":
                ports += _ports(_text(quelle), ort)
            else:
                gefunden = llm_clients(quelle, d.repo, d.pfad)
                if gefunden is None:
                    nicht_lesbar[art] += 1
                else:
                    clients += gefunden
        except (yaml.YAMLError, RecursionError, ValueError):
            nicht_lesbar[art] += 1
            dienste_unlesbar = dienste_unlesbar or art == "dienste"
        del quelle
    if dienste_unlesbar:
        notizen.append(f"{UNLESBAR_PRAEFIX}dienste: {dienste_pfad} nicht lesbar (zu groß, gesperrt oder kaputt): "
                       "keine Ports")
    elif not dienste_da:
        if dienste_konfiguriert:
            notizen.append(f"{UNLESBAR_PRAEFIX}dienste: {dienste_pfad} konfiguriert, aber nicht versioniert: "
                           "keine Ports")
        else:
            notizen.append(f"kein Dienst-Register ({dienste_pfad} nicht versioniert): keine Ports")
    for art, n in sorted(nicht_lesbar.items()):
        notizen.append(f"{n} {_DATEI_TEXT[art]} nicht lesbar oder nicht parsebar, übersprungen")
    if gesperrt:
        notizen.append(f"{gesperrt} Datei(en) nach der Geheimnis-Regel nicht geöffnet")
    if offen_am_ende:
        notizen.append(f"{offen_am_ende} Env-Beispieldatei(en) mit offenem Anführungszeichen am Ende: "
                       "Rest übersprungen")

    roh = (_env_eintraege(setzen, lesen, mit_tot=graph_ok, fremd=fremd, repos=repo_namen) + [(p, ()) for p in _port_eintraege(ports)]
           + [(c, ()) for c in _client_eintraege(
               sorted(clients, key=lambda c: (c.anbieter, c.fundstelle, c.modell or "")))])
    namen, verdacht = [], Counter[tuple[str, str]]()
    for n, schreib in roh:
        muster = pruefen("\n".join([n.name, *schreib, *n.gelesen, *n.gesetzt, *n.befund]))
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
        fehlende_submodule=sorted(fehlende),
        nicht_geoeffnet=gesperrt,
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
# Finale Review-Fixes: Grenze fuer den fertigen konfig-Issue-Text (GitHub nimmt hoechstens 65.536 Zeichen) und
# Reserve fuer alles ausser Inventar-Text und Belegen (Schluessel- und Repo-Marke ohne Repo-Namen, Schwere,
# Belege-Ueberschrift, Hinweis: zusammen ~230 Zeichen; der Repo-Name wird extra abgezogen)
KONFIG_ISSUE_MAX = 64000
KONFIG_ISSUE_RESERVE = 1000


def _kopf(inv: Inventar) -> str:
    arten, befunde = _zaehlen(inv)
    plan = konsolidierungsplan(inv)
    plan_text = "\n".join(f"{i}. {p}" for i, p in enumerate(plan, start=1)) or "Keine Konsolidierung nötig."
    kopf = (f"**Konfig-Inventar {inv.code_repo}**: nur Namen, nie Werte (D6).\n\n"
            "Namen: " + ", ".join(f"{a} {arten[a]}" for a in INVENTAR_ARTEN)
            + ". Befunde: " + ", ".join(f"{b} {befunde[b]}" for b in BEFUNDE)
            + f". Übersprungene Dateien: {inv.uebersprungen}.\n\n")
    if inv.fehlende_submodule:
        kopf += (f"**Unvollständig:** {len(inv.fehlende_submodule)} Submodule nicht ausgecheckt, nicht inventarisiert: "
                 + ", ".join(f"`{_zelle(s)}`" for s in inv.fehlende_submodule) + ".\n\n")
    kopf += f"### Konsolidierungsplan\n\n{plan_text}\n\n"
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


def inventar_fund(inv: Inventar, *, max_zeichen: int = 60000, max_issue: int = KONFIG_ISSUE_MAX) -> Fund:
    """Der eine `konfig`-Fund (Schwere mittel, Anker `konfig-inventar:<aussen>`, R-f; Titel ohne Zahlen, weil das
    Einzel-Issue (S5-11) nur seinen Text bearbeitet). Text = Kopf mit Plan und Tabelle "Name -> wo gelesen, wo
    gesetzt, Befund", gekappt wie in Task 3; `len(text) <= max_zeichen` gilt immer.

    Finale Review-Fixes: der fertige Issue-Text (Marken, Schwere, Belege, Hinweis) bleibt <= `max_issue`; dafuer
    zieht das Text-Budget den Belege-Block (bis 50 Belege) und eine feste Reserve fuer den Rest ab."""
    belege = tuple(sorted({_datei_von(o) for n in inv.namen if n.art != "llm_client" for o in n.gesetzt})[:50])
    belege_block = len("\n".join(f"- `{b}`" for b in belege))
    max_zeichen = max(0, min(max_zeichen, max_issue - belege_block - KONFIG_ISSUE_RESERVE - len(inv.code_repo)))
    kopf = _kopf(inv)
    zeilen = [_zeile(n, False) for n in _text_reihenfolge(inv.namen)]
    budget = max_zeichen - len(kopf) - len(_TABELLENKOPF) - KAPPEN_RESERVE - 1
    behalten, k = kappen(zeilen, budget)
    text = kopf + _TABELLENKOPF + "".join("\n" + z for z in behalten)
    if k:
        text += "\n\n" + kappen_zeile(k)
    text += "\n"
    if len(text) > max_zeichen:  # schon der Kopf passt nicht: hart kuerzen, Kappensatz fuer alle Zeilen
        rest = "\n\n" + kappen_zeile(len(zeilen)) + "\n"
        text = (text[: max(0, max_zeichen - len(rest))] + rest)[:max_zeichen]
    return Fund(
        art="konfig", schwere="mittel", code_repo=inv.code_repo, titel="Konfig-Inventar",
        text=text, belege=belege, anker=f"konfig-inventar:{inv.code_repo}")
