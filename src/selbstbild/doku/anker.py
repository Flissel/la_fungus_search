"""Belege: vier Ankerarten (symbol, sql, yaml_key, datei) mit Digest-Pruefung.

Pfadkonvention: `Beleg.file` ist IMMER repo-relativ (POSIX, relativ zu `konfig.repo`).
Manifest-Dokumente sind dagegen korpus-relativ; `repo_relativ` uebersetzt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tomllib
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import yaml

from . import korpus

if TYPE_CHECKING:  # zur Laufzeit lazy (Submodul)
    from ._manifest_gate2 import Manifest

Stand = Mapping[tuple[str, str, str], str]


@dataclass(frozen=True)
class Beleg:
    art: str  # symbol | sql | yaml_key | datei
    file: str  # repo-relativ, POSIX
    anker: str
    digest: str
    commit: str
    zeile: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "Beleg":
        zeile = d.get("zeile")
        return cls(
            art=str(d["art"]), file=str(d["file"]), anker=str(d["anker"]),
            digest=str(d["digest"]), commit=str(d["commit"]),
            zeile=int(zeile) if isinstance(zeile, int) else None,
        )


class SqlDefinitionLike(Protocol):
    name: str
    datei: str
    digest: str


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class UnsichererPfad(ValueError):
    """Final-Fix 22: ein Belegpfad mit `..`, absolut, mit Laufwerk/UNC oder ueber einen Link ausserhalb des Repos."""


_LAUFWERK = re.compile(r"^[A-Za-z]:")


def pfad_sicher(file: str) -> bool:
    """Repo-relativer Pfad ohne `..`-Segment, nicht absolut, ohne Laufwerk/UNC (reine Textpruefung, liest nichts).
    Belegpfade kommen aus KETTEN.md und sind nicht vertrauenswuerdig."""
    if not file or "\0" in file or file.startswith(("/", "\\")) or _LAUFWERK.match(file):
        return False
    return ".." not in file.replace("\\", "/").split("/")


def repo_pfad(repo: Path, file: str) -> Path:
    """`repo / file`, wenn `file` sicher ist und auch aufgeloest (Links) im Repo bleibt; sonst `UnsichererPfad`
    (`_im_repo`-artig, vor jedem Lesen)."""
    if not pfad_sicher(file):
        raise UnsichererPfad(file)
    ziel = Path(repo) / file
    try:
        ziel.resolve().relative_to(Path(repo).resolve())
    except ValueError as exc:
        raise UnsichererPfad(file) from exc
    return ziel


def repo_relativ(manifest: "Manifest", repo: Path, doc_file: str) -> str:
    """Korpus-relativen Dokumentpfad in einen repo-relativen POSIX-Pfad umrechnen."""
    absolut = Path(manifest.corpus_root) / doc_file
    try:
        return absolut.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        try:
            return absolut.relative_to(repo).as_posix()
        except ValueError:
            return absolut.as_posix()


def _korpus_relativ(manifest: "Manifest", repo: Path, file: str) -> str | None:
    wurzel = Path(manifest.corpus_root)
    if not wurzel.is_absolute():
        wurzel = repo / wurzel
    try:
        return (repo / file).resolve().relative_to(wurzel.resolve()).as_posix()
    except ValueError:
        return None


def stand_aus_manifest(m: "Manifest", repo: Path) -> dict[tuple[str, str, str], str]:
    """("symbol", repo-relative Datei, document_id) -> sha256(source).

    Mehrfach vergebene document_ids sind mehrdeutig und fehlen im Stand.
    """
    zaehler = Counter(d.document_id for d in m.documents)
    return {
        ("symbol", repo_relativ(m, repo, d.file), d.document_id): _sha(d.source)
        for d in m.documents
        if zaehler[d.document_id] == 1
    }


def stand_aus_sql(definitionen: Mapping[str, SqlDefinitionLike]) -> dict[tuple[str, str, str], str]:
    """("sql", datei, "schema.name") -> Digest der CREATE-Anweisung."""
    return {("sql", d.datei, d.name): d.digest for d in definitionen.values()}


Token = str | int | tuple[str, str]  # Schluessel | Position | Identitaet (schluessel, wert)
IDENTITAETS_SCHLUESSEL = ("capability", "name", "id", "key", "tool")

_POSITION = re.compile(r"\[(\d+)\]")
_IDENTITAET = re.compile(r'\[([^\]\["=.]+)=')
_NACKT = re.compile(r'[^.\[\]"]+')
_JSON = json.JSONDecoder()


def _json_string(text: str, i: int) -> tuple[str, int] | None:
    try:
        wert, j = _JSON.raw_decode(text, i)
    except ValueError:
        return None
    return (wert, j) if isinstance(wert, str) else None


def _tokens(pfad: str) -> list[Token] | None:
    """Pfad parsen: `a.b[2].c`, `[capability=x].targets`, `events["agentfarm.status"]`, `[name="a]b"]`."""
    if not pfad:
        return None
    tokens: list[Token] = []
    i, n = 0, len(pfad)
    while i < n:
        if pfad[i] == "[":
            if pfad.startswith('["', i):
                gelesen = _json_string(pfad, i + 1)
                if gelesen is None or not pfad.startswith("]", gelesen[1]):
                    return None
                tokens.append(gelesen[0])
                i = gelesen[1] + 1
                continue
            m = _POSITION.match(pfad, i)
            if m:
                tokens.append(int(m.group(1)))
                i = m.end()
                continue
            m = _IDENTITAET.match(pfad, i)
            if not m:
                return None
            j = m.end()
            if pfad.startswith('"', j):
                gelesen = _json_string(pfad, j)
                if gelesen is None:
                    return None
                wert, j = gelesen
            else:
                e = pfad.find("]", j)
                if e < 0:
                    return None
                wert, j = pfad[j:e], e
            if not pfad.startswith("]", j):
                return None
            tokens.append((m.group(1), wert))
            i = j + 1
            continue
        if pfad[i] == ".":
            if not tokens:
                return None
            i += 1
        elif tokens:
            return None
        m = _NACKT.match(pfad, i)
        if not m:
            return None
        tokens.append(m.group(0))
        i = m.end()
    return tokens


def pfad_text(tokens: list[Token]) -> str:
    """Gegenstueck zu `_tokens`: Schluessel mit `.`/`[`/`]`/`"` werden als `["..."]` geschrieben."""
    teile: list[str] = []
    for t in tokens:
        if isinstance(t, int):
            teile.append(f"[{t}]")
        elif isinstance(t, tuple):
            k, v = t
            wert = v if v and not set(v) & set('[]"') else json.dumps(v, ensure_ascii=False)
            teile.append(f"[{k}={wert}]")
        elif t and not set(t) & set('.[]"'):
            teile.append(f".{t}" if teile else t)
        else:
            teile.append(f"[{json.dumps(t, ensure_ascii=False)}]")
    return "".join(teile)


def _json_schluessel(k: object) -> str:
    """Text eines Zuordnungsschluessels wie bei `json.dumps` (Zahl -> "1", True -> "true", None -> "null")."""
    if isinstance(k, str):
        return k
    if k is None or isinstance(k, (bool, int, float)):
        return json.dumps(k)
    return str(k)


def _schluessel_als_text(wert: object) -> object:
    """Final-Fix C2: Zuordnungen mit gemischten Schluesseltypen (`{1: eins, a: zwei}` aus YAML) rekursiv auf
    Text-Schluessel bringen, damit `sort_keys` nicht mit TypeError abbricht. Deterministisch: Text-Schluessel bleiben,
    andere werden wie bei json geschrieben; kollidiert das mit einem vorhandenen Schluessel, bekommt er das
    Typpraefix `<typ>:`."""
    if isinstance(wert, dict):
        aus: dict[str, object] = {}
        for k, v in wert.items():
            if isinstance(k, str):
                aus[k] = _schluessel_als_text(v)
        for k, v in wert.items():
            if not isinstance(k, str):
                text = _json_schluessel(k)
                if text in aus:
                    text = f"{type(k).__name__}:{text}"
                aus[text] = _schluessel_als_text(v)
        return aus
    if isinstance(wert, (list, tuple)):
        return [_schluessel_als_text(v) for v in wert]
    return wert


def _json_text(wert: object) -> str:
    """Kanonischer JSON-Text eines Knotenwerts. Fuer einheitliche Schluessel byte-gleich mit dem bisherigen
    `json.dumps(sort_keys=True)` (keine neuen Digests); nur wenn das wirft, mit `_schluessel_als_text`."""
    try:
        return json.dumps(wert, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return json.dumps(_schluessel_als_text(wert), sort_keys=True, ensure_ascii=False, default=str)


def _digest_json(wert: object) -> str:
    return _sha(_json_text(wert))


def yaml_identitaet(knoten: yaml.Node, schluessel: str) -> str | None:
    """Skalarwert von `schluessel` in einem Listeneintrag (Mapping), sonst None."""
    if isinstance(knoten, yaml.MappingNode):
        for k, v in knoten.value:
            if isinstance(k, yaml.ScalarNode) and k.value == schluessel and isinstance(v, yaml.ScalarNode):
                return str(v.value)
    return None


def toml_identitaet(eintrag: object, schluessel: str) -> str | None:
    if isinstance(eintrag, dict):
        wert = eintrag.get(schluessel)
        if isinstance(wert, str):
            return wert
        if isinstance(wert, int) and not isinstance(wert, bool):
            return str(wert)
    return None


def _yaml_knoten(wurzel: yaml.Node, tokens: list[Token]) -> yaml.Node | None:
    knoten: yaml.Node | None = wurzel
    for t in tokens:
        if isinstance(knoten, yaml.MappingNode) and isinstance(t, str):
            knoten = next(
                (v for k, v in knoten.value if isinstance(k, yaml.ScalarNode) and k.value == t), None
            )
        elif isinstance(knoten, yaml.SequenceNode) and isinstance(t, int):
            knoten = knoten.value[t] if t < len(knoten.value) else None
        elif isinstance(knoten, yaml.SequenceNode) and isinstance(t, tuple):
            treffer = [v for v in knoten.value if yaml_identitaet(v, t[0]) == t[1]]
            knoten = treffer[0] if len(treffer) == 1 else None
        else:
            return None
        if knoten is None:
            return None
    return knoten


def _toml_knoten(wurzel: object, tokens: list[Token]) -> tuple[bool, object]:
    wert = wurzel
    for t in tokens:
        if isinstance(wert, dict) and isinstance(t, str) and t in wert:
            wert = wert[t]
        elif isinstance(wert, list) and isinstance(t, int) and t < len(wert):
            wert = wert[t]
        elif isinstance(wert, list) and isinstance(t, tuple):
            treffer = [e for e in wert if toml_identitaet(e, t[0]) == t[1]]
            if len(treffer) != 1:
                return False, None
            wert = treffer[0]
        else:
            return False, None
    return True, wert


def yaml_knoten_wert(datei: Path, schluesselpfad: str) -> tuple[bool, object]:
    """(gefunden, Wert) des Knotens `a.b[2].c` in einer YAML- oder TOML-Datei."""
    try:
        text = datei.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False, None
    return yaml_knoten_wert_text(text, datei.suffix, schluesselpfad)


def yaml_knoten_wert_text(text: str, endung: str, schluesselpfad: str) -> tuple[bool, object]:
    """Wie `yaml_knoten_wert`, aber auf einem Text (z. B. einem alten Stand aus `git show`); `endung` `.toml` ->
    TOML, sonst YAML."""
    tokens = _tokens(schluesselpfad)
    if tokens is None:
        return False, None
    try:
        if endung.lower() == ".toml":
            return _toml_knoten(tomllib.loads(text), tokens)
        wurzel = yaml.compose(text, Loader=yaml.SafeLoader)
        if wurzel is None:
            return False, None
        knoten = _yaml_knoten(wurzel, tokens)
        if knoten is None:
            return False, None
        lader = yaml.SafeLoader("")
        try:
            return True, lader.construct_object(knoten, deep=True)
        finally:
            lader.dispose()
    except (yaml.YAMLError, tomllib.TOMLDecodeError, RecursionError):
        return False, None


def yaml_knoten_digest(datei: Path, schluesselpfad: str) -> str | None:
    """Digest des Knotens `a.b[2].c` in einer YAML- oder TOML-Datei; None bei fehlendem Pfad/Fehler."""
    gefunden, wert = yaml_knoten_wert(datei, schluesselpfad)
    return _digest_json(wert) if gefunden else None


def knoten_text(wert: object) -> str:
    """Serialisierter Text eines Knotenwerts (fuer woertliche Tokensuche, ohne Kommentare)."""
    return _json_text(wert)


def datei_digest(datei: Path) -> str:
    return hashlib.sha256(datei.read_bytes()).hexdigest()


def _unter_wurzel(file: str, wurzeln: Sequence[str]) -> bool:
    for w in wurzeln:
        w = w.strip("/")
        if w and (file == w or file.startswith(w + "/")):
            return True
    return False


def beleg_pruefen(beleg: Beleg, stand: Stand, repo: Path, space_wurzeln: Sequence[str] = ()) -> str:
    """beleg_ok | beleg_geaendert | beleg_verschwunden. yaml_key/datei werden frisch gerechnet.

    `symbol`-Belege, die im Stand fehlen, werden nur ausserhalb der Space-Wurzeln direkt aus der Datei
    nachgerechnet (Ruling 29); im Space heisst ein fehlender Schluessel `beleg_verschwunden`.
    Ein unsicherer Pfad (`..`, absolut, Link nach aussen) wirft `UnsichererPfad`, bevor etwas gelesen wird.
    """
    repo_pfad(repo, beleg.file)
    if beleg.art in ("symbol", "sql"):
        aktuell = stand.get((beleg.art, beleg.file, beleg.anker))
        if aktuell is None and beleg.art == "symbol" and not _unter_wurzel(beleg.file, space_wurzeln):
            aktuell = symbol_digest_aus_datei(repo, beleg.file, beleg.anker)
    elif beleg.art == "yaml_key":
        aktuell = yaml_knoten_digest(repo / beleg.file, beleg.anker)
    elif beleg.art == "datei":
        try:
            aktuell = datei_digest(repo / beleg.file)
        except OSError:
            aktuell = None
    else:
        raise ValueError(f"unbekannte Ankerart: {beleg.art}")
    if aktuell is None:
        return "beleg_verschwunden"
    return "beleg_ok" if aktuell == beleg.digest else "beleg_geaendert"


def beleg_aus_zeile(
    file: str, zeile: int, manifest: "Manifest", commit: str, *, repo: Path
) -> Beleg | None:
    """Re-Verankerung: Definition ueber korpus.symbol_an; `file` ist repo-relativ.

    Anker = gepunktete document_id. None, wenn keine Definition umschliesst oder die
    document_id im Manifest mehrfach vorkommt (kein Anker statt eines falschen).
    """
    rel = _korpus_relativ(manifest, repo, file)
    if rel is None:
        return None
    doc = korpus.symbol_an(manifest, rel, zeile)
    if doc is None:
        return None
    if sum(1 for d in manifest.documents if d.document_id == doc.document_id) > 1:
        return None
    return Beleg("symbol", file, doc.document_id, _sha(doc.source), commit, zeile)


def beleg_fuer_datei(file: str, zeile: int | None, repo: Path, commit: str) -> Beleg:
    return Beleg("datei", file, "", datei_digest(repo / file), commit, zeile)


def korpus_relativ(manifest: "Manifest", repo: Path, file: str) -> str | None:
    """Repo-relativen Pfad in den korpus-relativen Pfad des Manifests umrechnen (None ausserhalb)."""
    return _korpus_relativ(manifest, repo, file)


@dataclass(frozen=True)
class Definition:
    """Eine Definition wie ein Manifest-Dokument (gleiche Zeilen- und source-Regeln)."""

    document_id: str
    start_line: int
    end_line: int
    source: str


def modulpfad(file: str) -> str:
    """`scripts/ketten_waechter.py` -> `scripts.ketten_waechter` (wie manifest._module_path, repo-relativ)."""
    return ".".join(Path(file).with_suffix("").parts)


def ast_definitionen(datei: Path, modul: str) -> list[Definition] | None:
    """Funktionen/Klassen (Methoden in Klassen) einer Datei wie `manifest._collect`; None, wenn unlesbar."""
    try:
        text = Path(datei).read_text(encoding="utf-8", errors="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
            baum = ast.parse(text)
    except (OSError, SyntaxError, ValueError, RecursionError):
        return None
    zeilen = text.splitlines()
    ergebnis: list[Definition] = []

    def sammeln(knoten: ast.AST, praefix: str) -> None:
        for kind in ast.iter_child_nodes(knoten):
            if not isinstance(kind, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            did = f"{praefix}.{kind.name}"
            ende = getattr(kind, "end_lineno", None) or kind.lineno
            ergebnis.append(Definition(did, kind.lineno, ende, "\n".join(zeilen[kind.lineno - 1:ende])))
            if isinstance(kind, ast.ClassDef):
                sammeln(kind, did)

    sammeln(baum, modul)
    return ergebnis


def symbol_digest_aus_datei(repo: Path, file: str, anker: str) -> str | None:
    """Digest eines Symbol-Ankers `<modulpfad>.<qualname>` direkt aus der Datei; None, wenn weg/mehrdeutig."""
    modul = modulpfad(file)
    if not anker.startswith(modul + "."):
        return None
    definitionen = ast_definitionen(repo / file, modul)
    if definitionen is None:
        return None
    treffer = [d for d in definitionen if d.document_id == anker]
    return _sha(treffer[0].source) if len(treffer) == 1 else None
