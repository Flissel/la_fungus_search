"""Konfiguration des Doku-Loops laden und validieren: `config/doku_loop.yaml` (`laden`) oder der Abschnitt
`doku:` der Regeldatei `selbstbild.yaml` (`aus_regeln`, Stufe 4)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _standard_pfad() -> Path:
    """Ohne --konfig: `config/doku_loop.yaml` unter `SELBSTBILD_VIBEMIND`, sonst unter dem Arbeitsordner."""
    wurzel = os.environ.get("SELBSTBILD_VIBEMIND")
    return (Path(wurzel) if wurzel else Path.cwd()) / "config" / "doku_loop.yaml"


STANDARD_AUSSCHLUSS_NAMEN = (
    ".git", ".venv", "venv", "env", "node_modules", "site-packages", "__pycache__", "dist", "build",
    ".fungus_cache", ".mypy_cache", ".pytest_cache", ".tox", ".worktrees",
)
MCP_MUSTER = frozenset({"decorator", "werkzeuge"})


class KonfigFehler(ValueError):
    """Die Konfiguration ist unvollstaendig oder widerspruechlich."""


@dataclass(frozen=True)
class SpaceKonfig:
    name: str
    wurzel: str  # repo-relativ, POSIX
    mcp_muster: str


@dataclass(frozen=True)
class Konfig:
    repo: Path
    spaces: dict[str, SpaceKonfig]
    ausschluss_namen: frozenset[str]
    ausschluss_pfade: frozenset[str]
    ketten_md: str
    weitere_doku: tuple[str, ...]
    compose_live: tuple[str, ...]
    sql_wurzeln: tuple[str, ...]
    dienste_yaml: str
    waechter_zustandsdatei: str
    waechter_zuordnung: dict[str, tuple[str, ...]]
    tick_minuten: int
    marker: Path
    agent: str
    judge_modell: str
    claude_bin: Path
    rowboat_wissen: Path
    rowboat_notizen: str
    rowboat_urteile: str
    # Final-Fix-Welle (Ruling 57), alle optional:
    zweig: str | None = None  # gesetzt -> geschrieben wird nur auf diesem Branch (sonst `falscher_zweig`)
    rowboat_notizen_aktiv: bool = True  # aus -> keine Rowboat-Notizen (Urteile werden trotzdem gelesen)
    tick_deadline_min: int = 10  # nach so vielen Minuten Laufzeit keine neuen Judge-Fragen mehr
    waechter_max_alter_min: int = 60  # aeltere Waechter-Zustandsdatei -> alle Messungen `keine_probe`
    # Selbstbild Stufe 4 (D4): Aussenabhaengigkeiten sind Konfiguration, beide repo-relativ
    budget_modul: str = "scripts/claude_budget.py"  # per Dateipfad geladen; fehlt es, gibt es kein LLM
    torwaechter: tuple[str, ...] = (".venv/Scripts/python.exe", "scripts/docs_drift_waechter.py")  # () = keiner
    # Selbstbild Stufe 4 (Plan P1): hier nur validiert, das Verhalten kommt in Task 4 bis 9
    doku_ziele: tuple[str, ...] = ()
    doku_nur_lesen: tuple[str, ...] = ()
    ereignisse: dict[str, Any] = field(default_factory=dict)
    space_dokumente: dict[str, Any] = field(default_factory=dict)
    rowboat_seiten_aktiv: bool = False
    rowboat_seiten_je_lauf: int = 3
    erstbefuellung: dict[str, Any] = field(default_factory=lambda: {"aktiv": False})


def _pfad(wert: str) -> Path:
    return Path(os.path.expandvars(wert))


def _brauche(roh: dict[str, Any], schluessel: str) -> Any:
    if schluessel not in roh:
        raise KonfigFehler(f"Schluessel fehlt: {schluessel}")
    return roh[schluessel]


def _ganzzahl(roh: dict[str, Any], schluessel: str, vorgabe: int) -> int:
    wert = roh.get(schluessel, vorgabe)
    if isinstance(wert, bool) or not isinstance(wert, int) or wert <= 0:
        raise KonfigFehler(f"{schluessel} muss eine positive ganze Zahl sein")
    return wert


def _liste(roh: dict[str, Any], schluessel: str) -> tuple[str, ...]:
    wert = roh.get(schluessel)
    if wert is None:
        return ()
    if not isinstance(wert, list) or not all(isinstance(x, str) for x in wert):
        raise KonfigFehler(f"{schluessel} muss eine Liste von Zeichenketten sein")
    return tuple(wert)


def _zuordnung(roh: dict[str, Any], schluessel: str, vorgabe: dict[str, Any]) -> dict[str, Any]:
    wert = roh.get(schluessel)
    if wert is None:
        return dict(vorgabe)
    if not isinstance(wert, dict):
        raise KonfigFehler(f"{schluessel} muss eine Zuordnung sein")
    return dict(wert)


def _wahrheit(roh: dict[str, Any], schluessel: str, vorgabe: bool) -> bool:
    wert = roh.get(schluessel, vorgabe)
    if not isinstance(wert, bool):
        raise KonfigFehler(f"{schluessel} muss true oder false sein")
    return wert


def _torwaechter(wert: Any) -> tuple[str, ...]:
    """`[]` = kein Torwaechter, sonst mindestens zwei Zeichenketten (Interpreter, Skript); alles andere ist ein Fehler."""
    if not isinstance(wert, list) or not all(isinstance(x, str) for x in wert) or len(wert) == 1:
        raise KonfigFehler("torwaechter muss [] oder eine Liste aus mindestens zwei Zeichenketten sein")
    return tuple(wert)


def _aussen(roh: dict[str, Any]) -> dict[str, Any]:
    """Optionale Schluessel der Stufe 4; fehlen sie, gelten die Vorgaben von `Konfig`."""
    aus: dict[str, Any] = {}
    if "budget_modul" in roh:
        if not isinstance(roh["budget_modul"], str) or not roh["budget_modul"].strip():
            raise KonfigFehler("budget_modul muss ein Pfad sein")
        aus["budget_modul"] = roh["budget_modul"]
    if "torwaechter" in roh:
        aus["torwaechter"] = _torwaechter(roh["torwaechter"])
    for k in ("doku_ziele", "doku_nur_lesen"):
        aus[k] = _liste(roh, k)
    aus["ereignisse"] = _zuordnung(roh, "ereignisse", {})
    aus["space_dokumente"] = _zuordnung(roh, "space_dokumente", {})
    aus["rowboat_seiten_aktiv"] = _wahrheit(roh, "rowboat_seiten_aktiv", False)
    aus["rowboat_seiten_je_lauf"] = _ganzzahl(roh, "rowboat_seiten_je_lauf", 3)
    erst = _zuordnung(roh, "erstbefuellung", {"aktiv": False})
    if not isinstance(erst.get("aktiv", False), bool):
        raise KonfigFehler("erstbefuellung.aktiv muss true oder false sein")
    erst.setdefault("aktiv", False)
    aus["erstbefuellung"] = erst
    return aus


# Schluessel des Abschnitts `doku:` (alle der alten config/doku_loop.yaml plus die der Stufe 4)
DOKU_SCHLUESSEL = frozenset({
    "spaces", "ausschluss_namen", "ausschluss_pfade", "ketten_md", "weitere_doku", "compose_live", "sql_wurzeln",
    "dienste_yaml", "waechter_zustandsdatei", "waechter_zuordnung", "tick_minuten", "zweig", "marker", "agent",
    "judge_modell", "claude_bin", "rowboat_notizen_aktiv", "rowboat_wissen", "rowboat_notizen", "rowboat_urteile",
    "tick_deadline_min", "waechter_max_alter_min",
    "budget_modul", "torwaechter", "doku_ziele", "doku_nur_lesen", "ereignisse", "space_dokumente",
    "rowboat_seiten_aktiv", "rowboat_seiten_je_lauf", "erstbefuellung",
})


def laden(pfad: Path | None = None) -> Konfig:
    """Alte Konfigurationsdatei `config/doku_loop.yaml`; `repo` ist zwei Ebenen ueber der Datei."""
    datei = Path(pfad) if pfad is not None else _standard_pfad()
    roh = yaml.safe_load(datei.read_text(encoding="utf-8"))
    if not isinstance(roh, dict):
        raise KonfigFehler(f"Keine Zuordnung in {datei}")
    return _aus_roh(roh, datei.resolve().parent.parent, relativ=False)


def _aus_roh(roh: dict[str, Any], repo: Path, *, relativ: bool) -> Konfig:
    """`relativ`: Pfade (marker, claude_bin, rowboat_wissen) ohne Laufwerk gelten ab `repo` (Regeldatei-Weg)."""

    def pfad(wert: str) -> Path:
        p = _pfad(wert)
        return repo / p if relativ and not p.is_absolute() else p

    spaces: dict[str, SpaceKonfig] = {}
    for name, eintrag in dict(_brauche(roh, "spaces")).items():
        if not isinstance(eintrag, dict):
            raise KonfigFehler(f"Space {name}: keine Zuordnung")
        for k in ("mcp_muster", "wurzel"):
            if k not in eintrag:
                raise KonfigFehler(f"Space {name}: Schluessel {k} fehlt")
        muster = str(eintrag["mcp_muster"])
        if muster not in MCP_MUSTER:
            raise KonfigFehler(f"Space {name}: unbekanntes mcp_muster {muster!r}")
        spaces[str(name)] = SpaceKonfig(str(name), str(eintrag["wurzel"]).strip("/"), muster)

    tick = int(_brauche(roh, "tick_minuten"))
    if tick <= 0:
        raise KonfigFehler("tick_minuten muss positiv sein")
    zweig = roh.get("zweig")
    zweig = str(zweig).strip() if zweig is not None else ""
    notizen_aktiv = roh.get("rowboat_notizen_aktiv", True)
    if not isinstance(notizen_aktiv, bool):
        raise KonfigFehler("rowboat_notizen_aktiv muss true oder false sein")
    deadline = _ganzzahl(roh, "tick_deadline_min", 10)
    max_alter = _ganzzahl(roh, "waechter_max_alter_min", 60)

    return Konfig(
        repo=repo,
        spaces=spaces,
        ausschluss_namen=frozenset(map(str, _brauche(roh, "ausschluss_namen"))),
        ausschluss_pfade=frozenset(str(p).strip("/") for p in _brauche(roh, "ausschluss_pfade")),
        ketten_md=str(_brauche(roh, "ketten_md")),
        weitere_doku=tuple(map(str, roh.get("weitere_doku") or ())),
        compose_live=tuple(map(str, _brauche(roh, "compose_live"))),
        sql_wurzeln=tuple(map(str, _brauche(roh, "sql_wurzeln"))),
        dienste_yaml=str(_brauche(roh, "dienste_yaml")),
        waechter_zustandsdatei=str(_brauche(roh, "waechter_zustandsdatei")),
        waechter_zuordnung={
            str(k): tuple(map(str, v)) for k, v in dict(_brauche(roh, "waechter_zuordnung")).items()
        },
        tick_minuten=tick,
        marker=pfad(str(_brauche(roh, "marker"))),
        agent=str(_brauche(roh, "agent")),
        judge_modell=str(_brauche(roh, "judge_modell")),
        claude_bin=pfad(str(_brauche(roh, "claude_bin"))),
        rowboat_wissen=pfad(str(_brauche(roh, "rowboat_wissen"))),
        rowboat_notizen=str(_brauche(roh, "rowboat_notizen")),
        rowboat_urteile=str(_brauche(roh, "rowboat_urteile")),
        zweig=zweig or None,
        rowboat_notizen_aktiv=notizen_aktiv,
        tick_deadline_min=deadline,
        waechter_max_alter_min=max_alter,
        **_aussen(roh),
    )


def _standard_konfig(wurzel: Path, regeln: Any) -> Konfig:
    """Universelle Vorgabe ohne Abschnitt `doku:`: Ziele sind die `doku_pfade`, kein KETTEN, keine Ereignisse,
    kein Rowboat, kein LLM (`agent` leer) und kein Torwaechter."""
    from ..hooks import _pfad_kennung, _stand_basis

    lokal = _stand_basis() / f"doku-{_pfad_kennung(wurzel)}"
    return Konfig(
        repo=wurzel, spaces={}, ausschluss_namen=frozenset(STANDARD_AUSSCHLUSS_NAMEN), ausschluss_pfade=frozenset(),
        ketten_md="", weitere_doku=(), compose_live=(), sql_wurzeln=(".",), dienste_yaml=regeln.dienste or "",
        waechter_zustandsdatei="",
        waechter_zuordnung={k: tuple(v) for k, v in regeln.waechter_zuordnung.items()},
        tick_minuten=30, marker=lokal / "anstoss", agent="", judge_modell="sonnet", claude_bin=Path("claude"),
        rowboat_wissen=lokal / "wissen", rowboat_notizen="Doku", rowboat_urteile="Doku-Urteile/Urteile.md",
        zweig=None, rowboat_notizen_aktiv=False, torwaechter=(), doku_ziele=tuple(regeln.doku_pfade),
    )


QUELLE_DOKU = "selbstbild.yaml (Abschnitt doku)"
QUELLE_OHNE_DOKU = "universelle Vorgabe (selbstbild.yaml ohne doku:)"
QUELLE_OHNE_DATEI = "universelle Vorgabe (keine selbstbild.yaml)"


def ist_vorgabe(quelle: str) -> bool:
    return quelle.startswith("universelle Vorgabe")


def aus_regeln_mit_quelle(wurzel: Path) -> tuple[Konfig, str]:
    """Wie `aus_regeln`, dazu die Quelle: `QUELLE_DOKU`, `QUELLE_OHNE_DOKU` oder `QUELLE_OHNE_DATEI`."""
    from ..regeln import REGELDATEI, regeln_laden

    wurzel = Path(wurzel)
    if not wurzel.is_dir():
        raise KonfigFehler("Wurzel ist kein Verzeichnis (Tippfehler im Pfad oder in SELBSTBILD_VIBEMIND?)")
    wurzel = wurzel.resolve()
    datei = wurzel / REGELDATEI
    text = datei.read_text(encoding="utf-8") if datei.is_file() else None
    try:
        regeln = regeln_laden(text)
    except ValueError as fehler:  # pydantic.ValidationError ist ein ValueError
        fehlerliste = getattr(fehler, "errors", None)
        ort = ".".join(map(str, fehlerliste()[0].get("loc", ()))) if callable(fehlerliste) and fehlerliste() else ""
        raise KonfigFehler(f"Regeldatei ungueltig: {type(fehler).__name__}" + (f" bei {ort}" if ort else "")) from None
    if regeln.doku is None:
        return _standard_konfig(wurzel, regeln), (QUELLE_OHNE_DOKU if text is not None else QUELLE_OHNE_DATEI)
    unbekannt = sorted(map(str, set(regeln.doku) - DOKU_SCHLUESSEL))
    if unbekannt:
        raise KonfigFehler(f"Unbekannte Schluessel unter doku: {', '.join(unbekannt)}")
    return _aus_roh(regeln.doku, wurzel, relativ=True), QUELLE_DOKU


def aus_regeln(wurzel: Path) -> Konfig:
    """Konfiguration aus `<wurzel>/selbstbild.yaml`, Abschnitt `doku:` (Arbeitsbaum). `Konfig.repo` ist `wurzel`.

    `wurzel` muss ein Verzeichnis sein (sonst `KonfigFehler`). Ohne Abschnitt oder ohne Datei gilt die universelle
    Vorgabe (der Kern laeuft auch ohne Regeldatei). Unbekannte Schluessel unter `doku:` sind ein `KonfigFehler`.

    Die Vorgaben unterscheiden sich: ohne Abschnitt sind `doku_ziele` die `doku_pfade` der Regeln und
    `torwaechter` ist `()`; mit Abschnitt, aber ohne den Schluessel, ist `doku_ziele` `()` und `torwaechter` der
    alte Standard. Leere Pfade (`ketten_md`, `waechter_zustandsdatei` = `""`) bedeuten: aus."""
    return aus_regeln_mit_quelle(wurzel)[0]
