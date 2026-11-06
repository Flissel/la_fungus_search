"""Fund-Sammler (Stufe 5, Task 2): Doku-Widersprueche, Doppelungen und toter Code.

Liest nur: die Pruefliste des Doku-Loops (`pruefliste_laden`) und `analyse.sqlite` des Selbstbild-Stands
(ueber `analyse.ergebnis.lesen`, das nur mit `mode=ro` oeffnet). Schwere nach D4, deterministisch, kein LLM.

Unlesbare Quellen (Fix-Runde 1): fehlt oder ist eine Quelle kaputt, bekommt der Hinweis das stabile Praefix
`quelle-unlesbar:doku` (Pruefliste) bzw. `quelle-unlesbar:analyse` (Analyse oder fehlerhafte Eintraege darin), gefolgt
von `: `. So kann der Aufrufer "keine Funde" von "Quelle nicht lesbar" unterscheiden (`UNLESBAR_PRAEFIX`).

Anker (Ruling S5-4) sind POSIX und repo-relativ: Doku `art|datei|aussage_id`, Doppelung die sortierten
Mitglieder-IDs mit `|`, toter Code die Symbol-ID.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from ..analyse.doppelt import Doppelung
from ..analyse.ergebnis import Analyse, digest_von, lesen
from ..analyse.lader import ANALYSE_VERSION
from ..analyse.lauf import _stand_meta, fremdcode_am_commit, parameter
from ..doku.konfig import Konfig
from ..doku.pruefliste import pruefliste_laden, pruefliste_pfad
from ..fremdcode import fremd_pruefer, ist_fremd_id
from ..regeln import REGELDATEI, regeln_laden
from ..speicher import pfad_kennung
from ..vermessen import stand_finden
from .goldsatz_tor import ToterCodeTor
from .typen import FremdAusgelassen, Fund, Schwere, fund_schluessel

# Ordnernamen, deren Inhalt "Tests, Beispiele, Vendor" ist (nicht produktiv), plus `fixtures`.
# Zusaetzlich zaehlen Dateinamen `test_*.py`, `*_test.py`, `*.spec.ts`, `*.test.ts` (siehe `ist_nicht_produktiv`).
NICHT_PRODUKTIV: tuple[str, ...] = (
    "test", "tests", "__tests__", "examples", "example", "vendor", "node_modules", "third_party", "fixtures")
_TESTNAME = re.compile(r"^(test_.*\.py|.*_test\.py|.*\.spec\.ts|.*\.test\.ts)$")

_DOKU_MITTEL = ("ereignis_nur_doku", "space_ohne_doku")
_JUDGE = ("judge_uneinig", "judge_unklar")
_ABWEICHEND = "ereignis_quellen_abweichend"
# Text von `ereignisse.abgleich` bei unterschiedlichem Werkzeug (der Praefix `steht nur in` ist der andere Fall).
# Gepinnt durch test_abweichend_text_stimmt_mit_echtem_abgleich.
_WERKZEUG_MARKE = "hat je Quelle ein anderes Werkzeug"
UNLESBAR_PRAEFIX = "quelle-unlesbar:"
_RANG: dict[str, int] = {"hoch": 0, "mittel": 1, "niedrig": 2}


def _posix(pfad: str) -> str:
    return pfad.replace("\\", "/")


def abweichend_ist_werkzeug(text: str) -> bool:
    """`ereignis_quellen_abweichend` mit unterschiedlichem Werkzeug (sonst: nur in einer Quelle)."""
    return _WERKZEUG_MARKE in text


def ist_nicht_produktiv(pfad: str) -> bool:
    teile = [t for t in _posix(pfad).split("/") if t]
    if not teile:
        return False
    return any(t in NICHT_PRODUKTIV for t in teile[:-1]) or _TESTNAME.match(teile[-1]) is not None


def _id_teile(graph_id: str) -> tuple[str, str]:
    """`symbol:<repo>:<pfad>#q` / `datei:<repo>:<pfad>` -> (innerstes Repo-Praefix, Pfad ohne `#q`)."""
    rest = _posix(graph_id).split(":", 1)[1] if ":" in graph_id else _posix(graph_id)
    repo, _, pfad = rest.partition(":")
    return repo, pfad.rpartition("#")[0] if "#" in pfad else pfad


def _gitmodules(ordner: Path) -> list[str]:
    try:
        text = (ordner / ".gitmodules").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [_posix(m.group(1).strip()).strip("/") for m in re.finditer(r"^\s*path\s*=\s*(.+?)\s*$", text, re.M)]


def _submodul_pfade(wurzel: Path, tiefe: int = 4) -> list[str]:
    """Repo-relative Pfade aller Submodule (rekursiv ueber die `.gitmodules` der Submodule), laengste zuerst."""
    aus: list[str] = []

    def lauf(ordner: Path, vor: str, rest: int) -> None:
        for p in _gitmodules(ordner):
            voll = f"{vor}{p}"
            aus.append(voll)
            if rest > 1 and (ordner / p).is_dir():
                lauf(ordner / p, voll + "/", rest - 1)

    lauf(wurzel, "", tiefe)
    return sorted(set(aus), key=lambda s: (-len(s), s))


def _aussen_name(wurzel: Path) -> str:
    try:
        text = (wurzel / REGELDATEI).read_text(encoding="utf-8")
        return regeln_laden(text).name or wurzel.name
    except (OSError, ValueError):
        return wurzel.name


def _doku_repo(datei: str | None, aussen: str, submodule: list[str]) -> str:
    if datei:
        d = _posix(datei)
        for s in submodule:
            if d == s or d.startswith(s + "/"):
                return f"{aussen}/{s}"
    return aussen


def _kurz(text: str, n: int = 90) -> str:
    z = " ".join(text.split())
    return z if len(z) <= n else z[: n - 1] + "…"


def _pruefliste_hinweis(konfig: Konfig) -> str:
    """`pruefliste_laden` verschluckt Fehler; hier wird (nur lesend) geprueft, ob die Datei da und lesbar ist."""
    pfad = pruefliste_pfad(konfig)
    if not pfad.is_file():
        return f"{UNLESBAR_PRAEFIX}doku: doku-pruefliste fehlt: {_posix(str(pfad))}"
    try:
        roh = json.loads(pfad.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        roh = None
    if not isinstance(roh, dict):
        return f"{UNLESBAR_PRAEFIX}doku: doku-pruefliste unlesbar: {_posix(str(pfad))}"
    return ""


def _doku_funde(konfig: Konfig, aussen: str, submodule: list[str]) -> list[Fund]:
    liste = pruefliste_laden(konfig)
    gemeinsam = liste["offen"] + liste["struktur"]
    laeufe: dict[str, set[str]] = defaultdict(set)
    for e in gemeinsam:
        if e.get("art") in _JUDGE and e.get("aussage_id") is not None:
            laeufe[str(e["aussage_id"])].add(str(e.get("lauf")))
    funde: dict[str, Fund] = {}
    for e in gemeinsam:
        art = str(e.get("art"))
        schwere: Schwere
        if art == _ABWEICHEND:
            schwere = "hoch" if abweichend_ist_werkzeug(str(e.get("text", ""))) else "mittel"
        elif art in _DOKU_MITTEL:
            schwere = "mittel"
        elif art in _JUDGE and len(laeufe.get(str(e.get("aussage_id")), ())) >= 2:
            schwere = "mittel"
        else:
            continue
        datei = _posix(str(e["datei"])) if e.get("datei") else ""
        aid = _posix(str(e["aussage_id"])) if e.get("aussage_id") else ""
        anker = f"{art}|{datei}|{aid}"
        if anker in funde:
            continue
        zeile = e.get("zeile")
        beleg = f"{datei}:{zeile}" if datei and zeile else (datei or aid)
        text = str(e.get("text", ""))
        funde[anker] = Fund(
            art="doku_widerspruch", schwere=schwere, code_repo=_doku_repo(datei, aussen, submodule),
            titel=f"{art}: {_kurz(text)}", text=text, belege=(beleg,) if beleg else (), anker=anker)
    return list(funde.values())


def _stand_ordner(wurzel: Path, stand_unter: Path | None) -> Path | None:
    if stand_unter is None:
        graph = stand_finden(wurzel)
        return graph.parent if graph is not None else None
    s = Path(stand_unter)
    if (s / "analyse.sqlite").is_file():
        return s
    if s.is_dir():
        treffer = sorted(o for o in s.glob(f"*-{pfad_kennung(wurzel)}") if (o / "analyse.sqlite").is_file())
        return treffer[-1] if treffer else None
    return None


def _analyse_lesen(wurzel: Path, stand_unter: Path | None) -> tuple[Analyse | None, str, list[str]]:
    """(Analyse, Hinweis, Fremdcode-Regeln am Graph-Commit); die Regeln sind leer, solange kein Commit bekannt ist."""
    stand = _stand_ordner(wurzel, stand_unter)
    fremdcode: list[str] = []
    if stand is None or not (stand / "analyse.sqlite").is_file():
        return None, f"{UNLESBAR_PRAEFIX}analyse: Analyse fehlt (erst `python -m selbstbild analysieren`): keine Doppelungen, kein toter Code", fremdcode
    try:
        commit, gemessen_um = _stand_meta(stand)
        if commit is None:
            return None, f"{UNLESBAR_PRAEFIX}analyse: Analyse nicht lesbar (Stand ohne Graph): keine Doppelungen, kein toter Code", fremdcode
        fremdcode = fremdcode_am_commit(wurzel, commit)
        for kolonie, budget in ((True, True), (True, False), (False, False)):
            a = lesen(stand, commit, ANALYSE_VERSION, digest_von(parameter(kolonie, budget, fremdcode)),
                      graph_gemessen_um=gemessen_um or "")
            if a is not None:
                return a, "", fremdcode
    except (RuntimeError, sqlite3.Error, OSError, ValueError, KeyError, TypeError) as fehler:
        return None, f"{UNLESBAR_PRAEFIX}analyse: Analyse nicht lesbar ({type(fehler).__name__}): keine Doppelungen, kein toter Code", fremdcode
    return None, f"{UNLESBAR_PRAEFIX}analyse: Analyse veraltet (Graph neu vermessen): keine Doppelungen, kein toter Code", fremdcode


_FEHLERHAFT = (AttributeError, KeyError, TypeError, ValueError, IndexError)


def _nur_fremd(d: Doppelung, fremd: Callable[[str], bool] | None) -> bool:
    """Alle Mitglieder sind Fremdcode (leere Doppelung nie)."""
    return fremd is not None and bool(d.mitglieder) and all(fremd(_posix(m)) for m in d.mitglieder)


def _doppel_funde(a: Analyse, fremd: Callable[[str], bool] | None = None) -> tuple[list[Fund], int]:
    """(Funde, Zahl uebersprungener fehlerhafter Eintraege). Doppelungen nur aus Fremdcode entfallen."""
    funde: dict[str, Fund] = {}
    kaputt = 0
    for d in a.doppelungen:
        try:
            if _nur_fremd(d, fremd):
                continue
            _doppel_eintrag(d, funde)
        except _FEHLERHAFT:
            kaputt += 1
    return list(funde.values()), kaputt


def _doppel_eintrag(d: Doppelung, funde: dict[str, Fund]) -> None:
    mitglieder = sorted({_posix(m) for m in d.mitglieder})
    if not mitglieder:
        return
    anker = "|".join(mitglieder)
    if anker in funde:
        return
    pfade = [_id_teile(m)[1] for m in mitglieder]
    mittel = d.aehnlichkeit >= 1.0 - 1e-9 and len(mitglieder) >= 3 \
        and any(not ist_nicht_produktiv(p) for p in pfade)
    schwere: Schwere = "mittel" if mittel else "niedrig"
    funde[anker] = Fund(
        art="doppelung", schwere=schwere, code_repo=_id_teile(mitglieder[0])[0],
        titel=f"{len(mitglieder)} Mitglieder, Ähnlichkeit {d.aehnlichkeit:.2f}: {pfade[0]}",
        text=f"{d.art}, Ähnlichkeit {d.aehnlichkeit:.4f}, {len(mitglieder)} Mitglieder.",
        belege=tuple(pfade), anker=anker)


def _fremd_funde(a: Analyse, fremd: Callable[[str], bool] | None) -> list[Fund]:
    """Die Funde, die `_doppel_funde` wegen Fremdcode auslaesst (je Anker einer)."""
    funde: dict[str, Fund] = {}
    for d in a.doppelungen:
        try:
            if _nur_fremd(d, fremd):
                _doppel_eintrag(d, funde)
        except _FEHLERHAFT:
            continue
    return list(funde.values())


def _tot_funde(a: Analyse) -> tuple[list[Fund], int]:
    funde: dict[str, Fund] = {}
    kaputt = 0
    for t in a.tot:
        try:
            if t.stufe != "sicher_tot":
                continue
            sym = _posix(t.symbol)
            if sym in funde:
                continue
            repo, pfad = _id_teile(sym)
            name = sym.rpartition("#")[2]
            funde[sym] = Fund(
                art="tot", schwere="mittel", code_repo=repo, titel=f"{pfad}#{name}",
                text="Sicher toter Code: " + ("; ".join(t.gruende) if t.gruende else "kein Aufrufer gefunden"),
                belege=(f"{pfad}#{name}",), anker=sym)
        except _FEHLERHAFT:
            kaputt += 1
    return list(funde.values()), kaputt


def _sortiert(funde: Iterable[Fund]) -> list[Fund]:
    return sorted(funde, key=lambda f: (_RANG[f.schwere], f.art, f.code_repo, f.anker))


def geprueft_repos(wurzel: Path, *, doku_konfig: Konfig) -> frozenset[str]:
    """Code-Repos, die ein Lauf ueber `wurzel` wirklich prueft (fuer das Schliessen, Task 4 Fix-Runde 1 I1).

    Gleiche Benennung wie `funde_sammeln`: der aeussere Name plus `<aussen>/<pfad>` je Submodul (rekursiv), aber
    nur Submodule mit Checkout (`.git`-Datei oder -Ordner). Ein nur in `.gitmodules` deklariertes Submodul wird nicht
    vermessen und steht deshalb nicht drin. `doku_konfig` wird nur gelesen: zeigt sie auf ein anderes Repo als
    `wurzel`, passen Doku-Funde und Repo-Namen nicht zusammen -- fail-closed die leere Menge (nichts wird geschlossen).
    """
    wurzel = Path(wurzel)
    try:
        if Path(doku_konfig.repo).resolve() != wurzel.resolve():
            return frozenset()
    except OSError:
        return frozenset()
    aussen = _aussen_name(wurzel)
    return frozenset({aussen, *(f"{aussen}/{s}" for s in _submodul_pfade(wurzel) if (wurzel / s / ".git").exists())})


def funde_sammeln(wurzel: Path, *, doku_konfig: Konfig, stand_unter: Path | None = None,
                  tor: ToterCodeTor,
                  ausgelassen: list[FremdAusgelassen] | None = None) -> tuple[list[Fund], list[str]]:
    """Funde (nach Schwere, Art, Repo, Anker sortiert) und Hinweise. Schreibt nichts.

    `ausgelassen`: optionaler Sammler; bekommt (nur bei Fremdcode-Auslassungen) ein `FremdAusgelassen` angehaengt,
    damit `abgleichen` Schliess-Kommentare ehrlich formulieren kann (F1)."""
    wurzel = Path(wurzel)
    aussen = _aussen_name(wurzel)
    notizen: list[str] = []
    hinweis_doku = _pruefliste_hinweis(doku_konfig)
    if hinweis_doku:
        notizen.append(hinweis_doku)
    funde = _doku_funde(doku_konfig, aussen, _submodul_pfade(wurzel))
    analyse, hinweis, fremdcode = _analyse_lesen(wurzel, stand_unter)
    if hinweis:
        notizen.append(hinweis)
    if analyse is not None:
        pruefer = fremd_pruefer(fremdcode, aussen)
        fremd = (lambda graph_id: ist_fremd_id(pruefer, graph_id)) if fremdcode else None
        neu, kaputt = _doppel_funde(analyse, fremd)
        funde += neu
        weg_funde = _fremd_funde(analyse, fremd)
        weg = len(weg_funde)
        if weg and ausgelassen is not None:
            ausgelassen.append(FremdAusgelassen(frozenset(fund_schluessel(f) for f in weg_funde),
                                                frozenset((f.art, f.code_repo) for f in weg_funde)))
        if weg:
            notizen.append(f"fremdcode: {weg} Doppelungen nur in Fremdcode ausgelassen")
        if kaputt:
            notizen.append(f"{UNLESBAR_PRAEFIX}analyse: {kaputt} fehlerhafte Doppelungs-Einträge übersprungen")
    if not tor.offen:
        notizen.append(f"toter Code: gesperrt ({tor.ja}/{tor.beurteilt} bestätigt)")
    elif analyse is not None:
        neu, kaputt = _tot_funde(analyse)
        funde += neu
        if kaputt:
            notizen.append(f"{UNLESBAR_PRAEFIX}analyse: {kaputt} fehlerhafte Tot-Einträge übersprungen")
    return _sortiert(funde), notizen
