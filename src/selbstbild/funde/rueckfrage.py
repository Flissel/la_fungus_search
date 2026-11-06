"""Rueckfragen aus dem Goldsatz als eigene Issue-Spur (Stufe 6a, Task 3).

Ein `tot`-Eintrag mit Urteil `unklar` wird ein GitHub-Issue (Labels `selbstbild` und `selbstbild:rueckfrage`). Der
User antwortet mit genau einem Label (`ANTWORTEN`); der naechste Lauf traegt die Antwort in die Goldsatz-YAML ein,
committet nur diese Datei im Repo, das sie enthaelt (Branch `main`, Datei sauber, ohne Push), und schliesst das Issue.

Dubletten nur ueber die Marke `<!-- selbstbild:rueckfrage=<sha256(symbol)[:16]> -->` im Issue-Text. Sie passt bewusst
nicht auf `sync._MARKE`/`sync._SAMMEL`, so fasst die Fund-Synchronisation Rueckfragen nie an.

Sonderfaelle (Spec 4.4, Punkt 4):
- mehrere Antwort-Labels: einmal kommentieren (Merker `stand["kommentiert"]`), nichts uebernehmen;
- kein Label und das Symbol ist nicht mehr toter Code: schliessen, der Goldsatz bleibt;
- ohne Antwort-Label geschlossen und noch `unklar`: einmal melden (Merker `stand["gemeldet"]`), nie wieder oeffnen;
- uebernommen wird nur in Eintraege, die noch `unklar` sind (R6). Steht dort schon genau die Antwort: schliessen mit
  „schon übernommen“; steht ein anderes Urteil: nichts uebernehmen, schliessen als erledigt;
- geschlossen mit genau einem Antwort-Label und noch `unklar`: uebernehmen wie offen, dann nur kommentieren (R7);
- leeres `noch_tot`: nichts wird als erledigt geschlossen (R8a).

Fehler von GitHub erscheinen nur als `GitHubFehler.art`; der erste Schreibfehler beendet den Lauf (`fehler`).
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..doku.schluessel import sieht_aus_wie_schluessel
from ..doku.schreibvorgang import git, kopf, kopf_pruefen
from .github import LABEL, LABEL_FARBE, LABEL_TEXT, GitHub, GitHubFehler
from .goldsatz_tor import _tot_eintraege, antworten_uebernehmen, unklare_eintraege
from .sammeln import _aussen_name, _id_teile
from .sync import _body, _labels
from .text import pruefen

RUECKFRAGE_LABEL = "selbstbild:rueckfrage"
ANTWORTEN: dict[str, tuple[str, bool]] = {
    "antwort:weg": ("ja", False),
    "antwort:bleibt": ("ja", True),
    "antwort:benutzt": ("nein", False),
}
MAX_NEU = 10
AUSSCHNITT_ZEILEN = 18
GOLDSATZ_ZWEIG = "main"
COMMIT_TEXT = "docs(selbstbild): Rückfrage-Antworten übernehmen"
DOPPELT_TEXT = "Bitte nur ein Antwort-Label setzen (antwort:weg, antwort:bleibt oder antwort:benutzt)."
SCHON = "schon übernommen"

# (Name, Farbe, Beschreibung) der Labels, die vor dem ersten Anlegen sichergestellt werden
_LABELS: tuple[tuple[str, str, str], ...] = (
    (LABEL, LABEL_FARBE, LABEL_TEXT),
    (RUECKFRAGE_LABEL, "fbca04", "Rückfrage von Selbstbild: mit genau einem antwort:-Label beantworten"),
    ("antwort:weg", "d73a4a", "Unbenutzt, kann weg"),
    ("antwort:bleibt", "0e8a16", "Unbenutzt, soll aber bleiben"),
    ("antwort:benutzt", "1d76db", "Wird doch benutzt"),
)
_MARKE = re.compile(r"<!--\s*selbstbild:rueckfrage\s*=\s*([0-9a-f]{16})\s*-->")
_COMMIT = re.compile(r"[0-9a-fA-F]{7,64}")
_GRUND = {
    "unerreichbar": "unerreichbar (kein Aufrufer gefunden)",
    "alt_90": "seit über 90 Tagen unverändert",
    "kein_doku_verweis": "keine Doku verweist darauf",
    "doku_verweis": "nur die Doku verweist darauf",
}
_ANLEITUNG = (
    "**So antwortest du:** Setze genau ein Label.\n\n"
    "- `antwort:weg`: unbenutzt, kann weg\n"
    "- `antwort:bleibt`: unbenutzt, soll aber bleiben\n"
    "- `antwort:benutzt`: wird doch benutzt\n\n"
    "Der nächste Lauf übernimmt die Antwort in den Goldsatz und schließt dieses Issue. "
    "Schließen ohne Label heißt: bleibt unklar."
)


@dataclass
class RueckfrageBericht:
    """Ergebnis eines Abgleichs. Nummern sind Issue-Nummern; `wartet` nennt je Issue den Grund (`#<nr> <name>:
    <grund>`); `links` hat je bekannte oder neue Rueckfrage die Issue-URL."""
    neu: list[int] = field(default_factory=list)
    uebernommen: list[int] = field(default_factory=list)
    geschlossen_ohne_antwort: list[int] = field(default_factory=list)
    erledigt: list[int] = field(default_factory=list)
    wartet: list[str] = field(default_factory=list)
    doppelt_beantwortet: list[int] = field(default_factory=list)
    notizen: list[str] = field(default_factory=list)
    fehler: str | None = None
    links: dict[int, str] = field(default_factory=dict)
    goldsatz_commit: str | None = None


def _hash(symbol: str) -> str:
    return hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:16]


def marke(symbol: str) -> str:
    return f"<!-- selbstbild:rueckfrage={_hash(symbol)} -->"


def _name(symbol: str) -> str:
    return symbol.rpartition("#")[2] if "#" in symbol else symbol


def _repo_kurz(repo: str) -> str:
    """Code-Repo ohne den Aussennamen (`vibemind/vibemind-os` -> `vibemind-os`); das aeussere Repo selbst bleibt."""
    kopf_, _, rest = repo.partition("/")
    return rest or kopf_


# ---------- Code-Ausschnitt ----------

def _suchen(baum: ast.Module, teile: list[str]) -> ast.AST | None:
    koerper: list[ast.stmt] = baum.body
    knoten: ast.AST | None = None
    for teil in teile:
        knoten = next((k for k in koerper if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       and k.name == teil), None)
        if knoten is None:
            return None
        koerper = knoten.body  # type: ignore[attr-defined]
    return knoten


def _fundstelle(wurzel: Path, symbol: str, n: int) -> tuple[int, str] | None:
    """(Zeile der Definition, Ausschnitt) aus dem committeten Stand (`HEAD`) des Code-Repos, sonst None.

    Repo und Pfad wie bei den Tot-Funden (`sammeln._id_teile`); das Code-Repo liegt unter `wurzel` am Repo-Pfad ohne
    das erste Segment (Aussenname). Nur `.py`, nur Funktionen/Klassen (auch verschachtelt `K.m`)."""
    if n <= 0 or not symbol.startswith("symbol:") or "#" not in symbol:
        return None
    repo, pfad = _id_teile(symbol)
    teile = repo.split("/")
    pfad_teile = pfad.split("/")
    if not pfad.endswith(".py") or teile[0] != _aussen_name(Path(wurzel)) \
            or any(t in ("", ".", "..") for t in teile[1:] + pfad_teile):
        return None
    checkout = Path(wurzel).joinpath(*teile[1:])
    if not (checkout / ".git").exists():  # sonst liefe `git -C` ins umgebende Repo hoch
        return None
    quelle = git(checkout, "show", f"HEAD:{pfad}")
    if quelle is None:
        return None
    try:
        baum = ast.parse(quelle)
    except (SyntaxError, ValueError):
        return None
    knoten = _suchen(baum, _name(symbol).split("."))
    if not isinstance(knoten, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    start = min([knoten.lineno, *(d.lineno for d in knoten.decorator_list)])
    ende = min(knoten.end_lineno or knoten.lineno, start + n - 1)
    code = "\n".join(quelle.splitlines()[start - 1: ende]).rstrip()
    if not code.strip() or sieht_aus_wie_schluessel(code):
        return None
    return knoten.lineno, code


def code_ausschnitt(wurzel: Path, symbol: str, n: int = AUSSCHNITT_ZEILEN) -> str | None:
    """Hoechstens `n` Zeilen der Definition aus `HEAD` (`git show`), ab dem ersten Dekorator; None bei
    Schluessel-Verdacht oder wenn etwas nicht lesbar ist."""
    fund = _fundstelle(wurzel, symbol, n)
    return fund[1] if fund else None


# ---------- Texte ----------

def _titel(symbol: str) -> str:
    repo, pfad = _id_teile(symbol)
    return f"[selbstbild] Rückfrage: {_name(symbol)} ({_repo_kurz(repo)}/{pfad})"


def _zaun(code: str) -> str:
    laengste = max((len(m.group(0)) for m in re.finditer(r"`+", code)), default=0)
    return "`" * max(3, laengste + 1)


def _text(symbol: str, eintrag: dict[str, Any], fund: tuple[int, str] | None) -> str:
    repo, pfad = _id_teile(symbol)
    ort = f"{_repo_kurz(repo)}/{pfad}" + (f":{fund[0]}" if fund else "")
    roh = eintrag.get("gruende")
    gruende = [_GRUND.get(str(g), str(g)) for g in roh] if isinstance(roh, list) else []
    teile = [
        "Selbstbild hält diesen Code für tot. Beim Prüfen des Goldsatzes war unklar, ob er noch gebraucht wird.",
        f"**Ort:** `{ort}`\n**Symbol:** `{symbol}`\n**Gründe:** {', '.join(gruende) or '(keine)'}",
    ]
    if fund:
        z = _zaun(fund[1])
        teile.append(f"**Ausschnitt:**\n\n{z}python\n{fund[1]}\n{z}")
    teile += ["**Wird das noch gebraucht?**", _ANLEITUNG, marke(symbol)]
    return "\n\n".join(teile) + "\n"


# ---------- Goldsatz-Commit ----------

def _goldsatz_repo(pfad: Path) -> tuple[Path | None, str | None]:
    """(Repo, None), wenn in die Goldsatz-Datei geschrieben und committet werden darf, sonst (None, Grund)."""
    top = git(pfad.parent, "rev-parse", "--show-toplevel")
    if top is None or not top.strip():
        return None, "kein_git_repo"
    repo = Path(top.strip())
    grund = kopf_pruefen(repo, GOLDSATZ_ZWEIG)
    if grund:
        return None, grund
    status = git(pfad.parent, "status", "--porcelain", "--", pfad.name)
    if status is None:
        return None, "git_fehler:status"
    if status.strip():
        return None, "goldsatz_unsauber"
    return repo, None


def _eingetragen(pfad: Path, antworten: dict[str, tuple[str, bool]]) -> bool:
    """Ist die geschriebene Datei gueltiges YAML, und hat jeder Eintrag jedes beantworteten Symbols genau das erwartete
    `urteil` und `bleibt`?"""
    try:
        eintraege = _tot_eintraege(pfad)
    except (OSError, ValueError, AttributeError, TypeError, yaml.YAMLError):
        return False
    gefunden: set[str] = set()
    for e in eintraege:
        symbol = e.get("symbol")
        if not isinstance(symbol, str) or symbol not in antworten:
            continue
        urteil, bleibt = antworten[symbol]
        if str(e.get("urteil")) != urteil or (e.get("bleibt") is True) != bleibt:
            return False
        gefunden.add(symbol)
    return gefunden == set(antworten)


def _uebernehmen_committen(pfad: Path, antworten: dict[str, tuple[str, bool]]) -> tuple[str | None, str | None]:
    """(sha7 oder `SCHON`, None) bei Erfolg, sonst (None, Grund). Geschrieben und committet wird nur `pfad`; scheitert
    der Commit, wird die Datei auf den committeten Stand zurueckgesetzt. Ist HEAD nach dem Commit nicht lesbar, gilt
    das als nicht bestaetigt (`commit_unbestaetigt`): die Datei bleibt, die Issues bleiben offen."""
    repo, grund = _goldsatz_repo(pfad)
    if repo is None:
        return None, grund
    ordner, name = pfad.parent, pfad.name
    try:
        diff = antworten_uebernehmen(pfad, antworten)
    except ValueError as e:  # vor dem Schreiben
        return None, f"goldsatz_fehler:{type(e).__name__}"
    except OSError as e:
        zurueck = git(ordner, "checkout", "HEAD", "--", name) is not None
        return None, f"goldsatz_fehler:{type(e).__name__}" + ("" if zurueck else ", Datei nicht zurückgesetzt")
    if not diff:
        return SCHON, None
    if not _eingetragen(pfad, antworten):  # R17-1: nie eine kaputte oder falsch eingetragene Datei committen
        zurueck = git(ordner, "checkout", "HEAD", "--", name) is not None
        return None, "yaml_kaputt" + ("" if zurueck else ", Datei nicht zurückgesetzt")
    if git(ordner, "add", "--", name) is None or git(ordner, "commit", "-q", "-m", COMMIT_TEXT, "--", name) is None:
        zurueck = git(ordner, "checkout", "HEAD", "--", name) is not None
        return None, "commit_fehlgeschlagen" + ("" if zurueck else ", Datei nicht zurückgesetzt")
    sha = kopf(repo)
    if not sha:  # Commit nicht bestaetigbar: Issues bleiben offen; der naechste Lauf schliesst mit `SCHON`
        return None, "commit_unbestaetigt"
    return sha[:7], None


# ---------- Abgleich ----------

def _uebernommen_text(antwort: tuple[str, bool], stand_commit: str) -> str:
    urteil, bleibt = antwort
    return f"Antwort übernommen: {urteil}{', bleibt' if bleibt else ''} (Goldsatz-Commit {stand_commit})."


def _zahlen(wert: object) -> set[int]:
    if not isinstance(wert, list):
        return set()
    return {x for x in wert if isinstance(x, int) and not isinstance(x, bool) and x > 0}


def _nummer(issue: dict[str, Any]) -> int | None:
    n = issue.get("number")
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None


def _link(gh: GitHub | None, repo: str, nummer: int) -> str:
    return f"https://{gh.host if gh is not None else 'github.com'}/{repo}/issues/{nummer}"


def rueckfragen_abgleichen(wurzel: Path, goldsatz_pfad: Path, gh: GitHub | None, repo: str, *, senden: bool,
                           noch_tot: frozenset[str], commit: str, stand: dict[str, Any],
                           oeffentlich: bool = False) -> RueckfrageBericht:
    """Gleicht die `unklar`-Eintraege des Goldsatzes mit den Rueckfrage-Issues ab (siehe Moduldoku).

    `oeffentlich`: die Konfiguration des Ziels (`Ziel.oeffentlich`). Code-Ausschnitte nur, wenn sie das nicht sagt UND
    GitHub das Repo als privat meldet (D6: Konfiguration ODER GitHub sagt oeffentlich -> kein Ausschnitt).

    `noch_tot`: Symbole, die die gespeicherte Analyse noch als toten Code fuehrt (jede Stufe).
    `commit`: SHA des vermessenen Stands (nur fuer den Erledigt-Kommentar, 7 Zeichen).
    `stand`: veraenderliches Teil-Dict (`auto-stand.json["rueckfrage"]`); hier gepflegt: `kommentiert` und
    `gemeldet` (je eine sortierte Liste von Issue-Nummern). Mit `senden=False` bleibt es unveraendert.
    Mit `senden=False` wird nichts geschrieben und nichts committet; der Bericht zeigt, was geschehen wuerde.
    """
    if senden and gh is None:
        raise ValueError("senden braucht ein GitHub-Objekt (Konto)")
    if not _COMMIT.fullmatch(commit or ""):
        raise ValueError("commit muss ein Git-SHA sein")
    sha7 = commit[:7].lower()
    goldsatz_pfad = Path(goldsatz_pfad)
    b = RueckfrageBericht()
    kommentiert = _zahlen(stand.get("kommentiert"))
    gemeldet = _zahlen(stand.get("gemeldet"))

    def merken(schluessel: str, menge: set[int]) -> None:
        if senden:
            stand[schluessel] = sorted(menge)

    def schreiben(methode: str, *args: Any) -> bool:
        """Ein Schreibaufruf (nur mit `senden`); False nach einem Fehler, der dann in `b.fehler` steht."""
        if not senden:
            return True
        try:
            getattr(gh, methode)(repo, *args)
        except GitHubFehler as f:
            b.fehler = f.art
            return False
        return True

    try:
        eintraege = _tot_eintraege(goldsatz_pfad)
        unklar = {str(e["symbol"]) for _, e in unklare_eintraege(goldsatz_pfad) if e.get("symbol")}
    except (OSError, ValueError, AttributeError, TypeError, yaml.YAMLError) as e:
        b.fehler = f"goldsatz_unlesbar:{type(e).__name__}"
        return b
    symbole: dict[str, str] = {}
    zustand: dict[str, tuple[str, bool]] = {}  # Symbol -> (urteil, bleibt) des ersten Eintrags
    for e in eintraege:
        if isinstance(e.get("symbol"), str) and e["symbol"]:
            symbole.setdefault(_hash(e["symbol"]), e["symbol"])
            zustand.setdefault(e["symbol"], (str(e.get("urteil", "offen")), e.get("bleibt") is True))

    # 1. Bestand (nur lesend); nur Issues mit Rueckfrage-Marke zaehlen
    if gh is None:
        bestand: list[dict[str, Any]] = []
        b.notizen.append("kein gh: Bestand als leer angenommen (offline)")
    else:
        try:
            bestand = gh.issues(repo)
        except GitHubFehler as f:
            b.fehler = f.art
            return b
    je_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eigene: list[tuple[int, dict[str, Any], str | None]] = []
    for issue in sorted(bestand, key=lambda i: _nummer(i) or 0):
        m = _MARKE.search(_body(issue))
        nr = _nummer(issue)
        if not m or nr is None:
            continue
        je_hash[m.group(1)].append(issue)
        eigene.append((nr, issue, symbole.get(m.group(1))))
        url = issue.get("html_url")
        b.links[nr] = url if isinstance(url, str) and url else _link(gh, repo, nr)

    # 2. Rueckfragen der Reihe nach. Uebernommen wird nur in Eintraege, die noch `unklar` sind (R6): eine Antwort
    #    des Users wird nie ueberschrieben. Geschlossene mit genau einem Antwort-Label zaehlen mit (R7).
    gesammelt: list[tuple[int, str, tuple[str, bool], bool]] = []  # (nr, symbol, antwort, offen)
    antwort_je_symbol: dict[str, set[tuple[str, bool]]] = defaultdict(set)
    noch_tot_leer = False
    for nr, issue, symbol in eigene:
        offen = issue.get("state") == "open"
        gesetzt = sorted(x for x in _labels(issue) if x in ANTWORTEN)
        if not offen:
            if symbol is not None and len(gesetzt) == 1 and symbol in unklar:
                antwort = ANTWORTEN[gesetzt[0]]
                gesammelt.append((nr, symbol, antwort, False))
                antwort_je_symbol[symbol].add(antwort)
            continue
        if symbol is None:
            b.notizen.append(f"Rückfrage #{nr}: Symbol nicht im Goldsatz, nichts übernommen")
            continue
        if len(gesetzt) == 1:
            antwort = ANTWORTEN[gesetzt[0]]
            if symbol in unklar:
                gesammelt.append((nr, symbol, antwort, True))
                antwort_je_symbol[symbol].add(antwort)
                continue
            if zustand.get(symbol) == antwort:
                if not schreiben("schliessen", nr, _uebernommen_text(antwort, SCHON)):
                    return b
                b.uebernommen.append(nr)
            else:
                urteil_jetzt = zustand.get(symbol, ("?", False))[0]
                text = f"Im Goldsatz inzwischen anders beurteilt ({urteil_jetzt}), Rückfrage erledigt."
                if not schreiben("schliessen", nr, text):
                    return b
                b.erledigt.append(nr)
        elif len(gesetzt) > 1:
            if nr not in kommentiert:
                if not schreiben("kommentieren", nr, DOPPELT_TEXT):
                    return b
                kommentiert.add(nr)
                merken("kommentiert", kommentiert)
            b.doppelt_beantwortet.append(nr)
        elif not noch_tot:
            noch_tot_leer = True  # R8a: leere Menge heisst eher "Analyse fehlt" als "alles lebt"
        elif symbol not in noch_tot:
            text = f"Nicht mehr als toter Code erkannt (Commit {sha7}), Rückfrage erledigt."
            if not schreiben("schliessen", nr, text):
                return b
            gemeldet.add(nr)  # nie als "ohne Antwort geschlossen" melden
            merken("gemeldet", gemeldet)
            b.erledigt.append(nr)

    if noch_tot_leer:
        b.notizen.append("noch_tot leer: keine Rückfrage als erledigt geschlossen")

    # 3. Antworten uebernehmen (ein Commit fuer alle), dann offene schliessen, geschlossene nur kommentieren
    widerspruch = {s for s, a in antwort_je_symbol.items() if len(a) > 1}
    for nr, symbol, _, _ in gesammelt:
        if symbol in widerspruch:
            b.notizen.append(f"Rückfrage #{nr} ({_name(symbol)}): widersprüchliche Antworten in mehreren Issues, "
                             "nichts übernommen")
    gesammelt = [g for g in gesammelt if g[1] not in widerspruch]
    if gesammelt:
        if senden:
            stand_commit, grund = _uebernehmen_committen(goldsatz_pfad, {s: a for _, s, a, _ in gesammelt})
        else:
            stand_commit, grund = (None, _goldsatz_repo(goldsatz_pfad)[1])
        if grund:
            b.wartet += [f"#{nr} {_name(symbol)}: {grund}" for nr, symbol, _, _ in gesammelt]
        else:
            if senden and stand_commit != SCHON:
                b.goldsatz_commit = stand_commit
            for nr, _, antwort, offen in gesammelt:
                text = _uebernommen_text(antwort, stand_commit or SCHON)
                if not schreiben("schliessen" if offen else "kommentieren", nr, text):
                    return b
                b.uebernommen.append(nr)

    # 4. Ohne Antwort-Label geschlossen und noch unklar: einmal melden, nie wieder oeffnen
    for nr, issue, symbol in eigene:
        if issue.get("state") == "open" or symbol is None or nr in gemeldet:
            continue
        if symbol in unklar and symbol in noch_tot and not (_labels(issue) & ANTWORTEN.keys()):
            b.geschlossen_ohne_antwort.append(nr)
            gemeldet.add(nr)
            merken("gemeldet", gemeldet)

    # 5. Neue Rueckfragen (nach dem Uebernehmen neu gelesen: beantwortete sind nicht mehr `unklar`)
    try:
        offen_unklar = unklare_eintraege(goldsatz_pfad)
    except (OSError, ValueError, AttributeError, TypeError, yaml.YAMLError) as e:
        b.fehler = f"goldsatz_unlesbar:{type(e).__name__}"
        return b
    kandidaten: list[tuple[str, dict[str, Any]]] = []
    gesehen: set[str] = set()
    nicht_tot = 0
    for _, e in offen_unklar:
        s = e.get("symbol")
        if not isinstance(s, str) or not s or s in gesehen or _hash(s) in je_hash:
            continue
        gesehen.add(s)
        if s not in noch_tot:
            nicht_tot += 1
            continue
        kandidaten.append((s, e))
    if nicht_tot:
        b.notizen.append(f"Rückfragen: {nicht_tot} unklare Einträge sind kein toter Code mehr, nicht gefragt")
    if len(kandidaten) > MAX_NEU:
        b.notizen.append(f"Rückfragen: {len(kandidaten) - MAX_NEU} weitere beim nächsten Lauf")
        kandidaten = kandidaten[:MAX_NEU]
    privat = False
    if kandidaten and senden and gh is not None:
        try:
            for name, farbe, beschreibung in _LABELS:
                gh.label_sicherstellen(repo, name, farbe, beschreibung)
            privat = not oeffentlich and gh.ist_privat(repo)  # Code-Ausschnitte nur in private Ziele (D6)
        except GitHubFehler as f:
            b.fehler = f.art
            return b
    for s, e in kandidaten:
        fund = _fundstelle(wurzel, s, AUSSCHNITT_ZEILEN) if privat else None
        titel, text = _titel(s), _text(s, e, fund)
        if fund and pruefen(titel + "\n" + text):
            text = _text(s, e, None)
        if pruefen(titel + "\n" + text):
            b.notizen.append(f"Rückfrage {_name(s)}: Schlüssel-Verdacht im Text, nicht gesendet")
            continue
        if not senden or gh is None:
            b.notizen.append(f"trocken: würde anlegen: {titel}")
            continue
        try:
            nr = gh.anlegen(repo, titel, text, labels=[RUECKFRAGE_LABEL])
        except GitHubFehler as f:
            b.fehler = f.art
            return b
        b.neu.append(nr)
        b.links[nr] = _link(gh, repo, nr)
    if not senden:
        b.notizen.append("trocken: nichts geschrieben, nichts committet")
    return b
