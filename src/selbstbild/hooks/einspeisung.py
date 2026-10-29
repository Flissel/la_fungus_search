"""UserPromptSubmit-Hook: speist den passenden Code-Kontext vor jedem Prompt ein (Stufe 3, T7).

Laeuft bei JEDEM Prompt, darum: nur Standardbibliothek, hoechstens 1,5 s (eigene Frist, danach keine Ausgabe),
hoechstens 8.000 Zeichen (~2.000 Tokens), bei jedem Fehler stille Ausgabe und Exit 0, schreibt nie Dateien.
Der Prompt wird weder ausgegeben noch protokolliert.

Bewusste Dopplung (Ruling U3): Tokenisierung, FTS-Abfrage, Datei-Dedup und Tot-Daempfung spiegeln
`suche.abfrage`/`suche.messung`, die Index-Frische spiegelt `suche.index.index_aktuell`. Importiert werden
`suche.*`/`analyse.*` hier nicht, weil sie igraph/numpy nachladen und den Start um Sekunden verlaengern wuerden.
`test_keine_schweren_importe` sichert das ab.

Ablauf (T12):
1. Arbeitsbereich: die zuletzt benutzten Dateien aus dem Transkript (`verlauf.letzte_dateien`, nur Pfade), auf den
   Hauptbaum abgebildet (`verlauf.hauptpfad`) und auf Index-Dateien; dazu ihre 1-Hop-Nachbarn ueber `kante`
   (importiert/ruft_auf, beide Richtungen, gedeckelt).
2. Prompt-Treffer: FTS5 wie bisher, dann Datei-Dedup und Tot-Daempfung.
3. Fusion: RRF (k=60) aus Prompt-Liste und Arbeitsbereich; Dateien in beiden Listen stehen oben. Dateien, die nur
   Arbeitsbereich sind, stehen im Abschnitt "Zuletzt bearbeitet und Nachbarn".
4. Schweigeschwelle: kein Block, wenn keine Kontextdatei im Index liegt und kein Prompt-Treffer stark ist
   (`MIN_TOKENS_STARK`, `BM25_STARK`). Ohne starken Treffer erscheinen Prompt-Treffer nur, wenn sie im
   Arbeitsbereich liegen.
5. Liegt eine Kontextdatei in einem Git-Worktree, warnt der Block ("zweiter Checkout").
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from . import REGELDATEI, blob_sha, regeln_lesen, repo_wurzeln, stand_fuer
from .verlauf import hauptpfad, letzte_dateien

FRIST_S = 1.5
MAX_ZEICHEN = 8000
MAX_BEREICHE = 3
MAX_STELLEN = 8
MAX_DOKU = 3
MAX_TOKENS = 12
_KANDIDATEN = 40
MAX_PROMPT = 20_000
MAX_GEAENDERT = 3
STANDARD_MAX_KB = 512
RRF_K = 60  # wie suche.abfrage
TOT_FAKTOR = 0.5  # wie suche.abfrage: sicher_tot halbiert den Wert
MAX_KONTEXT = 6  # Zeilen "zuletzt bearbeitet" im Abschnitt
MAX_NACHBARN_ZEILEN = 4  # Zeilen "Nachbar" im Abschnitt
MAX_WARNUNGEN = 3
NACHBARN_JE_DATEI = 5
NACHBARN_GESAMT = 40
NACHBAR_QUELLEN = 8  # Nachbarn werden fuer die neuesten 8 Kontextdateien gesucht
_KANTEN_JE_ABFRAGE = 2000
_TEXT_MAX = 200_000
_NACHBAR_ARTEN = ("importiert", "ruft_auf")

# Schweigeschwelle, vor der Re-Messung aus Prinzip festgelegt (T12), nicht an Messergebnisse angepasst.
# Ein Prompt-Treffer (eine Datei) ist "stark", wenn
#  a) mindestens MIN_TOKENS_STARK verschiedene Prompt-Tokens Name, Pfad oder Text derselben Datei treffen
#     (ein einzelnes Wort ist zu oft Zufall: "weiter", "test", "daten" stehen in tausenden Eintraegen), oder
#  b) der beste bm25-Wert der Anfrage hoechstens BM25_STARK ist. FTS5-bm25 ist negativ; ein Term traegt etwa
#     -IDF * Laengenfaktor bei, mit IDF = ln((N - n + 0.5) / (n + 0.5)) ~ ln(1/p) fuer den Anteil p der Eintraege,
#     die den Term enthalten (damit fast unabhaengig von der Indexgroesse N). Die Schwelle entspricht zwei Tokens
#     mit je p = 1 Promille (IDF ~ ln 1000 ~ 6.9) in einer Zeile durchschnittlicher Laenge (Faktor 1). Ein einzelnes
#     Token erreicht sie nur, wenn es nahezu eindeutig ist und in einer kurzen Zeile steht (Faktor bis
#     (k1+1)/(1+k1(1-b)) ~ 1.69 bei k1=1.2, b=0.75): dann ist es so spezifisch wie ein Bezeichner. In kleinen
#     Indizes (N < ~1000) ist die Schwelle unerreichbar; dort entscheidet allein a).
MIN_TOKENS_STARK = 2
BM25_STARK = -2 * math.log(1000)

# Spiegel von suche.abfrage (Stoppwoerter DE/EN, Camel-/snake-Zerlegung, >= 3 Zeichen, <= 12 Tokens).
_STOPP = frozenset("""
aber alle allem allen aller alles also auch auf aus bei bin bis bist dann das dass dein deine dem den der des
die dies diese diesem diesen dieser dieses doch dort durch ein eine einem einen einer eines euch euer für
gegen hab habe haben hat hast hatte hier ihr ihre ich ist jede jedem jeden jeder jedes jetzt kann kein keine
man mehr mein meine mit muss nach nicht noch nun nur oder ohne sein seine sich sie sind soll sollen sondern
über und uns unser unter vom von vor war waren was weil wenn wer werden wie wieder will wir wird wirst wo
wollen würde zum zur zwischen
about after all also and any are because been before being both but can could did does doing for from had
has have her here him his how into its just more most not off only other our out over she should some such
than that the their them then there these they this those through very was were what when where which while
who whom why will with would you your
""".split())
_GRENZE = re.compile(r"(?<=[a-zäöüß0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-zäöüß])")
_NICHT_WORT = re.compile(r"[\W_]+")
_SHA = re.compile(r"[0-9a-f]{40}")


class _Frist(Exception):
    pass


def _tokens(text: str) -> tuple[str, ...]:
    """Nur die ersten 20.000 Zeichen des Prompts; Sammeln endet bei MAX_TOKENS (kein quadratischer Aufwand)."""
    aus: list[str] = []
    gesehen: set[str] = set()
    for wort in _NICHT_WORT.split(_GRENZE.sub(" ", text[:MAX_PROMPT])):
        w = wort.lower()
        if len(w) >= 3 and w not in _STOPP and w not in gesehen:
            gesehen.add(w)
            aus.append(w)
            if len(aus) >= MAX_TOKENS:
                break
    return tuple(aus)


def _fts(tokens: tuple[str, ...]) -> str:
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def _ro(db: Path, pruefen: Callable[[], None]) -> sqlite3.Connection:
    """Schreibgeschuetzt oeffnen; sqlite bricht ab (OperationalError), sobald `pruefen` die Frist meldet."""
    c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)

    def fortschritt() -> int:
        try:
            pruefen()
        except _Frist:
            return 1
        return 0

    c.set_progress_handler(fortschritt, 1000)
    return c


def _git_head(wurzel: Path) -> str | None:
    """HEAD-Commit aus den Git-Dateien (kein Prozess); None, wenn irgendetwas unklar ist.

    Erwartet `.git` (Ordner oder gitdir-Datei) direkt in der Repo-Wurzel; liegt es woanders, ist das Ergebnis
    None und der Index gilt als veraltet (fail-closed).
    """
    try:
        git = wurzel / ".git"
        gemeinsam = git
        if git.is_file():  # Worktree/Submodul: "gitdir: <pfad>"
            z = git.read_text(encoding="utf-8").strip()
            if not z.startswith("gitdir:"):
                return None
            git = (wurzel / z[7:].strip()).resolve()
            gemeinsam = git
            cd = git / "commondir"
            if cd.is_file():
                gemeinsam = (git / cd.read_text(encoding="utf-8").strip()).resolve()
        kopf = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not kopf.startswith("ref:"):
            return kopf if _SHA.fullmatch(kopf) else None
        ref = kopf[4:].strip()
        for ordner in (git, gemeinsam):
            datei = ordner / ref
            if datei.is_file():
                sha = datei.read_text(encoding="utf-8").strip()
                return sha if _SHA.fullmatch(sha) else None
        gepackt = gemeinsam / "packed-refs"
        if gepackt.is_file():
            for z in gepackt.read_text(encoding="utf-8").splitlines():
                teile = z.split(" ", 1)
                if len(teile) == 2 and teile[1] == ref and _SHA.fullmatch(teile[0]):
                    return teile[0]
        return None
    except (OSError, ValueError):
        return None


def _graph_meta(graph: Path, pruefen: Callable[[], None]) -> dict[str, str]:
    """Schluessel/Werte von graph.sqlite sind JSON-kodiert (Speicher._dumps)."""
    with closing(_ro(graph, pruefen)) as c:
        roh = c.execute("SELECT k, v FROM meta").fetchall()
    aus: dict[str, str] = {}
    for k, v in roh:
        try:
            aus[str(json.loads(k))] = str(json.loads(v))
        except ValueError:
            continue
    return aus


def _pfad_im_arbeitsbaum(wurzel: Path, repo: str | None, pfad: str | None, name: str | None) -> Path | None:
    if not repo or not pfad:
        return None
    if repo == name:
        return wurzel / pfad
    if name and repo.startswith(name + "/"):  # Submodul heisst "<repo>/<pfad im Elternrepo>"
        return wurzel / repo[len(name) + 1:] / pfad
    return None


def _kurz(i: str) -> str:
    teile = i.split(":", 2)
    return (teile[2] if len(teile) == 3 else i)[:80]


def _warnungen(tot: str | None, warnung: str | None) -> str:
    teile: list[str] = []
    if tot == "sicher_tot":
        teile.append("⚠ tot")
    elif tot == "wahrscheinlich_tot":
        teile.append("⚠ wahrscheinlich tot")
    if warnung and warnung.startswith("kopie:"):
        ziele = [_kurz(x) for x in warnung[6:].split(",") if x][:2]
        if ziele:
            teile.append("⚠ Kopie von " + ", ".join(ziele))
    return " ".join(teile)


def _regel_zeile(r: dict) -> str:
    zeile = f"- {r.get('art', '?')} @ {r.get('ort', '?')}"
    if r.get("pflicht"):
        zeile += "; pflicht: " + ", ".join(r["pflicht"])
    if r.get("verboten"):
        zeile += "; verboten: " + ", ".join(r["verboten"])
    if r.get("hinweis"):
        zeile += " | " + r["hinweis"]
    return zeile[:300]


_AKZENT = str.maketrans("äöüàáâãåèéêëìíîïòóôõùúûýÿçñ", "aouaaaaaeeeeiiiioooouuuyycn")


def _ohne_akzent(text: str) -> str:
    """Naeherung an FTS5 `remove_diacritics`: kleingeschrieben, haeufige Akzente entfernt (`ä` -> `a`); C-schnell."""
    return text.lower().translate(_AKZENT)


def _trifft(token: str, text: str) -> bool:
    """`token` steht als ganzes Wort in `text` (beide normalisiert); Wortgrenze wie unicode61: nicht alphanumerisch."""
    i = text.find(token)
    while i >= 0:
        j = i + len(token)
        if (i == 0 or not text[i - 1].isalnum()) and (j >= len(text) or not text[j].isalnum()):
            return True
        i = text.find(token, i + 1)
    return False


def _datei_schluessel(i: str) -> str | None:
    """Eintrags-/Kantenende -> kanonische Datei-Id `datei:<repo>:<pfad>` (Spiegel von `messung._als_datei_id`)."""
    teile = i.split(":", 2)
    if len(teile) != 3 or teile[0] not in ("datei", "doku_seite", "symbol"):
        return None
    pfad = teile[2].rpartition("#")[0] if teile[0] == "symbol" else teile[2]
    return f"datei:{teile[1]}:{pfad}" if pfad else None


def _relativ_zu(pfad: str, wurzel: Path) -> str | None:
    """Absoluter Pfad -> Pfad relativ zur Repo-Wurzel mit `/`; None ausserhalb (Gross/klein wie das System)."""
    try:
        p = os.path.normpath(pfad)
        w = os.path.normpath(str(wurzel))
    except (TypeError, ValueError):
        return None
    if not os.path.normcase(p).startswith(os.path.normcase(w).rstrip(os.sep) + os.sep):
        return None
    rel = p[len(w.rstrip(os.sep)) + 1:].replace("\\", "/")
    return rel or None


def _anzeige(wurzel: Path, repo: str | None, pfad: str | None, name: str | None) -> str:
    """Pfad relativ zur Repo-Wurzel (auch fuer Submodule), sonst der Pfad im Repo."""
    ziel = _pfad_im_arbeitsbaum(wurzel, repo, pfad, name)
    if ziel is not None:
        rel = _relativ_zu(str(ziel), wurzel)
        if rel:
            return rel
    return pfad or "?"


_EINTRAG = "id, art, repo, pfad, zeile, blob, lebendigkeit, tot, bereich, warnung"


def _eintraege(c: sqlite3.Connection, ids: list[str]) -> dict[str, tuple]:
    aus: dict[str, tuple] = {}
    for i in range(0, len(ids), 500):
        teil = ids[i:i + 500]
        for z in c.execute(f"SELECT {_EINTRAG} FROM eintrag WHERE id IN ({','.join('?' * len(teil))})", teil):
            aus[z[0]] = z
    return aus


def _kontext_dateien(c: sqlite3.Connection, wurzel: Path, name: str | None,
                     pfade: list[str]) -> list[tuple[str, tuple, str | None, str]]:
    """Transkript-Pfade -> Index-Dateien in Reihenfolge des Verlaufs: (Datei-Id, Eintrag, Worktree-Name, Hauptpfad).

    Abbildung wie `_pfad_im_arbeitsbaum`, nur rueckwaerts: der laengste Repo-Praefix gewinnt (Submodule heissen
    `<name>/<pfad im Elternrepo>`). Nur Dateien mit Eintrag `datei` oder `doku_seite` zaehlen."""
    if not name:
        return []
    plan: list[tuple[list[tuple[str, str]], str | None, str]] = []
    alle: list[str] = []
    for p in pfade:
        haupt, wt = hauptpfad(p)
        rel = _relativ_zu(haupt, wurzel)
        if rel is None:
            continue
        teile = rel.split("/")
        kandidaten: list[tuple[str, str]] = []  # (Datei-Id, Eintrags-Id) vom tiefsten Repo zum aeussersten
        for k in range(len(teile) - 1, -1, -1):
            repo = name if k == 0 else name + "/" + "/".join(teile[:k])
            rest = "/".join(teile[k:])
            for art in ("datei", "doku_seite"):
                kandidaten.append((f"datei:{repo}:{rest}", f"{art}:{repo}:{rest}"))
        plan.append((kandidaten, wt, rel))
        alle += [e for _, e in kandidaten]
    if not plan:
        return []
    gefunden = _eintraege(c, list(dict.fromkeys(alle)))
    aus: list[tuple[str, tuple, str | None, str]] = []
    gesehen: set[str] = set()
    for kandidaten, wt, rel in plan:
        for d, e in kandidaten:
            if e in gefunden:
                if d not in gesehen:
                    gesehen.add(d)
                    aus.append((d, gefunden[e], wt, rel))
                break
    return aus


def _nachbarn(c: sqlite3.Connection, datei: str, ausschluss: set[str]) -> list[str]:
    """1-Hop-Nachbardateien ueber `importiert`/`ruft_auf` (beide Richtungen, Datei und ihre Symbole).

    Rangfolge: Zahl sicherer Kanten, dann aller Kanten, dann Id; hoechstens NACHBARN_JE_DATEI."""
    repo_pfad = datei.split(":", 1)[1]
    unten, oben = f"symbol:{repo_pfad}#", f"symbol:{repo_pfad}$"  # alle Symbole der Datei ('$' folgt auf '#')
    arten = ",".join("?" * len(_NACHBAR_ARTEN))
    zeilen: list[tuple[str, int]] = []
    for sql, args in (
            (f"SELECT nach, sicher FROM kante WHERE von=? AND art IN ({arten}) LIMIT ?",
             (datei, *_NACHBAR_ARTEN, _KANTEN_JE_ABFRAGE)),
            (f"SELECT von, sicher FROM kante WHERE nach=? AND art IN ({arten}) LIMIT ?",
             (datei, *_NACHBAR_ARTEN, _KANTEN_JE_ABFRAGE)),
            (f"SELECT nach, sicher FROM kante WHERE von>=? AND von<? AND art IN ({arten}) LIMIT ?",
             (unten, oben, *_NACHBAR_ARTEN, _KANTEN_JE_ABFRAGE)),
            (f"SELECT von, sicher FROM kante WHERE nach>=? AND nach<? AND art IN ({arten}) LIMIT ?",
             (unten, oben, *_NACHBAR_ARTEN, _KANTEN_JE_ABFRAGE))):
        zeilen += c.execute(sql, args).fetchall()
    zahl: dict[str, list[int]] = {}
    for anderes, sicher in zeilen:
        d = _datei_schluessel(anderes)
        if d is None or d == datei or d in ausschluss:
            continue
        z = zahl.setdefault(d, [0, 0])
        z[0] += 1 if sicher else 0
        z[1] += 1
    return sorted(zahl, key=lambda d: (-zahl[d][0], -zahl[d][1], d))[:NACHBARN_JE_DATEI]


def kontextblock(cwd: Path, prompt: str, *, frist_s: float = FRIST_S, transkript: Path | None = None,
                 transkript_ende: int | None = None) -> str | None:
    """Kontextblock zum Prompt oder None (keine Regeldatei, kein Index, nichts Passendes, Frist ueberschritten).

    `transkript` ist der `transcript_path` des Hooks; `transkript_ende` begrenzt den Verlauf auf die Bytes davor
    (Messung: Offset der Prompt-Zeile). Das Lesen des Transkripts zaehlt zur Frist."""
    start = time.monotonic()

    def pruefen() -> None:
        if time.monotonic() - start >= frist_s:
            raise _Frist

    try:
        return _block(Path(cwd), prompt, pruefen, transkript, transkript_ende)
    except _Frist:
        return None
    except sqlite3.OperationalError:  # u. a. vom Fortschritts-Handler abgebrochene Abfrage; auch Gesperrt/Defekt
        return None


def _block(cwd: Path, prompt: str, pruefen: Callable[[], None], transkript: Path | None,
           transkript_ende: int | None) -> str | None:
    pruefen()
    wurzeln = repo_wurzeln(cwd)
    if not wurzeln:
        return None
    tokens = _tokens(prompt)
    if not tokens and transkript is None:
        return None
    wurzel: Path | None = None
    stand: Path | None = None
    for w in reversed(wurzeln):  # innerstes Repo mit Stand und Index
        s = stand_fuer(w)
        if s is not None and (s.parent / "suche.sqlite").is_file():
            wurzel, stand = w, s
            break
    if wurzel is None or stand is None:
        return None
    pruefen()
    pfade = letzte_dateien(Path(transkript), ende=transkript_ende) if transkript is not None else []
    pruefen()

    graph = _graph_meta(stand, pruefen)
    name = graph.get("name")
    with closing(_ro(stand.parent / "suche.sqlite", pruefen)) as c:
        meta = dict(c.execute("SELECT k, v FROM meta"))
        commit = meta.get("commit", "")
        # Ruling U6: Index aelter/anders als der Stand -> stille Ausgabe.
        if not commit or commit != graph.get("commit") or meta.get("gemessen_um", "") != graph.get("gemessen_um", ""):
            return None
        pruefen()
        # 1. Arbeitsbereich: Kontextdateien (neueste zuerst) und ihre Nachbarn
        kontext = _kontext_dateien(c, wurzel, name, pfade)
        pruefen()
        kontext_ids = {d for d, *_ in kontext}
        nachbar_von: dict[str, str] = {}
        for d, eintrag, _wt, _rel in kontext[:NACHBAR_QUELLEN]:
            if eintrag[1] != "datei":
                continue
            for n in _nachbarn(c, d, kontext_ids | set(nachbar_von)):
                if len(nachbar_von) >= NACHBARN_GESAMT:
                    break
                nachbar_von[n] = d
            pruefen()
        arbeitsbereich = [d for d, *_ in kontext] + list(nachbar_von)
        rang_a = {d: r for r, d in enumerate(arbeitsbereich, start=1)}
        eintrag_von: dict[str, tuple] = {d: e for d, e, *_ in kontext}
        eintrag_von.update({k: v for k, v in _eintraege(c, [n for n in nachbar_von if n not in eintrag_von]).items()})

        # 2. Prompt-Treffer: FTS wie bisher, dann Datei-Dedup und Tot-Daempfung (Spiegel von suche.abfrage)
        zeilen = c.execute(
            f"SELECT e.id, e.art, e.repo, e.pfad, e.zeile, e.blob, e.lebendigkeit, e.tot, e.bereich, e.warnung,"
            f" bm25(volltext), volltext.name, volltext.pfad, substr(volltext.text, 1, {_TEXT_MAX})"
            " FROM volltext JOIN eintrag e ON e.id = volltext.id WHERE volltext MATCH ?"
            " ORDER BY bm25(volltext), e.id LIMIT ?", (_fts(tokens), _KANDIDATEN)).fetchall() if tokens else []
    pruefen()

    such = {_ohne_akzent(t) for t in tokens}
    beste: dict[str, tuple[float, int, tuple]] = {}  # Datei -> (Wert, Rang, Eintragszeile)
    getroffen: dict[str, set[str]] = {}
    bester_bm25 = 0.0
    for rang, z in enumerate(zeilen, start=1):
        i, art, pfad, tot, bm = z[0], z[1], z[3], z[7], z[10]
        if art not in ("datei", "symbol", "doku_seite") or not pfad:
            continue
        d = _datei_schluessel(i)
        if d is None:
            continue
        bester_bm25 = min(bester_bm25, float(bm))
        wert = 1.0 / (RRF_K + rang) * (TOT_FAKTOR if tot == "sicher_tot" else 1.0)
        if d not in beste or wert > beste[d][0]:
            beste[d] = (wert, rang, z[:10])
        text = " ".join(_ohne_akzent(x) for x in (z[11], z[12], z[13]) if x)
        getroffen.setdefault(d, set()).update(s for s in such if _trifft(s, text))
        pruefen()
    prompt_liste = sorted(beste, key=lambda d: (-beste[d][0], beste[d][1]))
    stark = bester_bm25 <= BM25_STARK or any(len(t) >= MIN_TOKENS_STARK for t in getroffen.values())

    # 4. Schweigeschwelle
    if not kontext and not stark:
        return None

    # 3. Fusion (RRF) aus Prompt-Liste und Arbeitsbereich
    rang_p = {d: r for r, d in enumerate(prompt_liste, start=1)}
    def rrf(d: str) -> float:
        return sum(1.0 / (RRF_K + r[d]) for r in (rang_p, rang_a) if d in r)

    gezeigt_p = [d for d in prompt_liste if stark or d in rang_a]
    gezeigt_p.sort(key=lambda d: (-rrf(d), d))

    frisch = commit == _git_head(wurzel)
    try:
        max_kb = int(regeln_lesen((wurzel / REGELDATEI).read_text(encoding="utf-8")).get("max_dateigroesse_kb",
                                                                                          STANDARD_MAX_KB))
    except (OSError, ValueError):
        max_kb = STANDARD_MAX_KB
    cache: dict[Path, tuple[str, str] | None] = {}

    def datei_frisch(repo: str | None, pfad: str | None, blob: str | None) -> bool:
        """Frische-Regel: Blob-SHA im Arbeitsbaum (roh oder CRLF->LF) == Index-Blob. Zu grosse Dateien gelten
        als geaendert, ohne gelesen zu werden."""
        ziel = _pfad_im_arbeitsbaum(wurzel, repo, pfad, name)
        if ziel is None or not blob:
            return False
        if ziel not in cache:
            try:
                cache[ziel] = None if ziel.stat().st_size > max_kb * 1024 else blob_sha(ziel)
            except OSError:
                cache[ziel] = None
        shas = cache[ziel]
        return shas is not None and blob in shas

    def rolle(d: str) -> str:
        if d in kontext_ids:
            return "zuletzt bearbeitet"
        if d in nachbar_von:
            return "Nachbar von " + _anzeige(wurzel, *_repo_pfad(nachbar_von[d]), name)
        return ""

    stellen: list[str] = []
    doku: list[str] = []
    geaendert: list[str] = []
    bereiche: list[str] = []
    for d in gezeigt_p:
        pruefen()
        i, art, repo, pfad, zeile, blob, leb, tot, bereich, warnung = beste[d][2]
        zeig = _anzeige(wurzel, repo, pfad, name)
        ort = zeig if "#" not in i else f"{i.split('#', 1)[1][:120]} ({zeig})"
        r = rolle(d)
        if art == "doku_seite":
            if len(doku) >= MAX_DOKU:
                continue
            if datei_frisch(repo, pfad, blob):
                doku.append(f"- {zeig}" + (f" – {r}" if r else ""))
            elif len(geaendert) < MAX_GEAENDERT:
                geaendert.append(f"- {ort} – seit Index geändert")
            continue
        if not datei_frisch(repo, pfad, blob):
            if len(geaendert) < MAX_GEAENDERT:
                geaendert.append(f"- {ort} – seit Index geändert")
            continue
        if len(stellen) >= MAX_STELLEN:
            continue
        text = f"- {zeig}:{zeile}" if zeile else f"- {zeig}"
        if "#" in i:
            text += " " + i.split("#", 1)[1][:120]
        if leb is not None:
            text += f" [Lebendigkeit {float(leb):.2f}]"
        w = _warnungen(tot, warnung)
        if w:
            text += " " + w
        if r:
            text += f" – {r}"
        stellen.append(text)
        if bereich and bereich not in bereiche and len(bereiche) < MAX_BEREICHE:
            bereiche.append(bereich)

    # Abschnitt "Zuletzt bearbeitet und Nachbarn": nur Arbeitsbereich, kein Prompt-Treffer
    kontext_zeilen: list[str] = []
    n_kontext = n_nachbar = 0
    regel_cache: dict[str, list[str]] = {}
    for d in arbeitsbereich:
        pruefen()
        if d in rang_p or d not in eintrag_von:
            continue
        ist_kontext = d in kontext_ids
        if (n_kontext if ist_kontext else n_nachbar) >= (MAX_KONTEXT if ist_kontext else MAX_NACHBARN_ZEILEN):
            continue
        i, art, repo, pfad, _zeile, blob, leb, _tot, bereich, warnung = eintrag_von[d]
        zeig = _anzeige(wurzel, repo, pfad, name)
        if ist_kontext:
            n_kontext += 1
        else:
            n_nachbar += 1
        if not datei_frisch(repo, pfad, blob):
            kontext_zeilen.append(f"- {zeig} – seit Index geändert")
            continue
        text = f"- {zeig} – {rolle(d)}"
        if bereich:
            text += f"; Bereich {bereich}"
            if bereich not in bereiche and len(bereiche) < MAX_BEREICHE:
                bereiche.append(bereich)
        arten = regel_cache.get(zeig)
        if arten is None:
            arten = _regel_arten(wurzeln, wurzel / zeig)
            regel_cache[zeig] = arten
        if arten:
            text += "; Regel " + ", ".join(arten)
        if leb is not None:
            text += f" [Lebendigkeit {float(leb):.2f}]"
        w = _warnungen(None, warnung)
        if w:
            text += " " + w
        kontext_zeilen.append(text[:400])

    warnungen: list[str] = []
    for d, _e, wt, rel in kontext:
        if wt is None or len(warnungen) >= MAX_WARNUNGEN or any(f"zweiter Checkout {wt}:" in x for x in warnungen):
            continue
        warnungen.append(f"- ⚠ zweiter Checkout {wt}: bearbeitet wird im Worktree, der Index kennt nur den"
                         f" committeten Hauptbaum ({rel}); Stellen und Zeilen können abweichen")

    # Ruling U6: ohne frischen Treffer (nur Namen "seit Index geändert") kein Block
    if not stellen and not doku and not any(" – seit Index geändert" not in z for z in kontext_zeilen):
        return None

    aus = [f"Selbstbild (Commit {commit[:7]}, Index {'frisch' if frisch else 'veraltet'})"]
    if warnungen:
        aus += ["Warnungen:", *warnungen]
    if bereiche:
        aus.append("Bereiche: " + ", ".join(bereiche))
    if stellen:
        aus += ["Stellen:", *stellen]
    if doku:
        aus += ["Doku:", *doku]
    if kontext_zeilen:
        aus += ["Zuletzt bearbeitet und Nachbarn:", *kontext_zeilen]
    if geaendert:
        aus += ["Seit Index geändert:", *geaendert]
    regeln: list[str] = []
    for w in wurzeln:
        try:
            r = regeln_lesen((w / REGELDATEI).read_text(encoding="utf-8"))
        except OSError:
            continue
        regeln += [_regel_zeile(x) for x in r.get("struktur", [])]
    if regeln:
        aus += ["Strukturregeln:", *regeln]
    pruefen()

    block = ""
    for z in aus:
        neu = (block + "\n" if block else "") + z
        if len(neu) > MAX_ZEICHEN:
            break
        block = neu
    return block


def _repo_pfad(d: str) -> tuple[str | None, str | None]:
    teile = d.split(":", 2)
    return (teile[1], teile[2]) if len(teile) == 3 else (None, None)


def _regel_arten(wurzeln: tuple[Path, ...], datei: Path) -> list[str]:
    """Arten der Strukturregeln, deren `ort` die Datei umfasst (Ruling U1: `struktur.regeln_fuer`, nur stdlib)."""
    from .. import struktur

    arten: list[str] = []
    for w in wurzeln:
        try:
            for r in struktur.regeln_fuer(w, str(datei)):
                a = str(r.get("art", "?"))
                if a not in arten:
                    arten.append(a)
        except (OSError, ValueError, re.error):
            continue
    return arten[:2]


def main() -> int:
    start = time.monotonic()
    try:
        eingabe = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
        prompt, cwd = eingabe.get("prompt"), eingabe.get("cwd")
        tr = eingabe.get("transcript_path")
        transkript = Path(tr) if isinstance(tr, str) and tr else None
        if isinstance(prompt, str) and isinstance(cwd, str):
            rest = FRIST_S - (time.monotonic() - start)
            ergebnis: list[str | None] = []

            def arbeit() -> None:
                try:
                    ergebnis.append(kontextblock(Path(cwd), prompt, frist_s=rest, transkript=transkript))
                except Exception:  # noqa: BLE001
                    pass

            # Harter Stopp: ein Daemon-Thread, auf den hoechstens bis zur Frist gewartet wird (blockierende
            # Dateizugriffe, die keinen Fristtest erreichen, koennen den Hook so nicht aufhalten).
            t = threading.Thread(target=arbeit, daemon=True)
            t.start()
            t.join(max(0.0, rest))
            block = ergebnis[0] if ergebnis and not t.is_alive() else None
            if block and time.monotonic() - start < FRIST_S:
                antwort = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": block}}
                sys.stdout.buffer.write(json.dumps(antwort).encode("ascii"))
                sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001 - der Hook schweigt bei jedem Fehler
        pass
    if os.environ.get("SELBSTBILD_HOOK_ZEIT") == "1":
        sys.stderr.write(f"hook {(time.monotonic() - start) * 1000:.0f} ms\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
