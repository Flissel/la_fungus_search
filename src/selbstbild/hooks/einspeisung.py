"""UserPromptSubmit-Hook: speist den passenden Code-Kontext vor jedem Prompt ein (Stufe 3, T7).

Laeuft bei JEDEM Prompt, darum: nur Standardbibliothek, hoechstens 1,5 s (eigene Frist, danach keine Ausgabe),
hoechstens 8.000 Zeichen (~2.000 Tokens), bei jedem Fehler stille Ausgabe und Exit 0, schreibt nie Dateien.
Der Prompt wird weder ausgegeben noch protokolliert.

Bewusste Dopplung (Ruling U3): Tokenisierung und FTS-Abfrage spiegeln `suche.abfrage`, die Index-Frische
spiegelt `suche.index.index_aktuell`. Importiert werden `suche.*`/`analyse.*` hier nicht, weil sie igraph/numpy
nachladen und den Start um Sekunden verlaengern wuerden. `test_keine_schweren_importe` sichert das ab.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from . import REGELDATEI, blob_sha, regeln_lesen, repo_wurzeln, stand_fuer

FRIST_S = 1.5
MAX_ZEICHEN = 8000
MAX_BEREICHE = 3
MAX_STELLEN = 8
MAX_DOKU = 3
MAX_TOKENS = 12
_KANDIDATEN = 40

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
    aus: list[str] = []
    for wort in _NICHT_WORT.split(_GRENZE.sub(" ", text)):
        w = wort.lower()
        if len(w) >= 3 and w not in _STOPP and w not in aus:
            aus.append(w)
    return tuple(aus[:MAX_TOKENS])


def _fts(tokens: tuple[str, ...]) -> str:
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def _git_head(wurzel: Path) -> str | None:
    """HEAD-Commit aus den Git-Dateien (kein Prozess); None, wenn irgendetwas unklar ist."""
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


def _graph_meta(graph: Path) -> dict[str, str]:
    """Schluessel/Werte von graph.sqlite sind JSON-kodiert (Speicher._dumps)."""
    with closing(_ro(graph)) as c:
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


def kontextblock(cwd: Path, prompt: str, *, frist_s: float = FRIST_S) -> str | None:
    """Kontextblock zum Prompt oder None (keine Regeldatei, kein Index, kein Treffer, Frist ueberschritten)."""
    start = time.monotonic()

    def pruefen() -> None:
        if time.monotonic() - start >= frist_s:
            raise _Frist

    try:
        return _block(Path(cwd), prompt, pruefen)
    except _Frist:
        return None


def _block(cwd: Path, prompt: str, pruefen: Callable[[], None]) -> str | None:
    pruefen()
    wurzeln = repo_wurzeln(cwd)
    if not wurzeln:
        return None
    tokens = _tokens(prompt)
    if not tokens:
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

    graph = _graph_meta(stand)
    with closing(_ro(stand.parent / "suche.sqlite")) as c:
        meta = dict(c.execute("SELECT k, v FROM meta"))
        pruefen()
        zeilen = c.execute(
            "SELECT e.id, e.art, e.pfad, e.zeile, e.blob, e.lebendigkeit, e.tot, e.bereich, e.warnung, e.repo"
            " FROM volltext JOIN eintrag e ON e.id = volltext.id WHERE volltext MATCH ?"
            " ORDER BY bm25(volltext), e.id LIMIT ?", (_fts(tokens), _KANDIDATEN)).fetchall()
    pruefen()
    if not zeilen:
        return None

    commit = meta.get("commit", "")
    frisch = (bool(commit) and commit == graph.get("commit")
              and meta.get("gemessen_um", "") == graph.get("gemessen_um", "")
              and commit == _git_head(wurzel))
    name = graph.get("name")

    cache: dict[Path, tuple[str, str] | None] = {}
    stellen: list[str] = []
    doku: list[str] = []
    bereiche: list[str] = []
    gesehen: set[str] = set()
    for (i, art, pfad, zeile, blob, leb, tot, bereich, warnung, repo) in zeilen:
        pruefen()
        if i in gesehen:
            continue
        gesehen.add(i)
        if art == "doku_seite":
            if pfad and len(doku) < MAX_DOKU:
                doku.append(f"- {pfad}")
            continue
        if art not in ("datei", "symbol") or len(stellen) >= MAX_STELLEN or not pfad:
            continue
        ziel = _pfad_im_arbeitsbaum(wurzel, repo, pfad, name)
        shas = None
        if ziel is not None and blob:
            if ziel not in cache:
                try:
                    cache[ziel] = blob_sha(ziel)
                except OSError:
                    cache[ziel] = None
            shas = cache[ziel]
        ort = i.split("#", 1)[1][:120] if "#" in i else pfad
        if shas is None or blob not in shas:
            stellen.append(f"- {ort} ({pfad}) – seit Index geändert")
            continue
        text = f"- {pfad}:{zeile}" if zeile else f"- {pfad}"
        if "#" in i:
            text += f" {ort}"
        if leb is not None:
            text += f" [Lebendigkeit {float(leb):.2f}]"
        w = _warnungen(tot, warnung)
        if w:
            text += " " + w
        stellen.append(text)
        if bereich and bereich not in bereiche and len(bereiche) < MAX_BEREICHE:
            bereiche.append(bereich)
    if not stellen and not doku:
        return None

    aus = [f"Selbstbild (Commit {commit[:7]}, Index {'frisch' if frisch else 'veraltet'})"]
    if bereiche:
        aus.append("Bereiche: " + ", ".join(bereiche))
    if stellen:
        aus += ["Stellen:", *stellen]
    if doku:
        aus += ["Doku:", *doku]
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


def main() -> int:
    start = time.monotonic()
    try:
        eingabe = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
        prompt, cwd = eingabe.get("prompt"), eingabe.get("cwd")
        if isinstance(prompt, str) and isinstance(cwd, str):
            rest = FRIST_S - (time.monotonic() - start)
            block = kontextblock(Path(cwd), prompt, frist_s=rest)
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
