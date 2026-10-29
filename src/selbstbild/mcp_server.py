"""MCP-Server `selbstbild`: Suche und Graph-Werkzeuge ueber den vermessenen Stand (Stufe 3, T6).

Start: `python -m selbstbild.mcp_server` (stdio) aus `.venv-selbstbild`; `--selftest` gibt Wurzeln,
Index-Zustand und Werkzeugnamen aus. Wurzeln kommen aus `SELBSTBILD_WURZELN` (mit `;` getrennt).

Entscheidungen:
- Jede Antwort ist ein dict mit `stand: {commit, index_aktuell}` oder `{"fehler": "<deutsche Meldung>"}`.
- `suche.sqlite` wird nur kurzlebig je Anfrage mit `mode=ro` geoeffnet (Windows sperrt offene Dateien, U4).
- Tot-Stufe, Bereich und Lebendigkeit kommen aus dem Index (`eintrag`), nicht aus der gespeicherten
  Analyse: der Server laedt so weder Graph noch Analyse. (`suche.index` zieht transitiv `igraph` ueber
  `analyse.lader`; `leidenalg` und `numpy` werden hier nicht geladen.)
- Regeln stammen aus dem committeten `selbstbild.yaml` des gemessenen Commits (wie die Vermessung, R21),
  nicht aus dem Arbeitsbaum. Welche Regel fuer einen Pfad gilt, entscheidet `struktur.regel_trifft`, derselbe
  Trefferbegriff wie beim Waechter.
- Antworten enthalten nie Env-Werte; Env-Knoten tragen im Graphen nur Namen.
"""
from __future__ import annotations

import os
import posixpath
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Literal

from .vermessen import stand_finden

Antwort = dict[str, Any]

# Windows vergleicht Pfade ohne Beachtung der Gross-/Kleinschreibung.
PFAD_OHNE_GROSSKLEIN: bool = os.name == "nt"


class _Meldung(Exception):
    """Klare deutsche Meldung an den Aufrufer (kein interner Fehler)."""


def wurzeln() -> list[Path]:
    roh = os.environ.get("SELBSTBILD_WURZELN", "")
    return [Path(t.strip()) for t in roh.split(";") if t.strip()]


def _teile(p: Path) -> list[str]:
    return [t.lower() if PFAD_OHNE_GROSSKLEIN else t for t in p.parts]


def _unter(ziel: Path, w: Path) -> bool:
    z, r = _teile(ziel.resolve()), _teile(w.resolve())
    return z[:len(r)] == r


def wurzel_waehlen(wurzel: str | None, pfad: str | None) -> Path | None:
    """Ausdruecklich genannte Wurzel; sonst die erste konfigurierte, die `pfad` enthaelt; sonst die erste."""
    if wurzel:
        return Path(wurzel)
    alle = wurzeln()
    if pfad:
        ziel = Path(pfad)
        if ziel.is_absolute():
            for w in alle:
                if _unter(ziel, w):
                    return w
    return alle[0] if alle else None


def relativ_zur_wurzel(wurzel: Path, eingabe: str) -> str:
    """Pfad relativ zur Wurzel mit `/`: absolute Pfade werden aufgeloest und (unter Windows ohne Beachtung der
    Schreibweise) relativ zur Wurzel genommen; relative Pfade werden nur normalisiert (`./`, Backslash, `..`).
    Ausserhalb der Wurzel: `_Meldung`. Die Wurzel selbst ergibt `""`."""
    if not eingabe:
        return ""
    ziel = Path(eingabe)
    if ziel.is_absolute():
        w = Path(wurzel).resolve()
        z = ziel.resolve()
        tz, tw = _teile(z), _teile(w)
        if tz[:len(tw)] != tw:
            raise _Meldung(f"Pfad liegt ausserhalb der Wurzel {wurzel}")
        rel = "/".join(z.parts[len(w.parts):])
    else:
        rel = eingabe.replace("\\", "/")
    norm = posixpath.normpath(rel)
    if norm == "." or norm == "":
        return ""
    if norm == ".." or norm.startswith("../") or norm.startswith("/"):
        raise _Meldung(f"Pfad liegt ausserhalb der Wurzel {wurzel}")
    return norm


def _gleich(a: str, b: str) -> bool:
    return a.lower() == b.lower() if PFAD_OHNE_GROSSKLEIN else a == b


def _beginnt(pfad: str, vorn: str) -> bool:
    """`pfad` ist `vorn` oder liegt darunter (Segmentgrenze)."""
    if vorn == "":
        return True
    p = pfad.lower() if PFAD_OHNE_GROSSKLEIN else pfad
    v = vorn.lower() if PFAD_OHNE_GROSSKLEIN else vorn
    return p == v or p.startswith(v + "/")


def _repo_und_pfad(stand: Path, rel: str) -> tuple[str | None, str]:
    """Wurzel-relativer Pfad -> (Repo, repo-relativer Pfad) im Index.

    Submodule heissen im Index `<wurzelrepo>/<submodulpfad>`; der laengste passende Praefix gewinnt.
    Ohne Treffer gehoert der Pfad zum Wurzel-Repo (`None`, falls dessen Name nicht ermittelbar ist)."""
    repos = [r for (r,) in _lesen(stand, "SELECT DISTINCT repo FROM eintrag WHERE repo IS NOT NULL")]
    if not repos:
        return None, rel
    from .suche.messung import wurzel_name
    kopf = wurzel_name(stand) or min(repos, key=lambda r: (len(r), r))
    beste = kopf
    for r in repos:
        if r.startswith(kopf + "/"):
            sub = r[len(kopf) + 1:]
            if _beginnt(rel, sub) and (len(sub) + len(kopf) + 1) > (len(beste)):
                beste = r
    sub = beste[len(kopf) + 1:] if beste != kopf else ""
    rest = rel[len(sub):].lstrip("/") if sub else rel
    return beste, rest


def _lage(wurzel: Path) -> tuple[Path, Antwort] | Antwort:
    """(Stand-Ordner, stand-Block) oder `{"fehler": ...}`."""
    from .analyse.lauf import _stand_meta
    from .suche.index import index_aktuell, index_pfad

    graph = stand_finden(Path(wurzel))
    if graph is None:
        return {"fehler": f"erst vermessen: python -m selbstbild vermessen {wurzel}"}
    stand = graph.parent
    if not index_pfad(stand).is_file():
        return {"fehler": f"Index fehlt: python -m selbstbild index {wurzel}"}
    if not index_aktuell(stand):
        return {"fehler": f"Index veraltet: python -m selbstbild index {wurzel}"}
    commit, _ = _stand_meta(stand)
    return stand, {"commit": commit, "index_aktuell": True}


def _mit_stand(wurzel: Path, fn: Callable[[Path], Antwort]) -> Antwort:
    try:
        lage = _lage(wurzel)
        if isinstance(lage, dict):
            return lage
        stand, block = lage
        return {"stand": block, **fn(stand)}
    except _Meldung as m:
        return {"fehler": str(m)}
    except (OSError, sqlite3.DatabaseError) as e:
        return {"fehler": f"Stand nicht lesbar ({type(e).__name__}); Index neu bauen: python -m selbstbild index {wurzel}"}
    except Exception as e:  # nie Meldungstext weitergeben: er koennte Pfade oder Werte tragen
        return {"fehler": f"{type(e).__name__}: interner Fehler"}


def _lesen(stand: Path, sql: str, args: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    from .suche.index import index_pfad

    ziel = index_pfad(stand)
    with closing(sqlite3.connect(f"{ziel.resolve().as_uri()}?mode=ro", uri=True)) as c:
        return c.execute(sql, args).fetchall()


def w_suchen(wurzel: Path, anfrage: str, top_k: int = 10) -> Antwort:
    def tun(stand: Path) -> Antwort:
        from .suche.abfrage import suchen

        return {"treffer": [
            {"id": t.id, "art": t.art, "pfad": t.pfad, "zeile": t.zeile, "wert": round(t.wert, 6),
             "herkunft": list(t.herkunft), "lebendigkeit": t.lebendigkeit, "tot": t.tot,
             "bereich": t.bereich, "warnung": t.warnung}
            for t in suchen(stand, anfrage, max(1, top_k))]}
    return _mit_stand(wurzel, tun)


def w_nachbarn(wurzel: Path, id: str, richtung: Literal["ein", "aus", "beide"] = "beide") -> Antwort:
    if richtung not in ("ein", "aus", "beide"):
        return {"fehler": "richtung muss ein, aus oder beide sein"}

    def tun(stand: Path) -> Antwort:
        from .suche.abfrage import nachbarn

        return {"kanten": [{"von": v, "nach": n, "sicher": s} for v, n, s in nachbarn(stand, id, richtung)]}
    return _mit_stand(wurzel, tun)


def _datei_eintrag(stand: Path, wurzel: Path, id_oder_pfad: str) -> tuple[Any, ...] | None:
    """Eintrag zu einer Knoten-id oder einem Pfad (relativ, absolut, Backslash, Schreibweise egal unter Windows)."""
    z = _lesen(stand, "SELECT id, pfad, bereich, lebendigkeit FROM eintrag WHERE id=? LIMIT 1", (id_oder_pfad,))
    if z:
        return z[0]
    repo, pfad = _repo_und_pfad(stand, relativ_zur_wurzel(wurzel, id_oder_pfad))
    zeilen = _lesen(stand, "SELECT id, pfad, bereich, lebendigkeit FROM eintrag WHERE art='datei' AND repo=? AND pfad"
                    + (" COLLATE NOCASE" if PFAD_OHNE_GROSSKLEIN else "") + "=? ORDER BY id LIMIT 1", (repo, pfad))
    return zeilen[0] if zeilen else None


def w_bereich(wurzel: Path, id_oder_pfad: str) -> Antwort:
    def tun(stand: Path) -> Antwort:
        z = _datei_eintrag(stand, wurzel, id_oder_pfad)
        if z is None:
            return {"fehler": f"unbekannt im Index: {id_oder_pfad}"}
        return {"id": z[0], "pfad": z[1], "bereich": z[2]}
    return _mit_stand(wurzel, tun)


def w_lebendigkeit(wurzel: Path, pfad: str) -> Antwort:
    def tun(stand: Path) -> Antwort:
        z = _datei_eintrag(stand, wurzel, pfad)
        if z is None:
            return {"fehler": f"unbekannt im Index: {pfad}"}
        return {"id": z[0], "pfad": z[1], "lebendigkeit": z[3]}
    return _mit_stand(wurzel, tun)


def w_tot(wurzel: Path, stufe: str = "sicher_tot", praefix: str = "", limit: int = 100) -> Antwort:
    """Tot-Kandidaten der Stufe unter einem Praefix (relativ oder absolut, relativ zur Wurzel)."""
    def tun(stand: Path) -> Antwort:
        rel = relativ_zur_wurzel(wurzel, praefix)
        if limit <= 0:
            return {"tot": []}
        from .suche.messung import wurzel_name
        repos = [r for (r,) in _lesen(stand, "SELECT DISTINCT repo FROM eintrag WHERE repo IS NOT NULL")]
        kopf = wurzel_name(stand) or (min(repos, key=lambda r: (len(r), r)) if repos else "")
        aus: list[dict[str, Any]] = []
        for i, repo, p, z, t in _lesen(stand, "SELECT id, repo, pfad, zeile, tot FROM eintrag WHERE tot=? ORDER BY id",
                                       (stufe,)):
            voll = (repo[len(kopf) + 1:] + "/" + p) if repo and repo.startswith(kopf + "/") and p else (p or "")
            if _beginnt(voll, rel):
                aus.append({"id": i, "pfad": voll, "zeile": z, "stufe": t})
                if len(aus) >= limit:
                    break
        return {"tot": aus}
    return _mit_stand(wurzel, tun)


def w_regeln(wurzel: Path, pfad: str) -> Antwort:
    """Strukturregeln des gemessenen Commits, die fuer den Pfad gelten: `ort` umfasst ihn (Wurzel-`ort` = ganzes
    Repo) oder ein `verboten`-Muster trifft ihn. Trefferbegriff = `struktur.regel_trifft` (wie der Waechter);
    gelesen wird aber die committete Regeldatei (R21), nicht der Arbeitsbaum (F5)."""
    def tun(stand: Path) -> Antwort:
        import yaml

        from . import struktur
        from .analyse.lauf import _stand_meta
        from .quelle import _regeln_aus_commit

        rel = relativ_zur_wurzel(wurzel, pfad)
        commit, _ = _stand_meta(stand)
        try:
            regeln = _regeln_aus_commit(Path(wurzel), commit or "HEAD")
        except (ValueError, yaml.YAMLError) as e:  # u. a. pydantic.ValidationError; Text enthaelt Inhalte
            raise _Meldung("Regeldatei ungueltig: selbstbild.yaml im gemessenen Commit") from e
        treffer = [d for d in (r.model_dump() for r in regeln.struktur)
                   if struktur.regel_trifft(d, rel, grossklein_egal=PFAD_OHNE_GROSSKLEIN)]
        return {"regeln": treffer}
    return _mit_stand(wurzel, tun)


def w_doku(wurzel: Path, id: str) -> Antwort:
    """Doku-Seiten, die den Namen erwaehnen (FTS5 ueber den committeten Text)."""
    def tun(stand: Path) -> Antwort:
        from .suche.abfrage import _fts_anfrage, anfrage_tokens

        name = id.rpartition("#")[2] if "#" in id else id.rpartition(":")[2] or id
        tokens = anfrage_tokens(name.rpartition("/")[2])
        if not tokens:
            return {"seiten": []}
        zeilen = _lesen(stand, "SELECT e.id, e.pfad FROM volltext v JOIN eintrag e ON e.id=v.id"
                        " WHERE volltext MATCH ? AND e.art='doku_seite' ORDER BY bm25(volltext), e.id LIMIT 10",
                        (_fts_anfrage(tokens),))
        return {"seiten": [{"id": i, "pfad": p} for i, p in zeilen]}
    return _mit_stand(wurzel, tun)


def _kein_wurzel() -> Antwort:
    return {"fehler": "keine Wurzel: SELBSTBILD_WURZELN setzen oder wurzel angeben"}


# --- MCP-Werkzeuge (eine Liste: Registrierung und WERKZEUGE kommen daraus) ---------------------------------

def selbstbild_suchen(anfrage: str, top_k: int = 10, wurzel: str | None = None) -> Antwort:
    """Sucht Dateien, Symbole und Doku im Selbstbild-Index (Stichwort + Struktur)."""
    w = wurzel_waehlen(wurzel, None)
    return w_suchen(w, anfrage, top_k) if w else _kein_wurzel()


def selbstbild_nachbarn(id: str, richtung: str = "beide", wurzel: str | None = None) -> Antwort:
    """Kanten um einen Knoten (ein, aus, beide) mit Sicherheit."""
    w = wurzel_waehlen(wurzel, None)
    return w_nachbarn(w, id, richtung) if w else _kein_wurzel()  # type: ignore[arg-type]


def selbstbild_bereich(id_oder_pfad: str, wurzel: str | None = None) -> Antwort:
    """Bereich (Community) einer Datei; Pfad relativ zur Wurzel oder absolut."""
    w = wurzel_waehlen(wurzel, id_oder_pfad)
    return w_bereich(w, id_oder_pfad) if w else _kein_wurzel()


def selbstbild_regeln(pfad: str, wurzel: str | None = None) -> Antwort:
    """Strukturregeln (committet), die fuer den Pfad gelten."""
    w = wurzel_waehlen(wurzel, pfad)
    return w_regeln(w, pfad) if w else _kein_wurzel()


def selbstbild_lebendigkeit(pfad: str, wurzel: str | None = None) -> Antwort:
    """Lebendigkeit (0..1) einer Datei."""
    w = wurzel_waehlen(wurzel, pfad)
    return w_lebendigkeit(w, pfad) if w else _kein_wurzel()


def selbstbild_tot(stufe: str = "sicher_tot", praefix: str = "", limit: int = 100,
                   wurzel: str | None = None) -> Antwort:
    """Tot-Kandidaten einer Stufe, optional unter einem Pfadpraefix."""
    w = wurzel_waehlen(wurzel, praefix)
    return w_tot(w, stufe, praefix, limit) if w else _kein_wurzel()


def selbstbild_doku(id: str, wurzel: str | None = None) -> Antwort:
    """Doku-Seiten, die den Namen der id erwaehnen."""
    w = wurzel_waehlen(wurzel, None)
    return w_doku(w, id) if w else _kein_wurzel()


_FUNKTIONEN: tuple[Callable[..., Antwort], ...] = (
    selbstbild_suchen, selbstbild_nachbarn, selbstbild_bereich, selbstbild_regeln,
    selbstbild_lebendigkeit, selbstbild_tot, selbstbild_doku,
)
WERKZEUGE: tuple[str, ...] = tuple(f.__name__ for f in _FUNKTIONEN)


def server_bauen() -> Any:
    from mcp.server.fastmcp import FastMCP  # erst hier: schwerer Import nur im Serverbetrieb

    server = FastMCP("selbstbild")
    for f in _FUNKTIONEN:
        server.tool()(f)
    return server


def _zustand(w: Path) -> str:
    lage = _lage(w)
    return lage["fehler"] if isinstance(lage, dict) else f"aktuell (commit {lage[1]['commit']})"


def _selftest() -> int:
    alle = wurzeln()
    print("wurzeln:", "; ".join(str(w) for w in alle) or "(keine)")
    for w in alle:
        try:
            print(f"index {w}: {_zustand(w)}")
        except Exception as e:  # Selbsttest meldet nur den Typ
            print(f"index {w}: nicht lesbar ({type(e).__name__})")
    print("werkzeuge:", ", ".join(WERKZEUGE))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--selftest" in args:
        return _selftest()
    server_bauen().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
