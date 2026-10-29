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
  nicht aus dem Arbeitsbaum. Bis T8 (`struktur.regeln_fuer`) sind es die rohen Strukturregeln (U1).
- Antworten enthalten nie Env-Werte; Env-Knoten tragen im Graphen nur Namen.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Literal

from ._glob import glob_regex
from .analyse.lauf import _stand_meta
from .quelle import _regeln_aus_commit
from .suche.abfrage import _fts_anfrage, anfrage_tokens, nachbarn, suchen
from .suche.index import index_aktuell, index_pfad
from .vermessen import stand_finden

WERKZEUGE: tuple[str, ...] = (
    "selbstbild_suchen", "selbstbild_nachbarn", "selbstbild_bereich", "selbstbild_regeln",
    "selbstbild_lebendigkeit", "selbstbild_tot", "selbstbild_doku",
)

Antwort = dict[str, Any]


def wurzeln() -> list[Path]:
    roh = os.environ.get("SELBSTBILD_WURZELN", "")
    return [Path(t.strip()) for t in roh.split(";") if t.strip()]


def wurzel_waehlen(wurzel: str | None, pfad: str | None) -> Path | None:
    """Ausdruecklich genannte Wurzel; sonst die erste konfigurierte, die `pfad` enthaelt; sonst die erste."""
    if wurzel:
        return Path(wurzel)
    alle = wurzeln()
    if pfad:
        ziel = Path(pfad)
        if ziel.is_absolute():
            for w in alle:
                try:
                    ziel.resolve().relative_to(w.resolve())
                except ValueError:
                    continue
                return w
    return alle[0] if alle else None


def _lage(wurzel: Path) -> tuple[Path, Antwort] | Antwort:
    """(Stand-Ordner, stand-Block) oder `{"fehler": ...}`."""
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
    except (OSError, sqlite3.DatabaseError) as e:
        return {"fehler": f"Stand nicht lesbar ({type(e).__name__}); Index neu bauen: python -m selbstbild index {wurzel}"}


def _lesen(stand: Path, sql: str, args: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    ziel = index_pfad(stand)
    with closing(sqlite3.connect(f"{ziel.resolve().as_uri()}?mode=ro", uri=True)) as c:
        return c.execute(sql, args).fetchall()


def w_suchen(wurzel: Path, anfrage: str, top_k: int = 10) -> Antwort:
    def tun(stand: Path) -> Antwort:
        return {"treffer": [
            {"id": t.id, "art": t.art, "pfad": t.pfad, "zeile": t.zeile, "wert": round(t.wert, 6),
             "herkunft": list(t.herkunft), "lebendigkeit": t.lebendigkeit, "tot": t.tot,
             "bereich": t.bereich, "warnung": t.warnung}
            for t in suchen(stand, anfrage, max(1, top_k))]}
    return _mit_stand(wurzel, tun)


def w_nachbarn(wurzel: Path, id: str, richtung: Literal["ein", "aus", "beide"] = "beide") -> Antwort:
    if richtung not in ("ein", "aus", "beide"):
        return {"fehler": "richtung muss ein, aus oder beide sein"}
    return _mit_stand(wurzel, lambda stand: {"kanten": [
        {"von": v, "nach": n, "sicher": s} for v, n, s in nachbarn(stand, id, richtung)]})


def _datei_eintrag(stand: Path, id_oder_pfad: str) -> tuple[Any, ...] | None:
    zeilen = _lesen(stand, "SELECT id, pfad, bereich, lebendigkeit FROM eintrag WHERE id=? OR (art='datei' AND pfad=?)"
                    " ORDER BY id LIMIT 1", (id_oder_pfad, id_oder_pfad.replace("\\", "/")))
    return zeilen[0] if zeilen else None


def w_bereich(wurzel: Path, id_oder_pfad: str) -> Antwort:
    def tun(stand: Path) -> Antwort:
        z = _datei_eintrag(stand, id_oder_pfad)
        if z is None:
            return {"fehler": f"unbekannt im Index: {id_oder_pfad}"}
        return {"id": z[0], "pfad": z[1], "bereich": z[2]}
    return _mit_stand(wurzel, tun)


def w_lebendigkeit(wurzel: Path, pfad: str) -> Antwort:
    def tun(stand: Path) -> Antwort:
        z = _datei_eintrag(stand, pfad)
        if z is None:
            return {"fehler": f"unbekannt im Index: {pfad}"}
        return {"id": z[0], "pfad": z[1], "lebendigkeit": z[3]}
    return _mit_stand(wurzel, tun)


def w_tot(wurzel: Path, stufe: str = "sicher_tot", praefix: str = "", limit: int = 100) -> Antwort:
    def tun(stand: Path) -> Antwort:
        if limit <= 0:
            return {"tot": []}
        zeilen = _lesen(stand, "SELECT id, pfad, zeile, tot FROM eintrag WHERE tot=? AND substr(coalesce(pfad,''),1,?)=?"
                        " ORDER BY id LIMIT ?", (stufe, len(praefix), praefix, limit))
        return {"tot": [{"id": i, "pfad": p, "zeile": z, "stufe": t} for i, p, z, t in zeilen]}
    return _mit_stand(wurzel, tun)


def w_regeln(wurzel: Path, pfad: str) -> Antwort:
    """Strukturregeln des gemessenen Commits, deren `ort` den Pfad umschliesst (roh, bis T8)."""
    def tun(stand: Path) -> Antwort:
        commit, _ = _stand_meta(stand)
        regeln = _regeln_aus_commit(Path(wurzel), commit or "HEAD")
        norm = pfad.replace("\\", "/").lstrip("./")
        treffer = [r.model_dump() for r in regeln.struktur
                   if glob_regex(r.ort.rstrip("/") + "/**").match(norm)]
        return {"regeln": treffer}
    return _mit_stand(wurzel, tun)


def w_doku(wurzel: Path, id: str) -> Antwort:
    """Doku-Seiten, die den Namen erwaehnen (FTS5 ueber den committeten Text)."""
    def tun(stand: Path) -> Antwort:
        name = id.rpartition("#")[2] if "#" in id else id.rpartition(":")[2] or id
        tokens = anfrage_tokens(name.rpartition("/")[2])
        if not tokens:
            return {"seiten": []}
        zeilen = _lesen(stand, "SELECT e.id, e.pfad FROM volltext v JOIN eintrag e ON e.id=v.id"
                        " WHERE volltext MATCH ? AND e.art='doku_seite' ORDER BY bm25(volltext), e.id LIMIT 10",
                        (_fts_anfrage(tokens),))
        return {"seiten": [{"id": i, "pfad": p} for i, p in zeilen]}
    return _mit_stand(wurzel, tun)


def _zustand(w: Path) -> str:
    lage = _lage(w)
    return lage["fehler"] if isinstance(lage, dict) else f"aktuell (commit {lage[1]['commit']})"


def _selftest() -> int:
    alle = wurzeln()
    print("wurzeln:", "; ".join(str(w) for w in alle) or "(keine)")
    for w in alle:
        try:
            print(f"index {w}: {_zustand(w)}")
        except (OSError, sqlite3.DatabaseError) as e:
            print(f"index {w}: nicht lesbar ({type(e).__name__})")
    print("werkzeuge:", ", ".join(WERKZEUGE))
    return 0


def _kein_wurzel() -> Antwort:
    return {"fehler": "keine Wurzel: SELBSTBILD_WURZELN setzen oder wurzel angeben"}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--selftest" in args:
        return _selftest()
    from mcp.server.fastmcp import FastMCP  # erst hier: schwerer Import nur im Serverbetrieb

    server = FastMCP("selbstbild")

    @server.tool()
    def selbstbild_suchen(anfrage: str, top_k: int = 10, wurzel: str | None = None) -> Antwort:
        """Sucht Dateien, Symbole und Doku im Selbstbild-Index (Stichwort + Struktur)."""
        w = wurzel_waehlen(wurzel, None)
        return w_suchen(w, anfrage, top_k) if w else _kein_wurzel()

    @server.tool()
    def selbstbild_nachbarn(id: str, richtung: str = "beide", wurzel: str | None = None) -> Antwort:
        """Kanten um einen Knoten (ein, aus, beide) mit Sicherheit."""
        w = wurzel_waehlen(wurzel, None)
        return w_nachbarn(w, id, richtung) if w else _kein_wurzel()  # type: ignore[arg-type]

    @server.tool()
    def selbstbild_bereich(id_oder_pfad: str, wurzel: str | None = None) -> Antwort:
        """Bereich (Community) einer Datei."""
        w = wurzel_waehlen(wurzel, id_oder_pfad)
        return w_bereich(w, id_oder_pfad) if w else _kein_wurzel()

    @server.tool()
    def selbstbild_regeln(pfad: str, wurzel: str | None = None) -> Antwort:
        """Strukturregeln (committet), die fuer den Pfad gelten."""
        w = wurzel_waehlen(wurzel, pfad)
        return w_regeln(w, pfad) if w else _kein_wurzel()

    @server.tool()
    def selbstbild_lebendigkeit(pfad: str, wurzel: str | None = None) -> Antwort:
        """Lebendigkeit (0..1) einer Datei."""
        w = wurzel_waehlen(wurzel, pfad)
        return w_lebendigkeit(w, pfad) if w else _kein_wurzel()

    @server.tool()
    def selbstbild_tot(stufe: str = "sicher_tot", praefix: str = "", limit: int = 100,
                       wurzel: str | None = None) -> Antwort:
        """Tot-Kandidaten einer Stufe, optional unter einem Pfadpraefix."""
        w = wurzel_waehlen(wurzel, praefix)
        return w_tot(w, stufe, praefix, limit) if w else _kein_wurzel()

    @server.tool()
    def selbstbild_doku(id: str, wurzel: str | None = None) -> Antwort:
        """Doku-Seiten, die den Namen der id erwaehnen."""
        w = wurzel_waehlen(wurzel, None)
        return w_doku(w, id) if w else _kein_wurzel()

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
