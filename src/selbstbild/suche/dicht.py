"""Optionale dichte Suche: Datei-Vektoren ueber einen lokalen HTTP-Embedder (F3), abgelegt in `suche.sqlite.vektor`.

Der Hook ruft nie ein Modell auf; nur `vektoren_bauen` und die Anfrage-Einbettung in `abfrage.suchen`
sprechen den Embedder an. Vektoren liegen normiert als float32-Bytes, gekennzeichnet mit dem Modellnamen.
Fehler tragen nur Typ und Status, nie Inhalte (Texte, Antworten). Nach `index_bauen` ist die Tabelle
`vektor` leer und wird erst hier befuellt: dieses Modul ist ausser `index_bauen` der einzige Schreiber.
"""
from __future__ import annotations

import json
import re
import sqlite3
import urllib.error
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .index import index_pfad

BATCH: int = 32
TEXT_MAX: int = 1000
KOPFZEILEN: int = 3
_ARTEN_MIT_VEKTOR = ("datei", "doku_seite")
_UEBERSCHRIFT = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")


@dataclass(frozen=True)
class Embedder:
    art: Literal["ollama", "dienst"]
    url: str
    modell: str


def _post(url: str, nutzlast: dict[str, object], timeout: float) -> object:
    anfrage = urllib.request.Request(url, data=json.dumps(nutzlast).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(anfrage, timeout=timeout) as antwort:
        return json.load(antwort)


def _fehler(e: Embedder, ursache: BaseException) -> RuntimeError:
    status = f", Status {ursache.code}" if isinstance(ursache, urllib.error.HTTPError) else ""
    return RuntimeError(f"Embedder {e.art} fehlgeschlagen: {type(ursache).__name__}{status}")


def _vektor(roh: object) -> list[float]:
    if not isinstance(roh, list) or not roh or not all(isinstance(x, (int, float)) for x in roh):
        raise ValueError("kein Vektor")
    return [float(x) for x in roh]


def einbetten(e: Embedder, texte: Sequence[str], *, timeout: float = 30.0) -> list[list[float]]:
    """Vektoren zu `texte` in Eingabereihenfolge, in Batches zu 32.

    ollama: `POST {url}/api/embed` mit `{"model","input":[...]}` -> `{"embeddings":[[...]]}`.
    dienst: `POST {url}/embed` mit `{"text"}` -> `{"vector":[...]}` (Protokoll von `HttpQueryEmbedder`,
    je Text eine Anfrage). Fehler -> `RuntimeError` nur mit Typ und Status.
    """
    basis = e.url.rstrip("/")
    aus: list[list[float]] = []
    try:
        for start in range(0, len(texte), BATCH):
            stapel = list(texte[start:start + BATCH])
            if e.art == "ollama":
                antwort = _post(f"{basis}/api/embed", {"model": e.modell, "input": stapel}, timeout)
                roh = antwort.get("embeddings") if isinstance(antwort, dict) else None
                if not isinstance(roh, list) or len(roh) != len(stapel):
                    raise ValueError("falsche Anzahl Vektoren")
                aus.extend(_vektor(v) for v in roh)
            else:
                for text in stapel:
                    antwort = _post(f"{basis}/embed", {"text": text}, timeout)
                    aus.append(_vektor(antwort.get("vector") if isinstance(antwort, dict) else None))
    except Exception as ursache:  # noqa: BLE001 - bewusst breit: Meldung traegt nur Typ und Status
        fehler = _fehler(e, ursache)
        if isinstance(ursache, urllib.error.HTTPError):
            ursache.close()  # Antwort-Socket freigeben
        raise fehler from None
    return aus


def _ro(stand: Path) -> sqlite3.Connection:
    ziel = index_pfad(stand)
    if not ziel.is_file():
        raise FileNotFoundError(f"kein Suchindex: {ziel}")
    return sqlite3.connect(f"{ziel.resolve().as_uri()}?mode=ro", uri=True)


def _text_aus(c: sqlite3.Connection, datei_id: str) -> str:
    z = c.execute("SELECT art, pfad FROM eintrag WHERE id=?", (datei_id,)).fetchone()
    if z is None:
        return ""
    art, pfad = z
    teile = [pfad or datei_id]
    if art == "datei":
        namen = [i.rpartition("#")[2] for (i,) in c.execute(
            "SELECT id FROM eintrag WHERE art='symbol' AND id LIKE ? ESCAPE '\\' ORDER BY id",
            (_like_praefix(datei_id),))]
        if namen:
            teile.append(" ".join(namen))
    else:
        text = (c.execute("SELECT text FROM volltext WHERE id=?", (datei_id,)).fetchone() or ("",))[0]
        zeilen = [m.group(1) for m in map(_UEBERSCHRIFT.match, text.splitlines()) if m]
        if not zeilen:
            zeilen = [z.strip() for z in text.splitlines() if z.strip()]
        teile.extend(zeilen[:KOPFZEILEN])
    return "\n".join(teile)[:TEXT_MAX]


def _like_praefix(datei_id: str) -> str:
    # symbol:<repo>:<pfad>#name  <->  datei:<repo>:<pfad>
    rest = datei_id.partition(":")[2]
    roh = f"symbol:{rest}#"
    return roh.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def datei_text(stand: Path, datei_id: str) -> str:
    """Pfad, Symbolnamen der Datei und die ersten 3 Ueberschriften bzw. Textzeilen (Doku), gekuerzt auf 1.000 Zeichen."""
    with closing(_ro(Path(stand))) as c:
        return _text_aus(c, datei_id)


def _normiert(v: Sequence[float]) -> np.ndarray | None:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return None if not np.isfinite(n) or n == 0.0 else (a / n).astype(np.float32)


def vektoren_bauen(stand: Path, e: Embedder) -> int:
    """Bettet alle Datei- und Doku-Eintraege ein und ersetzt `vektor` vollstaendig. Liefert die Anzahl."""
    stand = Path(stand)
    with closing(_ro(stand)) as c:
        ids = [i for (i,) in c.execute(
            f"SELECT id FROM eintrag WHERE art IN ({','.join('?' * len(_ARTEN_MIT_VEKTOR))}) ORDER BY id",
            _ARTEN_MIT_VEKTOR)]
        texte = [_text_aus(c, i) for i in ids]
    vektoren = einbetten(e, texte)
    zeilen: list[tuple[str, str, bytes]] = []
    for i, v in zip(ids, vektoren):
        n = _normiert(v)
        if n is not None:
            zeilen.append((i, e.modell, n.tobytes()))
    with closing(sqlite3.connect(str(index_pfad(stand)), isolation_level=None)) as c:
        c.execute("BEGIN")
        c.execute("DELETE FROM vektor")
        c.executemany("INSERT INTO vektor VALUES (?,?,?)", zeilen)
        c.execute("COMMIT")
    return len(zeilen)


def aehnlichste(stand: Path, anfrage_vektor: Sequence[float], top_k: int,
                modell: str) -> tuple[tuple[str, float], ...]:
    """Kosinus-Naechste unter den Vektoren des Modells, absteigend, Gleichstand nach id. Leer ohne Vektoren."""
    q = _normiert(anfrage_vektor)
    if q is None or top_k <= 0:
        return ()
    with closing(_ro(Path(stand))) as c:
        zeilen = c.execute("SELECT datei, v FROM vektor WHERE modell=? ORDER BY datei", (modell,)).fetchall()
    ids: list[str] = []
    reihen: list[np.ndarray] = []
    for datei, blob in zeilen:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape == q.shape:  # andere Dimension = anderes Modell, ueberspringen
            ids.append(datei)
            reihen.append(v)
    if not ids:
        return ()
    wert = np.vstack(reihen) @ q
    ordnung = sorted(range(len(ids)), key=lambda k: (-round(float(wert[k]), 6), ids[k]))
    return tuple((ids[k], float(wert[k])) for k in ordnung[:top_k])
