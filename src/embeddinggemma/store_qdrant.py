"""store_qdrant.py — Persistenz des MCMP-Index in einer Collection pro Codebase.

Ersetzt nur Speichern und Laden. Geladen wird in denselben Zustand, den die
Datei-Persistenz auch herstellt (Texte + Embedding-Matrix); FAISS baut der
Retriever daraus ohnehin neu auf. Die Suche selbst bleibt unangetastet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request

import numpy as np

_logger = logging.getLogger(__name__)

DEFAULT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6339")
BATCH = 256


def collection_for(codebase: str) -> str:
    """Stabiler Name pro Codebase-Pfad: lesbar vorne, kollisionssicher hinten.

    Voll case-gefaltet (auch im lesbaren Teil): auf Windows bezeichnen
    "...\\Laura", "...\\laura" und "...\\LAURA" denselben Ordner, muessen
    also auf denselben Namen abbilden. Stabilitaet geht vor Lesbarkeit der
    Original-Schreibweise.
    """
    norm = os.path.normcase(os.path.normpath(codebase)).replace("\\", "/").rstrip("/")
    kurz = norm.rsplit("/", 1)[-1] or "root"
    sauber = "".join(c if c.isalnum() else "_" for c in kurz)
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    return f"fungus_{sauber}_{digest}"


def verbunden(url: str = DEFAULT_URL) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/collections", timeout=5) as r:
            return json.loads(r.read().decode("utf-8")).get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def make_client(url: str = "", api_key: str = ""):
    from qdrant_client import QdrantClient
    return QdrantClient(url=url or DEFAULT_URL, api_key=api_key or os.environ.get("QDRANT_API_KEY"))


def ensure(client, name: str, dim: int) -> None:
    from qdrant_client.http.models import Distance, VectorParams
    vorhanden = [c.name for c in client.get_collections().collections]
    if name in vorhanden:
        info = client.get_collection(collection_name=name)
        if int(info.config.params.vectors.size) == int(dim):
            return
        _logger.warning("Collection %s hat Dimension %s statt %s - wird neu angelegt",
                        name, info.config.params.vectors.size, dim)
        client.delete_collection(name)
    client.create_collection(collection_name=name,
                             vectors_config=VectorParams(size=int(dim), distance=Distance.COSINE))


def save(client, name: str, texts: list[str], embeddings) -> int:
    """Speichert Texte + Embeddings in der Collection.

    Qdrant normalisiert Vektoren einer COSINE-Collection beim Speichern
    intern auf Einheitslaenge; was `scroll()`/`get_collection()` spaeter
    zurueckgeben, ist also nicht mehr das Original. Deshalb wird pro Punkt
    zusaetzlich die L2-Norm des Original-Embeddings in der Payload
    abgelegt (`"norm"`) - `load()` rechnet damit zurueck.
    """
    from qdrant_client.http.models import PointStruct
    mat = np.asarray(embeddings, dtype=np.float32)
    if mat.ndim != 2 or len(texts) != mat.shape[0]:
        raise ValueError(f"Texte ({len(texts)}) und Embeddings ({mat.shape}) passen nicht zusammen")
    ensure(client, name, int(mat.shape[1]))
    for start in range(0, len(texts), BATCH):
        punkte = [
            PointStruct(id=i, vector=mat[i].tolist(),
                       payload={"content": texts[i], "idx": i, "norm": float(np.linalg.norm(mat[i]))})
            for i in range(start, min(start + BATCH, len(texts)))
        ]
        client.upsert(collection_name=name, points=punkte)
    return len(texts)


def load(client, name: str):
    """Laedt Texte + Embeddings zurueck, mit der urspruenglichen Skalierung.

    Der von Qdrant gelieferte Vektor ist bei COSINE-Collections
    normiert; er wird mit der in `save()` abgelegten Norm zurueckskaliert,
    damit das Original-Embedding rauskommt statt der Einheitslaenge.
    Nullvektoren brauchen keine Sonderbehandlung: Qdrant normiert einen
    Nullvektor nicht (er bleibt [0, ..., 0]), und `vektor * 0 == 0` trifft
    exakt das Original - keine Division, also auch kein NaN-Risiko.
    Fehlt "norm" in der Payload (Collection aus einer Fassung vor diesem
    Feld), wird der gelieferte Vektor unveraendert benutzt statt
    abzustuerzen - dann bleibt die alte (unkorrigierte) Skalierung
    bestehen, ein sauberer Fehler ist besser als eine erfundene Zahl.
    """
    vorhanden = [c.name for c in client.get_collections().collections]
    if name not in vorhanden:
        return None
    gesammelt: list[tuple[int, str, list, "float | None"]] = []
    offset = None
    while True:
        punkte, offset = client.scroll(collection_name=name, limit=BATCH, offset=offset,
                                       with_vectors=True, with_payload=True)
        for p in punkte:
            payload = getattr(p, "payload", None) or {}
            vektor = getattr(p, "vector", None)
            gesammelt.append((int(payload.get("idx", getattr(p, "id", 0))),
                              payload.get("content", ""), vektor, payload.get("norm")))
        if offset is None:
            break
    if not gesammelt:
        return None
    gesammelt.sort(key=lambda t: t[0])
    texte = [t[1] for t in gesammelt]
    mat = np.array([t[2] for t in gesammelt], dtype=np.float32)
    for i, (_, _, _, norm) in enumerate(gesammelt):
        if norm is not None:
            mat[i] = mat[i] * float(norm)
    return texte, mat
