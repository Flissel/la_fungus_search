"""cachedir.py — ein Ort fuer die Aufloesung des Cache-Verzeichnisses.

Qdrant trennt die Vektoren, aber bm25 und multivec bleiben Dateien. Ohne
Unterverzeichnis pro Collection ueberschreibt die zweite Codebase die
Hybrid-Daten der ersten - ohne dass die Suche aufhoert, Treffer zu liefern.
"""

from __future__ import annotations

import os


def cache_dir(default_root: str, collection: str = "") -> str:
    basis = os.environ.get("FUNGUS_CACHE_DIR") or os.path.join(default_root, ".fungus_cache")
    return os.path.join(basis, collection) if collection else basis


def resolve_for_codebase(default_root: str, codebase: str) -> tuple[str, str]:
    """(collection, cache_dir) fuer eine Codebase.

    Die eine Stelle, an der Leseseite (mcp_server.py) und Schreibseite
    (repair_index.py, build_multivec_index.py) dieselbe Paarung bilden -
    beide rufen diese Funktion statt die Pruefung "FUNGUS_STORE=qdrant?"
    je einzeln nachzubauen. FUNGUS_STORE=qdrant schaltet die Collection
    scharf (case-gefaltet ueber store_qdrant.collection_for); sonst bleibt
    collection leer und cache_dir identisch zum alten, flachen Pfad.
    """
    from embeddinggemma.store_qdrant import collection_for

    collection = collection_for(codebase) if os.environ.get("FUNGUS_STORE", "").lower() == "qdrant" else ""
    return collection, cache_dir(default_root, collection)
