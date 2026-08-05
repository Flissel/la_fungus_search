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
