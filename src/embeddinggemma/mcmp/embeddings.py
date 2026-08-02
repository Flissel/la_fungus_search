"""OpenFang-backed embeddings for the active Fungus search path."""

from __future__ import annotations

from typing import Any

from vibemind_shared import get_embedding_model


EMBEDDING_ROLE = "fungus_search"


def load_embedding_model() -> Any:
    """Load the sole configured Fungus embedding model through OpenFang.

    The role fixes provider, model, and transport in the shared VibeMind
    configuration.  Errors deliberately propagate: local embedding fallbacks
    would produce incompatible vectors and bypass OpenFang accounting.
    """
    return get_embedding_model(EMBEDDING_ROLE)
