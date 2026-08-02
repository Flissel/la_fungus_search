"""OpenFang-backed embeddings for the active Fungus search path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vibemind_shared import get_config, get_embedding_config, get_embedding_model


EMBEDDING_ROLE = "fungus_search"
_EXPECTED_CONTRACT = {
    "driver": "openai",
    "provider": "openfang",
    "model": "text-embedding-3-large",
    "dim": 3072,
}


@dataclass(frozen=True)
class FungusEmbeddingContract:
    model: str
    dimension: int


def _validate_embedding_contract() -> FungusEmbeddingContract:
    config = get_config()
    embeddings = config.get("embeddings") if isinstance(config, Mapping) else None
    if not isinstance(embeddings, Mapping) or EMBEDDING_ROLE not in embeddings:
        raise RuntimeError(
            "Fungus OpenFang embedding contract requires explicit "
            "embeddings.fungus_search configuration."
        )

    resolved = get_embedding_config(EMBEDDING_ROLE)
    if not isinstance(resolved, Mapping):
        raise RuntimeError("Fungus OpenFang embedding contract must be a mapping.")
    if set(resolved) != set(_EXPECTED_CONTRACT):
        raise RuntimeError(
            "Fungus OpenFang embedding contract fields must be exactly "
            f"{sorted(_EXPECTED_CONTRACT)}; got {sorted(resolved)}."
        )

    for field, expected in _EXPECTED_CONTRACT.items():
        actual = resolved.get(field)
        if actual != expected:
            raise RuntimeError(
                "Fungus OpenFang embedding contract drift: "
                f"expected {field}={expected!r}, got {actual!r}."
            )

    return FungusEmbeddingContract(
        model=_EXPECTED_CONTRACT["model"],
        dimension=_EXPECTED_CONTRACT["dim"],
    )


def load_embedding_backend() -> tuple[Any, int]:
    """Load the configured gateway model with its validated vector dimension."""
    contract = _validate_embedding_contract()
    return get_embedding_model(EMBEDDING_ROLE), contract.dimension


def load_embedding_model() -> Any:
    """Load the sole configured Fungus embedding model through OpenFang.

    The role fixes provider, model, and transport in the shared VibeMind
    configuration.  Errors deliberately propagate: local embedding fallbacks
    would produce incompatible vectors and bypass OpenFang accounting.
    """
    return load_embedding_backend()[0]
