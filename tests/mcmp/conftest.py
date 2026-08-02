import os
import sys
import types
from pathlib import Path
import pytest
import numpy as np


def _ensure_src_on_path() -> None:
    """Ensure project src directory is on sys.path so imports work in tests."""
    tests_dir = Path(__file__).resolve().parent
    project_root = tests_dir.parent.parent
    src_dir = project_root / "src"
    src_str = str(src_dir)
    if src_dir.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()


@pytest.fixture(autouse=True)
def ensure_fake_sklearn(monkeypatch):
    """Provide a lightweight fallback for sklearn.cosine_similarity if sklearn is missing.

    simulation.update_document_relevance imports sklearn at runtime when CUDA is unavailable.
    To keep tests hermetic and avoid heavy deps, inject a tiny stub if sklearn is not installed.
    """
    try:
        import sklearn  # type: ignore # noqa: F401
        return
    except Exception:
        pass

    # Create module skeletons
    sklearn_mod = types.ModuleType("sklearn")
    metrics_mod = types.ModuleType("sklearn.metrics")
    pairwise_mod = types.ModuleType("sklearn.metrics.pairwise")

    def cosine_similarity(A, B):  # minimal compatible implementation
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return An @ Bn.T

    pairwise_mod.cosine_similarity = cosine_similarity
    metrics_mod.pairwise = pairwise_mod
    sklearn_mod.metrics = metrics_mod

    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.metrics"] = metrics_mod
    sys.modules["sklearn.metrics.pairwise"] = pairwise_mod


