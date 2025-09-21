from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import logging


_logger = logging.getLogger("MCMP.PCA")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


def pca_2d(doc_embeddings: np.ndarray, whiten: bool = False):
    _logger.debug("pca_2d: docs=%d whiten=%s", int(doc_embeddings.shape[0]), str(whiten))
    mean = doc_embeddings.mean(axis=0)
    X = doc_embeddings - mean
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comps = Vt[:2]
    except Exception:
        rng = np.random.default_rng(42)
        D = doc_embeddings.shape[1]
        comps = rng.normal(0, 1, size=(2, D))
        for i in range(2):
            comps[i] = comps[i] / (np.linalg.norm(comps[i]) or 1.0)
        S = None
    coords = X @ comps.T
    if whiten and S is not None:
        s = S[:2]
        safe = np.array([s[0] if s[0] != 0 else 1.0, s[1] if s[1] != 0 else 1.0])
        coords = coords / safe
    _logger.debug("pca_2d: done")
    return coords



def pca_fit_transform(doc_embeddings: np.ndarray, n_components: int = 2, whiten: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute PCA projection with components returned for projecting new points.

    Returns (coords, mean, components, singular_values).
    coords has shape (N, n_components), components has shape (n_components, D).
    """
    _logger.debug("pca_fit_transform: docs=%d comps=%d whiten=%s", int(doc_embeddings.shape[0]), int(n_components), str(whiten))
    mean = doc_embeddings.mean(axis=0)
    X = doc_embeddings - mean
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comps = Vt[:n_components]
    except Exception:
        rng = np.random.default_rng(42)
        D = doc_embeddings.shape[1]
        comps = rng.normal(0, 1, size=(n_components, D))
        for i in range(n_components):
            comps[i] = comps[i] / (np.linalg.norm(comps[i]) or 1.0)
        S = None
    coords = X @ comps.T
    if whiten and S is not None:
        s = S[:n_components]
        safe = np.array([sv if sv != 0 else 1.0 for sv in s])
        coords = coords / safe
    return coords, mean, comps, S

