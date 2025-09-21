from __future__ import annotations
from typing import Optional
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


