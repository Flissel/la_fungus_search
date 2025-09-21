import numpy as np


def test_pca_2d_shapes_and_whiten():
    from embeddinggemma.mcmp import pca

    rng = np.random.RandomState(42)
    X = rng.randn(50, 16).astype(np.float32)
    coords = pca.pca_2d(X, whiten=False)
    assert coords.shape == (50, 2)

    coords_w = pca.pca_2d(X, whiten=True)
    assert coords_w.shape == (50, 2)
    # Scale difference likely when whitening; just ensure numerically finite
    assert np.isfinite(coords_w).all()


