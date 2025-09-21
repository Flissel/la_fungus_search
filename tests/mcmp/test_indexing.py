import numpy as np
import pytest


def test_build_faiss_index_returns_none_when_unavailable(monkeypatch):
    from embeddinggemma.mcmp import indexing

    monkeypatch.setattr(indexing, "_FAISS_OK", False)
    embs = np.random.RandomState(0).randn(10, 8).astype(np.float32)
    idx = indexing.build_faiss_index(embs, dim=8)
    assert idx is None


def test_cosine_similarities_cpu_basic():
    from embeddinggemma.mcmp import indexing

    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    sims = indexing.cosine_similarities_cpu(q, docs)
    assert sims.shape == (3,)
    assert np.isclose(sims[0], 1.0, atol=1e-5)
    assert sims[1] < -0.99
    assert abs(sims[2]) < 1e-6


def test_cosine_similarities_torch_cpu_if_available():
    torch = pytest.importorskip("torch")
    from embeddinggemma.mcmp import indexing

    device = "cpu"
    doc_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    with torch.no_grad():
        doc_t = torch.tensor(doc_np, device=device).to(torch.float16)
        doc_tn = torch.nn.functional.normalize(doc_t, p=2, dim=1)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sims = indexing.cosine_similarities_torch(q, doc_tn, device=device)
        assert sims.shape == (2,)
        assert sims[0] > 0.99 and abs(sims[1]) < 1e-6


