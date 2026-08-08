import json
import random

import numpy as np
import pytest


def test_retriever_uses_shared_embedding_service_contract(monkeypatch):
    import embeddinggemma.mcmp_rag as mcmp_rag

    expected_model = object()
    calls = []

    def load_embedding_backend():
        calls.append(True)
        return expected_model, 3072

    monkeypatch.setattr(mcmp_rag, "load_embedding_backend", load_embedding_backend)

    retriever = mcmp_rag.MCPMRetriever()

    assert retriever.embedding_model is expected_model
    assert retriever._expected_embedding_dim == 3072
    assert calls == [True]


def test_retriever_propagates_embedding_service_failure(monkeypatch):
    import embeddinggemma.mcmp_rag as mcmp_rag

    def unavailable():
        raise RuntimeError("embedding-service unreachable")

    monkeypatch.setattr(mcmp_rag, "load_embedding_backend", unavailable)

    with pytest.raises(RuntimeError, match="embedding-service unreachable"):
        mcmp_rag.MCPMRetriever()


def test_persistent_index_with_legacy_dimension_requires_rebuild(monkeypatch, tmp_path):
    import embeddinggemma.mcmp_rag as mcmp_rag

    monkeypatch.setattr(mcmp_rag, "load_embedding_backend", lambda: (object(), 3072))
    monkeypatch.setattr(mcmp_rag.MCPMRetriever, "_CACHE_DIR", str(tmp_path))

    (tmp_path / "chunks.json").write_text(json.dumps(["legacy document"]), encoding="utf-8")
    np.savez_compressed(tmp_path / "embeddings.npz", embeddings=np.zeros((1, 384), dtype=np.float32))

    retriever = mcmp_rag.MCPMRetriever()

    with pytest.raises(RuntimeError, match="rebuild required.*384.*3072"):
        retriever.load_persistent_index()


def test_retriever_rejects_embedding_response_with_unexpected_dimension(monkeypatch):
    import embeddinggemma.mcmp_rag as mcmp_rag

    class WrongDimensionModel:
        def encode(self, _texts):
            return [[0.0] * 384]

    monkeypatch.setattr(mcmp_rag, "load_embedding_backend", lambda: (WrongDimensionModel(), 3072))

    retriever = mcmp_rag.MCPMRetriever()

    with pytest.raises(RuntimeError, match="expected dimension 3072.*got 384"):
        retriever.add_documents(["must not be silently indexed"], cache=False)


def test_retriever_rejects_embedding_response_count_mismatch(monkeypatch):
    import embeddinggemma.mcmp_rag as mcmp_rag

    class MissingVectorModel:
        def encode(self, _texts):
            return []

    monkeypatch.setattr(mcmp_rag, "load_embedding_backend", lambda: (MissingVectorModel(), 3072))

    retriever = mcmp_rag.MCPMRetriever()

    with pytest.raises(RuntimeError, match="response count does not match.*expected 1.*got 0"):
        retriever.add_documents(["must not be silently indexed"], cache=False)


def test_persistent_index_with_non_matrix_embeddings_requires_rebuild(monkeypatch, tmp_path):
    import embeddinggemma.mcmp_rag as mcmp_rag

    monkeypatch.setattr(mcmp_rag, "load_embedding_backend", lambda: (object(), 3072))
    monkeypatch.setattr(mcmp_rag.MCPMRetriever, "_CACHE_DIR", str(tmp_path))

    (tmp_path / "chunks.json").write_text(json.dumps(["corrupt document"]), encoding="utf-8")
    np.savez_compressed(tmp_path / "embeddings.npz", embeddings=np.zeros(384, dtype=np.float32))

    retriever = mcmp_rag.MCPMRetriever()

    with pytest.raises(RuntimeError, match="rebuild required.*two-dimensional"):
        retriever.load_persistent_index()


def test_faiss_neighbours_report_inner_product_as_cosine_similarity(monkeypatch):
    faiss = pytest.importorskip("faiss")
    import embeddinggemma.mcmp_rag as mcmp_rag

    random.seed(0)
    np.random.seed(0)
    monkeypatch.setattr(mcmp_rag, "load_embedding_backend", lambda: (object(), 2))

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    retriever = mcmp_rag.MCPMRetriever()
    retriever.documents = [
        mcmp_rag.Document(id=index, content=f"document-{index}", embedding=vector)
        for index, vector in enumerate(embeddings)
    ]
    cpu_index = faiss.IndexFlatIP(2)
    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)
    cpu_index.add(normalized_embeddings)
    retriever._faiss_index = cpu_index

    neighbours = retriever.find_nearest_documents(
        np.array([1.0, 0.0], dtype=np.float32),
        k=3,
    )

    assert [document.id for document, _similarity in neighbours] == [0, 1, 2]
    assert [similarity for _document, similarity in neighbours] == pytest.approx(
        [1.0, 0.0, -1.0]
    )


def test_retriever_accepts_explicit_embedding_backend():
    import embeddinggemma.mcmp_rag as mcmp_rag

    backend = object()
    retriever = mcmp_rag.MCPMRetriever(embedding_backend=(backend, 3))

    assert retriever.embedding_model is backend
    assert retriever._expected_embedding_dim == 3
