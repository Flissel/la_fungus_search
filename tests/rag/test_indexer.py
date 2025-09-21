def test_build_and_load_index(monkeypatch):
    from embeddinggemma.rag import indexer
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    docs = ["a", "b"]

    vs = QdrantVectorStore()
    embed_model = object()
    transformations = []

    idx = indexer.build_index(docs, vs, embed_model, transformations)
    assert hasattr(idx, "as_retriever")

    loaded = indexer.load_index("/tmp/nonexistent", vs, embed_model)
    assert hasattr(loaded, "as_retriever")


