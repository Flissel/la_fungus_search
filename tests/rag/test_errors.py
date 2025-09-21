def test_errors_inheritance():
    from embeddinggemma.rag.errors import RagError, VectorStoreError, IndexBuildError, GenerationError

    assert issubclass(VectorStoreError, RagError)
    assert issubclass(IndexBuildError, RagError)
    assert issubclass(GenerationError, RagError)


