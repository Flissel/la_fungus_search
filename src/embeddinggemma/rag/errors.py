class RagError(Exception):
    """Base class for Rag-related errors."""


class VectorStoreError(RagError):
    pass


class IndexBuildError(RagError):
    pass


class GenerationError(RagError):
    pass



