import sys
import types
from pathlib import Path
import pytest


def _ensure_src_on_path() -> None:
    tests_dir = Path(__file__).resolve().parent
    project_root = tests_dir.parent.parent
    src_dir = project_root / "src"
    src_str = str(src_dir)
    if src_dir.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()


@pytest.fixture(autouse=True)
def stub_sentence_transformers(monkeypatch):
    try:
        import sentence_transformers  # type: ignore # noqa: F401
        return
    except Exception:
        pass
    st_mod = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def encode(self, texts):
            return [[0.0] * 8 for _ in texts]

    st_mod.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = st_mod


@pytest.fixture(autouse=True)
def stub_llama_index(monkeypatch):
    """Provide a minimal llama_index stub for indexer/search tests."""
    core_name = "llama_index.core"
    vs_name = "llama_index.vector_stores.qdrant"

    if core_name not in sys.modules:
        core_mod = types.ModuleType(core_name)

        class StorageContext:
            @classmethod
            def from_defaults(cls, **kwargs):
                inst = cls()
                inst.kwargs = kwargs
                return inst

        class VectorStoreIndex:
            def __init__(self, documents=None, storage_context=None, embed_model=None, transformations=None, show_progress=False):
                self.documents = documents or []
                self.storage_context = storage_context
                self.embed_model = embed_model
                self.transformations = transformations
                self.show_progress = show_progress

            @classmethod
            def from_documents(cls, documents, storage_context=None, embed_model=None, transformations=None, show_progress=False):
                return cls(documents, storage_context, embed_model, transformations, show_progress)

            @classmethod
            def from_vector_store(cls, vector_store=None, storage_context=None, embed_model=None):
                return cls([], storage_context, embed_model, None, False)

            def as_retriever(self, similarity_top_k=10, node_postprocessors=None):
                class Retriever:
                    def retrieve(self_inner, query):
                        return []

                return Retriever()

        def load_index_from_storage(storage_context=None, embed_model=None):
            return VectorStoreIndex.from_vector_store(storage_context=storage_context, embed_model=embed_model)

        core_mod.StorageContext = StorageContext
        core_mod.VectorStoreIndex = VectorStoreIndex
        core_mod.load_index_from_storage = load_index_from_storage
        sys.modules[core_name] = core_mod

    if vs_name not in sys.modules:
        vs_mod = types.ModuleType(vs_name)

        class QdrantVectorStore:
            pass

        vs_mod.QdrantVectorStore = QdrantVectorStore
        sys.modules[vs_name] = vs_mod


@pytest.fixture(autouse=True)
def stub_qdrant_client(monkeypatch):
    """Provide a minimal qdrant_client stub for vectorstore tests."""
    root_name = "qdrant_client"
    models_name = "qdrant_client.http.models"

    if root_name in sys.modules and models_name in sys.modules:
        return

    root_mod = types.ModuleType(root_name)
    models_mod = types.ModuleType(models_name)

    class QdrantClient:
        pass

    class Distance:
        COSINE = "cosine"

    class VectorParams:
        def __init__(self, size: int, distance):
            self.size = size
            self.distance = distance

    class HnswConfigDiff:
        def __init__(self, m=None, ef_construct=None):
            self.m = m
            self.ef_construct = ef_construct

    class OptimizersConfigDiff:
        def __init__(self, indexing_threshold=None, memmap_threshold=None):
            self.indexing_threshold = indexing_threshold
            self.memmap_threshold = memmap_threshold

    class ScalarQuantizationConfig:
        class Scalar:
            def __init__(self, bits=None):
                self.bits = bits

        def __init__(self, scalar=None, always_ram=None):
            self.scalar = scalar
            self.always_ram = always_ram

    root_mod.QdrantClient = QdrantClient
    models_mod.Distance = Distance
    models_mod.VectorParams = VectorParams
    models_mod.HnswConfigDiff = HnswConfigDiff
    models_mod.OptimizersConfigDiff = OptimizersConfigDiff
    models_mod.ScalarQuantizationConfig = ScalarQuantizationConfig

    sys.modules[root_name] = root_mod
    sys.modules[models_name] = models_mod


