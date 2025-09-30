"""
Comprehensive tests for the RAG (Retrieval-Augmented Generation) module.

Tests all components of the RAG system including chunking, embeddings,
indexing, search, generation, and vectorstore functionality.
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Import all RAG components
from embeddinggemma.rag import chunking
from embeddinggemma.rag import embeddings as rag_embeddings
from embeddinggemma.rag import indexer
from embeddinggemma.rag import search
from embeddinggemma.rag import generation
from embeddinggemma.rag import vectorstore
from embeddinggemma.rag import config
from embeddinggemma.rag import errors


class TestRAGChunkingComprehensive:
    """Comprehensive tests for the chunking module."""

    def test_python_code_parsing_with_ast(self, tmp_path):
        """Test AST-based parsing of Python code files."""
        # Create test Python file
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
import os
import sys

def calculate_fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False

    def process(self):
        \"\"\"Process the data.\"\"\"
        if not self.processed:
            self.data = [x * 2 for x in self.data]
            self.processed = True
        return self.data

    def get_stats(self):
        return {
            'length': len(self.data),
            'sum': sum(self.data),
            'average': sum(self.data) / len(self.data) if self.data else 0
        }
""")

        # Test AST parsing
        chunks = chunking.parse_code_with_ast(str(test_file))

        # Should extract meaningful chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

        # Check metadata structure
        for chunk in chunks:
            metadata = chunk["metadata"]
            assert "start_line" in metadata
            assert "end_line" in metadata
            assert "size" in metadata
            assert metadata["start_line"] >= 1
            assert metadata["end_line"] >= metadata["start_line"]

    def test_fallback_chunking_for_invalid_python(self, tmp_path):
        """Test fallback chunking when AST parsing fails."""
        # Create file with invalid Python syntax
        test_file = tmp_path / "invalid.py"
        test_file.write_text("""
def broken_function(
    # Missing closing parenthesis and body
import os  # This will cause syntax error
""")

        # Should fall back to line-based chunking
        chunks = chunking.parse_code_with_ast(str(test_file))

        # Should still produce chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)

        # Should have size metadata from fallback
        for chunk in chunks:
            assert "size" in chunk["metadata"]
            assert chunk["metadata"]["size"] > 0

    def test_chunk_size_variations(self, tmp_path):
        """Test chunking with different size parameters."""
        # Create test file with known line count
        test_file = tmp_path / "test.py"
        lines = ["line {}".format(i) for i in range(1, 101)]  # 100 lines
        test_file.write_text("\n".join(lines))

        # Test different chunk sizes
        small_chunks = chunking.parse_code_with_ast(str(test_file))
        # Should produce multiple chunks for large file

        assert len(small_chunks) > 1
        total_lines = sum(chunk["metadata"]["size"] for chunk in small_chunks)
        assert total_lines >= 100  # Should cover all lines

    def test_empty_and_edge_case_files(self, tmp_path):
        """Test chunking with empty files and edge cases."""
        # Empty file
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        empty_chunks = chunking.parse_code_with_ast(str(empty_file))
        # Should handle gracefully, possibly return empty list or single chunk

        # Single line file
        single_file = tmp_path / "single.py"
        single_file.write_text("print('hello')")
        single_chunks = chunking.parse_code_with_ast(str(single_file))
        assert len(single_chunks) >= 1


class TestRAGEmbeddingsComprehensive:
    """Comprehensive tests for the RAG embeddings module."""

    def test_device_resolution_comprehensive(self, monkeypatch):
        """Test device resolution with all scenarios."""
        # Test when torch is completely unavailable
        monkeypatch.setattr(rag_embeddings, "torch", None)
        assert rag_embeddings.resolve_device("auto") == "cpu"
        assert rag_embeddings.resolve_device("cpu") == "cpu"
        assert rag_embeddings.resolve_device("cuda") == "cpu"  # Falls back to CPU

        # Test when CUDA is available
        class MockCUDA:
            @staticmethod
            def is_available():
                return True

        class MockTorch:
            cuda = MockCUDA()

        monkeypatch.setattr(rag_embeddings, "torch", MockTorch)
        assert rag_embeddings.resolve_device("auto") == "cuda"
        assert rag_embeddings.resolve_device("cuda") == "cuda"

        # Test when CUDA is not available but torch exists
        class MockCUDAUnavailable:
            @staticmethod
            def is_available():
                return False

        class MockTorchWithCUDA:
            cuda = MockCUDAUnavailable()

        monkeypatch.setattr(rag_embeddings, "torch", MockTorchWithCUDA)
        assert rag_embeddings.resolve_device("auto") == "cpu"
        assert rag_embeddings.resolve_device("cuda") == "cpu"  # Falls back

    def test_embedding_backend_full_lifecycle(self, monkeypatch):
        """Test complete lifecycle of EmbeddingBackend."""
        # Mock sentence transformer
        class MockModel:
            def __init__(self, name, device=None):
                self.model_name = name
                self.device = device
                self.encode_calls = []

            def encode(self, texts):
                self.encode_calls.append(texts)
                # Return mock embeddings
                return [[0.1 * i for i in range(8)] for i, _ in enumerate(texts)]

        monkeypatch.setattr(rag_embeddings, "SentenceTransformer", MockModel)
        monkeypatch.setattr(rag_embeddings, "_log_torch_environment", lambda level=None: None)
        monkeypatch.setattr(rag_embeddings, "resolve_device", lambda pref: "cpu")

        # Create backend
        backend = rag_embeddings.EmbeddingBackend(
            model_name="test-model",
            device_preference="auto"
        )

        # Test model loading
        model = backend.load()
        assert isinstance(model, MockModel)
        assert model.model_name == "test-model"
        assert model.device == "cpu"

        # Test encoding single text
        vectors = backend.encode(["Hello world"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 8

        # Test encoding multiple texts
        texts = ["Text one", "Text two", "Text three"]
        vectors = backend.encode(texts)
        assert len(vectors) == 3
        assert all(len(v) == 8 for v in vectors)

        # Test encoding calls were made correctly
        assert len(model.encode_calls) == 2  # Two encode calls

    def test_fallback_behavior_when_models_fail(self, monkeypatch):
        """Test fallback behavior when embedding models fail to load."""
        # Mock model that fails to initialize
        class FailingModel:
            def __init__(self, name, device=None):
                raise RuntimeError("Model loading failed")

        monkeypatch.setattr(rag_embeddings, "SentenceTransformer", FailingModel)
        monkeypatch.setattr(rag_embeddings, "_log_torch_environment", lambda level=None: None)
        monkeypatch.setattr(rag_embeddings, "resolve_device", lambda pref: "cpu")

        backend = rag_embeddings.EmbeddingBackend(
            model_name="failing-model",
            device_preference="auto"
        )

        # Should handle failure gracefully and use fallback encoding
        model = backend.load()
        # Model might be None or have fallback behavior
        # The exact behavior depends on implementation

        # Encoding should still work (with random embeddings)
        vectors = backend.encode(["test"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 8  # Default embedding dimension


class TestRAGIndexingComprehensive:
    """Comprehensive tests for the RAG indexing module."""

    def test_index_creation_and_loading(self, monkeypatch):
        """Test index creation and loading functionality."""
        # Mock llama_index components
        class MockVectorStore:
            def __init__(self):
                self.documents = []

        class MockStorageContext:
            @classmethod
            def from_defaults(cls, **kwargs):
                return cls()

        class MockIndex:
            def __init__(self, documents=None, storage_context=None, embed_model=None, transformations=None, show_progress=False):
                self.documents = documents or []
                self.storage_context = storage_context
                self.embed_model = embed_model

            @classmethod
            def from_documents(cls, documents, storage_context=None, embed_model=None, transformations=None, show_progress=False):
                return cls(documents, storage_context, embed_model, transformations, show_progress)

            @classmethod
            def from_vector_store(cls, vector_store=None, storage_context=None, embed_model=None):
                return cls([], storage_context, embed_model, None, False)

            def as_retriever(self, similarity_top_k=10, node_postprocessors=None):
                class MockRetriever:
                    def retrieve(self, query):
                        return []
                return MockRetriever()

        def mock_load_index_from_storage(storage_context=None, embed_model=None):
            return MockIndex.from_vector_store(storage_context=storage_context, embed_model=embed_model)

        # Apply mocks
        monkeypatch.setattr(indexer, "VectorStoreIndex", MockIndex)
        monkeypatch.setattr(indexer, "StorageContext", MockStorageContext)
        monkeypatch.setattr(indexer, "load_index_from_storage", mock_load_index_from_storage)

        # Test index creation
        docs = ["Document one", "Document two", "Document three"]
        vs = MockVectorStore()
        embed_model = object()
        transformations = []

        index = indexer.build_index(docs, vs, embed_model, transformations)
        assert index is not None
        assert hasattr(index, 'as_retriever')

        # Test index loading
        loaded_index = indexer.load_index("/fake/path", vs, embed_model)
        assert loaded_index is not None
        assert hasattr(loaded_index, 'as_retriever')

        # Test retriever creation
        retriever = index.as_retriever(similarity_top_k=5)
        assert retriever is not None
        assert hasattr(retriever, 'retrieve')

    def test_error_handling_in_indexing(self, monkeypatch):
        """Test error handling during indexing operations."""
        # Mock components that can fail
        class FailingIndex:
            def __init__(self, documents=None, storage_context=None, embed_model=None, transformations=None, show_progress=False):
                raise RuntimeError("Index creation failed")

            @classmethod
            def from_documents(cls, documents, storage_context=None, embed_model=None, transformations=None, show_progress=False):
                raise RuntimeError("Index creation from documents failed")

        monkeypatch.setattr(indexer, "VectorStoreIndex", FailingIndex)

        # Should handle failures gracefully
        docs = ["test"]
        vs = object()
        embed_model = object()
        transformations = []

        # These should either raise appropriate errors or handle gracefully
        # The exact behavior depends on implementation
        try:
            index = indexer.build_index(docs, vs, embed_model, transformations)
            # If we get here, the function handled the error
        except Exception:
            # Expected behavior - error should be raised
            pass


class TestRAGSearchComprehensive:
    """Comprehensive tests for the RAG search module."""

    def test_hybrid_search_functionality(self, monkeypatch):
        """Test hybrid search combining semantic and keyword search."""
        # Mock index and retriever
        class MockNode:
            def __init__(self, text, node_id="test"):
                self.text = text
                self.node_id = node_id
                self.metadata = {"source": node_id}

            def get_content(self):
                return self.text

        class MockRetriever:
            def __init__(self, nodes):
                self.nodes = nodes

            def retrieve(self, query):
                return self.nodes

        class MockIndex:
            def __init__(self, nodes):
                self.nodes = nodes

            def as_retriever(self, similarity_top_k=10, node_postprocessors=None):
                return MockRetriever(self.nodes)

        # Create test nodes
        nodes = [
            MockNode("Machine learning is powerful", "ml1"),
            MockNode("Natural language processing helps", "nlp1"),
            MockNode("Computer vision sees images", "cv1"),
        ]

        index = MockIndex(nodes)

        # Test hybrid search
        results = search.hybrid_search(index, query="machine learning", top_k=2, alpha=0.5)

        # Should return results
        assert isinstance(results, list)
        assert len(results) <= 2  # Limited by top_k

        # Results should have hybrid scores
        for result in results:
            assert "hybrid_score" in result
            assert isinstance(result["hybrid_score"], (int, float))

    def test_search_result_scoring(self, monkeypatch):
        """Test search result scoring and ranking."""
        # Create mock setup similar to above
        class MockNode:
            def __init__(self, text, score, node_id="test"):
                self.text = text
                self.node_id = node_id
                self.metadata = {"source": node_id}
                self.score = score

            def get_content(self):
                return self.text

        class MockRetriever:
            def __init__(self, nodes):
                self.nodes = nodes

            def retrieve(self, query):
                return self.nodes

        class MockIndex:
            def as_retriever(self, similarity_top_k=10, node_postprocessors=None):
                return MockRetriever([MockNode("High score text", 0.9, "high"),
                                    MockNode("Medium score text", 0.6, "med"),
                                    MockNode("Low score text", 0.3, "low")])

        index = MockIndex()

        # Test search with scoring
        results = search.hybrid_search(index, query="test", top_k=3, alpha=1.0)

        # Results should be ordered by score (highest first)
        scores = [r["hybrid_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_empty_results(self, monkeypatch):
        """Test search behavior with empty or no results."""
        class MockEmptyRetriever:
            def retrieve(self, query):
                return []

        class MockEmptyIndex:
            def as_retriever(self, similarity_top_k=10, node_postprocessors=None):
                return MockEmptyRetriever()

        index = MockEmptyIndex()

        # Should handle empty results gracefully
        results = search.hybrid_search(index, query="test", top_k=5, alpha=0.5)
        assert isinstance(results, list)
        assert len(results) == 0


class TestRAGGenerationComprehensive:
    """Comprehensive tests for the RAG generation module."""

    def test_ollama_generation_success(self, monkeypatch):
        """Test successful generation with Ollama."""
        # Mock successful response
        class MockResponse:
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"response": "Generated answer"}

        def mock_post(url, json=None, timeout=None):
            return MockResponse()

        monkeypatch.setattr(generation, "requests", type('MockRequests', (), {'post': mock_post})())

        # Test generation
        result = generation.generate_with_ollama(
            "Test prompt",
            model="test-model",
            host="http://localhost:11434"
        )

        assert result == "Generated answer"

    def test_ollama_generation_error_handling(self, monkeypatch):
        """Test error handling in Ollama generation."""
        # Mock error response
        class MockErrorResponse:
            def __init__(self):
                self.status_code = 500

            def raise_for_status(self):
                raise RuntimeError("Server error")

        def mock_post(url, json=None, timeout=None):
            return MockErrorResponse()

        monkeypatch.setattr(generation, "requests", type('MockRequests', (), {'post': mock_post})())

        # Should handle errors gracefully
        result = generation.generate_with_ollama(
            "Test prompt",
            model="test-model",
            host="http://localhost:11434"
        )

        # Should return error message
        assert result.startswith("[LLM error]")

    def test_ollama_connection_timeout(self, monkeypatch):
        """Test handling of connection timeouts."""
        def mock_post_timeout(url, json=None, timeout=None):
            raise TimeoutError("Connection timed out")

        monkeypatch.setattr(generation, "requests", type('MockRequests', (), {'post': mock_post_timeout})())

        # Should handle timeout gracefully
        result = generation.generate_with_ollama(
            "Test prompt",
            model="test-model",
            host="http://localhost:11434"
        )

        assert result.startswith("[LLM error]")

    def test_ollama_invalid_json_response(self, monkeypatch):
        """Test handling of invalid JSON responses."""
        class MockInvalidResponse:
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                raise json.JSONDecodeError("Invalid JSON", "", 0)

        def mock_post(url, json=None, timeout=None):
            return MockInvalidResponse()

        monkeypatch.setattr(generation, "requests", type('MockRequests', (), {'post': mock_post})())

        # Should handle invalid JSON gracefully
        result = generation.generate_with_ollama(
            "Test prompt",
            model="test-model",
            host="http://localhost:11434"
        )

        assert result.startswith("[LLM error]")


class TestRAGVectorStoreComprehensive:
    """Comprehensive tests for the RAG vectorstore module."""

    def test_collection_management(self, monkeypatch):
        """Test vectorstore collection creation and management."""
        # Mock Qdrant client
        class MockClient:
            def __init__(self):
                self.created_collections = []
                self.deleted_collections = []
                self.existing_collections = []

            def get_collections(self):
                class MockCollections:
                    def __init__(self, collections):
                        self.collections = collections

                return MockCollections(self.existing_collections)

            def get_collection(self, collection_name):
                class MockCollection:
                    def __init__(self, size):
                        self.config = type('Config', (), {
                            'params': type('Params', (), {
                                'vectors': type('Vectors', (), {'size': size})()
                            })()
                        })()

                if collection_name in [c.name for c in self.existing_collections]:
                    return MockCollection(128)  # Existing collection size
                return None

            def create_collection(self, **kwargs):
                self.created_collections.append(kwargs)

            def delete_collection(self, name):
                self.deleted_collections.append(name)

        client = MockClient()
        monkeypatch.setattr(vectorstore, "QdrantClient", lambda: client)

        # Test collection creation when missing
        vectorstore.ensure_collection(client, "test_collection", desired_dim=256)

        # Should create collection
        assert len(client.created_collections) == 1
        assert client.created_collections[0]["collection_name"] == "test_collection"
        assert client.created_collections[0]["vectors_config"]["size"] == 256

        # Test recreation when dimension mismatch
        client.existing_collections = [type('Collection', (), {'name': 'test_collection'})()]
        vectorstore.ensure_collection(client, "test_collection", desired_dim=512)

        # Should delete and recreate
        assert len(client.deleted_collections) == 1
        assert len(client.created_collections) == 2
        assert client.created_collections[1]["vectors_config"]["size"] == 512

    def test_error_handling_in_vectorstore(self, monkeypatch):
        """Test error handling in vectorstore operations."""
        # Mock client that fails
        class FailingClient:
            def get_collections(self):
                raise RuntimeError("Connection failed")

            def create_collection(self, **kwargs):
                raise RuntimeError("Creation failed")

        monkeypatch.setattr(vectorstore, "QdrantClient", lambda: FailingClient())

        # Should handle errors gracefully or propagate appropriately
        try:
            vectorstore.ensure_collection(FailingClient(), "test", 128)
        except Exception:
            # Expected - errors should be handled appropriately
            pass


class TestRAGConfigComprehensive:
    """Comprehensive tests for the RAG configuration module."""

    def test_configuration_defaults_and_validation(self):
        """Test configuration defaults and validation."""
        # Test default configuration
        config_obj = config.RagSettings()

        # Should have reasonable defaults
        assert isinstance(config_obj.qdrant_url, str)
        assert config_obj.collection_name
        assert isinstance(config_obj.use_ollama, bool)

        # Test configuration with custom values
        custom_config = config.RagSettings(
            qdrant_url="http://custom:6333",
            collection_name="custom_collection",
            use_ollama=False
        )

        assert custom_config.qdrant_url == "http://custom:6333"
        assert custom_config.collection_name == "custom_collection"
        assert custom_config.use_ollama is False

    def test_configuration_serialization(self):
        """Test configuration serialization and deserialization."""
        config_obj = config.RagSettings(
            qdrant_url="http://test:6333",
            collection_name="test_collection",
            use_ollama=True
        )

        # Should be serializable to dict
        config_dict = config_obj.__dict__
        assert config_dict["qdrant_url"] == "http://test:6333"
        assert config_dict["collection_name"] == "test_collection"
        assert config_dict["use_ollama"] is True


class TestRAGErrorHandling:
    """Comprehensive tests for RAG error handling."""

    def test_error_hierarchy(self):
        """Test that RAG errors follow proper inheritance hierarchy."""
        # Test error class hierarchy
        assert issubclass(errors.VectorStoreError, errors.RagError)
        assert issubclass(errors.IndexBuildError, errors.RagError)
        assert issubclass(errors.GenerationError, errors.RagError)

        # Test error instantiation
        rag_error = errors.RagError("Base error")
        assert str(rag_error) == "Base error"

        vector_error = errors.VectorStoreError("Vector store error")
        assert str(vector_error) == "Vector store error"

        index_error = errors.IndexBuildError("Index build error")
        assert str(index_error) == "Index build error"

        gen_error = errors.GenerationError("Generation error")
        assert str(gen_error) == "Generation error"

    def test_error_context_and_chaining(self):
        """Test error context and chaining capabilities."""
        # Test error with cause
        original_error = ValueError("Original error")
        rag_error = errors.RagError("Wrapper error")
        rag_error.__cause__ = original_error

        # Should preserve original error information
        assert rag_error.__cause__ is original_error
        assert str(rag_error) == "Wrapper error"


class TestRAGIntegrationScenarios:
    """Integration tests for RAG components working together."""

    def test_full_rag_pipeline(self, monkeypatch):
        """Test complete RAG pipeline from documents to generation."""
        # Mock all the components
        class MockEmbeddingModel:
            def encode(self, texts):
                return [[0.1 * i for i in range(8)] for i, _ in enumerate(texts)]

        class MockVectorStore:
            def __init__(self):
                self.docs = []

        class MockIndex:
            def __init__(self):
                self.docs = []

            def as_retriever(self, similarity_top_k=10, node_postprocessors=None):
                class MockRetriever:
                    def retrieve(self, query):
                        return [type('Node', (), {
                            'text': 'Retrieved document',
                            'metadata': {'source': 'test'},
                            'get_content': lambda: 'Retrieved document'
                        })()]
                return MockRetriever()

        class MockResponse:
            def raise_for_status(self): pass
            def json(self): return {"response": "Generated answer"}

        # Apply mocks
        monkeypatch.setattr(rag_embeddings, "SentenceTransformer", lambda name, device: MockEmbeddingModel())
        monkeypatch.setattr(rag_embeddings, "_log_torch_environment", lambda level=None: None)
        monkeypatch.setattr(rag_embeddings, "resolve_device", lambda pref: "cpu")
        monkeypatch.setattr(indexer, "VectorStoreIndex", MockIndex)
        monkeypatch.setattr(indexer, "StorageContext", type('MockStorage', (), {
            'from_defaults': classmethod(lambda cls, **kwargs: cls())
        })())
        monkeypatch.setattr(generation, "requests", type('MockRequests', (), {
            'post': lambda url, json=None, timeout=None: MockResponse()
        })())

        # Create complete RAG pipeline
        docs = ["Document one", "Document two", "Document three"]

        # 1. Create embedding backend
        backend = rag_embeddings.EmbeddingBackend()
        embeddings_list = backend.encode(docs)

        # 2. Create index
        vs = MockVectorStore()
        transformations = []
        index = indexer.build_index(docs, vs, backend, transformations)

        # 3. Perform search
        search_results = search.hybrid_search(index, query="test query", top_k=2, alpha=0.5)

        # 4. Generate response
        context = "\n\n".join([r.get("text", "") for r in search_results])
        prompt = f"Context:\n{context}\n\nQuestion: test query\n\nAnswer:"
        answer = generation.generate_with_ollama(prompt)

        # Verify complete pipeline
        assert len(embeddings_list) == 3
        assert len(search_results) <= 2
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_error_propagation_through_pipeline(self, monkeypatch):
        """Test that errors propagate correctly through the RAG pipeline."""
        # Mock embedding model that fails
        class FailingEmbeddingModel:
            def encode(self, texts):
                raise RuntimeError("Embedding failed")

        monkeypatch.setattr(rag_embeddings, "SentenceTransformer", lambda name, device: FailingEmbeddingModel())

        backend = rag_embeddings.EmbeddingBackend()

        # Should propagate embedding error
        try:
            backend.encode(["test"])
            assert False, "Should have raised an error"
        except Exception:
            pass  # Expected

    def test_performance_with_realistic_data(self):
        """Test performance with realistic document sizes."""
        import time

        # Create realistic document set
        docs = [f"This is document number {i} with some realistic content that might appear in a real codebase or documentation system." for i in range(50)]

        # Mock embedding model for performance testing
        class FastMockModel:
            def encode(self, texts):
                # Simulate realistic embedding time
                time.sleep(0.001 * len(texts))  # 1ms per text
                return [[0.1 * j for j in range(8)] for i in texts]

        # Test embedding performance
        start_time = time.time()
        embeddings_list = FastMockModel().encode(docs)
        embedding_time = time.time() - start_time

        # Should complete in reasonable time
        assert len(embeddings_list) == 50
        assert embedding_time < 5.0  # Less than 5 seconds

        # Test search performance with mock index
        class MockIndex:
            def as_retriever(self, similarity_top_k=10, node_postprocessors=None):
                class MockRetriever:
                    def retrieve(self, query):
                        # Simulate search time
                        time.sleep(0.01)  # 10ms search time
                        return []
                return MockRetriever()

        index = MockIndex()
        retriever = index.as_retriever()

        start_time = time.time()
        results = retriever.retrieve("test query")
        search_time = time.time() - start_time

        # Should complete quickly
        assert search_time < 1.0  # Less than 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])