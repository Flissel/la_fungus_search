"""
Comprehensive tests for the MCMP (Multi-Agent Cognitive Process Model) module.

Tests all components of the MCMP system including embeddings, simulation,
PCA, visualization, and indexing functionality.
"""

import numpy as np
import pytest
import time
from pathlib import Path
from typing import List, Dict, Any

# Import all MCMP components
from embeddinggemma.mcmp import embeddings
from embeddinggemma.mcmp import simulation
from embeddinggemma.mcmp import pca
from embeddinggemma.mcmp import visualize
from embeddinggemma.mcmp import indexing


class TestMCMPEmbeddingsComprehensive:
    """Comprehensive tests for the embeddings module."""

    def test_load_sentence_model_with_different_devices(self, monkeypatch):
        """Test loading sentence transformer models with different device preferences."""
        calls = []

        class MockModel:
            def __init__(self, name, device=None):
                calls.append((name, device))
                self.model_name = name
                self.device = device

            def encode(self, texts):
                return [[0.1] * 8 for _ in texts]

        monkeypatch.setattr(embeddings, "SentenceTransformer", MockModel)
        monkeypatch.setattr(embeddings, "_log_torch_env", lambda: None)

        # Test CPU preference
        calls.clear()
        model = embeddings.load_sentence_model("test-model", device_preference="cpu")
        assert len(calls) == 1
        assert calls[0] == ("test-model", "cpu")

        # Test CUDA preference with fallback
        calls.clear()
        def mock_resolve_device(pref):
            if pref == "cuda":
                return "cpu"  # Simulate CUDA unavailable
            return pref

        monkeypatch.setattr(embeddings, "resolve_device", mock_resolve_device)

        model = embeddings.load_sentence_model("test-model", device_preference="auto")
        assert len(calls) == 2  # Should try CUDA first, then fallback to CPU
        assert calls[0] == ("test-model", "cuda")
        assert calls[1] == ("test-model", "cpu")

    def test_embedding_backend_functionality(self, monkeypatch):
        """Test the EmbeddingBackend class functionality."""
        calls = []

        class MockModel:
            def __init__(self, name, device=None):
                calls.append((name, device))
                self.model_name = name
                self.device = device

            def encode(self, texts):
                return [[0.1] * 8 for _ in texts]

        monkeypatch.setattr(embeddings, "SentenceTransformer", MockModel)
        monkeypatch.setattr(embeddings, "_log_torch_environment", lambda level=None: None)
        monkeypatch.setattr(embeddings, "resolve_device", lambda pref: "cpu")

        backend = embeddings.EmbeddingBackend(
            model_name="test-model",
            device_preference="auto"
        )

        # Test model loading
        model = backend.load()
        assert isinstance(model, MockModel)
        assert model.model_name == "test-model"
        assert model.device == "cpu"

        # Test encoding
        texts = ["Hello world", "Test document"]
        vectors = backend.encode(texts)
        assert len(vectors) == 2
        assert all(len(v) == 8 for v in vectors)

    def test_device_resolution_logic(self, monkeypatch):
        """Test device resolution with various scenarios."""
        # Test auto resolution when torch is not available
        monkeypatch.setattr(embeddings, "torch", None)
        assert embeddings.resolve_device("auto") == "cpu"

        # Test auto resolution when CUDA is available
        class MockCUDA:
            @staticmethod
            def is_available():
                return True

        class MockTorch:
            cuda = MockCUDA()

        monkeypatch.setattr(embeddings, "torch", MockTorch)
        assert embeddings.resolve_device("auto") == "cuda"

        # Test explicit preferences
        assert embeddings.resolve_device("cpu") == "cpu"
        assert embeddings.resolve_device("cuda") == "cuda"

        # Test invalid preference
        assert embeddings.resolve_device("invalid") == "cpu"


class TestMCMPSimulationComprehensive:
    """Comprehensive tests for the simulation module."""

    def test_agent_position_updates(self):
        """Test agent position update mechanics."""
        # Create mock retriever
        class MockRetriever:
            def __init__(self):
                self.documents = [
                    type('Doc', (), {'id': i, 'embedding': np.random.randn(8)})()
                    for i in range(5)
                ]
                self.pheromone_trails = {}
                self.exploration_bonus = 0.1
                self.kw_lambda = 0.0
                self.kw_terms = set()

            def find_nearest_documents(self, position, k=3):
                # Simple mock implementation
                return [(self.documents[0], 0.5)]

        retriever = MockRetriever()

        # Create test agent
        agent = type('Agent', (), {
            'position': np.random.randn(8),
            'velocity': np.zeros(8),
            'exploration_factor': 0.1,
            'visited_docs': set(),
            'age': 0
        })()

        # Test position update
        initial_position = agent.position.copy()
        simulation.update_agent_position(retriever, agent, iteration=0)

        # Position should have changed
        assert not np.array_equal(agent.position, initial_position)

        # Test with multiple iterations
        for i in range(5):
            simulation.update_agent_position(retriever, agent, iteration=i)

        # Agent should still be in valid position
        assert np.isfinite(agent.position).all()
        assert agent.position.shape == (8,)

    def test_pheromone_mechanics(self):
        """Test pheromone deposition and decay mechanics."""
        class MockRetriever:
            def __init__(self):
                self.pheromone_trails = {}
                self.pheromone_decay = 0.9
                self.documents = [
                    type('Doc', (), {'id': i, 'visit_count': 0, 'last_visited': 0.0})()
                    for i in range(3)
                ]

        retriever = MockRetriever()

        # Create test agent that has visited documents
        agent = type('Agent', (), {
            'position': np.array([1.0, 0.0]),
            'visited_docs': {0, 1}
        })()

        # Initial pheromone trails should be empty
        assert len(retriever.pheromone_trails) == 0

        # Deposit pheromones
        simulation.deposit_pheromones(retriever, agent)

        # Should have created pheromone trails for visited docs
        assert len(retriever.pheromone_trails) > 0

        # All pheromone values should be positive
        for trail_key, strength in retriever.pheromone_trails.items():
            assert strength > 0

        # Test decay
        initial_count = len(retriever.pheromone_trails)
        simulation.decay_pheromones(retriever)

        # Trails should still exist but with reduced strength
        assert len(retriever.pheromone_trails) == initial_count
        for strength in retriever.pheromone_trails.values():
            assert strength < 1.0  # Should be decayed

    def test_document_relevance_update(self):
        """Test document relevance score updates."""
        class MockRetriever:
            def __init__(self):
                self.documents = [
                    type('Doc', (), {
                        'id': i,
                        'embedding': np.random.randn(8),
                        'relevance_score': 0.0,
                        'visit_count': 0,
                        'last_visited': 0.0
                    })()
                    for i in range(5)
                ]
                self.agents = [
                    type('Agent', (), {
                        'position': np.random.randn(8),
                        'visited_docs': {i}
                    })()
                    for i in range(3)
                ]
                self.kw_lambda = 0.1
                self.kw_terms = {"test", "query"}

        retriever = MockRetriever()
        query_embedding = np.random.randn(8)

        # Initial relevance scores should be 0
        for doc in retriever.documents:
            assert doc.relevance_score == 0.0

        # Update relevance
        simulation.update_document_relevance(retriever, query_embedding)

        # Relevance scores should have been updated
        total_relevance = sum(doc.relevance_score for doc in retriever.documents)
        assert total_relevance >= 0  # Should be non-negative

        # Test with different query embeddings
        new_query = np.random.randn(8)
        simulation.update_document_relevance(retriever, new_query)

        # Scores should potentially be different
        new_total = sum(doc.relevance_score for doc in retriever.documents)
        # The exact behavior depends on implementation, but shouldn't crash

    def test_agent_spawning(self):
        """Test agent spawning mechanics."""
        class MockRetriever:
            def __init__(self):
                self.num_agents = 5
                self.agents = []
                self.documents = [
                    type('Doc', (), {'embedding': np.random.randn(8)})()
                    for _ in range(3)
                ]
                self.Agent = lambda id, position, velocity, exploration_factor: type('Agent', (), {
                    'id': id,
                    'position': position,
                    'velocity': velocity,
                    'exploration_factor': exploration_factor,
                    'visited_docs': set(),
                    'age': 0,
                    'trail_strength': 1.0,
                    'energy': 1.0
                })()

        retriever = MockRetriever()
        query_embedding = np.random.randn(8)

        # Initially no agents
        assert len(retriever.agents) == 0

        # Spawn agents
        simulation.spawn_agents(retriever, query_embedding)

        # Should have created the expected number of agents
        assert len(retriever.agents) == 5

        # Each agent should have valid properties
        for agent in retriever.agents:
            assert hasattr(agent, 'id')
            assert hasattr(agent, 'position')
            assert hasattr(agent, 'velocity')
            assert hasattr(agent, 'exploration_factor')
            assert agent.position.shape == (8,)
            assert agent.velocity.shape == (8,)
            assert np.isfinite(agent.position).all()
            assert np.isfinite(agent.velocity).all()


class TestMCMPPCAAndVisualization:
    """Comprehensive tests for PCA and visualization components."""

    def test_pca_2d_with_different_parameters(self):
        """Test PCA with different parameters and edge cases."""
        # Test with simple data
        X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        coords = pca.pca_2d(X, whiten=False)
        assert coords.shape == (4, 2)

        # Test with whitening
        coords_whitened = pca.pca_2d(X, whiten=True)
        assert coords_whitened.shape == (4, 2)

        # Test with larger dimensionality
        X_large = np.random.randn(20, 10).astype(np.float32)
        coords_large = pca.pca_2d(X_large, whiten=False)
        assert coords_large.shape == (20, 2)

        # Test with single sample (edge case)
        X_single = np.array([[1, 2, 3, 4]], dtype=np.float32)
        coords_single = pca.pca_2d(X_single, whiten=False)
        assert coords_single.shape == (1, 2)

        # Test reproducibility with same random seed
        X1 = np.random.RandomState(42).randn(10, 5).astype(np.float32)
        X2 = np.random.RandomState(42).randn(10, 5).astype(np.float32)
        coords1 = pca.pca_2d(X1, whiten=False)
        coords2 = pca.pca_2d(X2, whiten=False)
        np.testing.assert_array_equal(coords1, coords2)

    def test_visualization_snapshot_building(self):
        """Test building visualization snapshots with various inputs."""
        # Create test data
        docs_xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        relevances = [0.1, 0.5, 0.9]
        trails = {(0, 1): 0.2, (1, 2): 0.3, (0, 2): 0.04}  # One below threshold

        # Build snapshot
        snapshot = visualize.build_snapshot(
            docs_xy,
            relevances,
            trails,
            max_edges=2
        )

        # Verify structure
        assert "documents" in snapshot
        assert "agents" in snapshot
        assert "edges" in snapshot
        assert "metadata" in snapshot

        # Check documents data
        assert len(snapshot["documents"]["xy"]) == 3
        assert len(snapshot["documents"]["relevance"]) == 3
        assert snapshot["documents"]["relevance"] == relevances

        # Check edges (should be limited by max_edges)
        assert len(snapshot["edges"]) <= 2

        # Test with no trails
        snapshot_no_trails = visualize.build_snapshot(
            docs_xy,
            relevances,
            {},
            max_edges=10
        )
        assert len(snapshot_no_trails["edges"]) == 0

        # Test with agents
        agents_xy = np.array([[0.5, 0.5]], dtype=float)
        snapshot_with_agents = visualize.build_snapshot(
            docs_xy,
            relevances,
            trails,
            max_edges=10,
            agents_xy=agents_xy
        )
        assert len(snapshot_with_agents["agents"]["xy"]) == 1


class TestMCMPIndexingComprehensive:
    """Comprehensive tests for the indexing module."""

    def test_faiss_index_building(self, monkeypatch):
        """Test FAISS index building with different scenarios."""
        # Mock FAISS as unavailable
        monkeypatch.setattr(indexing, "_FAISS_OK", False)
        embs = np.random.randn(10, 8).astype(np.float32)
        idx = indexing.build_faiss_index(embs, dim=8)
        assert idx is None

        # Mock FAISS as available
        monkeypatch.setattr(indexing, "_FAISS_OK", True)

        # Mock the actual FAISS index creation
        class MockIndex:
            def __init__(self, embs, dim):
                self.ntotal = len(embs)
                self.d = dim

        def mock_build_faiss(embs, dim):
            return MockIndex(embs, dim)

        monkeypatch.setattr(indexing, "_build_faiss_impl", mock_build_faiss)

        idx = indexing.build_faiss_index(embs, dim=8)
        assert idx is not None
        assert idx.ntotal == 10
        assert idx.d == 8

    def test_cosine_similarity_calculations(self):
        """Test cosine similarity calculations with various inputs."""
        # Test basic case
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        docs = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        sims = indexing.cosine_similarities_cpu(q, docs)
        assert sims.shape == (3,)
        assert np.isclose(sims[0], 1.0, atol=1e-5)  # Perfect match
        assert sims[1] < -0.99  # Opposite direction
        assert abs(sims[2]) < 1e-6  # Orthogonal

        # Test with identical vectors
        identical_docs = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        sims_identical = indexing.cosine_similarities_cpu(q[:2], identical_docs)
        assert np.allclose(sims_identical, [1.0, 1.0])

        # Test edge case: zero vector query
        zero_query = np.zeros(3, dtype=np.float32)
        sims_zero = indexing.cosine_similarities_cpu(zero_query, docs)
        assert np.allclose(sims_zero, 0.0)  # All similarities should be 0

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        # Test with large batch
        large_q = np.random.randn(100, 8).astype(np.float32)
        large_docs = np.random.randn(50, 8).astype(np.float32)

        start_time = time.time()
        sims = indexing.cosine_similarities_cpu(large_q, large_docs)
        elapsed = time.time() - start_time

        # Should complete in reasonable time and have correct shape
        assert sims.shape == (100, 50)
        assert elapsed < 5.0  # Should be fast for this size

        # Test memory efficiency - should not create extremely large intermediate arrays
        assert sims.dtype == np.float32


class TestMCMPIntegrationScenarios:
    """Integration tests for MCMP components working together."""

    def test_full_embedding_to_visualization_pipeline(self):
        """Test complete pipeline from embeddings to visualization."""
        # Create sample documents
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological systems.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret images.",
            "Deep learning uses multiple layers of neural networks."
        ]

        # Initialize retriever (using mock embedding model)
        retriever = type('MockRetriever', (), {
            'num_agents': 20,
            'max_iterations': 10,
            'exploration_bonus': 0.1,
            'pheromone_decay': 0.9,
            'documents': [],
            'agents': [],
            'pheromone_trails': {},
            'embedding_model': None,
            'Agent': lambda id, position, velocity, exploration_factor: type('Agent', (), {
                'id': id, 'position': position, 'velocity': velocity,
                'exploration_factor': exploration_factor, 'visited_docs': set(),
                'age': 0, 'trail_strength': 1.0, 'energy': 1.0
            })()
        })()

        # Add documents with mock embeddings
        for i, doc in enumerate(docs):
            embedding = np.random.randn(8).astype(np.float32)
            embedding /= np.linalg.norm(embedding)  # Normalize
            doc_obj = type('Document', (), {
                'id': i,
                'content': doc,
                'embedding': embedding,
                'relevance_score': 0.0,
                'visit_count': 0,
                'last_visited': 0.0,
                'metadata': {}
            })()
            retriever.documents.append(doc_obj)

        # Initialize simulation
        query_embedding = np.random.randn(8).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)

        simulation.spawn_agents(retriever, query_embedding)
        assert len(retriever.agents) == 20

        # Run simulation steps
        for _ in range(5):
            for agent in retriever.agents:
                simulation.update_agent_position(retriever, agent, 0)
                simulation.deposit_pheromones(retriever, agent)
            simulation.decay_pheromones(retriever)
            simulation.update_document_relevance(retriever, query_embedding)

        # Test visualization
        embeddings = np.array([d.embedding for d in retriever.documents])
        coords = pca.pca_2d(embeddings, whiten=False)
        assert coords.shape == (5, 2)

        # Build snapshot
        relevances = [d.relevance_score for d in retriever.documents]
        snapshot = visualize.build_snapshot(
            coords,
            relevances,
            retriever.pheromone_trails,
            max_edges=10
        )

        assert "documents" in snapshot
        assert "edges" in snapshot
        assert len(snapshot["documents"]["xy"]) == 5

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases across MCMP components."""
        # Test with empty data
        empty_embeddings = np.zeros((0, 8), dtype=np.float32)
        empty_coords = pca.pca_2d(empty_embeddings, whiten=False)
        assert empty_coords.shape == (0, 2)

        # Test with single dimension
        single_dim = np.array([[1], [2], [3]], dtype=np.float32)
        single_coords = pca.pca_2d(single_dim, whiten=False)
        assert single_coords.shape == (3, 2)

        # Test visualization with no data
        empty_snapshot = visualize.build_snapshot(
            np.zeros((0, 2)),
            [],
            {},
            max_edges=10
        )
        assert len(empty_snapshot["documents"]["xy"]) == 0
        assert len(empty_snapshot["edges"]) == 0

        # Test cosine similarity edge cases
        zero_vec = np.zeros(3, dtype=np.float32)
        normal_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        docs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        # Should handle zero vectors gracefully
        sims = indexing.cosine_similarities_cpu(zero_vec, docs)
        assert sims.shape == (1,)
        assert np.allclose(sims, 0.0)

        # Test with very small vectors
        tiny_vec = np.array([1e-10, 0.0, 0.0], dtype=np.float32)
        sims_tiny = indexing.cosine_similarities_cpu(tiny_vec, docs)
        assert sims_tiny.shape == (1,)

    def test_performance_characteristics(self):
        """Test performance characteristics of MCMP components."""
        # Test PCA performance with larger datasets
        large_data = np.random.randn(1000, 50).astype(np.float32)

        start_time = time.time()
        coords = pca.pca_2d(large_data, whiten=False)
        pca_time = time.time() - start_time

        assert coords.shape == (1000, 2)
        assert pca_time < 5.0  # Should be reasonably fast

        # Test cosine similarity performance
        queries = np.random.randn(100, 50).astype(np.float32)
        docs = np.random.randn(200, 50).astype(np.float32)

        start_time = time.time()
        sims = indexing.cosine_similarities_cpu(queries, docs)
        sim_time = time.time() - start_time

        assert sims.shape == (100, 200)
        assert sim_time < 10.0  # Should be reasonably fast

        # Test memory usage doesn't grow excessively
        import gc
        initial_objects = len(gc.get_objects())

        # Run multiple operations
        for _ in range(5):
            _ = pca.pca_2d(large_data, whiten=True)
            _ = indexing.cosine_similarities_cpu(queries, docs)

        final_objects = len(gc.get_objects())
        growth_ratio = final_objects / max(initial_objects, 1)
        assert growth_ratio < 2.0  # Reasonable memory growth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])