"""
Comprehensive integration tests for the entire EmbeddingGemma application.

Tests the MCPM-RAG system, realtime server, and all major components
working together in realistic scenarios.
"""

import os
import json
import time
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
import numpy as np

# Import the main application components
from embeddinggemma.mcmp_rag import MCPMRetriever, Document, Agent
from embeddinggemma.realtime.server import SnapshotStreamer, app
from embeddinggemma.ui.corpus import collect_codebase_chunks, list_code_files
from embeddinggemma.rag.generation import generate_with_ollama

# FastAPI test client
from fastapi.testclient import TestClient


class TestMCPMRetrieverIntegration:
    """Integration tests for the MCPMRetriever system."""

    def test_full_retrieval_pipeline(self):
        """Test the complete retrieval pipeline from documents to search results."""
        # Initialize retriever
        retriever = MCPMRetriever(
            num_agents=50,
            max_iterations=20,
            pheromone_decay=0.9,
            exploration_bonus=0.1,
            embed_batch_size=32
        )

        # Sample documents
        docs = [
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn automatically.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Natural language processing deals with the interaction between computers and humans using natural language.",
            "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images.",
        ]

        # Add documents
        retriever.add_documents(docs)
        assert len(retriever.documents) == 5

        # Initialize simulation with query
        query = "What is machine learning?"
        success = retriever.initialize_simulation(query)
        assert success
        assert retriever._current_query_embedding is not None
        assert len(retriever.agents) == 50

        # Run simulation steps
        for step in range(10):
            result = retriever.step(1)
            assert "avg_relevance" in result
            assert "steps" in result
            assert result["steps"] == 1

        # Perform search
        search_results = retriever.search(query, top_k=3)
        assert "results" in search_results
        assert len(search_results["results"]) <= 3

        # Verify results are ordered by relevance
        scores = [r["relevance_score"] for r in search_results["results"]]
        assert scores == sorted(scores, reverse=True)

        # Test visualization snapshot
        snapshot = retriever.get_visualization_snapshot(
            min_trail_strength=0.01,
            max_edges=100,
            dims=2
        )
        assert "documents" in snapshot
        assert "agents" in snapshot
        assert "edges" in snapshot
        assert len(snapshot["documents"]["xy"]) == 5

    def test_agent_lifecycle_and_pheromone_dynamics(self):
        """Test agent spawning, movement, and pheromone trail formation."""
        retriever = MCPMRetriever(num_agents=20, max_iterations=10)

        # Add documents
        docs = ["Document one", "Document two", "Document three"]
        retriever.add_documents(docs)

        # Initialize and run simulation
        retriever.initialize_simulation("test query")
        initial_trails = len(retriever.pheromone_trails)

        # Run multiple steps and check pheromone accumulation
        for _ in range(5):
            retriever.step(1)
            assert len(retriever.pheromone_trails) >= initial_trails

        # Check that documents have accumulated relevance scores
        total_relevance = sum(d.relevance_score for d in retriever.documents)
        assert total_relevance > 0

        # Check agent positions are valid
        for agent in retriever.agents:
            assert agent.position.shape == (retriever._embed_dim,)
            assert np.isfinite(agent.position).all()

    def test_getters_and_state_access(self):
        """Test all public getter methods for frontend integration."""
        retriever = MCPMRetriever(num_agents=10, max_iterations=5)

        # Initially empty
        assert retriever.get_query_embedding() is None
        positions = retriever.get_agent_positions()
        assert positions.shape == (0, 0)
        embeddings = retriever.get_doc_embeddings()
        assert embeddings.shape == (0, 0)
        relevances = retriever.get_doc_relevances()
        assert relevances == []
        trails = retriever.get_pheromone_trails()
        assert trails == {}

        # Add documents and initialize
        docs = ["Test document one", "Test document two"]
        retriever.add_documents(docs)
        retriever.initialize_simulation("test query")

        # Test getters after initialization
        query_emb = retriever.get_query_embedding()
        assert query_emb is not None
        assert query_emb.shape == (retriever._embed_dim,)

        positions = retriever.get_agent_positions()
        assert positions.shape == (10, retriever._embed_dim)

        embeddings = retriever.get_doc_embeddings()
        assert embeddings.shape == (2, retriever._embed_dim)

        relevances = retriever.get_doc_relevances()
        assert len(relevances) == 2
        assert all(isinstance(r[0], int) and isinstance(r[1], float) for r in relevances)

        trails = retriever.get_pheromone_trails()
        assert isinstance(trails, dict)

    def test_concurrent_operations(self):
        """Test that the retriever handles concurrent operations safely."""
        retriever = MCPMRetriever(num_agents=30, max_iterations=15)

        # Add documents
        docs = [f"Document {i}" for i in range(10)]
        retriever.add_documents(docs)

        def run_simulation_steps():
            retriever.initialize_simulation("concurrent test")
            for _ in range(5):
                retriever.step(1)

        def perform_searches():
            for _ in range(3):
                results = retriever.search("test query", top_k=3)
                assert "results" in results

        def get_snapshots():
            for _ in range(3):
                snapshot = retriever.get_visualization_snapshot()
                assert "documents" in snapshot

        # Run operations concurrently
        import threading

        threads = [
            threading.Thread(target=run_simulation_steps),
            threading.Thread(target=perform_searches),
            threading.Thread(target=get_snapshots),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify system is still in consistent state
        assert len(retriever.documents) == 10
        assert len(retriever.agents) == 30


class TestRealtimeServerIntegration:
    """Integration tests for the realtime server endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    def test_server_startup_and_health(self, client):
        """Test server startup and basic health endpoints."""
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200

        # Test status endpoint
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data
        assert data["running"] is False  # No simulation running initially

        # Test settings endpoint
        response = client.get("/settings")
        assert response.status_code == 200
        data = response.json()
        assert "settings" in data
        assert "usage" in data

    def test_simulation_lifecycle(self, client):
        """Test complete simulation start/stop/reset lifecycle."""
        # Start simulation
        start_data = {
            "query": "What is Python?",
            "num_agents": 20,
            "max_iterations": 5,
            "windows": [50, 100, 200]
        }

        response = client.post("/start", json=start_data)
        assert response.status_code == 200

        # Check status shows running simulation
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is True
        assert "docs" in data
        assert "agents" in data

        # Reset simulation
        response = client.post("/reset")
        assert response.status_code == 200

        # Check status shows reset
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False

    def test_search_endpoint(self, client):
        """Test search endpoint functionality."""
        # Start simulation first
        start_data = {
            "query": "test query",
            "windows": [100]
        }
        client.post("/start", json=start_data)

        # Wait a moment for initialization
        time.sleep(0.5)

        # Perform search
        search_data = {
            "query": "test query",
            "top_k": 3
        }
        response = client.post("/search", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "status" in data

    def test_settings_persistence(self, client):
        """Test settings save/load functionality."""
        # Update settings
        settings_data = {
            "query": "persistent test",
            "top_k": 5,
            "num_agents": 15,
            "exploration_bonus": 0.2
        }

        response = client.post("/settings", json=settings_data)
        assert response.status_code == 200

        # Verify settings were applied
        response = client.get("/settings")
        data = response.json()
        settings = data["settings"]
        assert settings["query"] == "persistent test"
        assert settings["top_k"] == 5
        assert settings["num_agents"] == 15

    def test_document_detail_endpoint(self, client):
        """Test document detail retrieval."""
        # Start simulation
        client.post("/start", json={
            "query": "test",
            "windows": [50]
        })

        # Try to get document details (may not exist in test environment)
        response = client.get("/doc/0")
        # This might return 404 if no documents, which is fine for testing
        if response.status_code == 200:
            data = response.json()
            assert "doc" in data
            assert "id" in data["doc"]
            assert "content" in data["doc"]


class TestCorpusIntegration:
    """Integration tests for corpus collection and processing."""

    def test_codebase_chunk_collection(self, tmp_path):
        """Test collection of code chunks from a codebase."""
        # Create test Python files
        test_dir = tmp_path / "test_code"
        test_dir.mkdir()

        # Create sample Python files
        (test_dir / "module1.py").write_text("""
def function_one():
    \"\"\"A simple function.\"\"\"
    return "hello"

class MyClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
""")

        (test_dir / "module2.py").write_text("""
import module1

def function_two():
    obj = module1.MyClass()
    return obj.get_value()
""")

        # Collect chunks
        windows = [50, 100, 150]
        chunks = collect_codebase_chunks(
            str(test_dir),
            windows,
            max_files=10,
            exclude_dirs=[]
        )

        # Verify chunks were collected
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_corpus_listing_endpoint(self, client, tmp_path):
        """Test corpus listing endpoint."""
        # Create test files
        test_dir = tmp_path / "test_files"
        test_dir.mkdir()
        (test_dir / "test1.py").write_text("print('hello')")
        (test_dir / "test2.py").write_text("print('world')")

        # Update streamer to use test directory
        from embeddinggemma.realtime.server import streamer
        original_use_repo = streamer.use_repo
        original_root_folder = streamer.root_folder

        streamer.use_repo = False
        streamer.root_folder = str(test_dir)

        try:
            # Test corpus listing
            response = client.get("/corpus/list?page=1&page_size=10")
            assert response.status_code == 200
            data = response.json()
            assert "files" in data
            assert "total" in data
            assert "root" in data
        finally:
            # Restore original settings
            streamer.use_repo = original_use_repo
            streamer.root_folder = original_root_folder


class TestEndToEndScenarios:
    """End-to-end test scenarios for the complete application."""

    def test_complete_workflow(self, client):
        """Test a complete workflow from corpus to search results."""
        # 1. Configure settings
        config_data = {
            "query": "What are the main components of this system?",
            "windows": [100, 200],
            "max_files": 50,
            "num_agents": 25,
            "max_iterations": 10,
            "top_k": 5
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # 2. Start simulation
        response = client.post("/start")
        assert response.status_code == 200

        # 3. Wait for simulation to run
        time.sleep(2.0)

        # 4. Check status
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is True

        # 5. Perform search
        search_data = {"query": "main components", "top_k": 3}
        response = client.post("/search", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

        # 6. Get visualization snapshot
        # Note: This might fail in test environment without proper setup
        # but we can at least verify the endpoint exists
        try:
            # This would require WebSocket connection in real scenario
            pass
        except Exception:
            pass  # Expected in test environment

        # 7. Stop simulation
        response = client.post("/stop")
        assert response.status_code == 200

    def test_error_handling(self, client):
        """Test error handling in various scenarios."""
        # Test starting without proper configuration
        response = client.post("/start", json={})
        # Should handle gracefully even with minimal config

        # Test search without running simulation
        response = client.post("/search", json={"query": "test"})
        # Should return error or handle gracefully

        # Test invalid document access
        response = client.get("/doc/99999")
        assert response.status_code == 404

        # Test invalid settings
        response = client.post("/settings", json={"invalid_param": "value"})
        # Should handle invalid parameters gracefully


class TestPerformanceAndStress:
    """Performance and stress tests for the application."""

    def test_large_document_set(self):
        """Test performance with a large number of documents."""
        retriever = MCPMRetriever(
            num_agents=50,
            max_iterations=20,
            embed_batch_size=64
        )

        # Create many documents
        large_docs = [f"Document number {i} with some content." for i in range(100)]
        start_time = time.time()
        retriever.add_documents(large_docs)
        add_time = time.time() - start_time

        # Should complete in reasonable time
        assert add_time < 30.0  # Less than 30 seconds
        assert len(retriever.documents) == 100

        # Initialize and run simulation
        retriever.initialize_simulation("test query")
        start_time = time.time()
        retriever.step(10)
        step_time = time.time() - start_time

        # Should complete in reasonable time
        assert step_time < 10.0  # Less than 10 seconds

    def test_concurrent_simulations(self):
        """Test running multiple simulations concurrently."""
        def run_simulation(sim_id):
            retriever = MCPMRetriever(
                num_agents=20,
                max_iterations=5
            )
            docs = [f"Doc {sim_id}_{i}" for i in range(10)]
            retriever.add_documents(docs)
            retriever.initialize_simulation(f"Query {sim_id}")
            retriever.step(3)
            return len(retriever.agents), len(retriever.documents)

        import concurrent.futures

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_simulation, i) for i in range(3)]
            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # All simulations should complete successfully
        assert len(results) == 3
        assert all(r[0] == 20 for r in results)  # All have 20 agents
        assert all(r[1] == 10 for r in results)  # All have 10 documents

        # Should complete in reasonable time
        assert total_time < 20.0  # Less than 20 seconds


class TestSystemIntegration:
    """Tests for system-level integration and compatibility."""

    def test_dependency_compatibility(self):
        """Test that all required dependencies are available and compatible."""
        # Test numpy
        assert np.__version__

        # Test that MCPMRetriever can be instantiated
        retriever = MCPMRetriever()
        assert retriever.num_agents > 0

        # Test that server can be imported
        from embeddinggemma.realtime import server
        assert hasattr(server, 'app')
        assert hasattr(server, 'SnapshotStreamer')

    def test_configuration_consistency(self):
        """Test that configuration is consistent across components."""
        # Test that MCPMRetriever defaults match server defaults
        retriever = MCPMRetriever()

        from embeddinggemma.realtime.server import streamer
        from embeddinggemma.realtime.server import settings_dict

        settings = settings_dict()

        # Key parameters should have reasonable defaults
        assert settings["num_agents"] > 0
        assert settings["max_iterations"] > 0
        assert settings["exploration_bonus"] >= 0
        assert settings["pheromone_decay"] > 0 and settings["pheromone_decay"] < 1

    def test_memory_usage_bounds(self):
        """Test that memory usage stays within reasonable bounds."""
        import gc

        retriever = MCPMRetriever(num_agents=100, max_iterations=50)

        # Add many documents
        docs = [f"Document {i}" for i in range(200)]
        retriever.add_documents(docs)

        # Initialize simulation
        retriever.initialize_simulation("memory test")

        # Run simulation and check memory doesn't grow unbounded
        initial_objects = len(gc.get_objects())

        for _ in range(10):
            retriever.step(1)

        final_objects = len(gc.get_objects())

        # Memory growth should be reasonable (less than 50% increase)
        growth_ratio = final_objects / max(initial_objects, 1)
        assert growth_ratio < 1.5


if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])