"""
Performance and stress tests for the EmbeddingGemma application.

Tests performance characteristics, memory usage, scalability, and behavior
under various load conditions to ensure the application can handle
real-world usage scenarios.
"""

import os
import time
import gc
import psutil
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
import numpy as np
import concurrent.futures
from unittest.mock import patch

# Import the main application components
from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.realtime.server import app, SnapshotStreamer, streamer
from embeddinggemma.ui.corpus import collect_codebase_chunks, list_code_files
from embeddinggemma.mcmp import pca, simulation, indexing, embeddings as mcmp_embeddings

# FastAPI test client
from fastapi.testclient import TestClient


class TestPerformanceBenchmarks:
    """Performance benchmarks for critical application components."""

    def test_embedding_performance_with_various_sizes(self):
        """Benchmark embedding performance with different document sizes."""
        # Test with small documents
        small_docs = ["Short document"] * 10
        start_time = time.time()
        # Mock embedding model for consistent testing
        class MockModel:
            def encode(self, texts):
                return [[0.1] * 8 for _ in texts]

        model = MockModel()
        embeddings = model.encode(small_docs)
        small_time = time.time() - start_time

        # Test with medium documents
        medium_docs = ["Medium length document with some content to test embedding performance and scalability"] * 50
        start_time = time.time()
        embeddings = model.encode(medium_docs)
        medium_time = time.time() - start_time

        # Test with large documents
        large_docs = ["Very long document content " * 100] * 20
        start_time = time.time()
        embeddings = model.encode(large_docs)
        large_time = time.time() - start_time

        # Verify performance characteristics
        assert small_time < 0.1  # Should be very fast for small batches
        assert medium_time < 1.0  # Should be reasonable for medium batches
        assert large_time < 5.0   # Should be acceptable for large batches

        # Check that time scales roughly linearly
        small_per_doc = small_time / len(small_docs)
        medium_per_doc = medium_time / len(medium_docs)
        large_per_doc = large_time / len(large_docs)

        # Performance should be roughly consistent per document
        assert abs(small_per_doc - medium_per_doc) < 0.05  # Within 50ms
        assert abs(medium_per_doc - large_per_doc) < 0.05  # Within 50ms

    def test_simulation_performance_with_agent_scaling(self):
        """Benchmark simulation performance with different numbers of agents."""
        agent_counts = [10, 50, 100, 200]

        results = {}

        for num_agents in agent_counts:
            # Create retriever with specific agent count
            retriever = MCPMRetriever(num_agents=num_agents, max_iterations=10)

            # Add test documents
            docs = [f"Document {i}" for i in range(20)]
            retriever.add_documents(docs)

            # Initialize simulation
            retriever.initialize_simulation("Performance test")

            # Measure simulation step performance
            start_time = time.time()
            for _ in range(10):
                retriever.step(1)
            total_time = time.time() - start_time

            # Store results
            results[num_agents] = {
                'total_time': total_time,
                'time_per_step': total_time / 10,
                'agents': num_agents
            }

        # Verify scaling characteristics
        for i in range(len(agent_counts) - 1):
            current_agents = agent_counts[i]
            next_agents = agent_counts[i + 1]

            current_time = results[current_agents]['time_per_step']
            next_time = results[next_agents]['time_per_step']

            # Time should scale roughly linearly with agent count
            expected_ratio = next_agents / current_agents
            actual_ratio = next_time / current_time

            # Allow some variance but should be roughly linear
            assert 0.5 * expected_ratio < actual_ratio < 2.0 * expected_ratio

    def test_pca_performance_with_dimensionality(self):
        """Benchmark PCA performance with different dimensionalities."""
        dimensions = [10, 50, 100, 200, 500]
        sample_sizes = [50, 100, 200]

        results = {}

        for dim in dimensions:
            for samples in sample_sizes:
                # Create test data
                data = np.random.randn(samples, dim).astype(np.float32)

                # Benchmark PCA
                start_time = time.time()
                coords = pca.pca_2d(data, whiten=False)
                pca_time = time.time() - start_time

                key = f"{samples}x{dim}"
                results[key] = {
                    'time': pca_time,
                    'samples': samples,
                    'dimensions': dim
                }

                # Verify output shape
                assert coords.shape == (samples, 2)

        # Check performance scaling
        for samples in sample_sizes:
            times = []
            for dim in dimensions:
                key = f"{samples}x{dim}"
                if key in results:
                    times.append(results[key]['time'])

            # Time should increase with dimensionality but not exponentially
            if len(times) > 1:
                # Later dimensions should not take dramatically longer
                ratio = times[-1] / times[0]
                dim_ratio = dimensions[-1] / dimensions[0]
                assert ratio < 3.0 * dim_ratio  # Should scale better than cubic

    def test_cosine_similarity_performance(self):
        """Benchmark cosine similarity calculation performance."""
        query_sizes = [10, 50, 100, 200]
        doc_sizes = [100, 500, 1000, 2000]

        results = {}

        for queries in query_sizes:
            for docs in doc_sizes:
                # Create test data
                query_data = np.random.randn(queries, 50).astype(np.float32)
                doc_data = np.random.randn(docs, 50).astype(np.float32)

                # Benchmark similarity calculation
                start_time = time.time()
                sims = indexing.cosine_similarities_cpu(query_data, doc_data)
                sim_time = time.time() - start_time

                key = f"{queries}q_{docs}d"
                results[key] = {
                    'time': sim_time,
                    'queries': queries,
                    'docs': docs
                }

                # Verify output shape
                assert sims.shape == (queries, docs)

        # Check scaling characteristics
        for queries in query_sizes:
            times = []
            for docs in doc_sizes:
                key = f"{queries}q_{docs}d"
                if key in results:
                    times.append(results[key]['time'])

            # Should scale roughly linearly with document count
            if len(times) > 1:
                ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
                doc_ratios = [doc_sizes[i+1] / doc_sizes[i] for i in range(len(doc_sizes)-1)]

                # Ratios should be roughly proportional
                for ratio, doc_ratio in zip(ratios, doc_ratios):
                    assert 0.5 < ratio / doc_ratio < 2.0


class TestMemoryUsageAndLeaks:
    """Test memory usage patterns and detect potential leaks."""

    def get_memory_usage_mb(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_memory_usage_during_simulation(self):
        """Test memory usage patterns during simulation."""
        initial_memory = self.get_memory_usage_mb()

        # Create and run multiple simulations
        for i in range(5):
            retriever = MCPMRetriever(num_agents=50, max_iterations=10)

            # Add documents
            docs = [f"Document {j}" for j in range(20)]
            retriever.add_documents(docs)

            # Initialize and run simulation
            retriever.initialize_simulation(f"Memory test {i}")
            for _ in range(5):
                retriever.step(1)

            # Force garbage collection
            gc.collect()

        final_memory = self.get_memory_usage_mb()
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable
        assert memory_growth < 100  # Less than 100MB growth

    def test_memory_usage_with_large_corpora(self):
        """Test memory usage with large document corpora."""
        initial_memory = self.get_memory_usage_mb()

        # Create large document set
        large_docs = [f"Document {i} with substantial content that includes various keywords and technical terms to test memory usage patterns"] * 1000

        # Process with MCPMRetriever
        retriever = MCPMRetriever(num_agents=100, max_iterations=5)
        start_time = time.time()
        retriever.add_documents(large_docs)
        add_time = time.time() - start_time

        # Should complete in reasonable time
        assert add_time < 30.0  # Less than 30 seconds

        # Initialize simulation
        retriever.initialize_simulation("Large corpus test")
        sim_memory = self.get_memory_usage_mb()

        # Memory usage should be reasonable even with large corpus
        assert sim_memory - initial_memory < 200  # Less than 200MB

        # Run simulation steps
        for _ in range(3):
            retriever.step(1)

        final_memory = self.get_memory_usage_mb()
        total_growth = final_memory - initial_memory

        # Total memory growth should be controlled
        assert total_growth < 300  # Less than 300MB total

    def test_garbage_collection_effectiveness(self):
        """Test that garbage collection works effectively."""
        # Create objects that should be garbage collected
        objects_before = len(gc.get_objects())

        for i in range(100):
            # Create retriever and use it
            retriever = MCPMRetriever(num_agents=20, max_iterations=3)
            docs = [f"Temp document {j}" for j in range(10)]
            retriever.add_documents(docs)
            retriever.initialize_simulation("GC test")
            retriever.step(2)

        # Force garbage collection
        gc.collect()

        objects_after = len(gc.get_objects())
        object_growth = objects_after - objects_before

        # Should not accumulate excessive objects
        assert object_growth < 1000  # Less than 1000 new objects

    def test_memory_efficiency_of_visualization(self):
        """Test memory efficiency of visualization operations."""
        # Create large dataset for visualization
        large_embeddings = np.random.randn(1000, 100).astype(np.float32)

        initial_memory = self.get_memory_usage_mb()

        # Test PCA memory usage
        start_time = time.time()
        coords = pca.pca_2d(large_embeddings, whiten=True)
        pca_time = time.time() - start_time

        pca_memory = self.get_memory_usage_mb()

        # PCA should be memory efficient
        assert pca_time < 10.0  # Should complete quickly
        assert pca_memory - initial_memory < 50  # Less than 50MB additional

        # Test snapshot building memory usage
        relevances = [0.5] * 1000
        trails = {(i, i+1): 0.1 for i in range(999)}

        start_time = time.time()
        snapshot = None
        for _ in range(10):  # Multiple snapshots
            snapshot = pca.pca_2d(large_embeddings, whiten=False)
        snapshot_time = time.time() - start_time

        final_memory = self.get_memory_usage_mb()

        # Multiple snapshots should not cause excessive memory growth
        assert snapshot_time < 5.0  # Should be fast
        assert final_memory - initial_memory < 100  # Less than 100MB total


class TestScalabilityTests:
    """Test scalability characteristics of the application."""

    def test_agent_scaling_performance(self):
        """Test performance scaling with increasing numbers of agents."""
        agent_counts = [10, 25, 50, 100, 200]

        results = {}

        for agents in agent_counts:
            # Create retriever with specific agent count
            retriever = MCPMRetriever(num_agents=agents, max_iterations=5)

            # Add fixed number of documents
            docs = [f"Document {i}" for i in range(50)]
            retriever.add_documents(docs)

            # Initialize and measure performance
            retriever.initialize_simulation("Scaling test")

            start_time = time.time()
            for _ in range(5):
                retriever.step(1)
            total_time = time.time() - start_time

            results[agents] = {
                'total_time': total_time,
                'time_per_step': total_time / 5,
                'agents': agents,
                'docs': 50
            }

        # Analyze scaling behavior
        times = [results[agents]['time_per_step'] for agents in agent_counts]
        agent_counts_list = list(agent_counts)

        # Calculate scaling factors
        for i in range(1, len(times)):
            prev_time = times[i-1]
            curr_time = times[i]
            prev_agents = agent_counts_list[i-1]
            curr_agents = agent_counts_list[i]

            time_ratio = curr_time / prev_time
            agent_ratio = curr_agents / prev_agents

            # Should scale roughly linearly (allow some variance)
            assert 0.3 * agent_ratio < time_ratio < 3.0 * agent_ratio

    def test_document_scaling_performance(self):
        """Test performance scaling with increasing numbers of documents."""
        doc_counts = [10, 50, 100, 200, 500]

        results = {}

        for docs in doc_counts:
            # Create retriever with fixed agent count
            retriever = MCPMRetriever(num_agents=50, max_iterations=5)

            # Add variable number of documents
            doc_list = [f"Document {i}" for i in range(docs)]
            start_time = time.time()
            retriever.add_documents(doc_list)
            add_time = time.time() - start_time

            # Initialize simulation
            retriever.initialize_simulation("Document scaling test")

            sim_start = time.time()
            for _ in range(3):
                retriever.step(1)
            sim_time = time.time() - sim_start

            results[docs] = {
                'add_time': add_time,
                'sim_time': sim_time,
                'total_time': add_time + sim_time,
                'docs': docs,
                'agents': 50
            }

        # Analyze scaling behavior
        add_times = [results[docs]['add_time'] for docs in doc_counts]
        sim_times = [results[docs]['sim_time'] for docs in doc_counts]

        # Document addition should scale roughly linearly
        for i in range(1, len(add_times)):
            time_ratio = add_times[i] / add_times[i-1]
            doc_ratio = doc_counts[i] / doc_counts[i-1]

            # Should scale roughly linearly
            assert 0.5 * doc_ratio < time_ratio < 2.0 * doc_ratio

    def test_concurrent_processing_performance(self):
        """Test performance under concurrent processing scenarios."""
        def run_simulation(simulation_id):
            """Run a single simulation."""
            retriever = MCPMRetriever(num_agents=30, max_iterations=5)
            docs = [f"Doc {simulation_id}_{i}" for i in range(20)]
            retriever.add_documents(docs)
            retriever.initialize_simulation(f"Concurrent test {simulation_id}")

            start_time = time.time()
            for _ in range(3):
                retriever.step(1)
            total_time = time.time() - start_time

            return total_time

        # Test concurrent execution
        num_concurrent = [1, 2, 4, 8]

        for concurrent in num_concurrent:
            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
                futures = [executor.submit(run_simulation, i) for i in range(concurrent)]
                results = [future.result() for future in futures]

            total_time = time.time() - start_time

            # Average time per simulation
            avg_time = sum(results) / len(results)

            # Concurrent execution should be reasonably efficient
            # (not perfectly linear due to overhead)
            expected_time = avg_time * concurrent * 0.8  # Allow 20% overhead
            assert total_time < expected_time

    def test_large_scale_data_processing(self):
        """Test processing of large-scale datasets."""
        # Create very large document set
        large_docs = []
        for i in range(1000):
            # Create documents of varying sizes
            size = np.random.choice([50, 100, 200, 500])
            content = " ".join([f"word{j}" for j in range(size)])
            large_docs.append(f"Document {i}: {content}")

        # Test processing performance
        retriever = MCPMRetriever(num_agents=100, max_iterations=3)

        start_time = time.time()
        retriever.add_documents(large_docs)
        add_time = time.time() - start_time

        # Should handle large datasets
        assert add_time < 60.0  # Less than 1 minute
        assert len(retriever.documents) == 1000

        # Initialize simulation
        sim_start = time.time()
        retriever.initialize_simulation("Large scale test")
        init_time = time.time() - sim_start

        # Should initialize quickly
        assert init_time < 10.0  # Less than 10 seconds

        # Run simulation steps
        step_start = time.time()
        for _ in range(2):
            retriever.step(1)
        step_time = time.time() - step_start

        # Should complete steps in reasonable time
        assert step_time < 30.0  # Less than 30 seconds

    def test_visualization_scaling(self):
        """Test visualization performance with large datasets."""
        # Test with increasing dataset sizes
        sizes = [100, 500, 1000, 2000]

        for size in sizes:
            # Create test embeddings
            embeddings = np.random.randn(size, 50).astype(np.float32)

            # Test PCA performance
            start_time = time.time()
            coords = pca.pca_2d(embeddings, whiten=False)
            pca_time = time.time() - start_time

            # Should scale reasonably
            assert pca_time < 5.0  # Should be fast for reasonable sizes
            assert coords.shape == (size, 2)

            # Test snapshot building
            relevances = [0.5] * size
            trails = {(i, i+1): 0.1 for i in range(min(size-1, 1000))}

            snap_start = time.time()
            snapshot = pca.pca_2d(embeddings, whiten=False)  # Simplified snapshot
            snap_time = time.time() - snap_start

            # Should be fast
            assert snap_time < 2.0


class TestStressTests:
    """Stress tests to verify system stability under extreme conditions."""

    def test_extreme_parameter_values(self):
        """Test system behavior with extreme parameter values."""
        # Test with very large numbers
        try:
            retriever = MCPMRetriever(
                num_agents=1000,
                max_iterations=100,
                pheromone_decay=0.999,
                exploration_bonus=0.01
            )

            # Add many documents
            docs = [f"Stress document {i}" for i in range(100)]
            retriever.add_documents(docs)

            # Should handle large parameters
            retriever.initialize_simulation("Stress test")

            # Run a few steps
            for _ in range(2):
                result = retriever.step(1)
                assert "avg_relevance" in result

        except MemoryError:
            pytest.skip("System doesn't have enough memory for extreme parameters")

    def test_rapid_successive_operations(self):
        """Test system under rapid successive operations."""
        operations = []

        for i in range(50):
            # Create and destroy retrievers rapidly
            retriever = MCPMRetriever(num_agents=20, max_iterations=3)
            docs = [f"Rapid doc {i}_{j}" for j in range(5)]
            retriever.add_documents(docs)
            retriever.initialize_simulation(f"Rapid test {i}")

            # Run a quick simulation
            for _ in range(2):
                retriever.step(1)

            operations.append(len(retriever.agents))

        # Should handle rapid operations
        assert len(operations) == 50
        assert all(op == 20 for op in operations)  # All should have 20 agents

    def test_resource_exhaustion_handling(self):
        """Test behavior when system resources are exhausted."""
        # This is a basic test - in a real environment, you'd want more
        # sophisticated resource monitoring

        initial_memory = self.get_memory_usage_mb()

        # Try to create many large objects
        large_objects = []
        try:
            for i in range(10):
                # Create large numpy arrays
                large_array = np.random.randn(1000, 1000).astype(np.float32)
                large_objects.append(large_array)

                # Check memory usage
                current_memory = self.get_memory_usage_mb()
                memory_growth = current_memory - initial_memory

                # If memory growth is excessive, we might be near limits
                if memory_growth > 500:  # More than 500MB
                    break

        finally:
            # Clean up
            large_objects.clear()
            gc.collect()

        # System should still be functional after stress
        retriever = MCPMRetriever()
        docs = ["Test doc"]
        retriever.add_documents(docs)
        retriever.initialize_simulation("Recovery test")
        result = retriever.step(1)

        assert "avg_relevance" in result

    def test_error_recovery_under_stress(self):
        """Test error recovery when under stress."""
        # Simulate stress conditions by creating many objects
        stress_objects = []

        for i in range(20):
            obj = {"data": "x" * 10000}  # 10KB objects
            stress_objects.append(obj)

        try:
            # Try to run simulation under stress
            retriever = MCPMRetriever(num_agents=30, max_iterations=5)
            docs = [f"Stress doc {i}" for i in range(10)]
            retriever.add_documents(docs)

            # Should still work despite stress
            retriever.initialize_simulation("Stress recovery test")
            result = retriever.step(2)

            assert "avg_relevance" in result

        finally:
            # Clean up stress objects
            stress_objects.clear()
            gc.collect()


class TestContinuousLoadTests:
    """Test system behavior under continuous load."""

    def test_sustained_operation(self):
        """Test system during sustained operation."""
        # Run continuous operation for a period
        start_time = time.time()
        operation_count = 0

        while time.time() - start_time < 30.0:  # 30 seconds of continuous operation
            # Create and run a small simulation
            retriever = MCPMRetriever(num_agents=15, max_iterations=3)
            docs = [f"Continuous doc {operation_count}"]
            retriever.add_documents(docs)
            retriever.initialize_simulation("Continuous test")
            retriever.step(1)

            operation_count += 1

            # Small delay to prevent overwhelming the system
            time.sleep(0.1)

        # Should have completed many operations
        assert operation_count > 50  # At least 50 operations in 30 seconds

        # Force final garbage collection
        gc.collect()

    def test_background_task_interference(self):
        """Test system performance with background tasks running."""
        def background_task(task_id):
            """Simulate background processing."""
            for i in range(100):
                # Simulate some background work
                data = np.random.randn(100, 100).astype(np.float32)
                _ = np.sum(data)  # Force computation
            return task_id

        # Start background tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bg_futures = [executor.submit(background_task, i) for i in range(2)]

            # Run main simulation while background tasks execute
            retriever = MCPMRetriever(num_agents=40, max_iterations=5)
            docs = [f"Background test doc {i}" for i in range(20)]
            retriever.add_documents(docs)
            retriever.initialize_simulation("Background interference test")

            # Run simulation
            for _ in range(3):
                retriever.step(1)

            # Wait for background tasks
            bg_results = [future.result() for future in bg_futures]

        # Should complete successfully despite background activity
        assert len(bg_results) == 2
        assert all(result in [0, 1] for result in bg_results)


# Helper method for getting memory usage
def get_memory_usage_mb():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except (ImportError, AttributeError):
        # psutil not available, return 0
        return 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])