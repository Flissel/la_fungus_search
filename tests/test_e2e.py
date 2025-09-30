"""
End-to-end tests for the complete EmbeddingGemma application.

Tests the entire application stack from HTTP endpoints to core functionality,
ensuring all components work together correctly in realistic scenarios.
"""

import os
import json
import time
import asyncio
import tempfile
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, List
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the main application components
from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.realtime.server import app, SnapshotStreamer, streamer
from embeddinggemma.ui.corpus import collect_codebase_chunks, list_code_files

# FastAPI test client
from fastapi.testclient import TestClient


class TestEndToEndScenarios:
    """Comprehensive end-to-end test scenarios."""

    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary test project with realistic code."""
        project_root = tmp_path / "test_project"
        project_root.mkdir()

        # Create realistic Python project structure
        (project_root / "__init__.py").write_text("")
        (project_root / "main.py").write_text("""
import argparse
import logging
from typing import List, Dict

def main() -> None:
    \"\"\"Main entry point for the application.\"\"\"
    parser = argparse.ArgumentParser(description="Test application")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    print("Application started successfully")

class DataProcessor:
    \"\"\"Handles data processing operations.\"\"\"

    def __init__(self, data: List[Dict]):
        self.data = data
        self.processed = False

    def process(self) -> List[Dict]:
        \"\"\"Process the input data.\"\"\"
        if not self.processed:
            self.data = [self._clean_item(item) for item in self.data]
            self.processed = True
        return self.data

    def _clean_item(self, item: Dict) -> Dict:
        \"\"\"Clean a single data item.\"\"\"
        return {k: v.strip() if isinstance(v, str) else v for k, v in item.items()}

if __name__ == "__main__":
    main()
""")

        (project_root / "utils.py").write_text("""
from typing import Optional, Union
import json

def load_config(config_path: str) -> dict:
    \"\"\"Load configuration from JSON file.\"\"\"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in {config_path}")

def validate_config(config: dict) -> bool:
    \"\"\"Validate configuration dictionary.\"\"\"
    required_keys = ['database_url', 'api_key']
    return all(key in config for key in required_keys)

class ConfigurationError(Exception):
    \"\"\"Raised when configuration is invalid.\"\"\"
    pass

def setup_logging(level: str = "INFO") -> None:
    \"\"\"Setup logging configuration.\"\"\"
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
""")

        (project_root / "models.py").write_text("""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class User:
    \"\"\"User data model.\"\"\"
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

@dataclass
class Post:
    \"\"\"Blog post data model.\"\"\"
    id: int
    title: str
    content: str
    author_id: int
    published_at: Optional[datetime] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class DatabaseConnection:
    \"\"\"Database connection manager.\"\"\"
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False

    def connect(self) -> bool:
        \"\"\"Establish database connection.\"\"\"
        # Simulate connection logic
        self.connected = True
        return True

    def disconnect(self) -> None:
        \"\"\"Close database connection.\"\"\"
        self.connected = False

    def execute_query(self, query: str) -> List[dict]:
        \"\"\"Execute SQL query.\"\"\"
        if not self.connected:
            raise RuntimeError("Not connected to database")
        return []  # Mock result
""")

        # Create subdirectory with more files
        api_dir = project_root / "api"
        api_dir.mkdir()
        (api_dir / "__init__.py").write_text("")
        (api_dir / "endpoints.py").write_text("""
from fastapi import FastAPI, HTTPException
from typing import List, Dict

app = FastAPI()

@app.get("/health")
async def health_check():
    \"\"\"Health check endpoint.\"\"\"
    return {"status": "healthy"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    \"\"\"Get user by ID.\"\"\"
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    return {"id": user_id, "name": f"User {user_id}"}

@app.post("/users")
async def create_user(user_data: Dict):
    \"\"\"Create new user.\"\"\"
    if "name" not in user_data:
        raise HTTPException(status_code=422, detail="Name is required")
    return {"id": 123, "name": user_data["name"]}
""")

        return project_root

    def test_complete_codebase_analysis_workflow(self, client, temp_project):
        """Test complete workflow from codebase analysis to query results."""
        # Step 1: Configure for codebase analysis
        config_data = {
            "query": "What does this application do?",
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [50, 100, 200],
            "max_files": 10,
            "num_agents": 20,
            "max_iterations": 5,
            "top_k": 3
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # Step 2: Start the simulation
        response = client.post("/start")
        assert response.status_code == 200

        # Step 3: Wait for simulation to initialize and run
        time.sleep(2.0)

        # Step 4: Check status
        response = client.get("/status")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["running"] is True
        assert "docs" in status_data
        assert "agents" in status_data

        # Step 5: Perform search
        search_data = {
            "query": "What are the main components?",
            "top_k": 3
        }
        response = client.post("/search", json=search_data)
        assert response.status_code == 200
        search_results = response.json()
        assert "results" in search_results
        assert "status" in search_results

        # Step 6: Get visualization data
        # Note: This would require WebSocket connection in real scenario

        # Step 7: Stop simulation
        response = client.post("/stop")
        assert response.status_code == 200

        # Verify the complete workflow worked
        assert status_data["docs"] > 0  # Should have found documents
        assert status_data["agents"] == 20  # Should have created agents
        assert len(search_results["results"]) <= 3  # Limited by top_k

    def test_real_time_visualization_workflow(self, client, temp_project):
        """Test real-time visualization and WebSocket communication."""
        # Configure for real-time visualization
        config_data = {
            "query": "Explain the architecture",
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [100],
            "max_files": 5,
            "num_agents": 15,
            "max_iterations": 3,
            "viz_dims": 2,
            "redraw_every": 1
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # Start simulation
        response = client.post("/start")
        assert response.status_code == 200

        # Wait for initialization
        time.sleep(1.0)

        # Check that simulation is running
        response = client.get("/status")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["running"] is True

        # Test WebSocket endpoint (simulated)
        # In a real scenario, we'd test actual WebSocket connections
        # For this test, we'll verify the endpoint exists and handles connections

        # Stop simulation
        response = client.post("/stop")
        assert response.status_code == 200

    def test_settings_persistence_across_sessions(self, client, temp_project):
        """Test that settings persist across simulation sessions."""
        # Set initial configuration
        initial_config = {
            "query": "First session query",
            "num_agents": 25,
            "max_iterations": 10,
            "exploration_bonus": 0.15,
            "pheromone_decay": 0.92
        }

        response = client.post("/settings", json=initial_config)
        assert response.status_code == 200

        # Verify settings were saved
        response = client.get("/settings")
        settings_data = response.json()
        saved_settings = settings_data["settings"]
        assert saved_settings["query"] == "First session query"
        assert saved_settings["num_agents"] == 25

        # Start first simulation
        start_data = {
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [50]
        }
        response = client.post("/start", json=start_data)
        assert response.status_code == 200

        time.sleep(1.0)

        # Stop first simulation
        response = client.post("/stop")
        assert response.status_code == 200

        # Start second simulation (settings should persist)
        response = client.post("/start", json=start_data)
        assert response.status_code == 200

        # Verify settings are still applied
        response = client.get("/settings")
        settings_data = response.json()
        current_settings = settings_data["settings"]
        assert current_settings["query"] == "First session query"
        assert current_settings["num_agents"] == 25

        # Stop second simulation
        response = client.post("/stop")
        assert response.status_code == 200

    def test_agent_management_endpoints(self, client, temp_project):
        """Test agent addition and resizing endpoints."""
        # Configure and start simulation
        config_data = {
            "query": "Test agent management",
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [50],
            "num_agents": 10
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        response = client.post("/start")
        assert response.status_code == 200

        time.sleep(1.0)

        # Test agent addition
        add_response = client.post("/agents/add", json={"n": 5})
        assert add_response.status_code == 200
        add_data = add_response.json()
        assert add_data["status"] == "ok"
        assert add_data["added"] == 5

        # Check status shows more agents
        response = client.get("/status")
        status_data = response.json()
        assert status_data["agents"] == 15  # 10 + 5

        # Test agent resizing
        resize_response = client.post("/agents/resize", json={"count": 8})
        assert resize_response.status_code == 200
        resize_data = resize_response.json()
        assert resize_data["status"] == "ok"
        assert resize_data["agents"] == 8

        # Test pause/resume functionality
        pause_response = client.post("/pause")
        assert pause_response.status_code == 200

        resume_response = client.post("/resume")
        assert resume_response.status_code == 200

        # Stop simulation
        response = client.post("/stop")
        assert response.status_code == 200

    def test_error_recovery_and_resilience(self, client):
        """Test error recovery and system resilience."""
        # Test starting with invalid configuration
        invalid_config = {
            "query": "",  # Empty query
            "num_agents": -1,  # Invalid number
            "max_iterations": 0  # Invalid iterations
        }

        response = client.post("/start", json=invalid_config)
        # Should handle gracefully or return appropriate error
        # The exact behavior depends on implementation

        # Test search without running simulation
        response = client.post("/search", json={"query": "test"})
        # Should handle gracefully

        # Test accessing non-existent document
        response = client.get("/doc/99999")
        assert response.status_code == 404

        # Test invalid JSON in requests
        response = client.post("/start", data="invalid json", headers={"content-type": "application/json"})
        # Should handle malformed requests gracefully

        # Test reset functionality after errors
        response = client.post("/reset")
        assert response.status_code == 200

    def test_concurrent_user_sessions(self, client, temp_project):
        """Test handling multiple concurrent user sessions."""
        def simulate_user_session(user_id):
            """Simulate a user session."""
            # Configure for this user
            config_data = {
                "query": f"User {user_id} query",
                "use_repo": False,
                "root_folder": str(temp_project),
                "windows": [50],
                "num_agents": 10,
                "max_iterations": 3
            }

            # Start simulation
            response = client.post("/start", json=config_data)
            assert response.status_code == 200

            # Wait a bit
            time.sleep(0.5)

            # Perform search
            search_response = client.post("/search", json={"query": f"User {user_id} search", "top_k": 2})
            assert search_response.status_code == 200

            # Stop simulation
            stop_response = client.post("/stop")
            assert stop_response.status_code == 200

            return True

        # Run multiple concurrent sessions
        import concurrent.futures

        num_users = 3
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(simulate_user_session, i) for i in range(num_users)]
            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # All sessions should complete successfully
        assert all(results)
        assert total_time < 10.0  # Should complete in reasonable time

    def test_large_scale_processing(self, client, tmp_path):
        """Test processing with large amounts of data."""
        # Create large test project
        large_project = tmp_path / "large_project"
        large_project.mkdir()

        # Create many files
        for i in range(20):
            (large_project / f"module_{i}.py").write_text(f"""
def function_{i}():
    \"\"\"Function {i} in module {i}.\"\"\"
    return f"result_{i}"

class Class{i}:
    \"\"\"Class {i} in module {i}.\"\"\"

    def __init__(self):
        self.value = {i}

    def get_value(self):
        return self.value
""")

        # Configure for large-scale processing
        config_data = {
            "query": "What is the overall architecture?",
            "use_repo": False,
            "root_folder": str(large_project),
            "windows": [100, 200],
            "max_files": 50,
            "num_agents": 50,
            "max_iterations": 5,
            "top_k": 5
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # Start processing
        start_time = time.time()
        response = client.post("/start")
        assert response.status_code == 200

        # Wait for processing
        time.sleep(3.0)

        # Check results
        response = client.get("/status")
        status_data = response.json()
        assert status_data["running"] is True

        # Should have processed many documents
        assert status_data["docs"] > 10

        # Perform search
        search_response = client.post("/search", json={"query": "architecture", "top_k": 3})
        assert search_response.status_code == 200

        total_time = time.time() - start_time

        # Should complete in reasonable time
        assert total_time < 15.0

        # Stop simulation
        response = client.post("/stop")
        assert response.status_code == 200

    def test_cross_platform_compatibility(self, client, temp_project):
        """Test compatibility across different environments."""
        # Test with various path formats
        config_data = {
            "query": "Test cross-platform paths",
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [50],
            "num_agents": 10,
            "max_iterations": 3
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # Start simulation
        response = client.post("/start")
        assert response.status_code == 200

        time.sleep(1.0)

        # Test various API endpoints
        endpoints_to_test = [
            ("/status", "GET"),
            ("/settings", "GET"),
            ("/search", "POST"),
        ]

        for endpoint, method in endpoints_to_test:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json={"query": "test", "top_k": 1})

            # Should not crash
            assert response.status_code in [200, 400, 404]  # Acceptable status codes

        # Stop simulation
        response = client.post("/stop")
        assert response.status_code == 200

    def test_memory_and_resource_management(self, client, temp_project):
        """Test memory usage and resource management."""
        import gc

        # Get initial memory state
        initial_objects = len(gc.get_objects())

        # Run multiple simulation cycles
        for cycle in range(3):
            config_data = {
                "query": f"Memory test cycle {cycle}",
                "use_repo": False,
                "root_folder": str(temp_project),
                "windows": [50],
                "num_agents": 20,
                "max_iterations": 3
            }

            response = client.post("/config", json=config_data)
            assert response.status_code == 200

            response = client.post("/start")
            assert response.status_code == 200

            time.sleep(1.0)

            response = client.post("/stop")
            assert response.status_code == 200

            # Force garbage collection
            gc.collect()

        # Check memory usage doesn't grow excessively
        final_objects = len(gc.get_objects())
        growth_ratio = final_objects / max(initial_objects, 1)

        # Memory growth should be reasonable
        assert growth_ratio < 2.0  # Less than 100% growth

    def test_api_rate_limiting_and_overload(self, client, temp_project):
        """Test behavior under high load and rapid requests."""
        config_data = {
            "query": "Load test",
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [50],
            "num_agents": 10,
            "max_iterations": 2
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # Send many rapid requests
        responses = []
        for i in range(20):
            response = client.post("/search", json={"query": f"Query {i}", "top_k": 2})
            responses.append(response.status_code)

        # Most requests should succeed or return appropriate errors
        successful_requests = sum(1 for status in responses if status == 200)
        assert successful_requests >= 10  # At least half should succeed

        # Start simulation for more realistic testing
        response = client.post("/start")
        assert response.status_code == 200

        time.sleep(1.0)

        # Send more requests while simulation is running
        for i in range(10):
            response = client.post("/search", json={"query": f"Running query {i}", "top_k": 2})
            # Should handle concurrent requests

        # Stop simulation
        response = client.post("/stop")
        assert response.status_code == 200


class TestRealWorldScenarios:
    """Test scenarios based on real-world usage patterns."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_code_review_assistance(self, client, temp_project):
        """Test scenario for code review assistance."""
        # Configure for code review
        config_data = {
            "query": "What are the potential issues in this codebase?",
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [100, 150],
            "max_files": 20,
            "num_agents": 30,
            "max_iterations": 8,
            "top_k": 5
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # Start analysis
        response = client.post("/start")
        assert response.status_code == 200

        time.sleep(2.0)

        # Get analysis results
        search_response = client.post("/search", json={
            "query": "potential bugs or issues",
            "top_k": 3
        })
        assert search_response.status_code == 200

        results = search_response.json()
        assert "results" in results

        # Stop analysis
        response = client.post("/stop")
        assert response.status_code == 200

    def test_documentation_generation(self, client, temp_project):
        """Test scenario for automatic documentation generation."""
        # Configure for documentation analysis
        config_data = {
            "query": "Generate documentation for this project",
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [200],
            "max_files": 15,
            "num_agents": 25,
            "max_iterations": 6,
            "top_k": 4
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # Start documentation analysis
        response = client.post("/start")
        assert response.status_code == 200

        time.sleep(2.0)

        # Get documentation insights
        search_response = client.post("/search", json={
            "query": "main functions and classes",
            "top_k": 4
        })
        assert search_response.status_code == 200

        # Test answer generation (if Ollama is available)
        try:
            answer_response = client.post("/answer", json={
                "query": "Summarize the main components",
                "top_k": 3
            })
            # Should handle gracefully even if Ollama is not available
        except Exception:
            pass  # Expected if Ollama is not configured

        # Stop analysis
        response = client.post("/stop")
        assert response.status_code == 200

    def test_performance_optimization_analysis(self, client, temp_project):
        """Test scenario for performance optimization suggestions."""
        # Configure for performance analysis
        config_data = {
            "query": "How can this code be optimized for performance?",
            "use_repo": False,
            "root_folder": str(temp_project),
            "windows": [80, 120],
            "max_files": 25,
            "num_agents": 35,
            "max_iterations": 7,
            "top_k": 4
        }

        response = client.post("/config", json=config_data)
        assert response.status_code == 200

        # Start performance analysis
        response = client.post("/start")
        assert response.status_code == 200

        time.sleep(2.0)

        # Get optimization suggestions
        search_response = client.post("/search", json={
            "query": "performance bottlenecks",
            "top_k": 3
        })
        assert search_response.status_code == 200

        # Stop analysis
        response = client.post("/stop")
        assert response.status_code == 200


class TestSystemIntegration:
    """Test system-level integration and compatibility."""

    def test_dependency_compatibility(self):
        """Test that all dependencies are compatible and available."""
        # Test that all required modules can be imported
        try:
            from embeddinggemma.mcmp_rag import MCPMRetriever
            from embeddinggemma.realtime.server import app
            from embeddinggemma.ui.corpus import collect_codebase_chunks
            from embeddinggemma.rag.generation import generate_with_ollama
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")

        # Test that MCPMRetriever can be instantiated
        retriever = MCPMRetriever()
        assert retriever.num_agents > 0

        # Test that FastAPI app can be created
        assert app is not None

    def test_configuration_consistency(self):
        """Test that configuration is consistent across all components."""
        from embeddinggemma.realtime.server import settings_dict

        settings = settings_dict()

        # Test that all required settings are present
        required_settings = [
            "query", "top_k", "num_agents", "max_iterations",
            "exploration_bonus", "pheromone_decay"
        ]

        for setting in required_settings:
            assert setting in settings
            assert settings[setting] is not None

        # Test that settings have reasonable value ranges
        assert settings["top_k"] > 0
        assert settings["num_agents"] > 0
        assert settings["max_iterations"] > 0
        assert 0 < settings["exploration_bonus"] <= 1.0
        assert 0 < settings["pheromone_decay"] < 1.0

    def test_error_propagation_and_handling(self):
        """Test that errors propagate correctly through the system."""
        # Test with invalid configuration
        from embeddinggemma.realtime.server import streamer

        # Save original state
        original_state = streamer.__dict__.copy()

        try:
            # Test invalid settings
            invalid_settings = {
                "num_agents": -1,
                "max_iterations": 0,
                "exploration_bonus": 2.0  # Out of range
            }

            # Should handle invalid settings gracefully
            from embeddinggemma.realtime.server import apply_settings
            apply_settings(invalid_settings)

        finally:
            # Restore original state
            streamer.__dict__.update(original_state)

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        from embeddinggemma.realtime.server import streamer
        import gc

        # Get initial state
        initial_running = streamer.running
        initial_retriever = streamer.retr

        try:
            # Start and stop simulation to test cleanup
            # This would require proper setup in a real test environment

            # Force garbage collection
            gc.collect()

            # Check that system state is clean
            # (This is a basic check - real cleanup testing would be more thorough)

        finally:
            # Restore any modified state
            streamer.running = initial_running
            streamer.retr = initial_retriever


if __name__ == "__main__":
    pytest.main([__file__, "-v"])