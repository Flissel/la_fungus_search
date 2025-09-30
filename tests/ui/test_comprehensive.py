"""
Comprehensive tests for the UI module components.

Tests all UI components including corpus management, state management,
query handling, and component integration.
"""

import tempfile
import shutil
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import pytest
import numpy as np

# Import UI components
from embeddinggemma.ui import corpus
from embeddinggemma.ui import state
from embeddinggemma.ui import queries
from embeddinggemma.ui import components
from embeddinggemma.ui import reports
from embeddinggemma.ui import mcmp_runner


class TestUICorpusComprehensive:
    """Comprehensive tests for the corpus management module."""

    def test_codebase_file_listing(self, tmp_path):
        """Test listing of code files in a codebase."""
        # Create test directory structure
        test_root = tmp_path / "test_codebase"
        test_root.mkdir()

        # Create various file types
        (test_root / "module1.py").write_text("print('hello')")
        (test_root / "module2.py").write_text("import os")
        (test_root / "data.json").write_text('{"key": "value"}')
        (test_root / "readme.md").write_text("# Test")

        # Create subdirectory
        subdir = test_root / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("def func(): pass")

        # Create excluded directory
        excluded_dir = test_root / ".venv"
        excluded_dir.mkdir()
        (excluded_dir / "venv_file.py").write_text("print('venv')")

        # Test file listing
        files = corpus.list_code_files(
            str(test_root),
            max_files=100,
            exclude_dirs=[".venv"]
        )

        # Should find Python files but not excluded ones
        assert len(files) >= 3  # At least the 3 Python files
        python_files = [f for f in files if f.endswith('.py')]
        assert len(python_files) == 3  # module1.py, module2.py, nested.py

        # Should not include files from excluded directories
        venv_files = [f for f in files if '.venv' in f]
        assert len(venv_files) == 0

    def test_chunk_collection_with_different_window_sizes(self, tmp_path):
        """Test chunk collection with various window sizes."""
        # Create test file with known content
        test_file = tmp_path / "test.py"
        lines = []
        for i in range(1, 21):  # 20 lines
            lines.append(f"# Line {i}")
            lines.append(f"def function_{i}():")
            lines.append("    pass")
            lines.append("")

        test_file.write_text("\n".join(lines))

        # Test different window sizes
        windows = [5, 10, 15]

        for window in windows:
            chunks = corpus.collect_codebase_chunks(
                str(tmp_path),
                [window],
                max_files=1,
                exclude_dirs=[]
            )

            # Should produce chunks
            assert len(chunks) > 0

            # Each chunk should be a string
            assert all(isinstance(chunk, str) for chunk in chunks)

            # Total content should be reasonable
            total_chars = sum(len(chunk) for chunk in chunks)
            assert total_chars > 0

    def test_corpus_collection_with_large_files(self, tmp_path):
        """Test corpus collection with large files."""
        # Create a large test file
        large_file = tmp_path / "large.py"
        large_content = []

        # Create 1000 lines of content
        for i in range(1000):
            large_content.append(f"# Comment {i}")
            large_content.append(f"def func_{i}():")
            large_content.append("    return i")
            large_content.append("")

        large_file.write_text("\n".join(large_content))

        # Test collection with reasonable limits
        chunks = corpus.collect_codebase_chunks(
            str(tmp_path),
            [50, 100],  # Window sizes
            max_files=1,
            exclude_dirs=[]
        )

        # Should handle large files gracefully
        assert len(chunks) > 0

        # Should not create excessively large chunks
        max_chunk_size = max(len(chunk) for chunk in chunks)
        assert max_chunk_size < 10000  # Reasonable upper bound

    def test_corpus_collection_error_handling(self, tmp_path):
        """Test error handling during corpus collection."""
        # Test with non-existent directory
        try:
            chunks = corpus.collect_codebase_chunks(
                "/non/existent/path",
                [50],
                max_files=10,
                exclude_dirs=[]
            )
            # Should handle gracefully or raise appropriate error
        except Exception:
            pass  # Expected behavior

        # Test with permission denied (simulate by creating unreadable file)
        restricted_file = tmp_path / "restricted.py"
        restricted_file.write_text("test content")

        # On Unix systems, we could test permissions, but for cross-platform
        # we'll just verify the function doesn't crash on normal files
        chunks = corpus.collect_codebase_chunks(
            str(tmp_path),
            [50],
            max_files=10,
            exclude_dirs=[]
        )
        assert isinstance(chunks, list)

    def test_corpus_metadata_extraction(self, tmp_path):
        """Test metadata extraction from collected chunks."""
        # Create test file with identifiable content
        test_file = tmp_path / "metadata_test.py"
        test_file.write_text("""
# This is a test module
import os
import sys

class TestClass:
    \"\"\"A test class for metadata extraction.\"\"\"

    def __init__(self):
        self.value = 42

    def get_value(self):
        \"\"\"Get the value.\"\"\"
        return self.value

def test_function():
    \"\"\"A test function.\"\"\"
    obj = TestClass()
    return obj.get_value()
""")

        chunks = corpus.collect_codebase_chunks(
            str(tmp_path),
            [100],
            max_files=1,
            exclude_dirs=[]
        )

        # Should extract meaningful chunks with metadata
        assert len(chunks) > 0

        # Check that chunks contain expected content
        chunk_text = " ".join(chunks).lower()
        assert "testclass" in chunk_text
        assert "test_function" in chunk_text
        assert "metadata extraction" in chunk_text


class TestUIStateComprehensive:
    """Comprehensive tests for the state management module."""

    def test_state_initialization_and_defaults(self):
        """Test state initialization with default values."""
        # Test default state creation
        default_state = state.UIState()

        # Should have reasonable defaults
        assert hasattr(default_state, 'query')
        assert hasattr(default_state, 'top_k')
        assert hasattr(default_state, 'num_agents')
        assert default_state.top_k > 0
        assert default_state.num_agents > 0

    def test_state_updates_and_validation(self):
        """Test state updates and validation."""
        ui_state = state.UIState()

        # Test valid updates
        ui_state.query = "New test query"
        ui_state.top_k = 10
        ui_state.num_agents = 50

        assert ui_state.query == "New test query"
        assert ui_state.top_k == 10
        assert ui_state.num_agents == 50

        # Test serialization
        state_dict = ui_state.to_dict()
        assert isinstance(state_dict, dict)
        assert state_dict["query"] == "New test query"
        assert state_dict["top_k"] == 10

        # Test deserialization
        new_state = state.UIState.from_dict(state_dict)
        assert new_state.query == ui_state.query
        assert new_state.top_k == ui_state.top_k

    def test_state_validation_constraints(self):
        """Test state validation constraints."""
        ui_state = state.UIState()

        # Test boundary values
        ui_state.top_k = 1  # Minimum
        ui_state.num_agents = 1  # Minimum

        # Should handle minimum values
        assert ui_state.top_k == 1
        assert ui_state.num_agents == 1

        # Test with very large values (should be handled gracefully)
        ui_state.top_k = 10000
        ui_state.num_agents = 10000

        # Should either accept or clamp to reasonable values
        assert ui_state.top_k > 0
        assert ui_state.num_agents > 0


class TestUIQueriesComprehensive:
    """Comprehensive tests for the queries module."""

    def test_query_processing_and_validation(self):
        """Test query processing and validation."""
        # Test basic query processing
        query = "What is machine learning?"
        processed = queries.process_query(query)

        # Should return processed query (implementation dependent)
        assert isinstance(processed, str)

        # Test query cleaning
        messy_query = "  What is   machine learning?  \n\n  "
        cleaned = queries.clean_query(messy_query)
        assert cleaned == "What is machine learning?"

        # Test empty query handling
        empty_result = queries.process_query("")
        # Should handle gracefully

    def test_query_history_management(self):
        """Test query history tracking and management."""
        # Test history initialization
        history = queries.QueryHistory()
        assert len(history.queries) == 0

        # Test adding queries
        history.add_query("First query")
        history.add_query("Second query")

        assert len(history.queries) == 2
        assert history.queries[0] == "First query"
        assert history.queries[1] == "Second query"

        # Test history limits
        for i in range(10):
            history.add_query(f"Query {i}")

        # Should maintain reasonable size (implementation dependent)
        assert len(history.queries) <= 20  # Reasonable upper bound

        # Test history serialization
        history_dict = history.to_dict()
        assert isinstance(history_dict, dict)
        assert "queries" in history_dict

    def test_query_similarity_detection(self):
        """Test query similarity detection for deduplication."""
        # Test similar queries
        similar_queries = [
            "What is Python?",
            "What is python?",
            "what is Python programming?",
            "Tell me about Python language"
        ]

        # Should detect similarities (implementation dependent)
        similarities = []
        for i, q1 in enumerate(similar_queries):
            for j, q2 in enumerate(similar_queries):
                if i != j:
                    sim = queries.calculate_query_similarity(q1, q2)
                    similarities.append(sim)

        # Should find some similarities
        assert len(similarities) > 0
        assert any(sim > 0.5 for sim in similarities)  # At least some similar


class TestUIComponentsComprehensive:
    """Comprehensive tests for the UI components module."""

    def test_component_initialization(self):
        """Test UI component initialization."""
        # Test basic component creation
        component = components.UIComponent()

        # Should have basic attributes
        assert hasattr(component, 'id')
        assert hasattr(component, 'name')
        assert hasattr(component, 'state')

        # Test component configuration
        component.id = "test_component"
        component.name = "Test Component"

        assert component.id == "test_component"
        assert component.name == "Test Component"

    def test_component_state_management(self):
        """Test component state management."""
        component = components.UIComponent()
        component.id = "test"

        # Test state updates
        component.state = {"key": "value"}
        assert component.state["key"] == "value"

        # Test state serialization
        state_dict = component.get_state()
        assert isinstance(state_dict, dict)

    def test_component_registry(self):
        """Test component registration and lookup."""
        # Clear any existing registry
        components._component_registry.clear()

        # Register test component
        test_component = components.UIComponent()
        test_component.id = "test_component"
        components.register_component(test_component)

        # Should be able to retrieve
        retrieved = components.get_component("test_component")
        assert retrieved is test_component

        # Test non-existent component
        missing = components.get_component("non_existent")
        assert missing is None

        # Test registry listing
        registry_list = components.list_components()
        assert len(registry_list) >= 1
        assert "test_component" in registry_list


class TestUIReportsComprehensive:
    """Comprehensive tests for the reports module."""

    def test_report_generation_and_formatting(self, tmp_path):
        """Test report generation and formatting."""
        # Create test data
        test_data = {
            "query": "What is machine learning?",
            "results": [
                {"content": "Machine learning is AI", "score": 0.9},
                {"content": "ML is subset of AI", "score": 0.8},
                {"content": "AI includes ML", "score": 0.7}
            ],
            "metadata": {"total_docs": 100, "processing_time": 1.2}
        }

        # Test report creation
        report = reports.generate_report(test_data, format_type="json")

        # Should generate valid report
        assert isinstance(report, str)
        assert len(report) > 0

        # Test JSON format specifically
        json_report = reports.generate_report(test_data, format_type="json")
        try:
            parsed = json.loads(json_report)
            assert "query" in parsed
            assert "results" in parsed
        except json.JSONDecodeError:
            pytest.fail("Report should be valid JSON")

    def test_report_export_functionality(self, tmp_path):
        """Test report export to files."""
        # Create test report data
        report_data = {
            "query": "Test query",
            "results": [{"content": "Test result"}],
            "timestamp": "2023-01-01T00:00:00"
        }

        # Test export to file
        export_path = tmp_path / "test_report.json"
        success = reports.export_report(report_data, str(export_path))

        # Should export successfully
        assert success
        assert export_path.exists()

        # Verify file contents
        with open(export_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["query"] == "Test query"
        assert len(loaded_data["results"]) == 1

    def test_report_aggregation(self):
        """Test report aggregation from multiple sources."""
        # Create multiple report data sets
        reports_data = [
            {
                "query": "Query 1",
                "results": [{"content": "Result 1"}],
                "score": 0.9
            },
            {
                "query": "Query 2",
                "results": [{"content": "Result 2"}],
                "score": 0.8
            },
            {
                "query": "Query 3",
                "results": [{"content": "Result 3"}],
                "score": 0.7
            }
        ]

        # Test aggregation
        aggregated = reports.aggregate_reports(reports_data)

        # Should combine results
        assert isinstance(aggregated, dict)
        assert "total_queries" in aggregated
        assert "average_score" in aggregated
        assert aggregated["total_queries"] == 3

        # Average should be reasonable
        avg_score = aggregated["average_score"]
        assert 0.6 <= avg_score <= 1.0

    def test_report_filtering_and_sorting(self):
        """Test report filtering and sorting capabilities."""
        # Create test reports with different scores and dates
        reports_data = [
            {"id": 1, "score": 0.9, "timestamp": "2023-01-01"},
            {"id": 2, "score": 0.7, "timestamp": "2023-01-02"},
            {"id": 3, "score": 0.8, "timestamp": "2023-01-03"},
            {"id": 4, "score": 0.6, "timestamp": "2023-01-04"},
        ]

        # Test filtering by score threshold
        filtered = reports.filter_reports(reports_data, min_score=0.75)
        assert len(filtered) == 2  # Should keep only high-scoring reports

        # Test sorting by score
        sorted_reports = reports.sort_reports(reports_data, sort_by="score", reverse=True)
        scores = [r["score"] for r in sorted_reports]
        assert scores == sorted(scores, reverse=True)


class TestUIMCMPRunnerComprehensive:
    """Comprehensive tests for the MCMP runner module."""

    def test_runner_initialization(self):
        """Test MCMP runner initialization."""
        # Test runner creation
        runner = mcmp_runner.MCMPRunner()

        # Should have default configuration
        assert hasattr(runner, 'config')
        assert hasattr(runner, 'state')
        assert isinstance(runner.config, dict)
        assert isinstance(runner.state, dict)

    def test_runner_configuration_management(self):
        """Test runner configuration management."""
        runner = mcmp_runner.MCMPRunner()

        # Test configuration updates
        new_config = {
            "num_agents": 50,
            "max_iterations": 100,
            "query": "Test configuration"
        }

        runner.update_config(new_config)
        assert runner.config["num_agents"] == 50
        assert runner.config["max_iterations"] == 100
        assert runner.config["query"] == "Test configuration"

        # Test configuration validation
        invalid_config = {"num_agents": -1}  # Invalid value
        # Should handle invalid config gracefully
        runner.update_config(invalid_config)

    def test_runner_state_tracking(self):
        """Test runner state tracking during execution."""
        runner = mcmp_runner.MCMPRunner()

        # Initialize state
        initial_state = {
            "status": "idle",
            "progress": 0,
            "current_step": 0
        }
        runner.update_state(initial_state)

        assert runner.state["status"] == "idle"
        assert runner.state["progress"] == 0

        # Update during execution
        runner.update_state({"status": "running", "progress": 50})
        assert runner.state["status"] == "running"
        assert runner.state["progress"] == 50

        # Test state serialization
        state_dict = runner.get_state()
        assert isinstance(state_dict, dict)
        assert state_dict["status"] == "running"

    def test_runner_error_handling(self):
        """Test error handling in runner operations."""
        runner = mcmp_runner.MCMPRunner()

        # Test with invalid configuration
        try:
            # This might fail depending on implementation
            runner.update_config({"invalid_key": "value"})
            # If it doesn't fail, that's also fine
        except Exception:
            pass  # Expected in some implementations

        # Test state access with missing keys
        try:
            _ = runner.state["non_existent_key"]
        except KeyError:
            pass  # Expected behavior


class TestUIIntegrationScenarios:
    """Integration tests for UI components working together."""

    def test_full_ui_workflow(self, tmp_path):
        """Test complete UI workflow from corpus to reports."""
        # 1. Set up test corpus
        test_root = tmp_path / "test_project"
        test_root.mkdir()

        # Create sample project files
        (test_root / "main.py").write_text("""
def main():
    print("Hello world")

if __name__ == "__main__":
    main()
""")

        (test_root / "utils.py").write_text("""
def helper():
    return "utility function"
""")

        # 2. Collect corpus
        chunks = corpus.collect_codebase_chunks(
            str(test_root),
            [50, 100],
            max_files=10,
            exclude_dirs=[]
        )

        assert len(chunks) > 0

        # 3. Create UI state
        ui_state = state.UIState()
        ui_state.query = "What does this project do?"
        ui_state.top_k = 3

        # 4. Process query
        processed_query = queries.process_query(ui_state.query)
        assert isinstance(processed_query, str)

        # 5. Create report from results
        mock_results = [
            {"content": chunk[:100] + "...", "score": 0.8}
            for chunk in chunks[:3]
        ]

        report_data = {
            "query": processed_query,
            "results": mock_results,
            "metadata": {"total_chunks": len(chunks)}
        }

        # 6. Generate report
        report = reports.generate_report(report_data, format_type="json")

        # 7. Export report
        export_path = tmp_path / "ui_test_report.json"
        export_success = reports.export_report(report_data, str(export_path))

        # Verify complete workflow
        assert len(chunks) > 0
        assert ui_state.query == "What does this project do?"
        assert len(report) > 0
        assert export_success
        assert export_path.exists()

    def test_ui_component_interaction(self):
        """Test interaction between UI components."""
        # Create multiple components
        corpus_component = components.UIComponent()
        corpus_component.id = "corpus"
        corpus_component.state = {"files_count": 0}

        query_component = components.UIComponent()
        query_component.id = "query"
        query_component.state = {"current_query": ""}

        report_component = components.UIComponent()
        report_component.id = "reports"
        report_component.state = {"reports_count": 0}

        # Register components
        components.register_component(corpus_component)
        components.register_component(query_component)
        components.register_component(report_component)

        # Test component communication
        # Update corpus state
        corpus_component.state["files_count"] = 100

        # Query component should be able to access corpus state
        corpus_state = components.get_component("corpus")
        assert corpus_state.state["files_count"] == 100

        # Update query and check report component
        query_component.state["current_query"] = "Test query"
        report_component.state["reports_count"] = 1

        # All components should be accessible
        all_components = components.list_components()
        assert "corpus" in all_components
        assert "query" in all_components
        assert "reports" in all_components

    def test_ui_error_recovery(self, tmp_path):
        """Test UI error recovery and resilience."""
        # Create scenario that might cause errors
        test_root = tmp_path / "problematic_project"
        test_root.mkdir()

        # Create mix of valid and problematic files
        (test_root / "valid.py").write_text("print('valid')")

        # Create a very large file that might cause issues
        large_file = test_root / "huge.py"
        large_content = ["line {}".format(i) for i in range(10000)]
        large_file.write_text("\n".join(large_content))

        # Create file with encoding issues (simulate)
        problematic_file = test_root / "encoding.py"
        try:
            # Try to write with problematic encoding
            with open(problematic_file, 'w', encoding='utf-8') as f:
                f.write("print('test')")
        except Exception:
            # If encoding fails, create simple file
            problematic_file.write_text("print('test')")

        # Test that corpus collection handles errors gracefully
        try:
            chunks = corpus.collect_codebase_chunks(
                str(test_root),
                [100, 200],
                max_files=100,
                exclude_dirs=[]
            )

            # Should either succeed or handle errors gracefully
            assert isinstance(chunks, list)

        except Exception as e:
            # If it fails, should be with a meaningful error
            assert "error" in str(e).lower() or "failed" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])