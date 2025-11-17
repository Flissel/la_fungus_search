"""
Unit tests for LangChain Memory Agent.

Tests the incremental memory creation, tool functions, and integration
with SupermemoryManager.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json


@pytest.fixture
def mock_llm():
    """Create mock LLM for testing."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value="Test response")
    return llm


@pytest.fixture
def mock_memory_manager():
    """Create mock SupermemoryManager for testing."""
    manager = MagicMock()
    manager.enabled = True
    manager.add_memory = AsyncMock(return_value=True)
    manager.update_memory = AsyncMock(return_value=True)
    manager.search_memory = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def langchain_agent(mock_llm, mock_memory_manager):
    """Create LangChainMemoryAgent instance for testing."""
    from src.embeddinggemma.agents.langchain_memory_agent import LangChainMemoryAgent

    return LangChainMemoryAgent(
        llm=mock_llm,
        memory_manager=mock_memory_manager,
        container_tag="test_run",
        model="gpt-4o-mini"
    )


class TestLangChainMemoryAgentInit:
    """Test LangChain Memory Agent initialization."""

    def test_init_with_valid_params(self, mock_llm, mock_memory_manager):
        """Test successful initialization."""
        from src.embeddinggemma.agents.langchain_memory_agent import LangChainMemoryAgent

        agent = LangChainMemoryAgent(
            llm=mock_llm,
            memory_manager=mock_memory_manager,
            container_tag="test_run",
            model="gpt-4o-mini"
        )

        assert agent.llm == mock_llm
        assert agent.memory_manager == mock_memory_manager
        assert agent.container_tag == "test_run"
        assert agent.model == "gpt-4o-mini"
        assert agent.memories_created == 0
        assert agent.memories_updated == 0
        assert agent.iterations_processed == 0

    def test_tools_created(self, langchain_agent):
        """Test that tools are created correctly."""
        assert len(langchain_agent.tools) == 3

        tool_names = [tool.name for tool in langchain_agent.tools]
        assert "add_memory" in tool_names
        assert "update_memory" in tool_names
        assert "search_memory" in tool_names


class TestAddMemoryTool:
    """Test add_memory tool function."""

    def test_add_memory_success(self, langchain_agent, mock_memory_manager):
        """Test successful memory addition."""
        input_data = {
            "content": "Test memory content",
            "type": "entry_point",
            "file_path": "src/test.py",
            "identifier": "main",
            "metadata": {"line": 42}
        }
        input_str = json.dumps(input_data)

        result = langchain_agent._add_memory_tool(input_str)

        assert "success" in result
        assert langchain_agent.memories_created == 1
        mock_memory_manager.add_memory.assert_called_once()

    def test_add_memory_invalid_json(self, langchain_agent):
        """Test handling of invalid JSON input."""
        result = langchain_agent._add_memory_tool("invalid json")

        assert "error" in result
        assert langchain_agent.memories_created == 0

    def test_add_memory_generates_custom_id(self, langchain_agent, mock_memory_manager):
        """Test that custom_id is generated correctly."""
        from src.embeddinggemma.memory.supermemory_client import SupermemoryManager

        input_data = {
            "content": "Test",
            "type": "entry_point",
            "file_path": "src/test.py",
            "identifier": "main",
            "metadata": {}
        }

        langchain_agent._add_memory_tool(json.dumps(input_data))

        # Verify add_memory was called with generated custom_id
        call_args = mock_memory_manager.add_memory.call_args
        custom_id = call_args.kwargs["custom_id"]

        expected_id = SupermemoryManager.generate_custom_id(
            "entry_point", "src/test.py", "main"
        )
        assert custom_id == expected_id


class TestUpdateMemoryTool:
    """Test update_memory tool function."""

    def test_update_memory_success(self, langchain_agent, mock_memory_manager):
        """Test successful memory update."""
        input_data = {
            "custom_id": "entry_point_src_test_py_main",
            "content": "Updated content",
            "metadata": {"line": 42, "version": 1}
        }
        input_str = json.dumps(input_data)

        result = langchain_agent._update_memory_tool(input_str)

        assert "success" in result
        assert langchain_agent.memories_updated == 1
        mock_memory_manager.update_memory.assert_called_once()

    def test_update_memory_missing_custom_id(self, langchain_agent):
        """Test error when custom_id is missing."""
        input_data = {
            "content": "Updated content",
            "metadata": {}
        }

        result = langchain_agent._update_memory_tool(json.dumps(input_data))

        assert "error" in result
        assert "custom_id is required" in result
        assert langchain_agent.memories_updated == 0


class TestSearchMemoryTool:
    """Test search_memory tool function."""

    def test_search_memory_success(self, langchain_agent, mock_memory_manager):
        """Test successful memory search."""
        # Mock search results
        mock_memory_manager.search_memory.return_value = [
            {
                "custom_id": "entry_point_src_test_py_main",
                "content": "Test content",
                "type": "entry_point",
                "version": 1
            }
        ]

        result = langchain_agent._search_memory_tool("test query")

        # Should return JSON string
        results = json.loads(result)
        assert len(results) == 1
        assert results[0]["custom_id"] == "entry_point_src_test_py_main"

    def test_search_memory_no_results(self, langchain_agent, mock_memory_manager):
        """Test search with no results."""
        mock_memory_manager.search_memory.return_value = []

        result = langchain_agent._search_memory_tool("nonexistent")

        results = json.loads(result)
        assert len(results) == 0


class TestProcessIteration:
    """Test process_iteration method."""

    @pytest.mark.asyncio
    async def test_process_iteration_disabled(self, langchain_agent):
        """Test iteration processing when memory manager is disabled."""
        langchain_agent.memory_manager.enabled = False

        result = await langchain_agent.process_iteration(
            query="test",
            code_chunks=[],
            judge_results=None
        )

        assert result["success"] is False
        assert "disabled" in result["reason"]
        assert langchain_agent.skipped_iterations == 1

    @pytest.mark.asyncio
    async def test_process_iteration_success(self, langchain_agent):
        """Test successful iteration processing."""
        # Mock agent executor
        langchain_agent.agent_executor.invoke = MagicMock(
            return_value={"output": "Created 1 memory"}
        )

        code_chunks = [
            {"file_path": "test.py", "content": "def main():"}
        ]
        judge_results = {
            1: {"is_relevant": True, "entry_point": True, "why": "Main function"}
        }

        result = await langchain_agent.process_iteration(
            query="find main",
            code_chunks=code_chunks,
            judge_results=judge_results
        )

        assert result["success"] is True
        assert langchain_agent.iterations_processed == 1

    @pytest.mark.asyncio
    async def test_process_iteration_error_handling(self, langchain_agent):
        """Test error handling in iteration processing."""
        # Mock agent executor to raise error
        langchain_agent.agent_executor.invoke = MagicMock(
            side_effect=Exception("Test error")
        )

        result = await langchain_agent.process_iteration(
            query="test",
            code_chunks=[],
            judge_results=None
        )

        assert result["success"] is False
        assert "Error" in result["reason"]


class TestSummarization:
    """Test summarization helper methods."""

    def test_summarize_code_chunks(self, langchain_agent):
        """Test code chunks summarization."""
        chunks = [
            {"file_path": "test1.py", "content": "x" * 150},
            {"file_path": "test2.py", "content": "y" * 150}
        ]

        summary = langchain_agent._summarize_code_chunks(chunks)

        assert "test1.py" in summary
        assert "test2.py" in summary
        # Content should be truncated
        assert len(summary) < len(chunks[0]["content"]) * len(chunks)

    def test_summarize_code_chunks_empty(self, langchain_agent):
        """Test summarization with no chunks."""
        summary = langchain_agent._summarize_code_chunks([])
        assert "No code chunks" in summary

    def test_summarize_judge_results(self, langchain_agent):
        """Test judge results summarization."""
        judge_results = {
            1: {"is_relevant": True, "entry_point": True, "why": "Main"},
            2: {"is_relevant": True, "entry_point": False, "why": "Helper"},
            3: {"is_relevant": False, "entry_point": False, "why": "Not relevant"}
        }

        summary = langchain_agent._summarize_judge_results(judge_results)

        assert "Total chunks evaluated: 3" in summary
        assert "Relevant chunks: 2" in summary
        assert "Entry points found: 1" in summary


class TestStatistics:
    """Test statistics tracking."""

    def test_get_stats(self, langchain_agent):
        """Test get_stats returns correct structure."""
        stats = langchain_agent.get_stats()

        assert "enabled" in stats
        assert "model" in stats
        assert "container_tag" in stats
        assert "iterations_processed" in stats
        assert "memories_created" in stats
        assert "memories_updated" in stats

        assert stats["model"] == "gpt-4o-mini"
        assert stats["container_tag"] == "test_run"

    def test_reset_stats(self, langchain_agent):
        """Test reset_stats clears counters."""
        # Set some stats
        langchain_agent.memories_created = 5
        langchain_agent.memories_updated = 3
        langchain_agent.iterations_processed = 10

        # Reset
        langchain_agent.reset_stats()

        # Verify reset
        assert langchain_agent.memories_created == 0
        assert langchain_agent.memories_updated == 0
        assert langchain_agent.iterations_processed == 0


class TestSupermemoryManagerIntegration:
    """Test integration with SupermemoryManager new methods."""

    def test_generate_custom_id(self):
        """Test custom_id generation."""
        from src.embeddinggemma.memory.supermemory_client import SupermemoryManager

        custom_id = SupermemoryManager.generate_custom_id(
            "entry_point",
            "src/server.py",
            "main"
        )

        assert custom_id == "entry_point_src_server_py_main"
        assert len(custom_id) <= 255

    def test_generate_custom_id_normalization(self):
        """Test that paths are normalized correctly."""
        from src.embeddinggemma.memory.supermemory_client import SupermemoryManager

        # Test with Windows path
        custom_id1 = SupermemoryManager.generate_custom_id(
            "pattern",
            "src\\utils\\helper.py",
            "my_function"
        )

        # Test with Unix path
        custom_id2 = SupermemoryManager.generate_custom_id(
            "pattern",
            "src/utils/helper.py",
            "my_function"
        )

        # Both should normalize to same ID
        assert custom_id1 == custom_id2
        assert "/" not in custom_id1
        assert "\\" not in custom_id1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
