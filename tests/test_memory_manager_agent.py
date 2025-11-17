"""
Unit tests for Memory Manager Agent.

Tests the LLM-powered decision making, conversation context handling,
and document ingestion logic.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.embeddinggemma.agents.memory_manager_agent import MemoryManagerAgent


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client for testing."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


@pytest.fixture
def mock_memory_manager():
    """Create mock SupermemoryManager for testing."""
    manager = MagicMock()
    manager.enabled = True
    manager.add_document = AsyncMock(return_value=True)
    manager.search_documents = AsyncMock(return_value=[])  # Return empty list (no duplicates)
    return manager


@pytest.fixture
def memory_agent(mock_llm_client, mock_memory_manager):
    """Create MemoryManagerAgent instance for testing."""
    return MemoryManagerAgent(
        llm_client=mock_llm_client,
        memory_manager=mock_memory_manager
    )


class TestMemoryManagerAgentInit:
    """Test Memory Manager Agent initialization."""

    def test_init_with_memory_manager(self, mock_llm_client, mock_memory_manager):
        """Test successful initialization with memory manager."""
        agent = MemoryManagerAgent(
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager
        )

        assert agent.enabled is True
        assert agent.llm_client == mock_llm_client
        assert agent.memory_manager == mock_memory_manager
        assert agent.decisions_made == 0
        assert agent.documents_ingested == 0
        assert agent.search_more_decisions == 0

    def test_init_without_memory_manager(self, mock_llm_client):
        """Test initialization without memory manager (disabled)."""
        agent = MemoryManagerAgent(
            llm_client=mock_llm_client,
            memory_manager=None
        )

        assert agent.enabled is False

    def test_init_with_disabled_memory_manager(self, mock_llm_client):
        """Test initialization with disabled memory manager."""
        disabled_manager = MagicMock()
        disabled_manager.enabled = False

        agent = MemoryManagerAgent(
            llm_client=mock_llm_client,
            memory_manager=disabled_manager
        )

        assert agent.enabled is False


class TestMemoryManagerAgentDecisions:
    """Test Memory Manager Agent decision making."""

    @pytest.mark.asyncio
    async def test_skip_when_disabled(self, mock_llm_client):
        """Test that agent skips when disabled."""
        agent = MemoryManagerAgent(
            llm_client=mock_llm_client,
            memory_manager=None
        )

        result = await agent.analyze_and_decide(
            query="test query",
            code_chunks=[{"content": "test"}],
            container_tag="test_run"
        )

        assert result["action"] == "skip"
        assert "disabled" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_skip_when_no_chunks(self, memory_agent):
        """Test that agent skips when no chunks provided."""
        result = await memory_agent.analyze_and_decide(
            query="test query",
            code_chunks=[],
            container_tag="test_run"
        )

        assert result["action"] == "skip"
        assert "no chunks" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_ingest_decision(self, memory_agent, mock_llm_client):
        """Test INGEST decision flow."""
        # Mock LLM response for INGEST decision
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "action": "ingest",
            "reason": "Complete understanding of authentication module",
            "confidence": 0.9,
            "documents": [
                {
                    "title": "Authentication Module - OAuth2",
                    "content": "Complete OAuth2 authentication flow...",
                    "type": "room",
                    "metadata": {
                        "file_path": "auth/oauth.py",
                        "exploration_status": "fully_explored",
                        "patterns": ["async/await", "OOP"],
                        "key_functions": ["authenticate", "validate"],
                        "key_classes": ["OAuth2Handler"]
                    }
                }
            ]
        }
        """

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        code_chunks = [
            {"content": "class OAuth2Handler:", "file_path": "auth/oauth.py"},
            {"content": "async def authenticate():", "file_path": "auth/oauth.py"},
        ]

        result = await memory_agent.analyze_and_decide(
            query="Find authentication",
            code_chunks=code_chunks,
            container_tag="test_run"
        )

        assert result["action"] == "ingest"
        assert result["confidence"] == 0.9
        assert len(result["documents"]) == 1
        assert result["documents"][0]["title"] == "Authentication Module - OAuth2"
        assert memory_agent.decisions_made == 1
        assert memory_agent.documents_ingested == 1

    @pytest.mark.asyncio
    async def test_search_more_decision(self, memory_agent, mock_llm_client):
        """Test SEARCH_MORE decision flow."""
        # Mock LLM response for SEARCH_MORE decision
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "action": "search_more",
            "reason": "Only entry point found, need implementation details",
            "confidence": 0.6,
            "suggested_queries": [
                "Explore OAuth2Handler methods",
                "Find token management"
            ]
        }
        """

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        code_chunks = [
            {"content": "class OAuth2Handler:", "file_path": "auth/oauth.py"}
        ]

        result = await memory_agent.analyze_and_decide(
            query="Find authentication",
            code_chunks=code_chunks,
            container_tag="test_run"
        )

        assert result["action"] == "search_more"
        assert len(result["suggested_queries"]) == 2
        assert memory_agent.decisions_made == 1
        assert memory_agent.search_more_decisions == 1


class TestConversationContext:
    """Test conversation context handling."""

    @pytest.mark.asyncio
    async def test_conversation_context_in_prompt(self, memory_agent, mock_llm_client):
        """Test that conversation context is included in prompt."""
        conversation_history = [
            {
                "step": 1,
                "query": "Find authentication",
                "discoveries": "OAuth2Handler found",
                "decision": "search_more: Need more context"
            },
            {
                "step": 2,
                "query": "Explore OAuth2Handler methods",
                "discoveries": "3 methods found",
                "decision": "search_more: Missing dependencies"
            }
        ]

        # Capture the prompt sent to LLM
        captured_prompt = None

        async def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get('messages', [{}])[1].get('content', '')
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"action": "skip", "reason": "test"}'
            return mock_response

        mock_llm_client.chat.completions.create = capture_prompt

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        await memory_agent.analyze_and_decide(
            query="Find token management",
            code_chunks=code_chunks,
            container_tag="test_run",
            conversation_history=conversation_history
        )

        # Verify conversation context is in prompt
        assert captured_prompt is not None
        assert "CONVERSATION HISTORY" in captured_prompt
        assert "Step 1" in captured_prompt
        assert "Step 2" in captured_prompt
        assert "Find authentication" in captured_prompt
        assert "OAuth2Handler found" in captured_prompt
        assert "WHAT WE'VE LEARNED SO FAR" in captured_prompt
        assert "Total exploration steps: 2" in captured_prompt

    @pytest.mark.asyncio
    async def test_no_conversation_context(self, memory_agent, mock_llm_client):
        """Test prompt when no conversation history provided."""
        captured_prompt = None

        async def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get('messages', [{}])[1].get('content', '')
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"action": "skip", "reason": "test"}'
            return mock_response

        mock_llm_client.chat.completions.create = capture_prompt

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        await memory_agent.analyze_and_decide(
            query="test query",
            code_chunks=code_chunks,
            container_tag="test_run",
            conversation_history=None
        )

        # Verify no conversation context in prompt
        assert captured_prompt is not None
        assert "CONVERSATION HISTORY" not in captured_prompt


class TestJudgeResultsIntegration:
    """Test integration with judge results."""

    @pytest.mark.asyncio
    async def test_judge_results_in_prompt(self, memory_agent, mock_llm_client):
        """Test that judge results are included in prompt."""
        judge_results = {
            1: {"is_relevant": True, "entry_point": True, "why": "Main server initialization"},
            2: {"is_relevant": True, "entry_point": False, "why": "Helper function"},
            3: {"is_relevant": False, "entry_point": False, "why": "Not relevant"}
        }

        captured_prompt = None

        async def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get('messages', [{}])[1].get('content', '')
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"action": "skip", "reason": "test"}'
            return mock_response

        mock_llm_client.chat.completions.create = capture_prompt

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        await memory_agent.analyze_and_decide(
            query="test query",
            code_chunks=code_chunks,
            judge_results=judge_results,
            container_tag="test_run"
        )

        # Verify judge results are in prompt
        assert captured_prompt is not None
        assert "JUDGE EVALUATION RESULTS" in captured_prompt
        assert "Total chunks evaluated: 3" in captured_prompt
        assert "Relevant chunks: 2" in captured_prompt
        assert "Entry points found: 1" in captured_prompt


class TestDocumentIngestion:
    """Test document ingestion logic."""

    @pytest.mark.asyncio
    async def test_ingest_documents_success(self, memory_agent, mock_memory_manager, mock_llm_client):
        """Test successful document ingestion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "action": "ingest",
            "reason": "Complete module",
            "documents": [
                {
                    "title": "Test Module",
                    "content": "Complete test module",
                    "type": "room",
                    "metadata": {"file_path": "test.py"}
                }
            ]
        }
        """

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        result = await memory_agent.analyze_and_decide(
            query="test",
            code_chunks=code_chunks,
            container_tag="test_run"
        )

        # Verify add_document was called
        mock_memory_manager.add_document.assert_called_once()
        call_args = mock_memory_manager.add_document.call_args

        assert call_args.kwargs["title"] == "Test Module"
        assert call_args.kwargs["content"] == "Complete test module"
        assert call_args.kwargs["doc_type"] == "room"
        assert call_args.kwargs["container_tag"] == "test_run"

        # Verify stats updated
        assert memory_agent.documents_ingested == 1


class TestStatistics:
    """Test statistics tracking."""

    def test_get_stats(self, memory_agent):
        """Test get_stats returns correct structure."""
        stats = memory_agent.get_stats()

        assert "enabled" in stats
        assert "decisions_made" in stats
        assert "documents_ingested" in stats
        assert "search_more_decisions" in stats

        assert stats["enabled"] is True
        assert stats["decisions_made"] == 0
        assert stats["documents_ingested"] == 0
        assert stats["search_more_decisions"] == 0

    def test_reset_stats(self, memory_agent):
        """Test reset_stats clears counters."""
        # Manually set stats
        memory_agent.decisions_made = 5
        memory_agent.documents_ingested = 3
        memory_agent.search_more_decisions = 2

        # Reset
        memory_agent.reset_stats()

        # Verify reset
        assert memory_agent.decisions_made == 0
        assert memory_agent.documents_ingested == 0
        assert memory_agent.search_more_decisions == 0


class TestErrorHandling:
    """Test error handling in Memory Manager Agent."""

    @pytest.mark.asyncio
    async def test_llm_error_handling(self, memory_agent, mock_llm_client):
        """Test graceful error handling when LLM fails."""
        mock_llm_client.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM API error")
        )

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        result = await memory_agent.analyze_and_decide(
            query="test",
            code_chunks=code_chunks,
            container_tag="test_run"
        )

        assert result["action"] == "skip"
        assert "Error" in result["reason"]

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, memory_agent, mock_llm_client):
        """Test handling of invalid JSON from LLM."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid JSON response"

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        result = await memory_agent.analyze_and_decide(
            query="test",
            code_chunks=code_chunks,
            container_tag="test_run"
        )

        # Should default to skip
        assert result["action"] == "skip"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
