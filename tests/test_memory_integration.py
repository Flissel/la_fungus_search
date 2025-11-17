"""
Integration tests for Memory Manager Agent with server workflow.

Tests the complete flow:
Server → Conversation History → Memory Manager Agent → Supermemory
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
    manager.search_documents = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def memory_agent(mock_llm_client, mock_memory_manager):
    """Create MemoryManagerAgent instance for testing."""
    return MemoryManagerAgent(
        llm_client=mock_llm_client,
        memory_manager=mock_memory_manager
    )


class TestServerConversationTracking:
    """Test conversation history tracking in server workflow."""

    @pytest.mark.asyncio
    async def test_conversation_history_builds_over_steps(self, memory_agent, mock_llm_client):
        """Test that conversation history accumulates across multiple steps."""
        conversation_history = []

        # Mock LLM to return search_more for first 2 steps, then ingest
        step_responses = [
            # Step 1: Search more
            {
                "action": "search_more",
                "reason": "Only found entry point, need implementation",
                "suggested_queries": ["Explore OAuth2Handler methods"]
            },
            # Step 2: Search more
            {
                "action": "search_more",
                "reason": "Found methods but missing dependencies",
                "suggested_queries": ["Find token management"]
            },
            # Step 3: Ingest
            {
                "action": "ingest",
                "reason": "Complete understanding of auth module",
                "documents": [
                    {
                        "title": "Authentication Module",
                        "content": "Complete OAuth2 implementation",
                        "type": "room",
                        "metadata": {"file_path": "auth/oauth.py"}
                    }
                ]
            }
        ]

        mock_responses = []
        for response_data in step_responses:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = str(response_data).replace("'", '"')
            mock_responses.append(mock_response)

        mock_llm_client.chat.completions.create = AsyncMock(side_effect=mock_responses)

        code_chunks = [{"content": "class OAuth2Handler:", "file_path": "auth/oauth.py"}]

        # Simulate 3 exploration steps
        queries = ["Find authentication", "Explore OAuth2Handler", "Find token management"]

        for i, query in enumerate(queries, 1):
            decision = await memory_agent.analyze_and_decide(
                query=query,
                code_chunks=code_chunks,
                container_tag="test_run",
                conversation_history=conversation_history
            )

            # Build conversation history entry (simulating server behavior)
            step_entry = {
                "step": i,
                "query": query,
                "discoveries": f"{len(code_chunks)} chunks found",
                "decision": f"{decision['action']}: {decision['reason']}"
            }
            conversation_history.append(step_entry)

        # Verify conversation history structure
        assert len(conversation_history) == 3
        assert conversation_history[0]["step"] == 1
        assert conversation_history[0]["query"] == "Find authentication"
        assert "ingest" in conversation_history[2]["decision"]

    @pytest.mark.asyncio
    async def test_conversation_history_influences_decisions(self, memory_agent, mock_llm_client):
        """Test that conversation history affects agent decisions."""
        # Simulate a conversation where we've already explored a lot
        extensive_history = [
            {"step": i, "query": f"Query {i}", "discoveries": "Found modules", "decision": "search_more"}
            for i in range(1, 11)
        ]

        captured_prompts = []

        async def capture_prompt(*args, **kwargs):
            captured_prompts.append(kwargs.get('messages', [{}])[1].get('content', ''))
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"action": "ingest", "reason": "Extensive exploration complete", "documents": []}'
            return mock_response

        mock_llm_client.chat.completions.create = capture_prompt

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        # First call without history
        await memory_agent.analyze_and_decide(
            query="test",
            code_chunks=code_chunks,
            container_tag="test_run",
            conversation_history=None
        )

        # Second call with extensive history
        await memory_agent.analyze_and_decide(
            query="test",
            code_chunks=code_chunks,
            container_tag="test_run",
            conversation_history=extensive_history
        )

        # Verify first prompt has no conversation context
        assert "CONVERSATION HISTORY" not in captured_prompts[0]

        # Verify second prompt includes conversation context
        assert "CONVERSATION HISTORY" in captured_prompts[1]
        assert "Total exploration steps: 10" in captured_prompts[1]


class TestMemoryAgentServerIntegration:
    """Test Memory Manager Agent integration with server components."""

    @pytest.mark.asyncio
    async def test_agent_stats_tracking(self, memory_agent, mock_llm_client, mock_memory_manager):
        """Test that agent tracks statistics correctly across multiple calls."""
        # Mock LLM responses: 2 ingest, 1 search_more, 1 skip
        responses = [
            '{"action": "ingest", "reason": "Complete", "documents": [{"title": "Doc1", "content": "test", "type": "room"}]}',
            '{"action": "search_more", "reason": "Need more", "suggested_queries": ["query1"]}',
            '{"action": "ingest", "reason": "Complete", "documents": [{"title": "Doc2", "content": "test", "type": "room"}]}',
            '{"action": "skip", "reason": "Not relevant"}'
        ]

        mock_llm_responses = []
        for resp in responses:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = resp
            mock_llm_responses.append(mock_response)

        mock_llm_client.chat.completions.create = AsyncMock(side_effect=mock_llm_responses)

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        # Execute 4 decisions
        for i in range(4):
            await memory_agent.analyze_and_decide(
                query=f"test {i}",
                code_chunks=code_chunks,
                container_tag="test_run"
            )

        # Verify statistics
        stats = memory_agent.get_stats()
        assert stats["decisions_made"] == 4
        assert stats["documents_ingested"] == 2  # 2 successful ingests
        assert stats["search_more_decisions"] == 1

    @pytest.mark.asyncio
    async def test_manifest_stats_integration(self, memory_agent, mock_llm_client):
        """Test that stats can be collected for manifest."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action": "ingest", "reason": "test", "documents": [{"title": "Test", "content": "test", "type": "room"}]}'

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        # Simulate 3 exploration steps
        conversation_history = []
        for i in range(3):
            decision = await memory_agent.analyze_and_decide(
                query=f"query {i}",
                code_chunks=code_chunks,
                container_tag="test_run",
                conversation_history=conversation_history
            )

            conversation_history.append({
                "step": i + 1,
                "query": f"query {i}",
                "discoveries": "test",
                "decision": decision["action"]
            })

        # Collect manifest stats (simulating server behavior)
        stats = memory_agent.get_stats()
        manifest_stats = {
            "agent_enabled": stats["enabled"],
            "agent_decisions": stats["decisions_made"],
            "agent_documents_ingested": stats["documents_ingested"],
            "agent_search_more_decisions": stats["search_more_decisions"],
            "conversation_history_steps": len(conversation_history)
        }

        # Verify manifest stats structure
        assert manifest_stats["agent_enabled"] is True
        assert manifest_stats["agent_decisions"] == 3
        assert manifest_stats["conversation_history_steps"] == 3


class TestDocumentIngestionFlow:
    """Test complete document ingestion flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_document_ingestion(self, memory_agent, mock_llm_client, mock_memory_manager):
        """Test complete flow from decision to document storage."""
        # Mock LLM to return ingest decision
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "action": "ingest",
            "reason": "Complete authentication module understanding",
            "confidence": 0.95,
            "documents": [
                {
                    "title": "OAuth2 Authentication Handler",
                    "content": "Complete OAuth2 implementation with token management",
                    "type": "room",
                    "metadata": {
                        "file_path": "auth/oauth.py",
                        "exploration_status": "fully_explored",
                        "patterns": ["async/await", "OOP"],
                        "key_functions": ["authenticate", "validate_token"],
                        "key_classes": ["OAuth2Handler"],
                        "dependencies": ["token_manager", "user_db"]
                    }
                },
                {
                    "title": "Token Management Module",
                    "content": "JWT token generation and validation",
                    "type": "module",
                    "metadata": {
                        "file_path": "auth/token.py",
                        "exploration_status": "fully_explored",
                        "patterns": ["JWT"],
                        "key_functions": ["generate_token", "validate_token"]
                    }
                }
            ]
        }
        """

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        code_chunks = [
            {"content": "class OAuth2Handler:", "file_path": "auth/oauth.py"},
            {"content": "async def authenticate():", "file_path": "auth/oauth.py"},
            {"content": "def generate_token():", "file_path": "auth/token.py"}
        ]

        result = await memory_agent.analyze_and_decide(
            query="Find authentication implementation",
            code_chunks=code_chunks,
            container_tag="test_run_123"
        )

        # Verify decision
        assert result["action"] == "ingest"
        assert result["confidence"] == 0.95
        assert result["ingested_count"] == 2

        # Verify add_document was called correctly
        assert mock_memory_manager.add_document.call_count == 2

        # Check first document call
        first_call = mock_memory_manager.add_document.call_args_list[0]
        assert first_call.kwargs["title"] == "OAuth2 Authentication Handler"
        assert first_call.kwargs["doc_type"] == "room"
        assert first_call.kwargs["container_tag"] == "test_run_123"
        assert "async/await" in first_call.kwargs["metadata"]["patterns"]

        # Check second document call
        second_call = mock_memory_manager.add_document.call_args_list[1]
        assert second_call.kwargs["title"] == "Token Management Module"
        assert second_call.kwargs["doc_type"] == "module"


class TestErrorRecovery:
    """Test error handling and recovery in integration scenarios."""

    @pytest.mark.asyncio
    async def test_partial_document_ingestion_failure(self, memory_agent, mock_llm_client, mock_memory_manager):
        """Test handling when some documents fail to ingest."""
        # Mock add_document to succeed for first doc, fail for second
        mock_memory_manager.add_document = AsyncMock(side_effect=[True, False])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "action": "ingest",
            "reason": "test",
            "documents": [
                {"title": "Doc1", "content": "test1", "type": "room"},
                {"title": "Doc2", "content": "test2", "type": "room"}
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

        # Should report only 1 successful ingestion
        assert result["ingested_count"] == 1
        assert memory_agent.documents_ingested == 1

    @pytest.mark.asyncio
    async def test_llm_timeout_recovery(self, memory_agent, mock_llm_client):
        """Test recovery when LLM call times out."""
        mock_llm_client.chat.completions.create = AsyncMock(
            side_effect=TimeoutError("LLM timeout")
        )

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        result = await memory_agent.analyze_and_decide(
            query="test",
            code_chunks=code_chunks,
            container_tag="test_run"
        )

        # Should gracefully fallback to skip
        assert result["action"] == "skip"
        assert "Error" in result["reason"] or "timeout" in result["reason"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
