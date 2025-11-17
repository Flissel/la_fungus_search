"""
End-to-end tests for Memory Manager Agent in full simulation workflow.

Simulates complete exploration runs to validate:
- Conversation history builds correctly across steps
- Agent makes appropriate ingest/search_more decisions
- Documents are stored with proper metadata
- Statistics are tracked accurately
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


class TestFullExplorationRun:
    """Test complete exploration run from start to finish."""

    @pytest.mark.asyncio
    async def test_progressive_exploration_to_ingest(self, memory_agent, mock_llm_client):
        """Test realistic exploration pattern: search_more → search_more → ingest."""

        # Simulate realistic LLM responses over 5 steps
        step_responses = [
            # Step 1: Find entry point
            {
                "action": "search_more",
                "reason": "Found entry point but need to explore implementation",
                "confidence": 0.4,
                "suggested_queries": ["Explore FastAPI route handlers", "Find WebSocket endpoints"]
            },
            # Step 2: Explore routes
            {
                "action": "search_more",
                "reason": "Found route definitions but missing business logic",
                "confidence": 0.5,
                "suggested_queries": ["Find code search logic", "Explore judge evaluation"]
            },
            # Step 3: Explore search logic
            {
                "action": "search_more",
                "reason": "Understanding search but need context on judge integration",
                "confidence": 0.6,
                "suggested_queries": ["Find judge implementation", "Explore LLM client setup"]
            },
            # Step 4: More complete understanding
            {
                "action": "search_more",
                "reason": "Almost complete - need to verify memory integration",
                "confidence": 0.7,
                "suggested_queries": ["Find Supermemory integration", "Explore memory manager"]
            },
            # Step 5: Complete understanding - ready to ingest
            {
                "action": "ingest",
                "reason": "Complete understanding of FastAPI server architecture with search, judge, and memory components",
                "confidence": 0.9,
                "documents": [
                    {
                        "title": "FastAPI Realtime Server - Complete Architecture",
                        "content": "Main server implementing code search with LLM judge and memory integration. Key components: WebSocket communication, code chunk retrieval, judge-based relevance evaluation, memory manager agent for knowledge storage.",
                        "type": "room",
                        "metadata": {
                            "file_path": "src/embeddinggemma/realtime/server.py",
                            "exploration_status": "fully_explored",
                            "patterns": ["async/await", "WebSocket", "REST API"],
                            "key_functions": ["_llm_judge", "_broadcast", "_search_code"],
                            "key_classes": ["RealtimeServer"],
                            "dependencies": ["judge", "memory_manager", "supermemory_client"]
                        }
                    },
                    {
                        "title": "Judge Evaluation System",
                        "content": "LLM-powered judge that evaluates code chunk relevance and generates follow-up queries",
                        "type": "module",
                        "metadata": {
                            "file_path": "src/embeddinggemma/judge.py",
                            "patterns": ["LLM integration"],
                            "key_functions": ["evaluate_relevance", "generate_queries"]
                        }
                    }
                ]
            }
        ]

        # Create mock responses
        mock_responses = []
        for response_data in step_responses:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            import json
            mock_response.choices[0].message.content = json.dumps(response_data)
            mock_responses.append(mock_response)

        mock_llm_client.chat.completions.create = AsyncMock(side_effect=mock_responses)

        # Simulate exploration queries
        exploration_queries = [
            "Find FastAPI server entry point",
            "Explore route handlers",
            "Find search logic implementation",
            "Explore judge integration",
            "Find memory manager integration"
        ]

        # Simulate code chunks found at each step
        code_chunks_per_step = [
            [{"content": "@app.get('/health')", "file_path": "server.py"}],
            [{"content": "@app.post('/search')", "file_path": "server.py"}],
            [{"content": "async def _search_code()", "file_path": "server.py"}],
            [{"content": "await self._llm_judge()", "file_path": "server.py"}],
            [{"content": "class MemoryManagerAgent", "file_path": "memory_manager_agent.py"}]
        ]

        # Run simulation
        conversation_history = []
        decisions = []

        for i, (query, chunks) in enumerate(zip(exploration_queries, code_chunks_per_step), 1):
            decision = await memory_agent.analyze_and_decide(
                query=query,
                code_chunks=chunks,
                container_tag="simulation_run_001",
                conversation_history=conversation_history
            )

            # Build conversation history entry
            step_entry = {
                "step": i,
                "query": query,
                "discoveries": f"{len(chunks)} chunks found, patterns identified",
                "decision": f"{decision['action']}: {decision['reason']}"
            }
            conversation_history.append(step_entry)
            decisions.append(decision)

        # Verify progression
        assert decisions[0]["action"] == "search_more"
        assert decisions[1]["action"] == "search_more"
        assert decisions[2]["action"] == "search_more"
        assert decisions[3]["action"] == "search_more"
        assert decisions[4]["action"] == "ingest"

        # Verify confidence increases over time
        assert decisions[0]["confidence"] < decisions[1]["confidence"]
        assert decisions[3]["confidence"] < decisions[4]["confidence"]

        # Verify final ingestion
        assert len(decisions[4]["documents"]) == 2
        assert "FastAPI Realtime Server" in decisions[4]["documents"][0]["title"]

        # Verify conversation history structure
        assert len(conversation_history) == 5
        assert conversation_history[-1]["step"] == 5
        assert "ingest" in conversation_history[-1]["decision"]

        # Verify stats
        stats = memory_agent.get_stats()
        assert stats["decisions_made"] == 5
        assert stats["search_more_decisions"] == 4
        assert stats["documents_ingested"] == 2

    @pytest.mark.asyncio
    async def test_early_ingest_when_complete(self, memory_agent, mock_llm_client):
        """Test agent ingests early when finding complete isolated module."""

        # Simulate finding a complete utility module immediately
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "action": "ingest",
            "reason": "Complete standalone utility module with clear purpose and no external dependencies",
            "confidence": 0.85,
            "documents": [
                {
                    "title": "String Utilities Module",
                    "content": "Self-contained string manipulation utilities including sanitization, formatting, and validation",
                    "type": "module",
                    "metadata": {
                        "file_path": "utils/string_utils.py",
                        "exploration_status": "fully_explored",
                        "patterns": ["functional programming"],
                        "key_functions": ["sanitize_string", "format_path", "validate_input"],
                        "dependencies": []
                    }
                }
            ]
        }
        """

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        code_chunks = [
            {"content": "def sanitize_string(s: str) -> str:", "file_path": "utils/string_utils.py"},
            {"content": "def format_path(path: str) -> str:", "file_path": "utils/string_utils.py"},
            {"content": "def validate_input(text: str) -> bool:", "file_path": "utils/string_utils.py"}
        ]

        conversation_history = []
        decision = await memory_agent.analyze_and_decide(
            query="Find string utilities",
            code_chunks=code_chunks,
            container_tag="simulation_run_002",
            conversation_history=conversation_history
        )

        # Should ingest immediately
        assert decision["action"] == "ingest"
        assert decision["confidence"] >= 0.8
        assert len(decision["documents"]) == 1
        assert decision["documents"][0]["title"] == "String Utilities Module"

        # Stats should show single decision leading to ingest
        stats = memory_agent.get_stats()
        assert stats["decisions_made"] == 1
        assert stats["search_more_decisions"] == 0
        assert stats["documents_ingested"] == 1


class TestConversationContextImpact:
    """Test how conversation context affects agent decisions."""

    @pytest.mark.asyncio
    async def test_context_prevents_redundant_ingestion(self, memory_agent, mock_llm_client):
        """Test that agent considers past ingestions to avoid redundancy."""

        # First ingestion
        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message.content = """
        {
            "action": "ingest",
            "reason": "Complete authentication module",
            "documents": [{"title": "Auth Module", "content": "OAuth2 implementation", "type": "room"}]
        }
        """

        # Second query on similar topic - should skip or search_more
        second_response = MagicMock()
        second_response.choices = [MagicMock()]
        second_response.choices[0].message.content = """
        {
            "action": "skip",
            "reason": "Already ingested complete authentication module in previous step",
            "confidence": 0.9
        }
        """

        mock_llm_client.chat.completions.create = AsyncMock(side_effect=[first_response, second_response])

        code_chunks = [{"content": "class OAuth2Handler:", "file_path": "auth.py"}]

        # First decision
        conversation_history = []
        decision1 = await memory_agent.analyze_and_decide(
            query="Find authentication",
            code_chunks=code_chunks,
            container_tag="test_run",
            conversation_history=conversation_history
        )

        conversation_history.append({
            "step": 1,
            "query": "Find authentication",
            "discoveries": "OAuth2Handler found",
            "decision": f"{decision1['action']}: {decision1['reason']}"
        })

        # Second decision with similar content
        decision2 = await memory_agent.analyze_and_decide(
            query="Explore OAuth implementation",
            code_chunks=code_chunks,
            container_tag="test_run",
            conversation_history=conversation_history
        )

        # First should ingest, second should skip
        assert decision1["action"] == "ingest"
        assert decision2["action"] == "skip"
        assert "already" in decision2["reason"].lower() or "previous" in decision2["reason"].lower()

    @pytest.mark.asyncio
    async def test_conversation_summary_in_prompt(self, memory_agent, mock_llm_client):
        """Test that long conversation history is properly summarized in prompt."""

        # Build extensive history (15 steps)
        extensive_history = []
        for i in range(1, 16):
            extensive_history.append({
                "step": i,
                "query": f"Query {i}: Exploring module {i}",
                "discoveries": f"Found {i*2} code chunks",
                "decision": "search_more: Need more context"
            })

        captured_prompt = None

        async def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get('messages', [{}])[1].get('content', '')
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"action": "ingest", "reason": "test", "documents": []}'
            return mock_response

        mock_llm_client.chat.completions.create = capture_prompt

        code_chunks = [{"content": "test", "file_path": "test.py"}]

        await memory_agent.analyze_and_decide(
            query="Final query",
            code_chunks=code_chunks,
            container_tag="test_run",
            conversation_history=extensive_history
        )

        # Verify conversation context is in prompt
        assert captured_prompt is not None
        assert "CONVERSATION HISTORY" in captured_prompt
        assert "Total exploration steps: 15" in captured_prompt
        assert "Recent queries explored: 10" in captured_prompt  # Only last 10 shown

        # Verify some recent steps are mentioned
        assert "Step 6" in captured_prompt or "Step 7" in captured_prompt


class TestManifestIntegration:
    """Test manifest statistics collection for simulation runs."""

    @pytest.mark.asyncio
    async def test_manifest_stats_complete_run(self, memory_agent, mock_llm_client):
        """Test collecting complete manifest stats after simulation run."""

        # Simulate varied decisions
        responses = [
            '{"action": "search_more", "reason": "Need more", "suggested_queries": ["q1"]}',
            '{"action": "ingest", "reason": "Complete", "documents": [{"title": "Doc1", "content": "test", "type": "room"}]}',
            '{"action": "skip", "reason": "Not relevant"}',
            '{"action": "search_more", "reason": "Need more", "suggested_queries": ["q2"]}',
            '{"action": "ingest", "reason": "Complete", "documents": [{"title": "Doc2", "content": "test", "type": "room"}, {"title": "Doc3", "content": "test", "type": "module"}]}'
        ]

        mock_responses = []
        for resp in responses:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = resp
            mock_responses.append(mock_response)

        mock_llm_client.chat.completions.create = AsyncMock(side_effect=mock_responses)

        # Run 5 steps
        conversation_history = []
        code_chunks = [{"content": "test", "file_path": "test.py"}]

        for i in range(5):
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
                "decision": f"{decision['action']}: {decision['reason']}"
            })

        # Collect manifest stats
        agent_stats = memory_agent.get_stats()
        manifest = {
            "run_id": "test_run",
            "memory_stats": {
                "agent_enabled": agent_stats["enabled"],
                "agent_decisions": agent_stats["decisions_made"],
                "agent_documents_ingested": agent_stats["documents_ingested"],
                "agent_search_more_decisions": agent_stats["search_more_decisions"],
                "conversation_history_steps": len(conversation_history)
            }
        }

        # Verify manifest structure
        assert manifest["memory_stats"]["agent_enabled"] is True
        assert manifest["memory_stats"]["agent_decisions"] == 5
        assert manifest["memory_stats"]["agent_documents_ingested"] == 3  # Doc1 + Doc2 + Doc3
        assert manifest["memory_stats"]["agent_search_more_decisions"] == 2
        assert manifest["memory_stats"]["conversation_history_steps"] == 5


class TestRealisticScenarios:
    """Test realistic exploration scenarios."""

    @pytest.mark.asyncio
    async def test_api_exploration_scenario(self, memory_agent, mock_llm_client):
        """Test exploring a REST API implementation."""

        scenario_responses = [
            # Step 1: Find API routes
            {
                "action": "search_more",
                "reason": "Found route definitions but need to understand handlers",
                "suggested_queries": ["Explore search endpoint handler", "Find WebSocket endpoint"]
            },
            # Step 2: Explore handlers
            {
                "action": "search_more",
                "reason": "Understanding handlers but need data models",
                "suggested_queries": ["Find request/response models", "Explore validation logic"]
            },
            # Step 3: Complete understanding
            {
                "action": "ingest",
                "reason": "Complete API structure with routes, handlers, and models",
                "documents": [
                    {
                        "title": "REST API - Search Endpoints",
                        "content": "Complete search API with routes, handlers, request validation, and response models",
                        "type": "room",
                        "metadata": {
                            "file_path": "api/search.py",
                            "patterns": ["REST", "async/await"],
                            "key_functions": ["search_handler", "validate_query"],
                            "dependencies": ["models", "validators"]
                        }
                    }
                ]
            }
        ]

        mock_responses = []
        for response_data in scenario_responses:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            import json
            mock_response.choices[0].message.content = json.dumps(response_data)
            mock_responses.append(mock_response)

        mock_llm_client.chat.completions.create = AsyncMock(side_effect=mock_responses)

        queries = [
            "Find API routes",
            "Explore request handlers",
            "Find data models"
        ]

        conversation_history = []
        final_decision = None

        for i, query in enumerate(queries, 1):
            decision = await memory_agent.analyze_and_decide(
                query=query,
                code_chunks=[{"content": f"code {i}", "file_path": "api.py"}],
                container_tag="api_exploration",
                conversation_history=conversation_history
            )

            conversation_history.append({
                "step": i,
                "query": query,
                "discoveries": f"Step {i} discoveries",
                "decision": f"{decision['action']}: {decision['reason']}"
            })

            final_decision = decision

        # Verify final ingestion
        assert final_decision["action"] == "ingest"
        assert "REST API" in final_decision["documents"][0]["title"]
        assert len(conversation_history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
