"""
LangChain Memory Agent for incremental knowledge building.

This agent creates and updates memories on EVERY iteration of exploration,
enabling progressive learning where the judge improves with each step.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

_logger = logging.getLogger(__name__)


class LangChainMemoryAgent:
    """
    LangChain-powered agent for incremental memory creation and updates.

    This agent runs on EVERY exploration iteration to:
    1. Analyze judge results and code chunks
    2. Check for existing similar memories
    3. Create new memories or update existing ones
    4. Track what has been learned so far

    Unlike the old Memory Manager Agent that batched storage when "complete",
    this agent builds knowledge incrementally, enabling progressive learning.
    """

    def __init__(
        self,
        llm,
        memory_manager,
        container_tag: str,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize LangChain Memory Agent.

        Args:
            llm: LangChain LLM instance (OpenAI, Ollama, etc.)
            memory_manager: SupermemoryManager instance
            container_tag: Container tag for this exploration run (run_id)
            model: LLM model name for logging
        """
        self.llm = llm
        self.memory_manager = memory_manager
        self.container_tag = container_tag
        self.model = model

        # Statistics
        self.memories_created = 0
        self.memories_updated = 0
        self.iterations_processed = 0
        self.skipped_iterations = 0

        # Build agent
        self.tools = self._build_tools()
        self.agent_prompt = self._build_prompt()
        self.agent_executor = self._create_agent()

        _logger.info(
            f"[LANGCHAIN-AGENT] LangChain Memory Agent initialized "
            f"(model: {model}, container: {container_tag})"
        )

    def _build_tools(self) -> list[Tool]:
        """Build tools for the LangChain agent."""
        return [
            Tool(
                name="add_memory",
                func=self._add_memory_tool,
                description=(
                    "Create a new memory about a code discovery. "
                    "Input: JSON with keys 'content', 'type', 'file_path', 'identifier', 'metadata'. "
                    "Types: entry_point, pattern, dependency, bug, security. "
                    "Returns: 'success' or 'error'."
                )
            ),
            Tool(
                name="update_memory",
                func=self._update_memory_tool,
                description=(
                    "Update an existing memory with new information. "
                    "Input: JSON with keys 'custom_id', 'content', 'metadata'. "
                    "Returns: 'success' or 'error'."
                )
            ),
            Tool(
                name="search_memory",
                func=self._search_memory_tool,
                description=(
                    "Search existing memories to check for duplicates or similar findings. "
                    "Input: search query string. "
                    "Returns: JSON list of matching memories with custom_id, content, version."
                )
            )
        ]

    def _build_prompt(self) -> PromptTemplate:
        """Build the ReAct prompt template for the agent."""
        template = """You are a memory management agent for code exploration. Your job is to analyze
code discoveries and create or update memories in a knowledge base.

CURRENT ITERATION CONTEXT:
Query: {query}
Code Chunks Found: {code_chunks_summary}
Judge Evaluation: {judge_summary}

MEMORY TYPES:
- entry_point: Main functions, API routes, CLI entrypoints
- pattern: Architectural patterns (async/await, DI, factories)
- dependency: Critical imports and external dependencies
- bug: Error patterns, suspicious code
- security: Authentication, authorization, vulnerabilities

YOUR TASK:
1. Analyze what was discovered in this iteration
2. Search for existing similar memories (avoid duplicates)
3. If similar memory exists: UPDATE it with new info
4. If no similar memory: CREATE a new memory
5. Prioritize significant discoveries (entry points, patterns, security issues)

GUIDELINES:
- Always search before creating to avoid duplicates
- Update existing memories when you find related new info
- Use descriptive content (what, where, why it matters)
- Include file_path, line numbers, and confidence in metadata
- Skip trivial findings (imports, comments, boilerplate)

You have access to the following tools:

{tools}

Use the following format:

Thought: What should I do with these discoveries?
Action: tool_name
Action Input: tool input
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now know what to do
Final Answer: Summary of memories created/updated

Begin!

Thought: {agent_scratchpad}"""

        return PromptTemplate.from_template(template)

    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent executor."""
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.agent_prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def _add_memory_tool(self, input_str: str) -> str:
        """Tool function for adding new memory."""
        try:
            # Parse JSON input
            data = json.loads(input_str)

            content = data.get("content", "")
            type_ = data.get("type", "discovery")
            file_path = data.get("file_path", "")
            identifier = data.get("identifier", "unknown")
            metadata = data.get("metadata", {})

            # Generate custom_id for deduplication
            from src.embeddinggemma.memory.supermemory_client import SupermemoryManager
            custom_id = SupermemoryManager.generate_custom_id(type_, file_path, identifier)

            # Add metadata fields
            metadata.update({
                "file_path": file_path,
                "identifier": identifier
            })

            # Call async method (need to handle async in sync context)
            import asyncio
            loop = asyncio.get_event_loop()
            success = loop.run_until_complete(
                self.memory_manager.add_memory(
                    content=content,
                    type=type_,
                    metadata=metadata,
                    custom_id=custom_id,
                    container_tag=self.container_tag
                )
            )

            if success:
                self.memories_created += 1
                return f"success: Created memory '{custom_id}'"
            else:
                return "error: Failed to create memory"

        except Exception as e:
            _logger.error(f"[LANGCHAIN-AGENT] Error in add_memory tool: {e}")
            return f"error: {str(e)}"

    def _update_memory_tool(self, input_str: str) -> str:
        """Tool function for updating existing memory."""
        try:
            # Parse JSON input
            data = json.loads(input_str)

            custom_id = data.get("custom_id", "")
            content = data.get("content", "")
            metadata = data.get("metadata", {})

            if not custom_id:
                return "error: custom_id is required for updates"

            # Call async method
            import asyncio
            loop = asyncio.get_event_loop()
            success = loop.run_until_complete(
                self.memory_manager.update_memory(
                    custom_id=custom_id,
                    content=content,
                    metadata=metadata,
                    container_tag=self.container_tag
                )
            )

            if success:
                self.memories_updated += 1
                return f"success: Updated memory '{custom_id}'"
            else:
                return "error: Failed to update memory"

        except Exception as e:
            _logger.error(f"[LANGCHAIN-AGENT] Error in update_memory tool: {e}")
            return f"error: {str(e)}"

    def _search_memory_tool(self, query: str) -> str:
        """Tool function for searching existing memories."""
        try:
            # Call async method
            import asyncio
            loop = asyncio.get_event_loop()
            memories = loop.run_until_complete(
                self.memory_manager.search_memory(
                    query=query,
                    container_tag=self.container_tag,
                    limit=5
                )
            )

            # Format results for agent
            results = []
            for memory in memories:
                results.append({
                    "custom_id": memory.get("custom_id", ""),
                    "content": memory.get("content", "")[:100],  # Truncate
                    "type": memory.get("type", "unknown"),
                    "version": memory.get("version", 1)
                })

            return json.dumps(results, indent=2)

        except Exception as e:
            _logger.error(f"[LANGCHAIN-AGENT] Error in search_memory tool: {e}")
            return f"error: {str(e)}"

    async def process_iteration(
        self,
        query: str,
        code_chunks: list[dict[str, Any]],
        judge_results: dict[int, dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Process a single exploration iteration to create/update memories.

        This is called on EVERY iteration of exploration, regardless of whether
        complete understanding has been achieved. The agent decides what to store.

        Args:
            query: The current exploration query
            code_chunks: Code chunks found in this iteration
            judge_results: Judge evaluation results (relevance, entry points, etc.)

        Returns:
            Result dict with success status, memories_created, memories_updated
        """
        if not self.memory_manager.enabled:
            self.skipped_iterations += 1
            return {
                "success": False,
                "reason": "Memory manager disabled",
                "memories_created": 0,
                "memories_updated": 0
            }

        try:
            self.iterations_processed += 1

            # Prepare summaries for agent
            code_chunks_summary = self._summarize_code_chunks(code_chunks)
            judge_summary = self._summarize_judge_results(judge_results)

            # Track stats before running agent
            created_before = self.memories_created
            updated_before = self.memories_updated

            # Run agent
            result = self.agent_executor.invoke({
                "query": query,
                "code_chunks_summary": code_chunks_summary,
                "judge_summary": judge_summary,
                "agent_scratchpad": ""
            })

            # Calculate what changed
            created_this_iteration = self.memories_created - created_before
            updated_this_iteration = self.memories_updated - updated_before

            _logger.info(
                f"[LANGCHAIN-AGENT] Iteration {self.iterations_processed}: "
                f"created {created_this_iteration}, updated {updated_this_iteration}"
            )

            return {
                "success": True,
                "memories_created": created_this_iteration,
                "memories_updated": updated_this_iteration,
                "agent_output": result.get("output", "")
            }

        except Exception as e:
            _logger.error(f"[LANGCHAIN-AGENT] Error processing iteration: {e}")
            return {
                "success": False,
                "reason": f"Error: {str(e)}",
                "memories_created": 0,
                "memories_updated": 0
            }

    def _summarize_code_chunks(self, code_chunks: list[dict[str, Any]]) -> str:
        """Summarize code chunks for agent prompt."""
        if not code_chunks:
            return "No code chunks found"

        summary_lines = []
        for i, chunk in enumerate(code_chunks[:5], 1):  # Limit to 5
            file_path = chunk.get("file_path", "unknown")
            content = chunk.get("content", "")[:100]  # Truncate
            summary_lines.append(f"{i}. {file_path}: {content}...")

        if len(code_chunks) > 5:
            summary_lines.append(f"... and {len(code_chunks) - 5} more chunks")

        return "\n".join(summary_lines)

    def _summarize_judge_results(self, judge_results: dict[int, dict[str, Any]] | None) -> str:
        """Summarize judge evaluation results for agent prompt."""
        if not judge_results:
            return "No judge evaluation available"

        relevant_count = sum(1 for r in judge_results.values() if r.get("is_relevant", False))
        entry_point_count = sum(1 for r in judge_results.values() if r.get("entry_point", False))

        summary_lines = [
            f"Total chunks evaluated: {len(judge_results)}",
            f"Relevant chunks: {relevant_count}",
            f"Entry points found: {entry_point_count}"
        ]

        # Add key findings
        for idx, result in list(judge_results.items())[:3]:  # Top 3
            if result.get("is_relevant"):
                why = result.get("why", "")[:100]
                entry = " [ENTRY POINT]" if result.get("entry_point") else ""
                summary_lines.append(f"- Chunk {idx}: {why}{entry}")

        return "\n".join(summary_lines)

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "enabled": self.memory_manager.enabled,
            "model": self.model,
            "container_tag": self.container_tag,
            "iterations_processed": self.iterations_processed,
            "memories_created": self.memories_created,
            "memories_updated": self.memories_updated,
            "skipped_iterations": self.skipped_iterations
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.memories_created = 0
        self.memories_updated = 0
        self.iterations_processed = 0
        self.skipped_iterations = 0
