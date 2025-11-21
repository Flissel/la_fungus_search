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
                    "Search existing memories to check for duplicates or find related findings. "
                    "Input: SPECIFIC search query string (e.g., 'server initialization', 'FastAPI routes', 'database connection'). "
                    "IMPORTANT: Be specific! Use file paths, function names, or technical terms, NOT generic queries. "
                    "Returns: JSON list of matching memories with custom_id, content, type, version."
                )
            )
        ]

    def _build_prompt(self) -> PromptTemplate:
        """Build the ReAct prompt template for the agent."""
        template = """You are the Memory Agent responsible for building architectural knowledge about codebases.

IMPORTANT: ALWAYS start by checking foundational knowledge before analyzing new discoveries.

FOUNDATIONAL KNOWLEDGE (check first):
- search_memory("codebase_module_tree") → Module structure map
- search_memory("codebase_entry_points") → Main executable files
- search_memory("module overview [area]") → Specific module details

CURRENT EXPLORATION TASK:
Query: {query}

CURRENT ITERATION RESULTS:
Code Chunks Found: {code_chunks_summary}
Judge Evaluation: {judge_summary}

MEMORY TYPES:
- entry_point: Main functions, API routes, CLI entrypoints, server initialization
- pattern: Architectural patterns (async/await, DI, factories, singleton)
- dependency: Critical imports and external dependencies
- module_architecture: High-level module organization and relationships
- bug: Error patterns, suspicious code
- security: Authentication, authorization, vulnerabilities

YOUR WORKFLOW:
1. FIRST: Search for foundational knowledge ("codebase_module_tree", "codebase_entry_points")
2. CONTEXT: Understand where current findings fit in the module structure
3. SEARCH: Look for existing memories related to current discoveries
4. DECIDE: Create NEW memory OR update EXISTING memory
5. PRIORITIZE: Entry points and architectural insights over trivial findings

CREATING ARCHITECTURAL MEMORIES:
- Reference module structure from foundational knowledge
- Connect findings to known modules/entry points
- Use clear custom_ids like "entry_point_server_initialization" or "module_agents_architecture"
- Include context: "This is part of the [module_name] module which handles [purpose]"

GUIDELINES:
- Always retrieve foundational knowledge first to understand codebase structure
- Build on existing architectural understanding, don't create isolated memories
- Update memories when you find related information
- Skip trivial findings (simple imports, comments, basic utility functions)
- Focus on architectural significance (how does this fit in the system?)

SEARCH QUERY RULES (CRITICAL):
- ❌ NEVER use generic queries like: "Classify the code", "modules", "architecture"
- ✅ ALWAYS use specific queries with concrete references:
  * File paths: "server.py initialization", "agents/langchain_memory_agent.py"
  * Function names: "start_exploration function", "add_memory implementation"
  * Technical terms: "FastAPI WebSocket routes", "Qdrant vector database setup"
  * Specific patterns: "async/await in realtime server", "LangChain agent tools"
- If you don't know what to search for, use the judge's insights or code chunk file paths as query terms
- Example: Instead of "Classify modules" → "src/embeddinggemma/realtime/server.py WebSocket handling"

You have access to the following tools:

{tools}

Tool Names: {tool_names}

Use the following format:

Thought: What should I do with these discoveries?
Action: tool_name
Action Input: tool input
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now know what to do
Final Answer: Summary of memories created/updated

Begin! Remember:
1. ALWAYS check foundational knowledge first with search_memory("codebase_module_tree")
2. Use SPECIFIC search queries with file paths, function names, or technical terms from the code chunks
3. NEVER use generic queries like "Classify the code" or "modules"

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

            _logger.info(
                f"[LANGCHAIN-AGENT] Tool called: add_memory | "
                f"type={type_}, file_path={file_path}, identifier={identifier}"
            )

            # Generate custom_id for deduplication
            from src.embeddinggemma.memory.supermemory_client import SupermemoryManager
            custom_id = SupermemoryManager.generate_custom_id(type_, file_path, identifier)

            # Add metadata fields
            metadata.update({
                "file_path": file_path,
                "identifier": identifier
            })

            # Direct sync call - no asyncio needed!
            success = self.memory_manager.add_memory(
                content=content,
                type=type_,
                metadata=metadata,
                custom_id=custom_id,
                container_tag=self.container_tag
            )

            if success:
                self.memories_created += 1
                result = f"success: Created memory '{custom_id}'"
                _logger.info(f"[LANGCHAIN-AGENT] Tool result: {result}")
                return result
            else:
                result = "error: Failed to create memory"
                _logger.warning(f"[LANGCHAIN-AGENT] Tool result: {result}")
                return result

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

            _logger.info(
                f"[LANGCHAIN-AGENT] Tool called: update_memory | "
                f"custom_id={custom_id[:50] if custom_id else 'MISSING'}"
            )

            if not custom_id:
                return "error: custom_id is required for updates"

            # Direct sync call - no asyncio needed!
            success = self.memory_manager.update_memory(
                custom_id=custom_id,
                content=content,
                metadata=metadata,
                container_tag=self.container_tag
            )

            if success:
                self.memories_updated += 1
                result = f"success: Updated memory '{custom_id}'"
                _logger.info(f"[LANGCHAIN-AGENT] Tool result: {result}")
                return result
            else:
                result = "error: Failed to update memory"
                _logger.warning(f"[LANGCHAIN-AGENT] Tool result: {result}")
                return result

        except Exception as e:
            _logger.error(f"[LANGCHAIN-AGENT] Error in update_memory tool: {e}")
            return f"error: {str(e)}"

    def _search_memory_tool(self, query: str) -> str:
        """Tool function for searching existing memories."""
        try:
            _logger.info(
                f"[LANGCHAIN-AGENT] Tool called: search_memory | "
                f"query={query[:50]}..."
            )

            # Direct sync call - no asyncio needed!
            memories = self.memory_manager.search_memory(
                query=query,
                container_tag=self.container_tag,
                limit=5
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

            result = json.dumps(results, indent=2)
            _logger.info(f"[LANGCHAIN-AGENT] Tool result: Found {len(results)} memories")
            return result

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
        """Summarize code chunks for agent prompt with specific technical details."""
        if not code_chunks:
            return "No code chunks found"

        summary_lines = []
        for i, chunk in enumerate(code_chunks[:5], 1):  # Limit to 5
            file_path = chunk.get("file_path", "unknown")
            content = chunk.get("content", "")

            # Extract function/class names from code for specific search context
            func_class_hints = []
            for line in content.split('\n')[:10]:  # First 10 lines
                line_stripped = line.strip()
                if line_stripped.startswith('def '):
                    func_name = line_stripped.split('(')[0].replace('def ', '').strip()
                    func_class_hints.append(f"def {func_name}")
                elif line_stripped.startswith('class '):
                    class_name = line_stripped.split('(')[0].split(':')[0].replace('class ', '').strip()
                    func_class_hints.append(f"class {class_name}")
                elif line_stripped.startswith('async def '):
                    func_name = line_stripped.split('(')[0].replace('async def ', '').strip()
                    func_class_hints.append(f"async def {func_name}")

            hints_str = ", ".join(func_class_hints[:3]) if func_class_hints else "code block"
            summary_lines.append(f"{i}. {file_path} ({hints_str})")

        if len(code_chunks) > 5:
            summary_lines.append(f"... and {len(code_chunks) - 5} more chunks")

        summary_lines.append("\n💡 TIP: Use these file paths and function names in your search queries!")

        return "\n".join(summary_lines)

    def _summarize_judge_results(self, judge_results: dict[int, dict[str, Any]] | None) -> str:
        """Summarize judge evaluation results with specific technical insights."""
        if not judge_results:
            return "No judge evaluation available"

        relevant_count = sum(1 for r in judge_results.values() if r.get("is_relevant", False))
        entry_point_count = sum(1 for r in judge_results.values() if r.get("entry_point", False))

        summary_lines = [
            f"Total chunks evaluated: {len(judge_results)}",
            f"Relevant chunks: {relevant_count}",
            f"Entry points found: {entry_point_count}",
            ""
        ]

        # Add key findings with MORE context
        summary_lines.append("Key Insights (use these terms in your searches):")
        for idx, result in list(judge_results.items())[:3]:  # Top 3
            if result.get("is_relevant"):
                why = result.get("why", "")[:200]  # More context
                entry = " [ENTRY POINT]" if result.get("entry_point") else ""
                summary_lines.append(f"- {why}{entry}")

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
