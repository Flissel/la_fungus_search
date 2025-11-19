"""
Memory Manager Agent - Decides what to ingest into Supermemory.

This agent is decoupled from the judge and focuses exclusively on:
1. Analyzing search results and code chunks
2. Deciding whether to ingest as a document or search more
3. Structuring knowledge into documents (rooms, modules, subsystems)
4. Managing the knowledge graph in Supermemory
"""
from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger(__name__)


class MemoryManagerAgent:
    """
    LLM-powered agent that decides what to ingest into Supermemory.

    Decoupled from the judge - focuses on memory completeness and organization.
    """

    def __init__(self, llm_client=None, memory_manager=None, model: str | None = None, min_chunks_for_ingest: int | None = None):
        """
        Initialize Memory Manager Agent.

        Args:
            llm_client: LLM client for making ingestion decisions (OpenAI, Ollama, etc.)
            memory_manager: SupermemoryManager instance for storing documents
            model: LLM model to use (defaults to gpt-4o-mini, or from MEMORY_AGENT_MODEL env var)
            min_chunks_for_ingest: Minimum chunks required for ingestion (defaults to 5, or from MEMORY_AGENT_MIN_CHUNKS env var)
        """
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.enabled = bool(memory_manager and memory_manager.enabled)

        # Configure LLM model and ingestion threshold
        import os
        self.model = model or os.getenv("MEMORY_AGENT_MODEL", "gpt-4o-mini")
        self.min_chunks_for_ingest = min_chunks_for_ingest or int(os.getenv("MEMORY_AGENT_MIN_CHUNKS", "5"))

        # Stats
        self.decisions_made = 0
        self.documents_ingested = 0
        self.search_more_decisions = 0

        if self.enabled:
            _logger.info(f"[MEMORY-AGENT] Memory Manager Agent initialized (model: {self.model}, min_chunks: {self.min_chunks_for_ingest})")
        else:
            _logger.info("[MEMORY-AGENT] Disabled (no memory manager)")

    async def analyze_and_decide(
        self,
        query: str,
        code_chunks: list[dict[str, Any]],
        judge_results: dict[int, dict] | None = None,
        container_tag: str = "default",
        conversation_history: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Analyze code chunks and decide: ingest as document OR search more.

        Args:
            query: Current exploration query
            code_chunks: List of code chunk dicts with content, file_path, etc.
            judge_results: Optional judge evaluation results
            container_tag: Container tag for memory isolation (run_id)
            conversation_history: Full conversation history from this run
                Format: [{"step": 1, "query": "...", "discoveries": "...", "decision": "..."}, ...]

        Returns:
            Decision dict with:
            - action: "ingest" | "search_more" | "skip"
            - reason: Explanation for the decision
            - documents: List of documents to ingest (if action="ingest")
            - suggested_queries: Follow-up queries (if action="search_more")
        """
        if not self.enabled or not code_chunks:
            return {"action": "skip", "reason": "Memory agent disabled or no chunks"}

        self.decisions_made += 1

        # Build prompt for Memory Manager Agent with conversation context
        prompt = self._build_decision_prompt(
            query,
            code_chunks,
            judge_results,
            conversation_history
        )

        try:
            # Call LLM to make ingestion decision
            response = await self._call_llm(prompt)
            decision = self._parse_decision(response)

            # If decision is to ingest, create documents
            if decision.get("action") == "ingest" and decision.get("documents"):
                ingested = await self._ingest_documents(
                    decision["documents"],
                    container_tag
                )
                decision["ingested_count"] = ingested
                self.documents_ingested += ingested
                _logger.info(
                    f"[MEMORY-AGENT] Ingested {ingested} documents for query: {query[:50]}..."
                )
            elif decision.get("action") == "search_more":
                self.search_more_decisions += 1
                _logger.debug(
                    f"[MEMORY-AGENT] Decided to search more: {decision.get('reason', 'N/A')}"
                )

            return decision

        except Exception as e:
            _logger.error(f"[MEMORY-AGENT] Error making decision: {e}")
            return {"action": "skip", "reason": f"Error: {str(e)}"}

    def _build_decision_prompt(
        self,
        query: str,
        code_chunks: list[dict[str, Any]],
        judge_results: dict[int, dict] | None = None,
        conversation_history: list[dict[str, Any]] | None = None
    ) -> str:
        """Build prompt for Memory Manager Agent to make ingestion decision."""

        # Summarize code chunks
        chunks_summary = []
        for i, chunk in enumerate(code_chunks[:10], 1):  # Limit to 10 for prompt
            file_path = chunk.get('file_path', chunk.get('path', 'unknown'))
            content_preview = (chunk.get('content', '')[:200] + '...'
                             if len(chunk.get('content', '')) > 200
                             else chunk.get('content', ''))
            chunks_summary.append(
                f"{i}. {file_path}\n{content_preview}\n"
            )

        chunks_text = "\n".join(chunks_summary)

        # Include judge results if available
        judge_context = ""
        if judge_results:
            relevant_count = sum(1 for r in judge_results.values() if r.get('is_relevant'))
            entry_points = [
                r.get('why', '') for r in judge_results.values()
                if r.get('entry_point')
            ]
            judge_context = f"""
**JUDGE EVALUATION RESULTS:**
- Total chunks evaluated: {len(judge_results)}
- Relevant chunks: {relevant_count}
- Entry points found: {len(entry_points)}
{('- Entry points: ' + ', '.join(entry_points[:3])) if entry_points else ''}
"""

        # Include conversation history for context
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-10:]  # Last 10 steps
            history_lines = []
            for i, step in enumerate(recent_history, 1):
                step_num = step.get('step', i)
                step_query = step.get('query', 'N/A')
                discoveries = step.get('discoveries', '')
                decision = step.get('decision', '')

                history_lines.append(
                    f"Step {step_num}: Query: {step_query[:80]}..."
                    f"{(' | Discoveries: ' + discoveries[:100]) if discoveries else ''}"
                    f"{(' | Decision: ' + decision) if decision else ''}"
                )

            conversation_context = f"""
**CONVERSATION HISTORY** (last {len(recent_history)} steps):
{chr(10).join(history_lines)}

**WHAT WE'VE LEARNED SO FAR:**
- Total exploration steps: {len(conversation_history)}
- Recent queries explored: {len(recent_history)}
- Progressive understanding: Building on previous discoveries

Use this history to:
1. Avoid redundant ingestion (check if we already stored similar info)
2. Understand exploration trajectory (where we've been, what we're looking for)
3. Make better decisions (ingest when we have cumulative complete understanding)
"""

        return f"""You are a Memory Manager Agent responsible for deciding what code knowledge to ingest into persistent memory.

**CURRENT EXPLORATION QUERY:** {query}
{conversation_context}

**CODE CHUNKS RETRIEVED ({len(code_chunks)} total):**
{chunks_text}

{judge_context}

**YOUR TASK:**
Analyze the code chunks and decide:

1. **INGEST** - If the chunks contain complete, valuable knowledge worth storing as documents:
   - Room/module is fully understood
   - Entry points are identified
   - Relationships to other modules are clear
   - Enough context to be useful in future searches

2. **SEARCH_MORE** - If more exploration is needed:
   - Chunks are incomplete or fragmented
   - Missing critical context (function implementations, dependencies)
   - Only have partial understanding of the module
   - Need to explore related files/classes

3. **SKIP** - If chunks are not relevant or already known

**OUTPUT FORMAT (JSON only):**
{{
  "action": "ingest" | "search_more" | "skip",
  "reason": "Clear explanation for your decision",
  "confidence": 0.0-1.0,
  "documents": [  // Only if action="ingest"
    {{
      "title": "Descriptive document title (e.g., 'FastAPI Server - Main Entry Point')",
      "content": "Complete summary of what this code does, its purpose, and key components",
      "type": "room" | "module" | "cluster" | "relationship",
      "metadata": {{
        "file_path": "path/to/file.py",
        "exploration_status": "fully_explored" | "partial",
        "patterns": ["async/await", "OOP", ...],
        "key_functions": ["function1", "function2"],
        "key_classes": ["Class1", "Class2"],
        "dependencies": ["module1", "module2"]
      }}
    }}
  ],
  "suggested_queries": [  // Only if action="search_more"
    "Follow-up query 1",
    "Follow-up query 2"
  ]
}}

**DECISION CRITERIA:**
- INGEST if: {self.min_chunks_for_ingest}+ chunks from same file, clear entry points, complete understanding
- SEARCH_MORE if: <{max(self.min_chunks_for_ingest - 2, 1)} chunks, fragmented info, missing critical context
- SKIP if: Not relevant to query, trivial code, already stored

Respond with JSON only.
"""

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM to make ingestion decision."""
        if not self.llm_client:
            raise RuntimeError("No LLM client configured for Memory Manager Agent")

        # Use the same LLM client as the judge
        # This could be OpenAI, Ollama, or any other provider
        try:
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                # OpenAI-style client
                response = await self.llm_client.chat.completions.create(
                    model=self.model,  # Use configurable model
                    messages=[
                        {"role": "system", "content": "You are a Memory Manager Agent. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            else:
                # Fallback for other clients
                raise RuntimeError("Unsupported LLM client type")

        except Exception as e:
            _logger.error(f"[MEMORY-AGENT] LLM call failed: {e}")
            raise

    def _parse_decision(self, response: str) -> dict[str, Any]:
        """Parse LLM response into decision dict."""
        try:
            decision = json.loads(response)

            # Validate required fields
            if "action" not in decision:
                decision["action"] = "skip"
            if "reason" not in decision:
                decision["reason"] = "No reason provided"

            # Ensure action is valid
            if decision["action"] not in ["ingest", "search_more", "skip"]:
                _logger.warning(
                    f"[MEMORY-AGENT] Invalid action: {decision['action']}, defaulting to skip"
                )
                decision["action"] = "skip"

            return decision

        except json.JSONDecodeError as e:
            _logger.error(f"[MEMORY-AGENT] Failed to parse LLM response: {e}")
            return {
                "action": "skip",
                "reason": f"Parse error: {str(e)}",
                "confidence": 0.0
            }

    async def _ingest_documents(
        self,
        documents: list[dict[str, Any]],
        container_tag: str
    ) -> int:
        """
        Ingest documents into Supermemory using /add-document API.

        Checks for existing documents before ingestion to prevent duplicates.

        Args:
            documents: List of document dicts with title, content, metadata
            container_tag: Container tag for isolation

        Returns:
            Number of successfully ingested documents
        """
        if not self.memory_manager:
            return 0

        ingested = 0
        for doc in documents:
            try:
                title = doc.get("title", "Untitled")
                doc_type = doc.get("type", "room")

                # Check for existing document to prevent duplicates
                # Search using title as query within the container
                existing_docs = await self.memory_manager.search_documents(
                    query=title,
                    container_tag=container_tag,
                    doc_type=doc_type,
                    limit=5
                )

                # Check if exact title match exists
                already_exists = any(
                    existing.get("title", "").strip() == title.strip()
                    for existing in existing_docs
                )

                if already_exists:
                    _logger.debug(
                        f"[MEMORY-AGENT] Document already exists, skipping: {title}"
                    )
                    continue

                # Use add_document method (to be implemented in SupermemoryManager)
                success = await self.memory_manager.add_document(
                    title=title,
                    content=doc.get("content", ""),
                    doc_type=doc_type,
                    container_tag=container_tag,
                    metadata=doc.get("metadata", {})
                )

                if success:
                    ingested += 1
                    _logger.debug(
                        f"[MEMORY-AGENT] Ingested document: {title}"
                    )

            except Exception as e:
                _logger.error(
                    f"[MEMORY-AGENT] Failed to ingest document '{doc.get('title')}': {e}"
                )

        return ingested

    def get_stats(self) -> dict[str, Any]:
        """Get Memory Manager Agent statistics."""
        return {
            "enabled": self.enabled,
            "decisions_made": self.decisions_made,
            "documents_ingested": self.documents_ingested,
            "search_more_decisions": self.search_more_decisions,
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.decisions_made = 0
        self.documents_ingested = 0
        self.search_more_decisions = 0
