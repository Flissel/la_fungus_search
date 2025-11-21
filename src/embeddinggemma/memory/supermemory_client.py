"""
Supermemory client wrapper for LLM judge persistent memory.

Provides async interface for storing and retrieving exploration insights,
enabling the judge to build cumulative knowledge across simulation steps.
"""
from __future__ import annotations

import logging
import os
from typing import Any

_logger = logging.getLogger(__name__)

# Supported insight types for code exploration
INSIGHT_TYPES = [
    "entry_point",     # Main functions, API routes, CLI entrypoints
    "pattern",         # Architectural patterns (async/await, DI, factories)
    "dependency",      # Critical imports and external dependencies
    "bug",            # Error patterns, suspicious code
    "security",       # Authentication, authorization, vulnerabilities
    "room",           # Code area/module comprehensive insight
    "relationship",   # Cross-room connections (imports, calls, dependencies)
    "cluster",        # Group of related rooms (subsystems)
    "discovery"       # General exploration finding
]


class SupermemoryManager:
    """
    Manages persistent memory for LLM judge using Supermemory API.

    The manager stores insights discovered during exploration and retrieves
    relevant context to inform future judge decisions.
    """

    def __init__(self, api_key: str | None = None, enabled: bool = True):
        """
        Initialize Supermemory manager.

        Args:
            api_key: Supermemory API key (defaults to SUPERMEMORY_API_KEY env var)
            enabled: Whether memory features are enabled (default: True)
        """
        self.api_key = api_key or os.getenv("SUPERMEMORY_API_KEY")
        self.base_url = os.getenv("SUPERMEMORY_BASE_URL", "https://api.supermemory.ai")
        self.enabled = enabled and bool(self.api_key)
        self.client = None

        # Stats tracking
        self.insights_stored = 0
        self.insights_retrieved = 0
        self.memory_queries = 0

        if self.enabled:
            try:
                from supermemory import AsyncSupermemory
                self.client = AsyncSupermemory(api_key=self.api_key, base_url=self.base_url)
                _logger.info(f"[MEMORY] Supermemory initialized successfully (base_url={self.base_url})")
            except ImportError:
                _logger.warning(
                    "[MEMORY] supermemory package not installed. "
                    "Install with: pip install supermemory"
                )
                self.enabled = False
            except Exception as e:
                _logger.error(f"[MEMORY] Failed to initialize Supermemory: {e}")
                self.enabled = False
        else:
            _logger.info("[MEMORY] Supermemory disabled (no API key)")

    async def add_insight(
        self,
        content: str,
        insight_type: str,
        container_tag: str,
        metadata: dict[str, Any] | None = None,
        confidence: float = 0.0
    ) -> bool:
        """
        Store an insight in Supermemory.

        Args:
            content: The insight text to store
            insight_type: Type of insight (entry_point, pattern, dependency, bug, security)
            container_tag: Container tag for isolation (e.g., run_id or exploration_goal)
            metadata: Additional metadata (file_path, phase, etc.)
            confidence: Confidence score (0.0-1.0)

        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            meta = metadata or {}
            meta.update({
                "type": insight_type,
                "confidence": confidence,
            })

            await self.client.memories.add(
                content=content,
                container_tags=[container_tag],
                metadata=meta
            )

            self.insights_stored += 1
            _logger.debug(
                f"[MEMORY] Stored insight: type={insight_type}, "
                f"container={container_tag}, confidence={confidence:.2f}"
            )
            return True

        except Exception as e:
            _logger.error(f"[MEMORY] Error storing insight: {e}")
            return False

    async def search_insights(
        self,
        query: str,
        container_tag: str,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search for relevant insights using semantic search.

        Args:
            query: Search query
            container_tag: Container tag to search within
            limit: Maximum number of results to return

        Returns:
            List of insight dictionaries with 'content' and 'metadata' keys
        """
        if not self.enabled or not self.client:
            return []

        try:
            self.memory_queries += 1

            results = await self.client.search.memories(
                q=query,
                container_tag=container_tag,
                limit=limit,
                threshold=0.6,
                rerank=True
            )

            if not results or not hasattr(results, 'results'):
                return []

            insights = []
            for r in results.results:
                insights.append({
                    "content": r.memory if hasattr(r, 'memory') else str(r),
                    "metadata": r.metadata if hasattr(r, 'metadata') else {}
                })

            self.insights_retrieved += len(insights)
            _logger.debug(
                f"[MEMORY] Retrieved {len(insights)} insights for query: {query[:50]}..."
            )
            return insights

        except Exception as e:
            _logger.error(f"[MEMORY] Error searching insights: {e}")
            return []

    async def get_context(
        self,
        query: str,
        container_tag: str,
        max_insights: int = 5
    ) -> str:
        """
        Get formatted context string from relevant past insights.

        Args:
            query: Current query to find relevant context for
            container_tag: Container tag to search within
            max_insights: Maximum number of insights to include

        Returns:
            Formatted context string to inject into judge prompt
        """
        insights = await self.search_insights(query, container_tag, max_insights)

        if not insights:
            return ""

        # Format insights into readable context
        lines = ["**RELEVANT PAST INSIGHTS:**"]
        for i, insight in enumerate(insights, 1):
            content = insight.get("content", "")
            meta = insight.get("metadata", {})
            insight_type = meta.get("type", "unknown")
            confidence = meta.get("confidence", 0.0)

            lines.append(
                f"{i}. [{insight_type.upper()}] (confidence: {confidence:.2f})\n"
                f"   {content}"
            )

        lines.append("")  # Empty line before main content
        return "\n".join(lines)

    async def add_bulk_insights(
        self,
        insights: list[dict[str, Any]],
        container_tag: str
    ) -> int:
        """
        Store multiple insights in bulk.

        Args:
            insights: List of insight dictionaries with 'content', 'type', and optional metadata
            container_tag: Container tag for all insights

        Returns:
            Number of successfully stored insights
        """
        stored = 0
        for insight in insights:
            content = insight.get("content", "")
            insight_type = insight.get("type", "discovery")
            metadata = insight.get("metadata", {})
            confidence = insight.get("confidence", 0.0)

            if content and await self.add_insight(
                content=content,
                insight_type=insight_type,
                container_tag=container_tag,
                metadata=metadata,
                confidence=confidence
            ):
                stored += 1

        return stored

    def get_stats(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "enabled": self.enabled,
            "insights_stored": self.insights_stored,
            "insights_retrieved": self.insights_retrieved,
            "memory_queries": self.memory_queries,
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.insights_stored = 0
        self.insights_retrieved = 0
        self.memory_queries = 0

    # ==================== Room-Specific Methods ====================

    async def add_room_insight(
        self,
        room_id: str,
        purpose: str,
        file_path: str,
        line_range: tuple[int, int] | list[int],
        exploration_status: str = "partial",
        patterns: list[str] | None = None,
        key_functions: list[str] | None = None,
        key_classes: list[str] | None = None,
        container_tag: str | None = None,
        confidence: float = 0.5,
        **extra_metadata
    ) -> bool:
        """
        Store comprehensive room (code area) insight.

        Args:
            room_id: Unique room identifier (e.g., 'server_py_main')
            purpose: What this code area does
            file_path: Path to the file
            line_range: Tuple/list of [start_line, end_line]
            exploration_status: 'entry_only', 'partial', or 'fully_explored'
            patterns: List of patterns found (e.g., ['async', 'WebSocket'])
            key_functions: Important functions in this area
            key_classes: Important classes in this area
            container_tag: Container tag for isolation
            confidence: Confidence score (0.0-1.0)
            **extra_metadata: Additional metadata fields

        Returns:
            True if stored successfully, False otherwise
        """
        content = f"Room '{room_id}': {purpose}"

        # Ensure line_range is a list
        if isinstance(line_range, tuple):
            line_range = list(line_range)

        metadata = {
            "room_id": room_id,
            "file_path": file_path,
            "line_range": line_range,
            "exploration_status": exploration_status,
            "patterns": patterns or [],
            "key_functions": key_functions or [],
            "key_classes": key_classes or [],
            **extra_metadata
        }

        return await self.add_insight(
            content=content,
            insight_type="room",
            container_tag=container_tag or "default",
            metadata=metadata,
            confidence=confidence
        )

    async def get_room_summary(
        self,
        room_id: str,
        container_tag: str
    ) -> dict[str, Any] | None:
        """
        Get comprehensive summary of a specific room.

        Args:
            room_id: Room identifier to retrieve
            container_tag: Container tag to search within

        Returns:
            Room summary dict or None if not found
        """
        insights = await self.search_insights(
            query=f"room:{room_id}",
            container_tag=container_tag,
            limit=5
        )

        if not insights:
            return None

        # Aggregate room insights
        summary = {
            "room_id": room_id,
            "purpose": insights[0].get("content", ""),
            "exploration_status": "unknown",
            "patterns": [],
            "key_functions": [],
            "key_classes": [],
            "relationships": [],
            "confidence": insights[0].get("confidence", 0.0)
        }

        for insight in insights:
            meta = insight.get("metadata", {})

            if meta.get("exploration_status"):
                summary["exploration_status"] = meta["exploration_status"]

            if meta.get("patterns"):
                summary["patterns"].extend(meta["patterns"])

            if meta.get("key_functions"):
                summary["key_functions"].extend(meta["key_functions"])

            if meta.get("key_classes"):
                summary["key_classes"].extend(meta["key_classes"])

        # Deduplicate lists
        summary["patterns"] = list(set(summary["patterns"]))
        summary["key_functions"] = list(set(summary["key_functions"]))
        summary["key_classes"] = list(set(summary["key_classes"]))

        return summary

    async def get_all_rooms(
        self,
        container_tag: str,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Get all discovered rooms in this exploration run.

        Args:
            container_tag: Container tag to search within
            limit: Maximum number of rooms to return

        Returns:
            List of room summary dicts
        """
        insights = await self.search_insights(
            query="type:room",
            container_tag=container_tag,
            limit=limit
        )

        rooms = []
        seen_room_ids = set()

        for insight in insights:
            meta = insight.get("metadata", {})
            room_id = meta.get("room_id", "unknown")

            # Deduplicate by room_id
            if room_id in seen_room_ids:
                continue
            seen_room_ids.add(room_id)

            rooms.append({
                "room_id": room_id,
                "file_path": meta.get("file_path", ""),
                "purpose": insight.get("content", ""),
                "status": meta.get("exploration_status", "unknown"),
                "confidence": meta.get("confidence", 0.0),
                "patterns": meta.get("patterns", []),
                "key_functions": meta.get("key_functions", []),
                "key_classes": meta.get("key_classes", [])
            })

        return rooms

    async def add_relationship_insight(
        self,
        from_room: str,
        to_room: str,
        relationship_type: str,
        content: str,
        container_tag: str,
        strength: str = "medium",
        confidence: float = 0.7,
        **extra_metadata
    ) -> bool:
        """
        Store relationship between two rooms.

        Args:
            from_room: Source room ID
            to_room: Target room ID
            relationship_type: 'imports', 'calls', 'depends_on', 'inherits'
            content: Description of the relationship
            container_tag: Container tag for isolation
            strength: 'weak', 'medium', 'strong'
            confidence: Confidence score (0.0-1.0)
            **extra_metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        metadata = {
            "from_room": from_room,
            "to_room": to_room,
            "relationship_type": relationship_type,
            "strength": strength,
            **extra_metadata
        }

        return await self.add_insight(
            content=content,
            insight_type="relationship",
            container_tag=container_tag,
            metadata=metadata,
            confidence=confidence
        )

    # ==================== Document Management Methods ====================

    async def add_document(
        self,
        title: str,
        content: str,
        doc_type: str = "room",
        container_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
        url: str | None = None
    ) -> bool:
        """
        Add a structured document to Supermemory using /add-document API.

        This is different from add_insight() - documents are structured knowledge
        units with titles, content, and rich metadata, ideal for storing complete
        code modules, rooms, or subsystems.

        Args:
            title: Document title (e.g., "FastAPI Server Module")
            content: Full document content (summary, analysis, code snippets)
            doc_type: Type of document (room, module, cluster, etc.)
            container_tag: Container tag for isolation (run_id)
            metadata: Additional metadata (patterns, functions, classes, etc.)
            url: Optional URL reference (file path or repo link)

        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            # Build metadata with type information
            meta = metadata or {}
            meta.update({
                "doc_type": doc_type,
                "container_tag": container_tag or "default",
                "title": title,  # Store title in metadata for retrieval
            })

            # Format content with title (Supermemory v3 API)
            # Include title as markdown header for better organization
            formatted_content = f"# {title}\n\n{content}"

            # Use Supermemory's memories.add endpoint (v3 API)
            # Create a custom_id for deduplication based on container + type + title
            custom_id = f"{container_tag or 'default'}_{doc_type}_{title}".replace(" ", "_")[:255]

            await self.client.memories.add(
                content=formatted_content,
                container_tags=[container_tag or "default"],  # Note: plural "tags"
                metadata=meta,
                custom_id=custom_id  # Enables updates instead of duplicates
            )

            self.insights_stored += 1  # Track as insight storage
            _logger.debug(
                f"[MEMORY] Added document: '{title}' (type={doc_type}, container={container_tag})"
            )
            return True

        except Exception as e:
            _logger.error(f"[MEMORY] Error adding document '{title}': {e}")
            return False

    async def search_documents(
        self,
        query: str,
        container_tag: str | None = None,
        doc_type: str | None = None,
        limit: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        filter_logic: str = "AND"
    ) -> list[dict[str, Any]]:
        """
        Search for documents by query with optional metadata filtering.

        Args:
            query: Search query
            container_tag: Filter by container tag
            doc_type: Filter by document type (room, module, cluster)
            limit: Maximum number of results
            metadata_filters: Optional dict of metadata key-value pairs to filter by
                Example: {"exploration_status": "fully_explored", "doc_type": "module"}
            filter_logic: "AND" (all must match) or "OR" (any must match)

        Returns:
            List of document dicts with title, content, metadata
        """
        if not self.enabled or not self.client:
            return []

        try:
            self.memory_queries += 1

            # Build search query with filters
            search_query = query
            if doc_type:
                search_query = f"type:{doc_type} {query}"

            # Use Supermemory's document search
            # Note: Use container_tags (plural) if filtering by container
            # Note: The Supermemory v3 API doesn't support threshold, rerank, or rewrite_query
            # parameters in search.documents(). These are available in other endpoints.
            search_kwargs = {
                "q": search_query,
                "limit": limit
            }
            if container_tag:
                search_kwargs["container_tags"] = [container_tag]  # Plural "tags"

            # Add metadata filters if supported by API
            # Note: Supermemory v3 API doesn't directly support metadata_filters in search,
            # so we'll filter results client-side
            results = await self.client.search.documents(**search_kwargs)

            if not results or not hasattr(results, 'results'):
                return []

            documents = []
            for r in results.results:
                doc = {
                    "title": r.title if hasattr(r, 'title') else "Untitled",
                    "content": r.content if hasattr(r, 'content') else str(r),
                    "url": r.url if hasattr(r, 'url') else "",
                    "metadata": r.metadata if hasattr(r, 'metadata') else {}
                }

                # Apply client-side metadata filtering if specified
                if metadata_filters:
                    matches = []
                    for key, value in metadata_filters.items():
                        metadata_value = doc["metadata"].get(key)
                        matches.append(metadata_value == value)

                    # Apply filter logic
                    if filter_logic == "AND":
                        if not all(matches):
                            continue  # Skip this document
                    else:  # OR logic
                        if not any(matches):
                            continue  # Skip this document

                documents.append(doc)

            self.insights_retrieved += len(documents)
            _logger.debug(
                f"[MEMORY] Retrieved {len(documents)} documents for query: {query[:50]}..."
            )
            return documents

        except Exception as e:
            _logger.error(f"[MEMORY] Error searching documents: {e}")
            return []

    async def get_document_by_title(
        self,
        title: str,
        container_tag: str
    ) -> dict[str, Any] | None:
        """
        Get a specific document by title.

        Args:
            title: Exact document title
            container_tag: Container tag to search within

        Returns:
            Document dict or None if not found
        """
        documents = await self.search_documents(
            query=f"title:{title}",
            container_tag=container_tag,
            limit=1
        )

        return documents[0] if documents else None

    # ==================== LangChain Memory Agent Methods ====================

    @staticmethod
    def generate_custom_id(type: str, file_path: str, identifier: str) -> str:
        """
        Generate deterministic custom_id for memory deduplication.

        The custom_id is used by Supermemory to identify existing memories for updates.
        Using the same custom_id will UPDATE the memory instead of creating a duplicate.

        Args:
            type: Memory type (entry_point, pattern, dependency, bug, security)
            file_path: File path where the discovery was made
            identifier: Unique identifier (function name, class name, etc.)

        Returns:
            Normalized custom_id string (max 255 chars)

        Example:
            >>> generate_custom_id("entry_point", "src/server.py", "main")
            'entry_point_src_server_py_main'
        """
        # Normalize path separators and dots for consistency
        normalized_path = file_path.replace("/", "_").replace("\\", "_").replace(".", "_")
        normalized_id = f"{type}_{normalized_path}_{identifier}"

        # Remove special characters that might cause issues
        normalized_id = normalized_id.replace(" ", "_").replace("-", "_")

        # Ensure max length 255 (Supermemory limit)
        return normalized_id[:255]

    async def add_memory(
        self,
        content: str,
        type: str,
        metadata: dict[str, Any],
        custom_id: str,
        container_tag: str
    ) -> bool:
        """
        Add or update memory in Supermemory (for LangChain agent).

        This method is used by the LangChain Memory Agent to create/update memories
        incrementally during exploration. It uses custom_id for deduplication -
        if a memory with the same custom_id exists, it will be updated instead of
        creating a duplicate.

        Args:
            content: Memory content (what was discovered)
            type: Memory type (entry_point, pattern, dependency, bug, security)
            metadata: Metadata dict (file_path, line, confidence, etc.)
            custom_id: Deterministic ID for deduplication (use generate_custom_id())
            container_tag: Container tag for isolation (run_id)

        Returns:
            True if stored successfully, False otherwise

        Example:
            >>> custom_id = SupermemoryManager.generate_custom_id(
            ...     "entry_point", "src/server.py", "main"
            ... )
            >>> await memory_manager.add_memory(
            ...     content="Main server entry point using FastAPI",
            ...     type="entry_point",
            ...     metadata={"file_path": "src/server.py", "line": 42, "confidence": 0.9},
            ...     custom_id=custom_id,
            ...     container_tag="run_abc123"
            ... )
        """
        if not self.enabled or not self.client:
            return False

        try:
            from datetime import datetime

            # Enhance metadata with memory-specific fields
            enhanced_metadata = {
                **metadata,
                "type": type,
                "custom_id": custom_id,
                "created_at": datetime.now().isoformat(),
                "version": 1  # Will be incremented on updates
            }

            # Add memory using Supermemory v3 API
            _logger.info(
                f"[MEMORY] API call: memories.add() | "
                f"container_tags={[container_tag]}, custom_id={custom_id[:50]}..."
            )

            await self.client.memories.add(
                content=content,
                container_tags=[container_tag],
                metadata=enhanced_metadata,
                custom_id=custom_id  # Enables updates instead of duplicates
            )

            self.insights_stored += 1
            _logger.info(
                f"[MEMORY] API response: Success | "
                f"type={type}, custom_id={custom_id[:50]}"
            )
            return True

        except Exception as e:
            _logger.error(f"[MEMORY] Error adding memory: {e}")
            return False

    async def update_memory(
        self,
        custom_id: str,
        content: str,
        metadata: dict[str, Any],
        container_tag: str
    ) -> bool:
        """
        Update existing memory by custom_id (for LangChain agent).

        This method updates an existing memory with new information. It uses the
        same custom_id as the original memory, which tells Supermemory to UPDATE
        instead of CREATE.

        Args:
            custom_id: Custom ID of memory to update
            content: Updated memory content
            metadata: Updated metadata (version will be incremented)
            container_tag: Container tag for isolation

        Returns:
            True if updated successfully, False otherwise

        Example:
            >>> await memory_manager.update_memory(
            ...     custom_id="entry_point_src_server_py_main",
            ...     content="Main server entry point using FastAPI with WebSocket support",
            ...     metadata={"file_path": "src/server.py", "line": 42, "confidence": 0.95},
            ...     container_tag="run_abc123"
            ... )
        """
        if not self.enabled or not self.client:
            return False

        try:
            from datetime import datetime

            # Extract type from custom_id (format: type_path_identifier)
            type_hint = custom_id.split("_")[0] if "_" in custom_id else "discovery"

            # Increment version in metadata
            current_version = metadata.get("version", 1)
            enhanced_metadata = {
                **metadata,
                "type": type_hint,
                "custom_id": custom_id,
                "updated_at": datetime.now().isoformat(),
                "version": current_version + 1
            }

            # Update using the SAME custom_id - Supermemory will replace the old memory
            _logger.info(
                f"[MEMORY] API call: memories.add() (UPDATE) | "
                f"container_tags={[container_tag]}, custom_id={custom_id[:50]}..."
            )

            await self.client.memories.add(
                content=content,
                container_tags=[container_tag],
                metadata=enhanced_metadata,
                custom_id=custom_id  # Same ID = UPDATE
            )

            _logger.info(
                f"[MEMORY] API response: Success (UPDATE) | "
                f"custom_id={custom_id[:50]}, version={current_version + 1}"
            )
            return True

        except Exception as e:
            _logger.error(f"[MEMORY] Error updating memory: {e}")
            return False

    async def search_memory(
        self,
        query: str,
        container_tag: str,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search for memories by query (for LangChain agent).

        This method is used by the LangChain Memory Agent to check if similar
        memories already exist before creating new ones.

        Args:
            query: Search query (e.g., "authentication module")
            container_tag: Container tag to search within
            limit: Maximum number of results

        Returns:
            List of memory dicts with content, metadata, custom_id

        Example:
            >>> memories = await memory_manager.search_memory(
            ...     query="authentication",
            ...     container_tag="run_abc123",
            ...     limit=5
            ... )
            >>> for memory in memories:
            ...     print(f"{memory['metadata']['custom_id']}: {memory['content'][:50]}...")
        """
        if not self.enabled or not self.client:
            return []

        try:
            self.memory_queries += 1

            _logger.info(
                f"[MEMORY] API call: search.memories() | "
                f"query={query[:50]}..., container_tag={container_tag}, limit={limit}"
            )

            # Search using Supermemory v3 API
            results = await self.client.search.memories(
                q=query,
                container_tag=container_tag,
                limit=limit,
                threshold=0.6,
                rerank=True
            )

            if not results or not hasattr(results, 'results'):
                _logger.info("[MEMORY] API response: No results returned")
                return []

            memories = []
            for r in results.results:
                meta = r.metadata if (hasattr(r, 'metadata') and r.metadata is not None) else {}
                memories.append({
                    "content": r.memory if hasattr(r, 'memory') else str(r),
                    "metadata": meta,
                    "custom_id": meta.get("custom_id", ""),
                    "type": meta.get("type", "unknown"),
                    "version": meta.get("version", 1)
                })

            self.insights_retrieved += len(memories)
            _logger.info(
                f"[MEMORY] API response: Success | Found {len(memories)} memories"
            )
            return memories

        except Exception as e:
            _logger.error(f"[MEMORY] Error searching memories: {e}")
            return []


# ============================================================================
# Synchronous Supermemory Manager for LangChain Tools
# ============================================================================


class SupermemoryManagerSync:
    """
    Synchronous wrapper for Supermemory API - designed for LangChain tools.

    This class uses the synchronous Supermemory client to avoid event loop
    conflicts when called from LangChain tool functions (which are sync but
    run inside an async FastAPI context).

    The async SupermemoryManager above uses AsyncSupermemory and is suitable
    for direct async/await usage. This sync version is specifically for
    LangChain ReAct agent tools.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize synchronous Supermemory manager.

        Args:
            api_key: Supermemory API key (falls back to env SUPERMEMORY_API_KEY)
        """
        import os
        from supermemory import Supermemory  # Sync client!

        self.api_key = api_key or os.getenv("SUPERMEMORY_API_KEY")
        self.base_url = os.getenv("SUPERMEMORY_BASE_URL", "https://api.supermemory.ai")
        self.enabled = bool(self.api_key)

        if not self.api_key:
            raise ValueError(
                "SUPERMEMORY_API_KEY not found in environment or constructor"
            )

        _logger.info(f"[MEMORY-SYNC] Initializing synchronous Supermemory client (base_url={self.base_url})")
        self.client = Supermemory(api_key=self.api_key, base_url=self.base_url)

    def add_memory(
        self,
        content: str,
        type: str | None = None,  # Note: SDK doesn't use this, kept for compatibility
        metadata: dict | None = None,
        custom_id: str | None = None,
        container_tag: str | None = None,
    ) -> bool:
        """
        Add a memory to Supermemory (synchronous).

        Args:
            content: Memory content text
            type: Memory type (DEPRECATED - SDK doesn't support this parameter)
            metadata: Optional metadata dict
            custom_id: Optional custom ID for deduplication
            container_tag: Optional container tag for isolation

        Returns:
            True if successful, False otherwise
        """
        if not content or not content.strip():
            _logger.warning("[MEMORY-SYNC] Empty content, skipping add")
            return False

        try:
            # Log the API call
            _logger.info(
                f"[MEMORY-SYNC] API call: memories.add() | "
                f"container_tags={[container_tag] if container_tag else []}, "
                f"custom_id={custom_id[:50] if custom_id else 'None'}..., "
                f"content_len={len(content)}"
            )

            # Direct sync call - no await or asyncio!
            # NOTE: SDK doesn't support 'type' parameter, removed it
            response = self.client.memories.add(
                content=content,
                metadata=metadata or {},
                custom_id=custom_id,
                container_tags=[container_tag] if container_tag else [],  # Plural list!
            )

            _logger.info(
                f"[MEMORY-SYNC] API response: Success | "
                f"id={response.id if hasattr(response, 'id') else 'N/A'}"
            )
            return True

        except Exception as e:
            _logger.error(f"[MEMORY-SYNC] Error adding memory: {e}", exc_info=True)
            return False

    def update_memory(
        self,
        custom_id: str,
        content: str,
        metadata: dict | None = None,
        container_tag: str | None = None,
    ) -> bool:
        """
        Update an existing memory by custom_id (synchronous).

        Uses the same add() API with the same custom_id to trigger an update.

        Args:
            custom_id: Custom ID of memory to update
            content: New content
            metadata: New metadata
            container_tag: Container tag

        Returns:
            True if successful, False otherwise
        """
        if not custom_id:
            _logger.warning("[MEMORY-SYNC] No custom_id provided for update")
            return False

        try:
            _logger.info(
                f"[MEMORY-SYNC] API call: memories.add() (update) | "
                f"custom_id={custom_id[:50]}..., content_len={len(content)}"
            )

            # Same API call with same custom_id = update
            # NOTE: SDK doesn't support 'type' parameter, removed it
            response = self.client.memories.add(
                content=content,
                metadata=metadata or {},
                custom_id=custom_id,
                container_tags=[container_tag] if container_tag else [],
            )

            _logger.info(
                f"[MEMORY-SYNC] API response: Updated | "
                f"id={response.id if hasattr(response, 'id') else 'N/A'}"
            )
            return True

        except Exception as e:
            _logger.error(f"[MEMORY-SYNC] Error updating memory: {e}", exc_info=True)
            return False

    def search_memory(
        self,
        query: str,
        container_tag: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Search memories (synchronous).

        Args:
            query: Search query
            container_tag: Optional container tag filter
            limit: Max results

        Returns:
            List of memory dicts with content, metadata, id
        """
        if not query or not query.strip():
            _logger.warning("[MEMORY-SYNC] Empty query, returning no results")
            return []

        try:
            _logger.info(
                f"[MEMORY-SYNC] API call: search.memories() | "
                f"q='{query[:100]}', container_tag={container_tag}, limit={limit}"
            )

            # Direct sync call - note: search.memories uses singular container_tag!
            results = self.client.search.memories(
                q=query,
                container_tag=container_tag,  # SINGULAR string, not list!
                limit=limit,
                threshold=0.6,
                rerank=True,
            )

            memories = []
            if results and hasattr(results, "memories"):
                for r in results.memories:
                    # Safe metadata handling
                    meta = (
                        r.metadata
                        if (hasattr(r, "metadata") and r.metadata is not None)
                        else {}
                    )
                    memories.append(
                        {
                            "content": r.content if hasattr(r, "content") else "",
                            "metadata": meta,
                            "id": r.id if hasattr(r, "id") else None,
                            "custom_id": r.custom_id if hasattr(r, "custom_id") else None,
                        }
                    )

            _logger.info(f"[MEMORY-SYNC] Search returned {len(memories)} results")
            return memories

        except Exception as e:
            _logger.error(f"[MEMORY-SYNC] Error searching memories: {e}", exc_info=True)
            return []
