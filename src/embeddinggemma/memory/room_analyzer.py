"""
Automatic room (code area) discovery and analysis.

Analyzes code chunks to identify and characterize "rooms" - cohesive code areas
that represent modules, files, or logical components of the codebase.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import logging

_logger = logging.getLogger(__name__)


class RoomAnalyzer:
    """
    Analyzes code chunks to automatically identify and characterize rooms.

    A "room" is a cohesive code area (typically a file or module) with:
    - Unique identifier
    - Purpose/responsibility
    - Key functions and classes
    - Detected patterns
    - Exploration status
    """

    def __init__(self):
        """Initialize room analyzer."""
        self.rooms_discovered = {}  # room_id -> room_info
        self.room_visits = defaultdict(int)  # room_id -> visit_count

    def analyze_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Analyze chunks to extract room insights.

        Args:
            chunks: List of code chunk dictionaries with content, file_path, etc.

        Returns:
            List of room insights ready for storage in Supermemory
        """
        insights = []

        # Group chunks by file (each file is a "room")
        file_groups = defaultdict(list)
        for chunk in chunks:
            file_path = chunk.get('file_path', '') or chunk.get('path', '')
            if file_path:
                file_groups[file_path].append(chunk)

        # Analyze each file as a potential room
        for file_path, file_chunks in file_groups.items():
            room_insight = self._analyze_room(file_path, file_chunks)
            if room_insight:
                insights.append(room_insight)
                # Cache discovered room
                room_id = room_insight["metadata"]["room_id"]
                self.rooms_discovered[room_id] = room_insight

        return insights

    def _analyze_room(
        self,
        file_path: str,
        chunks: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """
        Analyze a single room (file).

        Args:
            file_path: Path to the file
            chunks: List of chunks from this file

        Returns:
            Room insight dict or None if not enough information
        """
        if not file_path or not chunks:
            return None

        # Generate unique room_id from file path
        room_id = self._generate_room_id(file_path)

        # Track visit
        self.room_visits[room_id] += 1

        # Extract patterns from code
        patterns = self._detect_patterns(chunks)

        # Extract key functions/classes
        key_functions = self._extract_key_functions(chunks)
        key_classes = self._extract_key_classes(chunks)

        # Determine exploration status based on coverage
        chunk_count = len(chunks)
        if chunk_count >= 5:
            status = "fully_explored"
            confidence = 0.9
        elif chunk_count >= 2:
            status = "partial"
            confidence = 0.6
        else:
            status = "entry_only"
            confidence = 0.4

        # Build room purpose from available information
        purpose = self._infer_purpose(file_path, chunks, key_classes, key_functions)

        # Calculate line range coverage
        line_range = self._calculate_line_range(chunks)

        # Detect imports (potential relationships)
        imports = self._extract_imports(chunks)

        return {
            "type": "room",
            "content": f"Room '{room_id}': {purpose}",
            "confidence": confidence,
            "metadata": {
                "room_id": room_id,
                "file_path": file_path,
                "line_range": line_range,
                "exploration_status": status,
                "patterns": patterns,
                "key_functions": key_functions,
                "key_classes": key_classes,
                "visit_count": self.room_visits[room_id],
                "chunk_count": chunk_count,
                "imports": imports[:10]  # Top 10 imports
            }
        }

    def _generate_room_id(self, file_path: str) -> str:
        """Generate unique room identifier from file path."""
        # Normalize path separators
        normalized = file_path.replace('\\', '/').replace('/', '_')
        # Remove extension
        if normalized.endswith('.py'):
            normalized = normalized[:-3]
        # Remove leading path components if too long
        parts = normalized.split('_')
        if len(parts) > 5:
            # Keep last 5 parts (most specific)
            normalized = '_'.join(parts[-5:])
        return normalized.lower()

    def _detect_patterns(self, chunks: list[dict[str, Any]]) -> list[str]:
        """
        Detect code patterns in chunks.

        Args:
            chunks: List of code chunks

        Returns:
            List of detected pattern names
        """
        patterns = set()

        for chunk in chunks:
            content = chunk.get('content', '')

            # Async/await pattern
            if 'async def' in content or 'await ' in content:
                patterns.add('async/await')

            # Object-oriented programming
            if 'class ' in content and '__init__' in content:
                patterns.add('OOP')

            # FastAPI framework
            if 'FastAPI' in content or '@app.' in content:
                patterns.add('FastAPI')

            # WebSocket
            if 'WebSocket' in content or 'websocket' in content:
                patterns.add('WebSocket')

            # Event loop
            if 'asyncio' in content:
                patterns.add('event-loop')

            # Dependency injection
            if 'def __init__' in content and len(re.findall(r'self\.\w+\s*=', content)) > 2:
                patterns.add('dependency-injection')

            # Type hints
            if '->' in content and ':' in content:
                patterns.add('type-hints')

            # Decorator pattern
            if '@' in content and 'def ' in content:
                patterns.add('decorators')

            # Context managers
            if 'with ' in content or '__enter__' in content:
                patterns.add('context-managers')

            # Generator pattern
            if 'yield ' in content:
                patterns.add('generators')

        return sorted(list(patterns))

    def _extract_key_functions(self, chunks: list[dict[str, Any]]) -> list[str]:
        """
        Extract important function names.

        Args:
            chunks: List of code chunks

        Returns:
            List of key function names
        """
        functions = set()

        for chunk in chunks:
            content = chunk.get('content', '')

            # Match function definitions
            # Matches: def function_name(
            matches = re.findall(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)

            # Prioritize public functions (not starting with _)
            public_funcs = [f for f in matches if not f.startswith('_')]
            if public_funcs:
                functions.update(public_funcs[:3])  # Top 3 public
            else:
                # If no public functions, include private
                functions.update(matches[:2])  # Top 2 private

        return sorted(list(functions))[:5]  # Max 5 key functions

    def _extract_key_classes(self, chunks: list[dict[str, Any]]) -> list[str]:
        """
        Extract important class names.

        Args:
            chunks: List of code chunks

        Returns:
            List of key class names
        """
        classes = set()

        for chunk in chunks:
            content = chunk.get('content', '')

            # Match class definitions
            # Matches: class ClassName( or class ClassName:
            matches = re.findall(r'class ([a-zA-Z_][a-zA-Z0-9_]*)', content)
            classes.update(matches)

        return sorted(list(classes))[:3]  # Max 3 key classes

    def _calculate_line_range(self, chunks: list[dict[str, Any]]) -> list[int]:
        """
        Calculate line range coverage from chunks.

        Args:
            chunks: List of code chunks

        Returns:
            [start_line, end_line]
        """
        line_ranges = []
        for chunk in chunks:
            if 'line_range' in chunk:
                lr = chunk['line_range']
                if isinstance(lr, (list, tuple)) and len(lr) == 2:
                    line_ranges.append(lr)

        if line_ranges:
            min_line = min(r[0] for r in line_ranges)
            max_line = max(r[1] for r in line_ranges)
            return [min_line, max_line]

        return [0, 0]

    def _extract_imports(self, chunks: list[dict[str, Any]]) -> list[str]:
        """
        Extract import statements to identify dependencies/relationships.

        Args:
            chunks: List of code chunks

        Returns:
            List of imported module names
        """
        imports = set()

        for chunk in chunks:
            content = chunk.get('content', '')

            # Match: import module
            import_matches = re.findall(r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content)
            imports.update(import_matches)

            # Match: from module import ...
            from_matches = re.findall(r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import', content)
            imports.update(from_matches)

        return sorted(list(imports))

    def _infer_purpose(
        self,
        file_path: str,
        chunks: list[dict[str, Any]],
        classes: list[str],
        functions: list[str]
    ) -> str:
        """
        Infer the purpose of this room.

        Args:
            file_path: Path to the file
            chunks: List of code chunks
            classes: Detected class names
            functions: Detected function names

        Returns:
            Inferred purpose string
        """
        # Extract file name from path
        file_name = file_path.split('/')[-1].replace('\\', '/').split('/')[-1]
        file_name = file_name.replace('.py', '')

        # Build purpose from available information
        if classes:
            return f"{', '.join(classes)} - {file_name} module"
        elif functions:
            return f"{', '.join(functions[:2])} - {file_name} functionality"
        else:
            # Use docstring if available
            for chunk in chunks:
                content = chunk.get('content', '')
                # Look for module docstring
                docstring_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
                if docstring_match:
                    first_line = docstring_match.group(1).strip().split('\n')[0]
                    if len(first_line) < 100:
                        return first_line

            return f"{file_name} module"

    def get_room_stats(self) -> dict[str, Any]:
        """Get statistics about discovered rooms."""
        total_rooms = len(self.rooms_discovered)
        total_visits = sum(self.room_visits.values())

        # Count by exploration status
        status_counts = defaultdict(int)
        for room in self.rooms_discovered.values():
            status = room.get("metadata", {}).get("exploration_status", "unknown")
            status_counts[status] += 1

        return {
            "total_rooms": total_rooms,
            "total_visits": total_visits,
            "fully_explored": status_counts.get("fully_explored", 0),
            "partially_explored": status_counts.get("partial", 0),
            "entry_only": status_counts.get("entry_only", 0),
            "avg_visits_per_room": total_visits / total_rooms if total_rooms > 0 else 0
        }
