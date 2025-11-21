"""
Bootstrap foundational codebase knowledge into Supermemory.

This module scans the project directory structure and creates
initial architectural memories BEFORE any exploration begins.
"""

import os
import ast
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

_logger = logging.getLogger(__name__)


class CodebaseBootstrap:
    """Bootstrap foundational knowledge about codebase structure."""

    def __init__(self, root_dir: str, memory_manager):
        """
        Initialize bootstrap scanner.

        Args:
            root_dir: Root directory of the project
            memory_manager: SupermemoryManager instance
        """
        self.root_dir = Path(root_dir)
        self.memory_manager = memory_manager

    async def bootstrap(self, container_tag: str) -> Dict[str, Any]:
        """
        Create foundational memories about codebase structure.

        This scans the src/ directory to extract:
        - Module tree (all Python packages)
        - Entry points (main executable files)
        - Per-module summaries for significant modules

        Args:
            container_tag: Supermemory container tag (typically run_id)

        Returns:
            Statistics about created memories
        """
        _logger.info("[BOOTSTRAP] Scanning codebase structure...")
        memories_created = 0

        try:
            # 1. Scan module structure
            module_tree = self._scan_module_structure()
            _logger.info(f"[BOOTSTRAP] Found {len(module_tree)} modules")

            # 2. Create module tree memory
            success = await self.memory_manager.add_memory(
                content=self._format_module_tree(module_tree),
                type="codebase_structure",
                metadata={
                    "level": "foundational",
                    "category": "structure",
                    "modules_count": len(module_tree),
                    "auto_generated": True
                },
                custom_id="codebase_module_tree",
                container_tag=container_tag
            )
            if success:
                memories_created += 1
                _logger.debug("[BOOTSTRAP] Created codebase_module_tree memory")

            # 3. Find entry points
            entry_points = self._find_entry_points()
            _logger.info(f"[BOOTSTRAP] Found {len(entry_points)} entry points")

            # 4. Create entry points memory
            success = await self.memory_manager.add_memory(
                content=self._format_entry_points(entry_points),
                type="entry_points",
                metadata={
                    "level": "foundational",
                    "category": "entry_points",
                    "count": len(entry_points),
                    "auto_generated": True
                },
                custom_id="codebase_entry_points",
                container_tag=container_tag
            )
            if success:
                memories_created += 1
                _logger.debug("[BOOTSTRAP] Created codebase_entry_points memory")

            # 5. Create per-module summaries for significant modules
            for module_path, module_info in module_tree.items():
                if self._is_significant_module(module_info):
                    success = await self._create_module_memory(
                        module_path, module_info, container_tag
                    )
                    if success:
                        memories_created += 1

            _logger.info(f"[BOOTSTRAP] Created {memories_created} foundational memories")

            return {
                "success": True,
                "memories_created": memories_created,
                "module_tree": module_tree,
                "entry_points": entry_points
            }

        except Exception as e:
            _logger.error(f"[BOOTSTRAP] Failed: {e}", exc_info=True)
            return {
                "success": False,
                "memories_created": memories_created,
                "error": str(e)
            }

    def _scan_module_structure(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan src/ directory and extract module structure.

        Finds all Python packages (directories with __init__.py) and
        analyzes their contents.

        Returns:
            Dict mapping module paths to their metadata
        """
        modules = {}
        src_dir = self.root_dir / "src"

        if not src_dir.exists():
            _logger.warning(f"[BOOTSTRAP] src/ directory not found at {src_dir}")
            return modules

        # Find all __init__.py files (these define Python modules)
        for init_file in src_dir.rglob("__init__.py"):
            module_dir = init_file.parent

            try:
                relative_path = module_dir.relative_to(src_dir)
                module_name = str(relative_path).replace(os.sep, ".")

                # Find all .py files in this module (excluding __init__.py)
                py_files = [
                    f.name for f in module_dir.glob("*.py")
                    if f.name != "__init__.py"
                ]

                # Analyze module contents
                total_lines = 0
                main_classes = []
                main_functions = []
                imports = set()

                for py_file in module_dir.glob("*.py"):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.splitlines()
                            total_lines += len(lines)

                            # Parse AST to extract classes/functions
                            try:
                                tree = ast.parse(content)

                                # Extract top-level classes
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.ClassDef):
                                        main_classes.append(node.name)
                                    elif isinstance(node, ast.FunctionDef):
                                        # Only top-level functions (not methods)
                                        if node.col_offset == 0:
                                            main_functions.append(node.name)
                                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                                        # Track imports to understand dependencies
                                        if isinstance(node, ast.ImportFrom) and node.module:
                                            imports.add(node.module)

                            except SyntaxError:
                                # Skip files with syntax errors
                                pass

                    except Exception as e:
                        _logger.debug(f"[BOOTSTRAP] Error analyzing {py_file}: {e}")
                        continue

                # Store module info
                modules[module_name] = {
                    "path": str(relative_path),
                    "files": py_files,
                    "file_count": len(py_files),
                    "total_lines": total_lines,
                    "main_classes": list(set(main_classes))[:5],  # Top 5 unique classes
                    "main_functions": list(set(main_functions))[:5],  # Top 5 unique functions
                    "key_imports": list(imports)[:5]  # Top 5 external imports
                }

            except Exception as e:
                _logger.debug(f"[BOOTSTRAP] Error processing module {init_file}: {e}")
                continue

        return modules

    def _find_entry_points(self) -> List[Dict[str, str]]:
        """
        Find main entry points (server.py, __main__.py, etc.).

        Entry points are files that typically start the application or
        provide main CLI interfaces.

        Returns:
            List of entry point files with metadata
        """
        entry_points = []
        src_dir = self.root_dir / "src"

        if not src_dir.exists():
            return entry_points

        # Common entry point patterns
        patterns = [
            "**/server.py",
            "**/__main__.py",
            "**/app.py",
            "**/main.py",
            "**/cli.py"
        ]

        for pattern in patterns:
            for file_path in src_dir.glob(pattern):
                try:
                    relative_path = file_path.relative_to(src_dir)

                    # Try to extract docstring
                    description = ""
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            tree = ast.parse(content)
                            docstring = ast.get_docstring(tree)
                            if docstring:
                                # Get first line of docstring
                                description = docstring.split('\n')[0]
                    except Exception:
                        pass

                    entry_points.append({
                        "file": str(relative_path),
                        "type": file_path.stem,
                        "description": description[:200] if description else ""
                    })

                except Exception as e:
                    _logger.debug(f"[BOOTSTRAP] Error processing entry point {file_path}: {e}")
                    continue

        return entry_points

    def _format_module_tree(self, modules: Dict[str, Dict]) -> str:
        """
        Format module tree as readable text for memory storage.

        Args:
            modules: Module metadata dict

        Returns:
            Formatted string describing module structure
        """
        lines = ["CODEBASE MODULE STRUCTURE:\n"]
        lines.append("This is a foundational map of all Python packages in the codebase.\n")

        for module_name, info in sorted(modules.items()):
            lines.append(f"\n📦 {module_name}")
            lines.append(f"   Location: {info['path']}")
            lines.append(f"   Files: {info['file_count']} files")

            if info['files']:
                file_list = ', '.join(info['files'][:3])
                if len(info['files']) > 3:
                    file_list += f" ... (+{len(info['files']) - 3} more)"
                lines.append(f"   Python files: {file_list}")

            lines.append(f"   Size: ~{info['total_lines']} lines of code")

            if info['main_classes']:
                lines.append(f"   Key Classes: {', '.join(info['main_classes'][:3])}")

            if info['main_functions']:
                lines.append(f"   Key Functions: {', '.join(info['main_functions'][:3])}")

            if info['key_imports']:
                lines.append(f"   Dependencies: {', '.join(info['key_imports'][:3])}")

        return "\n".join(lines)

    def _format_entry_points(self, entry_points: List[Dict]) -> str:
        """
        Format entry points as readable text.

        Args:
            entry_points: List of entry point metadata

        Returns:
            Formatted string describing entry points
        """
        if not entry_points:
            return "CODEBASE ENTRY POINTS:\n\nNo main entry points detected."

        lines = ["CODEBASE ENTRY POINTS:\n"]
        lines.append("These files are the main entry points for running the application.\n")

        for ep in entry_points:
            lines.append(f"\n🚪 {ep['file']} ({ep['type']})")
            if ep['description']:
                lines.append(f"   Description: {ep['description'][:150]}")

        return "\n".join(lines)

    def _is_significant_module(self, module_info: Dict) -> bool:
        """
        Determine if module is significant enough for its own memory entry.

        Significant modules have:
        - Multiple files (2+)
        - Substantial code (100+ lines)
        - Multiple classes (2+)

        Args:
            module_info: Module metadata dict

        Returns:
            True if module should get dedicated memory
        """
        return (
            module_info['file_count'] >= 2 or
            module_info['total_lines'] >= 100 or
            len(module_info['main_classes']) >= 2
        )

    async def _create_module_memory(
        self,
        module_path: str,
        module_info: Dict,
        container_tag: str
    ) -> bool:
        """
        Create a memory entry for a significant module.

        Args:
            module_path: Dotted module path (e.g., embeddinggemma.agents)
            module_info: Module metadata
            container_tag: Supermemory container tag

        Returns:
            True if memory created successfully
        """
        content_lines = [
            f"MODULE OVERVIEW: {module_path}\n",
            f"Location: {module_info['path']}",
            f"Files ({module_info['file_count']}): {', '.join(module_info['files'])}",
            f"Size: ~{module_info['total_lines']} lines of code\n"
        ]

        if module_info['main_classes']:
            content_lines.append("Key Classes:")
            for cls in module_info['main_classes']:
                content_lines.append(f"  - {cls}")
            content_lines.append("")

        if module_info['main_functions']:
            content_lines.append("Key Functions:")
            for func in module_info['main_functions']:
                content_lines.append(f"  - {func}")
            content_lines.append("")

        if module_info['key_imports']:
            content_lines.append("External Dependencies:")
            for imp in module_info['key_imports']:
                content_lines.append(f"  - {imp}")

        content = "\n".join(content_lines)
        custom_id = f"module_{module_path.replace('.', '_')}"

        try:
            success = await self.memory_manager.add_memory(
                content=content,
                type="module_overview",
                metadata={
                    "level": "foundational",
                    "category": "module",
                    "module_path": module_path,
                    "file_count": module_info['file_count'],
                    "line_count": module_info['total_lines'],
                    "auto_generated": True
                },
                custom_id=custom_id,
                container_tag=container_tag
            )

            if success:
                _logger.debug(f"[BOOTSTRAP] Created memory for module: {module_path}")

            return success

        except Exception as e:
            _logger.error(f"[BOOTSTRAP] Failed to create memory for {module_path}: {e}")
            return False
