"""
Contextual query generator for goal-driven exploration.

Generates targeted follow-up queries based on:
- Current exploration goal and phase
- Previously discovered code (entry points, modules, patterns)
- Query history (to avoid repetition)
"""
from __future__ import annotations
from typing import Any

from .goals import get_goal, get_phase_info


def generate_contextual_queries(
    goal_type: str,
    phase_index: int,
    discoveries: dict[str, list[str]],
    recent_results: list[dict[str, Any]],
    query_history: list[str],
    max_queries: int = 5,
) -> list[str]:
    """
    Generate contextual follow-up queries based on current exploration state.

    Args:
        goal_type: Type of exploration goal (e.g., "architecture")
        phase_index: Current phase index
        discoveries: Dict of discovered items per category (e.g., {"entry_points": [...], "modules": [...]})
        recent_results: Recently retrieved code chunks
        query_history: List of previous queries to avoid repetition
        max_queries: Maximum number of queries to generate

    Returns:
        List of contextual query strings
    """
    goal = get_goal(goal_type)
    phase = get_phase_info(goal_type, phase_index)

    if not goal or not phase:
        return []

    queries: list[str] = []

    # 1. Extract discovered entities from recent results
    discovered_files = set()
    discovered_classes = set()
    discovered_functions = set()
    discovered_imports = set()

    for result in recent_results[-10:]:  # Last 10 results
        content = result.get("content", "")
        metadata = result.get("metadata", {})

        # Extract file path
        if "file_path" in metadata:
            discovered_files.add(metadata["file_path"])

        # Extract class names (simple pattern matching)
        import re
        classes = re.findall(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", content)
        discovered_classes.update(classes[:3])  # Top 3 classes

        # Extract function/method names
        functions = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", content)
        discovered_functions.update(functions[:3])  # Top 3 functions

        # Extract imports
        imports = re.findall(r"from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import", content)
        imports += re.findall(r"import\s+([A-Za-z_][A-Za-z0-9_.]*)", content)
        discovered_imports.update(imports[:5])  # Top 5 imports

    # 2. Generate phase-specific contextual queries
    phase_name = phase["name"]

    if "Entry Points" in phase_name:
        queries.extend(_generate_entry_point_queries(
            discovered_files, discovered_classes, discovered_functions
        ))

    elif "Core Modules" in phase_name:
        queries.extend(_generate_module_queries(
            discovered_files, discovered_classes, discovered_imports
        ))

    elif "Data Flow" in phase_name:
        queries.extend(_generate_data_flow_queries(
            discovered_classes, discovered_functions, discovered_imports
        ))

    elif "Dependencies" in phase_name or "Imports" in phase_name:
        queries.extend(_generate_dependency_queries(
            discovered_imports, discovered_files
        ))

    elif "Design Patterns" in phase_name or "Patterns" in phase_name:
        queries.extend(_generate_pattern_queries(
            discovered_classes, discovered_functions
        ))

    elif "Error Handling" in phase_name:
        queries.extend(_generate_error_handling_queries(
            discovered_files, discovered_functions
        ))

    elif "Security" in phase_name or "Auth" in phase_name:
        queries.extend(_generate_security_queries(
            discovered_functions, discovered_classes
        ))

    # 3. Filter out queries similar to history
    queries = _filter_by_history(queries, query_history)

    # 4. Return top N queries
    return queries[:max_queries]


def _generate_entry_point_queries(
    files: set[str], classes: set[str], functions: set[str]
) -> list[str]:
    """Generate queries to explore entry points further."""
    queries = []

    # Follow discovered entry points
    for func in list(functions)[:3]:
        queries.append(f"calls to {func} function invocations")
        queries.append(f"{func} implementation definition")

    for cls in list(classes)[:3]:
        queries.append(f"{cls} initialization constructor __init__")
        queries.append(f"{cls} methods interface")

    # General entry point expansion
    queries.extend([
        "application startup bootstrap initialization sequence",
        "CLI argument parser command line interface",
        "configuration loading settings initialization",
    ])

    return queries


def _generate_module_queries(
    files: set[str], classes: set[str], imports: set[str]
) -> list[str]:
    """Generate queries to explore module structure."""
    queries = []

    # Follow discovered modules
    for imp in list(imports)[:5]:
        module_parts = imp.split(".")
        if len(module_parts) > 1:
            queries.append(f"{module_parts[0]} module components")
            queries.append(f"{'.'.join(module_parts[:2])} submodule structure")

    for cls in list(classes)[:3]:
        queries.append(f"{cls} dependencies used by")

    # General module exploration
    queries.extend([
        "service layer business logic classes",
        "data models schemas definitions",
        "utility functions helper modules",
    ])

    return queries


def _generate_data_flow_queries(
    classes: set[str], functions: set[str], imports: set[str]
) -> list[str]:
    """Generate queries to trace data flow."""
    queries = []

    # Follow data transformations
    for func in list(functions)[:3]:
        queries.append(f"{func} input parameters arguments")
        queries.append(f"{func} return value output")

    for cls in list(classes)[:3]:
        queries.append(f"{cls} data attributes fields")
        queries.append(f"{cls} processing methods")

    # General data flow
    queries.extend([
        "request validation input schema",
        "response serialization output format",
        "database queries ORM operations",
        "data transformation pipeline processing",
    ])

    return queries


def _generate_dependency_queries(
    imports: set[str], files: set[str]
) -> list[str]:
    """Generate queries to map dependencies."""
    queries = []

    # Follow specific dependencies
    for imp in list(imports)[:5]:
        queries.append(f"{imp} usage examples")
        queries.append(f"imports from {imp}")

    # General dependency mapping
    queries.extend([
        "external library dependencies third party",
        "internal module cross-references imports",
        "configuration dependencies environment variables",
    ])

    return queries


def _generate_pattern_queries(
    classes: set[str], functions: set[str]
) -> list[str]:
    """Generate queries to identify design patterns."""
    queries = []

    # Follow discovered patterns
    for cls in list(classes)[:3]:
        queries.append(f"{cls} factory builder pattern")
        queries.append(f"{cls} singleton instance")

    # General pattern identification
    queries.extend([
        "dependency injection container",
        "repository pattern data access",
        "middleware decorators interceptors",
        "observer pattern event handlers",
    ])

    return queries


def _generate_error_handling_queries(
    files: set[str], functions: set[str]
) -> list[str]:
    """Generate queries for error handling analysis."""
    queries = []

    # Follow discovered error handling
    for func in list(functions)[:3]:
        queries.append(f"{func} error handling exceptions")
        queries.append(f"{func} validation checks")

    # General error analysis
    queries.extend([
        "custom exception classes error types",
        "error logging logger.error calls",
        "validation error messages",
        "exception propagation handling",
    ])

    return queries


def _generate_security_queries(
    functions: set[str], classes: set[str]
) -> list[str]:
    """Generate queries for security analysis."""
    queries = []

    # Follow discovered security components
    for func in list(functions)[:3]:
        if any(sec in func.lower() for sec in ["auth", "login", "verify", "check", "validate"]):
            queries.append(f"{func} implementation security")
            queries.append(f"{func} permission checks")

    for cls in list(classes)[:3]:
        if any(sec in cls.lower() for sec in ["auth", "user", "session", "token"]):
            queries.append(f"{cls} security methods")

    # General security analysis
    queries.extend([
        "authentication mechanism login flow",
        "authorization permission checks RBAC",
        "input sanitization validation",
        "password hashing security",
    ])

    return queries


def _filter_by_history(queries: list[str], history: list[str]) -> list[str]:
    """Filter out queries too similar to history."""
    if not history:
        return queries

    filtered = []
    history_lower = [h.lower() for h in history[-30:]]  # Last 30 queries

    for q in queries:
        q_lower = q.lower()
        # Check if query is too similar to any in history
        is_duplicate = False
        for h in history_lower:
            # Simple token overlap check
            q_tokens = set(q_lower.split())
            h_tokens = set(h.split())
            overlap = len(q_tokens & h_tokens)
            union = len(q_tokens | h_tokens)

            if union > 0 and (overlap / union) > 0.7:  # 70% token overlap
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(q)

    return filtered
