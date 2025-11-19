"""
Goal-oriented exploration templates for autonomous codebase analysis.

Each goal represents a high-level analysis task with phases, success criteria,
and query generation strategies.
"""
from __future__ import annotations
from typing import TypedDict, Literal


class ExplorationPhase(TypedDict):
    """A phase in goal-driven exploration."""
    name: str
    description: str
    initial_queries: list[str]
    success_criteria: dict[str, int | float]  # e.g., {"min_files": 5, "min_modules": 3}
    max_steps: int


class ExplorationGoal(TypedDict):
    """A high-level exploration goal with multiple phases."""
    goal_type: Literal["architecture", "bugs", "security", "quality", "dependencies", "testing"]
    description: str
    phases: list[ExplorationPhase]
    report_template: str  # How to structure the final report


# ============================================================================
# GOAL: Architecture Understanding
# ============================================================================

ARCHITECTURE_GOAL: ExplorationGoal = {
    "goal_type": "architecture",
    "description": "Understand the overall architecture, design patterns, and component relationships",
    "phases": [
        {
            "name": "Entry Points Discovery",
            "description": "Find main entry points, API routes, CLI commands, and service initializers",
            "initial_queries": [
                "main entry point application startup",
                "FastAPI app initialization routes",
                "CLI commands argument parser",
                "@app.get @app.post API endpoints",
                "server.py main() function",
                "__main__ entry point",
            ],
            "success_criteria": {
                "min_files": 3,
                "min_entry_points": 2,
            },
            "max_steps": 15,
        },
        {
            "name": "Core Modules Discovery",
            "description": "Identify major modules, services, and their responsibilities",
            "initial_queries": [
                "service classes business logic",
                "module structure package organization",
                "core functionality main components",
                "class definitions models schemas",
                "dependency injection service container",
            ],
            "success_criteria": {
                "min_files": 8,
                "min_modules": 5,
            },
            "max_steps": 20,
        },
        {
            "name": "Data Flow Analysis",
            "description": "Trace how data flows through the system",
            "initial_queries": [
                "request handling pipeline middleware",
                "data transformation processing",
                "database queries ORM models",
                "API request response flow",
                "validation schemas input output",
            ],
            "success_criteria": {
                "min_files": 6,
                "min_patterns": 3,
            },
            "max_steps": 20,
        },
        {
            "name": "Dependencies & Imports",
            "description": "Map external dependencies and internal imports",
            "initial_queries": [
                "import statements module dependencies",
                "external libraries third party packages",
                "internal module imports cross-references",
                "requirements.txt pyproject.toml dependencies",
            ],
            "success_criteria": {
                "min_files": 5,
                "min_dependencies": 10,
            },
            "max_steps": 15,
        },
        {
            "name": "Design Patterns",
            "description": "Identify architectural patterns and practices",
            "initial_queries": [
                "design patterns singleton factory",
                "dependency injection container",
                "repository pattern data access",
                "middleware decorators interceptors",
                "configuration management settings",
            ],
            "success_criteria": {
                "min_patterns": 3,
            },
            "max_steps": 15,
        },
    ],
    "report_template": """
# Architecture Report

## 1. System Overview
- Purpose: [High-level purpose]
- Architecture Style: [e.g., microservices, monolith, layered]
- Key Technologies: [List main frameworks/libraries]

## 2. Entry Points
- [List main entry points with file paths]

## 3. Core Modules
- [Module name]: [Responsibility]
  - Files: [file paths]
  - Key classes: [class names]

## 4. Data Flow
- [Describe request/response flow]
- [Describe data transformations]

## 5. Dependencies
- External: [List with versions if available]
- Internal: [Module dependency graph]

## 6. Design Patterns
- [Pattern name]: [Where used, purpose]

## 7. Key Insights
- [Notable architectural decisions]
- [Potential concerns or areas for improvement]
""",
}


# ============================================================================
# GOAL: Bug and Error Analysis
# ============================================================================

BUGS_GOAL: ExplorationGoal = {
    "goal_type": "bugs",
    "description": "Identify error handling, potential bugs, and code quality issues",
    "phases": [
        {
            "name": "Error Handling Discovery",
            "description": "Find all error handling code and exception patterns",
            "initial_queries": [
                "try except error handling",
                "raise exception error cases",
                "error logging logger.error",
                "HTTPException FastAPI errors",
                "validation errors ValueError TypeError",
            ],
            "success_criteria": {
                "min_files": 5,
                "min_error_handlers": 10,
            },
            "max_steps": 20,
        },
        {
            "name": "Input Validation",
            "description": "Analyze input validation and sanitization",
            "initial_queries": [
                "input validation pydantic BaseModel",
                "request body validation schema",
                "parameter validation type checking",
                "sanitize user input escape",
            ],
            "success_criteria": {
                "min_files": 4,
            },
            "max_steps": 15,
        },
        {
            "name": "Edge Cases",
            "description": "Identify potential edge cases and boundary conditions",
            "initial_queries": [
                "empty list None check",
                "zero division error boundary",
                "index out of range bounds",
                "null pointer None value",
            ],
            "success_criteria": {
                "min_issues": 5,
            },
            "max_steps": 15,
        },
    ],
    "report_template": """
# Bug and Error Analysis Report

## 1. Error Handling Coverage
- Files with error handling: [count]
- Common patterns: [list patterns]

## 2. Input Validation
- Validation strategy: [describe]
- Gaps: [list missing validations]

## 3. Potential Issues
- [Category]: [List specific issues with file:line references]

## 4. Recommendations
- [Priority]: [Specific recommendation]
""",
}


# ============================================================================
# GOAL: Security Analysis
# ============================================================================

SECURITY_GOAL: ExplorationGoal = {
    "goal_type": "security",
    "description": "Identify security vulnerabilities and authentication/authorization patterns",
    "phases": [
        {
            "name": "Authentication Discovery",
            "description": "Find authentication mechanisms and user management",
            "initial_queries": [
                "authentication login password",
                "JWT token OAuth",
                "user session management",
                "password hashing bcrypt",
            ],
            "success_criteria": {
                "min_files": 3,
            },
            "max_steps": 15,
        },
        {
            "name": "Authorization & Access Control",
            "description": "Analyze permission checks and access control",
            "initial_queries": [
                "authorization permission check",
                "role based access RBAC",
                "depends authentication required",
                "admin only protected route",
            ],
            "success_criteria": {
                "min_files": 3,
            },
            "max_steps": 15,
        },
        {
            "name": "Security Vulnerabilities",
            "description": "Look for common security issues",
            "initial_queries": [
                "SQL injection query parameters",
                "XSS cross site scripting",
                "CSRF token protection",
                "input sanitization escape",
                "secrets credentials hardcoded",
            ],
            "success_criteria": {
                "min_files": 4,
            },
            "max_steps": 20,
        },
    ],
    "report_template": """
# Security Analysis Report

## 1. Authentication
- Mechanism: [describe]
- Files: [list]

## 2. Authorization
- Strategy: [describe]
- Coverage: [assessment]

## 3. Vulnerabilities Found
- [Type]: [Location and description]

## 4. Security Recommendations
- [Priority]: [Specific action]
""",
}


# ============================================================================
# Goal Registry
# ============================================================================

GOALS: dict[str, ExplorationGoal] = {
    "architecture": ARCHITECTURE_GOAL,
    "bugs": BUGS_GOAL,
    "security": SECURITY_GOAL,
}


def get_goal(goal_type: str) -> ExplorationGoal | None:
    """Get exploration goal by type."""
    return GOALS.get(goal_type.lower())


def get_initial_queries(goal_type: str, phase_index: int = 0) -> list[str]:
    """Get initial queries for a specific goal and phase."""
    goal = get_goal(goal_type)
    if not goal or phase_index >= len(goal["phases"]):
        return []
    return goal["phases"][phase_index]["initial_queries"]


def get_phase_info(goal_type: str, phase_index: int = 0) -> ExplorationPhase | None:
    """Get information about a specific phase."""
    goal = get_goal(goal_type)
    if not goal or phase_index >= len(goal["phases"]):
        return None
    return goal["phases"][phase_index]
