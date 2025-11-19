"""Goal-oriented exploration module for autonomous codebase analysis."""
from __future__ import annotations

from .goals import (
    ExplorationGoal,
    ExplorationPhase,
    ARCHITECTURE_GOAL,
    BUGS_GOAL,
    SECURITY_GOAL,
    GOALS,
    get_goal,
    get_initial_queries,
    get_phase_info,
)
from .query_generator import generate_contextual_queries
from .report_builder import ExplorationReport

__all__ = [
    "ExplorationGoal",
    "ExplorationPhase",
    "ARCHITECTURE_GOAL",
    "BUGS_GOAL",
    "SECURITY_GOAL",
    "GOALS",
    "get_goal",
    "get_initial_queries",
    "get_phase_info",
    "generate_contextual_queries",
    "ExplorationReport",
]
