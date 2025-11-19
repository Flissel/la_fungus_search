"""
Exploration API endpoints for goal-driven autonomous codebase analysis.
"""
from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explore", tags=["exploration"])


# ============================================================================
# Request/Response Models
# ============================================================================

class StartExplorationRequest(BaseModel):
    """Request to start goal-driven exploration."""
    goal_type: str  # "architecture", "bugs", "security"
    top_k: int = 20
    max_iterations: int = 200


class ExplorationStatusResponse(BaseModel):
    """Current exploration status."""
    active: bool
    goal: str | None
    phase_index: int
    phase_name: str | None
    total_phases: int
    discoveries: dict[str, int]
    files_accessed: int
    queries_explored: int
    step: int


class PhaseAdvanceResponse(BaseModel):
    """Response after advancing phase."""
    success: bool
    new_phase_index: int
    new_phase_name: str | None
    message: str


class StopExplorationResponse(BaseModel):
    """Response after stopping exploration."""
    success: bool
    report_json_path: str | None
    report_markdown_path: str | None
    summary: dict[str, Any]


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/start", response_model=dict)
async def start_exploration(request: StartExplorationRequest):
    """
    Start goal-driven autonomous exploration.

    This initializes exploration mode with the specified goal, sets up
    phase tracking, and seeds the query pool with initial queries.
    """
    from embeddinggemma.realtime.server import streamer
    from embeddinggemma.exploration import get_goal, get_initial_queries, ExplorationReport

    try:
        # Validate goal type
        goal = get_goal(request.goal_type)
        if not goal:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid goal_type: {request.goal_type}. Must be one of: architecture, bugs, security"
            )

        # Check if already in exploration mode
        if streamer.exploration_mode:
            raise HTTPException(
                status_code=400,
                detail="Exploration already active. Stop current exploration before starting a new one."
            )

        # Check if simulation is running
        if not streamer.running:
            raise HTTPException(
                status_code=400,
                detail="Simulation must be running to start exploration. Start simulation first."
            )

        # Initialize exploration mode
        streamer.exploration_mode = True
        streamer.exploration_goal = request.goal_type
        streamer.exploration_phase = 0
        streamer._phase_discoveries = {}
        streamer._phase_files_accessed = {}

        # Update simulation parameters
        streamer.top_k = request.top_k
        streamer.max_iterations = request.max_iterations

        # Initialize exploration report
        run_id = streamer.run_id or "unknown"
        streamer._exploration_report = ExplorationReport(
            goal_type=request.goal_type,
            run_id=run_id
        )

        # Seed query pool with initial queries for phase 0
        initial_queries = get_initial_queries(request.goal_type, 0)
        for query in initial_queries:
            if query not in streamer._query_pool:
                streamer._query_pool.append(query)

        phase_name = goal["phases"][0]["name"]

        _logger.info(
            f"[EXPLORE] Started exploration: goal={request.goal_type}, "
            f"phase={phase_name}, initial_queries={len(initial_queries)}"
        )

        return {
            "success": True,
            "goal": request.goal_type,
            "phase_index": 0,
            "phase_name": phase_name,
            "total_phases": len(goal["phases"]),
            "initial_queries_added": len(initial_queries),
            "message": f"Exploration started for '{request.goal_type}' goal"
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error starting exploration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/status", response_model=ExplorationStatusResponse)
async def get_exploration_status():
    """Get current exploration status and progress."""
    from embeddinggemma.realtime.server import streamer
    from embeddinggemma.exploration import get_goal, get_phase_info

    try:
        if not streamer.exploration_mode:
            return ExplorationStatusResponse(
                active=False,
                goal=None,
                phase_index=0,
                phase_name=None,
                total_phases=0,
                discoveries={},
                files_accessed=0,
                queries_explored=0,
                step=streamer.step_i
            )

        goal = get_goal(streamer.exploration_goal)
        phase = get_phase_info(streamer.exploration_goal, streamer.exploration_phase)

        # Count discoveries
        discoveries_count = {}
        for category, items in streamer._phase_discoveries.items():
            discoveries_count[category] = len(items)

        # Count files accessed across all phases
        total_files = sum(
            len(files) for files in streamer._phase_files_accessed.values()
        )

        # Count queries explored
        queries_explored = len(streamer._query_pool)

        return ExplorationStatusResponse(
            active=True,
            goal=streamer.exploration_goal,
            phase_index=streamer.exploration_phase,
            phase_name=phase["name"] if phase else None,
            total_phases=len(goal["phases"]) if goal else 0,
            discoveries=discoveries_count,
            files_accessed=total_files,
            queries_explored=queries_explored,
            step=streamer.step_i
        )

    except Exception as e:
        _logger.error(f"Error getting exploration status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/next_phase", response_model=PhaseAdvanceResponse)
async def advance_to_next_phase():
    """Manually advance to the next exploration phase."""
    from embeddinggemma.realtime.server import streamer
    from embeddinggemma.exploration import get_goal

    try:
        if not streamer.exploration_mode:
            raise HTTPException(
                status_code=400,
                detail="No active exploration. Start exploration first."
            )

        goal = get_goal(streamer.exploration_goal)
        if not goal:
            raise HTTPException(status_code=500, detail="Invalid exploration goal")

        # Check if already at last phase
        if streamer.exploration_phase >= len(goal["phases"]) - 1:
            return PhaseAdvanceResponse(
                success=False,
                new_phase_index=streamer.exploration_phase,
                new_phase_name=goal["phases"][streamer.exploration_phase]["name"],
                message="Already at the last phase"
            )

        # Advance phase
        await streamer._advance_exploration_phase()

        new_phase = goal["phases"][streamer.exploration_phase]

        return PhaseAdvanceResponse(
            success=True,
            new_phase_index=streamer.exploration_phase,
            new_phase_name=new_phase["name"],
            message=f"Advanced to phase {streamer.exploration_phase}: {new_phase['name']}"
        )

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error advancing phase: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/stop", response_model=StopExplorationResponse)
async def stop_exploration():
    """Stop exploration and generate final report."""
    from embeddinggemma.realtime.server import streamer
    import os
    import json

    try:
        if not streamer.exploration_mode:
            raise HTTPException(
                status_code=400,
                detail="No active exploration to stop."
            )

        # Finalize report
        if hasattr(streamer, '_exploration_report'):
            report = streamer._exploration_report

            # Track final files accessed
            if hasattr(streamer, '_unique_docs_accessed'):
                for doc_id in streamer._unique_docs_accessed:
                    report.files_accessed.add(str(doc_id))

            # Track final queries
            if hasattr(streamer, '_query_pool'):
                for query in streamer._query_pool:
                    report.unique_queries.add(query)

            report.finalize()

            # Save reports
            run_dir = streamer._run_dir
            if run_dir:
                os.makedirs(run_dir, exist_ok=True)

                # Save JSON report
                json_path = os.path.join(run_dir, "exploration_report.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

                # Save Markdown report
                md_path = os.path.join(run_dir, "exploration_report.md")
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(report.to_markdown())

                _logger.info(f"[EXPLORE] Reports saved: {json_path}, {md_path}")

                summary = report.to_dict()["summary"]

                # Disable exploration mode
                streamer.exploration_mode = False
                streamer.exploration_goal = None
                streamer.exploration_phase = 0

                return StopExplorationResponse(
                    success=True,
                    report_json_path=json_path,
                    report_markdown_path=md_path,
                    summary=summary
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="No run directory available to save reports"
                )
        else:
            raise HTTPException(
                status_code=500,
                detail="No exploration report available"
            )

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error stopping exploration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
