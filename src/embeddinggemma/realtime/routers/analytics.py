"""Analytics router - Cross-run query and retrieval analytics endpoints."""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

# Import analytics functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from embeddinggemma.ui.analytics import (
    aggregate_query_corpus_mapping,
    analyze_coverage,
    analyze_query_patterns,
    get_run_summary
)

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/query-corpus-mapping")
async def query_corpus_mapping() -> JSONResponse:
    """Get aggregated query→document mappings across all runs.

    Returns mapping showing which queries retrieve which documents,
    useful for building training data for better retrievals.
    """
    try:
        result = aggregate_query_corpus_mapping()
        return JSONResponse({"status": "ok", "data": result})
    except Exception as e:
        _logger.error(f"Query-corpus mapping error: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.get("/coverage")
async def coverage_analysis() -> JSONResponse:
    """Analyze which documents are accessed vs never found across all runs."""
    try:
        result = analyze_coverage()
        return JSONResponse({"status": "ok", "data": result})
    except Exception as e:
        _logger.error(f"Coverage analysis error: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.get("/query-patterns")
async def query_patterns() -> JSONResponse:
    """Analyze common query patterns and characteristics across runs."""
    try:
        result = analyze_query_patterns()
        return JSONResponse({"status": "ok", "data": result})
    except Exception as e:
        _logger.error(f"Query patterns analysis error: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.get("/runs/{run_id}")
async def run_summary(run_id: str) -> JSONResponse:
    """Get comprehensive summary for a specific run.

    Args:
        run_id: The run identifier

    Returns:
        Combined data from manifest, queries, and retrievals for the run
    """
    try:
        result = get_run_summary(run_id)
        if 'error' in result:
            return JSONResponse({
                "status": "error",
                "message": result['error']
            }, status_code=404)
        return JSONResponse({"status": "ok", "data": result})
    except Exception as e:
        _logger.error(f"Run summary error: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.get("/overview")
async def analytics_overview() -> JSONResponse:
    """Get overview of all analytics data in one call."""
    try:
        overview = {
            'query_corpus_mapping': aggregate_query_corpus_mapping(),
            'coverage': analyze_coverage(),
            'query_patterns': analyze_query_patterns()
        }
        return JSONResponse({"status": "ok", "data": overview})
    except Exception as e:
        _logger.error(f"Analytics overview error: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)
