"""Offline Gate 1 MCMP ablation orchestration and evidence writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from benchmarks.mcmp.adapters import AdapterEvidence, run_faiss, run_mcmp
from benchmarks.mcmp.contracts import BenchmarkDataset, SearchRun
from benchmarks.mcmp.fixtures import build_synthetic_dataset
from benchmarks.mcmp.metrics import candidate_overlap, evaluate_run, query_geometry


_RUN_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A", ("q-main",)),
    ("B", ("q-main", "q-related")),
    ("C", ("q-main",)),
    ("D", ("q-main", "q-related")),
)


def run_gate1(
    seed: int, top_k: int, initial_k: int, num_agents: int, steps: int
) -> dict[str, object]:
    """Run the fixed A-D offline ablation and return its complete evidence."""
    dataset = build_synthetic_dataset(seed)
    runs: dict[str, dict[str, object]] = {}

    for method, query_ids in _RUN_SPECS:
        if method in {"A", "B"}:
            run, evidence = run_faiss(dataset, method, query_ids, top_k, initial_k)
        else:
            run, evidence = run_mcmp(
                dataset,
                method,
                query_ids,
                top_k,
                initial_k,
                seed,
                num_agents,
                steps,
            )
        runs[method] = _run_payload(dataset, run, evidence, top_k)

    payload: dict[str, object] = {
        "config": {
            "seed": seed,
            "top_k": top_k,
            "initial_k": initial_k,
            "num_agents": num_agents,
            "steps": steps,
        },
        "dataset": {
            "id": dataset.dataset_id,
            "digest": dataset.digest(),
            "document_ids": list(dataset.document_ids),
            "query_ids": list(dataset.query_ids),
            "document_vector_shape": list(dataset.document_vectors.shape),
            "query_vector_shape": list(dataset.query_vectors.shape),
        },
        "environment": _environment_payload(),
        "query_geometry": query_geometry(dataset),
        "runs": runs,
        "comparisons": _comparison_payload(runs),
    }
    novel_count = sum(
        len(runs[method]["metrics"]["novel_relevant_candidates"])
        for method in ("C", "D")
    )
    payload["conclusion"] = (
        "novel_relevant_observed" if novel_count > 0 else "no_novel_relevant_observed"
    )
    return payload


def write_gate1_result(payload: dict[str, object], path: Path) -> None:
    """Write a deterministic, human-reviewable Gate 1 JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_payload(
    dataset: BenchmarkDataset, run: SearchRun, evidence: AdapterEvidence, top_k: int
) -> dict[str, object]:
    return {
        "query_ids": list(run.query_ids),
        "independent_run_count": evidence.independent_run_count,
        "raw_ids": {
            "ranked_document_ids": list(run.ranked_document_ids),
            "initial_candidate_ids": sorted(run.initial_candidate_ids),
            "discovered_candidate_ids": sorted(run.discovered_candidate_ids),
            "per_query_candidate_ids": {
                query_id: sorted(candidate_ids)
                for query_id, candidate_ids in sorted(run.per_query_candidate_ids.items())
            },
            "per_query_ranked_document_ids": {
                query_id: list(ranking)
                for query_id, ranking in sorted(run.per_query_ranked_document_ids.items())
            },
        },
        "metrics": evaluate_run(dataset, run, top_k),
        "candidate_overlap": candidate_overlap(run),
        "timing": {"elapsed_ms": run.elapsed_ms},
        "candidate_comparisons": run.candidate_comparisons,
        "nearest_search_calls": evidence.nearest_search_calls,
        "mcmp_steps": run.mcmp_steps,
        "document_visits": dict(sorted(run.document_visits.items())),
        "pheromone_trails": run.pheromone_trails,
    }


def _environment_payload() -> dict[str, str]:
    try:
        import faiss  # type: ignore[import-not-found]

        faiss_version = getattr(faiss, "__version__", "unknown")
    except ImportError:
        faiss_version = "unavailable"
    return {
        "cpu_mode": "cpu",
        "faiss_version": str(faiss_version),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }


def _comparison_payload(runs: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "A_vs_C": _compare_runs(runs["A"], runs["C"]),
        "B_vs_D": _compare_runs(runs["B"], runs["D"]),
    }


def _compare_runs(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    left_metrics = left["metrics"]
    right_metrics = right["metrics"]
    assert isinstance(left_metrics, dict)
    assert isinstance(right_metrics, dict)
    return {
        "ranked_document_ids_equal": (
            left["raw_ids"]["ranked_document_ids"] == right["raw_ids"]["ranked_document_ids"]
        ),
        "novel_relevant_candidates": right_metrics["novel_relevant_candidates"],
        "recall_at_k_delta": right_metrics["recall_at_k"] - left_metrics["recall_at_k"],
        "mrr_delta": right_metrics["mrr"] - left_metrics["mrr"],
        "ndcg_at_k_delta": right_metrics["ndcg_at_k"] - left_metrics["ndcg_at_k"],
    }


def _has_complete_gate1_evidence(payload: dict[str, object]) -> bool:
    runs = payload.get("runs")
    if not isinstance(runs, dict) or list(runs) != ["A", "B", "C", "D"]:
        return False
    return all(
        isinstance(run, dict)
        and isinstance(run.get("metrics"), dict)
        and isinstance(run["metrics"].get("novel_relevant_candidates"), list)
        for run in runs.values()
    ) and payload.get("conclusion") in {
        "novel_relevant_observed",
        "no_novel_relevant_observed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--initial-k", type=int, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_gate1(args.seed, args.top_k, args.initial_k, args.num_agents, args.steps)
    write_gate1_result(payload, args.output)
    return 0 if _has_complete_gate1_evidence(payload) else 1


if __name__ == "__main__":
    raise SystemExit(main())
