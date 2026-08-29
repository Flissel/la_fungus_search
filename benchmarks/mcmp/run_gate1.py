"""Offline Gate 1 MCMP ablation orchestration and evidence writer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from collections.abc import Mapping

import numpy as np

from benchmarks.mcmp.adapters import (
    EXPLORATION_BONUS,
    PHEROMONE_DECAY,
    DETERMINISTIC_CLOCK_MODE,
    DETERMINISTIC_CLOCK_VALUE,
    AdapterEvidence,
    run_faiss,
    run_mcmp,
)
from benchmarks.mcmp.contracts import BenchmarkDataset, SearchRun
from benchmarks.mcmp.fixtures import FIXTURES, build_dataset
from benchmarks.mcmp.metrics import candidate_overlap, evaluate_run, query_geometry


_RUN_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A", ("q-main",)),
    ("B", ("q-main", "q-related")),
    ("C", ("q-main",)),
    ("D", ("q-main", "q-related")),
    ("E", ("q-main",)),
)


def run_gate1(
    seed: int,
    top_k: int,
    initial_k: int,
    num_agents: int,
    steps: int,
    fixture: str = "legacy",
) -> dict[str, object]:
    """Run the fixed A-D offline ablation and return its complete evidence."""
    dataset = build_dataset(fixture, seed)
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
                pool_only=method == "E",
            )
        runs[method] = _run_payload(
            dataset, run, evidence, seed, top_k, initial_k, num_agents, steps
        )

    payload: dict[str, object] = {
        "config": {
            "seed": seed,
            "top_k": top_k,
            "initial_k": initial_k,
            "num_agents": num_agents,
            "steps": steps,
        },
        "dataset": {
            "fixture": fixture,
            "id": dataset.dataset_id,
            "digest": dataset.digest(),
            "document_ids": list(dataset.document_ids),
            "query_ids": list(dataset.query_ids),
            "document_vector_shape": list(dataset.document_vectors.shape),
            "query_vector_shape": list(dataset.query_vectors.shape),
        },
        "environment": _environment_payload(),
        "execution": {
            "force_cpu": True,
            "faiss_factory": "Flat",
            "faiss_metric": "inner_product",
            "pheromone_decay": PHEROMONE_DECAY,
            "exploration_bonus": EXPLORATION_BONUS,
            "clock_mode": DETERMINISTIC_CLOCK_MODE,
            "clock_value": DETERMINISTIC_CLOCK_VALUE,
        },
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
    validate_gate1_evidence(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_payload(
    dataset: BenchmarkDataset,
    run: SearchRun,
    evidence: AdapterEvidence,
    seed: int,
    top_k: int,
    initial_k: int,
    num_agents: int,
    steps: int,
) -> dict[str, object]:
    execution = _run_execution_snapshot(
        run.method,
        run.query_ids,
        evidence.execution_backend,
        seed,
        top_k,
        initial_k,
        num_agents,
        steps,
        evidence.clock_mode,
        evidence.clock_value,
    )
    if evidence.per_query_random_seeds != execution["random_seed_provenance"]["per_query"]:
        raise ValueError("adapter random seed provenance does not match the run configuration")
    return {
        "query_ids": list(run.query_ids),
        "independent_run_count": evidence.independent_run_count,
        "execution_backend": evidence.execution_backend,
        "execution": execution,
        "raw_ids": {
            "ranked_document_ids": list(run.ranked_document_ids),
            "initial_candidate_ids": sorted(run.initial_candidate_ids),
            "per_query_initial_candidate_ids": {
                query_id: list(candidate_ids)
                for query_id, candidate_ids in evidence.per_query_initial_candidate_ids.items()
            },
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


def _run_execution_snapshot(
    method: str,
    query_ids: tuple[str, ...] | list[str],
    execution_backend: str,
    seed: int,
    top_k: int,
    initial_k: int,
    num_agents: int,
    steps: int,
    clock_mode: str = DETERMINISTIC_CLOCK_MODE,
    clock_value: float = DETERMINISTIC_CLOCK_VALUE,
) -> dict[str, object]:
    is_mcmp = method in {"C", "D", "E"}
    random_seed_provenance: dict[str, object] = {
        "python_random_seed": seed if is_mcmp else None,
        "numpy_random_seed": seed if is_mcmp else None,
        "per_query": (
            {
                query_id: {
                    "python_random_seed": seed + query_index,
                    "numpy_random_seed": seed + query_index,
                }
                for query_index, query_id in enumerate(query_ids)
            }
            if is_mcmp
            else None
        ),
    }
    return {
        "mode": "mcmp" if is_mcmp else "faiss",
        "seed": seed,
        "top_k": top_k,
        "initial_k": initial_k,
        "force_cpu": True,
        "faiss_factory": "Flat",
        "faiss_metric": "inner_product",
        "execution_backend": execution_backend,
        "num_agents": num_agents if is_mcmp else None,
        "steps": steps if is_mcmp else None,
        "pheromone_decay": PHEROMONE_DECAY if is_mcmp else None,
        "exploration_bonus": EXPLORATION_BONUS if is_mcmp else None,
        "clock_mode": clock_mode,
        "clock_value": clock_value,
        "random_seed_provenance": random_seed_provenance,
    }


def _environment_payload() -> dict[str, object]:
    try:
        import faiss  # type: ignore[import-not-found]

        faiss_version = getattr(faiss, "__version__", "unknown")
    except ImportError:
        faiss_version = "unavailable"
    return {
        "execution_backend": "faiss-cpu",
        "force_cpu": True,
        "faiss_factory": "Flat",
        "faiss_metric": "inner_product",
        "faiss_version": str(faiss_version),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }


def _comparison_payload(runs: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    comparisons = {
        "A_vs_C": _compare_runs(runs["A"], runs["C"]),
        "B_vs_D": _compare_runs(runs["B"], runs["D"]),
    }
    if "E" in runs:
        comparisons["A_vs_E"] = _compare_runs(runs["A"], runs["E"])
        comparisons["C_vs_E"] = _compare_runs(runs["C"], runs["E"])
    return comparisons


def _direct_initial_candidates(
    dataset: BenchmarkDataset, query_id: str, initial_k: int
) -> list[str]:
    """Recompute the CPU Flat inner-product initial ranking from fixture data."""
    query_index = list(dataset.query_ids).index(query_id)
    query_vector = dataset.query_vectors[query_index]
    scores = {
        document_id: float(np.dot(document_vector, query_vector))
        for document_id, document_vector in zip(
            dataset.document_ids, dataset.document_vectors, strict=True
        )
    }
    return [
        document_id
        for document_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:initial_k]
    ]


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


def validate_gate1_evidence(payload: Mapping[str, object]) -> None:
    """Fail closed unless a payload contains a coherent, complete A-D run."""
    _require_keys(payload, {"config", "dataset", "environment", "execution", "query_geometry", "runs", "comparisons", "conclusion"}, "payload")
    config = _mapping(payload["config"], "config")
    _require_keys(config, {"seed", "top_k", "initial_k", "num_agents", "steps"}, "config")
    for name in ("seed", "top_k", "initial_k", "num_agents", "steps"):
        _positive_int(config[name], name, allow_zero=name == "seed")
    execution = _mapping(payload["execution"], "execution")
    if not _strict_equal(dict(execution), {
        "force_cpu": True,
        "faiss_factory": "Flat",
        "faiss_metric": "inner_product",
        "pheromone_decay": PHEROMONE_DECAY,
        "exploration_bonus": EXPLORATION_BONUS,
        "clock_mode": DETERMINISTIC_CLOCK_MODE,
        "clock_value": DETERMINISTIC_CLOCK_VALUE,
    }):
        raise ValueError("execution settings are incomplete or inconsistent")

    dataset_payload = _mapping(payload["dataset"], "dataset")
    fixture = dataset_payload.get("fixture", "legacy")
    if not isinstance(fixture, str):
        raise ValueError("dataset.fixture must be a string")
    dataset = build_dataset(fixture, _integer(config["seed"], "seed"))
    required_dataset_keys = {"id", "digest", "document_ids", "query_ids", "document_vector_shape", "query_vector_shape"}
    if "fixture" in dataset_payload:
        required_dataset_keys = required_dataset_keys | {"fixture"}
    _require_keys(dataset_payload, required_dataset_keys, "dataset")
    expected_dataset = {
        "id": dataset.dataset_id,
        "digest": dataset.digest(),
        "document_ids": list(dataset.document_ids),
        "query_ids": list(dataset.query_ids),
        "document_vector_shape": list(dataset.document_vectors.shape),
        "query_vector_shape": list(dataset.query_vectors.shape),
    }
    if "fixture" in dataset_payload:
        expected_dataset["fixture"] = fixture
    if not _strict_equal(dict(dataset_payload), expected_dataset):
        raise ValueError("dataset evidence does not match the configured seed")

    environment = _mapping(payload["environment"], "environment")
    expected_environment = _environment_payload()
    if not _strict_equal(dict(environment), expected_environment):
        raise ValueError("environment provenance is incomplete or inconsistent")
    geometry = _mapping(payload["query_geometry"], "query_geometry")
    if not _strict_equal(dict(geometry), query_geometry(dataset)):
        raise ValueError("query geometry does not match the dataset")

    runs = _mapping(payload["runs"], "runs")
    if list(runs) not in (["A", "B", "C", "D"], ["A", "B", "C", "D", "E"]):
        raise ValueError("runs must be ordered A-D, optionally followed by E")
    present = set(runs)
    for method, query_ids in _RUN_SPECS:
        if method not in present:
            continue
        _validate_run_evidence(
            _mapping(runs[method], f"runs.{method}"),
            dataset,
            method,
            query_ids,
            _integer(config["seed"], "seed"),
            _integer(config["top_k"], "top_k"),
            _integer(config["initial_k"], "initial_k"),
            _integer(config["num_agents"], "num_agents"),
            _integer(config["steps"], "steps"),
        )

    comparisons = _mapping(payload["comparisons"], "comparisons")
    expected_pairs = (
        (("A_vs_C", "A", "C"), ("B_vs_D", "B", "D"), ("A_vs_E", "A", "E"), ("C_vs_E", "C", "E"))
        if "E" in present
        else (("A_vs_C", "A", "C"), ("B_vs_D", "B", "D"))
    )
    if list(comparisons) != [name for name, _left, _right in expected_pairs]:
        raise ValueError("comparisons must match the runs present")
    for name, left, right in expected_pairs:
        _validate_comparison(
            _mapping(comparisons[name], f"comparisons.{name}"),
            _mapping(runs[left], f"runs.{left}"),
            _mapping(runs[right], f"runs.{right}"),
        )
    novel_count = sum(
        len(_mapping(runs[method], f"runs.{method}")["metrics"]["novel_relevant_candidates"])
        for method in ("C", "D")
    )
    conclusion = "novel_relevant_observed" if novel_count > 0 else "no_novel_relevant_observed"
    if payload["conclusion"] != conclusion:
        raise ValueError("conclusion does not match C/D novel relevant evidence")


def _validate_run_evidence(
    run: Mapping[str, object],
    dataset: BenchmarkDataset,
    method: str,
    query_ids: tuple[str, ...],
    seed: int,
    top_k: int,
    initial_k: int,
    num_agents: int,
    steps: int,
) -> None:
    required = {"query_ids", "independent_run_count", "execution_backend", "execution", "raw_ids", "metrics", "candidate_overlap", "timing", "candidate_comparisons", "nearest_search_calls", "mcmp_steps", "document_visits", "pheromone_trails"}
    _require_keys(run, required, f"runs.{method}")
    if not _strict_equal(run["query_ids"], list(query_ids)) or _integer(run["independent_run_count"], "independent_run_count") != len(query_ids):
        raise ValueError(f"runs.{method} has incorrect query cardinality")
    if type(run["execution_backend"]) is not str or run["execution_backend"] != "faiss-cpu":
        raise ValueError(f"runs.{method} did not use the required FAISS CPU backend")
    execution = _mapping(run["execution"], f"runs.{method}.execution")
    expected_execution = _run_execution_snapshot(
        method, query_ids, "faiss-cpu", seed, top_k, initial_k, num_agents, steps
    )
    if not _strict_equal(dict(execution), expected_execution):
        raise ValueError(f"runs.{method} execution snapshot is inconsistent")
    raw_ids = _mapping(run["raw_ids"], f"runs.{method}.raw_ids")
    _require_keys(raw_ids, {"ranked_document_ids", "initial_candidate_ids", "per_query_initial_candidate_ids", "discovered_candidate_ids", "per_query_candidate_ids", "per_query_ranked_document_ids"}, f"runs.{method}.raw_ids")
    documents = set(dataset.document_ids)
    ranked = _string_list(raw_ids["ranked_document_ids"], "ranked_document_ids")
    initial = _string_list(raw_ids["initial_candidate_ids"], "initial_candidate_ids")
    discovered = _string_list(raw_ids["discovered_candidate_ids"], "discovered_candidate_ids")
    # Method E reranks only its FAISS pool, so it cannot return more documents than
    # the pool holds. Every other method ranks against the full corpus.
    expected_ranked_count = min(top_k, initial_k) if method == "E" else top_k
    if len(ranked) != expected_ranked_count or any(len(set(values)) != len(values) for values in (ranked, initial, discovered)) or not set(ranked) <= documents or not set(initial) <= documents or not set(discovered) <= documents:
        raise ValueError(f"runs.{method} contains invalid raw document ids")
    per_query_initials = _mapping(
        raw_ids["per_query_initial_candidate_ids"],
        f"runs.{method}.per_query_initial_candidate_ids",
    )
    if list(per_query_initials) != list(query_ids):
        raise ValueError(f"runs.{method} initial candidates do not match query ids")
    parsed_initials: dict[str, frozenset[str]] = {}
    for query_id, values in per_query_initials.items():
        identifiers = _string_list(values, "per-query initial candidate ids")
        if len(identifiers) != initial_k or len(set(identifiers)) != len(identifiers) or not set(identifiers) <= documents:
            raise ValueError(f"runs.{method} has invalid per-query initial candidates")
        expected_initial = _direct_initial_candidates(dataset, query_id, initial_k)
        if not _strict_equal(identifiers, expected_initial):
            raise ValueError(f"runs.{method} initial candidates do not match direct retrieval")
        parsed_initials[query_id] = frozenset(identifiers)
    if frozenset().union(*parsed_initials.values()) != frozenset(initial):
        raise ValueError(f"runs.{method} initial candidates do not match per-query evidence")
    per_query_candidates: dict[str, frozenset[str]] = {}
    per_query_rankings: dict[str, tuple[str, ...]] = {}
    for key in ("per_query_candidate_ids", "per_query_ranked_document_ids"):
        per_query = _mapping(raw_ids[key], f"runs.{method}.{key}")
        if list(per_query) != list(query_ids):
            raise ValueError(f"runs.{method}.{key} does not match query ids")
        for query_id, values in per_query.items():
            identifiers = _string_list(values, key)
            if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= documents:
                raise ValueError(f"runs.{method}.{key} contains unknown document ids")
            if key == "per_query_candidate_ids":
                per_query_candidates[query_id] = frozenset(identifiers)
            else:
                per_query_rankings[query_id] = tuple(identifiers)
    if frozenset().union(*per_query_candidates.values()) != frozenset(discovered):
        raise ValueError(f"runs.{method} discovered candidates do not match per-query evidence")
    if method in {"A", "B"} and frozenset(initial) != frozenset(discovered):
        raise ValueError(f"runs.{method} baseline candidates must not be novel")
    if any(len(ranking) != expected_ranked_count for ranking in per_query_rankings.values()):
        raise ValueError(f"runs.{method} per-query rankings must equal top_k")
    metrics = _mapping(run["metrics"], f"runs.{method}.metrics")
    _require_keys(metrics, {"recall_at_k", "reciprocal_rank", "mrr", "ndcg_at_k", "unique_relevant_documents", "candidate_count", "novel_candidates", "novel_relevant_candidates"}, f"runs.{method}.metrics")
    for name in ("recall_at_k", "reciprocal_rank", "mrr", "ndcg_at_k"):
        _finite_number(metrics[name], name)
    if _integer(metrics["unique_relevant_documents"], "unique_relevant_documents") < 0 or _integer(metrics["candidate_count"], "candidate_count") != len(discovered) or not set(_string_list(metrics["novel_candidates"], "novel_candidates")) <= set(discovered) or not set(_string_list(metrics["novel_relevant_candidates"], "novel_relevant_candidates")) <= documents:
        raise ValueError(f"runs.{method} has inconsistent metrics")
    overlap = _mapping(run["candidate_overlap"], f"runs.{method}.candidate_overlap")
    if set(overlap) != ({"q-main|q-related"} if len(query_ids) == 2 else set()):
        raise ValueError(f"runs.{method} has incorrect candidate overlap")
    for value in overlap.values():
        _finite_number(value, "candidate overlap")
    timing = _mapping(run["timing"], f"runs.{method}.timing")
    _require_keys(timing, {"elapsed_ms"}, f"runs.{method}.timing")
    if _finite_number(timing["elapsed_ms"], "elapsed_ms") < 0:
        raise ValueError("elapsed_ms must be nonnegative")
    candidate_comparisons = _positive_int(run["candidate_comparisons"], "candidate_comparisons", allow_zero=True)
    nearest_search_calls = _positive_int(run["nearest_search_calls"], "nearest_search_calls", allow_zero=True)
    if nearest_search_calls < len(query_ids):
        raise ValueError("nearest_search_calls is incomplete")
    if candidate_comparisons != nearest_search_calls * len(dataset.document_ids):
        raise ValueError("candidate comparisons do not match nearest-search evidence")
    expected_steps = 0 if method in {"A", "B"} else steps
    if _integer(run["mcmp_steps"], "mcmp_steps") != expected_steps:
        raise ValueError(f"runs.{method} has incorrect mcmp steps")
    visits = _mapping(run["document_visits"], f"runs.{method}.document_visits")
    if method in {"A", "B"} and visits:
        raise ValueError(f"runs.{method} must not report MCMP visits")
    if method in {"C", "D", "E"} and set(visits) != documents:
        raise ValueError(f"runs.{method} must report every document visit count")
    visit_counts = {
        document_id: _positive_int(value, "document visit", allow_zero=True)
        for document_id, value in visits.items()
    }
    pheromone_trails = _positive_int(run["pheromone_trails"], "pheromone_trails", allow_zero=True)
    total_visits = sum(visit_counts.values())
    if method in {"A", "B"}:
        if total_visits != 0 or pheromone_trails != 0:
            raise ValueError(f"runs.{method} must have zero MCMP state")
    else:
        if total_visits != num_agents * steps * len(query_ids):
            raise ValueError(f"runs.{method} visit total does not match execution snapshot")
        if frozenset(document_id for document_id, count in visit_counts.items() if count > 0) != frozenset(discovered):
            raise ValueError(f"runs.{method} discovery does not match positive visit counts")
        if pheromone_trails > total_visits:
            raise ValueError(f"runs.{method} pheromone trails exceed total visits")
    reconstructed = SearchRun(
        method=method,
        query_ids=query_ids,
        ranked_document_ids=tuple(ranked),
        initial_candidate_ids=frozenset(initial),
        discovered_candidate_ids=frozenset(discovered),
        per_query_candidate_ids=per_query_candidates,
        per_query_ranked_document_ids=per_query_rankings,
        elapsed_ms=_finite_number(timing["elapsed_ms"], "elapsed_ms"),
        candidate_comparisons=candidate_comparisons,
        mcmp_steps=_integer(run["mcmp_steps"], "mcmp_steps"),
        document_visits=visit_counts,
        pheromone_trails=pheromone_trails,
        per_query_initial_candidate_ids=parsed_initials,
    )
    if not _strict_equal(dict(metrics), evaluate_run(dataset, reconstructed, top_k)):
        raise ValueError(f"runs.{method} metrics do not match raw evidence")
    if not _strict_equal(dict(overlap), candidate_overlap(reconstructed)):
        raise ValueError(f"runs.{method} candidate overlap does not match raw evidence")


def _validate_comparison(comparison: Mapping[str, object], left: Mapping[str, object], right: Mapping[str, object]) -> None:
    _require_keys(comparison, {"ranked_document_ids_equal", "novel_relevant_candidates", "recall_at_k_delta", "mrr_delta", "ndcg_at_k_delta"}, "comparison")
    left_raw = _mapping(left["raw_ids"], "left raw ids")
    right_raw = _mapping(right["raw_ids"], "right raw ids")
    right_metrics = _mapping(right["metrics"], "right metrics")
    left_metrics = _mapping(left["metrics"], "left metrics")
    if type(comparison["ranked_document_ids_equal"]) is not bool or comparison["ranked_document_ids_equal"] != (left_raw["ranked_document_ids"] == right_raw["ranked_document_ids"]) or not _strict_equal(comparison["novel_relevant_candidates"], right_metrics["novel_relevant_candidates"]):
        raise ValueError("comparison does not match run evidence")
    for name, metric in (("recall_at_k_delta", "recall_at_k"), ("mrr_delta", "mrr"), ("ndcg_at_k_delta", "ndcg_at_k")):
        if _finite_number(comparison[name], name) != _finite_number(right_metrics[metric], metric) - _finite_number(left_metrics[metric], metric):
            raise ValueError("comparison metric delta is inconsistent")


def _require_keys(mapping: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{label} has incomplete or unexpected keys")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _positive_int(value: object, label: str, *, allow_zero: bool) -> int:
    result = _integer(value, label)
    if result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{label} must be positive")
    return result


def _finite_number(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _strict_equal(actual: object, expected: object) -> bool:
    """Compare persisted JSON values without Python's bool/int coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--initial-k", type=int, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", choices=sorted(FIXTURES), default="legacy")
    args = parser.parse_args(argv)
    payload = run_gate1(
        args.seed, args.top_k, args.initial_k, args.num_agents, args.steps, args.fixture
    )
    try:
        write_gate1_result(payload, args.output)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
