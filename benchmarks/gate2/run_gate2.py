"""Stage 2: run A-E over a real Gate 2 dataset across an agent budget sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.gate2.geometry import characterise, stage_two_is_justified
from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.snapshot import load_snapshot
from benchmarks.mcmp.adapters import run_faiss, run_mcmp
from benchmarks.mcmp.contracts import BenchmarkDataset
from benchmarks.mcmp.metrics import evaluate_run

AGENT_SWEEP = (24, 48, 96, 192, 384)

_RUN_SPECS = (("A", 1), ("B", 2), ("C", 1), ("D", 2), ("E", 1))


def run_retrieval(
    dataset: BenchmarkDataset,
    top_k: int,
    initial_k: int,
    seed: int,
    num_agents: int,
    steps: int,
) -> dict[str, object]:
    """Run every method once and report discovery separately from ranking."""
    runs: dict[str, object] = {}
    for method, query_count in _RUN_SPECS:
        query_ids = tuple(dataset.query_ids[:query_count])
        if method in {"A", "B"}:
            run, _evidence = run_faiss(dataset, method, query_ids, top_k, initial_k)
        else:
            run, _evidence = run_mcmp(
                dataset, method, query_ids, top_k, initial_k, seed, num_agents, steps,
                pool_only=method == "E",
            )
        relevant: set[str] = set()
        for query_id in query_ids:
            relevant |= set(dataset.relevant_by_query[query_id])
        metrics = evaluate_run(dataset, run, top_k)
        runs[method] = {
            "metrics": {key: value for key, value in metrics.items()},
            "discovered_relevant": len(set(run.discovered_candidate_ids) & relevant),
            "ranked_relevant": len(set(run.ranked_document_ids) & relevant),
            "relevant_total": len(relevant),
            "candidate_comparisons": run.candidate_comparisons,
            "ranked_document_ids": list(run.ranked_document_ids),
        }
    return {
        "config": {
            "top_k": top_k,
            "initial_k": initial_k,
            "seed": seed,
            "num_agents": num_agents,
            "steps": steps,
        },
        "dataset_id": dataset.dataset_id,
        "dataset_digest": dataset.digest(),
        "runs": runs,
    }


def write_result(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--initial-k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    snapshot = load_snapshot(args.snapshot, manifest_digest(manifest))
    dataset = build_gate2_dataset(manifest, snapshot, args.seed)

    geometry = characterise(dataset, top_k=args.top_k)
    write_result(geometry, args.output_dir / f"geometry-seed-{args.seed}.json")
    if not stage_two_is_justified(float(geometry["manifold_signature"])):
        parser.exit(
            0,
            "stage 1 signature below the pre-registered gate; stage 2 not run\n",
        )

    for num_agents in AGENT_SWEEP:
        payload = run_retrieval(
            dataset, args.top_k, args.initial_k, args.seed, num_agents, args.steps
        )
        write_result(
            payload, args.output_dir / f"retrieval-seed-{args.seed}-a{num_agents}.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
