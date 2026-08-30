"""Stage 2: run A-E over a real Gate 2 dataset across an agent budget sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.gate2.geometry import characterise, stage_two_is_justified
from benchmarks.gate2.manifest import Manifest, load_manifest, manifest_digest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.snapshot import Snapshot, load_snapshot
from benchmarks.mcmp.adapters import run_faiss, run_mcmp
from benchmarks.mcmp.contracts import BenchmarkDataset
from benchmarks.mcmp.metrics import evaluate_run

AGENT_SWEEP = (24, 48, 96, 192, 384)

_RUN_SPECS = (("A", 1), ("B", 2), ("C", 1), ("D", 2), ("E", 1))


def characterise_pooled(
    manifest: Manifest,
    snapshot: Snapshot,
    stage1_seeds: int,
    top_k: int,
    knn_k: int,
    max_hops: int,
    hop_threshold: float,
) -> dict[str, object]:
    """Pool stage 1 across seeds and score the pre-registered gate on the pool.

    The gate is pre-registered over "all (query, relevant) pairs". A single
    dataset is two queries, which is 2-4 pairs -- a population so small that
    the signature quantises to {0, .25, .5, .75, 1} and every threshold in
    (0, 0.25] decides identically. That does not wash out on the real corpus:
    two queries give the same 2-4 pairs over 249 documents as over 13.

    Pooling makes the measurement match the pre-registration; it does not
    change what is measured. ``characterise`` is therefore untouched, and each
    seed keeps its own summary alongside the pooled one.
    """
    if stage1_seeds < 1:
        raise ValueError("--stage1-seeds must be at least 1")

    per_seed: list[dict[str, object]] = []
    skipped_seeds: list[int] = []
    pooled_pairs: list[dict[str, object]] = []
    dataset_id = ""
    last_skip: ValueError | None = None

    for seed in range(stage1_seeds):
        try:
            dataset = build_gate2_dataset(manifest, snapshot, seed)
        except ValueError as error:
            # No query pair on this seed leaves both queries a non-empty
            # relevant set. Recorded in the evidence, not silently dropped.
            skipped_seeds.append(seed)
            last_skip = error
            continue
        report = characterise(
            dataset,
            top_k=top_k,
            knn_k=knn_k,
            max_hops=max_hops,
            hop_threshold=hop_threshold,
        )
        dataset_id = str(report["dataset_id"])
        per_seed.append(
            {
                "seed": seed,
                "pair_count": report["pair_count"],
                "far_count": report["far_count"],
                "far_and_reachable_count": report["far_and_reachable_count"],
                "manifold_signature": report["manifold_signature"],
            }
        )
        for pair in report["pairs"]:
            pooled_pairs.append({**pair, "seed": seed})

    if not per_seed:
        # Every seed failing is a structural fault in the manifest or snapshot,
        # not a run to report a 0.0 signature for. The last seed's own error is
        # chained so the real cause is not swallowed by the summary.
        raise ValueError(
            f"every stage 1 seed in range(0, {stage1_seeds}) was skipped, so the "
            "pooled manifold signature has no population to measure; the last "
            f"seed failed with: {last_skip}"
        ) from last_skip

    pair_count = len(pooled_pairs)
    far_count = sum(1 for pair in pooled_pairs if pair["far"])
    far_and_reachable = sum(
        1 for pair in pooled_pairs if pair["far"] and pair["chain_reachable"] is True
    )
    signature = far_and_reachable / pair_count if pair_count else 0.0
    return {
        "config": {
            "top_k": top_k,
            "knn_k": knn_k,
            "max_hops": max_hops,
            "hop_threshold": hop_threshold,
            "stage1_seeds": stage1_seeds,
        },
        "dataset_id": dataset_id,
        # Provenance, so a stub-derived evidence file and a production one are
        # not indistinguishable by inspection.
        "snapshot_backend": snapshot.backend,
        "snapshot_model": snapshot.model,
        "manifest_digest": snapshot.manifest_digest,
        "pair_count": pair_count,
        "far_count": far_count,
        "far_and_reachable_count": far_and_reachable,
        "manifold_signature": signature,
        "per_seed": per_seed,
        "skipped_seeds": skipped_seeds,
        "pairs": pooled_pairs,
    }


def run_retrieval(
    dataset: BenchmarkDataset,
    snapshot: Snapshot,
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
        # Provenance: the design's non-goal is explicit that no Gate 2
        # conclusion may be drawn from the test snapshot, which is
        # unenforceable if the evidence files do not say which one produced
        # them.
        "snapshot_backend": snapshot.backend,
        "snapshot_model": snapshot.model,
        "manifest_digest": snapshot.manifest_digest,
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
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--hop-threshold", type=float, default=0.0)
    parser.add_argument("--stage1-seeds", type=int, default=12)
    parser.add_argument("--initial-k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    snapshot = load_snapshot(args.snapshot, manifest_digest(manifest))

    geometry = characterise_pooled(
        manifest,
        snapshot,
        stage1_seeds=args.stage1_seeds,
        top_k=args.top_k,
        knn_k=args.knn_k,
        max_hops=args.max_hops,
        hop_threshold=args.hop_threshold,
    )
    write_result(geometry, args.output_dir / f"geometry-seed-{args.seed}.json")
    if not stage_two_is_justified(float(geometry["manifold_signature"])):
        parser.exit(
            0,
            "stage 1 signature below the pre-registered gate; stage 2 not run\n",
        )

    dataset = build_gate2_dataset(manifest, snapshot, args.seed)
    for num_agents in AGENT_SWEEP:
        payload = run_retrieval(
            dataset, snapshot, args.top_k, args.initial_k, args.seed, num_agents, args.steps
        )
        write_result(
            payload, args.output_dir / f"retrieval-seed-{args.seed}-a{num_agents}.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
