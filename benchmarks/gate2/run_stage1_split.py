"""Stage 1 under the split-sample protocol report section 16 makes binding.

The operating point `(knn_k, max_hops)` is not knowable in advance on a real
corpus, and picking it by scanning a grid for the most agreeable number is
post-hoc selection over that grid -- measured on the synthetic control as worth up
to +0.273 of apparent gap with nothing planted to find. So the grid is scanned on
one half of the query pairs and the gate is scored on a disjoint half.

The criterion was fixed before this ran, in report section 16.3 and in the Gate 2
design spec:

1. The quantity is the **real-minus-null `reach_given_far` gap**, not the raw
   reachability and not the signature. Reachability the null achieves just as
   easily is not evidence at any absolute value.
2. `knn_k` is drawn from the small end -- a dense graph reaches everything,
   related or not, which inflates the null rather than the signal.
3. Ties break toward the smaller `knn_k`, then the smaller `max_hops`. Stated
   here so a tie cannot be resolved by preference after the fact.

The halves are disjoint in *queries*, not merely in seeds. `build_gate2_dataset`
draws two query documents per seed from the same candidate pool, so disjoint seed
ranges would still share query documents and leak the selection into the
evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.gate2.geometry import (
    NULL_PERMUTATIONS,
    characterise,
    geometry_cache,
    permuted_labels,
    stage_two_is_justified,
)
from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.run_gate2 import characterise_pooled
from benchmarks.probes.sibling_oracle import build_sibling_dataset, sibling_map
from benchmarks.gate2.snapshot import load_snapshot

# Pre-registered grid. knn_k stops at 8 because section 16 measured the null
# inflating from there upward; including 12 would only offer a worse operating
# point a maximiser could still pick.
KNN_GRID = (2, 3, 4, 6, 8)
HOPS_GRID = (2, 3, 4, 6, 8)
SELECTION_PERMUTATIONS = 20


_BUILD = {"callgraph": None}


def _dataset(manifest, snapshot, seed):
    """The relation under measurement, chosen once in main()."""
    builder = _BUILD["callgraph"]
    if builder is None:
        return build_gate2_dataset(manifest, snapshot, seed)
    return builder(manifest, snapshot, seed)


def partition_seeds(
    manifest: object, snapshot: object, seed_count: int
) -> tuple[list[int], list[int], list[int]]:
    """Split seeds into two halves whose query documents do not overlap."""
    usable: list[tuple[int, frozenset[str]]] = []
    unusable: list[int] = []
    for seed in range(seed_count):
        try:
            dataset = _dataset(manifest, snapshot, seed)
        except ValueError:
            unusable.append(seed)
            continue
        usable.append((seed, frozenset(dataset.query_ids)))

    selection: list[int] = []
    evaluation: list[int] = []
    selection_queries: set[str] = set()
    evaluation_queries: set[str] = set()
    dropped: list[int] = list(unusable)
    for seed, queries in usable:
        to_selection = len(selection) <= len(evaluation)
        first, second = (
            (selection, selection_queries, evaluation_queries),
            (evaluation, evaluation_queries, selection_queries),
        )
        if not to_selection:
            first, second = second, first
        for bucket, own, other in (first, second):
            if not (queries & other):
                bucket.append(seed)
                own |= queries
                break
        else:
            # The pair straddles both halves; keeping it would leak.
            dropped.append(seed)
    return selection, evaluation, dropped


def _reach_given_far(report: dict[str, object]) -> float | None:
    far = int(report["far_count"])
    if far == 0:
        return None
    return int(report["far_and_reachable_count"]) / far


def sweep_selection(
    manifest: object,
    snapshot: object,
    seeds: list[int],
    top_k: int,
    hop_threshold: float,
    null_seed: int,
) -> list[dict[str, object]]:
    """Score every grid point on the selection half."""
    results: list[dict[str, object]] = []
    datasets = [_dataset(manifest, snapshot, seed) for seed in seeds]
    for knn_k in KNN_GRID:
        caches = [geometry_cache(dataset, knn_k) for dataset in datasets]
        for max_hops in HOPS_GRID:
            real_far = real_hits = 0
            for dataset, cache in zip(datasets, caches):
                report = characterise(
                    dataset, top_k, knn_k, max_hops, hop_threshold, cache=cache
                )
                real_far += int(report["far_count"])
                real_hits += int(report["far_and_reachable_count"])
            rng = np.random.default_rng(null_seed)
            null_rates: list[float] = []
            for _ in range(SELECTION_PERMUTATIONS):
                null_far = null_hits = 0
                for dataset, cache in zip(datasets, caches):
                    report = characterise(
                        permuted_labels(dataset, rng),
                        top_k,
                        knn_k,
                        max_hops,
                        hop_threshold,
                        cache=cache,
                    )
                    null_far += int(report["far_count"])
                    null_hits += int(report["far_and_reachable_count"])
                if null_far:
                    null_rates.append(null_hits / null_far)
            real = real_hits / real_far if real_far else None
            null = float(np.mean(null_rates)) if null_rates else None
            results.append(
                {
                    "knn_k": knn_k,
                    "max_hops": max_hops,
                    "real_reach_given_far": real,
                    "null_reach_given_far": null,
                    "gap": (real - null) if real is not None and null is not None else None,
                }
            )
    return results


def choose_operating_point(sweep: list[dict[str, object]]) -> dict[str, object]:
    """Maximise the gap; ties break to smaller knn_k, then smaller max_hops."""
    scored = [row for row in sweep if row["gap"] is not None]
    if not scored:
        raise ValueError("no grid point produced a measurable gap on the selection half")
    return min(
        scored,
        key=lambda row: (-float(row["gap"]), int(row["knn_k"]), int(row["max_hops"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-sample Gate 2 stage 1")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hop-threshold", type=float, default=0.0)
    parser.add_argument("--null-seed", type=int, default=0)
    parser.add_argument("--null-permutations", type=int, default=NULL_PERMUTATIONS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", choices=("callgraph", "siblings"), default="callgraph")
    parser.add_argument("--shared-minimum", type=int, default=1)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    # Fail-closed by design: a snapshot built for a different manifest is
    # rejected here rather than producing a quietly misaligned measurement.
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
    if arguments.oracle == "siblings":
        siblings = sibling_map(manifest, arguments.shared_minimum)
        _BUILD["callgraph"] = lambda m, s, seed: build_sibling_dataset(m, s, siblings, seed)
        print(f"oracle          : siblings, shared>={arguments.shared_minimum}, "
              f"{len(siblings)} documents")
    else:
        print("oracle          : call graph")

    selection, evaluation, dropped = partition_seeds(
        manifest, snapshot, arguments.seed_count
    )
    print(f"selection seeds : {len(selection)}")
    print(f"evaluation seeds: {len(evaluation)}")
    print(f"dropped         : {len(dropped)} (unusable or query-overlapping)")
    if not selection or not evaluation:
        raise SystemExit("split produced an empty half; raise --seed-count")

    sweep = sweep_selection(
        manifest,
        snapshot,
        selection,
        arguments.top_k,
        arguments.hop_threshold,
        arguments.null_seed,
    )
    print("\nselection-half sweep: real / null / gap")
    print(f"{'knn_k':>6}" + "".join(f"{f'hops={h}':>24}" for h in HOPS_GRID))
    for knn_k in KNN_GRID:
        cells = []
        for max_hops in HOPS_GRID:
            row = next(
                r for r in sweep if r["knn_k"] == knn_k and r["max_hops"] == max_hops
            )
            if row["gap"] is None:
                cells.append("   --  /  --  /   --  ")
            else:
                cells.append(
                    f"{row['real_reach_given_far']:6.3f}/"
                    f"{row['null_reach_given_far']:6.3f}/{row['gap']:+7.3f}"
                )
        print(f"{knn_k:>6}" + "".join(f"{c:>24}" for c in cells))

    point = choose_operating_point(sweep)
    print(
        f"\noperating point : knn_k={point['knn_k']} max_hops={point['max_hops']} "
        f"(selection gap {point['gap']:+.3f})"
    )

    print("\nevaluation half, pre-registered null:")
    geometry = characterise_pooled(
        manifest,
        snapshot,
        stage1_seeds=0,
        top_k=arguments.top_k,
        knn_k=int(point["knn_k"]),
        max_hops=int(point["max_hops"]),
        hop_threshold=arguments.hop_threshold,
        null_permutations=arguments.null_permutations,
        null_seed=arguments.null_seed,
        seeds=evaluation,
    )
    justified = stage_two_is_justified(
        float(geometry["manifold_signature"]), list(geometry["null_signatures"])
    )
    for key in (
        "pair_count",
        "far_rate",
        "reach_given_far",
        "null_reach_given_far_median",
        "manifold_signature",
        "null_median",
        "null_p95",
        "excess_over_null_median",
        "required_excess",
        "exceeds_null_p95",
        "meets_absolute_minimum",
        "meets_relative_excess",
    ):
        print(f"  {key:30} {geometry[key]}")
    print(f"\n  STAGE 2 JUSTIFIED: {justified}")

    payload = {
        "protocol": "split-sample, criterion pre-registered in report section 16.3",
        "oracle": arguments.oracle,
        "shared_minimum": arguments.shared_minimum if arguments.oracle == "siblings" else None,
        "snapshot_backend": geometry["snapshot_backend"],
        "snapshot_model": geometry["snapshot_model"],
        "selection_seeds": selection,
        "evaluation_seeds": evaluation,
        "dropped_seeds": dropped,
        "selection_sweep": sweep,
        "operating_point": point,
        "evaluation": geometry,
        "stage_two_justified": justified,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {arguments.output}")


if __name__ == "__main__":
    main()
