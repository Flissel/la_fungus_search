"""Does the walk reach call-graph neighbours that similarity alone misses?

This is **not** Gate 2 stage 2. Stage 1 closed in both measured embedding spaces
(report sections 17 and 18), so no retrieval claim may be drawn from this file.
The question here is different and the gate does not speak to it:

    Of the documents MCMP's walk *visits*, how many are genuine call-graph
    neighbours that the FAISS pool did not already contain?

Ranking is not involved, so the ceiling that dominates sections 14 and 15 cannot
bite. Cost is not involved either: a crawl is batch work, and the ~14 300x that
sinks MCMP as an interactive retriever is affordable when nothing is waiting on it.

What would make the mechanism worth keeping is a *non-empty difference*: relevant
documents the walk finds and the pool does not. Overlap means the call graph and
FAISS between them already have it covered, more cheaply.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.snapshot import load_snapshot
from benchmarks.mcmp.adapters import run_faiss, run_mcmp

TOP_K = 8
INITIAL_K = 8
STEPS = 50


def measure(
    dataset, seed: int, agents: int, expand_every: int, expand_k: int, frontier_cap: int
) -> dict[str, float]:
    query_ids = tuple(dataset.query_ids)
    relevant: set[str] = set()
    for query_id in query_ids:
        relevant |= set(dataset.relevant_by_query.get(query_id, frozenset()))

    faiss_run, _ = run_faiss(dataset, "A", query_ids[:1], TOP_K, INITIAL_K)
    pool = set(faiss_run.initial_candidate_ids)

    walk_run, _ = run_mcmp(
        dataset,
        "G",
        query_ids[:1],
        TOP_K,
        INITIAL_K,
        seed,
        agents,
        STEPS,
        frontier=True,
        expand_every=expand_every,
        expand_k=expand_k,
        frontier_cap=frontier_cap,
    )
    # The visited set, not the ranking. `document_visits` counts every document
    # the colony actually stood on; that is the crawl's output.
    visited = {document_id for document_id, count in walk_run.document_visits.items() if count > 0}

    # Only the first query's relevance, to match the single-query runs above.
    first_relevant = set(dataset.relevant_by_query.get(query_ids[0], frozenset()))
    return {
        "relevant": float(len(first_relevant)),
        "in_pool": float(len(first_relevant & pool)),
        "visited": float(len(visited)),
        "relevant_visited": float(len(first_relevant & visited)),
        # The number that decides whether the walk earns its keep.
        "novel_relevant": float(len(first_relevant & visited - pool)),
        "walk_comparisons": float(walk_run.candidate_comparisons),
        "faiss_comparisons": float(faiss_run.candidate_comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCMP-as-crawler probe on real code")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--agents", type=int, nargs="+", default=[96, 192, 384])
    # Section 13 measured G adding exactly 4.0 documents at every corpus size --
    # the round count binds, never the cap. A crawler needs room to actually
    # crawl, so these are the knobs a fair crawler test has to open.
    parser.add_argument("--expand-every", type=int, default=10)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--frontier-cap", type=int, default=64)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
    print(f"snapshot: {snapshot.backend} / {snapshot.model} / {snapshot.dimension}d")
    print(f"frontier: expand_every={arguments.expand_every} expand_k={arguments.expand_k} "
          f"cap={arguments.frontier_cap}")
    print(f"{'agents':>7}{'relevant':>10}{'in pool':>9}{'visited':>9}"
          f"{'rel.visited':>13}{'NOVEL rel.':>12}{'walk cmp':>12}")

    for agents in arguments.agents:
        rows = []
        for seed in range(arguments.seeds):
            try:
                dataset = build_gate2_dataset(manifest, snapshot, seed)
            except ValueError:
                continue
            rows.append(
                measure(
                    dataset,
                    seed,
                    agents,
                    arguments.expand_every,
                    arguments.expand_k,
                    arguments.frontier_cap,
                )
            )
        if not rows:
            print(f"{agents:>7}   no usable seed")
            continue
        print(
            f"{agents:>7}"
            f"{mean(r['relevant'] for r in rows):>10.2f}"
            f"{mean(r['in_pool'] for r in rows):>9.2f}"
            f"{mean(r['visited'] for r in rows):>9.1f}"
            f"{mean(r['relevant_visited'] for r in rows):>13.2f}"
            f"{mean(r['novel_relevant'] for r in rows):>12.2f}"
            f"{mean(r['walk_comparisons'] for r in rows):>12.0f}"
        )
    print("\nNOVEL rel. = call-graph neighbours the walk visited that the FAISS pool")
    print("did not contain. Zero means the walk found nothing the pool did not have.")


if __name__ == "__main__":
    main()
