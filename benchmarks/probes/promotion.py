"""Two measurements that decide whether any further MCMP work is justified.

Section 24 established that method C discovers a quarter more relevant documents
than FAISS and that not one of them enters the top 8. Section 23 established that
no visit-term change moves recall. Those two are in tension: there *is* extra
material to promote and nothing promotes it. Section 24.4 left the reason
unmeasured. This measures it.

**Question 1 -- where do the extra discoveries actually rank?** For every relevant
document, its position in FAISS's ordering and in MCMP's. If MCMP moves a document
from rank 40 to rank 12, a better reranker closes the gap and sections 14-15 still
matter. If it leaves it at 40, nothing will, and the ranking work is finished.

**Question 2 -- does the bounded frontier discover what the full corpus does?**
Method G was absent from `run_gate2._RUN_SPECS`, so its *discovery* on real code
has never been measured; section 23 only compared recall. C pays 57.2 million
comparisons for its extra 0.12 relevant documents per query. G pays 147 761 --
1/400th. If G discovers comparably, the economics change from absurd to arguable.

Ranks are taken over the full corpus ordering: `run_mcmp` truncates
`ranked_document_ids` at `top_k`, so `top_k` is set to the corpus size here. It
does not enter the walk -- only `initial_k` shapes the pool -- so this changes what
is reported, not what is run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.snapshot import load_snapshot
from benchmarks.mcmp.adapters import run_faiss, run_mcmp

INITIAL_K = 8
POOL_K = 8
STEPS = 50


def _ranks(ranked: list[str], targets: set[str]) -> dict[str, int]:
    position = {document_id: index + 1 for index, document_id in enumerate(ranked)}
    return {document_id: position[document_id] for document_id in targets if document_id in position}


def main() -> None:
    parser = argparse.ArgumentParser(description="promotion and frontier-discovery probe")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--agents", type=int, default=96)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))

    moved: list[tuple[int, int]] = []      # (faiss rank, mcmp rank) for far relevant docs
    discovery = {"A": [], "C": [], "G": []}
    costs = {"A": [], "C": [], "G": []}

    for seed in range(arguments.seeds):
        try:
            dataset = build_gate2_dataset(manifest, snapshot, seed)
        except ValueError:
            continue
        query_ids = tuple(dataset.query_ids[:1])
        relevant = set(dataset.relevant_by_query[query_ids[0]])
        if not relevant:
            continue
        full_k = len(dataset.document_ids)

        faiss_run, _ = run_faiss(dataset, "A", query_ids, full_k, INITIAL_K)
        faiss_ranks = _ranks(list(faiss_run.ranked_document_ids), relevant)
        pool_run, _ = run_faiss(dataset, "A", query_ids, POOL_K, INITIAL_K)
        discovery["A"].append(len(set(pool_run.discovered_candidate_ids) & relevant))
        costs["A"].append(pool_run.candidate_comparisons)

        walk_run, _ = run_mcmp(
            dataset, "C", query_ids, full_k, INITIAL_K, seed, arguments.agents, STEPS
        )
        walk_ranks = _ranks(list(walk_run.ranked_document_ids), relevant)
        discovery["C"].append(len(set(walk_run.discovered_candidate_ids) & relevant))
        costs["C"].append(walk_run.candidate_comparisons)

        frontier_run, _ = run_mcmp(
            dataset, "G", query_ids, POOL_K, INITIAL_K, seed, arguments.agents, STEPS,
            frontier=True,
        )
        discovery["G"].append(len(set(frontier_run.discovered_candidate_ids) & relevant))
        costs["G"].append(frontier_run.candidate_comparisons)

        for document_id, faiss_rank in faiss_ranks.items():
            if faiss_rank > POOL_K and document_id in walk_ranks:
                moved.append((faiss_rank, walk_ranks[document_id]))

    print(f"snapshot : {snapshot.backend} / {snapshot.model} / {snapshot.dimension}d")
    print(f"corpus   : {len(snapshot.document_ids)} documents, {arguments.agents} agents")

    print(f"\n=== Question 2: discovery and cost ===")
    print(f"{'method':>7}{'relevant discovered':>22}{'comparisons':>14}")
    for method in ("A", "C", "G"):
        if discovery[method]:
            print(f"{method:>7}{mean(discovery[method]):>22.2f}{mean(costs[method]):>14.0f}")

    print(f"\n=== Question 1: where relevant documents FAISS misses end up ===")
    if not moved:
        print("  no relevant document ranked deeper than top-8 in this sample")
        return
    improved = [(f, m) for f, m in moved if m < f]
    worsened = [(f, m) for f, m in moved if m > f]
    print(f"  far relevant documents          {len(moved)}")
    print(f"  median FAISS rank               {median(f for f, _ in moved):.0f}")
    print(f"  median MCMP rank                {median(m for _, m in moved):.0f}")
    print(f"  MCMP ranks it better            {len(improved)}")
    print(f"  MCMP ranks it worse             {len(worsened)}")
    reachable = [m for _, m in moved if m <= 16]
    print(f"  MCMP puts it in the top 16      {len(reachable)} of {len(moved)}")
    print(f"\n  the ten shallowest (FAISS rank -> MCMP rank):")
    for faiss_rank, mcmp_rank in sorted(moved)[:10]:
        arrow = "better" if mcmp_rank < faiss_rank else ("worse" if mcmp_rank > faiss_rank else "same")
        print(f"    {faiss_rank:>5} -> {mcmp_rank:>5}   {arrow}")


if __name__ == "__main__":
    main()
