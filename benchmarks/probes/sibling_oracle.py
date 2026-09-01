"""The one niche sections 17-19 left untested: related *without* a call edge.

Those sections used the call graph as the relevance oracle, which makes them
blind by construction to the category MCMP's remaining case rests on -- two
documents that belong together semantically while neither calls the other. A walk
that found exactly those would have scored zero there. Section 19.3 says so; this
module removes the excuse by supplying a second oracle for that relation, drawn
mechanically from the same manifest and requiring no judgement.

**The sibling relation.** Two documents are siblings when they share at least
``SHARED_MINIMUM`` callees or at least ``SHARED_MINIMUM`` callers *and* neither
calls the other. Sharing two callees is a real structural statement -- two
functions built out of the same pieces -- while sharing one could be `len`. Direct
call-graph neighbours are excluded outright, so the two oracles are disjoint by
construction and this cannot re-measure section 17 under a new name.

Measured on `embeddinggemma-local-v1`: 34 documents have siblings at threshold 2
(164 directed pairs, median 5 each), 128 at threshold 1 (872 pairs).

Two questions, the same two as before:

1. Geometry -- are siblings far, and are they chain-reachable above chance?
2. Crawl -- does the walk actually visit siblings the FAISS pool did not hold?
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np

from benchmarks.gate2.geometry import characterise, geometry_cache, permuted_labels
from benchmarks.gate2.manifest import Manifest, load_manifest, manifest_digest, relevant_for
from benchmarks.gate2.provider import QUERY_PREFIX
from benchmarks.gate2.snapshot import Snapshot, load_snapshot
from benchmarks.mcmp.adapters import run_faiss, run_mcmp
from benchmarks.mcmp.contracts import BenchmarkDataset

SHARED_MINIMUM = 2
TOP_K = 8
INITIAL_K = 8
STEPS = 50
NULL_PERMUTATIONS = 100


def sibling_map(manifest: Manifest, shared_minimum: int) -> dict[str, frozenset[str]]:
    """Documents sharing callees or callers, with direct call edges removed."""
    ids = [document.document_id for document in manifest.documents]
    callees = {i: set(manifest.callees_by_document.get(i, frozenset())) for i in ids}
    callers = {i: set(manifest.callers_by_document.get(i, frozenset())) for i in ids}
    direct = {i: set(relevant_for(manifest, i)) for i in ids}
    siblings: dict[str, frozenset[str]] = {}
    for a in ids:
        found = {
            b
            for b in ids
            if b != a
            and b not in direct[a]
            and (
                len(callees[a] & callees[b]) >= shared_minimum
                or len(callers[a] & callers[b]) >= shared_minimum
            )
        }
        if found:
            siblings[a] = frozenset(found)
    return siblings


def build_sibling_dataset(
    manifest: Manifest,
    snapshot: Snapshot,
    siblings: dict[str, frozenset[str]],
    seed: int,
) -> BenchmarkDataset:
    """Mirror `build_gate2_dataset`, with sibling relevance instead of call edges."""
    candidates = sorted(siblings)
    if len(candidates) < 2:
        raise ValueError("fewer than two documents have siblings")
    rng = np.random.default_rng(seed)
    chosen = [candidates[int(i)] for i in rng.choice(len(candidates), size=2, replace=False)]

    index_by_id = {document_id: position for position, document_id in enumerate(snapshot.document_ids)}
    # The query's own document must leave the corpus, exactly as in the Gate 2
    # provider: otherwise it ranks first at similarity 1.0 and is never relevant.
    corpus_ids = tuple(
        document.document_id
        for document in manifest.documents
        if document.document_id not in set(chosen)
    )
    corpus_set = set(corpus_ids)
    relevant_by_query = {
        f"{QUERY_PREFIX}{document_id}": frozenset(siblings[document_id] & corpus_set)
        for document_id in chosen
    }
    if any(not value for value in relevant_by_query.values()):
        raise ValueError("a chosen query has no sibling left in the corpus")

    dataset = BenchmarkDataset(
        dataset_id=f"siblings-{manifest.manifest_id}",
        seed=seed,
        document_ids=corpus_ids,
        document_vectors=np.stack([snapshot.vectors[index_by_id[i]] for i in corpus_ids]).astype(np.float32),
        query_ids=tuple(f"{QUERY_PREFIX}{i}" for i in chosen),
        query_vectors=np.stack([snapshot.vectors[index_by_id[i]] for i in chosen]).astype(np.float32),
        relevant_by_query=relevant_by_query,
    )
    dataset.validate()
    return dataset


def _datasets(manifest, snapshot, siblings, seed_count):
    built = []
    for seed in range(seed_count):
        try:
            built.append((seed, build_sibling_dataset(manifest, snapshot, siblings, seed)))
        except ValueError:
            continue
    return built


def geometry(built, knn_k: int, max_hops: int, null_seed: int) -> None:
    """Are siblings far, and reachable above chance?"""
    caches = [(seed, dataset, geometry_cache(dataset, knn_k)) for seed, dataset in built]
    pairs = []
    for _seed, dataset, cache in caches:
        pairs.extend(characterise(dataset, TOP_K, knn_k, max_hops, 0.0, cache=cache)["pairs"])
    total = len(pairs)
    far = sum(1 for p in pairs if p["far"])
    hits = sum(1 for p in pairs if p["far"] and p["chain_reachable"] is True)

    rng = np.random.default_rng(null_seed)
    null_signatures, null_reach = [], []
    for _ in range(NULL_PERMUTATIONS):
        null_pairs = []
        for _seed, dataset, cache in caches:
            null_pairs.extend(
                characterise(permuted_labels(dataset, rng), TOP_K, knn_k, max_hops, 0.0, cache=cache)["pairs"]
            )
        n_total = len(null_pairs)
        n_far = sum(1 for p in null_pairs if p["far"])
        n_hits = sum(1 for p in null_pairs if p["far"] and p["chain_reachable"] is True)
        null_signatures.append(n_hits / n_total if n_total else 0.0)
        null_reach.append(n_hits / n_far if n_far else 0.0)

    signature = hits / total if total else 0.0
    print(f"  pair_count                  {total}")
    print(f"  far_rate                    {far / total if total else 0.0:.3f}")
    print(f"  reach_given_far             {hits / far if far else 0.0:.3f}")
    print(f"  null reach_given_far median {float(np.median(null_reach)):.3f}")
    print(f"  signature                   {signature:.3f}")
    print(f"  null median / p95           {float(np.median(null_signatures)):.3f}"
          f" / {float(np.quantile(null_signatures, 0.95)):.3f}")
    print(f"  exceeds null p95            {signature > float(np.quantile(null_signatures, 0.95))}")


def crawl(built, agents: int, expand_every: int, expand_k: int, frontier_cap: int) -> None:
    """Does the walk visit siblings the FAISS pool did not already hold?"""
    rows = []
    for seed, dataset in built:
        query_ids = tuple(dataset.query_ids)
        faiss_run, _ = run_faiss(dataset, "A", query_ids[:1], TOP_K, INITIAL_K)
        pool = set(faiss_run.initial_candidate_ids)
        walk_run, _ = run_mcmp(
            dataset, "G", query_ids[:1], TOP_K, INITIAL_K, seed, agents, STEPS,
            frontier=True, expand_every=expand_every, expand_k=expand_k,
            frontier_cap=frontier_cap,
        )
        visited = {i for i, c in walk_run.document_visits.items() if c > 0}
        relevant = set(dataset.relevant_by_query[query_ids[0]])
        # The null for the crawl, without which "0.25 novel" is not a claim: the
        # walk and the pool are held exactly as measured and only the *labels*
        # move, drawn at the same size from the same corpus. Whatever a walk of
        # this shape would collect by chance shows up here.
        rng = np.random.default_rng(seed)
        corpus = list(dataset.document_ids)
        null_novel = []
        for _ in range(NULL_PERMUTATIONS):
            drawn = {
                corpus[int(i)]
                for i in rng.choice(len(corpus), size=len(relevant), replace=False)
            }
            null_novel.append(len(drawn & visited - pool))
        rows.append(
            {
                "relevant": len(relevant),
                "in_pool": len(relevant & pool),
                "visited": len(visited),
                "relevant_visited": len(relevant & visited),
                "novel_relevant": len(relevant & visited - pool),
                "null_novel": float(np.mean(null_novel)),
            }
        )
    if not rows:
        print("  no usable seed")
        return
    print(f"  relevant per query          {mean(r['relevant'] for r in rows):.2f}")
    print(f"  of those in FAISS pool      {mean(r['in_pool'] for r in rows):.2f}")
    print(f"  documents visited           {mean(r['visited'] for r in rows):.1f}")
    print(f"  relevant visited            {mean(r['relevant_visited'] for r in rows):.2f}")
    observed = mean(r["novel_relevant"] for r in rows)
    chance = mean(r["null_novel"] for r in rows)
    print(f"  NOVEL relevant              {observed:.3f}")
    print(f"  NOVEL under permuted labels {chance:.3f}")
    print(f"  ratio                       {observed / chance if chance else float('inf'):.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sibling-relation oracle probe")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--shared-minimum", type=int, default=SHARED_MINIMUM)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--agents", type=int, default=192)
    parser.add_argument("--expand-every", type=int, default=2)
    parser.add_argument("--expand-k", type=int, default=12)
    parser.add_argument("--frontier-cap", type=int, default=200)
    parser.add_argument("--null-seed", type=int, default=0)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
    siblings = sibling_map(manifest, arguments.shared_minimum)
    built = _datasets(manifest, snapshot, siblings, arguments.seeds)

    print(f"snapshot : {snapshot.backend} / {snapshot.model} / {snapshot.dimension}d")
    print(f"oracle   : siblings sharing >={arguments.shared_minimum} callees or callers, "
          f"no direct call edge")
    print(f"           {len(siblings)} documents have siblings; {len(built)} usable seeds")
    print(f"\ngeometry (knn_k={arguments.knn_k}, max_hops={arguments.max_hops}):")
    geometry(built, arguments.knn_k, arguments.max_hops, arguments.null_seed)
    print(f"\ncrawl (agents={arguments.agents}, expand_every={arguments.expand_every}, "
          f"expand_k={arguments.expand_k}, cap={arguments.frontier_cap}):")
    crawl(built, arguments.agents, arguments.expand_every, arguments.expand_k, arguments.frontier_cap)


if __name__ == "__main__":
    main()
