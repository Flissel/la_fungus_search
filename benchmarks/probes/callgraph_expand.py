"""A: deterministic call-graph expansion of the FAISS hits.

The strongest number in the whole investigation: ~60% of relevant documents are
call-graph neighbours ranked deeper than the FAISS top-8 (§17, §22). Every
stochastic attempt to reach them is measured shut (§25, §26), and the §26 spec
outcome names the reason a different approach must win: on a known graph, direct
queries beat walks. So: expand the *hits*, deterministically, one call-hop.

    FAISS top-8  ->  each hit contributes its callers and callees  ->  candidates

**No leakage.** Relevance is defined by the *query document's* edges. Expansion
only ever reads the edges of retrieved hits; the query document is not in the
corpus and its edge list is never consulted. An edge (hit, r) is genuine graph
information, not the label (query, r).

Primary metric: the equal-budget test that decided §26 — does the candidate set
hold more relevant documents than similarity's top-that-many? Secondary:
recall@k with candidates ranked by cosine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np

from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import QUERY_PREFIX, build_gate2_dataset
from benchmarks.gate2.snapshot import load_snapshot

INITIAL_K = 8


def main() -> None:
    parser = argparse.ArgumentParser(description="call-graph expansion of FAISS hits")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--hops", type=int, default=1)
    # The base retriever whose hits get expanded. bm25 exists because the C
    # probe measured it beating dense by 15 points on this oracle; expansion
    # should be tested on the strongest base, not the weakest.
    parser.add_argument("--base", choices=("dense", "bm25"), default="dense")
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
    bm25 = None
    bm25_position: dict[str, int] = {}
    source_of: dict[str, str] = {}
    if arguments.base == "bm25":
        from embeddinggemma.bm25_lite import BM25Lite

        source_of = {document.document_id: document.source for document in manifest.documents}
        fit_ids = [document.document_id for document in manifest.documents]
        bm25 = BM25Lite()
        bm25.fit([source_of[document_id] for document_id in fit_ids])
        bm25_position = {document_id: index for index, document_id in enumerate(fit_ids)}
    neighbours = {
        document.document_id: (
            set(manifest.callees_by_document.get(document.document_id, frozenset()))
            | set(manifest.callers_by_document.get(document.document_id, frozenset()))
        )
        for document in manifest.documents
    }

    rows = []
    for seed in range(arguments.seeds):
        try:
            dataset = build_gate2_dataset(manifest, snapshot, seed)
        except ValueError:
            continue
        query_id = dataset.query_ids[0]
        relevant = set(dataset.relevant_by_query[query_id])
        if not relevant:
            continue
        corpus = set(dataset.document_ids)
        query_vector = dataset.query_vectors[0]
        if arguments.base == "bm25":
            query_document = query_id[len(QUERY_PREFIX):]
            base_scores = bm25.score(source_of[query_document])
            score_of_id = {
                document_id: float(base_scores[bm25_position[document_id]])
                for document_id in dataset.document_ids
            }
            ranked_ids = sorted(dataset.document_ids, key=lambda d: -score_of_id[d])
        else:
            similarities = query_vector @ dataset.document_vectors.T
            order = np.argsort(-similarities)
            ranked_ids = [dataset.document_ids[int(i)] for i in order]
            score_of_id = {
                document_id: float(similarities[position])
                for position, document_id in enumerate(dataset.document_ids)
            }
        hits = ranked_ids[:INITIAL_K]

        # Expansion: the hits' own edges, restricted to the corpus. The query
        # document's edge list is never read.
        candidates = set(hits)
        frontier = set(hits)
        for _ in range(arguments.hops):
            grown: set[str] = set()
            for document_id in frontier:
                grown |= neighbours.get(document_id, set()) & corpus
            frontier = grown - candidates
            candidates |= grown

        budget = len(candidates)
        faiss_budget = set(ranked_ids[:budget])
        sim_of = score_of_id
        reranked = sorted(candidates, key=lambda d: -sim_of[d])

        # Reciprocal-rank fusion of two orderings of the same candidate set:
        # cosine, and graph provenance (the hits in retrieval order, then each
        # expansion under its best-ranked parent). Standard k=60. Chosen on the
        # selection half only; the evaluation half sees one rule.
        hit_rank = {document_id: index + 1 for index, document_id in enumerate(hits)}
        parent_rank: dict[str, int] = dict(hit_rank)
        for document_id in candidates - set(hits):
            parents = [hit_rank[h] for h in hits if document_id in neighbours.get(h, set())]
            parent_rank[document_id] = (min(parents) if parents else INITIAL_K) + INITIAL_K
        graph_order = sorted(candidates, key=lambda d: (parent_rank[d], -sim_of[d]))
        graph_pos = {d: i + 1 for i, d in enumerate(graph_order)}
        cosine_pos = {d: i + 1 for i, d in enumerate(reranked)}
        rrf = sorted(candidates, key=lambda d: -(1.0 / (60 + cosine_pos[d]) + 1.0 / (60 + graph_pos[d])))

        rows.append(
            {
                "relevant": len(relevant),
                "budget": budget,
                "relevant_in_candidates": len(relevant & candidates),
                "relevant_in_faiss_budget": len(relevant & faiss_budget),
                "recall8_faiss": sum(1 for d in ranked_ids[:8] if d in relevant) / len(relevant),
                "recall8_expand": sum(1 for d in reranked[:8] if d in relevant) / len(relevant),
                "recall16_faiss": sum(1 for d in ranked_ids[:16] if d in relevant) / len(relevant),
                "recall16_expand": sum(1 for d in reranked[:16] if d in relevant) / len(relevant),
                "recall32_faiss": sum(1 for d in ranked_ids[:32] if d in relevant) / len(relevant),
                "recall32_expand": sum(1 for d in reranked[:32] if d in relevant) / len(relevant),
                "recall8_rrf": sum(1 for d in rrf[:8] if d in relevant) / len(relevant),
                "recall16_rrf": sum(1 for d in rrf[:16] if d in relevant) / len(relevant),
                "seed": seed,
            }
        )

    print(f"snapshot : {snapshot.backend} / {snapshot.model} / {snapshot.dimension}d")
    print(f"seeds    : {len(rows)} usable, hops={arguments.hops}")
    print(f"\n=== equal budget (the §26 instrument) ===")
    print(f"  mean candidate-set size            {mean(r['budget'] for r in rows):.1f}")
    print(f"  relevant in expansion candidates   {mean(r['relevant_in_candidates'] for r in rows):.2f} "
          f"of {mean(r['relevant'] for r in rows):.2f}")
    print(f"  relevant in FAISS top-|candidates| {mean(r['relevant_in_faiss_budget'] for r in rows):.2f} "
          f"of {mean(r['relevant'] for r in rows):.2f}")
    print(f"\n=== ranked within candidates ===")
    for k in (8, 16, 32):
        rrf_column = (
            f"   rrf {mean(r[f'recall{k}_rrf'] for r in rows):.3f}" if k in (8, 16) else ""
        )
        print(f"  recall@{k:<2}  FAISS {mean(r[f'recall{k}_faiss'] for r in rows):.3f}   "
              f"cosine {mean(r[f'recall{k}_expand'] for r in rows):.3f}{rrf_column}")
    # The rule was designed once and is reported on both halves, so a lucky fit
    # to the first half cannot pass as a result.
    half = len(rows) // 2
    for name, part in (("selection half ", rows[:half]), ("evaluation half", rows[half:])):
        print(f"  [{name}] recall@8  FAISS {mean(r['recall8_faiss'] for r in part):.3f}"
              f"  rrf {mean(r['recall8_rrf'] for r in part):.3f}"
              f"   recall@16  FAISS {mean(r['recall16_faiss'] for r in part):.3f}"
              f"  rrf {mean(r['recall16_rrf'] for r in part):.3f}")


if __name__ == "__main__":
    main()
