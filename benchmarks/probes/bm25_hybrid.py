"""C: BM25 + dense fusion, measured — because serving does not fuse today.

Finding first: the `/query` path never touches BM25. `bm25_lite.BM25Lite` exists,
a 30 MB `bm25.npz` sits in the production cache, and the only "bm25" in
`realtime/server.py` is a comment calling the length prior "bm25-like" — the same
`len_prior` the §-ce4743b blend bug ranked by. This probe measures what the
missing fusion would buy on the call-graph oracle.

**One honesty note before the numbers.** The query here is the query function's
own source text, and that text names its callees, whose definitions carry the
same identifiers. BM25 therefore rediscovers part of the very relation the oracle
is built from — through raw text, with no access to the manifest. That is
production-realistic (a natural-language query containing an identifier behaves
the same way) and it is also why BM25 is an unusually strong contender on this
benchmark; both facts belong in the record.

Fusion is reciprocal-rank fusion, k=60, over the two full rankings — one rule,
no tuning, reported on both halves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np

from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import QUERY_PREFIX, build_gate2_dataset
from benchmarks.gate2.snapshot import load_snapshot
from embeddinggemma.bm25_lite import BM25Lite

RRF_K = 60


def _recall(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for document_id in ranked[:k] if document_id in relevant) / len(relevant)


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 + dense fusion on the call-graph oracle")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=96)
    # Control for authored-link leakage on prose corpora: the query note
    # names its targets inside [[...]], so BM25 may be scoring the link text
    # rather than the content. Stripping wikilinks from the *query only*
    # quantifies that share.
    parser.add_argument("--strip-query-links", action="store_true")
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
    source_of = {document.document_id: document.source for document in manifest.documents}
    fit_ids = [document.document_id for document in manifest.documents]
    bm25 = BM25Lite()
    bm25.fit([source_of[document_id] for document_id in fit_ids])
    bm25_position = {document_id: index for index, document_id in enumerate(fit_ids)}

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
        query_document = query_id[len(QUERY_PREFIX):]
        query_text = source_of[query_document]
        if arguments.strip_query_links:
            import re
            query_text = re.sub(r"\[\[[^\]]*\]\]", " ", query_text)

        similarities = dataset.query_vectors[0] @ dataset.document_vectors.T
        dense = [dataset.document_ids[int(i)] for i in np.argsort(-similarities)]

        scores = bm25.score(query_text)
        sparse = sorted(
            dataset.document_ids,
            key=lambda document_id: -float(scores[bm25_position[document_id]]),
        )

        dense_pos = {document_id: index + 1 for index, document_id in enumerate(dense)}
        sparse_pos = {document_id: index + 1 for index, document_id in enumerate(sparse)}
        fused = sorted(
            dataset.document_ids,
            key=lambda d: -(1.0 / (RRF_K + dense_pos[d]) + 1.0 / (RRF_K + sparse_pos[d])),
        )

        rows.append(
            {
                variant + str(k): _recall(ranking, relevant, k)
                for variant, ranking in (("dense", dense), ("bm25", sparse), ("rrf", fused))
                for k in (8, 16, 32)
            }
        )

    print(f"snapshot : {snapshot.backend} / {snapshot.model} / {snapshot.dimension}d")
    print(f"seeds    : {len(rows)} usable")
    print(f"\n{'':>12}{'recall@8':>10}{'recall@16':>11}{'recall@32':>11}")
    for variant in ("dense", "bm25", "rrf"):
        print(f"{variant:>12}" + "".join(f"{mean(r[variant + str(k)] for r in rows):>{w}.3f}"
                                          for k, w in ((8, 10), (16, 11), (32, 11))))
    half = len(rows) // 2
    for name, part in (("selection half ", rows[:half]), ("evaluation half", rows[half:])):
        print(f"  [{name}] recall@8  dense {mean(r['dense8'] for r in part):.3f}"
              f"  rrf {mean(r['rrf8'] for r in part):.3f}"
              f"   recall@16  dense {mean(r['dense16'] for r in part):.3f}"
              f"  rrf {mean(r['rrf16'] for r in part):.3f}")


if __name__ == "__main__":
    main()
