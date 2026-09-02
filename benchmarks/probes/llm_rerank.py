"""The first LLM inside the retrieval stack, measured against mechanical truth.

Section 27 left one number unmoved: recall@8. The union+expansion candidate set
provably holds more relevant documents than similarity's top-N, but they sit at
positions 9-12 and neither cosine nor RRF nor BM25 order them forward. Ordering
*twelve* documents is exactly the size of task a language model does well and
cheaply -- and, decisively for the standing ground-truth rule, it can be measured
here as a component under an independent oracle (the call graph), not as a judge
of itself. If it does not lift recall@8, the number says so.

Design:

- Candidates: identical to `callgraph_expand --base union` -- interleaved
  BM25+dense hits, one call-graph hop. All orderings rank the SAME set, so any
  difference is ordering, never candidate quality.
- One `claude -p` call per query (the SOM-planner headless pattern), prompt =
  query function source plus every candidate's source, task = rank all ids by
  how likely the candidate directly calls or is called by the query. That is
  deducible from the shown text -- the same text BM25 sees -- so it is
  reasoning, not leakage.
- Fail-closed parsing: the reply must contain a JSON array that is a permutation
  of the candidate ids; anything else counts as a failure and falls back to the
  BM25 ordering for that seed, reported separately rather than blended away.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from statistics import mean

import numpy as np

from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.snapshot import load_snapshot
from embeddinggemma.bm25_lite import BM25Lite

INITIAL_K = 8
QUERY_SOURCE_LINES = 80
CANDIDATE_SOURCE_LINES = 30


def _truncate(text: str, lines: int) -> str:
    rows = text.splitlines()
    if len(rows) <= lines:
        return text
    return "\n".join(rows[:lines]) + f"\n# ... ({len(rows) - lines} lines truncated)"


def build_candidates(dataset, manifest, bm25, bm25_position, source_of, neighbours):
    """Union hits + one hop, exactly the section-27 winning configuration."""
    query_id = dataset.query_ids[0]
    query_document = query_id[2:] if query_id.startswith("q:") else query_id
    scores = bm25.score(source_of[query_document])
    score_of = {
        document_id: float(scores[bm25_position[document_id]])
        for document_id in dataset.document_ids
    }
    bm25_ranked = sorted(dataset.document_ids, key=lambda d: -score_of[d])
    similarities = dataset.query_vectors[0] @ dataset.document_vectors.T
    dense_ranked = [dataset.document_ids[int(i)] for i in np.argsort(-similarities)]

    merged: list[str] = []
    for pair in zip(bm25_ranked, dense_ranked):
        for document_id in pair:
            if document_id not in merged:
                merged.append(document_id)
        if len(merged) >= INITIAL_K:
            break
    hits = merged[:INITIAL_K]

    corpus = set(dataset.document_ids)
    candidates = list(hits)
    for hit in hits:
        for neighbour in sorted(neighbours.get(hit, set()) & corpus):
            if neighbour not in candidates:
                candidates.append(neighbour)
    return query_document, hits, candidates, score_of


def rrf_order(candidates, hits, score_of, neighbours):
    hit_rank = {document_id: rank + 1 for rank, document_id in enumerate(hits)}
    parent_rank = dict(hit_rank)
    for document_id in candidates:
        if document_id in parent_rank:
            continue
        parents = [hit_rank[h] for h in hits if document_id in neighbours.get(h, set())]
        parent_rank[document_id] = (min(parents) if parents else INITIAL_K) + INITIAL_K
    score_sorted = sorted(candidates, key=lambda d: -score_of[d])
    graph_sorted = sorted(candidates, key=lambda d: (parent_rank[d], -score_of[d]))
    score_pos = {d: i + 1 for i, d in enumerate(score_sorted)}
    graph_pos = {d: i + 1 for i, d in enumerate(graph_sorted)}
    return sorted(
        candidates, key=lambda d: -(1.0 / (60 + score_pos[d]) + 1.0 / (60 + graph_pos[d]))
    )


def ask_llm(claude, model, query_source, candidates, source_of, timeout):
    listing = "\n\n".join(
        f"### candidate {index} : {document_id}\n{_truncate(source_of[document_id], CANDIDATE_SOURCE_LINES)}"
        for index, document_id in enumerate(candidates)
    )
    prompt = (
        "You are ranking code-search candidates for one query function.\n\n"
        "## query function\n"
        f"{_truncate(query_source, QUERY_SOURCE_LINES)}\n\n"
        "## candidates\n"
        f"{listing}\n\n"
        "## task\n"
        "Rank ALL candidates from most to least likely to be a DIRECT caller or "
        "callee of the query function (reading the sources above: does the query "
        "call the candidate's symbol, or does the candidate call the query's?). "
        f"Answer with ONLY a JSON array of all {len(candidates)} candidate ids, "
        "best first, nothing else."
    )
    completed = subprocess.run(
        [claude, "-p", "--output-format", "json", "--model", model, "--strict-mcp-config"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if completed.returncode != 0:
        return None, f"exit {completed.returncode}", 0.0
    try:
        envelope = json.loads(completed.stdout)
        reply = str(envelope.get("result", ""))
        cost = float(envelope.get("total_cost_usd", 0.0))
    except ValueError:
        return None, "outer JSON unparseable", 0.0
    match = re.search(r"\[[\d,\s]+\]", reply)
    if not match:
        return None, "no JSON array in reply", cost
    try:
        order = json.loads(match.group(0))
    except ValueError:
        return None, "array unparseable", cost
    if sorted(order) != list(range(len(candidates))):
        return None, "not a permutation of candidate ids", cost
    return [candidates[i] for i in order], "", cost


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM reranker over the 12-candidate set")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--claude", default=os.path.expanduser("~/.local/bin/claude.exe"))
    parser.add_argument("--timeout", type=int, default=180)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
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
    failures: list[str] = []
    total_cost = 0.0
    for seed in range(arguments.seeds):
        try:
            dataset = build_gate2_dataset(manifest, snapshot, seed)
        except ValueError:
            continue
        relevant = set(dataset.relevant_by_query[dataset.query_ids[0]])
        if not relevant:
            continue
        query_document, hits, candidates, score_of = build_candidates(
            dataset, manifest, bm25, bm25_position, source_of, neighbours
        )
        orders = {
            "bm25": sorted(candidates, key=lambda d: -score_of[d]),
            "rrf": rrf_order(candidates, hits, score_of, neighbours),
        }
        llm_order, error, cost = ask_llm(
            arguments.claude, arguments.model, source_of[query_document],
            candidates, source_of, arguments.timeout,
        )
        total_cost += cost
        if llm_order is None:
            failures.append(f"seed {seed}: {error}")
            orders["llm"] = orders["bm25"]
        else:
            orders["llm"] = llm_order

        row = {"llm_ok": llm_order is not None}
        for name, ordering in orders.items():
            for k in (4, 8):
                row[f"{name}{k}"] = sum(
                    1 for d in ordering[:k] if d in relevant
                ) / len(relevant)
        rows.append(row)

    print(f"model    : {arguments.model} | seeds evaluated: {len(rows)} | "
          f"llm ok: {sum(1 for r in rows if r['llm_ok'])} | cost: ${total_cost:.2f}")
    if failures:
        print("failures :", "; ".join(failures))
    print(f"\n{'ordering':>9}{'recall@4':>10}{'recall@8':>10}")
    for name in ("bm25", "rrf", "llm"):
        print(f"{name:>9}{mean(r[f'{name}4'] for r in rows):>10.3f}"
              f"{mean(r[f'{name}8'] for r in rows):>10.3f}")
    ok_rows = [r for r in rows if r["llm_ok"]]
    if ok_rows and len(ok_rows) != len(rows):
        print(f"  [nur erfolgreiche LLM-Aufrufe, n={len(ok_rows)}] "
              f"bm25@8 {mean(r['bm258'] for r in ok_rows):.3f}  "
              f"llm@8 {mean(r['llm8'] for r in ok_rows):.3f}")


if __name__ == "__main__":
    main()
