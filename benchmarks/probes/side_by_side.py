"""Side-by-side: retrieval v2 against the production-style window baseline.

Every number in report section 27 came from the function-as-query protocol; for
natural-language queries there is no oracle, and an LLM judge is ruled out by the
standing ground-truth rule. The legitimate evaluator is the operator with real
questions. This prints both engines' answers for one query, next to each other,
and adds no scores — judging is the point.

The baseline is BM25 over the production-style 200-line windows (the same window
set section 27.3 measured), standing in for today's serving unit. The v2 side is
exactly what `FUNGUS_RETRIEVAL_V2=1` serves: BM25 over function documents plus
one-hop call-graph expansion.

Run::

    PYTHONPATH=src .venv/Scripts/python.exe -m benchmarks.probes.side_by_side \
        "wie wird eine telegram nachricht gesendet?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The Windows console defaults to cp1252, and arbitrary source code does not fit
# in it. Degrade characters rather than crashing the comparison.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from embeddinggemma.bm25_lite import BM25Lite
from embeddinggemma.retrieval_v2 import HttpQueryEmbedder, RetrievalV2, load_index

MANIFEST = Path("benchmarks/gate2/manifests/brain-v1.json")
SNAPSHOT = Path("benchmarks/results/gate2/snapshot-brain-full.npz")
WINDOW_TEXTS = Path("benchmarks/results/gate2/windows-brain-texts.json")
WINDOW_META = Path("benchmarks/results/gate2/windows-brain-meta.json")


def _preview(text: str, lines: int = 3) -> str:
    rows = [row for row in text.splitlines() if row.strip()][:lines]
    return "\n".join("      " + row[:110] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 vs window baseline, one query")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()

    # With FUNGUS_V2_EMBEDDER_URL set (the local embedding service), the dense
    # arm arms and the comparison shows the full union configuration.
    import os
    embedder_url = os.environ.get("FUNGUS_V2_EMBEDDER_URL", "")
    engine = RetrievalV2(
        load_index(MANIFEST, SNAPSHOT),
        embed_query=HttpQueryEmbedder(embedder_url) if embedder_url else None,
    )
    windows = json.loads(WINDOW_TEXTS.read_text(encoding="utf-8"))
    meta = json.loads(WINDOW_META.read_text(encoding="utf-8"))
    window_bm25 = BM25Lite()
    window_bm25.fit(windows)

    print(f"query: {arguments.query!r}\n")
    print(f"=== V2 — Funktionen + Call-Graph-Expansion ({engine.engine}) ===")
    for row in engine.search(arguments.query, top_k=arguments.top_k)["results"]:
        m = row["metadata"]
        marker = "  [expanded]" if m["expanded"] else ""
        print(f"  {m['symbol']}  ({m['file']}:{m['start_line']}-{m['end_line']}){marker}")
        print(_preview(row["content"]))
    print(f"\n=== Produktions-Stil — 200-Zeilen-Fenster, BM25 ===")
    scores = window_bm25.score(arguments.query)
    for index in [int(i) for i in np.argsort(-scores)[: arguments.top_k]]:
        row = meta[index]
        print(f"  {row['file']}:{row['start']}-{row['end']}")
        print(_preview(windows[index], lines=3))


if __name__ == "__main__":
    main()
