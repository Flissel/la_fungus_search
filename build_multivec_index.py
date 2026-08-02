"""Build a ColBERT-Lite index through the canonical OpenFang embedding role."""

from __future__ import annotations

import json
import os
import sys
import time


sys.path.insert(0, "src")

BATCH_SIZE = int(os.environ.get("FUNGUS_MULTIVEC_BATCH", "32"))
CACHE = ".fungus_cache"
CHUNKS_JSON = os.path.join(CACHE, "chunks.json")
OUT = os.path.join(CACHE, "multivec.npz")

print("=== ColBERT-Lite multi-vector index build ===")
print("Embedding role: fungus_search (OpenFang)")
print(f"batch size: {BATCH_SIZE}")
print()

if not os.path.exists(CHUNKS_JSON):
    sys.exit(f"ERROR: {CHUNKS_JSON} not found — build the canonical Fungus index first.")

t0 = time.time()
with open(CHUNKS_JSON, encoding="utf-8") as handle:
    chunks = json.load(handle)
print(f"[1] Loaded {len(chunks)} chunks in {time.time() - t0:.1f}s")

from embeddinggemma.mcmp.embeddings import load_embedding_backend
from embeddinggemma.multivec import build_multivec_index, save_multivec, split_chunk_into_views

t0 = time.time()
model, expected_dimension = load_embedding_backend()
print(f"[2] OpenFang backend ready in {time.time() - t0:.1f}s (dim={expected_dimension})")

peek = chunks[:200]
view_counts = [len(split_chunk_into_views(chunk)) for chunk in peek]
print(
    f"[3] View-count sample (first {len(peek)} chunks): "
    f"min={min(view_counts)} avg={sum(view_counts) / len(view_counts):.1f} "
    f"max={max(view_counts)}"
)

t0 = time.time()
embeddings, chunk_ids = build_multivec_index(chunks, model, batch_size=BATCH_SIZE)
if embeddings.ndim != 2 or embeddings.shape[1] != expected_dimension:
    actual = embeddings.shape[1] if embeddings.ndim == 2 else "non-matrix"
    raise RuntimeError(
        "OpenFang embedding response has unexpected dimension: "
        f"expected {expected_dimension}, got {actual}."
    )
print(f"[4] Multi-vector embedding done in {time.time() - t0:.1f}s")
print(f"    embeddings.shape = {embeddings.shape}")
print(f"    chunk_ids.shape  = {chunk_ids.shape} (max={chunk_ids.max()})")

if embeddings.shape[0] == 0:
    sys.exit("ERROR: no embeddings produced — aborting save")

t0 = time.time()
save_multivec(OUT, embeddings, chunk_ids)
mb = os.path.getsize(OUT) / 1024**2
print(f"[5] Saved to {OUT} ({mb:.1f} MB) in {time.time() - t0:.1f}s")
print("\n=== Done ===")
