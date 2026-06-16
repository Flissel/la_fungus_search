"""Build a ColBERT-Lite multi-vector index over the existing chunks.

Reads .fungus_cache/chunks.json, splits each chunk into ~3 views, embeds
them with the same Qwen3 model used for the main index, and saves to
.fungus_cache/multivec.npz.

Usage:
    FUNGUS_DEVICE=cuda python build_multivec_index.py
"""
from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, "src")
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))

EMBED_MODEL = os.environ.get("FUNGUS_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEVICE = os.environ.get("FUNGUS_DEVICE", "cuda")
BATCH_SIZE = int(os.environ.get("FUNGUS_MULTIVEC_BATCH", "32"))
MAX_SEQ = int(os.environ.get("FUNGUS_MULTIVEC_SEQ", "512"))

CACHE = ".fungus_cache"
CHUNKS_JSON = os.path.join(CACHE, "chunks.json")
OUT = os.path.join(CACHE, "multivec.npz")


print("=== ColBERT-Lite multi-vector index build ===")
print(f"model:      {EMBED_MODEL}")
print(f"device:     {DEVICE}")
print(f"batch size: {BATCH_SIZE}")
print(f"max seq:    {MAX_SEQ}")
print()


# ── Load chunks ─────────────────────────────────────────────────────────
t0 = time.time()
if not os.path.exists(CHUNKS_JSON):
    sys.exit(f"ERROR: {CHUNKS_JSON} not found — run build_vibemind_index.py first.")
with open(CHUNKS_JSON, encoding="utf-8") as f:
    chunks = json.load(f)
print(f"[1] Loaded {len(chunks)} chunks in {time.time()-t0:.1f}s")


# ── Load embedder (same Qwen as main index) ─────────────────────────────
t0 = time.time()
from sentence_transformers import SentenceTransformer
import torch
if DEVICE == "cuda" and not torch.cuda.is_available():
    print("WARN: cuda requested but not available; falling back to cpu")
    DEVICE = "cpu"
model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
try:
    model.max_seq_length = MAX_SEQ
except Exception:
    pass
print(f"[2] Embedder loaded in {time.time()-t0:.1f}s "
      f"(dim={model.get_sentence_embedding_dimension()})")


# ── Build multi-vector index ────────────────────────────────────────────
from embeddinggemma.multivec import (
    build_multivec_index, save_multivec, split_chunk_into_views,
)

# Sanity peek: how many views per chunk on average?
peek = chunks[:200]
view_counts = [len(split_chunk_into_views(c)) for c in peek]
print(f"[3] View-count sample (first 200 chunks): "
      f"min={min(view_counts)} avg={sum(view_counts)/len(view_counts):.1f} max={max(view_counts)}")


t0 = time.time()
embeddings, chunk_ids = build_multivec_index(
    chunks, model, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ,
)
print(f"[4] Multi-vector embedding done in {time.time()-t0:.1f}s")
print(f"    embeddings.shape = {embeddings.shape}")
print(f"    chunk_ids.shape  = {chunk_ids.shape} (max={chunk_ids.max()})")

if embeddings.shape[0] == 0:
    sys.exit("ERROR: no embeddings produced — aborting save")


# ── Save ────────────────────────────────────────────────────────────────
t0 = time.time()
save_multivec(OUT, embeddings, chunk_ids)
mb = os.path.getsize(OUT) / 1024**2
print(f"[5] Saved to {OUT} ({mb:.1f} MB) in {time.time()-t0:.1f}s")

print()
print("=== Done ===")
print(f"To use: re-start the fungus-search MCP server (it auto-loads multivec.npz)")
print(f"       then call fungus_search_multivec(query='...').")
