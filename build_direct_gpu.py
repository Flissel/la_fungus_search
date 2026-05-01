"""Direct GPU-embedded Fungus index build — bypasses MCPMRetriever facade
which has been hanging during its .save_persistent_index() step.

Outputs to .fungus_cache/ so the existing Fungus search code can still
use it:
  - faiss.index  (1024-dim Qwen, IP metric)
  - embeddings.npz  (for rebuilds)
  - chunks.json  (chunk texts, in order)
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, "src")

CODEBASE = "C:/Users/User/Desktop/Vibemind_V1/vibemind-os"
EXCLUDE_DIRS = [
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", ".pitchdeck_chroma", ".playwright-mcp",
    "uv.lock", ".kilocode", ".vscode",
]
EMBED_MODEL = os.environ.get("FUNGUS_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
MAX_FILES = 15000
CHUNK_WINDOW = [200]
BATCH_SIZE = 16
MAX_CHARS = 1200   # truncate long chunks to bound memory
CHECKPOINT_EVERY = 2000  # flush partial results every N embeddings

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fungus_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

print("=== Direct GPU Fungus Index Build ===")
print(f"Codebase: {CODEBASE}")
print(f"Model: {EMBED_MODEL}")
print(f"Cache: {CACHE_DIR}")
print()

# 1. Collect chunks
t0 = time.time()
from embeddinggemma.ui.corpus import collect_codebase_chunks
raw = collect_codebase_chunks(
    root_dir=CODEBASE, windows=CHUNK_WINDOW,
    max_files=MAX_FILES, exclude_dirs=EXCLUDE_DIRS,
)
print(f"[1] raw chunks: {len(raw)} | {time.time()-t0:.1f}s", flush=True)

# 2. Filter + exact dedup
t0 = time.time()
seen = set()
chunks = []
for c in raw:
    if not isinstance(c, str):
        continue
    c = c.strip()
    if len(c) < 50:
        continue
    k = hash(c)
    if k in seen:
        continue
    seen.add(k)
    chunks.append(c)
print(f"[2] filtered+deduped: {len(chunks)} (from {len(raw)}) | {time.time()-t0:.1f}s", flush=True)

# 3. Load model (GPU)
t0 = time.time()
import torch
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"
m = SentenceTransformer(EMBED_MODEL, device=device)
print(f"[3] model loaded on {device} | {time.time()-t0:.1f}s", flush=True)

# 4. Embed in batches with progress
t0 = time.time()
vectors = np.empty((len(chunks), m.get_sentence_embedding_dimension()), dtype=np.float32)
n = len(chunks)

# Resume from partial checkpoint if present
resume_from = 0
_part = os.path.join(CACHE_DIR, "embeddings.partial.npz")
if os.path.exists(_part):
    try:
        _d = np.load(_part)
        _cnt = int(_d["count"]) if "count" in _d.files else _d["vectors"].shape[0]
        if _cnt > 0 and _cnt <= n and _d["vectors"].shape[1] == vectors.shape[1]:
            vectors[:_cnt] = _d["vectors"][:_cnt]
            resume_from = _cnt
            print(f"[4] resumed from checkpoint: {resume_from}/{n}", flush=True)
    except Exception as e:
        print(f"[4] could not resume ({e})", flush=True)
import torch as _torch
for start in range(resume_from, n, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n)
    batch = [c[:MAX_CHARS] for c in chunks[start:end]]
    try:
        v = m.encode(
            batch,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except _torch.cuda.OutOfMemoryError:
        _torch.cuda.empty_cache()
        # Retry batch one-by-one
        v = np.empty((len(batch), vectors.shape[1]), dtype=np.float32)
        for j, item in enumerate(batch):
            v[j] = m.encode(
                [item], batch_size=1, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )[0]
    vectors[start:end] = v.astype(np.float32)
    if (start // BATCH_SIZE) % 100 == 0:
        pct = end / n * 100
        rate = end / max(time.time() - t0, 0.01)
        eta = (n - end) / max(rate, 1)
        print(f"[4] embedded {end}/{n} ({pct:.1f}%) rate={rate:.0f}/s eta={eta:.0f}s", flush=True)
    # Periodic checkpoint so OOM crashes don't lose work
    if end % CHECKPOINT_EVERY < BATCH_SIZE:
        try:
            np.savez(os.path.join(CACHE_DIR, "embeddings.partial.npz"),
                     vectors=vectors[:end], count=end)
        except Exception:
            pass
print(f"[4] embedded all {n} chunks | total {time.time()-t0:.1f}s", flush=True)

# 5. Build FAISS index
t0 = time.time()
try:
    import faiss
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    faiss.write_index(idx, os.path.join(CACHE_DIR, "faiss.index"))
    print(f"[5] FAISS index written ({idx.ntotal} vectors) | {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"[5] FAISS build failed: {e}", flush=True)

# 6. Persist embeddings + chunks
t0 = time.time()
np.savez(os.path.join(CACHE_DIR, "embeddings.npz"), vectors=vectors)
with open(os.path.join(CACHE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)
print(f"[6] cache written | {time.time()-t0:.1f}s", flush=True)

# 7. Quick search test
t0 = time.time()
q = m.encode(["GPU CUDA device embedding sentence transformers"],
             normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
D, I = idx.search(q, 5)
print(f"[7] search test ({(time.time()-t0)*1000:.0f}ms):", flush=True)
for score, i in zip(D[0], I[0]):
    head = chunks[i][:120].replace("\n", " ")
    print(f"  {score:.3f}  {head}", flush=True)

print("\n=== DONE ===", flush=True)
