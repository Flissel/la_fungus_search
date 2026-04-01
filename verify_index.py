import numpy as np
import os
import json
import sys

cache = ".fungus_cache"

# Check embeddings
emb_file = os.path.join(cache, "embeddings.npz")
if os.path.exists(emb_file):
    d = np.load(emb_file)
    e = d["embeddings"]
    print(f"Embeddings: shape={e.shape}, dim={e.shape[1]}, dtype={e.dtype}")
    print(f"  Sample: {e[0][:5]}")
else:
    print("NO embeddings.npz")
    sys.exit(1)

# Check chunks
chunks_file = os.path.join(cache, "chunks.json")
if os.path.exists(chunks_file):
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Chunks: {len(chunks)}")
    print(f"  First: {chunks[0][:80]}...")
else:
    print("NO chunks.json")

# Check FAISS
faiss_file = os.path.join(cache, "faiss.index")
if os.path.exists(faiss_file):
    size = os.path.getsize(faiss_file) / (1024*1024)
    print(f"FAISS index: {size:.1f} MB")
    try:
        import faiss
        idx = faiss.read_index(faiss_file)
        print(f"  ntotal={idx.ntotal}, d={idx.d}")
    except ImportError:
        print("  (faiss not installed, can't verify)")
else:
    print("NO faiss.index (build may have skipped FAISS)")

# Test search if possible
if e.shape[1] > 100:
    print(f"\nDim {e.shape[1]} looks like real embeddings (not random 64-dim)")
    print("INDEX IS VALID")
else:
    print(f"\nWARNING: dim={e.shape[1]} looks like random fallback (expected 768)")
    print("INDEX NEEDS REBUILD with real embedding model")
