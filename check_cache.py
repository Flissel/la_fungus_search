import numpy as np
import os
import json

cache = ".fungus_cache"
emb_file = os.path.join(cache, "embeddings.npz")
chunks_file = os.path.join(cache, "chunks.json")
faiss_file = os.path.join(cache, "faiss.index")

print(f"=== Cache files ===")
for f in [emb_file, chunks_file, faiss_file]:
    size = os.path.getsize(f) / (1024*1024) if os.path.exists(f) else 0
    print(f"  {os.path.basename(f)}: {size:.1f} MB" if os.path.exists(f) else f"  {os.path.basename(f)}: MISSING")

if os.path.exists(emb_file):
    data = np.load(emb_file)
    embs = data["embeddings"]
    print(f"\n=== Embeddings ===")
    print(f"  Shape: {embs.shape}")
    print(f"  Dtype: {embs.dtype}")
    print(f"  Dim: {embs.shape[1]}")
    print(f"  Sample [0][:5]: {embs[0][:5]}")

if os.path.exists(chunks_file):
    with open(chunks_file, "r") as f:
        chunks = json.load(f)
    print(f"\n=== Chunks ===")
    print(f"  Count: {len(chunks)}")
    print(f"  First: {chunks[0][:100]}...")

try:
    import faiss
    print(f"\n=== FAISS ===")
    print(f"  faiss: {faiss.__version__ if hasattr(faiss, '__version__') else 'available'}")
    if os.path.exists(faiss_file):
        idx = faiss.read_index(faiss_file)
        print(f"  Index ntotal: {idx.ntotal}")
        print(f"  Index dim: {idx.d}")
except ImportError:
    print(f"\n=== FAISS: NOT INSTALLED ===")
