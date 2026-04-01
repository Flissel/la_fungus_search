"""Build persistent FAISS index for poc_injection_chain codebase."""
import sys
import time
import os

sys.path.insert(0, "src")
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))

print("Loading embedding model...")
start = time.time()

from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.ui.corpus import collect_codebase_chunks

CODEBASE = "C:/Users/User/Desktop/poc_injection_chain"
EXCLUDE = [".git", "__pycache__", "node_modules", ".venv", "target", ".next",
           ".pitchdeck_chroma", ".playwright-mcp", ".pytest_cache",
           "openfang/target", "multiseat-os/downloads"]

r = MCPMRetriever(
    embedding_model_name="all-MiniLM-L6-v2",
    num_agents=50,
    max_iterations=10,
    device_mode="auto",
    embed_batch_size=128,
)

model_time = time.time() - start
print(f"Model loaded in {model_time:.1f}s")
print(f"Embedding model: {r.embedding_model}")
if r.embedding_model:
    test = r.embedding_model.encode(["test"])
    print(f"Embedding dim: {test[0].shape}")
else:
    print("WARNING: No embedding model! Will use random vectors.")
    sys.exit(1)

print(f"\nCollecting chunks from {CODEBASE}...")
start = time.time()
chunks = collect_codebase_chunks(
    root_dir=CODEBASE,
    windows=[150, 300],
    max_files=2000,
    exclude_dirs=EXCLUDE,
)
chunk_time = time.time() - start
print(f"Collected {len(chunks)} chunks in {chunk_time:.1f}s")

print(f"\nEmbedding + indexing...")
start = time.time()
r.add_documents(chunks, cache=True)
embed_time = time.time() - start
print(f"Indexed in {embed_time:.1f}s")
print(f"Documents: {len(r.documents)}")
print(f"Embed dim: {r._embed_dim}")

# Verify cache
cache_dir = r._CACHE_DIR
for f in ["faiss.index", "embeddings.npz", "chunks.json"]:
    path = os.path.join(cache_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"  Cache: {f} = {size:.1f} MB")
    else:
        print(f"  Cache: {f} = MISSING")

# Test search
print(f"\nSearching...")
for q in ["gaze calibration eye tracking", "vulnerability CVE scanner", "intent classification event routing"]:
    start = time.time()
    results = r.search(q, top_k=3)
    st = time.time() - start
    hits = results.get("results", [])
    print(f"\n  '{q}' ({st:.3f}s, {len(hits)} hits)")
    for item in hits[:2]:
        score = item.get("relevance_score", 0)
        snippet = item.get("content", "")[:100].replace("\n", " ")
        print(f"    {score:.3f} | {snippet}")

print(f"\nDone. Total time: {model_time + chunk_time + embed_time:.1f}s")
