"""Test persistent FAISS index load + search."""
import time
import sys
sys.path.insert(0, "src")

start = time.time()
from embeddinggemma.mcmp_rag import MCPMRetriever

r = MCPMRetriever(
    embedding_model_name="google/embeddinggemma-300m",
    num_agents=10,
    max_iterations=5,
    device_mode="auto",
)

loaded = r.load_persistent_index()
load_time = time.time() - start
print(f"Loaded: {loaded} | Docs: {len(r.documents)} | Load time: {load_time:.1f}s")

if loaded and r.documents:
    queries = [
        "gaze calibration eye tracking cursor",
        "CVE vulnerability NIST scanner",
        "intent classification event routing",
    ]
    for q in queries:
        start = time.time()
        results = r.search(q, top_k=3)
        search_time = time.time() - start
        hits = results.get("results", [])
        print(f"\nQuery: '{q}' | {len(hits)} results | {search_time:.3f}s")
        for item in hits[:3]:
            score = item.get("relevance_score", 0)
            snippet = item.get("content", "")[:120].replace("\n", " ")
            print(f"  {score:.3f} | {snippet}")
else:
    print("No persistent index found or empty.")
