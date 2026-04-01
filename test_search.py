"""Test semantic search with persistent index."""
import time
import sys
sys.path.insert(0, "src")

from embeddinggemma.mcmp_rag import MCPMRetriever

start = time.time()
r = MCPMRetriever(
    embedding_model_name="all-MiniLM-L6-v2",
    num_agents=10,
    max_iterations=5,
    device_mode="auto",
)

loaded = r.load_persistent_index()
load_time = time.time() - start
print(f"Load: {loaded} | {len(r.documents)} docs | {load_time:.1f}s | dim={r._embed_dim}")

if not loaded:
    print("FAILED to load index")
    sys.exit(1)

queries = [
    "gaze calibration eye tracking cursor",
    "CVE vulnerability NIST scanner",
    "intent classification event routing",
    "OpenFang agent wrapper respond",
    "blink detection fatigue score",
]

for q in queries:
    start = time.time()
    results = r.search_direct(q, top_k=3)
    search_time = time.time() - start
    hits = results.get("results", [])
    print(f"\n'{q}' ({search_time:.2f}s)")
    for item in hits[:3]:
        score = item.get("relevance_score", 0)
        content = item.get("content", "")
        lines = content.split("\n")
        header = lines[0] if lines else ""
        header = header.encode("ascii", errors="replace").decode()
        print(f"  {score:.4f} | {header[:80]}")
