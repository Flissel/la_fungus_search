"""Build persistent FAISS index for the entire vibemind-os codebase."""
import sys
import time
import os
import re

sys.path.insert(0, "src")

CODEBASE = "C:/Users/User/Desktop/Vibemind_V1/vibemind-os"
EXCLUDE_DIRS = [
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", ".pitchdeck_chroma", ".playwright-mcp",
    "uv.lock", ".kilocode", ".vscode",
    # ── Opt-Stage-2 (2026-05-25): dead/duplicate trees that polluted top-K ──
    "Coding_engine",   # old copy under spaces/coding/Coding_engine/
    "_archive",
    "all_services",
]
MAX_FILES = 15000
CHUNK_WINDOW = [200]

print("=== Vibemind-OS Full Index Build ===")
print(f"Codebase: {CODEBASE}")
print("Embedding role: fungus_search (OpenFang)")
print()

# 1. Load model
t0 = time.time()
from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.ui.corpus import collect_codebase_chunks

r = MCPMRetriever(
    num_agents=50,
    max_iterations=10,
    embed_batch_size=32,
)
print(f"[1] OpenFang embedding backend ready: {time.time()-t0:.1f}s | "
      f"role=fungus_search | dim={r._expected_embedding_dim} | batch_size=32")

# 2. Collect chunks
t0 = time.time()
raw_chunks = collect_codebase_chunks(
    root_dir=CODEBASE,
    windows=CHUNK_WINDOW,
    max_files=MAX_FILES,
    exclude_dirs=EXCLUDE_DIRS,
)
print(f"[2] Raw chunks: {len(raw_chunks)} | {time.time()-t0:.1f}s")

# 3. Filter short/empty chunks AND oversized chunks (Opt-Stage-2)
# Oversized chunks (>20k chars ≈ >5k tokens) blow up attention quadratically
# and aren't useful for semantic search anyway (single huge JSON/lock-file blobs).
t0 = time.time()
MAX_CHARS = 20000
filtered = [c for c in raw_chunks if 50 <= len(c.strip()) <= MAX_CHARS]
short_n = sum(1 for c in raw_chunks if len(c.strip()) < 50)
oversized_n = sum(1 for c in raw_chunks if len(c.strip()) > MAX_CHARS)
print(f"[3] After filter: {len(filtered)} (removed {short_n} short, {oversized_n} oversized) | {time.time()-t0:.1f}s")

# 4. Deduplicate (exact-match only, skip simhash which segfaults on large corpora)
t0 = time.time()
seen = set()
deduped = []
dupe_count = 0
for c in filtered:
    if not isinstance(c, str):
        continue
    key = hash(c)
    if key in seen:
        dupe_count += 1
        continue
    seen.add(key)
    deduped.append(c)
print(f"[4] After exact-dedup: {len(deduped)} (removed {dupe_count} exact-dupes) | {time.time()-t0:.1f}s")

# 5. Embed + index
t0 = time.time()
r.add_documents(deduped, cache=True)
print(f"[5] Embedded + indexed: {len(r.documents)} docs, dim={r._embed_dim} | {time.time()-t0:.1f}s")

# 6. Verify cache
cache_dir = r._CACHE_DIR
for f in ["faiss.index", "embeddings.npz", "chunks.json"]:
    path = os.path.join(cache_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"  Cache: {f} = {size:.1f} MB")
    else:
        print(f"  Cache: {f} = MISSING")

# 7. Search quality test
print(f"\n=== Search Quality Test ===")
queries = [
    ("gaze calibration eye tracking cursor", "eyeterm OR gaze OR calibrat"),
    ("CVE vulnerability NIST scanner", "vuln OR cve OR scanner OR security"),
    ("intent classification event routing", "intent OR classif OR routing"),
    ("blink detection fatigue score", "blink OR fatigue OR detection"),
    ("WebSocket real-time messaging", "websocket OR socket OR messaging OR ws"),
    ("MCMP pheromone agent simulation", "mcmp OR pheromone OR simulation"),
    ("LLM prompt generation ollama", "llm OR prompt OR ollama OR generation"),
    ("React frontend component dashboard", "react OR component OR dashboard OR frontend"),
    ("FAISS vector embedding search", "faiss OR vector OR embedding OR search"),
    ("coding engine architect planner", "coding OR engine OR architect OR plan"),
    ("voice recognition speech input", "voice OR speech OR recogni"),
    ("brain radial attention network", "brain OR radial OR attention"),
    ("OpenFang agent bridge transport", "openfang OR bridge OR transport OR fang"),
    ("security canary honeypot detection", "canary OR honeypot OR security"),
    ("Redis pub/sub cache session", "redis OR pub OR cache OR session"),
]

total_time = 0
hits_found = 0
for query, expected_keywords in queries:
    t0 = time.time()
    results = r.search_direct(query, top_k=5)
    st = time.time() - t0
    total_time += st
    items = results.get("results", [])

    top = items[0] if items else {}
    score = top.get("relevance_score", 0)
    content = top.get("content", "")

    # Extract file from header
    m = re.search(r'# file: (.+?) \|', content)
    found_file = m.group(1) if m else content[:80].replace("\n", " ")

    # Check if any expected keyword appears in top-3 results
    top3_text = " ".join(it.get("content", "")[:500] for it in items[:3]).lower()
    kw_list = [k.strip().lower() for k in expected_keywords.split(" OR ")]
    matched = any(k in top3_text for k in kw_list)
    if matched:
        hits_found += 1
    tag = "OK" if matched else "MISS"

    print(f"  [{tag}] {score:.3f} | {query[:42]:42s} -> {found_file[:55]}")

avg_ms = (total_time / len(queries)) * 1000
accuracy = hits_found / len(queries) * 100
print(f"\n  Results: {hits_found}/{len(queries)} relevant ({accuracy:.0f}%)")
print(f"  Avg search: {avg_ms:.0f}ms | Total: {total_time:.2f}s")
print(f"\n=== Done ===")
