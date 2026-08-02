"""Focused index build — only Brain core + key Brain-adjacent code.

Reduces 84k full-codebase chunks to ~5-10k self-relevant chunks. CPU-feasible
in 5-15 min instead of 4-11h. Covers what discourse-agents would actually
QUERY about.

Subsystems indexed:
  - vibemind-os/brain/the_brain/core/      (architecture itself)
  - vibemind-os/brain/the_brain/web/       (HTTP layer)
  - vibemind-os/brain/the_brain/scripts/   (operational scripts)
  - vibemind-os/spaces/ideas/              (one autonomous space example)
  - vibemind-os/voice/python/data/         (embedding service shared by all)

Skipped: rowboat-knowledge data, openfang vendored deps, fungus itself,
mirofish (it's a black box behind HTTP), tests.
"""
import sys
import time
import os

sys.path.insert(0, "src")
CHUNK_WINDOW = [200]

# Curated focused roots
ROOTS = [
    "C:/Users/User/Desktop/Vibemind_V1/vibemind-os/brain/the_brain/core",
    "C:/Users/User/Desktop/Vibemind_V1/vibemind-os/brain/the_brain/web",
    "C:/Users/User/Desktop/Vibemind_V1/vibemind-os/brain/the_brain/scripts",
    "C:/Users/User/Desktop/Vibemind_V1/vibemind-os/spaces/ideas",
    "C:/Users/User/Desktop/Vibemind_V1/vibemind-os/voice/python/data",
]

EXCLUDE_DIRS = [
    ".git", "__pycache__", "node_modules", ".venv", "target",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", "tests", "test", ".vscode",
]

print("=== Brain-Focused Fungus Index Build (CPU) ===")
print("Embedding role: fungus_search (OpenFang)")
print(f"Roots: {len(ROOTS)}")
for r in ROOTS:
    print(f"  - {r}")
print()

t0 = time.time()
from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.ui.corpus import collect_codebase_chunks

BATCH = 16
retriever = MCPMRetriever(
    num_agents=20,
    max_iterations=5,
    embed_batch_size=BATCH,
)
print(f"  role=fungus_search batch={BATCH}")
print(f"[1] OpenFang embedding backend ready: {time.time()-t0:.1f}s | "
      f"dim={retriever._expected_embedding_dim}")

# Collect from each root, merge
t0 = time.time()
all_chunks = []
for root in ROOTS:
    if not os.path.exists(root):
        print(f"  [WARN] missing: {root}")
        continue
    chunks = collect_codebase_chunks(
        root_dir=root,
        windows=CHUNK_WINDOW,
        max_files=5000,
        exclude_dirs=EXCLUDE_DIRS,
    )
    print(f"  - {os.path.basename(root)}: {len(chunks)} chunks")
    all_chunks.extend(chunks)
print(f"[2] Total raw chunks: {len(all_chunks)} | {time.time()-t0:.1f}s")

# Filter + dedupe
t0 = time.time()
all_chunks = [c for c in all_chunks if len(c) >= 50]
seen = set()
unique = []
for c in all_chunks:
    h = hash(c)
    if h not in seen:
        seen.add(h)
        unique.append(c)
print(f"[3] After filter+dedup: {len(unique)} | {time.time()-t0:.1f}s")

# Embed
t0 = time.time()
print(f"[4] Embedding {len(unique)} chunks through OpenFang (batch={BATCH})...")
retriever.add_documents(unique)
print(f"[4] add_documents done: {time.time()-t0:.1f}s | docs={len(retriever.documents)}")

# Save
t0 = time.time()
ok = retriever.save_persistent_index()
print(f"[5] Persistent index saved: ok={ok} | {time.time()-t0:.1f}s")

print()
print(f"Done. Index lives in vibemind-os/la-fungus-search/.fungus_cache/")
