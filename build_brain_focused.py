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
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))
# Allow GPU if available (8GB free is enough for Qwen-Embedding-0.6B + this corpus).
# Override with FORCE_CPU=1 if Brain or another GPU consumer is running.
if os.environ.get("FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

EMBED_MODEL = os.environ.get("FUNGUS_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
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
print(f"Model: {EMBED_MODEL}")
print(f"Roots: {len(ROOTS)}")
for r in ROOTS:
    print(f"  - {r}")
print()

t0 = time.time()
from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.ui.corpus import collect_codebase_chunks

DEVICE = "cpu" if os.environ.get("FORCE_CPU") == "1" else "auto"
# Conservative batch — Qwen-Embedding-0.6B layer activations spike to ~800MB
# at batch 64 with long sequences. 16 fits in 4GB easily.
BATCH = 16 if DEVICE == "cpu" else 16
retriever = MCPMRetriever(
    embedding_model_name=EMBED_MODEL,
    num_agents=20,
    max_iterations=5,
    device_mode=DEVICE,
    embed_batch_size=BATCH,
)
print(f"  device={DEVICE} batch={BATCH}")
print(f"[1] Model loaded: {time.time()-t0:.1f}s | dim={retriever.embedding_model.get_sentence_embedding_dimension()}")

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
print(f"[4] Embedding {len(unique)} chunks (device={DEVICE}, batch={BATCH})...")
retriever.add_documents(unique)
print(f"[4] add_documents done: {time.time()-t0:.1f}s | docs={len(retriever.documents)}")

# Save
t0 = time.time()
ok = retriever.save_persistent_index()
print(f"[5] Persistent index saved: ok={ok} | {time.time()-t0:.1f}s")

print()
print(f"Done. Index lives in vibemind-os/la-fungus-search/.fungus_cache/")
