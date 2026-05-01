"""CPU-only build of the Vibemind-OS persistent FAISS index.

Slower than the GPU build but does not contend with Brain's CUDA stack.
Embeds 84k chunks at ~30-50 chunks/s on a modern CPU → 30-50 min total.

Usage:
    python build_vibemind_cpu.py
"""
import sys
import time
import os

sys.path.insert(0, "src")
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

print("=== Vibemind-OS Full Index Build (CPU) ===")
print(f"Codebase: {CODEBASE}")
print(f"Model: {EMBED_MODEL}")
print()

t0 = time.time()
from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.ui.corpus import collect_codebase_chunks

r = MCPMRetriever(
    embedding_model_name=EMBED_MODEL,
    num_agents=50,
    max_iterations=10,
    device_mode="cpu",
    embed_batch_size=32,  # smaller batch for CPU
)
print(f"[1] Model loaded: {time.time()-t0:.1f}s | dim={r.embedding_model.get_sentence_embedding_dimension()}")

t0 = time.time()
raw_chunks = collect_codebase_chunks(
    root_dir=CODEBASE,
    windows=CHUNK_WINDOW,
    max_files=MAX_FILES,
    exclude_dirs=EXCLUDE_DIRS,
)
print(f"[2] Raw chunks: {len(raw_chunks)} | {time.time()-t0:.1f}s")

t0 = time.time()
chunks = [c for c in raw_chunks if len(c) >= 50]
print(f"[3] After filter: {len(chunks)} (removed {len(raw_chunks)-len(chunks)} short) | {time.time()-t0:.1f}s")

t0 = time.time()
seen = set()
unique = []
for c in chunks:
    h = hash(c)
    if h not in seen:
        seen.add(h)
        unique.append(c)
print(f"[4] After exact-dedup: {len(unique)} (removed {len(chunks)-len(unique)} exact-dupes) | {time.time()-t0:.1f}s")

# Embed + index
t0 = time.time()
print(f"[5] Embedding {len(unique)} chunks (CPU, batch 32) — this will take 20-50 min...")
r.add_documents(unique)
print(f"[5] add_documents done: {time.time()-t0:.1f}s | docs={len(r.documents)}")

# Save
t0 = time.time()
ok = r.save_persistent_index()
print(f"[6] Persistent index saved: ok={ok} | {time.time()-t0:.1f}s")

print()
print("Done.")
