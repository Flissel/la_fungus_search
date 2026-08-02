"""Build the Fungus persistent index through the canonical OpenFang role.

This historical entrypoint retains its filename for existing operators.  It
does not load a local embedding model, select a device, or retry through a
second provider.  A missing or invalid OpenFang configuration is an error.
"""

import os
import sys
import time


sys.path.insert(0, "src")

CODEBASE = "C:/Users/User/Desktop/Vibemind_V1/vibemind-os"
EXCLUDE_DIRS = [
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", ".pitchdeck_chroma", ".playwright-mcp", "uv.lock",
    ".kilocode", ".vscode", "Coding_engine", "_archive", "all_services",
    "graphify-out", "temp-merge-parking",
]
MAX_FILES = 30000
CHUNK_WINDOW = [200]
EMBED_BATCH_SIZE = 16


print("=== Fungus OpenFang Index Build ===")
print(f"Codebase: {CODEBASE}")
print("Embedding role: fungus_search (OpenFang)")
print()

from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.ui.corpus import collect_codebase_chunks

t0 = time.time()
retriever = MCPMRetriever(
    num_agents=50,
    max_iterations=10,
    embed_batch_size=EMBED_BATCH_SIZE,
)
print(
    f"[1] OpenFang embedding backend ready: {time.time() - t0:.1f}s | "
    f"dim={retriever._expected_embedding_dim}",
    flush=True,
)

t0 = time.time()
raw = collect_codebase_chunks(
    root_dir=CODEBASE,
    windows=CHUNK_WINDOW,
    max_files=MAX_FILES,
    exclude_dirs=EXCLUDE_DIRS,
)
print(f"[2] raw chunks: {len(raw)} | {time.time() - t0:.1f}s", flush=True)

t0 = time.time()
seen = set()
chunks = []
for chunk in raw:
    if not isinstance(chunk, str):
        continue
    normalized = chunk.strip()
    if len(normalized) < 50 or normalized in seen:
        continue
    seen.add(normalized)
    chunks.append(normalized)
print(
    f"[3] filtered+deduped: {len(chunks)} (from {len(raw)}) | "
    f"{time.time() - t0:.1f}s",
    flush=True,
)

t0 = time.time()
retriever.add_documents(chunks, cache=True)
print(
    f"[4] embedded and persisted {len(retriever.documents)} chunks through "
    f"OpenFang | {time.time() - t0:.1f}s",
    flush=True,
)

t0 = time.time()
results = retriever.search_direct("OpenFang agent embedding search", top_k=5)
print(
    f"[5] search smoke test: {len(results.get('results', []))} hits | "
    f"{(time.time() - t0) * 1000:.0f}ms",
    flush=True,
)
print("\n=== DONE ===", flush=True)
