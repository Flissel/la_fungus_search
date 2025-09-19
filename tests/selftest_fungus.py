import os
import sys
import glob
import json
import re
import time


def read_cached_chunks(cache_dir: str, limit_files: int = 5, limit_lines: int = 1000):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No cache files found under {cache_dir}")
    docs = []
    used = 0
    for fp in files[:limit_files]:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= limit_lines:
                    break
                docs.append(line.rstrip("\n"))
        used += 1
    return docs, used, len(files)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root)

    cache_dir = os.path.join(root, ".fungus_cache", "chunks")
    print(f"🔎 Using cache dir: {cache_dir}")
    docs, used_files, total_files = read_cached_chunks(cache_dir)
    print(f"✅ Loaded {len(docs)} cached chunks from {used_files}/{total_files} files")

    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
        print(f"🟢 Torch CUDA available: {cuda} device: {gpu}")
    except Exception as e:
        print(f"⚠️ Torch not available or error: {e}")

    from mcmp_rag import MCPMRetriever

    retr = MCPMRetriever(num_agents=100, max_iterations=10, exploration_bonus=0.1, pheromone_decay=0.95)
    retr.log_every = 5

    t0 = time.time()
    ok = retr.add_documents(docs)
    t1 = time.time()
    if not ok:
        raise RuntimeError("add_documents failed")
    print(f"⏱️ add_documents: {(t1 - t0)*1000:.1f} ms for {len(docs)} chunks")

    query = "List functions, methods, and classes in the codebase"
    t2 = time.time()
    results = retr.search(query, top_k=5)
    t3 = time.time()
    nres = len(results.get("results", [])) if isinstance(results, dict) else 0
    print(f"⏱️ search: {(t3 - t2)*1000:.1f} ms, results: {nres}")
    print(json.dumps({"ok": True, "docs": len(docs), "results": results}, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()



