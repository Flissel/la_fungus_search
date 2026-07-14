"""Repair the .fungus_cache index in place — no re-embedding.

Three defects found 2026-07-14 after the full GPU rebuild:
  1. graphify-out/ (a generated KG dump) was 35% of the index — 30k chunks from
     a single graph.json — and dominated top-K for real code queries.
  2. bm25.npz was still the 2026-05-25 build, fitted on the OLD corpus, so its
     doc_ids no longer line up with the current documents -> the hybrid fusion
     mixed in misaligned BM25 scores.
  3. embeddings.npz was written with key "vectors" while MCPMRetriever reads
     "embeddings" (fixed in build_direct_gpu.py; we write both keys here).

Keeps the existing embeddings (they are correct), drops the noise rows, then
rebuilds FAISS + BM25 over exactly the surviving chunks.
"""

import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, "src")

CACHE = ".fungus_cache"
NOISE_MARKERS = ("graphify-out", "graphify_out")
_HEADER = re.compile(r"# file: (.+?) \|")


def _path_of(chunk) -> str:
    s = chunk if isinstance(chunk, str) else (chunk.get("content") or "")
    m = _HEADER.search(s)
    return m.group(1).replace("\\", "/") if m else ""


def main() -> int:
    t0 = time.time()
    chunks = json.load(open(os.path.join(CACHE, "chunks.json"), encoding="utf-8"))
    d = np.load(os.path.join(CACHE, "embeddings.npz"))
    key = "embeddings" if "embeddings" in d.files else "vectors"
    embs = d[key]
    d.close()
    print(f"[0] loaded: {len(chunks)} chunks, embeddings {embs.shape} | {time.time()-t0:.0f}s", flush=True)
    if len(chunks) != embs.shape[0]:
        print(f"ABORT: chunk/embedding count mismatch ({len(chunks)} vs {embs.shape[0]})")
        return 1

    # 1. keep-mask
    t0 = time.time()
    keep = np.array(
        [not any(k in _path_of(c) for k in NOISE_MARKERS) for c in chunks], dtype=bool
    )
    dropped = int((~keep).sum())
    print(f"[1] dropping {dropped} noise chunks ({100*dropped/len(chunks):.1f}%) "
          f"-> {int(keep.sum())} keep | {time.time()-t0:.0f}s", flush=True)

    new_chunks = [c for c, k in zip(chunks, keep) if k]
    new_embs = np.ascontiguousarray(embs[keep])
    del chunks, embs

    # 2. chunks.json
    t0 = time.time()
    tmp = os.path.join(CACHE, "chunks.tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new_chunks, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(CACHE, "chunks.json"))
    print(f"[2] chunks.json written ({len(new_chunks)}) | {time.time()-t0:.0f}s", flush=True)

    # 3. embeddings.npz — both keys (loader reads "embeddings")
    t0 = time.time()
    tmp = os.path.join(CACHE, "embeddings.tmp.npz")
    np.savez(tmp, embeddings=new_embs, vectors=new_embs)
    os.replace(tmp, os.path.join(CACHE, "embeddings.npz"))
    print(f"[3] embeddings.npz written {new_embs.shape} | {time.time()-t0:.0f}s", flush=True)

    # 4. FAISS (vectors are already L2-normalised -> inner product = cosine)
    t0 = time.time()
    import faiss
    idx = faiss.IndexFlatIP(new_embs.shape[1])
    idx.add(new_embs)
    faiss.write_index(idx, os.path.join(CACHE, "faiss.index"))
    print(f"[4] faiss.index written (ntotal={idx.ntotal}) | {time.time()-t0:.0f}s", flush=True)

    # 5. BM25 refit over exactly these chunks
    t0 = time.time()
    from embeddinggemma.bm25_lite import BM25Lite
    bm = BM25Lite()
    bm.fit([c if isinstance(c, str) else (c.get("content") or "") for c in new_chunks])
    bm.save(os.path.join(CACHE, "bm25.npz"))
    print(f"[5] bm25.npz refitted (N={bm.N}) | {time.time()-t0:.0f}s", flush=True)

    print("=== REPAIR DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
