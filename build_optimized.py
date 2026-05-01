"""
Build optimized FAISS index for poc_injection_chain.

Optimizations:
1. SimHash deduplication (removes near-duplicate chunks)
2. Exclude test/query strings that would pollute results
3. AST-chunking preferred, line-windows as fallback
4. Single window size (200 lines) to reduce overlap
5. FAISS IVF index for fast search (<50ms)
"""
import sys
import time
import os
import hashlib
import json
import re

sys.path.insert(0, "src")
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))

CODEBASE = os.environ.get("FUNGUS_CODEBASE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXCLUDE_DIRS = [
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".pitchdeck_chroma", ".playwright-mcp", ".pytest_cache", ".fungus_cache",
    "openfang/target", "multiseat-os/downloads", "deck_charts", "deck_images",
    ".pitchdeck_chroma",
]
# Files that contain our own test queries (would pollute search results)
EXCLUDE_FILES = [
    "test_search.py", "test_persistent.py", "build_index.py", "build_optimized.py",
    "check_cache.py", "verify_index.py",
]
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_FILES = 5000
CHUNK_WINDOW = [200]  # Single window to reduce overlap


def simhash(text, hashbits=64):
    """Compute SimHash fingerprint for near-duplicate detection."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    tokens = re.findall(r'\w+', text.lower())
    v = [0] * hashbits
    for token in tokens:
        try:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
        except Exception:
            continue
        if not isinstance(token_hash, int):
            print(f"[simhash] non-int token_hash: type={type(token_hash).__name__} val={repr(token_hash)[:80]} token={repr(token)[:40]}")
            continue
        for i in range(hashbits):
            try:
                bit = token_hash & (1 << i)
            except TypeError as e:
                print(f"[simhash] BOOM & at i={i} token_hash_type={type(token_hash).__name__} repr={repr(token_hash)[:80]}")
                raise
            if bit:
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(hashbits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(a, b):
    return bin(a ^ b).count('1')


def deduplicate_chunks(chunks, threshold=3):
    """Remove near-duplicate chunks using SimHash with bucket optimization.
    O(n * b) instead of O(n²), where b = average bucket size."""
    # Bucket by first 8 bits of simhash for fast lookup
    buckets = {}  # bucket_key -> [(simhash, index)]
    unique = []
    dupes = 0

    for chunk in chunks:
        h = simhash(chunk)
        bucket_key = h & 0xFF  # first 8 bits
        # Check nearby buckets (bucket_key ± 1 to catch edge cases)
        is_dupe = False
        for bk in [bucket_key, (bucket_key + 1) & 0xFF, (bucket_key - 1) & 0xFF]:
            for seen_hash, _ in buckets.get(bk, []):
                if hamming_distance(h, seen_hash) <= threshold:
                    is_dupe = True
                    dupes += 1
                    break
            if is_dupe:
                break
        if not is_dupe:
            buckets.setdefault(bucket_key, []).append((h, len(unique)))
            unique.append(chunk)
    return unique, dupes


def filter_chunks(chunks):
    """Remove chunks that are too short or from test artifacts."""
    filtered = []
    skipped = 0
    for chunk in chunks:
        # Skip very short chunks (< 50 chars of actual content)
        content = chunk.strip()
        if len(content) < 50:
            skipped += 1
            continue
        # Skip chunks from our own test/build scripts
        first_line = content.split('\n')[0] if content else ""
        skip = False
        for excl in EXCLUDE_FILES:
            if excl in first_line:
                skip = True
                break
        if skip:
            skipped += 1
            continue
        filtered.append(chunk)
    return filtered, skipped


def main():
    from embeddinggemma.mcmp_rag import MCPMRetriever
    from embeddinggemma.ui.corpus import collect_codebase_chunks

    print(f"=== Optimized Index Build ===")
    print(f"Codebase: {CODEBASE}")
    print(f"Model: {EMBED_MODEL}")
    print()

    # 1. Load model
    t0 = time.time()
    r = MCPMRetriever(
        embedding_model_name=EMBED_MODEL,
        num_agents=50,
        max_iterations=10,
        device_mode="auto",
        embed_batch_size=256,
    )
    print(f"[1] Model loaded: {time.time()-t0:.1f}s | dim={r.embedding_model.get_sentence_embedding_dimension()}")

    # 2. Collect chunks
    t0 = time.time()
    raw_chunks = collect_codebase_chunks(
        root_dir=CODEBASE,
        windows=CHUNK_WINDOW,
        max_files=MAX_FILES,
        exclude_dirs=EXCLUDE_DIRS,
    )
    print(f"[2] Raw chunks: {len(raw_chunks)} | {time.time()-t0:.1f}s")

    # 3. Filter
    t0 = time.time()
    filtered, filter_skipped = filter_chunks(raw_chunks)
    print(f"[3] After filter: {len(filtered)} (removed {filter_skipped} short/test) | {time.time()-t0:.1f}s")

    # 4. Deduplicate
    t0 = time.time()
    deduped, dupe_count = deduplicate_chunks(filtered, threshold=3)
    print(f"[4] After dedup: {len(deduped)} (removed {dupe_count} near-dupes) | {time.time()-t0:.1f}s")

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

    # 7. Test search quality
    print(f"\n=== Search Quality Test ===")
    queries = [
        ("gaze calibration eye tracking", "ga_calibrator.py or eyeterm"),
        ("vulnerability CVE NIST scanner", "poc_vuln_scanner"),
        ("intent classification event routing", "intent_classifier.py or intent_orchestrator"),
        ("blink detection fatigue score", "blink_tracker.py"),
        ("OpenFang agent send message", "openfang_tools.py or agent wrapper"),
        ("pitch deck generator slides", "pitch_deck_agent.py"),
        ("email personalizer template", "email_personalizer_agent.py"),
        ("canary honeypot file access", "canary.py"),
        ("brain radial attention network", "radial_attention.py"),
        ("websocket transport electron", "ws_transport.py or openfang_bridge"),
    ]

    total_time = 0
    for query, expected in queries:
        t0 = time.time()
        results = r.search_direct(query, top_k=3)
        st = time.time() - t0
        total_time += st
        hits = results.get("results", [])
        top = hits[0] if hits else {}
        score = top.get("relevance_score", 0)
        content = top.get("content", "")
        # Extract file from header
        m = re.search(r'# file: (.+?) \|', content)
        found_file = m.group(1) if m else content[:60].encode("ascii", errors="replace").decode()
        match = "OK" if any(e.lower() in found_file.lower() for e in expected.split(" or ")) else "MISS"
        print(f"  [{match}] {score:.3f} | {query[:40]:40s} -> {found_file[:50]}")

    avg = total_time / len(queries)
    print(f"\n  Avg search: {avg*1000:.0f}ms | Total: {total_time:.2f}s")


if __name__ == "__main__":
    main()
