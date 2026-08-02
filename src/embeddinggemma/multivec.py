"""ColBERT-Lite: cheap multi-vector index over the existing bi-encoder.

Concept: instead of 1 vector per chunk (which loses information when a chunk
contains multiple distinct concepts), we embed N "views" of each chunk:

    view 1: the header line + path tokens          (file identity)
    view 2: first half of the body                  (early code: signatures, imports)
    view 3: second half of the body                 (late code: implementations)

At query time we embed the user query (1 vector) and score each chunk by
the MAX cosine over its N views (late interaction; this is the ColBERT
"MaxSim" operator collapsed to chunk-level granularity).

Why this works better than single-vector for naming gaps:
    A chunk like `face_math/landmark_detector.py` will have a strong header
    view ("face landmark detector"), even if the body uses different vocab
    (MediaPipe API names). Single-vector averages everything; multi-vector
    keeps the strongest signal accessible.

Index footprint: 3× the original embeddings.npz (~250 MB for 45k chunks).
Build cost: 3× the embedder forward-pass time (~20 min on RTX 3060).
Search cost: same FAISS lookup over 3× larger flat index, then group-by-chunk.
"""
from __future__ import annotations
import os
import re
import time
import json
import logging
import numpy as np
from typing import List, Tuple

_logger = logging.getLogger("MultiVec")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


_HEADER_LINES = ("# file:", "# path:", "# tokens:")
_VIEWS_PER_CHUNK = 3  # header, body_first, body_second


def split_chunk_into_views(content: str) -> List[str]:
    """Return three views of a chunk for ColBERT-Lite indexing.

    view[0]: just the # path / # tokens header — strongest "identity" signal
    view[1]: first half of body
    view[2]: second half of body

    If the chunk is too short to split, returns 1 or 2 non-empty views.
    """
    lines = content.split("\n")
    header_parts: List[str] = []
    body_lines: List[str] = []
    for ln in lines:
        if ln.startswith(_HEADER_LINES):
            header_parts.append(ln)
        else:
            body_lines.append(ln)
    header = "\n".join(header_parts).strip()
    body = "\n".join(body_lines).strip()
    if not body:
        # All header (rare) — single view.
        return [header] if header else [content[:200]]
    # Split body in half by line count.
    mid = max(1, len(body_lines) // 2)
    body_first = "\n".join(body_lines[:mid]).strip()
    body_second = "\n".join(body_lines[mid:]).strip()
    views = []
    if header:
        views.append(header)
    if body_first:
        views.append(body_first)
    if body_second and body_second != body_first:
        views.append(body_second)
    return views


def build_multivec_index(chunks: List[str], embedding_model, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Compute multi-vector embeddings for all chunks.

    Returns:
        embeddings: (total_views, dim) float32 — all views concatenated
        chunk_ids:  (total_views,)  int32     — which chunk each view belongs to
    """
    t0 = time.time()
    # Pass 1: split each chunk into views, collect a flat list with chunk_ids
    all_views: List[str] = []
    chunk_ids: List[int] = []
    for i, chunk in enumerate(chunks):
        for v in split_chunk_into_views(chunk):
            all_views.append(v)
            chunk_ids.append(i)
    _logger.info("multivec build: %d chunks → %d views (%.1fx)",
                 len(chunks), len(all_views), len(all_views) / max(1, len(chunks)))

    # Pass 2: embed through the configured VibeMind embedding-service.  The client
    # owns transport, retries, and provider policy; local model kwargs would
    # be an accidental bypass.
    batches = []
    for start in range(0, len(all_views), max(1, batch_size)):
        batch = all_views[start:start + max(1, batch_size)]
        vectors = embedding_model.encode(batch)
        if len(vectors) != len(batch):
            raise RuntimeError(
                "embedding-service response count does not match the request: "
                f"expected {len(batch)}, got {len(vectors)}."
            )
        batches.append(np.asarray(vectors, dtype=np.float32))
    embs = np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)
    _logger.info("multivec build: embedded %d views in %.1fs → shape %s",
                 len(all_views), time.time() - t0, embs.shape)
    return embs, np.array(chunk_ids, dtype=np.int32)


def save_multivec(path_npz: str, embeddings: np.ndarray, chunk_ids: np.ndarray) -> None:
    np.savez_compressed(path_npz, embeddings=embeddings, chunk_ids=chunk_ids)
    _logger.info("multivec saved: %s (%.1f MB)", path_npz,
                 os.path.getsize(path_npz) / 1024**2)


def load_multivec(path_npz: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path_npz)
    return z["embeddings"], z["chunk_ids"]


# ── MaxSim retrieval ────────────────────────────────────────────────────
def maxsim_search(query_vec: np.ndarray,
                  view_embs: np.ndarray,
                  chunk_ids: np.ndarray,
                  top_k: int = 10) -> List[Tuple[int, float]]:
    """MaxSim over a single query vector against all chunk views.

    For each chunk, take the MAX cosine-similarity over all its views.
    Returns list of (chunk_id, max_sim) sorted desc, length top_k.

    All views are L2-normalised already (we set normalize_embeddings=True
    on build), so a single matrix-vector product gives cosine directly.
    """
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    # (n_views, dim) @ (1, dim).T = (n_views, 1)
    sims = (view_embs @ q.T).reshape(-1)
    # Group by chunk_id and take max.
    n_chunks = int(chunk_ids.max()) + 1 if chunk_ids.size else 0
    chunk_max = np.full(n_chunks, -np.inf, dtype=np.float32)
    np.maximum.at(chunk_max, chunk_ids, sims)
    # Top-k chunks by their max-sim.
    top_idx = np.argpartition(-chunk_max, min(top_k, n_chunks - 1))[:top_k]
    # Sort the partition.
    top_idx = top_idx[np.argsort(-chunk_max[top_idx])]
    return [(int(i), float(chunk_max[i])) for i in top_idx]


def maxsim_search_multi_query(query_vecs: np.ndarray,
                              view_embs: np.ndarray,
                              chunk_ids: np.ndarray,
                              top_k: int = 10) -> List[Tuple[int, float]]:
    """Full ColBERT-style MaxSim: multiple query vectors → sum-of-max over chunks.

    For each query "token" vector q_t, find the max cosine over the chunk's views.
    Then sum these maxes across all query tokens. This is the "Sum of MaxSim"
    operator that gives ColBERT its precision: each part of the query finds
    its best matching part of the chunk.
    """
    if query_vecs.ndim == 1:
        query_vecs = query_vecs[None, :]
    Q = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-12)
    # sims: (n_views, n_qtok)
    sims = view_embs @ Q.T
    # Group by chunk_id: for each q-tok, max over its views, then sum.
    n_chunks = int(chunk_ids.max()) + 1 if chunk_ids.size else 0
    n_qtok = sims.shape[1]
    chunk_scores = np.zeros(n_chunks, dtype=np.float32)
    for qi in range(n_qtok):
        col = sims[:, qi]
        chunk_max = np.full(n_chunks, -np.inf, dtype=np.float32)
        np.maximum.at(chunk_max, chunk_ids, col)
        # Replace -inf with 0 (chunks with no views matter only via their views).
        chunk_max[chunk_max == -np.inf] = 0.0
        chunk_scores += chunk_max
    top_idx = np.argpartition(-chunk_scores, min(top_k, n_chunks - 1))[:top_k]
    top_idx = top_idx[np.argsort(-chunk_scores[top_idx])]
    return [(int(i), float(chunk_scores[i])) for i in top_idx]


# ── Query "tokenisation" for multi-query MaxSim ─────────────────────────
def split_query_into_phrases(query: str, max_phrases: int = 6) -> List[str]:
    """Cheap query splitter: returns the full query + N noun-phrase-ish slices.

    Goal: produce N "views" of the query that each focus on one aspect.
    We don't have a real NP-chunker so we use heuristics:
    - the full query (catches the gestalt meaning)
    - each comma- or " and "-separated chunk
    - moving 3-gram windows over content words
    - rare capitalised tokens kept whole (likely identifiers)
    """
    phrases: List[str] = []
    seen: set[str] = set()

    def push(p: str):
        p = p.strip()
        if 4 <= len(p) <= 200 and p.lower() not in seen:
            seen.add(p.lower())
            phrases.append(p)

    push(query)
    for part in re.split(r",| and | or |;|\n", query):
        push(part)
    # 3-gram windows of content words (no stopwords)
    stop = {"the","and","for","with","from","this","that","what","is","are","to","of","a","an","in","on"}
    words = [w for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", query)
             if w.lower() not in stop]
    for i in range(len(words) - 2):
        push(" ".join(words[i:i+3]))
    # Preserve any identifier-like token (capital letter or _) on its own
    for w in words:
        if any(c.isupper() for c in w[1:]) or "_" in w:
            push(w)
    return phrases[:max_phrases]
