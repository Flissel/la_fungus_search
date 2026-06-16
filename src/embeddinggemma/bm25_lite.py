"""Lightweight BM25 scorer for hybrid search — pure numpy, no external dep.

Built for fungus' chunk store: ~45k chunks, ~300 tokens each. After init,
score() takes ~30ms for a single query against the full corpus.

Design:
- Tokenizer: lowercase, split on non-alphanumerics, drop tokens <3 chars.
  Code identifiers like `agent_task_v2` keep their underscores so a query
  for `"agent_task_v2"` gets exact match; we also split camelCase so a
  query for `"face landmark"` matches `FaceLandmarkDetector`.
- Stopwords: tiny English+German set (avoid filtering domain terms).
- IDF: smoothed Okapi BM25 (`log((N-df+0.5)/(df+0.5) + 1)`).
- Score: vectorised over all docs; returns float32 numpy array of length N.

Cached to disk alongside the FAISS index (`.fungus_cache/bm25.npz`).
"""
from __future__ import annotations
import os
import re
import math
import time
import logging
import numpy as np
from typing import List, Dict

_logger = logging.getLogger("BM25Lite")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


_STOPWORDS = {
    # tiny EN+DE set — only true stopwords, no domain terms
    "the", "and", "for", "with", "from", "this", "that", "what", "which",
    "where", "how", "who", "are", "is", "was", "were", "has", "have", "had",
    "can", "use", "all", "any", "not", "but", "into", "out", "by", "to", "in",
    "on", "at", "of", "as", "or", "an", "be", "if", "it", "we", "you", "he",
    "she", "do", "does", "did", "der", "die", "das", "und", "ist", "wie",
    "wo", "wer", "was", "ein", "eine", "den", "des", "dem", "von", "mit",
    "auf", "im", "zu", "es", "im",
}

# Split path-style identifiers + camelCase + snake_case → individual tokens.
# Keep underscore as a token boundary BUT also keep the joined identifier
# itself; we add both forms to the tokens list so exact-string queries hit.
_SPLIT_RE = re.compile(r"[^a-zA-Z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_")


def tokenize(text: str) -> List[str]:
    """Tokenise text → list of lowercase tokens length >= 3, no stopwords.

    Each compound identifier is kept verbatim AND split (so `FaceLandmarkDetector`
    yields ['facelandmarkdetector', 'face', 'landmark', 'detector']).
    """
    out: List[str] = []
    for tok in _SPLIT_RE.split(text or ""):
        if not tok:
            continue
        low = tok.lower()
        if len(low) >= 3 and low not in _STOPWORDS:
            out.append(low)
        # also split camel/snake compounds
        for part in _CAMEL_RE.split(tok):
            p = part.lower()
            if p and p != low and len(p) >= 3 and p not in _STOPWORDS:
                out.append(p)
    return out


class BM25Lite:
    """Inverted-index BM25 with vectorised query scoring."""
    k1: float = 1.5
    b: float = 0.75

    def __init__(self) -> None:
        self.N: int = 0
        # Per-token: doc_freq (df), and an inverted index {tok: {doc_id: tf}}.
        # For 45k chunks × ~300 tokens × ~50% unique, the inverted index is
        # roughly ~5 MB Python objects — fine.
        self.doc_freq: Dict[str, int] = {}
        self.postings: Dict[str, Dict[int, int]] = {}
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.int32)
        self.avgdl: float = 0.0
        self._idf_cache: Dict[str, float] = {}

    # --------------------------------------------------------------- build
    def fit(self, docs: List[str]) -> None:
        t0 = time.time()
        self.N = len(docs)
        self.doc_freq = {}
        self.postings = {}
        lens = np.zeros(self.N, dtype=np.int32)
        for i, text in enumerate(docs):
            toks = tokenize(text)
            lens[i] = len(toks)
            seen: Dict[str, int] = {}
            for t in toks:
                seen[t] = seen.get(t, 0) + 1
            for t, tf in seen.items():
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
                if t not in self.postings:
                    self.postings[t] = {}
                self.postings[t][i] = tf
        self.doc_len = lens
        self.avgdl = float(lens.mean()) if self.N else 0.0
        self._idf_cache = {}
        _logger.info("BM25 fit: N=%d vocab=%d avgdl=%.1f | %.1fs",
                     self.N, len(self.doc_freq), self.avgdl, time.time() - t0)

    # --------------------------------------------------------------- score
    def _idf(self, term: str) -> float:
        c = self._idf_cache.get(term)
        if c is not None:
            return c
        df = self.doc_freq.get(term, 0)
        if df == 0:
            self._idf_cache[term] = 0.0
            return 0.0
        v = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = v
        return v

    def score(self, query: str, candidate_ids: np.ndarray | None = None) -> np.ndarray:
        """BM25 scores for the query against either all docs or a candidate set.

        Returns float32 array. If candidate_ids is None → length N (all docs);
        otherwise length len(candidate_ids), in the same order.
        """
        if self.N == 0:
            return np.zeros(0, dtype=np.float32)
        terms = list(set(tokenize(query)))
        if candidate_ids is None:
            out = np.zeros(self.N, dtype=np.float32)
            idx_iter = None
        else:
            out = np.zeros(len(candidate_ids), dtype=np.float32)
            idx_iter = {int(c): k for k, c in enumerate(candidate_ids)}
        for t in terms:
            posting = self.postings.get(t)
            if not posting:
                continue
            idf = self._idf(t)
            if idf == 0.0:
                continue
            if idx_iter is None:
                for doc_id, tf in posting.items():
                    dl = float(self.doc_len[doc_id])
                    denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                    out[doc_id] += idf * tf * (self.k1 + 1.0) / (denom or 1.0)
            else:
                for doc_id, tf in posting.items():
                    out_pos = idx_iter.get(doc_id)
                    if out_pos is None:
                        continue
                    dl = float(self.doc_len[doc_id])
                    denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                    out[out_pos] += idf * tf * (self.k1 + 1.0) / (denom or 1.0)
        return out

    # ---------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        t0 = time.time()
        # Convert postings (dict of dicts) to two flat int32 arrays per term:
        # tokens[], offsets[len(tokens)+1], doc_ids[], tfs[].
        tokens: List[str] = []
        offsets: List[int] = [0]
        doc_ids: List[int] = []
        tfs: List[int] = []
        for tok, post in self.postings.items():
            tokens.append(tok)
            for d, c in post.items():
                doc_ids.append(d)
                tfs.append(c)
            offsets.append(len(doc_ids))
        np.savez_compressed(
            path,
            tokens=np.array(tokens, dtype=object),
            offsets=np.array(offsets, dtype=np.int64),
            doc_ids=np.array(doc_ids, dtype=np.int32),
            tfs=np.array(tfs, dtype=np.int32),
            doc_len=self.doc_len,
            meta=np.array([self.N, self.avgdl], dtype=np.float64),
        )
        _logger.info("BM25 saved: %s | %.1fs", path, time.time() - t0)

    @classmethod
    def load(cls, path: str) -> "BM25Lite":
        t0 = time.time()
        z = np.load(path, allow_pickle=True)
        self = cls()
        tokens = z["tokens"]
        offsets = z["offsets"]
        doc_ids = z["doc_ids"]
        tfs = z["tfs"]
        self.doc_len = z["doc_len"]
        meta = z["meta"]
        self.N = int(meta[0])
        self.avgdl = float(meta[1])
        self.postings = {}
        self.doc_freq = {}
        for i, tok in enumerate(tokens):
            s, e = int(offsets[i]), int(offsets[i + 1])
            post = {int(doc_ids[k]): int(tfs[k]) for k in range(s, e)}
            self.postings[str(tok)] = post
            self.doc_freq[str(tok)] = len(post)
        _logger.info("BM25 loaded: N=%d vocab=%d | %.1fs",
                     self.N, len(self.doc_freq), time.time() - t0)
        return self
