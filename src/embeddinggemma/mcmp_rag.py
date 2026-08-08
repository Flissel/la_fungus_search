#!/usr/bin/env python3
"""
MCPM-RAG (Deprecated facade)

Compatibility layer that exposes the historical `MCPMRetriever` API while
delegating all core functionality to the refactored `embeddinggemma.mcmp.*`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
import logging
import os
import time
import warnings
import numpy as np

from embeddinggemma.mcmp.embeddings import load_embedding_backend
from embeddinggemma.mcmp.simulation import (
    spawn_agents as _spawn_agents,
    update_agent_position as _update_agent_position,
    deposit_pheromones as _deposit_pheromones,
    decay_pheromones as _decay_pheromones,
    update_document_relevance as _update_document_relevance,
)
from embeddinggemma.mcmp.pca import pca_2d as _pca_2d
from embeddinggemma.mcmp.pca import pca_fit_transform as _pca_fit_transform
from embeddinggemma.mcmp.visualize import build_snapshot as _build_snapshot
from embeddinggemma.mcmp.indexing import build_faiss_index as _build_faiss
from embeddinggemma.mcmp.indexing import faiss_search as _faiss_search
from embeddinggemma.mcmp.indexing import save_faiss_index as _save_faiss
from embeddinggemma.mcmp.indexing import load_faiss_index as _load_faiss


_logger = logging.getLogger("MCMP.Facade")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


@dataclass
class Document:
    id: int
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    visit_count: int = 0
    last_visited: float = 0.0


@dataclass
class Agent:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    energy: float = 1.0
    trail_strength: float = 1.0
    visited_docs: Set[int] = field(default_factory=set)
    exploration_factor: float = 0.3
    age: int = 0


class MCPMRetriever:
    def __init__(self,
                 num_agents: int = 200,
                 max_iterations: int = 50,
                 pheromone_decay: float = 0.95,
                 exploration_bonus: float = 0.1,
                 embed_batch_size: int = 128,
                 build_faiss_after_add: bool = True,
                 force_cpu: bool = False,
                 embedding_backend: Optional[Tuple[Any, int]] = None,
                 time_source: Callable[[], float] | None = None):
        warnings.warn(
            "MCPMRetriever is deprecated as a facade. Internals are under embeddinggemma.mcmp.*",
            DeprecationWarning,
            stacklevel=2,
        )
        self.num_agents = int(num_agents)
        self.max_iterations = int(max_iterations)
        self.pheromone_decay = float(pheromone_decay)
        self.exploration_bonus = float(exploration_bonus)
        self.embed_batch_size = int(embed_batch_size)
        self.build_faiss_after_add = bool(build_faiss_after_add)
        self.force_cpu = bool(force_cpu)
        self.time_source = time.time if time_source is None else time_source
        if embedding_backend is None:
            self.embedding_model, self._expected_embedding_dim = load_embedding_backend()
        else:
            self.embedding_model, self._expected_embedding_dim = embedding_backend

        self.documents: List[Document] = []
        self.agents: List[Agent] = []
        self.pheromone_trails: Dict[Tuple[int, int], float] = {}
        self._current_query_embedding: Optional[np.ndarray] = None
        self._faiss_index = None
        self._embed_dim: Optional[int] = None
        self.log_every: int = 1

        # GPU cache used in mcmp.simulation.update_document_relevance if available
        self._doc_emb_torch = None  # type: ignore
        self._doc_emb_torch_norm = None  # type: ignore

        # Keyword knobs consumed by update_document_relevance
        self.kw_lambda: float = 0.0
        self.kw_terms: Set[str] = set()

        # Expose Agent constructor for simulation module
        # simulation.spawn_agents expects `retr.Agent` to be present
        self.Agent = Agent  # type: ignore[attr-defined]

    # ---- Persistence ----
    # Use absolute path based on package location so it works from any CWD
    _CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".fungus_cache")
    _FAISS_FILE = "faiss.index"
    _EMB_FILE = "embeddings.npz"
    _CHUNKS_FILE = "chunks.json"

    def _cache_path(self, filename: str) -> str:
        import os
        os.makedirs(self._CACHE_DIR, exist_ok=True)
        return os.path.join(self._CACHE_DIR, filename)

    def save_persistent_index(self) -> bool:
        """Save FAISS index + embeddings + chunk texts to disk."""
        import os
        if not self.documents:
            return False
        try:
            # Save FAISS index
            _save_faiss(self._faiss_index, self._cache_path(self._FAISS_FILE))
            # Save embeddings as compressed numpy
            mat = np.array([d.embedding for d in self.documents], dtype=np.float32)
            np.savez_compressed(self._cache_path(self._EMB_FILE), embeddings=mat)
            # Save chunk texts (for document reconstruction)
            import json
            texts = [d.content for d in self.documents]
            with open(self._cache_path(self._CHUNKS_FILE), "w", encoding="utf-8") as f:
                json.dump(texts, f, ensure_ascii=False)
            _logger.info("Persistent index saved: %d docs, %d dim", len(self.documents), mat.shape[1])
            return True
        except Exception as e:
            _logger.warning("Failed to save persistent index: %s", e)
            return False

    def load_persistent_index(self) -> bool:
        """Load FAISS index + embeddings + chunks from disk. Skips embedding computation."""
        import os, json
        faiss_path = self._cache_path(self._FAISS_FILE)
        emb_path = self._cache_path(self._EMB_FILE)
        chunks_path = self._cache_path(self._CHUNKS_FILE)

        # Embeddings + chunks are required; FAISS index is optional (rebuilt from embeddings)
        if not os.path.exists(emb_path) or not os.path.exists(chunks_path):
            return False

        try:
            # Load chunks
            with open(chunks_path, "r", encoding="utf-8") as f:
                texts = json.load(f)
            # Load embeddings
            data = np.load(emb_path)
            embs = data["embeddings"]
            if embs.ndim != 2:
                raise RuntimeError(
                    "Embedding cache rebuild required: cached embeddings must be a "
                    "two-dimensional matrix."
                )
            if len(texts) != embs.shape[0]:
                _logger.warning("Cache mismatch: %d texts vs %d embeddings", len(texts), embs.shape[0])
                return False
            cached_dim = int(embs.shape[1])
            if cached_dim != self._expected_embedding_dim:
                raise RuntimeError(
                    "Embedding cache rebuild required: cached dimension "
                    f"{cached_dim} does not match configured dimension "
                    f"{self._expected_embedding_dim}."
                )
            # Reconstruct documents
            self.documents = [
                Document(id=i, content=text, embedding=embs[i], metadata={})
                for i, text in enumerate(texts)
            ]
            self._embed_dim = cached_dim
            # Load FAISS index
            self._faiss_index = _load_faiss(faiss_path, gpu=True)
            if self._faiss_index is None:
                # Rebuild from cached embeddings (fast, no re-embedding needed)
                self._faiss_index = _build_faiss(embs, self._embed_dim)

            _logger.info("Persistent index loaded: %d docs, %d dim", len(self.documents), self._embed_dim)
            return True
        except RuntimeError as e:
            if str(e).startswith("Embedding cache rebuild required:"):
                raise
            _logger.warning("Failed to load persistent index: %s", e)
            return False
        except Exception as e:
            _logger.warning("Failed to load persistent index: %s", e)
            return False

    # ---- Public API ----
    def add_documents(self, docs: List[str], cache: bool = True) -> None:
        start_id = len(self.documents)
        contents = list(docs or [])
        embs: List[np.ndarray] = []
        bs = max(1, self.embed_batch_size)
        for i in range(0, len(contents), bs):
            batch = contents[i:i+bs]
            vecs = self.embedding_model.encode(batch)
            if len(vecs) != len(batch):
                raise RuntimeError(
                    "embedding-service response count does not match the request: "
                    f"expected {len(batch)}, got {len(vecs)}."
                )
            for vector in vecs:
                embedding = np.array(vector, dtype=np.float32)
                if embedding.ndim != 1 or embedding.shape[0] != self._expected_embedding_dim:
                    actual_dim = embedding.shape[0] if embedding.ndim == 1 else "non-vector"
                    raise RuntimeError(
                        "embedding-service response has unexpected dimension: "
                        f"expected dimension {self._expected_embedding_dim}, got {actual_dim}."
                    )
                embs.append(embedding)
        for i, (text, emb) in enumerate(zip(contents, embs)):
            self.documents.append(Document(id=start_id + i, content=text, embedding=emb, metadata={}))
        self._embed_dim = int(self.documents[0].embedding.shape[0]) if self.documents else None
        if self.build_faiss_after_add and self._embed_dim:
            try:
                mat = np.array([d.embedding for d in self.documents], dtype=np.float32)
                self._faiss_index = _build_faiss(
                    mat, int(self._embed_dim), force_cpu=self.force_cpu
                )
            except Exception as e:
                _logger.warning("FAISS index build failed: %s", e)
                self._faiss_index = None
        # Persist to disk for fast restart
        if cache and self.documents:
            self.save_persistent_index()

    def add_documents_incremental(self, docs: List[str]) -> Dict[str, int]:
        """Add only NEW documents (not already indexed). Returns stats."""
        existing = {d.content for d in self.documents}
        new_docs = [d for d in docs if d not in existing]
        if not new_docs:
            return {"new": 0, "total": len(self.documents), "skipped": len(docs)}
        self.add_documents(new_docs, cache=True)
        return {"new": len(new_docs), "total": len(self.documents), "skipped": len(docs) - len(new_docs)}

    def clear_documents(self) -> None:
        self.documents.clear()
        self._doc_emb_torch = None
        self._doc_emb_torch_norm = None
        self._faiss_index = None
        self._embed_dim = None

    def initialize_simulation(self, query: str) -> bool:
        if not self.documents:
            return False
        q = self.embedding_model.encode([query])[0]
        if len(q) != self._expected_embedding_dim:
            raise RuntimeError(
                "embedding-service response has unexpected dimension: "
                f"expected dimension {self._expected_embedding_dim}, got {len(q)}."
            )
        self._current_query_embedding = np.array(q, dtype=np.float32)
        self.spawn_agents(self._current_query_embedding)
        self.pheromone_trails = {}
        for d in self.documents:
            d.visit_count = 0
            d.last_visited = 0.0
            d.relevance_score = 0.0
        return True

    def step(self, n_steps: int = 1) -> Dict[str, Any]:
        if not self.documents or self._current_query_embedding is None or not self.agents:
            return {"error": "Simulation not initialized"}
        for _ in range(max(1, int(n_steps))):
            for a in self.agents:
                self.update_agent_position(a, 0)
                self.deposit_pheromones(a)
            self.update_document_relevance(self._current_query_embedding)
            self.decay_pheromones()
        avg_rel = float(np.mean([d.relevance_score for d in self.documents]))
        return {"avg_relevance": avg_rel, "steps": int(n_steps), "pheromone_trails": len(self.pheromone_trails)}

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        if not self.documents:
            return {"results": []}
        return self.search_direct(query, top_k)

    def search_direct(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Fast direct cosine search — no MCMP simulation needed.
        Uses FAISS if available, otherwise brute-force numpy."""
        if not self.documents:
            return {"results": []}
        q = np.array(self.embedding_model.encode([query])[0], dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self._expected_embedding_dim:
            actual_dim = q.shape[0] if q.ndim == 1 else "non-vector"
            raise RuntimeError(
                "embedding-service response has unexpected dimension: "
                f"expected dimension {self._expected_embedding_dim}, got {actual_dim}."
            )
        top_k = min(int(top_k), len(self.documents))

        # Try FAISS first
        if self._faiss_index is not None:
            try:
                distances, indices = _faiss_search(self._faiss_index, q, top_k)
                results = []
                for i, dist in zip(indices, distances):
                    i = int(i)
                    if 0 <= i < len(self.documents):
                        d = self.documents[i]
                        sim = max(0.0, float(dist))  # inner product = cosine for normalized
                        results.append({"content": d.content, "metadata": d.metadata, "relevance_score": sim})
                return {"results": results}
            except Exception as e:
                _logger.warning("search_direct FAISS failed, falling back: %s", e)

        # Brute-force cosine
        mat = np.array([d.embedding for d in self.documents], dtype=np.float32)
        mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        q_n = q / (np.linalg.norm(q) + 1e-12)
        sims = mat_n @ q_n
        idx = np.argsort(-sims)[:top_k]
        return {"results": [
            {"content": self.documents[int(i)].content, "metadata": self.documents[int(i)].metadata, "relevance_score": float(sims[int(i)])}
            for i in idx
        ]}

    def get_visualization_snapshot(self,
                                   min_trail_strength: float = 0.05,
                                   max_edges: int = 300,
                                   method: str = "pca",
                                   whiten: bool = False,
                                   spread: float = 1.0,
                                   jitter: float = 0.0,
                                   dims: int = 2) -> Dict[str, Any]:
        if not self.documents:
            return {"documents": {"xy": [], "relevance": []}, "agents": {"xy": []}, "edges": []}
        embs = np.array([d.embedding for d in self.documents], dtype=np.float32)
        k = 3 if int(dims) == 3 else 2
        # Cache PCA basis for stable layout and consistent agent projection
        if not hasattr(self, "_viz_pca"):
            self._viz_pca = {}
        if self._viz_pca.get("k") != k:
            self._viz_pca.clear()
        if not self._viz_pca:
            coords, mean, comps, S = _pca_fit_transform(embs, n_components=k, whiten=bool(whiten))
            self._viz_pca = {"mean": mean, "comps": comps, "S": S, "k": k, "whiten": bool(whiten)}
        else:
            mean = self._viz_pca["mean"]
            comps = self._viz_pca["comps"]
            S = self._viz_pca.get("S")
            coords = (embs - mean) @ comps.T
            if bool(self._viz_pca.get("whiten")) and S is not None:
                s = S[:k]
                safe = np.array([sv if sv != 0 else 1.0 for sv in s])
                coords = coords / safe
        coords = coords if coords is not None else np.zeros((len(self.documents), k), dtype=np.float32)
        rels = [float(d.relevance_score) for d in self.documents]
        meta = [{
            "id": int(d.id),
            "score": float(d.relevance_score),
            "visits": int(d.visit_count),
            "snippet": (d.content or '')[:140] if hasattr(d, 'content') else ''
        } for d in self.documents]
        trails = {k: v for k, v in (self.pheromone_trails or {}).items() if float(v) >= float(min_trail_strength)}
        agents_xy = None
        try:
            if self.agents and self._viz_pca.get("comps") is not None:
                import numpy as _np
                mean = self._viz_pca["mean"]
                comps = self._viz_pca["comps"]
                S = self._viz_pca.get("S")
                A = _np.array([getattr(a, 'position', None) for a in self.agents if getattr(a, 'position', None) is not None], dtype=_np.float32)
                if A.size:
                    agents_xy = (A - mean) @ comps.T
                    if bool(self._viz_pca.get("whiten")) and S is not None:
                        s = S[:k]
                        safe = _np.array([sv if sv != 0 else 1.0 for sv in s], dtype=_np.float32)
                        agents_xy = agents_xy / safe
        except Exception:
            agents_xy = None
        return _build_snapshot(coords, rels, trails, meta, agents_xy, max_edges=int(max_edges))

    # ---- Public getters for frontend/live updates ----
    def get_query_embedding(self) -> Optional[np.ndarray]:
        return None if self._current_query_embedding is None else self._current_query_embedding.copy()

    def get_agent_positions(self) -> np.ndarray:
        return np.array([a.position for a in self.agents], dtype=np.float32) if self.agents else np.zeros((0, 0), dtype=np.float32)

    def get_doc_embeddings(self) -> np.ndarray:
        return np.array([d.embedding for d in self.documents], dtype=np.float32) if self.documents else np.zeros((0, 0), dtype=np.float32)

    def get_doc_relevances(self) -> List[Tuple[int, float]]:
        return [(d.id, float(d.relevance_score)) for d in self.documents]

    def get_pheromone_trails(self) -> Dict[Tuple[int, int], float]:
        return dict(self.pheromone_trails)

    def get_snapshot(self,
                     min_trail_strength: float = 0.05,
                     max_edges: int = 300,
                     method: str = "pca",
                     whiten: bool = False,
                     dims: int = 2) -> Dict[str, Any]:
        return self.get_visualization_snapshot(
            min_trail_strength=min_trail_strength,
            max_edges=max_edges,
            method=method,
            whiten=whiten,
            dims=dims,
        )

    # ---- Delegates consumed by simulation.* ----
    def spawn_agents(self, query_embedding: np.ndarray) -> None:
        _spawn_agents(self, query_embedding)

    def update_agent_position(self, agent: Agent, iteration: int) -> None:
        _update_agent_position(self, agent, iteration)

    def deposit_pheromones(self, agent: Agent) -> None:
        _deposit_pheromones(self, agent)

    def decay_pheromones(self) -> None:
        _decay_pheromones(self)

    def update_document_relevance(self, query_embedding: np.ndarray) -> None:
        _update_document_relevance(self, query_embedding)

    # ---- Helpers for simulation.* ----
    def find_nearest_documents(self, position: np.ndarray, k: int = 3) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []
        k = min(int(k), len(self.documents))
        pos = np.array(position, dtype=np.float32).reshape(1, -1)

        # Fast path: use FAISS index if available (O(log n) vs O(n))
        if self._faiss_index is not None:
            try:
                similarities, indices = _faiss_search(self._faiss_index, pos.squeeze(), k)
                results = []
                for i, similarity in zip(indices, similarities):
                    i = int(i)
                    if 0 <= i < len(self.documents):
                        results.append((self.documents[i], float(similarity)))
                return results
            except Exception:
                pass  # Fall through to brute-force

        # Fallback: brute-force cosine similarity (slow for large N)
        mat = np.array([d.embedding for d in self.documents], dtype=np.float32)
        pos_n = pos.squeeze() / (np.linalg.norm(pos) + 1e-12)
        mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = mat_n @ pos_n
        idx = np.argsort(-sims)[:k]
        return [(self.documents[int(i)], float(sims[int(i)])) for i in idx]



