#!/usr/bin/env python3
"""
MCPM-RAG (Deprecated facade)

Compatibility layer that exposes the historical `MCPMRetriever` API while
delegating all core functionality to the refactored `embeddinggemma.mcmp.*`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
import logging
import os
import warnings
import numpy as np

from embeddinggemma.mcmp.embeddings import load_sentence_model
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
                 embedding_model_name: str = "google/embeddinggemma-300m",
                 num_agents: int = 200,
                 max_iterations: int = 50,
                 pheromone_decay: float = 0.95,
                 exploration_bonus: float = 0.1,
                 device_mode: str = "auto",
                 use_embedding_model: bool = True,
                 embed_batch_size: int = 128,
                 build_faiss_after_add: bool = True):
        warnings.warn(
            "MCPMRetriever is deprecated as a facade. Internals are under embeddinggemma.mcmp.*",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embedding_model_name = embedding_model_name
        self.num_agents = int(num_agents)
        self.max_iterations = int(max_iterations)
        self.pheromone_decay = float(pheromone_decay)
        self.exploration_bonus = float(exploration_bonus)
        self.device_mode = device_mode
        self.use_embedding_model = bool(use_embedding_model)
        self.embed_batch_size = int(embed_batch_size)
        self.build_faiss_after_add = bool(build_faiss_after_add)

        self.embedding_model = None
        if self.use_embedding_model:
            try:
                self.embedding_model = load_sentence_model(self.embedding_model_name, self.device_mode)
            except Exception as e:
                _logger.error("SentenceTransformer load failed: %s", e)
                self.embedding_model = None

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
            if len(texts) != embs.shape[0]:
                _logger.warning("Cache mismatch: %d texts vs %d embeddings", len(texts), embs.shape[0])
                return False
            # Reconstruct documents
            self.documents = [
                Document(id=i, content=text, embedding=embs[i], metadata={})
                for i, text in enumerate(texts)
            ]
            self._embed_dim = int(embs.shape[1])
            # Load FAISS index
            self._faiss_index = _load_faiss(faiss_path, gpu=True)
            if self._faiss_index is None:
                # Rebuild from cached embeddings (fast, no re-embedding needed)
                self._faiss_index = _build_faiss(embs, self._embed_dim)

            _logger.info("Persistent index loaded: %d docs, %d dim", len(self.documents), self._embed_dim)
            return True
        except Exception as e:
            _logger.warning("Failed to load persistent index: %s", e)
            return False

    # ---- Public API ----
    def add_documents(self, docs: List[str], cache: bool = True) -> None:
        start_id = len(self.documents)
        contents = list(docs or [])
        if self.embedding_model is not None:
            embs: List[np.ndarray] = []
            bs = max(1, self.embed_batch_size)
            for i in range(0, len(contents), bs):
                batch = contents[i:i+bs]
                vecs = self.embedding_model.encode(batch)
                embs.extend([np.array(v, dtype=np.float32) for v in vecs])
        else:
            rng = np.random.default_rng(42)
            embs = [rng.normal(0, 1, size=(64,)).astype(np.float32) for _ in contents]
        for i, (text, emb) in enumerate(zip(contents, embs)):
            self.documents.append(Document(id=start_id + i, content=text, embedding=emb, metadata={}))
        self._embed_dim = int(self.documents[0].embedding.shape[0]) if self.documents else None
        if self.build_faiss_after_add and self._embed_dim:
            try:
                mat = np.array([d.embedding for d in self.documents], dtype=np.float32)
                self._faiss_index = _build_faiss(mat, int(self._embed_dim))
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
        if self.embedding_model is None:
            dim = self.documents[0].embedding.shape  # type: ignore
            q = np.random.normal(0, 1, dim)
            q = q / (np.linalg.norm(q) or 1.0)
        else:
            q = self.embedding_model.encode([query])[0]
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
        # Fast path: if embedding model available, use direct cosine search
        # (bypasses slow MCMP simulation for large indices)
        if self.embedding_model is not None:
            return self.search_direct(query, top_k)
        # Fallback: MCMP simulation (requires embedding model for query encoding)
        if self._current_query_embedding is None:
            self.initialize_simulation(query)
            self.step(min(5, self.max_iterations))
        ranked = sorted(self.documents, key=lambda d: d.relevance_score, reverse=True)[:int(top_k)]
        return {"results": [
            {"content": d.content, "metadata": d.metadata, "relevance_score": float(d.relevance_score)}
            for d in ranked
        ]}

    def search_direct(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Fast direct cosine search — no MCMP simulation needed.
        Uses FAISS if available, otherwise brute-force numpy."""
        if not self.documents:
            return {"results": []}
        if self.embedding_model is None:
            _logger.warning("search_direct: no embedding model, cannot encode query")
            return {"results": []}

        q = np.array(self.embedding_model.encode([query])[0], dtype=np.float32)
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
                distances, indices = _faiss_search(self._faiss_index, pos.squeeze(), k)
                results = []
                for i, dist in zip(indices, distances):
                    i = int(i)
                    if 0 <= i < len(self.documents):
                        # FAISS L2 distance → approximate cosine similarity
                        sim = max(0.0, 1.0 - float(dist) / 2.0)
                        results.append((self.documents[i], sim))
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



