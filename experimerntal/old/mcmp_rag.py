#!/usr/bin/env python3
"""
MCPM-RAG: Monte Carlo Physarum Machine for Retrieval-Augmented Generation (Deprecated facade)

Summary
-------
This module implements a physarum-inspired Monte Carlo search to explore a
document space (embeddings) with many simple agents that leave pheromone
trails. Documents accrue relevance from query similarity, agent visits and
optional keyword boosts. The resulting pheromone network exposes transitive,
cross-file relations that nearest-neighbor search can miss.

Major parts
-----------
- Data structures: Document, Agent
- Retriever: MCPMRetriever configuration, embedding/FAISS setup
- Agent simulation: movement, pheromone deposit/decay
- Scoring: query similarity, visit/time bonuses, keyword boosts
- Extraction: relevance network and top-k selection
- Visualization helpers: 2D projections and network snapshot

Inspired by: https://github.com/CreativeCodingLab/Polyphorm
"""

import numpy as np
import logging
try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False
import os
import warnings
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import random
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict, deque
import time

@dataclass
class Document:
    """Document with metadata and search-time statistics for MCPM."""
    id: int
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    relevance_score: float = 0.0
    visit_count: int = 0
    last_visited: float = 0.0

@dataclass 
class Agent:
    """MCPM agent (physarum-inspired particle)."""
    id: int
    position: np.ndarray  # Position in the embedding space
    velocity: np.ndarray  # Direction of movement
    energy: float = 1.0   # Agent energy
    trail_strength: float = 1.0  # Pheromone trail strength
    visited_docs: Set[int] = field(default_factory=set)
    exploration_factor: float = 0.3  # Exploration vs. exploitation balance
    age: int = 0

class MCPMRetriever:
    """Monte Carlo Physarum Machine for RAG."""
    
    def __init__(self, 
                 embedding_model_name="google/embeddinggemma-300m",
                 num_agents=500,
                 max_iterations=100,
                 pheromone_decay=0.95,
                 exploration_bonus=0.1,
                 device_mode: str = "auto",
                 use_embedding_model: bool = True,
                 query_prompt_name: str = None,
                 doc_prompt_name: str = None,
                 prompts_config_path: str = "models/embeddinggemma/config_sentence_transformers.json",
                 embed_batch_size: int = 256,
                 build_faiss_after_add: bool = True):
        
        warnings.warn(
            "MCPMRetriever is deprecated as a facade. Internals are now under embeddinggemma.mcmp.*;"
            " this class will remain as a stable API wrapper.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embedding_model_name = embedding_model_name
        self.num_agents = num_agents
        self.max_iterations = max_iterations
        self.pheromone_decay = pheromone_decay
        self.exploration_bonus = exploration_bonus
        self.device_mode = device_mode  # "auto" | "cpu" | "cuda"
        self.use_embedding_model = use_embedding_model
        
        # ===== Embedding model and data =====
        self.embedding_model = None
        self.documents: List[Document] = []
        self.agents: List[Agent] = []
        
        # ===== MCPM state =====
        self.pheromone_trails = {}  # (doc_id, doc_id) -> strength
        self.relevance_network = nx.Graph()
        self.iteration_history = []
        # Cached torch doc embedding matrix for fast cosine on GPU
        self._doc_emb_torch = None  # type: ignore
        self._doc_emb_torch_norm = None  # type: ignore
        # Logging cadence
        self.log_every: int = 1
        # FAISS index for coarse retrieval
        self._faiss_index = None
        self._faiss_gpu_res = None
        self._embed_dim = None
        # Fast GPU stepping toggle (used by UI)
        self.fast_gpu_step: bool = False
        # ===== Embedding parameters =====
        self.embed_batch_size: int = int(embed_batch_size)
        self.build_faiss_after_add: bool = bool(build_faiss_after_add)
        # Keyword boost (optional)
        self.kw_lambda: float = 0.0
        self.kw_terms: set = set()
        
        # Performance Tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_convergence_time': 0,
            'network_density': 0,
            'unique_paths_found': 0
        }
        # Debug logging toggle
        self.debug_log: bool = False
        self._logger = logging.getLogger("MCPM")
        
        # Realtime Simulation / Visualization state
        self._current_query_embedding: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None

        # ===== Prompts configuration =====
        self._prompts = {}
        self._prompts_path = prompts_config_path
        try:
            if self._prompts_path and os.path.exists(self._prompts_path):
                with open(self._prompts_path, "r", encoding="utf-8", errors="ignore") as f:
                    cfg = json.load(f)
                    self._prompts = cfg.get("prompts", {}) or {}
        except Exception:
            self._prompts = {}
        # Defaults setzen mit Fallbacks
        self.query_prompt_name = query_prompt_name or ("Retrieval-query" if "Retrieval-query" in self._prompts else ("query" if "query" in self._prompts else None))
        self.doc_prompt_name = doc_prompt_name or ("Retrieval-document" if "Retrieval-document" in self._prompts else ("document" if "document" in self._prompts else None))

    def set_debug(self, enabled: bool = True, level: int = logging.INFO) -> None:
        """Enable/disable debug logging for the MCPM search loop."""
        self.debug_log = bool(enabled)
        try:
            self._logger.setLevel(int(level))
            if not self._logger.handlers:
                h = logging.StreamHandler()
                fmt = logging.Formatter('[%(levelname)s] %(message)s')
                h.setFormatter(fmt)
                self._logger.addHandler(h)
        except Exception:
            pass

    def _apply_prompt(self, name: Optional[str], text: str) -> str:
        if not name:
            return text
        prefix = self._prompts.get(name, "")
        return f"{prefix}{text}" if prefix else text
        
    def _resolve_device(self) -> str:
        """Resolve target device based on ``device_mode`` (auto|cpu|cuda)."""
        mode = (self.device_mode or "auto").lower()
        if mode == "cpu":
            return "cpu"
        if mode == "cuda":
            return "cuda"
        # auto
        try:
            import torch  # noqa: F401
            import torch.cuda as cuda
            return "cuda" if getattr(cuda, 'is_available', lambda: False)() else "cpu"
        except Exception:
            return "cpu"

    def load_embedding_model(self):
        """Load EmbeddingGemma model if enabled; return True on success."""
        if not self.use_embedding_model:
            return False
        if self.embedding_model is None:
            self._logger.info("Loading EmbeddingGemma for MCPM…")
            try:
                # Prefer shared loader if available
                try:
                    from embeddinggemma.mcmp.embeddings import load_sentence_model as _load_model  # type: ignore
                    self.embedding_model = _load_model(self.embedding_model_name, self.device_mode)
                except Exception:
                    device = self._resolve_device()
                    try:
                        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
                    except Exception as inner_e:
                        # Automatic fallback to CPU (e.g. when Torch was built without CUDA)
                        if device != "cpu":
                            self._logger.warning(f"CUDA not available ({inner_e}). Falling back to CPU…")
                            self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
                        else:
                            raise
                self._logger.info("EmbeddingGemma for MCPM loaded!")
                return True
            except Exception as e:
                self._logger.error(f"EmbeddingGemma error: {e}")
                return False
        return True
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the MCPM system with embeddings (or mock if unavailable)."""
        if not self.load_embedding_model():
            # Fallback: Mock-Embeddings erzeugen
            try:
                from mcmp_rag import Document as _Doc  # self-reference safe
                target_dim = 768
                if self.documents and self.documents[0].embedding is not None:
                    target_dim = int(self.documents[0].embedding.shape[0])
                if metadata is None:
                    metadata = [{} for _ in texts]
                for i, (text, meta) in enumerate(zip(texts, metadata)):
                    embedding = np.random.normal(0, 1, target_dim)
                    embedding = embedding / (np.linalg.norm(embedding) or 1.0)
                    doc = _Doc(
                        id=len(self.documents),
                        content=text,
                        embedding=embedding,
                        metadata=meta,
                    )
                    self.documents.append(doc)
                print(f"✅ {len(texts)} chunks added to MCPM (mock embeddings)")
                return True
            except Exception as e:
                print(f"❌ Mock-Embedding Fehler: {e}")
                return False
            
        if metadata is None:
            metadata = [{} for _ in texts]
            
        # Embeddings generieren (mit Prompting) in Batches, GPU-priorisiert
        prompted = [self._apply_prompt(self.doc_prompt_name, t) for t in texts]
        embeddings: np.ndarray
        if _TORCH_OK and torch.cuda.is_available():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            batches: List[torch.Tensor] = []
            bs = max(16, int(self.embed_batch_size))
            try:
                with torch.inference_mode():
                    for i in range(0, len(prompted), bs):
                        chunk = prompted[i:i+bs]
                        t = self.embedding_model.encode(
                            chunk,
                            batch_size=bs,
                            convert_to_tensor=True,
                            show_progress_bar=False,
                            device='cuda',
                            num_workers=0,
                            normalize_embeddings=True,
                        )
                        if not torch.is_floating_point(t):
                            t = t.float()
                        batches.append(t)
                emb_t = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
                # Cache GPU embeddings (fp16) and normalized version
                self._doc_emb_torch = emb_t.half().contiguous()
                self._doc_emb_torch_norm = torch.nn.functional.normalize(self._doc_emb_torch, p=2, dim=1)
                # Make compact CPU copy for per-document storage (fp16)
                embeddings = self._doc_emb_torch.float().cpu().numpy().astype(np.float16)
            except Exception:
                # CPU fallback batch path
                embeddings = self.embedding_model.encode(
                    prompted,
                    batch_size=bs,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    num_workers=0,
                    normalize_embeddings=True,
                ).astype(np.float16)
        else:
            # Pure CPU
            embeddings = self.embedding_model.encode(
                prompted,
                batch_size=max(16, int(self.embed_batch_size)),
                convert_to_numpy=True,
                show_progress_bar=False,
                num_workers=0,
                normalize_embeddings=True,
            ).astype(np.float16)
        
        # Dokumente erstellen
        for i, (text, embed, meta) in enumerate(zip(texts, embeddings, metadata)):
            doc = Document(
                id=len(self.documents),
                content=text,
                embedding=embed,  # store as float16 to reduce RAM
                metadata=meta
            )
            self.documents.append(doc)
            
        # Track embedding dim for FAISS
        try:
            if self.documents and hasattr(self.documents[0], 'embedding'):
                self._embed_dim = int(len(self.documents[0].embedding))
        except Exception:
            self._embed_dim = None

        # Build FAISS index (coarse) if available and requested
        if self.build_faiss_after_add and _FAISS_OK and self._embed_dim:
            try:
                # Use cached normalized GPU matrix to avoid Python loops
                if self._doc_emb_torch_norm is not None:
                    embs = self._doc_emb_torch_norm.float().cpu().numpy()
                else:
                    embs = np.array([d.embedding for d in self.documents], dtype=np.float32)
                    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                    embs = embs / norms
                try:
                    from embeddinggemma.mcmp.indexing import build_faiss_index as _build_faiss  # type: ignore
                    self._faiss_index = _build_faiss(embs, int(self._embed_dim))
                except Exception:
                    index = faiss.index_factory(self._embed_dim, "IVF4096,Flat", faiss.METRIC_INNER_PRODUCT)
                    if faiss.get_num_gpus() > 0:
                        self._faiss_gpu_res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_all_gpus(index)
                    index.train(embs)
                    index.add(embs)
                    try:
                        index.nprobe = 12
                    except Exception:
                        pass
                    self._faiss_index = index
            except Exception:
                self._faiss_index = None

        # Invalidate cached torch matrix
        self._doc_emb_torch = None
        self._doc_emb_torch_norm = None
        self._logger.info(f"{len(texts)} chunks added to MCPM")
        return True
    
    def spawn_agents(self, query_embedding: np.ndarray):
        """Spawn agents around the query position with small random offsets."""
        try:
            from embeddinggemma.mcmp.simulation import spawn_agents as _spawn  # type: ignore
            # Provide Agent class on retriever for the helper
            self.Agent = Agent  # type: ignore[attr-defined]
            _spawn(self, query_embedding)
        except Exception:
            self.agents = []
            for i in range(self.num_agents):
                noise = np.random.normal(0, 0.1, query_embedding.shape)
                start_pos = query_embedding + noise
                start_pos = start_pos / np.linalg.norm(start_pos)
                velocity = np.random.normal(0, 0.05, query_embedding.shape)
                agent = Agent(
                    id=i,
                    position=start_pos,
                    velocity=velocity,
                    exploration_factor=random.uniform(0.05, max(0.05, float(self.exploration_bonus))),
                )
                self.agents.append(agent)
    
    def find_nearest_documents(self, position: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Find k nearest documents to a given position in embedding space."""
        if not self.documents:
            return []
        # FAISS fast path
        if _FAISS_OK and self._faiss_index is not None and self._embed_dim:
            try:
                q = np.asarray(position, dtype=np.float32)
                q = q / (np.linalg.norm(q) or 1.0)
                D, I = self._faiss_index.search(q.reshape(1, -1), int(k))
                pairs = []
                for score, idx in zip(D[0], I[0]):
                    if idx >= 0 and idx < len(self.documents):
                        pairs.append((self.documents[int(idx)], float(score)))
                if pairs:
                    return pairs
            except Exception:
                pass
        # Torch fast path on GPU
        if _TORCH_OK and torch.cuda.is_available():
            # Build cached matrix if needed
            if self._doc_emb_torch is None:
                with torch.no_grad():
                    mat = torch.tensor(np.array([doc.embedding for doc in self.documents], dtype=np.float32), device='cuda')
                    mat = mat.to(torch.float16)
                    # normalize rows
                    mat_n = torch.nn.functional.normalize(mat, p=2, dim=1)
                    self._doc_emb_torch = mat
                    self._doc_emb_torch_norm = mat_n
            try:
                from embeddinggemma.mcmp.indexing import cosine_similarities_torch as _cos_torch  # type: ignore
                similarities = _cos_torch(position, self._doc_emb_torch_norm, device='cuda')
            except Exception:
                with torch.no_grad():
                    pos = torch.tensor(position.astype(np.float32), device='cuda').to(torch.float16)
                    pos_n = torch.nn.functional.normalize(pos, p=2, dim=0)
                    sims_t = self._doc_emb_torch_norm @ pos_n  # (N,)
                    similarities = sims_t.float().cpu().numpy()
        else:
            # CPU fallback
            doc_embeddings = np.array([doc.embedding for doc in self.documents])
            try:
                from embeddinggemma.mcmp.indexing import cosine_similarities_cpu as _cos_cpu  # type: ignore
                similarities = _cos_cpu(position, doc_embeddings)
            except Exception:
                similarities = cosine_similarity([position], doc_embeddings)[0]
        
        # Top k auswählen
        top_indices = np.argsort(similarities)[::-1][:k]
        return [(self.documents[idx], similarities[idx]) for idx in top_indices]
    
    def update_agent_position(self, agent: Agent, iteration: int):
        """Update agent position based on attraction, pheromones and exploration."""
        # Find nearby documents
        nearby_docs = self.find_nearest_documents(agent.position, k=3)
        
        if not nearby_docs:
            return
            
        # Attraction Force (zu relevanten Dokumenten)
        attraction_force = np.zeros_like(agent.position)
        for doc, similarity in nearby_docs:
            direction = doc.embedding - agent.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                # Stärke basierend auf Similarity und Dokument-Relevance
                force_strength = similarity * (1 + doc.relevance_score)
                attraction_force += (direction / distance) * force_strength
        
        # Pheromone Trail Force
        pheromone_force = self.calculate_pheromone_force(agent)
        
        # Exploration Force (Zufälligkeit)
        exploration_force = np.random.normal(0, agent.exploration_factor, agent.position.shape)
        
        # Combine forces
        total_force = (0.6 * attraction_force + 
                      0.3 * pheromone_force + 
                      0.1 * exploration_force)
        
        # Update velocity and position
        agent.velocity = 0.8 * agent.velocity + 0.2 * total_force
        agent.position += agent.velocity
        
        # Normalize to remain on the embedding hypersphere
        agent.position = agent.position / np.linalg.norm(agent.position)
        agent.age += 1
    
    def calculate_pheromone_force(self, agent: Agent) -> np.ndarray:
        """Compute pheromone trail influence vector for an agent."""
        try:
            from embeddinggemma.mcmp.simulation import calculate_pheromone_force as _calc  # type: ignore
            return _calc(self, agent)
        except Exception:
            if not self.pheromone_trails:
                return np.zeros_like(agent.position)
            force = np.zeros_like(agent.position)
            current_docs = self.find_nearest_documents(agent.position, k=1)
            if not current_docs:
                return force
            current_doc = current_docs[0][0]
            max_strength = 0
            best_direction = None
            for (doc_a, doc_b), strength in self.pheromone_trails.items():
                if doc_a == current_doc.id:
                    target_doc = next((d for d in self.documents if d.id == doc_b), None)
                    if target_doc:
                        direction = target_doc.embedding - agent.position
                        if strength > max_strength:
                            max_strength = strength
                            best_direction = direction
            if best_direction is not None and np.linalg.norm(best_direction) > 0:
                force = best_direction / np.linalg.norm(best_direction) * max_strength
            return force
    
    def deposit_pheromones(self, agent: Agent):
        """Deposit pheromone trail from the current document to recent ones."""
        try:
            from embeddinggemma.mcmp.simulation import deposit_pheromones as _deposit  # type: ignore
            _deposit(self, agent)
        except Exception:
            current_docs = self.find_nearest_documents(agent.position, k=1)
            if not current_docs:
                return
            current_doc = current_docs[0][0]
            current_doc.visit_count += 1
            current_doc.last_visited = time.time()
            agent.visited_docs.add(current_doc.id)
            for prev_doc_id in list(agent.visited_docs)[-3:]:
                if prev_doc_id != current_doc.id:
                    trail_key = tuple(sorted([current_doc.id, prev_doc_id]))
                    current_strength = self.pheromone_trails.get(trail_key, 0)
                    self.pheromone_trails[trail_key] = current_strength + agent.energy * agent.trail_strength * 0.1
    
    def decay_pheromones(self):
        """Apply exponential decay to all pheromone trails and prune weak ones."""
        try:
            from embeddinggemma.mcmp.simulation import decay_pheromones as _decay  # type: ignore
            _decay(self)
        except Exception:
            keys_to_remove = []
            for key in self.pheromone_trails:
                self.pheromone_trails[key] *= self.pheromone_decay
                if self.pheromone_trails[key] < 0.01:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.pheromone_trails[key]
    
    def update_document_relevance(self, query_embedding: np.ndarray):
        """Update document relevance from query similarity and agent activity (delegated)."""
        try:
            from embeddinggemma.mcmp.simulation import update_document_relevance as _upd  # type: ignore
            _upd(self, query_embedding)
        except Exception:
            pass
    
    def extract_relevance_network(self, min_strength=0.1) -> List[Tuple[Document, float]]:
        """Build a relevance network and return documents sorted by score."""
        # Sort documents by relevance
        relevant_docs = sorted(self.documents, key=lambda d: d.relevance_score, reverse=True)
        
        # Build network graph
        self.relevance_network.clear()
        for doc in relevant_docs:
            if doc.relevance_score > 0.1:  # keep only relevant nodes
                self.relevance_network.add_node(doc.id, 
                                               content=doc.content[:100] + "...",
                                               relevance=doc.relevance_score,
                                               visits=doc.visit_count)
        
        # Pheromone-trail-based edges
        for (doc_a, doc_b), strength in self.pheromone_trails.items():
            if strength > min_strength and doc_a in self.relevance_network.nodes and doc_b in self.relevance_network.nodes:
                self.relevance_network.add_edge(doc_a, doc_b, strength=strength)
        
        return [(doc, doc.relevance_score) for doc in relevant_docs if doc.relevance_score > 0.1]
    
    def search(self, query: str, top_k: int = 10, verbose: bool = True) -> Dict:
        """Main MCPM search function: run simulation and return ranked results."""
        if not self.documents:
            return {"error": "No documents loaded"}
        
        if not self.load_embedding_model():
            # Fallback: random query embedding with same dimensionality
            if not self.documents or self.documents[0].embedding is None:
                return {"error": "Embedding model not available"}
            dim = self.documents[0].embedding.shape
            query_embedding = np.random.normal(0, 1, dim)
            norm = np.linalg.norm(query_embedding) or 1.0
            query_embedding = query_embedding / norm
        else:
            # Query embedden
            q_text = self._apply_prompt(self.query_prompt_name, query)
            query_embedding = self.embedding_model.encode([q_text])[0]
        
        # Für Realtime-Funktionen verfügbar halten
        self._current_query_embedding = query_embedding
        
        # Spawn agents
        self.spawn_agents(query_embedding)
        
        if verbose:
            self._logger.info(f"MCPM search started: agents={self.num_agents} iterations={self.max_iterations}")
        if self.debug_log:
            try:
                print(f"[dbg] cfg | embed_model={self.embedding_model_name} | agents={self.num_agents} | iters={self.max_iterations} | pher_decay={self.pheromone_decay} | expl_bonus={self.exploration_bonus}")
            except Exception:
                pass

        # If FAISS available, temporarily restrict documents to top subset for speed
        original_documents = None
        if _FAISS_OK and self._faiss_index is not None and self._embed_dim:
            try:
                q = np.asarray(query_embedding, dtype=np.float32)
                q = q / (np.linalg.norm(q) or 1.0)
                # smaller prefilter to reduce CPU post-work
                topN = min(len(self.documents), max(int(top_k * 10), 500))
                D, I = self._faiss_index.search(q.reshape(1, -1), topN)
                idxs = [int(i) for i in I[0] if i >= 0]
                if idxs:
                    original_documents = self.documents
                    self.documents = [original_documents[i] for i in idxs]
                    # Invalidate cached torch matrices since we changed the set
                    self._doc_emb_torch = None
                    self._doc_emb_torch_norm = None
                    if verbose:
                        self._logger.info(f"FAISS coarse filter active: docs={len(self.documents)}")
            except Exception:
                pass
        
        # MCPM Iterationen
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Agent updates
            for agent in self.agents:
                self.update_agent_position(agent, iteration)
                self.deposit_pheromones(agent)
            
            # Update document relevance
            self.update_document_relevance(query_embedding)
            
            # Pheromone decay
            self.decay_pheromones()
            
            # Konvergenz messen
            avg_relevance = np.mean([doc.relevance_score for doc in self.documents])
            convergence_history.append(avg_relevance)
            
            # Progress
            if verbose and self.log_every and (iteration % int(self.log_every) == 0 or iteration == self.max_iterations - 1):
                self._logger.info(f"Iteration {iteration}: avg_relevance={avg_relevance:.3f}")
            # Debug metrics (aggregated)
            if self.debug_log and (iteration % max(1, int(self.log_every)) == 0 or iteration == self.max_iterations - 1):
                try:
                    avg_speed = float(np.mean([np.linalg.norm(a.velocity) for a in self.agents])) if self.agents else 0.0
                    trails = int(len(self.pheromone_trails))
                    max_trail = float(max(self.pheromone_trails.values())) if self.pheromone_trails else 0.0
                    visited = set()
                    for a in self.agents:
                        visited.update(a.visited_docs)
                    uniq_visited = int(len(visited))
                    self._logger.debug(f"iter={iteration} avg_rel={avg_relevance:.3f} avg_speed={avg_speed:.3f} trails={trails} max_trail={max_trail:.3f} uniq_visited={uniq_visited}")
                except Exception:
                    pass
        
        # Extract results
        relevant_docs = self.extract_relevance_network()
        
        # Top-K auswählen
        top_results = relevant_docs[:top_k]

        # Fallback: Falls keine relevanten Dokumente über Schwellwert
        if not top_results:
            try:
                doc_embeddings = np.array([doc.embedding for doc in self.documents])
                sims = cosine_similarity([query_embedding], doc_embeddings)[0]
                top_indices = np.argsort(sims)[::-1][:top_k]
                top_results = [(self.documents[idx], float(sims[idx])) for idx in top_indices]
            except Exception:
                top_results = []
        
        # Update stats
        self.search_stats['total_searches'] += 1
        self.search_stats['network_density'] = nx.density(self.relevance_network)
        self.search_stats['unique_paths_found'] = len(self.pheromone_trails)
        
        if verbose:
            self._logger.info(f"MCPM search finished: results={len(top_results)} density={self.search_stats['network_density']:.3f} trails={self.search_stats['unique_paths_found']}")
        if self.debug_log:
            try:
                tops = ", ".join([f"{score:.3f}" for _, score in top_results[:5]]) if top_results else ""
                self._logger.debug(f"top_scores={tops}")
            except Exception:
                pass
        # Restore original document set if filtered
        if original_documents is not None:
            self.documents = original_documents
        
        return {
            "query": query,
            "results": [
                {
                    "content": doc.content,
                    "relevance_score": score,
                    "visit_count": doc.visit_count,
                    "metadata": doc.metadata
                }
                for doc, score in top_results
            ],
            "network_stats": self.search_stats,
            "convergence_history": convergence_history,
            "pheromone_trails": len(self.pheromone_trails),
            "total_documents": len(self.documents)
        }

    # ===== Realtime Simulation & Visualization Utilities =====
    def initialize_simulation(self, query: str) -> bool:
        """Initialize agents and state for a real-time simulation."""
        if not self.documents:
            return False
        if not self.load_embedding_model():
            # Fallback query embedding
            dim = self.documents[0].embedding.shape
            query_embedding = np.random.normal(0, 1, dim)
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
        else:
            q_text = self._apply_prompt(self.query_prompt_name, query)
            query_embedding = self.embedding_model.encode([q_text])[0]
        
        self._current_query_embedding = query_embedding
        self.spawn_agents(query_embedding)
        
        # Reset trails and doc states
        self.pheromone_trails = {}
        for doc in self.documents:
            doc.visit_count = 0
            doc.last_visited = 0.0
            doc.relevance_score = 0.0
        
        # Reset PCA cache
        self._pca_mean = None
        self._pca_components = None
        return True
    
    def step(self, n_steps: int = 1) -> Dict:
        """Run ``n_steps`` simulation steps and return simple metrics."""
        if not self.documents or self._current_query_embedding is None or not self.agents:
            return {"error": "Simulation not initialized"}
        # Fast GPU stepping path (no pheromone calculation) for performance
        if _TORCH_OK and torch.cuda.is_available() and self.fast_gpu_step:
            try:
                return self._step_fast_gpu(n_steps)
            except Exception:
                # fallback to CPU path
                pass
        
        convergence_history = []
        for _ in range(max(1, n_steps)):
            for agent in self.agents:
                self.update_agent_position(agent, 0)
                self.deposit_pheromones(agent)
            self.update_document_relevance(self._current_query_embedding)
            self.decay_pheromones()
            avg_relevance = np.mean([doc.relevance_score for doc in self.documents])
            convergence_history.append(float(avg_relevance))
        
        return {
            "avg_relevance": convergence_history[-1] if convergence_history else 0.0,
            "steps": int(n_steps),
            "pheromone_trails": len(self.pheromone_trails)
        }

    def _step_fast_gpu(self, n_steps: int = 1) -> Dict:
        if self._doc_emb_torch_norm is None:
            # Build normalized doc matrix on GPU
            if self._doc_emb_torch is None:
                mat = torch.tensor(np.array([doc.embedding for doc in self.documents], dtype=np.float32), device='cuda')
                self._doc_emb_torch = mat.half().contiguous()
            self._doc_emb_torch_norm = torch.nn.functional.normalize(self._doc_emb_torch, p=2, dim=1)
        D = self._doc_emb_torch_norm.shape[1]
        # Build agent tensors
        pos = torch.stack([torch.tensor(a.position, dtype=torch.float32, device='cuda') for a in self.agents]).half()
        vel = torch.stack([torch.tensor(a.velocity, dtype=torch.float32, device='cuda') for a in self.agents]).half()
        expl = torch.tensor([float(a.exploration_factor) for a in self.agents], dtype=torch.float32, device='cuda').view(-1, 1)
        q = torch.tensor(self._current_query_embedding.astype(np.float32), device='cuda').half()
        qn = torch.nn.functional.normalize(q, p=2, dim=0)
        conv_hist = []
        for _ in range(max(1, int(n_steps))):
            # Normalize positions
            pos_n = torch.nn.functional.normalize(pos, p=2, dim=1)
            # Similarities to docs
            sims = pos_n @ self._doc_emb_torch_norm.T  # (M, N)
            vals, idxs = torch.topk(sims.float(), k=3, dim=1)  # (M,3)
            # Gather target embeddings
            targets = self._doc_emb_torch_norm.float()[idxs]  # (M,3,D)
            # Directions
            dirs = targets - pos_n.unsqueeze(1)  # (M,3,D)
            norms = torch.norm(dirs, dim=2, keepdim=True) + 1e-6
            dirs_n = dirs / norms
            # Attraction force (weight by similarity)
            att = (vals.unsqueeze(2) * dirs_n).sum(dim=1)  # (M,D)
            # Exploration force
            exp_force = torch.randn_like(pos_n.float()) * expl
            # No pheromone force in fast path
            total = (0.8 * att + 0.2 * exp_force).half()
            vel = (0.8 * vel + 0.2 * total).half()
            pos = (pos + vel).half()
            pos = torch.nn.functional.normalize(pos.float(), p=2, dim=1).half()
            # Update relevance once per step (GPU)
            sims_q = (self._doc_emb_torch_norm @ qn).float().cpu().numpy()
            for i, doc in enumerate(self.documents):
                query_sim = float(sims_q[i])
                visit_bonus = min(doc.visit_count * 0.1, 0.5)
                time_bonus = 0.0  # skip time bonus for speed
                doc.relevance_score = query_sim + visit_bonus + time_bonus
            conv_hist.append(float(np.mean([d.relevance_score for d in self.documents])))
        # Write back agent states (first few dims if needed)
        pos_cpu = pos.float().cpu().numpy()
        vel_cpu = vel.float().cpu().numpy()
        for a, p, v in zip(self.agents, pos_cpu, vel_cpu):
            a.position = p
            a.velocity = v
        return {
            "avg_relevance": conv_hist[-1] if conv_hist else 0.0,
            "steps": int(n_steps),
            "pheromone_trails": len(self.pheromone_trails)
        }
    
    def _ensure_pca(self):
        if self._pca_components is not None and self._pca_mean is not None:
            return
        if not self.documents:
            return
        doc_embeddings = np.array([doc.embedding for doc in self.documents])
        mean = doc_embeddings.mean(axis=0)
        X = doc_embeddings - mean
        try:
            # Stable SVD-based PCA without external dependencies
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            components = Vt[:2]  # (2, D)
        except Exception:
            # Fallback to random projection
            rng = np.random.default_rng(42)
            D = doc_embeddings.shape[1]
            components = rng.normal(0, 1, size=(2, D))
            for i in range(2):
                components[i] = components[i] / (np.linalg.norm(components[i]) or 1.0)
        self._pca_mean = mean
        self._pca_components = components
        # Speichere Singulärwerte für Whitened PCA
        try:
            self._pca_singular_values = S[:2]
        except Exception:
            self._pca_singular_values = None
    
    def project_documents_2d(self,
                             method: str = "pca",
                             whiten: bool = False,
                             spread: float = 1.0,
                             jitter: float = 0.0,
                             tsne_perplexity: float = 30.0,
                             tsne_early_exaggeration: float = 12.0,
                             tsne_n_iter: int = 1000,
                             random_state: int = 42) -> Optional[np.ndarray]:
        if not self.documents:
            return None
        doc_embeddings = np.array([doc.embedding for doc in self.documents])

        method = (method or "pca").lower()
        coords: Optional[np.ndarray] = None

        if method == "tsne":
            try:
                from sklearn.manifold import TSNE  # type: ignore
                coords = TSNE(
                    n_components=2,
                    perplexity=float(tsne_perplexity),
                    early_exaggeration=float(tsne_early_exaggeration),
                    n_iter=int(tsne_n_iter),
                    learning_rate="auto",
                    init="pca",
                    random_state=int(random_state),
                ).fit_transform(doc_embeddings)
            except Exception:
                coords = None

        if coords is None and method == "umap":
            # Prefer cuML UMAP if available (GPU), else fall back to umap-learn
            try:
                from cuml.manifold import UMAP  # type: ignore
                import cupy as cp  # type: ignore
                X_gpu = cp.asarray(doc_embeddings)
                reducer = UMAP(n_components=2, random_state=int(random_state))
                Y_gpu = reducer.fit_transform(X_gpu)
                coords = cp.asnumpy(Y_gpu)
            except Exception:
                try:
                    import umap  # type: ignore
                    reducer = umap.UMAP(n_components=2, random_state=int(random_state))
                    coords = reducer.fit_transform(doc_embeddings)
                except Exception:
                    coords = None

        if coords is None:
            # Prefer cuML PCA on GPU
            try:
                from cuml.decomposition import PCA  # type: ignore
                import cupy as cp  # type: ignore
                X_gpu = cp.asarray(doc_embeddings)
                pca = PCA(n_components=2, whiten=bool(whiten), random_state=int(random_state))
                Y_gpu = pca.fit_transform(X_gpu)
                coords = cp.asnumpy(Y_gpu)
            except Exception:
                # CPU PCA fallback (delegated if available)
                try:
                    from embeddinggemma.mcmp.pca import pca_2d as _pca_2d  # type: ignore
                    coords = _pca_2d(doc_embeddings, whiten=bool(whiten))
                except Exception:
                    self._ensure_pca()
                    X = doc_embeddings - self._pca_mean
                    coords = X @ self._pca_components.T  # (N, 2)
                    if whiten and getattr(self, "_pca_singular_values", None) is not None:
                        s = self._pca_singular_values
                        safe = np.array([s[0] if s[0] != 0 else 1.0, s[1] if s[1] != 0 else 1.0])
                        coords = coords / safe

        if spread and spread != 1.0:
            coords = coords * float(spread)
        if jitter and jitter > 0.0:
            coords = coords + np.random.normal(0, float(jitter), coords.shape)
        return coords
    
    def project_agents_2d(self,
                          method: str = "pca",
                          whiten: bool = False,
                          spread: float = 1.0,
                          jitter: float = 0.0) -> Optional[np.ndarray]:
        if not self.agents:
            return None
        # Agents are projected in the same linear space as PCA
        self._ensure_pca()
        A = np.array([agent.position for agent in self.agents]) - self._pca_mean
        coords = A @ self._pca_components.T
        if whiten and getattr(self, "_pca_singular_values", None) is not None:
            s = self._pca_singular_values
            safe = np.array([s[0] if s[0] != 0 else 1.0, s[1] if s[1] != 0 else 1.0])
            coords = coords / safe
        if spread and spread != 1.0:
            coords = coords * float(spread)
        if jitter and jitter > 0.0:
            coords = coords + np.random.normal(0, float(jitter), coords.shape)
        return coords
    
    def get_visualization_snapshot(self,
                                   min_trail_strength: float = 0.05,
                                   max_edges: int = 300,
                                   method: str = "pca",
                                   whiten: bool = False,
                                   spread: float = 1.0,
                                   jitter: float = 0.0,
                                   tsne_perplexity: float = 30.0,
                                   tsne_early_exaggeration: float = 12.0,
                                   tsne_n_iter: int = 1000) -> Dict:
        """Bereite Daten für 2D-Visualisierung vor."""
        docs_2d = self.project_documents_2d(method=method,
                                            whiten=whiten,
                                            spread=spread,
                                            jitter=jitter,
                                            tsne_perplexity=tsne_perplexity,
                                            tsne_early_exaggeration=tsne_early_exaggeration,
                                            tsne_n_iter=tsne_n_iter)
        agents_2d = self.project_agents_2d(method="pca", whiten=whiten, spread=spread, jitter=jitter)
        
        # Trails in 2D
        try:
            from embeddinggemma.mcmp.visualize import build_snapshot as _build_snapshot  # type: ignore
            return _build_snapshot(
                docs_2d if docs_2d is not None else np.zeros((0, 2), dtype=float),
                [float(doc.relevance_score) for doc in self.documents],
                self.pheromone_trails,
                max_edges=int(max_edges),
            )
        except Exception:
            edges = []
            if self.pheromone_trails and docs_2d is not None:
                for (a, b), strength in sorted(self.pheromone_trails.items(), key=lambda x: x[1], reverse=True):
                    if strength < min_trail_strength:
                        continue
                    if a < len(self.documents) and b < len(self.documents):
                        x0, y0 = docs_2d[a]
                        x1, y1 = docs_2d[b]
                        edges.append({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "s": float(strength)})
                    if len(edges) >= max_edges:
                        break
            return {
                "documents": {
                    "xy": docs_2d.tolist() if docs_2d is not None else [],
                    "relevance": [float(doc.relevance_score) for doc in self.documents]
                },
                "agents": {
                    "xy": agents_2d.tolist() if agents_2d is not None else []
                },
                "edges": edges
            }

    # ===== Document Management Utilities =====
    def clear_documents(self) -> None:
        """Entferne alle Dokumente und setze internen Zustand zurück."""
        self.documents = []
        self.agents = []
        self.pheromone_trails = {}
        self.relevance_network.clear()
        self.iteration_history = []
        # Stats teilweise zurücksetzen
        self.search_stats['network_density'] = 0
        self.search_stats['unique_paths_found'] = 0

    def delete_documents(self, indices: List[int]) -> int:
        """Lösche spezifische Dokumente anhand ihrer Listenindizes.

        Gibt die Anzahl gelöschter Dokumente zurück. Ids werden neu vergeben,
        Trails/Netzwerk werden geleert.
        """
        if not indices:
            return 0
        # Sicherstellen, dass Indizes gültig sind
        valid = [i for i in set(indices) if 0 <= i < len(self.documents)]
        if not valid:
            return 0
        # In absteigender Reihenfolge entfernen
        for i in sorted(valid, reverse=True):
            del self.documents[i]
        # Ids neu zuweisen
        for new_id, doc in enumerate(self.documents):
            doc.id = new_id
        # Trails/Netzwerk leeren, da Referenzen ungültig
        self.pheromone_trails = {}
        self.relevance_network.clear()
        return len(valid)
    
    def visualize_search_process(self, save_path: Optional[str] = None):
        """Visualisiere MCPM Suchprozess.

        Hinweis: matplotlib ist optional. Falls nicht installiert, wird eine
        verständliche Meldung ausgegeben und die Funktion beendet sich.
        """
        try:
            import matplotlib.pyplot as plt  # lazy import
        except Exception as e:
            self._logger.warning(f"Visualization not available (matplotlib missing): {e}")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Dokument-Relevance Verteilung
        relevances = [doc.relevance_score for doc in self.documents]
        ax1.hist(relevances, bins=20, alpha=0.7, color='blue')
        ax1.set_title('Dokument-Relevance Verteilung')
        ax1.set_xlabel('Relevance Score')
        ax1.set_ylabel('Anzahl Dokumente')
        
        # 2. Visit Pattern
        visits = [doc.visit_count for doc in self.documents]
        ax2.scatter(range(len(visits)), visits, alpha=0.6, c=relevances, cmap='viridis')
        ax2.set_title('Dokument-Besuche vs. Relevance')
        ax2.set_xlabel('Dokument ID')
        ax2.set_ylabel('Besuche')
        
        # 3. Pheromonspur-Stärke
        if self.pheromone_trails:
            strengths = list(self.pheromone_trails.values())
            ax3.hist(strengths, bins=15, alpha=0.7, color='green')
            ax3.set_title('Pheromonspur-Stärken')
            ax3.set_xlabel('Spur-Stärke')
            ax3.set_ylabel('Anzahl Pfade')
        
        # 4. Netzwerk-Visualisierung (vereinfacht)
        if self.relevance_network.nodes():
            pos = nx.spring_layout(self.relevance_network)
            node_sizes = [self.relevance_network.nodes[node].get('relevance', 0.1) * 1000 
                         for node in self.relevance_network.nodes()]
            
            nx.draw(self.relevance_network, pos, ax=ax4,
                   node_size=node_sizes,
                   node_color='lightblue',
                   with_labels=True,
                   font_size=8)
            ax4.set_title('Emergentes Relevanznetzwerk')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._logger.info(f"Visualization saved: {save_path}")
        else:
            plt.show()




