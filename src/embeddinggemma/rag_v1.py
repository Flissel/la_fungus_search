#!/usr/bin/env python3
"""
RAG v1 für Code-Analyse mit LlamaIndex, Qdrant und EmbeddingGemma
Skalierbare Lösung für große Codebasen mit AST-basiertem Chunking
"""

import os
import ast
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import json
import sys

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import transformers
import torch
import re
import requests

# Optional: Fungus (MCPMRetriever)
# Ensure project root on sys.path so that `mcmp_rag.py` at repo root is importable
try:
    _THIS_DIR = os.path.dirname(__file__)
    _ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
except Exception:
    pass
try:
    from mcmp_rag import MCPMRetriever  # type: ignore
except Exception as _merr:
    print(f"ℹ️ MCPMRetriever Import fehlgeschlagen: {_merr}")
    MCPMRetriever = None  # type: ignore

class RagV1:
    """
    RAG v1 (Enterprise-grade) für Code-Analyse mit semantischem Verständnis
    """
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6337",
                 qdrant_api_key: Optional[str] = None,
                 collection_name: str = "codebase",
                 embedding_model: str = "google/embeddinggemma-300m",
                 llm_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                 llm_device: str = "auto",
                 use_ollama: bool = False,
                 ollama_model: str = "qwen2.5-coder:7b",
                 ollama_host: str = "http://127.0.0.1:11434",
                 debug: bool = False):
        """
        Initialisiert die RAG v1 mit Qdrant und LlamaIndex
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm_device = llm_device  # "auto" | "cuda" | "cpu"
        self.use_ollama = bool(use_ollama)
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host.rstrip('/')
        self.debug = bool(debug)

        def _d(msg: str) -> None:
            if self.debug:
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[DEBUG {now}] {msg}")

        self._d = _d  # store logger
        
        # Qdrant Client initialisieren
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
            prefer_grpc=True
        )
        
        # Embedding Modell (EmbeddingGemma)
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # LLM für Generierung
        self.llm = self._build_llm(device_preference=self.llm_device)
        
        # LlamaIndex Index
        self.index = None
        self.vector_store = None
        
        # AST Parser für Code-Chunking (größere Chunks für vollständige Codeblocks)
        self.ast_parser = CodeSplitter(
            language="python",
            chunk_lines=120,
            chunk_lines_overlap=40,
            max_chars=5000
        )
        
        self._d(f"Init with qdrant_url={qdrant_url}, collection={collection_name}, embed={embedding_model}")
        self._d(f"LLM use_ollama={self.use_ollama} model={self.ollama_model} device_pref={self.llm_device}")
        print(f"✅ RagV1 initialisiert mit Qdrant: {qdrant_url}")

    def _build_llm(self, device_preference: str = "auto") -> HuggingFaceLLM:
        """
        Robust LLM builder that can fall back to CPU to avoid CUDA assert errors.
        """
        llm_model = self.llm_model
        def _tok_ids() -> Dict[str, int]:
            try:
                tok = transformers.AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
                eid = tok.eos_token_id or tok.pad_token_id or 0
                pid = tok.pad_token_id or eid
                return {"eos": int(eid), "pad": int(pid)}
            except Exception:
                return {"eos": 0, "pad": 0}
        ids = _tok_ids()
        want_cuda = (device_preference == "auto" and torch.cuda.is_available()) or (device_preference == "cuda")
        if want_cuda:
            try:
                return HuggingFaceLLM(
                    model_name=llm_model,
                    tokenizer_name=llm_model,
                    device_map="auto",
                    model_kwargs={"dtype": torch.float16, "attn_implementation": "eager"},
                    generate_kwargs={"pad_token_id": ids["pad"], "eos_token_id": ids["eos"], "temperature": 0.2}
                )
            except Exception as e:
                print(f"⚠️ CUDA LLM init failed ({e}). Falling back to CPU.")
        # CPU fallback
        return HuggingFaceLLM(
            model_name=llm_model,
            tokenizer_name=llm_model,
            device_map="cpu",
            model_kwargs={},
            generate_kwargs={"pad_token_id": ids["pad"], "eos_token_id": ids["eos"], "temperature": 0.2}
        )
    
    def _get_embedding_dim(self) -> int:
        """Ermittle robust die Embedding-Vektorgröße der aktiven Embeddings."""
        # 1) Versuche direkten Zugriff auf SentenceTransformer
        try:
            inner = getattr(self.embed_model, "_model", None)
            if inner is not None and hasattr(inner, "get_sentence_embedding_dimension"):
                d = int(inner.get_sentence_embedding_dimension())
                if d > 0:
                    return d
        except Exception:
            pass
        # 2) Probe über Query/Text Embedding
        for fn in ("embed_query", "embed"):
            try:
                vec = getattr(self.embed_model, fn)("dimension probe")
                d = int(len(vec))
                if d > 0:
                    return d
            except Exception:
                pass
        try:
            vec = self.embed_model.get_text_embedding("dimension probe")
            d = int(len(vec))
            if d > 0:
                return d
        except Exception:
            pass
        # 3) Fallback
        return 768

    # ---- Python helpers for method extraction and chunking ----
    def _extract_methods_from_python(self, file_path: str) -> List[str]:
        """Extract top-level functions and class methods from a Python file using AST."""
        methods: List[str] = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    methods.append(f"def {node.name}(...)")
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    for b in node.body:
                        if isinstance(b, ast.FunctionDef):
                            methods.append(f"class {class_name} -> def {b.name}(...)")
            return methods
        except Exception as e:
            print(f"⚠️ Methoden-Extraktion fehlgeschlagen für {file_path}: {e}")
            return []

    def extract_method_span(self, file_path: str, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Return the exact start/end line numbers and source code for a given top-level or class method.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            lines = source.splitlines()
            tree = ast.parse(source)
            target = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == method_name:
                    target = node
                    break
                if isinstance(node, ast.ClassDef):
                    for b in node.body:
                        if isinstance(b, ast.FunctionDef) and b.name == method_name:
                            target = b
                            break
                if target:
                    break
            if not target:
                return None
            start = getattr(target, 'lineno', 1)
            end = getattr(target, 'end_lineno', start)
            code = '\n'.join(lines[start-1:end])
            return {"start_line": start, "end_line": end, "code": code}
        except Exception as e:
            print(f"⚠️ extract_method_span fehlgeschlagen: {e}")
            return None

    def _chunk_python_file_for_fungus(self, file_path: str, windows: Optional[List[int]] = None) -> List[str]:
        """Create multi-window line chunks from a Python file for Fungus retrieval."""
        if windows is None:
            windows = [50, 100, 200, 300, 400]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️ Konnte Datei nicht lesen für Chunking: {e}")
            return []
        chunks: List[str] = []
        total = len(lines)
        rel = os.path.relpath(file_path)
        for w in windows:
            for i in range(0, total, w):
                start = i + 1
                end = min(i + w, total)
                body = ''.join(lines[i:end])
                if body.strip():
                    header = f"# file: {rel} | lines: {start}-{end} | window: {w}\n"
                    chunks.append(header + body)
        return chunks

    def create_collection(self):
        """Erstellt Qdrant Collection falls nicht vorhanden"""
        try:
            self._d("create_collection: fetching collections")
            # Collection Info abrufen
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            # Ziel-Vektorgröße ermitteln
            desired_vec_size = self._get_embedding_dim()

            if self.collection_name not in collection_names:
                # Collection erstellen
                self._d(f"creating collection {self.collection_name} size={desired_vec_size}")
                from qdrant_client.http.models import HnswConfigDiff, OptimizersConfigDiff, ScalarQuantizationConfig
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=desired_vec_size,
                        distance=Distance.COSINE
                    ),
                    hnsw_config=HnswConfigDiff(m=32, ef_construct=128),
                    optimizers_config=OptimizersConfigDiff(indexing_threshold=20000, memmap_threshold=200000),
                    quantization_config=ScalarQuantizationConfig(scalar=ScalarQuantizationConfig.Scalar(bits=8), always_ram=False)
                )
                print(f"✅ Qdrant Collection '{self.collection_name}' erstellt")
            else:
                # Prüfe bestehende Vektorgröße und gleiche ggf. an
                self._d(f"collection {self.collection_name} exists; checking dimensions")
                info = self.qdrant_client.get_collection(collection_name=self.collection_name)
                current_size = getattr(getattr(getattr(info, 'config', None), 'params', None), 'vectors', None)
                current_size = getattr(current_size, 'size', None)
                if isinstance(current_size, int) and current_size != desired_vec_size:
                    print(f"⚠️ Qdrant Collection '{self.collection_name}' hat Dimension {current_size}, erwartet {desired_vec_size}. Erstelle neu…")
                    self._d("deleting & recreating collection due to dim mismatch")
                    self.qdrant_client.delete_collection(self.collection_name)
                    from qdrant_client.http.models import HnswConfigDiff, OptimizersConfigDiff, ScalarQuantizationConfig
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=desired_vec_size,
                            distance=Distance.COSINE
                        ),
                        hnsw_config=HnswConfigDiff(m=32, ef_construct=128),
                        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000, memmap_threshold=200000),
                        quantization_config=ScalarQuantizationConfig(scalar=ScalarQuantizationConfig.Scalar(bits=8), always_ram=False)
                    )
                    print(f"✅ Qdrant Collection '{self.collection_name}' neu erstellt mit Dimension {desired_vec_size}")
                else:
                    print(f"ℹ️ Qdrant Collection '{self.collection_name}' existiert bereits")
                
        except Exception as e:
            print(f"❌ Fehler beim Erstellen der Collection: {e}")
    
    def parse_code_with_ast(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parst Python-Datei mit AST und extrahiert semantische Chunks
        
        Args:
            file_path: Pfad zur Python-Datei
            
        Returns:
            List von Chunks mit Metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # AST parsen
            tree = ast.parse(source_code)
            
            chunks = []
            current_chunk = {
                'file': os.path.relpath(file_path, 'src'),
                'type': 'file',
                'content': '',
                'metadata': {
                    'start_line': 1,
                    'end_line': 0,
                    'ast_nodes': []
                }
            }
            
            for node in ast.walk(tree):
                # Node-Typ extrahieren
                node_type = type(node).__name__
                start_line = getattr(node, 'lineno', 1)
                end_line = getattr(node, 'end_lineno', start_line)
                
                # Node-Repräsentation als String
                node_str = ast.get_source_segment(source_code, node)
                
                if node_str:
                    current_chunk['content'] += f"\n{node_str}"
                    current_chunk['metadata']['ast_nodes'].append({
                        'type': node_type,
                        'start_line': start_line,
                        'end_line': end_line,
                        'name': getattr(node, 'name', None) if hasattr(node, 'name') else None
                    })
                    current_chunk['metadata']['end_line'] = end_line
            
            # Chunk finalisieren
            if current_chunk['content'].strip():
                current_chunk['metadata']['size'] = len(current_chunk['content'])
                chunks.append(current_chunk)
            
            print(f"✅ {len(chunks)} AST-Chunks aus {file_path} extrahiert")
            return chunks
            
        except SyntaxError as e:
            print(f"⚠️ Syntax-Fehler in {file_path}: {e}")
            # Fallback: Zeilen-basierte Chunking
            return self._fallback_chunking(file_path)
        except Exception as e:
            print(f"❌ Fehler beim AST-Parsing von {file_path}: {e}")
            return []
    
    def _fallback_chunking(self, file_path: str) -> List[Dict[str, Any]]:
        """Fallback Chunking bei AST-Fehlern (Zeilen-basiert)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            chunks = []
            chunk_size = 20  # Zeilen pro Chunk
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i + chunk_size]
                chunk_content = ''.join(chunk_lines)
                
                if chunk_content.strip():
                    chunks.append({
                        'file': os.path.relpath(file_path, 'src'),
                        'type': 'line_chunk',
                        'content': chunk_content.strip(),
                        'metadata': {
                            'start_line': i + 1,
                            'end_line': min(i + chunk_size, len(lines)),
                            'size': len(chunk_content)
                        }
                    })
            
            print(f"ℹ️ Fallback-Chunking: {len(chunks)} Chunks aus {file_path}")
            return chunks
            
        except Exception as e:
            print(f"❌ Fallback-Chunking fehlgeschlagen: {e}")
            return []
    
    def build_index_from_directory(self, directory_path: str = "src"):
        """
        Baut den Enterprise-Index aus einem Verzeichnis mit AST-Parsing
        
        Args:
            directory_path: Verzeichnis mit Python-Dateien
        """
        print(f"🔄 Baue Enterprise-Index aus {directory_path}...")
        t0 = time.time()
        
        # Collection erstellen
        self.create_collection()
        
        # Qdrant Vector Store initialisieren
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embed_model=self.embed_model
        )
        
        # Storage Context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Dokumente laden und parsen
        documents = []
        python_files = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory_path)
                    
                    print(f"📄 Verarbeite: {relative_path}")
                    self._d(f"AST parse start: {relative_path}")
                    
                    # AST-basierte Chunking
                    chunks = self.parse_code_with_ast(file_path)
                    self._d(f"AST parse done: chunks={len(chunks)}")
                    
                    for chunk in chunks:
                        # LlamaIndex Document erstellen (Objekte statt Dict)
                        # Reduziere Metadaten (große Felder wie 'ast_nodes' entfernen)
                        meta_src = chunk['metadata']
                        meta_small = {
                            'file_path': chunk['file'],
                            'chunk_type': chunk['type'],
                            'start_line': meta_src.get('start_line'),
                            'end_line': meta_src.get('end_line'),
                            'size': meta_src.get('size'),
                            'timestamp': datetime.now().isoformat(),
                            'source': relative_path
                        }
                        doc = Document(
                            text=chunk['content'],
                            metadata=meta_small
                        )
                        documents.append(doc)
                    
                    python_files.append(relative_path)
        
        if not documents:
            print("❌ Keine Dokumente gefunden!")
            return
        
        print(f"📚 {len(documents)} Chunks extrahiert aus {len(python_files)} Dateien")
        self._d("Index build: storing into vector store")
        
        # Index bauen mit Retry bei Qdrant-Dimensionsfehler
        def _desired_dim() -> int:
            return self._get_embedding_dim()

        try:
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=[self.ast_parser],
                show_progress=True
            )
        except Exception as e:
            msg = str(e)
            if "Vector dimension error" in msg or "expected dim" in msg:
                dim = _desired_dim()
                print(f"⚠️ Qdrant Dimensionsfehler erkannt. Setze Collection '{self.collection_name}' auf {dim} neu…")
                try:
                    self.qdrant_client.delete_collection(self.collection_name)
                except Exception:
                    pass
                from qdrant_client.http.models import HnswConfigDiff, OptimizersConfigDiff, ScalarQuantizationConfig
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE
                    ),
                    hnsw_config=HnswConfigDiff(m=32, ef_construct=128),
                    optimizers_config=OptimizersConfigDiff(indexing_threshold=20000, memmap_threshold=200000),
                    quantization_config=ScalarQuantizationConfig(scalar=ScalarQuantizationConfig.Scalar(bits=8), always_ram=False)
                )
                # neuen VectorStore benutzen
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embed_model=self.embed_model
                )
                storage_context_retry = StorageContext.from_defaults(vector_store=self.vector_store)
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context_retry,
                    embed_model=self.embed_model,
                    transformations=[self.ast_parser],
                    show_progress=True
                )
            else:
                raise

        # Index persistent machen
        self.index.storage_context.persist(persist_dir="./enterprise_index")
        
        # Index persistent machen
        self.index.storage_context.persist(persist_dir="./enterprise_index")
        
        print(f"✅ Enterprise-Index gebaut mit {len(documents)} Chunks")
        self._d(f"Index build completed in {time.time()-t0:.2f}s")
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid-Suche: Kombiniert semantische und keyword-basierte Suche
        
        Args:
            query: Suchanfrage
            top_k: Anzahl der Ergebnisse
            alpha: Gewichtung semantisch vs. keyword (0-1)
            
        Returns:
            Liste von Ergebnissen mit Hybrid-Scores
        """
        if self.index is None:
            print("❌ Index muss zuerst gebaut werden!")
            return []
        
        self._d(f"hybrid_search start: top_k={top_k} alpha={alpha}")
        # Semantische Suche
        semantic_retriever = self.index.as_retriever(similarity_top_k=top_k * 2)
        semantic_nodes = semantic_retriever.retrieve(query)
        self._d(f"semantic retrieved: {len(semantic_nodes)} nodes")
        semantic_results = []
        
        for nws in semantic_nodes:
            node_obj = getattr(nws, 'node', nws)
            score = getattr(nws, 'score', 0.0) or 0.0
            # content & metadata
            content = getattr(node_obj, 'text', None)
            if not content and hasattr(node_obj, 'get_content'):
                content = node_obj.get_content()
            metadata = getattr(node_obj, 'metadata', {}) or {}
            source = metadata.get('file_path') or metadata.get('source') or getattr(node_obj, 'node_id', 'unknown')
            semantic_results.append({
                'content': content or '',
                'metadata': metadata,
                'semantic_score': float(score),
                'source': source
            })
        
        # Keyword-Suche (einfache String-Matching als Proxy)
        keyword_results = self._keyword_search(query, top_k * 2)
        self._d(f"keyword results: {len(keyword_results)}")
        
        # Hybrid-Scoring
        hybrid_results = []
        all_results = semantic_results + keyword_results
        
        # Deduplizierung nach Quelle
        seen_sources = set()
        for result in all_results:
            source = result.get('source', result.get('file_path', 'unknown'))
            if source not in seen_sources:
                seen_sources.add(source)
                
                # Hybrid Score berechnen
                semantic_score = result.get('semantic_score', 0.0)
                keyword_score = result.get('keyword_score', 0.0)
                hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score
                
                hybrid_result = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'hybrid_score': hybrid_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'source': source
                }
                hybrid_results.append(hybrid_result)
                
                if len(hybrid_results) >= top_k:
                    break
        
        # Nach Hybrid-Score sortieren
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        print(f"🔍 Hybrid-Suche: {len(hybrid_results)} Ergebnisse für '{query[:50]}...'")
        self._d("hybrid_search end")
        return hybrid_results[:top_k]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Einfache keyword-basierte Suche als Fallback"""
        if self.index is None:
            return []
        
        # Verwende LlamaIndex Text-Index für Keyword-Suche
        keyword_retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            node_postprocessors=[]  # Kein Reranking für Keywords
        )
        
        # Einfache String-Matching Scores
        nodes = keyword_retriever.retrieve(query)
        results = []
        
        for nws in nodes:
            node_obj = getattr(nws, 'node', nws)
            content = getattr(node_obj, 'text', None)
            if not content and hasattr(node_obj, 'get_content'):
                content = node_obj.get_content()
            text_lower = (content or '').lower()
            # Einfacher Keyword-Score basierend auf Vorkommen (token overlap)
            q_tokens = [t for t in query.lower().split() if t]
            token_hits = sum(1 for t in q_tokens if t in text_lower)
            keyword_score = min(token_hits / max(1, len(q_tokens)), 1.0)
            metadata = getattr(node_obj, 'metadata', {}) or {}
            source = metadata.get('file_path') or metadata.get('source') or getattr(node_obj, 'node_id', 'unknown')
            
            results.append({
                'content': content or '',
                'metadata': metadata,
                'keyword_score': keyword_score,
                'source': source
            })
        
        return results
    
    def generate_response(self, query: str, context_results: List[Dict[str, Any]], methods: Optional[List[str]] = None) -> str:
        """
        Generiert eine Antwort mit LLM basierend auf Retrieval-Ergebnissen
        
        Args:
            query: Originale Anfrage
            context_results: Retrieval-Ergebnisse
            methods: Optional per AST extrahierte Methodenliste für gezielte Antworten
            
        Returns:
            Generierte Antwort mit Citations
        """
        if not context_results:
            return "Keine relevanten Informationen gefunden."
        
        # Kontext aufbauen
        context = ""
        citations = []
        
        for i, result in enumerate(context_results[:3]):  # Top 3 Ergebnisse
            context += f"\n\nSnippet {i+1} (Score: {result['hybrid_score']:.3f}):\n"
            context += f"Datei: {result['metadata'].get('file_path', 'Unbekannt')}\n"
            context += f"{result['content'][:500]}..."  # Erste 500 Zeichen
            
            citations.append({
                'snippet': i+1,
                'file': result['metadata'].get('file_path', 'Unbekannt'),
                'score': result['hybrid_score']
            })
        
        # Prompt für LLM
        methods_block = "\n\n[AST-extracted methods]\n" + "\n".join(methods) if methods else ""
        prompt = f"""
Du bist ein Code-Analyst. Basierend auf dem folgenden Code-Kontext, beantworte die Frage präzise und mit Verweisen auf die Quellen.

Frage: {query}

Kontext:
{context}

Hinweis:
- Wenn eine Methodenliste aus dem AST bereitgestellt ist, gib GENAU diese Methoden als Antwort aus (ohne zu halluzinieren).\n- Formatiere die Ausgabe als einfache Liste.
{methods_block}

Antwort:
"""
        
        try:
            if getattr(self, 'use_ollama', False):
                self._d("LLM backend: Ollama generate start")
                # Generate via Ollama
                url = f"{self.ollama_host}/api/generate"
                payload = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 8192,
                        "num_predict": 512,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "gpu_layers": 999
                    }
                }
                try:
                    r = requests.post(url, json=payload, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    text = data.get('response', '')
                    self._d(f"Ollama generate ok: {len(text)} chars")
                except Exception as oe:
                    print(f"⚠️ Ollama Generation fehlgeschlagen: {oe}")
                    text = ""
                response_text = str(text) + "\n\n📚 Quellen:\n"
                for citation in citations:
                    response_text += f"- Snippet {citation['snippet']} ({citation['file']}, Score: {citation['score']:.3f})\n"
                return response_text
            # Default HF path
            self._d("LLM backend: HF generate start")
            response = self.llm.complete(prompt)
            self._d("HF generate done")
            response_text = str(response) + "\n\n📚 Quellen:\n"
            for citation in citations:
                response_text += f"- Snippet {citation['snippet']} ({citation['file']}, Score: {citation['score']:.3f})\n"
            return response_text
        except Exception as e:
            print(f"❌ LLM-Generierung fehlgeschlagen: {e}")
            # Retry once on CPU to avoid CUDA asserts
            try:
                self._d("Retry LLM on CPU")
                self.llm = self._build_llm(device_preference="cpu")
                response = self.llm.complete(prompt)
                response_text = str(response) + "\n\n📚 Quellen:\n"
                for citation in citations:
                    response_text += f"- Snippet {citation['snippet']} ({citation['file']}, Score: {citation['score']:.3f})\n"
                return response_text
            except Exception as e2:
                print(f"❌ LLM CPU-Retry fehlgeschlagen: {e2}")
                fallback_response = f"Zu '{query}':\n\n"
                for i, result in enumerate(context_results[:3]):
                    fallback_response += f"Snippet {i+1} (Score: {result['hybrid_score']:.3f}):\n"
                    fallback_response += f"Datei: {result['metadata'].get('file_path', 'Unbekannt')}\n"
                    fallback_response += f"{result['content'][:200]}...\n\n"
                return fallback_response
    
    def query(self, query: str, top_k: int = 5, alpha: float = 0.7, generate_response: bool = True) -> Dict[str, Any]:
        """
        Vollständige Query-Pipeline: Hybrid-Suche + Generierung
        
        Args:
            query: Suchanfrage
            top_k: Anzahl der Ergebnisse
            alpha: Hybrid-Gewichtung
            generate_response: Ob LLM-Antwort generieren
            
        Returns:
            Dictionary mit Ergebnissen und Antwort
        """
        print(f"🔍 Starte Enterprise-Query: '{query}'")
        self._d("query pipeline start")
        
        # Hybrid-Suche
        results = self.hybrid_search(query, top_k, alpha)
        self._d(f"retrieval done: {len(results)} results")
        # Optional: Methodenliste aus Query-Datei extrahieren
        methods: Optional[List[str]] = None
        try:
            file_match = re.search(r"([A-Za-z0-9_\-]+\.py)", query)
            candidate = file_match.group(1) if file_match else None
            if candidate:
                # Suche Datei unterhalb von 'src'
                found_path = None
                for root, dirs, files in os.walk('src'):
                    if candidate in files:
                        found_path = os.path.join(root, candidate)
                        break
                if found_path and os.path.exists(found_path):
                    methods = self._extract_methods_from_python(found_path)
        except Exception:
            methods = None
        
        if generate_response:
            # LLM-Generierung
            answer = self.generate_response(query, results, methods=methods)
            self._d("generation done")
        else:
            # Nur Ergebnisse
            answer = "Generierung deaktiviert. Hier sind die Top-Ergebnisse:\n\n"
            for i, result in enumerate(results):
                answer += f"#{i+1} (Score: {result['hybrid_score']:.3f})\n"
                answer += f"Datei: {result['metadata'].get('file_path', 'Unbekannt')}\n"
                answer += f"{result['content'][:150]}...\n\n"
            if methods:
                answer += "\nAST-Methoden:\n" + "\n".join(f"- {m}" for m in methods)
        
        out = {
            'query': query,
            'results': results,
            'answer': answer,
            'methods': methods,
            'top_k': top_k,
            'alpha': alpha,
            'timestamp': datetime.now().isoformat()
        }
        self._d("query pipeline end")
        return out

    # ---- Comparison with Fungus (MCPMRetriever) ----
    def _load_docs_from_markdown(self, file_path: str, include_text: bool = False) -> List[str]:
        path = os.path.abspath(file_path)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        code_pattern = re.compile(r"```([a-zA-Z0-9_+\-]*)\n([\s\S]*?)```", re.MULTILINE)
        documents: List[str] = []
        last_end = 0
        for match in code_pattern.finditer(content):
            lang = match.group(1) or ""
            code = match.group(2).strip('\n')
            full_block = f"```{lang}\n{code}\n```"
            documents.append(full_block)
            if include_text:
                prefix_text = content[last_end:match.start()].strip()
                if prefix_text:
                    for para in re.split(r"\n\n+", prefix_text):
                        para = para.strip()
                        if para:
                            documents.append(para)
            last_end = match.end()
        if include_text:
            tail = content[last_end:].strip()
            if tail:
                for para in re.split(r"\n\n+", tail):
                    para = para.strip()
                    if para:
                        documents.append(para)
        if not documents:
            documents = [content]
        return documents

    def compare_fungus_vs_enterprise(self, query: str, top_k: int = 5, alpha: float = 0.7,
                                     docs_file: Optional[str] = None,
                                     md_codeblocks: bool = True,
                                     md_include_text: bool = False) -> Dict[str, Any]:
        # Enterprise results
        ent_results = self.hybrid_search(query, top_k, alpha)

        # Fungus results
        fungus = None
        fungus_results: List[Dict[str, Any]] = []
        methods_info: Dict[str, Any] = {}
        if MCPMRetriever is not None:
            try:
                self._d("init fungus retriever")
                try:
                    fungus = MCPMRetriever(num_agents=200, max_iterations=60, device_mode=("cuda" if torch.cuda.is_available() else "cpu"))
                except Exception:
                    fungus = MCPMRetriever(num_agents=200, max_iterations=60)  # type: ignore

                docs: List[str] = []
                if docs_file and os.path.exists(docs_file):
                    self._d(f"fungus docs_file provided: {docs_file}")
                    if docs_file.endswith('.py'):
                        # AST method extraction and multi-window chunks for Fungus
                        methods = self._extract_methods_from_python(docs_file)
                        methods_info = {"file": os.path.relpath(docs_file), "methods": methods}
                        docs = self._chunk_python_file_for_fungus(docs_file)
                        if not docs:
                            # fallback to full file content
                            with open(docs_file, 'r', encoding='utf-8', errors='ignore') as f:
                                docs = [f.read()]
                    elif md_codeblocks:
                        docs = self._load_docs_from_markdown(docs_file, include_text=md_include_text)
                    else:
                        with open(docs_file, 'r', encoding='utf-8', errors='ignore') as f:
                            docs = [f.read()]
                # If no docs provided, try to use code files as plain text snippets
                if not docs:
                    # Fallback minimal corpus: filenames from enterprise build
                    docs = [r.get('content', '') for r in ent_results if r.get('content')]
                self._d(f"fungus corpus size: {len(docs)} units")
                if docs:
                    fungus.add_documents(docs)  # type: ignore
                    self._d("fungus search start")
                    f_out = fungus.search(query, top_k=top_k)  # type: ignore
                    items = f_out.get('results', []) if isinstance(f_out, dict) else []
                    self._d(f"fungus search done: {len(items)} items")
                    for it in items:
                        fungus_results.append({
                            'content': it.get('content', ''),
                            'score': float(it.get('relevance_score', 0.0)),
                            'source': 'fungus'
                        })
            except Exception as e:
                print(f"⚠️ Fungus Vergleich fehlgeschlagen: {e}")
        else:
            print("ℹ️ MCPMRetriever nicht verfügbar – überspringe Fungus-Vergleich")

        # Normalize and combine for hybrid
        def _norm(vals: List[float]) -> List[float]:
            if not vals:
                return []
            vmin, vmax = min(vals), max(vals)
            if vmax - vmin < 1e-9:
                return [1.0 for _ in vals]
            return [(v - vmin) / (vmax - vmin) for v in vals]

        ent_sc = [float(r.get('hybrid_score', r.get('semantic_score', 0.0))) for r in ent_results]
        ent_sc_n = _norm(ent_sc)
        ent_norm = [
            {
                'content': r.get('content', ''),
                'score': s,
                'source': r.get('metadata', {}).get('file_path', r.get('source', 'enterprise'))
            } for r, s in zip(ent_results, ent_sc_n)
        ]
        fun_sc = [fr.get('score', 0.0) for fr in fungus_results]
        fun_sc_n = _norm(fun_sc)
        fun_norm = [
            {**fr, 'score': s} for fr, s in zip(fungus_results, fun_sc_n)
        ]

        # Simple hybrid: take top_k by max of (ent_norm score, fun_norm score)
        hybrid_pool = ent_norm + fun_norm
        hybrid_sorted = sorted(hybrid_pool, key=lambda x: x['score'], reverse=True)[:top_k]

        return {
            'query': query,
            'enterprise': ent_norm[:top_k],
            'fungus': fun_norm[:top_k],
            'hybrid': hybrid_sorted,
            'methods': methods_info if methods_info else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def load_index(self, persist_dir: str = "./enterprise_index"):
        """Lädt einen gespeicherten Index"""
        try:
            # StorageContext mit Persist-Verzeichnis und aktuellem VectorStore
            if self.vector_store is None:
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embed_model=self.embed_model
                )

            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir,
                vector_store=self.vector_store
            )

            # Versuche vollständigen Indexzustand zu laden
            try:
                self.index = load_index_from_storage(
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )
            except Exception:
                # Fallback: nur VectorStore anbinden (z.B. wenn Persist-Index inkompatibel ist)
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )
            
            print(f"✅ Index aus {persist_dir} geladen")
            
        except Exception as e:
            print(f"❌ Fehler beim Laden des Index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über den Index zurück"""
        if self.index is None:
            return {"error": "Index nicht geladen"}
        
        try:
            # Qdrant Stats
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.collection_name
            )
            
            return {
                'collection_name': self.collection_name,
                'points_count': collection_info.points_count,
                'vectors_count': collection_info.vectors_count,
                'vector_size': collection_info.config.params.vectors.size,
                'index_loaded': self.index is not None,
                'embedding_model': self.embedding_model,
                'llm_model': self.llm_model
            }
            
        except Exception as e:
            print(f"❌ Fehler bei Stats: {e}")
            return {"error": str(e)}

# Backward-compatibility: alias class name
# After class RagV1 definition (above), ensure EnterpriseCodeRAG resolves to RagV1
EnterpriseCodeRAG = RagV1  # type: ignore

def main():
    """CLI für Enterprise-RAG"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise Code RAG mit Qdrant & LlamaIndex")
    subparsers = parser.add_subparsers(dest='command', help='Befehle')
    
    # Index bauen
    build_parser = subparsers.add_parser('build', help='Index aus Verzeichnis bauen')
    build_parser.add_argument('--directory', default='src', help='Verzeichnis zum Indexieren')
    
    # Query
    query_parser = subparsers.add_parser('query', help='Query ausführen')
    query_parser.add_argument('query_text', help='Suchanfrage')
    query_parser.add_argument('--top-k', type=int, default=5, help='Anzahl Ergebnisse')
    query_parser.add_argument('--alpha', type=float, default=0.7, help='Hybrid-Gewichtung')
    query_parser.add_argument('--no-generate', action='store_true', help='Keine LLM-Generierung')
    query_parser.add_argument('--llm-model', default=None, help='Überschreibt das LLM Modell')
    query_parser.add_argument('--llm-device', default=None, choices=['auto','cuda','cpu'], help='LLM Device überschreiben')
    query_parser.add_argument('--use-ollama', action='store_true', help='Ollama für Generierung nutzen')
    query_parser.add_argument('--ollama-model', default='qwen2.5-coder:7b', help='Ollama Modellname')
    query_parser.add_argument('--ollama-host', default='http://127.0.0.1:11434', help='Ollama Host URL')

    # Compare (Fungus vs Enterprise vs Hybrid)
    cmp_parser = subparsers.add_parser('compare', help='Vergleich: Fungus vs Enterprise vs Hybrid')
    cmp_parser.add_argument('query_text', help='Suchanfrage')
    cmp_parser.add_argument('--top-k', type=int, default=5, help='Anzahl Ergebnisse pro System')
    cmp_parser.add_argument('--alpha', type=float, default=0.7, help='Hybrid-Gewichtung für Enterprise')
    cmp_parser.add_argument('--docs-file', default=None, help='Optional: Markdown/Text Dokumente für Fungus')
    cmp_parser.add_argument('--md-codeblocks', action='store_true', help='Markdown Codeblöcke extrahieren')
    cmp_parser.add_argument('--md-include-text', action='store_true', help='Markdown Nicht‑Code Text einbeziehen')
    cmp_parser.add_argument('--llm-model', default=None, help='Überschreibt das LLM Modell')
    cmp_parser.add_argument('--llm-device', default=None, choices=['auto','cuda','cpu'], help='LLM Device überschreiben')
    cmp_parser.add_argument('--use-ollama', action='store_true', help='Ollama für Generierung nutzen')
    cmp_parser.add_argument('--ollama-model', default='qwen2.5-coder:7b', help='Ollama Modellname')
    cmp_parser.add_argument('--ollama-host', default='http://127.0.0.1:11434', help='Ollama Host URL')
    
    # Stats
    subparsers.add_parser('stats', help='Index-Statistiken anzeigen')
    
    # Load
    load_parser = subparsers.add_parser('load', help='Index laden')
    load_parser.add_argument('--dir', default='./enterprise_index', help='Persist-Directory')

    # Method span
    mspan = subparsers.add_parser('method', help='Gebe exakten Methodenspan aus einer Datei zurück')
    mspan.add_argument('file', help='Pfad zur Python-Datei (unter src)')
    mspan.add_argument('method', help='Methodenname')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Enterprise RAG initialisieren
    rag = RagV1(
        qdrant_url="http://localhost:6337",  # Custom Qdrant Port
        embedding_model="google/embeddinggemma-300m",
        llm_model=(args.llm_model if hasattr(args, 'llm_model') and args.llm_model else "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        llm_device=(args.llm_device if hasattr(args, 'llm_device') and args.llm_device else "auto"),
        use_ollama=bool(getattr(args, 'use_ollama', False)),
        ollama_model=(args.ollama_model if hasattr(args, 'ollama_model') and args.ollama_model else 'qwen2.5-coder:7b'),
        ollama_host=(args.ollama_host if hasattr(args, 'ollama_host') and args.ollama_host else 'http://127.0.0.1:11434')
    )
    
    try:
        if args.command == 'build':
            rag.build_index_from_directory(args.directory)
            
        elif args.command == 'query':
            # Index laden falls nicht vorhanden
            if rag.index is None:
                print("ℹ️ Lade Index...")
                rag.load_index()
            
            result = rag.query(
                query=args.query_text,
                top_k=args.top_k,
                alpha=args.alpha,
                generate_response=not args.no_generate
            )
            
            print("\n" + "="*80)
            print("🎯 ANTWORT:")
            print(result['answer'])
            print("="*80)
            
            # Stats ausgeben
            print(f"\n📊 Stats: {len(result['results'])} Ergebnisse, Alpha={args.alpha}")
            
        elif args.command == 'compare':
            # Ensure index for enterprise side
            if rag.index is None:
                print("ℹ️ Lade Index...")
                rag.load_index()
            cmp = rag.compare_fungus_vs_enterprise(
                query=args.query_text,
                top_k=args.top_k,
                alpha=args.alpha,
                docs_file=args.docs_file,
                md_codeblocks=bool(args.md_codeblocks),
                md_include_text=bool(args.md_include_text)
            )
            print(json.dumps(cmp, indent=2, ensure_ascii=False))

        elif args.command == 'method':
            # Resolve file under src
            candidate = None
            if os.path.isabs(args.file) and os.path.exists(args.file):
                candidate = args.file
            else:
                for root, dirs, files in os.walk('src'):
                    if os.path.basename(args.file) in files:
                        p = os.path.join(root, os.path.basename(args.file))
                        if p.endswith(os.path.basename(args.file)):
                            candidate = p
                            break
            if not candidate:
                print(json.dumps({"error": "file_not_found"}))
            else:
                span = rag.extract_method_span(candidate, args.method)
                print(json.dumps(span or {"error": "method_not_found"}, ensure_ascii=False, indent=2))

        elif args.command == 'stats':
            if rag.index is None:
                rag.load_index()
            
            stats = rag.get_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
        elif args.command == 'load':
            rag.load_index(args.dir)
            
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()