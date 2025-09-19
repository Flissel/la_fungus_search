#!/usr/bin/env python3
"""
Einfache RAG (Retrieval Augmented Generation) Pipeline mit EmbeddingGemma

LEGACY NOTICE:
- This SimpleRAG module is a minimal FAISS-based example.
- For production and large repos, prefer `embeddinggemma.enterprise_rag.EnterpriseCodeRAG`
  which provides Qdrant + LlamaIndex indexing, AST-based chunking, hybrid retrieval, and
  optional generation via HF or Ollama. The primary UI is `streamlit_fungus.py` with a
  dedicated "Rag" section for Enterprise RAG.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import pickle
from datetime import datetime

class SimpleRAG:
    """Einfache RAG Implementation mit EmbeddingGemma"""
    
    def __init__(self, model_name="google/embeddinggemma-300m", cache_dir="./rag_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_model(self):
        """EmbeddingGemma Modell laden"""
        if self.model is None:
            print("🔄 Lade EmbeddingGemma...")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Modell geladen!")
        return self.model
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Dokumente zur Wissensbasis hinzufügen"""
        print(f"📚 Füge {len(documents)} Dokumente hinzu...")
        
        model = self.load_model()
        
        # Embeddings erstellen
        embeddings = model.encode(
            documents, 
            prompt_name="document",
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # FAISS Index erstellen/erweitern
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product für Cosine
        
        # Normalisieren für Cosine Similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Dokumente und Metadata speichern
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"index": len(self.documents) + i} for i in range(len(documents))])
        
        print(f"✅ {len(documents)} Dokumente hinzugefügt. Total: {len(self.documents)}")
    
    def add_documents_from_file(self, file_path: str, encoding='utf-8'):
        """Dokumente aus Textdatei laden (eine Zeile = ein Dokument)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        with open(file_path, 'r', encoding=encoding) as f:
            documents = [line.strip() for line in f if line.strip()]
        
        metadata = [{"source": file_path, "line": i+1} for i in range(len(documents))]
        self.add_documents(documents, metadata)
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """Relevante Dokumente suchen"""
        if self.index is None or len(self.documents) == 0:
            print("❌ Keine Dokumente in der Wissensbasis!")
            return []
        
        model = self.load_model()
        
        # Query Embedding erstellen
        query_embedding = model.encode([query], prompt_name="query", convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Suche im Index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx],
                    'index': int(idx)
                })
        
        return results
    
    def save_knowledge_base(self, path: str):
        """Wissensbasis speichern"""
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat()
        }
        
        # JSON Daten speichern
        with open(f"{path}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # FAISS Index speichern
        if self.index is not None:
            faiss.write_index(self.index, f"{path}.faiss")
        
        print(f"💾 Wissensbasis gespeichert: {path}")
    
    def load_knowledge_base(self, path: str):
        """Wissensbasis laden"""
        # JSON Daten laden
        with open(f"{path}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = data['documents']
        self.metadata = data['metadata']
        
        # FAISS Index laden
        if os.path.exists(f"{path}.faiss"):
            self.index = faiss.read_index(f"{path}.faiss")
        
        print(f"📖 Wissensbasis geladen: {len(self.documents)} Dokumente")
    
    def interactive_search(self):
        """Interaktive Suche im Terminal"""
        print("🔍 Interaktive Suche gestartet (Ctrl+C zum Beenden)")
        print("-" * 50)
        
        try:
            while True:
                query = input("\n❓ Deine Frage: ").strip()
                if not query:
                    continue
                
                results = self.search(query, top_k=3)
                
                if results:
                    print(f"\n🎯 Top {len(results)} Ergebnisse:")
                    print("-" * 40)
                    
                    for i, result in enumerate(results):
                        print(f"\n#{i+1} (Score: {result['score']:.4f})")
                        print(f"📄 {result['document']}")
                        if 'source' in result['metadata']:
                            print(f"🔗 Quelle: {result['metadata']['source']}")
                else:
                    print("❌ Keine relevanten Dokumente gefunden")
        
        except KeyboardInterrupt:
            print("\n👋 Auf Wiedersehen!")

def main():
    """Hauptfunktion für CLI Nutzung"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple RAG mit EmbeddingGemma")
    subparsers = parser.add_subparsers(dest='command', help='Verfügbare Kommandos')
    
    # Add documents
    add_parser = subparsers.add_parser('add', help='Dokumente hinzufügen')
    add_parser.add_argument('--file', help='Textdatei mit Dokumenten')
    add_parser.add_argument('--docs', nargs='+', help='Dokumente als Argumente')
    add_parser.add_argument('--save', help='Wissensbasis speichern unter diesem Namen')
    
    # Search
    search_parser = subparsers.add_parser('search', help='In Wissensbasis suchen')
    search_parser.add_argument('query', help='Suchanfrage')
    search_parser.add_argument('--load', help='Wissensbasis laden')
    search_parser.add_argument('--top-k', type=int, default=5, help='Anzahl Ergebnisse')
    
    # Interactive
    interactive_parser = subparsers.add_parser('interactive', help='Interaktive Suche')
    interactive_parser.add_argument('--load', help='Wissensbasis laden')
    
    # Demo
    demo_parser = subparsers.add_parser('demo', help='Demo mit Beispieldaten')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    rag = SimpleRAG()
    
    try:
        if args.command == 'add':
            if args.file:
                rag.add_documents_from_file(args.file)
            elif args.docs:
                rag.add_documents(args.docs)
            else:
                print("❌ Bitte --file oder --docs angeben")
                return
            
            if args.save:
                rag.save_knowledge_base(args.save)
        
        elif args.command == 'search':
            if args.load:
                rag.load_knowledge_base(args.load)
            
            results = rag.search(args.query, top_k=args.top_k)
            
            if results:
                print(f"🔍 Suchergebnisse für: '{args.query}'")
                print("-" * 50)
                
                for i, result in enumerate(results):
                    print(f"\n#{i+1} (Score: {result['score']:.4f})")
                    print(f"📄 {result['document']}")
            else:
                print("❌ Keine Ergebnisse gefunden")
        
        elif args.command == 'interactive':
            if args.load:
                rag.load_knowledge_base(args.load)
            
            rag.interactive_search()
        
        elif args.command == 'demo':
            print("🎯 Demo wird gestartet...")
            
            # Beispieldokumente
            demo_docs = [
                "Python ist eine hochlevel Programmiersprache, die für ihre einfache Syntax bekannt ist.",
                "Machine Learning ist ein Teilbereich der künstlichen Intelligenz.",
                "Deep Learning nutzt neuronale Netze mit vielen Schichten.",
                "Natural Language Processing ermöglicht es Computern, menschliche Sprache zu verstehen.",
                "Computer Vision beschäftigt sich mit der automatischen Analyse von Bildern.",
                "Reinforcement Learning ist ein Lernverfahren, bei dem ein Agent durch Belohnungen lernt.",
                "TensorFlow ist ein beliebtes Framework für Machine Learning.",
                "PyTorch ist ein weiteres populäres Deep Learning Framework.",
                "Pandas ist eine Python-Bibliothek für Datenanalyse.",
                "NumPy bietet effiziente mathematische Operationen für Python."
            ]
            
            rag.add_documents(demo_docs)
            
            # Demo-Suchen
            demo_queries = [
                "Was ist Python?",
                "Erkläre mir Machine Learning",
                "Welche Frameworks gibt es?",
                "Wie funktioniert Deep Learning?"
            ]
            
            for query in demo_queries:
                print(f"\n🔍 Demo-Suche: '{query}'")
                print("-" * 40)
                
                results = rag.search(query, top_k=2)
                for i, result in enumerate(results):
                    print(f"#{i+1} ({result['score']:.3f}): {result['document']}")
            
            print("\n✨ Demo beendet! Du kannst jetzt 'interactive' verwenden.")
    
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()
