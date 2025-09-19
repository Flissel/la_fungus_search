#!/usr/bin/env python3
"""
Einfache Kommandozeilen-Version von EmbeddingGemma
"""

import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys
import os

class EmbeddingGemmaCLI:
    def __init__(self):
        self.model = None
        self.model_name = "google/embeddinggemma-300m"
    
    def load_model(self):
        """Modell laden"""
        if self.model is None:
            print("🔄 Lade EmbeddingGemma Modell...")
            try:
                self.model = SentenceTransformer(self.model_name)
                print("✅ Modell erfolgreich geladen!")
            except Exception as e:
                print(f"❌ Fehler beim Laden: {e}")
                sys.exit(1)
        return self.model
    
    def encode_text(self, text, prompt_type="document"):
        """Einzelnen Text encodieren"""
        model = self.load_model()
        return model.encode([text], prompt_name=prompt_type)[0]
    
    def encode_texts(self, texts, prompt_type="document"):
        """Mehrere Texte encodieren"""
        model = self.load_model()
        return model.encode(texts, prompt_name=prompt_type)
    
    def search_documents(self, query, documents, top_k=5):
        """Suche in Dokumenten"""
        print(f"🔍 Suche: '{query}'")
        print(f"📚 Durchsuche {len(documents)} Dokumente...")
        
        # Embeddings erstellen
        query_emb = self.encode_text(query, "query")
        doc_embs = self.encode_texts(documents, "document")
        
        # Similarity berechnen
        similarities = cosine_similarity([query_emb], doc_embs)[0]
        
        # Sortieren
        sorted_indices = np.argsort(similarities)[::-1]
        
        print(f"\n🎯 Top {min(top_k, len(documents))} Ergebnisse:")
        print("-" * 60)
        
        results = []
        for i, idx in enumerate(sorted_indices[:top_k]):
            score = similarities[idx]
            doc = documents[idx]
            
            print(f"#{i+1} (Score: {score:.4f})")
            print(f"📄 {doc}")
            print()
            
            results.append({
                'rank': i+1,
                'document': doc,
                'score': float(score),
                'index': int(idx)
            })
        
        return results
    
    def similarity_matrix(self, texts):
        """Similarity Matrix für Texte erstellen"""
        print(f"🧮 Berechne Similarity Matrix für {len(texts)} Texte...")
        
        embeddings = self.encode_texts(texts)
        matrix = cosine_similarity(embeddings)
        
        print("\n📊 Similarity Matrix:")
        print("-" * 50)
        
        # Header
        print("     ", end="")
        for i in range(len(texts)):
            print(f"T{i+1:2d}  ", end="")
        print()
        
        # Matrix
        for i, row in enumerate(matrix):
            print(f"T{i+1:2d}: ", end="")
            for val in row:
                print(f"{val:.2f} ", end="")
            print()
        
        return matrix.tolist()

def main():
    parser = argparse.ArgumentParser(description="EmbeddingGemma CLI Tool")
    subparsers = parser.add_subparsers(dest='command', help='Verfügbare Kommandos')
    
    # Search Command
    search_parser = subparsers.add_parser('search', help='Suche in Dokumenten')
    search_parser.add_argument('query', help='Suchanfrage')
    search_parser.add_argument('--docs', nargs='+', required=True, help='Dokumente zum Durchsuchen')
    search_parser.add_argument('--top-k', type=int, default=5, help='Anzahl der Top-Ergebnisse')
    search_parser.add_argument('--output', help='Output JSON Datei')
    
    # Similarity Command
    sim_parser = subparsers.add_parser('similarity', help='Similarity Matrix erstellen')
    sim_parser.add_argument('texts', nargs='+', help='Texte für Similarity Matrix')
    sim_parser.add_argument('--output', help='Output JSON Datei')
    
    # Encode Command
    encode_parser = subparsers.add_parser('encode', help='Text zu Embedding')
    encode_parser.add_argument('text', help='Text zum Encodieren')
    encode_parser.add_argument('--type', choices=['query', 'document'], default='document', help='Prompt Type')
    encode_parser.add_argument('--output', help='Output Datei für Embedding')
    
    # File Commands
    file_parser = subparsers.add_parser('file-search', help='Suche in Textdatei')
    file_parser.add_argument('query', help='Suchanfrage')
    file_parser.add_argument('file', help='Textdatei (eine Zeile = ein Dokument)')
    file_parser.add_argument('--top-k', type=int, default=5, help='Anzahl der Top-Ergebnisse')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = EmbeddingGemmaCLI()
    
    try:
        if args.command == 'search':
            results = cli.search_documents(args.query, args.docs, args.top_k)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({
                        'query': args.query,
                        'results': results
                    }, f, indent=2, ensure_ascii=False)
                print(f"💾 Ergebnisse gespeichert in {args.output}")
        
        elif args.command == 'similarity':
            matrix = cli.similarity_matrix(args.texts)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({
                        'texts': args.texts,
                        'similarity_matrix': matrix
                    }, f, indent=2, ensure_ascii=False)
                print(f"💾 Matrix gespeichert in {args.output}")
        
        elif args.command == 'encode':
            embedding = cli.encode_text(args.text, args.type)
            print(f"📊 Embedding Shape: {embedding.shape}")
            print(f"📈 Erste 5 Werte: {embedding[:5]}")
            
            if args.output:
                np.save(args.output, embedding)
                print(f"💾 Embedding gespeichert in {args.output}")
        
        elif args.command == 'file-search':
            if not os.path.exists(args.file):
                print(f"❌ Datei nicht gefunden: {args.file}")
                return
            
            with open(args.file, 'r', encoding='utf-8') as f:
                documents = [line.strip() for line in f if line.strip()]
            
            if not documents:
                print("❌ Keine Dokumente in der Datei gefunden")
                return
            
            results = cli.search_documents(args.query, documents, args.top_k)
    
    except KeyboardInterrupt:
        print("\n❌ Abgebrochen")
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()
