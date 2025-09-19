#!/usr/bin/env python3
"""
EmbeddingGemma Desktop Interface
Ein einfaches GUI für Google's EmbeddingGemma-300M Modell
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import json
import os
from datetime import datetime

# Konfiguration
MODEL_NAME = "google/embeddinggemma-300m"
CACHE_DIR = "./embedding_cache"

class EmbeddingService:
    """Service für EmbeddingGemma Operationen"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @st.cache_resource
    def load_model(_self):
        """Modell laden mit Streamlit Caching"""
        try:
            model = SentenceTransformer(MODEL_NAME)
            model = model.to(_self.device)
            return model
        except Exception as e:
            st.error(f"Fehler beim Laden des Modells: {str(e)}")
            st.info("Stelle sicher, dass du bei Hugging Face angemeldet bist und die EmbeddingGemma Lizenz akzeptiert hast.")
            return None
    
    def get_model(self):
        """Modell abrufen"""
        if self.model is None:
            self.model = self.load_model()
        return self.model
    
    def encode_texts(self, texts: List[str], prompt_type: str = "document") -> Optional[np.ndarray]:
        """Texte zu Embeddings konvertieren"""
        model = self.get_model()
        if model is None:
            return None
        
        try:
            embeddings = model.encode(
                texts, 
                prompt_name=prompt_type,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            st.error(f"Fehler beim Encoding: {str(e)}")
            return None
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Cosine Similarity berechnen"""
        return cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)[0]

def main():
    st.set_page_config(
        page_title="EmbeddingGemma Desktop",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 EmbeddingGemma Desktop Interface")
    st.markdown("*Powered by Google's EmbeddingGemma-300M*")
    
    # Service initialisieren
    embedding_service = EmbeddingService()
    
    # Sidebar für Einstellungen
    with st.sidebar:
        st.header("⚙️ Einstellungen")
        st.info(f"Device: {embedding_service.device}")
        
        # Embedding Dimensionen
        embed_dims = st.selectbox(
            "Embedding Dimensionen",
            [768, 512, 256, 128],
            index=0,
            help="Kleinere Dimensionen = schneller aber weniger präzise"
        )
        
        # Prompt Type
        default_prompt_type = st.selectbox(
            "Standard Prompt Type",
            ["document", "query"],
            index=0
        )
    
    # Hauptbereich mit Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Text Suche", "📊 Batch Analyse", "🎯 Similarity Matrix", "💾 Export"])
    
    with tab1:
        st.header("Text Suche & Similarity")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Suchanfrage")
            query = st.text_area(
                "Deine Suchanfrage:",
                height=100,
                placeholder="z.B. Wie funktioniert Machine Learning?"
            )
            
        with col2:
            st.subheader("📚 Dokumente")
            documents_text = st.text_area(
                "Dokumente (ein Dokument pro Zeile):",
                height=200,
                placeholder="Machine Learning Tutorial\nPython Programmierung\nKünstliche Intelligenz Grundlagen"
            )
        
        if st.button("🔍 Suche starten", type="primary"):
            if query and documents_text:
                documents = [doc.strip() for doc in documents_text.split('\n') if doc.strip()]
                
                with st.spinner("Embeddings werden berechnet..."):
                    # Query Embedding
                    query_emb = embedding_service.encode_texts([query], "query")
                    
                    # Document Embeddings
                    doc_embs = embedding_service.encode_texts(documents, "document")
                    
                    if query_emb is not None and doc_embs is not None:
                        # Dimensionen reduzieren wenn gewünscht
                        if embed_dims < 768:
                            query_emb = query_emb[:, :embed_dims]
                            doc_embs = doc_embs[:, :embed_dims]
                        
                        # Similarity berechnen
                        similarities = embedding_service.compute_similarity(query_emb[0], doc_embs)
                        
                        # Ergebnisse sortieren
                        sorted_indices = np.argsort(similarities)[::-1]
                        
                        st.subheader("🎯 Suchergebnisse")
                        
                        # Ergebnisse anzeigen
                        for i, idx in enumerate(sorted_indices[:5]):
                            score = similarities[idx]
                            doc = documents[idx]
                            
                            # Farbkodierung basierend auf Score
                            if score > 0.8:
                                color = "🟢"
                            elif score > 0.6:
                                color = "🟡"
                            else:
                                color = "🔴"
                            
                            with st.expander(f"{color} Rang {i+1}: {score:.3f} - {doc[:50]}..."):
                                st.write(f"**Dokument:** {doc}")
                                st.write(f"**Similarity Score:** {score:.4f}")
                                st.progress(float(score))
                        
                        # Visualisierung
                        fig = px.bar(
                            x=[f"Doc {i+1}" for i in range(len(documents))],
                            y=similarities,
                            title="Similarity Scores",
                            labels={'x': 'Dokumente', 'y': 'Cosine Similarity'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 Batch Text Analyse")
        
        # File Upload
        uploaded_file = st.file_uploader(
            "CSV Datei hochladen",
            type=['csv'],
            help="CSV mit einer 'text' Spalte"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("📄 Datenvorschau:")
                st.dataframe(df.head())
                
                text_column = st.selectbox(
                    "Text Spalte auswählen:",
                    df.columns.tolist()
                )
                
                if st.button("🚀 Batch Analyse starten"):
                    texts = df[text_column].astype(str).tolist()
                    
                    with st.spinner(f"Verarbeite {len(texts)} Texte..."):
                        embeddings = embedding_service.encode_texts(texts)
                        
                        if embeddings is not None:
                            if embed_dims < 768:
                                embeddings = embeddings[:, :embed_dims]
                            
                            # Similarity Matrix
                            similarity_matrix = cosine_similarity(embeddings)
                            
                            # Heatmap
                            fig = go.Figure(data=go.Heatmap(
                                z=similarity_matrix,
                                colorscale='Viridis'
                            ))
                            fig.update_layout(
                                title="Text Similarity Matrix",
                                xaxis_title="Text Index",
                                yaxis_title="Text Index"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistiken
                            st.subheader("📈 Statistiken")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Durchschnittliche Similarity", f"{similarity_matrix.mean():.3f}")
                            with col2:
                                st.metric("Max Similarity", f"{similarity_matrix.max():.3f}")
                            with col3:
                                st.metric("Min Similarity", f"{similarity_matrix.min():.3f}")
                            
                            # Ähnlichste Paare
                            st.subheader("🔗 Ähnlichste Text-Paare")
                            np.fill_diagonal(similarity_matrix, 0)  # Diagonal ausschließen
                            
                            # Top 5 ähnlichste Paare finden
                            flat_indices = np.unravel_index(
                                np.argsort(similarity_matrix.ravel())[-5:], 
                                similarity_matrix.shape
                            )
                            
                            for i, (row, col) in enumerate(zip(flat_indices[0][::-1], flat_indices[1][::-1])):
                                score = similarity_matrix[row, col]
                                st.write(f"**Paar {i+1}** (Score: {score:.3f})")
                                st.write(f"Text A: {texts[row][:100]}...")
                                st.write(f"Text B: {texts[col][:100]}...")
                                st.write("---")
            
            except Exception as e:
                st.error(f"Fehler beim Verarbeiten der Datei: {str(e)}")
    
    with tab3:
        st.header("🎯 Custom Similarity Matrix")
        
        custom_texts = st.text_area(
            "Texte eingeben (ein Text pro Zeile):",
            height=200,
            placeholder="Text 1\nText 2\nText 3"
        )
        
        if st.button("🧮 Matrix berechnen"):
            if custom_texts:
                texts = [text.strip() for text in custom_texts.split('\n') if text.strip()]
                
                if len(texts) >= 2:
                    with st.spinner("Similarity Matrix wird berechnet..."):
                        embeddings = embedding_service.encode_texts(texts)
                        
                        if embeddings is not None:
                            if embed_dims < 768:
                                embeddings = embeddings[:, :embed_dims]
                            
                            similarity_matrix = cosine_similarity(embeddings)
                            
                            # Interaktive Heatmap
                            fig = go.Figure(data=go.Heatmap(
                                z=similarity_matrix,
                                x=[f"Text {i+1}" for i in range(len(texts))],
                                y=[f"Text {i+1}" for i in range(len(texts))],
                                colorscale='RdYlBu_r',
                                text=similarity_matrix,
                                texttemplate="%{text:.3f}",
                                textfont={"size": 10},
                                hoverongaps=False
                            ))
                            
                            fig.update_layout(
                                title="Text Similarity Matrix",
                                xaxis_title="Texte",
                                yaxis_title="Texte",
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Matrix als DataFrame anzeigen
                            df_matrix = pd.DataFrame(
                                similarity_matrix,
                                index=[f"Text {i+1}" for i in range(len(texts))],
                                columns=[f"Text {i+1}" for i in range(len(texts))]
                            )
                            st.dataframe(df_matrix.round(4))
                else:
                    st.warning("Bitte mindestens 2 Texte eingeben.")
    
    with tab4:
        st.header("💾 Export & Speichern")
        
        st.subheader("📁 Embedding Cache")
        if os.path.exists(CACHE_DIR):
            cache_files = os.listdir(CACHE_DIR)
            st.write(f"Cache Dateien: {len(cache_files)}")
            
            if st.button("🗑️ Cache leeren"):
                for file in cache_files:
                    os.remove(os.path.join(CACHE_DIR, file))
                st.success("Cache geleert!")
                st.rerun()
        
        st.subheader("💻 System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**PyTorch Version:** {torch.__version__}")
            st.write(f"**Device:** {embedding_service.device}")
            st.write(f"**CUDA verfügbar:** {torch.cuda.is_available()}")
        
        with col2:
            if torch.cuda.is_available():
                st.write(f"**GPU Name:** {torch.cuda.get_device_name()}")
                st.write(f"**GPU Memory:** {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        st.subheader("📋 Verwendungshinweise")
        with st.expander("ℹ️ Tipps zur Nutzung"):
            st.markdown("""
            **Prompt Types:**
            - `query`: Für Suchanfragen optimiert
            - `document`: Für Dokumente optimiert
            
            **Embedding Dimensionen:**
            - 768: Beste Qualität (Standard)
            - 512/256/128: Schneller, weniger Speicher
            
            **Performance Tipps:**
            - Nutze kleinere Dimensionen für große Datenmengen
            - Cache wird automatisch verwendet
            - GPU beschleunigt die Verarbeitung erheblich
            """)

if __name__ == "__main__":
    main()
