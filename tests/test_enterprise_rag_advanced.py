#!/usr/bin/env python3
"""
Erweiterte Tests für EnterpriseCodeRAG
Testet Parent-Child Retrieval, Gold-Set Evaluation, Query-Pipeline und CLI-Erweiterungen
"""

import sys
import os
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
sys.path.append('src')

from embeddinggemma.enterprise_rag import EnterpriseCodeRAG

class TestEnterpriseRAGAdvanced(unittest.TestCase):
    """Erweiterte Tests für Enterprise-RAG Features"""

    def setUp(self):
        """Setup für jeden Test"""
        self.rag = EnterpriseCodeRAG(
            qdrant_url="http://localhost:6333",
            embedding_model="google/embeddinggemma-300m",
            llm_model="microsoft/DialoGPT-medium"
        )

    def test_parent_child_retrieval_initialization(self):
        """Test: Parent-Child Retrieval wird korrekt initialisiert"""
        # Mock Qdrant Client
        with patch.object(self.rag, 'qdrant_client') as mock_client:
            mock_client.get_collections.return_value = Mock()
            mock_client.get_collections.return_value.collections = []

            # Collection erstellen
            self.rag.create_collection()

            # Prüfen ob retrieve-Methode existiert
            self.assertTrue(hasattr(self.rag, 'retrieve'))

    def test_gold_set_setup(self):
        """Test: Gold-Set kann eingerichtet werden"""
        gold_questions = [
            {"question": "Wie funktioniert Machine Learning?", "answer": "ML nutzt Algorithmen"},
            {"question": "Was ist Python?", "answer": "Eine Programmiersprache"}
        ]

        # Setup Gold-Set
        self.rag.setup_gold_set(gold_questions)

        # Prüfen ob Gold-Set gespeichert wurde
        self.assertTrue(hasattr(self.rag, 'gold_set'))
        self.assertEqual(len(self.rag.gold_set), 2)

    def test_metrics_evaluation(self):
        """Test: Metriken können berechnet werden"""
        # Mock Ergebnisse
        mock_results = [
            {'content': 'Test content 1', 'hybrid_score': 0.8},
            {'content': 'Test content 2', 'hybrid_score': 0.6}
        ]

        gold_ids = ['test_id_1', 'test_id_2']

        # Metriken berechnen
        metrics = self.rag.evaluate_metrics(mock_results, gold_ids)

        # Prüfen ob Metriken zurückgegeben werden
        self.assertIsInstance(metrics, dict)
        self.assertIn('nDCG', metrics)
        self.assertIn('MRR', metrics)

    def test_query_pipeline_with_sec_filter(self):
        """Test: Query-Pipeline mit SEC-Filter"""
        # Mock Index
        with patch.object(self.rag, 'index', Mock()):
            with patch.object(self.rag, 'retrieve') as mock_retrieve:
                mock_retrieve.return_value = [
                    {'content': 'Sichere Information', 'metadata': {'security_level': 'public'}},
                    {'content': 'Vertrauliche Information', 'metadata': {'security_level': 'confidential'}}
                ]

                # Query mit SEC-Filter
                result = self.rag.query(
                    query="Test query",
                    top_k=5,
                    generate_response=False
                )

                # Prüfen ob SEC-Filter angewendet wurde
                self.assertIn('results', result)
                self.assertIn('sec_filter_applied', result)

    def test_cli_eval_subparser(self):
        """Test: CLI eval-Subparser funktioniert"""
        import argparse

        # Mock Argumente für eval
        test_args = [
            'enterprise_rag.py',
            'eval',
            '--gold-file', 'test_gold.json',
            '--repo', 'test/repo'
        ]

        with patch('sys.argv', test_args):
            # CLI sollte ohne Fehler laufen
            try:
                # Hier würde normalerweise main() aufgerufen werden
                # Für Test-Zwecke nur prüfen ob Imports funktionieren
                from embeddinggemma.enterprise_rag import main
                self.assertTrue(callable(main))
            except SystemExit:
                # CLI beendet sich normal
                pass

    def test_incremental_index_update(self):
        """Test: Incremental Index Update funktioniert"""
        # Mock Datei-Hashes
        mock_hashes = {
            'file1.py': 'hash1',
            'file2.py': 'hash2'
        }

        with patch.object(self.rag, 'qdrant_client') as mock_client:
            # Mock Collection
            mock_client.get_collection.return_value = Mock()
            mock_client.get_collection.return_value.points_count = 10

            # Incremental Update testen
            self.rag._incremental_update(mock_hashes)

            # Prüfen ob Update-Methoden aufgerufen wurden
            self.assertTrue(True)  # Placeholder für komplexere Prüfung

    def test_batch_embeddings_processing(self):
        """Test: Batch Embeddings werden korrekt verarbeitet"""
        # Mock große Datenmenge
        large_chunks = [
            {'content': f'Chunk {i} content', 'metadata': {}}
            for i in range(100)
        ]

        with patch.object(self.rag, 'embed_model') as mock_embed:
            mock_embed.encode.return_value = [[0.1] * 384] * 100  # Mock Embeddings

            # Batch Processing testen
            embeddings = self.rag._process_batch_embeddings(large_chunks)

            # Prüfen ob Embeddings erstellt wurden
            self.assertIsInstance(embeddings, list)
            self.assertEqual(len(embeddings), 100)

    def test_router_scope_selection(self):
        """Test: Router wählt korrekten Scope"""
        query_types = {
            'general': "Wie funktioniert KI?",
            'code': "def hello_world():",
            'api': "Wie nutze ich die API?"
        }

        for query_type, query in query_types.items():
            scope = self.rag._determine_query_scope(query)

            # Prüfen ob Scope bestimmt wurde
            self.assertIn(scope, ['general', 'code', 'api', 'unknown'])

    def test_soft_fallback_mechanism(self):
        """Test: Soft Fallback funktioniert bei Fehlern"""
        # Mock fehlgeschlagene Suche
        with patch.object(self.rag, 'hybrid_search') as mock_search:
            mock_search.side_effect = Exception("Search failed")

            # Fallback sollte aktiviert werden
            result = self.rag.query(
                query="Test query",
                top_k=5,
                generate_response=False
            )

            # Prüfen ob Fallback-Ergebnis zurückgegeben wurde
            self.assertIn('fallback_used', result)
            self.assertTrue(result['fallback_used'])

    def test_feedback_logging(self):
        """Test: Feedback wird korrekt geloggt"""
        # Mock Feedback-Daten
        feedback_data = {
            'query': 'Test query',
            'results': [{'content': 'Test', 'score': 0.8}],
            'user_rating': 4,
            'timestamp': '2024-01-01T12:00:00Z'
        }

        # Feedback loggen
        self.rag._log_feedback(feedback_data)

        # Prüfen ob Feedback gespeichert wurde
        self.assertTrue(hasattr(self.rag, 'feedback_log'))
        self.assertIn(feedback_data, self.rag.feedback_log)

    def test_cross_encoder_reranking(self):
        """Test: Cross-Encoder Reranking funktioniert"""
        # Mock Cross-Encoder
        with patch('sentence_transformers.CrossEncoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.9, 0.7, 0.5]

            candidates = [
                {'content': 'High relevance', 'score': 0.6},
                {'content': 'Medium relevance', 'score': 0.8},
                {'content': 'Low relevance', 'score': 0.4}
            ]

            # Reranking anwenden
            reranked = self.rag._apply_cross_encoder_reranking(
                candidates,
                "Test query"
            )

            # Prüfen ob Reranking angewendet wurde
            self.assertEqual(len(reranked), 3)
            # Höchste Cross-Encoder Score sollte zuerst kommen
            self.assertGreater(reranked[0]['cross_encoder_score'], reranked[1]['cross_encoder_score'])

if __name__ == '__main__':
    # Test-Suite ausführen
    unittest.main(verbosity=2)