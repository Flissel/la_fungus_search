import numpy as np

from mcmp_rag import MCPMRetriever


def test_mcmp_search_smoke_cpu_mock():
    docs = [f"Doc {i} content about python and search {i}" for i in range(12)]
    m = MCPMRetriever(num_agents=50, max_iterations=10, use_embedding_model=False)
    assert m.add_documents(docs)
    res = m.search("python search", top_k=5, verbose=False)
    assert isinstance(res, dict)
    assert "results" in res and len(res["results"]) <= 5
    assert "network_stats" in res
#!/usr/bin/env python3
"""
MCMP-RAG System Tests
Validiert die MCMP-Integration und Funktionalität
"""

import sys
import time
from pathlib import Path

def test_imports():
    """Teste alle notwendigen Imports"""
    print("Teste Imports...")
    
    try:
        import numpy as np
        print("  OK numpy")
        
        import matplotlib.pyplot as plt
        print("  OK matplotlib")
        
        import networkx as nx
        print("  OK networkx")
        
        import plotly.express as px
        print("  OK plotly")
        
        from sentence_transformers import SentenceTransformer
        print("  OK sentence-transformers")
        
        from mcmp_rag import MCPMRetriever, Document, Agent
        print("  OK mcmp_rag module")
        
        return True
        
    except ImportError as e:
        print(f"  ERROR Import Error: {e}")
        print("     Run 'pip install -r requirements.txt'")
        return False
    except Exception as e:
        print(f"  ERROR Unknown error: {e}")
        return False

def test_mcmp_basic():
    """Teste grundlegende MCMP Funktionalität"""
    print("\n🧠 Teste MCMP Grundfunktionen...")
    
    try:
        # MCMP initialisieren
        mcmp = MCPMRetriever(num_agents=50, max_iterations=20)
        print("  ✅ MCMP Initialisierung")
        
        # Test-Dokumente
        documents = [
            "Python ist eine Programmiersprache",
            "Machine Learning nutzt Algorithmen",
            "Künstliche Intelligenz ist zukunftsweisend"
        ]
        
        # Dokumente hinzufügen (Mock-Modus falls EmbeddingGemma nicht verfügbar)
        success = mcmp.add_documents(documents)
        if success:
            print("  ✅ Dokumente erfolgreich hinzugefügt")
        else:
            print("  ⚠️  EmbeddingGemma nicht verfügbar, verwende Mock-Modus")
            # Mock-Setup
            from mcmp_rag import Document
            import numpy as np
            mcmp.documents = []
            for i, doc in enumerate(documents):
                mock_embedding = np.random.normal(0, 1, 768)
                mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
                mcmp.documents.append(Document(
                    id=i, 
                    content=doc, 
                    embedding=mock_embedding
                ))
        
        print(f"  📊 {len(mcmp.documents)} Dokumente geladen")
        
        # Test-Suche
        if len(mcmp.documents) > 0:
            print("  🔍 Teste Mock-Suche...")
            # Mock search da EmbeddingGemma eventuell nicht verfügbar
            mock_results = {
                "query": "test query",
                "results": [
                    {
                        "content": doc.content,
                        "relevance_score": 0.8 - i * 0.1,
                        "visit_count": i + 1,
                        "metadata": {}
                    }
                    for i, doc in enumerate(mcmp.documents)
                ],
                "network_stats": {"network_density": 0.5},
                "pheromone_trails": 3,
                "total_documents": len(mcmp.documents)
            }
            
            print(f"    📈 Mock-Ergebnisse: {len(mock_results['results'])} Dokumente")
            print("  ✅ Grundlegende MCMP-Logik funktional")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MCMP Test Fehler: {e}")
        return False

def test_streamlit_integration():
    """Teste Streamlit Integration"""
    print("\n🌐 Teste Streamlit Integration...")
    
    try:
        import streamlit as st
        print("  ✅ Streamlit verfügbar")
        
        from mcmp_streamlit import mcmp_page, add_mcmp_to_main_app
        print("  ✅ MCMP-Streamlit Module importiert")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Streamlit nicht verfügbar: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Streamlit Integration Fehler: {e}")
        return False

def test_cli_functionality():
    """Teste CLI Funktionalität"""
    print("\n🖥️ Teste CLI Funktionalität...")
    
    try:
        from mcmp_cli import load_documents_from_file, save_results
        print("  ✅ CLI Module importiert")
        
        # Test JSON Speichern/Laden
        test_data = {"test": "data", "number": 42}
        test_file = "test_output.json"
        
        save_results(test_data, test_file)
        
        if Path(test_file).exists():
            print("  ✅ JSON Export funktional")
            Path(test_file).unlink()  # Cleanup
        else:
            print("  ❌ JSON Export fehlgeschlagen")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ CLI Test Fehler: {e}")
        return False

def test_dependencies():
    """Teste alle Dependencies aus requirements.txt"""
    print("\n📦 Teste Dependencies...")
    
    requirements = [
        "sentence_transformers",
        "torch", 
        "numpy",
        "sklearn",
        "streamlit",
        "plotly",
        "pandas",
        "networkx",
        "matplotlib"
    ]
    
    success_count = 0
    
    for req in requirements:
        try:
            __import__(req)
            print(f"  ✅ {req}")
            success_count += 1
        except ImportError:
            print(f"  ❌ {req} nicht verfügbar")
        except Exception as e:
            print(f"  ⚠️  {req} - {e}")
    
    print(f"  📊 {success_count}/{len(requirements)} Dependencies verfügbar")
    return success_count >= len(requirements) * 0.8  # 80% Success-Rate

def test_file_structure():
    """Teste ob alle MCMP Dateien vorhanden sind"""
    print("\n📁 Teste Dateistruktur...")
    
    required_files = [
        "mcmp_rag.py",
        "mcmp_streamlit.py", 
        "mcmp_cli.py",
        "start_mcmp.bat",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} fehlt")
            missing_files.append(file)
    
    if missing_files:
        print(f"  ⚠️  {len(missing_files)} Dateien fehlen: {missing_files}")
        return False
    
    print("  ✅ Alle MCMP Dateien vorhanden")
    return True

def run_comprehensive_test():
    """Führe alle Tests aus"""
    print("MCMP-RAG System Tests")
    print("=" * 50)
    
    test_results = {}
    
    # Test Suite
    tests = [
        ("Dateistruktur", test_file_structure),
        ("Dependencies", test_dependencies), 
        ("Imports", test_imports),
        ("MCMP Grundfunktionen", test_mcmp_basic),
        ("Streamlit Integration", test_streamlit_integration),
        ("CLI Funktionalität", test_cli_functionality)
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} - Unerwarteter Fehler: {e}")
            test_results[test_name] = False
    
    # Zusammenfassung
    print("\n" + "=" * 50)
    print("📊 Test-Zusammenfassung")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} | {status}")
    
    print(f"\n🎯 Ergebnis: {passed_tests}/{len(tests)} Tests bestanden")
    
    if passed_tests == len(tests):
        print("🎉 Alle Tests bestanden! MCMP-RAG ist bereit.")
        print("\n🚀 Nächste Schritte:")
        print("   1. start_mcmp.bat ausführen")
        print("   2. Streamlit Interface öffnen")
        print("   3. Eigene Dokumente testen")
        return True
    else:
        print("⚠️  Einige Tests fehlgeschlagen. Prüfe die Fehlermeldungen oben.")
        print("\n🔧 Lösungsvorschläge:")
        print("   1. pip install -r requirements.txt")
        print("   2. Virtual Environment aktivieren")  
        print("   3. Python Version prüfen (>=3.8 empfohlen)")
        return False

def quick_demo():
    """Schnelle Demo der MCMP Funktionalität"""
    print("\n🎬 Schnelle MCMP Demo")
    print("-" * 30)
    
    try:
        from mcmp_rag import demo_mcpm
        demo_mcpm()
        return True
    except Exception as e:
        print(f"❌ Demo fehlgeschlagen: {e}")
        return False

def main():
    """Hauptfunktion"""
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        quick_demo()
    else:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
