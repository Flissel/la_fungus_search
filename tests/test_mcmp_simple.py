#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMP-RAG System Tests (Windows-kompatibel)
Validiert die MCMP-Integration ohne Unicode-Probleme
"""

import sys
import os
import time
from pathlib import Path

def test_basic():
    """Grundlegende Funktions-Tests"""
    print("MCMP-RAG Basic Tests")
    print("=" * 30)
    
    # 1. Test Imports
    print("1. Testing imports...")
    try:
        import numpy as np
        import networkx as nx
        print("   [OK] Core libraries")
        
        from mcmp_rag import MCPMRetriever
        print("   [OK] MCMP module imported")
        
    except ImportError as e:
        print(f"   [ERROR] Import failed: {e}")
        return False
    
    # 2. Test MCMP Initialization  
    print("2. Testing MCMP initialization...")
    try:
        mcmp = MCPMRetriever(num_agents=10, max_iterations=5)
        print("   [OK] MCMP initialized")
    except Exception as e:
        print(f"   [ERROR] Init failed: {e}")
        return False
    
    # 3. Test Mock Documents
    print("3. Testing document handling...")
    try:
        # Mock documents with fake embeddings
        docs = ["Test doc 1", "Test doc 2", "Test doc 3"]
        
        # Since EmbeddingGemma might not be available, create mock setup
        from mcmp_rag import Document
        import numpy as np
        
        mcmp.documents = []
        for i, doc_text in enumerate(docs):
            # Create normalized random embedding
            embedding = np.random.normal(0, 1, 768)
            embedding = embedding / np.linalg.norm(embedding)
            
            doc = Document(id=i, content=doc_text, embedding=embedding)
            mcmp.documents.append(doc)
        
        print(f"   [OK] {len(mcmp.documents)} mock documents created")
        
    except Exception as e:
        print(f"   [ERROR] Document test failed: {e}")
        return False
    
    print("4. All basic tests passed!")
    return True

def test_files():
    """Test file structure"""
    print("\nFile Structure Test")
    print("=" * 20)
    
    files = [
        "mcmp_rag.py",
        "mcmp_streamlit.py", 
        "mcmp_cli.py",
        "start_mcmp.bat",
        "requirements.txt"
    ]
    
    all_good = True
    for f in files:
        if Path(f).exists():
            print(f"   [OK] {f}")
        else:
            print(f"   [MISSING] {f}")
            all_good = False
    
    return all_good

def test_streamlit():
    """Test streamlit integration"""
    print("\nStreamlit Integration Test")
    print("=" * 25)
    
    try:
        import streamlit
        print("   [OK] Streamlit available")
        
        # Test if our streamlit module imports
        from mcmp_streamlit import mcmp_page
        print("   [OK] MCMP Streamlit module")
        return True
        
    except ImportError as e:
        print(f"   [WARNING] Streamlit not available: {e}")
        return False
    except Exception as e:
        print(f"   [ERROR] Streamlit test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MCMP-RAG System Validation")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_files():
        tests_passed += 1
        
    if test_basic():
        tests_passed += 1
        
    if test_streamlit():
        tests_passed += 1
    
    # Summary
    print(f"\nResults: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n[SUCCESS] All tests passed!")
        print("MCMP-RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Run: start_mcmp.bat")
        print("2. Choose Web Interface option")
        print("3. Test with your documents")
        
    else:
        print("\n[WARNING] Some tests failed.")
        print("Check error messages above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Activate virtual environment")
        print("3. Check Python version (>=3.8)")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to continue...")
    sys.exit(0 if success else 1)
