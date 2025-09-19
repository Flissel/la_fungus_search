#!/usr/bin/env python3
"""
Schneller Test für Ollama Integration
"""

import requests
import json

def test_ollama_connection():
    """Teste Ollama Verbindung"""
    print("🔌 Teste Ollama Verbindung...")
    
    try:
        # Liste verfügbare Modelle
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()["models"]
            print(f"✅ Ollama läuft! Verfügbare Modelle:")
            for model in models:
                print(f"  - {model['name']} ({model['size'] // 1024**3:.1f}GB)")
            return models[0]["name"] if models else None
        else:
            print(f"❌ Ollama nicht erreichbar: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Verbindungsfehler: {e}")
        return None

def test_chat(model_name):
    """Teste Chat mit Modell"""
    print(f"\n💬 Teste Chat mit {model_name}...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Hallo! Antworte auf Deutsch mit maximal 20 Wörtern.",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            answer = response.json()["response"]
            print(f"🤖 Antwort: {answer}")
            return True
        else:
            print(f"❌ Chat Fehler: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat Fehler: {e}")
        return False

def main():
    print("🚀 Ollama Integration Test")
    print("=" * 40)
    
    # Teste Verbindung
    model_name = test_ollama_connection()
    
    if model_name:
        # Teste Chat
        success = test_chat(model_name)
        
        if success:
            print("\n✅ Ollama Integration erfolgreich!")
            print("\n🎯 Nächste Schritte:")
            print("1. streamlit run ollama_rag.py")
            print("2. Oder: python ollama_rag.py 'Deine Frage hier' --docs example_documents.txt")
        else:
            print("\n❌ Chat-Test fehlgeschlagen")
    else:
        print("\n❌ Ollama nicht verfügbar")
        print("\n💡 Lösungen:")
        print("1. Ollama starten: 'ollama serve'")
        print("2. Modell pullen: 'ollama pull llama3.2:1b'")

if __name__ == "__main__":
    main()
