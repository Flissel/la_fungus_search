# EmbeddingGemma Desktop Setup 🔍

Ein vollständiges Desktop-Interface für Google's EmbeddingGemma-300M Modell mit GUI, MCMP-RAG, RagV1 und Agent-Chat.

## Abstract

EmbeddingGemma kombiniert lokale Embeddings mit einem Physarum-inspirierten Multi‑Agenten‑Retriever (MCMP) und optionaler Enterprise‑RAG‑Suche:
- **Embeddings + UI**: Klassische semantische Suche und MCMP‑RAG Oberfläche (`streamlit_fungus_backup.py`).
- **MCMP‑RAG**: Viele Agenten bewegen sich im Embedding‑Raum, hinterlassen Pheromonspuren, dämpfen Trails und aktualisieren fortlaufend Dokument‑Relevanzen; am Ende werden Top‑K Chunks mit optionaler Diversität zurückgegeben.
- **Code‑Space Frontend**: `streamlit_fungus_backup.py` durchsucht Python‑Repos über mehrstufige Chunks (Header: `# file: … | lines: a-b | window: w`), unterstützt Multi‑Query (LLM‑generiert, grounded) und **Dedup**.
- **Agent‑Chat & Tools**: Chat‑Agent mit Tool‑Calls (z. B. Code‑Suche, Root‑Dir setzen), Hintergrund‑Reports mit Live‑Progress und optionaler Snapshot‑GIF‑Aufzeichnung.
- **Enterprise‑RAG**: Qdrant + LlamaIndex für persistente Indizes (Rag‑Modus im Fungus‑UI), Hybrid‑Scoring und Antwort‑Generierung.
- **Realtime API Server**: `src/embeddinggemma/realtime/server.py` FastAPI WebSocket server für Live-MCMP-Simulation mit React Frontend.

Siehe auch:
- Architektur & C4: docs/ARCHITECTURE.md
- Scripts overview: docs/SCRIPTS.md
- Simulation Details: docs/mcmp_simulation.md
- API Reference: docs/API_REFERENCE.md
- Config Reference: docs/CONFIG_REFERENCE.md

## 🚀 Schnellstart

### 1. Setup ausführen
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Editable install
pip install -e .
```

### 2. Hugging Face Setup
- Bei [Hugging Face](https://huggingface.co) registrieren
- [EmbeddingGemma Lizenz](https://huggingface.co/google/embeddinggemma-300m) akzeptieren
- Falls nötig: HF Token erstellen und setzen

### 3. Anwendung starten

#### 🌐 Streamlit Interface (Empfohlen)
```bash
# Direct run
streamlit run streamlit_fungus_backup.py

# Or using helper scripts
.\run-streamlit.ps1    # PowerShell
run-streamlit.cmd      # Windows CMD
```
Dann Browser öffnen: http://localhost:8501

#### 🚀 Realtime WebSocket Server + React Frontend
```bash
# 1. Start FastAPI server
uvicorn src.embeddinggemma.realtime.server:app --reload --port 8011

# 2. Start React frontend (separate terminal)
cd frontend
npm install  # first time only
npm run dev
```
- Server: http://localhost:8011  
- React Frontend: http://localhost:5173

#### 🛠️ RAG CLI (verfügbar in `experimerntal/old/`)
```bash
# Index aufbauen 
python experimerntal/old/rag_v1.py build --directory src

# Query gegen Index
python experimerntal/old/rag_v1.py query "RAG implementation details" --top-k 5
```

## 🔧 Features

### Streamlit Interface (streamlit_fungus_backup.py)
- Text/Code Suche über mehrstufige Chunks
- Multi-Query (LLM generiert, grounded auf eingebetteten Dateien)
- Dedup der Queries (Jaccard)
- Live-Logging, GIF-Snapshots, Agent-Chat (Tool-Calls)
- "Rag"-Sektion für Enterprise RAG (Qdrant + LlamaIndex)

### Realtime WebSocket Server (realtime/server.py)
- FastAPI mit WebSocket Live-Updates
- Pause/Resume/Reset Simulation
- Agents dynamisch hinzufügen/anpassen 
- Live Top-K Ergebnisse und Visualisierung
- Background Reports mit LLM Integration
- REST API für alle Konfigurationen

### RAG Module (rag/ directory)
- AST‑Chunking für Code
- Hybrid Retrieval (semantic + keyword)
- Qdrant VectorStore Integration
- Generierung via HF LLM oder Ollama

## 📁 Struktur
```
EmbeddingGemma/
├── streamlit_fungus_backup.py         # Streamlit Frontend (MCMP + RAG + Agent Chat)
├── src/embeddinggemma/
│   ├── realtime/server.py             # FastAPI WebSocket Server
│   ├── mcmp_rag.py                   # Core MCMP Retriever
│   ├── mcmp/                         # Simulation, embeddings, PCA
│   ├── rag/                          # RAG components (chunking, vectorstore)
│   ├── ui/                           # UI components for Streamlit
│   └── agents/agent_fungus_rag.py    # Agent with tool calls
├── frontend/                         # React frontend for realtime server
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SCRIPTS.md
│   ├── mcmp_simulation.md
│   ├── CONFIG_REFERENCE.md
│   └── API_REFERENCE.md
└── README.md
```

## ⚙️ Konfiguration

### Umgebungsvariablen
- `OLLAMA_MODEL`: LLM Model (default: qwen2.5-coder:7b)
- `OLLAMA_HOST`: Ollama Server URL (default: http://127.0.0.1:11434)
- `HF_TOKEN`: Hugging Face Token (falls benötigt)

### Hardware
- GPU für Embeddings (HF) empfohlen; MCMP Simulation läuft auf CPU
- Minimum 8GB RAM für größere Korpora

### Ports
- Streamlit: 8501 (default)
- Realtime Server: 8011 (empfohlen)
- Frontend Dev: 5173 (Vite)
- Ollama: 11434 (default)

## 🧠 MCMP-RAG: Schleimpilz-inspirierte Suche

MCMP erkundet den Dokumentenraum adaptiv, verstärkt Pfade durch Pheromone und bildet ein Netzwerk relevanter Verbindungen (multi‑hop). Das ermöglicht Abhängigkeits‑Tracing über Dateien hinweg – ergänzend zur klassischen NN‑Suche.

### Parameter (Beispiele)
- num_agents: 300
- max_iterations: 80
- pheromone_decay: 0.95
- exploration_bonus: 0.1

## 🔗 Links
- [EmbeddingGemma Model](https://huggingface.co/google/embeddinggemma-300m)
- [Polyphorm (Inspiration)](https://github.com/CreativeCodingLab/Polyphorm)
- [LlamaIndex](https://www.llamaindex.ai/)
- [Qdrant](https://qdrant.tech/)

