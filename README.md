# EmbeddingGemma Desktop Setup 🔍

Ein vollständiges Desktop-Interface für Google's EmbeddingGemma-300M Modell mit GUI, MCMP-RAG, RagV1 und Agent-Chat.

## Abstract

EmbeddingGemma kombiniert lokale Embeddings mit einem Physarum-inspirierten Multi‑Agenten‑Retriever (MCMP) und optionaler Enterprise‑RAG‑Suche:

- **Embeddings + UI**: Klassische semantische Suche und MCMP‑RAG Oberfläche (`streamlit_fungus_backup.py`).
- **MCMP‑RAG**: Viele Agenten bewegen sich im Embedding‑Raum, hinterlassen Pheromonspuren, dämpfen Trails und aktualisieren fortlaufend Dokument‑Relevanzen; am Ende werden Top‑K Chunks mit optionaler Diversität zurückgegeben.
- **Code‑Space Frontend**: `streamlit_fungus_backup.py` durchsucht Python‑Repos über mehrstufige Chunks (Header: `# file: … | lines: a-b | window: w`), unterstützt Multi‑Query (LLM‑generiert, grounded) und **Dedup**.
- **Agent‑Chat & Tools**: Chat‑Agent mit Tool‑Calls (z. B. Code‑Suche, Root‑Dir setzen), Hintergrund‑Reports mit Live‑Progress und optionaler Snapshot‑GIF‑Aufzeichnung.
- **RAG**: Qdrant + LlamaIndex für persistente Indizes (Rag‑Modus im Fungus‑UI), Hybrid‑Scoring und Antwort‑Generierung.
- **API & Coding Events**: `src/embeddinggemma/fungus_api.py` inkl. Endpoint zum Bauen von Code‑Edit‑Events aus Chunk‑Headern.

Siehe auch:

- Architektur & C4: docs/ARCHITECTURE.md
- Scripts overview: docs/SCRIPTS.md
- Demo & Insights: docs/DEMO.md

## 🚀 Schnellstart

### 1. Setup ausführen

```bash
# Doppelklick auf setup.bat oder im Terminal:
setup.bat
```

### 2. Hugging Face Setup

- Bei [Hugging Face](https://huggingface.co) registrieren
- [EmbeddingGemma Lizenz](https://huggingface.co/google/embeddinggemma-300m) akzeptieren
- Falls nötig: HF Token erstellen und setzen

### 3. Anwendung starten

#### 🌐 Web-Interface (Empfohlen)

```bash
streamlit run streamlit_fungus_backup.py
```

Dann Browser öffnen: http://localhost:8501

#### 🔧 RagV1 CLI (Index & Suche)

```bash
# Index aufbauen (unter src/)
python src/embeddinggemma/rag_v1.py build --directory src

# Query gegen vorhandenen Index
python src/embeddinggemma/rag_v1.py query "Wie ist RagV1 implementiert?" --top-k 5 --alpha 0.7

# Vergleich: Fungus vs RagV1 vs Hybrid (optional)
python src/embeddinggemma/rag_v1.py compare "Erkläre MCPMRetriever" --top-k 5

# Stats laden/anzeigen
python src/embeddinggemma/rag_v1.py load --dir ./enterprise_index
python src/embeddinggemma/rag_v1.py stats
```

## 🔧 Features

### Web-Interface (streamlit_fungus_backup.py)

- Text/Code Suche über mehrstufige Chunks
- Multi-Query (LLM generiert, grounded auf eingebetteten Dateien)
- Dedup der Queries (Jaccard)
- Live-Logging, GIF-Snapshots, Agent-Chat (Tool-Calls)
- “Rag”-Sektion für Enterprise RAG (Qdrant + LlamaIndex)

### RagV1 (rag_v1.py)

- AST‑Chunking für Code
- Hybrid Retrieval (semantic + keyword)
- Qdrant VectorStore
- Optionale Generierung via HF LLM oder Ollama

## 📁 Struktur

```
EmbeddingGemma/
├── streamlit_fungus_backup.py         # Primary Frontend (MCMP + Rag + Agent)
├── src/embeddinggemma/rag_v1.py
├── src/embeddinggemma/agents/agent_fungus_rag.py
├── docs/ARCHITECTURE.md
├── docs/SCRIPTS.md
├── docs/DEMO.md
└── README.md
```

## ⚙️ Konfiguration

- OLLAMA_MODEL und OLLAMA_HOST für LLM-Aufgaben (Multi-Query, Summaries, Chat)
- GPU für Embeddings (HF) empfohlen; MCMP selbst ist CPU‑lastig

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
