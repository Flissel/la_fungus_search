# EmbeddingGemma Desktop Setup 🔍

Ein vollständiges Desktop-Interface für Google's EmbeddingGemma-300M Modell mit GUI, CLI und RAG-System.

## Abstract

EmbeddingGemma kombiniert lokale Embeddings mit einem Physarum-inspirierten Multi‑Agenten‑Retriever (MCMP) und optionaler Enterprise‑RAG‑Suche:
- **Embeddings + UI**: Klassische semantische Suche (Streamlit `app.py`) und MCMP‑RAG Oberfläche (`mcmp_streamlit.py`, `streamlit_fungus.py`).
- **MCMP‑RAG**: Viele Agenten bewegen sich im Embedding‑Raum, hinterlassen Pheromonspuren, dämpfen Trails und aktualisieren fortlaufend Dokument‑Relevanzen; am Ende werden Top‑K Chunks mit optionaler Diversität zurückgegeben.
- **Code‑Space Frontend**: `streamlit_fungus.py` durchsucht Python‑Repos über mehrstufige Chunks (Header: `# file: … | lines: a-b | window: w`), unterstützt Multi‑Query, Auto‑Generierung (LLM, grounded auf eingebetteten Dateien) und **Dedup** der Queries.
- **Agent‑Chat & Tools**: Chat‑Agent mit Tool‑Calls (z. B. Code‑Suche, Root‑Dir setzen), Hintergrund‑Reports mit Live‑Progress und optionaler Snapshot‑GIF‑Aufzeichnung.
- **Enterprise‑RAG**: Qdrant + LlamaIndex für persistente Indizes (Rag‑Modus im Fungus‑UI), Hybrid‑Scoring und Antwort‑Generierung.
- **API & Coding Events**: `src/embeddinggemma/fungus_api.py` inkl. Endpoint zum Bauen von Code‑Edit‑Events aus Chunk‑Headern; optionales Publizieren in eine Queue.

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
streamlit run app.py
```
Dann Browser öffnen: http://localhost:8501

#### 🔍 Kommandozeile
```bash
# Einfache Suche
python cli.py search "Wie funktioniert KI?" --docs "Machine Learning Tutorial" "Python Guide" "AI Basics"

# Similarity Matrix
python cli.py similarity "Text 1" "Text 2" "Text 3"

# Aus Datei suchen
python cli.py file-search "Deine Frage" documents.txt
```

#### 📚 RAG System
```bash
# Demo starten
python rag.py demo

# Interaktive Suche
python rag.py interactive

# Dokumente hinzufügen und speichern
python rag.py add --file documents.txt --save meine_wissensbasis
```

## 🔧 Features

### Web-Interface (app.py)
- **Text Suche**: Query vs. Dokumente Similarity
- **Batch Analyse**: CSV-Dateien verarbeiten
- **Similarity Matrix**: Interaktive Heatmaps
- **Export Funktionen**: Cache-Management
- **Responsives Design**: Plotly Visualisierungen

### CLI Tool (cli.py)
- **Flexible Suche**: Ein-Zeilen-Kommandos
- **File-Support**: Textdateien durchsuchbar
- **JSON Export**: Strukturierte Ausgabe
- **Batch Processing**: Mehrere Texte gleichzeitig

### RAG System (rag.py)
- **Wissensbasis**: Dokumente persistent speichern
- **FAISS Integration**: Schnelle Vektorsuche
- **Interactive Mode**: Terminal-Chat
- **Metadata Support**: Erweiterte Dokumentinfos

## 📁 Struktur

```
EmbeddingGemma/
├── app.py              # Streamlit Web-Interface
├── cli.py              # Kommandozeilen-Tool
├── rag.py              # RAG System
├── requirements.txt    # Python Dependencies
├── setup.bat          # Windows Setup-Script
├── embedding_cache/   # Model Cache
├── rag_cache/         # RAG Wissensbasis Cache
└── README.md          # Diese Datei
```

## 🎯 Anwendungsfälle

### 📚 Dokumentensuche
Durchsuche große Textsammlungen semantisch:
```bash
python rag.py add --file meine_dokumente.txt --save firmen_kb
python rag.py search "Wie implementiere ich Feature X?" --load firmen_kb
```

### 🔍 Content Discovery
Finde ähnliche Inhalte in deiner Sammlung:
```bash
python cli.py similarity "Artikel 1" "Artikel 2" "Artikel 3" --output similarity.json
```

### 🤖 AI-Pipeline Integration
Nutze als Embedding-Service für größere Systeme:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("google/embeddinggemma-300m")
embeddings = model.encode(["Dein Text"])
```

## ⚙️ Konfiguration

### Embedding-Dimensionen
- **768**: Beste Qualität (Standard)
- **512**: Guter Kompromiss
- **256**: Schnell, weniger Speicher
- **128**: Sehr schnell, Mobile-optimiert

### Performance-Tipps
- **GPU**: Automatische CUDA-Nutzung falls verfügbar
- **Batch-Size**: Bei großen Datenmengen anpassen
- **Cache**: Wird automatisch verwendet
- **Dimensionen**: Für Speed/Storage-Tradeoff reduzieren

## 🐛 Troubleshooting

### Model Download Fehler
```bash
# HF Token setzen (falls nötig)
huggingface-cli login
```

### CUDA Fehler
```bash
# CPU-only Installation
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Import Fehler
```bash
# Dependencies neu installieren
pip install -r requirements.txt --force-reinstall
```

## 🔗 Links

- [EmbeddingGemma Model](https://huggingface.co/google/embeddinggemma-300m)
- [Google AI Blog](https://developers.googleblog.com/en/introducing-embeddinggemma/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)

## 📊 Model Info

- **Parameter**: 308M (100M Model + 200M Embedding)
- **Sprachen**: 100+ Sprachen
- **Context**: 2048 Token
- **Performance**: MTEB #1 unter 500M Parametern
- **Lizenz**: Gemma (kommerziell nutzbar)

## 🏃‍♂️ Quick Examples

### Web-Interface verwenden
1. `streamlit run app.py`
2. Text in "Suchanfrage" eingeben
3. Dokumente in "Dokumente" Bereich (eine Zeile = ein Dokument)
4. "Suche starten" klicken

### CLI für schnelle Tests
```bash
# Schnelle Similarity-Prüfung
python cli.py search "Python Tutorial" --docs "Learn Python Programming" "JavaScript Guide" "Java Basics"

# Output als JSON
python cli.py search "AI Tutorial" --docs "ML Guide" "DL Basics" --output results.json
```

### RAG für Projekte
```bash
# Wissensbasis aufbauen
echo "Python ist eine Programmiersprache" > docs.txt
echo "Machine Learning nutzt Algorithmen" >> docs.txt
python rag.py add --file docs.txt --save projekt_kb

# Suchen
python rag.py search "Was ist Python?" --load projekt_kb
```

Viel Spaß beim Experimentieren mit EmbeddingGemma! 🚀


## 🧠 MCMP-RAG: Schleimpilz-inspirierte Suche

**Neu!** Monte Carlo Physarum Machine (MCPM) für revolutionäre Dokumentenexploration, inspiriert vom Foraging-Verhalten des Schleimpilz *Physarum polycephalum*.

### 🌟 Was macht MCMP-RAG besonders?

**Statt statischer Vektorsuche:**
- **Adaptive Agenten** erkunden den Dokumentenraum dynamisch
- **Pheromonspur-basierte Pfadfindung** zwischen verwandten Dokumenten
- **Emergente Netzwerk-Strukturen** decken versteckte Verbindungen auf
- **Multi-hop Reasoning** über mehrere Dokumente hinweg

### 🚀 MCMP-RAG verwenden

#### 🌐 Web-Interface
```bash
# MCMP-RAG mit Streamlit starten
streamlit run mcmp_streamlit.py

# Oder über Batch-Script
start_mcmp.bat
```

#### 🖥️ Kommandozeile
```bash
# Schnelle Suche
python mcmp_cli.py search "Wie funktioniert KI?" --docs "AI Tutorial" "ML Guide" "DL Basics"

# Interaktiver Modus
python mcmp_cli.py interactive

# Benchmark verschiedener Konfigurationen
python mcmp_cli.py benchmark --file documents.txt --queries "Query 1" "Query 2"

# Demo ausführen
python mcmp_cli.py demo
```

#### 🧪 Python API
```python
from mcmp_rag import MCPMRetriever

# MCMP System initialisieren
mcmp = MCPMRetriever(num_agents=500, max_iterations=100)

# Dokumente hinzufügen
documents = ["Doc 1", "Doc 2", "Doc 3"]
mcmp.add_documents(documents)

# Suche mit Schleimpilz-Algorithmus
results = mcmp.search("Meine Frage", top_k=5)

# Ergebnisse analysieren
for result in results['results']:
    print(f"[{result['relevance_score']:.3f}] {result['content']}")

# Netzwerk visualisieren
mcmp.visualize_search_process("search_analysis.png")
```

### 🔧 MCMP Parameter

| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| `num_agents` | 300 | Anzahl MCPM-Agenten |
| `max_iterations` | 80 | Maximale Suchiterationen |
| `pheromone_decay` | 0.95 | Pheromonspur-Abklingrate |
| `exploration_bonus` | 0.1 | Exploration vs. Exploitation |

### 📊 MCMP vs. Standard RAG

| Feature | Standard RAG | MCMP-RAG |
|---------|-------------|----------|
| **Suchstrategie** | Statische Vektorsuche | Adaptive Agenten-Exploration |
| **Dokumentverbindungen** | Keine | Emergente Netzwerke |
| **Pfadfindung** | Single-hop | Multi-hop Reasoning |
| **Lernfähigkeit** | Statisch | Dynamische Anpassung |
| **Überraschungsfaktor** | Gering | Hoch (Serendipity) |

### 🎯 Anwendungsfälle

#### 📚 **Akademische Forschung**
- Literaturrecherche mit versteckten Verbindungen
- Cross-Domain Wissenstransfer
- Interdisziplinäre Konzeptentdeckung

#### 🏢 **Unternehmens-Wissensbasen**
- Komplexe Policy-Suche
- Innovative Lösungsansätze
- Team-übergreifende Expertise-Findung

#### 🔍 **Content Discovery**
- Unerwartete thematische Verbindungen
- Content-Gap-Analyse
- Trend-Vorhersage durch Netzwerk-Analyse

### 📈 Performance-Tipps

```python
# Für große Dokumentensammlungen (>1000 Docs)
mcmp = MCPMRetriever(
    num_agents=1000,       # Mehr Agenten für bessere Abdeckung
    max_iterations=150,    # Längere Exploration
    exploration_bonus=0.2  # Mehr Zufälligkeit
)

# Für schnelle Prototyping
mcmp = MCPMRetriever(
    num_agents=100,        # Weniger Agenten
    max_iterations=50,     # Kürzere Suche
    exploration_bonus=0.05 # Fokussierte Suche
)
```

### 🧬 Algorithmus-Details

MCPM basiert auf dem Foraging-Verhalten von *Physarum polycephalum*:

1. **Agent Spawning**: Agenten starten um Query-Embedding
2. **Attraction Forces**: Bewegung zu relevanten Dokumenten
3. **Pheromone Deposition**: Erfolgreiche Pfade werden markiert
4. **Trail Following**: Andere Agenten folgen starken Spuren
5. **Network Emergence**: Stabile Verbindungsstrukturen entstehen
6. **Relevance Extraction**: Finale Dokumenten-Rankings

### 📖 Inspiration & Referenzen

- **Polyphorm Paper**: [Polyphorm: Structural Analysis of Cosmological Datasets](https://arxiv.org/abs/2009.02441)
- **Original Repository**: [CreativeCodingLab/Polyphorm](https://github.com/CreativeCodingLab/Polyphorm)
- **Physarum-Algorithmen**: Jones, Jeff (2010). "Characteristics of pattern formation and evolution in approximations of Physarum transport networks"

---

## ✅ TODO: Coding Events & Queue Integration

Diese Sektion markiert anstehende Arbeiten rund um Code-Edit-Events, damit der Haupt‑Agent (Producer) eine Redis‑Queue speisen kann und ein Coder‑Agent (Consumer) gezielte Edits ausführt.

- [ ] API vervollständigen: Code‑Edit‑Event bauen und optional veröffentlichen
  - Endpoint (bereits vorhanden): `POST /api/edit/build_event` in `src/embeddinggemma/fungus_api.py`
  - Zweck: Aus einem RAG‑Chunk‑Header `# file: <pfad> | lines: a-b | window: w` exakte Dateipfade und Zeilengrenzen extrahieren, optional via AST auf eine Methode einschränken, und ein neutrales Edit‑Event erzeugen.
- [ ] Redis‑Integration finalisieren
  - Env: `REDIS_URL`, `CODER_EVENTS_KEY` (Standard: `coder:events`)
  - Zweck: Bei `publish=true` das Event als JSON in die Queue pushen, damit der Coder‑Agent es konsumieren kann.
- [ ] Event‑Schema konsolidieren (Platzhalter – Werte später befüllen)
  ```json
  {
    "id": "<uuid>",
    "type": "code_edit_request",
    "file_path": "<pfad.py>",
    "start_line": <int>,
    "end_line": <int>,
    "instructions": "<was zu ändern ist>",
    "before": "<optional: aktueller Code>",
    "bounds_source": "<chunk_header|explicit>",
    "chunk_header": "# file: <pfad> | lines: <a>-<b> | window: <w>",
    "prefer_method": "<optional: funktionsname>",
    "meta": { "query": "<ursprungsfrage>", "score": <float>, "source": "<rag|mcmp|manual>" },
    "routing": { "return_queue": "coder:results", "notify_webhook": "<optional_url>" },
    "control": { "priority": "<low|normal|high>", "timeout_s": <int>, "dry_run": <true|false> },
    "correlation_id": "<trace/run id>"
  }
  ```
- [ ] Client‑Snippet (Beispiel):
  ```bash
  curl -X POST http://localhost:8055/api/edit/build_event \
    -H "Content-Type: application/json" \
    -d '{
      "chunk_text": "# file: src\\embeddinggemma\\rag.py | lines: 1-277 | window: 800\n...",
      "prefer_method": "search",
      "instructions": "Refaktor: Benenne Variable x in query_text um.",
      "publish": true,
      "extra": {"issue_id": 123}
    }'
  ```

Hinweis: Separator‑Linien (`-----`) können in Datendateien für Lesbarkeit genutzt werden. Für das eigentliche Embedding oder die Edit‑Grenzen werden die Header‑Metadaten bevorzugt, weil sie exakte Datei‑ und Zeilenbereiche liefern.

