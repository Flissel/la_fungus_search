### Architecture (C4 diagrams)

#### C4: System Context (Level 1)
```mermaid
graph LR
  user[User] --> UI[React Frontend<br/>frontend/src/ui/App.tsx]
  UI -->|WebSocket| API[FastAPI Backend<br/>src/embeddinggemma/realtime/server.py]
  API -->|MCMP simulation| MCMP[MCPMRetriever<br/>src/embeddinggemma/mcmp_rag.py]
  MCMP -->|Document indexing| CORPUS[Corpus Builder<br/>src/embeddinggemma/ui/corpus.py]
  MCMP -->|LLM prompts| PROMPTS[Prompt Templates<br/>src/embeddinggemma/llm/prompts.py]
  API -->|Document search| SEARCH[Search Engine<br/>src/embeddinggemma/mcmp/*.py]
  API -->|LLM queries| LLM[LLM Providers<br/>src/embeddinggemma/llm/]
  API -->|Persistence| CACHE[.fungus_cache]
  API -->|Vector storage| QDRANT[Qdrant]
```

Key points:
- React + TypeScript frontend with Vite build system and WebSocket real-time updates
- FastAPI backend serving the React app and providing WebSocket API
- MCMP (Multi-agent Codebase Pattern Matching) simulation for intelligent code exploration
- Modular LLM integration supporting Ollama, OpenAI, Google, and Grok
- Corpus chunking and indexing with AST-aware parsing for Python files

#### C4: Container (Level 2)
```mermaid
graph TB
  subgraph Frontend [React Frontend]
    UI[React UI<br/>Plotly visualizations, WebSocket]
    WS[WebSocket Client<br/>Real-time updates]
    SETTINGS[Settings Manager<br/>Configuration persistence]
  end

  subgraph Backend [FastAPI Backend]
    API[REST API<br/>/start, /config, /search]
    SIM[Simulation Engine<br/>MCMP loop, agents, pheromones]
    CORPUS[Corpus Manager<br/>File scanning, chunking, caching]
    SEARCH[Search Engine<br/>Document retrieval, ranking]
    REPORTS[Report Generator<br/>LLM-powered analysis]
  end

  subgraph DataStores
    QDRANT[Qdrant<br/>Vector storage]
    CACHE[.fungus_cache<br/>Settings, reports, chunks]
    MODELS[Embedding Models<br/>SentenceTransformers, OpenAI]
  end

  subgraph LLMs
    OLLAMA[Ollama Server<br/>Local LLM inference]
    OPENAI[OpenAI API<br/>GPT models]
    GOOGLE[Google API<br/>Gemini models]
    GROK[Grok API<br/>x.ai models]
  end

  UI --> WS --> API
  API --> SIM --> CORPUS
  API --> SEARCH --> QDRANT
  API --> REPORTS --> CACHE
  SIM --> MODELS
  REPORTS --> OLLAMA
  REPORTS --> OPENAI
  REPORTS --> GOOGLE
  REPORTS --> GROK
```

#### C4: Component (Level 3) – Frontend internals
```mermaid
graph LR
  App[React App<br/>frontend/src/ui/App.tsx] --> Sidebar[Settings Sidebar<br/>Configuration controls]
  App --> Plot[Plotly Visualization<br/>Real-time pheromone network]
  App --> Log[Live Log Display<br/>Console output streaming]
  App --> Results[Results Panel<br/>Document snippets, metadata]
  App --> Metrics[Metrics Dashboard<br/>Performance charts]

  Sidebar --> Config[Configuration Manager<br/>Settings persistence, validation]
  Plot --> WS[WebSocket Client<br/>Real-time data streaming]
  WS --> Backend[FastAPI Backend<br/>src/embeddinggemma/realtime/server.py]
```

#### C4: Component (Level 3) – Backend internals
```mermaid
graph LR
  Server[FastAPI Server<br/>src/embeddinggemma/realtime/server.py] --> Streamer[SnapshotStreamer<br/>MCMP simulation orchestrator]
  Server --> API[REST Endpoints<br/>/start, /config, /search, /status]

  Streamer --> Retriever[MCPMRetriever<br/>src/embeddinggemma/mcmp_rag.py]
  Streamer --> Corpus[Corpus Manager<br/>src/embeddinggemma/ui/corpus.py]
  Streamer --> Prompts[Prompt Builder<br/>src/embeddinggemma/llm/prompts.py]
  Streamer --> LLM[LLM Dispatcher<br/>src/embeddinggemma/llm/dispatcher.py]

  Retriever --> Agents[Agent Simulation<br/>src/embeddinggemma/mcmp/simulation.py]
  Retriever --> Embeddings[Embedding Models<br/>src/embeddinggemma/mcmp/embeddings.py]
  Retriever --> PCA[Projection Engine<br/>src/embeddinggemma/mcmp/pca.py]
  Retriever --> Visualize[Snapshot Builder<br/>src/embeddinggemma/mcmp/visualize.py]
```

Key architectural principles:
- **Real-time WebSocket streaming** for live simulation updates
- **Modular LLM integration** supporting multiple providers (Ollama, OpenAI, Google, Grok)
- **AST-aware chunking** for intelligent code parsing and indexing
- **Multi-agent simulation** with pheromone-based exploration
- **Contextual steering** using LLM-powered relevance assessment

Performance considerations:
- Simulation and chunking are CPU-intensive operations
- Embedding models can leverage GPU acceleration when available
- WebSocket connections enable efficient real-time updates
- Corpus caching reduces redundant processing

Technology stack:
- **Frontend**: React 18, TypeScript, Vite, Plotly.js, Tailwind CSS
- **Backend**: FastAPI, Pydantic, WebSocket support
- **ML/AI**: SentenceTransformers, NumPy, scikit-learn, FAISS
- **External APIs**: Ollama, OpenAI, Google AI, Grok/x.ai
