### Architecture (C4 diagrams)

#### C4: System Context (Level 1)
```mermaid
graph LR
  user[User] --> UI[Streamlit Frontend<br/>streamlit_fungus_backup.py]
  user --> WS[Realtime WebSocket Server<br/>realtime/server.py]
  user --> Frontend[React Frontend<br/>frontend/]
  
  UI -->|MCMP search| MCMP[MCPMRetriever<br/>mcmp_rag.py]
  UI -->|Rag| RAG[RAG Module<br/>src/embeddinggemma/rag/]
  
  WS -->|Live MCMP| MCMP
  WS -->|WebSocket| Frontend
  
  RAG --> Qdrant[Qdrant]
  RAG --> HF[HF Embeddings]
  UI -->|Agent chat| LLM[LLM via Ollama or HF]
  WS -->|Reports| LLM
  UI -->|Background report| Cache[.fungus_cache]
  WS -->|Settings| Cache
```

Key points:
- Dual frontend: Streamlit UI (comprehensive) and React/WebSocket (realtime visualization)
- Realtime server provides live simulation updates, pause/resume, agent manipulation
- MCMP runs in-process; background reports use thread pool (Streamlit) or async (realtime server)
- RAG module supports both Qdrant persistence and in-memory operations
- LLMs via Ollama by default; HF LLM fallback where needed

#### C4: Container (Level 2)
```mermaid
graph TB
  subgraph StreamlitFE [Streamlit Frontend]
    UI[UI + Controls<br/>multi-query, dedup, logging]
    AG[Agent Chat<br/>LangChain / LangGraph]
    CORP[Corpus Builder<br/>chunkers / cache]
    BG[Background Reports<br/>ThreadPoolExecutor]
  end

  subgraph RealtimeServer [Realtime WebSocket Server]
    WS[WebSocket Handler<br/>live updates]
    API[REST API<br/>start/stop/config/search]
    STREAM[SnapshotStreamer<br/>async simulation loop]
  end

  subgraph ReactFE [React Frontend]
    VIZ[Live Visualization<br/>Plotly 2D/3D]
    CONTROLS[Controls<br/>pause/resume/agents]
  end

  subgraph Retrieval
    MCMP[MCPMRetriever<br/>simulation, scoring]
    RAG[RAG Module<br/>AST chunking, hybrid retrieval]
  end

  subgraph DataStores
    Q[Qdrant]
    C[.fungus_cache<br/>chunks / reports / gifs / settings]
  end

  subgraph LLMs
    OLL[Ollama Server]
    HFL[HF LLM fallback]
  end

  UI --> CORP --> MCMP
  UI --> RAG --> Q
  AG --> OLL
  RAG --> HFL
  UI --> BG --> C
  
  WS --> VIZ
  API --> STREAM --> MCMP
  STREAM --> OLL
  API --> C
  CONTROLS --> API
```

#### C4: Component (Level 3) – Frontend internals

**Streamlit Frontend Flow:**
```mermaid
graph LR
  Settings[Sidebar Settings] --> Corpus[Corpus Builder]
  Settings --> MQ[Multi-Query Generator<br/>Ollama]
  MQ --> Dedup[Dedup<br/>Jaccard]
  Corpus --> Runner[MCMP Runner<br/>sharded, logs, GIFs]
  Runner --> Raw[Raw Results<br/>snippets]
  Raw --> Summary[Agent Summary<br/>LLM]
  Settings --> Rag[Rag Section<br/>RAG Module]
  Rag --> Qdrant[Qdrant]
  Chat[Agent Chat] --> Tools[Tools<br/>search_code / get_settings / set_root_dir]
  Chat --> LLM[Ollama / LC LLM]
```

**Realtime Server Components:**
```mermaid
graph LR
  WSClient[WebSocket Client] --> Streamer[SnapshotStreamer]
  RestAPI[REST Endpoints] --> Streamer
  Streamer --> MCMP[MCPMRetriever]
  Streamer --> Reports[Background Reports<br/>async LLM]
  Streamer --> Settings[Settings Persistence]
  MCMP --> Viz[Visualization Data<br/>PCA, trails, agents]
  Viz --> WSClient
  Reports --> WSClient
```

CPU/GPU:
- Simulation and chunking are CPU-heavy; embeddings can use GPU.
- Not recommended to move the MCMP loop to GPU; keep HF embeddings on GPU if available.

Model upgrades:
- UI: set OLLAMA_MODEL; to use external providers, replace the local generator helper with your SDK.
- Agent chat: swap the LC chat model for one with tool-calls support.
- RagV1: set use_ollama=true or change llm_model/llm_device.

MCMP vs RAG:
- MCMP explores with agents and pheromone trails to reveal multi-hop traces (dependencies across files) that static nearest-neighbor search misses.
- RAG module provides traditional semantic + keyword hybrid search with persistent vector stores.

Architecture Evolution:
- **Current**: Dual interface (Streamlit comprehensive, React/WebSocket realtime)
- **Realtime Features**: Live simulation control, agent manipulation, streaming reports
- **Integration**: Both interfaces share core MCMP and RAG components
- **Future**: Enhanced agent orchestration, persistent trail analysis, collaborative editing integration
