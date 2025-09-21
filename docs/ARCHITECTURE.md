### Architecture (C4 diagrams)

#### C4: System Context (Level 1)
```mermaid
graph LR
  user[User] --> UI[Streamlit Frontend<br/>streamlit_fungus_backup.py]
  UI -->|MCMP search| MCMP[MCPMRetriever<br/>mcmp_rag.py]
  UI -->|Rag| RAG[RagV1<br/>src/embeddinggemma/rag_v1.py]
  RAG --> Qdrant[Qdrant]
  RAG --> HF[HF Embeddings]
  UI -->|Agent chat| LLM[LLM via Ollama or HF]
  UI -->|Background report| Cache[.fungus_cache]
  %% FastAPI API removed
```

Key points:
- Streamlit UI is primary; includes Rag (RagV1).
- MCMP runs in-process; background reports use a thread pool.
- RagV1 persists vectors to Qdrant; embeddings via EmbeddingGemma.
- LLMs via Ollama by default; HF LLM fallback where needed.

#### C4: Container (Level 2)
```mermaid
graph TB
  subgraph Frontend [Streamlit Frontend]
    UI[UI + Controls<br/>multi-query, dedup, logging]
    AG[Agent Chat<br/>LangChain / LangGraph]
    CORP[Corpus Builder<br/>chunkers / cache]
    BG[Background Reports<br/>ThreadPoolExecutor]
  end

  subgraph Retrieval
    MCMP[MCPMRetriever<br/>simulation, scoring]
    RAG[RagV1<br/>AST chunking, hybrid retrieval]
  end

  %% Services section removed (API deleted)

  subgraph DataStores
    Q[Qdrant]
    C[.fungus_cache<br/>chunks / reports / gifs]
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
  %% API links removed
```

#### C4: Component (Level 3) – Frontend internals
```mermaid
graph LR
  Settings[Sidebar Settings] --> Corpus[Corpus Builder]
  Settings --> MQ[Multi-Query Generator<br/>Ollama]
  MQ --> Dedup[Dedup<br/>Jaccard]
  Corpus --> Runner[MCMP Runner<br/>sharded, logs, GIFs]
  Runner --> Raw[Raw Results<br/>snippets]
  Raw --> Summary[Agent Summary<br/>LLM]
  Settings --> Rag[Rag Section<br/>RagV1]
  Rag --> Qdrant[Qdrant]
  Chat[Agent Chat] --> Tools[Tools<br/>search_code / get_settings / set_root_dir]
  Chat --> LLM[Ollama / LC LLM]
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

Current → Agent-based:
- Today: MCMP loop and optional agent chat.
- Target: orchestrate retrieval as agent tools (corpus slicing, MCMP passes, Enterprise hints), persist trails, feed a coder agent via the Edit Events API.
