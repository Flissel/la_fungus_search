## C4 Diagrams: Per-step LLM Reporting for MCMP

### System Context (Level 1)

```mermaid
graph LR
  user[User] --> FE[React Frontend<br/>localhost:5173]
  FE -->|HTTP API calls| BE[FastAPI Backend<br/>localhost:8011]
  FE <-->|WebSocket| BE
  BE -->|MCMP simulation| MCMP[MCPMRetriever<br/>Multi-agent exploration]
  BE -->|LLM queries| LLM[Multi-provider<br/>Ollama, OpenAI, Google, Grok]
  BE -->|Corpus indexing| CORPUS[Code chunking<br/>AST parsing, caching]
  BE -->|Vector storage| QDRANT[Qdrant database<br/>Optional vector backend]
  BE -->|Persistence| CACHE[.fungus_cache<br/>Settings, reports, artifacts]

  subgraph Simulation
    MCMP --> AGENTS[Agents<br/>Position, velocity, exploration]
    MCMP --> PHEROMONES[Pheromone trails<br/>Visit tracking, decay]
  end
```

### Container (Level 2)

```mermaid
graph TB
  subgraph Frontend [React Frontend]
    Settings[Configuration Sidebar<br/>Query, mode, corpus settings]
    Visualization[Plotly Network View<br/>Real-time 2D/3D rendering]
    Reports[Step Reports Panel<br/>LLM analysis results]
    Metrics[Performance Dashboard<br/>Simulation metrics, charts]
    Settings --> Visualization
    Settings --> Reports
    Settings --> Metrics
  end

  subgraph Backend [Backend 8011]
    API[/HTTP start, config, reset, search, answer/]
    WS[/WebSocket snapshot, metrics, results, report/]
    Streamer[SnapshotStreamer<br/>run_loop]
    Sim[MCPMRetriever<br/>step/agents/pheromones]
    Ranker[TopK Selector<br/>relevance]
    Reporter[LLM Report Orchestrator<br/>prompt by mode]
    Store[ReportStore<br/>write step_n.json]
    API --> Streamer
    Streamer --> Sim --> Ranker --> Reporter --> Store
    Streamer --> WS
    Store --> WS
  end

  Reporter --> LLM
  Store --> FS[(.fungus_cache)]
```

### Component (Level 3)

```mermaid
graph TB
  subgraph Backend Components
    Settings[SettingsModel<br/>query, top_k, mode, report_enabled]
    Loop[Simulation Loop<br/>step_i += 1]
    Snapshot[Build Snapshot<br/>PCA docs+agents]
    TopK[Select TopK<br/>by relevance]
    Prompt[Build Prompt by mode<br/>structured JSON]
    CallLLM[generate_with_ollama]
    Parse[Parse JSON -> Report Items]
    Persist[write step_i.json]
    Broadcast[WS report payload]
  end

  Settings --> Loop
  Loop --> Snapshot
  Loop --> TopK --> Prompt --> CallLLM --> Parse --> Persist --> Broadcast
```

### Report item (per doc, per step)

- code_chunk: text slice
- content: surrounding/full text used
- file_path: script/module path
- line_range: [start, end]
- code_purpose: brief intent/behavior
- code_dependencies: imports, callers/callees, globals
- file_type: py/test/json/etc.
- embedding_score: similarity to query
- relevance_to_query: rationale
- query_initial: original query string
- follow_up_queries: list derived by LLM
