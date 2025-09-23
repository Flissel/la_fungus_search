## C4 Diagrams: Per-step LLM Reporting for MCMP

### System Context (Level 1)

```mermaid
graph LR
  user[User] --> FE[React/Vite Frontend]
  FE -->|HTTP start/config/reset/search/answer| BE[FastAPI Backend]
  FE <-->|WebSocket snapshot/metrics/results/report| BE
  BE -->|LLM prompts| LLM[Ollama Server]
  BE --> FS[(.fungus_cache)]
  BE --> EMB[(Embeddings / FAISS)]

  subgraph Simulation
    BE --> RET[MCPMRetriever<br/>mcmp.simulation]
  end
```

### Container (Level 2)

```mermaid
graph TB
  subgraph Frontend [Frontend 5173]
    UI[Sidebar Controls<br/>TopK, Mode, Windows]
    Plot[Plotly 2D/3D]
    StepReport[Step Report Panel]
    UI --> Plot
    UI --> StepReport
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
