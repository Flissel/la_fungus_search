### Scripts overview

This project contains the following entry points and utilities.

- streamlit_fungus_backup.py
  - Status: Active (primary frontend)
  - Purpose: Full UI for MCPM search, agent chat, background reports, and the RagV1 section named "Rag".
  - When to use: Day-to-day interactive exploration, running multi-queries, viewing results and summaries.

- src/embeddinggemma/rag_v1.py
  - Status: Active
  - Purpose: RAG v1 pipelines with Qdrant + LlamaIndex, AST chunking, and hybrid retrieval.
  - When to use: Building/loading a persistent index; programmatic queries; CLI subcommands for build/query/compare/stats.

- src/embeddinggemma/agents/agent_fungus_rag.py
  - Status: Active (advanced / CLI agent)
  - Purpose: Combined tools agent (lines/collage/fungus/rag_v1) over a given file or corpus; useful for headless runs.
  - Run: `python -m embeddinggemma.agents.agent_fungus_rag ...`

  

  

- (Removed) src/embeddinggemma/fungus_api.py
  - FastAPI service has been removed.

- (Removed) src/embeddinggemma/app.py, src/embeddinggemma/cli.py
  - Streamlit demo app and simple CLI have been removed.

- src/embeddinggemma/codespace_analyzer.py
  - Status: Kept
  - Purpose: Code-space analyzer tool for quick scanning of `src`.


