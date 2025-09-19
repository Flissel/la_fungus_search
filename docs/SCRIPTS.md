### Scripts overview

This project contains multiple entry points. Use this guide to know which to keep using and which are legacy.

- streamlit_fungus.py
  - Status: Active (primary frontend)
  - Purpose: Full UI for MCPM search, agent chat, background reports, and the RagV1 section named "Rag".
  - When to use: Day-to-day interactive exploration, running multi-queries, viewing results and summaries.

- src/embeddinggemma/rag_v1.py
  - Status: Active
  - Purpose: RAG v1 pipelines with Qdrant + LlamaIndex, AST chunking, and hybrid retrieval.
  - When to use: Building/loading a persistent index; programmatic queries; CLI subcommands for build/query/compare/stats.

- src/embeddinggemma/agent_fungus_rag.py
  - Status: Active (advanced / CLI agent)
  - Purpose: Combined tools agent (lines/collage/fungus/rag_v1) over a given file or corpus; useful for headless runs.

- tools/rag.py (SimpleRAG demo)
  - Status: Legacy example
  - Purpose: Minimal FAISS-based RAG demo.
  - Recommendation: Prefer rag_v1.py.

- mcmp_cli.py
  - Status: Removed (legacy)
  - Recommendation: Use streamlit_fungus.py and the CLI in rag_v1.py.

- src/embeddinggemma/fungus_api.py
  - Status: Active
  - Purpose: FastAPI endpoints for Fungus/MCMP retrieval modes and code edit event builder.

- src/embeddinggemma/app.py, src/embeddinggemma/cli.py, src/embeddinggemma/codespace_analyzer.py
  - Status: Auxiliary
  - Purpose: Desktop demo UI, simple CLI, and a code-space analyzer tool. Keep as examples/utilities.


