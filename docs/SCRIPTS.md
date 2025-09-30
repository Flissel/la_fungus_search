### Scripts overview

This project contains the following entry points and utilities.

- streamlit_fungus_backup.py
  - Status: Active (primary comprehensive frontend)
  - Purpose: Full UI for MCMP search, agent chat, background reports, and RAG integration.
  - When to use: Day-to-day interactive exploration, running multi-queries, viewing results and summaries, agent conversations.
  - Run: `streamlit run streamlit_fungus_backup.py`

- src/embeddinggemma/realtime/server.py
  - Status: Active (realtime WebSocket server) 
  - Purpose: FastAPI server with WebSocket live updates, REST API for simulation control, agent manipulation.
  - When to use: Live visualization, real-time collaboration, programmatic API access, React frontend development.
  - Run: `uvicorn src.embeddinggemma.realtime.server:app --reload --port 8011`

- experimerntal/old/rag_v1.py
  - Status: Legacy (non-functional - missing dependencies)
  - Purpose: Previous RAG implementation, now replaced by integrated RAG components
  - Note: Use RAG features through Streamlit UI instead

- src/embeddinggemma/agents/agent_fungus_rag.py
  - Status: Active (advanced / CLI agent)
  - Purpose: Combined tools agent (lines/collage/fungus/rag_v1) over a given file or corpus; useful for headless runs.
  - Run: `python -m embeddinggemma.agents.agent_fungus_rag ...`

  

  

- frontend/
  - Status: Active (React frontend for realtime server)
  - Purpose: Live visualization of MCMP simulation with controls for pause/resume, agent manipulation.
  - When to use: Real-time monitoring, interactive visualization, demo purposes.
  - Run: `cd frontend && npm run dev` (requires realtime server running)

- src/embeddinggemma/codespace_analyzer.py
  - Status: Available (utility)
  - Purpose: Code-space analyzer tool for quick scanning and analysis of `src` directory.
  - When to use: Initial codebase exploration, generating corpus statistics.

## Key Components (Libraries)

- src/embeddinggemma/mcmp_rag.py
  - Core MCMP retriever implementation with agent-based simulation

- src/embeddinggemma/mcmp/
  - simulation.py: Agent movement, pheromone mechanics  
  - embeddings.py: HuggingFace embedding integration
  - visualize.py: PCA visualization, network graphs
  - pca.py: Dimensionality reduction utilities

- src/embeddinggemma/rag/
  - chunking.py: AST-based code chunking
  - vectorstore.py: Qdrant integration
  - search.py: Hybrid semantic + keyword search
  - generation.py: LLM response generation

- src/embeddinggemma/ui/
  - Streamlit UI components for corpus building, MCMP running, reports


