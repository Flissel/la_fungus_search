# TRAE C4 Architecture - Quick Reference

## 📋 Diagram Overview

| Diagram Level | Focus | File Section | Use Case |
|---------------|-------|--------------|----------|
| **Level 1: Context** | System boundaries & external actors | System Context | High-level system overview for stakeholders |
| **Level 2: Container** | Application containers & data stores | Container Diagram | Understanding system decomposition |
| **Level 3: Components** | Internal component structure | Component Diagrams | Detailed technical architecture |
| **Data Flow** | Process flows | Data Flow Diagram | Understanding MCMP simulation process |
| **Deployment** | Runtime environment | Deployment Diagram | Development environment setup |

## 🎯 Key Architectural Patterns

### Multi-Interface Pattern
- **React Frontend** (Port 5173): Modern, real-time visualization
- **Streamlit UI** (Port 8501): Comprehensive research interface
- **FastAPI Backend** (Port 8011): Real-time simulation orchestration

### Dual Retrieval Strategy
- **MCMP-RAG**: Multi-agent Physarum-inspired retrieval with emergent behavior
- **Enterprise RAG**: Traditional semantic search with Qdrant persistence

### Real-time Communication
- **WebSocket**: Live simulation updates, metrics, and visualization data
- **HTTP REST**: Configuration, control commands, and document access

## 🔌 Port Reference

```
┌─────────────────────────────────────────┐
│  TRAE System Port Configuration         │
├─────────────────────────────────────────┤
│  5173  │  React Dev (Vite)              │
│  8011  │  FastAPI Backend               │
│  8501  │  Streamlit UI                  │
│  6333  │  Qdrant Vector DB              │
│  11434 │  Ollama LLM Server             │
└─────────────────────────────────────────┘
```

## 🏗️ Component Responsibilities

### Frontend Components
- **React App**: Real-time visualization, WebSocket management
- **Streamlit UI**: Comprehensive controls, agent chat, multi-query

### Backend Components  
- **SnapshotStreamer**: Simulation orchestration, state management
- **MCPMRetriever**: Multi-agent simulation facade
- **Settings Manager**: Configuration validation and persistence

### Processing Components
- **Simulation Engine**: Agent dynamics, pheromone trails
- **Embedding Manager**: EmbeddingGemma model integration
- **Search Engine**: Hybrid semantic + keyword search

## 🔄 Data Flow Patterns

### Simulation Loop
```
Initialize → Spawn Agents → Move Agents → Deposit Pheromones → 
Update Relevance → Decay Trails → Create Snapshot → Broadcast → Loop
```

### Report Generation
```
Search Top-K → Build Prompt → Call Ollama → Parse JSON → 
Save Report → Broadcast to Frontend
```

### WebSocket Events
- `snapshot`: Visualization data (agents, documents, pheromone trails)
- `metrics`: Simulation statistics (step, relevance scores, trail count)
- `results`: Current top-K search results
- `report`: LLM-generated analysis per step
- `log`: Real-time logging and status updates

## 🛠️ Technology Stack Summary

### Core Technologies
- **Python 3.10+**: Backend processing, simulation, RAG
- **React 18 + TypeScript**: Modern frontend with type safety
- **FastAPI**: High-performance async web framework
- **Streamlit**: Rapid prototyping web interface

### ML/AI Stack
- **EmbeddingGemma-300M**: Google's embedding model
- **sentence-transformers**: Embedding pipeline
- **FAISS**: Fast similarity search
- **Ollama**: Local LLM inference

### Data & Storage
- **Qdrant**: Vector database for Enterprise RAG
- **File System**: Local caching (.fungus_cache)
- **NumPy**: Numerical computing for agent simulation

### Visualization
- **Plotly.js**: Interactive 2D/3D plotting
- **PCA**: Dimensionality reduction for visualization
- **WebSockets**: Real-time data streaming

## 📊 Performance Characteristics

### MCMP Simulation
- **CPU-intensive**: Agent movement calculations
- **Memory footprint**: Scales with document count and agent population
- **Real-time**: Live updates every N steps (configurable)

### Enterprise RAG
- **GPU-optional**: Embeddings can utilize GPU acceleration
- **Persistent**: Qdrant for durable vector storage
- **Hybrid**: Combines semantic and keyword search

## 🎮 User Interaction Flows

### React Frontend Flow
1. Configure simulation parameters in sidebar
2. Start simulation via `/start` endpoint
3. Watch real-time visualization via WebSocket
4. Click documents for detailed inspection
5. View step-by-step LLM reports

### Streamlit Flow
1. Set query and corpus parameters
2. Run MCMP simulation with live progress
3. View results and agent chat
4. Generate background reports
5. Switch to Enterprise RAG mode

## 🔍 Quick Navigation

- **Full diagrams**: See `/workspace/docs/c4_architecture.md`
- **System Context**: Start here for high-level overview
- **Container Diagram**: Understand service boundaries
- **Component Diagrams**: Dive into technical details
- **Data Flow**: Understand the MCMP process
- **Deployment**: Set up development environment

---

*Generated for TRAE (EmbeddingGemma) system documentation*