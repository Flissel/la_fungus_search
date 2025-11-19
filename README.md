# LA Fungus Search

**Multi-Agent Codebase Analysis using RAG and Mycelial Search**

A sophisticated code analysis tool that combines Retrieval-Augmented Generation (RAG) with multi-agent simulation to explore and understand large codebases through semantic search and intelligent question-answering.

---

## Overview

LA Fungus Search implements a novel **Mycelial Collective Multi-Perspective (MCMP) RAG** approach, where autonomous agents simulate a network of interconnected perspectives exploring a codebase. Like a mycelial network in nature, agents share discoveries, build upon each other's findings, and collectively construct comprehensive understanding of code structure and functionality.

### Key Features

- **Multi-Agent Simulation** - Autonomous agents explore codebases with different perspectives and queries
- **Semantic Code Search** - RAG-powered search using state-of-the-art embedding models
- **Real-Time Visualization** - Interactive web interface showing agent activity and discoveries
- **Flexible Backend** - Support for both in-memory FAISS and persistent Qdrant vector stores
- **LLM Integration** - Compatible with OpenAI, Ollama, and other OpenAI-compatible APIs
- **Modular Architecture** - Clean separation of concerns with 8 routers and 3 service layers
- **Production-Ready** - Comprehensive error handling, logging, and configuration management

---

## Architecture

LA Fungus Search follows a modern, modular architecture:

### Backend (Python/FastAPI)
```
src/embeddinggemma/realtime/
├── routers/           8 modular routers (collections, simulation, search, etc.)
├── services/          3 service layers (qdrant, settings, prompts)
└── server.py          Main FastAPI application
```

### Frontend (React/TypeScript)
```
frontend/src/
├── components/        10 UI components
├── hooks/             3 custom hooks (WebSocket, state, settings)
├── services/          API client
├── types/             TypeScript definitions
└── context/           Global state management
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture overview.

---

## Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18+ and npm
- **OpenAI API Key** or **Ollama** (for local LLM)

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd la_fungus_search

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root by copying `.env.example`:

```bash
cp .env.example .env
```

Then edit `.env` with your configuration. The `.env.example` file contains:
- **LLM Provider options**: OpenAI, Ollama (local), Google Gemini, or Grok
- **Embedding model configuration**: OpenAI or local models
- **Vector database**: Qdrant (persistent) or FAISS (in-memory, default)
- **Server ports and logging settings**

**Minimum required**: Choose one LLM provider and set its credentials. For OpenAI:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_actual_key_here
```

See `.env.example` for all available options and quick-start examples.

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 4. Start the Application

**Terminal 1 - Backend:**
```bash
# From project root
python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011
```

**Terminal 2 - Frontend:**
```bash
# From frontend directory
npm run dev
```

**Access the Application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8011
- API Documentation: http://localhost:8011/docs

---

## Usage

### 1. Configure Settings

Navigate to the **Settings** tab to configure:
- **Codebase Path** - Directory to analyze
- **Task Mode** - Overall analysis objective:
  - **architecture**: Map system components, layers, dependencies, and design patterns
  - **bugs**: Detect security vulnerabilities with severity ratings (CRITICAL/HIGH/MEDIUM/LOW)
  - **quality**: Assess code complexity, SOLID principles, and maintainability
  - **documentation**: Generate comprehensive API documentation
  - **features**: Trace feature implementations end-to-end through all layers
  - **deep/structure/exploratory/summary/repair**: Generic analysis modes
- **Judge Mode** - Steering strategy for simulation:
  - **steering**: Balanced exploration (default)
  - **focused**: Deep-first exploration following call chains
  - **exploratory**: Breadth-first discovery
- **Vector Backend** - Choose between memory (FAISS) or Qdrant
- **Simulation Parameters** - Number of agents, max queries, etc.
- **LLM Settings** - Model selection, temperature, token limits
- **Chunking Windows** - Default [2000, 4000, 8000] lines for larger context

### 2. Start Simulation

Click **Start** to begin the multi-agent simulation:
- Agents will autonomously generate and execute queries
- Real-time metrics and logs appear in the interface
- Reports are generated as agents discover insights
- Judge mode steers exploration to fulfill the Task Mode objective

### 3. Search and Query

Use the **Search** tab to:
- Perform semantic search across the codebase
- Get LLM-generated answers to specific questions
- View relevant code chunks with full context (no truncation)

### 4. View Results

- **Metrics Panel** - Real-time simulation statistics
- **Logs Panel** - Agent activity and system events
- **Results Panel** - Task-specific analysis reports with actionable insights
- **Visualization** - Charts showing agent activity and discoveries

### Example Use Cases

**Architecture Analysis:**
```
Task Mode: architecture
Query: "Explain the system architecture"
Result: Component hierarchy, design patterns, data flow diagrams
```

**Bug Detection:**
```
Task Mode: bugs
Query: "Find security vulnerabilities"
Result: Prioritized list with severity ratings and line numbers
```

**Code Quality Assessment:**
```
Task Mode: quality
Query: "Assess code quality"
Result: Complexity analysis, SOLID violations, refactoring recommendations
```

---

## Advanced Features

### Qdrant Integration

For persistent vector storage and better performance at scale:

1. **Start Qdrant:**
   ```bash
   docker-compose -f docker-compose.qdrant.yml up -d
   ```

2. **Update `.env`:**
   ```
   QDRANT_URL=http://localhost:6339
   ```

3. **Reindex Codebase:**
   Use the Corpus tab to reindex your codebase into Qdrant

### Prompt Customization

LA Fungus Search includes specialized prompts for different analysis objectives (added January 2025):

**Task-Specific Prompts:**
- **Architecture Mode** - 58-line detailed instructions for mapping system components, design patterns, and data flow
- **Bugs Mode** - 88-line comprehensive checklist for detecting security vulnerabilities with severity ratings
- **Quality Mode** - 68-line assessment framework for SOLID principles, complexity, and maintainability
- **Documentation Mode** - 74-line guide for extracting API documentation with parameters and examples
- **Features Mode** - 82-line template for tracing features end-to-end through all system layers

**Judge Prompts:**
- **Focused Mode** - Deep-first exploration strategy following call chains to build complete mental models
- **Steering Mode** - Balanced adaptive exploration (default)
- **Exploratory Mode** - Breadth-first discovery of patterns and connections

Prompts can be customized via the **Prompts** modal in the UI or by editing files in [src/embeddinggemma/modeprompts/](src/embeddinggemma/modeprompts/)

### Multiple Collections

Switch between different codebases or analysis sessions:
- Create collections in Qdrant
- Switch collections in the **Collections** tab
- Each collection maintains independent embeddings

---

## API Reference

The backend exposes 32 REST endpoints organized into 8 routers:

- **Collections** - Manage vector collections (`/collections`)
- **Simulation** - Control simulation lifecycle (`/start`, `/stop`, etc.)
- **Search** - Semantic search and Q&A (`/search`, `/answer`)
- **Agents** - Manage agent population (`/agents`)
- **Settings** - Configuration management (`/settings`)
- **Prompts** - Prompt customization (`/prompts`)
- **Corpus** - Codebase indexing (`/corpus`)
- **Misc** - Jobs, reports, introspection (`/jobs`, `/reports`)

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete API documentation.

---

## Development

### Running Tests

```bash
# Backend tests
pytest

# Frontend tests (if configured)
cd frontend
npm test
```

### Code Quality

```bash
# Lint Python code
ruff check .

# Format Python code
ruff format .

# Type check frontend
cd frontend
npm run type-check
```

### Adding New Features

See our development guides:
- [docs/MAINTENANCE.md](docs/MAINTENANCE.md) - Adding routers, services, components
- [docs/FRONTEND_ARCHITECTURE.md](docs/FRONTEND_ARCHITECTURE.md) - Frontend structure
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API documentation
- **[FRONTEND_ARCHITECTURE.md](docs/FRONTEND_ARCHITECTURE.md)** - Frontend component hierarchy
- **[MAINTENANCE.md](docs/MAINTENANCE.md)** - Development and deployment guides
- **[CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)** - Configuration options
- **[IMPROVEMENTS_2025.md](docs/IMPROVEMENTS_2025.md)** - January 2025 improvements (Task Modes, Judge Modes, better chunking)
- **[REFACTORING_HISTORY.md](docs/REFACTORING_HISTORY.md)** - Architecture evolution

---

## Project Structure

```
la_fungus_search/
├── src/embeddinggemma/
│   ├── realtime/          Backend server
│   │   ├── routers/       8 modular routers
│   │   ├── services/      3 service layers
│   │   └── server.py      Main FastAPI app
│   ├── ui/                CLI tools and corpus management
│   └── llm/               LLM integration and prompts
├── frontend/              React/TypeScript frontend
│   └── src/
│       ├── components/    10 UI components
│       ├── hooks/         Custom React hooks
│       ├── services/      API client
│       └── types/         TypeScript definitions
├── docs/                  Comprehensive documentation
├── models/                Embedding models (not in git)
└── tests/                 Test suites
```

---

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Qdrant** - Vector database (optional)
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings
- **OpenAI/Ollama** - LLM integration

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server
- **Axios** - HTTP client
- **Plotly** - Interactive visualizations

---

## Performance

- **Codebase Size** - Tested with repos up to 10,000+ Python files
- **Concurrent Agents** - Supports 200+ simultaneous agents
- **Search Latency** - < 100ms for typical queries
- **Embedding** - ~10-50 chunks per second depending on model

---

## Troubleshooting

### Common Issues

**Q: "API key not found" error**
- Ensure `.env` file exists with `OPENAI_API_KEY` set
- Restart the backend after creating `.env`

**Q: Frontend can't connect to backend**
- Verify backend is running on port 8011
- Check CORS settings in server.py

**Q: Slow embedding generation**
- Use CUDA/GPU if available
- Consider using smaller embedding models
- Adjust `max_files` setting to limit corpus size

**Q: Out of memory errors**
- Reduce number of agents
- Use Qdrant instead of in-memory FAISS
- Limit chunk size and window settings

See [docs/MAINTENANCE.md](docs/MAINTENANCE.md) for more troubleshooting help.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with FastAPI, React, and modern RAG techniques
- Inspired by mycelial networks in nature
- Uses state-of-the-art embedding models from Sentence Transformers

---

## Contact & Support

For questions, issues, or contributions:
- Review our [CONTRIBUTING.md](CONTRIBUTING.md) guide
- Check existing documentation in `docs/`
- Open an issue for bugs or feature requests

---

**Status:** Production-ready with comprehensive refactoring completed in January 2025
