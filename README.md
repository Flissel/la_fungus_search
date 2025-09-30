# EmbeddingGemma MCMP

**Multi-agent Codebase Pattern Matching (MCMP)** - An intelligent code exploration system that combines multi-agent simulation with modern LLM capabilities for advanced codebase analysis and retrieval.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

EmbeddingGemma MCMP is a sophisticated code analysis platform that uses multi-agent simulation with pheromone trails to explore and understand codebases. Unlike traditional RAG systems, MCMP agents traverse code through semantic connections, revealing complex dependency patterns and relationships that static similarity search cannot discover.

### Key Features

- **🧠 Multi-Agent Simulation**: Intelligent agents with pheromone-based exploration
- **⚡ Real-time Visualization**: Live WebSocket updates with Plotly visualizations
- **🔧 Modern Tech Stack**: React + TypeScript frontend, FastAPI backend
- **🤖 Multi-LLM Support**: Ollama, OpenAI, Google Gemini, Grok integration
- **📊 Advanced Analytics**: Contextual steering with LLM-powered relevance assessment
- **🔍 Smart Chunking**: AST-aware code parsing for Python projects

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 16+** (for frontend)
- **Ollama** (optional, for local LLM inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd embeddinggemma-mcmp
   ```

2. **Backend setup**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Frontend setup**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

1. **Start the backend**
   ```bash
   # Development mode with auto-reload
   python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --reload --port 8011

   # Or use the provided script (Windows PowerShell)
   ./run-realtime.ps1
   ```

2. **Start the frontend** (in a separate terminal)
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application**
   - **Main interface**: `http://localhost:5173` (frontend dev server)
   - **Backend API**: `http://localhost:8011` (serves static frontend)
   - **WebSocket**: `ws://localhost:8011/ws` (real-time updates)
   - **API docs**: `http://localhost:8011/docs` (FastAPI auto-generated)

## Architecture

### System Components

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   React UI      │◄──────────────► │   FastAPI       │
│   (Plotly viz) │                 │   Backend       │
└─────────────────┘                 └─────────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────┐                 ┌─────────────────┐
│  MCMP Simulation│                 │   LLM Providers │
│  (Multi-agent)  │                 │   (Ollama, etc) │
└─────────────────┘                 └─────────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────┐                 ┌─────────────────┐
│  Corpus Builder │                 │   Qdrant Vector │
│  (AST chunking) │                 │   Store         │
└─────────────────┘                 └─────────────────┘
```

### Core Technologies

- **Frontend**: React 18, TypeScript, Vite, Plotly.js, Tailwind CSS
- **Backend**: FastAPI, Pydantic, WebSocket support, AsyncIO
- **ML/AI**: SentenceTransformers, NumPy, scikit-learn, FAISS
- **External APIs**: Ollama, OpenAI GPT, Google Gemini, Grok/x.ai

## Usage

### Basic Workflow

1. **Configure settings** in the sidebar (corpus path, chunk sizes, agent count)
2. **Start simulation** with a query (e.g., "Explain the architecture")
3. **Watch agents explore** in real-time via the pheromone network visualization
4. **Review results** in the results panel with relevance scores and code snippets
5. **Generate reports** using LLM analysis for deeper insights

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/start` | POST | Initialize MCMP simulation with corpus |
| `/config` | POST | Update simulation parameters |
| `/search` | POST | Search for relevant code snippets |
| `/answer` | POST | Generate LLM answer from search results |
| `/status` | GET | Current simulation status |
| `/ws` | WebSocket | Real-time simulation updates |

## Configuration

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=ollama                    # ollama, openai, google, grok
OLLAMA_MODEL=qwen2.5-coder:7b         # Ollama model name
OLLAMA_HOST=http://127.0.0.1:11434    # Ollama server URL
OPENAI_API_KEY=your_key_here          # OpenAI API key
OPENAI_MODEL=gpt-4o-mini              # OpenAI model

# Embedding Configuration
EMBEDDING_MODEL=google/embeddinggemma-300m
DEVICE_MODE=auto                      # auto, cpu, cuda

# Vector Storage
VECTOR_BACKEND=memory                  # memory, qdrant
QDRANT_URL=http://localhost:6339      # Qdrant endpoint
QDRANT_COLLECTION=codebase            # Collection name
```

### Settings Reference

See [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) for complete configuration options including:
- Visualization parameters (dimensions, edge rendering)
- Simulation settings (agent count, pheromone decay, exploration)
- Corpus configuration (chunk sizes, file limits, exclusions)
- LLM provider settings and API configurations

## Advanced Features

### Multi-Agent Simulation

The MCMP system uses agents that:
- **Navigate** through embedding space toward relevant code
- **Deposit pheromones** on paths to mark important connections
- **Adapt exploration** based on relevance feedback
- **Collaborate** to discover complex dependency patterns

### Contextual Steering

LLM-powered relevance assessment provides:
- **Dynamic re-ranking** of search results
- **Follow-up query generation** for deeper exploration
- **Context-aware boosting** of related code sections
- **Stagnation detection** to prevent unproductive exploration

### Real-time Visualization

- **3D/2D projection** of code embeddings and agent positions
- **Pheromone trails** showing exploration paths
- **Live metrics** tracking simulation progress
- **Interactive exploration** of code snippets and metadata

## Development

### Project Structure

```
├── src/embeddinggemma/           # Core Python package
│   ├── mcmp/                     # Multi-agent simulation modules
│   ├── rag/                      # Traditional RAG pipeline
│   ├── llm/                      # LLM provider integrations
│   ├── ui/                       # UI utilities and corpus tools
│   ├── realtime/                 # FastAPI server
│   └── tools/                    # Development utilities
├── frontend/                     # React application
├── tests/                        # Test suites
├── docs/                         # Documentation
└── models/                       # Embedding model files
```

### Running Tests

```bash
# Unit tests
pytest -q

# End-to-end tests (requires running backend)
cd frontend && npm run test:e2e

# Specific test modules
pytest tests/mcmp/ -v
pytest tests/rag/ -v
```

### Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and component relationships
- **[Maintenance Guide](docs/MAINTENANCE.md)** - Development setup and troubleshooting
- **[Configuration Reference](docs/CONFIG_REFERENCE.md)** - Complete settings documentation
- **[Scripts Overview](docs/SCRIPTS.md)** - Entry points and utilities

## Examples

### Basic Search Query

```python
from embeddinggemma.mcmp_rag import MCPMRetriever

# Initialize retriever
retriever = MCPMRetriever(num_agents=200, max_iterations=100)

# Add code documents
retriever.add_documents(["def hello():\n    print('world')"])

# Run simulation
retriever.initialize_simulation("How does this function work?")
results = retriever.search("function implementation", top_k=5)

for result in results['results']:
    print(f"Score: {result['relevance_score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
```

### Custom LLM Integration

```python
from embeddinggemma.llm import generate_text

# Use different LLM providers
response = generate_text(
    provider="openai",
    prompt="Explain this code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    openai_model="gpt-4o-mini",
    openai_api_key="your-key"
)
```

## Contributing

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the existing code style
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Code Style

- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Include docstrings for public functions
- Add tests for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **EmbeddingGemma** by Google for the embedding models
- **SentenceTransformers** library for embedding utilities
- **FastAPI** for the modern web framework
- **React** ecosystem for the frontend interface
- **Plotly** for interactive visualizations

## Support

For issues, questions, or contributions, please:
1. Check the [documentation](docs/) first
2. Search existing [issues](../../issues)
3. Create a new issue with detailed information
4. Join our community discussions

---

**Happy coding with intelligent code exploration! 🚀**