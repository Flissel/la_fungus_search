# Configuration Reference

This document describes all configuration parameters available in EmbeddingGemma, including environment variables, simulation parameters, and API settings.

## Environment Variables

### LLM Configuration
- **OLLAMA_MODEL**: LLM model name for queries, summaries, and agent chat
  - Default: `qwen2.5-coder:7b`
  - Example: `llama3.1:7b`, `qwen2.5-coder:14b`
  
- **OLLAMA_HOST**: Ollama server URL
  - Default: `http://127.0.0.1:11434`
  - Example: `http://localhost:11434`

### HuggingFace Integration  
- **HF_TOKEN**: Hugging Face authentication token
  - Required for: Private model access, increased rate limits
  - Optional for: Public models like EmbeddingGemma

## Simulation Parameters

### Core Agent Settings
- **num_agents**: Number of agents in the simulation
  - Range: 1-10000
  - Default: 200
  - Higher values: Better exploration, more CPU usage

- **max_iterations**: Maximum simulation steps before auto-stop
  - Range: 1-5000  
  - Default: 200
  - Higher values: More thorough exploration, longer runtime

- **exploration_bonus**: Agent noise/exploration factor upper bound
  - Range: 0.01-1.0
  - Default: 0.1
  - Higher values: More random exploration, less focused search

- **pheromone_decay**: Trail decay rate per step
  - Range: 0.5-0.999
  - Default: 0.95
  - Higher values: Longer trail persistence, more memory

### Performance Settings
- **embed_batch_size**: Embedding batch size for GPU processing
  - Range: 1-4096
  - Default: 128
  - Adjust based on GPU memory

- **chunk_workers**: CPU threads for document chunking
  - Range: 1-128
  - Default: max(4, cpu_count)
  - Higher values: Faster corpus building

- **max_chunks_per_shard**: Chunks per shard in background jobs
  - Range: 0-100000
  - Default: 2000
  - 0 = no sharding, single job

## Corpus Configuration

### Source Control
- **use_repo**: Use 'src' directory vs custom root
  - Type: Boolean
  - Default: true
  - true: Search in 'src/' directory
  - false: Use root_folder setting

- **root_folder**: Custom corpus root directory
  - Type: String
  - Default: current working directory
  - Used when use_repo=false

- **max_files**: Maximum files to process
  - Range: 0-20000
  - Default: 500
  - 0 = no limit

- **exclude_dirs**: Directories to skip during corpus building
  - Type: List[String]
  - Default: [".venv", "node_modules", ".git", "external"]
  - Example: [".git", "__pycache__", "build"]

- **windows**: Chunk window sizes in lines
  - Type: List[Integer]
  - Default: Must be provided by frontend
  - Example: [1000, 2000, 4000]
  - Creates multi-scale chunks for better retrieval

## Visualization Parameters

### Display Settings
- **viz_dims**: Visualization dimensions
  - Values: 2 or 3
  - Default: 2
  - 2D: Faster rendering, better for large networks
  - 3D: More immersive, higher resource usage

- **redraw_every**: Steps between visualization updates
  - Range: 1-100
  - Default: 2
  - Higher values: Less frequent updates, better performance

- **min_trail_strength**: Minimum trail strength to display
  - Range: 0.0-1.0
  - Default: 0.05
  - Higher values: Show only strong connections

- **max_edges**: Maximum edges in visualization
  - Range: 10-5000
  - Default: 600
  - Higher values: More detail, slower rendering

## Query & Results

### Search Configuration
- **query**: Current search query
  - Type: String
  - Default: "Explain the architecture."
  - Updated via API or frontend

- **top_k**: Number of top results to return
  - Range: 1-200
  - Default: 10
  - Higher values: More comprehensive results

### Reporting
- **report_enabled**: Enable background LLM reports
  - Type: Boolean
  - Default: false
  - Generates structured analysis of results

- **report_every**: Steps between reports
  - Range: 1-100
  - Default: 5
  - Higher values: Less frequent, more efficient

- **report_mode**: Report generation style
  - Values: "deep", "structure", "exploratory", "summary", "repair"
  - Default: "deep"
  - deep: Detailed analysis
  - structure: Focus on architecture
  - exploratory: Broad coverage
  - summary: Concise overview
  - repair: Error-prone code focus

## Port Configuration

### Default Ports
- **Streamlit**: 8501
- **Realtime Server**: 8011 (recommended)
- **Frontend Dev**: 5173 (Vite default)
- **Ollama**: 11434

### Development Ports
When running multiple instances, use different ports:
```bash
# Realtime server on custom port
uvicorn src.embeddinggemma.realtime.server:app --port 8012

# Streamlit on custom port  
streamlit run streamlit_fungus_backup.py --server.port 8502
```

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 8GB
- Storage: 2GB for models

### Recommended
- CPU: 8+ cores
- RAM: 16GB+
- GPU: CUDA-capable for embeddings
- Storage: 5GB+ for larger corpora

## Configuration Files

### Settings Persistence
The realtime server saves settings to:
- Location: `.fungus_cache/settings.json`
- Format: JSON
- Auto-created on first run
- Updated via API calls

### Cache Directory
- Location: `.fungus_cache/`
- Contents:
  - `settings.json`: Parameter persistence
  - `reports/`: Background LLM reports
  - Corpus chunks and embeddings
  - GIF recordings (Streamlit)

## Usage Examples

### Basic Configuration
```json
{
  "num_agents": 300,
  "max_iterations": 100,
  "exploration_bonus": 0.15,
  "pheromone_decay": 0.95,
  "top_k": 10,
  "viz_dims": 2
}
```

### Performance Optimized
```json
{
  "num_agents": 100,
  "max_iterations": 50,
  "embed_batch_size": 256,
  "chunk_workers": 16,
  "redraw_every": 5,
  "max_edges": 300
}
```

### Exploration Focused  
```json
{
  "num_agents": 500,
  "max_iterations": 300,
  "exploration_bonus": 0.3,
  "pheromone_decay": 0.98,
  "min_trail_strength": 0.02,
  "max_edges": 1000
}
```

## Parameter Dependencies

### Related Settings
- `num_agents` ↔ `max_iterations`: More agents need fewer iterations
- `exploration_bonus` ↔ `pheromone_decay`: Balance exploration vs exploitation
- `viz_dims` ↔ `max_edges`: 3D can handle more edges visually
- `embed_batch_size` ↔ GPU memory: Adjust based on available VRAM
- `chunk_workers` ↔ CPU cores: Typically set to core count or 2x

### Performance Impact
- **CPU intensive**: num_agents, max_iterations, chunk_workers
- **Memory intensive**: embed_batch_size, max_files, max_edges  
- **GPU intensive**: embed_batch_size (if GPU available)
- **Network intensive**: redraw_every, report_every (WebSocket updates)