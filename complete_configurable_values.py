"""
COMPLETE Configurable Values in EmbeddingGemma
==============================================

After thorough code analysis, here are ALL configurable values found.

🎛️ This is the DEFINITIVE list of every parameter you can adjust!
"""

# =============================================================================
# ENVIRONMENT VARIABLES (External Configuration)
# =============================================================================

ENVIRONMENT_VARIABLES = {
    # Database/Vector Store
    "QDRANT_URL": "http://localhost:6337",
    "QDRANT_API_KEY": None,  # Optional
    "RAG_COLLECTION": "codebase", 
    "RAG_PERSIST_DIR": "./enterprise_index",
    
    # Embedding Models
    "EMBED_MODEL": "google/embeddinggemma-300m",
    
    # LLM Configuration
    "RAG_LLM_MODEL": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "RAG_LLM_DEVICE": "auto",  # auto|cuda|cpu
    
    # Ollama Configuration  
    "RAG_USE_OLLAMA": "0",  # "1" for true
    "OLLAMA_MODEL": "qwen2.5-coder:7b",
    "OLLAMA_HOST": "http://127.0.0.1:11434"
}

# =============================================================================
# MCMP RETRIEVER PARAMETERS (Core Simulation)
# =============================================================================

MCMP_RETRIEVER_CONFIG = {
    # Model Settings
    "embedding_model_name": "google/embeddinggemma-300m",
    "device_mode": "auto",  # auto|cuda|cpu
    "use_embedding_model": True,
    "embed_batch_size": 128,
    "build_faiss_after_add": True,
    
    # Agent Configuration
    "num_agents": 200,
    "max_iterations": 50,
    "exploration_bonus": 0.1,
    
    # Pheromone Settings
    "pheromone_decay": 0.95,
    
    # Internal Settings
    "log_every": 1,
    "kw_lambda": 0.0,  # Keyword scoring weight
    "kw_terms": set()  # Keyword terms set
}

# =============================================================================
# AGENT CLASS DEFAULTS
# =============================================================================

AGENT_DEFAULTS = {
    "energy": 1.0,
    "trail_strength": 1.0,
    "exploration_factor": 0.3,  # Note: Different from exploration_bonus!
    "age": 0,
    "visited_docs": set()
}

# =============================================================================
# DOCUMENT CLASS DEFAULTS  
# =============================================================================

DOCUMENT_DEFAULTS = {
    "relevance_score": 0.0,
    "visit_count": 0,
    "last_visited": 0.0,
    "embedding": None,
    "metadata": {}
}

# =============================================================================
# REALTIME SERVER CONFIGURATION
# =============================================================================

REALTIME_SERVER_CONFIG = {
    # Visualization Settings
    "redraw_every": 2,              # Visualization update frequency
    "min_trail_strength": 0.05,     # Minimum pheromone trail to display
    "max_edges": 600,               # Maximum edges in network viz
    "viz_dims": 2,                  # 2D or 3D visualization
    
    # Query Settings
    "query": "Explain the architecture.",
    "top_k": 10,
    
    # Corpus Processing
    "use_repo": True,
    "root_folder": ".",  # os.getcwd()
    "max_files": 500,
    "exclude_dirs": [".venv", "node_modules", ".git", "external"],
    "windows": [],  # Chunk window sizes
    "chunk_workers": 4,  # max(4, os.cpu_count() or 8)
    
    # Simulation Configuration
    "max_iterations": 200,
    "num_agents": 200,
    "exploration_bonus": 0.1,
    "pheromone_decay": 0.95,
    "embed_batch_size": 128,
    "max_chunks_per_shard": 2000,
    
    # Reporting
    "report_enabled": False,
    "report_every": 5,
    "report_mode": "deep",
    
    # Internal State
    "step_i": 0,
    "_paused": False,
    "_saved_state": None
}

# =============================================================================
# UI SETTINGS (Streamlit Interface)
# =============================================================================

UI_SETTINGS_CONFIG = {
    # Search Settings
    "mode": "deep",                    # deep|structure|exploratory|summary|repair
    "top_k": 5,
    "windows": [],                     # Chunk window sizes
    
    # Repository Settings
    "use_repo": True,
    "root_folder": "",
    "max_files": 1000,
    "exclude_dirs": [".venv", "node_modules", ".git", "external"],
    "docs_file": "",
    
    # MCMP Settings
    "num_agents": 200,
    "max_iterations": 60,
    "exploration_bonus": 0.1,
    "pheromone_decay": 0.95,
    "embed_bs": 64,                    # Note: Different name from embed_batch_size
    "max_chunks_per_shard": 2000,
    
    # Display Settings
    "show_tree": True,
    "show_network": True,
    "gen_answer": False,
    
    # Search Processing
    "div_alpha": 0.7,                  # Diversity weighting
    "dedup_tau": 0.92,                 # Deduplication threshold
    "per_folder_cap": 2,               # Max results per folder
    "pure_topk": False,                # Use diversity vs pure top-k
    "log_every": 10
}

# =============================================================================
# RAG SYSTEM CONFIGURATION
# =============================================================================

RAG_CONFIG = {
    # Database
    "qdrant_url": "http://localhost:6337",
    "qdrant_api_key": None,
    "collection_name": "codebase",
    "persist_dir": "./enterprise_index",
    
    # Models
    "embedding_model": "google/embeddinggemma-300m",
    "llm_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "llm_device": "auto",  # auto|cuda|cpu
    
    # Ollama
    "use_ollama": False,
    "ollama_model": "qwen2.5-coder:7b",
    "ollama_host": "http://127.0.0.1:11434"
}

# =============================================================================
# QDRANT VECTOR DATABASE CONFIGURATION
# =============================================================================

QDRANT_CONFIG = {
    # Vector Configuration
    "distance": "COSINE",              # Distance metric
    
    # HNSW Index Settings
    "hnsw_m": 32,                      # HNSW M parameter
    "hnsw_ef_construct": 128,          # HNSW ef_construct parameter
    
    # Optimizer Settings  
    "indexing_threshold": 20000,       # Indexing threshold
    "memmap_threshold": 200000,        # Memory mapping threshold
    
    # Quantization Settings
    "quantization_bits": 8,            # Scalar quantization bits
    "always_ram": False                # Keep quantized vectors in RAM
}

# =============================================================================
# SEARCH ALGORITHM PARAMETERS
# =============================================================================

SEARCH_CONFIG = {
    # Hybrid Search
    "top_k": 5,
    "alpha": 0.7,                      # Semantic vs keyword weighting
    "semantic_multiplier": 2,          # Retrieve top_k * 2 for semantic
    
    # Content Processing
    "content_preview_length": 2048,    # Content length for diversity calculation
    "random_seed": 42,                 # Fallback embedding seed
    "fallback_embedding_dim": 64       # Fallback embedding dimension
}

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

CHUNKING_CONFIG = {
    # File Processing
    "encoding": "utf-8",
    "lines_per_chunk": 20,             # Fallback chunking size
    "errors": "ignore",                # File reading error handling
    
    # AST Processing
    "include_ast_metadata": True
}

# =============================================================================
# LLM GENERATION PARAMETERS
# =============================================================================

LLM_GENERATION_CONFIG = {
    # Ollama Generation
    "timeout": 500,                    # Request timeout in seconds
    "stream": False,                   # Streaming response
    "temperature": 0.1,                # Generation temperature
    
    # Request Settings
    "host_default": "http://127.0.0.1:11434",
    "model_default": "qwen2.5-coder:7b"
}

# =============================================================================
# CORPUS PROCESSING PARAMETERS
# =============================================================================

CORPUS_CONFIG = {
    # File Collection (_collect_py_documents)
    "max_files_default": 300,
    "max_chars_per_file": 4000,
    "file_encoding": "utf-8",
    "file_errors": "ignore",
    
    # Parallel Processing (collect_codebase_chunks)
    "default_workers": 4,
    "max_workers_formula": "max(4, min(32, (os.cpu_count() or 8) * 2))",
    
    # Default Windows (mcmp_runner.py)
    "default_windows": [100, 200, 300]
}

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

VISUALIZATION_CONFIG = {
    # Snapshot Generation
    "min_trail_strength_default": 0.05,
    "max_edges_default": 300,          # Note: Different from server default!
    "method": "pca",
    "whiten": False,
    "spread": 1.0,
    "jitter": 0.0,
    "dims": 2,
    
    # PCA Processing
    "k_2d": 2,
    "k_3d": 3,
    "safe_singular_value": 1.0,        # Fallback for zero singular values
    
    # Trail Display Threshold (visualize.py)
    "min_trail_display": 0.05,
    
    # Content Preview
    "content_preview_chars": 100
}

# =============================================================================
# FRONTEND HTML DEFAULTS
# =============================================================================

FRONTEND_HTML_CONFIG = {
    # Control Defaults (index.html)
    "dims_default": 2,                 # 2D|3D
    "min_trail_input_step": 0.01,
    "min_trail_input_value": 0.05,
    "max_edges_input_step": 50,
    "max_edges_input_value": 600,
    "query_input_width": "320px",
    
    # Plotly Configuration
    "plotly_version": "2.35.2",
    "paper_bgcolor": "#121212",
    "plot_bgcolor": "#121212",
    "font_color": "#ddd",
    "margins": {"l": 10, "r": 10, "t": 30, "b": 10}
}

# =============================================================================
# HARDCODED PHYSICS CONSTANTS (Not Currently Configurable)
# =============================================================================

PHYSICS_CONSTANTS = {
    # Agent Spawning (simulation.py:spawn_agents)
    "spawn_noise_std": 0.1,            # Position noise standard deviation
    "spawn_velocity_std": 0.05,        # Velocity noise standard deviation
    "exploration_factor_min": 0.05,    # Minimum exploration factor
    
    # Force Weights (simulation.py:update_agent_position)
    "attraction_weight": 0.8,          # Document attraction force weight
    "pheromone_weight": 0.15,          # Pheromone trail force weight
    "exploration_weight": 0.05,        # Random exploration force weight
    
    # Velocity Update
    "velocity_decay": 0.85,            # Velocity momentum decay
    "force_application": 0.15,         # Force application rate
    
    # Document Relevance Scoring (simulation.py:update_document_relevance)
    "visit_bonus_multiplier": 0.1,     # Bonus per document visit
    "visit_bonus_max": 0.5,            # Maximum visit bonus
    "time_bonus": 0.1,                 # Bonus for recently visited docs
    "time_window": 1.0,                # Recency window (seconds)
    
    # Pheromone Trail Management (simulation.py:deposit_pheromones)
    "trail_deposit_multiplier": 0.1,   # Pheromone deposit amount
    "trail_min_strength": 0.01,        # Minimum strength before removal
    "trail_memory_length": 3,          # Previous docs to link with trails
    
    # Network Extraction (simulation.py:extract_relevance_network)
    "relevance_threshold": 0.1,        # Minimum relevance for inclusion
    "content_preview_length": 100,     # Characters in content preview
    
    # Normalization Constants
    "position_norm_epsilon": 1e-12,    # Avoid division by zero
    "embedding_norm_epsilon": 1e-12    # Avoid division by zero
}

# =============================================================================
# API VALIDATION RANGES (Pydantic Constraints)
# =============================================================================

API_VALIDATION_RANGES = {
    # Visualization
    "redraw_every": {"min": 1, "max": 100, "default": 2},
    "min_trail_strength": {"min": 0.0, "max": 1.0, "default": 0.05},
    "max_edges": {"min": 10, "max": 5000, "default": 600},
    "viz_dims": {"options": [2, 3], "default": 2},
    "top_k": {"min": 1, "max": 200, "default": 10},
    "report_every": {"min": 1, "max": 100, "default": 5},
    
    # Corpus Processing
    "max_files": {"min": 0, "max": 20000, "default": 500},
    "chunk_workers": {"min": 1, "max": 128, "default": 4},
    
    # Simulation
    "max_iterations": {"min": 1, "max": 5000, "default": 200},
    "num_agents": {"min": 1, "max": 10000, "default": 200},
    "exploration_bonus": {"min": 0.01, "max": 1.0, "default": 0.1},
    "pheromone_decay": {"min": 0.5, "max": 0.999, "default": 0.95},
    "embed_batch_size": {"min": 1, "max": 4096, "default": 128},
    "max_chunks_per_shard": {"min": 0, "max": 100000, "default": 2000}
}

# =============================================================================
# ALL CONFIGURABLE VALUES BY COMPONENT
# =============================================================================

ALL_CONFIGURABLE_VALUES = {
    
    # CORE SIMULATION PARAMETERS
    "num_agents": {
        "defaults": {
            "MCPMRetriever": 200,
            "UI_Settings": 200, 
            "Realtime_Server": 200,
            "mcmp_runner": 100  # quick search default
        },
        "range": [1, 10000],
        "description": "Number of agents in MCMP simulation"
    },
    
    "max_iterations": {
        "defaults": {
            "MCPMRetriever": 50,
            "UI_Settings": 60,
            "Realtime_Server": 200,
            "mcmp_runner": 20  # quick search default
        },
        "range": [1, 5000],
        "description": "Maximum simulation iterations"
    },
    
    "exploration_bonus": {
        "defaults": {
            "MCPMRetriever": 0.1,
            "UI_Settings": 0.1,
            "Realtime_Server": 0.1,
            "mcmp_runner": 0.1
        },
        "range": [0.01, 1.0],
        "description": "Controls agent exploration randomness (used in spawning)"
    },
    
    "exploration_factor": {
        "default": 0.3,
        "description": "Agent-specific exploration factor (different from exploration_bonus!)",
        "note": "Set per agent, ranges from exploration_factor_min to exploration_bonus"
    },
    
    "pheromone_decay": {
        "defaults": {
            "MCPMRetriever": 0.95,
            "UI_Settings": 0.95,
            "Realtime_Server": 0.95,
            "mcmp_runner": 0.95
        },
        "range": [0.5, 0.999],
        "description": "Rate at which pheromone trails fade each iteration"
    },
    
    # PERFORMANCE & PROCESSING
    "embed_batch_size": {
        "defaults": {
            "MCPMRetriever": 128,
            "Realtime_Server": 128,
            "UI_Settings": 64  # Named 'embed_bs'
        },
        "range": [1, 4096],
        "description": "Batch size for embedding computation"
    },
    
    "chunk_workers": {
        "default": 4,
        "formula": "max(4, os.cpu_count() or 8)",
        "corpus_formula": "max(4, min(32, (os.cpu_count() or 8) * 2))",
        "range": [1, 128],
        "description": "Number of parallel workers for processing"
    },
    
    "max_chunks_per_shard": {
        "defaults": {
            "UI_Settings": 2000,
            "Realtime_Server": 2000
        },
        "range": [0, 100000],
        "description": "Maximum chunks per processing shard"
    },
    
    "max_files": {
        "defaults": {
            "UI_Settings": 1000,
            "Realtime_Server": 500,
            "_collect_py_documents": 300,
            "mcmp_runner": 200  # quick search
        },
        "range": [0, 20000],
        "description": "Maximum files to process from repository"
    },
    
    # VISUALIZATION PARAMETERS
    "redraw_every": {
        "default": 2,
        "range": [1, 100],
        "description": "Visualization update frequency (iterations)"
    },
    
    "min_trail_strength": {
        "defaults": {
            "Realtime_Server": 0.05,
            "MCPMRetriever.get_snapshot": 0.05,
            "visualize.build_snapshot": 0.05
        },
        "range": [0.0, 1.0],
        "description": "Minimum pheromone trail strength to display"
    },
    
    "max_edges": {
        "defaults": {
            "Realtime_Server": 600,
            "MCPMRetriever.get_snapshot": 300,
            "visualize.build_snapshot": 300
        },
        "range": [10, 5000],
        "description": "Maximum edges in network visualization"
    },
    
    "viz_dims": {
        "default": 2,
        "options": [2, 3],
        "description": "Visualization dimensions (2D or 3D)"
    },
    
    # SEARCH & RETRIEVAL
    "top_k": {
        "defaults": {
            "UI_Settings": 5,
            "Realtime_Server": 10,
            "Search_Algorithm": 5,
            "MCPMRetriever.search": 10
        },
        "range": [1, 200],
        "description": "Number of top results to return"
    },
    
    "alpha": {
        "default": 0.7,
        "range": [0.0, 1.0],
        "description": "Hybrid search weighting (semantic vs keyword)"
    },
    
    "div_alpha": {
        "default": 0.7,
        "range": [0.0, 1.0],
        "description": "Diversity weighting in result selection"
    },
    
    "dedup_tau": {
        "default": 0.92,
        "range": [0.0, 1.0],
        "description": "Deduplication threshold (cosine similarity)"
    },
    
    "per_folder_cap": {
        "default": 2,
        "description": "Maximum results per folder"
    },
    
    # CORPUS & FILE PROCESSING
    "exclude_dirs": {
        "default": [".venv", "node_modules", ".git", "external"],
        "description": "Directories to exclude from processing"
    },
    
    "windows": {
        "defaults": {
            "UI_Settings": [],
            "Realtime_Server": [],
            "corpus_default": [100, 200, 300]
        },
        "description": "List of chunk window sizes (in lines)"
    },
    
    "lines_per_chunk": {
        "default": 20,
        "description": "Lines per chunk in fallback chunking"
    },
    
    "max_chars_per_file": {
        "default": 4000,
        "description": "Maximum characters to read per file"
    },
    
    "encoding": {
        "default": "utf-8",
        "description": "File encoding for reading"
    },
    
    # MODEL CONFIGURATION
    "embedding_model_name": {
        "default": "google/embeddinggemma-300m",
        "description": "Hugging Face embedding model"
    },
    
    "device_mode": {
        "default": "auto",
        "options": ["auto", "cuda", "cpu"],
        "description": "Device preference for model inference"
    },
    
    "use_embedding_model": {
        "default": True,
        "description": "Whether to use embedding model"
    },
    
    "build_faiss_after_add": {
        "default": True,
        "description": "Build FAISS index after adding documents"
    },
    
    # LLM SETTINGS
    "llm_model": {
        "default": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "description": "LLM model for text generation"
    },
    
    "llm_device": {
        "default": "auto",
        "options": ["auto", "cuda", "cpu"],
        "description": "Device for LLM inference"
    },
    
    "use_ollama": {
        "default": False,
        "description": "Use Ollama instead of Hugging Face"
    },
    
    "ollama_model": {
        "default": "qwen2.5-coder:7b",
        "description": "Ollama model name"
    },
    
    "ollama_host": {
        "default": "http://127.0.0.1:11434",
        "description": "Ollama server URL"
    },
    
    "timeout": {
        "defaults": {
            "ollama_generation": 500,
            "ui_queries": 180
        },
        "description": "Request timeout in seconds"
    },
    
    "temperature": {
        "default": 0.1,
        "description": "LLM generation temperature"
    },
    
    # DATABASE CONFIGURATION
    "qdrant_url": {
        "default": "http://localhost:6337",
        "description": "Qdrant vector database URL"
    },
    
    "qdrant_api_key": {
        "default": None,
        "description": "Optional Qdrant API key"
    },
    
    "collection_name": {
        "default": "codebase",
        "description": "Qdrant collection name"
    },
    
    "persist_dir": {
        "default": "./enterprise_index",
        "description": "Directory for persistent storage"
    },
    
    # REPORTING & LOGGING
    "report_enabled": {
        "default": False,
        "description": "Enable background reporting"
    },
    
    "report_every": {
        "default": 5,
        "range": [1, 100],
        "description": "Report generation frequency"
    },
    
    "report_mode": {
        "default": "deep",
        "options": ["deep", "structure", "exploratory", "summary", "repair"],
        "description": "Report generation mode"
    },
    
    "log_every": {
        "defaults": {
            "MCPMRetriever": 1,
            "UI_Settings": 10,
            "simulation_debug": 10  # max(1, int(getattr(retr, 'log_every', 10)))
        },
        "description": "Logging frequency for debug output"
    },
    
    # UI DISPLAY SETTINGS
    "mode": {
        "default": "deep",
        "options": ["deep", "structure", "exploratory", "summary", "repair"],
        "description": "Search mode in UI"
    },
    
    "show_tree": {
        "default": True,
        "description": "Show tree visualization in UI"
    },
    
    "show_network": {
        "default": True,
        "description": "Show network visualization in UI"
    },
    
    "gen_answer": {
        "default": False,
        "description": "Generate LLM answers in UI"
    },
    
    "pure_topk": {
        "default": False,
        "description": "Use pure top-k without diversity"
    },
    
    "use_repo": {
        "default": True,
        "description": "Use repository files vs uploaded docs"
    },
    
    "root_folder": {
        "defaults": {
            "UI_Settings": "",
            "Realtime_Server": ".",  # os.getcwd()
            "Codespace_Analyzer": "src"
        },
        "description": "Root folder for file processing"
    },
    
    "docs_file": {
        "default": "",
        "description": "Path to documents file"
    },
    
    "query": {
        "default": "Explain the architecture.",
        "description": "Default search query"
    },
    
    # KEYWORD SEARCH SETTINGS
    "kw_lambda": {
        "default": 0.0,
        "description": "Keyword scoring weight in relevance calculation"
    },
    
    "kw_terms": {
        "default": "set()",
        "description": "Set of keyword search terms"
    },
    
    # AGENT & DOCUMENT PROPERTIES
    "energy": {
        "default": 1.0,
        "description": "Agent energy level"
    },
    
    "trail_strength": {
        "default": 1.0,
        "description": "Agent pheromone trail deposit strength"
    },
    
    "age": {
        "default": 0,
        "description": "Agent age (iterations lived)"
    },
    
    "relevance_score": {
        "default": 0.0,
        "description": "Document relevance score"
    },
    
    "visit_count": {
        "default": 0,
        "description": "Number of times document was visited"
    },
    
    "last_visited": {
        "default": 0.0,
        "description": "Timestamp of last document visit"
    }
}

# =============================================================================
# CONFIGURATION BY LOCATION
# =============================================================================

CONFIGURATION_BY_FILE = {
    "src/embeddinggemma/mcmp_rag.py": {
        "MCPMRetriever.__init__": [
            "embedding_model_name", "num_agents", "max_iterations", 
            "pheromone_decay", "exploration_bonus", "device_mode",
            "use_embedding_model", "embed_batch_size", "build_faiss_after_add"
        ],
        "Agent dataclass": [
            "energy", "trail_strength", "exploration_factor", "age"
        ],
        "Document dataclass": [
            "relevance_score", "visit_count", "last_visited"
        ],
        "get_visualization_snapshot": [
            "min_trail_strength", "max_edges", "method", "whiten", "spread", "jitter", "dims"
        ]
    },
    
    "src/embeddinggemma/realtime/server.py": {
        "SnapshotStreamer.__init__": [
            "redraw_every", "min_trail_strength", "max_edges", "viz_dims",
            "query", "use_repo", "root_folder", "max_files", "exclude_dirs",
            "windows", "chunk_workers", "max_iterations", "num_agents",
            "exploration_bonus", "pheromone_decay", "embed_batch_size",
            "max_chunks_per_shard", "top_k", "report_enabled", "report_every", "report_mode"
        ],
        "SettingsModel (Pydantic)": [
            "All above with validation ranges"
        ],
        "_collect_py_documents": [
            "max_files", "max_chars"
        ]
    },
    
    "src/embeddinggemma/ui/state.py": {
        "Settings dataclass": [
            "mode", "top_k", "windows", "use_repo", "root_folder", "max_files",
            "exclude_dirs", "docs_file", "num_agents", "max_iterations",
            "show_tree", "show_network", "gen_answer", "div_alpha", "dedup_tau",
            "per_folder_cap", "pure_topk", "log_every", "exploration_bonus",
            "pheromone_decay", "embed_bs", "max_chunks_per_shard"
        ]
    },
    
    "src/embeddinggemma/rag/config.py": {
        "RagSettings dataclass": [
            "qdrant_url", "qdrant_api_key", "collection_name", "embedding_model",
            "llm_model", "llm_device", "use_ollama", "ollama_model", 
            "ollama_host", "persist_dir"
        ]
    },
    
    "src/embeddinggemma/rag/search.py": {
        "hybrid_search function": [
            "top_k", "alpha"
        ]
    },
    
    "src/embeddinggemma/rag/chunking.py": {
        "fallback_chunking function": [
            "lines_per_chunk"
        ]
    },
    
    "src/embeddinggemma/rag/generation.py": {
        "generate_with_ollama function": [
            "timeout"
        ]
    },
    
    "src/embeddinggemma/rag/vectorstore.py": {
        "_create_collection function": [
            "hnsw_m", "hnsw_ef_construct", "indexing_threshold", 
            "memmap_threshold", "quantization_bits", "always_ram"
        ]
    },
    
    "src/embeddinggemma/mcmp/simulation.py": {
        "Physics constants (hardcoded)": [
            "spawn_noise_std", "spawn_velocity_std", "exploration_factor_min",
            "attraction_weight", "pheromone_weight", "exploration_weight",
            "velocity_decay", "force_application", "visit_bonus_multiplier",
            "visit_bonus_max", "time_bonus", "time_window", "trail_deposit_multiplier",
            "trail_min_strength", "trail_memory_length", "relevance_threshold"
        ]
    },
    
    "src/embeddinggemma/ui/mcmp_runner.py": {
        "select_diverse_results function": [
            "top_k", "alpha", "dedup_tau", "per_folder_cap", "content_length"
        ],
        "quick_search_with_mcmp defaults": [
            "num_agents", "max_iterations", "exploration_bonus", "pheromone_decay", "embed_bs"
        ]
    },
    
    "src/embeddinggemma/ui/corpus.py": {
        "collect_codebase_chunks function": [
            "max_workers_formula"
        ]
    },
    
    "src/embeddinggemma/ui/queries.py": {
        "_ollama_generate function": [
            "timeout", "temperature"
        ]
    }
}

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

TOTAL_COUNTS = {
    "unique_configurable_parameters": len(ALL_CONFIGURABLE_VALUES),
    "environment_variables": len(ENVIRONMENT_VARIABLES),
    "hardcoded_physics_constants": len(PHYSICS_CONSTANTS),
    "api_validated_parameters": len(API_VALIDATION_RANGES),
    "configuration_files": len(CONFIGURATION_BY_FILE),
    "default_value_variations": sum(
        len(v.get("defaults", {})) if isinstance(v.get("defaults"), dict) else 1 
        for v in ALL_CONFIGURABLE_VALUES.values()
    )
}

if __name__ == "__main__":
    print("🎛️ COMPLETE EmbeddingGemma Configuration Values")
    print("=" * 60)
    
    for category, count in TOTAL_COUNTS.items():
        print(f"📊 {category.replace('_', ' ').title()}: {count}")
    
    print(f"\n🔧 Total Unique Parameters: {TOTAL_COUNTS['unique_configurable_parameters']}")
    print(f"🌍 Environment Variables: {TOTAL_COUNTS['environment_variables']}")
    print(f"⚡ Hardcoded Constants: {TOTAL_COUNTS['hardcoded_physics_constants']}")
    print(f"✅ API Validated: {TOTAL_COUNTS['api_validated_parameters']}")
    
    print("\n🎯 Most Critical Parameters:")
    critical = ["num_agents", "max_iterations", "exploration_bonus", "pheromone_decay", 
               "top_k", "embed_batch_size", "max_files", "redraw_every"]
    
    for param in critical:
        if param in ALL_CONFIGURABLE_VALUES:
            info = ALL_CONFIGURABLE_VALUES[param]
            defaults = info.get("defaults", info.get("default", "varies"))
            print(f"  • {param}: {defaults}")