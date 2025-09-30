"""
All Configurable Values in EmbeddingGemma
=========================================

This file lists EVERY configurable value found in the codebase,
organized by component and showing current defaults.

🎛️ Use this as your configuration reference!
"""

# =============================================================================
# ENVIRONMENT VARIABLES (External Configuration)
# =============================================================================

ENVIRONMENT_VARIABLES = {
    # Database Configuration
    "QDRANT_URL": "http://localhost:6337",
    "QDRANT_API_KEY": None,  # Optional
    "RAG_COLLECTION": "codebase",
    "RAG_PERSIST_DIR": "./enterprise_index",
    
    # Model Configuration
    "EMBED_MODEL": "google/embeddinggemma-300m",
    "RAG_LLM_MODEL": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "RAG_LLM_DEVICE": "auto",  # auto|cuda|cpu
    
    # Ollama Configuration
    "RAG_USE_OLLAMA": "0",  # "1" for true, "0" for false
    "OLLAMA_MODEL": "qwen2.5-coder:7b",
    "OLLAMA_HOST": "http://127.0.0.1:11434"
}

# =============================================================================
# MCMP RETRIEVER INITIALIZATION PARAMETERS
# =============================================================================

MCMP_RETRIEVER_DEFAULTS = {
    "embedding_model_name": "google/embeddinggemma-300m",
    "num_agents": 200,
    "max_iterations": 50,
    "pheromone_decay": 0.95,
    "exploration_bonus": 0.1,
    "device_mode": "auto",  # auto|cuda|cpu
    "use_embedding_model": True,
    "embed_batch_size": 128,
    "build_faiss_after_add": True
}

# =============================================================================
# UI SETTINGS (Streamlit Interface)
# =============================================================================

UI_SETTINGS_DEFAULTS = {
    "mode": "deep",
    "top_k": 5,
    "windows": [],  # List of chunk window sizes
    "use_repo": True,
    "root_folder": "",
    "max_files": 1000,
    "exclude_dirs": [".venv", "node_modules", ".git", "external"],
    "docs_file": "",
    "num_agents": 200,
    "max_iterations": 60,
    "show_tree": True,
    "show_network": True,
    "gen_answer": False,
    "div_alpha": 0.7,
    "dedup_tau": 0.92,
    "per_folder_cap": 2,
    "pure_topk": False,
    "log_every": 10,
    "exploration_bonus": 0.1,
    "pheromone_decay": 0.95,
    "embed_bs": 64,
    "max_chunks_per_shard": 2000
}

# =============================================================================
# REALTIME SERVER CONFIGURATION
# =============================================================================

REALTIME_SERVER_DEFAULTS = {
    # Visualization
    "redraw_every": 2,
    "min_trail_strength": 0.05,
    "max_edges": 600,
    "viz_dims": 2,
    "query": "Explain the architecture.",
    
    # Corpus
    "use_repo": True,
    "root_folder": ".",  # os.getcwd()
    "max_files": 500,
    "exclude_dirs": [".venv", "node_modules", ".git", "external"],
    "windows": [],  # Must come from frontend
    "chunk_workers": 4,  # max(4, os.cpu_count() or 8)
    
    # Simulation
    "max_iterations": 200,
    "num_agents": 200,
    "exploration_bonus": 0.1,
    "pheromone_decay": 0.95,
    "embed_batch_size": 128,
    "max_chunks_per_shard": 2000,
    
    # Results
    "top_k": 10,
    
    # Reporting
    "report_enabled": False,
    "report_every": 5,
    "report_mode": "deep"
}

# =============================================================================
# API VALIDATION RANGES (Pydantic Model Limits)
# =============================================================================

API_VALIDATION_RANGES = {
    # Visualization
    "redraw_every": {"min": 1, "max": 100, "default": 2},
    "min_trail_strength": {"min": 0.0, "max": 1.0, "default": 0.05},
    "max_edges": {"min": 10, "max": 5000, "default": 600},
    "viz_dims": {"allowed": [2, 3], "default": 2},
    "top_k": {"min": 1, "max": 200, "default": 10},
    "report_every": {"min": 1, "max": 100, "default": 5},
    
    # Corpus
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
# HARDCODED PHYSICS PARAMETERS (Currently Not Configurable)
# =============================================================================

PHYSICS_CONSTANTS = {
    # Agent Spawning (simulation.py:spawn_agents)
    "spawn_noise_std": 0.1,
    "spawn_velocity_std": 0.05,
    "exploration_factor_min": 0.05,
    
    # Force Weights (simulation.py:update_agent_position)
    "attraction_weight": 0.8,
    "pheromone_weight": 0.15,
    "exploration_weight": 0.05,
    
    # Velocity Update (simulation.py:update_agent_position)
    "velocity_decay": 0.85,
    "force_application": 0.15,
    
    # Document Relevance (simulation.py:update_document_relevance)
    "visit_bonus_multiplier": 0.1,
    "visit_bonus_max": 0.5,
    "time_bonus": 0.1,
    "time_window": 1.0,  # seconds
    
    # Pheromone Trails (simulation.py:deposit_pheromones)
    "trail_deposit_multiplier": 0.1,
    "trail_min_strength": 0.01,
    "trail_memory_length": 3,
    
    # Network Extraction (simulation.py:extract_relevance_network)
    "relevance_threshold": 0.1,
    "content_preview_length": 100
}

# =============================================================================
# SEARCH ALGORITHM PARAMETERS
# =============================================================================

SEARCH_DEFAULTS = {
    "top_k": 5,
    "alpha": 0.7,  # Semantic vs keyword weighting
    "semantic_multiplier": 2  # Retrieve top_k * 2 for semantic search
}

# =============================================================================
# CHUNKING PARAMETERS
# =============================================================================

CHUNKING_DEFAULTS = {
    "lines_per_chunk": 20,  # Fallback chunking
    "encoding": "utf-8",
    "include_ast_metadata": True
}

# =============================================================================
# AGENT DEFAULTS
# =============================================================================

AGENT_DEFAULTS = {
    "energy": 1.0,
    "trail_strength": 1.0,
    "exploration_factor": 0.1,
    "age": 0,
    "visited_docs": "set()"  # Empty set
}

# =============================================================================
# DOCUMENT DEFAULTS
# =============================================================================

DOCUMENT_DEFAULTS = {
    "relevance_score": 0.0,
    "visit_count": 0,
    "last_visited": 0.0,
    "embedding": None,
    "metadata": {}
}

# =============================================================================
# EMBEDDING BACKEND PARAMETERS
# =============================================================================

EMBEDDING_DEFAULTS = {
    "model_name": "google/embeddinggemma-300m",
    "device_preference": "auto"  # auto|cuda|cpu
}

# =============================================================================
# CODESPACE ANALYZER PARAMETERS
# =============================================================================

CODESPACE_ANALYZER_DEFAULTS = {
    "target_folder": "src"
}

# =============================================================================
# COMPLETE CONFIGURABLE VALUES SUMMARY
# =============================================================================

ALL_CONFIGURABLE_VALUES = {
    
    # CORE SIMULATION CONTROL
    "num_agents": {
        "default": 200,
        "range": [1, 10000],
        "description": "Number of agents in MCMP simulation",
        "locations": ["MCPMRetriever", "UI Settings", "Realtime Server"]
    },
    
    "max_iterations": {
        "default": 50,  # Varies: 50 (MCMP), 60 (UI), 200 (Realtime)
        "range": [1, 5000],
        "description": "Maximum simulation iterations before stopping",
        "locations": ["MCPMRetriever", "UI Settings", "Realtime Server"]
    },
    
    "exploration_bonus": {
        "default": 0.1,
        "range": [0.01, 1.0],
        "description": "Controls agent exploration randomness",
        "locations": ["MCPMRetriever", "UI Settings", "Realtime Server"]
    },
    
    "pheromone_decay": {
        "default": 0.95,
        "range": [0.5, 0.999],
        "description": "Rate at which pheromone trails fade",
        "locations": ["MCPMRetriever", "UI Settings", "Realtime Server"]
    },
    
    # PERFORMANCE SETTINGS
    "embed_batch_size": {
        "default": 128,  # Varies: 128 (MCMP/Realtime), 64 (UI)
        "range": [1, 4096],
        "description": "Batch size for embedding computation",
        "locations": ["MCPMRetriever", "UI Settings", "Realtime Server"]
    },
    
    "chunk_workers": {
        "default": 4,
        "range": [1, 128],
        "description": "Number of parallel workers for chunking",
        "locations": ["Realtime Server"]
    },
    
    "max_chunks_per_shard": {
        "default": 2000,
        "range": [0, 100000],
        "description": "Maximum chunks per processing shard",
        "locations": ["UI Settings", "Realtime Server"]
    },
    
    # VISUALIZATION CONTROL
    "redraw_every": {
        "default": 2,
        "range": [1, 100],
        "description": "Visualization update frequency (iterations)",
        "locations": ["Realtime Server"]
    },
    
    "min_trail_strength": {
        "default": 0.05,
        "range": [0.0, 1.0],
        "description": "Minimum pheromone strength to display",
        "locations": ["Realtime Server"]
    },
    
    "max_edges": {
        "default": 600,
        "range": [10, 5000],
        "description": "Maximum edges in network visualization",
        "locations": ["Realtime Server"]
    },
    
    "viz_dims": {
        "default": 2,
        "options": [2, 3],
        "description": "Visualization dimensions (2D or 3D)",
        "locations": ["Realtime Server"]
    },
    
    # SEARCH & RETRIEVAL
    "top_k": {
        "default": 5,  # Varies: 5 (UI), 10 (Realtime), 5 (Search)
        "range": [1, 200],
        "description": "Number of top results to return",
        "locations": ["UI Settings", "Realtime Server", "Search Algorithm"]
    },
    
    "alpha": {
        "default": 0.7,
        "range": [0.0, 1.0],
        "description": "Semantic vs keyword search weighting",
        "locations": ["Search Algorithm"]
    },
    
    "div_alpha": {
        "default": 0.7,
        "range": [0.0, 1.0],
        "description": "Diversity weighting in results",
        "locations": ["UI Settings"]
    },
    
    "dedup_tau": {
        "default": 0.92,
        "range": [0.0, 1.0],
        "description": "Deduplication threshold (Jaccard similarity)",
        "locations": ["UI Settings"]
    },
    
    # CORPUS PROCESSING
    "max_files": {
        "default": 500,  # Varies: 1000 (UI), 500 (Realtime)
        "range": [0, 20000],
        "description": "Maximum files to process from repository",
        "locations": ["UI Settings", "Realtime Server"]
    },
    
    "exclude_dirs": {
        "default": [".venv", "node_modules", ".git", "external"],
        "description": "Directories to exclude from processing",
        "locations": ["UI Settings", "Realtime Server"]
    },
    
    "windows": {
        "default": [],
        "description": "List of chunk window sizes (in lines)",
        "locations": ["UI Settings", "Realtime Server"]
    },
    
    "per_folder_cap": {
        "default": 2,
        "description": "Maximum results per folder",
        "locations": ["UI Settings"]
    },
    
    "lines_per_chunk": {
        "default": 20,
        "description": "Lines per chunk in fallback chunking",
        "locations": ["Chunking Algorithm"]
    },
    
    # MODEL CONFIGURATION
    "embedding_model_name": {
        "default": "google/embeddinggemma-300m",
        "description": "Hugging Face model name for embeddings",
        "locations": ["MCPMRetriever", "Environment Variables"]
    },
    
    "device_mode": {
        "default": "auto",
        "options": ["auto", "cuda", "cpu"],
        "description": "Device preference for model inference",
        "locations": ["MCPMRetriever", "Embedding Backend"]
    },
    
    "use_embedding_model": {
        "default": True,
        "description": "Whether to use embedding model",
        "locations": ["MCPMRetriever"]
    },
    
    "build_faiss_after_add": {
        "default": True,
        "description": "Build FAISS index after adding documents",
        "locations": ["MCPMRetriever"]
    },
    
    # LLM SETTINGS
    "llm_model": {
        "default": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "description": "LLM model for text generation",
        "locations": ["RAG Settings", "Environment Variables"]
    },
    
    "llm_device": {
        "default": "auto",
        "options": ["auto", "cuda", "cpu"],
        "description": "Device for LLM inference",
        "locations": ["RAG Settings"]
    },
    
    "use_ollama": {
        "default": False,
        "description": "Use Ollama instead of Hugging Face",
        "locations": ["RAG Settings", "Environment Variables"]
    },
    
    "ollama_model": {
        "default": "qwen2.5-coder:7b",
        "description": "Ollama model name",
        "locations": ["RAG Settings", "Environment Variables"]
    },
    
    "ollama_host": {
        "default": "http://127.0.0.1:11434",
        "description": "Ollama server URL",
        "locations": ["RAG Settings", "Environment Variables"]
    },
    
    # DATABASE SETTINGS
    "qdrant_url": {
        "default": "http://localhost:6337",
        "description": "Qdrant vector database URL",
        "locations": ["RAG Settings", "Environment Variables"]
    },
    
    "qdrant_api_key": {
        "default": None,
        "description": "Optional Qdrant API key",
        "locations": ["RAG Settings", "Environment Variables"]
    },
    
    "collection_name": {
        "default": "codebase",
        "description": "Qdrant collection name",
        "locations": ["RAG Settings", "Environment Variables"]
    },
    
    "persist_dir": {
        "default": "./enterprise_index",
        "description": "Directory for persistent storage",
        "locations": ["RAG Settings", "Environment Variables"]
    },
    
    # REPORTING & LOGGING
    "report_enabled": {
        "default": False,
        "description": "Enable background reporting",
        "locations": ["Realtime Server"]
    },
    
    "report_every": {
        "default": 5,
        "range": [1, 100],
        "description": "Report generation frequency",
        "locations": ["Realtime Server"]
    },
    
    "report_mode": {
        "default": "deep",
        "description": "Report generation mode",
        "locations": ["Realtime Server"]
    },
    
    "log_every": {
        "default": 10,  # Varies: 1 (MCMP), 10 (UI)
        "description": "Logging frequency for debug output",
        "locations": ["UI Settings", "MCPMRetriever"]
    },
    
    # UI DISPLAY SETTINGS
    "mode": {
        "default": "deep",
        "description": "Search mode in UI",
        "locations": ["UI Settings"]
    },
    
    "show_tree": {
        "default": True,
        "description": "Show tree visualization in UI",
        "locations": ["UI Settings"]
    },
    
    "show_network": {
        "default": True,
        "description": "Show network visualization in UI",
        "locations": ["UI Settings"]
    },
    
    "gen_answer": {
        "default": False,
        "description": "Generate LLM answers in UI",
        "locations": ["UI Settings"]
    },
    
    "pure_topk": {
        "default": False,
        "description": "Use pure top-k without diversity",
        "locations": ["UI Settings"]
    },
    
    "use_repo": {
        "default": True,
        "description": "Use repository files vs uploaded docs",
        "locations": ["UI Settings", "Realtime Server"]
    },
    
    "root_folder": {
        "default": "",
        "description": "Root folder for file processing",
        "locations": ["UI Settings", "Realtime Server"]
    },
    
    "docs_file": {
        "default": "",
        "description": "Path to documents file",
        "locations": ["UI Settings"]
    },
    
    "target_folder": {
        "default": "src",
        "description": "Target folder for codespace analysis",
        "locations": ["Codespace Analyzer"]
    },
    
    "query": {
        "default": "Explain the architecture.",
        "description": "Default search query",
        "locations": ["Realtime Server"]
    },
    
    "encoding": {
        "default": "utf-8",
        "description": "File encoding for reading",
        "locations": ["Chunking Algorithm"]
    }
}

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

CONFIGURATION_STATS = {
    "total_configurable_values": len(ALL_CONFIGURABLE_VALUES),
    "environment_variables": len(ENVIRONMENT_VARIABLES),
    "api_validated_parameters": len(API_VALIDATION_RANGES),
    "hardcoded_physics_constants": len(PHYSICS_CONSTANTS),
    "components_with_config": [
        "MCPMRetriever",
        "UI Settings", 
        "Realtime Server",
        "RAG Settings",
        "Search Algorithm",
        "Chunking Algorithm",
        "Embedding Backend",
        "Codespace Analyzer"
    ]
}

if __name__ == "__main__":
    print("🎛️ EmbeddingGemma - All Configurable Values")
    print("=" * 50)
    print(f"📊 Total configurable values: {CONFIGURATION_STATS['total_configurable_values']}")
    print(f"🌍 Environment variables: {CONFIGURATION_STATS['environment_variables']}")
    print(f"✅ API validated parameters: {CONFIGURATION_STATS['api_validated_parameters']}")
    print(f"⚡ Hardcoded physics constants: {CONFIGURATION_STATS['hardcoded_physics_constants']}")
    print(f"🏗️ Components with configuration: {len(CONFIGURATION_STATS['components_with_config'])}")
    
    print("\n🔧 Most Important Configurable Values:")
    important_params = ["num_agents", "max_iterations", "exploration_bonus", "pheromone_decay", 
                       "top_k", "embed_batch_size", "max_files"]
    for param in important_params:
        if param in ALL_CONFIGURABLE_VALUES:
            info = ALL_CONFIGURABLE_VALUES[param]
            print(f"  • {param}: {info['default']} - {info['description']}")