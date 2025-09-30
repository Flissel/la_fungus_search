"""
EmbeddingGemma Configuration Playground Reference
================================================

This file contains all configurable parameters found in the EmbeddingGemma codebase.
Use this as a reference for understanding what can be adjusted.

⚠️ This file is for REFERENCE ONLY - not included in the application!

Categories:
- Environment Variables
- MCMP Simulation Parameters  
- RAG Configuration
- UI/Frontend Settings
- Realtime Server Settings
- LLM Configuration
- Physics & Force Parameters
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

class EnvironmentVariables:
    """All environment variables used across the codebase"""
    
    # RAG/Database Configuration
    QDRANT_URL: str = "http://localhost:6337"
    QDRANT_API_KEY: Optional[str] = None
    RAG_COLLECTION: str = "codebase"
    RAG_PERSIST_DIR: str = "./enterprise_index"
    
    # Model Configuration
    EMBED_MODEL: str = "google/embeddinggemma-300m"
    RAG_LLM_MODEL: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    RAG_LLM_DEVICE: str = "auto"  # auto|cuda|cpu
    
    # Ollama Configuration
    RAG_USE_OLLAMA: str = "0"  # "1" for true
    OLLAMA_MODEL: str = "qwen2.5-coder:7b"
    OLLAMA_HOST: str = "http://127.0.0.1:11434"


# =============================================================================
# MCMP RETRIEVER CONFIGURATION
# =============================================================================

@dataclass
class MCPMRetrieverConfig:
    """MCPMRetriever initialization parameters"""
    
    # Model Settings
    embedding_model_name: str = "google/embeddinggemma-300m"
    device_mode: str = "auto"  # auto|cuda|cpu
    use_embedding_model: bool = True
    embed_batch_size: int = 128
    build_faiss_after_add: bool = True
    
    # Agent Configuration
    num_agents: int = 200
    max_iterations: int = 50
    exploration_bonus: float = 0.1  # Controls agent exploration randomness
    
    # Pheromone Trail Settings
    pheromone_decay: float = 0.95  # How fast pheromone trails fade
    
    # Internal Settings
    log_every: int = 1  # Logging frequency
    kw_lambda: float = 0.0  # Keyword scoring weight


# =============================================================================
# SIMULATION PHYSICS PARAMETERS
# =============================================================================

@dataclass 
class SimulationPhysicsConfig:
    """Hardcoded physics parameters in simulation.py"""
    
    # Agent Spawning
    spawn_noise_std: float = 0.1  # Standard deviation for spawn position noise
    spawn_velocity_std: float = 0.05  # Standard deviation for initial velocity
    exploration_factor_min: float = 0.05  # Minimum exploration factor
    
    # Force Weights (in update_agent_position)
    attraction_weight: float = 0.8   # Weight for document attraction force
    pheromone_weight: float = 0.15   # Weight for pheromone trail force  
    exploration_weight: float = 0.05 # Weight for random exploration force
    
    # Velocity Update
    velocity_decay: float = 0.85     # Velocity momentum decay
    force_application: float = 0.15  # How much new forces affect velocity
    
    # Document Relevance Scoring
    visit_bonus_multiplier: float = 0.1  # Bonus per document visit
    visit_bonus_max: float = 0.5         # Maximum visit bonus
    time_bonus: float = 0.1              # Bonus for recently visited docs
    time_window: float = 1.0             # Time window for recency bonus (seconds)
    
    # Pheromone Trail Management
    trail_deposit_multiplier: float = 0.1  # Pheromone deposit amount multiplier
    trail_min_strength: float = 0.01       # Minimum trail strength before removal
    trail_memory_length: int = 3           # How many previous docs to link
    
    # Network Extraction
    relevance_threshold: float = 0.1       # Minimum relevance for network inclusion
    content_preview_length: int = 100      # Characters in node content preview


# =============================================================================
# RAG CONFIGURATION  
# =============================================================================

@dataclass
class RagSettings:
    """RAG system configuration from rag/config.py"""
    
    # Database
    qdrant_url: str = "http://localhost:6337"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "codebase"
    persist_dir: str = "./enterprise_index"
    
    # Models
    embedding_model: str = "google/embeddinggemma-300m"
    llm_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    llm_device: str = "auto"  # auto|cuda|cpu
    
    # Ollama Integration
    use_ollama: bool = False
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_host: str = "http://127.0.0.1:11434"


# =============================================================================
# UI SETTINGS
# =============================================================================

@dataclass
class UISettings:
    """UI configuration from ui/state.py"""
    
    # Search Configuration
    mode: str = "deep"  # Search mode
    top_k: int = 5      # Number of top results to return
    windows: List[int] = field(default_factory=lambda: [])  # Chunk window sizes
    
    # Repository Settings
    use_repo: bool = True
    root_folder: str = ""
    max_files: int = 1000
    exclude_dirs: List[str] = field(default_factory=lambda: [".venv", "node_modules", ".git", "external"])
    docs_file: str = ""
    
    # MCMP Settings (UI overrides)
    num_agents: int = 200
    max_iterations: int = 60
    exploration_bonus: float = 0.1
    pheromone_decay: float = 0.95
    embed_bs: int = 64
    max_chunks_per_shard: int = 2000
    
    # Display Settings
    show_tree: bool = True
    show_network: bool = True
    gen_answer: bool = False
    
    # Search Processing
    div_alpha: float = 0.7      # Diversity weighting
    dedup_tau: float = 0.92     # Deduplication threshold
    per_folder_cap: int = 2     # Max results per folder
    pure_topk: bool = False     # Use pure top-k without diversity
    log_every: int = 10         # Logging frequency


# =============================================================================
# REALTIME SERVER CONFIGURATION
# =============================================================================

@dataclass
class RealtimeServerConfig:
    """Realtime server configuration from realtime/server.py"""
    
    # Visualization Settings
    redraw_every: int = 2              # Visualization update frequency
    min_trail_strength: float = 0.05   # Minimum pheromone trail strength to display
    max_edges: int = 600               # Maximum edges in visualization
    viz_dims: int = 2                  # Visualization dimensions (2 or 3)
    
    # Default Query
    query: str = "Explain the architecture."
    
    # Corpus Configuration  
    use_repo: bool = True
    root_folder: str = "."  # Will be set to os.getcwd()
    max_files: int = 500
    exclude_dirs: List[str] = field(default_factory=lambda: [".venv", "node_modules", ".git", "external"])
    windows: List[int] = field(default_factory=list)  # Must come from frontend
    chunk_workers: int = 4  # Will be max(4, os.cpu_count() or 8)
    
    # Simulation Configuration
    max_iterations: int = 200
    num_agents: int = 200
    exploration_bonus: float = 0.1
    pheromone_decay: float = 0.95
    embed_batch_size: int = 128
    max_chunks_per_shard: int = 2000
    
    # Results Configuration
    top_k: int = 10
    
    # Reporting Configuration
    report_enabled: bool = False
    report_every: int = 5
    report_mode: str = "deep"


# =============================================================================
# REALTIME API SETTINGS MODEL (Pydantic)
# =============================================================================

class RealtimeAPISettings:
    """Settings that can be configured via API (with validation ranges)"""
    
    # Visualization (with ranges)
    redraw_every: int = 2              # Range: 1-100
    min_trail_strength: float = 0.05   # Range: 0.0-1.0  
    max_edges: int = 600               # Range: 10-5000
    viz_dims: int = 2                  # Must be 2 or 3
    query: str = ""
    top_k: int = 10                    # Range: 1-200
    
    # Reporting
    report_enabled: bool = False
    report_every: int = 5              # Range: 1-100
    report_mode: str = "deep"
    
    # Corpus Settings
    use_repo: bool = True
    root_folder: str = ""
    max_files: int = 500               # Range: 0-20000
    exclude_dirs: List[str] = []
    windows: List[int] = []
    chunk_workers: int = 4             # Range: 1-128
    
    # Simulation Parameters  
    max_iterations: int = 200          # Range: 1-5000
    num_agents: int = 200              # Range: 1-10000
    exploration_bonus: float = 0.1     # Range: 0.01-1.0
    pheromone_decay: float = 0.95      # Range: 0.5-0.999
    embed_batch_size: int = 128        # Range: 1-4096
    max_chunks_per_shard: int = 2000   # Range: 0-100000


# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

@dataclass
class SearchConfig:
    """Search algorithm configuration from rag/search.py"""
    
    # Hybrid Search Parameters
    top_k: int = 5           # Number of results to return
    alpha: float = 0.7       # Semantic vs keyword search weighting
    semantic_multiplier: int = 2  # Retrieve top_k * 2 for semantic search


# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

@dataclass  
class ChunkingConfig:
    """Code chunking configuration from rag/chunking.py"""
    
    # Fallback Chunking
    lines_per_chunk: int = 20  # Lines per chunk in fallback mode
    
    # File Processing
    encoding: str = "utf-8"    # File encoding
    
    # AST Processing
    include_ast_metadata: bool = True  # Include AST node information


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Agent dataclass fields from mcmp_rag.py"""
    
    # Core Properties
    id: int
    position: Any  # np.ndarray
    velocity: Any  # np.ndarray
    energy: float = 1.0
    trail_strength: float = 1.0
    exploration_factor: float = 0.1
    age: int = 0
    visited_docs: set = field(default_factory=set)


# =============================================================================
# DOCUMENT CONFIGURATION  
# =============================================================================

@dataclass
class DocumentConfig:
    """Document dataclass fields from mcmp_rag.py"""
    
    # Core Properties
    id: int
    content: str
    embedding: Optional[Any] = None  # np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Dynamic Properties (added during simulation)
    relevance_score: float = 0.0
    visit_count: int = 0
    last_visited: float = 0.0


# =============================================================================
# EMBEDDING BACKEND CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingBackendConfig:
    """Embedding backend settings from rag/embeddings.py"""
    
    model_name: str = "google/embeddinggemma-300m"
    device_preference: str = "auto"  # auto|cuda|cpu


# =============================================================================
# CODESPACE ANALYZER CONFIGURATION
# =============================================================================

@dataclass
class CodespaceAnalyzerConfig:
    """Codespace analyzer settings from codespace_analyzer.py"""
    
    target_folder: str = "src"  # Default folder to analyze


# =============================================================================
# COMPLETE PARAMETER SUMMARY
# =============================================================================

class AllConfigurableParameters:
    """
    Complete reference of all configurable parameters in EmbeddingGemma
    
    This class serves as a comprehensive catalog of every parameter that can be
    adjusted in the system, organized by component and usage context.
    """
    
    # CORE SIMULATION PARAMETERS
    # -------------------------
    # These control the fundamental behavior of the MCMP simulation
    
    num_agents: int = 200              # Number of agents in simulation [1-10000]
    max_iterations: int = 50-200       # Simulation steps [1-5000]  
    exploration_bonus: float = 0.1     # Agent exploration randomness [0.01-1.0]
    pheromone_decay: float = 0.95      # Trail decay rate [0.5-0.999]
    
    # PHYSICS FORCE WEIGHTS
    # --------------------
    # These control how different forces affect agent movement
    
    attraction_weight: float = 0.8     # Document attraction force weight
    pheromone_weight: float = 0.15     # Pheromone trail force weight
    exploration_weight: float = 0.05   # Random exploration force weight
    velocity_decay: float = 0.85       # Velocity momentum decay
    force_application: float = 0.15    # Force application rate
    
    # AGENT SPAWNING PARAMETERS
    # ------------------------
    
    spawn_noise_std: float = 0.1       # Position noise when spawning
    spawn_velocity_std: float = 0.05   # Initial velocity noise
    exploration_factor_min: float = 0.05  # Minimum exploration factor
    
    # DOCUMENT RELEVANCE SCORING
    # -------------------------
    
    visit_bonus_multiplier: float = 0.1  # Bonus per document visit
    visit_bonus_max: float = 0.5         # Maximum visit bonus
    time_bonus: float = 0.1              # Bonus for recently visited
    time_window: float = 1.0             # Recency window (seconds)
    kw_lambda: float = 0.0               # Keyword scoring weight
    
    # PHEROMONE TRAIL MANAGEMENT
    # -------------------------
    
    trail_deposit_multiplier: float = 0.1  # Pheromone deposit amount
    trail_min_strength: float = 0.01       # Minimum before removal
    trail_memory_length: int = 3           # Previous docs to link
    
    # VISUALIZATION PARAMETERS
    # -----------------------
    
    redraw_every: int = 2              # Visualization update frequency [1-100]
    min_trail_strength: float = 0.05   # Minimum trail strength to show [0.0-1.0]
    max_edges: int = 600               # Maximum edges in visualization [10-5000]
    viz_dims: int = 2                  # Visualization dimensions (2 or 3)
    relevance_threshold: float = 0.1   # Minimum relevance for network
    content_preview_length: int = 100  # Characters in preview
    
    # SEARCH & RETRIEVAL
    # -----------------
    
    top_k: int = 5-10                  # Results to return [1-200]
    alpha: float = 0.7                 # Semantic vs keyword weight
    div_alpha: float = 0.7             # Diversity weighting  
    dedup_tau: float = 0.92            # Deduplication threshold
    pure_topk: bool = False            # Use pure top-k vs diversity
    
    # CORPUS PROCESSING
    # ----------------
    
    max_files: int = 500-1000          # Maximum files to process [0-20000]
    exclude_dirs: List[str] = [".venv", "node_modules", ".git", "external"]
    chunk_workers: int = 4-8           # Parallel workers [1-128]
    embed_batch_size: int = 64-128     # Embedding batch size [1-4096]
    max_chunks_per_shard: int = 2000   # Chunks per shard [0-100000]
    per_folder_cap: int = 2            # Max results per folder
    
    # CHUNKING CONFIGURATION
    # ---------------------
    
    windows: List[int] = []            # Chunk window sizes (lines)
    lines_per_chunk: int = 20          # Fallback chunking size
    encoding: str = "utf-8"            # File encoding
    
    # REPORTING & LOGGING
    # ------------------
    
    report_enabled: bool = False       # Enable background reporting
    report_every: int = 5              # Report frequency [1-100]
    report_mode: str = "deep"          # Report generation mode
    log_every: int = 1-10              # Logging frequency
    
    # MODEL & DEVICE SETTINGS
    # ----------------------
    
    embedding_model_name: str = "google/embeddinggemma-300m"
    device_mode: str = "auto"          # auto|cuda|cpu
    use_embedding_model: bool = True
    build_faiss_after_add: bool = True
    
    # LLM CONFIGURATION
    # ----------------
    
    llm_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    llm_device: str = "auto"           # auto|cuda|cpu
    use_ollama: bool = False
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_host: str = "http://127.0.0.1:11434"
    
    # DATABASE SETTINGS
    # ----------------
    
    qdrant_url: str = "http://localhost:6337"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "codebase"
    persist_dir: str = "./enterprise_index"


# =============================================================================
# PARAMETER USAGE MAPPING
# =============================================================================

PARAMETER_USAGE_MAP = {
    # Core simulation parameters used across multiple files
    "num_agents": [
        "mcmp/simulation.py (spawn_agents)", 
        "mcmp_rag.py (init MCPMRetriever)",
        "realtime/server.py (SnapshotStreamer)",
        "ui/state.py (Settings)"
    ],
    "max_iterations": [
        "realtime/server.py (simulation loop)", 
        "mcmp_rag.py (MCPMRetriever)",
        "ui/state.py (Settings)"
    ],
    "exploration_bonus": [
        "mcmp/simulation.py (agent exploration factor)",
        "mcmp_rag.py (MCPMRetriever)", 
        "realtime/server.py (SnapshotStreamer)",
        "ui/state.py (Settings)"
    ],
    "pheromone_decay": [
        "mcmp/simulation.py (decay_pheromones)",
        "mcmp_rag.py (MCPMRetriever)",
        "realtime/server.py (SnapshotStreamer)", 
        "ui/state.py (Settings)"
    ],
    "embed_batch_size": [
        "mcmp_rag.py (add_documents batched encode)",
        "realtime/server.py (SnapshotStreamer)",
        "ui/state.py (Settings as embed_bs)"
    ],
    
    # Visualization parameters
    "redraw_every": ["realtime/server.py (visualization update frequency)"],
    "min_trail_strength": ["realtime/server.py (pheromone trail display threshold)"],
    "max_edges": ["realtime/server.py (visualization edge limit)"],
    "viz_dims": ["realtime/server.py (2D vs 3D visualization)"],
    
    # Search parameters
    "top_k": ["realtime/server.py", "ui/state.py", "rag/search.py"],
    "alpha": ["rag/search.py (semantic vs keyword weight)"],
    "div_alpha": ["ui/state.py (diversity weighting)"],
    "dedup_tau": ["ui/state.py (deduplication threshold)"],
    
    # Corpus processing
    "max_files": ["realtime/server.py", "ui/state.py"],
    "exclude_dirs": ["realtime/server.py", "ui/state.py"],
    "chunk_workers": ["realtime/server.py"],
    "max_chunks_per_shard": ["realtime/server.py", "ui/state.py"],
    
    # Force physics (hardcoded in simulation.py)
    "attraction_weight": ["mcmp/simulation.py (update_agent_position)"],
    "pheromone_weight": ["mcmp/simulation.py (update_agent_position)"],
    "exploration_weight": ["mcmp/simulation.py (update_agent_position)"],
    "velocity_decay": ["mcmp/simulation.py (update_agent_position)"],
    "force_application": ["mcmp/simulation.py (update_agent_position)"],
}


# =============================================================================
# CONFIGURATION TIPS & RECOMMENDATIONS
# =============================================================================

class ConfigurationTips:
    """
    Guidelines for tuning parameters
    """
    
    PERFORMANCE_TUNING = {
        "For faster simulation": {
            "num_agents": "Reduce to 50-100 for faster processing",
            "max_iterations": "Reduce to 20-30 for quick results", 
            "embed_batch_size": "Increase to 256-512 if you have GPU memory",
            "chunk_workers": "Set to os.cpu_count() for parallel processing"
        },
        
        "For better accuracy": {
            "num_agents": "Increase to 300-500 for more thorough exploration",
            "max_iterations": "Increase to 100-200 for convergence",
            "exploration_bonus": "Reduce to 0.05 for less randomness",
            "pheromone_decay": "Increase to 0.98 for longer trail memory"
        },
        
        "For visualization": {
            "redraw_every": "Increase to 5-10 for smoother animation",
            "min_trail_strength": "Decrease to 0.01 to show more trails",
            "max_edges": "Increase to 1000+ for denser networks",
            "viz_dims": "Use 3 for complex datasets, 2 for simplicity"
        }
    }
    
    FORCE_TUNING = {
        "More exploration": {
            "exploration_weight": "Increase from 0.05 to 0.1-0.2",
            "exploration_bonus": "Increase from 0.1 to 0.2-0.3",
            "velocity_decay": "Reduce from 0.85 to 0.7-0.8"
        },
        
        "More focused search": {
            "attraction_weight": "Increase from 0.8 to 0.9",
            "pheromone_weight": "Increase from 0.15 to 0.2-0.3",
            "exploration_weight": "Reduce from 0.05 to 0.01-0.02"
        },
        
        "Stronger pheromone influence": {
            "pheromone_weight": "Increase from 0.15 to 0.3-0.5",
            "pheromone_decay": "Increase from 0.95 to 0.98-0.99",
            "trail_deposit_multiplier": "Increase from 0.1 to 0.2-0.3"
        }
    }
    
    CORPUS_TUNING = {
        "Large codebases": {
            "max_files": "Increase to 2000-5000",
            "max_chunks_per_shard": "Increase to 5000-10000", 
            "chunk_workers": "Set to cpu_count()",
            "embed_batch_size": "Increase to 256-512"
        },
        
        "Small focused search": {
            "max_files": "Reduce to 100-200",
            "per_folder_cap": "Increase to 5-10",
            "top_k": "Reduce to 3-5",
            "num_agents": "Reduce to 50-100"
        }
    }


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Fast Development Configuration
FAST_CONFIG = {
    "num_agents": 50,
    "max_iterations": 20,
    "exploration_bonus": 0.15,
    "pheromone_decay": 0.9,
    "redraw_every": 5,
    "top_k": 5,
    "max_files": 200,
    "embed_batch_size": 64
}

# High Accuracy Configuration  
ACCURACY_CONFIG = {
    "num_agents": 400,
    "max_iterations": 150,
    "exploration_bonus": 0.05,
    "pheromone_decay": 0.98,
    "redraw_every": 2,
    "top_k": 15,
    "max_files": 2000,
    "embed_batch_size": 256
}

# Visualization-Focused Configuration
VISUAL_CONFIG = {
    "redraw_every": 1,
    "min_trail_strength": 0.01,
    "max_edges": 1500,
    "viz_dims": 3,
    "num_agents": 300,
    "pheromone_decay": 0.97
}

# Large Codebase Configuration
LARGE_CODEBASE_CONFIG = {
    "max_files": 5000,
    "max_chunks_per_shard": 10000,
    "chunk_workers": 16,
    "embed_batch_size": 512,
    "num_agents": 500,
    "max_iterations": 200
}


if __name__ == "__main__":
    print("EmbeddingGemma Configuration Playground Reference")
    print("=" * 50)
    print(f"Total configurable parameter categories: {len([cls for cls in globals().values() if isinstance(cls, type) and cls.__name__.endswith('Config')])}")
    print(f"Environment variables: {len([attr for attr in dir(EnvironmentVariables) if not attr.startswith('_')])}")
    print(f"Parameter usage mappings: {len(PARAMETER_USAGE_MAP)}")
    print("\nUse this file as reference for understanding what can be configured!")