"""Settings management service - handles persistence and validation of application settings."""

from __future__ import annotations
from typing import Dict, Any
import os
import json
from pydantic import BaseModel, Field, validator

# Settings persistence directory
SETTINGS_DIR = os.path.join(os.getcwd(), ".fungus_cache")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")
os.makedirs(SETTINGS_DIR, exist_ok=True)


class SettingsModel(BaseModel):
    """Pydantic model for validating settings."""
    # visualization
    redraw_every: int | None = Field(default=None, ge=1, le=100)
    min_trail_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    max_edges: int | None = Field(default=None, ge=10, le=5000)
    viz_dims: int | None = Field(default=None)
    query: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=200)
    report_enabled: bool | None = None
    report_every: int | None = Field(default=None, ge=1, le=100)
    report_mode: str | None = None
    judge_mode: str | None = None
    # contextual steering weights and budgets
    alpha: float | None = Field(default=None, ge=0.0, le=2.0)
    beta: float | None = Field(default=None, ge=0.0, le=2.0)
    gamma: float | None = Field(default=None, ge=0.0, le=2.0)
    delta: float | None = Field(default=None, ge=0.0, le=2.0)
    epsilon: float | None = Field(default=None, ge=0.0, le=2.0)
    min_content_chars: int | None = Field(default=None, ge=0, le=20000)
    import_only_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    max_reports: int | None = Field(default=None, ge=0, le=1000)
    max_report_tokens: int | None = Field(default=None, ge=0, le=1000000)
    judge_enabled: bool | None = None
    # corpus
    use_repo: bool | None = None
    root_folder: str | None = None
    max_files: int | None = Field(default=None, ge=0, le=20000)
    exclude_dirs: list[str] | None = None
    windows: list[int] | None = None
    chunk_workers: int | None = Field(default=None, ge=1, le=128)
    # sim knobs
    max_iterations: int | None = Field(default=None, ge=1, le=5000)
    num_agents: int | None = Field(default=None, ge=1, le=10000)
    exploration_bonus: float | None = Field(default=None, ge=0.01, le=1.0)
    pheromone_decay: float | None = Field(default=None, ge=0.5, le=0.999)
    embed_batch_size: int | None = Field(default=None, ge=1, le=4096)
    max_chunks_per_shard: int | None = Field(default=None, ge=0, le=100000)
    # Vector backend
    vector_backend: str | None = None  # 'memory' | 'qdrant'
    qdrant_url: str | None = None
    qdrant_collection: str | None = None
    # run
    run_id: str | None = None
    # LLM (Ollama) configuration
    ollama_model: str | None = None
    ollama_host: str | None = None
    ollama_system: str | None = None
    ollama_num_gpu: int | None = Field(default=None, ge=0, le=128)
    ollama_num_thread: int | None = Field(default=None, ge=0, le=4096)
    ollama_num_batch: int | None = Field(default=None, ge=0, le=4096)
    # OpenAI
    llm_provider: str | None = None  # 'ollama' | 'openai'
    openai_model: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Google
    google_model: str | None = None
    google_api_key: str | None = None
    google_base_url: str | None = None
    google_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Grok
    grok_model: str | None = None
    grok_api_key: str | None = None
    grok_base_url: str | None = None
    grok_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Multi-query
    mq_enabled: bool | None = None
    mq_count: int | None = Field(default=None, ge=1, le=10)
    # Prompt overrides
    prompts_overrides: Dict[str, str] | None = None

    @validator('viz_dims')
    def _dims(cls, v):  # type: ignore
        if v is None:
            return 3
        return 3


def get_settings_dict(streamer: Any) -> dict:
    """
    Extract current settings from streamer object as a dictionary.

    Args:
        streamer: The SnapshotStreamer instance containing current settings

    Returns:
        Dictionary of all current settings
    """
    return {
        "query": streamer.query,
        "top_k": streamer.top_k,
        "report_enabled": streamer.report_enabled,
        "report_every": streamer.report_every,
        "report_mode": streamer.report_mode,
        "judge_mode": streamer.judge_mode,
        "alpha": streamer.alpha,
        "beta": streamer.beta,
        "gamma": streamer.gamma,
        "delta": streamer.delta,
        "epsilon": streamer.epsilon,
        "min_content_chars": streamer.min_content_chars,
        "import_only_penalty": streamer.import_only_penalty,
        "max_reports": streamer.max_reports,
        "max_report_tokens": streamer.max_report_tokens,
        "judge_enabled": streamer.judge_enabled,
        "redraw_every": streamer.redraw_every,
        "min_trail_strength": streamer.min_trail_strength,
        "max_edges": streamer.max_edges,
        "viz_dims": streamer.viz_dims,
        "use_repo": streamer.use_repo,
        "root_folder": streamer.root_folder,
        "max_files": streamer.max_files,
        "exclude_dirs": streamer.exclude_dirs,
        "windows": streamer.windows,
        "max_iterations": streamer.max_iterations,
        "num_agents": streamer.num_agents,
        "exploration_bonus": streamer.exploration_bonus,
        "pheromone_decay": streamer.pheromone_decay,
        "embed_batch_size": streamer.embed_batch_size,
        "max_chunks_per_shard": streamer.max_chunks_per_shard,
        # Vector backend
        "vector_backend": getattr(streamer, 'vector_backend', 'memory'),
        "qdrant_url": getattr(streamer, 'qdrant_url', ''),
        "qdrant_collection": getattr(streamer, 'qdrant_collection', ''),
        # LLM settings
        "ollama_model": streamer.ollama_model,
        "ollama_host": streamer.ollama_host,
        "ollama_system": streamer.ollama_system,
        "ollama_num_gpu": streamer.ollama_num_gpu,
        "ollama_num_thread": streamer.ollama_num_thread,
        "ollama_num_batch": streamer.ollama_num_batch,
        "llm_provider": streamer.llm_provider,
        "openai_model": streamer.openai_model,
        "openai_api_key": getattr(streamer, 'openai_api_key', ''),
        "openai_base_url": streamer.openai_base_url,
        "openai_temperature": streamer.openai_temperature,
        "google_model": streamer.google_model,
        "google_api_key": getattr(streamer, 'google_api_key', ''),
        "google_base_url": streamer.google_base_url,
        "google_temperature": streamer.google_temperature,
        "grok_model": streamer.grok_model,
        "grok_api_key": getattr(streamer, 'grok_api_key', ''),
        "grok_base_url": streamer.grok_base_url,
        "grok_temperature": streamer.grok_temperature,
        "mq_enabled": bool(getattr(streamer, 'mq_enabled', False)),
        "mq_count": min(int(getattr(streamer, 'mq_count', 5)), 3),
        "query_pool_cap": int(getattr(streamer, '_query_pool_cap', 100)),
        "token_cap": getattr(streamer, 'token_cap', None),
        "cost_cap_usd": getattr(streamer, 'cost_cap_usd', None),
        "stagnation_threshold": int(getattr(streamer, 'stagnation_threshold', 8)),
        # run
        "run_id": getattr(streamer, 'run_id', ''),
    }


def apply_settings_to_streamer(streamer: Any, settings_dict: dict) -> None:
    """
    Apply settings dictionary to streamer object.

    Args:
        streamer: The SnapshotStreamer instance to update
        settings_dict: Dictionary of settings to apply
    """
    try:
        sm = SettingsModel(**settings_dict)
        if sm.query is not None: streamer.query = sm.query
        if getattr(sm, 'top_k', None) is not None: streamer.top_k = int(getattr(sm, 'top_k'))
        if getattr(sm, 'report_enabled', None) is not None: streamer.report_enabled = bool(getattr(sm, 'report_enabled'))
        if getattr(sm, 'report_every', None) is not None: streamer.report_every = int(getattr(sm, 'report_every'))
        if getattr(sm, 'report_mode', None) is not None: streamer.report_mode = str(getattr(sm, 'report_mode'))
        if getattr(sm, 'alpha', None) is not None: streamer.alpha = float(getattr(sm, 'alpha'))
        if getattr(sm, 'beta', None) is not None: streamer.beta = float(getattr(sm, 'beta'))
        if getattr(sm, 'gamma', None) is not None: streamer.gamma = float(getattr(sm, 'gamma'))
        if getattr(sm, 'delta', None) is not None: streamer.delta = float(getattr(sm, 'delta'))
        if getattr(sm, 'epsilon', None) is not None: streamer.epsilon = float(getattr(sm, 'epsilon'))
        if getattr(sm, 'min_content_chars', None) is not None: streamer.min_content_chars = int(getattr(sm, 'min_content_chars'))
        if getattr(sm, 'import_only_penalty', None) is not None: streamer.import_only_penalty = float(getattr(sm, 'import_only_penalty'))
        if getattr(sm, 'max_reports', None) is not None: streamer.max_reports = int(getattr(sm, 'max_reports'))
        if getattr(sm, 'max_report_tokens', None) is not None: streamer.max_report_tokens = int(getattr(sm, 'max_report_tokens'))
        if getattr(sm, 'judge_enabled', None) is not None: streamer.judge_enabled = bool(getattr(sm, 'judge_enabled'))
        if sm.redraw_every is not None: streamer.redraw_every = int(sm.redraw_every)
        if sm.min_trail_strength is not None: streamer.min_trail_strength = float(sm.min_trail_strength)
        if sm.max_edges is not None: streamer.max_edges = int(sm.max_edges)
        if sm.viz_dims is not None: streamer.viz_dims = int(sm.viz_dims)
        if sm.use_repo is not None: streamer.use_repo = bool(sm.use_repo)
        if sm.root_folder is not None: streamer.root_folder = str(sm.root_folder)
        if sm.max_files is not None: streamer.max_files = int(sm.max_files)
        if sm.exclude_dirs is not None: streamer.exclude_dirs = [str(x) for x in sm.exclude_dirs]
        if sm.windows is not None: streamer.windows = [int(x) for x in sm.windows]
        if sm.max_iterations is not None: streamer.max_iterations = int(sm.max_iterations)
        if sm.num_agents is not None: streamer.num_agents = int(sm.num_agents)
        if sm.exploration_bonus is not None: streamer.exploration_bonus = float(sm.exploration_bonus)
        if sm.pheromone_decay is not None: streamer.pheromone_decay = float(sm.pheromone_decay)
        if sm.embed_batch_size is not None: streamer.embed_batch_size = int(sm.embed_batch_size)
        if sm.max_chunks_per_shard is not None: streamer.max_chunks_per_shard = int(sm.max_chunks_per_shard)
        # judge/report mode and LLM config
        if getattr(sm, 'judge_mode', None) is not None: streamer.judge_mode = str(getattr(sm, 'judge_mode'))
        if getattr(sm, 'ollama_model', None) is not None: streamer.ollama_model = str(getattr(sm, 'ollama_model'))
        if getattr(sm, 'ollama_host', None) is not None: streamer.ollama_host = str(getattr(sm, 'ollama_host'))
        if getattr(sm, 'ollama_system', None) is not None: streamer.ollama_system = str(getattr(sm, 'ollama_system'))
        if getattr(sm, 'ollama_num_gpu', None) is not None: streamer.ollama_num_gpu = int(getattr(sm, 'ollama_num_gpu'))
        if getattr(sm, 'ollama_num_thread', None) is not None: streamer.ollama_num_thread = int(getattr(sm, 'ollama_num_thread'))
        if getattr(sm, 'ollama_num_batch', None) is not None: streamer.ollama_num_batch = int(getattr(sm, 'ollama_num_batch'))
        if getattr(sm, 'mq_enabled', None) is not None: streamer.mq_enabled = bool(getattr(sm, 'mq_enabled'))
        if getattr(sm, 'mq_count', None) is not None: streamer.mq_count = int(getattr(sm, 'mq_count'))
        # vector backend
        if getattr(sm, 'vector_backend', None) is not None: streamer.vector_backend = str(getattr(sm, 'vector_backend'))
        if getattr(sm, 'qdrant_url', None) is not None: streamer.qdrant_url = str(getattr(sm, 'qdrant_url'))
        if getattr(sm, 'qdrant_collection', None) is not None: streamer.qdrant_collection = str(getattr(sm, 'qdrant_collection'))
        # run id
        if getattr(sm, 'run_id', None) is not None: streamer.run_id = str(getattr(sm, 'run_id'))
        if getattr(sm, 'llm_provider', None) is not None: streamer.llm_provider = str(getattr(sm, 'llm_provider'))
        if getattr(sm, 'openai_model', None) is not None: streamer.openai_model = str(getattr(sm, 'openai_model'))
        if getattr(sm, 'openai_api_key', None) is not None: streamer.openai_api_key = str(getattr(sm, 'openai_api_key'))
        if getattr(sm, 'openai_base_url', None) is not None: streamer.openai_base_url = str(getattr(sm, 'openai_base_url'))
        if getattr(sm, 'openai_temperature', None) is not None: streamer.openai_temperature = float(getattr(sm, 'openai_temperature'))
        if getattr(sm, 'google_model', None) is not None: streamer.google_model = str(getattr(sm, 'google_model'))
        if getattr(sm, 'google_api_key', None) is not None: streamer.google_api_key = str(getattr(sm, 'google_api_key'))
        if getattr(sm, 'google_base_url', None) is not None: streamer.google_base_url = str(getattr(sm, 'google_base_url'))
        if getattr(sm, 'google_temperature', None) is not None: streamer.google_temperature = float(getattr(sm, 'google_temperature'))
        if getattr(sm, 'grok_model', None) is not None: streamer.grok_model = str(getattr(sm, 'grok_model'))
        if getattr(sm, 'grok_api_key', None) is not None: streamer.grok_api_key = str(getattr(sm, 'grok_api_key'))
        if getattr(sm, 'grok_base_url', None) is not None: streamer.grok_base_url = str(getattr(sm, 'grok_base_url'))
        if getattr(sm, 'grok_temperature', None) is not None: streamer.grok_temperature = float(getattr(sm, 'grok_temperature'))
    except Exception:
        pass


def load_settings_from_disk(streamer: Any) -> None:
    """
    Load settings from disk and apply to streamer.

    Args:
        streamer: The SnapshotStreamer instance to update
    """
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings_dict = json.load(f)
                apply_settings_to_streamer(streamer, settings_dict)
    except Exception:
        pass


def save_settings_to_disk(streamer: Any) -> None:
    """
    Save current settings to disk.

    Args:
        streamer: The SnapshotStreamer instance containing settings to save
    """
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(get_settings_dict(streamer), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_settings_usage_lines(settings_dict: dict) -> list[str]:
    """
    Produce human-readable mapping of settings to their consuming scripts.

    Args:
        settings_dict: Dictionary of settings

    Returns:
        List of strings describing setting usage
    """
    usage_map = {
        "query": ["mcmp_rag.py (initialize_simulation/search)", "realtime/server.py (/start)"],
        "viz_dims": ["mcmp_rag.py (get_visualization_snapshot)", "frontend (Plotly 2D/3D)"],
        "min_trail_strength": ["mcmp/visualize.py (build_snapshot)", "mcmp_rag.py (get_visualization_snapshot)"],
        "max_edges": ["mcmp/visualize.py (build_snapshot)"],
        "redraw_every": ["realtime/server.py (_run_loop WS cadence)"],
        "num_agents": ["mcmp/simulation.py (spawn_agents)", "mcmp_rag.py (init MCPMRetriever)"],
        "max_iterations": ["realtime/server.py (jobs/start)", "streamlit_fungus_backup.py (loop)"],
        "exploration_bonus": ["mcmp/simulation.py (noise/force)"],
        "pheromone_decay": ["mcmp/simulation.py (decay_pheromones)"],
        "embed_batch_size": ["mcmp_rag.py (add_documents batched encode)"],
        "max_chunks_per_shard": ["realtime/server.py (jobs/start sharding)"],
        "use_repo": ["realtime/server.py (/start corpus path)"],
        "root_folder": ["realtime/server.py (/start corpus path)"],
        "max_files": ["ui/corpus.py (list_code_files)"],
        "exclude_dirs": ["ui/corpus.py (list_code_files)"],
        "windows": ["ui/corpus.py (chunk_python_file windows)"],
        "chunk_workers": ["ui/corpus.py (ThreadPoolExecutor)"],
        "top_k": ["mcmp_rag.py (search top_k)"],
        "report_enabled": ["realtime/server.py (_run_loop report)"],
        "report_every": ["realtime/server.py (_run_loop report cadence)"],
        "report_mode": ["realtime/server.py (prompt template)"],
        "judge_mode": ["realtime/server.py (judge prompt)"],
        "ollama_model": ["realtime/server.py (LLM model)"],
        "ollama_host": ["realtime/server.py (LLM host)"],
        "ollama_system": ["realtime/server.py (LLM system prompt)"],
        "ollama_num_gpu": ["realtime/server.py (LLM GPU opts)"],
        "ollama_num_thread": ["realtime/server.py (LLM CPU threads)"],
        "ollama_num_batch": ["realtime/server.py (LLM batch)"],
        "llm_provider": ["realtime/server.py (choose provider)"],
        "openai_model": ["realtime/server.py (OpenAI model)"],
        "openai_base_url": ["realtime/server.py (OpenAI endpoint)"],
        "openai_temperature": ["realtime/server.py (OpenAI temperature)"],
        "mode": ["frontend/UX (prompt style)", "streamlit_fungus_backup.py (mode prompt)"],
    }
    lines: list[str] = []
    for key, val in settings_dict.items():
        if key in usage_map:
            scripts = usage_map[key]
            lines.append(f"{key}: {val} -> Scripts: {', '.join(scripts)}")
    return lines


__all__ = [
    'SettingsModel',
    'SETTINGS_DIR',
    'SETTINGS_PATH',
    'get_settings_dict',
    'apply_settings_to_streamer',
    'load_settings_from_disk',
    'save_settings_to_disk',
    'get_settings_usage_lines',
]
