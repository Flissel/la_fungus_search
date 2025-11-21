from __future__ import annotations
from typing import List, Dict, Any, Set
import os
import asyncio
import json
import hashlib
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import re
from pydantic import BaseModel, Field, validator

import numpy as np

from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.ui.corpus import collect_codebase_chunks, list_code_files  # type: ignore
from embeddinggemma.llm import generate_text  # type: ignore
from embeddinggemma.llm.config import load_config  # type: ignore
from embeddinggemma.llm.prompts import get_report_instructions
from embeddinggemma.modeprompts import deep as _pm_deep  # type: ignore
from embeddinggemma.modeprompts import structure as _pm_structure  # type: ignore
from embeddinggemma.modeprompts import exploratory as _pm_exploratory  # type: ignore
from embeddinggemma.modeprompts import summary as _pm_summary  # type: ignore
from embeddinggemma.modeprompts import repair as _pm_repair  # type: ignore
from embeddinggemma.modeprompts import steering as _pm_steering  # type: ignore
from embeddinggemma.prompts import _default_instructions as _base_default_instructions  # type: ignore
from embeddinggemma.llm.prompts import build_report_prompt as prompts_build_report_prompt
from embeddinggemma.llm.prompts import build_judge_prompt as prompts_build_judge_prompt
from embeddinggemma.ui.queries import dedup_multi_queries  # type: ignore
from embeddinggemma.ui.reports import merge_reports_to_summary  # type: ignore
from embeddinggemma.memory import SupermemoryManager  # type: ignore
from embeddinggemma.memory.supermemory_client import SupermemoryManagerSync  # type: ignore
from embeddinggemma.memory.room_analyzer import RoomAnalyzer  # type: ignore
from embeddinggemma.agents import MemoryManagerAgent  # type: ignore

# Import service modules for settings and prompts management
from embeddinggemma.realtime.services import settings_manager
from embeddinggemma.realtime.services.settings_manager import SETTINGS_DIR
from embeddinggemma.realtime.services import prompts_manager

# Load .env early so env vars are present before initializing streamer/config
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Initialize logger
_logger = logging.getLogger(__name__)


def _collect_py_documents(root_dir: str, max_files: int = 300, max_chars: int = 4000) -> List[str]:
    docs: List[str] = []
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            try:
                with open(os.path.join(dirpath, fn), "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                    if txt:
                        docs.append(txt[:max_chars])
                        count += 1
                        if count >= max_files:
                            return docs
            except Exception:
                continue
    return docs


def _sha1_of_file(path: str) -> str:
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return ""


def _parse_chunk_header_line(first_line: str) -> tuple[str | None, int | None, int | None, int | None]:
    try:
        if not first_line.startswith('# file:'):
            return None, None, None, None
        parts = [p.strip() for p in first_line[1:].split('|')]
        file_part = parts[0].split(':', 1)[1].strip() if len(parts) > 0 and ':' in parts[0] else None
        lines_part = parts[1].split(':', 1)[1].strip() if len(parts) > 1 and ':' in parts[1] else None
        win_part = parts[2].split(':', 1)[1].strip() if len(parts) > 2 and ':' in parts[2] else None
        a, b = None, None
        if lines_part and '-' in lines_part:
            try:
                a = int(lines_part.split('-')[0].strip())
                b = int(lines_part.split('-')[1].strip())
            except Exception:
                a, b = None, None
        w = None
        try:
            if win_part:
                w = int(win_part)
        except Exception:
            w = None
        return file_part, a, b, w
    except Exception:
        return None, None, None, None


def _report_schema_hint() -> str:
    # deprecated: kept for backward compatibility; use prompts.report_schema_hint()
    from embeddinggemma.prompts import report_schema_hint
    return report_schema_hint()


def _build_report_prompt(mode: str, query: str, top_k: int, docs: list[dict]) -> str:
    # deprecated shim to central prompts module
    return prompts_build_report_prompt(mode, query, top_k, docs)


class SnapshotStreamer:
    def __init__(self) -> None:
        self.retr: MCPMRetriever | None = None
        self.clients: Set[WebSocket] = set()
        self.running: bool = False
        self.task: asyncio.Task | None = None
        # visualization config
        self.redraw_every: int = 2
        self.min_trail_strength: float = 0.05
        self.max_edges: int = 1500
        self.viz_dims: int = 3
        self.query: str = "Classify the code into modules."
        # corpus config
        self.use_repo: bool = True
        self.root_folder: str = os.getcwd()
        self.max_files: int = 500
        self.exclude_dirs: list[str] = [".venv", "node_modules", ".git", "external"]
        # windows (chunk sizes in lines) must come from frontend; no hard-coded defaults
        self.windows: list[int] = []
        self.chunk_workers: int = max(4, (os.cpu_count() or 8))
        self.corpus_file_count: int = 0  # Track number of source files (separate from chunk count)
        # metrics
        self.step_i: int = 0
        self.last_metrics: dict[str, float | int] | None = None
        # sim config
        self.max_iterations: int = 200
        self.num_agents: int = 200
        self.exploration_bonus: float = 0.1
        self.pheromone_decay: float = 0.95
        self.embed_batch_size: int = 128
        self.max_chunks_per_shard: int = 2000
        # vector backend
        self.vector_backend: str = os.environ.get('VECTOR_BACKEND', 'memory')  # 'memory' | 'qdrant'
        self.qdrant_url: str = os.environ.get('QDRANT_URL', 'http://localhost:6339')
        self.qdrant_api_key: str | None = os.environ.get('QDRANT_API_KEY')
        # Generate timestamped collection name based on root directory
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Extract directory name from root_folder for collection naming
        root_dir_name = self._get_root_dir_name()
        self.qdrant_collection: str = f"{root_dir_name}_{timestamp}"
        print(f"[DEBUG] SnapshotStreamer init: root_dir_name='{root_dir_name}' collection='{self.qdrant_collection}'", flush=True)
        # jobs
        self.jobs: dict[str, dict] = {}
        # hot state save for pause/resume
        self._paused: bool = False
        self._saved_state: dict | None = None
        # results / stability
        self.top_k: int = 10
        self._avg_rel_history: list[float] = []
        # reporting config
        self.report_enabled: bool = False
        self.report_every: int = 5
        self.report_mode: str = "deep"
        self.judge_mode: str = "steering"
        # contextual steering (experimental)
        self.alpha: float = 0.7  # cosine
        self.beta: float = 0.1   # visit_norm
        self.gamma: float = 0.1  # trail_degree
        self.delta: float = 0.1  # LLM_vote
        self.epsilon: float = 0.0  # length prior (bm25-like)
        self.min_content_chars: int = 80
        self.import_only_penalty: float = 0.4
        self.max_reports: int = 20
        self.max_report_tokens: int = 20000
        self._reports_sent: int = 0
        self._tokens_used: int = 0
        self.judge_enabled: bool = True
        self._judge_cache: dict[int, dict] = {}
        self._llm_vote: dict[int, int] = {}
        self._doc_boost: dict[int, float] = {}
        self._query_pool: list[str] = []
        self._query_pool_cap: int = 100
        self.mq_enabled: bool = False
        self.mq_count: int = 2
        # run budgets
        self.token_cap: int | None = None  # total token budget for run (approx)
        self.cost_cap_usd: float | None = None
        self._stagnant_steps: int = 0
        self.stagnation_threshold: int = 8  # steps without new targets before pausing reports
        self._seeds_queue: list[int] = []
        # corpus fingerprint to detect changes
        self._corpus_fingerprint: str | None = None
        # Run artifacts - initialized in start()
        self.run_id: str | None = None
        self._run_dir: str | None = None
        self._query_log_path: str | None = None
        self._retrieval_log_path: str | None = None
        self._manifest_path: str | None = None
        # Corpus metadata path (shared across runs)
        self._corpus_dir = os.path.join(SETTINGS_DIR, "corpus")
        self._corpus_metadata_path = os.path.join(self._corpus_dir, "metadata.json")
        # Run manifest tracking - initialized in start()
        self._run_start_time: float | None = None
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._unique_docs_accessed: set[int] = set()
        # Goal-driven exploration tracking
        self.exploration_goal: str | None = None  # e.g., "architecture", "bugs", "security"
        self.exploration_phase: int = 0  # Current phase index
        self.exploration_mode: bool = False  # Whether goal-driven mode is active
        self._phase_discoveries: dict[str, list[str]] = {}  # Track discoveries per phase
        self._phase_files_accessed: dict[int, set[int]] = {}  # Files accessed per phase
        # LLM configuration defaults (centralized)
        try:
            _cfg = load_config()
        except Exception:
            _cfg = None
        # Apply central config with env fallback
        if _cfg is not None:
            self.llm_provider: str = _cfg.provider
            self.ollama_model: str = _cfg.ollama.model
            self.ollama_host: str = _cfg.ollama.host
            self.ollama_system: str | None = _cfg.ollama.system
            self.ollama_num_gpu: int | None = _cfg.ollama.num_gpu
            self.ollama_num_thread: int | None = _cfg.ollama.num_thread
            self.ollama_num_batch: int | None = _cfg.ollama.num_batch
            self.openai_model: str = _cfg.openai.model
            self.openai_api_key: str | None = _cfg.openai.api_key
            self.openai_base_url: str | None = _cfg.openai.base_url
            self.openai_temperature: float = float(_cfg.openai.temperature)
            self.google_model: str = _cfg.google.model
            self.google_api_key: str | None = _cfg.google.api_key
            self.google_base_url: str | None = _cfg.google.base_url
            self.google_temperature: float = float(_cfg.google.temperature)
            self.grok_model: str = _cfg.grok.model
            self.grok_api_key: str | None = _cfg.grok.api_key
            self.grok_base_url: str | None = _cfg.grok.base_url
            self.grok_temperature: float = float(_cfg.grok.temperature)
            # Ensure API keys fall back to env if missing in config
            if not self.openai_api_key:
                self.openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not self.google_api_key:
                self.google_api_key = os.environ.get('GOOGLE_API_KEY')
            if not self.grok_api_key:
                self.grok_api_key = os.environ.get('GROK_API_KEY')
        else:
            # Fallback to envs directly if config fails
            try:
                self.ollama_model: str = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
                self.ollama_host: str = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
                self.ollama_system: str | None = os.environ.get('OLLAMA_SYSTEM')
                self.ollama_num_gpu: int | None = int(os.environ.get('OLLAMA_NUM_GPU')) if os.environ.get('OLLAMA_NUM_GPU') else None
                self.ollama_num_thread: int | None = int(os.environ.get('OLLAMA_NUM_THREAD')) if os.environ.get('OLLAMA_NUM_THREAD') else None
                self.ollama_num_batch: int | None = int(os.environ.get('OLLAMA_NUM_BATCH')) if os.environ.get('OLLAMA_NUM_BATCH') else None
            except Exception:
                self.ollama_model = 'qwen2.5-coder:7b'
                self.ollama_host = 'http://127.0.0.1:11434'
                self.ollama_system = None
                self.ollama_num_gpu = None
                self.ollama_num_thread = None
                self.ollama_num_batch = None
            self.llm_provider: str = os.environ.get('LLM_PROVIDER', 'ollama')
            self.openai_model: str = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
            self.openai_api_key: str | None = os.environ.get('OPENAI_API_KEY')
            self.openai_base_url: str | None = os.environ.get('OPENAI_BASE_URL')
            self.openai_temperature: float = float(os.environ.get('OPENAI_TEMPERATURE', '0.0'))
            self.google_model: str = os.environ.get('GOOGLE_MODEL', 'gemini-1.5-pro')
            self.google_api_key: str | None = os.environ.get('GOOGLE_API_KEY')
            self.google_base_url: str | None = os.environ.get('GOOGLE_BASE_URL')
            self.google_temperature: float = float(os.environ.get('GOOGLE_TEMPERATURE', '0.0'))
            self.grok_model: str = os.environ.get('GROK_MODEL', 'grok-2-latest')
            self.grok_api_key: str | None = os.environ.get('GROK_API_KEY')
            self.grok_base_url: str | None = os.environ.get('GROK_BASE_URL')
            self.grok_temperature: float = float(os.environ.get('GROK_TEMPERATURE', '0.0'))

        # Initialize Supermemory for judge persistent memory
        self.memory_manager = SupermemoryManager()

        # Initialize RoomAnalyzer for automatic room discovery
        self.room_analyzer = RoomAnalyzer()

        # Initialize Memory Manager Agent for ingestion decisions (LEGACY - being replaced)
        # Will be configured with LLM client after first judge call
        self.memory_agent: MemoryManagerAgent | None = None

        # Initialize LangChain Memory Agent for incremental knowledge building (NEW)
        # Will be configured with LLM client on first exploration step
        self.langchain_agent = None

        # Conversation history tracking for Memory Manager Agent context
        self.conversation_history: list[dict[str, Any]] = []

    @staticmethod
    def _parse_json_loose(raw: str) -> dict:
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            # Strip code fences
            s = raw.strip()
            if s.startswith('```'):
                s = "\n".join([ln for ln in s.splitlines() if not ln.strip().startswith('```')])
            # Find first JSON object or array
            start_obj = s.find('{')
            start_arr = s.find('[')
            start = max(0, min([p for p in [start_obj, start_arr] if p != -1])) if (start_obj != -1 or start_arr != -1) else -1
            end_obj = s.rfind('}')
            end_arr = s.rfind(']')
            end = max(end_obj, end_arr)
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            pass
        return {}

    def _doc_by_id(self, doc_id: int):
        try:
            if self.retr is None:
                return None
            return next((x for x in getattr(self.retr, 'documents', []) if int(getattr(x, 'id', -1)) == int(doc_id)), None)
        except Exception:
            return None

    def _trail_degree_map(self) -> dict[int, int]:
        try:
            if self.retr is None:
                return {}
            trails = getattr(self.retr, 'pheromone_trails', {}) or {}
            deg: dict[int, int] = {}
            for (a, b), _s in trails.items():
                deg[int(a)] = deg.get(int(a), 0) + 1
                deg[int(b)] = deg.get(int(b), 0) + 1
            return deg
        except Exception:
            return {}

    def _is_import_only(self, content: str) -> bool:
        if not content:
            return False
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines:
            return False
        non_comments = [ln for ln in lines if not ln.startswith('#')]
        if not non_comments:
            return True
        code_like = [ln for ln in non_comments if not (ln.startswith('import ') or ln.startswith('from ') or ln.startswith('"""') or ln.startswith("'''"))]
        return len(code_like) == 0

    # ============================================================================
    # Exploration Mode Methods
    # ============================================================================

    def _check_phase_completion(self) -> bool:
        """Check if current exploration phase completion criteria are met."""
        if not self.exploration_mode or self.exploration_goal is None:
            return False

        try:
            from embeddinggemma.exploration import get_phase_info

            phase = get_phase_info(self.exploration_goal, self.exploration_phase)
            if not phase:
                return False

            criteria = phase.get("success_criteria", {})
            if not criteria:
                return True  # No criteria = always complete

            # Check each criterion
            for key, required_value in criteria.items():
                if key == "min_files":
                    # Count unique files accessed in this phase
                    phase_files = self._phase_files_accessed.get(self.exploration_phase, set())
                    if len(phase_files) < required_value:
                        return False

                elif key == "min_entry_points":
                    # Count entry points discovered
                    entry_points = self._phase_discoveries.get("entry_points", [])
                    if len(entry_points) < required_value:
                        return False

                elif key == "min_modules":
                    # Count unique modules discovered
                    modules = self._phase_discoveries.get("modules", [])
                    if len(modules) < required_value:
                        return False

                elif key == "min_patterns":
                    # Count design patterns discovered
                    patterns = self._phase_discoveries.get("patterns", [])
                    if len(patterns) < required_value:
                        return False

                elif key == "min_dependencies":
                    # Count unique dependencies/imports
                    imports = self._phase_discoveries.get("imports", [])
                    if len(set(imports)) < required_value:
                        return False

            return True  # All criteria met

        except Exception as e:
            _logger.warning(f"Error checking phase completion: {e}")
            return False

    async def _advance_exploration_phase(self) -> None:
        """Advance to the next exploration phase."""
        if not self.exploration_mode or self.exploration_goal is None:
            return

        try:
            from embeddinggemma.exploration import get_goal, get_initial_queries

            goal = get_goal(self.exploration_goal)
            if not goal:
                return

            # Mark current phase complete in report
            if hasattr(self, '_exploration_report'):
                self._exploration_report.mark_phase_complete(self.exploration_phase)

            # Advance to next phase
            self.exploration_phase += 1

            # Check if we've completed all phases
            if self.exploration_phase >= len(goal["phases"]):
                _logger.info(f"[EXPLORE] All phases completed for goal '{self.exploration_goal}'")
                await self._broadcast({
                    "type": "exploration_complete",
                    "goal": self.exploration_goal,
                    "total_phases": len(goal["phases"])
                })
                return

            # Get new phase info
            phase = goal["phases"][self.exploration_phase]
            phase_name = phase["name"]

            _logger.info(f"[EXPLORE] Advanced to phase {self.exploration_phase}: {phase_name}")

            # Seed query pool with new phase's initial queries
            initial_queries = get_initial_queries(self.exploration_goal, self.exploration_phase)
            for query in initial_queries:
                if query not in self._query_pool:
                    self._query_pool.append(query)

            # Broadcast phase change
            await self._broadcast({
                "type": "exploration_phase_change",
                "goal": self.exploration_goal,
                "phase_index": self.exploration_phase,
                "phase_name": phase_name,
                "queries_added": len(initial_queries)
            })

        except Exception as e:
            _logger.error(f"Error advancing exploration phase: {e}", exc_info=True)

    def _track_phase_discovery(self, category: str, item: dict | str) -> None:
        """Track a discovery for the current exploration phase."""
        if not self.exploration_mode:
            return

        try:
            # Add to phase-specific discoveries
            if category not in self._phase_discoveries:
                self._phase_discoveries[category] = []
            self._phase_discoveries[category].append(item)

            # Add to exploration report if available
            if hasattr(self, '_exploration_report'):
                self._exploration_report.add_phase_discovery(
                    self.exploration_phase,
                    category,
                    item
                )

        except Exception as e:
            _logger.warning(f"Error tracking phase discovery: {e}")

    def _track_phase_file_access(self, doc_id: int) -> None:
        """Track file access for phase completion criteria."""
        if not self.exploration_mode:
            return

        try:
            if self.exploration_phase not in self._phase_files_accessed:
                self._phase_files_accessed[self.exploration_phase] = set()
            self._phase_files_accessed[self.exploration_phase].add(doc_id)
        except Exception as e:
            _logger.warning(f"Error tracking file access: {e}")

    def _log_query(self, query: str, source: str = "user", step: int = 0, parent_query: str | None = None) -> None:
        """Log a query to queries.jsonl"""
        try:
            _logger.info(f"[ANALYTICS] Logging query: query='{query[:50]}...', source={source}, step={step}, run_dir={self._run_dir}")
            os.makedirs(self._run_dir, exist_ok=True)
            import time
            entry = {
                "timestamp": time.time(),
                "query": query,
                "source": source,  # "user" | "llm" | "followup"
                "step": step,
                "parent_query": parent_query
            }
            with open(self._query_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            _logger.info(f"[ANALYTICS] Query logged successfully to {self._query_log_path}")
        except Exception as e:
            _logger.warning(f"Failed to log query: {e}", exc_info=True)

    def _log_retrieval(self, query: str, results: list[dict], step: int, retrieval_time_ms: float) -> None:
        """Log retrieval results to retrievals.jsonl"""
        try:
            # Defensive type checking
            if not isinstance(results, list):
                _logger.warning(f"_log_retrieval received non-list type: {type(results)}")
                return

            os.makedirs(self._run_dir, exist_ok=True)
            import time
            # Extract doc IDs and scores
            doc_ids = [int(r.get('id', r.get('doc_id', -1))) for r in results if isinstance(r, dict)]
            scores = [float(r.get('score', 0.0)) for r in results if isinstance(r, dict)]

            entry = {
                "timestamp": time.time(),
                "query": query,
                "step": step,
                "doc_ids": doc_ids,
                "scores": scores,
                "count": len(results),
                "retrieval_time_ms": retrieval_time_ms
            }
            with open(self._retrieval_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            # Track unique docs for coverage
            for doc_id in doc_ids:
                if doc_id >= 0:
                    self._unique_docs_accessed.add(doc_id)
        except Exception as e:
            _logger.warning(f"Failed to log retrieval: {e}")

    def _update_manifest(self, tokens: int = 0, cost: float = 0.0) -> None:
        """Update run manifest with current stats"""
        try:
            os.makedirs(self._run_dir, exist_ok=True)
            import time

            # Update accumulators
            self._total_tokens += tokens
            self._total_cost += cost

            # Calculate coverage
            total_docs = len(getattr(self.retr, 'documents', [])) if self.retr else 0
            coverage = (len(self._unique_docs_accessed) / total_docs) if total_docs > 0 else 0.0

            # Calculate runtime
            runtime_seconds = 0.0
            if self._run_start_time:
                runtime_seconds = time.time() - self._run_start_time

            # Get memory stats if available
            memory_stats = {}
            if hasattr(self, 'memory_manager') and self.memory_manager:
                memory_stats = self.memory_manager.get_stats()

                # Add room-specific stats
                if hasattr(self, 'room_analyzer') and self.room_analyzer:
                    room_stats = self.room_analyzer.get_room_stats()
                    memory_stats.update({
                        "rooms_discovered": room_stats.get("total_rooms", 0),
                        "total_room_visits": room_stats.get("total_visits", 0),
                        "rooms_fully_explored": room_stats.get("fully_explored", 0),
                        "rooms_partially_explored": room_stats.get("partially_explored", 0),
                        "rooms_entry_only": room_stats.get("entry_only", 0)
                    })

                # Add Memory Manager Agent stats (LEGACY)
                if hasattr(self, 'memory_agent') and self.memory_agent:
                    agent_stats = self.memory_agent.get_stats()
                    memory_stats.update({
                        "legacy_agent_enabled": agent_stats.get("enabled", False),
                        "legacy_agent_decisions": agent_stats.get("decisions_made", 0),
                        "legacy_agent_documents_ingested": agent_stats.get("documents_ingested", 0),
                        "legacy_agent_search_more_decisions": agent_stats.get("search_more_decisions", 0),
                        "conversation_history_steps": len(getattr(self, 'conversation_history', []))
                    })

                # Add LangChain Memory Agent stats (NEW)
                if hasattr(self, 'langchain_agent') and self.langchain_agent:
                    langchain_stats = self.langchain_agent.get_stats()
                    memory_stats.update({
                        "langchain_agent_enabled": langchain_stats.get("enabled", False),
                        "langchain_agent_model": langchain_stats.get("model", "unknown"),
                        "langchain_iterations_processed": langchain_stats.get("iterations_processed", 0),
                        "langchain_memories_created": langchain_stats.get("memories_created", 0),
                        "langchain_memories_updated": langchain_stats.get("memories_updated", 0),
                        "langchain_skipped_iterations": langchain_stats.get("skipped_iterations", 0)
                    })

            manifest = {
                "run_id": self.run_id,
                "query": self.query,
                "start_time": self._run_start_time,
                "runtime_seconds": runtime_seconds,
                "total_tokens": self._total_tokens,
                "total_cost": self._total_cost,
                "corpus_size": total_docs,
                "docs_accessed": len(self._unique_docs_accessed),
                "coverage_percent": round(coverage * 100, 2),
                "llm_provider": self.llm_provider,
                "llm_model": getattr(self, f"{self.llm_provider}_model", "unknown"),
                "memory_stats": memory_stats,
                "updated_at": time.time()
            }

            with open(self._manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _logger.warning(f"Failed to update manifest: {e}")

    def _save_corpus_metadata(self) -> None:
        """Save corpus metadata (document list with IDs, paths, sizes) to shared corpus directory."""
        try:
            if not self.retr or not hasattr(self.retr, 'documents'):
                _logger.warning("No retriever or documents available for corpus metadata")
                return

            os.makedirs(self._corpus_dir, exist_ok=True)
            import time
            import hashlib

            documents = getattr(self.retr, 'documents', [])
            doc_list = []

            for idx, doc in enumerate(documents):
                # Extract document metadata
                doc_id = idx
                content = getattr(doc, 'content', '')
                path = getattr(doc, 'path', '') or getattr(doc, 'metadata', {}).get('path', f'doc_{idx}')

                # Calculate content hash for deduplication checks
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

                doc_entry = {
                    "id": doc_id,
                    "path": path,
                    "content_length": len(content),
                    "content_hash": content_hash,
                    "has_embedding": hasattr(doc, 'embedding') and doc.embedding is not None
                }

                # Add optional metadata if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_entry["metadata"] = doc.metadata

                doc_list.append(doc_entry)

            # Save corpus metadata
            corpus_metadata = {
                "total_documents": len(doc_list),
                "created_at": time.time(),
                "fingerprint": self._corpus_fingerprint,
                "documents": doc_list
            }

            with open(self._corpus_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(corpus_metadata, f, indent=2, ensure_ascii=False)

            _logger.info(f"Corpus metadata saved: {len(doc_list)} documents to {self._corpus_metadata_path}")

        except Exception as e:
            _logger.warning(f"Failed to save corpus metadata: {e}")

    def _compute_blended_topk(self, results: list[dict]) -> list[dict]:
        if not isinstance(results, list):
            return []
        deg_map = self._trail_degree_map()
        blended: list[dict] = []
        for it in results:
            try:
                doc_id = int(it.get('id', it.get('doc_id', -1)))
                cosine = float(it.get('score', 0.0))
                content = str(it.get('content', ''))
                visits = 0.0
                if self.retr is not None:
                    d = self._doc_by_id(doc_id)
                    if d is not None:
                        visits = float(getattr(d, 'visit_count', 0))
                visit_norm = visits / max(1.0, visits + 10.0)
                trail_degree = float(deg_map.get(doc_id, 0))
                trail_degree = trail_degree / max(1.0, trail_degree + 10.0)
                llm_vote = float(self._llm_vote.get(doc_id, 0))  # -1,0,1
                len_prior = min(1.0, float(len(content)) / 800.0)
                # penalties
                if self._is_import_only(content):
                    len_prior *= (1.0 - float(self.import_only_penalty))
                if len(content) < int(self.min_content_chars):
                    len_prior *= 0.5
                boost = float(self._doc_boost.get(doc_id, 0.0))
                score = (
                    float(self.alpha) * cosine +
                    float(self.beta) * visit_norm +
                    float(self.gamma) * trail_degree +
                    float(self.delta) * llm_vote +
                    float(self.epsilon) * len_prior +
                    0.05 * boost
                )
                out = dict(it)
                out['blended_score'] = float(score)
                blended.append(out)
            except Exception:
                continue
        blended.sort(key=lambda x: x.get('blended_score', 0.0), reverse=True)
        return blended[: int(self.top_k)]

    async def _check_bootstrap_needed(self) -> bool:
        """
        Check if foundational bootstrap has been run for this container.

        Returns:
            True if bootstrap is needed, False if already exists
        """
        try:
            if not self.memory_manager or not self.memory_manager.enabled:
                return False

            container_tag = self.run_id or "default"

            # Search for codebase_module_tree memory (foundational bootstrap marker)
            memories = await self.memory_manager.search_memory(
                query="codebase_module_tree",
                container_tag=container_tag,
                limit=1
            )

            # If found and marked as auto-generated, bootstrap is done
            if memories:
                metadata = memories[0].get("metadata", {})
                if metadata.get("auto_generated"):
                    _logger.debug("[BOOTSTRAP] Foundational knowledge already exists")
                    return False

            # No bootstrap found, need to run
            _logger.info("[BOOTSTRAP] No foundational knowledge found, bootstrap needed")
            return True

        except Exception as e:
            _logger.warning(f"[BOOTSTRAP] Error checking bootstrap status: {e}")
            # On error, assume bootstrap is needed
            return True

    async def _run_bootstrap(self) -> None:
        """
        Run codebase bootstrap to create foundational knowledge.

        This scans the project structure and creates initial memories about:
        - Module tree (all Python packages)
        - Entry points (main executable files)
        - Module overviews for significant modules
        """
        try:
            if not self.memory_manager or not self.memory_manager.enabled:
                _logger.debug("[BOOTSTRAP] Skipped - memory manager disabled")
                return

            from embeddinggemma.memory.codebase_bootstrap import CodebaseBootstrap

            container_tag = self.run_id or "default"
            root_dir = os.getcwd()

            _logger.info("[BOOTSTRAP] Creating foundational codebase knowledge...")
            _ = asyncio.create_task(self._broadcast({
                "type": "log",
                "message": "bootstrap: scanning codebase structure..."
            }))

            # Create and run bootstrap
            bootstrapper = CodebaseBootstrap(
                root_dir=root_dir,
                memory_manager=self.memory_manager
            )

            result = await bootstrapper.bootstrap(container_tag=container_tag)

            if result.get('success'):
                memories_created = result.get('memories_created', 0)
                modules_count = len(result.get('module_tree', {}))
                entry_points_count = len(result.get('entry_points', []))

                _logger.info(
                    f"[BOOTSTRAP] Success - Created {memories_created} foundational memories "
                    f"({modules_count} modules, {entry_points_count} entry points)"
                )

                _ = asyncio.create_task(self._broadcast({
                    "type": "log",
                    "message": f"bootstrap: created {memories_created} foundational memories ({modules_count} modules)"
                }))
            else:
                error = result.get('error', 'Unknown error')
                _logger.error(f"[BOOTSTRAP] Failed: {error}")
                _ = asyncio.create_task(self._broadcast({
                    "type": "log",
                    "message": f"bootstrap: failed - {error}"
                }))

        except Exception as e:
            _logger.error(f"[BOOTSTRAP] Exception during bootstrap: {e}", exc_info=True)
            _ = asyncio.create_task(self._broadcast({
                "type": "log",
                "message": f"bootstrap: error - {str(e)[:100]}"
            }))

    async def _build_judge_prompt(self, query: str, results: list[dict]) -> str:
        # delegate to prompts module; prefer explicit judge_mode, then report_mode
        judge_mode = (getattr(self, 'judge_mode', None) or 'steering')
        task_mode = getattr(self, 'report_mode', None)  # Main task objective
        # Pass query history to avoid repetition
        query_history = list(getattr(self, '_query_pool', []))

        # Retrieve memory context from Supermemory (both legacy insights AND LangChain memories)
        memory_context = None
        try:
            if self.memory_manager and self.memory_manager.enabled:
                container_tag = self.run_id or "default"

                # Get legacy insights context (old method)
                legacy_context = await self.memory_manager.get_context(
                    query=query,
                    container_tag=container_tag,
                    max_insights=5
                )

                # Get LangChain memories (NEW - progressive learning)
                langchain_memories = await self.memory_manager.search_memory(
                    query=query,
                    container_tag=container_tag,
                    limit=5
                )

                # Format LangChain memories for judge prompt
                langchain_context = ""
                if langchain_memories:
                    langchain_lines = ["**ACCUMULATED KNOWLEDGE (from previous iterations):**"]
                    for i, mem in enumerate(langchain_memories, 1):
                        content = mem.get("content", "")
                        mem_type = mem.get("type", "unknown").upper()
                        version = mem.get("version", 1)
                        langchain_lines.append(
                            f"{i}. [{mem_type}] v{version}\n   {content[:200]}"
                        )
                    langchain_lines.append("")  # Empty line
                    langchain_context = "\n".join(langchain_lines)

                # Combine both contexts
                if legacy_context and langchain_context:
                    memory_context = langchain_context + "\n" + legacy_context
                elif langchain_context:
                    memory_context = langchain_context
                elif legacy_context:
                    memory_context = legacy_context

        except Exception as e:
            _logger.debug(f"[MEMORY] Failed to retrieve context: {e}")

        # Pass task_mode so judge understands the overall objective
        return prompts_build_judge_prompt(
            judge_mode, query, results,
            task_mode=task_mode,
            query_history=query_history,
            memory_context=memory_context
        )

    async def _llm_judge(self, results: list[dict]) -> dict[int, dict]:
        # Enforce token/step budget (approximate by characters)
        if int(self._reports_sent) >= int(self.max_reports):
            return {}
        # run-level budget checks
        try:
            if self.token_cap is not None and int(self._tokens_used) >= int(self.token_cap):
                return {}
        except Exception:
            pass
        judged: dict[int, dict] = {}
        try:
            prompt = await self._build_judge_prompt(self.query, results)
            self._tokens_used += len(prompt)
            # basic budget check
            if int(self._tokens_used) > int(self.max_report_tokens):
                return {}
            if self.token_cap is not None and int(self._tokens_used) > int(self.token_cap):
                return {}
            # pre-generation clarity log for judge
            try:
                ids = [int(it.get('id', it.get('doc_id', -1))) for it in (results or [])]
                _ = asyncio.create_task(self._broadcast({
                    "type": "log",
                    "message": f"judge: preparing doc_ids={ids} schema=items[doc_id,is_relevant,why,entry_point,missing_context[],follow_up_queries[],keywords[],inspect[]]"
                }))
            except Exception:
                pass
            try:
                _ = asyncio.create_task(self._broadcast({"type": "log", "message": f"judge: prompt_len={len(prompt)}"}))
            except Exception:
                pass
            try:
                _ = asyncio.create_task(self._broadcast({"type": "log", "message": "judge: generating..."}))
            except Exception:
                pass
            llm_opts = {}
            try:
                if isinstance(self.ollama_num_gpu, int):
                    llm_opts['num_gpu'] = int(self.ollama_num_gpu)
                if isinstance(self.ollama_num_thread, int):
                    llm_opts['num_thread'] = int(self.ollama_num_thread)
                if isinstance(self.ollama_num_batch, int):
                    llm_opts['num_batch'] = int(self.ollama_num_batch)
            except Exception:
                llm_opts = {}
            judge_prompt_path = os.path.join(SETTINGS_DIR, f"reports/judge_prompt_step_{int(self.step_i)}.txt")
            judge_usage_path = os.path.join(SETTINGS_DIR, f"runs/{str(getattr(self, 'run_id', 'run'))}/step_{int(self.step_i)}/judge_usage.json")
            text = generate_text(
                provider=(self.llm_provider or 'ollama'),
                prompt=prompt,
                system=self.ollama_system,
                # ollama
                ollama_model=self.ollama_model,
                ollama_host=self.ollama_host,
                ollama_options=(llm_opts or None),
                # openai
                openai_model=self.openai_model,
                openai_api_key=(self.openai_api_key or ''),
                openai_base_url=(self.openai_base_url or 'https://api.openai.com'),
                openai_temperature=float(getattr(self, 'openai_temperature', 0.0)),
                # google
                google_model=self.google_model,
                google_api_key=(self.google_api_key or ''),
                google_base_url=(self.google_base_url or 'https://generativelanguage.googleapis.com'),
                google_temperature=float(getattr(self, 'google_temperature', 0.0)),
                # grok
                grok_model=self.grok_model,
                grok_api_key=(self.grok_api_key or ''),
                grok_base_url=(self.grok_base_url or 'https://api.x.ai'),
                grok_temperature=float(getattr(self, 'grok_temperature', 0.0)),
                save_prompt_path=judge_prompt_path,
                save_usage_path=judge_usage_path,
            )
            raw = (text or "").strip()
            obj = self._parse_json_loose(raw)
            arr = obj.get('items', []) if isinstance(obj, dict) else []
            for j in arr:
                try:
                    did = int(j.get('doc_id'))
                    self._judge_cache[did] = j
                    judged[did] = j
                except Exception:
                    continue

            # Extract suggested_top_k if present (adaptive retrieval depth)
            try:
                if isinstance(obj, dict) and 'suggested_top_k' in obj:
                    suggested = int(obj['suggested_top_k'])
                    # Validate range (5-50)
                    if 5 <= suggested <= 50:
                        old_top_k = self.top_k
                        self.top_k = suggested
                        try:
                            _ = asyncio.create_task(self._broadcast({
                                "type": "adaptive_top_k",
                                "old_value": old_top_k,
                                "new_value": suggested,
                                "message": f"Judge adjusted top_k: {old_top_k} → {suggested}"
                            }))
                            _ = asyncio.create_task(self._broadcast({
                                "type": "log",
                                "message": f"judge: adaptive_top_k={suggested} (was {old_top_k})"
                            }))
                        except Exception:
                            pass
            except Exception:
                pass

            # Note: insights_to_store handling removed - now handled by Memory Manager Agent

            # AUTOMATIC ROOM DISCOVERY - analyze chunks to discover rooms
            try:
                if hasattr(self, 'room_analyzer') and self.room_analyzer and self.memory_manager and self.memory_manager.enabled:
                    auto_room_insights = self.room_analyzer.analyze_chunks(results)
                    if auto_room_insights:
                        container_tag = self.run_id or "default"
                        stored = await self.memory_manager.add_bulk_insights(
                            insights=auto_room_insights,
                            container_tag=container_tag
                        )
                        if stored > 0:
                            try:
                                _ = asyncio.create_task(self._broadcast({
                                    "type": "log",
                                    "message": f"memory: auto-discovered {stored} rooms"
                                }))
                            except Exception:
                                pass
            except Exception as e:
                _logger.debug(f"[MEMORY] Room auto-discovery failed: {e}")

            # LANGCHAIN MEMORY AGENT - create/update memories on EVERY iteration
            try:
                # Check if LangChain agent is enabled via environment
                langchain_enabled = os.environ.get('LANGCHAIN_MEMORY_ENABLED', 'true').lower() == 'true'

                # Lazy initialize LangChain Memory Agent
                # Create/recreate agent if: (1) not initialized, (2) run_id changed (new exploration session)
                needs_recreation = (
                    self.langchain_agent is None or
                    (self.langchain_agent and getattr(self.langchain_agent, 'container_tag', None) != self.run_id)
                )

                if needs_recreation and self.memory_manager and self.memory_manager.enabled and langchain_enabled:
                    try:
                        if self.llm_provider == 'openai':
                            from langchain_openai import ChatOpenAI
                            from embeddinggemma.agents.langchain_memory_agent import LangChainMemoryAgent

                            # Get model from environment or use default
                            langchain_model = os.environ.get('LANGCHAIN_MEMORY_MODEL', 'gpt-4o-mini')
                            max_iterations = int(os.environ.get('LANGCHAIN_MAX_ITERATIONS', '10'))

                            # Create LangChain LLM
                            llm = ChatOpenAI(
                                model=langchain_model,
                                api_key=self.openai_api_key,
                                base_url=self.openai_base_url,
                                temperature=0.0
                            )

                            # Create synchronous memory manager for LangChain tools
                            # (avoids event loop conflicts in sync tool functions)
                            sync_memory_manager = SupermemoryManagerSync()

                            # Create LangChain Memory Agent with sync manager
                            self.langchain_agent = LangChainMemoryAgent(
                                llm=llm,
                                memory_manager=sync_memory_manager,
                                container_tag=self.run_id or "default",
                                model=langchain_model
                            )

                            # Broadcast initialization
                            try:
                                _ = asyncio.create_task(self._broadcast({
                                    "type": "log",
                                    "message": f"langchain-agent: initialized (model: {langchain_model}, max_iterations: {max_iterations})"
                                }))
                            except Exception:
                                pass

                        # TODO: Add Ollama support
                        # elif self.llm_provider == 'ollama':
                        #     from langchain_ollama import ChatOllama
                        #     llm = ChatOllama(model=self.ollama_model, ...)

                    except Exception as e:
                        _logger.error(f"[LANGCHAIN-AGENT] Failed to initialize: {e}")
                        try:
                            _ = asyncio.create_task(self._broadcast({
                                "type": "log",
                                "message": f"langchain-agent: initialization failed - {str(e)[:100]}"
                            }))
                        except Exception:
                            pass

                # Process iteration with LangChain agent (runs on EVERY iteration)
                if self.langchain_agent and self.memory_manager.enabled:
                    try:
                        result = await self.langchain_agent.process_iteration(
                            query=self.query,
                            code_chunks=results,
                            judge_results=judged
                        )

                        if result.get("success"):
                            created = result.get("memories_created", 0)
                            updated = result.get("memories_updated", 0)

                            # Broadcast agent output
                            if created > 0 or updated > 0:
                                try:
                                    msg = f"langchain-agent: created {created}, updated {updated} memories"
                                    _ = asyncio.create_task(self._broadcast({
                                        "type": "log",
                                        "message": msg
                                    }))

                                    # Show agent reasoning if available
                                    agent_output = result.get("agent_output", "")
                                    if agent_output:
                                        _ = asyncio.create_task(self._broadcast({
                                            "type": "log",
                                            "message": f"  └─ {agent_output[:150]}"
                                        }))
                                except Exception:
                                    pass
                        else:
                            # Agent skipped or failed
                            reason = result.get("reason", "Unknown")
                            try:
                                _ = asyncio.create_task(self._broadcast({
                                    "type": "log",
                                    "message": f"langchain-agent: skipped - {reason}"
                                }))
                            except Exception:
                                pass

                    except Exception as e:
                        _logger.debug(f"[LANGCHAIN-AGENT] Iteration processing failed: {e}")

            except Exception as e:
                _logger.debug(f"[LANGCHAIN-AGENT] Error: {e}")

            try:
                _ = asyncio.create_task(self._broadcast({"type": "log", "message": f"judge: parsed={len(judged)}"}))
            except Exception:
                pass
            return judged
        except Exception as e:
            try:
                _ = asyncio.create_task(self._broadcast({"type": "log", "message": f"judge: LLM fallback due to: {e}"}))
            except Exception:
                pass
            # Fallback: heuristic
            for it in results:
                try:
                    doc_id = int(it.get('id', it.get('doc_id', -1)))
                    if doc_id in self._judge_cache:
                        judged[doc_id] = self._judge_cache[doc_id]
                        continue
                    content = str(it.get('content', ''))
                    score = float(it.get('score', 0.0))
                    relevant = score >= 0.5
                    entry_point = any(tok in content for tok in ['if __name__ == "__main__"', 'def main', 'class ', 'FastAPI(', 'app = FastAPI'])
                    judge = {
                        'doc_id': doc_id,
                        'is_relevant': bool(relevant),
                        'why': 'heuristic based on cosine score',
                        'entry_point': bool(entry_point),
                        'missing_context': [],
                        'follow_up_queries': [],
                        'keywords': [],
                        'inspect': [],
                    }
                    self._judge_cache[doc_id] = judge
                    judged[doc_id] = judge
                except Exception:
                    continue
            return judged

    def _enrich_results_with_ids(self, items: list[dict]) -> list[dict]:
        """Enrich results with document IDs.

        If results already have 'id' or 'doc_id', use those directly.
        Otherwise, fall back to content-based matching (legacy behavior).
        """
        if self.retr is None or not isinstance(items, list):
            return items or []
        try:
            out: list[dict] = []
            for it in items:
                new_it = dict(it)

                # Check if ID is already present (preferred method)
                existing_id = it.get('id') or it.get('doc_id')
                if existing_id is not None and existing_id != -1:
                    # ID already present, ensure both fields are set
                    new_it['id'] = int(existing_id)
                    new_it['doc_id'] = int(existing_id)
                    # Ensure score is present
                    if 'score' not in new_it:
                        new_it['score'] = float(it.get('relevance_score', it.get('embedding_score', 0.0)))
                    out.append(new_it)
                    continue

                # Fall back to content-based matching (legacy)
                c = str(it.get('content', ''))
                if not c:
                    # No content to match, use placeholder
                    new_it['id'] = -1
                    new_it['doc_id'] = -1
                    new_it['score'] = float(it.get('relevance_score', it.get('score', 0.0)))
                    out.append(new_it)
                    continue

                # Try to find document by content
                found = False
                for d in getattr(self.retr, 'documents', []):
                    if getattr(d, 'content', '') == c:
                        new_it['id'] = int(getattr(d, 'id', -1))
                        new_it['doc_id'] = int(getattr(d, 'id', -1))
                        new_it['score'] = float(getattr(d, 'relevance_score', 0.0))
                        found = True
                        break

                if not found:
                    # Could not match, log warning
                    _logger.warning(f"Could not find document ID for content (len={len(c)})")
                    new_it['id'] = -1
                    new_it['doc_id'] = -1
                    new_it['score'] = float(it.get('relevance_score', it.get('score', 0.0)))

                out.append(new_it)

            return out
        except Exception as e:
            _logger.error(f"_enrich_results_with_ids failed: {e}", exc_info=True)
            return items

    def _apply_judgements(self, judged: dict[int, dict]) -> None:
        # votes and boosts
        new_queries: list[str] = []
        for doc_id, j in judged.items():
            try:
                vote = 1 if bool(j.get('is_relevant')) else -1
                self._llm_vote[doc_id] = vote
                # update doc boost and optionally relevance
                boost_delta = 0.5 if vote > 0 else -0.2
                self._doc_boost[doc_id] = self._doc_boost.get(doc_id, 0.0) + boost_delta
                d = self._doc_by_id(doc_id)
                if d is not None:
                    cur = float(getattr(d, 'relevance_score', 0.0))
                    setattr(d, 'relevance_score', max(0.0, cur + (0.05 * boost_delta)))
                # seed queue
                if bool(j.get('entry_point')) and doc_id not in self._seeds_queue:
                    self._seeds_queue.append(doc_id)
                # collect new queries (keywords and follow-ups)
                for kw in j.get('keywords', []) or []:
                    if isinstance(kw, str) and kw.strip():
                        new_queries.append(kw.strip())
                for q in j.get('follow_up_queries', []) or []:
                    if isinstance(q, str) and q.strip():
                        new_queries.append(q.strip())
            except Exception:
                continue

        # Deduplicate new queries against existing pool using semantic similarity
        if new_queries:
            # First deduplicate within new queries themselves
            new_queries_deduped = dedup_multi_queries(new_queries, similarity_threshold=0.85)
            # Then filter out queries too similar to existing pool
            combined = list(self._query_pool) + new_queries_deduped
            all_deduped = dedup_multi_queries(combined, similarity_threshold=0.85)
            # Add only the truly new ones
            for q in all_deduped:
                if q not in self._query_pool:
                    self._query_pool.append(q)

        # Periodic global deduplication every 10 steps to prevent pool bloat
        if hasattr(self, 'step_i') and int(self.step_i) % 10 == 0:
            original_size = len(self._query_pool)
            self._query_pool = dedup_multi_queries(list(self._query_pool), similarity_threshold=0.85)
            if len(self._query_pool) < original_size:
                _logger.info(f"[DEDUP] Query pool cleaned: {original_size} -> {len(self._query_pool)} queries")

    def _parse_chunk_header(self, content: str) -> tuple[str | None, int | None, int | None, int | None]:
        try:
            # Header format: "# file: <rel> | lines: a-b | window: w"
            first = (content.splitlines() or [""])[0]
            if not first.startswith('# file:'):
                return None, None, None, None
            parts = [p.strip() for p in first[1:].split('|')]
            file_part = parts[0].split(':', 1)[1].strip() if len(parts) > 0 and ':' in parts[0] else None
            lines_part = parts[1].split(':', 1)[1].strip() if len(parts) > 1 and ':' in parts[1] else None
            win_part = parts[2].split(':', 1)[1].strip() if len(parts) > 2 and ':' in parts[2] else None
            a, b = None, None
            if lines_part and '-' in lines_part:
                try:
                    a = int(lines_part.split('-')[0].strip())
                    b = int(lines_part.split('-')[1].strip())
                except Exception:
                    a, b = None, None
            w = None
            try:
                if win_part:
                    w = int(win_part)
            except Exception:
                w = None
            return file_part, a, b, w
        except Exception:
            return None, None, None, None

    def _neighbors_for_doc(self, doc_id: int, line_radius: int = 100, max_neighbors: int = 20) -> list[int]:
        try:
            d = self._doc_by_id(doc_id)
            if d is None:
                return []
            f, a, b, _w = self._parse_chunk_header(getattr(d, 'content', ''))
            if f is None or a is None or b is None:
                return []
            lo = max(1, int(a) - int(line_radius))
            hi = int(b) + int(line_radius)
            out: list[int] = []
            for other in getattr(self.retr, 'documents', []) if self.retr is not None else []:
                if int(getattr(other, 'id', -1)) == int(doc_id):
                    continue
                f2, a2, b2, _w2 = self._parse_chunk_header(getattr(other, 'content', ''))
                if f2 == f and (a2 is not None and b2 is not None):
                    if not (b2 < lo or a2 > hi):
                        out.append(int(getattr(other, 'id', -1)))
                if len(out) >= int(max_neighbors):
                    break
            return out
        except Exception:
            return []

    def _apply_targeted_fetch(self, max_neighbors_per_seed: int = 10) -> None:
        # Boost neighbors around seeds and then drain the seed queue
        new_boosts = 0
        try:
            seeds = list(self._seeds_queue)
            self._seeds_queue.clear()
            for sid in seeds:
                neigh = self._neighbors_for_doc(int(sid), max_neighbors=int(max_neighbors_per_seed))
                for nid in neigh:
                    self._doc_boost[nid] = self._doc_boost.get(nid, 0.0) + 0.3
                    od = self._doc_by_id(nid)
                    if od is not None:
                        cur = float(getattr(od, 'relevance_score', 0.0))
                        setattr(od, 'relevance_score', max(0.0, cur + 0.02))
                        new_boosts += 1
        except Exception:
            return

    def _get_root_dir_name(self) -> str:
        """Extract a clean directory name from root_folder for collection naming.

        Returns:
            Sanitized directory name suitable for collection names
        """
        try:
            if self.use_repo:
                # When analyzing src folder, use parent directory name
                root_path = os.path.abspath(os.getcwd())
            else:
                root_path = os.path.abspath(self.root_folder)

            # Get the directory name (last component of the path)
            dir_name = os.path.basename(root_path)

            # Sanitize: replace spaces and special chars with underscores
            # Keep only alphanumeric, underscore, and hyphen
            import re
            sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', dir_name)

            # Fallback if empty after sanitization
            if not sanitized or sanitized == '_':
                sanitized = 'codebase'

            return sanitized.lower()
        except Exception:
            # Fallback to default if any error
            return 'codebase'

    def _apply_pruning(self) -> None:
        try:
            if self.retr is None:
                return
            for d in getattr(self.retr, 'documents', []):
                c = getattr(d, 'content', '')
                if len(c) < int(self.min_content_chars) or self._is_import_only(c):
                    cur = float(getattr(d, 'relevance_score', 0.0))
                    setattr(d, 'relevance_score', max(0.0, cur * (1.0 - float(self.import_only_penalty) * 0.5)))
        except Exception:
            pass

    async def start(self) -> None:
        if self.running:
            return

        # Use collection name as run_id for consistency
        # Collection name format: {root_dir_name}_{timestamp} (e.g., "codebase_20251110_175332")
        self.run_id = self.qdrant_collection

        # Clear LangChain agent to force recreation with latest code and new container_tag
        self.langchain_agent = None

        # Initialize logging paths for this run
        self._run_dir = os.path.join(SETTINGS_DIR, "runs", str(self.run_id))
        self._query_log_path = os.path.join(self._run_dir, "queries.jsonl")
        self._retrieval_log_path = os.path.join(self._run_dir, "retrievals.jsonl")
        self._manifest_path = os.path.join(self._run_dir, "manifest.json")

        # Reset run-specific tracking
        self._unique_docs_accessed = set()
        self._total_tokens = 0
        self._total_cost = 0.0
        self._run_start_time = time.time()
        self._update_manifest()  # Create initial manifest

        # Check if foundational knowledge bootstrap is needed
        bootstrap_needed = await self._check_bootstrap_needed()
        if bootstrap_needed:
            await self._run_bootstrap()

        # Build retriever and corpus
        # Ensure LLM API keys are present from env if not set
        try:
            if not getattr(self, 'openai_api_key', None):
                self.openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not getattr(self, 'google_api_key', None):
                self.google_api_key = os.environ.get('GOOGLE_API_KEY')
            if not getattr(self, 'grok_api_key', None):
                self.grok_api_key = os.environ.get('GROK_API_KEY')
        except Exception:
            pass
        embedding_model_name = os.environ.get('EMBEDDING_MODEL', 'google/embeddinggemma-300m')
        # Prefer OpenAI embeddings by default if API key present
        try:
            if not embedding_model_name or embedding_model_name.strip() == 'google/embeddinggemma-300m':
                if os.environ.get('OPENAI_API_KEY'):
                    embedding_model_name = 'openai:text-embedding-3-large'
        except Exception:
            pass
        device_mode = os.environ.get('DEVICE_MODE', 'auto')
        retr = MCPMRetriever(
            embedding_model_name=embedding_model_name,
            num_agents=self.num_agents,
            max_iterations=self.max_iterations,
            exploration_bonus=self.exploration_bonus,
            pheromone_decay=self.pheromone_decay,
            embed_batch_size=self.embed_batch_size,
            device_mode=device_mode,
        )
        if not self.windows or len(self.windows) == 0:
            # Fallback default windows if none provided or empty list
            # Larger windows = more context for LLM = better insights
            self.windows = [16000]
            try:
                await self._broadcast({"type": "log", "message": "windows: defaulted to [16000]"})
            except Exception:
                pass
        # Corpus loading: either from codebase (memory backend) or from Qdrant as text chunks
        if (self.vector_backend or 'memory').lower() == 'qdrant':
            try:
                from qdrant_client import QdrantClient  # type: ignore
                from embeddinggemma.rag.vectorstore import ensure_collection
                client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
                # Auto-create collection if it doesn't exist
                # Determine embedding dimension based on model
                embedding_dim = 3072  # Default for text-embedding-3-large
                if 'small' in embedding_model_name.lower():
                    embedding_dim = 1536
                elif 'embeddinggemma' in embedding_model_name.lower():
                    embedding_dim = 768
                ensure_collection(client, self.qdrant_collection, embedding_dim)
                # Broadcast collection name to frontend
                try:
                    await self._broadcast({"type": "collection", "name": self.qdrant_collection})
                except Exception:
                    pass
                # Pull recent points; assume payload.text contains content; fallback to 'text'/'snippet'
                res = client.scroll(collection_name=self.qdrant_collection, limit=int(self.max_files) or 10000, with_payload=True, with_vectors=False)
                points = (res[0] or []) if isinstance(res, tuple) else (res or [])
                docs = []
                for p in points:
                    try:
                        pl = getattr(p, 'payload', {}) or {}
                        txt = pl.get('text') or pl.get('snippet') or ''
                        if txt:
                            # prepend header if path/lines exist for neighbor finding
                            path = pl.get('path') or pl.get('file_path') or 'chunk'
                            start = pl.get('start') or 1
                            end = pl.get('end') or (start + len(str(txt).splitlines()))
                            header = f"# file: {path} | lines: {start}-{end} | window: {max(self.windows)}\n"
                            docs.append(header + str(txt))
                    except Exception:
                        continue
            except Exception as e:
                raise RuntimeError(f"Qdrant fetch failed: {e}")
        else:
            if self.use_repo:
                # Use absolute path to 'src' directory from project root
                # Go up 4 levels: server.py -> realtime -> embeddinggemma -> src -> project_root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                rf = os.path.join(project_root, 'src')
            else:
                rf = os.path.abspath(self.root_folder or os.getcwd())
            _logger.info(f"Loading codebase from: {rf}")
            _logger.info(f"Chunking parameters: windows={self.windows}, max_files={self.max_files}, exclude_dirs={self.exclude_dirs}")
            await self._broadcast({"type": "log", "message": f"Loading codebase from: {rf}"})
            await self._broadcast({"type": "log", "message": f"Chunking config: windows={self.windows}, max_files={self.max_files}"})
            docs, self.corpus_file_count = collect_codebase_chunks(rf, self.windows, int(self.max_files), self.exclude_dirs, self.chunk_workers)
            await self._broadcast({"type": "log", "message": f"Loaded {len(docs)} chunks from {self.corpus_file_count} files"})
            if len(docs) == 0:
                await self._broadcast({"type": "log", "message": "WARNING: No documents loaded! Check root_folder path and exclude_dirs"})
            # compute simple fingerprint of file list + sizes
            try:
                files = list_code_files(rf, int(self.max_files), self.exclude_dirs)
                h = hashlib.sha1()
                for p in sorted(files):
                    try:
                        h.update(p.encode('utf-8', errors='ignore'))
                        h.update(str(os.path.getsize(p)).encode('utf-8'))
                    except Exception:
                        continue
                self._corpus_fingerprint = h.hexdigest()
            except Exception:
                self._corpus_fingerprint = None
        retr.add_documents(docs)
        retr.initialize_simulation(self.query)
        self.retr = retr
        # Save corpus metadata after documents are loaded
        self._save_corpus_metadata()
        self.running = True
        self.task = asyncio.create_task(self._run_loop())
        # announce
        try:
            await self._broadcast({"type": "log", "message": f"started: docs={len(getattr(retr,'documents',[]))} agents={len(getattr(retr,'agents',[]))} windows={self.windows}"})
        except Exception:
            pass
        # Send initial snapshot immediately so UI isn't blank
        try:
            snap = retr.get_snapshot(
                min_trail_strength=self.min_trail_strength,
                max_edges=self.max_edges,
                method="pca",
                whiten=False,
                dims=int(self.viz_dims)
            )
            await self._broadcast({"type": "snapshot", "step": 0, "data": snap})
            # Also send initial metrics
            docs_count = len(getattr(retr, 'documents', []))
            agents_count = len(getattr(retr, 'agents', []))
            self.last_metrics = {
                "step": 0,
                "docs": docs_count,
                "files": self.corpus_file_count,
                "agents": agents_count,
                "avg_rel": 0.0,
                "max_rel": 0.0,
                "trails": 0
            }
            await self._broadcast({"type": "metrics", "data": self.last_metrics})
        except Exception as e:
            await self._broadcast({"type": "log", "message": f"initial snapshot error: {e}"})

    async def stop(self) -> None:
        self.running = False
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await asyncio.wait_for(self.task, timeout=5.0)
            except asyncio.CancelledError:
                _logger.info("Task cancelled successfully")
            except asyncio.TimeoutError:
                _logger.warning("Task did not stop within timeout")
            except Exception as e:
                _logger.error(f"Error during stop: {e}")
            finally:
                self.task = None
        elif self.task is not None:
            # Task already completed
            self.task = None

    async def _run_loop(self) -> None:
        assert self.retr is not None
        self.step_i = 0
        try:
            while self.running:
                if self._paused:
                    await asyncio.sleep(0.05)
                    continue
                # step
                try:
                    self.retr.step(1)
                except Exception as e:
                    await self._broadcast({"type": "log", "message": f"error in step: {e}"})
                    break
                self.step_i += 1
                # compute metrics
                try:
                    docs = getattr(self.retr, 'documents', [])
                    agents = getattr(self.retr, 'agents', [])
                    avg_rel = float(sum(getattr(d, 'relevance_score', 0.0) for d in docs) / max(1, len(docs))) if docs else 0.0
                    max_rel = float(max((getattr(d, 'relevance_score', 0.0) for d in docs), default=0.0)) if docs else 0.0
                    trails = len(getattr(self.retr, 'pheromone_trails', {}))
                    self.last_metrics = {"step": int(self.step_i), "docs": len(docs), "files": self.corpus_file_count, "agents": len(agents), "avg_rel": avg_rel, "max_rel": max_rel, "trails": trails}
                except Exception:
                    self.last_metrics = {"step": int(self.step_i), "files": self.corpus_file_count}

                if self.step_i % max(1, int(self.redraw_every)) == 0:
                    _logger.info(f"[ANALYTICS] redraw_every condition met: step={self.step_i}, redraw_every={self.redraw_every}")
                    try:
                        _logger.info(f"[ANALYTICS] about to call get_snapshot")
                        snap = self.retr.get_snapshot(min_trail_strength=self.min_trail_strength, max_edges=self.max_edges, method="pca", whiten=False, dims=int(self.viz_dims))
                        _logger.info(f"[ANALYTICS] get_snapshot succeeded")
                        await self._broadcast({"type": "snapshot", "step": int(self.step_i), "data": snap})
                        if self.last_metrics is not None:
                            await self._broadcast({"type": "metrics", "data": self.last_metrics})
                        # Always broadcast current Top-K results for this step
                        try:
                            if self.retr is not None:
                                # Log query and time retrieval
                                import time as _tm
                                _start = _tm.time()
                                self._log_query(self.query, source="user", step=int(self.step_i))
                                res_now = self.retr.search(self.query, top_k=int(self.top_k))
                                _elapsed = (_tm.time() - _start) * 1000
                                self._log_retrieval(self.query, res_now.get("results", []), step=int(self.step_i), retrieval_time_ms=_elapsed)
                                # If we have seeded follow-up queries and MQ is enabled, aggregate
                                if bool(getattr(self, 'mq_enabled', False)) and getattr(self, '_query_pool', []):
                                    try:
                                        pool = list(getattr(self, '_query_pool', []))
                                        try:
                                            await self._broadcast({"type": "log", "message": f"mq: pool_size={len(pool)} mq_count={min(int(getattr(self, 'mq_count', 5)), 3)}"})
                                        except Exception:
                                            pass
                                        extra = dedup_multi_queries(pool, similarity_threshold=0.8)[: min(int(getattr(self, 'mq_count', 5)), 3)]
                                        # Log a compact preview of selected extras to avoid overly long messages
                                        try:
                                            preview = extra[:3]
                                            more = max(0, len(extra) - len(preview))
                                            msg = f"mq: selected_extras={preview}" + (f" (+{more} more)" if more else "")
                                            await self._broadcast({"type": "log", "message": msg})
                                        except Exception:
                                            pass
                                        queries = [self.query] + extra
                                        agg: dict[str, dict] = {}
                                        per_counts = {}
                                        for q in queries:
                                            # Log follow-up query
                                            _start_fq = _tm.time()
                                            self._log_query(q, source="followup" if q != self.query else "user", step=int(self.step_i), parent_query=self.query if q != self.query else None)
                                            r = self.retr.search(q, top_k=int(self.top_k))
                                            _elapsed_fq = (_tm.time() - _start_fq) * 1000
                                            res_list = r.get('results', []) or []
                                            self._log_retrieval(q, res_list, step=int(self.step_i), retrieval_time_ms=_elapsed_fq)
                                            try:
                                                per_counts[q] = len(res_list)
                                            except Exception:
                                                per_counts[q] = 0
                                            for it in (r.get('results', []) or []):
                                                c = str(it.get('content', ''))
                                                if not c:
                                                    continue
                                                prev = agg.get(c)
                                                if prev is None or float(it.get('relevance_score', 0.0)) > float(prev.get('relevance_score', 0.0)):
                                                    agg[c] = it
                                        merged = sorted(agg.values(), key=lambda x: x.get('relevance_score', 0.0), reverse=True)[: int(self.top_k)]
                                        # Summarize aggregation in logs
                                        try:
                                            q_preview = queries[:3]
                                            more_q = max(0, len(queries) - len(q_preview))
                                            await self._broadcast({
                                                "type": "log",
                                                "message": (
                                                    f"mq: aggregated queries={q_preview}" + (f" (+{more_q} more)" if more_q else "") +
                                                    f" per_counts={per_counts} merged_results={len(merged)}"
                                                )
                                            })
                                        except Exception:
                                            pass
                                        await self._broadcast({"type": "results", "step": int(self.step_i), "data": merged})
                                        res_now = {"results": merged}
                                    except Exception as _e_mq:
                                        await self._broadcast({"type": "log", "message": f"mq: aggregate failed: {_e_mq}"})
                                else:
                                    await self._broadcast({"type": "results", "step": int(self.step_i), "data": res_now.get("results", [])})
                                # blended Top-K with contextual steering weights
                                try:
                                    blended = self._compute_blended_topk(res_now.get("results", []))
                                    await self._broadcast({"type": "results_blended", "step": int(self.step_i), "data": blended})
                                except Exception as _e_blend:
                                    await self._broadcast({"type": "log", "message": f"blend: failed: {_e_blend}"})
                                await self._broadcast({"type": "log", "message": f"top_k={self.top_k} results_count={len(res_now.get('results', []))}"})
                        except Exception as e:
                            _logger.error(f"[ANALYTICS] Exception in inner try block (logging code): {e}", exc_info=True)
                        # Per-step report (optional)
                        try:
                            if self.report_enabled and (int(self.step_i) % max(1, int(self.report_every)) == 0) and self.retr is not None:
                                # Log report generation query
                                _start_rpt = _tm.time()
                                res_top = self.retr.search(self.query, top_k=int(self.top_k))
                                _elapsed_rpt = (_tm.time() - _start_rpt) * 1000
                                self._log_retrieval(self.query, res_top.get("results", []), step=int(self.step_i), retrieval_time_ms=_elapsed_rpt)
                                docs = self._enrich_results_with_ids(res_top.get("results", []))
                                await self._broadcast({"type": "log", "message": f"report: step={self.step_i} mode={self.report_mode} top_k_items={len(docs)}"})
                                mode = (self.report_mode or "deep").lower()
                                # pre-generation clarity log
                                try:
                                    await self._broadcast({
                                        "type": "log",
                                        "message": (
                                            f"report: preparing mode={mode} items={len(docs)} "
                                            "fields=items[code_chunk,content,file_path,line_range,code_purpose,code_dependencies,file_type,embedding_score,relevance_to_query,query_initial,follow_up_queries]"
                                        )
                                    })
                                except Exception:
                                    pass
                                prompt = _build_report_prompt(mode, self.query, int(self.top_k), docs)
                                try:
                                    await self._broadcast({"type": "log", "message": f"report: prompt_len={len(prompt)}"})
                                except Exception:
                                    pass
                                try:
                                    await self._broadcast({"type": "log", "message": "report: generating..."})
                                    llm_opts = {}
                                    try:
                                        if isinstance(self.ollama_num_gpu, int):
                                            llm_opts['num_gpu'] = int(self.ollama_num_gpu)
                                        if isinstance(self.ollama_num_thread, int):
                                            llm_opts['num_thread'] = int(self.ollama_num_thread)
                                        if isinstance(self.ollama_num_batch, int):
                                            llm_opts['num_batch'] = int(self.ollama_num_batch)
                                    except Exception:
                                        llm_opts = {}
                                    prompt_path = os.path.join(SETTINGS_DIR, f"reports/prompt_step_{int(self.step_i)}.txt")
                                    usage_path = os.path.join(SETTINGS_DIR, f"runs/{str(getattr(self, 'run_id', 'run'))}/step_{int(self.step_i)}/usage.json")
                                    text = generate_text(
                                        provider=(self.llm_provider or 'ollama'),
                                        prompt=prompt,
                                        system=self.ollama_system,
                                        ollama_model=self.ollama_model,
                                        ollama_host=self.ollama_host,
                                        ollama_options=(llm_opts or None),
                                        openai_model=self.openai_model,
                                        openai_api_key=(self.openai_api_key or ''),
                                        openai_base_url=(self.openai_base_url or 'https://api.openai.com'),
                                        openai_temperature=float(getattr(self, 'openai_temperature', 0.0)),
                                        google_model=self.google_model,
                                        google_api_key=(self.google_api_key or ''),
                                        google_base_url=(self.google_base_url or 'https://generativelanguage.googleapis.com'),
                                        google_temperature=float(getattr(self, 'google_temperature', 0.0)),
                                        grok_model=self.grok_model,
                                        grok_api_key=(self.grok_api_key or ''),
                                        grok_base_url=(self.grok_base_url or 'https://api.x.ai'),
                                        grok_temperature=float(getattr(self, 'grok_temperature', 0.0)),
                                        save_prompt_path=prompt_path,
                                        save_usage_path=usage_path,
                                    )
                                except Exception as e:
                                    await self._broadcast({"type": "log", "message": f"report: LLM error: {e}"})
                                    text = "{}"
                                # Parse JSON best-effort
                                try:
                                    report_obj = self._parse_json_loose(text or "")
                                    if not isinstance(report_obj, dict):
                                        report_obj = {"items": []}
                                except Exception:
                                    report_obj = {"items": []}
                                # Save and broadcast
                                try:
                                    reports_dir = os.path.join(SETTINGS_DIR, "runs", str(getattr(self, 'run_id', 'run')))  # per-run directory
                                    os.makedirs(reports_dir, exist_ok=True)
                                    step_dir = os.path.join(reports_dir, f"step_{int(self.step_i)}")
                                    os.makedirs(step_dir, exist_ok=True)
                                    # Save exact prompt and raw response for debugging
                                    try:
                                        with open(os.path.join(step_dir, "prompt.txt"), "w", encoding="utf-8") as f_pr:
                                            f_pr.write(prompt)
                                    except Exception:
                                        pass
                                    try:
                                        with open(os.path.join(step_dir, "response_raw.txt"), "w", encoding="utf-8") as f_raw:
                                            f_raw.write(text or "")
                                    except Exception:
                                        pass
                                    # Save parsed JSON
                                    pth = os.path.join(step_dir, "report.json")
                                    with open(pth, "w", encoding="utf-8") as f:
                                        json.dump({"step": int(self.step_i), "mode": mode, "data": report_obj}, f, ensure_ascii=False, indent=2)
                                    await self._broadcast({"type": "log", "message": f"report: saved {pth}"})
                                except Exception as e:
                                    await self._broadcast({"type": "log", "message": f"report: save failed: {e}"})
                                await self._broadcast({"type": "report", "step": int(self.step_i), "data": report_obj})
                                try:
                                    await self._broadcast({"type": "log", "message": f"report: items={len(report_obj.get('items', [])) if isinstance(report_obj, dict) else 0}"})
                                except Exception:
                                    pass
                                # Seed follow-up queries from report into query pool (dedup)
                                try:
                                    items = report_obj.get('items', []) if isinstance(report_obj, dict) else []
                                    new_qs: list[str] = []
                                    for it in items:
                                        for q in (it.get('follow_up_queries', []) or []):
                                            if isinstance(q, str) and q.strip():
                                                new_qs.append(q.strip())
                                    # filter: require meaningful query (relaxed to allow exploratory queries)
                                    def _is_concrete(q: str) -> bool:
                                        s = (q or "").strip()
                                        if not s:
                                            return False
                                        # Too short or vague
                                        if len(s) < 10:
                                            return False
                                        # Generic/vague phrases to reject
                                        vague = [
                                            "more details", "tell me more", "what else", "anything else",
                                            "show me", "find", "search for", "look for", "get"
                                        ]
                                        s_lower = s.lower()
                                        if any(v in s_lower and len(s) < 25 for v in vague):
                                            return False

                                        # Accept concrete targets (high confidence)
                                        if re.search(r"\b(lines?\s*[:#-]?\s*\d+(-\d+)?)\b", s):
                                            return True
                                        if ("/" in s) or ("\\" in s):
                                            return True
                                        if re.search(r"\b(def|class|function|method|interface|type)\s+[A-Za-z_][A-Za-z0-9_]*", s):
                                            return True
                                        if re.search(r"@(app|router)\.(get|post|put|patch|delete)\(\s*['\"]", s):
                                            return True

                                        # Accept exploratory queries with domain keywords (medium confidence)
                                        exploratory_keywords = [
                                            "architecture", "pattern", "design", "structure", "flow",
                                            "dependency", "dependencies", "import", "imports", "module", "modules",
                                            "error", "exception", "handle", "handling", "validation",
                                            "api", "endpoint", "route", "router", "controller",
                                            "model", "schema", "database", "query", "repository",
                                            "service", "util", "helper", "config", "setting",
                                            "auth", "authentication", "authorization", "permission",
                                            "test", "tests", "testing", "mock", "fixture",
                                            "cache", "caching", "redis", "celery", "queue",
                                            "logging", "logger", "monitoring", "metric",
                                            "security", "vulnerability", "validate", "sanitize"
                                        ]
                                        if any(kw in s_lower for kw in exploratory_keywords):
                                            return True

                                        # Accept queries with specific verbs + nouns (low confidence)
                                        if re.search(r"\b(how|what|where|when|why)\s+(is|are|does|do)\b", s_lower):
                                            return True

                                        return False
                                    if new_qs:
                                        concrete = [q for q in new_qs if _is_concrete(q)]
                                        if concrete:
                                            qs = dedup_multi_queries(concrete + list(getattr(self, '_query_pool', [])), similarity_threshold=0.8)
                                            before = set(getattr(self, '_query_pool', []))
                                            added: list[str] = []
                                            for q in qs:
                                                if q not in before:
                                                    self._query_pool.append(q)
                                                    added.append(q)
                                                    if len(self._query_pool) > int(getattr(self, '_query_pool_cap', 100)):
                                                        self._query_pool = self._query_pool[-int(getattr(self, '_query_pool_cap', 100)) :]
                                            await self._broadcast({"type": "seed_queries", "added": added, "pool_size": len(self._query_pool)})
                                            await self._broadcast({"type": "log", "message": f"seed: added_follow_ups={len(added)} pool_size={len(self._query_pool)}"})
                                            # stagnation reset if we added concrete targets
                                            if added:
                                                self._stagnant_steps = 0
                                        else:
                                            await self._broadcast({"type": "log", "message": "seed: skipped non-concrete follow-ups"})
                                except Exception as _e_seed:
                                    await self._broadcast({"type": "log", "message": f"seed: follow_up_queries failed: {_e_seed}"})
                                # Contextual steering: judge + actions (budgeted)
                                try:
                                    # stagnation detection: if no new targets for N report windows, pause judge
                                    if hasattr(self, '_stagnant_steps') and hasattr(self, 'stagnation_threshold'):
                                        self._stagnant_steps += 1
                                    if self.judge_enabled and (self._reports_sent < int(self.max_reports)) and (self._stagnant_steps < int(getattr(self, 'stagnation_threshold', 8))):
                                        judgements = await self._llm_judge(docs)
                                        self._apply_judgements(judgements)
                                        self._reports_sent += 1
                                        # targeted neighborhood boosts and pruning
                                        self._apply_targeted_fetch(max_neighbors_per_seed=10)
                                        self._apply_pruning()
                                    elif self._stagnant_steps >= int(getattr(self, 'stagnation_threshold', 8)):
                                        await self._broadcast({"type": "log", "message": f"judge: paused due to stagnation (_stagnant_steps={self._stagnant_steps})"})
                                except Exception as _e_judge:
                                    await self._broadcast({"type": "log", "message": f"judge: failed: {_e_judge}"})

                                # ===== EXPLORATION MODE: Contextual Query Generation =====
                                try:
                                    if self.exploration_mode and self.exploration_goal:
                                        from embeddinggemma.exploration import generate_contextual_queries

                                        # Track file access for phase completion
                                        for doc in docs:
                                            doc_id = doc.get('id') or doc.get('doc_id')
                                            if doc_id and doc_id != -1:
                                                self._track_phase_file_access(int(doc_id))

                                        # Extract and track discoveries from report
                                        items = report_obj.get('items', []) if isinstance(report_obj, dict) else []
                                        for it in items:
                                            # Track entry points
                                            if it.get('entry_point') or 'main' in str(it.get('code_purpose', '')).lower():
                                                self._track_phase_discovery("entry_points", {
                                                    "name": it.get('code_chunk', 'unknown'),
                                                    "file_path": it.get('file_path', ''),
                                                    "description": it.get('code_purpose', '')
                                                })

                                            # Track modules (from file paths)
                                            file_path = it.get('file_path', '')
                                            if file_path:
                                                parts = file_path.replace('\\', '/').split('/')
                                                if len(parts) > 2:
                                                    module_name = '.'.join(parts[-3:-1])
                                                    self._track_phase_discovery("modules", {
                                                        "name": module_name,
                                                        "responsibility": it.get('code_purpose', ''),
                                                        "files": [file_path]
                                                    })

                                            # Track imports
                                            deps = it.get('code_dependencies', []) or []
                                            for dep in deps:
                                                if isinstance(dep, str):
                                                    self._track_phase_discovery("imports", dep)

                                        # Generate contextual follow-up queries
                                        contextual_queries = generate_contextual_queries(
                                            goal_type=self.exploration_goal,
                                            phase_index=self.exploration_phase,
                                            discoveries=self._phase_discoveries,
                                            recent_results=docs,
                                            query_history=list(self._query_pool),
                                            max_queries=5
                                        )

                                        # Add contextual queries to pool
                                        added_contextual = []
                                        for q in contextual_queries:
                                            if q not in self._query_pool:
                                                self._query_pool.append(q)
                                                added_contextual.append(q)

                                        if added_contextual:
                                            await self._broadcast({
                                                "type": "exploration_queries",
                                                "phase": self.exploration_phase,
                                                "queries": added_contextual
                                            })
                                            await self._broadcast({
                                                "type": "log",
                                                "message": f"explore: generated {len(added_contextual)} contextual queries"
                                            })

                                        # Check phase completion and advance if needed
                                        if self._check_phase_completion():
                                            await self._broadcast({
                                                "type": "log",
                                                "message": f"explore: phase {self.exploration_phase} complete, advancing..."
                                            })
                                            await self._advance_exploration_phase()

                                except Exception as _e_explore:
                                    await self._broadcast({"type": "log", "message": f"explore: failed: {_e_explore}"})
                                # ===== END EXPLORATION MODE =====

                        except Exception as e:
                            await self._broadcast({"type": "log", "message": f"report: unexpected failure: {e}"})
                        # Update stability window and broadcast results if stable
                        try:
                            avg_rel = float(self.last_metrics.get("avg_rel", 0.0)) if self.last_metrics else 0.0
                            self._avg_rel_history.append(avg_rel)
                            if len(self._avg_rel_history) > 5:
                                self._avg_rel_history = self._avg_rel_history[-5:]
                            if len(self._avg_rel_history) == 5:
                                m = sum(self._avg_rel_history) / 5.0
                                band = 0.05 * (m if m != 0.0 else 1.0)
                                stable = all(abs(v - m) <= band for v in self._avg_rel_history)
                                if stable and self.retr is not None:
                                    try:
                                        # Log stable results query
                                        _start_stable = _tm.time()
                                        res = self.retr.search(self.query, top_k=int(self.top_k))
                                        _elapsed_stable = (_tm.time() - _start_stable) * 1000
                                        self._log_retrieval(self.query, res.get("results", []), step=int(self.step_i), retrieval_time_ms=_elapsed_stable)
                                        await self._broadcast({"type": "results_stable", "step": int(self.step_i), "data": res.get("results", [])})
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    except Exception as e:
                        _logger.error(f"[ANALYTICS] Exception in outer try block (get_snapshot): {e}", exc_info=True)
                        await self._broadcast({"type": "log", "message": f"error in snapshot: {e}"})
                        break
                await asyncio.sleep(0)
        except Exception as e:
            try:
                await self._broadcast({"type": "log", "message": f"run_loop aborted: {e}"})
            except Exception:
                pass

    async def _broadcast(self, obj: Dict[str, Any]) -> None:
        if not self.clients:
            return
        msg = json.dumps(obj)
        stale: List[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            try:
                self.clients.remove(ws)
            except Exception:
                pass


streamer = SnapshotStreamer()
# Load persisted settings on startup
try:
    load_settings_from_disk()
except Exception:
    pass
app = FastAPI()

# Configure CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection: provide streamer instance to routers
def get_streamer() -> SnapshotStreamer:
    """Dependency that provides the streamer instance to endpoints."""
    return streamer

# Settings persistence - helper functions for backward compatibility
SETTINGS_DIR = settings_manager.SETTINGS_DIR
SETTINGS_PATH = settings_manager.SETTINGS_PATH

def settings_dict() -> dict:
    """Wrapper for backward compatibility - delegates to settings_manager."""
    return settings_manager.get_settings_dict(streamer)

def apply_settings(d: dict) -> None:
    """Wrapper for backward compatibility - delegates to settings_manager."""
    settings_manager.apply_settings_to_streamer(streamer, d)

def load_settings_from_disk() -> None:
    """Wrapper for backward compatibility - delegates to settings_manager."""
    settings_manager.load_settings_from_disk(streamer)

def save_settings_to_disk() -> None:
    """Wrapper for backward compatibility - delegates to settings_manager."""
    settings_manager.save_settings_to_disk(streamer)

# Static directory for serving files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Import and mount routers
from embeddinggemma.realtime.routers import collections as collections_router
from embeddinggemma.realtime.routers import simulation as simulation_router
from embeddinggemma.realtime.routers import search as search_router
from embeddinggemma.realtime.routers import agents as agents_router
from embeddinggemma.realtime.routers import settings as settings_router
from embeddinggemma.realtime.routers import prompts as prompts_router
from embeddinggemma.realtime.routers import corpus as corpus_router
from embeddinggemma.realtime.routers import misc as misc_router
from embeddinggemma.realtime.routers import analytics as analytics_router
from embeddinggemma.realtime.routers import exploration as exploration_router

# Configure router dependencies - set the module-level dependency functions
collections_router._get_streamer_dependency = get_streamer
simulation_router._get_streamer_dependency = get_streamer
simulation_router._save_settings_to_disk = save_settings_to_disk
search_router._get_streamer_dependency = get_streamer
agents_router._get_streamer_dependency = get_streamer
settings_router._get_streamer_dependency = get_streamer
settings_router._save_settings_to_disk = save_settings_to_disk
prompts_router._get_streamer_dependency = get_streamer
corpus_router._get_streamer_dependency = get_streamer
misc_router._get_streamer_dependency = get_streamer
misc_router._static_dir = static_dir
misc_router._app = app

# Include routers
app.include_router(collections_router.router)
app.include_router(simulation_router.router)
app.include_router(search_router.router)
app.include_router(agents_router.router)
app.include_router(settings_router.router)
app.include_router(prompts_router.router)
app.include_router(corpus_router.router)
app.include_router(misc_router.router)
app.include_router(analytics_router.router)
app.include_router(exploration_router.router)

# CORS for dev (Vite, direct origins) and dynamic port from env
PORT = int(os.environ.get('EMBEDDINGGEMMA_BACKEND_PORT', '8011'))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8011",
        "http://127.0.0.1:8011",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================================
# Introspect and run endpoints - MOVED TO routers/misc.py
# ============================================================================
# The following endpoints have been extracted to routers/misc.py:
#   - GET  /introspect/api  (http_introspect_api)
#   - POST /run/new         (http_run_new)

def _prompt_default_for_mode(mode: str) -> str:
    """Wrapper for backward compatibility - delegates to prompts_manager."""
    return prompts_manager.get_prompt_default_for_mode(mode)


# ============================================================================
# Prompts endpoints - MOVED TO routers/prompts.py
# ============================================================================
# The following endpoints have been extracted to routers/prompts.py:
#   - POST /prompts/save  (http_prompts_save)
#   - GET  /prompts       (http_prompts_get)


def settings_usage_lines(d: dict) -> list[str]:
    """Wrapper for backward compatibility - delegates to settings_manager."""
    return settings_manager.get_settings_usage_lines(d)


# Import SettingsModel from settings_manager for backward compatibility
from embeddinggemma.realtime.services.settings_manager import SettingsModel  # noqa: E402, F401


# ============================================================================
# Root endpoint - MOVED TO routers/misc.py
# ============================================================================
# The following endpoint has been extracted to routers/misc.py:
#   - GET  /  (index)


# ============================================================================
# Simulation control endpoints - MOVED TO routers/simulation.py
# ============================================================================
# The following 7 endpoints have been extracted to routers/simulation.py:
#   - POST /start       (http_start)
#   - POST /config      (http_config)
#   - POST /stop        (http_stop)
#   - POST /reset       (http_reset)
#   - POST /pause       (http_pause)
#   - POST /resume      (http_resume)
#   - GET  /status      (http_status)
#
# Agent management endpoints - MOVED TO routers/agents.py
# ============================================================================
# The following 2 endpoints have been extracted to routers/agents.py:
#   - POST /agents/add    (http_agents_add)
#   - POST /agents/resize (http_agents_resize)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    streamer.clients.add(ws)
    try:
        # send an ack
        await ws.send_text(json.dumps({"type": "hello", "message": "connected"}))
        while True:
            # allow clients to send small config updates
            data = await ws.receive_text()
            try:
                obj = json.loads(data)
                if isinstance(obj, dict) and obj.get("type") == "config":
                    if "viz_dims" in obj:
                        streamer.viz_dims = int(obj["viz_dims"])
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        try:
            streamer.clients.remove(ws)
        except Exception:
            pass


# ============================================================================
# Settings endpoints - MOVED TO routers/settings.py
# ============================================================================
# The following endpoints have been extracted to routers/settings.py:
#   - GET  /settings  (http_settings_get)
#   - POST /settings  (http_settings_post)


# ============================================================================
# Search endpoints - MOVED TO routers/search.py
# ============================================================================
# The following 3 endpoints have been extracted to routers/search.py:
#   - POST /search       (http_search)
#   - POST /answer       (http_answer)
#   - GET  /doc/{doc_id} (http_doc)

# ============================================================================
# Corpus endpoints - MOVED TO routers/corpus.py
# ============================================================================
# The following endpoints have been extracted to routers/corpus.py:
#   - GET  /corpus/list        (http_corpus_list)
#   - GET  /corpus/summary     (http_corpus_summary)
#   - POST /corpus/add_file    (http_corpus_add_file)
#   - POST /corpus/update_file (http_corpus_update_file)
#   - POST /corpus/reindex     (http_corpus_reindex)
#   - POST /corpus/index_repo  (http_corpus_index_repo)


# Collection management endpoints - MOVED TO routers/collections.py
# All 4 collections endpoints have been moved to routers/collections.py:
# - GET /collections/list
# - POST /collections/switch
# - GET /collections/{collection_name}/info
# - DELETE /collections/{collection_name}


# Helper functions for corpus operations moved to corpus router

# ============================================================================
# Jobs and reports endpoints - MOVED TO routers/misc.py
# ============================================================================
# The following endpoints have been extracted to routers/misc.py:
#   - POST /jobs/start      (http_jobs_start)
#   - GET  /jobs/status     (http_jobs_status)
#   - POST /reports/merge   (http_reports_merge)
