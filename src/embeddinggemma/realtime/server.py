from __future__ import annotations
from typing import List, Dict, Any, Set
import os
import asyncio
import json
import hashlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
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

# Load .env early so env vars are present before initializing streamer/config
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


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
        self.qdrant_collection: str = os.environ.get('QDRANT_COLLECTION', 'codebase')
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
        # run artifacts
        import time as _t
        self.run_id: str = os.environ.get('RUN_ID', f"run_{int(_t.time())}")
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

    def _build_judge_prompt(self, query: str, results: list[dict]) -> str:
        # delegate to prompts module; prefer explicit judge_mode, then report_mode
        mode = (getattr(self, 'judge_mode', None) or getattr(self, 'report_mode', None) or 'steering')
        return prompts_build_judge_prompt(mode, query, results)

    def _llm_judge(self, results: list[dict]) -> dict[int, dict]:
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
            prompt = self._build_judge_prompt(self.query, results)
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
        if self.retr is None or not isinstance(items, list):
            return items or []
        try:
            # map content to (id, score)
            cmap: dict[str, tuple[int, float]] = {}
            for d in getattr(self.retr, 'documents', []):
                cmap[getattr(d, 'content', '')] = (int(getattr(d, 'id', -1)), float(getattr(d, 'relevance_score', 0.0)))
            out: list[dict] = []
            for it in items:
                c = str(it.get('content', ''))
                did, sc = cmap.get(c, (-1, float(it.get('relevance_score', it.get('score', 0.0)))))
                new_it = dict(it)
                if did != -1:
                    new_it['id'] = did
                if 'score' not in new_it:
                    new_it['score'] = sc
                out.append(new_it)
            return out
        except Exception:
            return items

    def _apply_judgements(self, judged: dict[int, dict]) -> None:
        # votes and boosts
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
                # query pool
                for kw in j.get('keywords', []) or []:
                    if isinstance(kw, str) and kw and kw not in self._query_pool:
                        self._query_pool.append(kw)
                for q in j.get('follow_up_queries', []) or []:
                    if isinstance(q, str) and q and q not in self._query_pool:
                        self._query_pool.append(q)
            except Exception:
                continue

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
        if not self.windows:
            # Fallback default windows if none provided
            self.windows = [1000, 2000, 4000]
            try:
                await self._broadcast({"type": "log", "message": "windows: defaulted to [1000,2000,4000]"})
            except Exception:
                pass
        # Corpus loading: either from codebase (memory backend) or from Qdrant as text chunks
        if (self.vector_backend or 'memory').lower() == 'qdrant':
            try:
                from qdrant_client import QdrantClient  # type: ignore
                client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
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
                rf = 'src'
            else:
                rf = (self.root_folder or os.getcwd())
            docs = collect_codebase_chunks(rf, self.windows, int(self.max_files), self.exclude_dirs, self.chunk_workers)
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
        self.running = True
        self.task = asyncio.create_task(self._run_loop())
        # announce
        try:
            await self._broadcast({"type": "log", "message": f"started: docs={len(getattr(retr,'documents',[]))} agents={len(getattr(retr,'agents',[]))} windows={self.windows}"})
        except Exception:
            pass

    async def stop(self) -> None:
        self.running = False
        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout=1.0)
            except Exception:
                pass
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
                    self.last_metrics = {"step": int(self.step_i), "docs": len(docs), "agents": len(agents), "avg_rel": avg_rel, "max_rel": max_rel, "trails": trails}
                except Exception:
                    self.last_metrics = {"step": int(self.step_i)}

                if self.step_i % max(1, int(self.redraw_every)) == 0:
                    try:
                        snap = self.retr.get_snapshot(min_trail_strength=self.min_trail_strength, max_edges=self.max_edges, method="pca", whiten=False, dims=int(self.viz_dims))
                        await self._broadcast({"type": "snapshot", "step": int(self.step_i), "data": snap})
                        if self.last_metrics is not None:
                            await self._broadcast({"type": "metrics", "data": self.last_metrics})
                        # Always broadcast current Top-K results for this step
                        try:
                            if self.retr is not None:
                                res_now = self.retr.search(self.query, top_k=int(self.top_k))
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
                                            r = self.retr.search(q, top_k=int(self.top_k))
                                            try:
                                                per_counts[q] = len(r.get('results', []) or [])
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
                        except Exception:
                            pass
                        # Per-step report (optional)
                        try:
                            if self.report_enabled and (int(self.step_i) % max(1, int(self.report_every)) == 0) and self.retr is not None:
                                res_top = self.retr.search(self.query, top_k=int(self.top_k))
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
                                    # filter: require concrete target (file path, function/class, endpoint, or line range)
                                    def _is_concrete(q: str) -> bool:
                                        s = (q or "").strip()
                                        if not s:
                                            return False
                                        if re.search(r"\b(lines?\s*[:#-]?\s*\d+(-\d+)?)\b", s):
                                            return True
                                        if ("/" in s) or ("\\" in s):
                                            return True
                                        if re.search(r"\b(def|class)\s+[A-Za-z_][A-Za-z0-9_]*", s):
                                            return True
                                        if re.search(r"@app\.(get|post|put|patch|delete)\(\s*['\"]", s):
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
                                        judgements = self._llm_judge(docs)
                                        self._apply_judgements(judgements)
                                        self._reports_sent += 1
                                        # targeted neighborhood boosts and pruning
                                        self._apply_targeted_fetch(max_neighbors_per_seed=10)
                                        self._apply_pruning()
                                    elif self._stagnant_steps >= int(getattr(self, 'stagnation_threshold', 8)):
                                        await self._broadcast({"type": "log", "message": f"judge: paused due to stagnation (_stagnant_steps={self._stagnant_steps})"})
                                except Exception as _e_judge:
                                    await self._broadcast({"type": "log", "message": f"judge: failed: {_e_judge}"})
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
                                        res = self.retr.search(self.query, top_k=int(self.top_k))
                                        await self._broadcast({"type": "results_stable", "step": int(self.step_i), "data": res.get("results", [])})
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    except Exception as e:
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
app = FastAPI()

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

# serve simple static client
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/introspect/api")
async def http_introspect_api() -> JSONResponse:
    """Introspect server to list HTTP routes and WebSocket event types."""
    try:
        import inspect as _inspect
        import re as _re
        routes: list[dict] = []
        ws_events: set[str] = set()
        # Extract routes from FastAPI app
        for r in getattr(app, 'routes', []):
            try:
                path = getattr(r, 'path', None)
                methods = sorted(list(getattr(r, 'methods', []) or []))
                name = getattr(r, 'name', None)
                if not path or path.startswith('/openapi'):
                    continue
                if path in ['/', '/docs', '/redoc']:
                    continue
                routes.append({"path": path, "methods": methods, "name": name})
            except Exception:
                continue
        # Parse this module's source to find _broadcast({"type": "..."}) usages
        try:
            mod = _inspect.getmodule(http_introspect_api)  # type: ignore
            src = _inspect.getsource(mod) if mod else ""
        except Exception:
            src = ""
        if src:
            for m in _re.finditer(r"_broadcast\(\{[^\}]*\"type\"\s*:\s*\"([^\"]+)\"", src):
                try:
                    ev = m.group(1)
                    if ev:
                        ws_events.add(ev)
                except Exception:
                    continue
        return JSONResponse({"routes": routes, "ws_events": sorted(list(ws_events))})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/run/new")
async def http_run_new() -> JSONResponse:
    try:
        import time as _t
        streamer.run_id = f"run_{int(_t.time())}"
        return JSONResponse({"status": "ok", "run_id": streamer.run_id})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Settings persistence
SETTINGS_DIR = os.path.join(os.getcwd(), ".fungus_cache")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")
os.makedirs(SETTINGS_DIR, exist_ok=True)

def settings_dict() -> dict:
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
        "openai_base_url": streamer.openai_base_url,
        "openai_temperature": streamer.openai_temperature,
        "google_model": streamer.google_model,
        "google_base_url": streamer.google_base_url,
        "google_temperature": streamer.google_temperature,
        "grok_model": streamer.grok_model,
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

def apply_settings(d: dict) -> None:
    try:
        sm = SettingsModel(**d)
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
        if getattr(sm, 'openai_base_url', None) is not None: streamer.openai_base_url = str(getattr(sm, 'openai_base_url'))
        if getattr(sm, 'openai_temperature', None) is not None: streamer.openai_temperature = float(getattr(sm, 'openai_temperature'))
        if getattr(sm, 'google_model', None) is not None: streamer.google_model = str(getattr(sm, 'google_model'))
        if getattr(sm, 'google_base_url', None) is not None: streamer.google_base_url = str(getattr(sm, 'google_base_url'))
        if getattr(sm, 'google_temperature', None) is not None: streamer.google_temperature = float(getattr(sm, 'google_temperature'))
        if getattr(sm, 'grok_model', None) is not None: streamer.grok_model = str(getattr(sm, 'grok_model'))
        if getattr(sm, 'grok_base_url', None) is not None: streamer.grok_base_url = str(getattr(sm, 'grok_base_url'))
        if getattr(sm, 'grok_temperature', None) is not None: streamer.grok_temperature = float(getattr(sm, 'grok_temperature'))
    except Exception:
        pass

def load_settings_from_disk() -> None:
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
                apply_settings(d)
    except Exception:
        pass

def save_settings_to_disk() -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings_dict(), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _prompt_default_for_mode(mode: str) -> str:
    m = (mode or "deep").lower()
    try:
        if m == 'deep':
            return _pm_deep.instructions()  # type: ignore
        if m == 'structure':
            return _pm_structure.instructions()  # type: ignore
        if m == 'exploratory':
            return _pm_exploratory.instructions()  # type: ignore
        if m == 'summary':
            return _pm_summary.instructions()  # type: ignore
        if m == 'repair':
            return _pm_repair.instructions()  # type: ignore
        if m == 'steering':
            return _pm_steering.instructions()  # type: ignore
    except Exception:
        pass
    try:
        return _base_default_instructions(m)
    except Exception:
        return ""


@app.post("/prompts/save")
async def http_prompts_save(req: Request) -> JSONResponse:
    """Persist mode prompt overrides to .fungus_cache/prompts_overrides.json.

    Body: { overrides: { mode: instructions_text } }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    overrides = body.get("overrides", {}) or {}
    if not isinstance(overrides, dict):
        return JSONResponse({"status": "error", "message": "overrides must be an object"}, status_code=400)
    try:
        prompts_dir = os.path.join(SETTINGS_DIR)
        os.makedirs(prompts_dir, exist_ok=True)
        path = os.path.join(prompts_dir, "prompts_overrides.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({ str(k): str(v) for k,v in overrides.items() }, f, ensure_ascii=False, indent=2)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/prompts")
async def http_prompts_get() -> JSONResponse:
    try:
        path = os.path.join(SETTINGS_DIR, "prompts_overrides.json")
        overrides = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    overrides = {str(k): str(v) for k, v in obj.items() if isinstance(v, str)}
        modes = ['deep','structure','exploratory','summary','repair','steering']
        defaults = { m: _prompt_default_for_mode(m) for m in modes }
        return JSONResponse({"status":"ok", "overrides": overrides, "defaults": defaults, "modes": modes})
    except Exception as e:
        return JSONResponse({"status":"error", "message": str(e)}, status_code=500)


def settings_usage_lines(d: dict) -> list[str]:
    """Produce human-readable mapping: Param => Script(s)."""
    # Static knowledge of primary consumers; keep concise
    usage_map = {
        "query": ["mcmp_rag.py (initialize_simulation/search)", "realtime/server.py (/start)"],
        "viz_dims": ["mcmp_rag.py (get_visualization_snapshot)", "frontend (Plotly 2D/3D)"] ,
        "min_trail_strength": ["mcmp/visualize.py (build_snapshot)", "mcmp_rag.py (get_visualization_snapshot)"],
        "max_edges": ["mcmp/visualize.py (build_snapshot)"],
        "redraw_every": ["realtime/server.py (_run_loop WS cadence)"] ,
        "num_agents": ["mcmp/simulation.py (spawn_agents)", "mcmp_rag.py (init MCPMRetriever)"] ,
        "max_iterations": ["realtime/server.py (jobs/start)", "streamlit_fungus_backup.py (loop)"] ,
        "exploration_bonus": ["mcmp/simulation.py (noise/force)"] ,
        "pheromone_decay": ["mcmp/simulation.py (decay_pheromones)"] ,
        "embed_batch_size": ["mcmp_rag.py (add_documents batched encode)"] ,
        "max_chunks_per_shard": ["realtime/server.py (jobs/start sharding)"] ,
        "use_repo": ["realtime/server.py (/start corpus path)"] ,
        "root_folder": ["realtime/server.py (/start corpus path)"] ,
        "max_files": ["ui/corpus.py (list_code_files)"] ,
        "exclude_dirs": ["ui/corpus.py (list_code_files)"],
        "windows": ["ui/corpus.py (chunk_python_file windows)"] ,
        "chunk_workers": ["ui/corpus.py (ThreadPoolExecutor)"],
        "top_k": ["mcmp_rag.py (search top_k)"] ,
        "report_enabled": ["realtime/server.py (_run_loop report)"],
        "report_every": ["realtime/server.py (_run_loop report cadence)"],
        "report_mode": ["realtime/server.py (prompt template)"] ,
        "judge_mode": ["realtime/server.py (judge prompt)"] ,
        "ollama_model": ["realtime/server.py (LLM model)"] ,
        "ollama_host": ["realtime/server.py (LLM host)"] ,
        "ollama_system": ["realtime/server.py (LLM system prompt)"] ,
        "ollama_num_gpu": ["realtime/server.py (LLM GPU opts)"] ,
        "ollama_num_thread": ["realtime/server.py (LLM CPU threads)"] ,
        "ollama_num_batch": ["realtime/server.py (LLM batch)"] ,
        "llm_provider": ["realtime/server.py (choose provider)"] ,
        "openai_model": ["realtime/server.py (OpenAI model)"] ,
        "openai_base_url": ["realtime/server.py (OpenAI endpoint)"] ,
        "openai_temperature": ["realtime/server.py (OpenAI temperature)"] ,
        "mode": ["frontend/UX (prompt style)", "streamlit_fungus_backup.py (mode prompt)"] ,
    }
    lines: list[str] = []
    for k, v in d.items():
        if k in usage_map:
            scripts = usage_map[k]
            lines.append(f"{k}: {v} -> Scripts: {', '.join(scripts)}")
    return lines


class SettingsModel(BaseModel):
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


@app.get("/")
async def index() -> HTMLResponse:
    html_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(html_path):
        # Minimal landing page
        return HTMLResponse("""
<!doctype html>
<html><head><meta charset='utf-8'><title>MCMP Realtime</title></head>
<body>
<h3>MCMP Realtime</h3>
<p>Open <a href='/static/index.html'>client</a> to view the network.</p>
</body></html>
""")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/start")
async def http_start(req: Request) -> JSONResponse:
    raw = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    body = SettingsModel(**raw)
    try:
        await streamer._broadcast({"type": "log", "message": f"api:/start payload_keys={list(raw.keys()) if isinstance(raw, dict) else 'non-json'}"})
    except Exception:
        pass
    # log values which are provided
    try:
        applied_dict = {}
        try:
            applied_dict = body.model_dump(exclude_none=True)  # pydantic v2
        except Exception:
            applied_dict = {k: v for k, v in getattr(body, '__dict__', {}).items() if v is not None}
        await streamer._broadcast({"type": "log", "message": "api:/start applied: " + " ".join([f"{k}={applied_dict[k]}" for k in applied_dict])})
    except Exception:
        pass
    if body.query is not None:
        streamer.query = body.query
    if body.redraw_every is not None:
        streamer.redraw_every = int(body.redraw_every)
    if body.min_trail_strength is not None:
        streamer.min_trail_strength = float(body.min_trail_strength)
    if body.max_edges is not None:
        streamer.max_edges = int(body.max_edges)
    if body.viz_dims is not None:
        streamer.viz_dims = int(body.viz_dims)
    if body.use_repo is not None:
        streamer.use_repo = bool(body.use_repo)
    if body.root_folder is not None:
        streamer.root_folder = str(body.root_folder)
    if body.max_files is not None:
        streamer.max_files = int(body.max_files)
    if body.exclude_dirs is not None:
        streamer.exclude_dirs = [str(x) for x in body.exclude_dirs]
    if body.windows is not None:
        streamer.windows = [int(x) for x in body.windows]
    if body.chunk_workers is not None:
        streamer.chunk_workers = int(body.chunk_workers)
    if body.max_iterations is not None:
        streamer.max_iterations = int(body.max_iterations)
    if body.num_agents is not None:
        streamer.num_agents = int(body.num_agents)
    if body.exploration_bonus is not None:
        streamer.exploration_bonus = float(body.exploration_bonus)
    if body.pheromone_decay is not None:
        streamer.pheromone_decay = float(body.pheromone_decay)
    if body.embed_batch_size is not None:
        streamer.embed_batch_size = int(body.embed_batch_size)
    if body.max_chunks_per_shard is not None:
        streamer.max_chunks_per_shard = int(body.max_chunks_per_shard)
    if body.top_k is not None:
        streamer.top_k = int(body.top_k)
    if getattr(body, 'report_enabled', None) is not None:
        streamer.report_enabled = bool(getattr(body, 'report_enabled'))
    if getattr(body, 'report_every', None) is not None:
        streamer.report_every = int(getattr(body, 'report_every'))
    if getattr(body, 'report_mode', None) is not None:
        streamer.report_mode = str(getattr(body, 'report_mode'))
    if getattr(body, 'judge_mode', None) is not None:
        streamer.judge_mode = str(getattr(body, 'judge_mode'))
    if getattr(body, 'mq_enabled', None) is not None:
        streamer.mq_enabled = bool(getattr(body, 'mq_enabled'))
    if getattr(body, 'mq_count', None) is not None:
        streamer.mq_count = max(1, min(int(getattr(body, 'mq_count')), 3))
    # follow-up and budget controls
    if getattr(body, 'token_cap', None) is not None:
        try:
            streamer.token_cap = int(getattr(body, 'token_cap'))
        except Exception:
            streamer.token_cap = None
    if getattr(body, 'cost_cap_usd', None) is not None:
        try:
            streamer.cost_cap_usd = float(getattr(body, 'cost_cap_usd'))
        except Exception:
            streamer.cost_cap_usd = None
    if getattr(body, 'stagnation_threshold', None) is not None:
        streamer.stagnation_threshold = max(3, int(getattr(body, 'stagnation_threshold')))
    if getattr(body, 'query_pool_cap', None) is not None:
        streamer._query_pool_cap = max(10, int(getattr(body, 'query_pool_cap')))
    # contextual steering settings
    for k in ["alpha","beta","gamma","delta","epsilon","min_content_chars","import_only_penalty","max_reports","max_report_tokens","judge_enabled"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    # LLM configuration overrides
    for k in ["ollama_model","ollama_host","ollama_system","ollama_num_gpu","ollama_num_thread","ollama_num_batch"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    for k in ["llm_provider","openai_model","openai_api_key","openai_base_url","openai_temperature"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    for k in ["google_model","google_api_key","google_base_url","google_temperature","grok_model","grok_api_key","grok_base_url","grok_temperature"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    # OpenAI overrides
    for k in ["llm_provider","openai_model","openai_api_key","openai_base_url","openai_temperature"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    await streamer.start()
    save_settings_to_disk()
    return JSONResponse({"status": "ok"})


@app.post("/config")
async def http_config(req: Request) -> JSONResponse:
    raw = await req.json()
    body = SettingsModel(**raw)
    try:
        await streamer._broadcast({"type": "log", "message": f"api:/config payload_keys={list(raw.keys())}"})
    except Exception:
        pass
    # log values which will be applied
    try:
        applied_dict = {}
        try:
            applied_dict = body.model_dump(exclude_none=True)
        except Exception:
            applied_dict = {k: v for k, v in getattr(body, '__dict__', {}).items() if v is not None}
        await streamer._broadcast({"type": "log", "message": "api:/config applied: " + " ".join([f"{k}={applied_dict[k]}" for k in applied_dict])})
    except Exception:
        pass
    if body.redraw_every is not None:
        streamer.redraw_every = int(body.redraw_every)
    if body.min_trail_strength is not None:
        streamer.min_trail_strength = float(body.min_trail_strength)
    if body.max_edges is not None:
        streamer.max_edges = int(body.max_edges)
    if body.viz_dims is not None:
        streamer.viz_dims = int(body.viz_dims)
    if body.use_repo is not None:
        streamer.use_repo = bool(body.use_repo)
    if body.root_folder is not None:
        streamer.root_folder = str(body.root_folder)
    if body.max_files is not None:
        streamer.max_files = int(body.max_files)
    if body.exclude_dirs is not None:
        streamer.exclude_dirs = [str(x) for x in body.exclude_dirs]
    if body.windows is not None:
        streamer.windows = [int(x) for x in body.windows]
    if body.chunk_workers is not None:
        streamer.chunk_workers = int(body.chunk_workers)
    if body.max_iterations is not None:
        streamer.max_iterations = int(body.max_iterations)
    if body.num_agents is not None:
        streamer.num_agents = int(body.num_agents)
    if body.exploration_bonus is not None:
        streamer.exploration_bonus = float(body.exploration_bonus)
    if body.pheromone_decay is not None:
        streamer.pheromone_decay = float(body.pheromone_decay)
    if body.embed_batch_size is not None:
        streamer.embed_batch_size = int(body.embed_batch_size)
    if body.max_chunks_per_shard is not None:
        streamer.max_chunks_per_shard = int(body.max_chunks_per_shard)
    if body.top_k is not None:
        streamer.top_k = int(body.top_k)
    if getattr(body, 'report_enabled', None) is not None:
        streamer.report_enabled = bool(getattr(body, 'report_enabled'))
    if getattr(body, 'report_every', None) is not None:
        streamer.report_every = int(getattr(body, 'report_every'))
    if getattr(body, 'report_mode', None) is not None:
        streamer.report_mode = str(getattr(body, 'report_mode'))
    if getattr(body, 'judge_mode', None) is not None:
        streamer.judge_mode = str(getattr(body, 'judge_mode'))
    if getattr(body, 'mq_enabled', None) is not None:
        streamer.mq_enabled = bool(getattr(body, 'mq_enabled'))
    if getattr(body, 'mq_count', None) is not None:
        streamer.mq_count = int(getattr(body, 'mq_count'))
    # contextual steering settings
    for k in ["alpha","beta","gamma","delta","epsilon","min_content_chars","import_only_penalty","max_reports","max_report_tokens","judge_enabled"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    # LLM configuration overrides
    for k in ["ollama_model","ollama_host","ollama_system","ollama_num_gpu","ollama_num_thread","ollama_num_batch"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    save_settings_to_disk()
    return JSONResponse({"status": "ok"})


@app.post("/stop")
async def http_stop() -> JSONResponse:
    # Aggregate per-step usage files into a run summary on stop
    try:
        run_id = str(getattr(streamer, 'run_id', 'run'))
        base_dir = os.path.join(SETTINGS_DIR, "runs", run_id)
        total = {"by_provider": {}, "by_model": {}, "totals": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        import glob as _glob, json as _json
        for step_dir in sorted(_glob.glob(os.path.join(base_dir, "step_*"))):
            for name in ("usage.json", "judge_usage.json"):
                p = os.path.join(step_dir, name)
                try:
                    if os.path.exists(p):
                        with open(p, 'r', encoding='utf-8') as f:
                            u = _json.load(f)
                        prov = str(u.get('provider', 'unknown'))
                        model = str(u.get('model', 'unknown'))
                        # favor exact tokens; fall back to *_est
                        pt = int(u.get('prompt_tokens', u.get('prompt_tokens_est', 0)) or 0)
                        ct = int(u.get('completion_tokens', u.get('completion_tokens_est', 0)) or 0)
                        tt = int(u.get('total_tokens', u.get('total_tokens_est', pt + ct)) or (pt + ct))
                        total['totals']['prompt_tokens'] += pt
                        total['totals']['completion_tokens'] += ct
                        total['totals']['total_tokens'] += tt
                        total['by_provider'].setdefault(prov, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                        total['by_provider'][prov]['prompt_tokens'] += pt
                        total['by_provider'][prov]['completion_tokens'] += ct
                        total['by_provider'][prov]['total_tokens'] += tt
                        total['by_model'].setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                        total['by_model'][model]['prompt_tokens'] += pt
                        total['by_model'][model]['completion_tokens'] += ct
                        total['by_model'][model]['total_tokens'] += tt
                except Exception:
                    continue
        # Rough cost estimation (USD per 1K tokens); configurable via env
        PRICES = {
            'openai:gpt-4o': {"prompt": float(os.environ.get('PRICE_OPENAI_GPT4O_PROMPT', '0.005')), "completion": float(os.environ.get('PRICE_OPENAI_GPT4O_COMPLETION', '0.015'))},
            'openai:gpt-4o-mini': {"prompt": float(os.environ.get('PRICE_OPENAI_GPT4OM_PROMPT', '0.0005')), "completion": float(os.environ.get('PRICE_OPENAI_GPT4OM_COMPLETION', '0.0015'))},
        }
        costs = {"by_model": {}, "total_usd": 0.0}
        for model, v in total['by_model'].items():
            key = 'openai:' + model if not model.startswith('openai:') else model
            price = PRICES.get(key)
            if price:
                usd = (v['prompt_tokens'] / 1000.0) * price['prompt'] + (v['completion_tokens'] / 1000.0) * price['completion']
                costs['by_model'][model] = round(usd, 6)
                costs['total_usd'] += usd
        costs['total_usd'] = round(costs['total_usd'], 6)
        out = {"usage": total, "costs": costs}
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "run_costs.json"), 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        try:
            await streamer._broadcast({"type": "log", "message": f"run_costs: total_tokens={total['totals']['total_tokens']} total_usd={costs['total_usd']}"})
        except Exception:
            pass
    except Exception:
        pass
    await streamer.stop()
    return JSONResponse({"status": "stopped"})


@app.post("/reset")
async def http_reset() -> JSONResponse:
    # Fully stop and clear simulation state so a fresh /start rebuilds corpus and retriever
    try:
        await streamer.stop()
    except Exception:
        pass
    try:
        streamer.retr = None
        streamer.step_i = 0
        streamer.last_metrics = None
        streamer._paused = False
        streamer._saved_state = None
        streamer._avg_rel_history = []
        # Clear LLM/reporting related state and caches
        streamer._reports_sent = 0
        streamer._tokens_used = 0
        streamer._judge_cache = {}
        streamer._llm_vote = {}
        streamer._doc_boost = {}
        streamer._query_pool = []
        streamer._seeds_queue = []
        # Clear background jobs and force corpus rebuild detection
        streamer.jobs = {}
        streamer._corpus_fingerprint = None
        # Keep configuration (query, windows, etc.) so the next /start can reuse or override
        await streamer._broadcast({"type": "log", "message": "simulation reset"})
    except Exception:
        pass
    return JSONResponse({"status": "reset"})


@app.post("/pause")
async def http_pause() -> JSONResponse:
    streamer._paused = True
    # save minimal state
    try:
        retr = streamer.retr
        if retr is not None:
            streamer._saved_state = {
                "agents": [
                    {
                        "position": getattr(a, 'position', None).tolist() if getattr(a, 'position', None) is not None else None,
                        "velocity": getattr(a, 'velocity', None).tolist() if getattr(a, 'velocity', None) is not None else None,
                        "age": int(getattr(a, 'age', 0)),
                    }
                    for a in getattr(retr, 'agents', [])
                ]
            }
    except Exception:
        streamer._saved_state = None
    return JSONResponse({"status": "paused"})


@app.post("/resume")
async def http_resume() -> JSONResponse:
    streamer._paused = False
    # restore minimal state
    try:
        retr = streamer.retr
        if retr is not None and streamer._saved_state:
            agents = streamer._saved_state.get("agents", [])
            for i, a in enumerate(getattr(retr, 'agents', [])):
                if i < len(agents):
                    st = agents[i]
                    import numpy as _np
                    if st.get('position') is not None:
                        a.position = _np.array(st['position'], dtype=_np.float32)
                    if st.get('velocity') is not None:
                        a.velocity = _np.array(st['velocity'], dtype=_np.float32)
                    a.age = int(st.get('age', getattr(a, 'age', 0)))
    except Exception:
        pass
    return JSONResponse({"status": "resumed"})


@app.post("/agents/add")
async def http_agents_add(req: Request) -> JSONResponse:
    """Append N new agents with random positions/velocities in embedding space.
    Recommended to call while paused to avoid visual glitches.
    Body: { n: int }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    n = int(body.get("n", 0))
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    if not getattr(retr, "documents", []):
        return JSONResponse({"status": "error", "message": "no documents indexed"}, status_code=400)
    if n <= 0:
        return JSONResponse({"status": "error", "message": "n must be > 0"}, status_code=400)
    try:
        dim = int(retr.documents[0].embedding.shape[0])  # type: ignore
        import numpy as _np
        current_len = len(getattr(retr, 'agents', []))
        max_id = max([getattr(a, 'id', -1) for a in getattr(retr, 'agents', [])], default=-1)
        for i in range(n):
            pos = _np.random.normal(0, 1.0, size=(dim,)).astype(_np.float32)
            norm = float(_np.linalg.norm(pos)) or 1.0
            pos = pos / norm
            vel = _np.random.normal(0, 0.05, size=(dim,)).astype(_np.float32)
            agent = retr.Agent(  # type: ignore[attr-defined]
                id=int(max_id + 1 + i),
                position=pos,
                velocity=vel,
                exploration_factor=float(_np.random.uniform(0.05, max(0.05, float(getattr(retr, 'exploration_bonus', 0.1))))),
            )
            retr.agents.append(agent)
        retr.num_agents = int(len(retr.agents))
        return JSONResponse({"status": "ok", "added": int(n), "agents": int(len(retr.agents))})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/agents/resize")
async def http_agents_resize(req: Request) -> JSONResponse:
    """Resize agent population to target count. Adds random agents or trims list.
    Body: { count: int }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    target = int(body.get("count", -1))
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    if target < 0:
        return JSONResponse({"status": "error", "message": "count must be >= 0"}, status_code=400)
    cur = len(getattr(retr, 'agents', []))
    if target == cur:
        return JSONResponse({"status": "ok", "agents": cur})
    if target < cur:
        try:
            retr.agents = list(retr.agents)[:target]
            retr.num_agents = int(target)
            return JSONResponse({"status": "ok", "agents": int(len(retr.agents))})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    # Need to add agents
    to_add = int(target - cur)
    # Reuse add logic
    fake_req = Request({'type': 'http'})  # type: ignore
    fake_req._body = json.dumps({"n": to_add}).encode("utf-8")  # type: ignore
    return await http_agents_add(fake_req)


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


@app.get("/status")
async def http_status() -> JSONResponse:
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"running": False})
    return JSONResponse({
        "running": streamer.running,
        "docs": len(getattr(retr, 'documents', [])),
        "agents": len(getattr(retr, 'agents', [])),
        "metrics": streamer.last_metrics or {},
    })


@app.get("/settings")
async def http_settings_get() -> JSONResponse:
    sd = settings_dict()
    return JSONResponse({"settings": sd, "usage": settings_usage_lines(sd)})


@app.post("/settings")
async def http_settings_post(req: Request) -> JSONResponse:
    try:
        body = await req.json()
        apply_settings(body)
        save_settings_to_disk()
        sd = settings_dict()
        return JSONResponse({"status": "ok", "settings": sd, "usage": settings_usage_lines(sd)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.post("/search")
async def http_search(req: Request) -> JSONResponse:
    body = await req.json()
    query = str(body.get("query", ""))
    top_k = int(body.get("top_k", 5))
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    try:
        res = retr.search(query, top_k=top_k)
        return JSONResponse({"status": "ok", "results": res.get('results', [])})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/answer")
async def http_answer(req: Request) -> JSONResponse:
    body = await req.json()
    query = str(body.get("query", ""))
    top_k = int(body.get("top_k", 5))
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    res = retr.search(query, top_k=top_k)
    ctx = "\n\n".join([(it.get('content') or '')[:800] for it in res.get('results', [])])
    prompt = f"Kontext:\n{ctx}\n\nAufgabe:\n{query}\n\nAntwort:"
    llm_opts = {}
    try:
        if os.environ.get('OLLAMA_NUM_GPU'):
            llm_opts['num_gpu'] = int(os.environ.get('OLLAMA_NUM_GPU'))
        if os.environ.get('OLLAMA_NUM_THREAD'):
            llm_opts['num_thread'] = int(os.environ.get('OLLAMA_NUM_THREAD'))
        if os.environ.get('OLLAMA_NUM_BATCH'):
            llm_opts['num_batch'] = int(os.environ.get('OLLAMA_NUM_BATCH'))
    except Exception:
        llm_opts = {}
    answer_prompt_path = os.path.join(SETTINGS_DIR, "reports/answer_prompt.txt")
    try:
        # also write under run folder
        run_dir = os.path.join(SETTINGS_DIR, "runs", str(getattr(streamer, 'run_id', 'run')), f"step_{int(getattr(streamer, 'step_i', 0))}")
        os.makedirs(run_dir, exist_ok=True)
        answer_prompt_path = os.path.join(run_dir, "answer_prompt.txt")
    except Exception:
        pass
    text = generate_text(
        provider=(streamer.llm_provider or 'ollama'),
        prompt=prompt,
        system=streamer.ollama_system,
        ollama_model=streamer.ollama_model,
        ollama_host=streamer.ollama_host,
        ollama_options=(llm_opts or None),
        openai_model=streamer.openai_model,
        openai_api_key=(streamer.openai_api_key or ''),
        openai_base_url=(streamer.openai_base_url or 'https://api.openai.com'),
        openai_temperature=float(getattr(streamer, 'openai_temperature', 0.0)),
        google_model=streamer.google_model,
        google_api_key=(streamer.google_api_key or ''),
        google_base_url=(streamer.google_base_url or 'https://generativelanguage.googleapis.com'),
        google_temperature=float(getattr(streamer, 'google_temperature', 0.0)),
        grok_model=streamer.grok_model,
        grok_api_key=(streamer.grok_api_key or ''),
        grok_base_url=(streamer.grok_base_url or 'https://api.x.ai'),
        grok_temperature=float(getattr(streamer, 'grok_temperature', 0.0)),
        save_prompt_path=answer_prompt_path,
    )
    return JSONResponse({"status": "ok", "answer": text, "results": res.get('results', [])})


# Document detail endpoint
@app.get("/doc/{doc_id}")
async def http_doc(doc_id: int) -> JSONResponse:
    retr = streamer.retr
    if retr is None:
        return JSONResponse({"status": "error", "message": "retriever not started"}, status_code=400)
    try:
        d = next((x for x in getattr(retr, 'documents', []) if int(getattr(x, 'id', -1)) == int(doc_id)), None)
        if d is None:
            return JSONResponse({"status": "error", "message": "doc not found"}, status_code=404)
        emb = getattr(d, 'embedding', None)
        emb_list = emb.tolist() if emb is not None else []
        meta = getattr(d, 'metadata', {}) or {}
        return JSONResponse({
            "status": "ok",
            "doc": {
                "id": int(getattr(d, 'id', -1)),
                "content": getattr(d, 'content', ''),
                "embedding": emb_list,
                "relevance_score": float(getattr(d, 'relevance_score', 0.0)),
                "visit_count": int(getattr(d, 'visit_count', 0)),
                "metadata": meta,
            }
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Corpus listing endpoint
@app.get("/corpus/list")
async def http_corpus_list(root: str | None = None, page: int = 1, page_size: int = 200, exclude: str | None = None) -> JSONResponse:
    try:
        page = max(1, int(page))
        page_size = max(1, min(2000, int(page_size)))
        ex = [e.strip() for e in (exclude or "").split(',') if e and e.strip()]
        use_root = root or ('src' if streamer.use_repo else streamer.root_folder)
        files = list_code_files(use_root, 0, ex or streamer.exclude_dirs)
        total = len(files)
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        return JSONResponse({"root": use_root, "total": total, "page": page, "page_size": page_size, "files": files[start:end]})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/corpus/summary")
async def http_corpus_summary() -> JSONResponse:
    try:
        q_count = None
        try:
            if (streamer.vector_backend or 'memory').lower() == 'qdrant':
                from qdrant_client import QdrantClient  # type: ignore
                client = QdrantClient(url=streamer.qdrant_url, api_key=streamer.qdrant_api_key)
                cnt = client.count(collection_name=streamer.qdrant_collection, exact=True)
                q_count = int(getattr(cnt, 'count', None) or 0)
        except Exception:
            q_count = None
        sim_docs = len(getattr(streamer.retr, 'documents', [])) if streamer.retr else 0
        return JSONResponse({
            "vector_backend": streamer.vector_backend,
            "qdrant_url": streamer.qdrant_url,
            "qdrant_collection": streamer.qdrant_collection,
            "qdrant_points": q_count,
            "simulation_docs": sim_docs,
            "run_id": getattr(streamer, 'run_id', ''),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


def _load_embed_client():
    from embeddinggemma.mcmp.embeddings import load_sentence_model  # lazy import
    model_name = os.environ.get('EMBEDDING_MODEL', 'google/embeddinggemma-300m')
    try:
        if not model_name or model_name.strip() == 'google/embeddinggemma-300m':
            if os.environ.get('OPENAI_API_KEY'):
                model_name = 'openai:text-embedding-3-large'
    except Exception:
        pass
    device_mode = os.environ.get('DEVICE_MODE', 'auto')
    return load_sentence_model(model_name, device_mode)


def _encode_texts(embedder, texts: list[str]) -> list[list[float]]:
    try:
        # OpenAI adapter returns list[list[float]] directly
        vecs = embedder.encode(texts)
        if isinstance(vecs, list):
            return [list(map(float, v)) for v in vecs]
        # SentenceTransformers -> numpy array
        import numpy as _np
        if hasattr(vecs, 'tolist'):
            return [list(map(float, v)) for v in _np.asarray(vecs).tolist()]
        return [list(map(float, v)) for v in vecs]
    except Exception as e:
        raise RuntimeError(f"embedding failed: {e}")


@app.post("/corpus/add_file")
async def http_corpus_add_file(req: Request) -> JSONResponse:
    """Chunk a file, embed chunks, upsert to Qdrant, and reload simulation.

    Body: { path: string }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    path = str(body.get('path', '')).strip()
    if not path:
        return JSONResponse({"status": "error", "message": "path is required"}, status_code=400)
    if (streamer.vector_backend or 'memory').lower() != 'qdrant':
        return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)
    try:
        # Chunk
        from embeddinggemma.ui.corpus import chunk_python_file
        chunks = chunk_python_file(path, streamer.windows or [max(1, 1000)])
        if not chunks:
            return JSONResponse({"status": "error", "message": "no chunks produced"}, status_code=400)
        # Prepare payloads and texts
        payloads: list[dict] = []
        texts: list[str] = []
        for ch in chunks:
            first = (ch.splitlines() or [""])[0]
            p, a, b, _w = _parse_chunk_header_line(first)
            body_txt = "\n".join(ch.splitlines()[1:])
            payloads.append({"path": p or os.path.relpath(path), "start": a or 1, "end": b or 1 + len(body_txt.splitlines()), "text": body_txt})
            texts.append(body_txt)
        # Embed
        embedder = _load_embed_client()
        vectors = _encode_texts(embedder, texts)
        # Upsert
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        client = QdrantClient(url=streamer.qdrant_url, api_key=streamer.qdrant_api_key)
        pts = []
        import uuid as _uuid
        for vec, pl in zip(vectors, payloads):
            pts.append(PointStruct(id=str(_uuid.uuid4()), vector=vec, payload=pl))
        client.upsert(collection_name=streamer.qdrant_collection, points=pts)
        try:
            await streamer._broadcast({"type": "log", "message": f"qdrant: upserted chunks={len(pts)} for {path}"})
        except Exception:
            pass
        # Reload simulation
        await streamer.stop()
        await streamer.start()
        return JSONResponse({"status": "ok", "chunks": len(pts)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/corpus/update_file")
async def http_corpus_update_file(req: Request) -> JSONResponse:
    """Delete existing chunks for a path in Qdrant, then add_file flow, and reload."""
    try:
        body = await req.json()
    except Exception:
        body = {}
    path = str(body.get('path', '')).strip()
    if not path:
        return JSONResponse({"status": "error", "message": "path is required"}, status_code=400)
    if (streamer.vector_backend or 'memory').lower() != 'qdrant':
        return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = QdrantClient(url=streamer.qdrant_url, api_key=streamer.qdrant_api_key)
        flt = Filter(must=[FieldCondition(key="path", match=MatchValue(value=os.path.relpath(path)))])
        client.delete(collection_name=streamer.qdrant_collection, points_selector=flt)
        # reuse add_file logic by faking request
        fake = Request({'type': 'http'})  # type: ignore
        fake._body = json.dumps({"path": path}).encode('utf-8')  # type: ignore
        return await http_corpus_add_file(fake)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
@app.post("/corpus/reindex")
async def http_corpus_reindex(req: Request) -> JSONResponse:
    """Rebuild corpus and reinitialize retriever if files changed (or force).

    Body: { force: bool }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    force = bool(body.get("force", False))
    try:
        if streamer.use_repo:
            rf = 'src'
        else:
            rf = (streamer.root_folder or os.getcwd())
        files = list_code_files(rf, int(streamer.max_files), streamer.exclude_dirs)
        h = hashlib.sha1()
        for p in sorted(files):
            try:
                h.update(p.encode('utf-8', errors='ignore'))
                h.update(str(os.path.getsize(p)).encode('utf-8'))
            except Exception:
                continue
        new_fp = h.hexdigest()
        changed = (new_fp != streamer._corpus_fingerprint)
        if not changed and not force:
            return JSONResponse({"status": "ok", "changed": False, "message": "No file changes detected"})
        # stop if running
        try:
            await streamer._broadcast({"type": "log", "message": "reindex: starting"})
        except Exception:
            pass
        await streamer.stop()
        # rebuild docs and retriever
        docs = collect_codebase_chunks(rf, streamer.windows, int(streamer.max_files), streamer.exclude_dirs, streamer.chunk_workers)
        retr = MCPMRetriever(
            embedding_model_name=os.environ.get('EMBEDDING_MODEL', 'google/embeddinggemma-300m'),
            num_agents=streamer.num_agents,
            max_iterations=streamer.max_iterations,
            exploration_bonus=streamer.exploration_bonus,
            pheromone_decay=streamer.pheromone_decay,
            embed_batch_size=streamer.embed_batch_size,
            device_mode=os.environ.get('DEVICE_MODE', 'auto'),
        )
        retr.add_documents(docs)
        retr.initialize_simulation(streamer.query)
        streamer.retr = retr
        streamer._corpus_fingerprint = new_fp
        try:
            await streamer._broadcast({"type": "log", "message": f"reindex: complete docs={len(docs)}"})
        except Exception:
            pass
        return JSONResponse({"status": "ok", "changed": True, "docs": len(docs)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Bulk index repository into Qdrant and reload simulation
@app.post("/corpus/index_repo")
async def http_corpus_index_repo(req: Request) -> JSONResponse:
    """Chunk and embed all repo files to Qdrant, then restart simulation.

    Body: { root?: string, exclude_dirs?: string[] }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    if (streamer.vector_backend or 'memory').lower() != 'qdrant':
        return JSONResponse({"status": "error", "message": "vector_backend must be qdrant"}, status_code=400)
    root = str(body.get('root') or ('src' if streamer.use_repo else (streamer.root_folder or os.getcwd())))
    exclude = body.get('exclude_dirs') or streamer.exclude_dirs
    try:
        from embeddinggemma.ui.corpus import list_code_files, chunk_python_file  # type: ignore
        files = list_code_files(root, int(streamer.max_files), exclude)
        if not files:
            return JSONResponse({"status": "error", "message": "no files found to index"}, status_code=400)
        # Embedder and Qdrant client
        embedder = _load_embed_client()
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        client = QdrantClient(url=streamer.qdrant_url, api_key=streamer.qdrant_api_key)
        total_pts = 0
        for i, path in enumerate(files):
            try:
                chunks = chunk_python_file(path, streamer.windows or [max(1, 1000)])
                if not chunks:
                    continue
                payloads: list[dict] = []
                texts: list[str] = []
                for ch in chunks:
                    first = (ch.splitlines() or [""])[0]
                    p, a, b, _w = _parse_chunk_header_line(first)
                    body_txt = "\n".join(ch.splitlines()[1:])
                    payloads.append({"path": p or os.path.relpath(path), "start": a or 1, "end": b or 1 + len(body_txt.splitlines()), "text": body_txt})
                    texts.append(body_txt)
                vectors = _encode_texts(embedder, texts)
                import uuid as _uuid
                pts = [PointStruct(id=str(_uuid.uuid4()), vector=v, payload=pl) for v, pl in zip(vectors, payloads)]
                if pts:
                    client.upsert(collection_name=streamer.qdrant_collection, points=pts)
                    total_pts += len(pts)
                if i % 10 == 0:
                    try:
                        await streamer._broadcast({"type": "log", "message": f"qdrant: indexed files={i+1}/{len(files)} points={total_pts}"})
                    except Exception:
                        pass
            except Exception as e:
                try:
                    await streamer._broadcast({"type": "log", "message": f"index_repo: skipped {path}: {e}"})
                except Exception:
                    pass
                continue
        # Reload simulation
        await streamer.stop()
        await streamer.start()
        return JSONResponse({"status": "ok", "files": len(files), "points": total_pts})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Background shard job
@app.post("/jobs/start")
async def http_jobs_start(req: Request) -> JSONResponse:
    body = await req.json()
    q = str(body.get("query", streamer.query))
    job_id = str(len(streamer.jobs) + 1)
    streamer.jobs[job_id] = {"status": "running", "progress": 0}

    async def _job():
        try:
            # Build corpus
            if streamer.use_repo:
                docs = collect_codebase_chunks('src', streamer.windows, int(streamer.max_files), streamer.exclude_dirs)
            else:
                docs = collect_codebase_chunks(streamer.root_folder or os.getcwd(), streamer.windows, int(streamer.max_files), streamer.exclude_dirs)
            total_chunks = len(docs)
            shard_size = int(streamer.max_chunks_per_shard)
            if shard_size <= 0 or shard_size >= total_chunks:
                shard_ranges = [(0, total_chunks)]
            else:
                shard_ranges = [(i, min(i + shard_size, total_chunks)) for i in range(0, total_chunks, shard_size)]
            num_shards = max(1, len(shard_ranges))
            agg = []
            for idx, (s, e) in enumerate(shard_ranges):
                retr = MCPMRetriever(num_agents=streamer.num_agents, max_iterations=streamer.max_iterations, exploration_bonus=streamer.exploration_bonus, pheromone_decay=streamer.pheromone_decay, embed_batch_size=streamer.embed_batch_size)
                retr.add_documents(docs[s:e])
                retr.initialize_simulation(q)
                for _ in range(streamer.max_iterations):
                    retr.step(1)
                res = retr.search(q, top_k=5)
                agg.extend(res.get('results', []))
                pct = int(100 * (idx + 1) / num_shards)
                await streamer._broadcast({"type": "job_progress", "job_id": job_id, "percent": pct, "message": f"Processed shard {idx+1}/{num_shards}"})
            streamer.jobs[job_id] = {"status": "done", "progress": 100, "results": agg}
        except Exception as e:
            streamer.jobs[job_id] = {"status": "error", "message": str(e)}

    asyncio.create_task(_job())
    return JSONResponse({"status": "ok", "job_id": job_id})


@app.get("/jobs/status")
async def http_jobs_status(job_id: str):
    j = streamer.jobs.get(job_id)
    if not j:
        return JSONResponse({"status": "error", "message": "unknown job"}, status_code=404)
    return JSONResponse({"status": "ok", "job": j})


@app.post("/reports/merge")
async def http_reports_merge(req: Request) -> JSONResponse:
    """Merge multiple per-step or background reports into a single summary.json.

    Body (optional): {
      paths?: string[]  // explicit report.json paths; if omitted, auto-discover recent ones
      out_path?: string // output path for the summary (default: .fungus_cache/reports/summary.json)
      max_reports?: number // cap auto-discovery
    }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    paths = body.get('paths') if isinstance(body.get('paths'), list) else None
    out_path = body.get('out_path') if isinstance(body.get('out_path'), str) else None
    res = merge_reports_to_summary(paths=paths, out_path=out_path)
    # Best-effort: include the merged summary content for UI download
    summary_obj = None
    try:
        with open(res.get('summary_path') or '', 'r', encoding='utf-8') as f:
            summary_obj = json.load(f)
    except Exception:
        summary_obj = None
    return JSONResponse({"status": "ok", "data": res, "summary": summary_obj})


