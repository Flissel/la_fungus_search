from __future__ import annotations
from typing import List, Dict, Any, Set
import os
import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

import numpy as np

from embeddinggemma.mcmp_rag import MCPMRetriever
from embeddinggemma.ui.corpus import collect_codebase_chunks, list_code_files  # type: ignore
from embeddinggemma.rag.generation import generate_with_ollama  # type: ignore
from embeddinggemma.prompts import get_report_instructions


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


def _report_schema_hint() -> str:
    return (
        "Return ONLY JSON with key 'items' (array). Each item must include: "
        "code_chunk (str), content (str), file_path (str), line_range ([int,int]), "
        "code_purpose (str), code_dependencies (str[] or str), file_type (str), "
        "embedding_score (number), relevance_to_query (str), query_initial (str), follow_up_queries (str[])."
    )


def _build_report_prompt(mode: str, query: str, top_k: int, docs: list[dict]) -> str:
    mode = (mode or "deep").lower()
    ctx = "\n\n".join([(it.get("content") or "")[:1200] for it in docs])
    base = f"Mode: {mode}\nQuery: {query}\nTopK: {int(top_k)}\n\nContext begins:\n{ctx}\n\nContext ends.\n\n"
    schema = _report_schema_hint()
    instr = get_report_instructions(mode)
    return base + instr + "\n" + schema + " Answer with JSON only."


class SnapshotStreamer:
    def __init__(self) -> None:
        self.retr: MCPMRetriever | None = None
        self.clients: Set[WebSocket] = set()
        self.running: bool = False
        self.task: asyncio.Task | None = None
        # visualization config
        self.redraw_every: int = 2
        self.min_trail_strength: float = 0.05
        self.max_edges: int = 600
        self.viz_dims: int = 2
        self.query: str = "Explain the architecture."
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
        self._seeds_queue: list[int] = []

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
        # Keep prompt compact; pass only needed fields
        items = []
        for it in results:
            try:
                items.append({
                    'doc_id': int(it.get('id', it.get('doc_id', -1))),
                    'score': float(it.get('score', 0.0)),
                    'content': (it.get('content') or '')[:1200],
                })
            except Exception:
                continue
        schema = (
            "Return ONLY JSON with key 'items' (array). Each item must include: "
            "doc_id (int), is_relevant (bool), why (str), entry_point (bool), "
            "missing_context (str[]), follow_up_queries (str[]), keywords (str[]), inspect (str[])."
        )
        return (
            f"Query: {query}\n\n" +
            "Evaluate the following code chunks for relevance to the query. "
            "Mark entry_point for main functions, API routes, or top-level orchestrators.\n\n" +
            json.dumps({'chunks': items}, ensure_ascii=False) + "\n\n" +
            schema
        )

    def _llm_judge(self, results: list[dict]) -> dict[int, dict]:
        # Enforce token/step budget (approximate by characters)
        if int(self._reports_sent) >= int(self.max_reports):
            return {}
        judged: dict[int, dict] = {}
        try:
            prompt = self._build_judge_prompt(self.query, results)
            self._tokens_used += len(prompt)
            # basic budget check
            if int(self._tokens_used) > int(self.max_report_tokens):
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
                if os.environ.get('OLLAMA_NUM_GPU'):
                    llm_opts['num_gpu'] = int(os.environ.get('OLLAMA_NUM_GPU'))
                if os.environ.get('OLLAMA_NUM_THREAD'):
                    llm_opts['num_thread'] = int(os.environ.get('OLLAMA_NUM_THREAD'))
                if os.environ.get('OLLAMA_NUM_BATCH'):
                    llm_opts['num_batch'] = int(os.environ.get('OLLAMA_NUM_BATCH'))
            except Exception:
                llm_opts = {}
            judge_prompt_path = os.path.join(SETTINGS_DIR, f"reports/judge_prompt_step_{int(self.step_i)}.txt")
            text = generate_with_ollama(
                prompt,
                model=os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b'),
                host=os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434'),
                system=os.environ.get('OLLAMA_SYSTEM'),
                options=(llm_opts or None),
                save_prompt_path=judge_prompt_path,
            )
            raw = (text or "").strip()
            if raw.startswith("```"):
                raw = "\n".join([ln for ln in raw.splitlines() if not ln.strip().startswith("```")])
            obj = json.loads(raw)
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
        embedding_model_name = os.environ.get('EMBEDDING_MODEL', 'google/embeddinggemma-300m')
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
            raise RuntimeError("windows (chunk sizes) must be provided by frontend")
        if self.use_repo:
            docs = collect_codebase_chunks('src', self.windows, int(self.max_files), self.exclude_dirs, self.chunk_workers)
        else:
            rf = (self.root_folder or os.getcwd())
            docs = collect_codebase_chunks(rf, self.windows, int(self.max_files), self.exclude_dirs, self.chunk_workers)
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
                                        if os.environ.get('OLLAMA_NUM_GPU'):
                                            llm_opts['num_gpu'] = int(os.environ.get('OLLAMA_NUM_GPU'))
                                        if os.environ.get('OLLAMA_NUM_THREAD'):
                                            llm_opts['num_thread'] = int(os.environ.get('OLLAMA_NUM_THREAD'))
                                        if os.environ.get('OLLAMA_NUM_BATCH'):
                                            llm_opts['num_batch'] = int(os.environ.get('OLLAMA_NUM_BATCH'))
                                    except Exception:
                                        llm_opts = {}
                                    prompt_path = os.path.join(SETTINGS_DIR, f"reports/prompt_step_{int(self.step_i)}.txt")
                                    text = generate_with_ollama(
                                        prompt,
                                        model=os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b'),
                                        host=os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434'),
                                        system=os.environ.get('OLLAMA_SYSTEM'),
                                        options=(llm_opts or None),
                                        save_prompt_path=prompt_path,
                                    )
                                except Exception as e:
                                    await self._broadcast({"type": "log", "message": f"report: LLM error: {e}"})
                                    text = "{}"
                                # Parse JSON best-effort
                                try:
                                    raw = (text or "").strip()
                                    if raw.startswith("```"):
                                        raw = "\n".join([ln for ln in raw.splitlines() if not ln.strip().startswith("```")])
                                    report_obj = json.loads(raw)
                                except Exception:
                                    report_obj = {"items": []}
                                # Save and broadcast
                                try:
                                    reports_dir = os.path.join(SETTINGS_DIR, "reports")
                                    os.makedirs(reports_dir, exist_ok=True)
                                    pth = os.path.join(reports_dir, f"step_{int(self.step_i)}.json")
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
                                # Contextual steering: judge + actions (budgeted)
                                try:
                                    if self.judge_enabled and (self._reports_sent < int(self.max_reports)):
                                        judgements = self._llm_judge(docs)
                                        self._apply_judgements(judgements)
                                        self._reports_sent += 1
                                        # targeted neighborhood boosts and pruning
                                        self._apply_targeted_fetch(max_neighbors_per_seed=10)
                                        self._apply_pruning()
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

    @validator('viz_dims')
    def _dims(cls, v):  # type: ignore
        if v is None:
            return v
        if int(v) not in (2, 3):
            raise ValueError('viz_dims must be 2 or 3')
        return int(v)


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
    # contextual steering settings
    for k in ["alpha","beta","gamma","delta","epsilon","min_content_chars","import_only_penalty","max_reports","max_report_tokens","judge_enabled"]:
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
    # contextual steering settings
    for k in ["alpha","beta","gamma","delta","epsilon","min_content_chars","import_only_penalty","max_reports","max_report_tokens","judge_enabled"]:
        if getattr(body, k, None) is not None:
            setattr(streamer, k, getattr(body, k))
    save_settings_to_disk()
    return JSONResponse({"status": "ok"})


@app.post("/stop")
async def http_stop() -> JSONResponse:
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
    text = generate_with_ollama(
        prompt,
        model=os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b'),
        host=os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434'),
        system=os.environ.get('OLLAMA_SYSTEM'),
        options=(llm_opts or None),
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


