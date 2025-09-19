import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from .corpus import collect_codebase_chunks, list_code_files
from .mcmp_runner import select_diverse_results

REPORTS_DIR = os.path.join(".fungus_cache", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

BG_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _progress_path(job_id: str) -> str:
    return os.path.join(REPORTS_DIR, f"progress_{job_id}.json")


def write_progress(job_id: str, payload: Dict[str, Any]) -> None:
    try:
        payload = dict(payload)
        payload.setdefault("updated_at", datetime.utcnow().isoformat() + "Z")
        with open(_progress_path(job_id), 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def start_background_report(settings: Dict[str, Any], query_text: str):
    def _job(job_id: str, s: Dict[str, Any], q: str) -> str:
        try:
            from embeddinggemma.mcmp_rag import MCPMRetriever  # type: ignore
        except Exception:
            from mcmp_rag import MCPMRetriever  # type: ignore

        write_progress(job_id, {"status": "running", "percent": 0, "message": "Preparing corpus…"})
        if s.get('use_repo', True):
            docs = collect_codebase_chunks('src', s.get('windows', [50, 100, 200, 300, 400]), int(s.get('max_files', 1000)), s.get('exclude_dirs', []))
        else:
            rf = s.get('root_folder', os.getcwd())
            docs = collect_codebase_chunks(rf, s.get('windows', [50, 100, 200, 300, 400]), int(s.get('max_files', 1000)), s.get('exclude_dirs', []))
        write_progress(job_id, {"status": "running", "percent": 10, "message": f"Corpus ready: {len(docs)} chunks"})

        retr = MCPMRetriever(
            num_agents=int(s.get('num_agents', 200)),
            max_iterations=int(s.get('max_iterations', 60)),
            exploration_bonus=float(s.get('exploration_bonus', 0.1)),
            pheromone_decay=float(s.get('pheromone_decay', 0.95)),
            embed_batch_size=int(s.get('embed_bs', 64))
        )
        if not docs:
            write_progress(job_id, {"status": "error", "percent": 100, "message": "No documents to analyze"})
            raise RuntimeError("No documents to analyze")
        retr.add_documents(docs)
        total_chunks = len(docs)
        shard_size = int(s.get('max_chunks_per_shard', 2000))
        if shard_size <= 0 or shard_size >= total_chunks:
            shard_ranges = [(0, total_chunks)]
        else:
            shard_ranges = [(i, min(i + shard_size, total_chunks)) for i in range(0, total_chunks, shard_size)]
        aggregated_items: List[Dict[str, Any]] = []
        num_shards = max(1, len(shard_ranges))
        for shard_idx, (s_start, s_end) in enumerate(shard_ranges):
            shard = docs[s_start:s_end]
            try:
                retr.clear_documents()
            except Exception:
                pass
            retr.add_documents(shard)
            retr.initialize_simulation(q)
            for _ in range(int(s.get('max_iterations', 60))):
                retr.step(1)
            res_shard = retr.search(q, top_k=int(s.get('top_k', 5)))
            aggregated_items.extend(res_shard.get('results', []))
            pct = 10 + int(80 * float(shard_idx + 1) / float(num_shards))
            write_progress(job_id, {"status": "running", "percent": pct, "message": f"Processed shard {shard_idx+1}/{num_shards}"})
        if s.get('pure_topk', False):
            items = sorted(aggregated_items, key=lambda it: float(it.get('relevance_score', 0.0)), reverse=True)[:int(s.get('top_k', 5))]
        else:
            items = select_diverse_results(aggregated_items, retr, int(s.get('top_k', 5)), float(s.get('div_alpha', 0.7)), float(s.get('dedup_tau', 0.92)), int(s.get('per_folder_cap', 2)))
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": q,
            "settings": {k: v for k, v in s.items() if k not in {"docs"}},
            "results": items,
        }
        out_path = os.path.join(REPORTS_DIR, f"report_{job_id}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        write_progress(job_id, {"status": "done", "percent": 100, "message": "Report ready", "report_path": out_path})
        return out_path

    job_id = uuid.uuid4().hex[:12]
    write_progress(job_id, {"status": "running", "percent": 0, "message": "Queued", "started_at": datetime.utcnow().isoformat() + "Z"})
    fut = BG_EXECUTOR.submit(_job, job_id, dict(settings), str(query_text))
    return job_id, fut
