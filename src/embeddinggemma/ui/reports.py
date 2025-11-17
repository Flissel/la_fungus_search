import os
import json
import uuid
import hashlib
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
            docs = collect_codebase_chunks('src', s.get('windows', []), int(s.get('max_files', 1000)), s.get('exclude_dirs', []))
        else:
            rf = s.get('root_folder', os.getcwd())
            docs = collect_codebase_chunks(rf, s.get('windows', []), int(s.get('max_files', 1000)), s.get('exclude_dirs', []))
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


# ---- Reports merging helpers ----
def _discover_report_paths(max_reports: int = 200) -> List[str]:
    """Locate report.json files produced by per-step runs and background jobs.

    Searches in .fungus_cache/runs/**/step_*/report.json and .fungus_cache/reports/report_*.json
    Returns most recently modified first, up to max_reports.
    """
    roots: List[str] = []
    try:
        base = os.path.join('.fungus_cache', 'runs')
        for root, _dirs, files in os.walk(base):
            if 'report.json' in files:
                roots.append(os.path.join(root, 'report.json'))
    except Exception:
        pass
    try:
        # Background report outputs created by start_background_report
        rep_dir = os.path.join('.fungus_cache', 'reports')
        if os.path.isdir(rep_dir):
            for fn in os.listdir(rep_dir):
                if fn.startswith('report_') and fn.endswith('.json'):
                    roots.append(os.path.join(rep_dir, fn))
    except Exception:
        pass
    try:
        roots = sorted(roots, key=lambda p: os.path.getmtime(p), reverse=True)
    except Exception:
        pass
    return roots[: max(1, int(max_reports))]


def _norm_item(it: Dict[str, Any], fallback_query: str | None = None) -> Dict[str, Any]:
    """Normalize a single report item to a common schema."""
    out: Dict[str, Any] = {}
    out['code_chunk'] = it.get('code_chunk') or ''
    out['content'] = it.get('content') or ''
    out['file_path'] = it.get('file_path') or (it.get('metadata', {}) or {}).get('file_path') or ''
    lr = it.get('line_range')
    if isinstance(lr, list) and len(lr) == 2 and all(isinstance(x, int) for x in lr):
        out['line_range'] = [int(lr[0]), int(lr[1])]
    else:
        out['line_range'] = [1, max(1, len((out['content'] or '').splitlines()))]
    out['code_purpose'] = it.get('code_purpose') or ''
    out['code_dependencies'] = it.get('code_dependencies') or []
    out['file_type'] = it.get('file_type') or ''
    try:
        out['embedding_score'] = float(it.get('embedding_score', 0.0))
    except Exception:
        out['embedding_score'] = 0.0
    out['relevance_to_query'] = it.get('relevance_to_query') or ''
    out['query_initial'] = it.get('query_initial') or (fallback_query or '')
    fqs = it.get('follow_up_queries') or []
    out['follow_up_queries'] = [q for q in fqs if isinstance(q, str) and q.strip()]
    return out


def _item_key(it: Dict[str, Any]) -> str:
    """Stable key for deduplication: file_path + line_range or content hash."""
    fp = str(it.get('file_path') or '')
    lr = it.get('line_range') or []
    if fp and isinstance(lr, list) and len(lr) == 2:
        return f"{fp}:{int(lr[0])}-{int(lr[1])}"
    # Fallback to content/code hash
    txt = (it.get('code_chunk') or '') + '\n' + (it.get('content') or '')
    try:
        return 'sha1:' + hashlib.sha1(txt.encode('utf-8', errors='ignore')).hexdigest()
    except Exception:
        return 'sha1:'


def merge_reports_to_summary(paths: List[str] | None = None, out_path: str | None = None) -> Dict[str, Any]:
    """Merge multiple report.json files into a single summary document.

    - paths: optional explicit list of report file paths. If None, auto-discovers recent reports.
    - out_path: where to write the merged summary. Defaults to .fungus_cache/reports/summary.json

    Returns: { summary_path, total_reports, total_items, kept_items, deduped, sections }
    """
    if not paths:
        paths = _discover_report_paths()
    paths = [p for p in (paths or []) if isinstance(p, str) and os.path.isfile(p)]
    items_norm: List[Dict[str, Any]] = []
    errors: List[str] = []
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            step = int(data.get('step', -1)) if isinstance(data, dict) else -1
            mode = str(data.get('mode', '') or '') if isinstance(data, dict) else ''
            payload = data.get('data') if isinstance(data, dict) else None
            items = (payload or {}).get('items', []) if isinstance(payload, dict) else []
            # Try to propagate the main query if present in the file context
            fallback_query = None
            try:
                # Heuristic: initial query often present in first item
                if items:
                    q0 = items[0].get('query_initial')
                    if isinstance(q0, str) and q0.strip():
                        fallback_query = q0
            except Exception:
                fallback_query = None
            for it in (items or []):
                itn = _norm_item(it, fallback_query)
                itn['__source__'] = {
                    'path': p,
                    'step': step,
                    'mode': mode,
                }
                items_norm.append(itn)
        except Exception as e:
            errors.append(f"{p}: {e}")

    total_items = len(items_norm)
    # Deduplicate, keep highest embedding_score
    best_by_key: Dict[str, Dict[str, Any]] = {}
    for it in items_norm:
        k = _item_key(it)
        prev = best_by_key.get(k)
        if prev is None or float(it.get('embedding_score', 0.0)) > float(prev.get('embedding_score', 0.0)):
            best_by_key[k] = it
    merged_items = list(best_by_key.values())
    deduped = total_items - len(merged_items)

    # Aggregate follow_up_queries and basic sections
    all_followups: List[str] = []
    for it in merged_items:
        all_followups.extend(it.get('follow_up_queries', []) or [])
    # Sections by simple heuristics on file_path
    sections: Dict[str, List[Dict[str, Any]]] = {
        'routes': [],
        'websocket': [],
        'background_jobs': [],
        'indexing': [],
        'other': [],
    }
    for it in merged_items:
        fp = (it.get('file_path') or '').lower()
        if '/realtime/server.py' in fp or '\\realtime\\server.py' in fp:
            if '/ws' in (it.get('content') or '') or 'websocket' in (it.get('content') or '').lower():
                sections['websocket'].append(it)
            elif 'jobs' in (it.get('content') or '').lower():
                sections['background_jobs'].append(it)
            elif 'corpus/index' in (it.get('content') or '').lower() or 'qdrant' in (it.get('content') or '').lower():
                sections['indexing'].append(it)
            else:
                sections['routes'].append(it)
        else:
            sections['other'].append(it)

    # Build summary object
    summary = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'initial_query': next((it.get('query_initial') for it in merged_items if it.get('query_initial')), ''),
        'total_reports': len(paths),
        'total_items': total_items,
        'kept_items': len(merged_items),
        'deduped': deduped,
        'follow_up_queries': sorted(list({q for q in all_followups if isinstance(q, str) and q.strip()})),
        'sections': {k: [
            {kk: vv for kk, vv in it.items() if not kk.startswith('__')}
        for it in v] for k, v in sections.items()},
        'items': [
            {kk: vv for kk, vv in it.items() if not kk.startswith('__')}
        for it in merged_items],
        'sources': [p for p in paths],
        'errors': errors,
    }

    # Write output
    out = out_path or os.path.join('.fungus_cache', 'reports', 'summary.json')
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return {
        'summary_path': out,
        'total_reports': len(paths),
        'total_items': total_items,
        'kept_items': len(merged_items),
        'deduped': deduped,
        'errors': errors,
    }


def create_run_summary(run_id: str, query: str, result_items: List[Dict[str, Any]]) -> str:
    """Create a per-run summary that references corpus metadata instead of embedding content.

    Args:
        run_id: Unique run identifier
        query: The query for this run
        result_items: List of result items from retrieval (with doc_id references)

    Returns:
        Path to the created summary.json file

    This function creates a lightweight summary that:
    - Saves to .fungus_cache/runs/{run_id}/summary.json
    - References documents by ID instead of embedding full content
    - Links to corpus metadata for document details
    - Includes run-specific metrics and results
    """
    run_dir = os.path.join('.fungus_cache', 'runs', str(run_id))
    summary_path = os.path.join(run_dir, 'summary.json')
    corpus_metadata_path = os.path.join('.fungus_cache', 'corpus', 'metadata.json')

    # Load corpus metadata reference
    corpus_ref = {'path': corpus_metadata_path, 'exists': os.path.isfile(corpus_metadata_path)}
    if corpus_ref['exists']:
        try:
            with open(corpus_metadata_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
                corpus_ref['total_documents'] = corpus_data.get('total_documents', 0)
                corpus_ref['fingerprint'] = corpus_data.get('fingerprint')
        except Exception:
            pass

    # Transform result items to reference-only format (no full content embedding)
    result_refs = []
    for item in result_items:
        # Extract doc_id if available
        doc_id = item.get('id') or item.get('doc_id')
        if doc_id is None:
            # Skip items without doc IDs
            continue

        ref_item = {
            'doc_id': int(doc_id),
            'score': float(item.get('score', 0.0)),
            'embedding_score': float(item.get('embedding_score', 0.0)),
            'relevance_score': float(item.get('relevance_score', 0.0)),
        }

        # Include lightweight metadata but NOT full content
        if 'file_path' in item:
            ref_item['file_path'] = item['file_path']
        if 'line_range' in item:
            ref_item['line_range'] = item['line_range']
        if 'code_purpose' in item:
            ref_item['code_purpose'] = item['code_purpose']

        result_refs.append(ref_item)

    # Load manifest for run metrics
    manifest_path = os.path.join(run_dir, 'manifest.json')
    run_metrics = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                run_metrics = json.load(f)
        except Exception:
            pass

    # Build per-run summary
    summary = {
        'run_id': run_id,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'query': query,
        'corpus_metadata_ref': corpus_ref,
        'results_count': len(result_refs),
        'results': result_refs,
        'run_metrics': run_metrics,
    }

    # Write summary
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write run summary: {e}")

    return summary_path