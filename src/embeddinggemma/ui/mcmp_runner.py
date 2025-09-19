from __future__ import annotations
import os
from typing import List, Dict, Any
import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def _cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _normalize(a) @ _normalize(b).T


def _extract_source(meta: Dict[str, Any], content: str) -> str:
    src = (meta or {}).get('file_path') if isinstance(meta, dict) else None
    if not src:
        import re
        m = re.search(r"^# file: (.+?) \| lines:", content or "", re.MULTILINE)
        if m:
            src = m.group(1)
    return src or "unknown"


def select_diverse_results(results_items: List[Dict[str, Any]],
                           retr: Any,
                           top_k: int,
                           alpha: float,
                           dedup_tau: float,
                           per_folder_cap: int) -> List[Dict[str, Any]]:
    if not results_items:
        return []
    texts = [(it.get('content') or '')[:2048] for it in results_items]
    try:
        embs = retr.embedding_model.encode(texts)
        embs = np.array(embs, dtype=np.float32)
    except Exception:
        rng = np.random.default_rng(42)
        embs = rng.normal(0, 1, size=(len(texts), 64)).astype(np.float32)
    sims = _cos_sim(embs, embs)

    selected: List[int] = []
    folder_counts: Dict[str, int] = {}
    base_scores = np.array([float(it.get('relevance_score', 0.0)) for it in results_items], dtype=np.float32)
    order = np.argsort(-base_scores)
    for idx in order:
        if len(selected) >= int(top_k):
            break
        it = results_items[int(idx)]
        src = _extract_source(it.get('metadata', {}), it.get('content', ''))
        folder = os.path.dirname(src)
        if per_folder_cap > 0 and folder_counts.get(folder, 0) >= per_folder_cap:
            continue
        if selected:
            max_sim = float(np.max(sims[idx, selected]))
            if max_sim >= dedup_tau:
                continue
        diversity_penalty = float(np.max(sims[idx, selected])) if selected else 0.0
        _ = float(alpha * base_scores[idx] - (1.0 - alpha) * diversity_penalty)  # advisory
        selected.append(int(idx))
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    return [results_items[i] for i in selected]


def quick_search_with_mcmp(settings: Dict[str, Any], query_text: str, top_k: int) -> Dict[str, Any]:
    try:
        from embeddinggemma.mcmp_rag import MCPMRetriever  # type: ignore
    except Exception:
        try:
            from mcmp_rag import MCPMRetriever  # type: ignore
        except Exception as e:
            return {"error": f"MCPMRetriever unavailable: {e}"}

    from .corpus import collect_codebase_chunks, list_code_files

    corp = {
        "docs": [],
        "discovered_files": [],
        "loaded_files": [],
    }
    if settings.get('use_repo', True):
        corp["discovered_files"] = list_code_files('src', int(settings.get('max_files', 200)), settings.get('exclude_dirs', []))
        corp["docs"] = collect_codebase_chunks('src', settings.get('windows', [100, 200, 300]), int(settings.get('max_files', 200)), settings.get('exclude_dirs', []))
    else:
        rf = (settings.get('root_folder') or '').strip()
        if rf and os.path.isdir(rf):
            corp["discovered_files"] = list_code_files(rf, int(settings.get('max_files', 200)), settings.get('exclude_dirs', []))
            corp["docs"] = collect_codebase_chunks(rf, settings.get('windows', [100, 200, 300]), int(settings.get('max_files', 200)), settings.get('exclude_dirs', []))
    docs = corp.get('docs', [])
    if not docs:
        return {"error": "No documents found for quick search"}

    retr = MCPMRetriever(
        num_agents=int(settings.get('num_agents', 100)),
        max_iterations=int(settings.get('max_iterations', 20)),
        exploration_bonus=float(settings.get('exploration_bonus', 0.1)),
        pheromone_decay=float(settings.get('pheromone_decay', 0.95)),
        embed_batch_size=int(settings.get('embed_bs', 64))
    )
    try:
        retr.add_documents(docs)
        res = retr.search(str(query_text), top_k=int(top_k))
    except Exception as e:
        return {"error": f"search failed: {e}"}
    return {"results": res.get('results', []), "docs_count": len(docs)}
