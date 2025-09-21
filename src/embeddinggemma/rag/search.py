from __future__ import annotations
from typing import List, Dict, Any
import logging


_logger = logging.getLogger("Rag.Search")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)

def hybrid_search(index, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
    """Semantic + keyword hybrid search similar to RagV1.hybrid_search."""
    if index is None:
        return []
    _logger.info("hybrid_search: top_k=%d alpha=%.2f", int(top_k), float(alpha))
    semantic_retriever = index.as_retriever(similarity_top_k=top_k * 2)
    semantic_nodes = semantic_retriever.retrieve(query)
    semantic_results = []
    for nws in semantic_nodes:
        node_obj = getattr(nws, 'node', nws)
        score = float(getattr(nws, 'score', 0.0) or 0.0)
        content = getattr(node_obj, 'text', None) or (node_obj.get_content() if hasattr(node_obj, 'get_content') else None)
        metadata = getattr(node_obj, 'metadata', {}) or {}
        source = metadata.get('file_path') or metadata.get('source') or getattr(node_obj, 'node_id', 'unknown')
        semantic_results.append({'content': content or '', 'metadata': metadata, 'semantic_score': score, 'source': source})

    keyword_results = _keyword_search(index, query, top_k * 2)
    hybrid_results: List[Dict[str, Any]] = []
    seen = set()
    for result in semantic_results + keyword_results:
        src = result.get('source') or result.get('file_path', 'unknown')
        if src in seen:
            continue
        seen.add(src)
        semantic_score = result.get('semantic_score', 0.0)
        keyword_score = result.get('keyword_score', 0.0)
        hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score
        merged = dict(result)
        merged['hybrid_score'] = hybrid_score
        hybrid_results.append(merged)
        if len(hybrid_results) >= top_k:
            break
    hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    _logger.info("hybrid_search: results=%d", len(hybrid_results))
    return hybrid_results[:top_k]


def _keyword_search(index, query: str, top_k: int) -> List[Dict[str, Any]]:
    _logger.debug("keyword_search: top_k=%d", int(top_k))
    keyword_retriever = index.as_retriever(similarity_top_k=top_k, node_postprocessors=[])
    nodes = keyword_retriever.retrieve(query)
    res = []
    for nws in nodes:
        node_obj = getattr(nws, 'node', nws)
        content = getattr(node_obj, 'text', None) or (node_obj.get_content() if hasattr(node_obj, 'get_content') else None)
        text_lower = (content or '').lower()
        q_tokens = [t for t in query.lower().split() if t]
        token_hits = sum(1 for t in q_tokens if t in text_lower)
        keyword_score = min(token_hits / max(1, len(q_tokens)), 1.0)
        metadata = getattr(node_obj, 'metadata', {}) or {}
        source = metadata.get('file_path') or metadata.get('source') or getattr(node_obj, 'node_id', 'unknown')
        res.append({'content': content or '', 'metadata': metadata, 'keyword_score': keyword_score, 'source': source})
    return res


