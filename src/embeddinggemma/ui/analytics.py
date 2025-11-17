"""Analytics module for cross-run query and retrieval analysis."""

import os
import json
from typing import Dict, Any, List, Set
from collections import defaultdict


def aggregate_query_corpus_mapping() -> Dict[str, Any]:
    """Aggregate all query→document mappings across all runs.

    Returns a mapping showing which queries retrieve which documents,
    useful for building training data for better retrievals.

    Returns:
        {
            'total_runs': int,
            'total_queries': int,
            'unique_queries': int,
            'query_to_docs': {query: [doc_ids]},
            'doc_to_queries': {doc_id: [queries]},
            'query_frequency': {query: count}
        }
    """
    runs_dir = os.path.join('.fungus_cache', 'runs')
    if not os.path.isdir(runs_dir):
        return {
            'total_runs': 0,
            'total_queries': 0,
            'unique_queries': 0,
            'query_to_docs': {},
            'doc_to_queries': {},
            'query_frequency': {}
        }

    query_to_docs: Dict[str, Set[int]] = defaultdict(set)
    doc_to_queries: Dict[int, Set[str]] = defaultdict(set)
    query_frequency: Dict[str, int] = defaultdict(int)
    total_queries = 0
    run_count = 0

    # Scan all runs
    for run_id in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, run_id)
        if not os.path.isdir(run_dir):
            continue

        retrieval_log_path = os.path.join(run_dir, 'retrievals.jsonl')
        if not os.path.isfile(retrieval_log_path):
            continue

        run_count += 1

        # Parse retrieval log
        try:
            with open(retrieval_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        query = entry.get('query', '').strip()
                        doc_ids = entry.get('doc_ids', [])

                        if not query:
                            continue

                        total_queries += 1
                        query_frequency[query] += 1

                        for doc_id in doc_ids:
                            if isinstance(doc_id, int) and doc_id >= 0:
                                query_to_docs[query].add(doc_id)
                                doc_to_queries[doc_id].add(query)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    # Convert sets to lists for JSON serialization
    query_to_docs_list = {q: sorted(list(docs)) for q, docs in query_to_docs.items()}
    doc_to_queries_list = {doc_id: sorted(list(queries)) for doc_id, queries in doc_to_queries.items()}

    return {
        'total_runs': run_count,
        'total_queries': total_queries,
        'unique_queries': len(query_to_docs),
        'query_to_docs': query_to_docs_list,
        'doc_to_queries': doc_to_queries_list,
        'query_frequency': dict(query_frequency)
    }


def analyze_coverage() -> Dict[str, Any]:
    """Analyze which documents are accessed vs never found across all runs.

    Returns:
        {
            'total_corpus_docs': int,
            'accessed_docs': int,
            'never_accessed_docs': int,
            'coverage_percent': float,
            'access_frequency': {doc_id: count},
            'most_accessed': [(doc_id, count)],
            'never_accessed_ids': [doc_ids]
        }
    """
    corpus_metadata_path = os.path.join('.fungus_cache', 'corpus', 'metadata.json')

    # Load corpus metadata
    total_docs = 0
    corpus_doc_ids = set()
    try:
        with open(corpus_metadata_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            total_docs = corpus_data.get('total_documents', 0)
            for doc in corpus_data.get('documents', []):
                corpus_doc_ids.add(doc['id'])
    except Exception:
        # Corpus metadata not available
        return {
            'error': 'Corpus metadata not found',
            'total_corpus_docs': 0,
            'accessed_docs': 0,
            'never_accessed_docs': 0,
            'coverage_percent': 0.0,
            'access_frequency': {},
            'most_accessed': [],
            'never_accessed_ids': []
        }

    # Scan all runs for document access
    doc_access_count: Dict[int, int] = defaultdict(int)
    runs_dir = os.path.join('.fungus_cache', 'runs')

    if os.path.isdir(runs_dir):
        for run_id in os.listdir(runs_dir):
            run_dir = os.path.join(runs_dir, run_id)
            if not os.path.isdir(run_dir):
                continue

            retrieval_log_path = os.path.join(run_dir, 'retrievals.jsonl')
            if not os.path.isfile(retrieval_log_path):
                continue

            try:
                with open(retrieval_log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            doc_ids = entry.get('doc_ids', [])
                            for doc_id in doc_ids:
                                if isinstance(doc_id, int) and doc_id >= 0:
                                    doc_access_count[doc_id] += 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

    accessed_docs = set(doc_access_count.keys())
    never_accessed = corpus_doc_ids - accessed_docs
    coverage = (len(accessed_docs) / total_docs * 100) if total_docs > 0 else 0.0

    # Sort by access frequency
    most_accessed = sorted(doc_access_count.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        'total_corpus_docs': total_docs,
        'accessed_docs': len(accessed_docs),
        'never_accessed_docs': len(never_accessed),
        'coverage_percent': round(coverage, 2),
        'access_frequency': dict(doc_access_count),
        'most_accessed': most_accessed,
        'never_accessed_ids': sorted(list(never_accessed))
    }


def analyze_query_patterns() -> Dict[str, Any]:
    """Analyze common query patterns and characteristics across runs.

    Returns:
        {
            'total_queries': int,
            'unique_queries': int,
            'avg_query_length': float,
            'most_common_queries': [(query, count)],
            'query_sources': {source: count},
            'avg_results_per_query': float
        }
    """
    runs_dir = os.path.join('.fungus_cache', 'runs')
    if not os.path.isdir(runs_dir):
        return {
            'total_queries': 0,
            'unique_queries': 0,
            'avg_query_length': 0.0,
            'most_common_queries': [],
            'query_sources': {},
            'avg_results_per_query': 0.0
        }

    query_frequency: Dict[str, int] = defaultdict(int)
    query_sources: Dict[str, int] = defaultdict(int)
    query_lengths: List[int] = []
    total_results = 0
    total_queries = 0

    # Scan all queries
    for run_id in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, run_id)
        if not os.path.isdir(run_dir):
            continue

        queries_log_path = os.path.join(run_dir, 'queries.jsonl')
        retrievals_log_path = os.path.join(run_dir, 'retrievals.jsonl')

        # Parse query log
        if os.path.isfile(queries_log_path):
            try:
                with open(queries_log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            query = entry.get('query', '').strip()
                            source = entry.get('source', 'unknown')

                            if not query:
                                continue

                            query_frequency[query] += 1
                            query_sources[source] += 1
                            query_lengths.append(len(query))
                            total_queries += 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

        # Parse retrieval log for result counts
        if os.path.isfile(retrievals_log_path):
            try:
                with open(retrievals_log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            count = entry.get('count', 0)
                            total_results += count
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

    avg_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0.0
    avg_results = total_results / total_queries if total_queries > 0 else 0.0
    most_common = sorted(query_frequency.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        'total_queries': total_queries,
        'unique_queries': len(query_frequency),
        'avg_query_length': round(avg_query_length, 2),
        'most_common_queries': most_common,
        'query_sources': dict(query_sources),
        'avg_results_per_query': round(avg_results, 2)
    }


def get_run_summary(run_id: str) -> Dict[str, Any]:
    """Get comprehensive summary for a specific run.

    Args:
        run_id: The run identifier

    Returns:
        Combined data from manifest, queries, and retrievals for the run
    """
    run_dir = os.path.join('.fungus_cache', 'runs', run_id)
    if not os.path.isdir(run_dir):
        return {'error': 'Run not found'}

    manifest_path = os.path.join(run_dir, 'manifest.json')
    queries_path = os.path.join(run_dir, 'queries.jsonl')
    retrievals_path = os.path.join(run_dir, 'retrievals.jsonl')
    summary_path = os.path.join(run_dir, 'summary.json')

    result = {
        'run_id': run_id,
        'manifest': {},
        'queries': [],
        'retrievals': [],
        'summary': {}
    }

    # Load manifest
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                result['manifest'] = json.load(f)
        except Exception:
            pass

    # Load queries
    if os.path.isfile(queries_path):
        try:
            with open(queries_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            result['queries'].append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass

    # Load retrievals
    if os.path.isfile(retrievals_path):
        try:
            with open(retrievals_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            result['retrievals'].append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass

    # Load summary if exists
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                result['summary'] = json.load(f)
        except Exception:
            pass

    return result
