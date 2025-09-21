from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import ast


def parse_code_with_ast(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        tree = ast.parse(source_code)
        chunks = []
        current = {
            'file': os.path.relpath(file_path, 'src'),
            'type': 'file',
            'content': '',
            'metadata': {
                'start_line': 1,
                'end_line': 0,
                'ast_nodes': []
            }
        }
        for node in ast.walk(tree):
            node_str = ast.get_source_segment(source_code, node)
            if not node_str:
                continue
            start_line = getattr(node, 'lineno', 1)
            end_line = getattr(node, 'end_lineno', start_line)
            current['content'] += f"\n{node_str}"
            current['metadata']['ast_nodes'].append({
                'type': type(node).__name__,
                'start_line': start_line,
                'end_line': end_line,
                'name': getattr(node, 'name', None) if hasattr(node, 'name') else None
            })
            current['metadata']['end_line'] = end_line
        if current['content'].strip():
            current['metadata']['size'] = len(current['content'])
            chunks.append(current)
        return chunks
    except SyntaxError:
        return fallback_chunking(file_path)
    except Exception:
        return []


def fallback_chunking(file_path: str, lines_per_chunk: int = 20) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        out: List[Dict[str, Any]] = []
        for i in range(0, len(lines), lines_per_chunk):
            chunk = ''.join(lines[i:i + lines_per_chunk])
            if not chunk.strip():
                continue
            out.append({
                'file': os.path.relpath(file_path, 'src'),
                'type': 'line_chunk',
                'content': chunk.strip(),
                'metadata': {
                    'start_line': i + 1,
                    'end_line': min(i + lines_per_chunk, len(lines)),
                    'size': len(chunk)
                }
            })
        return out
    except Exception:
        return []


