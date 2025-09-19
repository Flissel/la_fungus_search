import os
import re
import hashlib
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = os.path.join(".fungus_cache", "chunks")
os.makedirs(CACHE_DIR, exist_ok=True)


def _file_sha1(path: str) -> str:
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return ""


def _cache_key(path: str, windows: List[int]) -> str:
    rel = os.path.relpath(path).replace(os.sep, "_")
    win_key = "-".join(str(int(w)) for w in sorted(set(int(w) for w in windows)))
    sha = _file_sha1(path)
    return os.path.join(CACHE_DIR, f"{rel}.{sha}.{win_key}.jsonl")


def _load_cached_chunks(path: str, windows: List[int]) -> List[str]:
    c = _cache_key(path, windows)
    try:
        if os.path.exists(c):
            with open(c, 'r', encoding='utf-8') as f:
                return [line.rstrip("\n") for line in f]
    except Exception:
        return []
    return []


def _save_cached_chunks(path: str, windows: List[int], chunks: List[str]) -> None:
    c = _cache_key(path, windows)
    try:
        os.makedirs(os.path.dirname(c), exist_ok=True)
        with open(c, 'w', encoding='utf-8') as f:
            for ch in chunks:
                f.write(ch.replace('\n', '\n') + '\n')
    except Exception:
        pass


def chunk_python_file(path: str, windows: List[int]) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return []
    chunks: List[str] = []
    total = len(lines)
    rel = os.path.relpath(path)
    for w in windows:
        for i in range(0, total, int(w)):
            start = i + 1
            end = min(i + int(w), total)
            body = ''.join(lines[i:end])
            if body.strip():
                header = f"# file: {rel} | lines: {start}-{end} | window: {w}\n"
                chunks.append(header + body)
    return chunks


def list_code_files(root_dir: str, max_files: int, exclude_dirs: List[str] = None) -> List[str]:
    files: List[str] = []
    count = 0
    for root, dirs, filenames in os.walk(root_dir):
        if exclude_dirs:
            dirs[:] = [d for d in dirs if not any(ex in os.path.join(root, d) for ex in exclude_dirs)]
        for fn in filenames:
            if fn.endswith('.py'):
                files.append(os.path.join(root, fn))
                count += 1
                if max_files and count >= max_files:
                    return files
    return files


def collect_codebase_chunks(root_dir: str, windows: List[int], max_files: int, exclude_dirs: List[str] = None) -> List[str]:
    files_to_process = list_code_files(root_dir, max_files, exclude_dirs)
    docs: List[str] = []
    with ThreadPoolExecutor(max_workers=max(4, min(32, (os.cpu_count() or 8) * 2))) as ex:
        futures: Dict[Any, str] = {}
        for p in files_to_process:
            cached = _load_cached_chunks(p, windows)
            if cached:
                docs.extend(cached)
            else:
                futures[ex.submit(chunk_python_file, p, windows)] = p
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                chunks = fut.result()
            except Exception:
                chunks = []
            if chunks:
                docs.extend(chunks)
                _save_cached_chunks(p, windows, chunks)
    return docs
