import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict


def list_py_files(root: str, max_files: int) -> List[str]:
    files: List[str] = []
    count = 0
    for r, _dirs, fns in os.walk(root):
        for fn in fns:
            if fn.endswith('.py'):
                files.append(os.path.join(r, fn))
                count += 1
                if max_files and count >= max_files:
                    return files
    return files


def chunk_file(path: str, windows: List[int]) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return []
    out: List[str] = []
    rel = os.path.relpath(path)
    n = len(lines)
    for w in windows:
        for i in range(0, n, w):
            start = i + 1
            end = min(i + w, n)
            body = ''.join(lines[i:end])
            if body.strip():
                header = f"# file: {rel} | lines: {start}-{end} | window: {w}\n"
                out.append(header + body)
    return out


def chunk_corpus(root: str, windows: List[int], workers: int, max_files: int) -> List[str]:
    files = list_py_files(root, max_files)
    docs: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(chunk_file, p, windows): p for p in files}
        for fut in as_completed(futs):
            try:
                docs.extend(fut.result())
            except Exception:
                pass
    return docs


def bench(root: str,
          windows: List[int],
          worker_grid: List[int],
          batch_grid: List[int],
          max_files: int) -> None:
    sys.path.append(os.path.abspath('.'))
    from mcmp_rag import MCPMRetriever

    results: List[Dict] = []

    for workers in worker_grid:
        t0 = time.time()
        docs = chunk_corpus(root, windows, workers, max_files)
        t1 = time.time()
        for bs in batch_grid:
            retr = MCPMRetriever(num_agents=50, max_iterations=5, embed_batch_size=bs)
            add_ms = None
            ok = True
            err = None
            try:
                t2 = time.time()
                retr.add_documents(docs)
                t3 = time.time()
                add_ms = int((t3 - t2) * 1000)
            except Exception as e:
                ok = False
                err = str(e)
            row = {
                "workers": workers,
                "batch": bs,
                "docs": len(docs),
                "chunk_ms": int((t1 - t0) * 1000),
                "add_ms": add_ms,
                "ok": ok,
                "err": err,
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    # Summary best configurations
    ok_rows = [r for r in results if r["ok"] and r["add_ms"] is not None]
    if ok_rows:
        best_add = min(ok_rows, key=lambda r: r["add_ms"])  # minimize add time
        best_chunk = min(results, key=lambda r: r["chunk_ms"])  # minimize chunk time
        print("==== BEST ====")
        print("fastest_add:", json.dumps(best_add, ensure_ascii=False))
        print("fastest_chunk:", json.dumps(best_chunk, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='src')
    ap.add_argument('--windows', default='50,100,200,300,400')
    ap.add_argument('--workers', default='4,8,16,32')
    ap.add_argument('--batches', default='64,128,256,384')
    ap.add_argument('--max-files', type=int, default=0)
    args = ap.parse_args()

    windows = [int(x) for x in args.windows.split(',') if x.strip().isdigit()]
    workers = [int(x) for x in args.workers.split(',') if x.strip().isdigit()]
    batches = [int(x) for x in args.batches.split(',') if x.strip().isdigit()]
    bench(args.root, windows, workers, batches, args.max_files)



