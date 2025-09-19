import os
import time
import json
import argparse
from typing import List, Dict


def run_enterprise(query: str) -> Dict:
    import subprocess
    cmd = [
        os.path.join('.venv312', 'Scripts', 'python.exe') if os.name == 'nt' else 'python',
        'src/embeddinggemma/enterprise_rag.py', 'query', query, '--top-k', '5', '--alpha', '0.7'
    ]
    t0 = time.time()
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=600)
        dur = time.time() - t0
        return {"ok": True, "ms": int(dur * 1000), "stdout": out.decode('utf-8', errors='ignore')}
    except subprocess.CalledProcessError as e:
        dur = time.time() - t0
        return {"ok": False, "ms": int(dur * 1000), "stdout": e.output.decode('utf-8', errors='ignore')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--queries', type=str, nargs='*', default=[
        'List all functions and methods in src/embeddinggemma/codespace_analyzer.py',
        'Where is MCPMRetriever initialized?',
        'Find import related code for qdrant in src/'
    ])
    args = parser.parse_args()

    results: List[Dict] = []
    for q in args.queries:
        for s in range(args.seeds):
            os.environ['PYTHONHASHSEED'] = str(s)
            r = run_enterprise(q)
            results.append({"query": q, "seed": s, **r})
            print(json.dumps(results[-1], ensure_ascii=False))

    print("==== SUMMARY ====")
    agg: Dict[str, List[int]] = {}
    for r in results:
        agg.setdefault(r['query'], []).append(r['ms'])
    for q, times in agg.items():
        print(f"{q}: n={len(times)} mean={sum(times)/len(times):.1f} ms min={min(times)} ms max={max(times)} ms")


if __name__ == '__main__':
    main()



