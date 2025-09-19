import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def extract_json(text: str):
    text = text.strip()
    # try direct json
    try:
        return json.loads(text), text
    except Exception:
        pass
    # try last json object via regex
    m = re.search(r"\{[\s\S]*\}$", text)
    if m:
        frag = m.group(0)
        try:
            return json.loads(frag), frag
        except Exception:
            return None, text
    return None, text


def run_agent(python_exec: str, agent_path: str, docs_file: str, query: str, model: str,
              device: str = "auto", windows: str = "400,300,200,100,50", strict: bool = True,
              include_text: bool = True) -> str:
    cmd = [
        python_exec,
        agent_path,
        "--docs-file", docs_file,
        "--md-codeblocks",
    ]
    if include_text:
        cmd.append("--md-include-text")
    cmd += [
        "--query", query,
        "--ollama-model", model,
        "--device", device,
        "--windows", windows,
        "--max-retry", "0",
    ]
    if strict:
        cmd.append("--strict-agent")

    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return (proc.stdout or "") + (proc.stderr or "")


def main():
    parser = argparse.ArgumentParser(description="Run a single query from JSONL against the agent and check output")
    parser.add_argument("--file", default="queries/elevenlabs_queries.jsonl")
    parser.add_argument("--index", type=int, default=0, help="0-based index of the query to run")
    parser.add_argument("--python", default=str(Path(".venv/Scripts/python.exe").resolve()))
    parser.add_argument("--agent", default="src/embeddinggemma/agent_fungus_rag.py")
    parser.add_argument("--docs", default="Elevenlabs_API_Codesheet_final.md")
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args()

    qpath = Path(args.file)
    if not qpath.exists():
        print(f"ERROR: file not found: {qpath}")
        sys.exit(2)

    lines = [ln for ln in qpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.index < 0 or args.index >= len(lines):
        print(f"ERROR: index out of range (0..{len(lines)-1})")
        sys.exit(2)

    obj = json.loads(lines[args.index])
    section = obj.get("section", "?")
    query = obj["query"]
    needles = obj.get("expected_contains", [])

    print(f"== Running [{args.index}] {section}: {query}")
    out = run_agent(args.python, args.agent, args.docs, query, args.model)
    data, raw = extract_json(out)
    hay = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else raw

    ok = all((needle in hay) for needle in needles)
    print("PASS" if ok else "FAIL")
    print(json.dumps({
        "section": section,
        "query": query,
        "ok": ok,
        "needles": needles,
    }, ensure_ascii=False, indent=2))

    # echo the parsed json for manual inspection
    if isinstance(data, dict):
        print("\n--- Parsed JSON ---")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("\n--- Raw Output (no JSON) ---")
        print(raw)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()




