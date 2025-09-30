from __future__ import annotations

import argparse
import json
import os
from typing import Any, List

from embeddinggemma.llm.prompts import build_judge_prompt
from embeddinggemma.llm import generate_text  # type: ignore


def _parse_json_loose(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        s = raw.strip()
        if s.startswith('```'):
            s = "\n".join([ln for ln in s.splitlines() if not ln.strip().startswith('```')])
        start_obj = s.find('{'); start_arr = s.find('[')
        start = max(0, min([p for p in [start_obj, start_arr] if p != -1])) if (start_obj != -1 or start_arr != -1) else -1
        end_obj = s.rfind('}'); end_arr = s.rfind(']')
        end = max(end_obj, end_arr)
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        pass
    return {}


def main() -> None:
    # Load .env so API keys/base URLs are available
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Isolated judge runner for steering mode")
    ap.add_argument('--query', required=True, help='User query')
    ap.add_argument('--results', required=True, help='Path to JSON file with items [{id, score, content}]')
    ap.add_argument('--mode', default='steering', help='Judge mode (default: steering)')
    ap.add_argument('--provider', default=os.environ.get('LLM_PROVIDER', 'ollama'))
    # Optional overrides
    ap.add_argument('--system', default=os.environ.get('OLLAMA_SYSTEM'))
    ap.add_argument('--ollama-model', dest='ollama_model', default=os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b'))
    ap.add_argument('--ollama-host', dest='ollama_host', default=os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434'))
    ap.add_argument('--openai-model', dest='openai_model', default=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'))
    ap.add_argument('--openai-temperature', dest='openai_temperature', type=float, default=float(os.environ.get('OPENAI_TEMPERATURE', '0.0')))
    ap.add_argument('--google-model', dest='google_model', default=os.environ.get('GOOGLE_MODEL', 'gemini-1.5-pro'))
    ap.add_argument('--google-temperature', dest='google_temperature', type=float, default=float(os.environ.get('GOOGLE_TEMPERATURE', '0.0')))
    ap.add_argument('--grok-model', dest='grok_model', default=os.environ.get('GROK_MODEL', 'grok-2-latest'))
    ap.add_argument('--grok-temperature', dest='grok_temperature', type=float, default=float(os.environ.get('GROK_TEMPERATURE', '0.0')))
    ap.add_argument('--out', default='-')
    args = ap.parse_args()

    with open(args.results, 'r', encoding='utf-8') as f:
        items: List[dict] = json.load(f)

    prompt = build_judge_prompt(args.mode, args.query, items)

    text = generate_text(
        provider=args.provider,
        prompt=prompt,
        system=args.system,
        # ollama
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        ollama_options=None,
        # openai
        openai_model=args.openai_model,
        openai_api_key=os.environ.get('OPENAI_API_KEY', ''),
        openai_base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com'),
        openai_temperature=float(args.openai_temperature),
        # google
        google_model=args.google_model,
        google_api_key=os.environ.get('GOOGLE_API_KEY', ''),
        google_base_url=os.environ.get('GOOGLE_BASE_URL', 'https://generativelanguage.googleapis.com'),
        google_temperature=float(args.google_temperature),
        # grok
        grok_model=args.grok_model,
        grok_api_key=os.environ.get('GROK_API_KEY', ''),
        grok_base_url=os.environ.get('GROK_BASE_URL', 'https://api.x.ai'),
        grok_temperature=float(args.grok_temperature),
        save_prompt_path=None,
    )

    obj = _parse_json_loose(text or "")
    out = json.dumps(obj, ensure_ascii=False, indent=2)
    if args.out == '-':
        print(out)
    else:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(out)


if __name__ == '__main__':
    main()


