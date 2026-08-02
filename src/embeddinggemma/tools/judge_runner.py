from __future__ import annotations

import argparse
import json
from typing import List

from vibemind_shared import get_client_sync, get_model
from embeddinggemma.llm.prompts import build_judge_prompt


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
    ap = argparse.ArgumentParser(description="Isolated judge runner for steering mode")
    ap.add_argument('--query', required=True, help='User query')
    ap.add_argument('--results', required=True, help='Path to JSON file with items [{id, score, content}]')
    ap.add_argument('--mode', default='steering', help='Judge mode (default: steering)')
    ap.add_argument('--out', default='-')
    args = ap.parse_args()

    with open(args.results, 'r', encoding='utf-8') as f:
        items: List[dict] = json.load(f)

    prompt = build_judge_prompt(args.mode, args.query, items)

    completion = get_client_sync("fungus_judge").chat.completions.create(
        model=get_model("fungus_judge"),
        messages=[{"role": "user", "content": prompt}],
    )
    text = completion.choices[0].message.content

    obj = _parse_json_loose(text or "")
    out = json.dumps(obj, ensure_ascii=False, indent=2)
    if args.out == '-':
        print(out)
    else:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(out)


if __name__ == '__main__':
    main()


