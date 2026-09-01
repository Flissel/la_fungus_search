"""Embed exported manifest sources with a locally cached model.

Runs in whichever interpreter has the model stack installed -- not necessarily the
Fungus venv, which deliberately has no torch. It reads the JSON written by
``build_local_snapshot.py export`` and writes a ``.npz`` with a single ``vectors``
array in the same order.

Offline by construction: ``HF_HUB_OFFLINE`` is set before the model loads, so a
missing or incomplete cache fails loudly instead of silently downloading a
different revision than the one that produced earlier vectors.

Usage::

    <python-with-torch> -m benchmarks.gate2.embed_local \
        --sources benchmarks/results/gate2/sources-local-v1.json \
        --out benchmarks/results/gate2/vectors-local-v1.npz \
        --model Qwen/Qwen3-Embedding-0.6B \
        --hf-home E:/huggingface_cache
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed manifest sources locally")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="embed at most this many documents this run (0 = all); the rest resume next run",
    )
    arguments = parser.parse_args()

    # Both must be set before the first transformers/sentence-transformers import,
    # which reads them at module scope.
    if arguments.hf_home:
        os.environ["HF_HOME"] = arguments.hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(arguments.hf_home) / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy as np
    from sentence_transformers import SentenceTransformer

    sources = json.loads(arguments.sources.read_text(encoding="utf-8"))
    print(f"documents     : {len(sources)}")

    # Resume support. A 0.6B model on CPU over a few hundred code documents
    # outlives a single foreground timeout, and the same run detached produced no
    # process at all -- so progress is checkpointed and re-entrant instead. The
    # file is the state: N vectors present means the first N sources are done.
    done = 0
    existing: np.ndarray | None = None
    if arguments.out.exists():
        with np.load(arguments.out) as payload:
            existing = payload["vectors"]
        done = int(existing.shape[0])
        print(f"resuming from : {done} already embedded")
    if done >= len(sources):
        print("nothing to do")
        return

    remaining = sources[done:]
    take = len(remaining) if arguments.chunk <= 0 else min(arguments.chunk, len(remaining))
    batch = remaining[:take]

    model = SentenceTransformer(arguments.model, device="cpu")
    # Code documents are long; the default 32-token truncation on some configs
    # would embed little more than a signature.
    model.max_seq_length = arguments.max_seq_length
    print(f"model         : {arguments.model}")
    print(f"max_seq_length: {model.max_seq_length}")
    print(f"this run      : documents {done}..{done + take - 1}")

    fresh = model.encode(
        batch,
        batch_size=arguments.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    vectors = fresh if existing is None else np.vstack([existing, fresh])
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.out, vectors=vectors)
    complete = vectors.shape[0] >= len(sources)
    print(f"vectors       : {vectors.shape} -> {arguments.out}")
    print(f"complete      : {complete} ({vectors.shape[0]}/{len(sources)})")


if __name__ == "__main__":
    main()
