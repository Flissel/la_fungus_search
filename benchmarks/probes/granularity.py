"""B: does function granularity beat the production 200-line windows?

Production fungus retrieves over ~200-line windows; the Gate 2 spec rejects that
unit for code because a window boundary cuts across call sites, and the
production `chunks.json` was separately found truncated mid-write. This measures
whether the objection is worth acting on: same files, same embedding model, same
call-graph oracle — only the document unit differs.

Fairness rules:

- The query is always the *function* (its embedding from the function snapshot),
  because that is the unit the oracle defines relevance for.
- On the window side, every window overlapping the query function's own span is
  excluded — the exact analog of dropping the query document from the function
  corpus, and for the same reason: it would match itself at similarity ~1.
- A relevant function counts as retrieved at k on the window side if any window
  overlapping its span is in the top k. That is the user-facing question — was
  the relevant code region shown — and it, if anything, *favours* windows, since
  a long function offers several windows that each count.

Two stages, same pattern as the snapshot pipeline:

    build     write window texts + spans (runs in the Fungus venv)
    evaluate  after embed_local has produced window vectors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from benchmarks.gate2.manifest import load_manifest, manifest_digest, query_candidates, relevant_for
from benchmarks.gate2.snapshot import load_snapshot

WINDOW_LINES = 200
TOP_KS = (8, 16)


def build_windows(corpus_root: Path, out_texts: Path, out_meta: Path) -> None:
    texts: list[str] = []
    meta: list[dict[str, object]] = []
    for path in sorted(corpus_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(corpus_root).as_posix()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for start in range(0, len(lines), WINDOW_LINES):
            body = "\n".join(lines[start : start + WINDOW_LINES])
            if not body.strip():
                continue
            first, last = start + 1, min(start + WINDOW_LINES, len(lines))
            # Production chunks carry this header inside the embedded text; the
            # comparison keeps it so the window side is measured as shipped.
            header = f"# file: {relative} | lines: {first}-{last} | window: {WINDOW_LINES}\n"
            texts.append(header + body)
            meta.append({"file": relative, "start": first, "end": last})
    out_texts.parent.mkdir(parents=True, exist_ok=True)
    out_texts.write_text(json.dumps(texts), encoding="utf-8")
    out_meta.write_text(json.dumps(meta), encoding="utf-8")
    print(f"windows       : {len(texts)}")
    print(f"texts         : {out_texts}")
    print(f"meta          : {out_meta}")


def evaluate(
    manifest_path: Path,
    snapshot_path: Path,
    meta_path: Path,
    window_vectors_path: Path,
    query_count: int,
    seed: int,
) -> None:
    manifest = load_manifest(manifest_path)
    snapshot = load_snapshot(snapshot_path, manifest_digest(manifest))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(window_vectors_path) as payload:
        window_vectors = payload["vectors"].astype(np.float32)
    norms = np.linalg.norm(window_vectors, axis=1, keepdims=True)
    window_vectors = window_vectors / np.clip(norms, 1e-9, None)
    if len(meta) != window_vectors.shape[0]:
        raise ValueError(f"{len(meta)} windows but {window_vectors.shape[0]} vectors")

    span_of = {
        document.document_id: (document.file, document.start_line, document.end_line)
        for document in manifest.documents
    }
    position = {document_id: index for index, document_id in enumerate(snapshot.document_ids)}
    windows_by_file: dict[str, list[int]] = {}
    for index, row in enumerate(meta):
        windows_by_file.setdefault(str(row["file"]), []).append(index)

    def overlapping_windows(document_id: str) -> set[int]:
        file, first, last = span_of[document_id]
        found = set()
        for index in windows_by_file.get(Path(file).as_posix(), []):
            row = meta[index]
            if int(row["start"]) <= last and int(row["end"]) >= first:
                found.add(index)
        return found

    rng = np.random.default_rng(seed)
    candidates = sorted(query_candidates(manifest))
    chosen = [candidates[int(i)] for i in rng.choice(len(candidates), size=query_count, replace=False)]

    function_matrix = snapshot.vectors
    rows = []
    skipped = 0
    for query_document in chosen:
        relevant = sorted(relevant_for(manifest, query_document))
        if not relevant:
            skipped += 1
            continue
        query_vector = snapshot.vectors[position[query_document]]

        # Function granularity: full corpus minus the query document itself.
        function_sims = function_matrix @ query_vector
        function_sims[position[query_document]] = -np.inf
        function_order = np.argsort(-function_sims)
        function_ranked = [snapshot.document_ids[int(i)] for i in function_order[: max(TOP_KS)]]

        # Window granularity: minus every window overlapping the query's span.
        own = overlapping_windows(query_document)
        window_sims = window_vectors @ query_vector
        for index in own:
            window_sims[index] = -np.inf
        window_order = np.argsort(-window_sims)

        relevant_windows = {document_id: overlapping_windows(document_id) for document_id in relevant}
        row = {}
        for k in TOP_KS:
            top_functions = set(function_ranked[:k])
            row[f"fn{k}"] = sum(1 for d in relevant if d in top_functions) / len(relevant)
            top_windows = set(int(i) for i in window_order[:k])
            row[f"win{k}"] = sum(
                1 for d in relevant if relevant_windows[d] & top_windows
            ) / len(relevant)
        # Equal shown-code budget. Top-8 windows display ~1600 lines; top-8
        # functions ~160. Comparing them slot-for-slot hands the window side ten
        # times the material, so the honest comparison fixes the *lines* budget:
        # how much relevant code reaches the reader per 1600 or 200 lines shown.
        function_lengths = {
            document.document_id: max(1, document.end_line - document.start_line + 1)
            for document in manifest.documents
        }
        deep_ranked = [snapshot.document_ids[int(i)] for i in function_order[:400]]
        for budget_lines, window_k in ((WINDOW_LINES * 8, 8), (WINDOW_LINES, 1)):
            shown = 0
            taken: set[str] = set()
            for document_id in deep_ranked:
                if shown >= budget_lines:
                    break
                taken.add(document_id)
                shown += function_lengths[document_id]
            row[f"fn_lines{budget_lines}"] = sum(1 for d in relevant if d in taken) / len(relevant)
            top_windows = set(int(i) for i in window_order[:window_k])
            row[f"win_lines{budget_lines}"] = sum(
                1 for d in relevant if relevant_windows[d] & top_windows
            ) / len(relevant)
        rows.append(row)

    print(f"snapshot : {snapshot.backend} / {snapshot.model} / {snapshot.dimension}d")
    print(f"queries  : {len(rows)} evaluated, {skipped} skipped (no relevant)")
    print(f"windows  : {len(meta)} of {WINDOW_LINES} lines; functions: {len(snapshot.document_ids)}")
    for k in TOP_KS:
        print(f"  recall@{k:<2} (units)  functions {mean(r[f'fn{k}'] for r in rows):.3f}   "
              f"windows {mean(r[f'win{k}'] for r in rows):.3f}")
    for budget_lines, window_k in ((WINDOW_LINES * 8, 8), (WINDOW_LINES, 1)):
        print(f"  recall @ {budget_lines:>4} shown lines:  "
              f"functions {mean(r[f'fn_lines{budget_lines}'] for r in rows):.3f}   "
              f"windows(top-{window_k}) {mean(r[f'win_lines{budget_lines}'] for r in rows):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="function vs window granularity")
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--corpus-root", type=Path, required=True)
    builder.add_argument("--texts", type=Path, required=True)
    builder.add_argument("--meta", type=Path, required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--manifest", type=Path, required=True)
    evaluator.add_argument("--snapshot", type=Path, required=True)
    evaluator.add_argument("--meta", type=Path, required=True)
    evaluator.add_argument("--window-vectors", type=Path, required=True)
    evaluator.add_argument("--query-count", type=int, default=96)
    evaluator.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build_windows(arguments.corpus_root, arguments.texts, arguments.meta)
    else:
        evaluate(
            arguments.manifest,
            arguments.snapshot,
            arguments.meta,
            arguments.window_vectors,
            arguments.query_count,
            arguments.seed,
        )


if __name__ == "__main__":
    main()
