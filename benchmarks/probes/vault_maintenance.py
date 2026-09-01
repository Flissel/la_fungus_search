"""a) Vault maintenance from the manifest: broken links, orphans, suggestions.

The wikilink manifest (`vault_manifest.py`) already carries the maintenance
findings as a by-product of fail-closed resolution; this turns them into a
report and adds the one feature that needs evidence before anyone trusts it —
**link suggestions** (similar notes that are not linked), evaluated by hold-out
link prediction rather than by anyone's judgement:

    remove a fixed fraction of the real edges, rank all unlinked pairs by
    embedding similarity, and count how many of the removed edges surface in
    the top suggestions. Embeddings never see the edges, so the held-out
    evaluation is clean by construction.

An LLM judge is ruled out by the standing ground-truth rule; the held-out edges
*are* the ground truth, authored by the vault's owner.

Output goes to the terminal and (optionally) into the gitignored results
directory. Note titles never belong in anything committed: the vault is private
and this repository's remote is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.snapshot import load_snapshot

HOLDOUT_FRACTION = 0.2
SUGGESTIONS_PER_NOTE = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="vault maintenance report")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suggest", type=int, default=10, help="print this many suggestions")
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
    documents = manifest.documents
    identifier = [document.document_id for document in documents]
    position = {document_id: index for index, document_id in enumerate(identifier)}

    # --- broken links: unresolved targets, attributed to the notes naming them.
    import re

    wikilink = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
    unresolved = set(manifest.unresolved_names)
    broken: dict[str, list[str]] = {}
    for document in documents:
        for match in wikilink.finditer(document.source):
            target = match.group(1).strip().lower()
            if target in unresolved:
                broken.setdefault(target, []).append(document.document_id)

    # --- orphans: no resolved link in either direction.
    orphans = [
        document.document_id
        for document in documents
        if not manifest.callees_by_document.get(document.document_id)
        and not manifest.callers_by_document.get(document.document_id)
    ]

    # --- existing undirected link pairs.
    linked: set[tuple[int, int]] = set()
    for origin, targets in manifest.callees_by_document.items():
        for target in targets:
            a, b = position[origin], position[target]
            linked.add((min(a, b), max(a, b)))

    vectors = snapshot.vectors
    similarities = vectors @ vectors.T

    # --- hold-out evaluation of similarity as a link-suggestion signal.
    rng = np.random.default_rng(arguments.seed)
    pairs = sorted(linked)
    held_count = max(1, int(len(pairs) * HOLDOUT_FRACTION))
    held = {pairs[int(i)] for i in rng.choice(len(pairs), size=held_count, replace=False)}
    remaining = linked - held

    candidate_scores: list[tuple[float, tuple[int, int]]] = []
    n = len(identifier)
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) in remaining:
                continue
            candidate_scores.append((float(similarities[a, b]), (a, b)))
    candidate_scores.sort(reverse=True)
    top = [pair for _score, pair in candidate_scores[: len(held)]]
    recovered = sum(1 for pair in top if pair in held)
    pool = len(candidate_scores)
    chance = len(held) * len(held) / pool

    # --- the actual suggestions, computed against the full link set.
    suggestions: list[tuple[float, str, str]] = []
    for score, (a, b) in candidate_scores:
        if (a, b) in held:  # it is a real link; suggesting it proves nothing
            continue
        suggestions.append((score, identifier[a], identifier[b]))
        if len(suggestions) >= arguments.suggest:
            break

    print(f"vault      : {manifest.corpus_root}")
    print(f"notes      : {len(documents)}  |  resolved edges: {len(linked)} undirected")
    print(f"\n=== broken links ({len(broken)} targets) ===")
    for target, origins in sorted(broken.items()):
        print(f"  [[{target}]]  <- {', '.join(sorted(origins)[:3])}"
              + (f" (+{len(origins) - 3})" if len(origins) > 3 else ""))
    print(f"\n=== orphan notes ({len(orphans)}) ===")
    for document_id in orphans:
        print(f"  {document_id}")
    print(f"\n=== link suggestions: does similarity deserve trust? (hold-out) ===")
    print(f"  edges held out          {len(held)} of {len(linked)}")
    print(f"  recovered in top-{len(held):<4}   {recovered}  ({recovered / len(held):.1%})")
    print(f"  chance would recover    {chance:.1f}  ({chance / len(held):.1%})")
    print(f"\n=== top {arguments.suggest} suggested links (unlinked, most similar) ===")
    for score, left, right in suggestions:
        print(f"  {score:5.3f}  {left}  <->  {right}")


if __name__ == "__main__":
    main()
