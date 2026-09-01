"""Prose oracle: a wikilink vault emitted in the Gate 2 manifest schema.

Section 27 measured the retrieval stack on code, where BM25 enjoys identifier
overlap and the call graph supplies a mechanical oracle. This converts a
markdown vault — notes as documents, resolved ``[[wikilinks]]`` as edges — into
the exact manifest schema the Gate 2 tooling reads, so the *entire* existing
measurement stack (bm25_hybrid, callgraph_expand, run_stage1_split) runs on
prose unchanged. What generalises and what was a code artifact becomes visible
by diffing two tables produced by the same instruments.

The oracle is **authored**, and that bias is part of the record: links measure
"finds what the writer already connected", which is weaker than a mechanical
call graph. Resolution is fail-closed like the code manifest's: a link target
matching more than one note stem is discarded and counted, never guessed.

**Privacy rule, load-bearing:** the vault is private and this repository's
remote is public. Everything this writes — manifest, sources, snapshots — goes
under ``benchmarks/results/vault/``, which is gitignored. Committing any of it
would publish personal notes; do not.

Run::

    PYTHONPATH=src python -m benchmarks.probes.vault_manifest \
        --vault C:/Users/User/Desktop/secondbrain \
        --out-dir benchmarks/results/vault
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")


def _commit_sha(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "no-git"


def build(vault: Path, out_dir: Path, manifest_id: str) -> None:
    notes: list[dict[str, object]] = []
    for path in sorted(vault.rglob("*.md")):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        relative = path.relative_to(vault).as_posix()
        notes.append(
            {
                "document_id": relative,
                "file": relative,
                "start_line": 1,
                "end_line": max(1, text.count("\n") + 1),
                "symbol": path.stem,
                "source": text,
            }
        )

    # Fail-closed stem resolution, exactly the code manifest's rule: a target
    # naming more than one note is discarded and counted, never guessed.
    stem_count = Counter(str(note["symbol"]).lower() for note in notes)
    note_by_stem = {
        str(note["symbol"]).lower(): str(note["document_id"])
        for note in notes
        if stem_count[str(note["symbol"]).lower()] == 1
    }
    known_ids = {str(note["document_id"]) for note in notes}

    callees: dict[str, set[str]] = {identifier: set() for identifier in known_ids}
    discarded: set[str] = set()
    unresolved: set[str] = set()
    for note in notes:
        origin = str(note["document_id"])
        for match in WIKILINK.finditer(str(note["source"])):
            target = match.group(1).strip().lower()
            if not target:
                continue
            if stem_count.get(target, 0) > 1:
                discarded.add(target)
            elif target in note_by_stem:
                resolved = note_by_stem[target]
                if resolved != origin:
                    callees[origin].add(resolved)
            else:
                unresolved.add(target)

    callers: dict[str, set[str]] = {identifier: set() for identifier in known_ids}
    for origin, targets in callees.items():
        for target in targets:
            callers[target].add(origin)

    payload = {
        "manifest_id": manifest_id,
        "corpus_root": str(vault),
        "commit_sha": _commit_sha(vault),
        "documents": notes,
        "callees_by_document": {key: sorted(value) for key, value in sorted(callees.items())},
        "callers_by_document": {key: sorted(value) for key, value in sorted(callers.items())},
        "discarded_names": sorted(discarded),
        "unresolved_names": sorted(unresolved),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{manifest_id}.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sources_path = out_dir / f"{manifest_id}-sources.json"
    sources_path.write_text(json.dumps([note["source"] for note in notes]), encoding="utf-8")

    linked = sum(1 for identifier in known_ids if callees[identifier] or callers[identifier])
    edge_count = sum(len(value) for value in callees.values())
    print(f"notes      : {len(notes)}")
    print(f"linked     : {linked}")
    print(f"edges      : {edge_count} resolved directed")
    print(f"discarded  : {len(discarded)} ambiguous stems")
    print(f"unresolved : {len(unresolved)} targets without a note")
    print(f"manifest   : {manifest_path}")
    print(f"sources    : {sources_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="vault -> Gate 2 manifest schema")
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-id", default="vault-v1")
    arguments = parser.parse_args()
    build(arguments.vault, arguments.out_dir, arguments.manifest_id)


if __name__ == "__main__":
    main()
