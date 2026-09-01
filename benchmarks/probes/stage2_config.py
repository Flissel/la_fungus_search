"""What stage 2 would measure, against what it should measure.

`run_gate2._RUN_SPECS` is `(A, B, C, D, E)`. Method G -- the bounded frontier that
section 13 established is the only variant surviving corpus scale -- is absent,
and `run_mcmp` is called with no visit-term override, so the walk ranks with the
`min(0.1 * visits, 0.5)` ceiling that sections 14 and 15 identified as the
bottleneck.

Gate 2 stage 1 became justified for the first time in section 22, on a
4 000-document corpus. Run as coded, stage 2 would therefore measure full-corpus
MCMP on the largest corpus yet attempted, in the configuration this report has
already shown fails there, and would report a strong negative that is a property
of the configuration rather than of the mechanism.

This probe runs the comparison the pre-registered stage 2 cannot: A (FAISS) and
C (full corpus) and G (bounded frontier), each with the shipped visit term and
with `normalised` at alpha 2, decoupled -- section 15's best candidate.

**This is not stage 2.** It is exploratory, it selects its own configuration, and
no gate decision may be read off it.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from statistics import mean
from typing import Iterator

import benchmarks.mcmp.adapters as adapters
from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.snapshot import load_snapshot
from benchmarks.mcmp.adapters import run_faiss, run_mcmp
from benchmarks.mcmp.metrics import evaluate_run
from benchmarks.probes.visit_term import _probe_class

TOP_K = 8
INITIAL_K = 8
STEPS = 50

# (label, term, alpha, decouple)
TERMS = (
    ("shipped", "capped", 0.5, False),
    ("norm a=2 dec", "normalised", 2.0, True),
)


@contextlib.contextmanager
def _term(term: str, alpha: float, decouple: bool) -> Iterator[None]:
    original = adapters.CountingRetriever
    adapters.CountingRetriever = _probe_class(False, term, alpha, decouple)
    try:
        yield
    finally:
        adapters.CountingRetriever = original


def _score(dataset, method: str, seed: int, agents: int) -> tuple[float, int]:
    query_ids = tuple(dataset.query_ids[:1])
    if method == "A":
        run, _ = run_faiss(dataset, "A", query_ids, TOP_K, INITIAL_K)
    else:
        run, _ = run_mcmp(
            dataset, method, query_ids, TOP_K, INITIAL_K, seed, agents, STEPS,
            frontier=method == "G",
        )
    return float(evaluate_run(dataset, run, TOP_K)["recall_at_k"]), run.candidate_comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="stage 2 configuration comparison")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--agents", type=int, default=96)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
    datasets = []
    for seed in range(arguments.seeds):
        try:
            datasets.append((seed, build_gate2_dataset(manifest, snapshot, seed)))
        except ValueError:
            continue

    print(f"snapshot : {snapshot.backend} / {snapshot.model} / {snapshot.dimension}d")
    print(f"corpus   : {len(snapshot.document_ids)} documents, "
          f"{len(datasets)} usable seeds, {arguments.agents} agents, {STEPS} steps")
    print(f"\n{'method':>7}{'visit term':>16}{'recall@8':>10}{'comparisons':>14}")

    for method in ("A", "C", "G"):
        for label, term, alpha, decouple in TERMS:
            if method == "A" and label != "shipped":
                continue  # FAISS does not read the visit term at all
            with _term(term, alpha, decouple):
                scored = [_score(dataset, method, seed, arguments.agents)
                          for seed, dataset in datasets]
            print(
                f"{method:>7}{(label if method != 'A' else '-'):>16}"
                f"{mean(r for r, _ in scored):>10.3f}"
                f"{mean(c for _, c in scored):>14.0f}"
            )


if __name__ == "__main__":
    main()
