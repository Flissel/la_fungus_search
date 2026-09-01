"""Where does chain reachability stop being vacuous?

Gate 2's stage-1 signature is the fraction of relevant documents that are both
*far* (ranked deeper than ``top_k``) and *chain-reachable* in the mutual k-NN
graph. On every corpus measurable offline so far, reachability sits at 1.000 at
the spec's defaults, which makes the signature numerically identical to the far
rate: the second condition contributes nothing and the gate silently tests rank
depth instead of reachability. The Gate 2 spec therefore makes a ``knn_k`` /
``max_hops`` sweep a binding pre-condition of the production run.

**The quantity that matters is not "is reachability below 1.0".** It is whether
reachability *separates the real labels from the permutation null*. If permuted
labels are reached just as easily, the measurement is vacuous no matter what its
absolute value is -- a densely connected graph reaches everything, related or
not. So this sweep reports both, and their gap.

The two fixtures validate the procedure itself before it is applied to anything
real: `manifold` has a planted chain, so a parameter setting that works must open
a gap there; `neutral` has no structure, so the same setting must show no gap.
A sweep that cannot tell those two apart cannot be trusted on a production corpus.

Run::

    PYTHONPATH=src .venv/Scripts/python.exe -m benchmarks.probes.reachability
"""

from __future__ import annotations

import argparse
from statistics import mean

import numpy as np

from benchmarks.gate2.geometry import characterise, geometry_cache, permuted_labels
from benchmarks.mcmp.fixtures import build_dataset

TOP_K = 8
SEEDS = (1, 2, 3, 4, 5, 6)
NULL_PERMUTATIONS = 10
KNN_VALUES = (2, 3, 4, 6, 8, 12)
HOP_VALUES = (1, 2, 3, 4, 6)
MANIFOLD_DOCUMENTS = 256


def _reachability_rate(measurement: dict[str, object]) -> float | None:
    """Fraction of *far* pairs that are chain-reachable, or None if none are far.

    Normalising over far pairs rather than all pairs is what separates the two
    conditions the signature multiplies together. The signature can fall simply
    because fewer documents are far; this cannot.
    """
    far = int(measurement["far_count"])
    if far == 0:
        return None
    return int(measurement["far_and_reachable_count"]) / far


def _sweep_cell(
    fixture: str, documents: int | None, knn_k: int, max_hops: int
) -> tuple[float | None, float | None]:
    real_rates: list[float] = []
    null_rates: list[float] = []
    for seed in SEEDS:
        dataset = build_dataset(fixture, seed, documents)
        # The cache holds the k-NN graph and the pairwise matrix, both of which
        # depend only on the vectors. Permuting labels leaves the vectors alone,
        # so one cache serves the real measurement and every permutation.
        cache = geometry_cache(dataset, knn_k)
        rate = _reachability_rate(
            characterise(dataset, TOP_K, knn_k, max_hops, 0.0, cache=cache)
        )
        if rate is not None:
            real_rates.append(rate)
        rng = np.random.default_rng(seed)
        for _ in range(NULL_PERMUTATIONS):
            null = permuted_labels(dataset, rng)
            null_rate = _reachability_rate(
                characterise(null, TOP_K, knn_k, max_hops, 0.0, cache=cache)
            )
            if null_rate is not None:
                null_rates.append(null_rate)
    return (
        mean(real_rates) if real_rates else None,
        mean(null_rates) if null_rates else None,
    )


def _format(value: float | None) -> str:
    return "  --  " if value is None else f"{value:6.3f}"


def sweep(fixture: str, documents: int | None) -> None:
    label = f"{fixture}" + (f", {documents} documents" if documents else "")
    print(
        f"\n{label}, top_k={TOP_K}, hop_threshold=0.0, seeds {SEEDS}, "
        f"{NULL_PERMUTATIONS} permutations per seed"
    )
    print("reachability among far pairs: real / null / gap")
    header = "".join(f"{f'hops={h}':>22}" for h in HOP_VALUES)
    print(f"{'knn_k':>6}{header}")
    for knn_k in KNN_VALUES:
        cells = []
        for max_hops in HOP_VALUES:
            real, null = _sweep_cell(fixture, documents, knn_k, max_hops)
            gap = (
                f"{real - null:+.3f}"
                if real is not None and null is not None
                else "  --  "
            )
            cells.append(f"{_format(real)}/{_format(null)}/{gap:>7}")
        print(f"{knn_k:>6}" + "".join(f"{c:>22}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chain-reachability saturation sweep")
    parser.add_argument(
        "--fixture", choices=("manifold", "neutral", "both"), default="both"
    )
    parser.add_argument("--documents", type=int, default=MANIFOLD_DOCUMENTS)
    arguments = parser.parse_args()
    if arguments.fixture in ("manifold", "both"):
        sweep("manifold", arguments.documents)
    if arguments.fixture in ("neutral", "both"):
        sweep("neutral", None)


if __name__ == "__main__":
    main()
