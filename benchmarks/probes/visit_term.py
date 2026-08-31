"""What actually carries MCMP's ranking, and what breaks it as the corpus grows.

Three experiments, each answering one question, none of them a gate. Nothing in
``src/`` is modified: every variant is an override applied for the duration of a
single measurement and reverted afterwards.

``defects``
    Two implementation defects in the pheromone mechanism, repaired and measured.
    A trail is stored under ``tuple(sorted(...))`` but followed only when the
    agent stands on the *lower* document id, so half the deposited signal is
    unreachable. And ``list(agent.visited_docs)[-3:]`` slices a *set*, so the
    "last three visited documents" are an arbitrary hash-ordered three.

``ablate``
    Remove the visit term from the relevance score and see what is left.

``terms``
    Replace the visit term's hard cap with shapes that cannot saturate.

Run::

    python -m benchmarks.probes.visit_term --experiment terms
"""

from __future__ import annotations

import argparse
import contextlib
from statistics import mean
from typing import Iterator

import numpy as np

import benchmarks.mcmp.adapters as adapters
from benchmarks.mcmp.adapters import CountingRetriever
from benchmarks.mcmp.fixtures import build_dataset
from benchmarks.mcmp.metrics import evaluate_run
from embeddinggemma.mcmp import simulation as sim

TOP_K = 8
INITIAL_K = 8
STEPS = 50
SEEDS = (1, 2, 3, 4, 5, 6)
AGENT_COUNTS = (96, 192, 384)

# recall@8 is measured over one query with three relevant documents across six
# seeds, so the smallest resolvable difference is 1/18. Treat a gap of that size
# as one document in one seed -- noise. The differences worth reading are large.
RECALL_RESOLUTION = 1.0 / (len(SEEDS) * 3)


def symmetric_pheromone_force(retriever: object, agent: object) -> np.ndarray:
    """``calculate_pheromone_force``, but a trail is followable from either end."""
    trails = retriever.pheromone_trails  # type: ignore[attr-defined]
    if not trails:
        return np.zeros_like(agent.position)  # type: ignore[attr-defined]
    force = np.zeros_like(agent.position)  # type: ignore[attr-defined]
    current = retriever.find_nearest_documents(agent.position, k=1)  # type: ignore[attr-defined]
    if not current:
        return force
    current_id = current[0][0].id
    max_strength = 0.0
    best_direction = None
    for (document_a, document_b), strength in trails.items():
        if document_a == current_id:
            other = document_b
        elif document_b == current_id:
            other = document_a
        else:
            continue
        if strength <= max_strength:
            continue
        target = next(
            (d for d in retriever.documents if d.id == other), None  # type: ignore[attr-defined]
        )
        if target is None:
            continue
        max_strength = float(strength)
        best_direction = target.embedding - agent.position  # type: ignore[attr-defined]
    if best_direction is not None and np.linalg.norm(best_direction) > 0:
        force = best_direction / np.linalg.norm(best_direction) * max_strength
    return force


def _apply_visit_term(retriever: CountingRetriever, term: str) -> None:
    """Swap the shipped ``min(0.1 * visits, 0.5)`` for another shape of the term."""
    if term == "capped":
        return
    visits = np.array([d.visit_count for d in retriever.documents], dtype=np.float64)
    peak = float(visits.max()) if visits.size else 0.0
    for document, count in zip(retriever.documents, visits):
        document.relevance_score -= min(count * 0.1, 0.5)
        if term == "none":
            continue
        if term == "uncapped":
            document.relevance_score += 0.1 * count
        elif peak <= 0.0:
            continue
        elif term == "log":
            document.relevance_score += 0.5 * float(np.log1p(count)) / float(np.log1p(peak))
        elif term == "normalised":
            document.relevance_score += 0.5 * (count / peak)
        else:
            raise ValueError(f"unknown visit term {term!r}")


def _probe_class(recency: bool, term: str) -> type[CountingRetriever]:
    class Probe(CountingRetriever):
        def deposit_pheromones(self, agent: object) -> None:  # type: ignore[override]
            if not recency:
                super().deposit_pheromones(agent)
                return
            current = self.find_nearest_documents(agent.position, k=1)  # type: ignore[attr-defined]
            if not current:
                return
            document = current[0][0]
            document.visit_count += 1
            document.last_visited = float(self.time_source())
            history = getattr(agent, "visit_history", None)
            if history is None:
                history = []
                agent.visit_history = history  # type: ignore[attr-defined]
            for previous in history[-3:]:
                if previous == document.id:
                    continue
                key = tuple(sorted([document.id, previous]))
                amount = agent.energy * agent.trail_strength * 0.1  # type: ignore[attr-defined]
                self.pheromone_trails[key] = self.pheromone_trails.get(key, 0.0) + amount
            agent.visited_docs.add(document.id)  # type: ignore[attr-defined]
            history.append(document.id)

        def update_document_relevance(self, query_embedding: np.ndarray) -> None:  # type: ignore[override]
            super().update_document_relevance(query_embedding)
            _apply_visit_term(self, term)

    return Probe


@contextlib.contextmanager
def _variant(
    symmetric: bool = False, recency: bool = False, term: str = "capped"
) -> Iterator[None]:
    original_force = sim.calculate_pheromone_force
    original_class = adapters.CountingRetriever
    if symmetric:
        # `update_agent_position` resolves this name in the simulation module at
        # call time, so rebinding it there reaches the running walk.
        sim.calculate_pheromone_force = symmetric_pheromone_force
    adapters.CountingRetriever = _probe_class(recency, term)
    try:
        yield
    finally:
        sim.calculate_pheromone_force = original_force
        adapters.CountingRetriever = original_class


def _recall(method: str, seed: int, agents: int, documents: int) -> float:
    dataset = build_dataset("manifold", seed, documents)
    run, _ = adapters.run_mcmp(
        dataset,
        method,
        ("q-main",),
        TOP_K,
        INITIAL_K,
        seed,
        agents,
        STEPS,
        frontier=method == "G",
    )
    return float(evaluate_run(dataset, run, TOP_K)["recall_at_k"])


def _mean_recall(method: str, agents: int, documents: int) -> float:
    return mean(_recall(method, seed, agents, documents) for seed in SEEDS)


def _header(documents: int) -> None:
    print(
        f"\nmanifold, {documents} documents, {STEPS} steps, seeds {SEEDS}, "
        f"recall@8 (resolution {RECALL_RESOLUTION:.3f})"
    )


def _table(documents: int, columns: tuple[str, ...], cell: object) -> None:
    _header(documents)
    print(f"{'method':>6} {'agents':>7}" + "".join(f"{name:>12}" for name in columns))
    for method in ("C", "G"):
        for agents in AGENT_COUNTS:
            values = [cell(method, agents, documents, name) for name in columns]
            print(f"{method:>6} {agents:>7}" + "".join(f"{v:>12.3f}" for v in values))


def experiment_defects(documents: int) -> None:
    """Repair the two pheromone defects, separately and together."""
    repairs = {
        "as built": (False, False),
        "symmetric": (True, False),
        "recency": (False, True),
        "both": (True, True),
    }

    def cell(method: str, agents: int, docs: int, name: str) -> float:
        symmetric, recency = repairs[name]
        with _variant(symmetric=symmetric, recency=recency):
            return _mean_recall(method, agents, docs)

    _table(documents, tuple(repairs), cell)


def experiment_ablate(documents: int) -> None:
    """Score with and without the visit term, everything else identical."""

    def cell(method: str, agents: int, docs: int, term: str) -> float:
        with _variant(term=term):
            return _mean_recall(method, agents, docs)

    _table(documents, ("capped", "none"), cell)


def experiment_terms(documents: int) -> None:
    """Compare the shipped cap against three shapes that cannot saturate."""

    def cell(method: str, agents: int, docs: int, term: str) -> float:
        with _variant(term=term):
            return _mean_recall(method, agents, docs)

    _table(documents, ("capped", "uncapped", "log", "normalised"), cell)


EXPERIMENTS = {
    "defects": experiment_defects,
    "ablate": experiment_ablate,
    "terms": experiment_terms,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MCMP relevance-term probes")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--documents", type=int, nargs="+", default=[256, 1024])
    arguments = parser.parse_args()
    for documents in arguments.documents:
        EXPERIMENTS[arguments.experiment](documents)


if __name__ == "__main__":
    main()
