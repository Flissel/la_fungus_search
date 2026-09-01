"""What actually carries MCMP's ranking, and what breaks it as the corpus grows.

Seven experiments, each answering one question, none of them a gate. Nothing in
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

``alpha``
    Sweep the replacement term's weight, which `terms` held at 0.5 throughout,
    and add two rank-based shapes that do not normalise against the peak.

``confirm``
    Carry the alpha sweep's winner across the agent budget, which is the other
    dimension along which the shipped ceiling collapses.

``decouple``
    The relevance score steers the walk and ranks the result at once. Apply the
    replacement to the ranking only, and see which of the two jobs it was doing.

``final``
    Both candidate fixes, coupled and decoupled, across the agent budget.

Every experiment takes ``--fixture``. Always run ``neutral`` as well as
``manifold``: relevance there is drawn from the FAISS top-16, so similarity is the
correct signal and a visit term that overrides it must *hurt*. A setting that wins
on manifold without that check is tuned to one fixture, not improved.

Run::

    python -m benchmarks.probes.visit_term --experiment alpha --fixture manifold
    python -m benchmarks.probes.visit_term --experiment alpha --fixture neutral
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


# Every measurement before 2026-09-01 held the term's weight at 0.5, chosen for
# comparability with the shipped ceiling. It is swept here, because "the shape
# does not help" and "0.5 of that shape does not help" are different claims.
DEFAULT_ALPHA = 0.5


def _visit_scores(visits: np.ndarray, term: str, alpha: float) -> np.ndarray:
    """The replacement visit term, as a vector over the working set.

    None of these can saturate the way the shipped ceiling does, but they fail in
    different regimes, and the failure is about *spread* rather than magnitude.
    Measured directly (`alpha` 0.5, so 0.5 is the full available range):

    | shape        | 12 docs, all visited | 1000 docs, 10 visited |
    |--------------|----------------------|-----------------------|
    | log          | 0.014                | 0.407                 |
    | normalised   | 0.071                | 0.488                 |
    | rank         | 0.500                | 0.005                 |
    | visited_rank | 0.500                | 0.500                 |

    `log` and `normalised` divide by the peak, so once a small working set is
    uniformly hammered they compress every document into a narrow band -- the
    ceiling's problem in another form. `rank` is the empirical CDF over the whole
    working set, which spreads that dense case fully but collapses when almost
    nothing is visited, because all the visited documents share the same top
    percentile. `visited_rank` ranks among the visited only, and is the one shape
    that keeps the full range in both regimes: the visited/unvisited distinction is
    already carried by the zero floor and need not consume the range as well.
    """
    if term == "none":
        return np.zeros_like(visits)
    if term == "uncapped":
        return 0.1 * visits
    peak = float(visits.max()) if visits.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(visits)
    if term == "log":
        return alpha * np.log1p(visits) / float(np.log1p(peak))
    if term == "normalised":
        return alpha * visits / peak
    if term == "rank":
        below = (visits[:, None] > visits[None, :]).sum(axis=1)
        return alpha * below / max(1, visits.size - 1)
    if term == "visited_rank":
        # `rank` over the whole working set is scale-free in the dense regime and
        # collapses in the sparse one: with 990 of 1000 documents unvisited, every
        # visited document sits in the same top percentile and they score within
        # 0.005 of each other. Ranking *among the visited* keeps the full spread in
        # both regimes, because the visited/unvisited split is already carried by
        # the zero floor and does not need to consume the range as well.
        visited = visits > 0
        count = int(visited.sum())
        scores = np.zeros_like(visits)
        if count == 0:
            return scores
        if count == 1:
            scores[visited] = alpha
            return scores
        below = ((visits[:, None] > visits[None, :]) & visited[None, :]).sum(axis=1)
        scores[visited] = alpha * below[visited] / (count - 1)
        return scores
    raise ValueError(f"unknown visit term {term!r}")


def _apply_visit_term(
    retriever: CountingRetriever, term: str, alpha: float = DEFAULT_ALPHA
) -> None:
    """Swap the shipped ``min(0.1 * visits, 0.5)`` for another shape of the term."""
    if term == "capped":
        return
    visits = np.array([d.visit_count for d in retriever.documents], dtype=np.float64)
    replacement = _visit_scores(visits, term, alpha)
    for document, count, score in zip(retriever.documents, visits, replacement):
        document.relevance_score -= min(count * 0.1, 0.5)
        document.relevance_score += float(score)


def _probe_class(
    recency: bool, term: str, alpha: float = DEFAULT_ALPHA, decouple: bool = False
) -> type[CountingRetriever]:
    """Build a retriever variant.

    ``decouple`` separates the two jobs the relevance score currently does at
    once. `update_document_relevance` sets `relevance_score`, which the attraction
    force reads through its `(1 + r)` weight *and* which the harness reads as the
    final ranking. With `decouple` set, every step but the last keeps the shipped
    term -- so the walk is steered exactly as method C steers it -- and the
    replacement term is applied only on the final call, where it affects the
    ranking alone. Any difference between the coupled and decoupled runs is the
    replacement term's effect on the walk rather than on the score.
    """

    class Probe(CountingRetriever):
        _relevance_calls = 0

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
            self._relevance_calls += 1
            if decouple and self._relevance_calls < STEPS:
                return
            _apply_visit_term(self, term, alpha)

    return Probe


@contextlib.contextmanager
def _variant(
    symmetric: bool = False,
    recency: bool = False,
    term: str = "capped",
    alpha: float = DEFAULT_ALPHA,
    decouple: bool = False,
) -> Iterator[None]:
    original_force = sim.calculate_pheromone_force
    original_class = adapters.CountingRetriever
    if symmetric:
        # `update_agent_position` resolves this name in the simulation module at
        # call time, so rebinding it there reaches the running walk.
        sim.calculate_pheromone_force = symmetric_pheromone_force
    adapters.CountingRetriever = _probe_class(recency, term, alpha, decouple)
    try:
        yield
    finally:
        sim.calculate_pheromone_force = original_force
        adapters.CountingRetriever = original_class


# The fixture under measurement. `neutral` is the control: its relevant documents
# are drawn from the FAISS top-16, so similarity is the correct signal there and a
# visit term that overrides similarity should *hurt*. Any setting that wins on
# manifold without being checked here is tuned to one fixture, not improved.
FIXTURE = "manifold"


def _recall(method: str, seed: int, agents: int, documents: int) -> float:
    # Only the manifold fixture accepts a corpus size; the others are fixed shape.
    size = documents if FIXTURE == "manifold" else None
    dataset = build_dataset(FIXTURE, seed, size)
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
    size = f"{documents} documents" if FIXTURE == "manifold" else "fixed size"
    print(
        f"\n{FIXTURE}, {size}, {STEPS} steps, seeds {SEEDS}, "
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


ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SWEEP_AGENTS = 192
SWEEP_AGENT_RANGE: tuple[int, ...] = (SWEEP_AGENTS,)
DECOUPLE = False
SWEEP_SHAPES = ("log", "rank", "visited_rank", "normalised")


def experiment_alpha(documents: int) -> None:
    """Sweep the term's weight, which every earlier measurement held at 0.5."""
    for agents in SWEEP_AGENT_RANGE:
        _header(documents)
        print(f"held: {agents} agents, decouple={DECOUPLE}")
        print(f"{'method':>6} {'shape':>13}" + "".join(f"{a:>9}" for a in ALPHAS))
        for method in ("C", "G"):
            for term in ("capped", "uncapped"):
                with _variant(term=term, decouple=DECOUPLE):
                    value = _mean_recall(method, agents, documents)
                pad = " " * (9 * (len(ALPHAS) - 1))
                print(f"{method:>6} {term:>13}{value:>9.3f}{pad}")
            for shape in SWEEP_SHAPES:
                row = []
                for alpha in ALPHAS:
                    with _variant(term=shape, alpha=alpha, decouple=DECOUPLE):
                        row.append(_mean_recall(method, agents, documents))
                print(f"{method:>6} {shape:>13}" + "".join(f"{v:>9.3f}" for v in row))


def experiment_decouple(documents: int) -> None:
    """Does the replacement term help by steering the walk, or by scoring it?

    Decoupled runs steer with the shipped term and score with the replacement, so
    the gap between the two columns is the term's effect on the walk itself.
    """
    _header(documents)
    print(f"held: {SWEEP_AGENTS} agents, alpha {DEFAULT_ALPHA}")
    print(f"{'method':>6} {'shape':>10}{'coupled':>10}{'decoupled':>11}")
    for method in ("C", "G"):
        for shape in ("uncapped", "log", "visited_rank"):
            values = []
            for decouple in (False, True):
                with _variant(term=shape, decouple=decouple):
                    values.append(_mean_recall(method, SWEEP_AGENTS, documents))
            print(f"{method:>6} {shape:>10}{values[0]:>10.3f}{values[1]:>11.3f}")


CANDIDATES = (("capped", DEFAULT_ALPHA), ("normalised", 1.0), ("normalised", 2.0))


def experiment_confirm(documents: int) -> None:
    """Carry the alpha sweep's winner across the agent budget.

    The alpha sweep held agents at 192. Agent count is the *other* dimension along
    which the shipped ceiling collapses (report section 10), so a term that only
    works at one budget has not fixed anything.
    """
    columns = tuple(f"{term} a={alpha:g}" for term, alpha in CANDIDATES)

    def cell(method: str, agents: int, docs: int, column: str) -> float:
        term, alpha = CANDIDATES[columns.index(column)]
        with _variant(term=term, alpha=alpha):
            return _mean_recall(method, agents, docs)

    _header(documents)
    print(f"{'method':>6} {'agents':>7}" + "".join(f"{name:>16}" for name in columns))
    for method in ("C", "G"):
        for agents in AGENT_COUNTS:
            values = [cell(method, agents, documents, name) for name in columns]
            print(f"{method:>6} {agents:>7}" + "".join(f"{v:>16.3f}" for v in values))


# (term, alpha, decouple). `uncapped` ignores alpha.
FINAL_CANDIDATES = (
    ("capped", DEFAULT_ALPHA, False),
    ("normalised", 2.0, False),
    ("uncapped", DEFAULT_ALPHA, True),
    ("normalised", 2.0, True),
)


def experiment_final(documents: int) -> None:
    """The two candidate fixes, coupled and decoupled, across the agent budget."""
    columns = tuple(
        f"{term[:6]} a={alpha:g}{' dec' if dec else ''}"
        for term, alpha, dec in FINAL_CANDIDATES
    )

    def cell(method: str, agents: int, docs: int, column: str) -> float:
        term, alpha, decouple = FINAL_CANDIDATES[columns.index(column)]
        with _variant(term=term, alpha=alpha, decouple=decouple):
            return _mean_recall(method, agents, docs)

    _header(documents)
    print(f"{'method':>6} {'agents':>7}" + "".join(f"{name:>18}" for name in columns))
    for method in ("C", "G"):
        for agents in AGENT_COUNTS:
            values = [cell(method, agents, documents, name) for name in columns]
            print(f"{method:>6} {agents:>7}" + "".join(f"{v:>18.3f}" for v in values))


EXPERIMENTS = {
    "defects": experiment_defects,
    "ablate": experiment_ablate,
    "terms": experiment_terms,
    "alpha": experiment_alpha,
    "confirm": experiment_confirm,
    "decouple": experiment_decouple,
    "final": experiment_final,
}


def main() -> None:
    global FIXTURE, SWEEP_AGENT_RANGE, DECOUPLE
    parser = argparse.ArgumentParser(description="MCMP relevance-term probes")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--documents", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--fixture", choices=("manifold", "neutral"), default="manifold")
    parser.add_argument("--agents", type=int, nargs="+", default=[SWEEP_AGENTS])
    parser.add_argument("--decouple", action="store_true")
    arguments = parser.parse_args()
    FIXTURE = arguments.fixture
    SWEEP_AGENT_RANGE = tuple(arguments.agents)
    DECOUPLE = arguments.decouple
    documents = arguments.documents if FIXTURE == "manifold" else [0]
    for size in documents:
        EXPERIMENTS[arguments.experiment](size)


if __name__ == "__main__":
    main()
