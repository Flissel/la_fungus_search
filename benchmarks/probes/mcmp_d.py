"""MCMP-D: the colony walks the mutual k-NN graph instead of continuous space.

Spec: docs/superpowers/specs/2026-09-01-mcmp-d-design.md. The one-line diagnosis
it implements: stage 1 proved the structure exists in the discrete graph (§22),
the continuous walk cannot traverse it (§25), so the colony moves onto the graph.

Three variants, because the design converges toward personalized PageRank and has
to prove it is more than that:

- ``ppr``    exact personalized PageRank, power iteration. Deterministic baseline.
- ``walk``   agent walk, pheromone off. Monte Carlo PPR; the §11 F-control.
- ``colony`` the full mechanism: hop probability weight x (1 + pheromone)^beta,
             symmetric edge pheromone, deposit and decay per step.

Evaluated against the call-graph oracle on the pure semantic graph (call edges in
the walk graph would be leakage — see the spec's oracle rule).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

import numpy as np

from benchmarks.gate2.geometry import chain_reachable, geometry_cache, knn_graph
from benchmarks.gate2.manifest import load_manifest, manifest_digest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.snapshot import load_snapshot
from benchmarks.mcmp.adapters import run_faiss

INITIAL_K = 8
TOP_K = 8
KNN_K = 8
ALPHA = 0.15
BETA = 1.0
DEPOSIT = 0.1
DECAY = 0.95
CEILING_HOPS = 6  # the stage-1 operating point the §22 gate opened at


def build_edges(
    vectors: np.ndarray, knn_k: int
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], int]:
    """Mutual k-NN adjacency with cosine weights, as per-node arrays.

    Reuses the audited `knn_graph` so the walk moves on *exactly* the graph the
    stage-1 reachability measurement certifies — not a near-relative of it.
    """
    graph = knn_graph(vectors, knn_k)
    similarities = vectors @ vectors.T
    edges: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    edge_count = 0
    for node, neighbours in graph.items():
        targets = np.asarray(sorted(neighbours), dtype=np.int64)
        weights = np.clip(similarities[node, targets], 0.0, None).astype(np.float64)
        edges[node] = (targets, weights)
        edge_count += len(targets)
    return edges, edge_count


def start_distribution(dataset, query_id: str) -> tuple[np.ndarray, np.ndarray]:
    similarities = dataset.query_vectors[list(dataset.query_ids).index(query_id)] @ dataset.document_vectors.T
    order = np.argsort(-similarities)[:INITIAL_K]
    # Normalise in float64: a float32 division leaves a sum that
    # `Generator.choice` rejects as "probabilities do not sum to 1".
    mass = np.clip(similarities[order].astype(np.float64), 1e-9, None)
    mass /= mass.sum()
    return order.astype(np.int64), mass


def exact_ppr(
    edges: dict[int, tuple[np.ndarray, np.ndarray]],
    node_count: int,
    start_nodes: np.ndarray,
    start_mass: np.ndarray,
    iterations: int = 60,
) -> np.ndarray:
    restart = np.zeros(node_count)
    restart[start_nodes] = start_mass
    mass = restart.copy()
    for _ in range(iterations):
        pushed = np.zeros(node_count)
        for node in range(node_count):
            value = mass[node]
            if value <= 0.0:
                continue
            targets, weights = edges.get(node, (None, None))
            if targets is None or len(targets) == 0 or weights.sum() <= 0.0:
                # Dangling mass restarts, so the total is conserved and the
                # iteration cannot leak probability out of the graph.
                pushed += value * restart
                continue
            pushed[targets] += value * (weights / weights.sum())
        mass = ALPHA * restart + (1.0 - ALPHA) * pushed
    return mass


def colony_walk(
    edges: dict[int, tuple[np.ndarray, np.ndarray]],
    node_count: int,
    start_nodes: np.ndarray,
    start_mass: np.ndarray,
    agents: int,
    steps: int,
    pheromone: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Discrete colony. Returns visit counts and the number of hops spent."""
    positions = rng.choice(start_nodes, size=agents, p=start_mass)
    visits = np.zeros(node_count, dtype=np.int64)
    trails: dict[tuple[int, int], float] = {}
    hops = 0
    for node in positions:
        visits[node] += 1
    for _ in range(steps):
        if pheromone and trails:
            for key in list(trails):
                trails[key] *= DECAY
                if trails[key] < 0.01:
                    del trails[key]
        for index in range(agents):
            node = int(positions[index])
            if rng.random() < ALPHA:
                target = int(rng.choice(start_nodes, p=start_mass))
            else:
                targets, weights = edges.get(node, (None, None))
                if targets is None or len(targets) == 0:
                    target = int(rng.choice(start_nodes, p=start_mass))
                else:
                    if pheromone:
                        boost = np.fromiter(
                            (
                                (1.0 + trails.get((min(node, int(t)), max(node, int(t))), 0.0)) ** BETA
                                for t in targets
                            ),
                            dtype=np.float64,
                            count=len(targets),
                        )
                        probabilities = weights * boost
                    else:
                        probabilities = weights
                    total = probabilities.sum()
                    if total <= 0.0:
                        target = int(rng.choice(start_nodes, p=start_mass))
                    else:
                        target = int(targets[rng.choice(len(targets), p=probabilities / total)])
                        if pheromone:
                            key = (min(node, target), max(node, target))
                            trails[key] = trails.get(key, 0.0) + DEPOSIT
                        hops += 1
            positions[index] = target
            visits[target] += 1
    return visits, hops


def _rank_of(mass: np.ndarray, similarities: np.ndarray, node: int) -> int:
    """Rank by mass, similarity as the deterministic tie-break."""
    order = np.lexsort((-similarities, -mass))
    return int(np.where(order == node)[0][0]) + 1


def scoring_variants(
    colony: np.ndarray, ppr: np.ndarray, similarities: np.ndarray
) -> dict[str, np.ndarray]:
    """Candidate signals for discriminating relevance *within* the reached set.

    Raw mass fails by construction: stationary mass decays geometrically with hop
    distance, so a four-hop document can never out-mass the start pool. Each
    variant here tries to remove the distance decay a different way.

    - ``mass``     the failing baseline, kept for comparison.
    - ``ratio``    colony visitation over the diffusion expectation (exact PPR).
                   Reinforcement should over-visit what diffusion alone would not.
    - ``support``  similarity, but only the visited set competes; the walk acts as
                   a candidate filter and similarity ranks inside it.
    """
    visits_norm = colony / max(1.0, colony.sum())
    visited = colony > 0
    ratio = np.where(visited, visits_norm / (ppr + 1e-12), 0.0)
    support_sim = np.where(visited, similarities, -np.inf)
    return {
        "mass": colony.astype(np.float64),
        "ratio": ratio,
        "support": support_sim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCMP-D on the mutual k-NN graph")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--agents", type=int, default=96)
    parser.add_argument("--steps", type=int, default=50)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    snapshot = load_snapshot(arguments.snapshot, manifest_digest(manifest))
    results: dict[str, list[dict[str, float]]] = {"faiss": [], "ppr": [], "walk": [], "colony": []}
    ceiling_reachable = 0
    ceiling_reached = {"ppr": 0, "walk": 0, "colony": 0}
    far_total = 0
    # The candidate-generator question, at equal budget: does the visited set
    # hold more of the relevant documents than FAISS's top-|support|?
    generator_rows: list[dict[str, float]] = []
    reachable_promotions: list[tuple[int, int]] = []
    promotions: dict[str, list[int]] = {"faiss": [], "ppr": [], "walk": [], "colony": []}
    hop_costs: list[int] = []

    for seed in range(arguments.seeds):
        try:
            dataset = build_gate2_dataset(manifest, snapshot, seed)
        except ValueError:
            continue
        query_id = dataset.query_ids[0]
        relevant_ids = dataset.relevant_by_query[query_id]
        if not relevant_ids:
            continue
        index_of = {document_id: position for position, document_id in enumerate(dataset.document_ids)}
        relevant = [index_of[document_id] for document_id in relevant_ids]
        node_count = len(dataset.document_ids)

        edges, _ = build_edges(dataset.document_vectors, KNN_K)
        start_nodes, start_mass = start_distribution(dataset, query_id)
        similarities = dataset.query_vectors[0] @ dataset.document_vectors.T

        faiss_order = np.argsort(-similarities)
        faiss_rank = {node: int(np.where(faiss_order == node)[0][0]) + 1 for node in relevant}
        far = [node for node in relevant if faiss_rank[node] > TOP_K]
        far_total += len(far)
        results["faiss"].append(
            {"recall": sum(1 for n in relevant if faiss_rank[n] <= TOP_K) / len(relevant)}
        )
        promotions["faiss"].extend(faiss_rank[n] for n in far)

        # The per-seed stage-1 ceiling: which far documents BFS proves reachable.
        cache = geometry_cache(dataset, KNN_K)
        reachable = [
            node
            for node in far
            if chain_reachable(
                dataset, query_id, dataset.document_ids[node], KNN_K, CEILING_HOPS, 0.0, cache=cache
            )
        ]
        ceiling_reachable += len(reachable)

        mass_by_variant: dict[str, np.ndarray] = {}
        mass_by_variant["ppr"] = exact_ppr(edges, node_count, start_nodes, start_mass)
        rng = np.random.default_rng(seed)
        walk_visits, _ = colony_walk(
            edges, node_count, start_nodes, start_mass, arguments.agents, arguments.steps, False, rng
        )
        mass_by_variant["walk"] = walk_visits.astype(np.float64)
        rng = np.random.default_rng(seed)
        colony_visits, hops = colony_walk(
            edges, node_count, start_nodes, start_mass, arguments.agents, arguments.steps, True, rng
        )
        mass_by_variant["colony"] = colony_visits.astype(np.float64)
        hop_costs.append(hops)

        for variant, mass in mass_by_variant.items():
            ranks = {node: _rank_of(mass, similarities, node) for node in relevant}
            results[variant].append(
                {"recall": sum(1 for n in relevant if ranks[n] <= TOP_K) / len(relevant)}
            )
            promotions[variant].extend(ranks[n] for n in far)
            support = mass > 0
            ceiling_reached[variant] += sum(1 for node in reachable if support[node])

        support_nodes = np.flatnonzero(mass_by_variant["walk"] > 0)
        support_size = int(len(support_nodes))
        support_set = set(int(n) for n in support_nodes)
        faiss_budget = set(int(n) for n in faiss_order[:support_size])
        generator_rows.append(
            {
                "support": float(support_size),
                "relevant_in_support": float(sum(1 for n in relevant if n in support_set)),
                "relevant_in_faiss_budget": float(sum(1 for n in relevant if n in faiss_budget)),
                "relevant_total": float(len(relevant)),
            }
        )
        support_sim = np.where(mass_by_variant["walk"] > 0, similarities, -np.inf)
        for node in reachable:
            reachable_promotions.append(
                (faiss_rank[node], _rank_of(support_sim, similarities, node))
            )

        for name, score in scoring_variants(
            mass_by_variant["colony"], mass_by_variant["ppr"], similarities
        ).items():
            key = f"score:{name}"
            results.setdefault(key, []).append(
                {
                    "recall": sum(
                        1 for n in relevant if _rank_of(score, similarities, n) <= TOP_K
                    )
                    / len(relevant)
                }
            )
            promotions.setdefault(key, []).extend(
                _rank_of(score, similarities, n) for n in far
            )

    print(f"snapshot : {snapshot.backend} / {snapshot.model} / {snapshot.dimension}d")
    print(f"graph    : mutual kNN k={KNN_K}, alpha={ALPHA}, agents={arguments.agents}, steps={arguments.steps}")
    print(f"ceiling  : {ceiling_reachable} of {far_total} far relevant documents are BFS-reachable "
          f"(knn {KNN_K}, hops {CEILING_HOPS})")
    print(f"cost     : mean {mean(hop_costs):.0f} hops/query "
          f"(G: 147 761 comparisons, C: 57.2M)")
    print(f"\n{'variant':>8}{'recall@8':>10}{'far-doc median rank':>21}{'reached of ceiling':>20}")
    for variant in ("faiss", "ppr", "walk", "colony", "score:mass", "score:ratio", "score:support"):
        rows = results[variant]
        if not rows:
            continue
        rank_column = (
            f"{median(promotions[variant]):>21.0f}" if promotions[variant] else f"{'--':>21}"
        )
        reached_column = (
            f"{ceiling_reached[variant]:>17} /{ceiling_reachable:>2}"
            if variant in ceiling_reached
            else f"{'--':>20}"
        )
        print(f"{variant:>8}{mean(r['recall'] for r in rows):>10.3f}{rank_column}{reached_column}")

    print()
    print("=== candidate generator, equal budget ===")
    if generator_rows:
        support = mean(r["support"] for r in generator_rows)
        in_support = mean(r["relevant_in_support"] for r in generator_rows)
        in_faiss = mean(r["relevant_in_faiss_budget"] for r in generator_rows)
        total = mean(r["relevant_total"] for r in generator_rows)
        print(f"  mean support size                {support:.0f} of {node_count}")
        print(f"  relevant in walk support         {in_support:.2f} of {total:.2f}")
        print(f"  relevant in FAISS top-|support|  {in_faiss:.2f} of {total:.2f}")
    print()
    print("=== reachable far documents: FAISS rank -> support-similarity rank ===")
    for faiss_position, support_position in sorted(reachable_promotions):
        direction = "better" if support_position < faiss_position else ("worse" if support_position > faiss_position else "same")
        print(f"    {faiss_position:>5} -> {support_position:>5}   {direction}")


if __name__ == "__main__":
    main()
