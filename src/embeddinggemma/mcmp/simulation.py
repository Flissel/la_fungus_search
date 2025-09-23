from __future__ import annotations
from typing import Any, List, Tuple
import numpy as np
import time
import logging


_logger = logging.getLogger("MCMP.Sim")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


def spawn_agents(retr: Any, query_embedding: np.ndarray) -> None:
    _logger.info("spawn_agents: start | num_agents=%d", getattr(retr, 'num_agents', -1))
    retr.agents = []
    for i in range(retr.num_agents):
        noise = np.random.normal(0, 0.1, query_embedding.shape)
        start_pos = query_embedding + noise
        start_pos = start_pos / np.linalg.norm(start_pos)
        velocity = np.random.normal(0, 0.05, query_embedding.shape)
        agent = retr.Agent(  # type: ignore[attr-defined]
            id=i,
            position=start_pos,
            velocity=velocity,
            exploration_factor=np.random.uniform(0.05, max(0.05, float(retr.exploration_bonus))),
        )
        retr.agents.append(agent)
    _logger.info("spawn_agents: done | agents=%d", len(retr.agents))


def calculate_pheromone_force(retr: Any, agent: Any) -> np.ndarray:
    _logger.debug("calculate_pheromone_force: agent_id=%s trails=%d", getattr(agent, 'id', '?'), len(getattr(retr, 'pheromone_trails', {})))
    if not retr.pheromone_trails:
        return np.zeros_like(agent.position)
    force = np.zeros_like(agent.position)
    current_docs = retr.find_nearest_documents(agent.position, k=1)
    if not current_docs:
        return force
    current_doc = current_docs[0][0]
    max_strength = 0.0
    best_direction = None
    for (doc_a, doc_b), strength in retr.pheromone_trails.items():
        if doc_a == current_doc.id:
            target_doc = next((d for d in retr.documents if d.id == doc_b), None)
            if target_doc is not None:
                direction = target_doc.embedding - agent.position
                if strength > max_strength:
                    max_strength = float(strength)
                    best_direction = direction
    if best_direction is not None and np.linalg.norm(best_direction) > 0:
        force = best_direction / np.linalg.norm(best_direction) * max_strength
    _logger.debug("calculate_pheromone_force: max_strength=%.4f", max_strength)
    return force


def update_agent_position(retr: Any, agent: Any, iteration: int) -> None:
    _logger.debug("update_agent_position: agent_id=%s iter=%d", getattr(agent, 'id', '?'), iteration)
    nearby_docs = retr.find_nearest_documents(agent.position, k=5)
    if not nearby_docs:
        return
    # Sphere-tangent attraction toward normalized doc embeddings
    p = agent.position / (np.linalg.norm(agent.position) + 1e-12)
    attraction_force = np.zeros_like(p)
    for doc, similarity in nearby_docs:
        nd = doc.embedding / (np.linalg.norm(doc.embedding) + 1e-12)
        tangential = nd - (p @ nd) * p  # remove radial component
        norm_t = np.linalg.norm(tangential)
        if norm_t > 0:
            tangential = tangential / norm_t
            weight = float(similarity) * (1.0 + float(doc.relevance_score))
            attraction_force += weight * tangential
    pheromone_force = calculate_pheromone_force(retr, agent)
    exploration_force = np.random.normal(0, agent.exploration_factor, agent.position.shape)
    total_force = 0.8 * attraction_force + 0.15 * pheromone_force + 0.05 * exploration_force
    agent.velocity = 0.85 * agent.velocity + 0.15 * total_force
    agent.position = p + agent.velocity
    agent.position = agent.position / (np.linalg.norm(agent.position) + 1e-12)
    agent.age += 1
    if iteration % max(1, int(getattr(retr, 'log_every', 10))) == 0:
        _logger.debug(
            "update_agent_position: speed=%.4f att=%.4f pher=%.4f expl=%.4f energy=%.3f",
            float(np.linalg.norm(agent.velocity)),
            float(np.linalg.norm(attraction_force)),
            float(np.linalg.norm(pheromone_force)),
            float(np.linalg.norm(exploration_force)),
            float(getattr(agent, 'energy', 0.0)),
        )


def deposit_pheromones(retr: Any, agent: Any) -> None:
    current_docs = retr.find_nearest_documents(agent.position, k=1)
    if not current_docs:
        return
    current_doc = current_docs[0][0]
    current_doc.visit_count += 1
    current_doc.last_visited = time.time()
    agent.visited_docs.add(current_doc.id)
    for prev_doc_id in list(agent.visited_docs)[-3:]:
        if prev_doc_id != current_doc.id:
            trail_key = tuple(sorted([current_doc.id, prev_doc_id]))
            current_strength = retr.pheromone_trails.get(trail_key, 0.0)
            amount = agent.energy * agent.trail_strength * 0.1
            retr.pheromone_trails[trail_key] = current_strength + amount
    _logger.debug(
        "deposit_pheromones: doc_id=%s visit_count=%d total_trails=%d",
        getattr(current_doc, 'id', '?'),
        int(current_doc.visit_count),
        len(retr.pheromone_trails),
    )


def decay_pheromones(retr: Any) -> None:
    keys_to_remove: List[Tuple[int, int]] = []
    for key in retr.pheromone_trails:
        retr.pheromone_trails[key] *= retr.pheromone_decay
        if retr.pheromone_trails[key] < 0.01:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del retr.pheromone_trails[key]
    if keys_to_remove:
        _logger.debug("decay_pheromones: pruned=%d remaining=%d", len(keys_to_remove), len(retr.pheromone_trails))


def update_document_relevance(retr: Any, query_embedding: np.ndarray) -> None:
    _logger.debug("update_document_relevance: start | docs=%d kw_lambda=%.3f",
                  len(getattr(retr, 'documents', [])), float(getattr(retr, 'kw_lambda', 0.0)))
    try:
        import torch  # type: ignore
        _TORCH_OK = True
    except Exception:
        torch = None  # type: ignore
        _TORCH_OK = False
    if _TORCH_OK and torch.cuda.is_available():
        with torch.no_grad():
            if retr._doc_emb_torch_norm is None:
                if retr._doc_emb_torch is None:
                    mat = torch.tensor(np.array([doc.embedding for doc in retr.documents], dtype=np.float32), device='cuda')
                    mat = mat.to(torch.float16)
                    retr._doc_emb_torch = mat
                retr._doc_emb_torch_norm = torch.nn.functional.normalize(retr._doc_emb_torch, p=2, dim=1)
            q = torch.tensor(query_embedding.astype(np.float32), device='cuda').to(torch.float16)
            qn = torch.nn.functional.normalize(q, p=2, dim=0)
            sims = (retr._doc_emb_torch_norm @ qn).float().cpu().numpy()
    else:
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        doc_embeddings = np.array([doc.embedding for doc in retr.documents])
        sims = cosine_similarity([query_embedding], doc_embeddings)[0]
    max_rel = 0.0
    for i, doc in enumerate(retr.documents):
        query_sim = float(sims[i])
        visit_bonus = min(doc.visit_count * 0.1, 0.5)
        time_bonus = 0.1 if time.time() - doc.last_visited < 1.0 else 0.0
        kw_bonus = 0.0
        if retr.kw_lambda and retr.kw_terms and getattr(doc, 'content', None):
            try:
                text = (doc.content or '').lower()
                if text:
                    hits = sum(1 for t in retr.kw_terms if t and t in text)
                    if hits:
                        kw_bonus = retr.kw_lambda * float(hits) / float(max(1, len(retr.kw_terms)))
            except Exception:
                kw_bonus = 0.0
        doc.relevance_score = query_sim + visit_bonus + time_bonus + kw_bonus
        if doc.relevance_score > max_rel:
            max_rel = float(doc.relevance_score)
    if retr.documents:
        mean_rel = float(np.mean([d.relevance_score for d in retr.documents]))
        top_rel = float(np.max([d.relevance_score for d in retr.documents]))
        _logger.debug("update_document_relevance: mean=%.4f max=%.4f top=%.4f", mean_rel, max_rel, top_rel)


def extract_relevance_network(retr: Any, min_strength: float = 0.1):
    # Sort documents by relevance
    relevant_docs = sorted(retr.documents, key=lambda d: d.relevance_score, reverse=True)
    retr.relevance_network.clear()
    for doc in relevant_docs:
        if doc.relevance_score > 0.1:
            retr.relevance_network.add_node(doc.id, content=(doc.content[:100] + "..."), relevance=doc.relevance_score, visits=doc.visit_count)
    edge_count = 0
    for (doc_a, doc_b), strength in retr.pheromone_trails.items():
        if strength > min_strength and doc_a in retr.relevance_network.nodes and doc_b in retr.relevance_network.nodes:
            retr.relevance_network.add_edge(doc_a, doc_b, strength=strength)
            edge_count += 1
    _logger.debug("extract_relevance_network: nodes=%d edges=%d", len(retr.relevance_network.nodes), edge_count)
    return [(doc, doc.relevance_score) for doc in relevant_docs if doc.relevance_score > 0.1]



def log_simulation_step(retr: Any, step_idx: int) -> None:
    """Log aggregated step-level metrics for diagnostics/telemetry."""
    try:
        num_docs = len(getattr(retr, 'documents', []))
        num_agents = len(getattr(retr, 'agents', []))
        avg_rel = float(np.mean([d.relevance_score for d in retr.documents])) if num_docs else 0.0
        max_rel = float(np.max([d.relevance_score for d in retr.documents])) if num_docs else 0.0
        trails = len(getattr(retr, 'pheromone_trails', {}))
        avg_speed = float(np.mean([np.linalg.norm(getattr(a, 'velocity', 0.0)) for a in retr.agents])) if num_agents else 0.0
        _logger.info(
            "step=%d docs=%d agents=%d avg_rel=%.4f max_rel=%.4f trails=%d avg_speed=%.4f",
            int(step_idx), num_docs, num_agents, avg_rel, max_rel, trails, avg_speed,
        )
    except Exception as e:
        _logger.debug("log_simulation_step failed: %s", e)

