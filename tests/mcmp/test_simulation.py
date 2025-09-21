import time
import types
import sys
import numpy as np


class DummyDoc:
    def __init__(self, id, embedding, content=""):
        self.id = id
        self.embedding = np.asarray(embedding, dtype=float)
        self.relevance_score = 0.0
        self.visit_count = 0
        self.last_visited = 0.0
        self.content = content


class DummyAgent:
    def __init__(self, id, position, velocity, exploration_factor):
        self.id = id
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.exploration_factor = float(exploration_factor)
        self.trail_strength = 1.0
        self.energy = 1.0
        self.age = 0
        self.visited_docs = set()


class DummyRetriever:
    def __init__(self):
        self.num_agents = 3
        self.exploration_bonus = 0.1
        self.documents = [
            DummyDoc(0, [1.0, 0.0, 0.0], content="hello world"),
            DummyDoc(1, [0.0, 1.0, 0.0], content="foo bar"),
            DummyDoc(2, [0.0, 0.0, 1.0], content="baz qux"),
        ]
        self.pheromone_trails = {}
        self.kw_lambda = 0.0
        self.kw_terms = []
        self._doc_emb_torch = None
        self._doc_emb_torch_norm = None
        self.log_every = 1
        self.pheromone_decay = 0.9
        self.relevance_network = types.SimpleNamespace(
            clear=lambda: None,
            add_node=lambda *args, **kwargs: None,
            add_edge=lambda *args, **kwargs: None,
            nodes={},
        )
        self.Agent = DummyAgent

    def find_nearest_documents(self, position, k=1):
        # Return list of (doc, similarity) using cosine
        position = np.asarray(position, dtype=float)
        sims = []
        for d in self.documents:
            sim = float(np.dot(position, d.embedding) / ((np.linalg.norm(position) + 1e-12) * (np.linalg.norm(d.embedding) + 1e-12)))
            sims.append((d, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]


def test_spawn_agents_and_deposit_pheromones():
    from embeddinggemma.mcmp import simulation as sim

    retr = DummyRetriever()
    query = np.array([1.0, 0.0, 0.0], dtype=float)

    # Should create agents without requiring retriever to define class attributes beyond Agent
    sim.spawn_agents(retr, query)
    assert len(retr.agents) == retr.num_agents

    # Run a few steps: update, deposit, decay
    for step in range(3):
        for agent in retr.agents:
            sim.update_agent_position(retr, agent, iteration=step)
            sim.deposit_pheromones(retr, agent)
        sim.decay_pheromones(retr)

    # Trails should accumulate and decay but remain non-negative
    assert isinstance(retr.pheromone_trails, dict)
    for key, val in retr.pheromone_trails.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert val >= 0.0


def test_update_document_relevance_cpu_path(monkeypatch):
    from embeddinggemma.mcmp import simulation as sim

    retr = DummyRetriever()
    query = np.array([1.0, 0.0, 0.0], dtype=float)

    # Force CPU path: if torch exists, make cuda unavailable; otherwise CPU path is used by default
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore
    if torch is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sim.update_document_relevance(retr, query)
    # Nearest doc aligned with query should have highest score
    scores = [d.relevance_score for d in retr.documents]
    assert scores[0] == max(scores)


