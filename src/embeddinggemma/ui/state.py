from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Settings:
    mode: str = "deep"
    top_k: int = 5
    windows: List[int] = field(default_factory=lambda: [50, 100, 200, 300, 400])
    use_repo: bool = True
    root_folder: str = ""
    max_files: int = 1000
    exclude_dirs: List[str] = field(default_factory=lambda: [".venv", "node_modules", ".git", "external"]) 
    docs_file: str = ""
    num_agents: int = 200
    max_iterations: int = 60
    show_tree: bool = True
    show_network: bool = True
    gen_answer: bool = False
    div_alpha: float = 0.7
    dedup_tau: float = 0.92
    per_folder_cap: int = 2
    pure_topk: bool = False
    log_every: int = 10
    exploration_bonus: float = 0.1
    pheromone_decay: float = 0.95
    embed_bs: int = 64
    max_chunks_per_shard: int = 2000

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}  # type: ignore


SESSION_KEYS = {
    "generated_multi_queries": [],
    "report_job_id": None,
    "report_future": None,
    "report_started_at": None,
    "root_folder_agent_override": None,
    "last_snapshot": None,
}


def init_session(session):
    for k, v in SESSION_KEYS.items():
        if k not in session:
            session[k] = v
