from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import logging


_logger = logging.getLogger("MCMP.Visualize")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)


def build_snapshot(docs_xy: np.ndarray,
                   relevances: List[float],
                   pheromone_trails: Dict[tuple, float],
                   max_edges: int = 300) -> Dict[str, Any]:
    _logger.debug("build_snapshot: docs=%d trails=%d", int(len(docs_xy)), int(len(pheromone_trails or {})))
    edges = []
    if pheromone_trails is not None and len(docs_xy):
        dim = int(docs_xy.shape[1]) if hasattr(docs_xy, 'shape') and len(docs_xy.shape) == 2 else 2
        for (a, b), strength in sorted(pheromone_trails.items(), key=lambda x: x[1], reverse=True):
            if strength < 0.05:
                continue
            if a < len(docs_xy) and b < len(docs_xy):
                if dim >= 3:
                    x0, y0, z0 = docs_xy[a]
                    x1, y1, z1 = docs_xy[b]
                    edges.append({
                        "x0": float(x0), "y0": float(y0), "z0": float(z0),
                        "x1": float(x1), "y1": float(y1), "z1": float(z1),
                        "s": float(strength)
                    })
                else:
                    x0, y0 = docs_xy[a]
                    x1, y1 = docs_xy[b]
                    edges.append({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "s": float(strength)})
            if len(edges) >= max_edges:
                break
    return {
        "documents": {"xy": docs_xy.tolist(), "relevance": [float(r) for r in relevances]},
        "agents": {"xy": []},
        "edges": edges,
    }


