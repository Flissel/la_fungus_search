import numpy as np


def test_build_snapshot_edges_and_limits():
    from embeddinggemma.mcmp import visualize

    docs_xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    relevances = [0.1, 0.5, 0.9]
    trails = {(0, 1): 0.2, (1, 2): 0.3, (0, 2): 0.04}  # one below threshold
    snap = visualize.build_snapshot(docs_xy, relevances, trails, max_edges=1)

    assert "documents" in snap and "agents" in snap and "edges" in snap
    assert len(snap["documents"]["xy"]) == 3
    assert len(snap["edges"]) == 1  # max_edges enforced
    assert all(k in snap["edges"][0] for k in ("x0", "y0", "x1", "y1", "s"))


