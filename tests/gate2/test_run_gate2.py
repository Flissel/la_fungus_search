from __future__ import annotations

import json
from pathlib import Path

from benchmarks.gate2.manifest import build_manifest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.run_gate2 import AGENT_SWEEP, run_retrieval, write_result
from benchmarks.gate2.snapshot import build_stub_snapshot


def _dataset(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"def f{index}():\n    return f{index + 1}()\n\n" for index in range(12)
    )
    (root / "alpha.py").write_text(body + "def f12():\n    return 0\n", encoding="utf-8")
    manifest = build_manifest(root, commit_sha="sha", manifest_id="m1")
    return build_gate2_dataset(manifest, build_stub_snapshot(manifest, dimension=16), seed=1)


def test_agent_sweep_covers_the_gate_one_range() -> None:
    assert AGENT_SWEEP == (24, 48, 96, 192, 384)


def test_run_reports_all_five_methods(tmp_path: Path) -> None:
    payload = run_retrieval(
        _dataset(tmp_path), top_k=4, initial_k=4, seed=1, num_agents=4, steps=2
    )

    assert list(payload["runs"]) == ["A", "B", "C", "D", "E"]


def test_discovery_and_ranking_are_reported_separately(tmp_path: Path) -> None:
    payload = run_retrieval(
        _dataset(tmp_path), top_k=4, initial_k=4, seed=1, num_agents=4, steps=2
    )

    for run in payload["runs"].values():
        assert "discovered_relevant" in run
        assert "ranked_relevant" in run
        assert isinstance(run["discovered_relevant"], int)
        assert isinstance(run["ranked_relevant"], int)
        assert 0 <= run["discovered_relevant"] <= run["relevant_total"]
        assert 0 <= run["ranked_relevant"] <= run["relevant_total"]


def test_result_round_trips(tmp_path: Path) -> None:
    payload = run_retrieval(
        _dataset(tmp_path), top_k=4, initial_k=4, seed=1, num_agents=4, steps=2
    )
    path = tmp_path / "out" / "retrieval.json"

    write_result(payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
