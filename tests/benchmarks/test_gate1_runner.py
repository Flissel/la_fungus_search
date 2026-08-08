from __future__ import annotations

import json

from benchmarks.mcmp.run_gate1 import run_gate1, write_gate1_result


def test_gate1_runner_orchestrates_fixed_ablation_and_round_trips_evidence(tmp_path) -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)

    assert list(payload["runs"]) == ["A", "B", "C", "D"]
    assert payload["config"] == {
        "seed": 7,
        "top_k": 4,
        "initial_k": 1,
        "num_agents": 24,
        "steps": 10,
    }
    assert payload["conclusion"] in {
        "novel_relevant_observed",
        "no_novel_relevant_observed",
    }
    assert payload["runs"]["D"]["independent_run_count"] == 2

    output_path = tmp_path / "gate1.json"
    write_gate1_result(payload, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
