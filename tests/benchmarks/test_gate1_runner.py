from __future__ import annotations

from copy import deepcopy
import json

import pytest

from benchmarks.mcmp import run_gate1 as gate1_module
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


def test_gate1_payload_records_forced_cpu_and_reproducibility_settings() -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)

    assert payload["execution"]["pheromone_decay"] == 0.95
    assert payload["execution"]["exploration_bonus"] == 0.1
    assert payload["environment"]["execution_backend"] == "faiss-cpu"
    assert payload["environment"]["faiss_factory"] == "Flat"
    assert payload["environment"]["faiss_metric"] == "inner_product"
    assert payload["environment"]["force_cpu"] is True
    assert {run["execution_backend"] for run in payload["runs"].values()} == {"faiss-cpu"}
    assert payload["runs"]["A"]["execution"] == {
        "mode": "faiss",
        "seed": 7,
        "top_k": 4,
        "initial_k": 1,
        "force_cpu": True,
        "faiss_factory": "Flat",
        "faiss_metric": "inner_product",
        "execution_backend": "faiss-cpu",
        "num_agents": None,
        "steps": None,
        "pheromone_decay": None,
        "exploration_bonus": None,
    }
    assert payload["runs"]["D"]["execution"] == {
        "mode": "mcmp",
        "seed": 7,
        "top_k": 4,
        "initial_k": 1,
        "force_cpu": True,
        "faiss_factory": "Flat",
        "faiss_metric": "inner_product",
        "execution_backend": "faiss-cpu",
        "num_agents": 24,
        "steps": 10,
        "pheromone_decay": 0.95,
        "exploration_bonus": 0.1,
    }


def test_gate1_content_is_deterministic_except_elapsed_timing() -> None:
    first = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)
    second = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)
    for payload in (first, second):
        for run in payload["runs"].values():
            del run["timing"]["elapsed_ms"]

    assert first == second


def test_writer_rejects_incomplete_or_inconsistent_evidence_before_creating_nested_path(tmp_path) -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)
    payload["conclusion"] = "no_novel_relevant_observed"
    output_path = tmp_path / "nested" / "gate1.json"

    with pytest.raises(ValueError, match="conclusion"):
        write_gate1_result(payload, output_path)

    assert not output_path.exists()


def test_writer_rejects_tampered_nested_metric_before_persisting(tmp_path) -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)
    payload["runs"]["C"]["metrics"]["mrr"] = 0.0
    output_path = tmp_path / "nested" / "gate1.json"

    with pytest.raises(ValueError, match="metrics"):
        write_gate1_result(payload, output_path)

    assert not output_path.exists()


def test_writer_rejects_inconsistent_search_counts_before_persisting(tmp_path) -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)
    payload["runs"]["D"]["candidate_comparisons"] = 0
    output_path = tmp_path / "nested" / "gate1.json"

    with pytest.raises(ValueError, match="candidate comparisons"):
        write_gate1_result(payload, output_path)

    assert not output_path.exists()


def test_writer_persists_full_schema_with_sorted_format_and_final_newline(tmp_path) -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)
    output_path = tmp_path / "nested" / "gate1.json"

    write_gate1_result(payload, output_path)

    text = output_path.read_text(encoding="utf-8")
    reloaded = json.loads(text)
    assert text.endswith("\n")
    assert text.startswith('{\n  "comparisons"')
    assert reloaded == payload
    assert list(reloaded["runs"]) == ["A", "B", "C", "D"]
    assert reloaded["runs"]["D"]["independent_run_count"] == 2
    assert reloaded["comparisons"].keys() == {"A_vs_C", "B_vs_D"}
    for run in reloaded["runs"].values():
        assert {"raw_ids", "metrics", "timing", "candidate_comparisons", "nearest_search_calls", "document_visits", "pheromone_trails"} <= run.keys()


def test_cli_maps_invalid_evidence_to_nonzero_without_writing(monkeypatch, tmp_path) -> None:
    invalid_payload = {"runs": {}}
    output_path = tmp_path / "gate1.json"
    monkeypatch.setattr(gate1_module, "run_gate1", lambda *args: invalid_payload)

    with pytest.raises(SystemExit) as error:
        gate1_module.main([
            "--seed", "7", "--top-k", "4", "--initial-k", "1", "--num-agents", "24", "--steps", "10", "--output", str(output_path),
        ])

    assert error.value.code == 2
    assert not output_path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "initial_k",
        "num_agents",
        "visit_count",
        "pheromone_trails",
        "independent_run_count_bool",
        "force_cpu_int",
    ],
)
def test_writer_rejects_forged_configuration_and_execution_evidence(mutation, tmp_path) -> None:
    payload = deepcopy(run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10))
    if mutation == "initial_k":
        payload["config"]["initial_k"] = 2
    elif mutation == "num_agents":
        payload["config"]["num_agents"] = 25
    elif mutation == "visit_count":
        payload["runs"]["C"]["document_visits"]["main-bridge"] = 0
    elif mutation == "pheromone_trails":
        payload["runs"]["C"]["pheromone_trails"] = 999
    elif mutation == "independent_run_count_bool":
        payload["runs"]["A"]["independent_run_count"] = True
    else:
        payload["execution"]["force_cpu"] = 1
    output_path = tmp_path / mutation / "gate1.json"

    with pytest.raises(ValueError):
        write_gate1_result(payload, output_path)

    assert not output_path.exists()
