from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

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
    assert payload["execution"]["clock_mode"] == "fixed"
    assert payload["execution"]["clock_value"] == 2.0
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
        "clock_mode": "fixed",
        "clock_value": 2.0,
        "random_seed_provenance": {
            "python_random_seed": None,
            "numpy_random_seed": None,
            "per_query": None,
        },
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
        "clock_mode": "fixed",
        "clock_value": 2.0,
        "random_seed_provenance": {
            "python_random_seed": 7,
            "numpy_random_seed": 7,
            "per_query": {
                "q-main": {"python_random_seed": 7, "numpy_random_seed": 7},
                "q-related": {"python_random_seed": 8, "numpy_random_seed": 8},
            },
        },
    }


def test_gate1_content_is_deterministic_except_elapsed_timing() -> None:
    payloads = [
        run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)
        for _ in range(8)
    ]
    for payload in payloads:
        for run in payload["runs"].values():
            del run["timing"]["elapsed_ms"]

    assert payloads[1:] == [payloads[0]] * 7


def test_gate1_deterministic_path_does_not_read_wall_clock(monkeypatch) -> None:
    from embeddinggemma.mcmp import simulation

    def forbidden_wall_clock() -> float:
        raise AssertionError("deterministic benchmark must not read wall clock")

    monkeypatch.setattr(
        simulation, "time", SimpleNamespace(time=forbidden_wall_clock)
    )

    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)

    assert payload["execution"]["clock_mode"] == "fixed"


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
        "clock_value",
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
    elif mutation == "clock_value":
        payload["runs"]["C"]["execution"]["clock_value"] = 3.0
    else:
        payload["execution"]["force_cpu"] = 1
    output_path = tmp_path / mutation / "gate1.json"

    with pytest.raises(ValueError):
        write_gate1_result(payload, output_path)

    assert not output_path.exists()


def test_gate1_persists_actual_per_query_initial_candidates() -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10)

    assert payload["runs"]["C"]["raw_ids"]["per_query_initial_candidate_ids"] == {
        "q-main": ["main-top"],
    }
    assert payload["runs"]["D"]["raw_ids"]["per_query_initial_candidate_ids"] == {
        "q-main": ["main-top"],
        "q-related": ["related-top"],
    }


def test_gate1_preserves_score_ranked_initial_candidates_when_initial_k_is_two() -> None:
    payload = run_gate1(seed=7, top_k=4, initial_k=2, num_agents=24, steps=10)

    assert payload["runs"]["D"]["raw_ids"]["per_query_initial_candidate_ids"] == {
        "q-main": ["main-top", "main-near"],
        "q-related": ["related-top", "related-near"],
    }


def test_cli_persists_score_ranked_initial_candidates_when_initial_k_is_two(tmp_path) -> None:
    output_path = tmp_path / "gate1.json"

    assert gate1_module.main([
        "--seed", "7", "--top-k", "4", "--initial-k", "2", "--num-agents", "24", "--steps", "10", "--output", str(output_path),
    ]) == 0

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["runs"]["D"]["raw_ids"]["per_query_initial_candidate_ids"] == {
        "q-main": ["main-top", "main-near"],
        "q-related": ["related-top", "related-near"],
    }


@pytest.mark.parametrize("mutation", ["initial_k", "top_k", "forged_c_initials", "comparison_delta"])
def test_writer_rejects_self_consistent_raw_binding_forgery(mutation, tmp_path) -> None:
    payload = deepcopy(run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10))
    if mutation == "initial_k":
        payload["config"]["initial_k"] = 2
        for run in payload["runs"].values():
            run["execution"]["initial_k"] = 2
    elif mutation == "top_k":
        payload["config"]["top_k"] = 5
        for run in payload["runs"].values():
            run["execution"]["top_k"] = 5
    elif mutation == "forged_c_initials":
        raw_ids = payload["runs"]["C"]["raw_ids"]
        raw_ids["per_query_initial_candidate_ids"]["q-main"] = ["opposite"]
        raw_ids["initial_candidate_ids"] = ["opposite"]
    else:
        payload["comparisons"]["A_vs_C"]["mrr_delta"] += 5e-16
    output_path = tmp_path / mutation / "gate1.json"

    with pytest.raises(ValueError):
        write_gate1_result(payload, output_path)

    assert not output_path.exists()


def test_writer_rejects_reordered_score_ranked_initial_evidence(tmp_path) -> None:
    payload = deepcopy(run_gate1(seed=7, top_k=4, initial_k=2, num_agents=24, steps=10))
    payload["runs"]["D"]["raw_ids"]["per_query_initial_candidate_ids"]["q-main"].reverse()
    output_path = tmp_path / "reordered" / "gate1.json"

    with pytest.raises(ValueError, match="direct retrieval"):
        write_gate1_result(payload, output_path)

    assert not output_path.exists()


def test_writer_rejects_forged_python_or_numpy_seed_provenance(tmp_path) -> None:
    payload = deepcopy(run_gate1(seed=7, top_k=4, initial_k=1, num_agents=24, steps=10))
    payload["runs"]["D"]["execution"]["random_seed_provenance"]["per_query"][
        "q-related"
    ]["python_random_seed"] = 99
    output_path = tmp_path / "forged-seed-provenance.json"

    with pytest.raises(ValueError, match="execution snapshot"):
        write_gate1_result(payload, output_path)

    assert not output_path.exists()
