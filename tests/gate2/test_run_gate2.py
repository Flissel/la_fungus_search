from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.gate2 import run_gate2 as run_gate2_module
from benchmarks.gate2.manifest import build_manifest, save_manifest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.run_gate2 import (
    AGENT_SWEEP,
    characterise_pooled,
    run_retrieval,
    write_result,
)
from benchmarks.gate2.snapshot import build_stub_snapshot, save_snapshot


def _build_manifest(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"def f{index}():\n    return f{index + 1}()\n\n" for index in range(12)
    )
    (root / "alpha.py").write_text(body + "def f12():\n    return 0\n", encoding="utf-8")
    return build_manifest(root, commit_sha="sha", manifest_id="m1")


def _dataset_and_snapshot(tmp_path: Path):
    manifest = _build_manifest(tmp_path)
    snapshot = build_stub_snapshot(manifest, dimension=16)
    return build_gate2_dataset(manifest, snapshot, seed=1), snapshot


def _write_manifest_and_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    """Build the same corpus as ``_dataset_and_snapshot`` and persist it for main()."""
    manifest = _build_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    snapshot = build_stub_snapshot(manifest, dimension=16)
    snapshot_path = tmp_path / "snapshot.npz"
    save_snapshot(snapshot, snapshot_path)
    return manifest_path, snapshot_path


def test_agent_sweep_covers_the_gate_one_range() -> None:
    assert AGENT_SWEEP == (24, 48, 96, 192, 384)


def test_run_reports_all_five_methods(tmp_path: Path) -> None:
    dataset, snapshot = _dataset_and_snapshot(tmp_path)

    payload = run_retrieval(
        dataset, snapshot, top_k=4, initial_k=4, seed=1, num_agents=4, steps=2
    )

    assert list(payload["runs"]) == ["A", "B", "C", "D", "E"]


def test_discovery_and_ranking_are_reported_separately(tmp_path: Path) -> None:
    dataset, snapshot = _dataset_and_snapshot(tmp_path)

    payload = run_retrieval(
        dataset, snapshot, top_k=4, initial_k=4, seed=1, num_agents=4, steps=2
    )

    for run in payload["runs"].values():
        assert "discovered_relevant" in run
        assert "ranked_relevant" in run
        assert isinstance(run["discovered_relevant"], int)
        assert isinstance(run["ranked_relevant"], int)
        assert 0 <= run["discovered_relevant"] <= run["relevant_total"]
        assert 0 <= run["ranked_relevant"] <= run["relevant_total"]


def test_retrieval_payload_records_snapshot_provenance(tmp_path: Path) -> None:
    """A stub-derived evidence file must not be mistakable for a production one."""
    dataset, snapshot = _dataset_and_snapshot(tmp_path)

    payload = run_retrieval(
        dataset, snapshot, top_k=4, initial_k=4, seed=1, num_agents=4, steps=2
    )

    assert payload["snapshot_backend"] == "stub"
    assert payload["snapshot_model"] == "sha256-gaussian"
    assert payload["manifest_digest"] == snapshot.manifest_digest


def test_result_round_trips(tmp_path: Path) -> None:
    dataset, snapshot = _dataset_and_snapshot(tmp_path)
    payload = run_retrieval(
        dataset, snapshot, top_k=4, initial_k=4, seed=1, num_agents=4, steps=2
    )
    path = tmp_path / "out" / "retrieval.json"

    write_result(payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_stage_one_pools_pairs_across_seeds(tmp_path: Path) -> None:
    """One dataset is two queries, so its pair population is 2-4 -- too coarse
    for a 10% threshold to discriminate. The pooled population is the sum over
    every seed that produced a dataset, and it is strictly larger than any
    single seed's.
    """
    manifest = _build_manifest(tmp_path)
    snapshot = build_stub_snapshot(manifest, dimension=16)

    report = characterise_pooled(
        manifest, snapshot, stage1_seeds=12, top_k=8, knn_k=8, max_hops=6, hop_threshold=0.0
    )

    per_seed_counts = [entry["pair_count"] for entry in report["per_seed"]]
    assert report["config"]["stage1_seeds"] == 12
    assert len(report["per_seed"]) + len(report["skipped_seeds"]) == 12
    assert max(per_seed_counts) <= 4
    assert report["pair_count"] == sum(per_seed_counts)
    assert report["pair_count"] > max(per_seed_counts)
    assert report["manifold_signature"] == pytest.approx(
        report["far_and_reachable_count"] / report["pair_count"]
    )
    assert {pair["seed"] for pair in report["pairs"]} == {
        entry["seed"] for entry in report["per_seed"]
    }
    assert report["snapshot_backend"] == "stub"
    assert report["snapshot_model"] == "sha256-gaussian"
    assert report["manifest_digest"] == snapshot.manifest_digest


def test_stage_one_fails_closed_when_every_seed_skips(tmp_path: Path) -> None:
    """helper and caller are each other's only neighbour, so every seed's pair
    leaves both queries an empty relevant set and no dataset can be built.
    """
    root = tmp_path / "tiny"
    root.mkdir(parents=True, exist_ok=True)
    (root / "alpha.py").write_text(
        "def helper():\n    return 1\n\ndef caller():\n    return helper()\n",
        encoding="utf-8",
    )
    manifest = build_manifest(root, commit_sha="sha", manifest_id="m1")
    snapshot = build_stub_snapshot(manifest, dimension=16)

    with pytest.raises(ValueError, match="every stage 1 seed") as excinfo:
        characterise_pooled(
            manifest, snapshot, stage1_seeds=4, top_k=8, knn_k=8, max_hops=6, hop_threshold=0.0
        )

    # The underlying cause is named, not swallowed by the summary.
    assert "non-empty relevant set" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_stage_one_rejects_a_seed_count_below_one(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    snapshot = build_stub_snapshot(manifest, dimension=16)

    with pytest.raises(ValueError, match="stage1-seeds must be at least 1"):
        characterise_pooled(
            manifest, snapshot, stage1_seeds=0, top_k=8, knn_k=8, max_hops=6, hop_threshold=0.0
        )


def test_main_records_all_four_geometry_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The design fixes all four as CLI parameters recorded in the output.

    Pinning three of them in source is the mistake this research already made
    three times: hold a parameter fixed, then generalise over it.
    """
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        run_gate2_module, "stage_two_is_justified", lambda signature, null: False
    )

    with pytest.raises(SystemExit) as excinfo:
        run_gate2_module.main(
            [
                "--manifest", str(manifest_path),
                "--snapshot", str(snapshot_path),
                "--seed", "1",
                "--top-k", "3",
                "--knn-k", "5",
                "--max-hops", "2",
                "--hop-threshold", "0.25",
                "--stage1-seeds", "4",
                "--null-permutations", "5",
                "--exploratory",
                "--steps", "2",
                "--output-dir", str(output_dir),
            ]
        )

    assert excinfo.value.code == 0
    payload = json.loads(
        (output_dir / "geometry-m1.json").read_text(encoding="utf-8")
    )
    assert payload["config"] == {
        "top_k": 3,
        "knn_k": 5,
        "max_hops": 2,
        "hop_threshold": 0.25,
        "stage1_seeds": 4,
        "null_permutations": 5,
        "null_seed": 0,
        "exploratory": True,
    }


def test_main_skips_stage_two_when_the_gate_is_not_met(tmp_path: Path) -> None:
    """No pairwise similarity in this stub geometry reaches 0.9, so a hop
    threshold of 0.9 prunes every hop and no far pair is chain-reachable.
    Signature and null are both a real, unmocked 0.0, and `0.0 > 0.0` is False,
    so the significance condition closes the gate. This is the one place the
    degenerate-null path runs end to end through the CLI.
    """
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)
    output_dir = tmp_path / "out"

    with pytest.raises(SystemExit) as excinfo:
        run_gate2_module.main(
            [
                "--manifest", str(manifest_path),
                "--snapshot", str(snapshot_path),
                "--seed", "6",
                "--hop-threshold", "0.9",
                "--stage1-seeds", "4",
                "--null-permutations", "5",
                "--exploratory",
                "--steps", "2",
                "--output-dir", str(output_dir),
            ]
        )

    assert excinfo.value.code == 0
    payload = json.loads(
        (output_dir / "geometry-m1.json").read_text(encoding="utf-8")
    )
    assert payload["manifold_signature"] == 0.0
    assert payload["far_and_reachable_count"] == 0
    assert list(output_dir.glob("retrieval-seed-*")) == []


def test_main_runs_the_sweep_when_the_gate_is_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(run_gate2_module, "AGENT_SWEEP", (4,))
    # Whether the gate computes the right verdict is geometry.py's concern and
    # is covered there; what this test covers is the CLI's branch wiring. On the
    # stub corpus the real gate now correctly refuses to open -- see
    # test_main_refuses_to_open_the_gate_on_a_structureless_corpus.
    seen: dict[str, object] = {}

    def _record(signature: float, null: list[float]) -> bool:
        seen["signature"] = signature
        seen["null"] = null
        return True

    monkeypatch.setattr(run_gate2_module, "stage_two_is_justified", _record)

    exit_code = run_gate2_module.main(
        [
            "--manifest", str(manifest_path),
            "--snapshot", str(snapshot_path),
            "--seed", "1",
            "--top-k", "4",
            "--initial-k", "4",
            "--stage1-seeds", "4",
            "--null-permutations", "5",
                "--exploratory",
            "--steps", "2",
            "--output-dir", str(output_dir),
        ]
    )

    assert exit_code == 0
    geometry = json.loads(
        (output_dir / "geometry-m1.json").read_text(encoding="utf-8")
    )
    # The gate is decided on the pooled population, not one seed's 2-4 pairs.
    assert geometry["pair_count"] > max(
        entry["pair_count"] for entry in geometry["per_seed"]
    )
    assert geometry["snapshot_backend"] == "stub"
    assert geometry["manifest_digest"]
    # The mock hides the verdict, so assert main() handed the predicate the
    # pooled signature and the pooled null -- not a per-seed value or another
    # run's list.
    assert seen["signature"] == geometry["manifold_signature"]
    assert seen["null"] == geometry["null_signatures"]

    retrieval_path = output_dir / "retrieval-seed-1-a4.json"
    assert retrieval_path.exists()
    payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    assert list(payload["runs"]) == ["A", "B", "C", "D", "E"]
    assert payload["snapshot_backend"] == "stub"


def test_geometry_payload_carries_the_null_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema only. The verdict is pinned closed so this test cannot depend on it.

    A 7-permutation null is deliberately weak -- its 95th percentile is barely
    more than the maximum of seven draws -- and the gate does open on it. The
    default is 100 for that reason. What is asserted here is that the evidence
    file carries the null and both sub-decisions, whichever way they fall.
    """
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        run_gate2_module, "stage_two_is_justified", lambda signature, null: False
    )

    with pytest.raises(SystemExit):
        run_gate2_module.main(
            [
                "--manifest", str(manifest_path),
                "--snapshot", str(snapshot_path),
                "--seed", "1",
                "--stage1-seeds", "4",
                "--null-permutations", "7",
                "--exploratory",
                "--steps", "2",
                "--output-dir", str(output_dir),
            ]
        )

    payload = json.loads((output_dir / "geometry-m1.json").read_text(encoding="utf-8"))
    assert len(payload["null_signatures"]) == 7
    assert payload["null_median"] == pytest.approx(
        sorted(payload["null_signatures"])[len(payload["null_signatures"]) // 2]
    )
    assert payload["excess_over_null_median"] == pytest.approx(
        payload["manifold_signature"] - payload["null_median"]
    )
    assert isinstance(payload["exceeds_null_p95"], bool)
    assert isinstance(payload["meets_absolute_minimum"], bool)
    assert isinstance(payload["meets_relative_excess"], bool)


def test_main_refuses_to_open_the_gate_on_a_structureless_corpus(tmp_path: Path) -> None:
    """The regression against the defect the null exists to fix.

    The stub embedder derives each vector from a text digest, so the corpus
    carries no geometric structure at all. The old bare 10% threshold was
    cleared three to four times over on exactly this input.

    Run at the PRE-REGISTERED defaults -- 12 stage-1 seeds, 100 permutations --
    and across several null seeds, because an earlier version of this test
    passed by 0.002 at a hand-picked configuration that nobody runs, while the
    CLI's own defaults opened the gate.
    """
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)

    for null_seed in (0, 1, 2):
        output_dir = tmp_path / f"out-{null_seed}"
        with pytest.raises(SystemExit) as excinfo:
            run_gate2_module.main(
                [
                    "--manifest", str(manifest_path),
                    "--snapshot", str(snapshot_path),
                    "--seed", "1",
                    "--null-seed", str(null_seed),
                    "--steps", "2",
                    "--output-dir", str(output_dir),
                ]
            )

        assert excinfo.value.code == 0
        assert list(output_dir.glob("retrieval-*")) == []
        payload = json.loads(
            (output_dir / "geometry-m1.json").read_text(encoding="utf-8")
        )
        assert payload["config"]["stage1_seeds"] == 12
        assert payload["config"]["null_permutations"] == 100
        assert payload["config"]["exploratory"] is False
        # The old gate would have opened: the raw signature clears 0.10 easily.
        assert payload["manifold_signature"] >= 0.10
        # Against its own null it does not clear all three conditions.
        assert not (
            payload["exceeds_null_p95"]
            and payload["meets_absolute_minimum"]
            and payload["meets_relative_excess"]
        )


def test_a_weakened_null_is_refused_without_the_exploratory_flag(tmp_path: Path) -> None:
    """A 7-permutation null opens the gate on structureless input, so a run
    below the pre-registered count must not be mistakable for the real thing."""
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)

    with pytest.raises(ValueError, match="below the pre-registered"):
        run_gate2_module.main(
            [
                "--manifest", str(manifest_path),
                "--snapshot", str(snapshot_path),
                "--seed", "1",
                "--null-permutations", "7",
                "--steps", "2",
                "--output-dir", str(tmp_path / "out"),
            ]
        )


def test_the_evidence_decomposes_the_signature_into_its_two_factors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The signature is far_rate x reach_given_far. If reachability saturates at
    1.000 the signature silently becomes the far rate, and the permutation test
    compares rank depth rather than reachability. The evidence has to show that.
    """
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        run_gate2_module, "stage_two_is_justified", lambda signature, null: False
    )

    with pytest.raises(SystemExit):
        run_gate2_module.main(
            [
                "--manifest", str(manifest_path),
                "--snapshot", str(snapshot_path),
                "--seed", "1",
                "--stage1-seeds", "4",
                "--null-permutations", "5",
                "--exploratory",
                "--steps", "2",
                "--output-dir", str(output_dir),
            ]
        )

    payload = json.loads((output_dir / "geometry-m1.json").read_text(encoding="utf-8"))
    assert payload["manifold_signature"] == pytest.approx(
        payload["far_rate"] * payload["reach_given_far"]
    )
    assert "null_far_rate_median" in payload
    assert "null_reach_given_far_median" in payload
    assert payload["required_excess"] == pytest.approx(
        0.10 * (1.0 - payload["null_median"])
    )
