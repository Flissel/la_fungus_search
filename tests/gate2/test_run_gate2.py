from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.gate2 import run_gate2 as run_gate2_module
from benchmarks.gate2.manifest import build_manifest, save_manifest
from benchmarks.gate2.provider import build_gate2_dataset
from benchmarks.gate2.run_gate2 import AGENT_SWEEP, run_retrieval, write_result
from benchmarks.gate2.snapshot import build_stub_snapshot, save_snapshot


def _build_manifest(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"def f{index}():\n    return f{index + 1}()\n\n" for index in range(12)
    )
    (root / "alpha.py").write_text(body + "def f12():\n    return 0\n", encoding="utf-8")
    return build_manifest(root, commit_sha="sha", manifest_id="m1")


def _dataset(tmp_path: Path):
    manifest = _build_manifest(tmp_path)
    return build_gate2_dataset(manifest, build_stub_snapshot(manifest, dimension=16), seed=1)


def _write_manifest_and_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    """Build the same corpus as ``_dataset`` and persist it to disk for main()."""
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


def test_main_skips_stage_two_when_the_gate_is_not_met(tmp_path: Path) -> None:
    """Seed 6 on this corpus naturally yields a manifold signature of 0.0 at
    the default top-k=8 -- well below the 0.10 gate -- with no patching.
    That keeps this a real end-to-end exercise of the CLI's gate decision,
    not an assertion about a mocked boolean.
    """
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)
    output_dir = tmp_path / "out"

    with pytest.raises(SystemExit) as excinfo:
        run_gate2_module.main(
            [
                "--manifest", str(manifest_path),
                "--snapshot", str(snapshot_path),
                "--seed", "6",
                "--steps", "2",
                "--output-dir", str(output_dir),
            ]
        )

    assert excinfo.value.code == 0
    assert (output_dir / "geometry-seed-6.json").exists()
    assert list(output_dir.glob("retrieval-seed-*")) == []


def test_main_runs_the_sweep_when_the_gate_is_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, snapshot_path = _write_manifest_and_snapshot(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(run_gate2_module, "stage_two_is_justified", lambda signature: True)
    monkeypatch.setattr(run_gate2_module, "AGENT_SWEEP", (4,))

    exit_code = run_gate2_module.main(
        [
            "--manifest", str(manifest_path),
            "--snapshot", str(snapshot_path),
            "--seed", "1",
            "--top-k", "4",
            "--initial-k", "4",
            "--steps", "2",
            "--output-dir", str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "geometry-seed-1.json").exists()
    retrieval_path = output_dir / "retrieval-seed-1-a4.json"
    assert retrieval_path.exists()
    payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    assert list(payload["runs"]) == ["A", "B", "C", "D", "E"]
