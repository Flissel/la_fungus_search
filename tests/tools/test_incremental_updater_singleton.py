import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPDATER_PATH = ROOT / "incremental_updater.py"


def _load_updater():
    spec = importlib.util.spec_from_file_location("fungus_incremental_updater", UPDATER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_machine_singleton_rejects_second_holder_and_recovers_after_release(tmp_path):
    updater = _load_updater()
    assert hasattr(updater, "acquire_singleton_lock"), (
        "incremental_updater must expose the machine-wide singleton boundary"
    )

    lock_path = tmp_path / "fungus-incremental-updater.lock"
    first = updater.acquire_singleton_lock(lock_path)
    assert first is not None
    try:
        assert updater.acquire_singleton_lock(lock_path) is None
    finally:
        updater.release_singleton_lock(first)

    recovered = updater.acquire_singleton_lock(lock_path)
    assert recovered is not None
    updater.release_singleton_lock(recovered)
