import importlib.util
import subprocess
import sys
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


def test_machine_singleton_rejects_separate_process_and_recovers_after_exit(tmp_path):
    """The OS-owned lock, not Python module state, fences another process."""
    lock_path = tmp_path / "fungus-incremental-updater.lock"
    holder = """
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("fungus_lock_holder", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
handle = module.acquire_singleton_lock(Path(sys.argv[2]))
if handle is None:
    raise SystemExit(2)
print("locked", flush=True)
if sys.stdin.readline().strip() != "release":
    raise SystemExit(3)
module.release_singleton_lock(handle)
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", holder, str(UPDATER_PATH), str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"

    updater = _load_updater()
    assert updater.acquire_singleton_lock(lock_path) is None

    stdout, stderr = process.communicate("release\n", timeout=5)
    assert process.returncode == 0, stdout + stderr
    recovered = updater.acquire_singleton_lock(lock_path)
    assert recovered is not None
    updater.release_singleton_lock(recovered)
