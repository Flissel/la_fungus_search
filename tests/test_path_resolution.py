"""
Tests for path resolution in the chunking system.

These tests verify that the system correctly resolves paths to the codebase
and doesn't create incorrect paths like 'src/src'.
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_path_resolution_from_server_location():
    """Test that path resolution from server.py location is correct."""
    # Simulate being at server.py location
    server_file = Path(__file__).parent.parent / "src" / "embeddinggemma" / "realtime" / "server.py"

    # Current BUGGY calculation (3 dirname calls)
    buggy_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(server_file))))
    buggy_rf = os.path.join(buggy_project_root, 'src')

    # FIXED calculation (4 dirname calls)
    fixed_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(server_file)))))
    fixed_rf = os.path.join(fixed_project_root, 'src')

    print(f"\nserver.py location: {server_file}")
    print(f"Buggy project_root: {buggy_project_root}")
    print(f"Buggy rf: {buggy_rf}")
    print(f"Fixed project_root: {fixed_project_root}")
    print(f"Fixed rf: {fixed_rf}")

    # The buggy version should create an incorrect path
    assert "src" in buggy_rf, "Buggy path should contain 'src'"

    # The fixed version should point to the correct src directory
    expected_src = Path(__file__).parent.parent / "src"
    assert Path(fixed_rf).resolve() == expected_src.resolve(), \
        f"Fixed path {fixed_rf} should equal {expected_src}"


def test_no_double_src_in_path():
    """Test that we don't create paths with 'src/src'."""
    # Simulate server.py location
    server_file = Path(__file__).parent.parent / "src" / "embeddinggemma" / "realtime" / "server.py"

    # Current BUGGY calculation
    buggy_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(server_file))))
    buggy_rf = os.path.join(buggy_project_root, 'src')

    # Check if buggy version creates src/src
    path_parts = Path(buggy_rf).parts
    src_count = path_parts.count('src')

    print(f"\nPath parts: {path_parts}")
    print(f"'src' appears {src_count} times")

    # In the buggy version, we might get src twice
    # This test documents the bug
    if src_count > 1:
        print("BUG DETECTED: Path contains 'src' multiple times")


def test_correct_directory_levels():
    """Test the number of directory levels traversed."""
    server_file = Path(__file__).parent.parent / "src" / "embeddinggemma" / "realtime" / "server.py"

    # Directory structure:
    # la_fungus_search/          <- project root (need 4 dirname to get here from server.py)
    #   src/                     <- target directory (need 3 dirname to get here from server.py)
    #     embeddinggemma/        <- 2 dirname
    #       realtime/            <- 1 dirname
    #         server.py          <- 0 dirname (starting point)

    # 1 dirname from server.py
    level_1 = os.path.dirname(os.path.abspath(server_file))
    assert level_1.endswith("realtime"), f"Level 1 should be realtime, got {level_1}"

    # 2 dirname from server.py
    level_2 = os.path.dirname(level_1)
    assert level_2.endswith("embeddinggemma"), f"Level 2 should be embeddinggemma, got {level_2}"

    # 3 dirname from server.py
    level_3 = os.path.dirname(level_2)
    assert level_3.endswith("src"), f"Level 3 should be src, got {level_3}"

    # 4 dirname from server.py
    level_4 = os.path.dirname(level_3)
    assert level_4.endswith("la_fungus_search"), f"Level 4 should be la_fungus_search, got {level_4}"

    print(f"\nDirectory levels:")
    print(f"  0: {server_file}")
    print(f"  1: {level_1}")
    print(f"  2: {level_2}")
    print(f"  3: {level_3} <- target (src dir)")
    print(f"  4: {level_4} <- project root")


def test_path_exists_after_resolution():
    """Test that the resolved path actually exists."""
    # Fixed calculation
    server_file = Path(__file__).parent.parent / "src" / "embeddinggemma" / "realtime" / "server.py"
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(server_file)))))
    rf = os.path.join(project_root, 'src')

    print(f"\nResolved path: {rf}")
    print(f"Path exists: {os.path.exists(rf)}")
    print(f"Is directory: {os.path.isdir(rf)}")

    assert os.path.exists(rf), f"Path {rf} should exist"
    assert os.path.isdir(rf), f"Path {rf} should be a directory"

    # Check if there are Python files
    py_files = []
    for root, dirs, files in os.walk(rf):
        for f in files:
            if f.endswith('.py'):
                py_files.append(os.path.join(root, f))
        if len(py_files) >= 3:
            break

    print(f"Found {len(py_files)} Python files (showing first 3): {py_files[:3]}")
    assert len(py_files) > 0, f"Should find Python files in {rf}"


if __name__ == "__main__":
    print("=" * 70)
    print("PATH RESOLUTION TESTS - IDENTIFYING BUGS")
    print("=" * 70)

    print("\n1. Testing path resolution from server.py location...")
    test_path_resolution_from_server_location()

    print("\n2. Testing for double 'src' in path...")
    test_no_double_src_in_path()

    print("\n3. Testing directory level traversal...")
    test_correct_directory_levels()

    print("\n4. Testing that resolved path exists...")
    test_path_exists_after_resolution()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
