"""
Tests for the chunking system to identify breaking points.

These tests verify that files are discovered, chunked correctly,
and that empty windows lists are handled properly.
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddinggemma.ui.corpus import (
    list_code_files,
    chunk_python_file,
    _chunk_line_windows,
    _chunk_python_file_ast,
    collect_codebase_chunks,
)


def test_empty_windows_produces_no_chunks_line_based(sample_python_file):
    """
    BUG TEST: Empty windows list produces no chunks in line-based chunking.
    This test documents the current buggy behavior.
    """
    chunks = _chunk_line_windows(sample_python_file, [])

    print(f"\nEmpty windows test (line-based):")
    print(f"  Input file: {sample_python_file}")
    print(f"  Windows: []")
    print(f"  Chunks produced: {len(chunks)}")

    # This is the BUG - empty windows produces 0 chunks
    assert len(chunks) == 0, "BUG CONFIRMED: Empty windows produces no chunks"
    print("  BUG CONFIRMED: Empty windows list produces 0 chunks")


def test_empty_windows_with_fallback_line_based(sample_python_file):
    """Test that providing a fallback window works."""
    # Simulate what the fix should do
    windows = []
    if not windows or len(windows) == 0:
        windows = [1000]  # Fallback

    chunks = _chunk_line_windows(sample_python_file, windows)

    print(f"\nEmpty windows with fallback test (line-based):")
    print(f"  Input file: {sample_python_file}")
    print(f"  Windows (after fallback): {windows}")
    print(f"  Chunks produced: {len(chunks)}")

    assert len(chunks) > 0, "With fallback, should produce chunks"
    print(f"  SUCCESS: Fallback produced {len(chunks)} chunks")


def test_ast_chunking_with_empty_windows(sample_python_file):
    """Test AST chunking with empty windows."""
    chunks = _chunk_python_file_ast(sample_python_file, [])

    print(f"\nEmpty windows test (AST-based):")
    print(f"  Input file: {sample_python_file}")
    print(f"  Windows: []")
    print(f"  Chunks produced: {len(chunks)}")

    # AST chunking has internal fallback to 1000
    if len(chunks) > 0:
        print("  AST chunking handled empty windows correctly")
    else:
        print("  WARNING: AST chunking also failed with empty windows")


def test_valid_windows_produces_chunks(sample_python_file):
    """Test that valid windows list produces chunks."""
    windows = [10, 20]
    chunks = _chunk_line_windows(sample_python_file, windows)

    print(f"\nValid windows test:")
    print(f"  Input file: {sample_python_file}")
    print(f"  Windows: {windows}")
    print(f"  Chunks produced: {len(chunks)}")

    assert len(chunks) > 0, "Valid windows should produce chunks"
    print(f"  SUCCESS: Produced {len(chunks)} chunks")

    # Verify chunk format
    if chunks:
        print(f"  Sample chunk header: {chunks[0].split(chr(10))[0]}")


def test_list_code_files_finds_files(temp_codebase):
    """Test that list_code_files discovers Python files."""
    src_dir = os.path.join(temp_codebase, "src")
    files = list_code_files(src_dir, max_files=100, exclude_dirs=None)

    print(f"\nFile discovery test:")
    print(f"  Root dir: {src_dir}")
    print(f"  Files found: {len(files)}")
    print(f"  Sample files: {[os.path.basename(f) for f in files[:5]]}")

    assert len(files) > 0, "Should find Python files in test codebase"
    print(f"  SUCCESS: Found {len(files)} Python files")


def test_list_code_files_with_exclude(temp_codebase):
    """Test that exclude_dirs filter works."""
    src_dir = os.path.join(temp_codebase, "src")

    # First, get all files
    all_files = list_code_files(src_dir, max_files=100, exclude_dirs=None)

    # Then exclude the ui directory
    filtered_files = list_code_files(src_dir, max_files=100, exclude_dirs=["ui"])

    print(f"\nExclude dirs test:")
    print(f"  All files: {len(all_files)}")
    print(f"  After excluding 'ui': {len(filtered_files)}")

    assert len(filtered_files) < len(all_files), "Excluding dirs should reduce file count"
    print(f"  SUCCESS: Filter reduced files from {len(all_files)} to {len(filtered_files)}")


def test_list_code_files_empty_directory(empty_directory):
    """Test behavior when no Python files exist."""
    files = list_code_files(empty_directory, max_files=100, exclude_dirs=None)

    print(f"\nEmpty directory test:")
    print(f"  Root dir: {empty_directory}")
    print(f"  Files found: {len(files)}")

    assert len(files) == 0, "Empty directory should return no files"
    print("  SUCCESS: Empty directory returns empty list")


def test_collect_codebase_chunks_with_empty_windows_BUGGY(temp_codebase):
    """
    BUG TEST: collect_codebase_chunks with empty windows produces no chunks.
    This test documents the current buggy behavior.
    """
    src_dir = os.path.join(temp_codebase, "src")
    chunks = collect_codebase_chunks(src_dir, windows=[], max_files=100, exclude_dirs=None)

    print(f"\nFull pipeline test with empty windows (BUGGY):")
    print(f"  Root dir: {src_dir}")
    print(f"  Windows: []")
    print(f"  Chunks collected: {len(chunks)}")

    # This is the BUG - empty windows produces 0 chunks even though files exist
    if len(chunks) == 0:
        print("  BUG CONFIRMED: Empty windows in full pipeline produces 0 chunks")
        print("  This is the ROOT CAUSE of the chunking failure")
    else:
        print(f"  Unexpected: Produced {len(chunks)} chunks")


def test_collect_codebase_chunks_with_valid_windows(temp_codebase):
    """Test that collect_codebase_chunks works with valid windows."""
    src_dir = os.path.join(temp_codebase, "src")
    chunks = collect_codebase_chunks(src_dir, windows=[100, 200], max_files=100, exclude_dirs=None)

    print(f"\nFull pipeline test with valid windows:")
    print(f"  Root dir: {src_dir}")
    print(f"  Windows: [100, 200]")
    print(f"  Chunks collected: {len(chunks)}")

    assert len(chunks) > 0, "Valid windows should produce chunks"
    print(f"  SUCCESS: Produced {len(chunks)} chunks")

    # Show sample chunks
    if chunks:
        print(f"  Sample chunk (first 100 chars): {chunks[0][:100]}...")


def test_chunk_python_file_integration(sample_python_file):
    """Test the high-level chunk_python_file function."""
    # Test with empty windows
    chunks_empty = chunk_python_file(sample_python_file, [])
    print(f"\nIntegration test chunk_python_file:")
    print(f"  With empty windows: {len(chunks_empty)} chunks")

    # Test with valid windows
    chunks_valid = chunk_python_file(sample_python_file, [10, 20])
    print(f"  With valid windows [10, 20]: {len(chunks_valid)} chunks")

    if len(chunks_empty) == 0 and len(chunks_valid) > 0:
        print("  BUG CONFIRMED: Empty windows causes failure at high level")


def test_windows_fallback_logic():
    """Test the windows fallback logic pattern."""
    print(f"\nWindows fallback logic test:")

    # Test 1: Empty list
    windows = []
    if not windows or len(windows) == 0:
        windows = [1000]
    print(f"  Empty list [] -> {windows}")
    assert windows == [1000], "Empty list should trigger fallback"

    # Test 2: None
    windows = None
    if not windows or (windows is not None and len(windows) == 0):
        windows = [1000]
    print(f"  None -> {windows}")
    assert windows == [1000], "None should trigger fallback"

    # Test 3: Valid list
    windows = [500, 1000]
    original = windows.copy()
    if not windows or len(windows) == 0:
        windows = [1000]
    print(f"  Valid list {original} -> {windows}")
    assert windows == original, "Valid list should not change"

    print("  SUCCESS: Fallback logic works correctly")


if __name__ == "__main__":
    import pytest

    print("=" * 70)
    print("CHUNKING SYSTEM TESTS - IDENTIFYING BUGS")
    print("=" * 70)

    # Run with pytest if available, otherwise inform user
    print("\nTo run these tests properly, use: pytest tests/test_chunking_system.py -v")
    print("\nOr run individual tests with pytest:")
    print("  pytest tests/test_chunking_system.py::test_empty_windows_produces_no_chunks_line_based -v")
    print("\n" + "=" * 70)
