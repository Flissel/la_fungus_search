"""
Test fixtures for chunking system tests.
"""
import os
import tempfile
import pytest
from pathlib import Path


@pytest.fixture
def temp_codebase():
    """Create a temporary directory with sample Python files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample codebase structure
        src_dir = Path(tmpdir) / "src"
        src_dir.mkdir()

        # Create embeddinggemma package structure
        pkg_dir = src_dir / "embeddinggemma"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        # Create a realtime subpackage
        realtime_dir = pkg_dir / "realtime"
        realtime_dir.mkdir()
        (realtime_dir / "__init__.py").write_text("")

        # Create sample server.py with some functions and classes
        server_content = '''"""Sample server module."""
import os
from typing import Optional

class StreamHandler:
    """Sample stream handler class."""

    def __init__(self, config: dict):
        self.config = config
        self.windows = []

    def start(self):
        """Start the handler."""
        if not self.windows:
            self.windows = [1000, 2000, 4000]
        return self._load_data()

    def _load_data(self):
        """Internal method to load data."""
        return {"status": "ok"}

def calculate_path():
    """Calculate project root path."""
    current = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current))))
'''
        (realtime_dir / "server.py").write_text(server_content)

        # Create ui subpackage
        ui_dir = pkg_dir / "ui"
        ui_dir.mkdir()
        (ui_dir / "__init__.py").write_text("")

        # Create sample corpus.py
        corpus_content = '''"""Sample corpus module."""
import os
from typing import List

def list_code_files(root_dir: str, max_files: int, exclude_dirs: List[str] = None) -> List[str]:
    """List Python files in directory."""
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        if exclude_dirs:
            dirs[:] = [d for d in dirs if not any(ex in os.path.join(root, d) for ex in exclude_dirs)]
        for fn in filenames:
            if fn.endswith('.py'):
                files.append(os.path.join(root, fn))
                if max_files and len(files) >= max_files:
                    return files
    return files

def chunk_python_file(path: str, windows: List[int]) -> List[str]:
    """Chunk a Python file by line windows."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return []

    chunks = []
    for w in windows:
        step = max(1, int(w))
        for i in range(0, len(lines), step):
            start = i + 1
            end = min(i + step, len(lines))
            body = ''.join(lines[i:end])
            if body.strip():
                header = f"# file: {path} | lines: {start}-{end}\\n"
                chunks.append(header + body)
    return chunks
'''
        (ui_dir / "corpus.py").write_text(corpus_content)

        # Create another sample file with more complex content
        util_content = '''"""Utility functions."""

def helper_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

class DataProcessor:
    """Process data."""

    def process(self, data):
        """Process the data."""
        return [x * 2 for x in data]

    def validate(self, data):
        """Validate the data."""
        if not data:
            raise ValueError("Empty data")
        return True
'''
        (pkg_dir / "utils.py").write_text(util_content)

        yield tmpdir


@pytest.fixture
def empty_directory():
    """Create an empty temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_python_file():
    """Create a single Python file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write('''"""Sample Python file for testing."""

def test_function():
    """A test function."""
    x = 1
    y = 2
    return x + y

class TestClass:
    """A test class."""

    def method_one(self):
        """Method one."""
        pass

    def method_two(self):
        """Method two."""
        return 42
''')
        filepath = f.name

    yield filepath

    # Cleanup
    try:
        os.unlink(filepath)
    except:
        pass
