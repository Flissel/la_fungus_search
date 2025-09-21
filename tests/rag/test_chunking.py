import os
from pathlib import Path


def test_parse_code_with_ast_and_fallback(tmp_path: Path):
    from embeddinggemma.rag import chunking

    # Valid python
    p_good = tmp_path / "good.py"
    p_good.write_text("""
def foo():
    return 1
""".strip())
    chunks = chunking.parse_code_with_ast(str(p_good))
    assert isinstance(chunks, list) and len(chunks) >= 1
    assert chunks[0]["metadata"]["start_line"] == 1

    # Invalid python -> fallback
    p_bad = tmp_path / "bad.py"
    p_bad.write_text("""
def oops(
    x = 1
""".strip())
    chunks2 = chunking.parse_code_with_ast(str(p_bad))
    assert isinstance(chunks2, list)
    # fallback yields line_chunk items with metadata size
    assert all("size" in c["metadata"] for c in chunks2)


