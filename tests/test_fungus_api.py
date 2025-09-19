from fastapi.testclient import TestClient
from src.embeddinggemma.fungus_api import app


def test_build_event_from_chunk_header():
    client = TestClient(app)
    chunk = "# file: src/embeddinggemma/rag.py | lines: 10-40 | window: 200\nprint('x')\n"
    r = client.post("/api/edit/build_event", json={"chunk_text": chunk, "instructions": "Add log"})
    assert r.status_code == 200
    data = r.json()
    assert data["event"]["file_path"].endswith("src/embeddinggemma/rag.py")
    assert data["event"]["start_line"] == 10
    assert data["event"]["end_line"] == 40

