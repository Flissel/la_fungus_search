"""The local embedding service and the v2 dense arm it feeds — torch-free tests.

The backend's `encode` is injected, so no test loads a model; what is tested is
the HTTP contract, and that the v2 arm arms against it, disarms with a recorded
reason when it dies, and never takes the endpoint down with it.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from benchmarks.gate2.manifest import build_manifest, save_manifest
from benchmarks.gate2.snapshot import build_stub_snapshot, save_snapshot
from embeddinggemma.local_embedding_service import EmbeddingBackend, serve
from embeddinggemma.retrieval_v2 import HttpQueryEmbedder, RetrievalV2, load_index


@pytest.fixture()
def service():
    backend = EmbeddingBackend(
        model="stub-16d",
        device="cpu",
        max_seq_length=64,
        encode=lambda texts: [[0.25] * 16 for _ in texts],
    )
    server = serve(backend, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def _post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_health_names_the_real_model(service: str) -> None:
    with urllib.request.urlopen(f"{service}/health", timeout=5) as response:
        payload = json.load(response)
    assert payload["model"] == "stub-16d"
    assert payload["backend"] == "local-transformers"


def test_embed_contract(service: str) -> None:
    status, payload = _post(f"{service}/embed", {"text": "hello"})
    assert status == 200 and len(payload["vector"]) == 16
    status, payload = _post(f"{service}/embed", {"text": ""})
    assert status == 400
    status, payload = _post(f"{service}/embed/batch", {"texts": ["a", "b"]})
    assert status == 200 and len(payload["vectors"]) == 2


@pytest.fixture()
def v2_assets(tmp_path: Path) -> dict[str, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text(
        "def parse_config(path):\n    return normalise_settings(path)\n\n"
        "def normalise_settings(raw):\n    return dict(raw)\n",
        encoding="utf-8",
    )
    manifest = build_manifest(corpus, "sha", "svc-test")
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    snapshot_path = tmp_path / "snapshot.npz"
    save_snapshot(build_stub_snapshot(manifest, dimension=16), snapshot_path)
    return {"manifest": manifest_path, "snapshot": snapshot_path}


def test_dense_arm_arms_against_the_live_service(service: str, v2_assets: dict[str, Path]) -> None:
    index = load_index(v2_assets["manifest"], v2_assets["snapshot"])
    engine = RetrievalV2(index, embed_query=HttpQueryEmbedder(service))
    result = engine.search("parse_config", top_k=2)
    assert result["results"]
    assert engine.engine.startswith("v2:union+expand")


def test_dead_embedder_disarms_and_keeps_serving(v2_assets: dict[str, Path]) -> None:
    index = load_index(v2_assets["manifest"], v2_assets["snapshot"])
    # A port nothing listens on: the embed call raises, the arm disarms with the
    # reason recorded, and BM25 + expansion still answer.
    engine = RetrievalV2(
        index, embed_query=HttpQueryEmbedder("http://127.0.0.1:9", timeout=0.3)
    )
    result = engine.search("parse_config", top_k=2)
    assert result["results"], "BM25 must keep serving"
    assert "query embedder failed" in index.dense_disabled_reason
    assert "dense off" in engine.engine
