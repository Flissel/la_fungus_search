"""T2: the evidence CLI, end to end against a temporary repo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "embeddinggemma.maintainer_evidence", *arguments],
        capture_output=True,
        text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "audio.py").write_text(
        "def transcribe_audio(path):\n"
        "    model = load_whisper_model()\n"
        "    return model.run(path)\n"
        "\n"
        "def load_whisper_model():\n"
        "    return FasterWhisper('base')\n",
        encoding="utf-8",
    )
    return repo


def test_cli_answers_with_digested_hits(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(["laeuft ueber faster-whisper", "transcribe_audio"]), encoding="utf-8")

    result = _run(
        ["--repo", str(repo), "--queries", str(queries), "--cache-dir", str(tmp_path / "cache")]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["document_count"] == 2
    assert payload["engine"].startswith("v2:")
    by_query = {row["query"]: row["hits"] for row in payload["results"]}
    symbols = {hit["symbol"] for hit in by_query["laeuft ueber faster-whisper"]}
    assert "load_whisper_model" in symbols, "the whisper claim must find its function"
    for hits in by_query.values():
        for hit in hits:
            assert len(hit["digest"]) == 64, "every hit carries a sha256 source digest"
            assert hit["file"] and hit["start_line"] >= 1

    # The expansion arm shows: querying one function surfaces its call neighbour.
    expanded = {hit["symbol"]: hit["expanded"] for hit in by_query["transcribe_audio"]}
    assert "load_whisper_model" in expanded


def test_cli_fails_closed_on_missing_repo(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(["x"]), encoding="utf-8")
    result = _run(
        ["--repo", str(tmp_path / "nope"), "--queries", str(queries), "--cache-dir", str(tmp_path)]
    )
    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_cache_is_per_repo(tmp_path: Path) -> None:
    repo_a = _make_repo(tmp_path)
    repo_b = tmp_path / "other"
    repo_b.mkdir()
    (repo_b / "misc.py").write_text("def unrelated():\n    return 1\n", encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(["unrelated"]), encoding="utf-8")
    cache = tmp_path / "cache"

    first = _run(["--repo", str(repo_a), "--queries", str(queries), "--cache-dir", str(cache)])
    second = _run(["--repo", str(repo_b), "--queries", str(queries), "--cache-dir", str(cache)])
    assert first.returncode == 0 and second.returncode == 0
    # Two distinct cache slots: indexing repo B must not have touched repo A's.
    slots = [p for p in cache.iterdir() if p.is_dir()]
    assert len(slots) == 2
    payload = json.loads(second.stdout)
    assert payload["document_count"] == 1, "repo B answers from its own index"
