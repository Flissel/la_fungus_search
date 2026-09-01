"""A local embedding service, so retrieval v2's dense arm can arm.

The production embedding-service is a wrapper over OpenAI `text-embedding-3-large`
and is dead at the account level (429 insufficient_quota, report section 17.1).
This serves a locally cached model over the same JSON contract — `GET /health`,
`POST /embed {"text"}`, `POST /embed/batch {"texts"}` — so nothing downstream has
to learn a second protocol, while `/health` names the real model and dimension so
the two services can never be mistaken for each other.

Deliberately stdlib-only on the HTTP side: the interpreter that has torch (the
model side) is not the interpreter that has FastAPI, and a threading HTTP server
is entirely adequate for a localhost, single-operator service. The model import
is lazy, so importing this module — for tests, or from the torch-free Fungus
venv — costs nothing.

Run from an interpreter with torch + sentence-transformers::

    python -m embeddinggemma.local_embedding_service --port 8091 \
        --model Qwen/Qwen3-Embedding-0.6B --hf-home E:/huggingface_cache --device cuda

Then arm v2's dense side with ``FUNGUS_V2_EMBEDDER_URL=http://127.0.0.1:8091``.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Sequence


class EmbeddingBackend:
    """Lazy, thread-safe wrapper. Tests inject `encode` and never touch torch."""

    def __init__(
        self,
        model: str,
        device: str,
        max_seq_length: int,
        encode: Callable[[Sequence[str]], list[list[float]]] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.max_seq_length = max_seq_length
        self._encode = encode
        self._lock = threading.Lock()
        self.dimension: int | None = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            if self._encode is None:
                from sentence_transformers import SentenceTransformer

                loaded = SentenceTransformer(self.model, device=self.device)
                loaded.max_seq_length = self.max_seq_length
                self._encode = lambda batch: loaded.encode(
                    list(batch), convert_to_numpy=True, normalize_embeddings=False
                ).tolist()
            vectors = self._encode(texts)
        if vectors and self.dimension is None:
            self.dimension = len(vectors[0])
        return vectors


def make_handler(backend: EmbeddingBackend) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:  # quiet by default
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server contract)
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(
                200,
                {
                    "status": "ok",
                    "model": backend.model,
                    "backend": "local-transformers",
                    "device": backend.device,
                    "dimension": backend.dimension,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                self._send(400, {"error": "invalid JSON"})
                return
            try:
                if self.path == "/embed":
                    text = payload.get("text")
                    if not isinstance(text, str) or not text:
                        self._send(400, {"error": "text must be a non-empty string"})
                        return
                    self._send(200, {"vector": backend.encode([text])[0]})
                elif self.path == "/embed/batch":
                    texts = payload.get("texts")
                    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                        self._send(400, {"error": "texts must be a list of strings"})
                        return
                    self._send(200, {"vectors": backend.encode(texts) if texts else []})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as error:  # model failure is a 502, like the original
                self._send(502, {"error": f"embedding failed: {error}"})

    return Handler


def serve(backend: EmbeddingBackend, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(backend))
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="local embedding service")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--warm", action="store_true", help="load the model before serving")
    arguments = parser.parse_args()
    if arguments.hf_home:
        os.environ["HF_HOME"] = arguments.hf_home
        os.environ["HF_HUB_CACHE"] = os.path.join(arguments.hf_home, "hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    backend = EmbeddingBackend(arguments.model, arguments.device, arguments.max_seq_length)
    if arguments.warm:
        backend.encode(["warmup"])
        print(f"model loaded: {backend.model} ({backend.dimension} dims, {backend.device})")
    server = serve(backend, arguments.port)
    print(f"serving on http://127.0.0.1:{arguments.port} (model {arguments.model})")
    server.serve_forever()


if __name__ == "__main__":
    main()
