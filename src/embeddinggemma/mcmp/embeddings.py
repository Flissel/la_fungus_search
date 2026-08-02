"""Fail-closed HTTP embeddings for the VibeMind embedding-service."""

from __future__ import annotations

import os
import time
from typing import Any, Sequence
from urllib.parse import urlparse

import requests


EMBEDDING_DIMENSION = 3072


class EmbeddingServiceError(RuntimeError):
    """The embedding-service returned an invalid or non-retryable result."""


class EmbeddingServiceUnavailable(EmbeddingServiceError):
    """The embedding-service remained unreachable after bounded retries."""


class EmbeddingServiceClient:
    """Small adapter for the shared embedding-service batch contract.

    This class intentionally has no provider SDK or local-model fallback.  The
    service owns credentials and provider selection; callers only receive
    validated vectors in the one supported 3072-dimensional space.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        session: requests.Session | Any | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
    ) -> None:
        configured_url = base_url if base_url is not None else os.environ.get("EMBEDDING_SERVICE_URL", "")
        if not configured_url or not configured_url.strip():
            raise EmbeddingServiceError(
                "EMBEDDING_SERVICE_URL is required; set a URL reachable from this Fungus runtime"
            )
        if any(character.isspace() for character in configured_url):
            raise EmbeddingServiceError("EMBEDDING_SERVICE_URL must be an absolute HTTP(S) URL without whitespace")
        try:
            parsed_url = urlparse(configured_url)
        except ValueError as exc:
            raise EmbeddingServiceError("EMBEDDING_SERVICE_URL must be an absolute HTTP(S) URL") from exc
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise EmbeddingServiceError("EMBEDDING_SERVICE_URL must be an absolute HTTP(S) URL")
        self._base_url = configured_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = float(timeout if timeout is not None else os.environ.get("EMBEDDING_HTTP_TIMEOUT", "30"))
        self._max_retries = max(0, int(max_retries if max_retries is not None else os.environ.get("EMBEDDING_HTTP_MAX_RETRIES", "2")))
        self._retry_backoff = max(0.0, float(retry_backoff if retry_backoff is not None else os.environ.get("EMBEDDING_HTTP_RETRY_BACKOFF", "0.5")))

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text in order through ``POST /embed/batch`` only."""
        values = list(texts)
        if not values:
            return []
        response = self._post_with_retry("/embed/batch", {"texts": values})
        try:
            payload = response.json()
            vectors = payload["vectors"]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingServiceError("embedding-service returned malformed batch response") from exc
        self._validate_vectors(vectors, len(values))
        return vectors

    def _post_with_retry(self, path: str, payload: dict[str, list[str]]) -> Any:
        attempts = self._max_retries + 1
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                response = self._session.post(
                    f"{self._base_url}{path}", json=payload, timeout=self._timeout
                )
                response.raise_for_status()
                return response
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_error = exc
            except requests.exceptions.HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                if not isinstance(status_code, int) or status_code < 500:
                    raise EmbeddingServiceError(
                        f"embedding-service request failed with status {status_code}"
                    ) from exc
                last_error = exc
            if attempt < attempts - 1:
                time.sleep(self._retry_backoff * (attempt + 1))
        raise EmbeddingServiceUnavailable(
            f"embedding-service unavailable after {attempts} attempts; no local fallback is configured"
        ) from last_error

    @staticmethod
    def _validate_vectors(vectors: Any, expected_count: int) -> None:
        if not isinstance(vectors, list) or len(vectors) != expected_count:
            actual_count = len(vectors) if isinstance(vectors, list) else "non-list"
            raise EmbeddingServiceError(
                "embedding-service response count mismatch: "
                f"expected {expected_count}, got {actual_count}"
            )
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
                actual_dimension = len(vector) if isinstance(vector, list) else "non-vector"
                raise EmbeddingServiceError(
                    "embedding-service returned unexpected dimension: "
                    f"expected {EMBEDDING_DIMENSION}, got {actual_dimension}; reindex is required "
                    "before this vector space can be used."
                )


def load_embedding_backend() -> tuple[EmbeddingServiceClient, int]:
    """Return the sole Fungus embedding backend and its fixed vector contract."""
    return EmbeddingServiceClient(), EMBEDDING_DIMENSION


def load_embedding_model() -> EmbeddingServiceClient:
    """Compatibility helper for callers using the historical model accessor."""
    return load_embedding_backend()[0]
