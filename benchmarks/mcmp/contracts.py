"""Fail-closed data contracts for the offline MCMP benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from collections.abc import Sequence

import numpy as np


def _validate_ids(ids: Sequence[str], *, name: str) -> None:
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError(f"{name} must contain non-empty string ids")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} must be unique")


def _validate_matrix(matrix: np.ndarray, *, name: str) -> None:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        canonical_matrix = np.asarray(matrix, dtype="<f4")
    if not np.all(np.isfinite(canonical_matrix)):
        raise ValueError(f"{name} must remain finite when normalized to float32")


def _validate_nonnegative_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _update_string(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(struct.pack("<I", len(encoded)))
    digest.update(encoded)


def _update_strings(digest: hashlib._Hash, values: Sequence[str]) -> None:
    digest.update(struct.pack("<I", len(values)))
    for value in values:
        _update_string(digest, value)


@dataclass
class BenchmarkDataset:
    dataset_id: str
    seed: int
    document_ids: Sequence[str]
    document_vectors: np.ndarray
    query_ids: Sequence[str]
    query_vectors: np.ndarray
    relevant_by_query: dict[str, frozenset[str]]

    def validate(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        _validate_ids(self.document_ids, name="document ids")
        _validate_ids(self.query_ids, name="query ids")
        _validate_matrix(self.document_vectors, name="document vectors")
        _validate_matrix(self.query_vectors, name="query vectors")
        if self.document_vectors.shape[0] != len(self.document_ids):
            raise ValueError("document vector rows must match document ids")
        if self.query_vectors.shape[0] != len(self.query_ids):
            raise ValueError("query vector rows must match query ids")
        if self.document_vectors.shape[1] != self.query_vectors.shape[1]:
            raise ValueError("document and query vector dimensions must match")

        document_ids = set(self.document_ids)
        query_ids = set(self.query_ids)
        for query_id, relevant_ids in self.relevant_by_query.items():
            if query_id not in query_ids:
                raise ValueError("unknown relevant query")
            if not isinstance(relevant_ids, frozenset):
                raise ValueError("relevant documents must be frozensets")
            unknown_ids = set(relevant_ids) - document_ids
            if unknown_ids:
                raise ValueError("unknown relevant document")

    def digest(self) -> str:
        self.validate()
        result = hashlib.sha256()
        result.update(b"BenchmarkDataset/v1\0")
        _update_string(result, self.dataset_id)
        result.update(struct.pack("<q", int(self.seed)))
        _update_strings(result, self.document_ids)
        _update_strings(result, self.query_ids)
        for matrix in (self.document_vectors, self.query_vectors):
            result.update(struct.pack("<II", *matrix.shape))
            result.update(np.ascontiguousarray(matrix, dtype="<f4").tobytes())
        for query_id in sorted(self.relevant_by_query):
            _update_string(result, query_id)
            _update_strings(result, sorted(self.relevant_by_query[query_id]))
        return result.hexdigest()


@dataclass(frozen=True)
class SearchRun:
    method: str
    query_ids: Sequence[str]
    ranked_document_ids: Sequence[str]
    initial_candidate_ids: frozenset[str]
    discovered_candidate_ids: frozenset[str]
    per_query_candidate_ids: dict[str, frozenset[str]]
    per_query_ranked_document_ids: dict[str, Sequence[str]]
    elapsed_ms: float
    candidate_comparisons: int | None
    mcmp_steps: int
    document_visits: dict[str, int]
    pheromone_trails: int
    per_query_initial_candidate_ids: dict[str, frozenset[str]] | None = None

    def validate(self, dataset: BenchmarkDataset) -> None:
        dataset.validate()
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string")
        _validate_ids(self.query_ids, name="run query ids")
        dataset_query_ids = set(dataset.query_ids)
        unknown_query_ids = set(self.query_ids) - dataset_query_ids
        if unknown_query_ids:
            raise ValueError("unknown run query id")

        document_ids = set(dataset.document_ids)
        self._validate_document_ids(self.ranked_document_ids, document_ids, "ranked document")
        self._validate_candidate_ids(self.initial_candidate_ids, document_ids)
        self._validate_candidate_ids(self.discovered_candidate_ids, document_ids)

        expected_query_ids = set(self.query_ids)
        if set(self.per_query_candidate_ids) != expected_query_ids:
            raise ValueError("per-query candidate ids must match run query ids")
        if set(self.per_query_ranked_document_ids) != expected_query_ids:
            raise ValueError("per-query ranked ids must match run query ids")
        if self.per_query_initial_candidate_ids is not None and set(
            self.per_query_initial_candidate_ids
        ) != expected_query_ids:
            raise ValueError("per-query initial ids must match run query ids")
        for query_id, candidate_ids in self.per_query_candidate_ids.items():
            self._validate_candidate_ids(candidate_ids, document_ids)
        for query_id, ranking in self.per_query_ranked_document_ids.items():
            self._validate_document_ids(ranking, document_ids, "ranked document")
        if self.per_query_initial_candidate_ids is not None:
            for candidate_ids in self.per_query_initial_candidate_ids.values():
                self._validate_candidate_ids(candidate_ids, document_ids)

        for document_id, visits in self.document_visits.items():
            if document_id not in document_ids:
                raise ValueError("unknown visited document")
            _validate_nonnegative_integer(visits, name="document visits")
        if not isinstance(
            self.elapsed_ms, (int, float, np.integer, np.floating)
        ) or isinstance(self.elapsed_ms, (bool, np.bool_)):
            raise ValueError("elapsed_ms must be finite and nonnegative")
        if not math.isfinite(float(self.elapsed_ms)) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and nonnegative")
        if self.candidate_comparisons is not None:
            _validate_nonnegative_integer(self.candidate_comparisons, name="candidate comparisons")
        _validate_nonnegative_integer(self.mcmp_steps, name="mcmp steps")
        _validate_nonnegative_integer(self.pheromone_trails, name="pheromone trails")

    @staticmethod
    def _validate_document_ids(
        ranked_ids: Sequence[str], document_ids: set[str], label: str
    ) -> None:
        if any(document_id not in document_ids for document_id in ranked_ids):
            raise ValueError(f"unknown {label}")
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError(f"duplicate {label}s")

    @staticmethod
    def _validate_candidate_ids(
        candidate_ids: frozenset[str], document_ids: set[str]
    ) -> None:
        if not isinstance(candidate_ids, frozenset):
            raise ValueError("candidate ids must be frozensets")
        if not candidate_ids <= document_ids:
            raise ValueError("unknown candidate document")
