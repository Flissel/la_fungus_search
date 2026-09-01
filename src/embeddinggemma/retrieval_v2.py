"""Retrieval v2: the measured stack, behind a flag, fail-closed.

What ships here is exactly the configuration the ablation harness measured best
(report section 27, probe `benchmarks/probes/callgraph_expand.py --base union`):

1. **Candidate union, not full-list fusion.** BM25 and the dense ranking each
   contribute interleaved hits. Full-list RRF was measured *worse* than BM25
   alone (0.613 vs 0.638 recall@8), so it is deliberately not used.
2. **One-hop call-graph expansion of the hits.** Each hit contributes its direct
   callers and callees from the manifest. Only the hits' edges are read.
3. **Function-granularity documents.** Per shown line of code, functions beat the
   200-line production windows 0.653 to 0.477; the misleading per-unit comparison
   is in report section 27.3 next to it.

Measured on the call-graph oracle: recall@8 0.712 / recall@16 0.833, against
0.487 / 0.606 for the dense ranking the v1 path serves. The protocol was
function-as-query; natural-language queries shift the dense/BM25 balance, which
is why this runs behind `FUNGUS_RETRIEVAL_V2` for side-by-side comparison rather
than replacing v1 outright.

Fail-closed rules, in the spirit of the snapshot pipeline this grew out of:

- A snapshot whose manifest digest does not match the manifest is refused.
- A dense arm whose embedder dimension does not match the snapshot is disarmed
  and the reason is recorded; BM25 and expansion still serve.
- Assets that fail to load raise; the caller decides whether that is a 500. An
  enabled flag with broken assets must never silently degrade to v1.

The manifest and snapshot formats are the Gate 2 ones. Their readers are inlined
here rather than imported, because `benchmarks/` is not a serving dependency; the
digest algorithm is replicated bit-for-bit and covered by a cross-check test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from embeddinggemma.bm25_lite import BM25Lite

HIT_BUDGET = 8


@dataclass(frozen=True)
class V2Document:
    document_id: str
    file: str
    start_line: int
    end_line: int
    symbol: str
    source: str


@dataclass
class V2Index:
    documents: tuple[V2Document, ...]
    neighbours: dict[str, frozenset[str]]
    manifest_digest: str
    bm25: BM25Lite
    vectors: np.ndarray | None = None
    dense_model: str = ""
    dense_disabled_reason: str = ""
    position: dict[str, int] = field(default_factory=dict)


def _digest(payload: dict) -> str:
    """The Gate 2 manifest digest, replicated bit-for-bit.

    Covered by a test that compares against `benchmarks.gate2.manifest`'s own
    implementation, so drift between the two fails loudly instead of letting a
    stale snapshot pass the check.
    """
    result = hashlib.sha256()
    result.update(b"Gate2Manifest/v1\0")
    for part in (payload["manifest_id"], payload["corpus_root"], payload["commit_sha"]):
        result.update(str(part).encode("utf-8") + b"\0")
    for document in payload["documents"]:
        result.update(str(document["document_id"]).encode("utf-8") + b"\0")
        result.update(str(document["source"]).encode("utf-8") + b"\0")
    callees = payload["callees_by_document"]
    for key in sorted(callees):
        result.update(key.encode("utf-8") + b":")
        result.update(",".join(sorted(callees[key])).encode("utf-8") + b"\0")
    return result.hexdigest()


def load_index(
    manifest_path: Path,
    snapshot_path: Path | None = None,
) -> V2Index:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    documents = tuple(
        V2Document(
            document_id=str(row["document_id"]),
            file=str(row["file"]),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            symbol=str(row["symbol"]),
            source=str(row["source"]),
        )
        for row in payload["documents"]
    )
    if not documents:
        raise ValueError(f"manifest {manifest_path} contains no documents")
    callees = {k: set(v) for k, v in payload["callees_by_document"].items()}
    callers = {k: set(v) for k, v in payload["callers_by_document"].items()}
    neighbours = {
        document.document_id: frozenset(
            callees.get(document.document_id, set()) | callers.get(document.document_id, set())
        )
        for document in documents
    }
    digest = _digest(payload)

    bm25 = BM25Lite()
    bm25.fit([document.source for document in documents])

    vectors: np.ndarray | None = None
    dense_model = ""
    if snapshot_path is not None:
        with np.load(Path(snapshot_path)) as stored:
            stored_digest = str(stored["manifest_digest"])
            if stored_digest != digest:
                raise ValueError(
                    "snapshot manifest digest does not match the manifest; a stale "
                    "or foreign snapshot must not serve"
                )
            stored_ids = [str(value) for value in stored["document_ids"]]
            if stored_ids != [document.document_id for document in documents]:
                raise ValueError("snapshot document order does not match the manifest")
            vectors = np.asarray(stored["vectors"], dtype=np.float32)
            dense_model = str(stored["model"])

    return V2Index(
        documents=documents,
        neighbours=neighbours,
        manifest_digest=digest,
        bm25=bm25,
        vectors=vectors,
        dense_model=dense_model,
        position={document.document_id: index for index, document in enumerate(documents)},
    )


class RetrievalV2:
    def __init__(
        self,
        index: V2Index,
        embed_query: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        self._index = index
        self._embed_query = embed_query
        if index.vectors is None:
            index.dense_disabled_reason = "no snapshot configured"
        elif embed_query is None:
            index.dense_disabled_reason = "no query embedder configured"

    @property
    def engine(self) -> str:
        if self._dense_armed():
            return f"v2:union+expand ({self._index.dense_model})"
        return f"v2:bm25+expand (dense off: {self._index.dense_disabled_reason})"

    def _dense_armed(self) -> bool:
        return (
            self._index.vectors is not None
            and self._embed_query is not None
            and not self._index.dense_disabled_reason
        )

    def _dense_ranking(self, query: str) -> list[int] | None:
        if not self._dense_armed():
            return None
        try:
            raw = np.asarray(self._embed_query(query), dtype=np.float32).reshape(-1)
        except Exception as error:
            # Same treatment as a dimension mismatch: disarm with the reason
            # recorded and keep serving. A dead embedder degrades the ranking,
            # never the endpoint.
            self._index.dense_disabled_reason = f"query embedder failed: {error}"
            return None
        expected = self._index.vectors.shape[1]
        if raw.shape[0] != expected:
            # Disarm rather than crash: the BM25 and expansion arms are still the
            # measured configuration for exactly this situation (report 27.1, the
            # bm25-base row). Recorded so /engine surfaces it.
            self._index.dense_disabled_reason = (
                f"query embedder returned {raw.shape[0]} dims, snapshot has {expected}"
            )
            return None
        query_vector = raw / max(float(np.linalg.norm(raw)), 1e-9)
        similarities = self._index.vectors @ query_vector
        return [int(i) for i in np.argsort(-similarities)]

    def search(self, query: str, top_k: int = 10) -> dict[str, object]:
        index = self._index
        scores = index.bm25.score(query)
        bm25_ranking = [int(i) for i in np.argsort(-scores)]
        dense_ranking = self._dense_ranking(query)

        # Interleaved union: half the hit budget from each arm, deduplicated,
        # so the candidate set stays the size the single-arm variants had.
        merged: list[int] = []
        if dense_ranking is None:
            merged = bm25_ranking[:HIT_BUDGET]
        else:
            for pair in zip(bm25_ranking, dense_ranking):
                for position in pair:
                    if position not in merged:
                        merged.append(position)
                if len(merged) >= HIT_BUDGET:
                    break
            merged = merged[:HIT_BUDGET]

        # One hop along the call graph, hits' edges only.
        candidates = set(merged)
        for position in list(candidates):
            document_id = index.documents[position].document_id
            for neighbour_id in index.neighbours.get(document_id, frozenset()):
                neighbour_position = index.position.get(neighbour_id)
                if neighbour_position is not None:
                    candidates.add(neighbour_position)

        # Rank within candidates by BM25 score -- the rule that measured 0.712
        # recall@8 -- with document order as the deterministic tie-break.
        ordered = sorted(candidates, key=lambda position: (-float(scores[position]), position))
        results = []
        for position in ordered[: max(1, int(top_k))]:
            document = index.documents[position]
            results.append(
                {
                    "content": document.source,
                    "metadata": {
                        "document_id": document.document_id,
                        "file": document.file,
                        "start_line": document.start_line,
                        "end_line": document.end_line,
                        "symbol": document.symbol,
                        "expanded": position not in merged,
                    },
                    "relevance_score": float(scores[position]),
                }
            )
        return {"results": results, "engine": self.engine}


class HttpQueryEmbedder:
    """Query embeddings from the local embedding service's `/embed` contract.

    Any failure raises; `RetrievalV2._dense_ranking` treats a raise the same way
    it treats a dimension mismatch -- disarm the dense arm with the reason
    recorded, keep serving BM25 + expansion. A dead embedder degrades the
    ranking, never the endpoint.
    """

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __call__(self, text: str) -> Sequence[float]:
        import urllib.request

        request = urllib.request.Request(
            f"{self.base_url}/embed",
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        vector = payload.get("vector")
        if not isinstance(vector, list) or not vector:
            raise ValueError("embedder returned no vector")
        return vector


def build_from_env(environ: dict[str, str]) -> RetrievalV2 | None:
    """The single wiring point. Returns None when the flag is off; raises when
    the flag is on and the assets are broken -- an enabled v2 must never
    silently degrade."""
    if environ.get("FUNGUS_RETRIEVAL_V2", "0") != "1":
        return None
    manifest_path = environ.get("FUNGUS_V2_MANIFEST", "")
    if not manifest_path:
        raise ValueError("FUNGUS_RETRIEVAL_V2=1 but FUNGUS_V2_MANIFEST is not set")
    snapshot_value = environ.get("FUNGUS_V2_SNAPSHOT", "")
    index = load_index(Path(manifest_path), Path(snapshot_value) if snapshot_value else None)
    embedder_url = environ.get("FUNGUS_V2_EMBEDDER_URL", "")
    embed_query = HttpQueryEmbedder(embedder_url) if embedder_url else None
    return RetrievalV2(index, embed_query=embed_query)
