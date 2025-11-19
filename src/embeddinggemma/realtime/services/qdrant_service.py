"""Qdrant service - centralized vector database operations."""

from __future__ import annotations
from typing import Any
import logging

_logger = logging.getLogger(__name__)


class QdrantService:
    """Centralized Qdrant client management and operations.

    This service provides a single point of access for all Qdrant operations,
    eliminating the need for repeated client initialization across endpoints.
    """

    def __init__(self, url: str, api_key: str | None = None, collection: str | None = None):
        """Initialize Qdrant service.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            collection: Default collection name
        """
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        """Lazy-loaded Qdrant client instance.

        Returns:
            QdrantClient instance
        """
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
                _logger.info(f"Qdrant client initialized: url={self.url}")
            except Exception as e:
                _logger.error(f"Failed to initialize Qdrant client: {e}")
                raise
        return self._client

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections with metadata.

        Returns:
            List of collection info dictionaries
        """
        try:
            # Get collection names only (CollectionDescription has only name in newer versions)
            collections = self.client.get_collections().collections
            result = []

            # Fetch detailed info for each collection
            for c in collections:
                try:
                    # Get full collection info which has vectors_count and points_count
                    info = self.client.get_collection(collection_name=c.name)
                    result.append({
                        "name": c.name,
                        "vectors_count": getattr(info, 'vectors_count', 0) or 0,
                        "points_count": getattr(info, 'points_count', 0) or 0,
                    })
                except Exception as e:
                    _logger.warning(f"Could not get info for collection {c.name}: {e}")
                    # Fallback with no counts
                    result.append({
                        "name": c.name,
                        "vectors_count": 0,
                        "points_count": 0,
                    })

            return result
        except Exception as e:
            _logger.error(f"Failed to list collections: {e}")
            return []

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name

        Returns:
            True if collection exists
        """
        try:
            collections = self.client.get_collections().collections
            return any(c.name == name for c in collections)
        except Exception:
            return False

    def get_collection_info(self, name: str) -> dict[str, Any] | None:
        """Get detailed information about a collection.

        Args:
            name: Collection name

        Returns:
            Collection info dict or None if not found
        """
        try:
            info = self.client.get_collection(collection_name=name)
            return {
                "name": name,
                "vectors_count": getattr(info, 'vectors_count', 0) or 0,
                "points_count": getattr(info, 'points_count', 0) or 0,
                "config": {
                    "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None,
                    "distance": info.config.params.vectors.distance.name if hasattr(info.config.params, 'vectors') else None,
                }
            }
        except Exception as e:
            _logger.error(f"Failed to get collection info for {name}: {e}")
            return None

    def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(collection_name=name)
            _logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            _logger.error(f"Failed to delete collection {name}: {e}")
            return False

    def upsert_points(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """Upsert points to a collection in batches.

        Args:
            collection_name: Target collection name
            points: List of point dictionaries with id, vector, payload
            batch_size: Batch size for upsert operations

        Returns:
            Number of points upserted
        """
        try:
            from qdrant_client.models import PointStruct

            total = 0
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                point_structs = [
                    PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p.get("payload", {})
                    )
                    for p in batch
                ]
                self.client.upsert(
                    collection_name=collection_name,
                    points=point_structs
                )
                total += len(batch)

            _logger.info(f"Upserted {total} points to collection {collection_name}")
            return total
        except Exception as e:
            _logger.error(f"Failed to upsert points: {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in a collection.

        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of search results with id, score, and payload
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload or {}
                }
                for r in results
            ]
        except Exception as e:
            _logger.error(f"Search failed: {e}")
            raise

    def scroll_all_points(
        self,
        collection_name: str,
        batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Retrieve all points from a collection using scroll API.

        Args:
            collection_name: Collection name
            batch_size: Batch size for scrolling

        Returns:
            List of all points with id, vector, and payload
        """
        try:
            all_points = []
            offset = None

            while True:
                results, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True
                )

                all_points.extend([
                    {
                        "id": p.id,
                        "vector": p.vector,
                        "payload": p.payload or {}
                    }
                    for p in results
                ])

                if offset is None:
                    break

            _logger.info(f"Retrieved {len(all_points)} points from {collection_name}")
            return all_points
        except Exception as e:
            _logger.error(f"Failed to scroll points: {e}")
            raise


def create_qdrant_service(url: str, api_key: str | None = None, collection: str | None = None) -> QdrantService:
    """Factory function to create a QdrantService instance.

    Args:
        url: Qdrant server URL
        api_key: Optional API key
        collection: Optional default collection name

    Returns:
        QdrantService instance
    """
    return QdrantService(url=url, api_key=api_key, collection=collection)


__all__ = ["QdrantService", "create_qdrant_service"]
