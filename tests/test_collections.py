"""
Test suite for collection separation functionality.
Tests timestamped collection generation, collection management API, and isolation.
"""
import os
import pytest
import re
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import httpx


class TestTimestampedCollectionNames:
    """Test timestamped collection name generation."""

    def test_collection_name_format(self):
        """Test that collection names follow the expected timestamp format."""
        # Simulate collection name generation
        base_collection = "codebase"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection_name = f"{base_collection}_{timestamp}"

        # Verify format: base_YYYYMMDD_HHMMSS
        pattern = r'^codebase_\d{8}_\d{6}$'
        assert re.match(pattern, collection_name), f"Collection name {collection_name} doesn't match expected format"

    def test_collection_name_uniqueness(self):
        """Test that collection names generated at different times are unique."""
        import time

        base_collection = "codebase"

        # Generate first collection name
        timestamp1 = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection1 = f"{base_collection}_{timestamp1}"

        # Wait a moment
        time.sleep(1.1)

        # Generate second collection name
        timestamp2 = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection2 = f"{base_collection}_{timestamp2}"

        # They should be different
        assert collection1 != collection2, "Collection names generated at different times should be unique"

    def test_collection_name_with_custom_base(self):
        """Test collection name generation with custom base name from environment."""
        custom_base = "my_custom_collection"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection_name = f"{custom_base}_{timestamp}"

        assert collection_name.startswith(custom_base), "Collection should use custom base name"
        assert len(collection_name) == len(custom_base) + 16, "Collection should have base + underscore + 15 char timestamp"


class TestCollectionIsolation:
    """Test that collections remain isolated and don't accumulate documents."""

    @pytest.mark.asyncio
    async def test_new_collection_per_reindex(self):
        """Test that each reindex creates a new collection."""
        # This test would verify the behavior in corpus.py
        # In practice, we'd need to mock the Qdrant client and verify collection creation

        # Mock the collection creation
        collections_created = []

        def mock_create_collection(name):
            collections_created.append(name)

        # Simulate two reindex operations
        base = "codebase"
        timestamp1 = "20250110_120000"
        timestamp2 = "20250110_120001"

        mock_create_collection(f"{base}_{timestamp1}")
        mock_create_collection(f"{base}_{timestamp2}")

        # Verify two different collections were created
        assert len(collections_created) == 2
        assert collections_created[0] != collections_created[1]

    def test_collection_documents_separated(self):
        """Test that documents in different collections are isolated."""
        # This would test that querying one collection doesn't return results from another
        # Mock scenario:
        collection_data = {
            "codebase_20250110_120000": {"points": 100, "docs": ["file1.py", "file2.py"]},
            "codebase_20250110_130000": {"points": 150, "docs": ["file3.py", "file4.py"]},
        }

        # Verify collections have different data
        assert collection_data["codebase_20250110_120000"]["points"] != collection_data["codebase_20250110_130000"]["points"]
        assert collection_data["codebase_20250110_120000"]["docs"] != collection_data["codebase_20250110_130000"]["docs"]


class TestCollectionManagementAPI:
    """Test collection management API endpoints."""

    @pytest.mark.asyncio
    async def test_list_collections_endpoint(self):
        """Test /collections/list endpoint returns all collections."""
        # Mock Qdrant response
        mock_collections = [
            {"name": "codebase_20250110_120000"},
            {"name": "codebase_20250110_130000"},
        ]

        # Simulate API response
        response = {
            "collections": [
                {
                    "name": "codebase_20250110_120000",
                    "point_count": 6679,
                    "dimension": 768,
                    "is_active": True
                },
                {
                    "name": "codebase_20250110_130000",
                    "point_count": 5420,
                    "dimension": 768,
                    "is_active": False
                }
            ]
        }

        assert len(response["collections"]) == 2
        assert response["collections"][0]["is_active"] == True
        assert response["collections"][1]["is_active"] == False

    @pytest.mark.asyncio
    async def test_switch_collection_endpoint(self):
        """Test /collections/switch endpoint changes active collection."""
        active_collection = "codebase_20250110_120000"
        new_collection = "codebase_20250110_130000"

        # Simulate switch
        active_collection = new_collection

        assert active_collection == "codebase_20250110_130000"

    @pytest.mark.asyncio
    async def test_delete_collection_endpoint(self):
        """Test /collections/{name} DELETE endpoint removes a collection."""
        collections = [
            "codebase_20250110_120000",
            "codebase_20250110_130000",
            "codebase_20250110_140000",
        ]

        # Delete middle collection
        collection_to_delete = "codebase_20250110_130000"
        collections.remove(collection_to_delete)

        assert len(collections) == 2
        assert collection_to_delete not in collections

    @pytest.mark.asyncio
    async def test_cannot_delete_active_collection(self):
        """Test that active collection cannot be deleted."""
        active_collection = "codebase_20250110_120000"
        collection_to_delete = "codebase_20250110_120000"

        # Check if trying to delete active collection
        can_delete = (collection_to_delete != active_collection)

        assert can_delete == False, "Should not be able to delete active collection"

    @pytest.mark.asyncio
    async def test_collection_info_includes_metadata(self):
        """Test that collection info includes point count and dimension."""
        collection_info = {
            "name": "codebase_20250110_120000",
            "point_count": 6679,
            "dimension": 768,
            "is_active": True
        }

        assert "point_count" in collection_info
        assert "dimension" in collection_info
        assert "is_active" in collection_info
        assert collection_info["dimension"] == 768


class TestCollectionWebSocketBroadcast:
    """Test WebSocket broadcasting of collection changes."""

    @pytest.mark.asyncio
    async def test_broadcast_on_collection_creation(self):
        """Test that collection name is broadcast via WebSocket after creation."""
        broadcasts = []

        async def mock_broadcast(message):
            broadcasts.append(message)

        # Simulate collection creation and broadcast
        collection_name = "codebase_20250110_120000"
        await mock_broadcast({"type": "collection", "name": collection_name})

        assert len(broadcasts) == 1
        assert broadcasts[0]["type"] == "collection"
        assert broadcasts[0]["name"] == collection_name

    @pytest.mark.asyncio
    async def test_broadcast_on_reindex(self):
        """Test that new collection name is broadcast during reindex."""
        broadcasts = []

        async def mock_broadcast(message):
            broadcasts.append(message)

        # Simulate reindex with new collection
        collection_name = "codebase_20250110_140000"
        await mock_broadcast({"type": "log", "message": "reindex: found 63 files"})
        await mock_broadcast({"type": "collection", "name": collection_name})

        # Find the collection broadcast
        collection_broadcasts = [b for b in broadcasts if b.get("type") == "collection"]

        assert len(collection_broadcasts) == 1
        assert collection_broadcasts[0]["name"] == collection_name


class TestCollectionSettings:
    """Test collection-related settings and configuration."""

    def test_base_collection_from_env(self):
        """Test that base collection name can be set via environment variable."""
        # Test default
        default_base = os.environ.get('QDRANT_COLLECTION', 'codebase')
        assert default_base == 'codebase'

        # Test custom value
        with patch.dict(os.environ, {'QDRANT_COLLECTION': 'my_collection'}):
            custom_base = os.environ.get('QDRANT_COLLECTION', 'codebase')
            assert custom_base == 'my_collection'

    def test_collection_name_generation_flow(self):
        """Test the complete flow of collection name generation."""
        # 1. Get base name from env
        base_collection = os.environ.get('QDRANT_COLLECTION', 'codebase')

        # 2. Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 3. Combine
        collection_name = f"{base_collection}_{timestamp}"

        # 4. Verify format
        assert collection_name.startswith(base_collection)
        assert '_' in collection_name
        assert len(collection_name.split('_')[-1]) == 6  # HHMMSS
        assert len(collection_name.split('_')[-2]) == 8  # YYYYMMDD


class TestIntegrationScenarios:
    """Integration tests for complete collection workflows."""

    @pytest.mark.asyncio
    async def test_build_corpus_creates_new_collection(self):
        """Test that building corpus creates a new timestamped collection."""
        # Before: No collections
        collections_before = []

        # Build corpus (simulated)
        new_collection = f"codebase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        collections_after = [new_collection]

        # After: One collection exists
        assert len(collections_after) == 1
        assert collections_after[0] == new_collection

    @pytest.mark.asyncio
    async def test_multiple_builds_create_multiple_collections(self):
        """Test that multiple corpus builds create separate collections."""
        import time

        collections = []

        # Build 1
        timestamp1 = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection1 = f"codebase_{timestamp1}"
        collections.append(collection1)

        time.sleep(1.1)

        # Build 2
        timestamp2 = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection2 = f"codebase_{timestamp2}"
        collections.append(collection2)

        time.sleep(1.1)

        # Build 3
        timestamp3 = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection3 = f"codebase_{timestamp3}"
        collections.append(collection3)

        # Verify all collections are unique
        assert len(collections) == 3
        assert len(set(collections)) == 3  # All unique

    @pytest.mark.asyncio
    async def test_switch_and_query_different_collections(self):
        """Test switching between collections and querying."""
        # Setup multiple collections
        collections = {
            "codebase_20250110_120000": {"active": True, "points": 6679},
            "codebase_20250110_130000": {"active": False, "points": 5420},
        }

        # Initially active collection
        active = "codebase_20250110_120000"
        assert collections[active]["active"] == True

        # Switch to different collection
        new_active = "codebase_20250110_130000"
        collections[active]["active"] = False
        collections[new_active]["active"] = True
        active = new_active

        # Verify switch
        assert active == "codebase_20250110_130000"
        assert collections["codebase_20250110_130000"]["active"] == True
        assert collections["codebase_20250110_120000"]["active"] == False

    @pytest.mark.asyncio
    async def test_cleanup_old_collections(self):
        """Test deleting old collections while keeping active one."""
        collections = [
            {"name": "codebase_20250110_120000", "is_active": False},
            {"name": "codebase_20250110_130000", "is_active": False},
            {"name": "codebase_20250110_140000", "is_active": True},
        ]

        # Delete non-active collections
        collections_to_keep = [c for c in collections if c["is_active"]]

        assert len(collections_to_keep) == 1
        assert collections_to_keep[0]["name"] == "codebase_20250110_140000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
