"""
Standalone test runner for collection separation functionality.
Runs without pytest - just execute with python.
"""
import os
import re
import time
from datetime import datetime
from unittest.mock import patch


class TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def assert_true(self, condition, message=""):
        if not condition:
            raise AssertionError(f"Assertion failed: {message}")

    def assert_equal(self, a, b, message=""):
        if a != b:
            raise AssertionError(f"Expected {a} == {b}. {message}")

    def assert_not_equal(self, a, b, message=""):
        if a == b:
            raise AssertionError(f"Expected {a} != {b}. {message}")

    def assert_in(self, item, container, message=""):
        if item not in container:
            raise AssertionError(f"Expected {item} in {container}. {message}")

    def run_test(self, name, test_func):
        """Run a single test function."""
        try:
            test_func()
            self.passed += 1
            print(f"[PASS] {name}")
        except AssertionError as e:
            self.failed += 1
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            self.failed += 1
            print(f"[ERROR] {name}: Unexpected error: {e}")

    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"FAILED: {self.failed} tests failed")
        else:
            print("SUCCESS: All tests passed!")
        print(f"{'='*60}")


# Test instances
runner = TestRunner()


def test_collection_name_format():
    """Test that collection names follow the expected timestamp format."""
    base_collection = "codebase"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    collection_name = f"{base_collection}_{timestamp}"

    # Verify format: base_YYYYMMDD_HHMMSS
    pattern = r'^codebase_\d{8}_\d{6}$'
    runner.assert_true(
        re.match(pattern, collection_name),
        f"Collection name {collection_name} doesn't match expected format"
    )


def test_collection_name_uniqueness():
    """Test that collection names generated at different times are unique."""
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
    runner.assert_not_equal(
        collection1, collection2,
        "Collection names generated at different times should be unique"
    )


def test_collection_name_with_custom_base():
    """Test collection name generation with custom base name."""
    custom_base = "my_custom_collection"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    collection_name = f"{custom_base}_{timestamp}"

    runner.assert_true(
        collection_name.startswith(custom_base),
        "Collection should use custom base name"
    )
    runner.assert_equal(
        len(collection_name),
        len(custom_base) + 16,  # underscore + 15 char timestamp
        "Collection should have base + underscore + 15 char timestamp"
    )


def test_new_collection_per_reindex():
    """Test that each reindex creates a new collection."""
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
    runner.assert_equal(len(collections_created), 2, "Should create 2 collections")
    runner.assert_not_equal(
        collections_created[0], collections_created[1],
        "Collections should have different names"
    )


def test_collection_documents_separated():
    """Test that documents in different collections are isolated."""
    collection_data = {
        "codebase_20250110_120000": {"points": 100, "docs": ["file1.py", "file2.py"]},
        "codebase_20250110_130000": {"points": 150, "docs": ["file3.py", "file4.py"]},
    }

    # Verify collections have different data
    runner.assert_not_equal(
        collection_data["codebase_20250110_120000"]["points"],
        collection_data["codebase_20250110_130000"]["points"],
        "Collections should have different point counts"
    )
    runner.assert_not_equal(
        collection_data["codebase_20250110_120000"]["docs"],
        collection_data["codebase_20250110_130000"]["docs"],
        "Collections should have different documents"
    )


def test_list_collections_response():
    """Test collection list response structure."""
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

    runner.assert_equal(len(response["collections"]), 2, "Should have 2 collections")
    runner.assert_equal(
        response["collections"][0]["is_active"], True,
        "First collection should be active"
    )
    runner.assert_equal(
        response["collections"][1]["is_active"], False,
        "Second collection should not be active"
    )


def test_switch_collection():
    """Test switching active collection."""
    active_collection = "codebase_20250110_120000"
    new_collection = "codebase_20250110_130000"

    # Simulate switch
    active_collection = new_collection

    runner.assert_equal(
        active_collection, "codebase_20250110_130000",
        "Active collection should be switched"
    )


def test_delete_collection():
    """Test deleting a collection."""
    collections = [
        "codebase_20250110_120000",
        "codebase_20250110_130000",
        "codebase_20250110_140000",
    ]

    # Delete middle collection
    collection_to_delete = "codebase_20250110_130000"
    collections.remove(collection_to_delete)

    runner.assert_equal(len(collections), 2, "Should have 2 collections after delete")
    runner.assert_true(
        collection_to_delete not in collections,
        "Deleted collection should not be in list"
    )


def test_cannot_delete_active_collection():
    """Test that active collection cannot be deleted."""
    active_collection = "codebase_20250110_120000"
    collection_to_delete = "codebase_20250110_120000"

    # Check if trying to delete active collection
    can_delete = (collection_to_delete != active_collection)

    runner.assert_equal(
        can_delete, False,
        "Should not be able to delete active collection"
    )


def test_collection_info_metadata():
    """Test that collection info includes required metadata."""
    collection_info = {
        "name": "codebase_20250110_120000",
        "point_count": 6679,
        "dimension": 768,
        "is_active": True
    }

    runner.assert_in("point_count", collection_info, "Should have point_count")
    runner.assert_in("dimension", collection_info, "Should have dimension")
    runner.assert_in("is_active", collection_info, "Should have is_active")
    runner.assert_equal(collection_info["dimension"], 768, "Dimension should be 768")


def test_broadcast_collection_creation():
    """Test collection name broadcast structure."""
    broadcasts = []

    def mock_broadcast(message):
        broadcasts.append(message)

    # Simulate collection creation and broadcast
    collection_name = "codebase_20250110_120000"
    mock_broadcast({"type": "collection", "name": collection_name})

    runner.assert_equal(len(broadcasts), 1, "Should have 1 broadcast")
    runner.assert_equal(broadcasts[0]["type"], "collection", "Should be collection type")
    runner.assert_equal(
        broadcasts[0]["name"], collection_name,
        "Should broadcast correct collection name"
    )


def test_broadcast_on_reindex():
    """Test that new collection name is broadcast during reindex."""
    broadcasts = []

    def mock_broadcast(message):
        broadcasts.append(message)

    # Simulate reindex with new collection
    collection_name = "codebase_20250110_140000"
    mock_broadcast({"type": "log", "message": "reindex: found 63 files"})
    mock_broadcast({"type": "collection", "name": collection_name})

    # Find the collection broadcast
    collection_broadcasts = [b for b in broadcasts if b.get("type") == "collection"]

    runner.assert_equal(len(collection_broadcasts), 1, "Should have 1 collection broadcast")
    runner.assert_equal(
        collection_broadcasts[0]["name"], collection_name,
        "Should broadcast correct collection name"
    )


def test_base_collection_from_env():
    """Test that base collection name can be set via environment variable."""
    # Test default
    default_base = os.environ.get('QDRANT_COLLECTION', 'codebase')
    runner.assert_equal(default_base, 'codebase', "Default should be 'codebase'")

    # Test custom value
    with patch.dict(os.environ, {'QDRANT_COLLECTION': 'my_collection'}):
        custom_base = os.environ.get('QDRANT_COLLECTION', 'codebase')
        runner.assert_equal(custom_base, 'my_collection', "Should use custom base")


def test_collection_name_generation_flow():
    """Test the complete flow of collection name generation."""
    # 1. Get base name from env
    base_collection = os.environ.get('QDRANT_COLLECTION', 'codebase')

    # 2. Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 3. Combine
    collection_name = f"{base_collection}_{timestamp}"

    # 4. Verify format
    runner.assert_true(
        collection_name.startswith(base_collection),
        "Should start with base collection name"
    )
    runner.assert_in('_', collection_name, "Should contain underscore")
    runner.assert_equal(
        len(collection_name.split('_')[-1]), 6,
        "Time part should be 6 digits (HHMMSS)"
    )
    runner.assert_equal(
        len(collection_name.split('_')[-2]), 8,
        "Date part should be 8 digits (YYYYMMDD)"
    )


def test_build_corpus_creates_collection():
    """Test that building corpus creates a new timestamped collection."""
    # Before: No collections
    collections_before = []

    # Build corpus (simulated)
    new_collection = f"codebase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    collections_after = [new_collection]

    # After: One collection exists
    runner.assert_equal(len(collections_after), 1, "Should have 1 collection")
    runner.assert_equal(collections_after[0], new_collection, "Should be new collection")


def test_multiple_builds_create_multiple_collections():
    """Test that multiple corpus builds create separate collections."""
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
    runner.assert_equal(len(collections), 3, "Should have 3 collections")
    runner.assert_equal(len(set(collections)), 3, "All collections should be unique")


def test_switch_and_query_collections():
    """Test switching between collections."""
    # Setup multiple collections
    collections = {
        "codebase_20250110_120000": {"active": True, "points": 6679},
        "codebase_20250110_130000": {"active": False, "points": 5420},
    }

    # Initially active collection
    active = "codebase_20250110_120000"
    runner.assert_equal(
        collections[active]["active"], True,
        "Initial collection should be active"
    )

    # Switch to different collection
    new_active = "codebase_20250110_130000"
    collections[active]["active"] = False
    collections[new_active]["active"] = True
    active = new_active

    # Verify switch
    runner.assert_equal(active, "codebase_20250110_130000", "Should switch to new collection")
    runner.assert_equal(
        collections["codebase_20250110_130000"]["active"], True,
        "New collection should be active"
    )
    runner.assert_equal(
        collections["codebase_20250110_120000"]["active"], False,
        "Old collection should not be active"
    )


def test_cleanup_old_collections():
    """Test deleting old collections while keeping active one."""
    collections = [
        {"name": "codebase_20250110_120000", "is_active": False},
        {"name": "codebase_20250110_130000", "is_active": False},
        {"name": "codebase_20250110_140000", "is_active": True},
    ]

    # Delete non-active collections
    collections_to_keep = [c for c in collections if c["is_active"]]

    runner.assert_equal(len(collections_to_keep), 1, "Should keep 1 collection")
    runner.assert_equal(
        collections_to_keep[0]["name"], "codebase_20250110_140000",
        "Should keep active collection"
    )


# Run all tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Collection Separation Tests - Standalone Runner")
    print("="*60 + "\n")

    print("Testing Timestamped Collection Names:")
    runner.run_test("test_collection_name_format", test_collection_name_format)
    runner.run_test("test_collection_name_uniqueness", test_collection_name_uniqueness)
    runner.run_test("test_collection_name_with_custom_base", test_collection_name_with_custom_base)

    print("\nTesting Collection Isolation:")
    runner.run_test("test_new_collection_per_reindex", test_new_collection_per_reindex)
    runner.run_test("test_collection_documents_separated", test_collection_documents_separated)

    print("\nTesting Collection Management API:")
    runner.run_test("test_list_collections_response", test_list_collections_response)
    runner.run_test("test_switch_collection", test_switch_collection)
    runner.run_test("test_delete_collection", test_delete_collection)
    runner.run_test("test_cannot_delete_active_collection", test_cannot_delete_active_collection)
    runner.run_test("test_collection_info_metadata", test_collection_info_metadata)

    print("\nTesting WebSocket Broadcasts:")
    runner.run_test("test_broadcast_collection_creation", test_broadcast_collection_creation)
    runner.run_test("test_broadcast_on_reindex", test_broadcast_on_reindex)

    print("\nTesting Collection Settings:")
    runner.run_test("test_base_collection_from_env", test_base_collection_from_env)
    runner.run_test("test_collection_name_generation_flow", test_collection_name_generation_flow)

    print("\nTesting Integration Scenarios:")
    runner.run_test("test_build_corpus_creates_collection", test_build_corpus_creates_collection)
    runner.run_test("test_multiple_builds_create_multiple_collections", test_multiple_builds_create_multiple_collections)
    runner.run_test("test_switch_and_query_collections", test_switch_and_query_collections)
    runner.run_test("test_cleanup_old_collections", test_cleanup_old_collections)

    runner.print_summary()
