"""
Test script to diagnose Supermemory API 404 errors.
"""
import os
from supermemory import Supermemory

def test_supermemory():
    print("=== Testing Supermemory SDK ===\n")

    # Initialize client
    api_key = os.getenv("SUPERMEMORY_API_KEY")
    base_url = os.getenv("SUPERMEMORY_BASE_URL", "https://api.supermemory.ai")

    print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT SET")
    print(f"Base URL: {base_url}\n")

    try:
        client = Supermemory(api_key=api_key, base_url=base_url)
        print(f"[OK] Client initialized")
        print(f"   Base URL from client: {client.base_url}\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize client: {e}\n")
        return

    # Test 1: Add a memory
    print("Test 1: Adding a memory...")
    try:
        result = client.memories.add(
            content="This is a test memory to verify API connectivity",
            container_tags=["test_404_fix"],
            metadata={"test": True, "purpose": "debugging"}
        )
        print(f"[OK] Memory added successfully")
        print(f"   ID: {result.id if hasattr(result, 'id') else 'N/A'}")
        print(f"   Response: {result}\n")
    except Exception as e:
        print(f"[ERROR] Failed to add memory:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Full error: {repr(e)}\n")

    # Test 2: Search memories
    print("Test 2: Searching memories...")
    try:
        results = client.search.memories(
            q="test memory",
            container_tag="test_404_fix",
            limit=5,
            threshold=0.6,
            rerank=True
        )
        print(f"[OK] Search completed successfully")
        if hasattr(results, 'memories'):
            print(f"   Found {len(results.memories)} memories")
            for i, memory in enumerate(results.memories[:3], 1):
                print(f"   Memory {i}: {memory.content[:50]}...")
        else:
            print(f"   Results: {results}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to search memories:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Full error: {repr(e)}\n")

if __name__ == "__main__":
    test_supermemory()
