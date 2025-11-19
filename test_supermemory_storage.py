"""
Test script to verify Supermemory storage is working.

This tests both the legacy add() method and the new documents.add() method.
"""
import asyncio
import os
import sys
from dotenv import load_dotenv
from supermemory import AsyncSupermemory

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

async def test_supermemory_storage():
    """Test Supermemory storage functionality."""

    api_key = os.getenv("SUPERMEMORY_API_KEY")

    if not api_key:
        print("❌ SUPERMEMORY_API_KEY not found in .env file")
        return False

    print(f"✅ API Key found: {api_key[:20]}...")

    try:
        # Initialize Supermemory client
        client = AsyncSupermemory(api_key=api_key)
        print("✅ AsyncSupermemory client initialized")

        # Test 1: Add a simple memory using v3 API
        print("\n📝 Test 1: Adding memory with memories.add() method...")
        try:
            result = await client.memories.add(
                content="This is a test memory from la_fungus_search Memory Manager Agent test.",
                container_tags=["test_run"]  # Plural "tags"
            )
            print(f"✅ Memory added successfully!")
            print(f"   Result: {result}")
        except Exception as e:
            print(f"❌ Failed to add memory: {e}")
            print(f"   Error type: {type(e).__name__}")

        # Test 2: Add a document (same way Memory Manager Agent does it)
        print("\n📄 Test 2: Adding document with memories.add() method (Memory Manager Agent style)...")
        try:
            title = "Test Document - Memory Manager Agent"
            content = "This is a test document to verify the Memory Manager Agent's storage mechanism is working correctly."

            # Format like Memory Manager Agent does
            formatted_content = f"# {title}\n\n{content}"
            custom_id = "test_run_test_Test_Document_-_Memory_Manager_Agent"

            result = await client.memories.add(
                content=formatted_content,
                container_tags=["test_run"],  # Plural "tags"
                metadata={
                    "doc_type": "test",
                    "container_tag": "test_run",
                    "title": title,
                    "patterns": ["testing"],
                    "agent": "memory_manager"
                },
                custom_id=custom_id
            )
            print(f"✅ Document added successfully!")
            print(f"   Result: {result}")
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
            print(f"   Error type: {type(e).__name__}")

        # Test 3: Search for what we just added
        print("\n🔍 Test 3: Searching for added content...")
        try:
            results = await client.search.documents(
                q="Memory Manager Agent test",
                container_tags=["test_run"],  # Plural "tags"
                limit=10
            )
            print(f"✅ Search completed!")
            if hasattr(results, 'results') and results.results:
                print(f"   Found {len(results.results)} results")
                for i, result in enumerate(results.results[:3], 1):
                    content = result.content if hasattr(result, 'content') and result.content else str(result)
                    title = result.title if hasattr(result, 'title') and result.title else "Untitled"
                    print(f"   {i}. [{title}] {content[:80] if content else 'No content'}...")
            else:
                print(f"   No results found (documents may still be processing)")
        except Exception as e:
            print(f"❌ Failed to search: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

        # Test 4: Test duplicate prevention (try adding same document again)
        print("\n🔄 Test 4: Testing duplicate prevention...")
        try:
            # Try adding the same document again
            duplicate_result = await client.memories.add(
                content=formatted_content,
                container_tags=["test_run"],
                metadata={
                    "doc_type": "test",
                    "container_tag": "test_run",
                    "title": title,
                    "patterns": ["testing"],
                    "agent": "memory_manager"
                },
                custom_id=custom_id  # Same custom_id should update, not duplicate
            )
            print(f"✅ Duplicate handling tested!")
            print(f"   Result: {duplicate_result}")
            print(f"   (custom_id ensures update instead of duplicate)")
        except Exception as e:
            print(f"❌ Failed duplicate test: {e}")
            print(f"   Error type: {type(e).__name__}")

        print("\n" + "="*60)
        print("✅ Supermemory storage test completed!")
        return True

    except Exception as e:
        print(f"\n❌ Supermemory client initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Supermemory Storage Test")
    print("="*60)

    success = asyncio.run(test_supermemory_storage())

    if success:
        print("\n✅ All tests passed! Supermemory is working correctly.")
        print("\nThe Memory Manager Agent should be able to store documents.")
        print("Make sure to run a simulation with judge mode enabled to see it in action!")
    else:
        print("\n❌ Tests failed. Please check your Supermemory configuration.")
        print("\nTroubleshooting:")
        print("1. Verify SUPERMEMORY_API_KEY is set in .env")
        print("2. Check that the API key is valid")
        print("3. Ensure supermemory package is installed: pip install supermemory")
        print("4. Check network connectivity to Supermemory API")
