"""
Test the CodebaseBootstrap functionality.

This script tests the bootstrap system without needing to run the full server.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from embeddinggemma.memory.supermemory_client import SupermemoryManager
from embeddinggemma.memory.codebase_bootstrap import CodebaseBootstrap


async def test_bootstrap():
    """Test bootstrap functionality."""
    print("=" * 60)
    print("TESTING CODEBASE BOOTSTRAP")
    print("=" * 60)

    # Initialize memory manager
    print("\n1. Initializing SupermemoryManager...")
    memory_manager = SupermemoryManager()

    # Check if memory manager is enabled
    if not memory_manager.enabled:
        print("[FAIL] Memory manager is not enabled!")
        print("   Make sure SUPERMEMORY_API_KEY and SUPERMEMORY_BASE_URL are set in .env")
        return False

    print(f"[OK] Memory manager enabled")
    print(f"   Base URL: {memory_manager.base_url}")

    # Create bootstrap instance
    print("\n2. Creating CodebaseBootstrap instance...")
    root_dir = os.getcwd()
    bootstrapper = CodebaseBootstrap(
        root_dir=root_dir,
        memory_manager=memory_manager
    )
    print(f"✅ Bootstrap created (root: {root_dir})")

    # Run bootstrap
    print("\n3. Running bootstrap...")
    container_tag = "test_bootstrap_container"
    result = await bootstrapper.bootstrap(container_tag=container_tag)

    if not result.get('success'):
        print(f"❌ Bootstrap failed: {result.get('error')}")
        return False

    print(f"✅ Bootstrap succeeded!")

    # Display results
    print("\n4. Bootstrap Results:")
    print(f"   Memories created: {result.get('memories_created', 0)}")

    module_tree = result.get('module_tree', {})
    print(f"   Modules found: {len(module_tree)}")
    if module_tree:
        print("\n   Module list:")
        for module_name in sorted(module_tree.keys())[:10]:
            info = module_tree[module_name]
            print(f"      - {module_name} ({info['file_count']} files, ~{info['total_lines']} lines)")
        if len(module_tree) > 10:
            print(f"      ... and {len(module_tree) - 10} more modules")

    entry_points = result.get('entry_points', [])
    print(f"\n   Entry points found: {len(entry_points)}")
    if entry_points:
        for ep in entry_points:
            print(f"      - {ep['file']} ({ep['type']})")

    # Test retrieval
    print("\n5. Testing memory retrieval...")
    memories = await memory_manager.search_memory(
        query="codebase_module_tree",
        container_tag=container_tag,
        limit=1
    )

    if memories:
        print(f"✅ Found codebase_module_tree memory")
        mem = memories[0]
        content = mem.get('content', '')
        print(f"   Content preview: {content[:200]}...")
    else:
        print(f"❌ Could not retrieve codebase_module_tree memory")
        return False

    # Test entry points retrieval
    memories = await memory_manager.search_memory(
        query="codebase_entry_points",
        container_tag=container_tag,
        limit=1
    )

    if memories:
        print(f"✅ Found codebase_entry_points memory")
        mem = memories[0]
        content = mem.get('content', '')
        print(f"   Content preview: {content[:200]}...")
    else:
        print(f"⚠️ Could not retrieve codebase_entry_points memory")

    print("\n" + "=" * 60)
    print("BOOTSTRAP TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_bootstrap())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
