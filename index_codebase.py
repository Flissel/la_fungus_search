#!/usr/bin/env python3
"""Index the codebase before running simulation"""

import requests
import time

API_BASE = "http://localhost:8011"

print("Triggering codebase reindex...")
print("=" * 60)

# Trigger reindex
response = requests.post(
    f"{API_BASE}/corpus/reindex",
    json={"force": True}
)

if response.status_code == 200:
    result = response.json()
    print(f"[OK] Reindex triggered")
    print(f"     Status: {result.get('status')}")
    print(f"     Changed: {result.get('changed')}")
    print(f"     Docs: {result.get('docs')}")

    if result.get('status') == 'ok':
        print("\n[SUCCESS] Codebase indexed successfully!")
        print("\nYou can now run your simulation.")

        # Verify collection exists
        print("\nVerifying collection...")
        time.sleep(2)  # Wait for collection to be created

        from qdrant_client import QdrantClient
        import os

        qdrant_path = os.path.join(".fungus_cache", "qdrant")
        client = QdrantClient(path=qdrant_path)

        collections = client.get_collections()
        if collections.collections:
            latest = sorted([c.name for c in collections.collections], reverse=True)[0]
            print(f"[OK] Latest collection: {latest}")

            info = client.get_collection(latest)
            print(f"     Points: {info.points_count}")
            print(f"     Vector size: {info.config.params.vectors.size}")
        else:
            print("[WARNING] No collections found yet. Check server logs.")
    else:
        print(f"\n[ERROR] Reindex failed: {result}")

else:
    print(f"[ERROR] HTTP {response.status_code}")
    print(f"        {response.text}")
    print("\nIs the server running? Check: http://localhost:8011/health")
