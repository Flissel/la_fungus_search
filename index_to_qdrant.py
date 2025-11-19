#!/usr/bin/env python3
"""Index codebase to Qdrant (not in-memory)"""

import requests
import time

API_BASE = "http://localhost:8011"

print("Indexing codebase to Qdrant...")
print("=" * 60)

# Trigger Qdrant indexing
response = requests.post(
    f"{API_BASE}/corpus/index_repo",
    json={"root": "src", "exclude_dirs": []}
)

if response.status_code == 200:
    result = response.json()
    print(f"[OK] Indexing triggered")
    print(f"     Status: {result.get('status')}")
    print(f"     Files: {result.get('files')}")
    print(f"     Points: {result.get('points')}")

    if result.get('status') == 'ok':
        print("\n[SUCCESS] Codebase indexed to Qdrant!")

        # Verify
        print("\nVerifying Qdrant collection...")
        time.sleep(2)

        from qdrant_client import QdrantClient
        import os

        qdrant_path = os.path.join(".fungus_cache", "qdrant")
        client = QdrantClient(path=qdrant_path)

        collections = client.get_collections()
        if collections.collections:
            print(f"\n[OK] Found {len(collections.collections)} collection(s):")
            for coll in collections.collections:
                info = client.get_collection(coll.name)
                print(f"     - {coll.name}: {info.points_count} points")
        else:
            print("\n[WARNING] No collections found")

        # Check retriever status
        print("\nChecking retriever status...")
        status_response = requests.get(f"{API_BASE}/corpus/summary")
        if status_response.status_code == 200:
            data = status_response.json()
            print(f"[OK] Simulation docs: {data.get('simulation_docs', 0)}")
            print(f"     Qdrant points: {data.get('qdrant_points', 0)}")
    else:
        print(f"\n[ERROR] Indexing failed: {result}")

else:
    print(f"[ERROR] HTTP {response.status_code}")
    print(f"        {response.text}")
