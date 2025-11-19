#!/usr/bin/env python3
"""Check if the retriever has documents loaded via the API"""

import requests

API_BASE = "http://localhost:8011"

print("Checking retriever status...")
print("=" * 60)

# Try to get corpus summary
try:
    response = requests.get(f"{API_BASE}/corpus/summary")

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Corpus summary retrieved")
        print(f"     Vector backend: {data.get('vector_backend', 'unknown')}")
        print(f"     Qdrant collection: {data.get('qdrant_collection', 'none')}")
        print(f"     Qdrant points: {data.get('qdrant_points', 0)}")
        print(f"     Simulation docs: {data.get('simulation_docs', 0)}")
        print(f"     Run ID: {data.get('run_id', 'none')}")

        sim_docs = data.get('simulation_docs', 0)
        if sim_docs > 0:
            print(f"\n[SUCCESS] Retriever has {sim_docs} documents loaded!")
            print("\nYou can now run your simulation and it will retrieve actual code.")
        else:
            print("\n[WARNING] No documents in retriever")
            print("           This means the reindex may have been lost (server restart?)")
    else:
        print(f"[ERROR] HTTP {response.status_code}")
        print(f"        {response.text}")

except Exception as e:
    print(f"[ERROR] {e}")
    print("\nIs the server running?")

print("\n" + "=" * 60)
print("To run a simulation:")
print("  1. Go to http://localhost:5174")
print("  2. Enter a query in the Simulation tab")
print("  3. Click 'Start'")
print("  4. Wait for completion")
print("  5. Check .fungus_cache/runs/{run_id}/summary.json for results")
