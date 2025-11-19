#!/usr/bin/env python3
"""Check Qdrant collection status"""

from qdrant_client import QdrantClient
import os

# Connect to Qdrant
qdrant_path = os.path.join(".fungus_cache", "qdrant")
client = QdrantClient(path=qdrant_path)

# List all collections
collections = client.get_collections()
print("Available collections:")
print("=" * 60)

if not collections.collections:
    print("  [EMPTY] No collections found!")
else:
    for coll in collections.collections:
        print(f"\n[Collection: {coll.name}]")

        # Get collection info
        try:
            info = client.get_collection(coll.name)
            print(f"  Points count: {info.points_count}")
            print(f"  Vector size: {info.config.params.vectors.size}")
            print(f"  Distance: {info.config.params.vectors.distance}")

            if info.points_count == 0:
                print(f"  [WARNING] Collection is EMPTY!")
            else:
                # Sample a few points
                sample = client.scroll(
                    collection_name=coll.name,
                    limit=3,
                    with_payload=True
                )
                print(f"  Sample points: {len(sample[0])}")
                for i, point in enumerate(sample[0], 1):
                    payload = point.payload
                    file_path = payload.get('file_path', 'unknown')
                    print(f"    {i}. ID={point.id}, file={file_path}")

        except Exception as e:
            print(f"  [ERROR] {e}")

print("\n" + "=" * 60)
print("\nRecent run collections:")
print("=" * 60)

# Check for recent run directories
runs_dir = os.path.join(".fungus_cache", "runs")
if os.path.exists(runs_dir):
    run_dirs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))], reverse=True)[:5]

    for run_id in run_dirs:
        # Check if this collection exists
        exists = any(coll.name == run_id for coll in collections.collections)
        print(f"  {run_id}: {'[EXISTS]' if exists else '[MISSING]'}")
