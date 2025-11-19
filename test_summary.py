#!/usr/bin/env python3
"""Quick test to verify create_run_summary works"""

from embeddinggemma.ui.reports import create_run_summary

# Test data
run_id = "test_run_123"
query = "test query"
result_items = [
    {
        "id": 1,
        "score": 0.95,
        "embedding_score": 0.92,
        "relevance_score": 0.94,
        "content": "Test content 1"
    },
    {
        "id": 2,
        "score": 0.85,
        "embedding_score": 0.82,
        "relevance_score": 0.84,
        "content": "Test content 2"
    }
]

try:
    summary_path = create_run_summary(run_id, query, result_items)
    print(f"✓ Summary created successfully at: {summary_path}")

    # Verify the file exists
    import os
    if os.path.exists(summary_path):
        print(f"✓ File exists")
        with open(summary_path, 'r') as f:
            import json
            data = json.load(f)
            print(f"✓ Summary contains {len(data.get('results', []))} results")
            print(f"✓ Run ID: {data.get('run_id')}")
            print(f"✓ Query: {data.get('query')}")
    else:
        print(f"✗ File NOT found at {summary_path}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
