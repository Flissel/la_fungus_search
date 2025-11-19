#!/usr/bin/env python3
"""Test that retrieval logging and summary creation work after fixes"""

import os
import json
import time

# Check if there are any run directories
cache_dir = ".fungus_cache"
runs_dir = os.path.join(cache_dir, "runs")

if not os.path.exists(runs_dir):
    print("✗ No runs directory found")
    exit(1)

# List all runs
run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
print(f"Found {len(run_dirs)} run directories:")

for run_id in run_dirs:
    run_path = os.path.join(runs_dir, run_id)
    print(f"\n[Run: {run_id}]")

    # Check for key files
    files_to_check = {
        "queries.jsonl": "Query log",
        "retrievals.jsonl": "Retrieval log",
        "manifest.json": "Manifest",
        "summary.json": "Summary",
        "run_costs.json": "Cost tracking"
    }

    for filename, description in files_to_check.items():
        filepath = os.path.join(run_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)

            # For JSONL files, count lines
            if filename.endswith('.jsonl'):
                with open(filepath, 'r') as f:
                    line_count = sum(1 for _ in f)
                print(f"  [OK] {description:20s} - {line_count} entries ({size} bytes)")
            # For JSON files, show structure
            elif filename.endswith('.json'):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        keys = list(data.keys())[:5]  # First 5 keys
                        print(f"  [OK] {description:20s} - {len(data)} keys ({size} bytes)")
                        if filename == "summary.json" and "results" in data:
                            print(f"       -> Contains {len(data['results'])} results")
                    elif isinstance(data, list):
                        print(f"  [OK] {description:20s} - {len(data)} items ({size} bytes)")
                except json.JSONDecodeError as e:
                    print(f"  [ERR] {description:20s} - Invalid JSON: {e}")
        else:
            print(f"  [MISSING] {description:20s}")

print("\n" + "="*60)
print("Summary:")
print("="*60)

# Count runs with summaries
runs_with_summaries = sum(1 for run_id in run_dirs
                         if os.path.exists(os.path.join(runs_dir, run_id, "summary.json")))
runs_with_retrievals = sum(1 for run_id in run_dirs
                          if os.path.exists(os.path.join(runs_dir, run_id, "retrievals.jsonl")))

print(f"Total runs: {len(run_dirs)}")
print(f"Runs with retrievals.jsonl: {runs_with_retrievals}")
print(f"Runs with summary.json: {runs_with_summaries}")

if runs_with_summaries > 0:
    print("\n[SUCCESS] Summaries are being created!")
else:
    print("\n[WARNING] No summaries found yet. Try running a simulation.")
