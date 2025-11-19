# Indexing Issue - No Documents in Collection

## Root Cause

The simulation was running but retrieving **0 documents** because the Qdrant collection was **empty**. No codebase indexing had been performed.

## Evidence

```
Available collections:
============================================================
  [EMPTY] No collections found!
```

Server logs showed:
- Collection creation: `ensure_collection: name=la_fungus_search_20251112_150202 dim=3072`
- But no retrieval results (all searches returned empty)
- LLM was generating responses without any code context

## Solution

**You must index your codebase BEFORE running the simulation.**

### Method 1: Using the Frontend (Easiest)

1. Open the Corpus tab: http://localhost:5174 (navigate to Corpus section)
2. Click the **"Reindex"** button
3. Wait for indexing to complete (watch the logs)
4. Once complete, run your simulation

### Method 2: Using the API

```bash
curl -X POST http://localhost:8011/corpus/reindex \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

### Method 3: Using the Script

```bash
python index_codebase.py
```

## What Happens During Indexing

1. **File Discovery** - Scans `src/` directory for code files
2. **Chunking** - Breaks files into semantic chunks
3. **Embedding** - Generates vector embeddings using OpenAI `text-embedding-3-large`
4. **Storage** - Stores vectors in Qdrant collection
5. **Initialization** - Prepares retriever for search

## Expected Output

After successful indexing:

```
[OK] Reindex triggered
     Status: ok
     Changed: True
     Docs: 1234  # Number of chunks indexed

[SUCCESS] Codebase indexed successfully!

[OK] Latest collection: la_fungus_search_20251112_152410
     Points: 1234
     Vector size: 3072
```

## Verification

Check if indexing worked:

```bash
python check_collection.py
```

Should show:
- Collection with matching name
- Non-zero points count
- Sample documents with file paths

## Important Notes

- **Indexing takes time** - Large codebases can take several minutes
- **API key required** - Make sure `OPENAI_API_KEY` is set in your `.env`
- **Automatic collection naming** - Each reindex creates a new collection with timestamp
- **Previous collections** - Old collections remain in Qdrant until manually cleaned up

## Workflow

The correct workflow is:

1. **Start server** → `python run-realtime.ps1 -Port 8011`
2. **Index codebase** → Use Corpus tab or API
3. **Wait for completion** → Watch logs for "reindex: complete"
4. **Run simulation** → Use Simulation tab
5. **Stop simulation** → Get per-run summary in `.fungus_cache/runs/{run_id}/summary.json`

## Why This Happened

The server starts without indexing by default. This allows you to:
- Configure settings first
- Choose what to index
- Avoid unnecessary re-indexing on every startup

But it means **you must explicitly trigger indexing** before running simulations.
