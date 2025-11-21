# Supermemory Integration Fixes

This document describes the fixes applied to resolve Supermemory API integration issues and improve the LangChain Memory Agent behavior.

## Problem Summary

The LangChain Memory Agent was failing with **404 Not Found** errors when trying to add memories to Supermemory, and was generating **generic search queries** that didn't provide meaningful context.

---

## Root Cause Analysis

### Issue 1: Supermemory SDK 404 Errors

**Symptom:**
```
[LANGCHAIN-AGENT] Error processing iteration: <html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx</center>
</body>
</html>
```

**Root Cause:**
The Supermemory Python SDK's `memories.add()` method does **NOT** accept a `type` parameter. The actual SDK signature is:
```python
(*, content, container_tag, container_tags, custom_id, metadata, ...)
```

Passing the unsupported `type` parameter caused the SDK to call invalid API endpoints, resulting in nginx 404 responses.

**Verification:**
```python
from supermemory import Supermemory
import inspect
c = Supermemory(api_key='test')
print(inspect.signature(c.memories.add))
# Output: NO 'type' parameter in signature
```

### Issue 2: Agent Caching Preventing Code Updates

**Symptom:**
Code changes to `supermemory_client.py` and `langchain_memory_agent.py` weren't being applied even after server reload.

**Root Cause:**
The `LangChainMemoryAgent` instance and its `SupermemoryManagerSync` were created once and **cached indefinitely**:
- `/reset` endpoint did NOT clear `langchain_agent`
- `start()` method did NOT reset the agent
- Restarting exploration reused the old cached agent with stale code

### Issue 3: Generic Search Queries

**Symptom:**
Agent generating queries like `"Classify the code into modules"` instead of specific technical queries.

**Root Cause:**
- Prompt instructions weren't explicit enough about query requirements
- Code chunk summaries didn't include function/class names for context
- No runtime validation to reject generic queries

---

## Fixes Applied

### Fix 1: Remove `type` Parameter from SDK Calls

**File:** `src/embeddinggemma/memory/supermemory_client.py`

**Changes (Lines 975-980, 1025-1030):**
```python
# BEFORE (broken):
response = self.client.memories.add(
    content=content,
    type=type,  # <-- SDK doesn't support this!
    metadata=metadata or {},
    custom_id=custom_id,
    container_tags=[container_tag] if container_tag else [],
)

# AFTER (fixed):
response = self.client.memories.add(
    content=content,
    metadata=metadata or {},
    custom_id=custom_id,
    container_tags=[container_tag] if container_tag else [],
    # NOTE: SDK doesn't support 'type' parameter, removed it
)
```

### Fix 2: Force Agent Recreation on Exploration Start

**File:** `src/embeddinggemma/realtime/server.py`

**Change 1 - `start()` method (Line 1395):**
```python
# Clear LangChain agent to force recreation with latest code and new container_tag
self.langchain_agent = None
```

**Change 2 - `_llm_judge()` method (Lines 1031-1037):**
```python
# Create/recreate agent if: (1) not initialized, (2) run_id changed (new exploration session)
needs_recreation = (
    self.langchain_agent is None or
    (self.langchain_agent and getattr(self.langchain_agent, 'container_tag', None) != self.run_id)
)

if needs_recreation and self.memory_manager and self.memory_manager.enabled and langchain_enabled:
    # ... create new agent
```

### Fix 3: Clear Agent on Reset

**File:** `src/embeddinggemma/realtime/routers/simulation.py`

**Change (Line 407):**
```python
# Clear LangChain agent to force recreation with latest code
streamer.langchain_agent = None
```

### Fix 4: Improved Agent Prompts for Specific Queries

**File:** `src/embeddinggemma/agents/langchain_memory_agent.py`

**Change 1 - Search Query Rules (Lines 151-159):**
```python
SEARCH QUERY RULES (CRITICAL):
- NEVER use generic queries like: "Classify the code", "modules", "architecture"
- ALWAYS use specific queries with concrete references:
  * File paths: "server.py initialization", "agents/langchain_memory_agent.py"
  * Function names: "start_exploration function", "add_memory implementation"
  * Technical terms: "FastAPI WebSocket routes", "Qdrant vector database setup"
  * Specific patterns: "async/await in realtime server", "LangChain agent tools"
- If you don't know what to search for, use the judge's insights or code chunk file paths as query terms
- Example: Instead of "Classify modules" -> "src/embeddinggemma/realtime/server.py WebSocket handling"
```

**Change 2 - Enhanced Code Chunk Summaries (Lines 398-430):**
```python
# Now extracts function/class names from code chunks:
# Output: "1. src/server.py (class SnapshotStreamer, async def start_exploration, def process_chunk)"
```

**Change 3 - Enhanced Judge Results (Lines 432-455):**
- Increased context from 100 to 200 characters
- Added header: "Key Insights (use these terms in your searches):"

---

## Configuration Changes

### `.env` / `.env.example`

Added Supermemory base URL configuration:
```bash
SUPERMEMORY_API_KEY=sm_your_api_key_here
SUPERMEMORY_BASE_URL=https://api.supermemory.ai
```

---

## Testing

### Test Script: `test_supermemory_api.py`

Created a test script to verify Supermemory SDK connectivity:

```python
from supermemory import Supermemory

client = Supermemory(api_key=api_key, base_url=base_url)

# Test 1: Add memory (without type parameter)
result = client.memories.add(
    content="Test memory",
    container_tags=["test"],
    metadata={"test": True}
)
# Expected: MemoryAddResponse(id='...', status='queued')

# Test 2: Search memories
results = client.search.memories(
    q="test memory",
    container_tag="test",
    limit=5,
    threshold=0.6,
    rerank=True
)
```

**Test Results:**
```
[OK] Memory added successfully
   ID: hSapu2Mhh1oj4ijNAAy7rJ
   Response: MemoryAddResponse(id='hSapu2Mhh1oj4ijNAAy7rJ', status='queued')
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/embeddinggemma/memory/supermemory_client.py` | Removed `type` parameter from SDK calls, added `base_url` support |
| `src/embeddinggemma/agents/langchain_memory_agent.py` | Improved prompts, enhanced code chunk/judge summaries |
| `src/embeddinggemma/realtime/server.py` | Agent cache clearing in `start()` and `_llm_judge()` |
| `src/embeddinggemma/realtime/routers/simulation.py` | Agent cache clearing in `/reset` endpoint |
| `.env.example` | Added `SUPERMEMORY_BASE_URL` |
| `test_supermemory_api.py` | New test script for SDK verification |

---

## How It Works Now

1. **On `/start` or `/reset`**: `langchain_agent` is set to `None`
2. **On first judge call**: Agent is recreated with fresh code and correct `container_tag`
3. **Memory operations**: SDK calls use correct parameters (no `type`)
4. **Search queries**: Agent is guided to use specific file paths, function names, and technical terms

---

## Future Improvements

- [ ] Add runtime query validation to reject generic queries
- [ ] Implement query enhancement to auto-append context from code chunks
- [ ] Add memory deduplication based on semantic similarity
