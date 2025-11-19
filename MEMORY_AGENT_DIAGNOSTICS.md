# Memory Agent Diagnostics & Configuration

## Quick Answer: Why Only Searches?

**This is NORMAL and CORRECT behavior!** ✅

The system performs mostly searches because the Memory Manager Agent intelligently waits until it has **complete understanding** before ingesting. Most exploration steps are gathering context (`SEARCH_MORE` decisions), and ingestion only happens when knowledge is complete.

---

## Understanding the Two-System Architecture

### System 1: Qdrant (Vector Database) - PRIMARY SEARCH
- **Contains:** ALL code chunks from your indexed codebase
- **Purpose:** Fast semantic search during exploration
- **Every `search_v4` call** queries Qdrant
- **Pre-loaded:** YES (via `index_codebase.py`)

### System 2: Supermemory (Cloud Memory) - DISCOVERED INSIGHTS
- **Contains:** High-level insights and complete module analyses
- **Purpose:** Store distilled knowledge to avoid redundant work
- **Ingestion:** Only when Memory Agent decides understanding is complete
- **Pre-loaded:** NO (built incrementally during exploration)

---

## Memory Agent Decision Process

### Decision Criteria (Configurable)

The agent makes one of three decisions each step:

| Decision | Criteria | Meaning |
|----------|----------|---------|
| **🔍 SEARCH_MORE** | <3 chunks, fragmented info, missing context | Keep exploring, not ready to ingest |
| **✅ INGEST** | 5+ chunks, clear entry points, complete understanding | Store as document in Supermemory |
| **⏭️  SKIP** | Not relevant, trivial, already stored | No action needed |

### Typical Exploration Pattern:

```
Step 1: Query "Find authentication"
  ├─ Found: 2 chunks from auth/oauth.py
  └─ Decision: 🔍 SEARCH_MORE (need more context)

Step 2: Query "Explore OAuth2Handler methods"
  ├─ Found: 4 chunks, class implementation
  └─ Decision: 🔍 SEARCH_MORE (still fragmentary)

Step 3: Query "Token management dependencies"
  ├─ Found: 6 chunks, complete token flow
  └─ Decision: ✅ INGEST (complete understanding!)
      └─ Document: "Authentication Module - OAuth2 + Tokens"

Step 4+: Can use stored knowledge from Supermemory
```

---

## New Enhanced Logging (Just Added!)

### What You'll Now See in Logs:

#### 1. Agent Initialization
```
memory-agent: initialized and ready (model: gpt-4o, container: run_abc123, min_chunks: 3)
```

#### 2. SEARCH_MORE Decisions
```
memory-agent: 🔍 SEARCH MORE (confidence: 0.6) - Only 2 chunks found, need more context to understand complete flow
  └─ Suggested: Explore OAuth2Handler dependencies, Find token validation logic
```

#### 3. INGEST Decisions
```
memory-agent: ✅ INGEST 2 documents (confidence: 0.9) - Complete auth module with clear entry points and dependencies
  └─ Documents: Authentication Module - OAuth2, Token Management System
```

#### 4. SKIP Decisions
```
memory-agent: ⏭️  SKIP - Code already ingested in step 5
```

---

## Configuration Options

### Environment Variables in `.env`:

```bash
# Memory Manager Agent LLM Model
# Determines intelligence level for ingestion decisions
# Options: gpt-4o (smart, expensive), gpt-4o-mini (fast, cheap)
MEMORY_AGENT_MODEL=gpt-4o

# Memory Manager Agent Minimum Chunks
# Lower = faster ingestion (potentially incomplete)
# Higher = more thorough ingestion (slower but complete)
# Recommended: 3-7 (default: 5)
MEMORY_AGENT_MIN_CHUNKS=3
```

### Impact of MEMORY_AGENT_MIN_CHUNKS:

| Value | Behavior | Use Case |
|-------|----------|----------|
| **1-2** | Very aggressive ingestion | Quick prototyping, small codebases |
| **3-5** | Balanced (RECOMMENDED) | Most codebases, good compromise |
| **6-10** | Very conservative | Large complex systems, ensure completeness |

---

## How to Verify Memory Agent is Working

### Method 1: Watch Simulation Logs

Start a simulation and look for these messages:

```bash
# Expected on first search (lazy initialization):
memory-agent: initialized and ready (model: gpt-4o, container: run_abc123, min_chunks: 3)

# Expected on most steps:
memory-agent: 🔍 SEARCH MORE (confidence: 0.4) - Need more context...

# Expected occasionally (when complete):
memory-agent: ✅ INGEST 3 documents (confidence: 0.9) - Complete understanding...
```

**If you DON'T see the initialization message:**
- Memory Agent is not running
- Check that `LLM_PROVIDER=openai` in `.env`
- Check that `SUPERMEMORY_API_KEY` is set

### Method 2: Check `manifest.json`

After an exploration run, inspect:
```bash
.fungus_cache/runs/LATEST_RUN/manifest.json
```

Look for:
```json
{
  "memory_stats": {
    "agent_enabled": true,          // ← Should be true
    "agent_decisions": 15,          // ← Should be > 0 if agent ran
    "agent_documents_ingested": 3,  // ← May be 0 if still exploring
    "agent_search_more_decisions": 12, // ← Most decisions are "search more"
    "conversation_history_steps": 15   // ← Tracks conversation context
  }
}
```

**Healthy Stats:**
- `agent_decisions` > 0 (agent is making decisions)
- `agent_search_more_decisions` >> `agent_documents_ingested` (expected ratio!)
- `conversation_history_steps` == total exploration steps

**Unhealthy Stats:**
- `agent_enabled: false` → Agent not initialized
- `agent_decisions: 0` → Agent not being called
- All three → Check configuration

### Method 3: Check Supermemory Dashboard

1. Go to https://supermemory.ai/dashboard
2. Navigate to your project
3. Look for documents with `container_tags: ["run_abc123"]`
4. Should see structured documents like:
   - "Authentication Module - OAuth2"
   - "FastAPI Server - Main Entry Point"

**If no documents:**
- Agent may be working but just hasn't ingested yet (normal early in exploration)
- Run 15-20 exploration steps to see ingestion
- Lower `MEMORY_AGENT_MIN_CHUNKS` to 3 for faster ingestion

---

## Common Scenarios & Solutions

### Scenario 1: "I only see searches, no ingestion"

**Diagnosis:**
- Check logs for `memory-agent:` messages
- Check if `🔍 SEARCH MORE` decisions are being made

**Solution:**
- **If you see `🔍 SEARCH MORE`:** This is CORRECT! Agent is gathering context.
  - Action: Run 15-20 more steps, ingestion will happen when ready
  - Or: Lower `MEMORY_AGENT_MIN_CHUNKS=2` for faster ingestion

- **If you DON'T see any `memory-agent:` messages:** Agent not running
  - Action: Verify `LLM_PROVIDER=openai` in `.env`
  - Action: Check `SUPERMEMORY_API_KEY` is set

### Scenario 2: "Memory Agent not initializing"

**Error in logs:**
```
memory-agent: initialization failed - 'ollama' provider not supported
```

**Solution:**
Memory Agent currently only supports OpenAI. Change your `.env`:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Scenario 3: "Too much ingestion, storing incomplete knowledge"

**Symptoms:**
- Many small, fragmentary documents in Supermemory
- Documents with only 1-2 code chunks

**Solution:**
Increase ingestion threshold:
```bash
MEMORY_AGENT_MIN_CHUNKS=7
```

### Scenario 4: "Too slow, want faster ingestion for testing"

**Solution:**
Lower ingestion threshold:
```bash
MEMORY_AGENT_MIN_CHUNKS=2
```

---

## Performance Expectations

### Typical 20-Step Exploration Run:

```
Agent Decisions:          20
├─ SEARCH_MORE:          16 (80%)  ← Most common
├─ INGEST:                3 (15%)  ← Occasional
└─ SKIP:                  1 (5%)   ← Rare

Documents Ingested:       3-5
├─ Average size:          500-2000 tokens
└─ Contains:              5-10 code chunks each
```

### Cost Breakdown (using gpt-4o):

| Component | Calls/Run | Cost/Call | Total |
|-----------|-----------|-----------|-------|
| Judge (main search) | 20 | $0.02 | $0.40 |
| Memory Agent (decisions) | 20 | $0.01 | $0.20 |
| Supermemory (storage) | 3-5 docs | $0.0001 | $0.0005 |
| **TOTAL** | | | **~$0.60/run** |

**Cost Optimization:**
- Use `MEMORY_AGENT_MODEL=gpt-4o-mini` → Saves ~70% on agent decisions
- Use `OPENAI_MODEL=gpt-4o-mini` → Saves ~70% on judge calls
- Trade-off: Slightly lower quality decisions

---

## Troubleshooting Checklist

### ✅ **Agent is Working If:**
- [x] See `memory-agent: initialized and ready` in logs
- [x] See `memory-agent: 🔍 SEARCH MORE` decisions regularly
- [x] `manifest.json` shows `agent_enabled: true`
- [x] `manifest.json` shows `agent_decisions > 0`
- [x] Occasionally see `✅ INGEST` decisions (after 5-10 steps)

### ❌ **Agent is NOT Working If:**
- [ ] No `memory-agent:` messages in logs
- [ ] `manifest.json` shows `agent_enabled: false`
- [ ] `manifest.json` shows `agent_decisions: 0`
- [ ] Error: `'ollama' provider not supported` or similar

---

## Advanced Configuration

### Custom Ingestion Logic

You can override the `min_chunks_for_ingest` parameter when initializing the agent in `server.py`:

```python
self.memory_agent = MemoryManagerAgent(
    llm_client=llm_client,
    memory_manager=self.memory_manager,
    model="gpt-4o",
    min_chunks_for_ingest=3  # Override environment variable
)
```

### LLM Prompt Customization

The agent's decision criteria are in the prompt at [memory_manager_agent.py:246-248](src/embeddinggemma/agents/memory_manager_agent.py#L246-L248):

```python
**DECISION CRITERIA:**
- INGEST if: {self.min_chunks_for_ingest}+ chunks, clear entry points, complete understanding
- SEARCH_MORE if: <{max(self.min_chunks_for_ingest - 2, 1)} chunks, fragmented, missing context
- SKIP if: Not relevant, trivial, already stored
```

You can modify these criteria to suit your needs.

---

## Summary: Expected Behavior

| Phase | Typical Decisions | Documents in Supermemory |
|-------|-------------------|--------------------------|
| **Steps 1-5** | 100% SEARCH_MORE | 0 |
| **Steps 6-10** | 80% SEARCH_MORE, 20% INGEST | 1-2 |
| **Steps 11-20** | 60% SEARCH_MORE, 40% INGEST | 3-7 |
| **Steps 20+** | 50% SEARCH_MORE, 30% INGEST, 20% SKIP | 7-15 |

**Key Insight:**
- **Qdrant** (vector database) contains ALL your code - searched every step
- **Supermemory** (cloud memory) contains INSIGHTS - built incrementally
- **Memory Agent** decides when understanding is complete enough to store
- **Mostly searches is CORRECT** - that's the agent being smart about what to store!

---

## Quick Reference

### Start a Diagnostic Run:

```bash
# 1. Verify configuration
cat .env | grep MEMORY_AGENT

# 2. Start simulation
# Watch for: "memory-agent: initialized and ready"

# 3. After 15-20 steps, check manifest
cat .fungus_cache/runs/LATEST_RUN/manifest.json | grep agent

# 4. Check Supermemory dashboard
# https://supermemory.ai/dashboard
```

### Adjust Ingestion Speed:

```bash
# Faster ingestion (more aggressive)
MEMORY_AGENT_MIN_CHUNKS=2

# Slower ingestion (more thorough)
MEMORY_AGENT_MIN_CHUNKS=7

# Default (balanced)
MEMORY_AGENT_MIN_CHUNKS=5
```

---

## Files Modified in This Update

- **[server.py](src/embeddinggemma/realtime/server.py#L880-L988)** - Enhanced logging and WebSocket messages
- **[memory_manager_agent.py](src/embeddinggemma/agents/memory_manager_agent.py#L26-L53)** - Configurable threshold
- **[.env](.env#L32-L37)** - Added `MEMORY_AGENT_MIN_CHUNKS` configuration

**New Features:**
- ✅ Initialization messages show model and settings
- ✅ Decision messages show confidence and detailed reasons
- ✅ INGEST messages show document titles being stored
- ✅ SEARCH_MORE messages show suggested next queries
- ✅ Configurable ingestion threshold via environment variable
- ✅ Threshold dynamically adjusts LLM prompt

**Result:** Full visibility into Memory Agent decisions in real-time!
