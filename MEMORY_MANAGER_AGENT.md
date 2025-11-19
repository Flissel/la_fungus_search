# Memory Manager Agent - Decoupled Architecture with Conversation Context

## Overview

The **Memory Manager Agent** is a dedicated LLM-powered component that handles all memory ingestion decisions, completely decoupled from the judge's search/retrieval logic. It has full access to the **conversation history** from the current run, enabling context-aware decisions about when to ingest knowledge vs. continue exploring.

**Key Features:**
- **Decoupled from Judge**: Clean separation between search and ingestion
- **Conversation-Aware**: Full context of exploration history (queries, discoveries, decisions)
- **LLM-Powered**: Smart decisions about completeness and value
- **Document-Based**: Structured knowledge storage using Supermemory `/add-document` API

**Architecture:**
- **Judge**: Evaluates code relevance and generates follow-up queries (search-only)
- **Memory Manager Agent**: Decides what to ingest based on conversation context and code analysis

## Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌──────────────────┐
│              │      │                  │      │                  │
│    Judge     │─────▶│  Memory Manager  │─────▶│   Supermemory    │
│              │      │      Agent       │      │                  │
│  (Search)    │      │  (Ingestion)     │      │   (Documents)    │
│              │      │                  │      │                  │
└──────────────┘      └──────────────────┘      └──────────────────┘
       │                       │
       │                       │
       ▼                       ▼
  Relevance              Ingest vs.
  Follow-up              Search More
  Keywords               Decisions
```

## Key Benefits

### 1. Separation of Concerns

**Before (Coupled):**
```python
Judge:
  - Evaluate relevance ✓
  - Generate queries ✓
  - Store insights ✓ (coupling!)
  - Track rooms ✓ (coupling!)
```

**After (Decoupled):**
```python
Judge:
  - Evaluate relevance ✓
  - Generate queries ✓

Memory Manager Agent:
  - Store insights ✓
  - Track rooms ✓
  - Decide ingest vs. search ✓
```

### 2. Conversation-Aware Decision Making

The Memory Manager Agent has full access to the **conversation history**, enabling smarter decisions:

**Conversation Context Includes:**
- **Exploration History**: All queries executed in this run
- **Discovery Pattern**: What was found at each step
- **Decision Trail**: Previous agent decisions (ingest vs. search more)
- **Progressive Understanding**: How knowledge builds over time

**How It Uses Context:**
- **Avoid Redundancy**: Don't ingest if we already stored similar information
- **Assess Completeness**: Use cumulative discoveries to decide if knowledge is complete
- **Understand Trajectory**: Know where exploration is heading to make better decisions
- **Detect Convergence**: Recognize when enough context has been gathered

**Example:**
```
Step 1: Query "Find authentication" → Found auth/oauth.py entry
Step 2: Query "Explore OAuth2Handler" → Found 3 methods, partial understanding
Step 3: Query "Token management" → Found token.py module

Agent Decision with Context:
"INGEST - We now have complete auth flow:
- OAuth2Handler (entry point)
- Token management (dependencies)
- 3 related modules discovered across steps
- Ready to document as complete auth subsystem"
```

### 3. Document-Based Storage

Uses Supermemory's `/add-document` API instead of flat insights:

```python
# Old approach (insights)
await memory_manager.add_insight(
    content="FastAPI server found",
    insight_type="entry_point",
    ...
)

# New approach (documents)
await memory_manager.add_document(
    title="FastAPI Server - Main Entry Point",
    content="Complete summary with context...",
    doc_type="room",
    metadata={
        "exploration_status": "fully_explored",
        "patterns": ["async/await", "WebSocket"],
        ...
    }
)
```

### 4. Scalability

Easy to add multiple specialized Memory Manager Agents:

- **Room Manager**: Specializes in room/module documentation
- **Relationship Manager**: Tracks dependencies and connections
- **Cluster Manager**: Identifies subsystems and architectural layers

## Implementation Details

### Files Created

1. **`src/embeddinggemma/agents/__init__.py`** (11 lines)
   - Module exports for agent subsystem

2. **`src/embeddinggemma/agents/memory_manager_agent.py`** (350+ lines)
   - `MemoryManagerAgent` class
   - LLM-powered ingestion decisions
   - Document structuring logic
   - Statistics tracking

3. **`MEMORY_MANAGER_AGENT.md`** (this file)
   - Architecture documentation
   - Usage examples
   - Integration guide

### Files Modified

1. **`src/embeddinggemma/memory/supermemory_client.py`**
   - Added `add_document()` - Store structured documents
   - Added `search_documents()` - Search documents
   - Added `get_document_by_title()` - Retrieve specific documents

2. **`src/embeddinggemma/realtime/server.py`**
   - Import `MemoryManagerAgent`
   - Initialize agent with LLM client
   - Call agent after judge evaluation
   - Broadcast agent decisions via WebSocket
   - Track agent stats in manifest

3. **`src/embeddinggemma/prompts/__init__.py`**
   - Simplified `judge_schema_hint()` to remove `insights_to_store`
   - Removed room/relationship insight examples
   - Added note about Memory Manager Agent handling ingestion

## How It Works

### Step-by-Step Flow

```
1. User Query: "Find authentication implementation"
   ↓
2. Judge evaluates code chunks
   ├─ is_relevant: true/false
   ├─ entry_point: true/false
   ├─ follow_up_queries: [...]
   ├─ keywords: [...]
   └─ suggested_top_k: 25
   ↓
3. Memory Manager Agent analyzes results
   ├─ Input: code chunks + judge results
   ├─ Decision: INGEST vs. SEARCH_MORE vs. SKIP
   └─ Reason: "Complete understanding of auth module"
   ↓
4a. If INGEST: Create structured documents
    ├─ Title: "Authentication Module - OAuth2 Handler"
    ├─ Content: Complete summary + analysis
    ├─ Type: "room"
    ├─ Metadata: patterns, functions, classes
    └─ Store to Supermemory via /add-document
   ↓
4b. If SEARCH_MORE: Return suggested queries
    └─ ["Explore token validation logic", "Find user session management"]
```

### Decision Criteria

The Memory Manager Agent uses these criteria:

**INGEST** - When:
- ✅ 5+ code chunks from same file/module
- ✅ Clear entry points identified
- ✅ Complete understanding of purpose
- ✅ Key functions/classes extracted
- ✅ Dependencies mapped

**SEARCH_MORE** - When:
- ⚠️ <3 chunks (fragmented information)
- ⚠️ Missing critical context
- ⚠️ Incomplete implementation details
- ⚠️ Unclear relationships to other modules

**SKIP** - When:
- ❌ Not relevant to query
- ❌ Trivial/boilerplate code
- ❌ Already stored (redundant)

## Usage Example

### Simulation Run

```bash
# Start simulation with judge mode enabled
# Memory Manager Agent automatically activates
```

**WebSocket Messages:**

```javascript
// Judge evaluates chunks
{"type": "log", "message": "judge: generating..."}
{"type": "log", "message": "judge: parsed=15"}

// Memory Manager Agent makes decision
{"type": "log", "message": "memory-agent: ingested 2 documents (Complete auth module understanding)"}

// Manifest updated with stats
{
  "memory_stats": {
    "agent_enabled": true,
    "agent_decisions": 5,
    "agent_documents_ingested": 8,
    "agent_search_more_decisions": 2
  }
}
```

### Programmatic Access

```python
# Query ingested documents
documents = await memory_manager.search_documents(
    query="authentication",
    container_tag="run_20251115_140000",
    doc_type="room",
    limit=10
)

for doc in documents:
    print(f"Title: {doc['title']}")
    print(f"Type: {doc['metadata']['doc_type']}")
    print(f"Patterns: {doc['metadata']['patterns']}")
    print(f"---")
```

## Configuration

### Environment Variables

No additional configuration needed! The Memory Manager Agent:

- ✅ Uses existing `SUPERMEMORY_API_KEY`
- ✅ Uses existing LLM provider (`LLM_PROVIDER=openai`)
- ✅ Uses existing OpenAI model (defaults to `gpt-4o-mini` for decisions)

### Enabling/Disabling

The Memory Manager Agent is automatically enabled when:

1. Supermemory is enabled (`SUPERMEMORY_API_KEY` is set)
2. Judge mode is enabled during simulation
3. LLM provider is configured (OpenAI, Ollama, etc.)

To disable, simply don't set `SUPERMEMORY_API_KEY`.

## Statistics & Analytics

### Manifest Tracking

Each simulation run includes Memory Manager Agent statistics:

```json
{
  "run_id": "la_fungus_search_20251115_140000",
  "memory_stats": {
    "enabled": true,
    "insights_stored": 0,           // Legacy (now unused)
    "insights_retrieved": 15,
    "memory_queries": 10,
    "rooms_discovered": 5,           // RoomAnalyzer (still active)
    "rooms_fully_explored": 3,
    "rooms_partially_explored": 2,
    "agent_enabled": true,                    // NEW
    "agent_decisions": 8,                     // NEW
    "agent_documents_ingested": 12,           // NEW
    "agent_search_more_decisions": 3,         // NEW
    "conversation_history_steps": 10          // NEW: Conversation context tracking
  }
}
```

### Agent Stats

```python
# From MemoryManagerAgent
stats = memory_agent.get_stats()
# Returns:
{
  "enabled": True,
  "decisions_made": 8,
  "documents_ingested": 12,
  "search_more_decisions": 3
}
```

## API Reference

### MemoryManagerAgent

```python
class MemoryManagerAgent:
    """LLM-powered agent for memory ingestion decisions."""

    def __init__(
        self,
        llm_client=None,        # OpenAI AsyncClient or similar
        memory_manager=None     # SupermemoryManager instance
    ):
        """Initialize Memory Manager Agent."""

    async def analyze_and_decide(
        self,
        query: str,                                   # Current exploration query
        code_chunks: list[dict],                      # Retrieved code chunks
        judge_results: dict[int, dict] = None,        # Judge evaluation results
        container_tag: str = "default",               # Container for isolation
        conversation_history: list[dict] = None       # NEW: Full conversation history
    ) -> dict[str, Any]:
        """
        Analyze code chunks and decide: ingest OR search more.

        Returns:
            {
                "action": "ingest" | "search_more" | "skip",
                "reason": "Explanation",
                "confidence": 0.0-1.0,
                "documents": [...],           # If action="ingest"
                "suggested_queries": [...],   # If action="search_more"
                "ingested_count": 0           # Number of docs ingested
            }
        """
```

### SupermemoryManager (Enhanced)

```python
async def add_document(
    self,
    title: str,                           # Document title
    content: str,                         # Full content + analysis
    doc_type: str = "room",              # room | module | cluster
    container_tag: str | None = None,    # run_id for isolation
    metadata: dict[str, Any] | None = None, # Patterns, functions, etc.
    url: str | None = None               # Optional file path reference
) -> bool:
    """Add structured document to Supermemory."""

async def search_documents(
    self,
    query: str,                          # Search query
    container_tag: str | None = None,   # Filter by container
    doc_type: str | None = None,        # Filter by type
    limit: int = 10                      # Max results
) -> list[dict[str, Any]]:
    """Search documents by query."""

async def get_document_by_title(
    self,
    title: str,                         # Exact title
    container_tag: str                  # Container to search
) -> dict[str, Any] | None:
    """Get specific document by title."""
```

## Future Enhancements

### Phase 1: Multi-Agent System
- **Room Manager Agent**: Specializes in room/module docs
- **Relationship Manager Agent**: Tracks dependencies
- **Cluster Manager Agent**: Identifies subsystems

### Phase 2: Smart Deduplication
- Check if similar document already exists
- Update existing document instead of creating duplicate
- Version tracking for document updates

### Phase 3: Context7 Integration
- Use Context7 MCP tool for advanced document management
- Leverage Context7's document versioning
- Enhanced search with Context7 semantic features

### Phase 4: Agent Coordination
- Multiple agents collaborate on ingestion
- Consensus-based decisions (2+ agents agree)
- Specialized agents for different code types

## Comparison: Before vs. After

### Before (Coupled Judge + Insights)

```json
// Judge schema (complex, coupled)
{
  "items": [{"doc_id": 1, "is_relevant": true, ...}],
  "suggested_top_k": 25,
  "insights_to_store": [
    {"type": "room", "content": "...", "metadata": {...}}
  ]
}
```

**Problems:**
- ❌ Judge has too many responsibilities
- ❌ Insight storage tightly coupled to search logic
- ❌ Difficult to optimize ingestion separately
- ❌ No way to decide "search more" vs. "ingest"

### After (Decoupled Architecture)

```json
// Judge schema (simple, focused)
{
  "items": [{"doc_id": 1, "is_relevant": true, ...}],
  "suggested_top_k": 25
}

// Memory Agent decision (separate)
{
  "action": "ingest",
  "documents": [
    {"title": "...", "content": "...", "type": "room"}
  ]
}
```

**Benefits:**
- ✅ Judge focuses on search only
- ✅ Memory Agent optimized for ingestion
- ✅ Clean separation of concerns
- ✅ Explicit "ingest" vs. "search more" decisions

## Testing

### Validation Checklist

1. **Start Simulation** with judge mode enabled
2. **Verify Initialization**:
   ```
   [MEMORY-AGENT] Memory Manager Agent initialized
   ```

3. **Watch for Agent Decisions**:
   ```
   memory-agent: ingested 3 documents (Complete module understanding)
   memory-agent: search more (Fragmented info, need more context)
   ```

4. **Check Manifest**:
   ```bash
   cat .fungus_cache/runs/LATEST_RUN/manifest.json
   ```
   Look for:
   ```json
   {
     "agent_enabled": true,
     "agent_decisions": >0,
     "agent_documents_ingested": >0
   }
   ```

5. **Query Documents**:
   ```python
   docs = await memory_manager.search_documents(
       query="authentication",
       container_tag=run_id
   )
   ```

## Troubleshooting

### Agent Not Initializing

**Symptom**: No "Memory Manager Agent initialized" log

**Solutions**:
- Verify `SUPERMEMORY_API_KEY` is set in `.env`
- Check `LLM_PROVIDER=openai` (or other provider)
- Ensure `OPENAI_API_KEY` is valid

### No Documents Ingested

**Symptom**: `agent_documents_ingested: 0` in manifest

**Possible Causes**:
- Agent decided to "search more" (check logs)
- Code chunks were too fragmented
- Query not finding relevant code

**Solutions**:
- Try more specific query
- Let simulation run longer (more steps)
- Check WebSocket logs for agent decisions

### Document API Errors

**Symptom**: Errors when calling `add_document()`

**Solutions**:
- Update `supermemory` package: `pip install --upgrade supermemory`
- Verify Supermemory API supports `/add-document` endpoint
- Check API key permissions

## Summary

The Memory Manager Agent architecture provides:

- 🎯 **Clear Separation**: Judge handles search, Agent handles ingestion
- 📚 **Document-Based**: Structured documents instead of flat insights
- 🤖 **LLM-Powered**: Smart decisions about what to ingest
- 📊 **Rich Analytics**: Detailed stats on ingestion decisions
- 🔧 **Easy Integration**: Drop-in replacement for old insights system

The judge can now focus on what it does best—finding relevant code—while the Memory Manager Agent handles the complex decision of when and how to store knowledge! 🧠✨
