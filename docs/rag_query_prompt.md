## Strategic RAG Query Master Prompt (Python, line-sliced corpora)

Use this prompt to drive structured, parallel semantic queries over a Python codebase split into line-range chunks. Start broad, narrow iteratively, and switch to line-precision for final retrieval.

### Objectives
- Map structure (imports, classes, functions, entry points)
- Trace relationships (callers/callees, imports, exceptions)
- Extract implementations end-to-end (with helpers)
- Retrieve precise slices by line numbers

---

### Phase 1 — Discovery (Structure)
- "module-level imports and dependencies"
- "function and class definitions with signatures"
- "main entry points and CLI parsing (if __name__ == '__main__')"
- "docstrings above functions/classes and module headers"
- "global config variables and constants"

### Phase 2 — Components
Functions
- "function definitions with parameters and return types"
- "async def and await usage"
- "decorators and wrapper functions"
- "generator functions (yield)"

Classes
- "class definitions with base classes"
- "__init__, __str__, __repr__, __eq__, __call__"
- "@property, @classmethod, @staticmethod"
- "dataclass definitions"

Data Ops
- "file I/O open/read/write"
- "json/csv serialization"
- "dictionary/list comprehensions"

### Phase 3 — Relationships
- "calls to function {NAME}"
- "usages of class {NAME}"
- "imports for module {MODULE}"
- "try/except handling for {ErrorType}"
- "variable assignments and usages of {VAR}"

### Phase 4 — Patterns
Concurrency
- "asyncio usage (async def, await, tasks)"
- "threading primitives (Thread, Lock, Semaphore)"
- "multiprocessing (Process, Pool, Queue)"

Error Handling
- "try/except/finally patterns"
- "raise statements and asserts"
- "custom exception classes"

Context Managers
- "with statements and __enter__/__exit__ implementations"
- "contextlib utilities"

### Phase 5 — Iterative Refinement
1. "main function or script entry"
2. "functions called from main"
3. "helpers/utilities used by {FUNC}"
4. "complete implementation of {FUNC} including helpers"
5. "nearby comments/docstrings around line {N} (+/- 15)"

### Phase 6 — Semantic (What Code Does)
- "HTTP handlers (FastAPI endpoints, WebSocket)"
- "vector search/indexing (FAISS)"
- "embedding generation (SentenceTransformers)"
- "PCA projection and visualization"
- "logging and error reporting"

### Phase 7 — Parallel Batches (Run together)
Architecture Overview
- "all class definitions in repo"
- "all async function definitions"
- "external imports"
- "global variable declarations"

Realtime Feature Set (this project)
- "FastAPI endpoints (/start,/config,/pause,/resume,/ws,/doc)"
- "SnapshotStreamer._run_loop and broadcasting"
- "MCPMRetriever usage and simulation step"
- "collect_codebase_chunks and windows handling"
- "Plotly 2D/3D figure generation in React"

Error Handling
- "custom exception classes"
- "critical path try/except in server"
- "logging statements for errors"
- "fallbacks and retries (ollama generation)"

### Phase 8 — Line-Precision (Sliced Corpora)
- "lines {A}-{B} in {FILE}"
- "context around line {N} (+/- 15)"
- "complete def starting at line {N}"
- "class starting at line {N} until next class/EOF"

---

### Ready-to-Use Templates

Structure
- "All function and class definitions"
- "Module-level imports and dependencies"
- "Main execution blocks and entry points"

Functionality
- "Code for {FEATURE}: implementation and helpers"
- "Error handling for {OPERATION}"
- "Tests and validation for {COMPONENT}"

Debugging
- "Code around line {NUM} with context"
- "Function {NAME} and all its callers"
- "Variable {NAME} definition and usage"

---

### Example Workflow (This Repo)
1) "@app.post('/start') implementation and app initialization"
2) "SnapshotStreamer._run_loop broadcasting and metrics"
3) "MCPMRetriever.step and simulation.update_agent_position"
4) "collect_codebase_chunks windows → chunk_python_file slicing"
5) "React Plotly onClick customdata for doc selection"



