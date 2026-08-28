# la_fungus_search

Bio-inspired semantic code search. Instead of ranking chunks by cosine similarity alone, a
colony of agents walks the codebase leaving pheromone trails — paths that keep proving useful
get reinforced, paths that do not decay. What surfaces is not just the closest chunk, but the
neighbourhood a question actually lives in.

Built for navigating large, unfamiliar repositories where you know what you want but not what
it is called.

## How it works

Two retrieval paths, side by side:

- **MCMP** (`src/embeddinggemma/mcmp_rag.py`) — the pheromone simulation. Agents traverse the
  chunk graph, deposit and evaporate signal, and converge on regions rather than single hits.
- **RagV1** (`src/embeddinggemma/rag/`) — conventional hybrid retrieval with AST-aware chunking
  (`ast_scan.py`), so a function arrives whole instead of split across a window boundary. Dense
  vectors via EmbeddingGemma, sparse via BM25 (`bm25_lite.py`), persisted to Qdrant.

`mcp_server.py` exposes the same retrieval as an MCP server, so an agent can call it as a tool.
LLM calls default to a local Ollama server, with a Hugging Face fallback.

## Running it

The current entry point is a realtime server (FastAPI + websockets) with a Vite frontend:

```powershell
./run-realtime.ps1 -Port 8011
```

It expects `VIBEMIND_CONFIG_DIR` to point at a directory containing `llm_config.yml`, and a
virtualenv at `.venv`. The frontend lives in `frontend/` (`npm install && npm run dev`).

A Streamlit UI is also present under `src/embeddinggemma/ui`.

Build an index first:

```bash
uv sync                 # or: pip install -r requirements.txt
python build_index.py
```

Optional, for persistent vectors:

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

Configuration lives in `.env` — copy `_.env.example` and fill it in. See
[`docs/ENV.md`](docs/ENV.md) for every variable and
[`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) for tuning the simulation.

## Index variants

Several builders ship with the repo, for different hardware and corpora:

| Script | Use |
|---|---|
| `build_index.py` | default path |
| `build_direct_gpu.py` | GPU embedding, largest corpora |
| `build_optimized.py` | memory-constrained machines |
| `build_multivec_index.py` | multi-vector retrieval |
| `build_brain_focused.py` | scoped to a subtree |
| `incremental_updater.py` | re-index only what changed |
| `repair_index.py`, `verify_index.py` | consistency checks |

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — C4 diagrams, container and component views
- [`docs/MCMP_FULL_PIPELINE_ANALYSIS.md`](docs/MCMP_FULL_PIPELINE_ANALYSIS.md) — the simulation in detail
- [`docs/mcmp_simulation.md`](docs/mcmp_simulation.md) — parameters and behaviour
- [`docs/SCRIPTS.md`](docs/SCRIPTS.md) — every entry point
- [`docs/MAINTENANCE.md`](docs/MAINTENANCE.md) — cache and index upkeep

Benchmarks live in `benchmarks/`. Tests run with `pytest`.

## Status

Research-grade and actively used, not a packaged product. Parts of `docs/` still describe the
earlier Streamlit-first layout. Some scripts assume a Windows host.

Part of the [VibeMind](https://github.com/Vibemind-LAB/Vibemind_V1) stack.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
