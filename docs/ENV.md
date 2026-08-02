# Runtime environment

`VIBEMIND_CONFIG_DIR` is required for every Fungus runtime that uses the LLM
summary or judge paths. It must point to the canonical VibeMind configuration
directory and contain `llm_config.yml`. The service contains no local
`llm_config.yml` and no provider credentials.

The canonical configuration must define these exact OpenFang contracts:

- `roles.fungus_summary` with `provider: openfang`.
- `roles.fungus_judge` with `provider: openfang`.
- `providers.openfang.base_url: ${OPENFANG_URL}/v1`, with `OPENFANG_URL`
  set to the OpenFang origin (for example `http://127.0.0.1:4200`).

Missing configuration or any contract drift is a hard error. Fungus does not
use a local provider or a provider fallback for the LLM paths.

Embeddings are a separate HTTP contract: `EMBEDDING_SERVICE_URL` is required
and must name an endpoint reachable from the current Fungus runtime. A host
MCP must set its own reachable URL. `http://embedding-service:8080` is valid
only when the runtime shares a Docker/Swarm network with that service; it is
not an implicit default. The embedding-service owns its provider, cost, and
approval configuration, which this repository neither configures nor proves.

Optional vector settings:

- `VECTOR_BACKEND=qdrant`
- `QDRANT_URL=http://localhost:6339`
- `QDRANT_COLLECTION=codebase`

`FUNGUS_RERANKER_DEVICE` controls only the optional local cross-encoder
reranker; it never selects the embedding provider or model. It is deliberately
absent from the default install. To enable it explicitly, install
`pip install ".[reranker]"` (or `uv sync --extra reranker`). The reranker is
loaded lazily only for a query that uses reranking, never during MCP import;
missing the extra therefore leaves the default heavy-free rather than acting as
an embedding fallback.

No deployment, cache rebuild, or cache deletion is implied by this document.
