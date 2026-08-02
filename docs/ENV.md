# Runtime environment

`VIBEMIND_CONFIG_DIR` is required for every Fungus runtime that uses LLMs or
embeddings. It must point to the canonical VibeMind configuration directory
and contain `llm_config.yml`. The service contains no local `llm_config.yml`
and no provider credentials.

The canonical configuration must define these exact OpenFang contracts:

- `roles.fungus_summary` with `provider: openfang`.
- `roles.fungus_judge` with `provider: openfang`.
- `embeddings.fungus_search` with exactly `driver: openai`,
  `provider: openfang`, `model: text-embedding-3-large`, and `dim: 3072`.
- `providers.openfang.base_url: ${OPENFANG_URL}/v1`, with `OPENFANG_URL`
  set to the OpenFang origin (for example `http://127.0.0.1:4200`).

Missing configuration or any contract drift is a hard error. Fungus does not
use a local provider, a local embedding model, or a provider fallback.

Optional vector settings:

- `VECTOR_BACKEND=qdrant`
- `QDRANT_URL=http://localhost:6339`
- `QDRANT_COLLECTION=codebase`

`FUNGUS_RERANKER_DEVICE` controls only the optional local cross-encoder
reranker; it never selects the embedding provider or model.

No deployment, cache rebuild, or cache deletion is implied by this document.
