# Fungus embedding-service contract

`la-fungus-search` sends every index and query embedding request to the
VibeMind `embedding-service`; it never loads an embedding model and never
calls an upstream provider directly.

Configure its base URL explicitly with `EMBEDDING_SERVICE_URL`; there is no
default. The host MCP process must set a URL it can reach. A deployment may
explicitly set `http://embedding-service:8080` only when it shares that
Docker/Swarm network. Requests use `POST /embed/batch` with
`{"texts": ["..."]}` and require `{"vectors": [[...], ...]}` in matching
order. `EMBEDDING_HTTP_TIMEOUT`, `EMBEDDING_HTTP_MAX_RETRIES` (default 2), and
`EMBEDDING_HTTP_RETRY_BACKOFF` bound retries for connection, timeout, and 5xx
failures only. All other failures are hard errors; there is no local or direct
provider fallback.

The active vector contract is 3072 dimensions. Existing cache files or Qdrant
collections with another dimension cause a clear rebuild-required error. This
repository does not rebuild collections, migrate dimensions, or reindex data;
those are separate deployment/data operations. The embedding-service's
provider, cost, and approval authority are outside this repository and are not
verified by this client contract.
