# Context MCP: Multi-Korpus-Suche mit Quittung, für jede Agenten-Session

`src/embeddinggemma/context_mcp.py` serviert den gemessenen Retrieval-Stack
(Report Sektionen 27/28) über MCP/stdio. Zwei Tools:

- `context_corpora()` — was durchsuchbar ist: Korpus, Engine, Dokumentzahl,
  `rank_rule`, Manifest-Digest.
- `context_search(query, corpus="", top_k=8)` — Treffer mit Quittung: Korpus,
  Datei, Zeilenspanne, Symbol, `expanded`-Markierung. Leerer `corpus` sucht
  alle, fusioniert per Rang.

Laden dauert ~2 s (gegen ~50 s Kaltstart des chunk-basierten `mcp_server.py`),
ein Index je Korpus, kein Embedding-Dienst nötig — der Dense-Arm armiert sich
nur, wenn `FUNGUS_V2_EMBEDDER_URL` auf den lokalen Embedding-Dienst zeigt
(`local_embedding_service.py`).

## Konfiguration

`FUNGUS_V2_CORPORA` zeigt auf eine JSON-Liste:

```json
[
  {"name": "brain-code",
   "manifest": "<repo>/benchmarks/gate2/manifests/brain-v1.json",
   "snapshot": "<repo>/benchmarks/results/gate2/snapshot-brain-full.npz"},
  {"name": "secondbrain",
   "manifest": "<privat, gitignoriert>",
   "snapshot": "<privat, gitignoriert>",
   "rank_rule": "rrf"}
]
```

`rank_rule`: `bm25` für Code, `rrf` für Prosa — Sektion 28, ein Knopf je Korpus.
Snapshot optional; ohne ihn läuft BM25+Expansion.

## Registrierung in Claude Code

```bash
claude mcp add context-v2 \
  --env PYTHONPATH=<repo>/src \
  --env FUNGUS_V2_CORPORA=<pfad>/corpora.json \
  -- <repo>/.venv/Scripts/python.exe -m embeddinggemma.context_mcp
```

Oder als Eintrag in einem `--mcp-config`-Profil (Muster `.claude/mcp-profiles/`):

```json
{"mcpServers": {"context-v2": {
  "command": "<repo>/.venv/Scripts/python.exe",
  "args": ["-m", "embeddinggemma.context_mcp"],
  "env": {"PYTHONPATH": "<repo>/src",
          "FUNGUS_V2_CORPORA": "<pfad>/corpora.json"}}}}
```

## Selbsttest

```bash
FUNGUS_V2_CORPORA=<pfad>/corpora.json \
  python -m embeddinggemma.context_mcp --selftest
```

Lädt die Konfiguration fail-closed und druckt die Korpora samt Digests. Ein
echter stdio-Roundtrip (initialize → list_tools → call_tool) ist im Commit-Log
dokumentiert; die Demo-Query „wie wird eine notiz gegen den code verifiziert"
beantwortete sich aus Vault-Spec und Brain-Code zugleich.

## Grenzen, ehrlich

Vault-Manifeste enthalten private Notiztexte — sie bleiben gitignoriert, die
Korpora-Konfiguration zeigt lokal darauf. Die Zahlen hinter dem Stack stammen
vom Funktion-als-Query-Protokoll; für natürlichsprachliche Queries gilt Sektion
27.5/28.3. Der Server erfindet keine Frische: Manifeste altern mit dem Repo und
werden per `--rebuild` der Evidence-CLI bzw. Neuexport aktualisiert.
