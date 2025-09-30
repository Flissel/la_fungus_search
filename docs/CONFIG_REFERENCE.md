# MCMP Realtime – Konfigurationsreferenz

Diese Referenz dokumentiert alle im Frontend und Backend verfügbaren Einstellungen. Jede Einstellung nennt den technischen Schlüssel (Backend-Name), Typ, Default (falls bekannt) und Zweck. Werte können via API (/start, /config), per .env, oder über das UI gesetzt werden. Backend-Quelle: src/embeddinggemma/realtime/server.py (SettingsModel, settings_dict, apply_settings).

## Allgemein
- query (string, default: "Classify the code into modules.")
  - Zweck: Such-/Steuer-Query für Simulation, Ranking und Berichte.
- top_k (int, default: 10)
  - Zweck: Anzahl der Top-Ergebnisse pro Schritt/Antwort.
- mode/report_mode (string, default: "deep")
  - Werte: deep, structure, exploratory, summary, repair, steering. Prompt-Stil für Berichte (LLM).
- judge_mode (string, default: "steering")
  - Zweck: Prompt-Stil für LLM-Judge (Kontextsteuerung).
- mq_enabled (bool, default: false)
  - Zweck: Multi-Query Modus aktivieren (LLM-unterstützte Zusatzabfragen).
- mq_count (int, default: 5)
  - Zweck: Anzahl generierter Zusatz-Queries im Multi-Query Modus.

## Visualisierung
- viz_dims (int, default: 3)
  - Werte: 2 oder 3; 2D/3D Projektion (PCA) im Snapshot.
- min_trail_strength (float, default: 0.05)
  - Zweck: Schwellenwert zum Anzeigen von Pheromon-Kanten.
- max_edges (int, default: 1500)
  - Zweck: Kantenlimit im Snapshot.
- redraw_every (int, default: 2)
  - Zweck: Schrittintervall für Snapshot-/Metrics-Broadcast via WebSocket.

## Corpus & Chunking
- use_repo (bool, default: true)
  - Zweck: src als Wurzel nutzen; sonst root_folder.
- root_folder (string, default: Arbeitsverzeichnis)
  - Zweck: Projektwurzel, wenn use_repo=false.
- max_files (int, default: 500)
  - Zweck: Obergrenze eingelesener Dateien.
- exclude_dirs (string[], default: [".venv","node_modules",".git","external"])
  - Zweck: Verzeichnisse ausschließen.
- windows (int[], default: none – Pflichtfeld im UI)
  - Zweck: Zeilenfenster für Chunking (z. B. 20, 50, 100).
- chunk_workers (int, default: CPU-basiert)
  - Zweck: Thread-Anzahl beim Chunking.
- embed_batch_size (int, default: 128)
  - Zweck: Batchgröße beim Embedding von Chunks (SentenceTransformers).
- max_chunks_per_shard (int, default: 2000)
  - Zweck: Shardgröße für Batch-Läufe (Jobs).

## Simulation
- num_agents (int, default: 200)
  - Zweck: Anzahl Agenten in der Simulation.
- max_iterations (int, default: 200)
  - Zweck: Maximale Schrittzahl (Läufe, Jobs).
- exploration_bonus (float, default: 0.1)
  - Zweck: Gewichtung explorativer Agentenbewegung.
- pheromone_decay (float, default: 0.95)
  - Zweck: Zerfallfaktor der Pheromon-Trails pro Schritt.

## Reporting & Judge (LLM‑gesteuerte Kontextsteuerung)
- report_enabled (bool, default: false)
  - Zweck: Periodische Schrittberichte aktivieren.
- report_every (int, default: 5)
  - Zweck: Intervall (in Schritten) für Berichte.
- report_mode (string, s. oben)
  - Zweck: Berichtsprompt-Modus.
- judge_enabled (bool, default: true)
  - Zweck: LLM-Judge aktivieren (Kontextsteuerung).
- max_reports (int, default: 20)
  - Zweck: Budgetlimit Berichts-/Judge-Schritte.
- max_report_tokens (int, default: 20000; Zeichen approximiert)
  - Zweck: Grobes Token-/Zeichenbudget für Berichte/Judge.

### Blended Scoring / Pruning
- alpha (float, default: 0.7)
  - Zweck: Gewicht Kosinus-Similarität.
- beta (float, default: 0.1)
  - Zweck: Gewicht Besuchsnorm (visit_norm).
- gamma (float, default: 0.1)
  - Zweck: Gewicht Trail-Degree.
- delta (float, default: 0.1)
  - Zweck: Gewicht LLM-Vote (−1/0/1).
- epsilon (float, default: 0.0)
  - Zweck: Länge-/Prior-Gewicht (bm25-ähnlich).
- min_content_chars (int, default: 80)
  - Zweck: Minimum Zeichenlänge für Chunk-Bewertung/Pruning.
- import_only_penalty (float, default: 0.4)
  - Zweck: Strafgewicht für reine Import-Chunks.

## LLM Provider (zentrale Defaults unter src/embeddinggemma/llm/config.py)
- llm_provider (string, default: ollama)
  - Werte: ollama, openai, google, grok.

### Ollama
- ollama_model (string, default: qwen2.5-coder:7b)
- ollama_host (string, default: http://127.0.0.1:11434)
- ollama_system (string|null, default: null)
- ollama_num_gpu (int|null, default: env/None)
- ollama_num_thread (int|null, default: env/None)
- ollama_num_batch (int|null, default: env/None)

### OpenAI
- openai_model (string, default: gpt-4o-mini)
- openai_api_key (string|null)
- openai_base_url (string, default: https://api.openai.com)
- openai_temperature (float, default: 0.0)

### Google
- google_model (string, default: gemini-1.5-pro)
- google_api_key (string|null)
- google_base_url (string, default: https://generativelanguage.googleapis.com)
- google_temperature (float, default: 0.0)

### Grok
- grok_model (string, default: grok-2-latest)
- grok_api_key (string|null)
- grok_base_url (string, default: https://api.x.ai)
- grok_temperature (float, default: 0.0)

## Aktionen (UI)
Diese ändern keinen Dauerzustand, sind aber Workflow-relevant:
- Apply (POST /config)
- Start (POST /start – initialisiert Corpus/Simulation)
- Stop (POST /stop)
- Reset (POST /reset – setzt Simulation komplett zurück, Konfiguration bleibt)
- Pause/Resume (POST /pause, /resume)
- Add Agents/Resize Agents (POST /agents/add, /agents/resize)
- Corpus Listing (GET /corpus/list)
- Shard Run (POST /jobs/start)
- Search/Answer (POST /search, /answer)

## Mapping & Verwendung
Die wichtigsten Verwendungen pro Einstellung sind in settings_usage_lines im Backend hinterlegt. Beispiele:
- viz_dims → Projektion & UI (PCA 2D/3D)
- num_agents, pheromone_decay, exploration_bonus → mcmp/simulation.py
- report_*, judge_*, alpha..epsilon, min_content_chars, import_only_penalty → LLM-Steuerung/Blended-Score in realtime/server.py
- Provider-spezifische Felder → LLM‑Dispatcher src/embeddinggemma/llm/dispatcher.py und Aufrufe in realtime/server.py

Hinweis: Um Defaults zentral zu pflegen, kann src/embeddinggemma/llm/config.py angepasst werden. .env Werte überschreiben diese Defaults. Das Frontend sendet nur gesetzte/abweichende Werte.



