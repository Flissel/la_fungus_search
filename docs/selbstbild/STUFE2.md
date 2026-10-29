# Selbstbild Stufe 2: Mathe auf dem Codegraphen (Abnahme)

Stand: 29.10.2026. Paket `src/selbstbild/analyse/`, Lauf in `analyse/lauf.py`, CLI in `selbstbild/__main__.py`.
Stufe 2 liest nur den Graphen aus Stufe 1 (`graph.sqlite`, nur lesend) und die Blobs am vermessenen
Commit. Sie schreibt `analyse.sqlite` neben den Stand. Kein LLM, kein Netz.

## Befehle (PowerShell, im la_fungus-Arbeitsbaum)

```powershell
$env:PYTHONPATH="src"
# Analyse des vermessenen Stands. Ergebnis wird in analyse.sqlite gespeichert, Zusammenfassung als JSON
.\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --export "$env:TEMP\vibemind-analyse.json"
.\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --ohne-kolonie
.\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --pruefen-reproduzierbar   # rechnet alles zweimal
# Goldsatz-Kandidaten aus der gespeicherten Analyse, danach der Vergleich
.\.venv-selbstbild\Scripts\python -m selbstbild goldsatz C:\Users\User\Desktop\Vibemind_V1 docs\selbstbild\goldsatz-vibemind.yaml
.\.venv-selbstbild\Scripts\python -m selbstbild vergleichen C:\Users\User\Desktop\Vibemind_V1 docs\selbstbild\goldsatz-vibemind.yaml
```

- `analysieren` ohne vermessenen Stand endet mit Exit 1 und dem Hinweis „erst vermessen“.
- `goldsatz` und `vergleichen` rechnen nicht neu. Sie lesen den gespeicherten Lauf zum aktuellen
  Graph-Commit, bevorzugt den mit Kolonie. Fehlt er, enden sie mit Exit 1 und „erst analysieren“.
  Ein ungültiges Goldsatz-YAML führt zu Exit 2 mit der Meldung aus `vergleichen`.
- Zur Zusammenfassung von `analysieren` gehören: Zähler je Fundart, die 10 größten Bereiche, die 10
  obersten Hotspots, `luecken`, `statistik`, Kolonie gegen Mathe (Überlappungen), `dauer_ms`,
  `speicher_spitze_mb` (Peak Working Set des Prozesses) und `speicher_privat_spitze_mb` (Peak privater Commit).

**Determinismus und Laufzeiten.** `statistik` enthält `laufzeit_laden_ms`, `laufzeit_mathe_ms` (alles
nach dem Laden außer der Kolonie) und `laufzeit_kolonie_ms`. Schlüssel mit dem Präfix `laufzeit_` stehen
**nicht** im Export, sonst wäre `analyse.json` nicht byte-gleich. Sie liegen in `analyse.sqlite` in der
Nebentabelle `messung`, und `vergleichen` liest sie von dort. `reproduzierbar_mathe` und
`reproduzierbar_kolonie` (je 1 oder 0) gibt es nur mit `--pruefen-reproduzierbar`. Weil sie
deterministisch sind, stehen sie im Export. Ein Lauf mit diesem Schalter unterscheidet sich vom Lauf
ohne ihn also genau um diese beiden Zeilen.

## Abnahme-Läufe

Host: Windows 11, 32 GB RAM, freier Commit vor jedem Lauf 5,1 bis 5,4 GB. Jeder Lauf lief unter einem
Wächter: Abbruch, wenn der Working Set des Prozessbaums 2,5 GB übersteigt oder der Lauf länger als
30 min dauert, Vorabprüfung „freier Commit ≥ 3 GB“. Kein Lauf wurde abgebrochen.

| Lauf | Commit | Dauer | Peak WS Python (selbst gemessen) | Peak WS Prozessbaum (abgetastet, inkl. `git`-Kinder) | Peak privater Commit |
|---|---|---|---|---|---|
| a0) `vermessen` VibeMind, inkrementell (Stand war `6f8e4e0f`) | `1e409b44` | 201 s | 0,85 GB | 1,01 GB | – |
| a1) `analysieren` VibeMind | `1e409b44` | **98 s** (Laden 4,0 s, Mathe 92,3 s, Kolonie 0,76 s) | **1,15 GB** | 1,44 GB | 1,87 GB |
| a2) `analysieren` VibeMind `--pruefen-reproduzierbar` | `1e409b44` | 193 s | 1,16 GB | 1,44 GB | 1,88 GB |
| b) `analysieren` sales-claw (Stand aktuell) | `c8679622` | 4,2 s (Mathe 3,5 s, Kolonie 0,04 s) | 0,13 GB | 0,14 GB | 0,85 GB |
| c) `goldsatz` VibeMind (vor S8, ersetzt) | `1e409b44` | 1,1 s | – | 0,05 GB | – |
| a3) `analysieren` VibeMind mit Ruling S8, `--pruefen-reproduzierbar` (ersetzt) | `1e409b44` | 326 s (Laden 65,7 s, Mathe 130,4 s, Kolonie 0,75 s, je einmal gerechnet) | 1,16 GB | 1,54 GB | 1,88 GB |
| c2) `goldsatz` VibeMind nach S8 (ersetzt) | `1e409b44` | 1,1 s | – | 0,05 GB | – |
| **a4) `analysieren` VibeMind mit S8 + S9, `--pruefen-reproduzierbar` (gültiger Stand)** | `1e409b44` | 263 s (Laden 4,0 s, Mathe 128,6 s, Kolonie 0,76 s, je einmal gerechnet) | 1,16 GB | 1,45 GB | 1,88 GB |
| **c3) `goldsatz` VibeMind nach S9 (gültig)** | `1e409b44` | 1,0 s | – | 0,05 GB | – |

- Das Speicherbudget von 2,5 GB Spitze hält. Der VibeMind-Lauf braucht 1,15 GB im Python-Prozess. Die
  1,44 GB im Prozessbaum zählen die `git cat-file`-Kindprozesse der Doppelungssuche mit.
- Der private Commit liegt höher als der Working Set (auch bei sales-claw 0,85 GB). Vermutlich
  reservieren numpy, BLAS und igraph Puffer, ohne sie zu belegen. Bei knappem Commit-Limit des Hosts
  zählt dieser Wert.
- a0 war fast ein Neulauf: 19 606 von 19 608 Dateien wurden neu geparst, nur 2 kamen aus dem Cache.
  Das ist **erwartet**. Task 1 von Stufe 2 hat `PARSER_VERSION` von 2 auf 3 erhöht
  (`selbstbild/sprachen.py`), und der Cache-Schlüssel enthält die Version. Geprüft in
  `Speicher._cache_schluessel` (`speicher.py`): `[PARSER_VERSION, GRAMMATIK_VERSION, Python-Version,
  blob, sprache, pfad]`. Damit verfällt der ganze Parse-Cache. Graph danach: 234 882 Knoten,
  482 618 Kanten (23 416 unsicher), 89 fehlerhafte Dateien.
- a3 enthält den S8-Durchlauf über alle Code-Blobs (Namensverweise). Die Mathe-Zeit steigt deshalb von
  92 auf 130 s. Das Laden dauerte diesmal 65,7 s statt 4 s, bei gleichem Graphen. Der freie Commit lag
  vor dem Lauf bei 4,7 GB statt über 5 GB. Vermutlich hat der Host ausgelagert, gemessen ist das nicht.
  Der Speicher bleibt bei 1,16 GB im Python-Prozess.
- **Determinismus:** Der Export von a1 (16,1 MB, sha256 `c17d3ecf…a91645`) und der von a2 stammen aus
  zwei Prozessen. Sie unterscheiden sich nur in den zwei erwarteten `reproduzierbar_*`-Zeilen. Im
  selben Prozess ergab die zweite Rechnung `reproduzierbar_mathe = 1` und `reproduzierbar_kolonie = 1`.
  a3 (S8) und a4 (S8 + S9) ergaben ebenfalls `reproduzierbar_mathe = 1` und `reproduzierbar_kolonie = 1`.
  S8 und S9 ändern nur die Funde zu totem Code. Bereiche, Zentralität, Hotspots, Doppelungen und Kolonie
  sind in a1, a3 und a4 gleich (geprüft).
- `luecken` auf VibeMind: `kante_ohne_knoten = 0` und `symbol_ohne_datei = 0`, auf sales-claw ebenso.
  Der Schlüssel `symbol_ohne_datei` kommt aus `dateigraph`. Er ist da, weil `luecken` erst nach allen
  Analysen gebaut wird (Ruling S3, abgesichert durch einen Test).

## Ergebnisse VibeMind (`1e409b44`, 20 Repos, 19 608 Dateien, 205 k Symbole)

### Toter Code je Stufe (nach Ruling S8 und S9, Lauf a4)

Zwei Rulings verschärfen nachträglich die Regeln in `analyse/tot.py`. Beide wirken nur in die sichere
Richtung, also gegen falsches `sicher_tot`:

- **S8 Namensverweis:** Ein `sicher_tot`-Kandidat, dessen einfacher Name (mindestens 3 Zeichen) als
  Bezeichner in einer anderen committeten Code-Datei desselben oder eines Vorfahren-Repos steht, wird
  `wahrscheinlich_tot` mit Grund `namensverweis`. In der eigenen Datei zählt nur ein Vorkommen außerhalb
  der eigenen Definitionszeilen. Geprüft werden nur Code-Dateien. Doku, HTML, YAML, JSON und SQL zählen
  nicht.
- **S8 Vererbung:** Verschachteltes erbt die S4- und R22-Ausnahmen seiner Vorfahren (Qualname-Präfix in
  derselben Datei). Damit fallen auch Handler in JSX-Komponenten unter `jsx_komponente`.
- **S9 Generiert:** Symbole in Build- oder generierter Ausgabe liefern keinen Fund. Gemeint sind
  Ordnersegmente `dist`, `build`, `out`, `.next`, `.nuxt`, `coverage`, `node_modules`, `dist-*` und
  `*-dist` sowie Dateinamen `*.min.*`, `*.bundle.*` und `*-bundle.*`. Sie werden in `statistik` als
  `tot_generiert_uebersprungen` gezählt, auf VibeMind **709**. Helfer: `ist_generiert()` in
  `analyse/__init__.py`.
- **S9 Konvention:** Namen, die Frameworks per Konvention aufrufen, sind nie `sicher_tot`. Das sind
  `do_*`, `visit_*`, `depart_*`, `generic_visit`, `on_*`, `handle_*`, die `setUp`/`tearDown`-Familie,
  `setup_method`/`teardown_method`, `__init_subclass__`, `__class_getitem__` und `test*`-Methoden. Sie
  werden `wahrscheinlich_tot` mit Grund `konvention`, auf VibeMind 316 Funde.

| Stufe | vor S8 (a1) | nach S8 (a3) | **nach S9 (a4, gültig)** | häufigste Gründe (a4) | Repos mit den meisten Funden (a4) |
|---|---|---|---|---|---|
| `sicher_tot` | 11 788 (2 361 Dateien) | 1 267 (590) | **1 184** (582 Dateien) | per Definition alle `unerreichbar` + `alt_90` + `kein_doku_verweis` | coding-engine 384, hermes-agent 229, vibemind-os 168, swe_desgine 125, voice 103, openclaw 50 |
| `wahrscheinlich_tot` | 19 720 | 30 241 | 29 855 | `unerreichbar` 27 580, `alt_90` 24 273, `kein_doku_verweis` 19 880, **`namensverweis` 9 199**, `doku_verweis` 8 426, `rust_r22` 4 291, `jsx_komponente` 3 839, `ts_klasse` 3 181, **`konvention` 316** | hermes-agent 7 170, vibemind-os 4 917, coding-engine 4 570, openfang 4 403 |
| `verdaechtig` | 21 912 | 21 912 | 21 911 | `nur_test` (einziger Grund) | hermes-agent 9 730, openfang 3 863, coding-engine 2 479, openclaw 1 715 |

Wegen S9 sinkt die Summe aller Funde von 53 420 auf 52 950. Die 1 184 `sicher_tot` verteilen sich so:
880 `.py` (91 großgeschriebene oberste Namen, 398 darunter verschachtelt, 381 oberste Funktionen),
180 `.ts`, 108 `.js`, 11 `.tsx`, 4 `.jsx`, 1 `.mjs`.

Je 5 Beispiele zum schnellen Gegenprüfen (fester Seed, Lauf a4):

- `sicher_tot`
  - `symbol:vibemind/vibemind-os/coding-engine:mcp_plugins/servers/grpc_host/grpc_client.py#EventFixTeamClient.postgres_execute`
  - `symbol:vibemind/vibemind-os/coding-engine:mcp_plugins/servers/grpc_host/tools/fix_migrate_tool.py#FixMigrateTool.get_fix`
  - `symbol:vibemind/vibemind-os/openclaw:extensions/nextcloud-talk/src/format.ts#formatNextcloudTalkCodeBlock`
  - `symbol:vibemind/vibemind-os:brain/the_brain/core/config.py#groq_key`
  - `symbol:vibemind/vibemind-os:spaces/mirofish/mirofish/frontend/src/api/simulation.js#getRunStatusDetail`
- `wahrscheinlich_tot`
  - `symbol:vibemind/vibemind-os/spaces/captain_cook:minibook/frontend/src/lib/tag-colors.ts#hashString` (`namensverweis`)
  - `symbol:vibemind/vibemind-os/spaces/learning:mcp/tools/materials.py#MaterialImportGateway` (jung: kein `alt_90`)
  - `symbol:vibemind/vibemind-os/spaces/shuttles/swe_desgine/external/arch_team:backend/routers/clarification_router.py#QuestionResponse` (`namensverweis`)
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/ledger/store.py#LedgerStore.get_run` (`namensverweis`)
  - `symbol:vibemind/vibemind-os/voice:electron-app/renderer/multiverse.js#MultiverseApp.onContainerMouseMove` (`ts_klasse`)
- `verdaechtig` (`nur_test`)
  - `symbol:vibemind/vibemind-os/coding-engine:src/services/pipeline_data_coalescer.py#PipelineDataCoalescer.__init__`
  - `symbol:vibemind/vibemind-os/spaces/captain_cook:agenten/execution/process.py#CapabilityProjection`
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/analysis/crop_controller.py#target_crop_size`
  - `symbol:vibemind/vibemind-os/spaces/captain_cook:agenten/execution/codex_supervisor.py#CodexSupervisor.run`
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/interchange/fcp7_xml.py#timeline_to_fcp7_xml`

#### Stichproben (`git grep -w` über alle Dateiarten am gepinnten Commit des jeweiligen Repos)

- **Vor S8** (4 Beispiele aus a1): 2 falsch. Die Enum `GateStatus` wird nur per Attribut und Annotation
  benutzt. `MediaRange.expected_seq_length` wird nur aus Tests aufgerufen, deren Import aus dem
  src-Layout nicht aufgelöst wird. Seit S8 sind beide `wahrscheinlich_tot`.
- **Nach S8** (a3, 10 Stück, `random.Random(20261028).sample` über die sortierten `sicher_tot`-Ids):
  **8 von 10 wirklich tot.** Falsch waren `MediaHandler.do_OPTIONS` (`http.server` dispatcht per
  `"do_" + Methode`) und `vendor-CQAN8hwX.js#pa` (minifiziertes Build-Artefakt).
- **Dieselben 10 nach S9** (a4): `pa` ist kein Fund mehr (generiert), `do_OPTIONS` ist
  `wahrscheinlich_tot` mit `konvention`. Die übrigen 8 sind weiter `sicher_tot`, und alle 8 sind
  wirklich tot (der Name steht im ganzen Repo nur an der eigenen Definition). Damit stimmen **8 von 8**.
- **Neue Stichprobe nach S9** (a4, 10 Stück, Seed `20261028 + 1 = 20261029`, keine Überschneidung mit
  der alten): **9 von 10 wirklich tot.** Bei allen neun steht der Name nur an der Definition, und keine
  davon trägt einen Decorator (geprüft):

| Symbol | Urteil |
|---|---|
| coding-engine `src/colony/lifecycle_controller.py#LifecycleController.add_transition_hook` | tot |
| coding-engine `src/tools/ollama_tool.py#ollama_generate` | tot |
| hermes-agent `hermes_cli/middleware.py#apply_api_request_middleware` | tot |
| captain_cook `agenten/functions/vizualisation.py#load_blockchain` | tot |
| swe_desgine `requirements_engineer/memory/mcp_memory_client.py#MCPMemoryClient.get_mcp_entity_for_story` | tot |
| laura `services/local-api/src/laura/api/demo.py#build_preview_items` | tot |
| voice `python/electron_backend.py#get_backend` | tot |
| voice `python/llm_config.py#list_roles` | tot |
| vibemind-os `brain/the_brain/core/transformer_ctm.py#TransformerCTMDistiller.distill_step` | tot |
| openfang `crates/openfang-api/static/js/pages/hands.js#handsPage` | **falsch**: wird aus HTML aufgerufen (`<div x-data="handsPage">`, Alpine.js). HTML zählt beim Namensverweis nicht |

Zusammen: Von 18 beurteilten `sicher_tot` nach S9 stimmen 17. Die verbleibende Fehlerquelle sind Namen,
die nur aus HTML oder Templates referenziert werden (`x-data`, `onclick=`). Ob die 27 von 30 halten,
misst endgültig der Goldsatz.

### Bereiche (Leiden auf dem Dateigraphen)

428 Bereiche. 8 387 der 19 608 Dateien (43 %) haben keine Import- oder Aufrufkante und stehen im
Sammelbereich `(isoliert)`. Die größten echten Bereiche (der Name ist der gemeinsame Pfad. Bei
Bereichen aus einem einzigen Repo **ohne** Repo-Präfix, deshalb steht das Repo hier dabei):

| Id | Dateien | Kohäsion | Name | Repo |
|---|---|---|---|---|
| bereich:2 | 829 | 0,9501 | apps/desktop/electron | hermes-agent |
| bereich:3 | 745 | 0,9126 | src/services | coding-engine |
| bereich:4 | 552 | 0,7750 | tests/gateway | hermes-agent |
| bereich:5 | 518 | 0,9918 | services/local-api | laura |
| bereich:6 | 485 | 0,7574 | tests/agent | hermes-agent |
| bereich:7 | 471 | 0,9859 | extensions/msteams/src | openclaw |
| bereich:8 | 456 | 0,9491 | src/agents | coding-engine |
| bereich:9 | 455 | 0,7199 | tests/hermes_cli | hermes-agent |
| bereich:10 | 451 | 0,9290 | brain/the_brain/core | vibemind-os |
| bereich:11 | 388 | 0,9682 | spaces/marketing/claw/tests | vibemind-os |

### Hotspots (Perzentil Änderung × Größe × Zentralität)

| Wert | Datei |
|---|---|
| 0,99735 | hermes-agent `run_agent.py` |
| 0,99628 | hermes-agent `tests/test_tui_gateway_server.py` |
| 0,99577 | sales-claw `sales-mcp/tests/test_ui.py` |
| 0,99559 | openfang `crates/openfang-kernel/src/kernel.rs` |
| 0,99552 | hermes-agent `tests/hermes_cli/test_web_server.py` |
| 0,99521 | hermes-agent `tests/run_agent/test_run_agent.py` |
| 0,99506 | hermes-agent `agent/auxiliary_client.py` |
| 0,99486 | hermes-agent `hermes_cli/auth.py` |
| 0,99483 | vibemind-os `spaces/marketing/tests/test_chat_worker.py` |
| 0,99476 | hermes-agent `plugins/platforms/telegram/adapter.py` |

Testdateien werden nicht ausgeschlossen. Sie sind groß, ändern sich oft und rufen viel auf, deshalb
stellen sie 5 der Top 10. Betweenness (Dateigraph, cutoff 4): Rang 1 `run_agent.py`, dann
`gateway/platforms/base.py`, `hermes_cli/plugins.py` und `hermes_cli/config.py`. Die Top 10 kommen
alle aus hermes-agent. PageRank: Rang 1 ist hermes-agent `ui-tui/src/components/branding.tsx#SessionPanel.strip`
(0,0114, doppelt so hoch wie Rang 2). Vermutlich ist das ein Artefakt der Namensauflösung, denn
`strip` ist ein sehr häufiger Name. Nicht geprüft. Danach folgen coding-engine `cell_cli`
(`list`, `ensure_config_dir`).

### Doppelungen

| Art | Gruppen | größtes Beispiel |
|---|---|---|
| `gleiche_datei` | 261 | `llm_client.py` fünfmal byte-gleich (`business/`, `devops/`, `ops/` …) |
| `gleiches_symbol` | 1 329 | `on_change` in 114 `src/services/agent_*.py` der coding-engine (generierte Services) |
| `aehnliche_datei` (Cluster, Jaccard ≥ 0,9) | 201 | 28 `tests/test_agent_task_*.py` (0,9046), 22 Composio-Controller in rowboat (0,9017) |

Übersprungene LSH-Eimer (> 50 Blobs, Boilerplate, Ruling S7): **36**.

### Kolonie gegen Mathe

| | Mathe | Kolonie |
|---|---|---|
| Laufzeit | 92,3 s vor S8, 128,6 s mit S8 + S9 (Teilzeiten nicht einzeln gemessen. Vermutlich dominiert das Blob-Lesen für Doku, Namensverweise und Doppelungen) | **0,76 s** |
| Reproduzierbar (zweiter Lauf im Prozess) | ja | ja |
| Zentrum | PageRank-Top-50 (auf Dateien abgebildet) | 50 meistbesuchte Dateien |
| Überlappung Zentrum | **7 von 50** gemeinsam | |
| Abdeckung | alle 19 608 Dateien | **486 besucht (2,5 %)**, 19 122 „tote Ecken“ |
| tote Ecken gegen toten Code | 5 093 von 5 347 Dateien mit Totfund liegen in toten Ecken, bei `sicher_tot` (a4) 505 von 582 Dateien | |
| Bereiche | 428 | 2 (4 und 2 Dateien, coding-engine `cell_cli`/`mind` und ein `dist-embedded`-Bundle) |

Fazit: Die Kolonie ist schnell und deterministisch. Mit den MCMP-Standardwerten (192 Agenten, 400
Schritte) bleibt sie auf dem VibeMind-Graphen aber in wenigen Einstiegsnachbarschaften der
coding-engine hängen. Dass sie fast alle toten Dateien „findet“, liegt nur daran, dass sie 97,5 % aller
Dateien nie besucht. Ein Signal ist das nicht. Als Herausforderer verliert sie nach diesen Zahlen bei
Bereichen und Zentrum klar. Gemessen wird das endgültig mit `vergleichen` am Goldsatz.

## Ergebnisse sales-claw (`c8679622`, 599 Dateien, 9 068 Knoten)

- Toter Code: `sicher_tot` **0**, `wahrscheinlich_tot` 2 570, `verdaechtig` 312 (`nur_test`). 0 sicher
  tot, weil kaum etwas älter als 90 Tage ist: die Stufe „alt“ greift nicht. Unter den
  `wahrscheinlich_tot` sind 697 `ts_klasse` und 352 `jsx_komponente`.
  - Beispiele `wahrscheinlich_tot`: `symbol:sales-claw:sales-mcp/verlinken.py#enthaelt_link`,
    `symbol:sales-claw:sales-mcp/ui_marke.py#routen.vorschlag_aktion.handler`,
    `symbol:sales-claw:sales-mcp/ui_marketing.py#routen.entwuerfe.haken`,
    `symbol:sales-claw:sales-mcp/ui_kalender.py#_mit_hinweisen`,
    `symbol:sales-claw:editor/src/App/PultLeiste.tsx#PultLeiste.neueFassungLaden`
  - Beispiele `verdaechtig`: `symbol:sales-claw:kalender/src/api.ts#aendern`,
    `symbol:sales-claw:kalender/src/fenster.ts#mitSperre`, `symbol:sales-claw:kalender/src/fenster.ts#startWerte`,
    `symbol:sales-claw:editor/src/chatKontext.ts#chipGibtEs`,
    `symbol:sales-claw:editor/src/App/Seite/SeitePruefung.tsx#befundZusammenfassung`
  - Dass `routen.*.handler` unerreichbar ist, zeigt: Die Routen-Registrierung (`routen(app)` aus
    `ui.py`) wird nicht als Einstieg erkannt. Ein typischer Fall von dynamischem Aufruf, den die
    Regeldatei über `einstiegspunkte` nachtragen kann.
- Bereiche: 14, davon `(isoliert)` 330 Dateien. Größte echte Bereiche: `sales-mcp/tests` (66, 0,8869),
  `editor` (37, 0,7156), `editor` (32), `editor/src` (30), `editor/src` (28).
- Top-Hotspots: `sales-mcp/tests/test_ui.py`, `sales-mcp/ui.py`, `sales-mcp/server.py`.
- Doppelungen: `gleiches_symbol` 28 (größte Gruppe: Fixture `saubere_tabellen` in 19 Testdateien),
  `gleiche_datei` 0, `aehnliche_datei` 0, übersprungene Eimer 0.
- Kolonie: 53 von 599 Dateien besucht, Zentrum ∩ PageRank-Top-50 = 14 von 50, 1 Bereich (10 Dateien).

## Bekannte Grenzen

- **E1:** Doppelungen nur textuell und strukturell. Gleiches mit anderem Wortlaut kommt erst mit den
  Embeddings in Stufe 3.
- **E2:** Die Kolonie ist die MCMP-Mechanik auf dem Dateigraphen, nicht der Original-Code im
  Embedding-Raum. Mit den Standardwerten deckt sie große Graphen kaum ab (s. o.).
- **E3:** Betweenness nur auf dem Dateigraphen mit cutoff 4, also lokal begrenzt.
- **E4:** Methoden folgen ihrer Klasse. Tote Methoden in lebenden Klassen bleiben unentdeckt (vorsichtige
  Richtung). Umgekehrt würde eine fälschlich tote Klasse alle ihre Methoden mitziehen. Seit S8 rettet
  der Namensverweis Methoden, deren Name anderswo vorkommt (`obj.methode()`).
- **S1:** Hotspot-Größe = Anzahl Symbole, nicht Zeilen. **S2:** Testdateien nur nach Namensmuster
  (plus `conftest.py`). **S3:** `luecken` nach allen Analysen. **S4:** Go-Methoden, TS-Klassen und
  JSX-Komponenten nie `sicher_tot`. Seit S8 erben verschachtelte Symbole diese Ausnahmen. **S5:** Ein
  Doku-Verweis zählt auch aus der Doku von Vorfahren-Repos. **S6:** Zeilenregel für `gleiches_symbol`
  über erneutes Parsen der Kandidaten. **S7:** Ähnlichkeits-Cluster per Union-Find, Eimer über 50 Blobs
  übersprungen (36 bei VibeMind). **S8:** Namensverweis-Wächter und vererbte S4/R22-Ausnahmen (s. o.).
  Er ist absichtlich grob: Ein gleichnamiger Bezeichner irgendwo im Code genügt zum Abstufen, auch ein
  zufällig gleicher Name. Das ist die sichere Richtung. Namen unter 3 Zeichen werden nicht geprüft.
  Doku-, HTML- und Datendateien (YAML, JSON, SQL) zählen nicht, dort registrierte Namen retten also
  nicht (Beispiel `handsPage`, s. o.). **S9:** generierte Dateien ohne Fund (`tot_generiert_uebersprungen`),
  Konventionsnamen nie `sicher_tot` (Grund `konvention`). Die Muster sind feste Listen. Andere
  Konventionen (z. B. camelCase-`onClick`) deckt erst der Namensverweis ab.
- **R15:** TS-Namensimporte sind unsicher. TS gilt als erreichbar, sobald irgendeine Kante trifft.
- **R22:** Rust ist nie `sicher_tot` (4 291 Rust-Symbole stehen in `wahrscheinlich_tot`).
- **Dynamische Aufrufe:** Registries, Plugins, String-Dispatch, Routen-Registrierung, Callbacks und
  JSX lassen lebenden Code tot aussehen. Die drei Stufen dämpfen das, die Regeldatei kann Einstiegspunkte
  nachtragen. Seit S8 fängt der Namensverweis die meisten dieser Fälle ab, darunter Python-Klassen nur
  per Attribut oder Typ und Testimporte aus src-Layouts. Seit S9 sind auch Dispatch-Präfixe und
  Build-Ausgabe abgedeckt. Übrig bleiben Verweise aus HTML und Templates (Alpine `x-data`, `onclick=`).
- Bereichsnamen tragen bei Bereichen aus einem Repo kein Repo-Präfix (`src/services` kann in jedem Repo
  liegen).
- Laufzeit: `laufzeit_mathe_ms` ist nicht nach Teilschritten aufgeschlüsselt. Mit
  `--pruefen-reproduzierbar` verdoppelt sich die Laufzeit.

## Nächster Schritt (für dich)

Die Spec-Abnahme „27 von 30 bestätigt“ ist **noch nicht gemessen**. Sie entsteht erst mit deinen Urteilen.

1. `docs/selbstbild/goldsatz-vibemind.yaml` öffnen. Er ist neu erzeugt nach S9 aus Lauf a4. Er enthält 40 `sicher_tot`-Kandidaten, 15
   Doppelungen und die 3 größten Bereiche. Alle Urteile stehen auf `offen`.
   - Bei `tot` und `doppelt` je Eintrag `urteil: ja` (stimmt) oder `urteil: nein` (stimmt nicht)
     setzen. Unsichere Einträge auf `offen` lassen, sie zählen nicht.
   - Bei `bereiche` unter `erwartet` Pfad-Präfixe eintragen, die deiner Meinung nach dazugehören.
   - Optional unter `zentral` Datei-Ids (`datei:<repo>:<pfad>`), die du für zentral hältst.
2. Vergleich laufen lassen (rechnet nicht neu, liest den gespeicherten Lauf mit Laufzeiten und
   Reproduzierbarkeit):

```powershell
cd C:\Users\User\ClaudeWork\wt-fungus-gate1
$env:PYTHONPATH="src"; .\.venv-selbstbild\Scripts\python -m selbstbild vergleichen C:\Users\User\Desktop\Vibemind_V1 docs\selbstbild\goldsatz-vibemind.yaml
```

`tot.mathe` ist die Präzision von `sicher_tot` (ja / (ja + nein)), `n_mathe` die Zahl der Urteile.
Bestanden ist die Abnahme bei mindestens 27 „ja“ unter 30 Urteilen, also einer Präzision von 0,9 oder
mehr. Ist der Graph inzwischen neu vermessen, vorher `analysieren` laufen lassen, sonst meldet
`vergleichen` „erst analysieren“:

```powershell
$env:PYTHONPATH="src"; .\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --pruefen-reproduzierbar
```

Achtung: Ein neuer Lauf kann andere Kandidaten ziehen. Der Goldsatz gilt für den Commit `1e409b44`.
Einträge, deren Symbol nicht mehr in der Analyse steht, zählt `vergleichen` als `ignoriert`.
