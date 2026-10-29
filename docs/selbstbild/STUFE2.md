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
# Ausgefuellten Goldsatz gegen die gespeicherte Analyse messen
.\.venv-selbstbild\Scripts\python -m selbstbild vergleichen C:\Users\User\Desktop\Vibemind_V1 docs\selbstbild\goldsatz-vibemind.yaml
```

**Nur zum Neu-Erzeugen** der Goldsatz-Kandidaten (nicht für den normalen Ablauf). Ein vorhandener
Goldsatz wird ohne `--ueberschreiben` **nicht** angefasst (Exit 2), damit eingetragene Urteile erhalten
bleiben. Mit dem Schalter gehen sie verloren:

```text
python -m selbstbild goldsatz <wurzel> <ziel.yaml> [--ueberschreiben]
```

- `analysieren` ohne vermessenen Stand endet mit Exit 1 und dem Hinweis „erst vermessen“.
- `goldsatz` und `vergleichen` rechnen nicht neu. Sie lesen den gespeicherten Lauf zum aktuellen
  Graphen, bevorzugt den mit Kolonie. Dabei müssen Commit **und** `gemessen_um` des Graphen
  übereinstimmen (Tabelle `herkunft`), ein neu vermessener Graph verlangt also eine neue Analyse. Fehlt
  der Lauf, enden sie mit Exit 1 und „erst analysieren“.
- Exit-Codes: 1 heißt nicht vermessen oder nicht analysiert. 3 heißt Stand ohne brauchbaren Graphen
  (neu vermessen). 2 heißt ungültiges oder unlesbares Goldsatz-YAML, vorhandenes Goldsatz-Ziel ohne
  `--ueberschreiben` oder nicht schreibbares Ziel (`--export`, `goldsatz`).
- Der Goldsatz trägt `commit:`. Weicht er vom Commit der Analyse ab, warnt `vergleichen` auf stderr und
  nennt die Zahl der tot-Einträge, deren Symbol verschwunden ist (sie zählen zu `ignoriert`).
- Zur Zusammenfassung von `analysieren` gehören: Zähler je Fundart, die 10 größten Bereiche, die 10
  obersten Hotspots, `luecken`, `statistik`, Kolonie gegen Mathe (Überlappungen), `dauer_ms`,
  `speicher_spitze_mb` (Peak Working Set des Prozesses) und `speicher_privat_spitze_mb` (Peak privater Commit).

**Determinismus und Messwerte.** `statistik` enthält `laufzeit_laden_ms`, `laufzeit_mathe_ms` (alles
nach dem Laden außer der Kolonie) und `laufzeit_kolonie_ms`. Mit `--pruefen-reproduzierbar` kommen
`reproduzierbar_mathe` und `reproduzierbar_kolonie` dazu (je 1 oder 0). Alle Schlüssel mit dem Präfix
`laufzeit_` oder `reproduzierbar_` sind Messwerte und stehen **nicht** im Export. Sie liegen in
`analyse.sqlite` in der Nebentabelle `messung`, und `vergleichen` liest sie von dort. Gleiche Parameter
ergeben damit immer einen byte-gleichen `analyse.json`, mit oder ohne Prüf-Schalter. Ein Test prüft
das über zwei Prozesse mit verschiedenem `PYTHONHASHSEED`. Im Export bleiben aus `statistik` nur
`doppelt_eimer_uebersprungen` und `tot_generiert_uebersprungen`.

## Abnahme-Läufe

Host: Windows 11, 32 GB RAM, freier Commit vor jedem Lauf 5,1 bis 5,4 GB. Jeder Lauf lief unter einem
Wächter: Abbruch, wenn der Working Set des Prozessbaums 2,5 GB übersteigt oder der Lauf länger als
30 min dauert, Vorabprüfung „freier Commit ≥ 3 GB“. Kein Analyselauf wurde abgebrochen. Ein erster
Startversuch von a5 scheiterte, bevor Python lief: Der Wächter selbst (PowerShell) bekam beim Start
keinen Speicher (CoreCLR `0x8007000E`), weil der Host kurz am Commit-Limit war. Der Wiederholungslauf
startete mit 5,9 GB freiem Commit.

| Lauf | Commit | Dauer | Peak WS Python (selbst gemessen) | Peak WS Prozessbaum (abgetastet, inkl. `git`-Kinder) | Peak privater Commit |
|---|---|---|---|---|---|
| a0) `vermessen` VibeMind, inkrementell (Stand war `6f8e4e0f`) | `1e409b44` | 201 s | 0,85 GB | 1,01 GB | – |
| a1) `analysieren` VibeMind | `1e409b44` | **98 s** (Laden 4,0 s, Mathe 92,3 s, Kolonie 0,76 s) | **1,15 GB** | 1,44 GB | 1,87 GB |
| a2) `analysieren` VibeMind `--pruefen-reproduzierbar` | `1e409b44` | 193 s | 1,16 GB | 1,44 GB | 1,88 GB |
| b) `analysieren` sales-claw (Stand aktuell) | `c8679622` | 4,2 s (Mathe 3,5 s, Kolonie 0,04 s) | 0,13 GB | 0,14 GB | 0,85 GB |
| c) `goldsatz` VibeMind (vor S8, ersetzt) | `1e409b44` | 1,1 s | – | 0,05 GB | – |
| a3) `analysieren` VibeMind mit Ruling S8, `--pruefen-reproduzierbar` (ersetzt) | `1e409b44` | 326 s (Laden 65,7 s, Mathe 130,4 s, Kolonie 0,75 s, je einmal gerechnet) | 1,16 GB | 1,54 GB | 1,88 GB |
| c2) `goldsatz` VibeMind nach S8 (ersetzt) | `1e409b44` | 1,1 s | – | 0,05 GB | – |
| a4) `analysieren` VibeMind mit S8 + S9, `--pruefen-reproduzierbar` (ersetzt) | `1e409b44` | 263 s (Laden 4,0 s, Mathe 128,6 s, Kolonie 0,76 s, je einmal gerechnet) | 1,16 GB | 1,45 GB | 1,88 GB |
| c3) `goldsatz` VibeMind nach S9 (ersetzt) | `1e409b44` | 1,0 s | – | 0,05 GB | – |
| **a5) `analysieren` VibeMind nach Fix-Runde 3 (S8 mit HTML/Vorlagen), `--pruefen-reproduzierbar` (gültiger Stand)** | `1e409b44` | 281 s (Laden 22,4 s, Mathe 129,0 s, Kolonie 0,74 s, je einmal gerechnet) | 1,16 GB | 1,57 GB | 1,88 GB |
| **c4) `goldsatz` VibeMind nach Fix-Runde 3, `--ueberschreiben` (gültig)** | `1e409b44` | 1 s | – | – | – |

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
  zwei Prozessen. Damals unterschieden sie sich nur in den zwei `reproduzierbar_*`-Zeilen. Seit Fix-Runde 3
  stehen diese nicht mehr im Export, und der Export von a5 enthält sie nicht. Im selben Prozess ergaben
  a2, a3, a4 und a5 jeweils `reproduzierbar_mathe = 1` und `reproduzierbar_kolonie = 1`. S8 und S9
  ändern nur die Funde zu totem Code. Bereiche, Zentralität, Hotspots, Doppelungen und Kolonie sind in
  a1, a3, a4 und a5 gleich (geprüft).
- `luecken` auf VibeMind: `kante_ohne_knoten = 0` und `symbol_ohne_datei = 0`, auf sales-claw ebenso.
  Der Schlüssel `symbol_ohne_datei` kommt aus `dateigraph`. Er ist da, weil `luecken` erst nach allen
  Analysen gebaut wird (Ruling S3, abgesichert durch einen Test).

## Ergebnisse VibeMind (`1e409b44`, 20 Repos, 19 608 Dateien, 205 k Symbole)

### Toter Code je Stufe (nach S8, S9 und Fix-Runde 3, Lauf a5)

Zwei Rulings und eine Fix-Runde verschärfen nachträglich die Regeln in `analyse/tot.py`. Alle wirken nur
in die sichere Richtung, also gegen falsches `sicher_tot`:

- **S8 Namensverweis:** Ein `sicher_tot`-Kandidat, dessen einfacher Name (mindestens 3 Zeichen) als
  Bezeichner in einer anderen committeten Datei desselben oder eines Vorfahren-Repos steht, wird
  `wahrscheinlich_tot` mit Grund `namensverweis`. In der eigenen Datei zählt nur ein Vorkommen außerhalb
  der eigenen Definitionszeilen. Gelesen werden Code-Dateien (python, ts, tsx, js, go, rust, bash,
  powershell). Seit Fix-Runde 3 kommen HTML und Vorlagen dazu (`.html`, `.htm`, `.vue`, `.svelte`,
  `.jinja`, `.jinja2`, `.j2`, `.hbs`, `.ejs`, `.erb`, `.twig`), damit auch Alpine `x-data="f"` und
  `onclick="f()"` zählen. Generierte Dateien (S9) verweisen nie. Doku, YAML, JSON und SQL zählen nicht.
- **S8 Vererbung:** Verschachteltes erbt die S4- und R22-Ausnahmen seiner Vorfahren (Qualname-Präfix in
  derselben Datei). Damit fallen auch Handler in JSX-Komponenten unter `jsx_komponente`.
- **S9 Generiert:** Symbole in Build- oder generierter Ausgabe liefern keinen Fund. Gemeint sind
  Ordnersegmente `dist`, `build`, `out`, `.next`, `.nuxt`, `coverage`, `node_modules`, `dist-*` und
  `*-dist` sowie Dateinamen `*.min.*`, `*.bundle.*` und `*-bundle.*`. Ihre Zahl steht in `statistik` als
  `tot_generiert_uebersprungen`, auf VibeMind **709** Symbole. Helfer: `ist_generiert()` in
  `analyse/__init__.py`.
- **S9 Konvention:** Namen, die Frameworks per Konvention aufrufen, sind nie `sicher_tot`. Das sind
  `do_*`, `visit_*`, `depart_*`, `generic_visit`, `on_*`, `handle_*`, die `setUp`/`tearDown`-Familie,
  `setup_method`/`teardown_method`, `__init_subclass__` und `__class_getitem__`. Dazu kommen Methoden
  (Symbolart `methode`), deren Name mit `test` beginnt. Freie Funktionen `test*` außerhalb von
  Testdateien fallen **nicht** darunter, Testdateien sind ohnehin nie `sicher_tot` (S2). Diese Funde
  werden `wahrscheinlich_tot` mit Grund `konvention`, auf VibeMind 316.

| Stufe | vor S8 (a1) | nach S8 (a3) | nach S9 (a4) | **nach Fix-Runde 3 (a5, gültig)** | häufigste Gründe (a5) | Repos mit den meisten Funden (a5) |
|---|---|---|---|---|---|---|
| `sicher_tot` | 11 788 (2 361 Dateien) | 1 267 (590) | 1 184 (582) | **1 151** (563 Dateien) | per Definition alle `unerreichbar` + `alt_90` + `kein_doku_verweis` | coding-engine 384, hermes-agent 229, vibemind-os 152, swe_desgine 125, voice 103, openclaw 50 |
| `wahrscheinlich_tot` | 19 720 | 30 241 | 29 855 | 29 888 | `unerreichbar` 27 613, `alt_90` 24 306, `kein_doku_verweis` 19 913, **`namensverweis` 9 232**, `doku_verweis` 8 426, `rust_r22` 4 291, `jsx_komponente` 3 839, `ts_klasse` 3 181, **`konvention` 316** | hermes-agent 7 170, vibemind-os 4 933, coding-engine 4 570, openfang 4 420 |
| `verdaechtig` | 21 912 | 21 912 | 21 911 | 21 911 | `nur_test` (einziger Grund) | hermes-agent 9 730, openfang 3 863, coding-engine 2 479, openclaw 1 715 |

- **S9 und die Summe:** Von a3 nach a4 fiel die Summe aller Funde von 53 420 auf 52 950, also um 470.
  Das sind genau die a3-Funde in generierten Dateien (406 `wahrscheinlich_tot`, 63 `sicher_tot`,
  1 `verdaechtig`). Die übrigen 239 der 709 übersprungenen Symbole waren schon vorher kein Fund, weil
  sie erreichbar sind.
- **Fix-Runde 3:** Durch HTML und Vorlagen gingen 33 weitere `sicher_tot` nach `wahrscheinlich_tot`
  (1 184 → 1 151), darunter `handsPage`.
- **Aufteilung der 1 151 `sicher_tot` nach Endung:** 880 `.py`, 180 `.ts`, 75 `.js`, 11 `.tsx`, 4 `.jsx`, 1 `.mjs`.
- **Die 880 `.py` im Einzelnen:** 91 oberste Namen mit großem Anfangsbuchstaben (vermutlich Klassen),
  398 darunter verschachtelt (Methoden toter Klassen), 381 oberste Funktionen, 10 in Funktionen
  verschachtelt (Closures und innere Funktionen toter Funktionen).
- **Kurze Namen:** 14 `sicher_tot` haben einen Namen unter 3 Zeichen (schon nach S9 waren es 14). Für
  sie greift der Namensverweis nicht.

Je 5 Beispiele zum schnellen Gegenprüfen (fester Seed, Lauf a5):

- `sicher_tot`
  - `symbol:vibemind/vibemind-os/coding-engine:mcp_plugins/servers/grpc_host/grpc_client.py#EventFixTeamClient.postgres_execute`
  - `symbol:vibemind/vibemind-os/coding-engine:mcp_plugins/servers/grpc_host/tools/fix_migrate_tool.py#FixMigrateTool.get_fix`
  - `symbol:vibemind/vibemind-os/openclaw:extensions/nextcloud-talk/src/format.ts#formatNextcloudTalkCodeBlock`
  - `symbol:vibemind/vibemind-os:brain/the_brain/core/ewc_regularization.py#MemoryAwareConsolidation.add_task`
  - `symbol:vibemind/vibemind-os:spaces/rowboat/rowboat/apps/x/apps/renderer/src/lib/toast.ts#removeToast`
- `wahrscheinlich_tot`
  - `symbol:vibemind/vibemind-os/spaces/captain_cook:minibook/frontend/src/components/ui/dropdown-menu.tsx#DropdownMenuSubContent` (`jsx_komponente`)
  - `symbol:vibemind/vibemind-os/spaces/learning:mcp/tools/generation.py#GeneratedCourseRouter` (jung: kein `alt_90`)
  - `symbol:vibemind/vibemind-os/spaces/shuttles/swe_desgine/external/arch_team:backend/routers/arch_team_router.py#EvaluateSingleRequest` (`namensverweis`)
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/interchange/validate.py#validate_export` (`nur_unsicher`)
  - `symbol:vibemind/vibemind-os/voice:electron-app/renderer/multiverse.js#MultiverseApp.exitBubbleWithAnimation` (`ts_klasse`)
- `verdaechtig` (`nur_test`)
  - `symbol:vibemind/vibemind-os/coding-engine:src/services/pipeline_data_coalescer.py#PipelineDataCoalescer.__init__`
  - `symbol:vibemind/vibemind-os/spaces/captain_cook:agenten/execution/process.py#CapabilityProjection`
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/analysis/crop_controller.py#target_crop_size`
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/interchange/fcpx_xml.py#_frame_duration`
  - `symbol:vibemind/vibemind-os:spaces/marketing/claw/logo_bearbeiten.py#_abstand`

#### Stichproben (`git grep -w` über alle Dateiarten am gepinnten Commit des jeweiligen Repos)

- **Vor S8** (4 Beispiele aus a1): 2 falsch. Die Enum `GateStatus` wird nur per Attribut und Annotation
  benutzt. `MediaRange.expected_seq_length` wird nur aus Tests aufgerufen, deren Import aus dem
  src-Layout nicht aufgelöst wird. Seit S8 sind beide `wahrscheinlich_tot`.
- **Erste 10** (aus a3, Seed `20261028`): nach S8 waren **8 von 10** wirklich tot. Falsch waren
  `MediaHandler.do_OPTIONS` (`http.server` dispatcht per `"do_" + Methode`) und `vendor-CQAN8hwX.js#pa`
  (minifiziertes Build-Artefakt). Seit S9 ist `pa` kein Fund mehr und `do_OPTIONS` ist
  `wahrscheinlich_tot` mit `konvention`.
- **Zweite 10** (aus a4, Seed `20261029`): nach S9 waren **9 von 10** wirklich tot. Bei den neun steht
  der Name nur an der Definition, und keine trägt einen Decorator (geprüft). Falsch war openfang
  `crates/openfang-api/static/js/pages/hands.js#handsPage`: Er wird aus HTML aufgerufen
  (`<div x-data="handsPage">`, Alpine.js).
- **Gezielt nachgeprüft in a5:** `handsPage` ist jetzt `wahrscheinlich_tot` mit `namensverweis`, weil
  `index_body.html` mitgelesen wird.
- **Stand in a5 für alle 20 gezogenen Symbole:** 17 sind weiter `sicher_tot`, und genau diese 17 waren
  als wirklich tot beurteilt. Die drei falschen sind keine `sicher_tot` mehr. Zur zweiten Stichprobe
  gehören auch `add_transition_hook`, `ollama_generate`, `apply_api_request_middleware`,
  `load_blockchain`, `get_mcp_entity_for_story`, `build_preview_items`, `get_backend`, `list_roles` und
  `distill_step`.

Ob die 27 von 30 halten, misst endgültig der Goldsatz. Die bekannten Restquellen sind dynamische Aufrufe
ohne Namen im Code (Registries in YAML oder JSON, `getattr` mit zusammengesetzten Strings) und die 14
kurzen Namen.

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
| Laufzeit | 92,3 s vor S8, 129,0 s in a5 (Teilzeiten nicht einzeln gemessen. Vermutlich dominiert das Blob-Lesen für Doku, Namensverweise und Doppelungen) | **0,76 s** |
| Reproduzierbar (zweiter Lauf im Prozess) | ja | ja |
| Zentrum | PageRank-Top-50 (auf Dateien abgebildet) | 50 meistbesuchte Dateien |
| Überlappung Zentrum | **7 von 50** gemeinsam | |
| Abdeckung | alle 19 608 Dateien | **486 besucht (2,5 %)**, 19 122 „tote Ecken“ |
| tote Ecken gegen toten Code (Lauf a5) | 5 090 von 5 341 Dateien mit Totfund liegen in toten Ecken, bei `sicher_tot` 486 von 563 Dateien | |
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
  Seit Fix-Runde 3 zählen auch HTML und Vorlagen (Beispiel `handsPage`, s. o.). Doku- und Datendateien
  (Markdown, YAML, JSON, SQL) zählen nicht, dort registrierte Namen retten also nicht. Generierte Dateien
  verweisen nie. **S9:** generierte Dateien ohne Fund (`tot_generiert_uebersprungen`),
  Konventionsnamen nie `sicher_tot` (Grund `konvention`). Die Muster sind feste Listen. Andere
  Konventionen (z. B. camelCase-`onClick`) deckt erst der Namensverweis ab.
- **R15:** TS-Namensimporte sind unsicher. TS gilt als erreichbar, sobald irgendeine Kante trifft.
- **R22:** Rust ist nie `sicher_tot` (4 291 Rust-Symbole stehen in `wahrscheinlich_tot`).
- **Dynamische Aufrufe:** Registries, Plugins, String-Dispatch, Routen-Registrierung, Callbacks und
  JSX lassen lebenden Code tot aussehen. Die drei Stufen dämpfen das, die Regeldatei kann Einstiegspunkte
  nachtragen. Seit S8 fängt der Namensverweis die meisten dieser Fälle ab, darunter Python-Klassen nur
  per Attribut oder Typ und Testimporte aus src-Layouts. Seit S9 sind auch Dispatch-Präfixe und
  Build-Ausgabe abgedeckt, seit Fix-Runde 3 auch Verweise aus HTML und Vorlagen. Übrig bleiben Namen,
  die nur in YAML oder JSON registriert sind, aus Strings zusammengesetzt werden (`getattr(x, "do_" + m)`
  mit unbekanntem Präfix) oder kürzer als 3 Zeichen sind (14 `sicher_tot`).
- Bereichsnamen tragen bei Bereichen aus einem Repo kein Repo-Präfix (`src/services` kann in jedem Repo
  liegen).
- Laufzeit: `laufzeit_mathe_ms` ist nicht nach Teilschritten aufgeschlüsselt. Mit
  `--pruefen-reproduzierbar` verdoppelt sich die Laufzeit.

## Nächster Schritt (für dich)

Die Spec-Abnahme „27 von 30 bestätigt“ ist **noch nicht gemessen**. Sie entsteht erst mit deinen Urteilen.

1. `docs/selbstbild/goldsatz-vibemind.yaml` öffnen. Er wurde nach Fix-Runde 3 aus Lauf a5 neu erzeugt
   und trägt `commit: 1e409b44…`. Er enthält 40 `sicher_tot`-Kandidaten, 15 Doppelungen und die 3
   größten Bereiche. Alle Urteile stehen auf `offen`. **Nicht** erneut `goldsatz` darauf laufen
   lassen: Ohne `--ueberschreiben` lehnt der Befehl ab, mit dem Schalter wären deine Urteile weg.
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

Achtung: Der Goldsatz gilt für den Commit `1e409b44`. Bei einem anderen Commit warnt `vergleichen` auf
stderr und nennt die Zahl der verschwundenen Symbole. Diese Einträge zählen als `ignoriert`.
Einen neuen Goldsatz erzeugst du nicht über die ausgefüllte Datei, sondern in eine neue Datei.
