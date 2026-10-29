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
| c) `goldsatz` VibeMind | `1e409b44` | 1,1 s | – | 0,05 GB | – |

- Das Speicherbudget von 2,5 GB Spitze hält. Der VibeMind-Lauf braucht 1,15 GB im Python-Prozess. Die
  1,44 GB im Prozessbaum zählen die `git cat-file`-Kindprozesse der Doppelungssuche mit.
- Der private Commit liegt höher als der Working Set (auch bei sales-claw 0,85 GB). Vermutlich
  reservieren numpy, BLAS und igraph Puffer, ohne sie zu belegen. Bei knappem Commit-Limit des Hosts
  zählt dieser Wert.
- a0 war nur fast ein Neulauf: 19 606 von 19 608 Dateien wurden neu geparst, nur 2 kamen aus dem Cache.
  Der Parse-Cache des Stands von `6f8e4e0f` passte also nicht mehr. Das gehört zu Stufe 1 und wurde
  hier nicht untersucht. Graph danach: 234 882 Knoten, 482 618 Kanten (23 416 unsicher), 89 fehlerhafte Dateien.
- **Determinismus:** Der Export von a1 (16,1 MB, sha256 `c17d3ecf…a91645`) und der von a2 stammen aus
  zwei Prozessen. Sie unterscheiden sich nur in den zwei erwarteten `reproduzierbar_*`-Zeilen. Im
  selben Prozess ergab die zweite Rechnung `reproduzierbar_mathe = 1` und `reproduzierbar_kolonie = 1`.
- `luecken` auf VibeMind: `kante_ohne_knoten = 0` und `symbol_ohne_datei = 0`, auf sales-claw ebenso.
  Der Schlüssel `symbol_ohne_datei` kommt aus `dateigraph`. Er ist da, weil `luecken` erst nach allen
  Analysen gebaut wird (Ruling S3, abgesichert durch einen Test).

## Ergebnisse VibeMind (`1e409b44`, 20 Repos, 19 608 Dateien, 205 k Symbole)

### Toter Code je Stufe

| Stufe | Anzahl | häufigste Gründe | Repos mit den meisten Funden |
|---|---|---|---|
| `sicher_tot` | **11 788** (in 2 361 Dateien) | per Definition alle `unerreichbar` + `alt_90` + `kein_doku_verweis` | hermes-agent 4 638, coding-engine 2 473, vibemind-os 1 323, swe_desgine 1 034, voice 768, openclaw 529 |
| `wahrscheinlich_tot` | 19 720 | `unerreichbar` 17 445, `alt_90` 14 138, `kein_doku_verweis` 9 581, `doku_verweis` 8 590, `rust_r22` 4 291, `ts_klasse` 3 145, `jsx_komponente` 2 364, `nur_unsicher` 2 275 | openfang 4 349, vibemind-os 3 762, hermes-agent 2 892, coding-engine 2 819 |
| `verdaechtig` | 21 912 | `nur_test` (einziger Grund dieser Stufe) | hermes-agent 9 731, openfang 3 863, coding-engine 2 479, openclaw 1 715 |

Je 5 Beispiele zum schnellen Gegenprüfen (gezogen mit festem Seed):

- `sicher_tot`
  - `symbol:vibemind/vibemind-os/coding-engine:mcp_plugins/servers/time/user_interaction_utils.py#create_ask_user_tool.AskUserArgs`
  - `symbol:vibemind/vibemind-os/spaces/shuttles/swe_desgine:requirements_engineer/gates/quality_gate.py#GateStatus`
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/timebase/ranges.py#MediaRange.expected_seq_length`
  - `symbol:vibemind/vibemind-os/voice:python/publishing/n8n_publisher.py#N8nPublisher.publish_all_workflows`
  - `symbol:vibemind/vibemind-os:spaces/rowboat/rowboat/apps/rowboat/app/projects/[projectId]/manage-triggers/components/triggers-tab.tsx#TriggersTab.handleBackToList`
- `wahrscheinlich_tot`
  - `symbol:vibemind/vibemind-os/coding-engine:src/agents/redis_pubsub_agent.py#RedisPubSubAgent._create_claude_code_tools.generate_redis_code` (`alt_90`, `nur_unsicher`)
  - `symbol:vibemind/vibemind-os/openfang:crates/openfang-skills/src/config_injection.rs#SkillConfigVar` (`rust_r22`)
  - `symbol:vibemind/vibemind-os:spaces/rowboat/rowboat/apps/rowboat/src/infrastructure/repositories/mongodb.conversations.repository.ts#MongoDBConversationsRepository` (`ts_klasse`)
  - `symbol:vibemind/vibemind-os:spaces/rowboat/rowboat/apps/x/apps/renderer/src/components/settings-dialog.tsx#SettingsDialog.handleTabChange` (jung: kein `alt_90`)
  - `symbol:vibemind/vibemind-os:spaces/rowboat/rowboat/apps/x/apps/renderer/src/extensions/chart-block.tsx#addNodeView` (`ts_klasse`)
- `verdaechtig` (`nur_test`)
  - `symbol:vibemind/vibemind-os/coding-engine:src/services/pipeline_data_coalescer.py#PipelineDataCoalescer.__init__`
  - `symbol:vibemind/vibemind-os/spaces/captain_cook:agenten/execution/codex_supervisor.py#CodexSupervisor.run`
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/analysis/crop_controller.py#face_detection_available`
  - `symbol:vibemind/vibemind-os/spaces/video/laura:services/local-api/src/laura/interchange/fcp7_xml.py#timeline_to_fcp7_xml`
  - `symbol:vibemind/vibemind-os:spaces/marketing/claw/logo_bearbeiten.py#LogoFehler`

**Stichprobe vor dem Goldsatz (eigene Prüfung per `git grep` am Commit, 4 der 5 `sicher_tot`-Beispiele):**

- `GateStatus` ist **falsch**. Die Enum wird nur per Attribut benutzt (`GateStatus.PASS`), als
  Annotation und als Re-Export in `gates/__init__.py`. Nichts davon erzeugt eine `ruft_auf`-Kante.
- `MediaRange.expected_seq_length` ist **falsch**. Die Methode wird von `tests/test_ranges.py` aufgerufen
  und hätte damit `verdaechtig` sein müssen. Der Testimport `laura.…` aus dem src-Layout
  (`services/local-api/src`) wird nicht aufgelöst, deshalb fehlt auch die Testkante.
- `N8nPublisher.publish_all_workflows` ist plausibel richtig: kein Aufrufer im Repo.
- `create_ask_user_tool.AskUserArgs` (Kopie im `time`-Server) ist plausibel richtig. Die Server
  importieren jeweils ihre eigene Kopie, und `time` nutzt sie nicht.

Daraus ergeben sich zwei systematische Fehlerquellen für falsches `sicher_tot`, die der Goldsatz messen wird:

1. **Python-Klassen ohne Aufruf:** Enums, Dataclasses nur als Typ, Basisklassen, `isinstance` und
   Konstanten-Klassen. Unter den 6 270 `sicher_tot` in `.py` haben 1 241 einen großgeschriebenen
   obersten Namen (vermutlich Klassen). Weitere 2 579 sind darunter verschachtelt (Methoden toter
   Klassen, E4). Bis zu ein Drittel aller `sicher_tot` hängt also an „Klasse hat keine eingehende
   Aufrufkante“.
2. **Verschachteltes in JSX-Komponenten und TS-Klassen:** Ruling S4 nimmt die Komponente oder Klasse
   selbst von `sicher_tot` aus, nicht aber ihre verschachtelten Handler (`TriggersTab.handleBackToList`).
   Ist die Komponente unerreichbar, weil `<Foo/>` keine Kante erzeugt, gelten ihre Handler als
   unerreichbar. Das betrifft 1 229 `sicher_tot` in TS/JS mit großgeschriebenem Elternsymbol. Weitere
   4 031 sind oberste TS/JS-Funktionen. Viele davon werden vermutlich als Wert übergeben (Callbacks,
   Exporte), ohne dass eine Kante entsteht.

Beides ist **nicht behoben** (Task 11 verdrahtet und misst, die Regeln stammen aus Task 4). Es ist der
naheliegende nächste Fix, falls der Goldsatz die Präzision unter 27/30 drückt. Für einen Fix spricht:
Verschachteltes erbt die Ausnahmen `jsx_komponente` und `ts_klasse` seines Elternsymbols, und
Python-Klassen werden nie `sicher_tot`, solange es nur Attribut- oder Typverweise geben kann.

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
| Laufzeit | 92,3 s (Teilzeiten nicht einzeln gemessen. Vermutlich dominiert das Blob-Lesen für Doku und Doppelungen) | **0,76 s** |
| Reproduzierbar (zweiter Lauf im Prozess) | ja | ja |
| Zentrum | PageRank-Top-50 (auf Dateien abgebildet) | 50 meistbesuchte Dateien |
| Überlappung Zentrum | **7 von 50** gemeinsam | |
| Abdeckung | alle 19 608 Dateien | **486 besucht (2,5 %)**, 19 122 „tote Ecken“ |
| tote Ecken gegen toten Code | 5 093 von 5 347 Dateien mit Totfund liegen in toten Ecken, bei `sicher_tot` 2 179 von 2 361 | |
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
  Richtung). Umgekehrt werden alle Methoden einer fälschlich toten Klasse mit `sicher_tot` (s. o.).
- **S1:** Hotspot-Größe = Anzahl Symbole, nicht Zeilen. **S2:** Testdateien nur nach Namensmuster
  (plus `conftest.py`). **S3:** `luecken` nach allen Analysen. **S4:** Go-Methoden, TS-Klassen und
  JSX-Komponenten nie `sicher_tot`, deren verschachtelte Symbole aber schon (Lücke, s. o.). **S5:** Ein
  Doku-Verweis zählt auch aus der Doku von Vorfahren-Repos. **S6:** Zeilenregel für `gleiches_symbol`
  über erneutes Parsen der Kandidaten. **S7:** Ähnlichkeits-Cluster per Union-Find, Eimer über 50 Blobs
  übersprungen (36 bei VibeMind).
- **R15:** TS-Namensimporte sind unsicher. TS gilt als erreichbar, sobald irgendeine Kante trifft.
- **R22:** Rust ist nie `sicher_tot` (4 291 Rust-Symbole stehen in `wahrscheinlich_tot`).
- **Dynamische Aufrufe:** Registries, Plugins, String-Dispatch, Routen-Registrierung, Callbacks und
  JSX lassen lebenden Code tot aussehen. Die drei Stufen dämpfen das, die Regeldatei kann Einstiegspunkte
  nachtragen. Neu gemessen: Python-Klassen, die nur per Attribut oder Typ benutzt werden, und
  Testimporte aus src-Layouts, die nicht aufgelöst werden.
- Bereichsnamen tragen bei Bereichen aus einem Repo kein Repo-Präfix (`src/services` kann in jedem Repo
  liegen).
- Laufzeit: `laufzeit_mathe_ms` ist nicht nach Teilschritten aufgeschlüsselt. Mit
  `--pruefen-reproduzierbar` verdoppelt sich die Laufzeit.

## Nächster Schritt (für dich)

Die Spec-Abnahme „27 von 30 bestätigt“ ist **noch nicht gemessen**. Sie entsteht erst mit deinen Urteilen.

1. `docs/selbstbild/goldsatz-vibemind.yaml` öffnen. Er enthält 40 `sicher_tot`-Kandidaten, 15
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
