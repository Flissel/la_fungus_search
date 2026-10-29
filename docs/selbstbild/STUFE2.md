# Selbstbild Stufe 2: Mathe auf dem Codegraphen (Abnahme)

Stand: 29.10.2026 (Final-Fix-Welle, gültiger Lauf a7). Paket `src/selbstbild/analyse/`, Lauf in `analyse/lauf.py`, CLI in `selbstbild/__main__.py`.
Stufe 2 liest nur den Graphen aus Stufe 1 (`graph.sqlite`, nur lesend) und die Blobs am vermessenen
Commit. Sie schreibt `analyse.sqlite` neben den Stand. Kein LLM, kein Netz.

**Python 3.11 bis 3.12.** Der Selbstbild-Stack braucht `numpy==1.26.4`, und dafür gibt es keine
Wheels für Python 3.13 und neuer. Erprobt ist er nur auf 3.11, mit den Windows-Wheels von igraph und
leidenalg. Ein Extra kann kein eigenes `requires-python` haben, deshalb steht die Untergrenze als
Kommentar beim Extra `selbstbild` in `pyproject.toml`.

## Befehle (PowerShell, im la_fungus-Arbeitsbaum)

```powershell
$env:PYTHONPATH="src"
# Analyse des vermessenen Stands. Ergebnis wird in analyse.sqlite gespeichert, Zusammenfassung als JSON
.\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --export "$env:TEMP\vibemind-analyse.json"
.\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --ohne-kolonie
.\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --pruefen-reproduzierbar   # rechnet alles zweimal
.\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --kolonie-budget   # plus budgetgleiche Kolonie (S11)
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
- Exit-Codes: 1 heißt nicht vermessen, nicht analysiert oder `graph.sqlite` unlesbar
  (`StandUnlesbar`, z. B. gesperrt oder falsches Schema, mit dem Text des DB-Fehlers). 3 heißt Stand ohne brauchbaren Graphen
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
`laufzeit_` oder `reproduzierbar_` sind Messwerte und stehen **nicht** im Export (dazu zählen auch
`laufzeit_kolonie_budget_ms` und `reproduzierbar_kolonie_budget`). Sie liegen in
`analyse.sqlite` in der Nebentabelle `messung`, und `vergleichen` liest sie von dort. Gleiche Parameter
ergeben damit immer einen byte-gleichen `analyse.json`, mit oder ohne Prüf-Schalter. Ein Test prüft
das über zwei Prozesse mit verschiedenem `PYTHONHASHSEED`. Im Export bleiben aus `statistik`
`doppelt_eimer_uebersprungen`, `tot_generiert_uebersprungen`, `pagerank_quersprache_ausgelassen`
(S14) und mit `--kolonie-budget` die Kennzahlen `kolonie_budget_*` (S11). `parameter` enthält unter
`regeln` den Digest der Regelkonstanten (S13). Ändert sich eine Regel, ändert sich der Schlüssel des
gespeicherten Laufs, und es braucht eine neue Analyse.

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

## Final-Fix-Welle (Rulings S10 bis S14, Lauf a7, gültiger Stand)

Die Gesamtprüfung von Stufe 2 fand weitere Fehlerquellen. Diese Welle behebt sie, danach wurde
VibeMind neu vermessen (a6) und neu analysiert (a7). **a7 ersetzt a5.** Die Abschnitte weiter unten
mit „Lauf a5“ sind die Vorgeschichte und nicht mehr gültig.

Was sich geändert hat:

- **S10 Decorators (Stufe 1 und 2).** Die Parser für Python und TS halten je Symbol die
  Decorator-Köpfe ohne Argumente fest, sortiert als Attr `dekoriert=kopf.timer,property`.
  `PARSER_VERSION` steigt dafür von 3 auf 4. Ein dekoriertes Symbol ist nie `sicher_tot` (Grund
  `dekorator`), außer **alle** seine Decorators sind inert: `property`, `staticmethod`,
  `classmethod`, `dataclass`, `dataclasses.dataclass`, `functools.*`, `abstractmethod`,
  `abc.abstractmethod`, `overload`, `typing.overload`, `cached_property`, `contextmanager`,
  `contextlib.contextmanager`, `wraps`, `total_ordering`, `final`, `typing.final` und `override`.
  Verschachteltes erbt die Ausnahme, zum Beispiel die Methoden einer per Decorator registrierten
  Klasse. Das ist die vorsichtige Richtung.
- **S2, zweite Erweiterung.** `ist_testdatei` erkennt zusätzlich die Ordner `test/`, `spec/`, `e2e/`
  und `__mocks__/` sowie die Dateinamen `*.test.(jsx|mjs|cjs|mts|cts)` und `*.stories.*`.
- **S9, Konventionen ergänzt.** Neu dabei sind `pytest_*`, `upgrade`/`downgrade` (alembic),
  `lambda_handler` und `handle` (Django). Go `init` zählt nur in `.go`-Dateien.
- **S8, mehr gelesene Dateien.** Der Namensverweis-Scan liest jetzt auch JSON und YAML (ohne
  Lock-Dateien wie `package-lock.json` oder `pnpm-lock.yaml`), dazu `*.toml`, `*.cfg`, `*.ini`,
  `Dockerfile*`, `Makefile` und `Procfile`. Bezeichner sind Unicode-fähig (`[^\W\d]\w*`), auch in
  den Doku-Verweisen.
- **S11 Kolonie.** Die Agenten starten gestreut über alle Wurzeldateien, gemischt mit
  `rng.permutation`. Rundlauf gibt es nur, wenn es mehr Agenten als Wurzeln gibt. Wer in eine
  Sackgasse läuft, startet auf der nächsten Wurzel eines gemeinsamen Rundlauf-Zählers über alle
  Wurzeln neu, nicht mehr auf seiner eigenen. Neu ist ein budgetgleicher Lauf: Agenten-Schritte
  ≈ 10 × Dateien, also `ceil(10 × Dateien / 192)` Schritte, höchstens 5000. Er läuft nur mit
  `--kolonie-budget`, seine Kennzahlen stehen in `statistik` unter `kolonie_budget_*`.
- **S12 Goldsatz-Metrik.** Gewertet werden nur Gold-Einträge, die im aktuellen Lauf `sicher_tot`
  sind. Die übrigen zählen als `ignoriert`. Für die Kolonie zählt jetzt die Übereinstimmung über
  **alle** beurteilten Einträge: ((ja ∧ tote Ecke) + (nein ∧ keine tote Ecke)) / (ja + nein). Die
  alte Metrik konnte nicht verlieren.
- **S13.** `parameter()` trägt unter `regeln` einen Digest aller Regelkonstanten: Konventionen,
  Vorlagen- und Konfig-Endungen, Code-Sprachen, Generiert- und Testmuster, inerte Decorators,
  Sprachfamilien und Bezeichner-Regex. Das ersetzt das manuelle `tot_regeln=s9`. Ändert sich eine
  Regel, sind gespeicherte Läufe ungültig, und `vergleichen`/`goldsatz` verlangen „erst analysieren“.
- **S14 PageRank** (Befund siehe unten). `ruft_auf`-Kanten zwischen verschiedenen Sprachfamilien
  gehen nicht mehr in PageRank und Hotspots ein. Die Familien sind Python, JS/TS, Rust, Go und Rest.
  Ihre Zahl steht in `statistik` als `pagerank_quersprache_ausgelassen`. Unsichere Kanten zählen
  weiter mit 0,5.
- **Weitere Korrekturen:**
  - Hotspots schließen Testdateien standardmäßig aus (`mit_tests=False`, auch in den Perzentilen).
  - „Top-50“ meint überall 50 **Dateien** (`zentral.dateien_top`).
  - Bereichsnamen tragen immer das Repo (`<repo>/<präfix>`), Submodul-Repos teilen den Pfad ihres
    Elternrepos.
  - Die Goldsatz-Bereiche sind die drei größten unter `spaces/`, auch in Submodul-Repos. Gibt es
    weniger, wird aufgefüllt.
  - Doppelungen zählen nur Code. `gleiche_datei` und `aehnliche_datei` ignorieren json, yaml,
    markdown, sql, dotenv und Dateien ohne Sprache.
  - Ein Ähnlichkeits-Cluster aus nur einer Blob-SHA entfällt. Das ist ein Schutz, nach der
    Union-Find-Logik kann so ein Cluster gar nicht entstehen.
  - Die Zeilenregel parst über `_parsen_sicher`.
  - `laden` meldet nur bei fehlender Tabelle `StandOhneGraph`. Andere DB-Fehler gehen mit ihrem Text
    als `StandUnlesbar` weiter (CLI Exit 1).
  - `gemessen_um` kommt aus derselben Lese-Transaktion wie der Graph.

### Abnahme der Welle (`1e409b44`)

| Lauf | Dauer | Peak WS Python (selbst gemessen) | Peak WS Prozessbaum (abgetastet) | Peak privater Commit |
|---|---|---|---|---|
| a6) `vermessen` VibeMind, inkrementell nach `PARSER_VERSION` 4 | 227 s | nicht gemessen (Wächter-Fehler, s. u.) | nicht gemessen | – |
| **a7) `analysieren` VibeMind `--pruefen-reproduzierbar --kolonie-budget`** | **251 s** (Laden 4,0 s, Mathe 122,3 s, Kolonie 0,96 s, budgetgleiche Kolonie 0,76 s, je einmal gerechnet) | **1,10 GB** | 1,47 GB | 1,82 GB |
| **c5) `goldsatz` VibeMind `--ueberschreiben`** | ~1 s | – | – | – |

- Vor jedem Lauf war der freie Commit über 3 GB, vor a6 4,9 GB und vor a7 4,8 GB. Der Wächter bricht
  ab bei mehr als 2,5 GB im Prozessbaum oder nach mehr als 30 min. Keiner der Läufe wurde
  abgebrochen. Das Budget von 2,5 GB hält, die Spitze liegt bei 1,47 GB.
- **a6 ohne Speichermessung.** Der Wächter hatte einen Fehler im eigenen Skript: Bei
  `@([double]$ws / 1GB, …)` bindet in PowerShell das Komma stärker als `/`. Jede Abtastung warf deshalb
  eine Ausnahme, und die wurde verschluckt. Für a6 gibt es also keine Speicherwerte, und der
  Abbruch-Wächter war während a6 wirkungslos. Vor a7 war der Fehler behoben. Vergleichswert: Der
  letzte volle Neulauf a0 brauchte 0,85 GB.
- a6 parste 19 606 Dateien neu, also fast alle. Das ist erwartet, weil `PARSER_VERSION` erhöht wurde.
  Der Graph ist unverändert groß: 234 882 Knoten, 482 618 Kanten (23 416 unsicher), 89 fehlerhafte
  Dateien.
- Reproduzierbar im selben Prozess: `reproduzierbar_mathe`, `reproduzierbar_kolonie` und
  `reproduzierbar_kolonie_budget` sind alle 1. Export a7: 15,7 MB, sha256 `87cf1d6d…afd51cd`.

### Toter Code (a7)

| Stufe | a5 (vorher) | **a7** | häufigste Gründe (a7) |
|---|---|---|---|
| `sicher_tot` | 1 151 (563 Dateien) | **1 109** (546 Dateien) | per Definition alle `unerreichbar` + `alt_90` + `kein_doku_verweis` |
| `wahrscheinlich_tot` | 29 888 | 29 923 | `unerreichbar` 27 648, `alt_90` 24 345, `kein_doku_verweis` 19 951, `namensverweis` 9 163, `doku_verweis` 8 423, `rust_r22` 4 291, `jsx_komponente` 3 839, `ts_klasse` 3 181, `nur_unsicher` 2 275, `methode_lebender_klasse` 1 448, `dunder` 696, **`dekorator` 479**, `fehlerhaft` 453, **`konvention` 348**, `spezial_fehlt` 179 |
| `verdaechtig` | 21 911 | 21 900 | `nur_test` |

- **Übergänge von a5 nach a7:** Von den 1 151 alten `sicher_tot` bleiben 1 108. 33 wurden
  `wahrscheinlich_tot` wegen `dekorator`, 10 wegen `namensverweis` (durch die neu gelesenen
  Konfig-/Datendateien). Ein Symbol ist neu `sicher_tot`, nämlich coding-engine
  `redis_tool.py#RedisTool.llen`. In a5 hat es nur ein Tokenizer-Artefakt gerettet: Der ASCII-Regex
  zerlegte „Füllen“ in `F` und `llen`, belegt in `playwright_agent.py:317`. Mit Unicode-Bezeichnern
  fällt dieser Scheintreffer weg. Das ist richtig so.
- 109 Funde haben `dekorator` als einzige Ausnahme. 33 davon waren vorher `sicher_tot`, die übrigen
  76 hatte schon der Namensverweis gerettet.
- **Die zwei bekannten Fehlgriffe** sind jetzt `wahrscheinlich_tot` mit `dekorator`. Beide sind
  weiter `unerreichbar`, `alt_90` und `kein_doku_verweis`:
  - coding-engine `cell_cli/commands/marketplace.py#list_installed`, ein Click-Kommando
  - coding-engine `src/colony/k8s/operator.py#cell_health_timer`, ein `kopf.timer`
- Nach Endung verteilen sich die 1 109 so: 839 `.py`, 180 `.ts`, 75 `.js`, 11 `.tsx` und 4 `.jsx`.
  Die meisten liegen in coding-engine (372), danach folgen hermes-agent (222), vibemind-os (149),
  swe_desgine (125), voice (103) und openclaw (50). 14 Namen haben weniger als 3 Zeichen.

### S14: Befund zu den PageRank-Spitzen (a5, nur lesend belegt)

Untersucht wurden die 10 größten PageRank-Symbole aus a5. Für jedes wurden die eingehenden Kanten
nach Art, Sicherheit und Sprachfamilie der Quelle gezählt, nur lesend aus `graph.sqlite`.

| Rang a5 | Symbol | Eingang | Herkunft |
|---|---|---|---|
| 1 (0,0114) | hermes-agent `ui-tui/src/components/branding.tsx#SessionPanel.strip` | 2 172 | **2 169 unsichere `ruft_auf` aus Python-Dateien** (`.strip()` per Namens-Rückfall auf eine TSX-Methode), 2 aus TS |
| 2 (0,0055) | coding-engine `cell_cli/commands/cells.py#list` | 1 626 | 1 624 unsichere `ruft_auf` aus Python, gleiche Familie (eingebautes `list(...)` per Name auf die Funktion `list`) |
| 3, 4, 7, 10 | coding-engine `cell_cli/config.py#ensure_config_dir`, `cells.py#get_auth_headers`, `config.py#get_credentials`, `config.py#get_config` | 5 / 4 / 8 / 25 | wenige Kanten, sicher oder unsicher, alle Python |
| 5, 6, 9 | hermes-agent `get_hermes_home`, `PlatformConfig`, `load_config` | 304 / 943 / 462 | fast alle sicher, Python, echte Zentralität |
| 8 (0,0018) | hermes-agent `apps/desktop/…/markdown-text.tsx#useOpenMediaFile.open` | 425 | **420 unsichere `ruft_auf` aus Python** (`open(...)` → TSX-Methode `open`) |

**Entscheidung:** Rang 1 und Rang 8 stammen aus sprachübergreifenden, unsicheren Rückfallkanten.
Deshalb gilt die Regel aus S14. Auf VibeMind fallen damit 5 405 `ruft_auf`-Kanten aus PageRank und
Hotspots. Danach führen in a7 coding-engine `cells.py#list` (0,0056), `config.py#ensure_config_dir`,
hermes-agent `get_hermes_home`, `cells.py#get_auth_headers` und `PlatformConfig`.

**Offen:** Rang 1 in a7, `cells.py#list`, ist weiterhin ein Artefakt, diesmal innerhalb derselben
Sprache. Das eingebaute `list(...)` wird über den Namen der gleichnamigen Funktion zugeordnet. Die
Kanten sind unsicher und zählen nur mit 0,5, die Regel aus S14 greift aber nicht. Ein Fix gehört in die
Namensauflösung von Stufe 1, etwa „keine Rückfallkante für Python-Builtins“, und wäre ein eigenes
Ruling.

### Weitere Zahlen (a7)

- **Hotspots ohne Testdateien:**
  - Top 10: hermes-agent `run_agent.py` (0,99819), `hermes_cli/config.py`, `agent/auxiliary_client.py`,
    openfang `crates/openfang-kernel/src/kernel.rs`, hermes-agent `hermes_cli/auth.py`,
    `plugins/platforms/telegram/adapter.py`, `hermes_state.py`, `plugins/platforms/discord/adapter.py`,
    `apps/desktop/electron/main.ts` und `gateway/platforms/base.py`.
  - In a5 waren 5 der Top 10 Testdateien.
- **Betweenness** ist unverändert, Rang 1 bleibt `run_agent.py`.
- **Bereiche:**
  - 428 Bereiche, `(isoliert)` mit 8 387 Dateien. Leiden ist unverändert, nur die Namen tragen jetzt
    das Repo.
  - Die größten sind `…/hermes-agent/apps/desktop/electron` (829), `…/coding-engine/src/services`
    (745), `…/hermes-agent/tests/gateway` (552) und `…/spaces/video/laura/services/local-api` (518).
  - Der Goldsatz wählt `bereich:2`, `bereich:4` und `bereich:5`, alle drei unter `spaces/`.
- **Doppelungen** (nur noch Code):
  - `gleiche_datei` 188, vorher 261. Größtes Beispiel ist `llm_client.py` fünfmal.
  - `gleiches_symbol` 1 329, unverändert. Größtes Beispiel ist `on_change` in 114 coding-engine-Services.
  - `aehnliche_datei` 142 Cluster, vorher 201. Größtes Beispiel sind 28 `tests/test_agent_task_*.py`
    (0,9046).
  - 36 LSH-Eimer wurden übersprungen.
- `luecken` ist 0/0. 709 generierte Symbole wurden übersprungen.

### Kolonie gegen Mathe (a7)

| | Mathe | Kolonie Standard (192 × 400) | Kolonie budgetgleich (192 × 1 022 ≈ 10 × 19 608) |
|---|---|---|---|
| Laufzeit | 122,3 s | 0,96 s | 0,76 s |
| Reproduzierbar | ja | ja | ja |
| Abdeckung | alle 19 608 Dateien | **2 893 besucht (14,8 %)**, 16 715 tote Ecken (a5: 486 besucht) | 2 897 besucht (14,8 %), 16 711 tote Ecken |
| Zentrum ∩ PageRank-Top-50-Dateien | – | **13 von 50** (a5: 7) | 13 von 50 |
| Dateien mit `sicher_tot` in toten Ecken | – | 371 von 546 | 370 von 546 |
| Dateien mit Totfund in toten Ecken | – | 4 324 von 5 334 | 4 321 von 5 334 |
| Bereiche | 428 | 1 (2 Dateien) | 1 |

**Fazit (ehrlich):**

- Der gestreute Start mit rundlaufendem Neustart behebt den Startfehler. Die Kolonie besucht
  sechsmal so viele Dateien wie in a5, und die Überlappung mit dem Zentrum steigt von 7 auf 13.
- Mehr Schritte bringen fast nichts. Der budgetgleiche Lauf mit 2,6-mal so vielen Schritten besucht
  nur 4 Dateien mehr. Die Abdeckung ist also durch das begrenzt, was von den Wurzeln über Kanten
  erreichbar ist, nicht durch das Budget. 85 % der Dateien bleiben tote Ecken.
- Dass die toten Ecken viele Totfunde enthalten (371 von 546 Dateien mit `sicher_tot`), ist darum kein
  eigenes Signal. Nicht besucht wird ohnehin fast alles.
- Ins Kolonie-Zentrum geraten auch 3 generierte Bundles (`dist-embedded/*.js`). Die Kolonie kennt die
  Regel aus S9 nicht.
- Als Herausforderer bleibt die Kolonie bei Bereichen (1 gegen 428) und Zentrum klar hinter der
  Mathe. Endgültig misst das `vergleichen` am Goldsatz. Seit S12 kann die Kolonie dort auch verlieren.

## Vorgeschichte: Ergebnisse VibeMind vor der Final-Fix-Welle (Lauf a5, ersetzt; `1e409b44`, 20 Repos, 19 608 Dateien, 205 k Symbole)

Alles in diesem Abschnitt beschreibt den Stand **vor** S10 bis S14. Die gültigen Zahlen stehen oben im
Abschnitt zur Final-Fix-Welle.

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

Fazit (a5, überholt; Startfehler seit S11 behoben, siehe Final-Fix-Welle): Die Kolonie ist schnell und deterministisch. Mit den MCMP-Standardwerten (192 Agenten, 400
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
- **S1:** Hotspot-Größe = Anzahl Symbole, nicht Zeilen. Testdateien zählen in den Hotspots
  standardmäßig nicht. **S2:** Testdateien nur nach Namens- und Ordnermuster (plus `conftest.py`,
  seit der Final-Fix-Welle auch `test/`, `spec/`, `e2e/`, `__mocks__/`, `*.stories.*`). **S3:** `luecken` nach allen Analysen. **S4:** Go-Methoden, TS-Klassen und
  JSX-Komponenten nie `sicher_tot`. Seit S8 erben verschachtelte Symbole diese Ausnahmen. **S5:** Ein
  Doku-Verweis zählt auch aus der Doku von Vorfahren-Repos. **S6:** Zeilenregel für `gleiches_symbol`
  über erneutes Parsen der Kandidaten. **S7:** Ähnlichkeits-Cluster per Union-Find, Eimer über 50 Blobs
  übersprungen (36 bei VibeMind). **S8:** Namensverweis-Wächter und vererbte S4/R22-Ausnahmen (s. o.).
  Er ist absichtlich grob: Ein gleichnamiger Bezeichner irgendwo im Code genügt zum Abstufen, auch ein
  zufällig gleicher Name. Das ist die sichere Richtung. Namen unter 3 Zeichen werden nicht geprüft.
  Seit Fix-Runde 3 zählen auch HTML und Vorlagen (Beispiel `handsPage`, s. o.). Seit der
  Final-Fix-Welle kommen JSON und YAML (ohne Lock-Dateien) sowie `*.toml`, `*.cfg`, `*.ini`,
  `Dockerfile*`, `Makefile` und `Procfile` dazu. Markdown und SQL zählen weiter nicht. Generierte Dateien
  verweisen nie. **S9:** generierte Dateien ohne Fund (`tot_generiert_uebersprungen`),
  Konventionsnamen nie `sicher_tot` (Grund `konvention`). Die Muster sind feste Listen. Andere
  Konventionen (z. B. camelCase-`onClick`) deckt erst der Namensverweis ab.
- **S10:** Dekorierte Symbole sind nie `sicher_tot`, außer alle Decorators sind inert. Auch tote
  dekorierte Funktionen erscheinen deshalb nur als `wahrscheinlich_tot`. Erkannt wird nur der
  Ausdruckskopf: `@register(...)` und `@app.register(...)` gelten als verschiedene Namen, beide als
  nicht inert. **S11:** Mehr Schritte erhöhen die Abdeckung der Kolonie kaum (s. o.). **S12:** Der
  Goldsatz wertet nur Einträge, die aktuell `sicher_tot` sind. **S13:** Jede Regeländerung erzwingt
  eine Neuanalyse. **S14:** Echte sprachübergreifende Aufrufe (selten) zählen nicht in PageRank und
  Hotspots. Gleichsprachige Builtin-Artefakte wie `cells.py#list` bleiben bestehen.
- **R15:** TS-Namensimporte sind unsicher. TS gilt als erreichbar, sobald irgendeine Kante trifft.
- **R22:** Rust ist nie `sicher_tot` (4 291 Rust-Symbole stehen in `wahrscheinlich_tot`).
- **Dynamische Aufrufe:** Registries, Plugins, String-Dispatch, Routen-Registrierung, Callbacks und
  JSX lassen lebenden Code tot aussehen. Die drei Stufen dämpfen das, die Regeldatei kann Einstiegspunkte
  nachtragen. Seit S8 fängt der Namensverweis die meisten dieser Fälle ab, darunter Python-Klassen nur
  per Attribut oder Typ und Testimporte aus src-Layouts. Seit S9 sind auch Dispatch-Präfixe und
  Build-Ausgabe abgedeckt, seit Fix-Runde 3 auch Verweise aus HTML und Vorlagen. Übrig bleiben Namen,
  die nur in YAML oder JSON registriert sind, aus Strings zusammengesetzt werden (`getattr(x, "do_" + m)`
  mit unbekanntem Präfix) oder kürzer als 3 Zeichen sind (14 `sicher_tot`).
- Bereichsnamen tragen seit der Final-Fix-Welle immer das Repo (`<repo>/<präfix>`). Die Tabelle im
  Abschnitt „Vorgeschichte“ zeigt noch die alten Namen ohne Präfix.
- Laufzeit: `laufzeit_mathe_ms` ist nicht nach Teilschritten aufgeschlüsselt. Mit
  `--pruefen-reproduzierbar` verdoppelt sich die Laufzeit.

## Nächster Schritt (für dich)

Die Spec-Abnahme „27 von 30 bestätigt“ ist **noch nicht gemessen**. Sie entsteht erst mit deinen Urteilen.

1. `docs/selbstbild/goldsatz-vibemind.yaml` öffnen. Er wurde nach der Final-Fix-Welle aus Lauf a7
   neu erzeugt und trägt `commit: 1e409b44…`. Er enthält 40 `sicher_tot`-Kandidaten, 15 Doppelungen
   und die 3 größten Bereiche unter `spaces/` (`bereich:2`, `bereich:4` und `bereich:5`). Alle Urteile stehen auf `offen`. **Nicht** erneut `goldsatz` darauf laufen
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
$env:PYTHONPATH="src"; .\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --pruefen-reproduzierbar --kolonie-budget
```

Achtung: Der Goldsatz gilt für den Commit `1e409b44`. Bei einem anderen Commit warnt `vergleichen` auf
stderr und nennt die Zahl der verschwundenen Symbole. Diese Einträge zählen als `ignoriert`.
Einen neuen Goldsatz erzeugst du nicht über die ausgefüllte Datei, sondern in eine neue Datei.
