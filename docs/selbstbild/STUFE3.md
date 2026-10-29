# Selbstbild Stufe 3: Suche und Einspeisung (Abnahme)

Stand: 29.10.2026 (Task 11). Code aus Task 1 bis 10 (561 Tests), la_fungus `main` @ `cfed30f`. Für diese Abnahme
wurde **kein Code geändert**. Gemessen wurde VibeMind (`C:\Users\User\Desktop\Vibemind_V1`) am Commit `2de425b9`.

Nachtrag Task 12 (29.10.2026): Der Einspeisungs-Hook nutzt jetzt den Gesprächsverlauf und hat eine
Schweigeschwelle. Das Transkript-Gold zählt nur noch erreichbare Ziele (la_fungus `2d66e8f`, `acadb91`, 587 Tests).
Die Re-Messung steht in **Schritt 4b**. Die Messung aus Schritt 4 bleibt als Historie stehen.

Stufe 3 liest Graph (Stufe 1) und Analyse (Stufe 2) nur lesend und baut daraus `suche.sqlite` (FTS5, Kanten und
optional Datei-Vektoren). Darauf sitzen die Suche (`stichwort`, `struktur`, `dicht`, `fusion`), der MCP-Server, der
Einspeisungs-Hook (UserPromptSubmit), der Schreib-Wächter (PreToolUse), `pruefen` und der erzeugte Skill
`neuer-space`.

**Kurzfassung:**

| Ziel | Ergebnis | Status |
|---|---|---|
| `cells.py#list` nach S16 nicht mehr an der PageRank-Spitze | nicht mehr in den Top-500 | **erreicht** |
| Einspeisung: erwartete Datei im Block, ≥ 16 von 20 | T11: **0 von 20**; nach T12 (Verlauf, bereinigtes Gold): **3 von 20**, davon 0 „neu“ | **verfehlt** (Ursachen s. Schritt 4 und 4b) |
| Einspeisung: kein Block statt eines unpassenden (T12) | schweigt bei 2 von 20 (mit Verlauf) und 4 von 10 (ohne Verlauf); 14 der 17 Blöcke ohne erwartete Datei | **nur teilweise** (die Schwelle trennt kurz von lang, nicht passend von unpassend, s. 4b) |
| Einspeisung: nie `sicher_tot` ohne Warnung | T11: 0 ohne, 2 mit Warnung; T12: 0 ohne, 0 mit Warnung | erreicht |
| Einspeisung: Latenz ≤ 1,5 s | T11: Median 93 ms, Maximum 235 ms; T12 inkl. Transkript: Median 92 ms, Maximum 333 ms (im Prozess), Hook-Prozess Median 134 ms, Maximum 171 ms | erreicht |
| E1: `fusion` schlägt `struktur` um ≥ 5 Punkte | bestes Modell +3,8 Punkte (KETTEN) | **dichte Suche bleibt aus** |
| Wächter: Root-Space deny mit Hinweis, Submodul-Space still | beides wie gefordert | erreicht |
| `pruefen` | 0 Verstöße (auch 0 weiche) | erreicht |
| Schritt 7 (Hooks und MCP in VibeMind) | vorbereitet, **nicht ausgeführt** | offen, wartet auf User-OK |

## Befehle (PowerShell, im la_fungus-Arbeitsbaum)

```powershell
cd C:\Users\User\ClaudeWork\wt-fungus-gate1
$env:PYTHONPATH="src"
.\.venv-selbstbild\Scripts\python -m selbstbild vermessen C:\Users\User\Desktop\Vibemind_V1     # inkrementell
.\.venv-selbstbild\Scripts\python -m selbstbild analysieren C:\Users\User\Desktop\Vibemind_V1 --kolonie-budget
.\.venv-selbstbild\Scripts\python -m selbstbild index C:\Users\User\Desktop\Vibemind_V1         # suche.sqlite
.\.venv-selbstbild\Scripts\python -m selbstbild suchen C:\Users\User\Desktop\Vibemind_V1 "kampagnenplan marketing" --top 10
.\.venv-selbstbild\Scripts\python -m selbstbild suche-messen C:\Users\User\Desktop\Vibemind_V1 --transkripte <ordner> `
    --ollama http://localhost:11434 --modell qwen3-embedding:0.6b
.\.venv-selbstbild\Scripts\python -m selbstbild pruefen C:\Users\User\Desktop\Vibemind_V1
.\.venv-selbstbild\Scripts\python -m selbstbild skill neuer-space C:\Users\User\Desktop\Vibemind_V1 <ziel>\SKILL.md --ueberschreiben
```

Für die Messung lief ein eigener Treiber statt `suche-messen`, weil die laufende Session (ihr Transkript handelt von
Selbstbild selbst) ausgeschlossen werden musste. Er ruft dieselben Funktionen auf (`_kandidaten`, `gold_aus_ketten`,
`messen_quellen`) und zieht die Stichprobe wie `gold_aus_transkripten`: Kandidaten nach Dateiname und Position
sortiert, `random.Random(20261028).sample(…, 20)`. Prompt-Texte lebten nur im Speicher. In diesem Dokument stehen nur
Nummern und Zahlen.

## Läufe und Speicher

Host: Windows 11, 32 GB RAM, parallel andere Sessions. Jeder Lauf startete nur bei ≥ 3 GB freiem Commit und lief
unter einem Wächter (PowerShell, Abtastung alle 3 s über den eigenen Prozessbaum): Abbruch nur des eigenen Baums bei
mehr als 2,5 GB Working Set, bei weniger als 1,5 GB freiem Commit oder nach Zeitablauf. **Kein Lauf wurde
abgebrochen.**

| Lauf | Dauer | Peak WS Prozessbaum | freier Commit vor / Minimum |
|---|---|---|---|
| v1) `vermessen` inkrementell (Stand war `1e409b44`) | 110 s (`dauer_s` 107,8) | 0,93 GB | 5,70 / 3,17 GB |
| a8) `analysieren --kolonie-budget` | 134 s (Laden 4,0 s, Mathe 126,9 s, Kolonie 0,73 s, budgetgleich 0,76 s); Peak WS Python selbst gemessen 1,09 GB, privater Commit 1,80 GB | 1,46 GB | 5,19 / 3,52 GB |
| i1) `index` | 20 s | 1,26 GB | 5,67 / 3,88 GB |
| m1) Messung `stichwort`/`struktur` + Schritt 4 | 27 s | 0,17 GB (0,45 GB im Wiederholungslauf) | 5,72 / 4,77 GB |
| d1) Vektoren + `dicht`/`fusion` `qwen3-embedding:0.6b` | 1 455 s | 0,71 GB | 5,46 / 3,63 GB |
| d2) Vektoren + `dicht`/`fusion` `nomic-embed-text:latest` | 1 193 s | 0,40 GB | 5,56 / **2,33 GB** (unter 3 GB, aber über der Abbruchgrenze 1,5 GB; Ollama lud nomic nach, andere Sessions liefen) |
| t12) Re-Messung Einspeisung (Schritt 4b) | 13,5 s | 0,08 GB | 5,47 / 4,57 GB |
| t12d) Diagnose zu Schritt 4b | 10,1 s | 0,17 GB | 5,85 / 4,70 GB |

Ollama: Nur schon installierte Embedder, nichts gezogen. `qwen3-embedding:0.6b` war bereits geladen (5,8 GB VRAM)
und blieb geladen. `nomic-embed-text` wurde für d2 von Ollama nachgeladen. Andere Modelle wurden nicht angefasst.

## Schritt 1: Neu vermessen, analysieren, indizieren

- **S16 und inkrementell:** Ein inkrementeller Lauf löst die Kanten immer neu auf. `vermessen()` liest nur die
  Parse-Ergebnisse aus dem Cache und ruft danach `aufloesen()` über **alle** Dateien auf (`vermessen.py`, Abschnitt
  „Auflösen“). S16 änderte nur `aufloesen.py`, nicht `PARSER_VERSION`. Ein voller Lauf oder ein anderer
  Rückweg war deshalb nicht nötig. Beleg: 4 Dateien neu geparst, 19 606 aus dem Cache, trotzdem andere Kantenzahlen.
- Graph `2de425b9`: 19 610 Dateien, 234 885 Knoten, **480 177 Kanten (20 973 unsicher)**. Vorher (`1e409b44`):
  482 618 Kanten (23 416 unsicher). S16 entfernt also 2 443 unsichere Rückfallkanten. 89 fehlerhafte Dateien,
  7 fehlende Submodule wie bisher.
- Analyse a8: `sicher_tot` 1 109, `wahrscheinlich_tot` 29 926, `verdaechtig` 21 909. 468 Bereiche (a7: 428; weniger
  Rückfallkanten zerlegen Bereiche feiner). `pagerank_quersprache_ausgelassen` 4 586 (a7: 5 405). Doppelungen
  unverändert (188 / 1 329 / 142). `luecken` 0/0. Kolonie: 2 821 Dateien besucht, Zentrum ∩ PageRank-Top-50 = 12.
- Index: 234 707 Einträge, 460 567 Kanten, 4 001 Doku-Texte, mit Analyse. `index_aktuell` und der MCP-Selbsttest
  melden „aktuell (commit 2de425b9…)“.

## Schritt 2: PageRank-Spitze nach S16

`cells.py#list` ist weg, nicht nur von Rang 1, sondern aus den gesamten Top-500.

| Rang | Symbol (Repo) | PageRank | Eingang `ruft_auf` |
|---|---|---|---|
| 1 | `hermes_constants.py#get_hermes_home` (hermes-agent) | 0,00325 | 303, alle sicher |
| 2 | `gateway/config.py#PlatformConfig` (hermes-agent) | 0,00284 | 942, alle sicher |
| 3 | `hermes_constants.py#_get_platform_default_hermes_home` (hermes-agent) | 0,00196 | 3 sicher (erbt von Rang 1) |
| 4 | `hermes_cli/config.py#load_config` (hermes-agent) | 0,00180 | 461, alle sicher |
| 5 | `hermes_cli/config.py#_load_config_impl` (hermes-agent) | 0,00164 | 2 sicher (erbt von Rang 4) |

Die Top-5 sind echte Zentralität. Rang 6 bis 9 ebenso (`get_hermes_home_override`, coding-engine
`src/mind/event_bus.py#Event` mit 396 sicheren und 66 unsicheren Eingängen, `SessionSource`, `AIAgent`).

**Neuer Restbefund, Rang 10:** hermes-agent
`optional-skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py#read_text` (0,00086) hat
**576 unsichere** und nur 4 sichere `ruft_auf`-Eingänge, alle aus hermes-agent-Python. Vermutlich ist das dasselbe
Muster wie bei `cells.py#list`, diesmal für eine **Methode**: `pfad.read_text()` (`pathlib`) wird über den Namen der
gleichnamigen freien Funktion zugeordnet. S16 deckt nur Builtins und JS-Globale ab. Gezählt habe ich nur aus `kante`
in `suche.sqlite`, die Aufrufstellen habe ich nicht einzeln geprüft. Vorschlag für ein eigenes Ruling: Für
Attributaufrufe `x.name()` mit unbekanntem Empfänger keinen Namens-Rückfall auf **freie** Funktionen (höchstens auf
Methoden).

## Schritt 3: Messung der Suche

Gold:
- **KETTEN** (`docs/spaces/KETTEN.md` im äußeren Repo): 105 Hops mit Pfad.
- **Transkripte:** 95 Transkripte des Projekts (ohne die aktuelle Session, ohne `agent-*`), daraus 203 Kandidaten.
  Gezogen wurden 20, Indizes 1, 9, 11, 36, 54, 55, 59, 67, 117, 127, 138, 142, 148, 149, 152, 158, 168, 175, 176
  und 185.

recall@10 = Anteil der Anfragen mit mindestens einer erwarteten Datei unter den ersten 10 Dateien (nach Datei-Dedup).

| Variante | KETTEN recall@10 | KETTEN MRR | Transkripte recall@10 | Transkripte MRR |
|---|---|---|---|---|
| `stichwort` | 0,581 | 0,353 | 0,05 | 0,017 |
| `struktur` | 0,581 | 0,430 | 0,05 | 0,013 |
| `dicht:qwen3-embedding:0.6b` | 0,429 | 0,270 | 0,00 | 0,000 |
| `fusion:qwen3-embedding:0.6b` | **0,619** | **0,471** | 0,00 | 0,000 |
| `dicht:nomic-embed-text:latest` | 0,210 | 0,120 | 0,05 | 0,007 |
| `fusion:nomic-embed-text:latest` | 0,590 | 0,395 | 0,05 | 0,025 |

`dicht_leer` war in allen dichten Läufen 0. Die Fusion fiel also nie still auf die Struktur-Suche zurück.

**Modellwahl (E1): Die dichte Suche bleibt aus.** Die beste `fusion`-recall@10 hat `qwen3-embedding:0.6b` mit 0,619 auf KETTEN. Das sind nur **+3,8 Punkte** über `struktur` (0,581), gefordert sind 5. nomic kommt auf +1,0 Punkte (0,590). Auf dem Transkript-Gold bringt keine Fusion etwas (0,00 bzw. 0,05 gegen 0,05). Die Fusion hebt zwar die MRR (KETTEN 0,471 gegen 0,430), aber E1 entscheidet über recall@10. `Embedder` bleibt also im MCP-Server und in der Messung optional, als Standard gilt keine dichte Suche. In `suche.sqlite.vektor` liegen nach d2 noch die nomic-Vektoren. Sie stören nicht, denn `aehnlichste` filtert nach Modell, und der nächste `index`-Lauf leert die Tabelle.

Weitere Befunde:
- **`struktur` hebt die Rangfolge, nicht den Recall.** Auf KETTEN steigt MRR von 0,353 auf 0,430 bei gleichem
  recall@10.
- **KETTEN:** 6 von 105 Hops haben kein Ziel im Index. Bei 38 Hops (darunter diese 6) findet `stichwort` keine erwartete Datei unter den
  ersten 500 Treffern. Die übrigen 61 liegen in den Top-10 und 6 auf Rang 11 bis 50.
- **Dauer der dichten Läufe:** Die Vorab-Prognose (nur Einbettung von 128 Proben) sagte 7,9 min für qwen und
  2,7 min für nomic. Real dauerte d1 24 min. Etwa 14 min davon sind Textaufbau: `dicht._text_aus` sucht die
  Symbolnamen je Datei mit `id LIKE ? ESCAPE '\'`, und SQLite nutzt dafür keinen Bereichszugriff
  (`SCAN eintrag USING INDEX sqlite_autoindex_eintrag_1`, 0,035 s je Datei, 23 611 Dateien). Fix-Vorschlag: Bereichsabfrage
  `id >= 'symbol:<repo>:<pfad>#' AND id < 'symbol:<repo>:<pfad>$'` (`$` folgt auf `#`) oder einmal alle Symbole
  sortiert lesen und nach Datei gruppieren.

## Schritt 4: Spec-Abnahme Einspeisung

Aufruf `selbstbild.hooks.einspeisung.kontextblock(cwd=C:\Users\User\Desktop\Vibemind_V1, prompt=…)` für die
20 Transkript-Prompts aus Schritt 3, danach dieselben 20 noch einmal über den echten Hook als Subprozess
(`python -m selbstbild.hooks.einspeisung`, Hook-JSON auf stdin). Gewertet wird über die Stellen- und Doku-Zeilen des
Blocks. Jede Blockzeile wurde einer Index-Zeile zugeordnet (in allen 20 Blöcken 0 nicht zugeordnete Zeilen). Damit
gilt der Abgleich über die volle Datei-Id, nicht nur über den Pfad.

| Nr | Block | Stellen + Doku | „seit Index geändert“ | erwartete Datei im Block | `sicher_tot` mit / ohne Warnung | ms (im Prozess) | ms (Hook-Prozess) |
|---|---|---|---|---|---|---|---|
| 1 | ja | 9 | 0 | nein | 0 / 0 | 235 | 146 |
| 2 | ja | 10 | 1 | nein | 0 / 0 | 142 | 113 |
| 3 | ja | 9 | 0 | nein | 0 / 0 | 87 | 139 |
| 4 | ja | 10 | 0 | nein | 1 / 0 | 197 | 145 |
| 5 | ja | 11 | 0 | nein | 0 / 0 | 48 | 62 |
| 6 | ja | 3 | 0 | nein | 0 / 0 | 16 | 61 |
| 7 | ja | 8 | 0 | nein | 0 / 0 | 168 | 141 |
| 8 | ja | 8 | 0 | nein | 0 / 0 | 81 | 137 |
| 9 | ja | 10 | 0 | nein | 1 / 0 | 90 | 146 |
| 10 | ja | 8 | 0 | nein | 0 / 0 | 186 | 148 |
| 11 | ja | 8 | 0 | nein | 0 / 0 | 222 | 69 |
| 12 | ja | 11 | 0 | nein | 0 / 0 | 218 | 126 |
| 13 | ja | 11 | 0 | nein | 0 / 0 | 115 | 93 |
| 14 | ja | 8 | 0 | nein | 0 / 0 | 69 | 68 |
| 15 | ja | 9 | 0 | nein | 0 / 0 | 89 | 78 |
| 16 | ja | 8 | 3 | nein | 0 / 0 | 151 | 63 |
| 17 | ja | 3 | 2 | nein | 0 / 0 | 28 | 57 |
| 18 | ja | 10 | 3 | nein | 0 / 0 | 73 | 60 |
| 19 | ja | 8 | 0 | nein | 0 / 0 | 96 | 73 |
| 20 | ja | 8 | 3 | nein | 0 / 0 | 51 | 63 |

| Ziel | Ergebnis |
|---|---|
| erwartete Datei im Block, ≥ 16/20 | **0/20** (auch 0/20, wenn „seit Index geändert“ mitzählt) |
| nie `sicher_tot` ohne Warnung | **erfüllt**: 2 `sicher_tot`-Stellen erschienen, beide mit „⚠ tot“. Im Index stehen 1 109 `sicher_tot`, genau so viele wie in der Analyse. |
| Latenz ≤ 1,5 s | **erfüllt**: im Prozess Median 93 ms, Maximum 235 ms (erster, kalter Aufruf). Ein Wiederholungslauf ergab 32 / 95 ms. Hook als eigener Prozess (inkl. Python-Start): Median 86 ms, Maximum 148 ms (Wiederholungslauf 149 ms). |
| Blockgröße ≤ 8 000 Zeichen | erfüllt, höchstens 1 783 Zeichen |

**Das Ziel 16/20 ist klar verfehlt.** Ursachen, belegt nur mit Zahlen:

1. **Das Transkript-Gold ist größtenteils nicht erreichbar.** Bei **13 der 20** Prompts liegt *keine* erwartete Datei
   im Index. Die Ziele liegen dort, wo Selbstbild per Entwurf nicht hinsieht:
   - `vibemind-os/.worktrees/learning-wiring/…` (54 Zieldateien) und `.worktrees/setup-agent/…` (2). Das sind Git-Worktrees
     im Submodul-Ordner. `_relativ` bildet sie auf das Repo `vibemind-os` ab, vermessen wird aber nur der committete Baum.
   - `docs/superpowers/…` (3) und `vibemind-os/spaces/sales-claw/docs/superpowers/…` (8): Pläne und Specs, die erst
     nach dem Prompt entstanden oder nie committet wurden. Im äußeren Repo liegen 12 solche Dateien heute noch
     untracked.
   - `.superpowers/sdd/…` (4): gitignored.

   Über alle 203 Kandidaten haben nur 82 überhaupt ein indiziertes Ziel. Höchstens 7 der 20 Prompts waren also
   lösbar.
2. **Die Prompts beschreiben selten die Datei.** 10 der 20 Prompts sind kürzer als 80 Zeichen (Fortsetzungen wie
   Zustimmung oder „weiter“, 2 bis 7 Tokens nach Stoppwörtern). Die bearbeitete Datei ergibt sich hier aus dem
   Gesprächsverlauf, nicht aus dem Prompt. Lange Prompts (bis 4 482 Zeichen) gehen nur mit ihren ersten
   12 Tokens in die Suche (`MAX_TOKENS`). Von den 7 lösbaren Prompts findet `stichwort` nur bei einem eine erwartete
   Datei unter den ersten 500 Treffern (Rang 3). In den Hook-Block kam sie trotzdem nicht. Der Hook nimmt die 40 besten
   FTS-Zeilen auf Symbol-Ebene, ohne Datei-Dedup, Lebendigkeit und Tot-Dämpfung, und davon nur die ersten
   8 frischen Stellen.
3. **Zusatzmessung mit bereinigtem Gold** (nur zur Einordnung, nicht als Abnahme): Ziele auf indizierte Dateien
   beschränkt, Prompts ohne solches Ziel gestrichen (82 Kandidaten), gleiche Ziehung mit gleichem Seed. Ergebnis:
   `stichwort` und `struktur` recall@10 je 0,05, Einspeisung **1/20**, 0 `sicher_tot` ohne Warnung (4 mit Warnung),
   Latenz Median 93 ms, Maximum 267 ms. Das Problem ist also nicht nur das Gold. Prompt-zu-Datei per Stichwort trägt
   auf diesen Prompts nicht.
4. **Der Hook liefert immer einen Block (20/20), auch zu 2-Token-Prompts.** Weil keiner davon eine der später
   bearbeiteten Dateien enthielt, kosteten fast alle Blöcke Kontext (je etwa 1 400 Zeichen), ohne die Aufgabe zu
   treffen. Review Focus 2 („kein irreführender Block, sondern nichts“) ist damit nur formal erfüllt. Der Block ist
   nicht falsch, aber unpassend.

**Vorschläge (nicht umgesetzt, keine Anpassung an die Abnahme):**
- **Gold (T5):** `gold_aus_transkripten` soll nur Ziele zählen, die im Index liegen bzw. in einem vermessenen Repo
  committet sind (Worktree-, ignorierte und untracked Pfade verwerfen). Prompts ohne solches Ziel sollen entfallen,
  ebenso Prompts unter etwa 80 Zeichen (Fortsetzungen). Sonst misst „16/20“ vor allem das Gold.
- **Hook-Schwelle:** Einen Block nur ausgeben, wenn mindestens 2 Prompt-Tokens in Name oder Pfad derselben Datei
  treffen oder der beste BM25-Wert eine Schwelle überschreitet. Sonst nichts ausgeben.
- **Hook-Ranking:** Die Datei-Dedup und die Tot-Dämpfung aus `suche.abfrage` auch im Hook spiegeln, wie bei Tokens
  und FTS (Ruling U3).
- **Zielbild prüfen:** Die Spec-Zahl 16/20 setzt voraus, dass ein Prompt seine Zieldatei benennt. Für
  Fortsetzungs-Prompts bräuchte der Hook den Gesprächskontext, und den hat er per Entwurf nicht.

## Schritt 4b: Einspeisung nach Task 12

Task 12 hat den Hook umgebaut (la_fungus `2d66e8f`) und das Gold bereinigt (`acadb91`). Danach wurde neu gemessen,
auf demselben Index (`2de425b9`, Graph = Index, äußerer HEAD `8f7be186`, also „Index veraltet“ nach Ruling U6). Der
Index musste nicht neu gebaut werden, denn der Hook schweigt nicht pauschal (17 von 20 Blöcken).

**Was der Hook jetzt tut:**
- **Arbeitsbereich:** `hooks/verlauf.letzte_dateien` liest höchstens 256 KB vor dem Prompt aus dem Transkript
  (`transcript_path` des Hooks, in der Messung zusätzlich `transkript_ende` = Byte-Offset der Prompt-Zeile). Es
  liefert nur Pfade aus `tool_use`-Eingaben (`file_path`, `notebook_path`, `path`), nie Texte. `hauptpfad` bildet
  `…/.worktrees/<name>/…` und `…/.claude/worktrees/<name>/…` auf den Hauptbaum ab. Die Pfade werden wie im Index auf
  Datei-Ids abgebildet (längster Repo-Präfix, Submodule als `<name>/<pfad>`). Dazu kommen 1-Hop-Nachbarn über
  `kante` (`importiert`/`ruft_auf`, beide Richtungen, Datei und ihre Symbole), höchstens 5 je Datei und 40 insgesamt.
- **Prompt-Treffer:** FTS5 wie bisher (40 Zeilen), dann Datei-Dedup und Tot-Dämpfung (×0,5) wie in `suche.abfrage`.
- **Fusion:** RRF (k = 60) aus Prompt-Liste und Arbeitsbereich. Treffer, die auch im Arbeitsbereich liegen, stehen
  oben und tragen „zuletzt bearbeitet“ bzw. „Nachbar von …“. Reine Kontextdateien stehen im eigenen Abschnitt
  „Zuletzt bearbeitet und Nachbarn“ mit Bereich, passender Strukturregel und Warnungen (höchstens 6 + 4 Zeilen).
- **Schweigeschwelle**, vor der Messung festgelegt und committet (`2d66e8f`), danach nicht verändert: kein Block, wenn
  keine Kontextdatei im Index liegt und kein Prompt-Treffer stark ist. Stark heißt, dass mindestens 2 verschiedene
  Prompt-Tokens Name, Pfad oder Text derselben Datei treffen, oder dass der beste bm25-Wert höchstens
  −2 · ln 1000 ≈ −13,8 ist. Begründung: Ein Term trägt etwa −IDF bei, und IDF ≈ ln(1/p) für den Anteil p der
  Einträge, die ihn enthalten. Das ist fast unabhängig von der Indexgröße. Die Schwelle entspricht damit zwei
  Tokens mit je 1 ‰ Häufigkeit in einer Zeile durchschnittlicher Länge. Ohne starken Treffer erscheinen
  Prompt-Treffer nur, wenn sie im Arbeitsbereich liegen.
- **Zweiter Checkout:** Liegt eine Kontextdatei in einem Worktree, nennt der Block Worktree-Name und Hauptpfad.
- Unverändert: Frist 1,5 s (jetzt inklusive Transkript-Lesen), höchstens 8 000 Zeichen, still bei jedem Problem,
  nur Standardbibliothek (plus `selbstbild.struktur` für die Regel je Datei, Ruling U1), Frische nach U6 (ein Block
  nur aus „seit Index geändert“-Namen wird nicht ausgegeben), nie `sicher_tot` ohne Warnung.

**Gold (bereinigt):** Ziele werden mit `hauptpfad` normalisiert, und es zählen nur Ziele im Index. Prompts ohne
solches Ziel entfallen, kurze Prompts bleiben. Aus 95 Transkripten (ohne die laufende Session und ohne `agent-*`)
gab es 203 Kandidaten, davon 106 mit erreichbarem Ziel (T11 ohne `hauptpfad`: 82). Gezogen wurden 20 mit Seed
20261028, sortiert wie bisher (`transkript_stichproben`). 7 der 20 Prompts sind kürzer als 80 Zeichen. Die 20
stammen aus 6 Transkripten. Die Stichprobe unterscheidet sich von der aus Schritt 4, weil die Kandidatenliste
gefiltert ist.

| Nr | < 80 Zeichen | Verlaufspfade / im Index (256 KB) | Ziel im Verlauf | Block | Stellen / Doku / Kontext / geändert | erwartete Datei im Block | ms |
|---|---|---|---|---|---|---|---|
| 1 | ja | 0 / 0 | nein | **nein** | – | nein | 10 |
| 2 | ja | 5 / 3 | nein | ja | 0 / 0 / 7 / 0 | nein | 74 |
| 3 | nein | 8 / 0 | nein | ja | 8 / 0 / 0 / 0 | nein | 101 |
| 4 | nein | 0 / 0 | nein | ja | 1 / 3 / 0 / 0 | nein | 47 |
| 5 | ja | 0 / 0 | nein | **nein** | – | nein | 18 |
| 6 | ja | 3 / 1 | ja (Rang 1) | ja | 0 / 0 / 1 / 0 | **ja** | 27 |
| 7 | nein | 1 / 1 | nein | ja | 8 / 3 / 1 / 1 | nein | 122 |
| 8 | nein | 0 / 0 | nein | ja | 8 / 3 / 0 / 0 | nein | 269 |
| 9 | ja | 0 / 0 | nein | ja | 8 / 0 / 0 / 0 | nein | 205 |
| 10 | ja | 4 / 0 | nein | ja | 8 / 0 / 0 / 3 | nein | 189 |
| 11 | nein | 0 / 0 | nein | ja | 8 / 3 / 0 / 0 | nein | 77 |
| 12 | nein | 0 / 0 | nein | ja | 8 / 3 / 0 / 1 | nein | 333 |
| 13 | nein | 5 / 1 | ja (Rang 1) | ja | 8 / 3 / 1 / 0 | **ja** | 82 |
| 14 | nein | 0 / 0 | nein | ja | 8 / 2 / 0 / 0 | nein | 96 |
| 15 | nein | 1 / 0 | nein | ja | 8 / 1 / 0 / 0 | nein | 124 |
| 16 | ja | 0 / 0 | nein | ja | 0 / 3 / 0 / 0 | nein | 37 |
| 17 | nein | 3 / 1 | nein | ja | 8 / 2 / 1 / 0 | nein | 109 |
| 18 | nein | 2 / 0 | nein | ja | 8 / 0 / 0 / 0 | nein | 87 |
| 19 | nein | 7 / 5 | ja (Rang 2) | ja | 8 / 2 / 9 / 0 | **ja** | 123 |
| 20 | nein | 4 / 0 | nein | ja | 8 / 0 / 0 / 0 | nein | 88 |

| | Ergebnis | Ziel |
|---|---|---|
| (a) erwartete Datei im Block | **3/20** (frisch mit Zeile bzw. Bereich: 3/20; mit „seit Index geändert“: 3/20) | ≥ 16/20, **verfehlt** |
| (b) davon „neu“ (erwartete Datei, die nicht schon unter den letzten Dateien vor dem Prompt war) | **0/20** | ohne Ziel |
| (c) Hook schweigt | 2/20 | – |
| (d) `sicher_tot` ohne Warnung | 0 (auch 0 mit Warnung) | nie, **erfüllt** |
| (e) Latenz inkl. Transkript-Lesen | im Prozess Median 92 ms, Maximum 333 ms; Hook als Prozess (Python-Start, Transkripte bis 239 MB) Median 134 ms, Maximum 171 ms | ≤ 1,5 s, **erfüllt** |
| (f) Blockgröße | Median 1 788, Maximum 2 991 Zeichen | ≤ 8 000, erfüllt |
| ohne Verlauf (Nr. 1–10, `transkript=None`) | schweigt 4/10, trifft 0/10, Median 25 ms, Maximum 99 ms | – |

Jede Stellen-Zeile ließ sich einem Index-Eintrag zuordnen (0 nicht zuordenbar). Die Hook-Prozess-Läufe bekamen den
ganzen `transcript_path` (ohne `transkript_ende`), sie messen also nur die Latenz.

**Ursachen, belegt mit einer Diagnose (keine Abnahme, gleiche 20 Prompts):**

1. **Das 256-KB-Fenster ist zu klein für diese Transkripte.** Vor den Prompts liegen 1,8 bis 224 MB Verlauf, und
   die letzten 256 KB enthalten oft nur Werkzeugergebnisse ohne Dateipfade (9 von 20 Prompts: 0 Pfade). Eine
   erwartete Datei unter den ersten 6 Kontextdateien gäbe es bei 256 KB für 3, bei 1 MB für 6, bei 4 MB für 9 und
   bei 16 MB oder 64 MB ebenfalls für 9 Prompts. Das Lesen kostet bei 256 KB im Median 0,7 ms, bei 1 MB 2,9 ms und
   bei 4 MB 9,6 ms (Maximum 12,7 ms). Das Fenster stammt aus der Schnittstelle des Briefs und wurde **nicht** an die
   Messung angepasst.
2. **Viele Ziele gibt es vor dem Prompt noch nicht.** Bei 5 der 20 Prompts sind alle Ziele Dateien, die erst danach
   angelegt wurden (erste Berührung ist `Write`, vorher nie im Verlauf). Bei Nr. 1 gilt das für 13 der 14 Ziele.
   Kein Hook kann eine noch nicht existierende Datei nennen.
3. **Obergrenze des Verlaufs:** Im *gesamten* Verlauf vor dem Prompt liegt eine erwartete Datei bei 12 der 20
   Prompts. Bei 3 davon steht sie weit hinten (Rang 11, 23 und 28 der Verlaufsdateien) und fiele auch bei großem
   Fenster aus den 6 Kontextzeilen. Mit diesem Gold liegt die erreichbare Zahl für Verlauf plus Stichwort deshalb
   bei etwa 12 bis 13 von 20, nicht bei 16.
4. **Stichwort und Nachbarn finden fast nichts Neues.** Unter den 40 FTS-Zeilen des Prompts liegt eine erwartete
   Datei bei 1 von 20 Prompts, unter 500 Zeilen bei 2 von 20. 1-Hop-Nachbarn der Verlaufsdateien treffen bei 0 von
   20. Daher (b) = 0.
5. **Die Schweigeschwelle trennt kurz von lang, nicht passend von unpassend.** Stark sind 16 der 20 Prompts. Alle
   Prompts mit 12 Tokens (der Deckel `MAX_TOKENS`) erfüllen beide Kriterien (bester bm25 −15,5 bis −22,3, 3 bis
   9 Tokens in derselben Datei). Bei den 4 kurzen Prompts, die schweigen dürften, fehlt beides. Mehrere Prompts
   teilen denselben besten bm25-Wert (−16,59 viermal, −16,14 und −15,5 je zweimal). Ihre ersten 12 Tokens sind also
   wohl gleich (gemeinsamer Vorspann in einem Transkript). Von den 17 Blöcken enthalten 14 keine erwartete Datei.
   Nicht abbildbare Verlaufspfade (256 KB, alle 20 Prompts): 14 in anderen Arbeitsbäumen unter `ClaudeWork`, 5 unter
   `.claude` (Memory), 1 im Temp-Ordner und 11 anderswo außerhalb des Repos. Kein Pfad im Repo ging bei der
   Abbildung verloren.

**Vorschläge (nicht umgesetzt, keine Anpassung an die Abnahme):**
- **Fenster vergrößern:** `letzte_dateien` mit 4 MB statt 256 KB aufrufen. Laut Diagnose wären das 9 statt 3 von 20,
  bei etwa 10 ms Lesezeit. Das sollte als eigene Entscheidung mit neuer Messung auf einer frischen Stichprobe
  laufen, nicht als Nachjustierung dieser Abnahme.
- **Schwelle schärfen:** Tokens nach Seltenheit gewichten oder nur Treffer in Name und Pfad zählen, nicht im Text.
  Lange Prompts treffen über Symbolnamen im Datei-Text fast immer 2 Tokens. Auch hier gilt: erst festlegen, dann
  auf frischer Stichprobe messen.
- **Zielbild der Abnahme anpassen:** „16 von 20“ setzt voraus, dass das Ziel vor dem Prompt existiert und erkennbar
  ist. Vorschlag: neu angelegte Ziele aus dem Gold nehmen oder getrennt zählen, und (b) als eigentliche Kennzahl
  für „findet die richtige Stelle“ führen.

## Schritt 5: Wächter-Durchstich (ohne zu schreiben)

Wächter als Subprozess (`python -m selbstbild.hooks.waechter`), Hook-JSON mit `tool_name: Write` auf stdin,
`cwd` = VibeMind. Keine Zieldatei existierte vorher oder nachher.

| Fall | Ziel | Ergebnis | ms |
|---|---|---|---|
| Root-Space | `…\Vibemind_V1\spaces\testspace\main.py` | **deny**: `kein_space_im_root: 'spaces/testspace/main.py' ist laut Regel verboten (spaces/*/). Spaces gehoeren nach vibemind-os/spaces/<name>/ (Vertrag: …)` | 61 |
| Submodul-Space | `…\Vibemind_V1\vibemind-os\spaces\testspace\main.py` | **keine Ausgabe** | 58 |
| Root-Space, `/` und klein geschrieben | `c:/users/user/desktop/vibemind_v1/spaces/testspace/main.py` | deny (wie oben) | 60 |
| Root-Space über `..` | `…\Vibemind_V1\docs\..\spaces\testspace\main.py` | deny (wie oben) | 58 |
| `capabilities.yaml` im Root-Unterordner | `…\Vibemind_V1\foo\capabilities.yaml` | deny (`*/capabilities.yaml`) | 59 |
| `capabilities.yaml` im Submodul-Space | `…\vibemind-os\spaces\marketing\capabilities.yaml` | keine Ausgabe | 64 |
| außerhalb jedes Repos | `%TEMP%\irgendwo\spaces\x\main.py` | keine Ausgabe | 65 |

Alle mit Exit 0. `python -m selbstbild pruefen C:\Users\User\Desktop\Vibemind_V1`: **0 Verstöße, Exit 0, 0,2 s, also auch
0 weiche Verstöße.** Das ist erwartet: Die Regeln des äußeren Repos haben leere `pflicht`-Listen, und die Submodule
haben keine eigenen Pflichtteile.

## Skill `neuer-space`

Nach dem Index neu erzeugt (`skill neuer-space … --ueberschreiben`). Er hat jetzt den Abschnitt **Vorbilder**:
`vibemind-os/spaces/captain_cook/` (Lebendigkeit 1 548,42, 7 161 Dateien), `vibemind-os/spaces/marketing/` (435,23,
366) und `vibemind-os/spaces/sales-claw/` (314,07, 228). Committet im äußeren Repo als `8f7be186`
(`docs: Skill neuer-space mit Vorbildern`, nur diese Datei).

Anmerkung: Die Rangfolge nach der **Summe** der Lebendigkeit bevorzugt große Ordner. `captain_cook` enthält das
eingebettete hermes-agent-Submodul (der größte Teil der 7 161 Dateien) und ist als Vorlage für einen neuen Space
eher untypisch. Ein Mittelwert oder eine Obergrenze je Ordner wäre die Alternative. Geändert habe ich nichts.

Nebenwirkung: Seit diesem Commit ist der äußere HEAD `8f7be186`, der Index steht auf `2de425b9`. Der Hook
kennzeichnet Blöcke deshalb als „Index veraltet“, liefert aber weiter frische Stellen (Ruling U6). Alle Messungen
oben liefen vor diesem Commit, nur d2 lief danach. Die Suche selbst hängt nicht am HEAD.

## Schritt 7: offen, wartet auf User-OK

Nicht ausgeführt: kein `hooks einspielen` in die echte `settings.local.json`, keine MCP-Registrierung, keine `.pth`
im echten venv. Vorbereitet in `C:\Users\User\Desktop\Vibemind_V1\.superpowers\sdd\2026-10-29-selbstbild-stufe3-suche-einspeisung\`:

- `step7-hooks.diff`: Ausgabe von `hooks einspielen …\.claude\settings.local.json --trocken --python
  …\.venv-selbstbild\Scripts\python.exe`. Sie hängt `hooks.UserPromptSubmit` (Einspeisung) und
  `hooks.PreToolUse` mit matcher `Write|Edit|MultiEdit|NotebookEdit` (Wächter) an, beide mit `timeout: 5`. Die Datei
  blieb unverändert (Hash vorher und nachher gleich).
- **`.pth` ist nötig.** `selbstbild` ist aus `.venv-selbstbild` ohne `PYTHONPATH` nicht importierbar
  (`find_spec('selbstbild')` ist `None`, in site-packages liegen nur `distutils-precedence.pth` und `pywin32.pth`).
  Die Hook-Befehle setzen kein `PYTHONPATH`. Ohne `hooks pth` enden beide Hooks deshalb mit **Exit 1** und
  `No module named 'selbstbild'` (geprüft). Das ist kein stilles Exit 0, wie es die Hook-Regel verlangt, denn der
  Fehler liegt vor dem Hook-Code. Claude Code zeigt ihn als nicht blockierenden Hook-Fehler an. Darum muss
  `hooks pth` zuerst laufen.
- `step7-mcp.txt`: der Selbsttest (Index aktuell, 7 Werkzeuge) und der Befehl
  `claude mcp add --scope local --env SELBSTBILD_WURZELN=C:\Users\User\Desktop\Vibemind_V1 selbstbild --
  …\.venv-selbstbild\Scripts\python.exe -m selbstbild.mcp_server` (nach `hooks pth`, sonst zusätzlich
  `--env PYTHONPATH=…\src`).

Reihenfolge nach OK: `hooks pth` → `hooks einspielen` (ohne `--trocken`) → `claude mcp add` → in neuer Session
`/mcp` prüfen.

## Bekannte Grenzen und offene Punkte

- **Einspeisung 0/20 (T11), 3/20 nach T12** gegen das Spec-Ziel 16/20 (s. Schritt 4 und 4b). Die Abnahme dieses
  Ziels ist **nicht bestanden**. Mit diesem Gold liegt die Obergrenze für Verlauf plus Stichwort bei etwa 12 von 20,
  denn bei 5 der 20 Prompts wurden alle Ziele erst danach angelegt.
- **E1:** Die dichte Suche bleibt aus. Fusion mit qwen 0.6b liegt auf KETTEN 3,8 Punkte über `struktur`, gefordert
  sind 5.
- **Restartefakt PageRank Rang 10** (`read_text`, 576 unsichere Methoden-Rückfallkanten). Vorschlag: eigenes Ruling.
- **`gold_aus_transkripten`** las bis T11 jede JSONL-Datei ganz in den Speicher. Seit T12 liest es zeilenweise
  (Peak der Re-Messung 0,08 GB).
- **Prognose dichter Läufe** unterschätzt die Dauer um den Textaufbau (s. Schritt 3).
- Die Messung ist einmalig. Wiederholt habe ich nur die Messung m1 (gleiche Zahlen, kürzere Latenzen).
