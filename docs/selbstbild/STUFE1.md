# Selbstbild Stufe 1: Abnahme

Stand 2026-10-28, nach der Final-Fix-Welle (R18–R22). Paket `selbstbild` (`src/selbstbild/`), Tests
`tests/selbstbild/` (204 grün mit `-W error`). Alle Läufe lesen nur committeten Code, schreiben nur in den lokalen Stand
(`%LOCALAPPDATA%\selbstbild\<name>-<hash>\graph.sqlite`) und nie in die vermessenen Arbeitsbäume.

## Bedienung

PowerShell, im Arbeitsbaum `C:\Users\User\ClaudeWork\wt-fungus-gate1` (kein editable Install,
R6, deshalb `PYTHONPATH=src`):

```powershell
cd C:\Users\User\ClaudeWork\wt-fungus-gate1
$env:PYTHONPATH = "src"
.\.venv-selbstbild\Scripts\python -m selbstbild vermessen C:\Users\User\Desktop\Vibemind_V1           # Bericht als JSON
.\.venv-selbstbild\Scripts\python -m selbstbild vermessen C:\Users\User\Desktop\Vibemind_V1 --voll    # Parse-Cache umgehen
.\.venv-selbstbild\Scripts\python -m selbstbild vermessen C:\Users\User\Desktop\Vibemind_V1 --commit HEAD~1
.\.venv-selbstbild\Scripts\python -m selbstbild export C:\Users\User\Desktop\Vibemind_V1 "$env:TEMP\vibemind-export.json"
.\.venv-selbstbild\Scripts\python -m selbstbild status C:\Users\User\Desktop\Vibemind_V1             # Name, Commit, Zähler, Alter
.\.venv-selbstbild\Scripts\python -m selbstbild -v vermessen C:\Users\User\Desktop\Vibemind_V1        # Fortschritt je Repo nach stderr
```

`export` und `status` legen nichts an. Sie suchen den vorhandenen Stand über die Pfadkennung
der Wurzel (`<name>-<sha256(Pfad)[:12]>`), finden ihn also auch nach einer Namensänderung in
der `selbstbild.yaml`. Der vermessene Name steht in der Meta `name`. Ohne Stand: Meldung
`noch nicht vermessen`, Exit 1. Stand ohne Graph (z. B. ein Stand im alten Format, dessen
Graph beim Öffnen verworfen wurde): Meldung `neu vermessen nötig`, Exit 3. Es wird dann nie
ein leerer Graph exportiert.

## Abnahme-Läufe

Host: Windows 11, 32 GB RAM. Python 3.11, tree-sitter 0.26.0, Language-Pack 1.22.1.

| Lauf | Commit | Dateien | geparst | aus Cache | übersprungen (Grund: Anzahl) | Knoten | Kanten | unsicher | fehlerhafte Dateien | fehlende Submodule | Dauer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a) `Vibemind_V1 --voll`, 20 Repos (äußeres Repo + 19 Submodule) | `6f8e4e0f` | 19 607 | 19 607 | 0 | keine_sprache 2004, binaer 332, zu_gross 92, ausgenommen 41, submodul_fehlt 7, lfs 4, symlink 2 | 234 874 | 483 389 | 25 800 (5,3 %) | 87 | 7 (s. u.) | **302 s** voll |
| b) zweiter Lauf, gleicher Commit, ohne `--voll` | `6f8e4e0f` | 19 607 | **0** | 19 607 | wie a) | 234 874 | 483 389 | 25 800 | 87 | 7 | **62 s** inkrementell |
| c) `vibemind-os/spaces/sales-claw --voll` (eigenes Git-Repo, keine Regeldatei, also Vorgaben) | `c8679622` | 599 | 599 | 0 | keine_sprache 49, binaer 25, ausgenommen 2, zu_gross 1 | 9 068 | 21 119 | 2 288 (10,8 %) | 1 | 0 | 7,3 s |
| c') Kontrolllauf nach der Final-Fix-Welle, wie c), frischer Stand | `c8679622` | 599 | 599 | 0 | keine_sprache 74, ausgenommen 2, zu_gross 1 | 9 067 | 21 728 | 2 154 (9,9 %) | 2 | 0 | 9,1 s voll, **1,3 s** inkrementell |

Die Dauer ist `Bericht.dauer_s`, also nur `vermessen()` ohne Prozessstart und Export.
Lauf c) endete ohne Änderung am Kern. Lauf c) misst den `HEAD` des sales-claw-Checkouts, nicht
den Pin aus vibemind-os, deshalb 599 statt 228 Dateien wie in a).

Lauf c') zeigt die Wirkung der Final-Fix-Welle auf denselben Commit. Die 25 Bilder ohne
Sprach-Endung zählen jetzt als `keine_sprache`, weil die Sprache vor dem Lesen geprüft wird.
Die zweite fehlerhafte Datei ist `sales-mcp/ui.py`. Sie nutzt f-String-Syntax aus Python 3.12
(PEP 701), die `ast` unter 3.11 nicht liest. Vorher fehlten ihre Routen still, jetzt trägt sie
`spezial_fehlt=1`. Kantenarten in c'): ruft_auf 10 790 (1 129 unsicher), enthaelt 8 508,
liest_schreibt 1 027, importiert 676, liest_env 616, registriert 98, ruft_http 9, verbindet 4.
Die Läufe a) und b) wurden nicht neu erhoben.

Die Zahlen stammen aus der Messung nach Fix-Runde 1 (streamender Export, R17). Die erste
Messung (vor den Fixes) ergab dieselben Zähler bei 290 s voll und 46 s inkrementell. Der
inkrementelle Lauf ist jetzt langsamer, weil der zeitlich geordnete Symbolverlauf je Repo
einmal ganz gelesen wird.

**Determinismus:** Der Export von a) und der nach b) sind byte-gleich (357 MB, sha256
`9fa19484…e1c517f`, ohne `seit`). Vor den Fixes waren schon zwei getrennte volle Läufe und b)
byte-gleich (mit `seit`, 366 MB).

**Speicher (Peak Working Set, `K32GetProcessMemoryInfo`, ganzer Prozess inkl. Export):**

| Prozess | Peak nach Fix (streamender Export) | Peak vorher |
|---|---|---|
| a) `vermessen` voll + `export_json` | **0,96 GB** | 2,83 GB |
| b) `vermessen` inkrementell + `export_json` | **1,08 GB** | 2,83 GB |
| c) sales-claw voll + Export | 0,09 GB | 0,15 GB |

Der Export (22 s für a) hebt den Peak nicht mehr an: Der Wert vor dem Export ist gleich dem
Endwert. `graph.sqlite` für a) ist jetzt 1,38 GB groß (Parse-Cache, Graph mit Sortierspalten
und Indizes, Symbolverlauf). Darin stecken auch die freien Seiten der beim Umbau verworfenen
alten Tabellen, denn es läuft kein VACUUM.

**Fehlende Submodule in a)** (nicht initialisiert, als `submodul_fehlt` gemeldet, der Rest wurde
vermessen): `vibemind-os/spaces/desktop/Automation_ui`, `vibemind-os/spaces/sales`,
`vibemind-os/spaces/learning/learnhouse`, `vibemind-os/spaces/learning/penecho`,
`vibemind-os/voice/external/Autogen_AgentFarm`, `vibemind-os/voice/external/openclaw`,
`vibemind-os/voice/external/zeroclaw`.

**Graph a) nach Art:** 205 038 Symbole, 19 607 Dateien, 3 999 Doku-Seiten, 3 406 Env-Namen,
1 379 Routen, 670 MCP-Werkzeuge, 446 Tabellen, 158 SQL-Funktionen, 151 Dienste, 20 Repos.

| Kantenart | Anzahl | davon unsicher |
|---|---|---|
| enthaelt | 224 645 | 0 |
| ruft_auf | 220 598 | 23 006 (10,4 %) |
| importiert | 24 723 | 1 009 (4,1 %) |
| liest_env | 7 889 | 0 |
| liest_schreibt | 2 743 | 1 776 (64,7 %) |
| registriert | 673 | 0 |
| definiert_route | 1 512 | 0 |
| nutzt_dienst | 457 | 0 |
| verbindet | 149 | 9 |

Die Zeile `nutzt_dienst` stammt aus der Messung vor R18. Darin stecken noch die
HTTP-Aufrufstellen, die jetzt als eigene Kantenart `ruft_http` gezählt werden.

## d) Secret-Probe (startet der User selbst)

`scripts/secret_probe.py` liest die Werte einer `.env` nur im Speicher und sucht jeden Wert
(roh und JSON-escaped, als UTF-8-Bytes) in `graph.sqlite`, einer evtl. vorhandenen
`graph.sqlite-wal` und im Export. Ausgegeben werden nur Variablennamen mit Treffer und Zähler.
Exit-Codes: 0 kein Treffer, 1 mindestens ein Treffer, 2 Prüfung nicht möglich (`--env`,
`--graph` oder `--export` fehlt oder ist unlesbar, `--graph` ist keine lesbare
SQLite-Datenbank). Die Probe schlägt also geschlossen fehl: Ein Fehler wird nie als „0 Treffer“
gemeldet. Werte unter 6 Zeichen werden ignoriert, weil sie als gewöhnliche Teilstrings überall
vorkommen.

Abweichung von Spec 12a: Statt sha256-Vergleich mit Kandidaten-Strings wird direkt nach
Teilstrings gesucht (Ruling R16). Das ist strenger, weil es auch Werte findet, die in
längere Strings eingebettet sind.

Zusätzlich liest das Skript alle Text- und Blob-Spalten aller Tabellen über sqlite3 (nur
lesend). Das findet auch Werte, die in der Datei über Overflow-Seiten verteilt sind. Die
Dateien werden blockweise mit Überlappung durchsucht. Auf mmap wird verzichtet, weil
`mmap.find` unter dem hier genutzten CPython 3.11.0 unter Windows abstürzt. Unquotierte Werte enden vor ` #` (Inline-Kommentar), in
Quotes gehört `#` zum Wert.

Aufruf (PowerShell, nach Lauf a):

```powershell
cd C:\Users\User\ClaudeWork\wt-fungus-gate1
$env:PYTHONPATH = "src"
.\.venv-selbstbild\Scripts\python -m selbstbild export C:\Users\User\Desktop\Vibemind_V1 "$env:TEMP\vibemind-export.json"
.\.venv-selbstbild\Scripts\python scripts\secret_probe.py --env C:\Users\User\Desktop\Vibemind_V1\.env --graph "$env:LOCALAPPDATA\selbstbild\vibemind-8ac5c0f3db0b\graph.sqlite" --export "$env:TEMP\vibemind-export.json"
```

Der Implementierer hat das Skript nur mit einer Fake-`.env` in den Unit-Tests ausgeführt
(`tests/selbstbild/test_secret_probe.py`), nie mit echten Werten. In den Tests von `vermessen`
prüft `test_keine_secret_werte` dieselbe Eigenschaft für `.env`, `.env.example`, Compose,
PowerShell und Python.

## Bekannte Grenzen

1. **Aufrufkanten über Paket- und Repo-Grenzen.** Es gibt null `ruft_auf`-Kanten zwischen zwei
   Repos, denn die Auflösung läuft je Repo. Brücken über Repos entstehen nur über geteilte
   Ressourcen (Env-Namen, Dienste, Tabellen, Routen). Von den 60 016 Aufrufkanten zwischen
   verschiedenen Top-Level-Ordnern sind 8 462 (14 %) unsicher. TS-Namensimporte bleiben
   unsicher (R15), relative Python-Importe über mehr als eine Ebene ebenfalls (R13).
   Absolute Python-Importe aus einer Datei außerhalb eines Pakets suchen zuerst im eigenen
   Ordner (`sys.path[0]`-Semantik, Skript-Ordner), ein Treffer dort ist sicher. Innerhalb eines
   Pakets (Ordner mit `__init__.py`) gilt das nicht. Der unsichere Rückfall „Name existiert im
   Repo genau einmal“ zielt nie auf Methoden.
2. **TS: Barrels eine Ebene, keine Pfad-Aliase.** `export { X } from "./x"` und
   `export * from "./x"` gelten als Importe. Der Auflöser folgt einem Re-Export genau eine
   Ebene weit (Ergebnis immer unsicher, R15). Ketten über mehrere Barrels und
   `export { X as Y }` unter dem neuen Namen Y werden nicht aufgelöst. Pfad-Aliase (`@/…`,
   tsconfig `paths`) und Paketnamen bleiben unaufgelöst. Solche Importe gelten als extern.
3. **Tabellenzugriffe meist unsicher** (65 % in a, 99,8 % in sales-claw). Die Tabellen sind
   nicht im selben Repo per SQL definiert (Supabase-Schema liegt woanders), und die Regex
   erkennt nur SQL mit führendem Verb (R9). `TABLE IF [NOT] EXISTS name` ergibt `name`. Das
   Innere von `EXTRACT`/`SUBSTRING`/`TRIM`/`OVERLAY(...)` nennt keine Tabelle.
4. **`seit` ist Stand-Wissen, nicht Teil des Exports (R17).** Der Symbolverlauf wird je Lauf
   aus den Digests fortgeschrieben (Spec 6.1). `seit` ist der Beginn der Strecke des Digests,
   der zur Commit-Zeit des Laufs gilt. Es steht als Attribut der Symbolknoten in der Tabelle
   `knoten` des Stands und fehlt im Export. Der Export bleibt damit eine reine Funktion von
   (Commit, Regeln): inkrementell == voll byte-gleich, auch bei verschiedenen Commit-Zeiten.
   Der Verlauf ist nach Commit-Zeit geordnet, je Repo (Bereich) getrennt:
   - Ein später nachgemessener älterer Commit wird zeitlich einsortiert. Der bisherige
     Zustand am nächsten erfassten Lauf wird festgeschrieben, spätere Strecken bleiben richtig.
   - Ein Lauf auf einer schon erfassten Zeit ist idempotent.
   - Fehlt ein Symbol in einem erfassten Lauf, beginnt seine Strecke danach neu, auch mit
     gleichem Digest. Ohne erfassten Lauf dazwischen gibt es keine Lücke.
   - `seit` ist nur so genau wie die erfassten Läufe: Ein frischer Stand kennt nur den
     aktuellen Commit, dann ist `seit` dessen Commit-Zeit. Zwei Commits mit gleicher
     Commit-Zeit gelten als derselbe Lauf.
   - Submodul-Symbole tragen die Commit-Zeit ihres Submoduls.
   - Der alte, nach Einfügereihenfolge geführte Verlauf (vor R17) wird beim ersten Öffnen
     verworfen. Ebenso ein Graph ohne Sortierspalten. Dabei werden auch die Meta-Einträge
     `commit` und `bericht` gelöscht, `status`/`export` melden dann `neu vermessen nötig`.
5. **Export streamt, aus einem Snapshot.** `export_json` schreibt Kopf, Kanten und Knoten
   nacheinander aus sortierten Abfragen (Sortierspalten als UTF-8-BLOB, Bytevergleich =
   Codepunkt-Ordnung). Alle drei Lesevorgänge laufen in einer Lese-Transaktion (WAL-Snapshot).
   Ein paralleles `graph_ersetzen` mischt sich also nicht hinein. Das Format bleibt
   `json.dumps(sort_keys=True, indent=1)`, deshalb stehen die Kanten vor den Knoten. `vermessen`
   selbst hält während des Laufs alle Parser-Ergebnisse und den Graphen im Speicher (Peak etwa
   1 GB bei 20 Repos und 19 600 Dateien). SQLite läuft mit `synchronous=NORMAL` im WAL-Modus.
6. **Regeln gelten je Repo (R21).** Die `selbstbild.yaml` im äußeren Repo wirkt nur auf dessen
   Pfade. Globs auf Submodul-Pfade (`vibemind-os/...`) erreichen das Submodul nie. Deshalb
   wurden sie aus der äußeren Datei entfernt (doku_pfade, Strukturregel `ort`,
   Einstiegspunkte). Im Submodul gelten dessen eigene Regeln bzw. die Vorgaben (`**/*.md` als
   Doku). Die Regeldatei für `vibemind-os` mit der Space-Regel kommt mit dem Plan für Stufe 3
   (Wächter), nicht in Stufe 1. Das Dienst-Register (`config/dienste.yaml`) wird dagegen über
   alle Repos gemischt, das äußere gewinnt bei Doppelungen. Strukturregeln und `issue_ziel`
   werden in Stufe 1 nur geladen, nicht ausgewertet.
7. **Fehlerhafte Dateien:** 87 in a), davon 51 YAML (Vorlagen und Mehrfach-Dokumente), 35 TS/TSX/JSX
   und 1 Python. Was tree-sitter erkennt, wird übernommen. Eine Parser-Ausnahme bricht den Lauf
   nicht ab. Die Datei wird nur für diesen Lauf als fehlerhaft markiert und nicht gecacht, der
   nächste Lauf parst sie neu. In den Läufen a und b trat keine Ausnahme auf. Kann `ast` eine
   Python-Datei nicht lesen, die tree-sitter parst (z. B. Syntax aus Python 3.12 unter 3.11),
   fehlen ihre Routen, MCP-Werkzeuge und HTTP-Aufrufe. Die Datei wird dann `fehlerhaft` und
   trägt das Attribut `spezial_fehlt=1`. Die Python-Version (`sys.version_info[:2]`) steht
   deshalb im Cache-Schlüssel.
8. **Blobs werden nur bei Bedarf gelesen.** Blobs ohne Sprache werden nie gelesen. Für einen
   Cache-Treffer (Blob, Sprache, Pfad) liest `baum_lesen` den Inhalt ebenfalls nicht, denn die
   Binär- und LFS-Prüfung hat er schon bestanden. Der erste Lauf liest jeden Sprach-Blob
   zweimal (Klassifikation, dann Parsen). Die 62 s aus b) sind vor dieser Änderung gemessen.
   Neu gemessen ist nur sales-claw: 1,3 s ohne Änderung (c').
9. **Parse-Cache** ist pfadabhängig (R11, `pfad` ist Pflichtparameter). Umbenennungen werden
   deshalb neu geparst. Der Cache wird nie aufgeräumt und wächst mit jedem neuen Blob.
10. Gleiche Blobs am gleichen Pfad in zwei Repos teilen einen Cache-Eintrag. Das ist gewollt,
    denn Parser-Ergebnisse sind repo-frei (R10). In a) sind es 19 605 Einträge für 19 607 Dateien.
11. **Zeitachse auf der First-Parent-Linie (R20).** Die Zeitachse liest
    `git log -m --first-parent`, weil die Rückwärts-Abbildung der Umbenennungen eine lineare
    Historie braucht. Arbeit auf einem Branch zählt deshalb als eine Änderung zum
    Merge-Zeitpunkt, mit der Person, die gemergt hat. Die einzelnen Branch-Commits und ihre
    Autoren tauchen nicht auf.
12. **Rust-Makrokörper werden nicht durchlaufen (R22).** Für tree-sitter ist ein Makrokörper
    nur ein Token-Baum. Aufrufe darin (`vec![…]`, `tokio::select! {…}`, eigene `macro_rules!`)
    fehlen, nur die Env-Makros werden gelesen. Stufe 2 darf
    Rust-Symbole deshalb nicht als „sicher tot“ einstufen, solange Makrokörper fehlen.
13. **HTTP-Aufrufe sind eine eigene Kantenart (R18).** `ruft_http` führt von der Aufrufstelle
    (Symbol oder Datei) zu `env:<BASE_URL>`. Methode und Pfad stehen im Beleg-Anker.
    `nutzt_dienst` bedeutet nur noch Symbol bzw. Datei → `dienst:<name>` aus dem Dienst-Register.
14. **Herkunft der Beleg-Digests (R19).** `Beleg.digest` belegt die Quelle, nicht den Anker:
    - Aufruf-, Env-, Import- und Tabellenkanten: Digest des umschließenden Symbolkörpers
      (`Symbol.digest`), auf Modulebene die Git-Blob-SHA der Datei.
    - Routen, MCP-Werkzeuge, HTTP-Aufrufe: sha256 des AST-Quellausschnitts (Dekorator bzw.
      Aufruf), Zeilenenden normalisiert.
    - Compose: sha256 des Dienstblocks, ersatzweise die Blob-SHA.
    - Symbole und SQL-Objekte: wie bisher der Digest ihres eigenen Körpers.

    Bei doppelten Qualnames (`@property` und `@x.setter`) stammen Beleg, Kanten-Digests und
    Symbolverlauf aus der ersten Definition.
15. **Keine `compose`- und `aufgabe`-Einstiege.** Die Einstiegsarten gibt es im Typ. Erzeugt
    werden sie aber erst mit dem Plan für Stufe 2. Compose-Dienste und geplante Aufgaben machen
    Code in Stufe 1 also nicht erreichbar.
16. **Express-Routen nur mit Handler.** Eine TS/JS-Route gilt nur, wenn das zweite Argument ein
    Bezeichner oder eine Funktion ist (`app.get("/x", h)`). `cache.delete("/tmp")` ist keine
    Route. Env-Namen aus `const { A, X: y } = process.env` werden gelesen (A, X).
